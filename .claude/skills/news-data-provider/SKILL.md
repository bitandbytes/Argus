---
name: news-data-provider
description: "Use this skill when implementing or modifying the real NewsDataProvider (Phase 3.1). Triggers on: 'implement NewsDataProvider', 'Alpha Vantage NEWS_SENTIMENT', 'Finnhub company news', 'real news headlines', 'news caching', 'rate limit handling', 'news provider fallback', 'replace news stub', 'wire news into FinBERT', or any task that fetches financial news headlines from an external API and caches them for FinBERT processing. Do NOT use for FinBERT inference itself (use finbert-integration) or for LLM-based news analysis (use llm-validator)."
---

# News Data Provider Skill

This skill guides Claude in replacing the Phase 1 `NewsDataProvider` stub with a real implementation that fetches headlines from Alpha Vantage `NEWS_SENTIMENT`, falls back to Finnhub, and caches results on disk. The output feeds the `FinBERTEnricher` plugin.

## When to Use This Skill

- Implementing `src/data/news_data.py` (replacing the `[]`-returning stub)
- Adding Alpha Vantage `NEWS_SENTIMENT` client
- Adding Finnhub `/company-news` fallback
- Implementing `(ticker, date)`-keyed disk cache with 12-hour TTL
- Writing fixture-based unit tests for both providers
- Wiring the real provider into `FinBERTEnricher._load_recent_headlines()`

Do NOT use this skill for:
- FinBERT inference or feature computation → use `finbert-integration`
- LLM-based deep news analysis → use `llm-validator`
- Earnings calendar fetching → use `event-filter`

## Architecture Context

The `NewsDataProvider` sits in the data ingestion layer. It produces a list of `{text, date, source}` dicts per ticker per day. These feed into `FinBERTEnricher.batch_enrich()`, which scores them and produces rolling sentiment features.

```
NewsDataProvider.get_headlines(ticker, start, end)
        │
        ▼
FinBERTEnricher.analyze_batch_cached(headlines)
        │
        ▼
FeatureStore.update_sentiment(ticker, features)
```

The Phase 1 stub returns `[]`, which causes `FinBERTEnricher` to emit all-zero sentiment features. The Phase 3 implementation replaces only `NewsDataProvider` — no changes to `FinBERTEnricher` are needed.

## API Overview

### Alpha Vantage NEWS_SENTIMENT (primary)

- **Free tier**: 25 calls/day, 5 calls/minute
- **Endpoint**: `GET https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&time_from={YYYYMMDDTHHMM}&apikey={key}`
- **Response**: JSON with `feed[]` array; each item has `title`, `time_published`, `overall_sentiment_score`, `ticker_sentiment[]`
- **Advantage**: Pre-tagged with tickers, includes a built-in sentiment score (useful as a cross-check against FinBERT)
- **Lookback**: ~2 years of history on free tier

### Finnhub company-news (fallback)

- **Free tier**: 60 calls/minute (generous)
- **Endpoint**: `GET https://finnhub.io/api/v1/company-news?symbol={symbol}&from={YYYY-MM-DD}&to={YYYY-MM-DD}&token={key}`
- **Response**: JSON array; each item has `headline`, `datetime` (Unix timestamp), `source`
- **When to use**: Alpha Vantage rate-limited (429), empty result, or key not configured

## Step 1: Implement NewsDataProvider

```python
# src/data/news_data.py

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
import requests
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_TTL_HOURS = 12


@dataclass
class Headline:
    text: str
    date: str        # ISO date string: "2024-01-15"
    source: str
    raw_sentiment: Optional[float] = None  # Alpha Vantage's own score, if available


@dataclass
class NewsDataProvider:
    """Fetches financial news headlines from Alpha Vantage with Finnhub fallback.

    Cache: data/raw/news/{ticker}/{YYYY-MM-DD}.json (12-hour TTL)
    Rate limiting: 5 calls/minute for Alpha Vantage (enforced internally)
    Empty results: returned as [] — FinBERTEnricher handles this gracefully
    """
    alpha_vantage_key: Optional[str] = None
    finnhub_key: Optional[str] = None
    cache_root: Path = Path("data/raw/news")
    _last_av_call: float = field(default=0.0, init=False, repr=False)
    _av_min_interval: float = field(default=12.0, init=False, repr=False)  # 5/min = 12s gap

    def get_headlines(
        self,
        ticker: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> list[Headline]:
        """Return headlines for a ticker between start and end (inclusive).

        Checks disk cache first. On cache miss, calls Alpha Vantage with Finnhub fallback.
        Empty list is returned (not raised) on API failure.

        Args:
            ticker: Stock symbol, e.g. "AAPL"
            start: First date to fetch (default: 7 days ago)
            end: Last date to fetch (default: today)
        """
        if end is None:
            end = date.today()
        if start is None:
            start = end - timedelta(days=7)

        headlines = []
        current = start
        while current <= end:
            cached = self._load_cache(ticker, current)
            if cached is not None:
                headlines.extend(cached)
            else:
                fetched = self._fetch_for_date(ticker, current)
                self._save_cache(ticker, current, fetched)
                headlines.extend(fetched)
            current += timedelta(days=1)

        return headlines

    def get_macro_news(self, start: Optional[date] = None, end: Optional[date] = None) -> list[Headline]:
        """Fetch macro/market-wide news. Stub — returns [] until Phase 3+ implementation."""
        return []

    # -------------------------------------------------------------------------
    # Fetching
    # -------------------------------------------------------------------------

    def _fetch_for_date(self, ticker: str, for_date: date) -> list[Headline]:
        """Fetch headlines for a single ticker on a single date."""
        if self.alpha_vantage_key:
            try:
                return self._fetch_alpha_vantage(ticker, for_date)
            except RateLimitError:
                logger.warning("Alpha Vantage rate limited for %s on %s — falling back to Finnhub", ticker, for_date)
            except Exception as e:
                logger.warning("Alpha Vantage failed for %s on %s: %s — falling back", ticker, for_date, e)

        if self.finnhub_key:
            try:
                return self._fetch_finnhub(ticker, for_date)
            except Exception as e:
                logger.warning("Finnhub failed for %s on %s: %s", ticker, for_date, e)

        return []

    def _fetch_alpha_vantage(self, ticker: str, for_date: date) -> list[Headline]:
        """Fetch from Alpha Vantage NEWS_SENTIMENT endpoint."""
        self._throttle_av()

        time_from = for_date.strftime("%Y%m%dT0000")
        time_to = for_date.strftime("%Y%m%dT2359")
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT&tickers={ticker}"
            f"&time_from={time_from}&time_to={time_to}"
            f"&limit=50&apikey={self.alpha_vantage_key}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "Note" in data or "Information" in data:
            raise RateLimitError("Alpha Vantage rate limit or key issue")

        feed = data.get("feed", [])
        headlines = []
        for item in feed:
            # Filter to items that mention this specific ticker
            ticker_sentiments = {
                ts["ticker"]: ts.get("ticker_sentiment_score")
                for ts in item.get("ticker_sentiment", [])
            }
            if ticker not in ticker_sentiments and item.get("feed"):
                continue

            pub_date = item.get("time_published", "")[:8]  # "YYYYMMDD"
            if len(pub_date) == 8:
                iso_date = f"{pub_date[:4]}-{pub_date[4:6]}-{pub_date[6:8]}"
            else:
                iso_date = str(for_date)

            headlines.append(Headline(
                text=item.get("title", ""),
                date=iso_date,
                source=item.get("source", "alpha_vantage"),
                raw_sentiment=ticker_sentiments.get(ticker),
            ))

        return headlines

    def _fetch_finnhub(self, ticker: str, for_date: date) -> list[Headline]:
        """Fetch from Finnhub /company-news endpoint."""
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={for_date}&to={for_date}"
            f"&token={self.finnhub_key}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        items = resp.json()

        if not isinstance(items, list):
            return []

        return [
            Headline(
                text=item.get("headline", ""),
                date=str(pd.Timestamp(item.get("datetime", 0), unit="s").date()),
                source=item.get("source", "finnhub"),
                raw_sentiment=None,
            )
            for item in items
            if item.get("headline")
        ]

    def _throttle_av(self) -> None:
        """Enforce 5 calls/minute rate limit for Alpha Vantage."""
        elapsed = time.monotonic() - self._last_av_call
        if elapsed < self._av_min_interval:
            time.sleep(self._av_min_interval - elapsed)
        self._last_av_call = time.monotonic()

    # -------------------------------------------------------------------------
    # Cache
    # -------------------------------------------------------------------------

    def _cache_path(self, ticker: str, for_date: date) -> Path:
        return self.cache_root / ticker / f"{for_date}.json"

    def _load_cache(self, ticker: str, for_date: date) -> Optional[list[Headline]]:
        path = self._cache_path(ticker, for_date)
        if not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > CACHE_TTL_HOURS and for_date >= date.today():
            # Stale cache for today — re-fetch (but keep stale cache for past dates)
            return None
        with open(path) as f:
            raw = json.load(f)
        return [Headline(**item) for item in raw]

    def _save_cache(self, ticker: str, for_date: date, headlines: list[Headline]) -> None:
        path = self._cache_path(ticker, for_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([h.__dict__ for h in headlines], f, indent=2)


class RateLimitError(Exception):
    pass
```

## Step 2: Wire into FinBERTEnricher

Replace the `NotImplementedError` stub in `FinBERTEnricher._load_recent_headlines()`:

```python
# src/plugins/enrichers/finbert.py

from src.data.news_data import NewsDataProvider, Headline
from datetime import date, timedelta

class FinBERTEnricher(DataEnricher):

    def __init__(self, ..., news_provider: Optional[NewsDataProvider] = None):
        # ... existing init ...
        self.news_provider = news_provider  # injectable for testing

    def _load_recent_headlines(self, ticker: str, days_back: int) -> list[dict]:
        if self.news_provider is None:
            return []  # Phase 1 behavior: no news provider → zero sentiment

        end = date.today()
        start = end - timedelta(days=days_back)
        headlines: list[Headline] = self.news_provider.get_headlines(ticker, start, end)

        return [
            {"text": h.text, "timestamp": pd.Timestamp(h.date)}
            for h in headlines
            if h.text.strip()
        ]
```

## Step 3: Add API keys to .env

```bash
# .env
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
```

Update `.env.example` to document both keys.

## Step 4: Write unit tests with recorded fixtures

Use pre-recorded API responses to test both providers without network calls:

```python
# tests/data/test_news_data.py
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.data.news_data import NewsDataProvider
from datetime import date

FIXTURE_DIR = Path("tests/fixtures/news")

@pytest.fixture
def av_success_response():
    return {
        "feed": [
            {
                "title": "Apple reports record iPhone sales",
                "time_published": "20240115T093000",
                "source": "Reuters",
                "ticker_sentiment": [
                    {"ticker": "AAPL", "ticker_sentiment_score": "0.35"}
                ],
            }
        ]
    }

@pytest.fixture
def av_rate_limit_response():
    return {"Note": "Thank you for using Alpha Vantage! API call frequency is 5 calls per minute."}

@pytest.fixture
def finnhub_success_response():
    return [
        {
            "headline": "Apple unveils new AI features",
            "datetime": 1705312200,  # 2024-01-15
            "source": "Bloomberg",
        }
    ]

def test_av_success(av_success_response, tmp_path):
    provider = NewsDataProvider(alpha_vantage_key="test", cache_root=tmp_path)
    provider._av_min_interval = 0  # disable throttle in tests

    mock_resp = MagicMock()
    mock_resp.json.return_value = av_success_response
    mock_resp.raise_for_status = lambda: None

    with patch("requests.get", return_value=mock_resp):
        headlines = provider.get_headlines("AAPL", start=date(2024, 1, 15), end=date(2024, 1, 15))

    assert len(headlines) == 1
    assert "iPhone" in headlines[0].text
    assert headlines[0].raw_sentiment == "0.35"

def test_av_rate_limit_falls_back_to_finnhub(av_rate_limit_response, finnhub_success_response, tmp_path):
    provider = NewsDataProvider(
        alpha_vantage_key="test",
        finnhub_key="test",
        cache_root=tmp_path,
    )
    provider._av_min_interval = 0

    responses = [
        MagicMock(json=lambda: av_rate_limit_response, raise_for_status=lambda: None),
        MagicMock(json=lambda: finnhub_success_response, raise_for_status=lambda: None),
    ]

    with patch("requests.get", side_effect=responses):
        headlines = provider.get_headlines("AAPL", start=date(2024, 1, 15), end=date(2024, 1, 15))

    assert len(headlines) == 1
    assert headlines[0].source == "Bloomberg"

def test_cache_prevents_second_api_call(av_success_response, tmp_path):
    provider = NewsDataProvider(alpha_vantage_key="test", cache_root=tmp_path)
    provider._av_min_interval = 0

    mock_resp = MagicMock(json=lambda: av_success_response, raise_for_status=lambda: None)

    with patch("requests.get", return_value=mock_resp) as mock_get:
        provider.get_headlines("AAPL", start=date(2024, 1, 15), end=date(2024, 1, 15))
        provider.get_headlines("AAPL", start=date(2024, 1, 15), end=date(2024, 1, 15))

    # Second call should hit cache — only 1 API call total
    assert mock_get.call_count == 1

def test_both_keys_missing_returns_empty(tmp_path):
    provider = NewsDataProvider(cache_root=tmp_path)
    headlines = provider.get_headlines("AAPL")
    assert headlines == []
```

## Rate Limit Strategy

Alpha Vantage free tier: 25 requests/day, 5/minute.

| Scenario | Strategy |
|---|---|
| 25 tickers, 7-day lookback | 25 × 7 = 175 calls needed. Must use cache aggressively — past dates never change so they cache permanently. |
| Per-day cache | Past dates: permanent (never re-fetched). Today: 12-hour TTL (re-fetch once per run). |
| Rate limit hit | Fall back to Finnhub (60/min — much more generous) |
| Both keys missing | Return `[]` — FinBERTEnricher emits zero features (Phase 1 behavior preserved) |

For a 50-ticker watchlist with daily runs:
- First run: up to 50 API calls (1 per ticker for today) — well within limits
- Subsequent runs: all cache hits (past dates), 1 fresh call per ticker for today

## Validation Checklist

- [ ] `get_headlines("AAPL", start, end)` returns non-empty list with real keys
- [ ] Rate-limit response from Alpha Vantage triggers Finnhub fallback (not an exception)
- [ ] Requesting the same (ticker, date) twice only calls the API once (cache hit)
- [ ] Past dates (before today) cache permanently (TTL check is skipped)
- [ ] Today's cache is refreshed if older than 12 hours
- [ ] Missing API keys return `[]` without raising
- [ ] `FinBERTEnricher` produces non-zero sentiment features when provider returns headlines
- [ ] Unit tests use recorded fixtures — no network calls in CI
