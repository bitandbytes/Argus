"""
News data provider — Phase 1 stub.

Returns empty results for all methods. The interface is designed to match
what FinBERTEnricher will expect in Phase 2/3 when a real provider
(Alpha Vantage, Finnhub) is wired up.

Expected headline dict schema (for future implementors):
    {
        "ticker":       str,          # e.g. "AAPL"
        "headline":     str,          # raw headline text
        "published_at": datetime,     # UTC publication timestamp
        "source":       str,          # news source name
        "url":          str,          # article URL
    }
"""

import logging
from datetime import datetime  # noqa: F401  (kept for type-hint reference in docstrings)
from typing import Optional

logger = logging.getLogger(__name__)


class NewsDataProvider:
    """
    Fetches news headlines for use with FinBERT sentiment scoring.

    **Phase 1 stub:** both methods return ``[]``. No network calls are made.
    Phase 2/3 will replace the method bodies with Alpha Vantage or Finnhub
    calls; the interface (method names, argument names, return type) will
    remain identical so callers need no changes.

    Args:
        api_key: API key for the underlying news provider (unused in Phase 1).
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key
        if api_key is None:
            logger.debug("NewsDataProvider initialised without an API key — stub mode (Phase 1)")

    def get_headlines(self, ticker: str, days_back: int = 7) -> list[dict]:
        """
        Return recent headlines for *ticker*.

        Phase 1 stub: always returns ``[]``.

        Args:
            ticker: Ticker symbol (e.g. ``"AAPL"``).
            days_back: How many calendar days of history to retrieve.

        Returns:
            List of headline dicts with keys
            ``ticker, headline, published_at, source, url``.
            Returns ``[]`` in Phase 1.
        """
        logger.debug(
            "NewsDataProvider.get_headlines('%s', days_back=%d) — stub, returning []",
            ticker,
            days_back,
        )
        return []

    def get_macro_news(self, days_back: int = 7) -> list[dict]:
        """
        Return recent macro-level news (not ticker-specific).

        Phase 1 stub: always returns ``[]``.

        Args:
            days_back: How many calendar days of history to retrieve.

        Returns:
            List of headline dicts with keys
            ``ticker, headline, published_at, source, url``.
            Returns ``[]`` in Phase 1.
        """
        logger.debug(
            "NewsDataProvider.get_macro_news(days_back=%d) — stub, returning []",
            days_back,
        )
        return []
