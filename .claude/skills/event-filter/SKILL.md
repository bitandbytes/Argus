---
name: event-filter
description: "Use this skill when implementing or modifying event-driven exit logic (Phase 3.3). Triggers on: 'add an EventFilter', 'earnings blackout exit', 'news shock exit', 'ATR stop filter', 'implement EarningsBlackoutFilter', 'implement NewsShockFilter', 'implement AtrStopFilter', 'add exit_reason to TradeSignal', 'replace earnings calendar stub', or any task that evaluates open positions against discrete events to trigger exits. Do NOT use for ATR-based position sizing (use risk-manager) or for earnings data in fundamentals signals (use fundamentals-plugin)."
---

# Event Filter Skill

This skill guides Claude in implementing the `EventFilter` plugin type — Phase 3.3's discrete event-driven exit layer. Event filters run on open positions after each daily bar and decide whether to exit before the next open. They complement the confidence-based exit in `RiskManager` by catching categorical risks (upcoming earnings, news shocks, hard ATR breaches) that a smooth confidence score can miss.

## When to Use This Skill

- Adding the `EventFilter` ABC to `src/plugins/base.py`
- Implementing any filter in `src/risk/event_filter.py`
- Replacing the `MarketDataProvider.get_earnings_calendar()` stub with a real Alpha Vantage client
- Adding the `exits:` config block to `config/settings.yaml`
- Adding `exit_reason` to `TradeSignal`
- Writing unit tests for exit filters against synthetic inputs

Do NOT use this skill for:
- ATR-based position sizing at entry → use `risk-manager`
- Earnings data as a valuation signal → use `fundamentals-plugin`
- Technical indicator plugins → use `plugin-author`

## Architecture Context

Event filters sit between the signal generation layers and order execution (see ADR-0012). They are evaluated in `RiskManager.evaluate_exits()` after the daily bar closes:

```
Open position + today's bar
        │
        ▼
 EarningsBlackoutFilter.should_exit() ──► exit("earnings_blackout")
        │ (if no exit)
        ▼
 NewsShockFilter.should_exit()        ──► exit("news_shock")
        │ (if no exit)
        ▼
 AtrStopFilter.should_exit()          ──► exit("atr_stop")
        │ (if no exit)
        ▼
 RiskManager confidence exit          ──► exit("low_confidence") or hold
```

First filter to return `True` wins. `exit_reason` is logged and surfaced in Telegram alerts.

## Step 1: Add EventFilter ABC to base.py

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class PositionContext:
    """Information about an open position passed to each EventFilter."""
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    current_price: float
    direction: int          # +1 long, -1 short
    shares: float
    unrealized_pnl_pct: float

@dataclass
class BarContext:
    """Single daily bar plus derived features."""
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    atr_14: float           # pre-computed ATR(14)
    sentiment_score: float  # from FinBERT enricher
    news_volume_ratio: float

class EventFilter(ABC):
    """Plugin that evaluates an open position and decides whether to exit.

    Unlike SignalFilter (which transforms entry signals), EventFilter
    is evaluated after a bar closes against currently open positions.
    All registered EventFilters are evaluated in order; the first
    that returns (True, reason) triggers an exit.

    Class attributes:
        name: Unique identifier matching config/settings.yaml::exits key.
        enabled_by_default: Whether the filter is active without explicit config.
    """

    name: str = ""
    enabled_by_default: bool = True

    @abstractmethod
    def should_exit(
        self,
        position: PositionContext,
        bar: BarContext,
        context: dict,
    ) -> tuple[bool, Optional[str]]:
        """Evaluate whether to exit the position.

        Args:
            position: Current position metadata.
            bar: Today's completed bar with pre-computed features.
            context: Extra data dict — may include earnings_calendar,
                     upcoming_events, etc. Keys depend on the filter.

        Returns:
            (should_exit: bool, reason: Optional[str])
            reason is a short slug used in logs/alerts, e.g. "earnings_blackout".
            Return (False, None) to hold.
        """
```

## Step 2: Register EventFilter in PluginRegistry

```python
# src/plugins/registry.py
from src.plugins.base import EventFilter

class PluginRegistry:
    def get_all_event_filters(self) -> list[EventFilter]:
        return list(self._event_filters)
```

Add `exits:` block to `config/plugins.yaml` (or load directly from `config/settings.yaml`):

```yaml
exits:
  earnings_blackout:
    enabled: true
    class: "src.risk.event_filter.EarningsBlackoutFilter"
    pre_earnings_days: 2
    post_earnings_days: 1
  news_shock:
    enabled: true
    class: "src.risk.event_filter.NewsShockFilter"
    sentiment_threshold: 0.75
    volume_ratio_threshold: 3.0
  atr_stop:
    enabled: true
    class: "src.risk.event_filter.AtrStopFilter"
    atr_stop_multiple: 2.0
    atr_tp_multiple: 4.0
```

## Step 3: Implement the three filters

Create `src/risk/event_filter.py`:

```python
import pandas as pd
from typing import Optional
from src.plugins.base import EventFilter, PositionContext, BarContext

class EarningsBlackoutFilter(EventFilter):
    """Exit T-2 before and T+1 after earnings to avoid gap risk.

    Reads earnings dates from context["earnings_calendar"] — a dict of
    {ticker: list[pd.Timestamp]} loaded by the orchestrator.

    +1: Exit T-2 before earnings (pre-blackout)
    +2: Hold through earnings if already in blackout
    -1: Exit T+1 after earnings (post-blackout)
    """
    name = "earnings_blackout"

    def __init__(self, pre_earnings_days: int = 2, post_earnings_days: int = 1):
        self.pre_days = pre_earnings_days
        self.post_days = post_earnings_days

    def should_exit(
        self,
        position: PositionContext,
        bar: BarContext,
        context: dict,
    ) -> tuple[bool, Optional[str]]:
        earnings_dates: list[pd.Timestamp] = (
            context.get("earnings_calendar", {}).get(position.ticker, [])
        )
        for earnings_dt in earnings_dates:
            days_to = (earnings_dt - bar.date).days
            days_since = (bar.date - earnings_dt).days
            if 0 < days_to <= self.pre_days:
                return True, f"earnings_blackout_pre_{days_to}d"
            if 0 <= days_since <= self.post_days:
                return True, f"earnings_blackout_post_{days_since}d"
        return False, None


class NewsShockFilter(EventFilter):
    """Exit on extreme sentiment swing with abnormal news volume.

    Condition: |sentiment_score| > threshold AND news_volume_ratio > ratio_threshold.
    Both conditions must be true — sentiment alone can be noisy.
    """
    name = "news_shock"

    def __init__(self, sentiment_threshold: float = 0.75, volume_ratio_threshold: float = 3.0):
        self.sentiment_threshold = sentiment_threshold
        self.volume_ratio_threshold = volume_ratio_threshold

    def should_exit(
        self,
        position: PositionContext,
        bar: BarContext,
        context: dict,
    ) -> tuple[bool, Optional[str]]:
        sentiment_extreme = abs(bar.sentiment_score) > self.sentiment_threshold
        volume_spike = bar.news_volume_ratio > self.volume_ratio_threshold

        if sentiment_extreme and volume_spike:
            direction = "negative" if bar.sentiment_score < 0 else "positive"
            return True, f"news_shock_{direction}"
        return False, None


class AtrStopFilter(EventFilter):
    """Hard stop-loss and take-profit based on ATR(14).

    Checks whether today's bar violated the ATR-based hard stop or TP
    set at entry. These are floor/ceiling limits independent of the
    confidence-based soft exit in RiskManager.

    Stop:  entry_price - atr_stop_multiple × ATR_14  (for longs)
    TP:    entry_price + atr_tp_multiple  × ATR_14  (for longs)
    """
    name = "atr_stop"

    def __init__(self, atr_stop_multiple: float = 2.0, atr_tp_multiple: float = 4.0):
        self.stop_multiple = atr_stop_multiple
        self.tp_multiple = atr_tp_multiple

    def should_exit(
        self,
        position: PositionContext,
        bar: BarContext,
        context: dict,
    ) -> tuple[bool, Optional[str]]:
        atr = bar.atr_14
        if atr <= 0:
            return False, None

        d = position.direction
        stop_price = position.entry_price - d * self.stop_multiple * atr
        tp_price = position.entry_price + d * self.tp_multiple * atr

        # Use intraday low/high to check if barrier was breached during the bar
        if d == 1:  # long
            if bar.low <= stop_price:
                return True, "atr_stop_loss"
            if bar.high >= tp_price:
                return True, "atr_take_profit"
        elif d == -1:  # short
            if bar.high >= stop_price:
                return True, "atr_stop_loss"
            if bar.low <= tp_price:
                return True, "atr_take_profit"

        return False, None
```

## Step 4: Replace earnings calendar stub

The Phase 1 `MarketDataProvider.get_earnings_calendar()` returns `[]`. Replace it:

```python
# src/data/market_data.py
import requests
from pathlib import Path
import pandas as pd

EARNINGS_CACHE = Path("data/features/events/earnings_calendar.parquet")

def get_earnings_calendar(
    self,
    tickers: list[str],
    horizon_days: int = 30,
) -> dict[str, list[pd.Timestamp]]:
    """Fetch upcoming earnings dates for a list of tickers.

    Uses Alpha Vantage EARNINGS_CALENDAR endpoint.
    Falls back to yfinance .calendar property.
    Cache: data/features/events/earnings_calendar.parquet (24-hour TTL).
    """
    if self._earnings_cache_fresh():
        return self._load_earnings_cache(tickers)

    results: dict[str, list[pd.Timestamp]] = {}
    for ticker in tickers:
        try:
            dates = self._fetch_av_earnings(ticker)
        except Exception:
            dates = self._fetch_yf_earnings(ticker)
        results[ticker] = [d for d in dates if d >= pd.Timestamp.today()]

    self._save_earnings_cache(results)
    return results

def _earnings_cache_fresh(self) -> bool:
    if not EARNINGS_CACHE.exists():
        return False
    age = pd.Timestamp.now() - pd.Timestamp(EARNINGS_CACHE.stat().st_mtime, unit="s")
    return age.total_seconds() < 86400  # 24 hours

def _fetch_av_earnings(self, ticker: str) -> list[pd.Timestamp]:
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=EARNINGS_CALENDAR&symbol={ticker}&horizon=3month"
        f"&apikey={self.alpha_vantage_key}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    # Response is CSV; parse reportDate column
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    return pd.to_datetime(df["reportDate"]).tolist()
```

## Step 5: Add exit_reason to TradeSignal

```python
# src/models/trade_signal.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TradeSignal:
    # ... existing fields ...
    exit_reason: Optional[str] = None  # e.g. "earnings_blackout_pre_2d", "atr_stop_loss"
```

## Step 6: Wire into RiskManager.evaluate_exits()

```python
# src/risk/risk_manager.py
def evaluate_exits(
    self,
    open_positions: list[PositionContext],
    bar: BarContext,
    context: dict,
) -> list[tuple[PositionContext, str]]:
    """Return list of (position, exit_reason) for positions that should exit."""
    exits = []
    filters = self.registry.get_all_event_filters()

    for position in open_positions:
        for f in filters:
            should, reason = f.should_exit(position, bar, context)
            if should:
                exits.append((position, reason))
                break  # first filter wins; don't evaluate remaining

    return exits
```

## Unit Tests

```python
# tests/risk/test_event_filter.py
import pytest
import pandas as pd
from src.risk.event_filter import EarningsBlackoutFilter, NewsShockFilter, AtrStopFilter
from src.plugins.base import PositionContext, BarContext

@pytest.fixture
def long_position():
    return PositionContext(
        ticker="AAPL", entry_date=pd.Timestamp("2024-01-01"),
        entry_price=150.0, current_price=155.0,
        direction=1, shares=10, unrealized_pnl_pct=0.033,
    )

@pytest.fixture
def neutral_bar():
    return BarContext(
        date=pd.Timestamp("2024-01-15"), open=155.0, high=157.0,
        low=153.0, close=156.0, volume=50_000_000,
        atr_14=3.0, sentiment_score=0.1, news_volume_ratio=1.0,
    )

def test_earnings_blackout_triggers_2d_before(long_position, neutral_bar):
    f = EarningsBlackoutFilter(pre_earnings_days=2)
    context = {"earnings_calendar": {"AAPL": [pd.Timestamp("2024-01-17")]}}
    should_exit, reason = f.should_exit(long_position, neutral_bar, context)
    assert should_exit
    assert "pre" in reason

def test_earnings_blackout_no_trigger_outside_window(long_position, neutral_bar):
    f = EarningsBlackoutFilter(pre_earnings_days=2)
    context = {"earnings_calendar": {"AAPL": [pd.Timestamp("2024-01-25")]}}
    should_exit, _ = f.should_exit(long_position, neutral_bar, context)
    assert not should_exit

def test_news_shock_requires_both_conditions(long_position, neutral_bar):
    f = NewsShockFilter(sentiment_threshold=0.75, volume_ratio_threshold=3.0)
    # Only sentiment extreme — no exit
    bar = neutral_bar
    bar_high_sent = BarContext(**{**bar.__dict__, "sentiment_score": -0.9, "news_volume_ratio": 1.0})
    should_exit, _ = f.should_exit(long_position, bar_high_sent, {})
    assert not should_exit

    # Both conditions — exit
    bar_shock = BarContext(**{**bar.__dict__, "sentiment_score": -0.9, "news_volume_ratio": 4.0})
    should_exit, reason = f.should_exit(long_position, bar_shock, {})
    assert should_exit
    assert "news_shock" in reason

def test_atr_stop_loss_long(long_position, neutral_bar):
    f = AtrStopFilter(atr_stop_multiple=2.0)
    # entry 150, ATR 3, stop = 150 - 2*3 = 144. Bar low = 143 → triggered
    bar_hit = BarContext(**{**neutral_bar.__dict__, "low": 143.0})
    should_exit, reason = f.should_exit(long_position, bar_hit, {})
    assert should_exit
    assert reason == "atr_stop_loss"

def test_atr_stop_no_trigger_within_range(long_position, neutral_bar):
    f = AtrStopFilter(atr_stop_multiple=2.0)
    # entry 150, stop = 144, bar low = 145 → no exit
    should_exit, _ = f.should_exit(long_position, neutral_bar, {})
    assert not should_exit
```

## Validation Checklist

- [ ] `EventFilter` ABC in `base.py` with correct `should_exit` signature
- [ ] `PluginRegistry.get_all_event_filters()` returns filters in config order
- [ ] `EarningsBlackoutFilter` exits on T-2 and T+1, holds on T-3 and T+2
- [ ] `NewsShockFilter` requires both extreme sentiment AND volume spike
- [ ] `AtrStopFilter` checks intraday low (not close) against stop for longs
- [ ] `exit_reason` field on `TradeSignal` is populated by `RiskManager.evaluate_exits()`
- [ ] Earnings calendar cache is refreshed daily; returns empty list on API failure
- [ ] All three filters tested with synthetic data covering True/False branches
