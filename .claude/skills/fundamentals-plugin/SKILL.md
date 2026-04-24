---
name: fundamentals-plugin
description: "Use this skill when implementing or modifying the fundamentals signal pillar (Phase 3.2). Triggers on: 'add fundamentals', 'implement FundamentalsDataProvider', 'create a FundamentalIndicatorPlugin', 'add P/E z-score', 'add FCF yield', 'add earnings growth', 'add earnings surprise', 'wire fundamentals into the quant engine', 'ETF fundamental exclusion', or any task that fetches balance-sheet / income-statement / cash-flow data and turns it into a normalized signal. Do NOT use for technical indicators (use plugin-author) or for the earnings blackout exit logic (use event-filter)."
---

# Fundamentals Plugin Skill

This skill guides Claude in building the fundamentals signal pillar — Layer 2's third input channel alongside technicals and sentiment. Fundamentals are implemented as a distinct plugin type (`FundamentalIndicatorPlugin`) so they can be tuned, weighted per regime, and excluded for ETFs without touching core pipeline code.

## When to Use This Skill

- Implementing `FundamentalsDataProvider` in `src/data/fundamentals_data.py`
- Adding the `FundamentalIndicatorPlugin` abstract base to `src/plugins/base.py`
- Writing any plugin under `src/plugins/fundamentals/`
- Wiring the `fundamental_score` channel into `QuantEngine._weighted_composite`
- Configuring ETF exclusion or per-stock `fundamentals_weight_override`
- Exposing `fundamentals_weight` for Bayesian tuning

Do NOT use this skill for:
- Technical indicator plugins → use `plugin-author`
- Earnings calendar fetching for exit logic → use `event-filter`
- Backtesting the fundamentals channel → use `backtest-runner`

## Architecture Context

The v1.2 architecture (ADR-0011) adds fundamentals as a **first-class signal pillar**. The QuantEngine composite becomes:

```
composite = Σ(tech_weight_i × tech_signal_i) + fund_weight × fundamental_score + sent_weight × sentiment_score
```

Where all weights still sum to 1.0 per regime, and `fundamental_score` is the mean of all enabled `FundamentalIndicatorPlugin` outputs. ETFs force `fund_weight = 0` and the remaining weights are re-normalized.

## Step 1: Add FundamentalIndicatorPlugin to base.py

Add this ABC to `src/plugins/base.py` alongside the existing four plugin types:

```python
from abc import ABC, abstractmethod
import pandas as pd

class FundamentalIndicatorPlugin(ABC):
    """Plugin that derives a normalized signal from financial statement data.

    Unlike IndicatorPlugin (which operates on OHLCV), this plugin receives
    a fundamentals DataFrame (quarterly/annual statements) alongside a price
    DataFrame, and returns a normalized signal in [-1, +1].

    Class attributes:
        name: Unique plugin identifier. Must match config/plugins.yaml key.
        category: One of "valuation", "quality", "growth", "surprise".
        version: SemVer string.
        applies_to_etfs: Set False (default). ETFs have no income statements.
    """

    name: str = ""
    category: str = ""
    version: str = "1.0.0"
    applies_to_etfs: bool = False  # hard ETF exclusion at registry level

    @abstractmethod
    def compute(self, fundamentals_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.Series:
        """Compute the raw fundamental signal aligned to the price_df index.

        Args:
            fundamentals_df: Quarterly or annual financial statements.
                             Columns vary by plugin; see FundamentalsDataProvider.
            price_df: OHLCV DataFrame with DatetimeIndex. Used for market-cap,
                      price normalization, and date alignment.

        Returns:
            pd.Series with the same DatetimeIndex as price_df. Values are raw
            (un-normalized). NaN is acceptable for periods before first report.
        """

    @abstractmethod
    def normalize(self, values: pd.Series) -> pd.Series:
        """Normalize the raw signal to [-1, +1].

        +1 = strong fundamental buy signal
        -1 = strong fundamental sell signal
         0 = neutral / no signal

        Must handle NaN gracefully (forward-fill then clamp, or return NaN).
        """

    @abstractmethod
    def get_tunable_params(self) -> dict:
        """Return ParamSpec entries that BayesianTuner can search.

        Fundamentals plugins typically expose lookback windows and z-score
        thresholds. fundamentals_weight itself is exposed at the QuantEngine
        level, not here.
        """

    def get_default_params(self) -> dict:
        """Return default parameter values."""
        return {}
```

## Step 2: Register the new type in PluginRegistry

In `src/plugins/registry.py`, add discovery and retrieval for `FundamentalIndicatorPlugin`:

```python
from src.plugins.base import FundamentalIndicatorPlugin

class PluginRegistry:
    def __init__(self):
        # existing lists ...
        self._fundamentals: list[FundamentalIndicatorPlugin] = []

    def get_all_fundamentals(self) -> list[FundamentalIndicatorPlugin]:
        return list(self._fundamentals)
```

Add a `fundamentals:` block to `config/plugins.yaml`:

```yaml
fundamentals:
  enabled:
    - name: "pe_zscore"
      class: "src.plugins.fundamentals.pe_zscore.PEZscorePlugin"
      active: true
    - name: "fcf_yield"
      class: "src.plugins.fundamentals.fcf_yield.FCFYieldPlugin"
      active: true
    - name: "earnings_growth"
      class: "src.plugins.fundamentals.earnings_growth.EarningsGrowthPlugin"
      active: true
    - name: "earnings_surprise"
      class: "src.plugins.fundamentals.earnings_surprise.EarningsSurprisePlugin"
      active: true
```

## Step 3: Implement FundamentalsDataProvider

Create `src/data/fundamentals_data.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

@dataclass
class FundamentalsDataProvider:
    """Fetches and caches quarterly/annual financial statements.

    Primary source: Alpha Vantage OVERVIEW, INCOME_STATEMENT, CASH_FLOW, EARNINGS.
    Fallback: yfinance .financials / .quarterly_financials.
    Cache: data/features/fundamentals/{ticker}/{fiscal_period}.parquet
    Refresh: monthly, or within 7 days of a scheduled earnings date.
    ETF short-circuit: returns empty DataFrame for tickers in etf_tickers set.
    """
    cache_root: Path = Path("data/features/fundamentals")
    alpha_vantage_key: Optional[str] = None
    etf_tickers: set[str] = field(default_factory=set)

    def get_statements(self, ticker: str) -> pd.DataFrame:
        """Return quarterly financial statements aligned to announcement dates.

        Returns an empty DataFrame for ETFs (no API call made).
        Columns: fiscal_date, report_date, revenue, net_income, eps,
                 eps_estimate, free_cash_flow, total_debt, shares_outstanding.
        """
        if ticker in self.etf_tickers:
            return pd.DataFrame()

        cached = self._load_cache(ticker)
        if cached is not None and not self._is_stale(ticker, cached):
            return cached

        try:
            data = self._fetch_alpha_vantage(ticker)
        except Exception as e:
            logger.warning("Alpha Vantage failed for %s: %s — falling back to yfinance", ticker, e)
            data = self._fetch_yfinance(ticker)

        if not data.empty:
            self._save_cache(ticker, data)
        return data

    def _fetch_alpha_vantage(self, ticker: str) -> pd.DataFrame:
        """Fetch INCOME_STATEMENT + CASH_FLOW + EARNINGS from Alpha Vantage."""
        if not self.alpha_vantage_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not set")
        # Implementation: requests to:
        #   https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={key}
        #   https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={key}
        #   https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={key}
        # Merge on fiscalDateEnding; parse quarterlyReports arrays.
        raise NotImplementedError

    def _fetch_yfinance(self, ticker: str) -> pd.DataFrame:
        """Fallback: yfinance quarterly financials."""
        t = yf.Ticker(ticker)
        fin = t.quarterly_financials.T  # rows = quarters
        cf = t.quarterly_cashflow.T
        # Align and return standardized columns
        # EPS estimate not available from yfinance — set to NaN
        raise NotImplementedError

    def _load_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        path = self.cache_root / ticker / "statements.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def _save_cache(self, ticker: str, df: pd.DataFrame) -> None:
        path = self.cache_root / ticker / "statements.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def _is_stale(self, ticker: str, cached: pd.DataFrame) -> bool:
        """Stale if >30 days old, or within 7 days of next earnings date."""
        path = self.cache_root / ticker / "statements.parquet"
        age_days = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days
        return age_days > 30  # TODO: also check proximity to earnings date
```

## Step 4: Implement the four initial plugins

All four live in `src/plugins/fundamentals/`. Each follows the same structure:

### pe_zscore.py — Trailing P/E vs sector 5-year z-score

```python
import pandas as pd
import numpy as np
from src.plugins.base import FundamentalIndicatorPlugin, ParamSpec

class PEZscorePlugin(FundamentalIndicatorPlugin):
    """P/E ratio normalized as a z-score vs its own 5-year history.

    +1: P/E is 2σ below its 5-year mean (cheap)
    -1: P/E is 2σ above its 5-year mean (expensive)
     0: P/E is at its historical mean
    """
    name = "pe_zscore"
    category = "valuation"
    version = "1.0.0"
    applies_to_etfs = False

    def compute(self, fundamentals_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.Series:
        params = self.get_default_params()
        if fundamentals_df.empty or "eps" not in fundamentals_df.columns:
            return pd.Series(np.nan, index=price_df.index)

        # Forward-fill EPS from quarterly reports to daily index
        eps_daily = fundamentals_df["eps"].reindex(price_df.index, method="ffill")
        pe = price_df["close"] / eps_daily.replace(0, np.nan)

        # Rolling z-score over lookback window
        window = params.get("zscore_window", 252 * 5)
        pe_mean = pe.rolling(window, min_periods=60).mean()
        pe_std = pe.rolling(window, min_periods=60).std()
        return (pe - pe_mean) / pe_std.replace(0, np.nan)

    def normalize(self, values: pd.Series) -> pd.Series:
        # z-score of -2 → +1 (cheap), z-score of +2 → -1 (expensive)
        return (-values / 2.0).clip(-1, 1).fillna(0.0)

    def get_tunable_params(self) -> dict:
        return {
            "zscore_window": ParamSpec(
                name="zscore_window", type="int",
                low=252, high=252 * 7, default=252 * 5,
                description="Rolling window (days) for P/E z-score baseline",
            ),
        }

    def get_default_params(self) -> dict:
        return {"zscore_window": 252 * 5}
```

### fcf_yield.py — Free cash flow / market cap

```python
class FCFYieldPlugin(FundamentalIndicatorPlugin):
    """FCF yield = trailing 12-month FCF / market cap.

    +1: FCF yield >= 8% (strong cash generation)
    -1: FCF yield <= 0% (burning cash)
     0: ~3% yield (market average)
    """
    name = "fcf_yield"
    category = "quality"
    # ... similar structure; normalize via sigmoid centered at 0.03
```

### earnings_growth.py — YoY EPS growth from last 4 quarters

```python
class EarningsGrowthPlugin(FundamentalIndicatorPlugin):
    """YoY EPS growth vs year-ago quarter.

    +1: >30% YoY growth
    -1: <-30% decline
     0: flat
    """
    name = "earnings_growth"
    category = "growth"
    # normalize via tanh(growth / 0.30) → maps ±30% to ±0.76
```

### earnings_surprise.py — Actual EPS vs consensus estimate

```python
class EarningsSurprisePlugin(FundamentalIndicatorPlugin):
    """(Actual EPS - Estimate EPS) / |Estimate EPS|.

    +1: beat by >10%
    -1: missed by >10%
     0: inline with estimates
    Note: this is a point-in-time signal; forward-fill between earnings dates.
    """
    name = "earnings_surprise"
    category = "surprise"
    # normalize via (surprise / 0.10).clip(-1, 1)
```

## Step 5: Wire fundamental_score into QuantEngine

In `src/signals/quant_engine.py`, add the fundamentals aggregation channel alongside the existing sentiment channel:

```python
def _compute_fundamental_score(
    self,
    ticker: str,
    fundamentals_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> float:
    """Mean of all enabled FundamentalIndicatorPlugin outputs for one bar."""
    plugins = self.registry.get_all_fundamentals()
    if not plugins:
        return 0.0

    scores = []
    for plugin in plugins:
        raw = plugin.compute(fundamentals_df, price_df)
        if raw.empty or raw.isna().all():
            continue
        latest_raw = raw.iloc[-1]
        if pd.isna(latest_raw):
            continue
        scores.append(plugin.normalize(pd.Series([latest_raw])).iloc[0])

    return float(np.mean(scores)) if scores else 0.0

def _weighted_composite(
    self,
    indicator_signals: dict[str, float],
    sentiment_score: float,
    fundamental_score: float,
    regime_weights: dict[str, float],
    f_fund: float,  # 0.0 for ETFs, 1.0 otherwise
) -> float:
    """Compute composite with re-normalization when fundamentals are excluded."""
    raw_fund_weight = regime_weights.get("fundamentals_weight", 0.0) * f_fund

    # Re-normalize technical + sentiment weights when fund weight is zeroed
    tech_sent_raw = {k: v for k, v in regime_weights.items() if k != "fundamentals_weight"}
    tech_sent_sum = sum(tech_sent_raw.values())
    scale = (1.0 - raw_fund_weight) / tech_sent_sum if tech_sent_sum > 0 else 1.0

    composite = raw_fund_weight * fundamental_score
    composite += regime_weights.get("sentiment_weight", 0.0) * scale * sentiment_score
    for name, signal in indicator_signals.items():
        w = regime_weights.get(name, 0.0) * scale
        composite += w * signal

    return float(np.clip(composite, -1.0, 1.0))
```

## Step 6: Configure regime weights

Add `fundamentals_weight` to `config/cluster_params/cluster_default.yaml`:

```yaml
regime_weights:
  TRENDING_UP:
    fundamentals_weight: 0.15
    sentiment_weight: 0.10
    # ... existing indicator weights sum to 0.75
  TRENDING_DOWN:
    fundamentals_weight: 0.10
    sentiment_weight: 0.10
    # ...
  RANGING:
    fundamentals_weight: 0.20
    sentiment_weight: 0.10
    # ...
  VOLATILE:
    fundamentals_weight: 0.05
    sentiment_weight: 0.10
    # ...
```

Expose `fundamentals_weight` as tunable in the QuantEngine or in each regime weight config so BayesianTuner can search `[0.0, 0.30]`.

## ETF Exclusion Rules

1. Load the `etfs:` list from `config/watchlist.yaml` in the orchestrator.
2. Pass the set to `FundamentalsDataProvider(etf_tickers=etf_set)` — the provider returns `pd.DataFrame()` with no API call.
3. In QuantEngine, set `f_fund = 0.0` for ETF tickers; `f_fund = 1.0` for stocks.
4. `_weighted_composite` re-normalizes the remaining weights automatically.
5. Validate: `fundamental_score` must be exactly `0.0` for all ETF bars — add an assertion in tests.

Optional: a per-stock override in `watchlist.yaml`:
```yaml
stocks:
  AAPL:
    fundamentals_weight_override: 0.0   # treat like an ETF
```

## Anti-Lookahead Rules (Critical)

Fundamentals have a **reporting lag** of 30–60 days after the fiscal quarter ends. Ignoring this causes severe lookahead bias.

- **Use `report_date` (when the 10-Q was filed), not `fiscal_date`** when aligning to the price index.
- `pd.Series.reindex(price_df.index, method="ffill")` is correct — it forward-fills the most recent FILED report, not the most recent quarter.
- Never use a fiscal quarter that ended after the bar date.
- Test: run `FundamentalsDataProvider.get_statements("AAPL")`, verify the first non-NaN date in each series is ≥ 45 days after the corresponding fiscal quarter end.

## Unit Tests

```python
# tests/plugins/fundamentals/test_pe_zscore.py
def test_etf_returns_zero_score(monkeypatch):
    """ETF tickers must produce fundamental_score == 0.0."""
    provider = FundamentalsDataProvider(etf_tickers={"ITA"})
    assert provider.get_statements("ITA").empty

def test_pe_zscore_cheap_stock():
    plugin = PEZscorePlugin()
    # Construct a scenario where P/E is 2σ below its own mean
    pe = pd.Series([20.0] * 252 * 5)
    pe.iloc[-1] = 10.0  # suddenly cheap
    raw = pd.Series((pe - pe.mean()) / pe.std())
    normalized = plugin.normalize(raw)
    assert normalized.iloc[-1] > 0.5  # cheap → positive signal

def test_no_lookahead_bias():
    """report_date-aligned ffill must not expose future quarters."""
    # ...
```

## Validation Checklist

- [ ] `FundamentalIndicatorPlugin` ABC in `base.py` with all abstract methods
- [ ] `PluginRegistry.get_all_fundamentals()` returns all enabled plugins
- [ ] `FundamentalsDataProvider.get_statements("ITA")` returns `pd.DataFrame()` immediately
- [ ] `FundamentalsDataProvider.get_statements("AAPL")` returns non-empty DataFrame with Alpha Vantage or yfinance fallback
- [ ] `PEZscorePlugin.normalize()` maps expensive → negative, cheap → positive
- [ ] `QuantEngine._weighted_composite()` with `f_fund=0` produces same result as weight re-normalized without fundamentals
- [ ] Backtest A/B: fundamentals-on vs fundamentals-off Sharpe difference < 10% (overfitting guard)
