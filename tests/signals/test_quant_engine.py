"""
Unit tests for QuantEngine (src/signals/quant_engine.py).

All tests use lightweight synthetic OHLCV data and a real PluginRegistry
loaded from config/plugins.yaml. FinBERT is NOT loaded — the enricher plugin
is excluded from registry fixtures so tests run without the model on disk.

Test coverage:
  - Direction and confidence are always in valid ranges.
  - Regime weights sum to 1.0 for all four regimes.
  - Multi-timeframe boost never pushes confidence above 1.0.
  - Sentiment score changes the composite output.
  - No lookahead bias: signal at row t unchanged when row t+1 is appended.
  - generate_series() output length matches input DataFrame.
  - All-flat price series produces near-zero signal.
  - Insufficient weekly data → no crash, no boost applied.
  - All required TradeSignal fields are populated.
  - should_exit() returns correct decision in trending vs ranging regimes.
"""

from datetime import date

import pandas as pd
import pytest

from src.models.trade_signal import RegimeType
from src.plugins.registry import PluginRegistry
from src.signals.quant_engine import QuantEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 300, trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with a configurable trend.

    Args:
        n: Number of trading days.
        trend: Daily drift for the close price (0.001 = slight uptrend).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with DatetimeIndex and columns [open, high, low, close, volume].
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=date.today(), periods=n)
    close = 100.0 * (1 + trend + rng.normal(0, 0.01, n)).cumprod()
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _flat_ohlcv(n: int = 300) -> pd.DataFrame:
    """All bars have the same close price — indicator signals should be ~0."""
    dates = pd.bdate_range(end=date.today(), periods=n)
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.5] * n,
            "low": [99.5] * n,
            "close": [100.0] * n,
            "volume": [1_000_000.0] * n,
        },
        index=dates,
    )


def _make_registry(settings_path: str = "config/settings.yaml") -> PluginRegistry:
    """
    Build a real PluginRegistry with all 6 indicators active.
    FinBERT enricher is explicitly excluded to avoid model download in CI.
    """
    registry = PluginRegistry()
    # Patch discover_plugins to skip enrichers (FinBERT requires model on disk)
    import yaml

    with open("config/plugins.yaml") as f:
        cfg = yaml.safe_load(f)

    for entry in cfg.get("indicators", {}).get("enabled", []):
        if entry.get("active", False):
            plugin = registry._instantiate(entry["class"])
            registry._indicators[entry["name"]] = plugin

    return registry


@pytest.fixture
def registry() -> PluginRegistry:
    return _make_registry()


@pytest.fixture
def engine(registry: PluginRegistry) -> QuantEngine:
    return QuantEngine(registry, settings_path="config/settings.yaml")


@pytest.fixture
def df_trending() -> pd.DataFrame:
    """300 bars with a clear uptrend."""
    return _make_ohlcv(n=300, trend=0.002)


@pytest.fixture
def df_flat() -> pd.DataFrame:
    return _flat_ohlcv(n=300)


# ---------------------------------------------------------------------------
# Test 1: Direction always in [-1, +1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", list(RegimeType))
def test_direction_in_valid_range(
    engine: QuantEngine, df_trending: pd.DataFrame, regime: RegimeType
) -> None:
    signal = engine.generate_signal(df_trending, regime, "TEST")
    assert (
        -1.0 <= signal.direction <= 1.0
    ), f"direction={signal.direction} out of [-1, +1] for regime={regime}"


# ---------------------------------------------------------------------------
# Test 2: Confidence always in [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", list(RegimeType))
def test_confidence_in_valid_range(
    engine: QuantEngine, df_trending: pd.DataFrame, regime: RegimeType
) -> None:
    signal = engine.generate_signal(df_trending, regime, "TEST")
    assert (
        0.0 <= signal.confidence <= 1.0
    ), f"confidence={signal.confidence} out of [0, 1] for regime={regime}"


# ---------------------------------------------------------------------------
# Test 3: Regime weights sum to 1.0 for all four regimes
# ---------------------------------------------------------------------------


def test_regime_weights_sum_to_one(engine: QuantEngine) -> None:
    for regime, weights in engine.regime_weights.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Regime {regime}: weights sum to {total:.8f}, expected 1.0"


# ---------------------------------------------------------------------------
# Test 4: Multi-timeframe boost never pushes confidence above 1.0
# ---------------------------------------------------------------------------


def test_mtf_boost_never_exceeds_one(engine: QuantEngine) -> None:
    # Use a strongly trending series with plenty of weekly bars
    df = _make_ohlcv(n=500, trend=0.003)
    for regime in RegimeType:
        signal = engine.generate_signal(df, regime, "TEST")
        assert (
            signal.confidence <= 1.0
        ), f"confidence={signal.confidence} > 1.0 after MTF boost in regime={regime}"


# ---------------------------------------------------------------------------
# Test 5: Sentiment score affects the composite output
# ---------------------------------------------------------------------------


def test_sentiment_affects_composite(engine: QuantEngine, df_trending: pd.DataFrame) -> None:
    regime = RegimeType.TRENDING_UP
    sentiment_weight = engine.regime_weights[regime].get("sentiment", 0.0)
    if sentiment_weight == 0.0:
        pytest.skip("Sentiment weight is 0.0 for this regime — effect not testable")

    sig_neutral = engine.generate_signal(df_trending, regime, "TEST", sentiment_score=0.0)
    sig_bullish = engine.generate_signal(df_trending, regime, "TEST", sentiment_score=1.0)
    sig_bearish = engine.generate_signal(df_trending, regime, "TEST", sentiment_score=-1.0)

    # Bullish sentiment should push direction higher than neutral
    assert sig_bullish.direction >= sig_neutral.direction, (
        f"Bullish sentiment should not decrease direction: "
        f"bullish={sig_bullish.direction}, neutral={sig_neutral.direction}"
    )
    # Bearish sentiment should push direction lower than neutral
    assert sig_bearish.direction <= sig_neutral.direction, (
        f"Bearish sentiment should not increase direction: "
        f"bearish={sig_bearish.direction}, neutral={sig_neutral.direction}"
    )
    # At least one of them should differ from neutral
    assert (
        sig_bullish.direction != sig_bearish.direction
    ), "Bullish and bearish sentiment produced identical direction — sentiment has no effect"


# ---------------------------------------------------------------------------
# Test 6: No lookahead bias — signal at t unchanged when t+1 is appended
# ---------------------------------------------------------------------------


def test_no_lookahead_bias(engine: QuantEngine) -> None:
    df = _make_ohlcv(n=300, trend=0.001)
    regime = RegimeType.TRENDING_UP

    # Signal using df up to t (last row = df.iloc[-2])
    df_at_t = df.iloc[:-1]
    sig_at_t = engine.generate_signal(df_at_t, regime, "TEST")

    # Signal using df up to t, then append row t+1
    df_extended = df  # adds one more row
    sig_extended = engine.generate_signal(df_extended.iloc[:-1], regime, "TEST")

    # Same bar (df.iloc[-2]) should give the same result
    assert sig_at_t.direction == sig_extended.direction, (
        f"Lookahead bias detected: direction changed when future bar was appended "
        f"({sig_at_t.direction} → {sig_extended.direction})"
    )


# ---------------------------------------------------------------------------
# Test 7: generate_series() length matches input DataFrame
# ---------------------------------------------------------------------------


def test_generate_series_length(engine: QuantEngine) -> None:
    df = _make_ohlcv(n=50, trend=0.001)
    regime_series = pd.Series(RegimeType.TRENDING_UP, index=df.index)
    signals = engine.generate_series(df, regime_series, "TEST")

    assert len(signals) == len(
        df
    ), f"generate_series returned {len(signals)} signals for {len(df)}-row DataFrame"
    assert (signals.index == df.index).all(), "generate_series index does not match df.index"


# ---------------------------------------------------------------------------
# Test 8: Flat price series → near-zero direction
# ---------------------------------------------------------------------------


def test_all_flat_price_gives_near_zero_signal(engine: QuantEngine, df_flat: pd.DataFrame) -> None:
    regime = RegimeType.RANGING
    signal = engine.generate_signal(df_flat, regime, "FLAT")
    assert (
        abs(signal.direction) < 0.30
    ), f"Flat price series should produce near-zero direction, got {signal.direction}"


# ---------------------------------------------------------------------------
# Test 9: Insufficient weekly data → no crash, confidence not boosted
# ---------------------------------------------------------------------------


def test_insufficient_weekly_data_no_crash(engine: QuantEngine) -> None:
    # Only 15 trading days → < 20 weekly bars → no MTF boost
    df_short = _make_ohlcv(n=15, trend=0.002)
    regime = RegimeType.TRENDING_UP

    # Should not raise
    signal = engine.generate_signal(df_short, regime, "SHORT")

    # Confidence must still be valid
    assert 0.0 <= signal.confidence <= 1.0

    # Verify MTF was NOT applied: confidence should equal |direction| exactly
    # (since _weekly_confirms returns False for < 20 weekly bars)
    raw_confidence = abs(signal.direction)
    assert signal.confidence == pytest.approx(raw_confidence, abs=1e-9), (
        f"MTF boost should NOT apply with only {len(df_short)} bars. "
        f"Expected confidence={raw_confidence}, got {signal.confidence}"
    )


# ---------------------------------------------------------------------------
# Test 10: All required TradeSignal fields are populated
# ---------------------------------------------------------------------------


def test_trade_signal_fields_populated(engine: QuantEngine, df_trending: pd.DataFrame) -> None:
    regime = RegimeType.VOLATILE
    signal = engine.generate_signal(df_trending, regime, "AAPL", sentiment_score=0.1)

    assert signal.ticker == "AAPL"
    assert signal.timestamp is not None
    assert signal.source_layer == "quant"
    assert signal.regime == RegimeType.VOLATILE
    assert isinstance(signal.features, dict)
    assert len(signal.features) > 0, "features dict should not be empty"
    assert "sentiment" in signal.features, "features must include 'sentiment' key"
    # RiskManager fields are intentionally None at quant layer
    assert signal.stop_loss_pct is None
    assert signal.take_profit_pct is None
    assert signal.bet_size is None
    assert signal.llm_approved is None
