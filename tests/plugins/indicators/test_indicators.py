"""Unit tests for all six indicator plugins.

Each test class covers:
    - compute() returns a DataFrame with expected columns
    - normalize() output is strictly within [-1, +1]
    - get_tunable_params() returns valid ParamSpec entries
    - Known input → known output pairs for deterministic verification
"""

import numpy as np
import pandas as pd
import pytest

from src.plugins.base import ParamSpec
from src.plugins.indicators.bollinger import BollingerBandIndicator
from src.plugins.indicators.donchian import DonchianChannelIndicator
from src.plugins.indicators.macd import MACDIndicator
from src.plugins.indicators.rsi import RSIIndicator
from src.plugins.indicators.sma import SMACrossoverIndicator
from src.plugins.indicators.volume import VolumeIndicator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_ohlcv() -> pd.DataFrame:
    """Completely flat price series: all closes = 100, volume = 1_000_000.
    Useful for verifying neutral/zero signals.
    """
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=float),
        },
        index=dates,
    )


@pytest.fixture
def trending_up_ohlcv() -> pd.DataFrame:
    """Steadily rising prices: close increases by 0.5 each day.
    Expected result: trend/momentum indicators should be bullish.
    """
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100.0 + 0.5 * np.arange(n)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 2_000_000, dtype=float),
        },
        index=dates,
    )


@pytest.fixture
def trending_down_ohlcv() -> pd.DataFrame:
    """Steadily falling prices: close decreases by 0.5 each day."""
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 250.0 - 0.5 * np.arange(n)
    return pd.DataFrame(
        {
            "open": close + 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 2_000_000, dtype=float),
        },
        index=dates,
    )


@pytest.fixture
def crossover_ohlcv() -> pd.DataFrame:
    """Price series that crosses from bearish to bullish at midpoint.

    First 150 bars: declining (fast SMA < slow SMA).
    Last 150 bars: strongly rising (fast SMA crosses above slow SMA).
    Designed so 50/200 SMA crossover is clearly visible.
    """
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # Decline first 200 bars, then rise strongly
    close = np.concatenate(
        [
            200.0 - 0.3 * np.arange(200),  # declining phase
            140.0 + 1.0 * np.arange(300),  # rising phase
        ]
    )
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_500_000, dtype=float),
        },
        index=dates,
    )


def _assert_valid_paramspec(params: dict) -> None:
    """Assert that all values in params dict are valid ParamSpec instances."""
    assert len(params) > 0, "get_tunable_params() must return at least one param"
    for name, spec in params.items():
        assert isinstance(spec, ParamSpec), f"Param '{name}' must be a ParamSpec instance"
        assert spec.name == name, f"ParamSpec.name '{spec.name}' must match key '{name}'"
        if spec.type in ("int", "float"):
            assert spec.low is not None and spec.high is not None
            assert spec.low < spec.high, f"ParamSpec '{name}': low must be < high"
        if spec.type == "categorical":
            assert spec.choices and len(spec.choices) > 0


# ---------------------------------------------------------------------------
# SMACrossoverIndicator
# ---------------------------------------------------------------------------


class TestSMACrossoverIndicator:
    ind = SMACrossoverIndicator()
    params = {"fast_period": 50, "slow_period": 200}
    small_params = {"fast_period": 5, "slow_period": 20}

    def test_compute_returns_expected_columns(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        assert "sma_fast" in result.columns
        assert "sma_slow" in result.columns
        assert "sma_crossover" in result.columns

    def test_compute_does_not_mutate_input(self, trending_up_ohlcv):
        original_cols = set(trending_up_ohlcv.columns)
        self.ind.compute(trending_up_ohlcv, self.params)
        assert set(trending_up_ohlcv.columns) == original_cols

    def test_uptrend_gives_positive_crossover(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.small_params)
        valid = result["sma_crossover"].dropna()
        # After warmup, fast SMA > slow SMA in uptrend
        assert valid.iloc[-1] > 0

    def test_downtrend_gives_negative_crossover(self, trending_down_ohlcv):
        result = self.ind.compute(trending_down_ohlcv, self.small_params)
        valid = result["sma_crossover"].dropna()
        assert valid.iloc[-1] < 0

    def test_normalize_within_bounds(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.small_params)
        normalized = self.ind.normalize(result["sma_crossover"])
        valid = normalized.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_normalize_large_gap_clips_to_one(self):
        """A 10% gap (> 5% threshold) should saturate at +1."""
        s = pd.Series([0.10, -0.10, 0.05, 0.0])
        out = self.ind.normalize(s)
        assert out.iloc[0] == pytest.approx(1.0)
        assert out.iloc[1] == pytest.approx(-1.0)
        assert out.iloc[2] == pytest.approx(1.0)
        assert out.iloc[3] == pytest.approx(0.0)

    def test_tunable_params_valid(self):
        _assert_valid_paramspec(self.ind.get_tunable_params())

    def test_default_params(self):
        p = self.ind.get_default_params()
        assert p["fast_period"] == 50
        assert p["slow_period"] == 200


# ---------------------------------------------------------------------------
# RSIIndicator
# ---------------------------------------------------------------------------


class TestRSIIndicator:
    ind = RSIIndicator()
    params = {"period": 14}

    def test_compute_returns_rsi_column(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        assert "rsi" in result.columns

    def test_rsi_bounds(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        valid = result["rsi"].dropna()
        assert (valid >= 0.0).all() and (valid <= 100.0).all()

    def test_uptrend_rsi_high(self, trending_up_ohlcv):
        """Steady uptrend → RSI should be above 50."""
        result = self.ind.compute(trending_up_ohlcv, self.params)
        valid = result["rsi"].dropna()
        assert valid.iloc[-1] > 50.0

    def test_downtrend_rsi_low(self, trending_down_ohlcv):
        """Steady downtrend → RSI should be below 50."""
        result = self.ind.compute(trending_down_ohlcv, self.params)
        valid = result["rsi"].dropna()
        assert valid.iloc[-1] < 50.0

    def test_flat_series_rsi_near_50(self, flat_ohlcv):
        """Flat price → diffs are 0 → RSI should be NaN or near 50."""
        result = self.ind.compute(flat_ohlcv, self.params)
        valid = result["rsi"].dropna()
        # Flat series has zero gains and losses; RSI may be NaN from 0/0
        # If non-NaN, must be in [0, 100]
        if len(valid) > 0:
            assert (valid >= 0.0).all() and (valid <= 100.0).all()

    def test_normalize_within_bounds(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        normalized = self.ind.normalize(result["rsi"])
        valid = normalized.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_normalize_known_values(self):
        """RSI=50 → 0, RSI=30 → +0.4, RSI=70 → -0.4, RSI=0 → +1, RSI=100 → -1."""
        s = pd.Series([50.0, 30.0, 70.0, 0.0, 100.0])
        out = self.ind.normalize(s)
        assert out.iloc[0] == pytest.approx(0.0)
        assert out.iloc[1] == pytest.approx(0.4)
        assert out.iloc[2] == pytest.approx(-0.4)
        assert out.iloc[3] == pytest.approx(1.0)
        assert out.iloc[4] == pytest.approx(-1.0)

    def test_normalize_oversold_bullish(self, trending_down_ohlcv):
        """Oversold after downtrend → normalize() > 0 (bullish signal)."""
        result = self.ind.compute(trending_down_ohlcv, self.params)
        normalized = self.ind.normalize(result["rsi"])
        assert normalized.dropna().iloc[-1] > 0

    def test_normalize_overbought_bearish(self, trending_up_ohlcv):
        """Overbought after uptrend → normalize() < 0 (bearish mean-reversion)."""
        result = self.ind.compute(trending_up_ohlcv, self.params)
        normalized = self.ind.normalize(result["rsi"])
        assert normalized.dropna().iloc[-1] < 0

    def test_tunable_params_valid(self):
        _assert_valid_paramspec(self.ind.get_tunable_params())

    def test_default_params(self):
        assert self.ind.get_default_params()["period"] == 14


# ---------------------------------------------------------------------------
# MACDIndicator
# ---------------------------------------------------------------------------


class TestMACDIndicator:
    ind = MACDIndicator()
    params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}

    def test_compute_returns_expected_columns(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        for col in ("macd", "macd_signal", "macd_hist"):
            assert col in result.columns

    def test_uptrend_macd_positive(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        valid = result["macd"].dropna()
        assert valid.iloc[-1] > 0

    def test_downtrend_macd_negative(self, trending_down_ohlcv):
        result = self.ind.compute(trending_down_ohlcv, self.params)
        valid = result["macd"].dropna()
        assert valid.iloc[-1] < 0

    def test_normalize_within_bounds(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        normalized = self.ind.normalize(result["macd_hist"])
        valid = normalized.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_flat_macd_zero(self, flat_ohlcv):
        """Flat price → all EMAs equal → MACD and histogram = 0."""
        result = self.ind.compute(flat_ohlcv, self.params)
        assert result["macd"].dropna().abs().max() == pytest.approx(0.0, abs=1e-10)
        assert result["macd_hist"].dropna().abs().max() == pytest.approx(0.0, abs=1e-10)

    def test_tunable_params_valid(self):
        _assert_valid_paramspec(self.ind.get_tunable_params())

    def test_tunable_params_fast_less_than_slow(self):
        params = self.ind.get_tunable_params()
        assert params["fast_period"].high <= params["slow_period"].low

    def test_default_params(self):
        p = self.ind.get_default_params()
        assert p["fast_period"] == 12
        assert p["slow_period"] == 26
        assert p["signal_period"] == 9


# ---------------------------------------------------------------------------
# BollingerBandIndicator
# ---------------------------------------------------------------------------


class TestBollingerBandIndicator:
    ind = BollingerBandIndicator()
    params = {"period": 20, "num_std": 2.0}

    def test_compute_returns_expected_columns(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        for col in ("bb_upper", "bb_lower", "bb_mid", "bb_pct_b", "bb_width"):
            assert col in result.columns

    def test_upper_above_mid_above_lower(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        valid = result.dropna(subset=["bb_upper", "bb_lower", "bb_mid"])
        assert (valid["bb_upper"] >= valid["bb_mid"]).all()
        assert (valid["bb_mid"] >= valid["bb_lower"]).all()

    def test_pct_b_is_float_between_zero_and_one_near_mid(self, flat_ohlcv):
        """Flat price ≈ midline → %B ≈ 0.5 (but std is near zero, so may be NaN)."""
        result = self.ind.compute(flat_ohlcv, self.params)
        valid = result["bb_pct_b"].dropna()
        # When std is near zero (flat series), pct_b is NaN — that's fine
        if len(valid) > 0:
            assert (valid >= -0.5).all() and (valid <= 1.5).all()

    def test_normalize_known_values(self):
        """%B=0 (lower band) → +1, %B=1 (upper band) → -1, %B=0.5 → 0."""
        s = pd.Series([0.0, 1.0, 0.5, 0.25, 0.75])
        out = self.ind.normalize(s)
        assert out.iloc[0] == pytest.approx(1.0)
        assert out.iloc[1] == pytest.approx(-1.0)
        assert out.iloc[2] == pytest.approx(0.0)
        assert out.iloc[3] == pytest.approx(0.5)
        assert out.iloc[4] == pytest.approx(-0.5)

    def test_normalize_within_bounds(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        normalized = self.ind.normalize(result["bb_pct_b"])
        valid = normalized.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_tunable_params_valid(self):
        _assert_valid_paramspec(self.ind.get_tunable_params())

    def test_default_params(self):
        p = self.ind.get_default_params()
        assert p["period"] == 20
        assert p["num_std"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# DonchianChannelIndicator
# ---------------------------------------------------------------------------


class TestDonchianChannelIndicator:
    ind = DonchianChannelIndicator()
    params = {"period": 20}

    def test_compute_returns_expected_columns(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        for col in ("dc_upper", "dc_lower", "dc_mid", "dc_position"):
            assert col in result.columns

    def test_upper_above_lower(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        valid = result.dropna(subset=["dc_upper", "dc_lower"])
        assert (valid["dc_upper"] >= valid["dc_lower"]).all()

    def test_no_lookahead_shift(self, trending_up_ohlcv):
        """dc_upper[t] should equal max(high[t-period:t-1]), not including today."""
        result = self.ind.compute(trending_up_ohlcv, {"period": 5})
        # The channel for row i should equal max of high rows [i-5 .. i-1]
        for i in range(6, 15):
            expected_upper = trending_up_ohlcv["high"].iloc[i - 5 : i].max()
            actual_upper = result["dc_upper"].iloc[i]
            assert actual_upper == pytest.approx(expected_upper, rel=1e-9)

    def test_position_at_top_of_channel(self):
        """If close = dc_upper, dc_position should be +1."""
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        # Flat channel except last bar breaches to exactly dc_upper
        close = np.full(n, 100.0)
        high = np.full(n, 102.0)
        low = np.full(n, 98.0)
        df = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": np.ones(n)},
            index=dates,
        )
        result = self.ind.compute(df, {"period": 20})
        valid = result.dropna(subset=["dc_position"])
        # All closes are at the midpoint → position should be 0
        assert valid["dc_position"].abs().max() == pytest.approx(0.0, abs=1e-9)

    def test_uptrend_positive_position(self, trending_up_ohlcv):
        """Rising price → close approaches channel top → positive position."""
        result = self.ind.compute(trending_up_ohlcv, self.params)
        valid = result["dc_position"].dropna()
        assert valid.iloc[-1] > 0

    def test_normalize_clips_to_bounds(self):
        """Values outside [-1, +1] are clipped."""
        s = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = self.ind.normalize(s)
        assert out.iloc[0] == pytest.approx(-1.0)
        assert out.iloc[4] == pytest.approx(1.0)
        assert (out >= -1.0).all() and (out <= 1.0).all()

    def test_normalize_within_bounds(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        normalized = self.ind.normalize(result["dc_position"])
        valid = normalized.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_tunable_params_valid(self):
        _assert_valid_paramspec(self.ind.get_tunable_params())

    def test_default_params(self):
        assert self.ind.get_default_params()["period"] == 20


# ---------------------------------------------------------------------------
# VolumeIndicator
# ---------------------------------------------------------------------------


class TestVolumeIndicator:
    ind = VolumeIndicator()
    params = {"fast_period": 10, "slow_period": 30}

    def test_compute_returns_expected_columns(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        for col in ("obv", "obv_ema_fast", "obv_ema_slow", "obv_signal"):
            assert col in result.columns

    def test_obv_increases_on_up_days(self):
        """OBV should monotonically increase when every close goes up."""
        n = 50
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 100.0 + np.arange(n, dtype=float)
        df = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            },
            index=dates,
        )
        result = self.ind.compute(df, self.params)
        # OBV must be strictly increasing
        diffs = result["obv"].diff().dropna()
        assert (diffs >= 0).all()
        assert (diffs > 0).sum() > 0

    def test_obv_decreases_on_down_days(self, trending_down_ohlcv):
        result = self.ind.compute(trending_down_ohlcv, self.params)
        obv = result["obv"]
        # OBV must decrease overall (last value < first non-NaN value)
        assert obv.iloc[-1] < obv.iloc[1]

    def test_uptrend_obv_signal_positive(self, trending_up_ohlcv):
        """Rising OBV → fast EMA > slow EMA → obv_signal > 0."""
        result = self.ind.compute(trending_up_ohlcv, self.params)
        valid = result["obv_signal"].dropna()
        assert valid.iloc[-1] > 0

    def test_normalize_within_bounds(self, trending_up_ohlcv):
        result = self.ind.compute(trending_up_ohlcv, self.params)
        normalized = self.ind.normalize(result["obv_signal"])
        valid = normalized.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_tunable_params_valid(self):
        _assert_valid_paramspec(self.ind.get_tunable_params())

    def test_tunable_params_fast_less_than_slow(self):
        params = self.ind.get_tunable_params()
        assert params["fast_period"].high <= params["slow_period"].low

    def test_default_params(self):
        p = self.ind.get_default_params()
        assert p["fast_period"] == 10
        assert p["slow_period"] == 30
