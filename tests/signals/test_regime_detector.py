"""
Unit tests for RegimeDetector.

All tests use synthetic OHLCV data — no network calls, no real market data.
The HMM uses random_state=42 for reproducibility.

Test strategy:
  - Deterministic synthetic series (linear ramp, sine wave) for reliable
    ADX values.
  - Stochastic random-walk series for HMM training (with fixed seed).
  - _reconcile() is tested directly so reconciliation logic has 100% branch
    coverage independent of HMM internals.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.trade_signal import RegimeType
from src.signals.regime_detector import RegimeDetector

# ---------------------------------------------------------------------------
# Synthetic OHLCV helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(closes: np.ndarray, spread: float = 0.3) -> pd.DataFrame:
    """Convert a close-price array into a minimal OHLCV DataFrame."""
    dates = pd.date_range("2020-01-02", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes - 0.1,
            "high": closes + spread,
            "low": closes - spread,
            "close": closes,
            "volume": np.full(len(closes), 1_000_000),
        },
        index=dates,
    )


def _make_trending_up(n: int = 300) -> pd.DataFrame:
    """Perfectly linear uptrend — ADX converges to ~100."""
    closes = 100.0 + np.arange(n, dtype=float) * 0.5
    return _make_ohlcv(closes, spread=0.1)


def _make_trending_down(n: int = 300) -> pd.DataFrame:
    """Perfectly linear downtrend — ADX converges to ~100."""
    closes = 300.0 - np.arange(n, dtype=float) * 0.5
    return _make_ohlcv(closes, spread=0.1)


def _make_ranging(n: int = 300) -> pd.DataFrame:
    """
    Perfectly alternating ±0.5 market — ADX converges to ~0.

    Each bar alternates direction, so within any 14-bar ADX window
    up-moves and down-moves balance exactly (+DI ≈ -DI → DX ≈ 0).
    """
    changes = np.where(np.arange(n) % 2 == 0, 0.5, -0.5)
    closes = 100.0 + np.cumsum(changes)
    return _make_ohlcv(closes, spread=0.1)


def _make_random_walk(n: int = 300, seed: int = 42, daily_vol: float = 0.01) -> pd.DataFrame:
    """Random-walk price series with realistic daily volatility."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0003, daily_vol, n)  # slight upward drift
    closes = 100.0 * np.exp(np.cumsum(log_rets))
    return _make_ohlcv(closes, spread=closes * 0.002)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_detector(tmp_path_factory):
    """
    RegimeDetector fitted on a 300-bar random-walk series.

    Scoped to module for speed — HMM fitting takes ~0.5 s per fit.
    """
    tmp = tmp_path_factory.mktemp("hmm")
    df = _make_random_walk(n=300, seed=42)
    rd = RegimeDetector(model_dir=str(tmp))
    rd.fit(df, ticker="test")
    return rd


@pytest.fixture()
def unfitted_detector():
    """RegimeDetector that has NOT been fitted (but has state_labels populated)."""
    rd = RegimeDetector()
    # Inject labels so _reconcile() can be tested without HMM
    rd._state_labels = {0: "bull", 1: "bear", 2: "sideways"}
    return rd


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_n_components(self) -> None:
        assert RegimeDetector().n_components == 3

    def test_default_trend_threshold(self) -> None:
        assert RegimeDetector().trend_threshold == 25.0

    def test_default_range_threshold(self) -> None:
        assert RegimeDetector().range_threshold == 20.0

    def test_default_uncertainty_threshold(self) -> None:
        assert RegimeDetector().uncertainty_threshold == 0.40

    def test_default_adx_period(self) -> None:
        assert RegimeDetector().adx_period == 14

    def test_default_lookback_days(self) -> None:
        assert RegimeDetector().lookback_days == 504

    def test_not_fitted_on_construction(self) -> None:
        assert not RegimeDetector()._is_fitted


# Expose private params as read-only properties for the init tests above.
# Patch the class to add them (avoids modifying production code):
RegimeDetector.n_components = property(lambda self: self._n_components)
RegimeDetector.trend_threshold = property(lambda self: self._trend_threshold)
RegimeDetector.range_threshold = property(lambda self: self._range_threshold)
RegimeDetector.uncertainty_threshold = property(lambda self: self._uncertainty_threshold)
RegimeDetector.adx_period = property(lambda self: self._adx_period)
RegimeDetector.lookback_days = property(lambda self: self._lookback_days)


# ---------------------------------------------------------------------------
# 2. Feature computation
# ---------------------------------------------------------------------------


class TestFeatureComputation:
    def test_feature_shape_is_n_minus_2_by_2(self) -> None:
        df = _make_random_walk(n=50)
        features, valid_index = RegimeDetector()._compute_features(df)
        # Row 0: log_return NaN; Row 1: realized_vol std(single value)=NaN → both dropped
        assert features.shape == (48, 2)

    def test_no_nan_in_returned_features(self) -> None:
        df = _make_random_walk(n=50)
        features, _ = RegimeDetector()._compute_features(df)
        assert not np.isnan(features).any()

    def test_valid_index_length_matches_feature_rows(self) -> None:
        df = _make_random_walk(n=50)
        features, valid_index = RegimeDetector()._compute_features(df)
        assert len(valid_index) == len(features)

    def test_log_return_sign_correct_for_uptrend(self) -> None:
        df = _make_trending_up(n=50)
        features, _ = RegimeDetector()._compute_features(df)
        # Linear uptrend: all log returns positive
        assert (features[:, 0] > 0).all()

    def test_log_return_sign_correct_for_downtrend(self) -> None:
        df = _make_trending_down(n=50)
        features, _ = RegimeDetector()._compute_features(df)
        assert (features[:, 0] < 0).all()

    def test_realized_vol_is_nonnegative(self) -> None:
        df = _make_random_walk(n=100)
        features, _ = RegimeDetector()._compute_features(df)
        assert (features[:, 1] >= 0).all()

    def test_small_df_returns_empty_if_only_one_row(self) -> None:
        df = _make_trending_up(n=1)
        features, valid_index = RegimeDetector()._compute_features(df)
        assert len(features) == 0
        assert len(valid_index) == 0


# ---------------------------------------------------------------------------
# 3. ADX computation
# ---------------------------------------------------------------------------


class TestADXComputation:
    def test_returns_series_aligned_to_df_index(self) -> None:
        df = _make_trending_up(n=100)
        adx = RegimeDetector()._compute_adx(df)
        assert isinstance(adx, pd.Series)
        assert adx.index.equals(df.index)

    def test_trending_up_gives_high_adx(self) -> None:
        """Strong linear uptrend: ADX should exceed 25 after convergence."""
        df = _make_trending_up(n=300)
        adx = RegimeDetector()._compute_adx(df)
        last_valid = adx.dropna().iloc[-1]
        assert last_valid > 25.0, f"Expected ADX > 25 for trending up, got {last_valid:.1f}"

    def test_trending_down_gives_high_adx(self) -> None:
        """ADX measures trend strength regardless of direction."""
        df = _make_trending_down(n=300)
        adx = RegimeDetector()._compute_adx(df)
        last_valid = adx.dropna().iloc[-1]
        assert last_valid > 25.0, f"Expected ADX > 25 for trending down, got {last_valid:.1f}"

    def test_ranging_gives_low_adx(self) -> None:
        """Sine-wave oscillation: ADX should converge below 20."""
        df = _make_ranging(n=300)
        adx = RegimeDetector()._compute_adx(df)
        last_valid = adx.dropna().iloc[-1]
        assert last_valid < 20.0, f"Expected ADX < 20 for ranging, got {last_valid:.1f}"

    def test_early_rows_are_nan(self) -> None:
        """ADX requires warm-up — first several rows must be NaN."""
        df = _make_trending_up(n=100)
        adx = RegimeDetector()._compute_adx(df)
        assert adx.iloc[:10].isna().all()

    def test_insufficient_data_returns_nan_series(self) -> None:
        """Fewer than 2 × adx_period rows → pandas-ta returns None or all NaN."""
        df = _make_trending_up(n=5)
        adx = RegimeDetector()._compute_adx(df)
        assert adx.isna().all()


# ---------------------------------------------------------------------------
# 4. fit()
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_self(self, tmp_path) -> None:
        df = _make_random_walk(n=100)
        rd = RegimeDetector(model_dir=str(tmp_path))
        result = rd.fit(df)
        assert result is rd

    def test_is_fitted_after_fit(self, tmp_path) -> None:
        df = _make_random_walk(n=100)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df)
        assert rd._is_fitted

    def test_state_labels_has_n_components_keys(self, tmp_path) -> None:
        df = _make_random_walk(n=100)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df)
        assert len(rd._state_labels) == 3

    def test_state_labels_values_are_valid(self, tmp_path) -> None:
        df = _make_random_walk(n=100)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df)
        assert set(rd._state_labels.values()) == {"bull", "bear", "sideways"}

    def test_transition_matrix_shape(self, tmp_path) -> None:
        df = _make_random_walk(n=100)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df)
        tm = rd.get_transition_matrix()
        assert tm.shape == (3, 3)

    def test_transition_matrix_rows_sum_to_one(self, tmp_path) -> None:
        df = _make_random_walk(n=100)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df)
        row_sums = rd.get_transition_matrix().sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_model_file_created(self, tmp_path) -> None:
        df = _make_random_walk(n=100)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df, ticker="AAPL")
        assert (tmp_path / "AAPL.pkl").exists()

    def test_raises_on_insufficient_data(self, tmp_path) -> None:
        # n=4 → n_valid = 3 < n_components+1 = 4 → should raise
        df = _make_trending_up(n=4)
        rd = RegimeDetector(model_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Insufficient data"):
            rd.fit(df)


# ---------------------------------------------------------------------------
# 5. State label ordering
# ---------------------------------------------------------------------------


class TestStateLabelOrdering:
    def test_bull_state_has_highest_mean_return(self, fitted_detector: RegimeDetector) -> None:
        """The state labelled 'bull' must have the highest mean log return."""
        rd = fitted_detector
        df = _make_random_walk(n=300, seed=42)
        features, _ = rd._compute_features(df)
        states = rd._hmm.predict(features)

        bull_id = [k for k, v in rd._state_labels.items() if v == "bull"][0]
        bear_id = [k for k, v in rd._state_labels.items() if v == "bear"][0]

        bull_mean = features[states == bull_id, 0].mean()
        bear_mean = features[states == bear_id, 0].mean()
        assert bull_mean > bear_mean

    def test_bear_state_has_lowest_mean_return(self, fitted_detector: RegimeDetector) -> None:
        rd = fitted_detector
        df = _make_random_walk(n=300, seed=42)
        features, _ = rd._compute_features(df)
        states = rd._hmm.predict(features)

        bear_id = [k for k, v in rd._state_labels.items() if v == "bear"][0]
        sideways_id = [k for k, v in rd._state_labels.items() if v == "sideways"][0]

        bear_mean = features[states == bear_id, 0].mean()
        sideways_mean = features[states == sideways_id, 0].mean()
        assert bear_mean < sideways_mean


# ---------------------------------------------------------------------------
# 6. detect() — current-day classification
# ---------------------------------------------------------------------------


class TestDetect:
    def test_returns_regime_type(self, fitted_detector: RegimeDetector) -> None:
        df = _make_random_walk(n=100, seed=99)
        result = fitted_detector.detect(df)
        assert isinstance(result, RegimeType)

    def test_raises_if_not_fitted(self) -> None:
        rd = RegimeDetector()
        with pytest.raises(RuntimeError, match="must be fitted"):
            rd.detect(_make_random_walk(n=50))

    def test_trending_up_not_classified_as_ranging(self, tmp_path) -> None:
        """Strong uptrend should NOT return RANGING (high ADX rules it out)."""
        df = _make_trending_up(n=300)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df)
        result = rd.detect(df)
        assert result != RegimeType.RANGING, f"Strong uptrend should not be RANGING, got {result}"

    def test_trending_down_not_classified_as_ranging(self, tmp_path) -> None:
        df = _make_trending_down(n=300)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df)
        result = rd.detect(df)
        assert result != RegimeType.RANGING

    def test_ranging_not_classified_as_trending(self, tmp_path) -> None:
        """Sine-wave oscillation should NOT return TRENDING_UP or TRENDING_DOWN."""
        df = _make_ranging(n=300)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df)
        result = rd.detect(df)
        assert result not in (
            RegimeType.TRENDING_UP,
            RegimeType.TRENDING_DOWN,
        ), f"Ranging market should not be TRENDING_*, got {result}"


# ---------------------------------------------------------------------------
# 7. detect_series() — bulk classification
# ---------------------------------------------------------------------------


class TestDetectSeries:
    def test_returns_series(self, fitted_detector: RegimeDetector) -> None:
        df = _make_random_walk(n=100, seed=10)
        result = fitted_detector.detect_series(df)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, fitted_detector: RegimeDetector) -> None:
        df = _make_random_walk(n=80, seed=11)
        result = fitted_detector.detect_series(df)
        assert len(result) == len(df)

    def test_index_matches_input(self, fitted_detector: RegimeDetector) -> None:
        df = _make_random_walk(n=80, seed=12)
        result = fitted_detector.detect_series(df)
        assert result.index.equals(df.index)

    def test_all_values_are_regime_type(self, fitted_detector: RegimeDetector) -> None:
        df = _make_random_walk(n=80, seed=13)
        result = fitted_detector.detect_series(df)
        assert all(isinstance(v, RegimeType) for v in result)

    def test_raises_if_not_fitted(self) -> None:
        rd = RegimeDetector()
        with pytest.raises(RuntimeError, match="must be fitted"):
            rd.detect_series(_make_random_walk(n=50))

    def test_trending_series_has_mostly_consistent_regime(self, tmp_path) -> None:
        """≥ 70% of rows in a strong trending series should share one regime."""
        df = _make_trending_up(n=300)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df)
        series = rd.detect_series(df)

        # Count the most common regime in the second half (after ADX warm-up)
        tail_series = series.iloc[150:]
        dominant_count = tail_series.value_counts().iloc[0]
        assert dominant_count / len(tail_series) >= 0.70, (
            f"Expected ≥70% dominant regime in trending data, "
            f"got {dominant_count / len(tail_series):.1%}"
        )


# ---------------------------------------------------------------------------
# 8. Reconciliation logic — direct unit tests (no HMM required)
# ---------------------------------------------------------------------------


class TestReconciliation:
    """Test every branch of _reconcile() in isolation."""

    def test_high_uncertainty_returns_volatile(self, unfitted_detector: RegimeDetector) -> None:
        rd = unfitted_detector
        assert rd._reconcile(30.0, 0, 0.45) == RegimeType.VOLATILE

    def test_uncertainty_exactly_at_threshold_is_not_volatile(
        self, unfitted_detector: RegimeDetector
    ) -> None:
        rd = unfitted_detector
        # uncertainty == threshold (0.40): boundary — NOT volatile (strictly >)
        result = rd._reconcile(30.0, 0, 0.40)
        assert result != RegimeType.VOLATILE

    def test_low_adx_returns_ranging_regardless_of_hmm(
        self, unfitted_detector: RegimeDetector
    ) -> None:
        rd = unfitted_detector
        # Even with bull state, ADX < 20 → RANGING
        assert rd._reconcile(15.0, 0, 0.10) == RegimeType.RANGING

    def test_high_adx_bull_state_returns_trending_up(
        self, unfitted_detector: RegimeDetector
    ) -> None:
        rd = unfitted_detector
        assert rd._reconcile(28.0, 0, 0.10) == RegimeType.TRENDING_UP  # state 0 = bull

    def test_high_adx_bear_state_returns_trending_down(
        self, unfitted_detector: RegimeDetector
    ) -> None:
        rd = unfitted_detector
        assert rd._reconcile(28.0, 1, 0.10) == RegimeType.TRENDING_DOWN  # state 1 = bear

    def test_high_adx_sideways_state_returns_ranging(
        self, unfitted_detector: RegimeDetector
    ) -> None:
        rd = unfitted_detector
        assert rd._reconcile(28.0, 2, 0.10) == RegimeType.RANGING  # state 2 = sideways

    def test_transition_zone_adx_bull_returns_trending_up(
        self, unfitted_detector: RegimeDetector
    ) -> None:
        """ADX in 20–25 transition zone: direction from HMM."""
        rd = unfitted_detector
        assert rd._reconcile(22.0, 0, 0.10) == RegimeType.TRENDING_UP

    def test_uncertainty_takes_precedence_over_low_adx(
        self, unfitted_detector: RegimeDetector
    ) -> None:
        """VOLATILE override applies even when ADX < 20."""
        rd = unfitted_detector
        assert rd._reconcile(10.0, 2, 0.50) == RegimeType.VOLATILE

    def test_unknown_state_id_defaults_to_ranging(self, unfitted_detector: RegimeDetector) -> None:
        """State ID not in state_labels dict defaults to 'sideways' → RANGING."""
        rd = unfitted_detector
        assert rd._reconcile(28.0, 99, 0.10) == RegimeType.RANGING


# ---------------------------------------------------------------------------
# 9. Persistence (save / load round-trip)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_creates_file(self, fitted_detector: RegimeDetector, tmp_path) -> None:
        dest = str(tmp_path / "roundtrip.pkl")
        fitted_detector.save(dest)
        import pathlib

        assert pathlib.Path(dest).exists()

    def test_load_returns_regime_detector(self, fitted_detector: RegimeDetector, tmp_path) -> None:
        dest = str(tmp_path / "roundtrip.pkl")
        fitted_detector.save(dest)
        loaded = RegimeDetector.load(dest)
        assert isinstance(loaded, RegimeDetector)

    def test_loaded_detector_is_fitted(self, fitted_detector: RegimeDetector, tmp_path) -> None:
        dest = str(tmp_path / "roundtrip.pkl")
        fitted_detector.save(dest)
        loaded = RegimeDetector.load(dest)
        assert loaded._is_fitted

    def test_save_load_produces_same_regime(
        self, fitted_detector: RegimeDetector, tmp_path
    ) -> None:
        df = _make_random_walk(n=100, seed=77)
        original_regime = fitted_detector.detect(df)

        dest = str(tmp_path / "roundtrip.pkl")
        fitted_detector.save(dest)
        loaded = RegimeDetector.load(dest)
        loaded_regime = loaded.detect(df)

        assert loaded_regime == original_regime

    def test_fit_auto_saves_model(self, tmp_path) -> None:
        df = _make_random_walk(n=100)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df, ticker="AUTO")
        assert (tmp_path / "AUTO.pkl").exists()


# ---------------------------------------------------------------------------
# 10. Regime stability
# ---------------------------------------------------------------------------


class TestRegimeStability:
    def test_regime_series_has_temporal_structure(self, tmp_path) -> None:
        """
        detect_series on structured data (three distinct regime phases) must
        produce temporal structure — not random noise.

        **Setup:** 300 bars split into three 100-bar phases:
          - Phase 1: strong uptrend (high drift, low vol)
          - Phase 2: flat / ranging (zero drift, high vol)
          - Phase 3: strong downtrend (negative drift, low vol)

        With 4 possible regime values and 260 post-warmup bars, a purely
        random assignment would produce an expected maximum run of only
        ~4 days (log(260) / log(4)). A working detector must produce
        at least one regime run >= 30 consecutive days.
        """
        warmup = 40
        rng = np.random.default_rng(99)
        phase1 = rng.normal(0.010, 0.004, 100)  # strong uptrend, low vol
        phase2 = rng.normal(0.000, 0.020, 100)  # flat, high vol
        phase3 = rng.normal(-0.010, 0.004, 100)  # strong downtrend, low vol

        log_rets = np.concatenate([phase1, phase2, phase3])
        closes = 100.0 * np.exp(np.cumsum(log_rets))
        df = _make_ohlcv(closes, spread=closes * 0.001)

        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df)
        series = rd.detect_series(df)

        post_warmup = series.tolist()[warmup:]

        # Compute all run lengths in the post-warmup sequence.
        runs: list[int] = []
        if post_warmup:
            current, length = post_warmup[0], 1
            for v in post_warmup[1:]:
                if v == current:
                    length += 1
                else:
                    runs.append(length)
                    current, length = v, 1
            runs.append(length)

        max_run = max(runs) if runs else 0
        assert max_run >= 30, (
            f"Longest regime run {max_run} < 30 days — detector lacks temporal "
            f"structure on 3-phase data. "
            f"Top 5 run lengths: {sorted(runs, reverse=True)[:5]}"
        )

    def test_regime_series_not_all_volatile(self, tmp_path) -> None:
        """A 300-bar random-walk series should produce some non-VOLATILE regimes."""
        df = _make_random_walk(n=300, seed=42)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df)
        series = rd.detect_series(df)
        non_volatile = (series != RegimeType.VOLATILE).sum()
        assert non_volatile > 50, f"Expected many non-VOLATILE rows, got {non_volatile}"


# ---------------------------------------------------------------------------
# 11. Minimum data requirements
# ---------------------------------------------------------------------------


class TestMinimumData:
    def test_fit_raises_on_4_row_df(self, tmp_path) -> None:
        """4 rows → 3 valid features (< n_components + 1 = 4) → ValueError."""
        df = _make_trending_up(n=4)
        rd = RegimeDetector(model_dir=str(tmp_path))
        with pytest.raises(ValueError):
            rd.fit(df)

    def test_detect_returns_volatile_on_empty_features(self, tmp_path) -> None:
        """detect() with 1-row df → no valid features → VOLATILE."""
        df_train = _make_random_walk(n=100)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df_train)

        df_single = _make_trending_up(n=1)
        result = rd.detect(df_single)
        assert result == RegimeType.VOLATILE

    def test_detect_series_returns_all_volatile_on_tiny_df(self, tmp_path) -> None:
        """detect_series with 1-row df → all VOLATILE."""
        df_train = _make_random_walk(n=100)
        rd = RegimeDetector(model_dir=str(tmp_path))
        rd.fit(df_train)

        df_single = _make_trending_up(n=1)
        result = rd.detect_series(df_single)
        assert all(v == RegimeType.VOLATILE for v in result)
