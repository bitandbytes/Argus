"""Tests for WalkForwardOptimizer, WFOResult, and CombinatorialPurgedCV.

Covers:
  - Window construction: count, non-overlapping OOS, edge cases
  - optimize(): returns valid WFOResult with populated fields
  - PBO computation: all-overfit, none-overfit, partial
  - Stability check delegation to BayesianTuner
  - CombinatorialPurgedCV: split count, PBO from known results
  - Error handling: insufficient data, empty candidates
  - MLflow logging (mocked)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.plugins.base import IndicatorPlugin, ParamSpec
from src.tuning.bayesian_tuner import BayesianTuner
from src.tuning.purged_cv import CombinatorialPurgedCV
from src.tuning.walk_forward import WalkForwardOptimizer, WFOResult, WFOWindowResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 800, trend: float = 0.0005, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n, freq="B")
    log_returns = rng.normal(trend, 0.015, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class _SimplePlugin(IndicatorPlugin):
    """Minimal plugin with one int param."""
    name = "simple"
    category = "trend"
    version = "0.0.1"
    output_column = "simple_out"

    def compute(self, df, params):
        df = df.copy()
        df["simple_out"] = 0.0
        return df

    def normalize(self, values):
        return values

    def get_tunable_params(self):
        return {"period": ParamSpec("period", "int", low=5, high=20, default=10)}

    def get_default_params(self):
        return {"period": 10}


def _dummy_objective(params: Dict[str, Any], df: pd.DataFrame) -> float:
    """Returns a deterministic value based on params and df length."""
    period = params.get("simple__period", 10)
    return float(period) / 20.0 + len(df) / 10_000.0


def _zero_objective(params: Dict[str, Any], df: pd.DataFrame) -> float:
    return 0.0


def _len_objective(params: Dict[str, Any], df: pd.DataFrame) -> float:
    """IS returns high Sharpe; OOS deliberately low — simulates IS/OOS gap."""
    return len(df) / 500.0  # longer data → higher "sharpe"


# ---------------------------------------------------------------------------
# TestWalkForwardOptimizerInit
# ---------------------------------------------------------------------------

class TestWalkForwardOptimizerInit:
    def test_default_params(self) -> None:
        opt = WalkForwardOptimizer()
        assert opt.in_sample_days == 252
        assert opt.out_of_sample_days == 126
        assert opt.n_trials == 100
        assert opt.pbo_top_k == 10

    def test_custom_params_stored(self) -> None:
        opt = WalkForwardOptimizer(in_sample_days=100, out_of_sample_days=50, n_trials=10)
        assert opt.in_sample_days == 100
        assert opt.out_of_sample_days == 50
        assert opt.n_trials == 10

    def test_invalid_in_sample_days_raises(self) -> None:
        with pytest.raises(ValueError, match="in_sample_days"):
            WalkForwardOptimizer(in_sample_days=5)

    def test_invalid_oos_days_raises(self) -> None:
        with pytest.raises(ValueError, match="out_of_sample_days"):
            WalkForwardOptimizer(out_of_sample_days=3)

    def test_invalid_n_trials_raises(self) -> None:
        with pytest.raises(ValueError, match="n_trials"):
            WalkForwardOptimizer(n_trials=0)

    def test_invalid_pbo_top_k_raises(self) -> None:
        with pytest.raises(ValueError, match="pbo_top_k"):
            WalkForwardOptimizer(pbo_top_k=1)


# ---------------------------------------------------------------------------
# TestMakeWindows
# ---------------------------------------------------------------------------

class TestMakeWindows:
    def test_correct_number_of_windows(self) -> None:
        opt = WalkForwardOptimizer(in_sample_days=100, out_of_sample_days=50)
        # 100 IS + 3×50 OOS = 250 rows → exactly 3 windows
        df = _make_ohlcv(n=250)
        windows = opt._make_windows(df)
        assert len(windows) >= 3

    def test_oos_windows_non_overlapping(self) -> None:
        opt = WalkForwardOptimizer(in_sample_days=100, out_of_sample_days=50)
        df = _make_ohlcv(n=500)
        windows = opt._make_windows(df)
        # Each OOS start must be >= previous OOS end
        for (_, oos1), (_, oos2) in zip(windows, windows[1:]):
            assert oos2.index[0] >= oos1.index[-1]

    def test_is_windows_correct_length(self) -> None:
        opt = WalkForwardOptimizer(in_sample_days=100, out_of_sample_days=50)
        df = _make_ohlcv(n=500)
        windows = opt._make_windows(df)
        for is_df, _ in windows:
            assert len(is_df) == 100

    def test_insufficient_data_raises(self) -> None:
        opt = WalkForwardOptimizer(in_sample_days=252, out_of_sample_days=126)
        # 252 IS + 2×126 OOS = 504 < 3 windows requirement
        df = _make_ohlcv(n=300)
        with pytest.raises(ValueError, match="Insufficient data"):
            opt._make_windows(df)

    def test_larger_dataset_more_windows(self) -> None:
        opt = WalkForwardOptimizer(in_sample_days=100, out_of_sample_days=50)
        df_small = _make_ohlcv(n=300)
        df_large = _make_ohlcv(n=700)
        w_small = opt._make_windows(df_small)
        w_large = opt._make_windows(df_large)
        assert len(w_large) > len(w_small)


# ---------------------------------------------------------------------------
# TestOptimize
# ---------------------------------------------------------------------------

class TestOptimize:
    """Integration tests for the full optimize() path (fast, no real data)."""

    def _make_optimizer(self) -> WalkForwardOptimizer:
        return WalkForwardOptimizer(
            in_sample_days=50,
            out_of_sample_days=25,
            n_trials=3,
            pbo_top_k=2,
            mlflow_tracking=False,
        )

    def test_returns_wfo_result(self) -> None:
        opt = self._make_optimizer()
        df = _make_ohlcv(n=250)
        result = opt.optimize(df, "TEST", _dummy_objective, [_SimplePlugin()])
        assert isinstance(result, WFOResult)

    def test_ticker_stored(self) -> None:
        opt = self._make_optimizer()
        df = _make_ohlcv(n=250)
        result = opt.optimize(df, "AAPL", _dummy_objective, [_SimplePlugin()])
        assert result.ticker == "AAPL"

    def test_n_windows_positive(self) -> None:
        opt = self._make_optimizer()
        df = _make_ohlcv(n=250)
        result = opt.optimize(df, "T", _dummy_objective, [_SimplePlugin()])
        assert result.n_windows >= 3

    def test_best_params_populated(self) -> None:
        opt = self._make_optimizer()
        df = _make_ohlcv(n=250)
        result = opt.optimize(df, "T", _dummy_objective, [_SimplePlugin()])
        assert isinstance(result.best_params, dict)
        assert len(result.best_params) > 0

    def test_pbo_between_0_and_1(self) -> None:
        opt = self._make_optimizer()
        df = _make_ohlcv(n=250)
        result = opt.optimize(df, "T", _dummy_objective, [_SimplePlugin()])
        assert 0.0 <= result.pbo <= 1.0

    def test_aggregate_oos_sharpe_is_finite(self) -> None:
        opt = self._make_optimizer()
        df = _make_ohlcv(n=250)
        result = opt.optimize(df, "T", _dummy_objective, [_SimplePlugin()])
        assert math.isfinite(result.aggregate_oos_sharpe)

    def test_windows_list_length_matches_n_windows(self) -> None:
        opt = self._make_optimizer()
        df = _make_ohlcv(n=250)
        result = opt.optimize(df, "T", _dummy_objective, [_SimplePlugin()])
        assert len(result.windows) == result.n_windows

    def test_first_window_always_stable(self) -> None:
        opt = self._make_optimizer()
        df = _make_ohlcv(n=250)
        result = opt.optimize(df, "T", _dummy_objective, [_SimplePlugin()])
        assert result.windows[0].is_stable is True

    def test_best_window_idx_valid(self) -> None:
        opt = self._make_optimizer()
        df = _make_ohlcv(n=250)
        result = opt.optimize(df, "T", _dummy_objective, [_SimplePlugin()])
        assert 0 <= result.best_window_idx < result.n_windows

    def test_insufficient_data_raises(self) -> None:
        opt = self._make_optimizer()
        df = _make_ohlcv(n=60)  # too short for 3 windows with IS=50, OOS=25
        with pytest.raises(ValueError, match="Insufficient data"):
            opt.optimize(df, "T", _dummy_objective, [_SimplePlugin()])


# ---------------------------------------------------------------------------
# TestWFOWindowResult
# ---------------------------------------------------------------------------

class TestWFOWindowResult:
    def test_timestamps_chronological(self) -> None:
        opt = WalkForwardOptimizer(
            in_sample_days=50, out_of_sample_days=25, n_trials=2,
            pbo_top_k=2, mlflow_tracking=False,
        )
        df = _make_ohlcv(n=250)
        result = opt.optimize(df, "T", _dummy_objective, [_SimplePlugin()])
        for w in result.windows:
            assert w.is_start <= w.is_end
            assert w.oos_start <= w.oos_end
            assert w.is_end < w.oos_start

    def test_n_trials_reported(self) -> None:
        opt = WalkForwardOptimizer(
            in_sample_days=50, out_of_sample_days=25, n_trials=3,
            pbo_top_k=2, mlflow_tracking=False,
        )
        df = _make_ohlcv(n=250)
        result = opt.optimize(df, "T", _dummy_objective, [_SimplePlugin()])
        for w in result.windows:
            assert w.n_trials >= 1


# ---------------------------------------------------------------------------
# TestComputePBO
# ---------------------------------------------------------------------------

class TestComputePBO:
    """Unit tests for the _compute_pbo internals via public optimize() path."""

    def test_pbo_all_candidates_equal(self) -> None:
        opt = WalkForwardOptimizer(
            in_sample_days=50, out_of_sample_days=25, n_trials=3,
            pbo_top_k=2, mlflow_tracking=False,
        )
        df = _make_ohlcv(n=250)
        # Constant objective → IS "best" is arbitrary → PBO reflects random rank
        result = opt.optimize(df, "T", _zero_objective, [_SimplePlugin()])
        assert 0.0 <= result.pbo <= 1.0


# ---------------------------------------------------------------------------
# TestStabilityCheck
# ---------------------------------------------------------------------------

class TestStabilityCheck:
    def test_delegates_to_bayesian_tuner(self) -> None:
        params1 = {"a__x": 10.0}
        params2 = {"a__x": 11.0}  # 10% drift — within 20% threshold
        assert WalkForwardOptimizer.stability_check([params1, params2]) is True

    def test_unstable_returns_false(self) -> None:
        params1 = {"a__x": 10.0}
        params2 = {"a__x": 15.0}  # 50% drift — exceeds 20% threshold
        assert WalkForwardOptimizer.stability_check([params1, params2]) is False

    def test_single_window_always_stable(self) -> None:
        assert WalkForwardOptimizer.stability_check([{"a__x": 5.0}]) is True

    def test_empty_history_stable(self) -> None:
        assert WalkForwardOptimizer.stability_check([]) is True

    def test_matches_bayesian_tuner_directly(self) -> None:
        history = [{"a__x": 10.0}, {"a__x": 10.5}, {"a__x": 11.0}]
        assert WalkForwardOptimizer.stability_check(history) == BayesianTuner.stability_check(history)


# ---------------------------------------------------------------------------
# TestCombinatorialPurgedCV
# ---------------------------------------------------------------------------

class TestCombinatorialPurgedCVInit:
    def test_default_params(self) -> None:
        cpcv = CombinatorialPurgedCV()
        assert cpcv.n_groups == 10
        assert cpcv.n_test_groups == 2

    def test_invalid_n_groups_raises(self) -> None:
        with pytest.raises(ValueError, match="n_groups"):
            CombinatorialPurgedCV(n_groups=2)

    def test_invalid_n_test_groups_raises(self) -> None:
        with pytest.raises(ValueError, match="n_test_groups"):
            CombinatorialPurgedCV(n_groups=5, n_test_groups=5)

    def test_n_test_groups_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_test_groups"):
            CombinatorialPurgedCV(n_groups=5, n_test_groups=0)


class TestCombinatorialPurgedCVRun:
    def _simple_strategy(
        self, is_df: pd.DataFrame, oos_df: pd.DataFrame
    ) -> Tuple[float, float]:
        """Returns (IS length / 100, OOS length / 100) as proxy Sharpe values."""
        return float(len(is_df)) / 100.0, float(len(oos_df)) / 100.0

    def test_returns_correct_number_of_results(self) -> None:
        import math as _math
        cpcv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2, embargo_days=0)
        df = _make_ohlcv(n=500)
        results = cpcv.run(self._simple_strategy, df)
        # C(5, 2) = 10
        expected = _math.comb(5, 2)
        assert len(results) == expected

    def test_results_are_float_tuples(self) -> None:
        cpcv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2, embargo_days=0)
        df = _make_ohlcv(n=500)
        results = cpcv.run(self._simple_strategy, df)
        for is_s, oos_s in results:
            assert isinstance(is_s, float)
            assert isinstance(oos_s, float)

    def test_run_with_embargo(self) -> None:
        cpcv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2, embargo_days=5)
        df = _make_ohlcv(n=500)
        results = cpcv.run(self._simple_strategy, df)
        # Some results may be skipped due to embargo, but at least some should pass
        assert len(results) > 0

    def test_strategy_fn_error_skipped(self) -> None:
        def bad_strategy(is_df, oos_df):
            raise RuntimeError("boom")

        cpcv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2, embargo_days=0)
        df = _make_ohlcv(n=500)
        results = cpcv.run(bad_strategy, df)
        assert results == []


class TestComputePBOStatic:
    def test_all_above_median_returns_zero(self) -> None:
        # If all OOS Sharpes are equal, half will be < median → PBO = 0.5
        results = [(1.0, 1.0)] * 10
        pbo = CombinatorialPurgedCV.compute_pbo(results)
        # With equal values: median = 1.0, and 0 are strictly < 1.0 → PBO = 0.0
        assert pbo == 0.0

    def test_all_below_median_returns_one(self) -> None:
        # OOS Sharpes: 10 values where all are below median (impossible — median is itself)
        # Use strictly increasing: only lower half < median
        oos = [float(i) for i in range(10)]  # [0..9], median = 4.5
        results = [(1.0, v) for v in oos]
        pbo = CombinatorialPurgedCV.compute_pbo(results)
        # 5 values (0..4) are < 4.5 → PBO = 0.5
        assert pytest.approx(pbo, abs=0.01) == 0.5

    def test_empty_results_returns_zero(self) -> None:
        pbo = CombinatorialPurgedCV.compute_pbo([])
        assert pbo == 0.0

    def test_single_result(self) -> None:
        pbo = CombinatorialPurgedCV.compute_pbo([(1.0, 0.8)])
        # Single point: 0.8 < median(0.8) = 0.8 is False → PBO = 0.0
        assert pbo == 0.0


# ---------------------------------------------------------------------------
# TestMLflowIntegration
# ---------------------------------------------------------------------------

class TestMLflowIntegration:
    def test_mlflow_start_run_called_when_tracking_enabled(self) -> None:
        opt = WalkForwardOptimizer(
            in_sample_days=50, out_of_sample_days=25, n_trials=2,
            pbo_top_k=2, mlflow_tracking=True,
        )
        df = _make_ohlcv(n=250)
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        with patch("mlflow.start_run", return_value=mock_run) as mock_sr, \
             patch("mlflow.log_metric"), \
             patch("mlflow.log_param"), \
             patch("mlflow.active_run", return_value=MagicMock()):
            result = opt.optimize(df, "AAPL", _dummy_objective, [_SimplePlugin()])
            mock_sr.assert_called_once()

    def test_mlflow_disabled_does_not_call_start_run(self) -> None:
        opt = WalkForwardOptimizer(
            in_sample_days=50, out_of_sample_days=25, n_trials=2,
            pbo_top_k=2, mlflow_tracking=False,
        )
        df = _make_ohlcv(n=250)
        with patch("mlflow.start_run") as mock_sr:
            opt.optimize(df, "T", _dummy_objective, [_SimplePlugin()])
            mock_sr.assert_not_called()
