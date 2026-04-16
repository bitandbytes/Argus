"""Tests for BayesianTuner, OptimizationResult, and stability_check.

Covers:
  - Search space construction from plugins (namespace, deduplication)
  - tune() correctness: n_trials, best_value in search space, param history
  - MLflow per-trial logging (mocked to avoid real MLflow server)
  - stability_check: stable / unstable pairs, edge cases
  - unpack_params: flat → nested conversion
  - compute_sharpe utility
  - Input validation errors
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.plugins.base import IndicatorPlugin, ParamSpec, SmoothingPlugin, SmoothResult
from src.tuning.bayesian_tuner import (
    BayesianTuner,
    OptimizationResult,
    compute_sharpe,
)


# ---------------------------------------------------------------------------
# Minimal plugin stubs
# ---------------------------------------------------------------------------

class _IntPlugin(IndicatorPlugin):
    """One int param: x ∈ [1, 10]."""
    name = "plugin_a"
    category = "momentum"
    version = "0.0.1"
    output_column = "a"

    def compute(self, df, params): return df
    def normalize(self, values): return values

    def get_tunable_params(self):
        return {"x": ParamSpec("x", "int", low=1, high=10, default=5)}

    def get_default_params(self): return {"x": 5}


class _FloatPlugin(IndicatorPlugin):
    """One float param: y ∈ [0.0, 1.0]."""
    name = "plugin_b"
    category = "trend"
    version = "0.0.1"
    output_column = "b"

    def compute(self, df, params): return df
    def normalize(self, values): return values

    def get_tunable_params(self):
        return {"y": ParamSpec("y", "float", low=0.0, high=1.0, default=0.5)}

    def get_default_params(self): return {"y": 0.5}


class _CatPlugin(IndicatorPlugin):
    """One categorical param: mode ∈ {fast, slow}."""
    name = "plugin_c"
    category = "trend"
    version = "0.0.1"
    output_column = "c"

    def compute(self, df, params): return df
    def normalize(self, values): return values

    def get_tunable_params(self):
        return {
            "mode": ParamSpec(
                "mode", "categorical", choices=["fast", "slow"], default="fast"
            )
        }

    def get_default_params(self): return {"mode": "fast"}


class _SmoothPlugin(SmoothingPlugin):
    """Smoothing plugin with one int param."""
    name = "smoother_x"
    version = "0.0.1"

    def smooth(self, series, params): ...
    def get_tunable_params(self):
        return {"window": ParamSpec("window", "int", low=2, high=20, default=5)}


class _MultiParamPlugin(IndicatorPlugin):
    """Two params: fast ∈ [5, 50] and slow ∈ [20, 200]."""
    name = "multi"
    category = "trend"
    version = "0.0.1"
    output_column = "m"

    def compute(self, df, params): return df
    def normalize(self, values): return values

    def get_tunable_params(self):
        return {
            "fast": ParamSpec("fast", "int", low=5, high=50, default=12),
            "slow": ParamSpec("slow", "int", low=20, high=200, default=26),
        }

    def get_default_params(self): return {"fast": 12, "slow": 26}


# ---------------------------------------------------------------------------
# Objective helpers
# ---------------------------------------------------------------------------

def _quadratic_objective(params: Dict[str, Any]) -> float:
    """Maximised at plugin_a__x=5, plugin_b__y=0.5 → returns 1.0 at optimum."""
    x = params.get("plugin_a__x", 5)
    y = params.get("plugin_b__y", 0.5)
    return -(((x - 5) / 5.0) ** 2 + ((y - 0.5) / 0.5) ** 2)


def _constant_objective(params: Dict[str, Any]) -> float:
    return 0.42


def _crashing_objective(params: Dict[str, Any]) -> float:
    raise RuntimeError("Objective exploded!")


# ---------------------------------------------------------------------------
# TestSearchSpaceConstruction
# ---------------------------------------------------------------------------

class TestSearchSpaceConstruction:
    def test_single_plugin_namespaced_key(self) -> None:
        tuner = BayesianTuner([_IntPlugin()])
        assert "plugin_a__x" in tuner.search_space

    def test_multiple_plugins_all_keys_present(self) -> None:
        tuner = BayesianTuner([_IntPlugin(), _FloatPlugin()])
        space = tuner.search_space
        assert "plugin_a__x" in space
        assert "plugin_b__y" in space

    def test_multi_param_plugin_all_params_present(self) -> None:
        tuner = BayesianTuner([_MultiParamPlugin()])
        space = tuner.search_space
        assert "multi__fast" in space
        assert "multi__slow" in space

    def test_paramspec_types_preserved(self) -> None:
        tuner = BayesianTuner([_IntPlugin(), _FloatPlugin(), _CatPlugin()])
        space = tuner.search_space
        assert space["plugin_a__x"].type == "int"
        assert space["plugin_b__y"].type == "float"
        assert space["plugin_c__mode"].type == "categorical"

    def test_smoothing_plugin_included(self) -> None:
        tuner = BayesianTuner([_SmoothPlugin()])
        assert "smoother_x__window" in tuner.search_space

    def test_search_space_returns_copy(self) -> None:
        tuner = BayesianTuner([_IntPlugin()])
        space1 = tuner.search_space
        space1["injected"] = None  # mutate the copy
        assert "injected" not in tuner.search_space

    def test_raises_on_empty_plugins(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            BayesianTuner([])


# ---------------------------------------------------------------------------
# TestBayesianTunerTune
# ---------------------------------------------------------------------------

class TestBayesianTunerTune:
    def test_returns_optimization_result(self) -> None:
        tuner = BayesianTuner([_IntPlugin()], n_trials=5, mlflow_tracking=False)
        result = tuner.tune(_constant_objective)
        assert isinstance(result, OptimizationResult)

    def test_n_trials_respected(self) -> None:
        tuner = BayesianTuner([_IntPlugin()], n_trials=8, mlflow_tracking=False)
        result = tuner.tune(_constant_objective)
        assert result.n_trials == 8

    def test_best_value_is_highest_observed(self) -> None:
        """Constant objective → best_value should equal the constant."""
        tuner = BayesianTuner([_IntPlugin()], n_trials=10, mlflow_tracking=False)
        result = tuner.tune(_constant_objective)
        assert abs(result.best_value - 0.42) < 1e-6

    def test_best_params_keys_match_search_space(self) -> None:
        tuner = BayesianTuner(
            [_IntPlugin(), _FloatPlugin()], n_trials=5, mlflow_tracking=False
        )
        result = tuner.tune(_constant_objective)
        for key in result.best_params:
            assert key in tuner.search_space, f"Unexpected key '{key}' in best_params"

    def test_param_history_length_matches_n_trials(self) -> None:
        tuner = BayesianTuner([_IntPlugin()], n_trials=7, mlflow_tracking=False)
        result = tuner.tune(_constant_objective)
        assert len(result.param_history) == 7
        assert len(result.value_history) == 7

    def test_value_history_equals_objective_values(self) -> None:
        tuner = BayesianTuner([_IntPlugin()], n_trials=5, mlflow_tracking=False)
        result = tuner.tune(_constant_objective)
        assert all(abs(v - 0.42) < 1e-6 for v in result.value_history)

    def test_study_name_preserved_in_result(self) -> None:
        tuner = BayesianTuner(
            [_IntPlugin()], n_trials=3, study_name="my_study", mlflow_tracking=False
        )
        result = tuner.tune(_constant_objective)
        assert result.study_name == "my_study"

    def test_maximises_objective(self) -> None:
        """TPE should converge toward the quadratic's maximum."""
        tuner = BayesianTuner(
            [_IntPlugin(), _FloatPlugin()], n_trials=50, seed=0, mlflow_tracking=False
        )
        result = tuner.tune(_quadratic_objective)
        # Best value should be negative and close to 0 (maximum = 0 at x=5, y=0.5)
        assert result.best_value > -0.5, (
            f"Expected optimiser to improve, best_value={result.best_value:.4f}"
        )

    def test_crashing_objective_does_not_abort_study(self) -> None:
        """Trials that raise must be silently skipped, study must still return."""
        tuner = BayesianTuner([_IntPlugin()], n_trials=5, mlflow_tracking=False)
        # All trials will fail → best_params empty, best_value -inf
        result = tuner.tune(_crashing_objective)
        assert isinstance(result, OptimizationResult)

    def test_categorical_param_suggested_correctly(self) -> None:
        tuner = BayesianTuner([_CatPlugin()], n_trials=5, mlflow_tracking=False)
        result = tuner.tune(_constant_objective)
        assert result.best_params["plugin_c__mode"] in ("fast", "slow")

    def test_same_seed_same_best_params(self) -> None:
        """Determinism: same seed → same best params for same objective."""
        kwargs = dict(n_trials=10, seed=7, mlflow_tracking=False)
        result1 = BayesianTuner([_IntPlugin()], **kwargs).tune(_quadratic_objective)
        result2 = BayesianTuner([_IntPlugin()], **kwargs).tune(_quadratic_objective)
        assert result1.best_params == result2.best_params


# ---------------------------------------------------------------------------
# TestUnpackParams
# ---------------------------------------------------------------------------

class TestUnpackParams:
    def test_basic_unpack(self) -> None:
        tuner = BayesianTuner([_IntPlugin(), _FloatPlugin()], mlflow_tracking=False)
        flat = {"plugin_a__x": 7, "plugin_b__y": 0.3}
        unpacked = tuner.unpack_params(flat)
        assert unpacked == {"plugin_a": {"x": 7}, "plugin_b": {"y": 0.3}}

    def test_multi_param_plugin_unpack(self) -> None:
        tuner = BayesianTuner([_MultiParamPlugin()], mlflow_tracking=False)
        flat = {"multi__fast": 10, "multi__slow": 50}
        unpacked = tuner.unpack_params(flat)
        assert unpacked["multi"] == {"fast": 10, "slow": 50}

    def test_key_without_double_underscore(self) -> None:
        """Keys without __ go into 'unknown' namespace."""
        tuner = BayesianTuner([_IntPlugin()], mlflow_tracking=False)
        unpacked = tuner.unpack_params({"orphan_key": 99})
        assert unpacked["unknown"]["orphan_key"] == 99


# ---------------------------------------------------------------------------
# TestStabilityCheck
# ---------------------------------------------------------------------------

class TestStabilityCheck:
    def test_single_window_always_stable(self) -> None:
        assert BayesianTuner.stability_check([{"a__x": 10}]) is True

    def test_empty_history_stable(self) -> None:
        assert BayesianTuner.stability_check([]) is True

    def test_identical_params_stable(self) -> None:
        params = {"a__x": 10, "b__y": 0.5}
        assert BayesianTuner.stability_check([params, params]) is True

    def test_within_threshold_stable(self) -> None:
        """10 % drift on a 0.20 threshold → stable."""
        p1 = {"a__x": 10}
        p2 = {"a__x": 11}   # 10% drift
        assert BayesianTuner.stability_check([p1, p2], threshold=0.20) is True

    def test_exceeds_threshold_unstable(self) -> None:
        """50 % drift on a 0.20 threshold → unstable."""
        p1 = {"a__x": 10}
        p2 = {"a__x": 15}   # 50% drift
        assert BayesianTuner.stability_check([p1, p2], threshold=0.20) is False

    def test_three_windows_all_stable(self) -> None:
        windows = [{"a__x": 10}, {"a__x": 11}, {"a__x": 12}]
        assert BayesianTuner.stability_check(windows, threshold=0.20) is True

    def test_last_pair_unstable_returns_false(self) -> None:
        windows = [{"a__x": 10}, {"a__x": 11}, {"a__x": 100}]
        assert BayesianTuner.stability_check(windows, threshold=0.20) is False

    def test_zero_to_nonzero_unstable(self) -> None:
        """When prev == 0 and curr != 0, mark as unstable."""
        p1 = {"a__x": 0}
        p2 = {"a__x": 1}
        assert BayesianTuner.stability_check([p1, p2]) is False

    def test_zero_to_zero_stable(self) -> None:
        p1 = {"a__x": 0}
        p2 = {"a__x": 0}
        assert BayesianTuner.stability_check([p1, p2]) is True

    def test_categorical_params_ignored(self) -> None:
        """Non-numeric params are skipped; numeric drift decides stability."""
        p1 = {"mode": "fast", "a__x": 10}
        p2 = {"mode": "slow", "a__x": 10}
        assert BayesianTuner.stability_check([p1, p2]) is True

    def test_custom_threshold(self) -> None:
        p1 = {"a__x": 100}
        p2 = {"a__x": 150}  # 50% drift
        assert BayesianTuner.stability_check([p1, p2], threshold=0.60) is True
        assert BayesianTuner.stability_check([p1, p2], threshold=0.40) is False


# ---------------------------------------------------------------------------
# TestMLflowLogging
# ---------------------------------------------------------------------------

class TestMLflowLogging:
    def test_log_called_per_trial(self) -> None:
        """With an active MLflow run, log_metric should be called once per trial."""
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)

        tuner = BayesianTuner([_IntPlugin()], n_trials=3, mlflow_tracking=True)

        with (
            patch("mlflow.active_run", return_value=mock_run),
            patch("mlflow.log_metric") as mock_log_metric,
            patch("mlflow.log_param"),
        ):
            tuner.tune(_constant_objective)

        # At minimum the "tuning/objective" metric should be logged 3 times (once
        # per trial) plus 1 for the summary best_value.
        objective_calls = [
            c for c in mock_log_metric.call_args_list if "objective" in c.args[0]
        ]
        assert len(objective_calls) >= 3

    def test_no_log_when_no_active_run(self) -> None:
        """When no MLflow run is active, log_metric must never be called."""
        tuner = BayesianTuner([_IntPlugin()], n_trials=3, mlflow_tracking=True)

        with (
            patch("mlflow.active_run", return_value=None),
            patch("mlflow.log_metric") as mock_log_metric,
        ):
            tuner.tune(_constant_objective)

        mock_log_metric.assert_not_called()

    def test_no_log_when_tracking_disabled(self) -> None:
        tuner = BayesianTuner([_IntPlugin()], n_trials=3, mlflow_tracking=False)

        with patch("mlflow.log_metric") as mock_log_metric:
            tuner.tune(_constant_objective)

        mock_log_metric.assert_not_called()


# ---------------------------------------------------------------------------
# TestComputeSharpe
# ---------------------------------------------------------------------------

class TestComputeSharpe:
    def test_positive_returns_positive_sharpe(self) -> None:
        import numpy as np
        rng = np.random.default_rng(0)
        # Positive mean with some noise so std > 0
        returns = list(0.01 + rng.normal(0, 0.005, 252))
        assert compute_sharpe(returns) > 0.0

    def test_zero_std_returns_zero(self) -> None:
        assert compute_sharpe([0.0, 0.0, 0.0]) == 0.0

    def test_single_observation_returns_zero(self) -> None:
        assert compute_sharpe([0.05]) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert compute_sharpe([]) == 0.0

    def test_annualisation_scales_correctly(self) -> None:
        """Double annualisation factor → Sharpe scales by √2."""
        import math
        returns = [0.001] * 100
        s252 = compute_sharpe(returns, annualisation=252.0)
        s1008 = compute_sharpe(returns, annualisation=1008.0)
        assert abs(s1008 / s252 - math.sqrt(4)) < 1e-6
