"""Bayesian hyperparameter tuning via Optuna (Task 2.5).

Wraps Optuna's TPE sampler with:
  - Automatic search-space construction from each plugin's
    ``get_tunable_params()`` declaration.
  - Per-trial MLflow logging (best-effort; never blocks optimization).
  - A ``stability_check()`` utility that flags parameter sets that drift
    more than a configurable threshold across adjacent walk-forward windows.

The ``BayesianTuner`` is a pure search engine — it knows nothing about
OHLCV data or trade simulation. Callers (e.g. ``WalkForwardOptimizer``)
are responsible for supplying an ``objective_fn(params) -> float`` that
scores a parameter set on OOS data and returns a Sharpe ratio (or any
scalar metric to maximise).

Typical usage::

    from src.plugins.indicators.rsi import RSIIndicator
    from src.plugins.indicators.macd import MACDIndicator
    from src.tuning.bayesian_tuner import BayesianTuner

    plugins = [RSIIndicator(), MACDIndicator()]
    tuner   = BayesianTuner(plugins, n_trials=100)

    def my_objective(params):
        # params == {"rsi__period": 14, "macd__fast_period": 12, ...}
        per_plugin = tuner.unpack_params(params)
        # ... run strategy, return OOS Sharpe ...
        return sharpe

    result = tuner.tune(my_objective)
    print(result.best_params, result.best_value)

    # Parameter stability across two walk-forward windows:
    stable = BayesianTuner.stability_check(
        [window1_best_params, window2_best_params]
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import optuna

from src.plugins.base import IndicatorPlugin, ParamSpec, SmoothingPlugin

logger = logging.getLogger(__name__)

# Silence Optuna's per-trial logging by default; callers may override.
optuna.logging.set_verbosity(optuna.logging.WARNING)

_Plugin = Union[IndicatorPlugin, SmoothingPlugin]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Outcome of a single :meth:`BayesianTuner.tune` call.

    Attributes:
        best_params: Flat namespaced dict ``{plugin__param: value}`` of the
            best trial's parameters.
        best_value: Objective value (e.g. Sharpe ratio) of the best trial.
        n_trials: Number of trials actually evaluated.
        study_name: Name of the underlying Optuna study.
        param_history: List of param dicts — one entry per completed trial,
            in evaluation order.
        value_history: Corresponding list of objective values.
    """

    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    study_name: str
    param_history: List[Dict[str, Any]] = field(default_factory=list)
    value_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BayesianTuner
# ---------------------------------------------------------------------------

class BayesianTuner:
    """Optuna-backed Bayesian hyperparameter optimiser.

    Builds a unified search space by collecting ``get_tunable_params()``
    from every plugin and prefixing each parameter name with the plugin's
    ``name`` attribute (e.g. ``"rsi__period"``, ``"macd__fast_period"``).
    This avoids name collisions across plugins that share parameter names
    like ``period``.

    Args:
        plugins: List of indicator or smoothing plugins whose parameters
            should be included in the search space.
        n_trials: Number of Optuna trials to run (default 100).
        study_name: Name of the Optuna study.  Auto-generated when ``None``.
        mlflow_tracking: Whether to log each trial to the active MLflow run
            (best-effort; never raises).
        seed: Random seed passed to the TPE sampler for reproducibility.

    Raises:
        ValueError: If ``plugins`` is empty (no search space to optimise).
    """

    def __init__(
        self,
        plugins: List[_Plugin],
        n_trials: int = 100,
        study_name: Optional[str] = None,
        mlflow_tracking: bool = True,
        seed: int = 42,
    ) -> None:
        if not plugins:
            raise ValueError("plugins must not be empty.")
        self._plugins = list(plugins)
        self._n_trials = n_trials
        self._study_name = study_name or _auto_study_name()
        self._mlflow_tracking = mlflow_tracking
        self._seed = seed
        self._space: Dict[str, ParamSpec] = self._build_search_space()

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    @property
    def search_space(self) -> Dict[str, ParamSpec]:
        """Return a copy of the aggregated parameter search space."""
        return dict(self._space)

    def tune(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        timeout: Optional[float] = None,
    ) -> OptimizationResult:
        """Run the Optuna study and return the best result.

        Each trial calls ``objective_fn(params)`` where ``params`` is a flat
        dict with namespaced keys (``plugin__param``).  If a trial raises an
        exception, it is marked as failed and the study continues.

        After all trials complete, best params and best value are logged to
        the active MLflow run (if one exists and ``mlflow_tracking=True``).

        Args:
            objective_fn: Callable that maps a param dict → scalar metric to
                **maximise** (typically OOS Sharpe ratio).
            timeout: Optional wall-clock timeout in seconds.

        Returns:
            :class:`OptimizationResult` with best params, best value, and
            per-trial history.
        """
        sampler = optuna.samplers.TPESampler(seed=self._seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=self._study_name,
        )

        param_history: List[Dict[str, Any]] = []
        value_history: List[float] = []

        def _objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            try:
                value = float(objective_fn(params))
            except Exception as exc:
                logger.debug(
                    "Trial %d raised %s: %s — marking failed.",
                    trial.number, type(exc).__name__, exc,
                )
                raise optuna.exceptions.TrialPruned() from exc

            param_history.append(params)
            value_history.append(value)

            if self._mlflow_tracking:
                self._log_trial(trial.number, params, value)

            return value

        study.optimize(
            _objective,
            n_trials=self._n_trials,
            timeout=timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )

        # Retrieve best result; guard against all-failed studies.
        # study.best_trial raises ValueError (not returns None) when no trial completed.
        try:
            best_params: Dict[str, Any] = dict(study.best_params)
            best_value = float(study.best_value)
        except ValueError:
            best_params = {}
            best_value = float("-inf")

        result = OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            n_trials=len(study.trials),
            study_name=self._study_name,
            param_history=param_history,
            value_history=value_history,
        )

        # Log summary to MLflow
        if self._mlflow_tracking:
            self._log_summary(result)

        logger.info(
            "BayesianTuner: %d trials, best_value=%.4f, study=%s",
            result.n_trials, result.best_value, result.study_name,
        )
        return result

    def unpack_params(
        self, flat_params: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Convert flat namespaced params to a per-plugin dict.

        Example::

            flat = {"rsi__period": 14, "macd__fast_period": 12}
            tuner.unpack_params(flat)
            # → {"rsi": {"period": 14}, "macd": {"fast_period": 12}}

        Args:
            flat_params: Flat dict as returned in :attr:`OptimizationResult.best_params`.

        Returns:
            Nested dict ``{plugin_name: {param_name: value}}``.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for key, value in flat_params.items():
            if "__" in key:
                plugin_name, param_name = key.split("__", 1)
            else:
                plugin_name, param_name = "unknown", key
            result.setdefault(plugin_name, {})[param_name] = value
        return result

    @staticmethod
    def stability_check(
        param_histories: List[Dict[str, Any]],
        threshold: float = 0.20,
    ) -> bool:
        """Check that best parameters are stable across adjacent windows.

        For each consecutive pair of parameter dicts (representing best params
        from adjacent walk-forward folds), computes the relative drift of
        every numeric parameter.  Returns ``False`` if any param drifts more
        than ``threshold`` (e.g. 0.20 = 20 %).

        Args:
            param_histories: Ordered list of best-param dicts — one per
                walk-forward fold (earliest first).
            threshold: Maximum allowed relative drift per parameter
                (default 0.20).

        Returns:
            ``True`` if all adjacent pairs are within the drift limit;
            ``False`` if any parameter exceeds it.
        """
        if len(param_histories) < 2:
            return True

        for prev, curr in zip(param_histories, param_histories[1:]):
            shared = set(prev.keys()) & set(curr.keys())
            for key in shared:
                pv, cv = prev[key], curr[key]
                if not isinstance(pv, (int, float)) or not isinstance(cv, (int, float)):
                    continue
                if pv == 0:
                    # Any change from zero counts as unstable if curr != 0
                    if cv != 0:
                        return False
                    continue
                drift = abs(cv - pv) / abs(pv)
                if drift > threshold:
                    logger.debug(
                        "Stability check failed for '%s': prev=%s curr=%s drift=%.2f%%",
                        key, pv, cv, drift * 100,
                    )
                    return False
        return True

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_search_space(self) -> Dict[str, ParamSpec]:
        """Aggregate tunable params from all plugins into a flat namespaced dict."""
        space: Dict[str, ParamSpec] = {}
        for plugin in self._plugins:
            plugin_name = plugin.name
            for param_name, spec in plugin.get_tunable_params().items():
                key = f"{plugin_name}__{param_name}"
                if key in space:
                    logger.warning(
                        "Duplicate search-space key '%s' from plugin '%s' — overwriting.",
                        key, plugin_name,
                    )
                space[key] = spec
        return space

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Ask Optuna to sample one value for each parameter in the space."""
        params: Dict[str, Any] = {}
        for key, spec in self._space.items():
            if spec.type == "int":
                params[key] = trial.suggest_int(key, int(spec.low), int(spec.high))
            elif spec.type == "float":
                params[key] = trial.suggest_float(key, float(spec.low), float(spec.high))
            elif spec.type == "categorical":
                params[key] = trial.suggest_categorical(key, list(spec.choices))
        return params

    def _log_trial(
        self,
        trial_num: int,
        params: Dict[str, Any],
        value: float,
    ) -> None:
        """Log a single trial to the active MLflow run (best-effort)."""
        try:
            import mlflow

            if mlflow.active_run() is None:
                return
            mlflow.log_metric("tuning/objective", value, step=trial_num)
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"tuning/{k}", float(v), step=trial_num)
        except Exception as exc:
            logger.debug("MLflow trial %d logging failed: %s", trial_num, exc)

    def _log_summary(self, result: OptimizationResult) -> None:
        """Log best-trial summary to the active MLflow run (best-effort)."""
        try:
            import mlflow

            if mlflow.active_run() is None:
                return
            mlflow.log_metric("tuning/best_value", result.best_value)
            mlflow.log_metric("tuning/n_trials", float(result.n_trials))
            for k, v in result.best_params.items():
                if isinstance(v, (int, float)):
                    mlflow.log_param(f"best/{k}", v)
                else:
                    mlflow.log_param(f"best/{k}", str(v))
        except Exception as exc:
            logger.debug("MLflow summary logging failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_study_name() -> str:
    """Generate a study name from the current timestamp."""
    from datetime import datetime

    return f"bayesian_tuning_{datetime.now():%Y%m%d_%H%M%S}"


def compute_sharpe(returns: Any, annualisation: float = 252.0) -> float:
    """Compute the annualised Sharpe ratio from a daily-returns series.

    Convenience function for use inside ``objective_fn`` implementations.

    Args:
        returns: Array-like of daily returns (not log-returns).
        annualisation: Trading days per year (default 252).

    Returns:
        Annualised Sharpe ratio.  Returns 0.0 if ``std == 0`` or fewer
        than 2 observations.
    """
    import numpy as np

    arr = np.asarray(returns, dtype=float)
    if len(arr) < 2:
        return 0.0
    std = arr.std(ddof=1)
    if std == 0.0:
        return 0.0
    return float(arr.mean() / std * (annualisation ** 0.5))
