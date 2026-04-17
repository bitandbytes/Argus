"""Walk-Forward Optimization (Task 2.6).

Implements ``WalkForwardOptimizer``: a rolling-window parameter optimization
framework that separates in-sample (IS) tuning from out-of-sample (OOS)
evaluation to produce overfitting-resistant parameter sets for each cluster.

Design:
  - Caller supplies an ``objective_fn(params, data_df) -> float`` that runs any
    backtest and returns a Sharpe ratio.  The WFO is completely agnostic to the
    internal implementation of that function.
  - For each rolling window, Optuna is invoked via ``BayesianTuner`` on the IS
    slice.  The top-k IS candidates are also evaluated on the OOS slice to
    compute the Probability of Backtest Overfitting (PBO).
  - PBO = fraction of windows where the IS-best parameter set ranks below the
    median OOS Sharpe across the top-k candidates.  A PBO < 0.40 is required
    before cluster parameters are promoted to production.
  - Stability is checked via ``BayesianTuner.stability_check`` across adjacent
    windows; any parameter drifting > 20 % triggers an ``is_stable=False`` flag.

Typical usage::

    from src.tuning.walk_forward import WalkForwardOptimizer

    optimizer = WalkForwardOptimizer(in_sample_days=252, out_of_sample_days=126)
    result = optimizer.optimize(df, ticker="AAPL", objective_fn=my_obj, plugins=plugins)
    if result.pbo < 0.40 and result.is_stable:
        print("Parameters validated:", result.best_params)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.plugins.base import IndicatorPlugin, SmoothingPlugin
from src.tuning.bayesian_tuner import BayesianTuner, OptimizationResult

logger = logging.getLogger(__name__)

_Plugin = Union[IndicatorPlugin, SmoothingPlugin]
_ParamDict = Dict[str, Any]
_ObjectiveFn = Callable[[_ParamDict, pd.DataFrame], float]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WFOWindowResult:
    """Results from a single in-sample / out-of-sample window.

    Attributes:
        window_idx: Zero-based index of this window.
        is_start: First date of the in-sample slice.
        is_end: Last date of the in-sample slice.
        oos_start: First date of the out-of-sample slice.
        oos_end: Last date of the out-of-sample slice.
        best_params: Flat namespaced params (``{plugin__param: value}``) that
            yielded the best IS Sharpe.
        is_sharpe: Best IS Sharpe from the Optuna study.
        oos_sharpe: OOS Sharpe when ``best_params`` are applied to the OOS slice.
        n_trials: Number of Optuna trials actually evaluated.
        is_stable: ``True`` if parameter drift vs the previous window is < 20 %.
            Always ``True`` for window 0 (no prior window to compare against).
    """

    window_idx: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    best_params: _ParamDict
    is_sharpe: float
    oos_sharpe: float
    n_trials: int
    is_stable: bool


@dataclass
class WFOResult:
    """Aggregate results from a full walk-forward optimization run.

    Attributes:
        ticker: Stock symbol this WFO was run for.
        windows: Ordered list of per-window results (earliest first).
        best_params: Params from the window with the highest OOS Sharpe.
        aggregate_oos_sharpe: Mean OOS Sharpe across all windows.
        pbo: Probability of Backtest Overfitting ∈ [0, 1].  Computed from the
            top-k IS candidate parameter sets per window.
        is_stable: ``True`` iff every window passes the parameter-drift check.
        n_windows: Total number of rolling windows evaluated.
        best_window_idx: Index of the window that produced ``best_params``.
    """

    ticker: str
    windows: List[WFOWindowResult]
    best_params: _ParamDict
    aggregate_oos_sharpe: float
    pbo: float
    is_stable: bool
    n_windows: int
    best_window_idx: int = field(default=0)


# ---------------------------------------------------------------------------
# WalkForwardOptimizer
# ---------------------------------------------------------------------------

class WalkForwardOptimizer:
    """Rolling-window parameter optimizer with PBO computation.

    Splits a time-series DataFrame into sequential in-sample (IS) and
    out-of-sample (OOS) windows, optimises parameters on each IS window via
    ``BayesianTuner``, evaluates on the OOS window, and computes the
    Probability of Backtest Overfitting (PBO) from the top-k IS candidate
    parameter sets.

    Args:
        in_sample_days: Number of trading days in each IS window (default 252 ≈ 1 year).
        out_of_sample_days: Number of trading days in each OOS window (default 126 ≈ 6 months).
            Also the step size — windows advance by this amount each iteration so
            OOS periods are non-overlapping.
        n_trials: Number of Optuna trials per IS window (default 100).
        pbo_top_k: Number of top IS candidates to evaluate on OOS for PBO
            computation (default 10).  Must be ≥ 2 for meaningful PBO.
        seed: Random seed for reproducibility (passed to ``BayesianTuner``).
        mlflow_tracking: Whether to log per-run metrics to MLflow (default True).
    """

    def __init__(
        self,
        in_sample_days: int = 252,
        out_of_sample_days: int = 126,
        n_trials: int = 100,
        pbo_top_k: int = 10,
        seed: int = 42,
        mlflow_tracking: bool = True,
    ) -> None:
        if in_sample_days < 30:
            raise ValueError(f"in_sample_days must be >= 30, got {in_sample_days}")
        if out_of_sample_days < 10:
            raise ValueError(f"out_of_sample_days must be >= 10, got {out_of_sample_days}")
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")
        if pbo_top_k < 2:
            raise ValueError(f"pbo_top_k must be >= 2 for meaningful PBO, got {pbo_top_k}")

        self.in_sample_days = in_sample_days
        self.out_of_sample_days = out_of_sample_days
        self.n_trials = n_trials
        self.pbo_top_k = pbo_top_k
        self.seed = seed
        self.mlflow_tracking = mlflow_tracking

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def optimize(
        self,
        df: pd.DataFrame,
        ticker: str,
        objective_fn: _ObjectiveFn,
        plugins: List[_Plugin],
        run_name_prefix: Optional[str] = None,
    ) -> WFOResult:
        """Run walk-forward optimization and return validated parameters.

        For each rolling window the method:
          1. Calls ``BayesianTuner.tune()`` on the IS slice.
          2. Evaluates the top-k IS candidates on the OOS slice.
          3. Checks parameter stability against the previous window.

        After all windows, PBO is computed and the result with the highest OOS
        Sharpe is selected as ``best_params``.

        If ``mlflow_tracking=True`` the entire run is wrapped in a single
        ``mlflow.start_run()`` with per-window metrics logged as steps.

        Args:
            df: Full OHLCV DataFrame indexed by date.  Must be long enough for
                at least 3 rolling windows (≥ ``in_sample_days + 3 * out_of_sample_days``
                rows).
            ticker: Stock symbol — used for logging and MLflow run names.
            objective_fn: ``Callable[[params, data_df], float]`` that evaluates a
                parameter set on a given data slice and returns a Sharpe ratio.
                Receives flat namespaced params ``{plugin__param: value}``.
            plugins: Plugins whose ``get_tunable_params()`` defines the Optuna
                search space.
            run_name_prefix: Optional prefix for the MLflow run name.

        Returns:
            :class:`WFOResult` with per-window details, aggregate OOS Sharpe,
            PBO, stability flag, and the best validated parameter set.

        Raises:
            ValueError: If ``df`` is too short for at least 3 windows.
        """
        windows = self._make_windows(df)
        logger.info(
            "WFO starting: ticker=%s, %d windows (%d IS + %d OOS days each)",
            ticker, len(windows), self.in_sample_days, self.out_of_sample_days,
        )

        run_name = f"{run_name_prefix or 'wfo'}_{ticker}"

        def _run_all() -> WFOResult:
            window_results: List[WFOWindowResult] = []
            per_window_candidates: List[List[Tuple[_ParamDict, float, float]]] = []
            prev_best_params: Optional[_ParamDict] = None

            for idx, (is_df, oos_df) in enumerate(windows):
                logger.info(
                    "Window %d/%d: IS %s→%s, OOS %s→%s",
                    idx + 1, len(windows),
                    is_df.index[0].date(), is_df.index[-1].date(),
                    oos_df.index[0].date(), oos_df.index[-1].date(),
                )
                win_result, candidates = self._run_window(
                    is_df, oos_df, idx, objective_fn, plugins, prev_best_params
                )
                window_results.append(win_result)
                per_window_candidates.append(candidates)
                prev_best_params = win_result.best_params

                if self.mlflow_tracking:
                    self._log_window(idx, win_result)

            pbo = self._compute_pbo(per_window_candidates)
            oos_sharpes = [w.oos_sharpe for w in window_results]
            agg_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
            best_idx = int(np.argmax(oos_sharpes)) if oos_sharpes else 0
            overall_stable = all(w.is_stable for w in window_results)

            result = WFOResult(
                ticker=ticker,
                windows=window_results,
                best_params=window_results[best_idx].best_params,
                aggregate_oos_sharpe=agg_oos,
                pbo=pbo,
                is_stable=overall_stable,
                n_windows=len(window_results),
                best_window_idx=best_idx,
            )

            if self.mlflow_tracking:
                self._log_summary(result)

            logger.info(
                "WFO complete: ticker=%s, windows=%d, agg_oos_sharpe=%.4f, pbo=%.4f, stable=%s",
                ticker, result.n_windows, result.aggregate_oos_sharpe,
                result.pbo, result.is_stable,
            )
            return result

        if self.mlflow_tracking:
            try:
                import mlflow
                with mlflow.start_run(run_name=run_name):
                    return _run_all()
            except Exception as exc:
                logger.warning("MLflow run failed (%s) — running without tracking.", exc)

        return _run_all()

    @staticmethod
    def stability_check(
        param_history: List[_ParamDict],
        threshold: float = 0.20,
    ) -> bool:
        """Check parameter stability across adjacent walk-forward windows.

        Delegates to :meth:`BayesianTuner.stability_check`.

        Args:
            param_history: Ordered list of best-param dicts (one per window).
            threshold: Maximum allowed relative drift per parameter (default 0.20).

        Returns:
            ``True`` if all adjacent pairs are within the drift limit.
        """
        return BayesianTuner.stability_check(param_history, threshold)

    # ------------------------------------------------------------------ #
    # Window construction                                                  #
    # ------------------------------------------------------------------ #

    def _make_windows(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Construct non-overlapping IS/OOS window pairs.

        The first window starts at row 0.  Each subsequent window advances by
        ``out_of_sample_days`` so OOS periods are strictly non-overlapping.
        The last window is dropped if its OOS slice is shorter than half of
        ``out_of_sample_days``.

        Args:
            df: Full OHLCV DataFrame.

        Returns:
            List of ``(is_df, oos_df)`` tuples, ordered chronologically.

        Raises:
            ValueError: If fewer than 3 complete windows can be constructed.
        """
        windows: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        step = self.out_of_sample_days
        total = len(df)
        i = 0

        while True:
            is_end = i + self.in_sample_days
            oos_end = is_end + self.out_of_sample_days

            if is_end >= total:
                break

            actual_oos_end = min(oos_end, total)
            oos_len = actual_oos_end - is_end

            if oos_len < max(self.out_of_sample_days // 2, 10):
                break

            is_df = df.iloc[i:is_end]
            oos_df = df.iloc[is_end:actual_oos_end]
            windows.append((is_df, oos_df))
            i += step

        if len(windows) < 3:
            min_rows = self.in_sample_days + 3 * self.out_of_sample_days
            raise ValueError(
                f"Insufficient data for walk-forward optimization: {total} rows available, "
                f"need at least {min_rows} rows for 3 windows "
                f"(in_sample_days={self.in_sample_days}, "
                f"out_of_sample_days={self.out_of_sample_days})."
            )

        return windows

    # ------------------------------------------------------------------ #
    # Per-window optimization                                              #
    # ------------------------------------------------------------------ #

    def _run_window(
        self,
        is_df: pd.DataFrame,
        oos_df: pd.DataFrame,
        window_idx: int,
        objective_fn: _ObjectiveFn,
        plugins: List[_Plugin],
        prev_best_params: Optional[_ParamDict],
    ) -> Tuple[WFOWindowResult, List[Tuple[_ParamDict, float, float]]]:
        """Optimize on IS data and evaluate top-k candidates on OOS data.

        Args:
            is_df: In-sample data slice.
            oos_df: Out-of-sample data slice.
            window_idx: Zero-based window index.
            objective_fn: ``(params, data_df) -> float`` callable.
            plugins: Plugins defining the Optuna search space.
            prev_best_params: Best params from the previous window (used for
                stability check).  ``None`` for window 0.

        Returns:
            Tuple of:
              - :class:`WFOWindowResult` for this window.
              - List of ``(params, is_sharpe, oos_sharpe)`` for the top-k
                IS candidates (used for PBO computation).
        """
        study_name = f"wfo_window_{window_idx}_{id(is_df)}"
        tuner = BayesianTuner(
            plugins,
            n_trials=self.n_trials,
            study_name=study_name,
            mlflow_tracking=False,  # WFO handles its own MLflow logging
            seed=self.seed + window_idx,  # different seed per window
        )

        def is_objective(params: _ParamDict) -> float:
            return objective_fn(params, is_df)

        opt_result: OptimizationResult = tuner.tune(is_objective)
        logger.debug(
            "Window %d IS: best_value=%.4f, n_trials=%d",
            window_idx, opt_result.best_value, opt_result.n_trials,
        )

        # Build top-k IS candidates from Optuna's trial history.
        candidates = self._select_top_k_candidates(
            opt_result.param_history,
            opt_result.value_history,
            opt_result.best_params,
        )

        # Evaluate each top-k candidate on OOS data.
        candidates_with_oos: List[Tuple[_ParamDict, float, float]] = []
        for params, is_sharpe in candidates:
            try:
                oos_sharpe = float(objective_fn(params, oos_df))
            except Exception as exc:
                logger.debug("OOS eval failed for candidate: %s", exc)
                oos_sharpe = float("-inf")
            candidates_with_oos.append((params, is_sharpe, oos_sharpe))

        # OOS Sharpe for the IS-best (first candidate after sorting).
        best_oos_sharpe = candidates_with_oos[0][2] if candidates_with_oos else float("-inf")

        # Parameter stability vs the previous window.
        if prev_best_params and opt_result.best_params:
            is_stable = BayesianTuner.stability_check(
                [prev_best_params, opt_result.best_params]
            )
        else:
            is_stable = True  # window 0: no prior to compare

        win_result = WFOWindowResult(
            window_idx=window_idx,
            is_start=is_df.index[0],
            is_end=is_df.index[-1],
            oos_start=oos_df.index[0],
            oos_end=oos_df.index[-1],
            best_params=opt_result.best_params,
            is_sharpe=opt_result.best_value,
            oos_sharpe=best_oos_sharpe,
            n_trials=opt_result.n_trials,
            is_stable=is_stable,
        )

        return win_result, candidates_with_oos

    def _select_top_k_candidates(
        self,
        param_history: List[_ParamDict],
        value_history: List[float],
        best_params: _ParamDict,
    ) -> List[Tuple[_ParamDict, float]]:
        """Return top-k IS candidates sorted by IS Sharpe (descending).

        If the trial history contains fewer than ``pbo_top_k`` entries,
        the best params are duplicated to ensure we always have at least 2
        candidates for a meaningful PBO computation.

        Args:
            param_history: Per-trial parameter dicts (Optuna order).
            value_history: Corresponding IS Sharpe values.
            best_params: The Optuna study's reported best params.

        Returns:
            List of ``(params, is_sharpe)`` tuples, at most ``pbo_top_k`` long.
        """
        if param_history and value_history:
            paired = sorted(
                zip(param_history, value_history),
                key=lambda x: x[1],
                reverse=True,
            )
            top = list(paired[: self.pbo_top_k])
        else:
            top = []

        # Ensure the reported best is always at the front.
        if best_params:
            best_is = float(value_history[0]) if value_history else 0.0
            top = [(best_params, best_is)] + [
                (p, v) for p, v in top if p != best_params
            ][: self.pbo_top_k - 1]

        # Pad if we have fewer than 2 candidates.
        while len(top) < 2 and best_params:
            top.append((best_params, top[0][1] if top else 0.0))

        return top

    # ------------------------------------------------------------------ #
    # PBO computation                                                      #
    # ------------------------------------------------------------------ #

    def _compute_pbo(
        self,
        per_window_candidates: List[List[Tuple[_ParamDict, float, float]]],
    ) -> float:
        """Compute Probability of Backtest Overfitting.

        For each window we have a list of ``(params, is_sharpe, oos_sharpe)``
        tuples.  The IS-best candidate is the one with the highest IS Sharpe.
        We check whether the IS-best also ranks at or above the median OOS
        Sharpe among all candidates in that window.

        A window counts as "overfit" when the IS-best candidate ranks below
        the median OOS (i.e., IS optimization did NOT predict OOS performance).

        Args:
            per_window_candidates: Per-window lists of ``(params, is_sharpe,
                oos_sharpe)`` tuples.

        Returns:
            PBO ∈ [0, 1].  Lower is better.  Returns 0.0 when no window has
            at least 2 candidates.
        """
        n_overfit = 0
        n_qualifying = 0

        for candidates in per_window_candidates:
            if len(candidates) < 2:
                continue
            n_qualifying += 1

            # Candidates are already sorted: first = IS-best.
            is_best_oos = candidates[0][2]
            all_oos = [c[2] for c in candidates]
            median_oos = float(np.median(all_oos))

            if is_best_oos < median_oos:
                n_overfit += 1

        if n_qualifying == 0:
            logger.warning(
                "PBO could not be computed (no window had ≥ 2 candidates); returning 0.0"
            )
            return 0.0

        pbo = n_overfit / n_qualifying
        logger.info("PBO = %.4f (%d overfit / %d qualifying windows)", pbo, n_overfit, n_qualifying)
        return pbo

    # ------------------------------------------------------------------ #
    # MLflow helpers                                                       #
    # ------------------------------------------------------------------ #

    def _log_window(self, window_idx: int, result: WFOWindowResult) -> None:
        """Log per-window metrics to the active MLflow run (best-effort)."""
        try:
            import mlflow

            if mlflow.active_run() is None:
                return
            step = window_idx
            mlflow.log_metric("wfo/is_sharpe", result.is_sharpe, step=step)
            mlflow.log_metric("wfo/oos_sharpe", result.oos_sharpe, step=step)
            mlflow.log_metric("wfo/is_stable", float(result.is_stable), step=step)
            mlflow.log_metric("wfo/n_trials", float(result.n_trials), step=step)
        except Exception as exc:
            logger.debug("MLflow window %d logging failed: %s", window_idx, exc)

    def _log_summary(self, result: WFOResult) -> None:
        """Log aggregate WFO summary to the active MLflow run (best-effort)."""
        try:
            import mlflow

            if mlflow.active_run() is None:
                return
            mlflow.log_metric("wfo/aggregate_oos_sharpe", result.aggregate_oos_sharpe)
            mlflow.log_metric("wfo/pbo", result.pbo)
            mlflow.log_metric("wfo/n_windows", float(result.n_windows))
            mlflow.log_metric("wfo/is_stable", float(result.is_stable))
            for k, v in result.best_params.items():
                if isinstance(v, (int, float)):
                    mlflow.log_param(f"wfo_best/{k}", v)
                else:
                    mlflow.log_param(f"wfo_best/{k}", str(v))
        except Exception as exc:
            logger.debug("MLflow WFO summary logging failed: %s", exc)
