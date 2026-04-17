"""Purged cross-validation for financial time series (Task 2.3).

Standard k-fold and even ``TimeSeriesSplit`` are unsafe for meta-labeling
because labels are forward-looking: a label generated at time *t* depends on
prices up to *t + max_holding_days*.  Without purging, training samples whose
labels "see into" the test window leak future information.

``PurgedCrossValidator`` wraps ``timeseriescv.CombPurgedKFoldCV`` with a
standard sklearn CV interface (``split(X, y)``), making it a drop-in
replacement for ``TimeSeriesSplit`` inside ``CalibratedClassifierCV``.

Usage with MetaLabelModel::

    labels = triple_barrier_labels(prices, sig_times, sig_dirs)
    pred_times = pd.Series(labels.index, index=labels.index)
    eval_times = pd.Series(labels["exit_time"].values, index=labels.index)

    model = MetaLabelModel()
    model.train(X, y, pred_times=pred_times, eval_times=eval_times)

Usage standalone::

    cv = PurgedCrossValidator(pred_times, eval_times, n_splits=5, embargo_days=5)
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        ...
"""

import itertools
import logging
from typing import Callable, Iterator, List, Tuple

import numpy as np
import pandas as pd
from timeseriescv.cross_validation import CombPurgedKFoldCV

logger = logging.getLogger(__name__)


class PurgedCrossValidator:
    """
    Sklearn-compatible purged k-fold cross-validator for time series.

    Wraps ``timeseriescv.CombPurgedKFoldCV`` and stores ``pred_times`` /
    ``eval_times`` at construction so that the standard ``split(X, y)``
    signature is exposed — making this a valid ``cv=`` argument for
    ``sklearn.calibration.CalibratedClassifierCV``.

    Purging removes training samples whose *eval_time* (when the label became
    known) overlaps with the test fold's *pred_times* (when predictions are
    made).  An additional embargo gap further prevents leakage from correlated
    returns near the fold boundary.

    Args:
        pred_times: Series of timestamps when each signal was generated
            (entry time).  Must share its index with the feature matrix ``X``
            passed to :meth:`split`.
        eval_times: Series of timestamps when each label became known
            (exit time = entry + holding period).  Same index as
            ``pred_times``.  All values must be ``>= pred_times``.
        n_splits: Total number of folds to partition the data into (default 5).
        embargo_days: Additional gap (in calendar days) between the last
            training sample and the first test sample.  Should equal
            ``max_holding_days`` used in :func:`triple_barrier_labels`
            (default 5).

    Raises:
        TypeError: If ``pred_times`` or ``eval_times`` are not
            ``pd.Series``.
        ValueError: If lengths differ or any ``eval_time < pred_time``.
    """

    def __init__(
        self,
        pred_times: pd.Series,
        eval_times: pd.Series,
        n_splits: int = 5,
        embargo_days: int = 5,
    ) -> None:
        self._validate(pred_times, eval_times)
        self._pred_times = pred_times
        self._eval_times = eval_times
        self._n_splits = n_splits
        self._embargo_days = embargo_days
        self._cv = CombPurgedKFoldCV(
            n_splits=n_splits,
            n_test_splits=1,
            embargo_td=pd.Timedelta(days=embargo_days),
        )

    # ------------------------------------------------------------------ #
    # Sklearn CV protocol                                                   #
    # ------------------------------------------------------------------ #

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield ``(train_indices, test_indices)`` for each purged fold.

        Args:
            X: Feature matrix whose length must match ``pred_times`` /
                ``eval_times``.
            y: Ignored (kept for sklearn API compatibility).
            groups: Ignored (kept for sklearn API compatibility).

        Yields:
            Tuples of ``(train_idx, test_idx)`` as positional
            ``np.ndarray`` of integers.

        Raises:
            ValueError: If ``len(X)`` differs from ``len(pred_times)``.
        """
        if len(X) != len(self._pred_times):
            raise ValueError(
                f"X has {len(X)} rows but pred_times has "
                f"{len(self._pred_times)} entries."
            )
        yield from self._cv.split(
            X,
            pred_times=self._pred_times,
            eval_times=self._eval_times,
        )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits (folds)."""
        return self._n_splits

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate(pred_times: pd.Series, eval_times: pd.Series) -> None:
        """Validate pred_times and eval_times before storing."""
        for name, series in (("pred_times", pred_times), ("eval_times", eval_times)):
            if not isinstance(series, pd.Series):
                raise TypeError(
                    f"{name} must be a pd.Series, got {type(series).__name__}."
                )

        if len(pred_times) != len(eval_times):
            raise ValueError(
                f"pred_times ({len(pred_times)}) and eval_times "
                f"({len(eval_times)}) must have the same length."
            )

        if (eval_times.values < pred_times.values).any():
            raise ValueError(
                "All eval_times must be >= pred_times "
                "(a label cannot resolve before the signal is generated)."
            )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def purged_walk_forward_splits(
    X: pd.DataFrame,
    pred_times: pd.Series,
    eval_times: pd.Series,
    n_splits: int = 5,
    embargo_days: int = 5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience wrapper: create a :class:`PurgedCrossValidator` and iterate.

    Equivalent to::

        cv = PurgedCrossValidator(pred_times, eval_times, n_splits, embargo_days)
        yield from cv.split(X)

    Args:
        X: Feature matrix (n_samples × n_features).
        pred_times: Signal generation timestamps.
        eval_times: Label evaluation timestamps (entry + holding period).
        n_splits: Number of folds (default 5).
        embargo_days: Gap between train end and test start (default 5).

    Yields:
        ``(train_idx, test_idx)`` positional index arrays.
    """
    cv = PurgedCrossValidator(pred_times, eval_times, n_splits, embargo_days)
    yield from cv.split(X)


# ---------------------------------------------------------------------------
# CombinatorialPurgedCV — CPCV-based PBO calculator
# ---------------------------------------------------------------------------

class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation for Probability of Backtest Overfitting.

    Implements the CPCV framework from López de Prado's "Advances in Financial
    Machine Learning" (Chapter 12).  The data is divided into ``n_groups``
    sequential groups; all ``C(n_groups, n_test_groups)`` combinations of test
    groups are generated.  For each combination the caller-supplied
    ``strategy_fn(is_df, oos_df)`` is invoked, returning a
    ``(is_sharpe, oos_sharpe)`` pair.

    ``compute_pbo()`` then measures what fraction of paths have an OOS Sharpe
    below the median — indicating that the strategy is likely overfit.

    Usage::

        from src.tuning.purged_cv import CombinatorialPurgedCV

        cpcv = CombinatorialPurgedCV(n_groups=10, n_test_groups=2, embargo_days=5)
        results = cpcv.run(strategy_fn, data)
        pbo = cpcv.compute_pbo(results)
        assert pbo < 0.40, "Strategy likely overfit — do not deploy"

    Args:
        n_groups: Number of sequential groups to split the data into (default 10).
        n_test_groups: Number of groups to use as the test set per combination
            (default 2).  Must be < ``n_groups``.
        embargo_days: Calendar-day gap applied between the IS end and OOS start
            to prevent leakage from correlated returns (default 5).
    """

    def __init__(
        self,
        n_groups: int = 10,
        n_test_groups: int = 2,
        embargo_days: int = 5,
    ) -> None:
        if n_groups < 3:
            raise ValueError(f"n_groups must be >= 3, got {n_groups}")
        if n_test_groups < 1 or n_test_groups >= n_groups:
            raise ValueError(
                f"n_test_groups must be in [1, n_groups), got {n_test_groups}"
            )
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_days = embargo_days

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        strategy_fn: Callable[[pd.DataFrame, pd.DataFrame], Tuple[float, float]],
        data: pd.DataFrame,
    ) -> List[Tuple[float, float]]:
        """Evaluate ``strategy_fn`` on all CPCV combinations.

        The data is split into ``n_groups`` equal-sized sequential slices.
        For each ``C(n_groups, n_test_groups)`` combination the test slices are
        concatenated as OOS data; all remaining slices are concatenated as IS
        data with a ``embargo_days``-day gap applied at each IS/OOS boundary.

        Args:
            strategy_fn: ``Callable[[is_df, oos_df], Tuple[float, float]]``
                returning ``(IS_sharpe, OOS_sharpe)``.  Receives contiguous
                pandas DataFrames indexed by date.
            data: Full OHLCV (or feature) DataFrame indexed by date.

        Returns:
            List of ``(IS_sharpe, OOS_sharpe)`` tuples, one per CPCV
            combination.  Length = ``C(n_groups, n_test_groups)``.
        """
        groups = self._split_groups(data)
        results: List[Tuple[float, float]] = []

        for test_indices in itertools.combinations(range(self.n_groups), self.n_test_groups):
            test_idx_set = set(test_indices)
            train_idx_list = [i for i in range(self.n_groups) if i not in test_idx_set]

            is_df = pd.concat([groups[i] for i in train_idx_list])
            oos_df = pd.concat([groups[i] for i in sorted(test_indices)])

            # Apply embargo: drop IS rows within embargo_days of OOS start.
            if self.embargo_days > 0 and len(oos_df) > 0:
                oos_start = oos_df.index[0]
                cutoff = oos_start - pd.Timedelta(days=self.embargo_days)
                is_df = is_df[is_df.index <= cutoff]

            if len(is_df) < 2 or len(oos_df) < 2:
                logger.debug("Skipping CPCV combination %s: insufficient data after embargo", test_indices)
                continue

            try:
                pair = strategy_fn(is_df, oos_df)
                results.append((float(pair[0]), float(pair[1])))
            except Exception as exc:
                logger.warning("strategy_fn failed for CPCV combination %s: %s", test_indices, exc)

        logger.info(
            "CombinatorialPurgedCV: %d combinations evaluated, %d succeeded",
            len(list(itertools.combinations(range(self.n_groups), self.n_test_groups))),
            len(results),
        )
        return results

    @staticmethod
    def compute_pbo(results: List[Tuple[float, float]]) -> float:
        """Compute PBO from a list of (IS_sharpe, OOS_sharpe) pairs.

        PBO = fraction of paths where the OOS Sharpe is below the median OOS
        Sharpe across all paths.  A value above 0.50 means the strategy is more
        likely to fail live than succeed.

        Args:
            results: List of ``(IS_sharpe, OOS_sharpe)`` pairs as returned by
                :meth:`run`.

        Returns:
            PBO ∈ [0, 1].  Returns 0.0 for empty results.
        """
        if not results:
            logger.warning("compute_pbo called with empty results — returning 0.0")
            return 0.0

        oos_sharpes = np.array([r[1] for r in results], dtype=float)
        median_oos = float(np.median(oos_sharpes))
        pbo = float(np.mean(oos_sharpes < median_oos))
        logger.info("CPCV PBO = %.4f (median OOS Sharpe = %.4f)", pbo, median_oos)
        return pbo

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _split_groups(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Split ``data`` into ``n_groups`` sequential equal-length groups."""
        arrays = np.array_split(np.arange(len(data)), self.n_groups)
        return [data.iloc[arr] for arr in arrays if len(arr) > 0]
