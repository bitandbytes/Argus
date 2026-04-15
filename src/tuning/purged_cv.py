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

import logging
from typing import Iterator, Tuple

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
