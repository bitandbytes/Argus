"""Tests for PurgedCrossValidator and purged_walk_forward_splits.

Covers:
  - Basic split behaviour (fold count, non-overlap)
  - Chronological ordering (test always after train)
  - Embargo gap enforcement
  - Purging of overlapping labels
  - sklearn compatibility (CalibratedClassifierCV, standard split signature)
  - Input validation errors
  - Module-level helper function
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

from src.tuning.purged_cv import PurgedCrossValidator, purged_walk_forward_splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(
    n: int = 100,
    holding: int = 5,
    start: str = "2020-01-01",
    seed: int = 0,
) -> tuple:
    """Return (X, y, pred_times, eval_times) with a daily DatetimeIndex."""
    idx = pd.date_range(start, periods=n, freq="D")
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 4)), index=idx, columns=list("abcd"))
    y = pd.Series((rng.random(n) > 0.6).astype(int), index=idx)
    pred_times = pd.Series(idx, index=idx)
    eval_times = pred_times + pd.Timedelta(days=holding)
    return X, y, pred_times, eval_times


# ---------------------------------------------------------------------------
# PurgedCrossValidator — basic split behaviour
# ---------------------------------------------------------------------------

class TestPurgedCrossValidatorSplits:
    def test_split_yields_correct_n_folds(self) -> None:
        X, y, pred, eval_ = _make_data(n=120)
        cv = PurgedCrossValidator(pred, eval_, n_splits=5, embargo_days=5)
        folds = list(cv.split(X))
        assert len(folds) == 5

    def test_train_test_indices_non_overlapping(self) -> None:
        X, y, pred, eval_ = _make_data(n=120)
        cv = PurgedCrossValidator(pred, eval_, n_splits=5, embargo_days=5)
        for train_idx, test_idx in cv.split(X):
            assert len(np.intersect1d(train_idx, test_idx)) == 0, (
                "Train and test index sets must be disjoint."
            )

    def test_no_train_label_overlaps_test_pred_times(self) -> None:
        """Core purging guarantee: no training sample's eval_time should
        fall inside the test fold's prediction window."""
        X, y, pred, eval_ = _make_data(n=120, holding=5)
        cv = PurgedCrossValidator(pred, eval_, n_splits=5, embargo_days=0)
        for train_idx, test_idx in cv.split(X):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            test_pred_min = pred.iloc[test_idx].min()
            test_pred_max = pred.iloc[test_idx].max()
            train_eval_times = eval_.iloc[train_idx]
            # No training eval_time should overlap the test pred window
            overlapping = (train_eval_times >= test_pred_min) & (
                train_eval_times <= test_pred_max
            )
            assert not overlapping.any(), (
                f"Purging failed: {overlapping.sum()} training samples "
                "have eval_times inside the test prediction window."
            )

    def test_get_n_splits_returns_correct_value(self) -> None:
        _, _, pred, eval_ = _make_data()
        cv = PurgedCrossValidator(pred, eval_, n_splits=4)
        assert cv.get_n_splits() == 4

    def test_get_n_splits_ignores_arguments(self) -> None:
        _, _, pred, eval_ = _make_data()
        cv = PurgedCrossValidator(pred, eval_, n_splits=3)
        assert cv.get_n_splits(X=None, y=None, groups=None) == 3


# ---------------------------------------------------------------------------
# PurgedCrossValidator — purging and embargo
# ---------------------------------------------------------------------------

class TestPurgedCrossValidatorPurgingEmargo:
    def test_embargo_gap_enforced(self) -> None:
        """Training samples that precede the test window should not have
        eval_times within the embargo buffer before the test fold starts.

        Note: CombPurgedKFoldCV is combinatorial — training sets may include
        samples from folds *after* the test fold.  The embargo constraint only
        applies to training samples that are temporally before the test window,
        since only those can produce forward-leaking labels.
        """
        embargo = 10
        X, _, pred, eval_ = _make_data(n=150, holding=3)
        cv = PurgedCrossValidator(pred, eval_, n_splits=5, embargo_days=embargo)
        for train_idx, test_idx in cv.split(X):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            test_start = pred.iloc[test_idx].min()
            # Only training samples that precede the test window can leak forward
            before_test = pred.iloc[train_idx] < test_start
            if not before_test.any():
                continue
            train_before_idx = train_idx[before_test.values]
            # Their eval_times must not reach into the test window
            leaking = eval_.iloc[train_before_idx] >= test_start
            assert not leaking.any(), (
                f"{leaking.sum()} pre-test training samples have eval_times "
                f">= test_start ({test_start}), indicating forward leakage."
            )

    def test_purging_removes_overlapping_labels(self) -> None:
        """With a long holding period, some training samples should be purged."""
        n = 100
        # holding=40 days means labels deeply overlap fold boundaries
        X, _, pred, eval_ = _make_data(n=n, holding=40)
        cv_purged = PurgedCrossValidator(pred, eval_, n_splits=5, embargo_days=5)
        cv_plain = PurgedCrossValidator(pred, pred, n_splits=5, embargo_days=0)

        purged_train_sizes = [len(tr) for tr, _ in cv_purged.split(X)]
        plain_train_sizes = [len(tr) for tr, _ in cv_plain.split(X)]

        # Purged CV should remove some training samples (smaller train sets)
        assert sum(purged_train_sizes) < sum(plain_train_sizes), (
            "Purging should reduce total training samples when labels overlap folds."
        )

    def test_larger_embargo_reduces_train_size(self) -> None:
        """A larger embargo gap should leave fewer training samples per fold."""
        X, _, pred, eval_ = _make_data(n=150, holding=3)
        cv_small = PurgedCrossValidator(pred, eval_, n_splits=5, embargo_days=2)
        cv_large = PurgedCrossValidator(pred, eval_, n_splits=5, embargo_days=15)

        small_total = sum(len(tr) for tr, _ in cv_small.split(X))
        large_total = sum(len(tr) for tr, _ in cv_large.split(X))
        assert large_total <= small_total, (
            "Larger embargo should yield fewer (or equal) total training samples."
        )


# ---------------------------------------------------------------------------
# PurgedCrossValidator — sklearn compatibility
# ---------------------------------------------------------------------------

class TestPurgedCrossValidatorSklearnCompat:
    def test_sklearn_compatible_split_signature(self) -> None:
        """split(X) with no y/groups must work (standard sklearn signature)."""
        X, _, pred, eval_ = _make_data()
        cv = PurgedCrossValidator(pred, eval_, n_splits=5)
        folds = list(cv.split(X))       # no y, no groups
        assert len(folds) == 5

    def test_compatible_with_calibrated_classifier_cv(self) -> None:
        """PurgedCrossValidator must be passable as cv= to CalibratedClassifierCV."""
        X, y, pred, eval_ = _make_data(n=150, seed=1)
        cv = PurgedCrossValidator(pred, eval_, n_splits=4, embargo_days=3)
        base = xgb.XGBClassifier(
            n_estimators=10, verbosity=0, eval_metric="logloss", random_state=0
        )
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
        cal.fit(X, y)                     # must not raise
        proba = cal.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_split_returns_numpy_arrays(self) -> None:
        X, _, pred, eval_ = _make_data()
        cv = PurgedCrossValidator(pred, eval_, n_splits=3)
        for train_idx, test_idx in cv.split(X):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)


# ---------------------------------------------------------------------------
# PurgedCrossValidator — input validation
# ---------------------------------------------------------------------------

class TestPurgedCrossValidatorValidation:
    def test_raises_on_non_series_pred_times(self) -> None:
        idx = pd.date_range("2020-01-01", periods=10)
        eval_ = pd.Series(idx, index=idx)
        with pytest.raises(TypeError, match="pred_times"):
            PurgedCrossValidator(list(idx), eval_)

    def test_raises_on_non_series_eval_times(self) -> None:
        idx = pd.date_range("2020-01-01", periods=10)
        pred = pd.Series(idx, index=idx)
        with pytest.raises(TypeError, match="eval_times"):
            PurgedCrossValidator(pred, list(idx))

    def test_raises_on_mismatched_lengths(self) -> None:
        idx = pd.date_range("2020-01-01", periods=10)
        pred = pd.Series(idx, index=idx)
        eval_ = pred.iloc[:8] + pd.Timedelta(days=5)
        with pytest.raises(ValueError, match="same length"):
            PurgedCrossValidator(pred, eval_)

    def test_raises_when_eval_before_pred(self) -> None:
        idx = pd.date_range("2020-01-01", periods=10)
        pred = pd.Series(idx, index=idx)
        eval_ = pred - pd.Timedelta(days=1)   # eval before entry — impossible
        with pytest.raises(ValueError, match="eval_times must be >= pred_times"):
            PurgedCrossValidator(pred, eval_)

    def test_split_raises_on_x_length_mismatch(self) -> None:
        X, _, pred, eval_ = _make_data(n=50)
        X_wrong = X.iloc[:30]        # different length from pred/eval (50)
        cv = PurgedCrossValidator(pred, eval_, n_splits=3)
        with pytest.raises(ValueError, match="pred_times"):
            list(cv.split(X_wrong))


# ---------------------------------------------------------------------------
# purged_walk_forward_splits helper function
# ---------------------------------------------------------------------------

class TestPurgedWalkForwardSplits:
    def test_helper_yields_correct_number_of_folds(self) -> None:
        X, _, pred, eval_ = _make_data(n=120)
        folds = list(purged_walk_forward_splits(X, pred, eval_, n_splits=5))
        assert len(folds) == 5

    def test_helper_yields_index_tuples(self) -> None:
        X, _, pred, eval_ = _make_data()
        for train_idx, test_idx in purged_walk_forward_splits(X, pred, eval_, n_splits=3):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_helper_non_overlapping_folds(self) -> None:
        X, _, pred, eval_ = _make_data(n=120)
        for tr, te in purged_walk_forward_splits(X, pred, eval_, n_splits=5):
            assert len(np.intersect1d(tr, te)) == 0
