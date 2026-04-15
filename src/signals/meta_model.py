"""XGBoost meta-labeling classifier — Layer 3 of the cascade pipeline.

The ``MetaLabelModel`` does **not** predict price direction. Given the same
features the quant engine used plus the quant engine's own prediction
(direction + confidence), it estimates the calibrated probability that the
quant signal will be correct (i.e. that the position will hit take-profit
before stop-loss within ``max_holding_days`` bars).

Training workflow (Task 2.2)::

    # 1. Generate historical quant signals
    signals = quant_engine.generate_series(df, regime_series, ticker)

    # 2. Label each signal via triple-barrier
    prices = df["close"]
    sig_times  = pd.DatetimeIndex([s.timestamp for s in signals])
    sig_dirs   = pd.Series([s.direction for s in signals], index=sig_times)
    labels = triple_barrier_labels(prices, sig_times, sig_dirs)

    # 3. Assemble feature matrix (aligns signals with label rows)
    X = build_feature_matrix([signals[i] for i in labels.index])
    y = labels["meta_label"]

    # 4. Train and save
    model = MetaLabelModel()
    metrics = model.train(X, y)
    model.save()

Production inference::

    proba, uncertainty = model.predict(X_live)
    if proba[0] > 0.55 and uncertainty[0] < 0.15:
        bet_size = (proba[0] - 0.5) * 2   # maps [0.5, 1.0] → [0.0, 1.0]
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# Default XGBoost hyperparameters (overridden by config/settings.yaml at runtime).
_XGB_DEFAULTS: Dict[str, Any] = {
    "max_depth": 5,
    "learning_rate": 0.01,
    "n_estimators": 500,
    "reg_lambda": 7.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": 42,
}


class MetaLabelModel:
    """
    XGBoost binary classifier wrapped in Platt-scaling calibration.

    Trained on triple-barrier meta-labels; outputs calibrated
    P(quant_signal_correct) for bet-size computation.

    **Anti-overfitting:**  ``train()`` uses ``TimeSeriesSplit`` inside
    ``CalibratedClassifierCV`` (no future leakage during calibration) and a
    separate walk-forward pass with an embargo gap to compute honest OOS
    F1 and Brier score.

    **Versioning:** ``save()`` auto-increments the version number from
    existing ``v*.pkl`` files in ``model_dir`` and writes a
    ``metadata.json`` summary alongside every model file.

    Args:
        model_dir: Directory for versioned model files.
            Created if it does not exist.
        settings_path: Path to ``config/settings.yaml`` (XGBoost params).
        **xgb_kwargs: Override any XGBoost hyperparameter.
    """

    def __init__(
        self,
        model_dir: str = "data/models/meta_model",
        settings_path: str = "config/settings.yaml",
        **xgb_kwargs: Any,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        params = self._load_xgb_params(settings_path)
        params.update(xgb_kwargs)
        self._xgb_params: Dict[str, Any] = params

        self._calibrated: Optional[CalibratedClassifierCV] = None
        self._metrics: Dict[str, Any] = {}
        self._version: Optional[int] = None
        self._trained_at: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        embargo_days: int = 5,
    ) -> Dict[str, Any]:
        """
        Train XGBoost with Platt calibration on meta-labeled signal data.

        Steps:

        1. Validate inputs and compute ``scale_pos_weight`` from class ratio.
        2. Fit ``CalibratedClassifierCV(TimeSeriesSplit(n_splits))`` — Platt
           scaling with time-ordered CV.
        3. Compute OOS F1 and Brier score via a separate walk-forward pass
           (each fold trains on past data only; ``embargo_days`` gap separates
           training end from test start to prevent label leakage).
        4. Log metrics and params to MLflow (best-effort; never fails training).

        Args:
            X: Feature matrix (n_signals × ``len(FEATURE_COLUMNS)``).
            y: Binary meta-labels: 1 = signal was correct, 0 = wrong/timeout.
            n_splits: Number of time-series CV folds (default 5).
            embargo_days: Bars gap between train end and test start in the
                walk-forward evaluation (default 5).

        Returns:
            Dict with keys:

            * ``f1_score`` — OOS F1 score (weighted average)
            * ``brier_score`` — OOS Brier score (lower is better)
            * ``class_balance`` — fraction of positive labels
            * ``n_train`` — total training samples
            * ``feature_importance`` — pd.Series of XGBoost importances

        Raises:
            ValueError: Mismatched lengths, non-binary labels, or < 20 samples.
        """
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length, got X={len(X)}, y={len(y)}."
            )
        unique_labels = set(y.unique())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"y must be binary {{0, 1}}, got {sorted(unique_labels)}."
            )
        if len(X) < 20:
            raise ValueError(
                f"Need at least 20 training samples, got {len(X)}."
            )

        pos = int(y.sum())
        neg = int((y == 0).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        logger.info(
            "Training MetaLabelModel on %d samples "
            "(pos=%d, neg=%d, scale_pos_weight=%.2f)",
            len(X), pos, neg, scale_pos_weight,
        )

        params = {**self._xgb_params, "scale_pos_weight": scale_pos_weight}
        base_model = xgb.XGBClassifier(**params, verbosity=0)
        cv = TimeSeriesSplit(n_splits=n_splits)
        self._calibrated = CalibratedClassifierCV(
            base_model, method="sigmoid", cv=cv
        )
        self._calibrated.fit(X, y)

        # OOS evaluation via walk-forward with embargo
        oos_probas, oos_trues = self._walk_forward_eval(
            X, y, n_splits, embargo_days
        )
        f1 = float(
            f1_score(oos_trues, (oos_probas >= 0.5).astype(int), zero_division=0)
        )
        brier = float(brier_score_loss(oos_trues, oos_probas))

        feat_imp = self._extract_feature_importance(list(X.columns))
        self._trained_at = datetime.now().strftime("%Y-%m-%d")
        self._metrics = {
            "f1_score": f1,
            "brier_score": brier,
            "class_balance": pos / len(y),
            "n_train": len(X),
            "feature_importance": feat_imp,
        }

        self._log_to_mlflow(params, self._metrics)
        logger.info(
            "MetaLabelModel training complete — F1=%.3f, Brier=%.3f",
            f1, brier,
        )
        return self._metrics

    def _walk_forward_eval(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int,
        embargo_days: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Walk-forward OOS evaluation with embargo gap between folds."""
        n = len(X)
        fold_size = max(n // (n_splits + 1), 1)

        all_probas: List[float] = []
        all_trues: List[int] = []

        for k in range(1, n_splits + 1):
            train_end = k * fold_size
            test_start = train_end + embargo_days
            test_end = min(test_start + fold_size, n)

            if test_start >= n or test_end <= test_start:
                continue

            X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
            X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

            pos_k = int(y_tr.sum())
            neg_k = int((y_tr == 0).sum())
            spw_k = neg_k / pos_k if pos_k > 0 else 1.0
            fold_params = {**self._xgb_params, "scale_pos_weight": spw_k}

            fold_base = xgb.XGBClassifier(**fold_params, verbosity=0)
            fold_cal = CalibratedClassifierCV(fold_base, method="sigmoid", cv=3)
            try:
                fold_cal.fit(X_tr, y_tr)
                probas = fold_cal.predict_proba(X_te)[:, 1]
                all_probas.extend(probas.tolist())
                all_trues.extend(y_te.tolist())
            except Exception as exc:
                logger.warning("Walk-forward fold %d failed: %s", k, exc)

        if not all_probas:
            # Fallback: in-sample — only used when data is too small to split
            logger.warning(
                "Walk-forward evaluation produced no OOS predictions; "
                "falling back to in-sample metrics."
            )
            all_probas = self._calibrated.predict_proba(X)[:, 1].tolist()
            all_trues = y.tolist()

        return np.array(all_probas), np.array(all_trues)

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def predict(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return calibrated probability and uncertainty for each row in X.

        Args:
            X: Feature matrix with the same columns as training data.

        Returns:
            Tuple ``(proba, uncertainty)`` where:

            * ``proba`` — shape ``(n,)``, calibrated P(signal_correct) ∈ [0, 1].
            * ``uncertainty`` — shape ``(n,)``, Bernoulli variance
              ``p * (1 − p)``, maximum 0.25 at ``p = 0.5``.
              Compare against ``meta_model.uncertainty_threshold`` in
              ``config/settings.yaml`` (default 0.15) before trading.

        Raises:
            RuntimeError: If called before :meth:`train` or :meth:`load`.
        """
        if self._calibrated is None:
            raise RuntimeError(
                "MetaLabelModel is not trained. Call train() or load() first."
            )
        proba = self._calibrated.predict_proba(X)[:, 1]
        uncertainty = proba * (1.0 - proba)
        return proba, uncertainty

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self) -> Path:
        """
        Persist the trained model and metadata to *model_dir*.

        The file is named ``v{N}_{YYYY-MM-DD}.pkl`` where N is
        auto-incremented from existing pkl files in the directory.
        A ``metadata.json`` file is always written (or overwritten).

        Returns:
            :class:`pathlib.Path` of the written ``.pkl`` file.

        Raises:
            RuntimeError: If called before :meth:`train`.
        """
        if self._calibrated is None:
            raise RuntimeError(
                "Cannot save: MetaLabelModel is not trained. Call train() first."
            )

        version = self._next_version()
        self._version = version
        date_str = self._trained_at or datetime.now().strftime("%Y-%m-%d")

        model_path = self._model_dir / f"v{version}_{date_str}.pkl"
        joblib.dump(self._calibrated, model_path)

        # metadata.json — scalar values only (exclude pd.Series)
        meta: Dict[str, Any] = {
            "version": version,
            "training_date": date_str,
            "f1_score": self._metrics.get("f1_score"),
            "brier_score": self._metrics.get("brier_score"),
            "class_balance": self._metrics.get("class_balance"),
            "n_train": self._metrics.get("n_train"),
            "xgboost_params": {
                k: v
                for k, v in self._xgb_params.items()
                if isinstance(v, (int, float, str, bool))
            },
        }
        with open(self._model_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("MetaLabelModel saved → %s", model_path)
        return model_path

    @classmethod
    def load(
        cls,
        path: str,
        model_dir: Optional[str] = None,
    ) -> "MetaLabelModel":
        """
        Load a previously saved model from a ``.pkl`` file.

        Args:
            path: Path to a versioned ``.pkl`` file written by :meth:`save`.
            model_dir: Override the model directory (default: parent of path).

        Returns:
            :class:`MetaLabelModel` instance ready to call :meth:`predict`.
        """
        model_path = Path(path)
        instance: "MetaLabelModel" = cls.__new__(cls)
        instance._calibrated = joblib.load(model_path)
        instance._xgb_params = dict(_XGB_DEFAULTS)
        instance._metrics = {}
        instance._version = None
        instance._trained_at = None
        instance._model_dir = Path(model_dir) if model_dir else model_path.parent

        meta_path = instance._model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            instance._metrics = {
                k: meta.get(k) for k in ("f1_score", "brier_score")
            }
            instance._version = meta.get("version")
            instance._trained_at = meta.get("training_date")

        logger.info("MetaLabelModel loaded ← %s", model_path)
        return instance

    def get_calibration_metrics(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """
        Compute calibration diagnostics on held-out data.

        Args:
            X: Feature matrix.
            y: True binary meta-labels.

        Returns:
            Dict with keys:

            * ``brier_score`` — mean squared error of predictions
            * ``f1_at_threshold_0_5`` — F1 at the default 0.5 cut-off
            * ``calibration_bins`` — dict with ``predicted`` and ``observed``
              lists (10 equal-width bins)
            * ``n_samples`` — number of samples evaluated

        Raises:
            RuntimeError: If called before :meth:`train` or :meth:`load`.
        """
        if self._calibrated is None:
            raise RuntimeError(
                "MetaLabelModel is not trained. Call train() or load() first."
            )

        proba, _ = self.predict(X)
        preds = (proba >= 0.5).astype(int)

        bins = np.linspace(0, 1, 11)
        bin_indices = np.clip(np.digitize(proba, bins) - 1, 0, 9)
        bin_predicted: List[float] = []
        bin_observed: List[float] = []
        for b in range(10):
            mask = bin_indices == b
            if mask.sum() > 0:
                bin_predicted.append(float(proba[mask].mean()))
                bin_observed.append(float(y.values[mask].mean()))

        return {
            "brier_score": float(brier_score_loss(y, proba)),
            "f1_at_threshold_0_5": float(f1_score(y, preds, zero_division=0)),
            "calibration_bins": {
                "predicted": bin_predicted,
                "observed": bin_observed,
            },
            "n_samples": len(y),
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load_xgb_params(self, settings_path: str) -> Dict[str, Any]:
        """Load XGBoost hyperparams from settings.yaml; fall back to defaults."""
        params = dict(_XGB_DEFAULTS)
        try:
            with open(settings_path) as f:
                cfg = yaml.safe_load(f)
            xgb_cfg = cfg.get("meta_model", {}).get("xgboost", {})
            params.update(xgb_cfg)
        except (FileNotFoundError, KeyError):
            logger.warning(
                "Could not load XGBoost params from %s; using defaults.",
                settings_path,
            )
        return params

    def _next_version(self) -> int:
        """Auto-increment version from existing pkl files in model_dir."""
        existing = sorted(self._model_dir.glob("v*.pkl"))
        if not existing:
            return 1
        last_stem = existing[-1].stem  # e.g. "v3_2026-04-15"
        try:
            return int(last_stem.split("_")[0].lstrip("v")) + 1
        except ValueError:
            return len(existing) + 1

    def _extract_feature_importance(
        self, feature_names: List[str]
    ) -> pd.Series:
        """Extract sorted feature importances from the first calibrated estimator."""
        try:
            base = self._calibrated.calibrated_classifiers_[0].estimator
            imp = pd.Series(base.feature_importances_, index=feature_names)
            return imp.sort_values(ascending=False)
        except Exception:
            return pd.Series(dtype=float)

    def _log_to_mlflow(
        self, params: Dict[str, Any], metrics: Dict[str, Any]
    ) -> None:
        """Log training run to MLflow (best-effort; never interrupts training)."""
        try:
            run_name = f"meta_model_{self._trained_at}"
            with mlflow.start_run(run_name=run_name):
                loggable = {
                    k: v
                    for k, v in params.items()
                    if isinstance(v, (int, float, str, bool))
                }
                mlflow.log_params(loggable)
                mlflow.log_metrics({
                    "f1_score": float(metrics.get("f1_score") or 0.0),
                    "brier_score": float(metrics.get("brier_score") or 0.0),
                    "class_balance": float(metrics.get("class_balance") or 0.0),
                })
        except Exception as exc:
            logger.warning("MLflow logging skipped: %s", exc)
