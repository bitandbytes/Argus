"""Tests for MetaLabelModel and build_feature_matrix.

Covers:
  - build_feature_matrix: column order, regime one-hot, edge cases
  - MetaLabelModel.train: valid run, metrics, input validation errors
  - MetaLabelModel.predict: shapes, bounds, untrained guard
  - MetaLabelModel.save / load: versioning, metadata.json, roundtrip
  - MetaLabelModel.get_calibration_metrics: keys, Brier on separable data
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.models.trade_signal import RegimeType, TradeSignal
from src.signals.feature_assembler import FEATURE_COLUMNS, build_feature_matrix
from src.signals.meta_model import MetaLabelModel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_xy(
    n: int = 200,
    seed: int = 42,
    pos_rate: float = 0.35,
) -> tuple:
    """Random feature matrix + imbalanced binary labels (time-indexed)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    y = pd.Series(
        (rng.random(n) < pos_rate).astype(int),
        index=X.index,
    )
    return X, y


def _make_signal(
    direction: float = 0.5,
    confidence: float = 0.7,
    regime: RegimeType = RegimeType.TRENDING_UP,
    timestamp: datetime = None,
    features: dict = None,
) -> TradeSignal:
    """Build a minimal TradeSignal for feature-assembly tests."""
    if timestamp is None:
        timestamp = datetime(2020, 1, 1)
    if features is None:
        features = {
            "sma_crossover": 0.3,
            "macd":          0.2,
            "rsi":          -0.1,
            "bollinger":     0.0,
            "donchian":      0.1,
            "volume":        0.2,
            "sentiment":     0.0,
        }
    return TradeSignal(
        ticker="TEST",
        timestamp=timestamp,
        direction=direction,
        confidence=confidence,
        source_layer="quant",
        regime=regime,
        features=features,
    )


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_column_order_matches_feature_columns(self) -> None:
        df = build_feature_matrix([_make_signal()])
        assert list(df.columns) == FEATURE_COLUMNS

    def test_regime_one_hot_trending_up(self) -> None:
        df = build_feature_matrix([_make_signal(regime=RegimeType.TRENDING_UP)])
        assert df["regime_trending_up"].iloc[0] == 1.0
        assert df["regime_trending_down"].iloc[0] == 0.0
        assert df["regime_ranging"].iloc[0] == 0.0
        assert df["regime_volatile"].iloc[0] == 0.0

    def test_regime_one_hot_trending_down(self) -> None:
        df = build_feature_matrix([_make_signal(regime=RegimeType.TRENDING_DOWN)])
        assert df["regime_trending_down"].iloc[0] == 1.0
        assert df["regime_trending_up"].iloc[0] == 0.0

    def test_regime_one_hot_ranging(self) -> None:
        df = build_feature_matrix([_make_signal(regime=RegimeType.RANGING)])
        assert df["regime_ranging"].iloc[0] == 1.0

    def test_regime_one_hot_volatile(self) -> None:
        df = build_feature_matrix([_make_signal(regime=RegimeType.VOLATILE)])
        assert df["regime_volatile"].iloc[0] == 1.0
        assert (
            df[["regime_trending_up", "regime_trending_down", "regime_ranging"]]
            .iloc[0]
            .sum()
        ) == 0.0

    def test_direction_and_confidence_populated(self) -> None:
        df = build_feature_matrix([_make_signal(direction=0.8, confidence=0.9)])
        assert df["direction"].iloc[0] == pytest.approx(0.8)
        assert df["confidence"].iloc[0] == pytest.approx(0.9)

    def test_missing_features_default_to_zero(self) -> None:
        sig = _make_signal(features={})
        df = build_feature_matrix([sig])
        for col in ["sma_crossover", "macd", "rsi", "bollinger", "donchian", "volume", "sentiment"]:
            assert df[col].iloc[0] == pytest.approx(0.0), f"{col} should default to 0.0"

    def test_empty_signals_returns_correct_columns(self) -> None:
        df = build_feature_matrix([])
        assert list(df.columns) == FEATURE_COLUMNS
        assert len(df) == 0

    def test_multiple_signals_indexed_by_timestamp(self) -> None:
        t0, t1 = datetime(2020, 1, 1), datetime(2020, 1, 2)
        df = build_feature_matrix([_make_signal(timestamp=t0), _make_signal(timestamp=t1)])
        assert len(df) == 2
        assert df.index.name == "timestamp"

    def test_index_dtype_is_datetimeindex(self) -> None:
        df = build_feature_matrix([_make_signal()])
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_sentiment_from_features(self) -> None:
        sig = _make_signal(features={"sentiment": 0.42})
        df = build_feature_matrix([sig])
        assert df["sentiment"].iloc[0] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# MetaLabelModel — training
# ---------------------------------------------------------------------------

class TestMetaLabelModelTraining:
    def test_train_returns_required_keys(self, tmp_path) -> None:
        X, y = _make_xy()
        model = MetaLabelModel(model_dir=str(tmp_path))
        metrics = model.train(X, y)
        for key in ("f1_score", "brier_score", "class_balance", "n_train", "feature_importance"):
            assert key in metrics, f"Missing metrics key: {key}"

    def test_train_metrics_in_valid_range(self, tmp_path) -> None:
        X, y = _make_xy()
        metrics = MetaLabelModel(model_dir=str(tmp_path)).train(X, y)
        assert 0.0 <= metrics["f1_score"]     <= 1.0
        assert 0.0 <= metrics["brier_score"]  <= 1.0
        assert 0.0 <= metrics["class_balance"] <= 1.0
        assert metrics["n_train"] == len(X)

    def test_train_with_30pct_positive_class(self, tmp_path) -> None:
        """30 % positive rate (realistic for quant signals) must not raise."""
        X, y = _make_xy(n=200, pos_rate=0.30)
        metrics = MetaLabelModel(model_dir=str(tmp_path)).train(X, y)
        assert metrics["f1_score"] >= 0.0

    def test_feature_importance_is_series(self, tmp_path) -> None:
        X, y = _make_xy()
        metrics = MetaLabelModel(model_dir=str(tmp_path)).train(X, y)
        assert isinstance(metrics["feature_importance"], pd.Series)

    def test_train_raises_mismatched_lengths(self, tmp_path) -> None:
        X, y = _make_xy(n=100)
        with pytest.raises(ValueError, match="same length"):
            MetaLabelModel(model_dir=str(tmp_path)).train(X, y.iloc[:50])

    def test_train_raises_non_binary_labels(self, tmp_path) -> None:
        X, y = _make_xy(n=100)
        y_bad = y.copy()
        y_bad.iloc[0] = 2
        with pytest.raises(ValueError, match="binary"):
            MetaLabelModel(model_dir=str(tmp_path)).train(X, y_bad)

    def test_train_raises_too_few_samples(self, tmp_path) -> None:
        X, y = _make_xy(n=10)
        with pytest.raises(ValueError, match="20 training samples"):
            MetaLabelModel(model_dir=str(tmp_path)).train(X, y)


# ---------------------------------------------------------------------------
# MetaLabelModel — prediction
# ---------------------------------------------------------------------------

class TestMetaLabelModelPrediction:
    @pytest.fixture(scope="class")
    def trained(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("model")
        X, y = _make_xy()
        model = MetaLabelModel(model_dir=str(tmp))
        model.train(X, y)
        return model, X

    def test_predict_output_shapes(self, trained) -> None:
        model, X = trained
        proba, uncertainty = model.predict(X)
        assert proba.shape == (len(X),)
        assert uncertainty.shape == (len(X),)

    def test_predict_proba_in_unit_interval(self, trained) -> None:
        model, X = trained
        proba, _ = model.predict(X)
        assert (proba >= 0.0).all() and (proba <= 1.0).all()

    def test_predict_uncertainty_non_negative(self, trained) -> None:
        model, X = trained
        _, uncertainty = model.predict(X)
        assert (uncertainty >= 0.0).all()

    def test_predict_uncertainty_max_at_half(self, trained) -> None:
        """Bernoulli variance p*(1-p) is at most 0.25."""
        model, X = trained
        _, uncertainty = model.predict(X)
        assert float(uncertainty.max()) <= 0.25 + 1e-9

    def test_predict_before_train_raises(self, tmp_path) -> None:
        model = MetaLabelModel(model_dir=str(tmp_path))
        X, _ = _make_xy(n=5)
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(X)


# ---------------------------------------------------------------------------
# MetaLabelModel — save / load
# ---------------------------------------------------------------------------

class TestMetaLabelModelPersistence:
    def test_save_creates_versioned_pkl(self, tmp_path) -> None:
        X, y = _make_xy()
        model = MetaLabelModel(model_dir=str(tmp_path))
        model.train(X, y)
        path = model.save()
        assert path.exists()
        assert path.suffix == ".pkl"
        assert path.name.startswith("v1_")

    def test_save_creates_metadata_json(self, tmp_path) -> None:
        X, y = _make_xy()
        model = MetaLabelModel(model_dir=str(tmp_path))
        model.train(X, y)
        model.save()
        meta_path = tmp_path / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["version"] == 1
        assert "f1_score" in meta
        assert "brier_score" in meta
        assert "xgboost_params" in meta

    def test_save_increments_version(self, tmp_path) -> None:
        X, y = _make_xy()
        last_path = None
        for _ in range(3):
            model = MetaLabelModel(model_dir=str(tmp_path))
            model.train(X, y)
            last_path = model.save()
        assert last_path.name.startswith("v3_")

    def test_load_predictions_match_original(self, tmp_path) -> None:
        X, y = _make_xy()
        model = MetaLabelModel(model_dir=str(tmp_path))
        model.train(X, y)
        saved_path = model.save()
        proba_orig, _ = model.predict(X)

        loaded = MetaLabelModel.load(str(saved_path))
        proba_loaded, _ = loaded.predict(X)
        np.testing.assert_allclose(proba_orig, proba_loaded, rtol=1e-5)

    def test_load_restores_metrics_from_metadata(self, tmp_path) -> None:
        X, y = _make_xy()
        model = MetaLabelModel(model_dir=str(tmp_path))
        model.train(X, y)
        saved_path = model.save()

        loaded = MetaLabelModel.load(str(saved_path))
        assert "f1_score" in loaded._metrics
        assert "brier_score" in loaded._metrics

    def test_save_before_train_raises(self, tmp_path) -> None:
        model = MetaLabelModel(model_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="not trained"):
            model.save()

    def test_load_predict_before_predict_guard_not_triggered(self, tmp_path) -> None:
        """Loaded model should be immediately usable (no 'not trained' error)."""
        X, y = _make_xy()
        model = MetaLabelModel(model_dir=str(tmp_path))
        model.train(X, y)
        saved = model.save()
        loaded = MetaLabelModel.load(str(saved))
        proba, unc = loaded.predict(X[:10])
        assert len(proba) == 10


# ---------------------------------------------------------------------------
# MetaLabelModel — calibration metrics
# ---------------------------------------------------------------------------

class TestMetaLabelModelCalibration:
    def test_get_calibration_metrics_keys(self, tmp_path) -> None:
        X, y = _make_xy()
        model = MetaLabelModel(model_dir=str(tmp_path))
        model.train(X, y)
        cal = model.get_calibration_metrics(X, y)
        for key in ("brier_score", "f1_at_threshold_0_5", "calibration_bins", "n_samples"):
            assert key in cal

    def test_n_samples_matches_input(self, tmp_path) -> None:
        X, y = _make_xy(n=150)
        model = MetaLabelModel(model_dir=str(tmp_path))
        model.train(X, y)
        cal = model.get_calibration_metrics(X, y)
        assert cal["n_samples"] == 150

    def test_brier_low_on_separable_data(self, tmp_path) -> None:
        """Well-separated classes should produce Brier score well below 0.25.

        Classes are interleaved (alternating 0/1) so every time-series fold
        contains both labels — avoiding TimeSeriesSplit single-class failures.
        """
        rng = np.random.default_rng(7)
        n = 200
        # Interleave: even rows → class 0 (mean −2), odd rows → class 1 (mean +2)
        labels = np.array([i % 2 for i in range(n)])
        means = np.where(labels[:, None] == 0, -2.0, 2.0)
        features = means + rng.standard_normal((n, len(FEATURE_COLUMNS))) * 0.4
        X_sep = pd.DataFrame(
            features,
            columns=FEATURE_COLUMNS,
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )
        y_sep = pd.Series(labels, index=X_sep.index)

        model = MetaLabelModel(model_dir=str(tmp_path))
        model.train(X_sep, y_sep)
        cal = model.get_calibration_metrics(X_sep, y_sep)
        assert cal["brier_score"] < 0.20, (
            f"Expected Brier < 0.20 on separable data, got {cal['brier_score']:.3f}"
        )

    def test_calibration_bins_have_matching_lengths(self, tmp_path) -> None:
        X, y = _make_xy()
        model = MetaLabelModel(model_dir=str(tmp_path))
        model.train(X, y)
        cal = model.get_calibration_metrics(X, y)
        bins = cal["calibration_bins"]
        assert len(bins["predicted"]) == len(bins["observed"])

    def test_get_calibration_metrics_before_train_raises(self, tmp_path) -> None:
        X, y = _make_xy(n=50)
        model = MetaLabelModel(model_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="not trained"):
            model.get_calibration_metrics(X, y)
