"""Tests for StockClusterer and module-level feature helpers.

Covers:
  - Feature extraction correctness (Hurst, ADX, autocorr, vol, etc.)
  - Clustering with fixed k and auto-selected k
  - Reproducibility (same seed → same labels)
  - YAML save/load round-trip
  - Input validation and edge cases
  - predict() for unseen tickers
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tuning.stock_clusterer import (
    StockClusterer,
    compute_hurst_exponent,
    compute_lag1_autocorr,
    compute_mean_adx,
    compute_mean_reversion_speed,
    compute_realized_vol,
    compute_volume_profile_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    n: int = 600,
    trend: float = 0.0003,
    noise: float = 0.015,
    seed: int = 0,
    volume_base: int = 1_000_000,
) -> pd.DataFrame:
    """Generate synthetic OHLCV with known statistical properties."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-04-15", periods=n)
    close = 100.0 * np.cumprod(1 + trend + rng.normal(0.0, noise, n))
    high = close * (1 + rng.uniform(0.001, 0.015, n))
    low = close * (1 - rng.uniform(0.001, 0.015, n))
    open_ = close * (1 + rng.normal(0.0, 0.005, n))
    volume = rng.integers(volume_base, volume_base * 5, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_ohlcv_dict(
    n_stocks: int = 10,
    n_bars: int = 600,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Return a dict of synthetic OHLCV DataFrames for n_stocks tickers."""
    tickers = [f"STK{i:02d}" for i in range(n_stocks)]
    rng = np.random.default_rng(seed)
    return {
        t: _make_ohlcv(
            n=n_bars,
            trend=rng.uniform(-0.0005, 0.001),
            noise=rng.uniform(0.008, 0.025),
            seed=rng.integers(0, 10_000),
        )
        for t in tickers
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_hurst_random_walk_near_half(self) -> None:
        """White-noise log returns should yield H ≈ 0.5."""
        rng = np.random.default_rng(1)
        prices = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, 600)))
        h = compute_hurst_exponent(prices)
        assert 0.1 <= h <= 0.9, f"Expected H near 0.5 for white noise, got {h:.3f}"

    def test_hurst_persistent_series_above_half(self) -> None:
        """Returns with strong positive autocorrelation should yield H > 0.5.

        R/S analysis removes the within-chunk mean, so a constant drift is
        irrelevant.  To produce H > 0.5 we need *positively autocorrelated*
        returns, constructed here via a long moving-average of white noise.
        """
        rng = np.random.default_rng(2)
        n, window = 800, 30
        # MA(30) of white noise → returns positively autocorrelated up to lag 30
        raw = rng.normal(0.0, 0.01, n + window)
        ma_returns = np.convolve(raw, np.ones(window) / window, mode="valid")[:n]
        prices = pd.Series(100.0 * np.cumprod(1.0 + 0.001 + ma_returns))
        h = compute_hurst_exponent(prices)
        assert h > 0.5, f"Expected H > 0.5 for MA-persistent series, got {h:.3f}"

    def test_hurst_insufficient_data_returns_half(self) -> None:
        prices = pd.Series([100.0, 101.0, 100.5])
        h = compute_hurst_exponent(prices)
        assert h == 0.5

    def test_hurst_result_in_unit_interval(self) -> None:
        rng = np.random.default_rng(3)
        prices = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.02, 300)))
        h = compute_hurst_exponent(prices)
        assert 0.0 <= h <= 1.0

    def test_realized_vol_positive(self) -> None:
        df = _make_ohlcv()
        vol = compute_realized_vol(df["close"])
        assert vol > 0.0

    def test_realized_vol_annualised(self) -> None:
        """Volatility should be in a plausible annualised range (5%–200%)."""
        df = _make_ohlcv(noise=0.015)
        vol = compute_realized_vol(df["close"])
        assert 0.05 <= vol <= 2.0, f"Unexpected vol: {vol:.4f}"

    def test_lag1_autocorr_within_range(self) -> None:
        df = _make_ohlcv()
        ac = compute_lag1_autocorr(df["close"])
        assert -1.0 <= ac <= 1.0

    def test_lag1_autocorr_short_series(self) -> None:
        """Fewer than 3 log-return points → returns 0.0 without error."""
        close = pd.Series([100.0, 101.0])
        ac = compute_lag1_autocorr(close)
        assert ac == 0.0

    def test_mean_adx_positive(self) -> None:
        df = _make_ohlcv()
        adx = compute_mean_adx(df["high"], df["low"], df["close"])
        assert adx >= 0.0

    def test_mean_reversion_speed_returns_float(self) -> None:
        df = _make_ohlcv()
        speed = compute_mean_reversion_speed(df["close"])
        assert isinstance(speed, float)
        assert not math.isnan(speed)

    def test_volume_profile_ratio_positive(self) -> None:
        df = _make_ohlcv()
        ratio = compute_volume_profile_ratio(df["close"], df["volume"])
        assert ratio > 0.0

    def test_volume_profile_ratio_rising_volume(self) -> None:
        """Series where recent volume >> past volume → ratio > 1."""
        n = 200
        vol = np.concatenate([np.ones(150) * 1_000, np.ones(50) * 10_000])
        close = pd.Series(np.ones(n) * 100.0)
        volume = pd.Series(vol.astype(float))
        ratio = compute_volume_profile_ratio(close, volume)
        assert ratio > 1.0, f"Expected ratio > 1, got {ratio:.3f}"


# ---------------------------------------------------------------------------
# StockClusterer — fit
# ---------------------------------------------------------------------------

class TestStockClustererFit:
    def test_fit_returns_self(self) -> None:
        """fit() should return self for chaining."""
        ohlcv = _make_ohlcv_dict(n_stocks=8)
        sc = StockClusterer(n_clusters=4, random_state=42)
        result = sc.fit(ohlcv)
        assert result is sc

    def test_labels_all_tickers_assigned(self) -> None:
        ohlcv = _make_ohlcv_dict(n_stocks=8)
        sc = StockClusterer(n_clusters=4, random_state=42)
        sc.fit(ohlcv)
        assert set(sc.labels_.keys()) == set(ohlcv.keys())

    def test_cluster_count_with_fixed_k(self) -> None:
        ohlcv = _make_ohlcv_dict(n_stocks=10)
        sc = StockClusterer(n_clusters=4, random_state=42)
        sc.fit(ohlcv)
        distinct = set(sc.labels_.values())
        assert len(distinct) == 4

    def test_auto_k_selects_within_range(self) -> None:
        """Without a fixed k, auto-selection should pick k ∈ [4..8]."""
        ohlcv = _make_ohlcv_dict(n_stocks=10)
        sc = StockClusterer(random_state=42)   # n_clusters=None
        sc.fit(ohlcv)
        k = len(set(sc.labels_.values()))
        assert 4 <= k <= 8, f"Auto-selected k={k} outside [4..8]"

    def test_silhouette_score_available_after_fit(self) -> None:
        ohlcv = _make_ohlcv_dict(n_stocks=8)
        sc = StockClusterer(n_clusters=4, random_state=42)
        sc.fit(ohlcv)
        score = sc.silhouette_score_
        assert -1.0 <= score <= 1.0

    def test_skips_short_stocks_with_warning(self, caplog) -> None:
        """Stocks with < _MIN_BARS rows are skipped (warning logged)."""
        import logging

        ohlcv = _make_ohlcv_dict(n_stocks=8)
        ohlcv["SHORT"] = _make_ohlcv(n=10)   # only 10 bars

        with caplog.at_level(logging.WARNING, logger="src.tuning.stock_clusterer"):
            sc = StockClusterer(n_clusters=4, random_state=42)
            sc.fit(ohlcv)

        assert "SHORT" not in sc.labels_
        assert any("SHORT" in r.message for r in caplog.records)

    def test_raises_with_too_few_usable_stocks(self) -> None:
        """< 2 stocks with enough data → RuntimeError."""
        ohlcv = {"ONLY_ONE": _make_ohlcv(n=600)}
        sc = StockClusterer(n_clusters=4)
        with pytest.raises(RuntimeError, match="at least 2"):
            sc.fit(ohlcv)

    def test_labels_before_fit_raises(self) -> None:
        sc = StockClusterer()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            _ = sc.labels_


# ---------------------------------------------------------------------------
# StockClusterer — stability
# ---------------------------------------------------------------------------

class TestStockClustererStability:
    def test_same_seed_same_labels(self) -> None:
        """Fitting twice on the same data with the same seed → identical labels."""
        ohlcv = _make_ohlcv_dict(n_stocks=10, seed=7)

        sc1 = StockClusterer(n_clusters=5, random_state=42)
        sc1.fit(ohlcv)

        sc2 = StockClusterer(n_clusters=5, random_state=42)
        sc2.fit(ohlcv)

        assert sc1.labels_ == sc2.labels_, "Labels differ with the same seed."

    def test_get_cluster_members_consistent(self) -> None:
        """get_cluster_members(id) should return the same tickers as labels_."""
        ohlcv = _make_ohlcv_dict(n_stocks=8, seed=3)
        sc = StockClusterer(n_clusters=4, random_state=42)
        sc.fit(ohlcv)

        for cid in set(sc.labels_.values()):
            members = set(sc.get_cluster_members(cid))
            from_labels = {t for t, c in sc.labels_.items() if c == cid}
            assert members == from_labels, (
                f"Cluster {cid}: get_cluster_members mismatch."
            )

    def test_different_seeds_may_differ(self) -> None:
        """Different random seeds may (and usually do) produce different labels."""
        ohlcv = _make_ohlcv_dict(n_stocks=10, seed=99)
        sc1 = StockClusterer(n_clusters=5, random_state=0)
        sc1.fit(ohlcv)
        sc2 = StockClusterer(n_clusters=5, random_state=999)
        sc2.fit(ohlcv)
        # Not asserting they differ — just asserting no crash and both are valid
        for lbl in (sc1.labels_, sc2.labels_):
            assert len(lbl) == len(ohlcv)


# ---------------------------------------------------------------------------
# StockClusterer — YAML I/O
# ---------------------------------------------------------------------------

class TestStockClustererIO:
    def test_save_creates_yaml_file(self, tmp_path: Path) -> None:
        ohlcv = _make_ohlcv_dict(n_stocks=8)
        sc = StockClusterer(n_clusters=4, random_state=42)
        sc.fit(ohlcv)
        out = tmp_path / "assignments.yaml"
        sc.save(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_returns_path(self, tmp_path: Path) -> None:
        ohlcv = _make_ohlcv_dict(n_stocks=8)
        sc = StockClusterer(n_clusters=4, random_state=42)
        sc.fit(ohlcv)
        returned = sc.save(tmp_path / "out.yaml")
        assert isinstance(returned, Path)

    def test_save_load_roundtrip_labels(self, tmp_path: Path) -> None:
        """Labels written to YAML must match those read back."""
        ohlcv = _make_ohlcv_dict(n_stocks=8, seed=5)
        sc = StockClusterer(n_clusters=4, random_state=42)
        sc.fit(ohlcv)

        p = tmp_path / "clusters.yaml"
        sc.save(p)
        loaded = StockClusterer.load(p)

        assert loaded.labels_ == sc.labels_

    def test_save_load_metadata_preserved(self, tmp_path: Path) -> None:
        ohlcv = _make_ohlcv_dict(n_stocks=8)
        sc = StockClusterer(n_clusters=4, random_state=42)
        sc.fit(ohlcv)

        p = tmp_path / "meta.yaml"
        sc.save(p)
        loaded = StockClusterer.load(p)

        assert loaded._fitted_k == sc._fitted_k
        assert loaded._method == sc._method
        assert loaded._lookback_days == sc._lookback_days

    def test_save_load_silhouette_preserved(self, tmp_path: Path) -> None:
        ohlcv = _make_ohlcv_dict(n_stocks=8)
        sc = StockClusterer(n_clusters=4, random_state=42)
        sc.fit(ohlcv)

        p = tmp_path / "sil.yaml"
        sc.save(p)
        loaded = StockClusterer.load(p)

        assert abs(loaded.silhouette_score_ - sc.silhouette_score_) < 1e-5

    def test_yaml_contains_all_tickers(self, tmp_path: Path) -> None:
        """Every fitted ticker should appear in the YAML file."""
        import yaml as _yaml

        ohlcv = _make_ohlcv_dict(n_stocks=8)
        sc = StockClusterer(n_clusters=4, random_state=42)
        sc.fit(ohlcv)
        p = tmp_path / "c.yaml"
        sc.save(p)

        with open(p) as f:
            data = _yaml.safe_load(f)

        saved_tickers = set(data["ticker_to_cluster"].keys())
        assert saved_tickers == set(ohlcv.keys())

    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        sc = StockClusterer()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            sc.save(tmp_path / "x.yaml")


# ---------------------------------------------------------------------------
# StockClusterer — predict
# ---------------------------------------------------------------------------

class TestStockClustererPredict:
    def test_predict_returns_valid_cluster_id(self) -> None:
        """predict() must return one of the known cluster IDs."""
        ohlcv = _make_ohlcv_dict(n_stocks=8, seed=10)
        sc = StockClusterer(n_clusters=4, random_state=42)
        sc.fit(ohlcv)

        new_df = _make_ohlcv(n=600, seed=999)
        cluster_id = sc.predict("NEW_TICKER", new_df)

        valid_ids = set(sc.labels_.values())
        assert cluster_id in valid_ids, (
            f"predict() returned {cluster_id} which is not a valid cluster ID."
        )

    def test_predict_before_fit_raises(self) -> None:
        sc = StockClusterer()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            sc.predict("AAPL", _make_ohlcv())
