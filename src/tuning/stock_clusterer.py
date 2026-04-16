"""Stock clustering for the hybrid cluster-based tuning pipeline (Task 2.4).

Groups stocks by behavioral similarity so that walk-forward optimization
(Task 2.6) can pool training data within each cluster, giving each cluster
its own baseline parameter set.

Features extracted per stock (504-day lookback):
  - Hurst exponent       — trend persistence (R/S rescaled-range analysis)
  - Mean ADX             — momentum / directional strength
  - Lag-1 autocorrelation — mean-reversion propensity
  - Realized volatility  — annualised return std
  - Mean reversion speed — exponential decay rate of lag-k autocorrelation
  - Volume profile ratio — recent quarter vs. full-lookback average volume

Algorithms:
  - ``method="kmeans"`` — standard K-Means (sklearn)
  - ``method="dtw"``    — DTW-based TimeSeriesKMeans (tslearn)

k is either fixed at construction time or auto-selected via silhouette score
over the range k ∈ [4..8].

Usage::

    sc = StockClusterer(random_state=42)          # auto-select k
    sc.fit(ohlcv_dict)                            # {ticker: OHLCV DataFrame}
    print(sc.labels_)                             # {ticker: cluster_id}
    sc.save("config/cluster_assignments.yaml")

Re-clustering quarterly::

    sc.fit(updated_ohlcv_dict)
    sc.save("config/cluster_assignments.yaml")
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pandas_ta as ta
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Minimum bars needed for reliable Hurst / ADX computation
_MIN_BARS: int = 50
# Lag range for mean reversion speed estimate
_MR_MAX_LAG: int = 10
# k search range
_K_MIN: int = 4
_K_MAX: int = 8


# ---------------------------------------------------------------------------
# Feature helpers (module-level for testability)
# ---------------------------------------------------------------------------

def compute_hurst_exponent(prices: pd.Series, min_lags: int = 10) -> float:
    """Estimate Hurst exponent using R/S (rescaled-range) analysis.

    Args:
        prices: Price series (close prices, not returns).
        min_lags: Minimum chunk size for the R/S calculation.

    Returns:
        Hurst exponent in [0, 1].  0.5 ≈ random walk; >0.5 trending;
        <0.5 mean-reverting.  Returns 0.5 on insufficient data.
    """
    returns = np.log(prices / prices.shift(1)).dropna().values
    T = len(returns)

    if T < max(min_lags * 2, _MIN_BARS):
        return 0.5

    max_lag = T // 2
    if max_lag <= min_lags:
        return 0.5

    # Sample ~50 lag sizes logarithmically between min_lags and max_lag
    lags = np.unique(
        np.geomspace(min_lags, max_lag, num=50, dtype=int)
    )
    rs_values: List[float] = []
    valid_lags: List[int] = []

    for k in lags:
        n_chunks = T // k
        if n_chunks == 0:
            continue
        chunk_rs: List[float] = []
        for i in range(n_chunks):
            chunk = returns[i * k : (i + 1) * k]
            if len(chunk) < 2:
                continue
            mean_c = chunk.mean()
            Y = np.cumsum(chunk - mean_c)
            R = Y.max() - Y.min()
            S = chunk.std(ddof=1)
            if S > 0:
                chunk_rs.append(R / S)
        if chunk_rs:
            rs_values.append(np.mean(chunk_rs))
            valid_lags.append(k)

    if len(valid_lags) < 2:
        return 0.5

    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_values)
    H = float(np.polyfit(log_lags, log_rs, 1)[0])
    return float(np.clip(H, 0.0, 1.0))


def compute_mean_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> float:
    """Compute the mean ADX over the available history.

    Args:
        high, low, close: Price series with a shared DatetimeIndex.
        length: ADX period (default 14).

    Returns:
        Mean ADX value (float ≥ 0).  Returns 0.0 on error.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adx_df = ta.adx(high=high, low=low, close=close, length=length)
        if adx_df is None or adx_df.empty:
            return 0.0
        adx_col = f"ADX_{length}"
        if adx_col not in adx_df.columns:
            adx_col = [c for c in adx_df.columns if c.startswith("ADX")][0]
        return float(adx_df[adx_col].dropna().mean())
    except Exception:
        return 0.0


def compute_lag1_autocorr(close: pd.Series) -> float:
    """Lag-1 autocorrelation of log returns.

    Args:
        close: Close price series.

    Returns:
        Autocorrelation in [-1, 1].  Returns 0.0 if undefined.
    """
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < 3:
        return 0.0
    val = log_returns.autocorr(lag=1)
    return 0.0 if pd.isna(val) else float(val)


def compute_realized_vol(close: pd.Series) -> float:
    """Annualised realised volatility from log returns.

    Args:
        close: Close price series.

    Returns:
        Annualised volatility (float ≥ 0).
    """
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < 2:
        return 0.0
    return float(log_returns.std(ddof=1) * np.sqrt(252))


def compute_mean_reversion_speed(close: pd.Series, max_lag: int = _MR_MAX_LAG) -> float:
    """Estimate mean-reversion speed as the OLS decay rate of |autocorr(lag=k)|.

    A steep negative slope means autocorrelation decays quickly → fast mean
    reversion.  A flat slope → persistent (trending) behaviour.

    Args:
        close: Close price series.
        max_lag: Maximum lag to compute autocorrelation for.

    Returns:
        Decay slope (float, typically negative or near-zero).
    """
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < max_lag + 2:
        return 0.0

    lags = np.arange(1, max_lag + 1)
    ac_vals = np.array([
        log_returns.autocorr(lag=int(k)) for k in lags
    ])
    valid = ~np.isnan(ac_vals)
    if valid.sum() < 2:
        return 0.0

    slope = float(np.polyfit(lags[valid], np.abs(ac_vals[valid]), 1)[0])
    return slope


def compute_volume_profile_ratio(close: pd.Series, volume: pd.Series) -> float:
    """Ratio of recent-quarter average volume to full-history average volume.

    Values > 1 indicate rising volume (accumulation / distribution phase);
    values < 1 indicate drying volume.

    Args:
        close: Unused (kept for a consistent signature with other helpers).
        volume: Volume series.

    Returns:
        Ratio (float > 0).  Returns 1.0 if undefined.
    """
    v = volume.replace(0, np.nan).dropna()
    if len(v) < 2:
        return 1.0
    q = max(1, len(v) // 4)
    recent_mean = v.iloc[-q:].mean()
    total_mean = v.mean()
    if total_mean == 0:
        return 1.0
    return float(recent_mean / total_mean)


# ---------------------------------------------------------------------------
# StockClusterer
# ---------------------------------------------------------------------------

class StockClusterer:
    """Cluster stocks by behavioural similarity.

    Computes 6 statistical features per stock from raw OHLCV data, normalises
    them, and applies K-Means (default) or DTW-based clustering.

    Args:
        n_clusters: Number of clusters.  ``None`` → auto-select k ∈ [4..8]
            via silhouette score.
        method: ``"kmeans"`` (default) or ``"dtw"`` (requires tslearn).
        lookback_days: Calendar days of history used for feature extraction
            (default 504 ≈ 2 years).
        random_state: Seed for reproducibility.

    Raises:
        ValueError: If ``method`` is not ``"kmeans"`` or ``"dtw"``.
        RuntimeError: If fewer than 2 stocks have enough data to cluster.
    """

    FEATURE_NAMES: List[str] = [
        "hurst_exponent",
        "mean_adx",
        "lag1_autocorr",
        "realized_vol",
        "mean_reversion_speed",
        "volume_profile_ratio",
    ]

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        method: str = "kmeans",
        lookback_days: int = 504,
        random_state: int = 42,
    ) -> None:
        if method not in ("kmeans", "dtw"):
            raise ValueError(f"method must be 'kmeans' or 'dtw', got {method!r}.")
        self._n_clusters = n_clusters
        self._method = method
        self._lookback_days = lookback_days
        self._random_state = random_state

        # Set after fit()
        self._labels: Dict[str, int] = {}
        self._silhouette: float = float("nan")
        self._fitted_k: Optional[int] = None
        self._scaler: Optional[StandardScaler] = None
        self._km: Optional[object] = None  # KMeans or TimeSeriesKMeans
        self._feature_matrix: Optional[np.ndarray] = None
        self._fitted_tickers: List[str] = []

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, ohlcv_dict: Dict[str, pd.DataFrame]) -> "StockClusterer":
        """Extract features and cluster all provided tickers.

        Args:
            ohlcv_dict: Mapping ``{ticker: DataFrame}`` where each DataFrame
                has columns ``["open", "high", "low", "close", "volume"]``
                and a ``DatetimeIndex``.  At least ``_MIN_BARS`` rows required
                per stock (shorter ones are skipped with a warning).

        Returns:
            self (for chaining).

        Raises:
            RuntimeError: If fewer than 2 stocks are usable.
        """
        features: Dict[str, np.ndarray] = {}

        for ticker, df in ohlcv_dict.items():
            df = self._tail_lookback(df)
            if len(df) < _MIN_BARS:
                logger.warning(
                    "Skipping %s: only %d bars (need %d).", ticker, len(df), _MIN_BARS
                )
                continue
            try:
                vec = self._extract_features(df)
                features[ticker] = vec
            except Exception as exc:
                logger.warning("Skipping %s: feature extraction failed: %s", ticker, exc)

        if len(features) < 2:
            raise RuntimeError(
                f"Need at least 2 usable stocks to cluster, got {len(features)}."
            )

        tickers = list(features.keys())
        X = np.array([features[t] for t in tickers])

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        if self._method == "dtw":
            labels, k, score = self._fit_dtw(X_scaled, tickers, ohlcv_dict)
        else:
            labels, k, score = self._fit_kmeans(X_scaled)

        self._labels = dict(zip(tickers, labels.tolist()))
        self._fitted_k = k
        self._silhouette = score
        self._feature_matrix = X_scaled
        self._fitted_tickers = tickers

        logger.info(
            "StockClusterer: k=%d, silhouette=%.4f, method=%s",
            k, score, self._method,
        )
        return self

    def predict(self, ticker: str, ohlcv: pd.DataFrame) -> int:
        """Assign one ticker to the nearest cluster (post-fit).

        Args:
            ticker: Ticker symbol (informational only).
            ohlcv: OHLCV DataFrame with the standard columns.

        Returns:
            Integer cluster ID.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        self._require_fitted()
        df = self._tail_lookback(ohlcv)
        vec = self._extract_features(df).reshape(1, -1)
        vec_scaled = self._scaler.transform(vec)
        cluster_id = int(self._km.predict(vec_scaled)[0])
        return cluster_id

    def get_cluster_members(self, cluster_id: int) -> List[str]:
        """Return sorted list of tickers assigned to ``cluster_id``.

        Args:
            cluster_id: Cluster integer label.

        Returns:
            Sorted list of ticker strings.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        self._require_fitted()
        return sorted(t for t, c in self._labels.items() if c == cluster_id)

    def save(self, path: Union[str, Path]) -> Path:
        """Persist cluster assignments and metadata to a YAML file.

        The file is written atomically (temp file → rename) to avoid
        corrupt output on interruption.

        Args:
            path: Target path (e.g. ``"config/cluster_assignments.yaml"``).

        Returns:
            Resolved ``Path`` of the written file.
        """
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build cluster → members map
        cluster_ids = sorted(set(self._labels.values()))
        clusters: Dict[str, List[str]] = {
            str(cid): self.get_cluster_members(cid) for cid in cluster_ids
        }

        payload: dict = {
            "metadata": {
                "n_clusters": self._fitted_k,
                "method": self._method,
                "features": self.FEATURE_NAMES,
                "silhouette_score": round(float(self._silhouette), 6)
                if not np.isnan(self._silhouette)
                else None,
                "lookback_days": self._lookback_days,
                "generated_date": date.today().isoformat(),
            },
            "ticker_to_cluster": {t: int(c) for t, c in sorted(self._labels.items())},
            "clusters": clusters,
        }

        # Atomic write
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".yaml.tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(
                    f"# Stock cluster assignments — generated by StockClusterer\n"
                    f"# silhouette_score: {payload['metadata']['silhouette_score']}\n"
                    f"# generated_date:   {payload['metadata']['generated_date']}\n\n"
                )
                yaml.dump(payload, fh, default_flow_style=False, sort_keys=False)
            os.replace(tmp, path)
        except Exception:
            os.unlink(tmp)
            raise

        logger.info("Cluster assignments saved to %s", path)
        return path.resolve()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "StockClusterer":
        """Reconstruct a :class:`StockClusterer` from a saved YAML file.

        Only ``labels_``, ``silhouette_score_``, and metadata are restored.
        The underlying KMeans model is *not* restored (``predict()`` will
        raise until ``fit()`` is called again).

        Args:
            path: Path to a YAML file previously written by :meth:`save`.

        Returns:
            Partially-restored :class:`StockClusterer` instance.
        """
        with open(path) as fh:
            data = yaml.safe_load(fh)

        meta = data.get("metadata", {})
        inst = cls(
            n_clusters=meta.get("n_clusters"),
            method=meta.get("method", "kmeans"),
            lookback_days=meta.get("lookback_days", 504),
        )
        inst._labels = {
            t: int(c) for t, c in data.get("ticker_to_cluster", {}).items()
        }
        inst._fitted_k = meta.get("n_clusters")
        raw_score = meta.get("silhouette_score")
        inst._silhouette = float(raw_score) if raw_score is not None else float("nan")
        return inst

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def labels_(self) -> Dict[str, int]:
        """Return ``{ticker → cluster_id}`` mapping after :meth:`fit`."""
        self._require_fitted()
        return dict(self._labels)

    @property
    def silhouette_score_(self) -> float:
        """Silhouette coefficient of the fitted model (float in [-1, 1])."""
        self._require_fitted()
        return self._silhouette

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _tail_lookback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the last ``lookback_days`` calendar days of ``df``."""
        if df.empty:
            return df
        end = df.index[-1]
        start = end - pd.Timedelta(days=self._lookback_days)
        return df.loc[df.index >= start]

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute the 6-element feature vector for a single stock.

        Args:
            df: OHLCV DataFrame with at least ``_MIN_BARS`` rows and columns
                ``high``, ``low``, ``close``, ``volume``.

        Returns:
            ``np.ndarray`` of shape ``(6,)``.
        """
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        hurst = compute_hurst_exponent(close)
        mean_adx = compute_mean_adx(high, low, close)
        autocorr = compute_lag1_autocorr(close)
        real_vol = compute_realized_vol(close)
        mr_speed = compute_mean_reversion_speed(close)
        vol_profile = compute_volume_profile_ratio(close, volume)

        return np.array([hurst, mean_adx, autocorr, real_vol, mr_speed, vol_profile],
                        dtype=float)

    def _fit_kmeans(
        self, X_scaled: np.ndarray
    ) -> tuple[np.ndarray, int, float]:
        """Fit K-Means, auto-selecting k if needed.

        Returns:
            ``(labels, k, silhouette_score)``
        """
        if self._n_clusters is not None:
            k = self._n_clusters
            km = KMeans(n_clusters=k, random_state=self._random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = (
                silhouette_score(X_scaled, labels)
                if len(set(labels)) >= 2
                else float("nan")
            )
            self._km = km
            return labels, k, float(score)

        # Auto-select k
        best_k, best_score, best_km, best_labels = _K_MIN, -2.0, None, None
        for k in range(_K_MIN, _K_MAX + 1):
            if k > len(X_scaled):
                break
            km = KMeans(n_clusters=k, random_state=self._random_state, n_init=10)
            lbl = km.fit_predict(X_scaled)
            if len(set(lbl)) < 2:
                continue
            score = silhouette_score(X_scaled, lbl)
            if score > best_score:
                best_score, best_k, best_km, best_labels = score, k, km, lbl

        if best_km is None:
            # Fallback: just use _K_MIN
            best_km = KMeans(n_clusters=_K_MIN, random_state=self._random_state, n_init=10)
            best_labels = best_km.fit_predict(X_scaled)
            best_score = float("nan")
            best_k = _K_MIN

        self._km = best_km
        return best_labels, best_k, float(best_score)

    def _fit_dtw(
        self,
        X_scaled: np.ndarray,
        tickers: List[str],
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> tuple[np.ndarray, int, float]:
        """Fit DTW-based TimeSeriesKMeans (tslearn).

        Uses the normalised log-return time series as input (not the aggregated
        feature vector, which is meaningless for DTW).

        Returns:
            ``(labels, k, silhouette_score)``
        """
        try:
            from tslearn.clustering import TimeSeriesKMeans
            from tslearn.preprocessing import TimeSeriesScalerMeanVariance
        except ImportError as exc:
            raise ImportError(
                "tslearn is required for method='dtw'. "
                "Install it with: pip install tslearn"
            ) from exc

        # Build time-series matrix (n_stocks × T × 1) using shortest series length
        series_list = []
        min_len = None
        for ticker in tickers:
            df = self._tail_lookback(ohlcv_dict[ticker])
            lr = np.log(df["close"] / df["close"].shift(1)).dropna().values
            series_list.append(lr)
            min_len = len(lr) if min_len is None else min(min_len, len(lr))

        if min_len is None or min_len < 2:
            raise RuntimeError("Insufficient data for DTW clustering.")

        X_ts = np.array([s[-min_len:] for s in series_list]).reshape(
            len(series_list), min_len, 1
        )
        scaler_ts = TimeSeriesScalerMeanVariance()
        X_ts_scaled = scaler_ts.fit_transform(X_ts)

        k = self._n_clusters or self._auto_select_k_dtw(X_ts_scaled)
        km = TimeSeriesKMeans(
            n_clusters=k,
            metric="dtw",
            random_state=self._random_state,
            n_init=3,
            n_jobs=1,
        )
        labels = km.fit_predict(X_ts_scaled)
        self._km = km

        score = (
            silhouette_score(X_scaled, labels)
            if len(set(labels)) >= 2
            else float("nan")
        )
        return labels, k, float(score)

    def _auto_select_k_dtw(self, X_ts: np.ndarray) -> int:
        """Select best k for DTW via inertia elbow (silhouette on DTW is slow)."""
        try:
            from tslearn.clustering import TimeSeriesKMeans
        except ImportError:
            return _K_MIN

        best_k = _K_MIN
        prev_inertia: Optional[float] = None
        for k in range(_K_MIN, min(_K_MAX + 1, len(X_ts))):
            km = TimeSeriesKMeans(
                n_clusters=k, metric="dtw", random_state=self._random_state, n_init=2
            )
            km.fit(X_ts)
            inertia = float(km.inertia_)
            if prev_inertia is not None and inertia > prev_inertia * 0.8:
                break
            best_k = k
            prev_inertia = inertia
        return best_k

    def _require_fitted(self) -> None:
        if not self._labels:
            raise RuntimeError(
                "StockClusterer has not been fitted yet. Call fit() first."
            )
