"""
Market regime detector — HMM + ADX dual approach.

Layer 1 of the cascade pipeline. Runs before the QuantEngine to classify
the current market regime and supply regime-specific indicator weights.

Reconciliation logic (in priority order):
    1. HMM uncertainty > 0.40              → VOLATILE
    2. ADX < 20                            → RANGING
    3. ADX ≥ 20, HMM state = bull         → TRENDING_UP
    4. ADX ≥ 20, HMM state = bear         → TRENDING_DOWN
    5. ADX ≥ 20, HMM state = sideways     → RANGING
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from hmmlearn.hmm import GaussianHMM

from src.models.trade_signal import RegimeType

logger = logging.getLogger(__name__)

_BULL = "bull"
_BEAR = "bear"
_SIDEWAYS = "sideways"


class RegimeDetector:
    """
    Classifies the current market regime using a dual HMM + ADX approach.

    **HMM features:** log daily return and 20-day realised volatility (2 columns).
    GaussianHMM with 3 hidden states is trained on up to ``lookback_days`` of
    history. States are labelled "bull", "bear", "sideways" by their mean log
    return — the state with the highest mean is "bull", lowest is "bear".

    **ADX:** The fast classifier (Wilder's ADX-14). ADX > 25 = trending,
    ADX < 20 = ranging. ADX is used to override the HMM direction when the
    market is clearly ranging.

    **Persistence:** Fitted detectors are saved to ``{model_dir}/{ticker}.pkl``
    via joblib. Call :meth:`load` to reuse without retraining.

    **Phase 1 lookahead note:** :meth:`detect_series` uses the HMM Viterbi and
    forward-backward algorithms over the full observation sequence, so past
    states are informed by future observations. This is standard for Phase 1
    research. Phase 3 should replace with an online (forward-only) filter.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        lookback_days: int = 504,
        adx_period: int = 14,
        trend_threshold: float = 25.0,
        range_threshold: float = 20.0,
        uncertainty_threshold: float = 0.40,
        model_dir: str = "data/models/hmm",
        random_state: int = 42,
    ) -> None:
        """
        Configure regime detector parameters.

        All defaults match ``config/settings.yaml`` (regime section). Zero-arg
        construction gives production-ready defaults.

        Args:
            n_components: Number of HMM hidden states (default 3).
            covariance_type: GaussianHMM covariance type — ``"full"`` gives
                each state its own 2×2 covariance matrix.
            n_iter: Maximum EM iterations for HMM fitting.
            lookback_days: Training window size (~2 years of trading days).
            adx_period: ADX calculation period (Wilder's default = 14).
            trend_threshold: ADX above this → trending (default 25).
            range_threshold: ADX below this → ranging (default 20).
            uncertainty_threshold: HMM max-state-prob below ``1 - threshold``
                triggers VOLATILE (default 0.40).
            model_dir: Directory for persisted ``.pkl`` model files.
            random_state: Seed for HMM weight initialisation.
        """
        self._n_components = n_components
        self._covariance_type = covariance_type
        self._n_iter = n_iter
        self._lookback_days = lookback_days
        self._adx_period = adx_period
        self._trend_threshold = trend_threshold
        self._range_threshold = range_threshold
        self._uncertainty_threshold = uncertainty_threshold
        self._model_dir = Path(model_dir)
        self._random_state = random_state

        self._hmm: Optional[GaussianHMM] = None
        self._state_labels: Dict[int, str] = {}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame, ticker: str = "default") -> "RegimeDetector":
        """
        Train GaussianHMM on log returns + realised volatility.

        Uses the last ``lookback_days`` rows of *df*. States are labelled
        "bull", "bear", "sideways" by their mean log return. The fitted
        detector is saved to ``{model_dir}/{ticker}.pkl`` automatically.

        Args:
            df: OHLCV DataFrame with columns ``open``, ``high``, ``low``,
                ``close``, ``volume`` and a DatetimeIndex. At least
                ``n_components + 2`` rows required after NaN removal.
            ticker: Key for the persistence filename.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If fewer valid feature rows than ``n_components + 1``
                remain after removing NaN.
        """
        train_df = df.tail(self._lookback_days) if len(df) > self._lookback_days else df.copy()
        features, _ = self._compute_features(train_df)

        min_rows = self._n_components + 1
        if len(features) < min_rows:
            raise ValueError(
                f"Insufficient data: need at least {min_rows} valid feature rows "
                f"(after NaN removal), got {len(features)}. "
                f"Provide at least {min_rows + 1} rows of OHLCV data."
            )

        logger.info(
            "Fitting GaussianHMM(n_components=%d) on %d rows for '%s'",
            self._n_components,
            len(features),
            ticker,
        )

        self._hmm = GaussianHMM(
            n_components=self._n_components,
            covariance_type=self._covariance_type,
            n_iter=self._n_iter,
            random_state=self._random_state,
        )
        self._hmm.fit(features)
        self._state_labels = self._label_states(features)
        self._is_fitted = True

        self._model_dir.mkdir(parents=True, exist_ok=True)
        save_path = self._model_dir / f"{ticker}.pkl"
        joblib.dump(self, save_path)
        logger.info("RegimeDetector saved to '%s'", save_path)

        return self

    def detect(self, df: pd.DataFrame) -> RegimeType:
        """
        Classify the current (most recent day's) market regime.

        Uses HMM forward-backward posteriors on the last ``lookback_days``
        rows. The final row's state probability vector determines the regime.

        Args:
            df: OHLCV DataFrame.

        Returns:
            :class:`RegimeType` enum value for today.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        self._require_fitted("detect")

        window = df.tail(self._lookback_days) if len(df) > self._lookback_days else df
        features, _ = self._compute_features(window)

        if len(features) == 0:
            logger.warning("No valid features; defaulting to VOLATILE.")
            return RegimeType.VOLATILE

        proba = self._hmm.predict_proba(features)
        last_proba = proba[-1]
        state = int(np.argmax(last_proba))
        uncertainty = float(1.0 - last_proba.max())

        adx_series = self._compute_adx(window)
        adx_valid = adx_series.dropna()
        adx = float(adx_valid.iloc[-1]) if not adx_valid.empty else float(self._trend_threshold - 1)

        return self._reconcile(adx, state, uncertainty)

    def detect_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify the regime for every row in df (bulk / backtesting use).

        Rows that lack valid HMM features (early NaN rows) default to VOLATILE.
        ADX rows that are NaN (first ~28 rows) also default to VOLATILE.

        Args:
            df: OHLCV DataFrame.

        Returns:
            ``pd.Series`` of :class:`RegimeType` aligned to ``df.index``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        self._require_fitted("detect_series")

        features, valid_index = self._compute_features(df)
        adx_series = self._compute_adx(df)

        # Default everything to VOLATILE; overwrite valid rows below.
        regimes = pd.Series(RegimeType.VOLATILE, index=df.index, dtype=object)

        if len(features) == 0:
            return regimes

        states = self._hmm.predict(features)
        proba = self._hmm.predict_proba(features)
        uncertainties = 1.0 - proba.max(axis=1)

        adx_at_valid = adx_series.reindex(valid_index)

        for i, idx in enumerate(valid_index):
            adx_val = adx_at_valid.iloc[i]
            adx = float(adx_val) if pd.notna(adx_val) else float(self._trend_threshold - 1)
            regimes.at[idx] = self._reconcile(adx, int(states[i]), float(uncertainties[i]))

        return regimes

    def save(self, path: str) -> None:
        """
        Persist this detector to disk (explicit save, separate from fit()).

        Args:
            path: File path for the ``.pkl`` file.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        logger.debug("RegimeDetector saved to '%s'", dest)

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        """
        Load a persisted RegimeDetector from disk.

        Args:
            path: Path to a ``.pkl`` file created by :meth:`fit` or :meth:`save`.

        Returns:
            Loaded, already-fitted RegimeDetector.
        """
        detector: "RegimeDetector" = joblib.load(path)
        logger.debug("RegimeDetector loaded from '%s'", path)
        return detector

    def get_transition_matrix(self) -> np.ndarray:
        """
        Return the HMM state transition probability matrix.

        Returns:
            Array of shape ``(n_components, n_components)`` where entry
            ``[i, j]`` = P(next_state=j | current_state=i). Rows sum to 1.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        self._require_fitted("get_transition_matrix")
        assert self._hmm is not None
        return self._hmm.transmat_

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        """
        Build the 2-column HMM feature matrix from OHLCV data.

        Columns:
            - ``log_return_1d``: daily log return
            - ``realized_vol_20d``: 20-day rolling std of log returns × √252

        Args:
            df: OHLCV DataFrame.

        Returns:
            ``(features, valid_index)`` where ``features`` has shape
            ``(n_valid, 2)`` and ``valid_index`` is the DatetimeIndex of the
            included rows (NaN rows are excluded).
        """
        log_ret = np.log(df["close"] / df["close"].shift(1))
        realized_vol = log_ret.rolling(window=20, min_periods=1).std() * np.sqrt(252)

        feature_df = pd.DataFrame(
            {"log_ret": log_ret, "realized_vol": realized_vol},
            index=df.index,
        )
        valid_mask = feature_df.notna().all(axis=1)
        valid_df = feature_df.loc[valid_mask]

        return valid_df.to_numpy(dtype=np.float64), valid_df.index

    def _label_states(self, features: np.ndarray) -> Dict[int, str]:
        """
        Assign "bull", "bear", "sideways" labels to HMM state IDs.

        State IDs (0, 1, … n_components-1) are arbitrary after fitting.
        This method provides a stable semantic mapping by ranking states on
        their mean log return (column 0 of the feature matrix).

        Args:
            features: Training feature matrix, shape ``(n, 2)``.

        Returns:
            Dict mapping state_id (int) → label (str).
        """
        assert self._hmm is not None
        states = self._hmm.predict(features)

        mean_returns: Dict[int, float] = {
            state_id: (
                float(features[states == state_id, 0].mean()) if (states == state_id).any() else 0.0
            )
            for state_id in range(self._n_components)
        }

        sorted_ids = sorted(mean_returns, key=lambda k: mean_returns[k])
        # sorted_ids[0]  = lowest mean return  → bear
        # sorted_ids[-1] = highest mean return → bull
        # anything in between                  → sideways
        labels: Dict[int, str] = {
            sorted_ids[0]: _BEAR,
            sorted_ids[-1]: _BULL,
        }
        for sid in sorted_ids[1:-1]:
            labels[sid] = _SIDEWAYS

        logger.debug(
            "HMM state labels: %s | mean returns: %s",
            labels,
            {k: f"{v:.5f}" for k, v in mean_returns.items()},
        )
        return labels

    def _compute_adx(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute ADX using pandas-ta and return a Series aligned to df.index.

        Returns NaN for the first ~28 rows (ADX warm-up period). If pandas-ta
        returns ``None`` (insufficient data), returns an all-NaN Series.

        Args:
            df: OHLCV DataFrame with ``high``, ``low``, ``close`` columns.

        Returns:
            ``pd.Series`` of ADX values.
        """
        adx_result = ta.adx(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            length=self._adx_period,
        )

        if adx_result is None:
            return pd.Series(np.nan, index=df.index, name="ADX")

        adx_col = f"ADX_{self._adx_period}"
        if adx_col not in adx_result.columns:
            candidates = [c for c in adx_result.columns if str(c).startswith("ADX")]
            if not candidates:
                return pd.Series(np.nan, index=df.index, name="ADX")
            adx_col = candidates[0]

        return adx_result[adx_col].reindex(df.index)

    def _reconcile(self, adx: float, hmm_state: int, uncertainty: float) -> RegimeType:
        """
        Combine ADX signal and HMM state into a final RegimeType.

        Priority order:
            1. ``uncertainty > uncertainty_threshold`` → VOLATILE
            2. ``adx < range_threshold``               → RANGING
            3. HMM label determines direction:
               - "bull"     → TRENDING_UP
               - "bear"     → TRENDING_DOWN
               - "sideways" → RANGING

        Note: rules 3 applies for both the confirmed trending zone (ADX > 25)
        and the transition zone (20 ≤ ADX < 25). The ADX threshold only gates
        the RANGING override — above it, the HMM state determines direction.

        Args:
            adx: Current ADX(14) value.
            hmm_state: Most-probable HMM state ID (argmax of posteriors).
            uncertainty: ``1 - max(state_probabilities)`` for this timestep.

        Returns:
            :class:`RegimeType`.
        """
        if uncertainty > self._uncertainty_threshold:
            return RegimeType.VOLATILE

        if adx < self._range_threshold:
            return RegimeType.RANGING

        label = self._state_labels.get(hmm_state, _SIDEWAYS)
        if label == _BULL:
            return RegimeType.TRENDING_UP
        if label == _BEAR:
            return RegimeType.TRENDING_DOWN
        return RegimeType.RANGING

    def _require_fitted(self, method: str) -> None:
        """Raise RuntimeError if the model has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"RegimeDetector must be fitted before calling {method}(). "
                "Call fit(df) or load a pre-trained detector."
            )
