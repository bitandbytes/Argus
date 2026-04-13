"""
Classical Quant Engine — Layer 2 of the cascade pipeline.

Combines normalized indicator signals into a regime-weighted composite score,
optionally boosted by multi-timeframe (weekly) confirmation. Outputs a
TradeSignal for every bar requested.

Design principles (from quant-engine-dev skill):
  1. Never imports specific plugins — iterates IndicatorPlugin instances from registry.
  2. Composite = Σ(weight_i × normalized_score_i), always clipped to [-1, +1].
  3. Weights per regime must sum to 1.0 (validated on load; normalized if not).
  4. Multi-timeframe boost only increases confidence; never pushes above 1.0.
  5. Sentiment is treated as a synthetic indicator key injected by the caller.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from src.models.trade_signal import RegimeType, TradeSignal
from src.plugins.base import IndicatorPlugin
from src.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)

# Fallback weights used when no config file is found.
# Derived directly from architecture doc §4.2 (equal-spread VOLATILE row).
_HARDCODED_FALLBACK: Dict[str, Dict[str, float]] = {
    "TRENDING_UP": {
        "sma_crossover": 0.35,
        "macd": 0.25,
        "rsi": 0.10,
        "bollinger": 0.10,
        "donchian": 0.05,
        "volume": 0.05,
        "sentiment": 0.10,
    },
    "TRENDING_DOWN": {
        "sma_crossover": 0.35,
        "macd": 0.25,
        "rsi": 0.10,
        "bollinger": 0.10,
        "donchian": 0.05,
        "volume": 0.05,
        "sentiment": 0.10,
    },
    "RANGING": {
        "sma_crossover": 0.10,
        "macd": 0.10,
        "rsi": 0.25,
        "bollinger": 0.25,
        "donchian": 0.05,
        "volume": 0.10,
        "sentiment": 0.15,
    },
    "VOLATILE": {
        "sma_crossover": 0.15,
        "macd": 0.15,
        "rsi": 0.15,
        "bollinger": 0.20,
        "donchian": 0.10,
        "volume": 0.10,
        "sentiment": 0.15,
    },
}

_DEFAULT_PARAMS_PATH = "config/cluster_params/cluster_default.yaml"


class QuantEngine:
    """
    Layer 2 — Classical quant signal generator.

    Combines plugin-based indicator signals into a composite directional score
    weighted by the current market regime. Sentiment (FinBERT) is wired in as
    a synthetic indicator key — the caller passes ``sentiment_score`` directly
    rather than the engine calling the enricher.

    **Lookahead safety:** ``generate_signal`` operates on the last row of
    whatever DataFrame is passed in. ``generate_series`` slices the DataFrame
    forward-only (``df.iloc[:i+1]``) so each bar only sees past data.

    **Weight loading:** Tries ``params_path`` → ``cluster_default.yaml`` →
    hardcoded dict. The engine always starts, even with missing config files.

    Args:
        registry: Populated ``PluginRegistry`` instance.
        settings_path: Path to ``config/settings.yaml``.
        params_path: Path to a regime-weight YAML (e.g., a cluster or stock
            override file). ``None`` uses the cluster default.
    """

    def __init__(
        self,
        registry: PluginRegistry,
        settings_path: str = "config/settings.yaml",
        params_path: Optional[str] = None,
    ) -> None:
        self._indicators: List[IndicatorPlugin] = registry.get_all_indicators()

        with open(settings_path) as f:
            cfg = yaml.safe_load(f)

        quant_cfg = cfg.get("quant", {})
        self._entry_threshold: float = float(quant_cfg.get("entry_confidence_threshold", 0.30))
        self._exit_threshold: float = float(quant_cfg.get("exit_confidence_threshold", 0.20))
        self._mtf_boost: float = float(quant_cfg.get("multi_timeframe_boost", 1.15))

        self.regime_weights: Dict[RegimeType, Dict[str, float]] = self._load_regime_weights(
            params_path
        )

        logger.info(
            "QuantEngine ready — %d indicators, entry_threshold=%.2f, mtf_boost=%.2f",
            len(self._indicators),
            self._entry_threshold,
            self._mtf_boost,
        )

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def generate_signal(
        self,
        df: pd.DataFrame,
        regime: RegimeType,
        ticker: str,
        sentiment_score: float = 0.0,
    ) -> TradeSignal:
        """
        Generate a trade signal for the most recent bar in df.

        Args:
            df: OHLCV DataFrame indexed by date. Must contain at least enough
                rows for the slowest indicator to produce a valid value.
            regime: Current market regime (from RegimeDetector).
            ticker: Stock ticker symbol (used for logging and TradeSignal).
            sentiment_score: FinBERT score in [-1, +1]. Defaults to 0.0
                (Phase 1 stub — wire up FinBERTEnricher in Phase 3).

        Returns:
            :class:`TradeSignal` with direction, confidence, and all indicator
            scores captured in ``features``.
        """
        scores: Dict[str, float] = self._compute_scores(df, ticker)
        scores["sentiment"] = float(sentiment_score)

        composite = self._weighted_composite(scores, regime)
        confidence = self._apply_mtf_boost(abs(composite), df, composite)

        return TradeSignal(
            ticker=ticker,
            timestamp=(
                df.index[-1].to_pydatetime()
                if hasattr(df.index[-1], "to_pydatetime")
                else df.index[-1]
            ),
            direction=composite,
            confidence=confidence,
            source_layer="quant",
            regime=regime,
            features=scores,
        )

    def generate_series(
        self,
        df: pd.DataFrame,
        regime_series: "pd.Series[RegimeType]",
        ticker: str,
        sentiment_series: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Generate a TradeSignal for every bar in df (forward-only, for backtesting).

        Each bar at index ``i`` only sees ``df.iloc[:i+1]`` — strictly no
        lookahead. Bars that lack sufficient data for any indicator return a
        neutral signal (direction=0, confidence=0).

        Args:
            df: Full OHLCV DataFrame.
            regime_series: Series of RegimeType aligned to df.index.
            ticker: Stock ticker symbol.
            sentiment_series: Optional FinBERT scores aligned to df.index.
                If None, uses 0.0 for every bar.

        Returns:
            ``pd.Series`` of :class:`TradeSignal` aligned to ``df.index``.
        """
        signals = []
        for i, idx in enumerate(df.index):
            window = df.iloc[: i + 1]
            regime = regime_series.at[idx] if idx in regime_series.index else RegimeType.VOLATILE
            sent = (
                float(sentiment_series.at[idx])
                if sentiment_series is not None and idx in sentiment_series.index
                else 0.0
            )
            try:
                sig = self.generate_signal(window, regime, ticker, sent)
            except Exception:
                sig = TradeSignal(
                    ticker=ticker,
                    timestamp=idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
                    direction=0.0,
                    confidence=0.0,
                    source_layer="quant",
                    regime=regime,
                )
            signals.append(sig)

        return pd.Series(signals, index=df.index, name="signal")

    def should_exit(
        self,
        df: pd.DataFrame,
        position_direction: int,
        regime: RegimeType,
        ticker: str,
        sentiment_score: float = 0.0,
    ) -> bool:
        """
        Determine whether to exit an open position.

        In trending regimes, hold until the signal flips direction.
        In ranging/volatile regimes, exit when confidence falls below the
        exit threshold.

        Args:
            df: Current OHLCV DataFrame.
            position_direction: +1 (long) or -1 (short).
            regime: Current market regime.
            ticker: Stock ticker symbol.
            sentiment_score: Current FinBERT score.

        Returns:
            True if the position should be closed.
        """
        signal = self.generate_signal(df, regime, ticker, sentiment_score)

        if regime in (RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN):
            return (position_direction > 0 and signal.direction < -0.20) or (
                position_direction < 0 and signal.direction > 0.20
            )
        return signal.confidence < self._exit_threshold

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_scores(self, df: pd.DataFrame, ticker: str) -> Dict[str, float]:
        """
        Run each indicator, normalize its output, and return the latest value.

        Indicators that produce NaN at the latest bar (insufficient history)
        contribute 0.0 with a warning rather than crashing the engine.

        Args:
            df: OHLCV DataFrame.
            ticker: Used only for warning messages.

        Returns:
            Dict mapping indicator name → normalized score in [-1, +1].
        """
        scores: Dict[str, float] = {}
        for ind in self._indicators:
            params = ind.get_default_params()
            try:
                enriched = ind.compute(df, params)
                col = ind.output_column
                if col not in enriched.columns:
                    logger.warning(
                        "%s: indicator '%s' did not produce column '%s' — scoring 0.0",
                        ticker,
                        ind.name,
                        col,
                    )
                    scores[ind.name] = 0.0
                    continue

                raw = enriched[col]
                normalized = ind.normalize(raw)
                latest = normalized.iloc[-1]

                if pd.isna(latest):
                    logger.debug(
                        "%s: indicator '%s' returned NaN at latest bar — scoring 0.0",
                        ticker,
                        ind.name,
                    )
                    scores[ind.name] = 0.0
                else:
                    scores[ind.name] = float(latest)

            except Exception as exc:
                logger.warning(
                    "%s: indicator '%s' raised %s — scoring 0.0: %s",
                    ticker,
                    ind.name,
                    type(exc).__name__,
                    exc,
                )
                scores[ind.name] = 0.0

        return scores

    def _weighted_composite(self, scores: Dict[str, float], regime: RegimeType) -> float:
        """
        Compute the regime-weighted composite signal.

        Unknown indicator names in the weights dict get weight 0.0 (logged at
        DEBUG). Weights are pre-normalized on load so the composite is already
        bounded by [-1, +1] when all indicators agree, but an explicit clip is
        applied for safety.

        Args:
            scores: Dict of indicator name → normalized score.
            regime: Current market regime.

        Returns:
            Composite score in [-1, +1].
        """
        weights = self.regime_weights.get(regime, {})
        total = 0.0
        for name, weight in weights.items():
            score = scores.get(name, 0.0)
            total += weight * score

        return max(-1.0, min(1.0, total))

    def _apply_mtf_boost(self, confidence: float, df: pd.DataFrame, composite: float) -> float:
        """
        Boost confidence by ``multi_timeframe_boost`` when weekly trend aligns.

        The weekly check uses the last 4 vs last 10 weekly closes (SMA on
        weekly bars). Requires at least 20 weekly bars; returns unmodified
        confidence otherwise.

        Args:
            confidence: Current confidence (= |composite|).
            df: OHLCV DataFrame with DatetimeIndex (needed for weekly resample).
            composite: Daily composite direction.

        Returns:
            Confidence in [0, 1], boosted if weekly agrees.
        """
        if self._weekly_confirms(df, composite):
            confidence = min(1.0, confidence * self._mtf_boost)
        return confidence

    def _weekly_confirms(self, df: pd.DataFrame, daily_composite: float) -> bool:
        """
        Check whether the weekly SMA trend agrees with the daily composite direction.

        Uses ``resample("W-FRI")`` so each weekly bar closes on Friday, matching
        standard weekly chart conventions.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            daily_composite: The computed daily composite score.

        Returns:
            True if weekly trend direction matches daily composite sign.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return False

        weekly = df["close"].resample("W-FRI").last().dropna()
        if len(weekly) < 20:
            return False

        sma_short = weekly.rolling(4).mean().iloc[-1]
        sma_long = weekly.rolling(10).mean().iloc[-1]

        if pd.isna(sma_short) or pd.isna(sma_long):
            return False

        weekly_direction = 1 if sma_short > sma_long else -1
        return (weekly_direction > 0 and daily_composite > 0) or (
            weekly_direction < 0 and daily_composite < 0
        )

    def _load_regime_weights(
        self, params_path: Optional[str]
    ) -> Dict[RegimeType, Dict[str, float]]:
        """
        Load and validate regime weights from a YAML file.

        Resolution order:
          1. ``params_path`` (if provided and exists)
          2. ``config/cluster_params/cluster_default.yaml``
          3. Hardcoded fallback dict

        Each regime's weights are normalized to sum exactly to 1.0.

        Args:
            params_path: Optional explicit path to a weights YAML.

        Returns:
            Dict mapping RegimeType → {indicator_name: weight}.
        """
        candidates = []
        if params_path:
            candidates.append(Path(params_path))
        candidates.append(Path(_DEFAULT_PARAMS_PATH))

        raw: Optional[Dict] = None
        for candidate in candidates:
            if candidate.exists():
                with open(candidate) as f:
                    data = yaml.safe_load(f)
                raw = data.get("indicators", {}).get("weights", {})
                logger.info("Loaded regime weights from '%s'", candidate)
                break

        if raw is None:
            logger.warning("No regime weight config found — using hardcoded fallback defaults.")
            raw = _HARDCODED_FALLBACK

        result: Dict[RegimeType, Dict[str, float]] = {}
        for regime_key, weights in raw.items():
            try:
                regime = RegimeType(regime_key)
            except ValueError:
                logger.warning("Unknown regime key '%s' in weights config — skipping.", regime_key)
                continue

            weights = dict(weights)
            total = sum(weights.values())
            if abs(total - 1.0) > 1e-6:
                logger.warning(
                    "Regime '%s' weights sum to %.6f, not 1.0 — normalizing.", regime_key, total
                )
                weights = {k: v / total for k, v in weights.items()}

            result[regime] = weights

        # Ensure all four regimes are present — fill missing with equal spread.
        for regime in RegimeType:
            if regime not in result:
                logger.warning(
                    "Regime '%s' missing from weights config — using equal spread.", regime.value
                )
                active_names = [ind.name for ind in self._indicators] + ["sentiment"]
                n = len(active_names)
                result[regime] = {name: 1.0 / n for name in active_names}

        return result
