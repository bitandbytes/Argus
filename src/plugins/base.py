"""
Abstract base classes for the plugin system.

The trading pipeline supports four plugin types:
    - IndicatorPlugin: Computes a technical indicator and normalizes to [-1, +1]
    - SmoothingPlugin: Pre-processes price/volume series (e.g., Kalman filter)
    - DataEnricher: Adds external context features (e.g., FinBERT sentiment)
    - SignalFilter: Post-processes TradeSignals (e.g., LLM validator)

All plugins are discovered via config/plugins.yaml at startup. Core pipeline
code never imports plugin implementations directly.

See .claude/skills/plugin-author/SKILL.md for the full plugin development guide.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import pandas as pd


# ============================================================================
# Supporting Types
# ============================================================================


@dataclass
class ParamSpec:
    """
    Specification for a tunable parameter.

    Used by Optuna to construct the search space for Bayesian hyperparameter
    optimization. Every IndicatorPlugin and SmoothingPlugin must declare its
    tunable params via get_tunable_params().
    """

    name: str
    type: Literal["int", "float", "categorical"]
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Any = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.type in ("int", "float") and (self.low is None or self.high is None):
            raise ValueError(f"ParamSpec {self.name}: int/float types require low and high")
        if self.type == "categorical" and not self.choices:
            raise ValueError(f"ParamSpec {self.name}: categorical type requires choices")


@dataclass
class SmoothResult:
    """
    Output of a SmoothingPlugin.

    Contains the smoothed series plus auxiliary signals that downstream
    components (indicators, regime detector, exit logic) may use.
    """

    smoothed: pd.Series  # The smoothed price/volume series
    trend: pd.Series  # Underlying trend estimate (often equals smoothed for simple smoothers)
    velocity: pd.Series  # Rate of change of the trend (derivative)
    noise_estimate: pd.Series  # Estimated noise/uncertainty at each timestep
    confidence: pd.Series  # Inverse of noise — high confidence = low noise


# ============================================================================
# Plugin Base Classes
# ============================================================================


class IndicatorPlugin(ABC):
    """
    Abstract base class for technical indicator plugins.

    An IndicatorPlugin computes a technical indicator from OHLCV data and
    normalizes its output to [-1, +1] for use in the composite quant signal.

    Example: RSI, MACD, Bollinger Bands, SMA crossover.

    Subclasses must define:
        - name: str (unique identifier, snake_case)
        - category: str (one of: trend, momentum, volatility, volume, filter)
        - version: str (semver)
        - output_column: str (the DataFrame column passed to normalize())

    See .claude/skills/plugin-author/SKILL.md for implementation examples.
    """

    name: str
    category: Literal["trend", "momentum", "volatility", "volume", "filter"]
    version: str
    output_column: str  # Column produced by compute() that is passed to normalize()

    @abstractmethod
    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute the indicator and add output column(s) to the dataframe.

        Args:
            df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                indexed by date. Must NOT contain future data.
            params: Dictionary of parameter values (e.g., {"period": 14}).

        Returns:
            A copy of df with new column(s) added containing the indicator values.

        CRITICAL: This method must NEVER use future data. All rolling windows
        must use center=False (the default). When in doubt, ask: "Could this
        code know today's value before today's close happened?"
        """
        ...

    @abstractmethod
    def normalize(self, values: pd.Series) -> pd.Series:
        """
        Normalize the indicator output to the range [-1, +1].

        The convention is:
            +1.0 = strong bullish signal
            -1.0 = strong bearish signal
             0.0 = no signal / neutral

        Args:
            values: The raw indicator output series.

        Returns:
            Series with values clipped to [-1, +1].
        """
        ...

    @abstractmethod
    def get_tunable_params(self) -> Dict[str, ParamSpec]:
        """
        Return the parameter search space for Optuna tuning.

        Returns:
            Dict mapping parameter name to ParamSpec.
        """
        ...

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """
        Return default parameter values.

        Used when no tuned parameters are available for a stock.
        """
        ...


class SmoothingPlugin(ABC):
    """
    Abstract base class for smoothing plugins.

    A SmoothingPlugin pre-processes a time series (price, volume, or any
    derived series) to remove noise. Examples: Kalman filter, exponential
    smoothing, Holt-Winters.

    Smoothing plugins run BEFORE indicator computation and can be applied
    to specific columns via the `apply_to` config field.

    Phase 4 plans to add KalmanSmoother as the first SmoothingPlugin
    implementation. See architecture document Section 14.1.
    """

    name: str
    version: str

    @abstractmethod
    def smooth(self, series: pd.Series, params: Dict[str, Any]) -> SmoothResult:
        """
        Smooth a time series and return the smoothed result with auxiliary signals.

        Args:
            series: The input time series (e.g., closing prices).
            params: Dictionary of parameter values.

        Returns:
            SmoothResult with smoothed series, trend, velocity, noise, and confidence.

        CRITICAL: Must not use future data. For online smoothers like Kalman,
        each output timestep depends only on observations up to and including
        that timestep.
        """
        ...

    @abstractmethod
    def get_tunable_params(self) -> Dict[str, ParamSpec]:
        """Return the parameter search space for Optuna tuning."""
        ...


class DataEnricher(ABC):
    """
    Abstract base class for data enricher plugins.

    A DataEnricher adds external context features to the FeatureVector.
    Examples: FinBERT sentiment, options flow, macro indicators, cross-asset
    correlation.

    Enrichers can run in two modes:
        - Per-ticker: enrich(ticker, features) → returns dict of new features
        - Batch: batch_enrich(tickers) → returns dict of dicts (more efficient)

    See .claude/skills/finbert-integration/SKILL.md for the canonical example.
    """

    name: str
    data_type: Literal["sentiment", "macro", "alternative", "cross_asset"]
    version: str

    @abstractmethod
    def enrich(self, ticker: str, features: Any) -> Dict[str, float]:
        """
        Compute enrichment features for a single ticker.

        Args:
            ticker: Stock ticker symbol.
            features: Existing FeatureVector for context (may be unused).

        Returns:
            Dict mapping feature name to value. Feature names should be
            prefixed with the enricher's domain (e.g., "sentiment_score",
            "options_pcr", "macro_vix_level").
        """
        ...

    def batch_enrich(self, tickers: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Process multiple tickers efficiently.

        Default implementation calls enrich() in a loop. Override for
        enrichers that benefit from batching (e.g., FinBERT batch inference).
        """
        return {ticker: self.enrich(ticker, None) for ticker in tickers}


class SignalFilter(ABC):
    """
    Abstract base class for signal filter plugins.

    A SignalFilter post-processes TradeSignals at one of three pipeline stages:
        - "pre_quant": Before the quant engine generates signals
        - "post_quant": After quant signal, before ML meta-model
        - "post_meta": After ML meta-model approves the trade

    Examples:
        - LLMValidator (post_meta): Final veto gate using GPT-4o-mini
        - AttentionWeighter (post_quant): Dynamic indicator reweighting
        - RLPositionSizer (post_meta): Reinforcement learning for bet sizing
    """

    name: str
    stage: Literal["pre_quant", "post_quant", "post_meta"]
    version: str

    @abstractmethod
    def filter(self, signal: Any, context: Dict[str, Any]) -> Any:
        """
        Filter or modify a TradeSignal.

        Args:
            signal: The TradeSignal to evaluate.
            context: Auxiliary context (features, market data, portfolio state).

        Returns:
            The original signal, a modified signal, or None to suppress the trade.
        """
        ...

    def should_activate(self, context: Dict[str, Any]) -> bool:
        """
        Decide whether this filter should run for the current context.

        Default: always run. Override for conditional filters (e.g., LLM
        validator only runs if cost budget allows).
        """
        return True
