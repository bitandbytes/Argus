"""
Core data models for the trading pipeline.

These dataclasses define the contracts between pipeline layers. All inter-layer
communication uses these typed structures, never raw dicts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class RegimeType(str, Enum):
    """Market regime classification used by the regime detector."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"


@dataclass
class TradeSignal:
    """
    The output of the signal generation pipeline.

    Flows through the cascade: QuantEngine → MetaLabelModel → LLMValidator → RiskManager.
    Each layer may modify or annotate the signal.
    """

    ticker: str
    timestamp: datetime
    direction: float  # -1.0 (strong sell) to +1.0 (strong buy)
    confidence: float  # 0.0 to 1.0
    source_layer: str  # "quant" | "meta_model" | "llm_validator"
    regime: RegimeType
    features: Dict[str, float] = field(default_factory=dict)

    # Set by RiskManager (Phase 3)
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    bet_size: Optional[float] = None

    # Set by LLMValidator (Phase 3)
    llm_approved: Optional[bool] = None
    llm_reasoning: Optional[str] = None

    def __post_init__(self) -> None:
        if not -1.0 <= self.direction <= 1.0:
            raise ValueError(f"direction must be in [-1, +1], got {self.direction}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class FeatureVector:
    """
    Container for all features available at a given timestamp for a ticker.

    Built by the feature engineering pipeline from raw OHLCV data, sentiment
    scores, regime labels, and any DataEnricher outputs.
    """

    ticker: str
    timestamp: datetime
    technical: Dict[str, float] = field(default_factory=dict)
    sentiment: Dict[str, float] = field(default_factory=dict)
    derived: Dict[str, float] = field(default_factory=dict)
    regime: Optional[RegimeType] = None

    def to_dict(self) -> Dict[str, Any]:
        """Flatten all feature groups into a single dict for ML model input."""
        result = {}
        result.update({f"tech_{k}": v for k, v in self.technical.items()})
        result.update({f"sent_{k}": v for k, v in self.sentiment.items()})
        result.update({f"deriv_{k}": v for k, v in self.derived.items()})
        if self.regime:
            result["regime"] = self.regime.value
        return result


@dataclass
class MetaDecision:
    """Output of the ML meta-labeling model (Layer 3)."""

    should_trade: bool
    calibrated_proba: float
    uncertainty: float
    bet_size_fraction: float


@dataclass
class LLMResponse:
    """Output of the LLM Validator (Layer 4)."""

    decision: str  # "APPROVE" | "VETO"
    confidence: float
    reasoning: str
    risk_flags: list[str] = field(default_factory=list)
    impact_horizon: str = ""
