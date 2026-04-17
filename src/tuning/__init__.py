"""Tuning package — hyperparameter optimization and validation utilities."""

from src.tuning.bayesian_tuner import BayesianTuner, OptimizationResult, compute_sharpe
from src.tuning.promotion_gate import (
    DECISION_DEMOTE,
    DECISION_KEEP_CLUSTER,
    DECISION_PROMOTE,
    PromotionDecision,
    PromotionGate,
)
from src.tuning.purged_cv import CombinatorialPurgedCV, PurgedCrossValidator, purged_walk_forward_splits
from src.tuning.stock_clusterer import StockClusterer
from src.tuning.walk_forward import WalkForwardOptimizer, WFOResult, WFOWindowResult

__all__ = [
    "BayesianTuner",
    "OptimizationResult",
    "compute_sharpe",
    "CombinatorialPurgedCV",
    "PurgedCrossValidator",
    "purged_walk_forward_splits",
    "PromotionGate",
    "PromotionDecision",
    "DECISION_PROMOTE",
    "DECISION_KEEP_CLUSTER",
    "DECISION_DEMOTE",
    "StockClusterer",
    "WalkForwardOptimizer",
    "WFOResult",
    "WFOWindowResult",
]
