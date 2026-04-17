"""
Per-stock parameter tuning + promotion gate — Task 2.7.

Runs ``WalkForwardOptimizer`` for each ticker in
``config/cluster_assignments.yaml``, compares the result against the
stock's cluster baseline (``config/cluster_params/cluster_{id}.yaml``) via
:class:`PromotionGate`, and promotes qualifying stocks by writing
``config/stock_overrides/{ticker}.yaml``.  Every decision — promote, keep,
demote — is appended to ``config/promotion_log.yaml``.

Usage:
    # Evaluate every ticker in cluster_assignments.yaml
    python scripts/tune_individual.py

    # Limit to a single cluster
    python scripts/tune_individual.py --cluster 0

    # Tune specific tickers (bypasses cluster_assignments)
    python scripts/tune_individual.py --tickers AAPL MSFT --cluster 0

    # Dry-run: evaluate and print decisions but do NOT write overrides or log
    python scripts/tune_individual.py --tickers AAPL --n-trials 5 --dry-run

    # Quick smoke-test
    python scripts/tune_individual.py --tickers AAPL --n-trials 5 --no-mlflow
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure project root is on the path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

from src.data.market_data import MarketDataProvider
from src.plugins.registry import PluginRegistry
from src.signals.quant_engine import QuantEngine
from src.signals.regime_detector import RegimeDetector
from src.tuning.bayesian_tuner import BayesianTuner, compute_sharpe
from src.tuning.promotion_gate import (
    DECISION_KEEP_CLUSTER,
    DECISION_PROMOTE,
    PromotionDecision,
    PromotionGate,
)
from src.tuning.walk_forward import WalkForwardOptimizer, WFOResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SETTINGS_PATH = "config/settings.yaml"
_PLUGINS_PATH = "config/plugins.yaml"
_CLUSTER_ASSIGNMENTS_PATH = "config/cluster_assignments.yaml"
_CLUSTER_PARAMS_DIR = Path("config/cluster_params")


# ─────────────────────────────────────────────────────────────────────────── #
# CLI
# ─────────────────────────────────────────────────────────────────────────── #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-stock WFO + PromotionGate evaluation (Task 2.7).",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Evaluate only these tickers (must still belong to a cluster "
             "unless --cluster is also given).",
    )
    p.add_argument(
        "--cluster",
        type=int,
        default=None,
        help="Limit evaluation to this cluster ID (and use it for --tickers).",
    )
    p.add_argument(
        "--start",
        default="2015-01-01",
        help="Start of data window for WFO (default: 2015-01-01).",
    )
    p.add_argument(
        "--end",
        default=None,
        help="End of data window (default: today).",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Optuna trials per IS window.  Overrides settings.yaml.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and print decisions but do NOT write override files "
             "or append to promotion_log.yaml.",
    )
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging.",
    )
    p.add_argument(
        "--mlflow-tracking-uri",
        default="data/mlflow.db",
        help="SQLite DB for MLflow (default: data/mlflow.db).",
    )
    p.add_argument(
        "--mlflow-experiment",
        default="argus_promotion",
        help="MLflow experiment name (default: argus_promotion).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────── #
# Config loading
# ─────────────────────────────────────────────────────────────────────────── #

def _load_settings() -> Dict[str, Any]:
    with open(_SETTINGS_PATH) as f:
        return yaml.safe_load(f)


def _load_cluster_assignments() -> Dict[int, List[str]]:
    """Return ``{cluster_id: [tickers]}`` from ``cluster_assignments.yaml``."""
    path = Path(_CLUSTER_ASSIGNMENTS_PATH)
    if not path.exists():
        logger.error(
            "Cluster assignments file %s not found — run scripts/tune_clusters.py first.",
            _CLUSTER_ASSIGNMENTS_PATH,
        )
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    raw_clusters = data.get("clusters", {}) or {}
    return {int(k): list(v) for k, v in raw_clusters.items()}


def _lookup_cluster_id(ticker: str, assignments: Dict[int, List[str]]) -> Optional[int]:
    for cid, members in assignments.items():
        if ticker in members:
            return cid
    return None


# ─────────────────────────────────────────────────────────────────────────── #
# Cluster baseline loading
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class _ClusterBaseline:
    """Hydrated cluster baseline for a stock's promotion evaluation.

    Only the fields consumed by :meth:`PromotionGate.evaluate` are populated
    on the :class:`WFOResult` stub — everything else defaults to zero/empty.
    ``weights`` is carried separately for :meth:`PromotionGate.promote`.
    """

    cluster_id: int
    wfo_stub: WFOResult
    weights: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _load_cluster_baseline(cluster_id: int) -> Optional[_ClusterBaseline]:
    """Load a cluster's tuned params YAML and materialise a baseline stub.

    Reads ``metadata.aggregate_oos_sharpe`` (written by ``tune_clusters.py``)
    and ``indicators.weights``.  When the cluster YAML is missing, returns
    ``None`` — the caller short-circuits to ``KEEP_CLUSTER`` without WFO.
    """
    path = _CLUSTER_PARAMS_DIR / f"cluster_{cluster_id}.yaml"
    if not path.exists():
        logger.warning(
            "Cluster %d params file %s not found — run tune_clusters.py first.",
            cluster_id, path,
        )
        return None

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    metadata = data.get("metadata", {}) or {}
    agg_sharpe = float(metadata.get("aggregate_oos_sharpe", 0.0))

    stub = WFOResult(
        ticker=f"cluster_{cluster_id}",
        windows=[],
        best_params={},
        aggregate_oos_sharpe=agg_sharpe,
        pbo=float(metadata.get("pbo", 0.0)),
        is_stable=bool(metadata.get("is_stable", False)),
        n_windows=int(metadata.get("n_windows", 0)),
    )
    weights = (data.get("indicators", {}) or {}).get("weights", {}) or {}
    return _ClusterBaseline(cluster_id=cluster_id, wfo_stub=stub, weights=weights)


# ─────────────────────────────────────────────────────────────────────────── #
# Objective function factory (matches scripts/tune_clusters.py)
# ─────────────────────────────────────────────────────────────────────────── #

def _make_objective(registry: PluginRegistry, ticker: str) -> Any:
    """Create a WFO objective function (identical shape to tune_clusters)."""
    def objective(params: Dict[str, Any], df: pd.DataFrame) -> float:
        if len(df) < 50:
            return 0.0

        per_plugin = BayesianTuner.unpack_params_static(params)

        regime_detector = RegimeDetector()
        warmup_end = len(df) // 3
        try:
            regime_detector.fit(df.iloc[:warmup_end], ticker=ticker)
        except Exception:
            return 0.0

        engine = QuantEngine(registry, settings_path=_SETTINGS_PATH)

        regime_series = regime_detector.detect_series(df)
        try:
            signals = engine.generate_series(
                df, regime_series, ticker, plugin_params=per_plugin
            )
        except Exception:
            return 0.0

        close = df["close"]
        directions = pd.Series(
            [s.direction for s in signals], index=df.index, dtype=float
        )
        fwd_returns = close.pct_change().shift(-1).fillna(0.0)
        strategy_returns = fwd_returns * np.sign(directions)
        trade_mask = directions.diff().abs() > 0.1
        strategy_returns -= trade_mask.astype(float) * 0.001
        return compute_sharpe(strategy_returns.values)

    return objective


# ─────────────────────────────────────────────────────────────────────────── #
# Per-ticker evaluation
# ─────────────────────────────────────────────────────────────────────────── #

def _evaluate_ticker(
    ticker: str,
    cluster_id: int,
    baseline: _ClusterBaseline,
    gate: PromotionGate,
    optimizer: WalkForwardOptimizer,
    registry: PluginRegistry,
    mdp: MarketDataProvider,
    start: str,
    end: Optional[str],
    in_sample_days: int,
    oos_days: int,
    dry_run: bool,
    mlflow_tracking: bool,
) -> Optional[PromotionDecision]:
    """Run WFO → PromotionGate.evaluate for one ticker. Returns the decision."""
    logger.info("Ticker %s | cluster %d: fetching data...", ticker, cluster_id)
    try:
        df = mdp.fetch_ohlcv(ticker, start=start, end=end or str(datetime.now().date()))
    except Exception as exc:
        logger.warning("Ticker %s: data fetch failed (%s) — skipping.", ticker, exc)
        return None

    # Cheap short-circuit: if history is below the promotion history floor,
    # skip WFO entirely and record a KEEP_CLUSTER decision.
    if len(df) < gate.min_history_bars:
        logger.info(
            "Ticker %s: only %d bars < %d required — short-circuit KEEP_CLUSTER.",
            ticker, len(df), gate.min_history_bars,
        )
        decision = PromotionDecision(
            ticker=ticker,
            decision=DECISION_KEEP_CLUSTER,
            reasons=[
                f"insufficient history: {len(df)} bars < "
                f"{gate.min_history_bars} required (no WFO performed)"
            ],
            metrics={
                "n_bars_available": int(len(df)),
                "cluster_oos_sharpe": float(baseline.wfo_stub.aggregate_oos_sharpe),
            },
        )
        if not dry_run:
            gate.log_decision(decision)
        return decision

    # Need enough rows for WFO to produce at least one window.
    min_rows = in_sample_days + 3 * oos_days
    if len(df) < min_rows:
        logger.warning(
            "Ticker %s: only %d rows (WFO needs %d) — skipping.",
            ticker, len(df), min_rows,
        )
        return None

    obj_fn = _make_objective(registry, ticker)
    plugins = registry.get_all_indicators()

    try:
        individual_result = optimizer.optimize(
            df=df,
            ticker=ticker,
            objective_fn=obj_fn,
            plugins=plugins,
            run_name_prefix=f"individual_{ticker}",
        )
    except ValueError as exc:
        logger.warning("Ticker %s: WFO skipped (%s).", ticker, exc)
        return None
    except Exception as exc:
        logger.error("Ticker %s: WFO failed: %s", ticker, exc, exc_info=True)
        return None

    decision = gate.evaluate(
        ticker=ticker,
        cluster_wfo=baseline.wfo_stub,
        individual_wfo=individual_result,
        n_bars_available=len(df),
    )

    logger.info(
        "Ticker %s → %s  (cluster_s=%.3f, indiv_s=%.3f, pbo=%s)",
        ticker,
        decision.decision,
        baseline.wfo_stub.aggregate_oos_sharpe,
        individual_result.aggregate_oos_sharpe,
        decision.metrics.get("pbo_used", "n/a"),
    )

    if decision.decision == DECISION_PROMOTE and not dry_run:
        gate.promote(
            ticker=ticker,
            individual_wfo=individual_result,
            cluster_id=cluster_id,
            cluster_weights=baseline.weights,
            decision=decision,
        )

    if not dry_run:
        gate.log_decision(decision)

    if mlflow_tracking:
        _log_decision_to_mlflow(decision)

    return decision


def _log_decision_to_mlflow(decision: PromotionDecision) -> None:
    try:
        with mlflow.start_run(run_name=f"promotion_{decision.ticker}", nested=True):
            mlflow.log_param("ticker", decision.ticker)
            mlflow.log_param("decision", decision.decision)
            numeric = {
                f"promotion/{k}": float(v)
                for k, v in decision.metrics.items()
                if isinstance(v, (int, float)) and v is not None
            }
            if numeric:
                mlflow.log_metrics(numeric)
    except Exception as exc:
        logger.debug("MLflow promotion logging failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────── #
# Main
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    args = _parse_args()
    settings = _load_settings()

    tuning_cfg = settings.get("tuning", {}) or {}
    in_sample_days = int(tuning_cfg.get("in_sample_days", 252))
    oos_days = int(tuning_cfg.get("out_of_sample_days", 126))
    n_trials = args.n_trials or int(tuning_cfg.get("n_trials", 100))

    cpcv_cfg = tuning_cfg.get("cpcv", {}) or {}
    cpcv_enabled = bool(cpcv_cfg.get("enabled", True))
    cpcv_n_groups = int(cpcv_cfg.get("n_groups", 10))
    cpcv_n_test_groups = int(cpcv_cfg.get("n_test_groups", 2))
    cpcv_embargo_days = int(cpcv_cfg.get("embargo_days", 5))

    assignments = _load_cluster_assignments()
    if not assignments:
        logger.error("No cluster assignments available — aborting.")
        sys.exit(1)

    # Resolve the (ticker → cluster_id) list we will evaluate.
    evaluation_list: List[Tuple[str, int]] = []
    if args.tickers and args.cluster is not None:
        for t in args.tickers:
            evaluation_list.append((t, args.cluster))
    elif args.tickers:
        for t in args.tickers:
            cid = _lookup_cluster_id(t, assignments)
            if cid is None:
                logger.warning("Ticker %s not in any cluster — skipping.", t)
                continue
            evaluation_list.append((t, cid))
    elif args.cluster is not None:
        if args.cluster not in assignments:
            logger.error(
                "Cluster %d not found. Available: %s",
                args.cluster, sorted(assignments.keys()),
            )
            sys.exit(1)
        evaluation_list.extend((t, args.cluster) for t in assignments[args.cluster])
    else:
        for cid, members in sorted(assignments.items()):
            evaluation_list.extend((t, cid) for t in members)

    if not evaluation_list:
        logger.error("No tickers to evaluate.")
        sys.exit(1)

    logger.info(
        "Evaluating %d ticker(s) across %d cluster(s); dry_run=%s.",
        len(evaluation_list),
        len({c for _, c in evaluation_list}),
        args.dry_run,
    )

    # Cache cluster baselines so we don't re-load the YAML per ticker.
    baselines: Dict[int, Optional[_ClusterBaseline]] = {}

    # One registry, one data provider, one optimizer — reused across tickers.
    registry = PluginRegistry()
    registry.discover_plugins(_PLUGINS_PATH)
    mdp = MarketDataProvider(cache_dir="data/raw")

    optimizer = WalkForwardOptimizer(
        in_sample_days=in_sample_days,
        out_of_sample_days=oos_days,
        n_trials=n_trials,
        pbo_top_k=min(10, max(2, n_trials // 10)),
        mlflow_tracking=False,  # outer run handles logging
        run_cpcv=cpcv_enabled,
        cpcv_n_groups=cpcv_n_groups,
        cpcv_n_test_groups=cpcv_n_test_groups,
        cpcv_embargo_days=cpcv_embargo_days,
    )

    gate = PromotionGate()

    if not args.no_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    run_ctx = (
        mlflow.start_run(run_name=f"tune_individual_{datetime.now():%Y%m%d_%H%M%S}")
        if not args.no_mlflow
        else _null_context()
    )

    decisions: List[Tuple[str, int, Optional[PromotionDecision]]] = []
    with run_ctx:
        for ticker, cluster_id in evaluation_list:
            if cluster_id not in baselines:
                baselines[cluster_id] = _load_cluster_baseline(cluster_id)
            baseline = baselines[cluster_id]
            if baseline is None:
                logger.warning(
                    "Ticker %s: no cluster_%d baseline — skipping.",
                    ticker, cluster_id,
                )
                decisions.append((ticker, cluster_id, None))
                continue

            decision = _evaluate_ticker(
                ticker=ticker,
                cluster_id=cluster_id,
                baseline=baseline,
                gate=gate,
                optimizer=optimizer,
                registry=registry,
                mdp=mdp,
                start=args.start,
                end=args.end,
                in_sample_days=in_sample_days,
                oos_days=oos_days,
                dry_run=args.dry_run,
                mlflow_tracking=not args.no_mlflow,
            )
            decisions.append((ticker, cluster_id, decision))

    _print_summary(decisions)

    # Exit 1 only when every ticker failed to produce a decision.
    if all(d is None for _, _, d in decisions):
        logger.error("No decisions produced — see warnings above.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────── #
# Summary printing
# ─────────────────────────────────────────────────────────────────────────── #

def _print_summary(
    decisions: List[Tuple[str, int, Optional[PromotionDecision]]],
) -> None:
    header = (
        f"{'ticker':<8}{'cluster':<9}{'decision':<16}"
        f"{'cluster_s':<12}{'indiv_s':<10}{'pbo':<10}reason"
    )
    print("\n" + "=" * 100)
    print(header)
    print("-" * 100)
    for ticker, cid, decision in decisions:
        if decision is None:
            print(f"{ticker:<8}{cid:<9}{'SKIPPED':<16}{'':<12}{'':<10}{'':<10}"
                  "data/WFO error")
            continue
        m = decision.metrics
        cs = m.get("cluster_oos_sharpe")
        is_s = m.get("individual_oos_sharpe")
        pbo = m.get("pbo_used")
        cs_str = f"{cs:.3f}" if isinstance(cs, (int, float)) else ""
        is_str = f"{is_s:.3f}" if isinstance(is_s, (int, float)) else ""
        pbo_str = f"{pbo:.3f}" if isinstance(pbo, (int, float)) else ""
        reason = decision.reasons[-1] if decision.reasons else ""
        print(
            f"{ticker:<8}{cid:<9}{decision.decision:<16}"
            f"{cs_str:<12}{is_str:<10}{pbo_str:<10}{reason}"
        )
    print("=" * 100 + "\n")


# ─────────────────────────────────────────────────────────────────────────── #
# Helper: null context manager for when MLflow is disabled
# ─────────────────────────────────────────────────────────────────────────── #

class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    main()
