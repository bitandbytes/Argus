"""
Cluster parameter tuning via Walk-Forward Optimization — Task 2.6.

Runs WalkForwardOptimizer for each cluster in config/cluster_assignments.yaml,
aggregates OOS Sharpe across cluster members, and writes validated parameter
files to config/cluster_params/cluster_{id}.yaml.

Usage:
    # Tune all clusters (reads cluster_assignments.yaml)
    python scripts/tune_clusters.py --mode wfo

    # Tune a specific cluster
    python scripts/tune_clusters.py --mode wfo --cluster 0

    # Tune with specific tickers (useful for testing — creates a temporary cluster 0)
    python scripts/tune_clusters.py --mode wfo --tickers AAPL MSFT

    # Quick smoke-test with fewer Optuna trials
    python scripts/tune_clusters.py --mode wfo --tickers AAPL --n-trials 5 --no-mlflow

    # Override data window
    python scripts/tune_clusters.py --mode wfo --start 2018-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
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
_DEFAULT_PARAMS_PATH = "config/cluster_params/cluster_default.yaml"


# ─────────────────────────────────────────────────────────────────────────── #
# CLI
# ─────────────────────────────────────────────────────────────────────────── #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tune cluster parameters via Walk-Forward Optimization (Task 2.6).",
    )
    p.add_argument(
        "--mode",
        choices=["wfo"],
        default="wfo",
        help="Tuning mode (currently only 'wfo' is supported).",
    )
    p.add_argument(
        "--cluster",
        type=int,
        default=None,
        help="Tune only this cluster ID.  Omit to tune all clusters.",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Override cluster assignments: treat these tickers as cluster 0.",
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
        default="argus_wfo",
        help="MLflow experiment name (default: argus_wfo).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────── #
# Config loading
# ─────────────────────────────────────────────────────────────────────────── #

def _load_settings() -> Dict[str, Any]:
    with open(_SETTINGS_PATH) as f:
        return yaml.safe_load(f)


def _load_cluster_assignments() -> Dict[int, List[str]]:
    """Return {cluster_id: [tickers]} from cluster_assignments.yaml.

    Falls back to all watchlist tickers in cluster 0 if the assignments file
    doesn't exist yet (e.g. before StockClusterer has been run).
    """
    path = Path(_CLUSTER_ASSIGNMENTS_PATH)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
        raw_clusters = data.get("clusters", {})
        return {int(k): list(v) for k, v in raw_clusters.items()}

    logger.warning(
        "%s not found — falling back to watchlist tickers in cluster 0.",
        _CLUSTER_ASSIGNMENTS_PATH,
    )
    watchlist_path = Path("config/watchlist.yaml")
    if watchlist_path.exists():
        with open(watchlist_path) as f:
            wl = yaml.safe_load(f)
        tickers = [s["ticker"] for s in wl.get("stocks", [])]
        tickers += [e["ticker"] for e in wl.get("etfs", [])]
        return {0: tickers}
    return {0: ["AAPL"]}


def _load_default_regime_weights() -> Dict[str, Dict[str, float]]:
    """Load regime weights from cluster_default.yaml for carry-forward."""
    path = Path(_DEFAULT_PARAMS_PATH)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
        return data.get("indicators", {}).get("weights", {})
    return {}


# ─────────────────────────────────────────────────────────────────────────── #
# Objective function factory
# ─────────────────────────────────────────────────────────────────────────── #

def _make_objective(
    registry: PluginRegistry,
    ticker: str,
) -> Any:
    """Create a WFO objective function for a given ticker.

    The objective:
    1. Unpacks flat namespaced params into per-plugin dicts.
    2. Creates a fresh RegimeDetector and QuantEngine with those plugin params.
    3. Generates signals on the provided data slice using forward-only slicing.
    4. Computes a vectorized daily-return Sharpe from the generated signals.

    This avoids PyBroker overhead (which is too slow for 100 Optuna trials per
    window) while still exercising the full QuantEngine signal path.

    Args:
        registry: Loaded plugin registry.
        ticker: Stock ticker symbol.

    Returns:
        ``Callable[[params, df], float]`` for use as ``WalkForwardOptimizer``'s
        ``objective_fn``.
    """
    def objective(params: Dict[str, Any], df: pd.DataFrame) -> float:
        if len(df) < 50:
            return 0.0

        # Convert flat namespaced params → per-plugin dict.
        per_plugin = BayesianTuner.unpack_params_static(params)

        # Create a fresh regime detector for this slice.
        regime_detector = RegimeDetector()
        warmup_end = len(df) // 3
        try:
            regime_detector.fit(df.iloc[:warmup_end], ticker=ticker)
        except Exception:
            return 0.0

        # Create a QuantEngine with default weights (WFO tunes plugin params, not weights).
        engine = QuantEngine(registry, settings_path=_SETTINGS_PATH)

        # Generate signals forward-only (generate_series uses iloc[:i+1] internally).
        regime_series = regime_detector.detect_series(df)
        try:
            signals = engine.generate_series(
                df, regime_series, ticker, plugin_params=per_plugin
            )
        except Exception:
            return 0.0

        # Vectorized Sharpe: daily returns × sign(signal direction).
        close = df["close"]
        directions = pd.Series(
            [s.direction for s in signals], index=df.index, dtype=float
        )
        # 1-bar forward return aligned to signal direction.
        fwd_returns = close.pct_change().shift(-1).fillna(0.0)
        strategy_returns = fwd_returns * np.sign(directions)
        # Apply a minimal round-trip cost (0.1% per trade, half per side).
        trade_mask = directions.diff().abs() > 0.1
        strategy_returns -= trade_mask.astype(float) * 0.001
        return compute_sharpe(strategy_returns.values)

    return objective


# ─────────────────────────────────────────────────────────────────────────── #
# Output writing
# ─────────────────────────────────────────────────────────────────────────── #

def _write_cluster_params(
    cluster_id: int,
    best_params: Dict[str, Any],
    metadata: Dict[str, Any],
    default_weights: Dict[str, Dict[str, float]],
) -> Path:
    """Write validated cluster parameters to config/cluster_params/cluster_{id}.yaml.

    The output format preserves the existing ``indicators.weights`` section
    (regime weights are not tuned by WFO) and adds an ``indicators.params``
    section with the optimized plugin-level parameters.

    Args:
        cluster_id: Integer cluster identifier.
        best_params: Flat namespaced best params from WFO.
        metadata: Dict of metadata fields (pbo, sharpe, etc.).
        default_weights: Regime weight dict carried forward from cluster_default.yaml.

    Returns:
        Path of the written YAML file.
    """
    _CLUSTER_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _CLUSTER_PARAMS_DIR / f"cluster_{cluster_id}.yaml"

    # Convert flat params to per-plugin dict for human-readable output.
    per_plugin = BayesianTuner.unpack_params_static(best_params)

    doc: Dict[str, Any] = {
        "metadata": {
            "generated_date": datetime.now().strftime("%Y-%m-%d"),
            "cluster_id": cluster_id,
            **metadata,
        },
        "indicators": {
            "weights": default_weights or {},
            "params": per_plugin,
        },
    }

    # Atomic write: write to temp then rename.
    tmp = output_path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
    tmp.rename(output_path)

    logger.info("Wrote cluster params to %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────── #
# Per-cluster WFO runner
# ─────────────────────────────────────────────────────────────────────────── #

def _run_cluster_wfo(
    cluster_id: int,
    tickers: List[str],
    start: str,
    end: Optional[str],
    settings: Dict[str, Any],
    n_trials_override: Optional[int],
    mlflow_tracking: bool,
) -> Tuple[Optional[Path], Optional[WFOResult]]:
    """Run WFO for all tickers in a cluster and write the output YAML.

    Returns the output file path and the WFOResult for the representative
    ticker (ticker with highest aggregate OOS Sharpe).
    """
    tuning_cfg = settings.get("tuning", {})
    in_sample_days: int = int(tuning_cfg.get("in_sample_days", 252))
    oos_days: int = int(tuning_cfg.get("out_of_sample_days", 126))
    n_trials: int = n_trials_override or int(tuning_cfg.get("n_trials", 100))
    pbo_threshold: float = float(
        tuning_cfg.get("promotion", {}).get("pbo_threshold", 0.40)
    )

    cpcv_cfg = tuning_cfg.get("cpcv", {}) or {}
    cpcv_enabled: bool = bool(cpcv_cfg.get("enabled", True))
    cpcv_n_groups: int = int(cpcv_cfg.get("n_groups", 10))
    cpcv_n_test_groups: int = int(cpcv_cfg.get("n_test_groups", 2))
    cpcv_embargo_days: int = int(cpcv_cfg.get("embargo_days", 5))

    mdp = MarketDataProvider(cache_dir="data/raw")
    registry = PluginRegistry()
    registry.discover_plugins(_PLUGINS_PATH)
    plugins = registry.get_all_indicators()

    default_weights = _load_default_regime_weights()

    optimizer = WalkForwardOptimizer(
        in_sample_days=in_sample_days,
        out_of_sample_days=oos_days,
        n_trials=n_trials,
        pbo_top_k=min(10, max(2, n_trials // 10)),
        mlflow_tracking=False,  # outer MLflow run handles logging
        run_cpcv=cpcv_enabled,
        cpcv_n_groups=cpcv_n_groups,
        cpcv_n_test_groups=cpcv_n_test_groups,
        cpcv_embargo_days=cpcv_embargo_days,
    )

    all_results: List[Tuple[str, WFOResult]] = []

    for ticker in tickers:
        logger.info("Cluster %d | ticker %s: fetching data...", cluster_id, ticker)
        try:
            df = mdp.fetch_ohlcv(ticker, start=start, end=end or str(datetime.now().date()))
        except Exception as exc:
            logger.warning("Cluster %d | ticker %s: data fetch failed (%s) — skipping.", cluster_id, ticker, exc)
            continue

        min_rows = in_sample_days + 3 * oos_days
        if len(df) < min_rows:
            logger.warning(
                "Cluster %d | ticker %s: only %d rows (need %d) — skipping.",
                cluster_id, ticker, len(df), min_rows,
            )
            continue

        obj_fn = _make_objective(registry, ticker)

        try:
            result = optimizer.optimize(
                df=df,
                ticker=ticker,
                objective_fn=obj_fn,
                plugins=plugins,
                run_name_prefix=f"cluster_{cluster_id}",
            )
            all_results.append((ticker, result))
            logger.info(
                "Cluster %d | ticker %s: oos_sharpe=%.4f, pbo=%.4f, stable=%s",
                cluster_id, ticker, result.aggregate_oos_sharpe, result.pbo, result.is_stable,
            )
        except ValueError as exc:
            logger.warning("Cluster %d | ticker %s: WFO skipped (%s).", cluster_id, ticker, exc)
        except Exception as exc:
            logger.error("Cluster %d | ticker %s: WFO failed: %s", cluster_id, ticker, exc, exc_info=True)

    if not all_results:
        logger.error("Cluster %d: no tickers produced WFO results — skipping output.", cluster_id)
        return None, None

    # Pick the representative ticker: highest aggregate OOS Sharpe.
    best_ticker, best_result = max(all_results, key=lambda x: x[1].aggregate_oos_sharpe)
    agg_oos = float(np.mean([r.aggregate_oos_sharpe for _, r in all_results]))
    agg_pbo = float(np.mean([r.pbo for _, r in all_results]))

    # Aggregate CPCV PBO across tickers that successfully produced it.
    cpcv_values = [r.cpcv_pbo for _, r in all_results if r.cpcv_pbo is not None]
    agg_cpcv_pbo: Optional[float] = (
        float(np.mean(cpcv_values)) if cpcv_values else None
    )

    # Promotion gate: prefer CPCV PBO when available — it is the López de
    # Prado-recommended measure.  Fall back to the per-window PBO otherwise.
    gate_pbo = agg_cpcv_pbo if agg_cpcv_pbo is not None else agg_pbo
    gate_source = "cpcv_pbo" if agg_cpcv_pbo is not None else "pbo"
    pbo_pass = gate_pbo < pbo_threshold
    if not pbo_pass:
        logger.warning(
            "Cluster %d: %s=%.4f exceeds threshold %.2f — parameters may be overfit.",
            cluster_id, gate_source, gate_pbo, pbo_threshold,
        )

    metadata: Dict[str, Any] = {
        "n_tickers": len(all_results),
        "representative_ticker": best_ticker,
        "aggregate_oos_sharpe": round(agg_oos, 4),
        "pbo": round(agg_pbo, 4),
        "cpcv_pbo": round(agg_cpcv_pbo, 4) if agg_cpcv_pbo is not None else None,
        "pbo_pass": pbo_pass,
        "pbo_source": gate_source,
        "is_stable": best_result.is_stable,
        "n_windows": best_result.n_windows,
        "in_sample_days": in_sample_days,
        "out_of_sample_days": oos_days,
        "n_trials": n_trials,
    }

    output_path = _write_cluster_params(
        cluster_id=cluster_id,
        best_params=best_result.best_params,
        metadata=metadata,
        default_weights=default_weights,
    )

    if mlflow_tracking:
        try:
            metrics = {
                f"cluster_{cluster_id}/oos_sharpe": agg_oos,
                f"cluster_{cluster_id}/pbo": agg_pbo,
                f"cluster_{cluster_id}/is_stable": float(best_result.is_stable),
                f"cluster_{cluster_id}/n_tickers": float(len(all_results)),
            }
            if agg_cpcv_pbo is not None:
                metrics[f"cluster_{cluster_id}/cpcv_pbo"] = agg_cpcv_pbo
            mlflow.log_metrics(metrics)
            mlflow.log_param(f"cluster_{cluster_id}/best_ticker", best_ticker)
            mlflow.log_param(f"cluster_{cluster_id}/pbo_source", gate_source)
        except Exception as exc:
            logger.debug("MLflow cluster logging failed: %s", exc)

    return output_path, best_result


# ─────────────────────────────────────────────────────────────────────────── #
# Main
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    args = _parse_args()
    settings = _load_settings()

    if not args.no_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    if args.tickers:
        cluster_assignments = {0: args.tickers}
        logger.info("Using manually specified tickers as cluster 0: %s", args.tickers)
    else:
        cluster_assignments = _load_cluster_assignments()

    if args.cluster is not None:
        if args.cluster not in cluster_assignments:
            logger.error("Cluster %d not found in assignments. Available: %s", args.cluster, list(cluster_assignments.keys()))
            sys.exit(1)
        cluster_assignments = {args.cluster: cluster_assignments[args.cluster]}

    logger.info(
        "Starting WFO for %d cluster(s): %s",
        len(cluster_assignments),
        {k: len(v) for k, v in cluster_assignments.items()},
    )

    run_ctx = (
        mlflow.start_run(run_name=f"tune_clusters_{datetime.now():%Y%m%d_%H%M%S}")
        if not args.no_mlflow
        else _null_context()
    )

    with run_ctx:
        for cluster_id, tickers in sorted(cluster_assignments.items()):
            logger.info("=" * 60)
            logger.info("Cluster %d | %d tickers: %s", cluster_id, len(tickers), tickers)
            logger.info("=" * 60)

            output_path, result = _run_cluster_wfo(
                cluster_id=cluster_id,
                tickers=tickers,
                start=args.start,
                end=args.end,
                settings=settings,
                n_trials_override=args.n_trials,
                mlflow_tracking=not args.no_mlflow,
            )

            if output_path:
                logger.info("Cluster %d: output written to %s", cluster_id, output_path)
            else:
                logger.warning("Cluster %d: no output produced.", cluster_id)

    logger.info("WFO complete.")


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
