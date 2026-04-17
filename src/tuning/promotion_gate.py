"""Hybrid cluster/individual promotion gate (Task 2.7).

Implements the ``PromotionGate`` that decides whether a stock's individually
tuned Walk-Forward Optimization result should be promoted to
``config/stock_overrides/{ticker}.yaml`` (taking precedence over the
cluster default) or kept on the cluster baseline.

Criteria (from ``docs/architecture.md`` §5.4 and ``config/settings.yaml``
``tuning.promotion``):

  1. History ≥ ``min_history_bars`` (default 2500 daily bars ≈ 10 years)
  2. Individual OOS Sharpe > Cluster OOS Sharpe × ``sharpe_improvement_threshold``
     (default 1.20)
  3. Parameter stability across adjacent WFO windows (``is_stable`` flag — the
     drift threshold is applied inside :class:`BayesianTuner.stability_check`)
  4. Probability of Backtest Overfitting < ``pbo_threshold`` (default 0.40).
     Prefers the CPCV PBO when available (``WFOResult.cpcv_pbo``), falls back
     to the per-window PBO (``WFOResult.pbo``).

Demotion: :meth:`PromotionGate.check_demotion` returns ``True`` when a
promoted stock's rolling live/paper Sharpe drops > ``demotion_sharpe_gap``
(default 0.30) below the cluster baseline.  Phase 3 wires this into the
daily pipeline; Phase 2 exposes the helper so it can be unit-tested now.

Typical usage::

    gate = PromotionGate()
    decision = gate.evaluate(
        ticker="AAPL",
        cluster_wfo=cluster_result,
        individual_wfo=individual_result,
        n_bars_available=len(df),
    )
    if decision.decision == "PROMOTE":
        gate.promote(
            ticker="AAPL",
            individual_wfo=individual_result,
            cluster_id=0,
            cluster_weights=cluster_weights,
            decision=decision,
        )
    gate.log_decision(decision)
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.tuning.bayesian_tuner import BayesianTuner
from src.tuning.walk_forward import WFOResult

logger = logging.getLogger(__name__)

_DEFAULT_SETTINGS_PATH = "config/settings.yaml"
_DEFAULT_OVERRIDES_DIR = "config/stock_overrides"
_DEFAULT_LOG_PATH = "config/promotion_log.yaml"

# Decision string literals — kept as plain strings (not Enum) for simple YAML
# serialisation and grep-friendly diffs in promotion_log.yaml.
DECISION_PROMOTE = "PROMOTE"
DECISION_KEEP_CLUSTER = "KEEP_CLUSTER"
DECISION_DEMOTE = "DEMOTE"


# ---------------------------------------------------------------------------
# PromotionDecision dataclass
# ---------------------------------------------------------------------------

@dataclass
class PromotionDecision:
    """Outcome of a :meth:`PromotionGate.evaluate` call.

    Attributes:
        ticker: Stock symbol.
        decision: One of ``"PROMOTE"``, ``"KEEP_CLUSTER"``, ``"DEMOTE"``.
        reasons: Human-readable list of criteria outcomes (both passes and
            failures) for the audit trail.
        metrics: Dict of numeric evidence — cluster/individual Sharpes,
            ratio, PBO, stability flag, history length, etc.
        timestamp: UTC ISO-8601 timestamp when the decision was made.
    """

    ticker: str
    decision: str
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


# ---------------------------------------------------------------------------
# PromotionGate
# ---------------------------------------------------------------------------

class PromotionGate:
    """Hybrid cluster-vs-individual promotion decision maker.

    Loads thresholds from ``config/settings.yaml::tuning.promotion`` at
    construction time.  Exposes:

      - :meth:`evaluate` — cluster + individual WFO → :class:`PromotionDecision`.
      - :meth:`promote` — write ``config/stock_overrides/{ticker}.yaml``.
      - :meth:`demote` — delete an existing override file.
      - :meth:`check_demotion` — live-performance-based demotion predicate.
      - :meth:`log_decision` — append a :class:`PromotionDecision` to
        ``config/promotion_log.yaml`` (atomic write).
      - :meth:`is_promoted` / :meth:`list_promoted` — introspection of the
        current override set.
      - :meth:`resolve_params_path` — fallback-chain resolver used by the
        backtester and daily pipeline to pick the right YAML for a ticker.

    Args:
        settings_path: Path to ``settings.yaml`` (default
            ``config/settings.yaml``).  Override for testing.
        overrides_dir: Directory that holds per-stock override YAMLs.
            Default ``config/stock_overrides``.
        log_path: Path to the append-only YAML audit trail.
            Default ``config/promotion_log.yaml``.
    """

    def __init__(
        self,
        settings_path: str = _DEFAULT_SETTINGS_PATH,
        overrides_dir: str = _DEFAULT_OVERRIDES_DIR,
        log_path: str = _DEFAULT_LOG_PATH,
    ) -> None:
        self.overrides_dir = Path(overrides_dir)
        self.log_path = Path(log_path)

        cfg = self._load_settings(settings_path)
        promo = cfg.get("tuning", {}).get("promotion", {}) or {}
        self.min_history_bars: int = int(promo.get("min_history_bars", 2500))
        self.sharpe_improvement_threshold: float = float(
            promo.get("sharpe_improvement_threshold", 1.20)
        )
        self.param_drift_threshold: float = float(
            promo.get("param_drift_threshold", 0.20)
        )
        self.pbo_threshold: float = float(promo.get("pbo_threshold", 0.40))
        self.min_oos_trades: int = int(promo.get("min_oos_trades", 30))
        self.demotion_lookback_days: int = int(
            promo.get("demotion_lookback_days", 60)
        )
        self.demotion_sharpe_gap: float = float(
            promo.get("demotion_sharpe_gap", 0.30)
        )

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        ticker: str,
        cluster_wfo: WFOResult,
        individual_wfo: WFOResult,
        n_bars_available: int,
    ) -> PromotionDecision:
        """Apply all promotion criteria and return a decision.

        The criteria are checked in the order from the architecture doc
        flowchart (§5.4).  A single failure short-circuits to
        ``KEEP_CLUSTER`` — all other criteria are still reported in
        ``reasons`` for the audit log.

        Args:
            ticker: Stock symbol.
            cluster_wfo: The cluster baseline :class:`WFOResult`.
            individual_wfo: The stock's individually tuned WFO result.
            n_bars_available: Number of daily bars in the stock's history.
                Usually ``len(df)`` from the ticker's full OHLCV frame.

        Returns:
            :class:`PromotionDecision` with ``decision`` set and ``metrics``
            populated with every numeric input used.
        """
        reasons: List[str] = []
        metrics: Dict[str, Any] = {
            "n_bars_available": int(n_bars_available),
            "cluster_oos_sharpe": float(cluster_wfo.aggregate_oos_sharpe),
            "individual_oos_sharpe": float(individual_wfo.aggregate_oos_sharpe),
            "individual_pbo": float(individual_wfo.pbo),
            "individual_cpcv_pbo": individual_wfo.cpcv_pbo,
            "individual_is_stable": bool(individual_wfo.is_stable),
        }

        # 1. History length
        if n_bars_available < self.min_history_bars:
            reasons.append(
                f"insufficient history: {n_bars_available} bars < "
                f"{self.min_history_bars} required"
            )
            return self._keep_cluster(ticker, reasons, metrics)
        reasons.append(
            f"history OK: {n_bars_available} bars >= {self.min_history_bars}"
        )

        # 2. Sharpe improvement
        # Guard against zero or negative cluster Sharpe — require an absolute
        # individual lead of at least the threshold×|cluster| or 0.10 Sharpe
        # when the cluster baseline is ≤ 0.
        cluster_s = float(cluster_wfo.aggregate_oos_sharpe)
        indiv_s = float(individual_wfo.aggregate_oos_sharpe)
        ratio: Optional[float] = None
        if cluster_s > 0:
            ratio = indiv_s / cluster_s
            metrics["sharpe_ratio"] = float(ratio)
            if ratio <= self.sharpe_improvement_threshold:
                reasons.append(
                    f"Sharpe lift too small: {indiv_s:.3f} / {cluster_s:.3f} = "
                    f"{ratio:.3f} <= {self.sharpe_improvement_threshold}"
                )
                return self._keep_cluster(ticker, reasons, metrics)
            reasons.append(
                f"Sharpe lift OK: ratio={ratio:.3f} > "
                f"{self.sharpe_improvement_threshold}"
            )
        else:
            min_abs_lift = 0.10
            metrics["sharpe_abs_lift"] = float(indiv_s - cluster_s)
            if indiv_s - cluster_s < min_abs_lift:
                reasons.append(
                    f"cluster Sharpe <= 0 ({cluster_s:.3f}); individual lift "
                    f"{indiv_s - cluster_s:.3f} < {min_abs_lift}"
                )
                return self._keep_cluster(ticker, reasons, metrics)
            reasons.append(
                f"cluster Sharpe <= 0; individual abs lift "
                f"{indiv_s - cluster_s:.3f} >= {min_abs_lift}"
            )

        # 3. Parameter stability
        if not individual_wfo.is_stable:
            reasons.append(
                f"params unstable across WFO windows (drift > "
                f"{self.param_drift_threshold * 100:.0f}%)"
            )
            return self._keep_cluster(ticker, reasons, metrics)
        reasons.append("params stable across WFO windows")

        # 4. PBO — prefer CPCV when available, fall back to per-window PBO
        pbo, pbo_source = self._select_pbo(individual_wfo)
        metrics["pbo_used"] = float(pbo)
        metrics["pbo_source"] = pbo_source
        if pbo >= self.pbo_threshold:
            reasons.append(
                f"{pbo_source}={pbo:.3f} >= {self.pbo_threshold} "
                f"(overfitting risk too high)"
            )
            return self._keep_cluster(ticker, reasons, metrics)
        reasons.append(f"{pbo_source}={pbo:.3f} < {self.pbo_threshold}")

        # All criteria satisfied.
        return PromotionDecision(
            ticker=ticker,
            decision=DECISION_PROMOTE,
            reasons=reasons,
            metrics=metrics,
        )

    def check_demotion(
        self,
        ticker: str,
        rolling_sharpe_60d: float,
        cluster_baseline_sharpe: float,
    ) -> bool:
        """Return ``True`` when live performance warrants reverting to cluster.

        Implements the demotion rule from architecture §5.4:
        *rolling 60-day live Sharpe drops > ``demotion_sharpe_gap`` below
        the cluster baseline.*

        This helper does **not** itself delete the override file — callers
        (typically ``scripts/check_demotions.py``) should record two
        consecutive months of breach before calling :meth:`demote`.

        Args:
            ticker: Stock symbol (used only for logging).
            rolling_sharpe_60d: Recent rolling Sharpe from live or paper
                fills.  NaN/inf is treated as "no data" → returns ``False``.
            cluster_baseline_sharpe: The cluster's OOS Sharpe from the most
                recent WFO run.

        Returns:
            ``True`` if the Sharpe gap exceeds ``demotion_sharpe_gap``.
        """
        try:
            gap = float(cluster_baseline_sharpe) - float(rolling_sharpe_60d)
        except (TypeError, ValueError):
            return False
        if gap != gap or gap in (float("inf"), float("-inf")):  # NaN / inf guard
            return False
        breach = gap > self.demotion_sharpe_gap
        logger.info(
            "Demotion check for %s: cluster=%.3f, rolling=%.3f, gap=%.3f, "
            "threshold=%.3f → %s",
            ticker, cluster_baseline_sharpe, rolling_sharpe_60d, gap,
            self.demotion_sharpe_gap, "DEMOTE" if breach else "KEEP",
        )
        return breach

    # ------------------------------------------------------------------ #
    # Override file I/O                                                    #
    # ------------------------------------------------------------------ #

    def promote(
        self,
        ticker: str,
        individual_wfo: WFOResult,
        cluster_id: int,
        cluster_weights: Dict[str, Dict[str, float]],
        decision: Optional[PromotionDecision] = None,
    ) -> Path:
        """Write an individual parameter override file for ``ticker``.

        The override file inherits the cluster's regime weights (those are not
        tuned by WFO) and supplies the stock-specific ``indicators.params``
        block from :attr:`WFOResult.best_params`.  Atomic write via temp file.

        Args:
            ticker: Stock symbol — used as the filename (``{ticker}.yaml``).
            individual_wfo: The stock's tuned WFO result.
            cluster_id: The cluster the stock was assigned to (recorded in
                metadata so demotion can find the right cluster YAML).
            cluster_weights: Regime weight dict carried forward from the
                cluster's ``indicators.weights`` section.
            decision: Optional :class:`PromotionDecision` — if supplied, the
                metrics and reasons are embedded in the override's metadata
                for post-hoc auditing.

        Returns:
            The path of the written file.
        """
        self.overrides_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.overrides_dir / f"{ticker}.yaml"

        per_plugin = BayesianTuner.unpack_params_static(individual_wfo.best_params)

        metadata: Dict[str, Any] = {
            "generated_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "ticker": ticker,
            "cluster_id": int(cluster_id),
            "individual_oos_sharpe": round(
                float(individual_wfo.aggregate_oos_sharpe), 4
            ),
            "pbo": round(float(individual_wfo.pbo), 4),
            "cpcv_pbo": (
                round(float(individual_wfo.cpcv_pbo), 4)
                if individual_wfo.cpcv_pbo is not None
                else None
            ),
            "is_stable": bool(individual_wfo.is_stable),
            "n_windows": int(individual_wfo.n_windows),
        }
        if decision is not None:
            metadata["decision_reasons"] = list(decision.reasons)
            for key in ("cluster_oos_sharpe", "sharpe_ratio", "sharpe_abs_lift"):
                if key in decision.metrics:
                    metadata[key] = (
                        round(float(decision.metrics[key]), 4)
                        if isinstance(decision.metrics[key], (int, float))
                        else decision.metrics[key]
                    )

        doc: Dict[str, Any] = {
            "metadata": metadata,
            "indicators": {
                "weights": cluster_weights or {},
                "params": per_plugin,
            },
        }

        self._atomic_yaml_write(output_path, doc)
        logger.info("Promoted %s → %s", ticker, output_path)
        return output_path

    def demote(self, ticker: str) -> bool:
        """Remove an existing override file (revert to cluster params).

        Args:
            ticker: Stock symbol whose override should be removed.

        Returns:
            ``True`` if a file was deleted; ``False`` if no override existed.
        """
        path = self.overrides_dir / f"{ticker}.yaml"
        if not path.exists():
            logger.debug("No override to demote for %s (file %s missing).", ticker, path)
            return False
        path.unlink()
        logger.info("Demoted %s (removed %s).", ticker, path)
        return True

    def is_promoted(self, ticker: str) -> bool:
        """Return ``True`` iff a stock override file currently exists."""
        return (self.overrides_dir / f"{ticker}.yaml").exists()

    def list_promoted(self) -> List[str]:
        """Return an alphabetically sorted list of currently promoted tickers."""
        if not self.overrides_dir.exists():
            return []
        return sorted(p.stem for p in self.overrides_dir.glob("*.yaml"))

    def resolve_params_path(
        self,
        ticker: str,
        cluster_id: Optional[int] = None,
        cluster_params_dir: str = "config/cluster_params",
    ) -> Optional[str]:
        """Resolve a ticker's params-YAML path via the fallback chain.

        Priority (matches architecture §5.5):
          1. ``{overrides_dir}/{ticker}.yaml``
          2. ``{cluster_params_dir}/cluster_{cluster_id}.yaml`` (when
             ``cluster_id`` is given)
          3. ``{cluster_params_dir}/cluster_default.yaml``

        Args:
            ticker: Stock symbol.
            cluster_id: Cluster assignment from
                ``config/cluster_assignments.yaml``.
            cluster_params_dir: Where cluster YAMLs live.

        Returns:
            The first path that exists, as a string.  ``None`` only when even
            the default file is missing (which would indicate a broken repo).
        """
        override = self.overrides_dir / f"{ticker}.yaml"
        if override.exists():
            return str(override)

        cluster_dir = Path(cluster_params_dir)
        if cluster_id is not None:
            cluster_file = cluster_dir / f"cluster_{cluster_id}.yaml"
            if cluster_file.exists():
                return str(cluster_file)

        default_file = cluster_dir / "cluster_default.yaml"
        if default_file.exists():
            return str(default_file)

        return None

    # ------------------------------------------------------------------ #
    # Audit log                                                            #
    # ------------------------------------------------------------------ #

    def log_decision(self, decision: PromotionDecision) -> None:
        """Append a :class:`PromotionDecision` to ``promotion_log.yaml``.

        Uses a read-modify-atomic-write pattern: reads the existing log (if
        any), appends the new entry, and writes the whole document through a
        temp file + rename.  A ``decisions`` list is the only top-level key.

        Args:
            decision: The decision to record.
        """
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        doc: Dict[str, Any]
        if self.log_path.exists():
            with open(self.log_path) as f:
                loaded = yaml.safe_load(f) or {}
            doc = loaded if isinstance(loaded, dict) else {}
        else:
            doc = {}

        entries: List[Dict[str, Any]] = list(doc.get("decisions") or [])
        entries.append(
            {
                "timestamp": decision.timestamp,
                "ticker": decision.ticker,
                "decision": decision.decision,
                "reasons": list(decision.reasons),
                "metrics": {
                    k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in decision.metrics.items()
                },
            }
        )
        doc["decisions"] = entries

        self._atomic_yaml_write(self.log_path, doc)
        logger.info(
            "Logged %s decision for %s (%d total entries).",
            decision.decision, decision.ticker, len(entries),
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_settings(settings_path: str) -> Dict[str, Any]:
        path = Path(settings_path)
        if not path.exists():
            logger.warning(
                "Settings file %s not found — PromotionGate using built-in defaults.",
                settings_path,
            )
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _atomic_yaml_write(path: Path, doc: Dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
        # shutil.move handles cross-filesystem tmpfs/real-FS edge cases that
        # Path.rename() does not.
        shutil.move(str(tmp), str(path))

    def _keep_cluster(
        self,
        ticker: str,
        reasons: List[str],
        metrics: Dict[str, Any],
    ) -> PromotionDecision:
        return PromotionDecision(
            ticker=ticker,
            decision=DECISION_KEEP_CLUSTER,
            reasons=reasons,
            metrics=metrics,
        )

    def _select_pbo(self, wfo: WFOResult) -> tuple[float, str]:
        """Return (pbo_value, source_name) — CPCV when available."""
        if wfo.cpcv_pbo is not None:
            return float(wfo.cpcv_pbo), "cpcv_pbo"
        return float(wfo.pbo), "pbo"
