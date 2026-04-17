"""Tests for PromotionGate, PromotionDecision, and the params resolver.

Covers:
  - Constructor loads thresholds from settings.yaml (with fallback defaults).
  - ``evaluate()`` — each criterion's pass/fail path short-circuits correctly.
  - PBO selection prefers CPCV when available.
  - ``promote()`` writes a valid YAML with inherited weights + tuned params.
  - ``demote()`` removes an existing override and is idempotent.
  - ``check_demotion()`` respects the sharpe_gap threshold, handles NaN/inf.
  - ``log_decision()`` appends to the YAML audit log (atomic round-trip).
  - ``resolve_params_path()`` — override > cluster_id > cluster_default chain.
  - ``is_promoted()`` / ``list_promoted()`` introspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from src.tuning.promotion_gate import (
    DECISION_KEEP_CLUSTER,
    DECISION_PROMOTE,
    PromotionDecision,
    PromotionGate,
)
from src.tuning.walk_forward import WFOResult, WFOWindowResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wfo_result(
    *,
    ticker: str = "AAPL",
    aggregate_oos_sharpe: float = 0.90,
    pbo: float = 0.30,
    cpcv_pbo: float = 0.25,
    is_stable: bool = True,
    n_windows: int = 5,
    best_params: Dict[str, Any] = None,
) -> WFOResult:
    """Build a WFOResult fixture with sensible defaults."""
    import pandas as pd

    best_params = best_params if best_params is not None else {"rsi__period": 14}
    # One dummy window so downstream code that inspects ``windows`` doesn't crash.
    win = WFOWindowResult(
        window_idx=0,
        is_start=pd.Timestamp("2020-01-01"),
        is_end=pd.Timestamp("2020-12-31"),
        oos_start=pd.Timestamp("2021-01-01"),
        oos_end=pd.Timestamp("2021-06-30"),
        best_params=best_params,
        is_sharpe=aggregate_oos_sharpe,
        oos_sharpe=aggregate_oos_sharpe,
        n_trials=10,
        is_stable=is_stable,
    )
    return WFOResult(
        ticker=ticker,
        windows=[win] * n_windows,
        best_params=best_params,
        aggregate_oos_sharpe=aggregate_oos_sharpe,
        pbo=pbo,
        is_stable=is_stable,
        n_windows=n_windows,
        best_window_idx=0,
        cpcv_pbo=cpcv_pbo,
        cpcv_n_paths=10,
    )


@pytest.fixture
def gate(tmp_path: Path) -> PromotionGate:
    """A PromotionGate pointing at tmp_path for overrides + log."""
    settings = {
        "tuning": {
            "promotion": {
                "min_history_bars": 2500,
                "sharpe_improvement_threshold": 1.20,
                "param_drift_threshold": 0.20,
                "pbo_threshold": 0.40,
                "min_oos_trades": 30,
                "demotion_lookback_days": 60,
                "demotion_sharpe_gap": 0.30,
            }
        }
    }
    settings_path = tmp_path / "settings.yaml"
    with open(settings_path, "w") as f:
        yaml.dump(settings, f)
    return PromotionGate(
        settings_path=str(settings_path),
        overrides_dir=str(tmp_path / "stock_overrides"),
        log_path=str(tmp_path / "promotion_log.yaml"),
    )


@pytest.fixture
def cluster_wfo() -> WFOResult:
    return _make_wfo_result(ticker="cluster_0", aggregate_oos_sharpe=0.60)


@pytest.fixture
def strong_individual_wfo() -> WFOResult:
    """Individual WFO that passes every criterion."""
    return _make_wfo_result(
        ticker="AAPL",
        aggregate_oos_sharpe=0.90,  # 0.90 / 0.60 = 1.50 > 1.20
        pbo=0.30,
        cpcv_pbo=0.25,
        is_stable=True,
    )


# ---------------------------------------------------------------------------
# TestPromotionGateInit
# ---------------------------------------------------------------------------

class TestPromotionGateInit:
    def test_loads_thresholds_from_settings(self, gate: PromotionGate) -> None:
        assert gate.min_history_bars == 2500
        assert gate.sharpe_improvement_threshold == 1.20
        assert gate.param_drift_threshold == 0.20
        assert gate.pbo_threshold == 0.40
        assert gate.demotion_lookback_days == 60
        assert gate.demotion_sharpe_gap == 0.30

    def test_missing_settings_uses_defaults(self, tmp_path: Path) -> None:
        g = PromotionGate(
            settings_path=str(tmp_path / "nonexistent.yaml"),
            overrides_dir=str(tmp_path / "overrides"),
            log_path=str(tmp_path / "log.yaml"),
        )
        assert g.min_history_bars == 2500
        assert g.sharpe_improvement_threshold == 1.20
        assert g.pbo_threshold == 0.40

    def test_custom_thresholds_loaded(self, tmp_path: Path) -> None:
        cfg = {
            "tuning": {
                "promotion": {
                    "min_history_bars": 1000,
                    "sharpe_improvement_threshold": 1.50,
                    "pbo_threshold": 0.35,
                    "demotion_sharpe_gap": 0.50,
                }
            }
        }
        p = tmp_path / "s.yaml"
        with open(p, "w") as f:
            yaml.dump(cfg, f)
        g = PromotionGate(
            settings_path=str(p),
            overrides_dir=str(tmp_path / "o"),
            log_path=str(tmp_path / "l.yaml"),
        )
        assert g.min_history_bars == 1000
        assert g.sharpe_improvement_threshold == 1.50
        assert g.pbo_threshold == 0.35
        assert g.demotion_sharpe_gap == 0.50


# ---------------------------------------------------------------------------
# TestEvaluate — criterion-by-criterion
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_promote_when_all_criteria_pass(
        self, gate: PromotionGate, cluster_wfo: WFOResult,
        strong_individual_wfo: WFOResult,
    ) -> None:
        decision = gate.evaluate(
            "AAPL", cluster_wfo, strong_individual_wfo, n_bars_available=3000
        )
        assert decision.decision == DECISION_PROMOTE
        assert decision.ticker == "AAPL"
        assert decision.metrics["pbo_source"] == "cpcv_pbo"
        assert decision.metrics["pbo_used"] == 0.25

    def test_keep_cluster_when_history_too_short(
        self, gate: PromotionGate, cluster_wfo: WFOResult,
        strong_individual_wfo: WFOResult,
    ) -> None:
        decision = gate.evaluate(
            "AAPL", cluster_wfo, strong_individual_wfo, n_bars_available=1000
        )
        assert decision.decision == DECISION_KEEP_CLUSTER
        assert any("insufficient history" in r for r in decision.reasons)

    def test_keep_cluster_when_sharpe_lift_too_small(
        self, gate: PromotionGate, cluster_wfo: WFOResult,
    ) -> None:
        # Individual Sharpe 0.65, cluster 0.60 → ratio 1.083 < 1.20
        individual = _make_wfo_result(aggregate_oos_sharpe=0.65)
        decision = gate.evaluate("AAPL", cluster_wfo, individual, n_bars_available=3000)
        assert decision.decision == DECISION_KEEP_CLUSTER
        assert any("Sharpe lift" in r for r in decision.reasons)
        assert decision.metrics["sharpe_ratio"] == pytest.approx(0.65 / 0.60, abs=0.01)

    def test_keep_cluster_when_unstable(
        self, gate: PromotionGate, cluster_wfo: WFOResult,
    ) -> None:
        individual = _make_wfo_result(aggregate_oos_sharpe=0.90, is_stable=False)
        decision = gate.evaluate("AAPL", cluster_wfo, individual, n_bars_available=3000)
        assert decision.decision == DECISION_KEEP_CLUSTER
        assert any("unstable" in r for r in decision.reasons)

    def test_keep_cluster_when_pbo_too_high(
        self, gate: PromotionGate, cluster_wfo: WFOResult,
    ) -> None:
        individual = _make_wfo_result(
            aggregate_oos_sharpe=0.90, pbo=0.55, cpcv_pbo=0.50,
        )
        decision = gate.evaluate("AAPL", cluster_wfo, individual, n_bars_available=3000)
        assert decision.decision == DECISION_KEEP_CLUSTER
        assert any("overfitting risk" in r for r in decision.reasons)

    def test_pbo_prefers_cpcv_when_available(
        self, gate: PromotionGate, cluster_wfo: WFOResult,
    ) -> None:
        individual = _make_wfo_result(
            aggregate_oos_sharpe=0.90, pbo=0.50, cpcv_pbo=0.20,
        )
        decision = gate.evaluate("AAPL", cluster_wfo, individual, n_bars_available=3000)
        # Per-window PBO alone would fail (0.50 >= 0.40); CPCV saves it.
        assert decision.decision == DECISION_PROMOTE
        assert decision.metrics["pbo_source"] == "cpcv_pbo"

    def test_pbo_falls_back_when_cpcv_missing(
        self, gate: PromotionGate, cluster_wfo: WFOResult,
    ) -> None:
        individual = _make_wfo_result(
            aggregate_oos_sharpe=0.90, pbo=0.30, cpcv_pbo=None,
        )
        decision = gate.evaluate("AAPL", cluster_wfo, individual, n_bars_available=3000)
        assert decision.metrics["pbo_source"] == "pbo"
        assert decision.decision == DECISION_PROMOTE

    def test_cluster_sharpe_nonpositive_uses_abs_lift(
        self, gate: PromotionGate,
    ) -> None:
        """When cluster baseline <= 0, ratio is meaningless — require abs lift."""
        cluster = _make_wfo_result(aggregate_oos_sharpe=-0.10)
        # Individual must beat cluster by >= 0.10 absolute.
        individual_pass = _make_wfo_result(aggregate_oos_sharpe=0.05)
        individual_fail = _make_wfo_result(aggregate_oos_sharpe=-0.05)

        d_pass = gate.evaluate("X", cluster, individual_pass, n_bars_available=3000)
        d_fail = gate.evaluate("X", cluster, individual_fail, n_bars_available=3000)

        assert d_pass.decision == DECISION_PROMOTE
        assert d_fail.decision == DECISION_KEEP_CLUSTER

    def test_metrics_populated_on_every_decision(
        self, gate: PromotionGate, cluster_wfo: WFOResult,
        strong_individual_wfo: WFOResult,
    ) -> None:
        decision = gate.evaluate(
            "AAPL", cluster_wfo, strong_individual_wfo, n_bars_available=3000
        )
        assert "n_bars_available" in decision.metrics
        assert "cluster_oos_sharpe" in decision.metrics
        assert "individual_oos_sharpe" in decision.metrics
        assert "individual_pbo" in decision.metrics
        assert "individual_is_stable" in decision.metrics


# ---------------------------------------------------------------------------
# TestPromote / TestDemote
# ---------------------------------------------------------------------------

class TestPromoteDemote:
    def test_promote_writes_yaml(
        self, gate: PromotionGate, strong_individual_wfo: WFOResult,
    ) -> None:
        weights = {
            "TRENDING_UP": {"rsi": 0.3, "macd": 0.7},
        }
        path = gate.promote(
            ticker="AAPL",
            individual_wfo=strong_individual_wfo,
            cluster_id=2,
            cluster_weights=weights,
        )
        assert path.exists()
        with open(path) as f:
            doc = yaml.safe_load(f)

        assert doc["metadata"]["ticker"] == "AAPL"
        assert doc["metadata"]["cluster_id"] == 2
        assert doc["metadata"]["individual_oos_sharpe"] == 0.9
        assert doc["metadata"]["is_stable"] is True
        assert doc["indicators"]["weights"] == weights
        assert doc["indicators"]["params"] == {"rsi": {"period": 14}}

    def test_promote_embeds_decision_reasons(
        self, gate: PromotionGate, strong_individual_wfo: WFOResult,
        cluster_wfo: WFOResult,
    ) -> None:
        decision = gate.evaluate(
            "AAPL", cluster_wfo, strong_individual_wfo, n_bars_available=3000
        )
        path = gate.promote(
            ticker="AAPL",
            individual_wfo=strong_individual_wfo,
            cluster_id=0,
            cluster_weights={},
            decision=decision,
        )
        with open(path) as f:
            doc = yaml.safe_load(f)
        assert "decision_reasons" in doc["metadata"]
        assert len(doc["metadata"]["decision_reasons"]) > 0

    def test_promote_atomic_write(
        self, gate: PromotionGate, strong_individual_wfo: WFOResult,
    ) -> None:
        """No .tmp file should survive a successful promote."""
        gate.promote(
            ticker="AAPL",
            individual_wfo=strong_individual_wfo,
            cluster_id=0,
            cluster_weights={},
        )
        leftovers = list(gate.overrides_dir.glob("*.tmp"))
        assert leftovers == []

    def test_demote_removes_file(
        self, gate: PromotionGate, strong_individual_wfo: WFOResult,
    ) -> None:
        gate.promote(
            ticker="AAPL",
            individual_wfo=strong_individual_wfo,
            cluster_id=0,
            cluster_weights={},
        )
        assert gate.is_promoted("AAPL")
        assert gate.demote("AAPL") is True
        assert not gate.is_promoted("AAPL")

    def test_demote_idempotent(self, gate: PromotionGate) -> None:
        assert gate.demote("NOT_PROMOTED") is False

    def test_list_promoted_sorted(
        self, gate: PromotionGate, strong_individual_wfo: WFOResult,
    ) -> None:
        for t in ["MSFT", "AAPL", "GOOGL"]:
            gate.promote(
                ticker=t, individual_wfo=strong_individual_wfo,
                cluster_id=0, cluster_weights={},
            )
        assert gate.list_promoted() == ["AAPL", "GOOGL", "MSFT"]

    def test_list_promoted_empty_when_dir_missing(
        self, tmp_path: Path,
    ) -> None:
        g = PromotionGate(
            settings_path=str(tmp_path / "missing.yaml"),
            overrides_dir=str(tmp_path / "never_created"),
            log_path=str(tmp_path / "log.yaml"),
        )
        assert g.list_promoted() == []


# ---------------------------------------------------------------------------
# TestCheckDemotion
# ---------------------------------------------------------------------------

class TestCheckDemotion:
    def test_gap_exceeds_threshold_returns_true(self, gate: PromotionGate) -> None:
        # cluster 0.80, rolling 0.40 → gap 0.40 > 0.30 → DEMOTE
        assert gate.check_demotion("AAPL", rolling_sharpe_60d=0.40,
                                   cluster_baseline_sharpe=0.80) is True

    def test_gap_below_threshold_returns_false(self, gate: PromotionGate) -> None:
        # cluster 0.80, rolling 0.60 → gap 0.20 < 0.30 → KEEP
        assert gate.check_demotion("AAPL", rolling_sharpe_60d=0.60,
                                   cluster_baseline_sharpe=0.80) is False

    def test_rolling_above_cluster_returns_false(self, gate: PromotionGate) -> None:
        assert gate.check_demotion("AAPL", rolling_sharpe_60d=1.20,
                                   cluster_baseline_sharpe=0.80) is False

    def test_nan_returns_false(self, gate: PromotionGate) -> None:
        assert gate.check_demotion("AAPL", rolling_sharpe_60d=float("nan"),
                                   cluster_baseline_sharpe=0.80) is False

    def test_inf_returns_false(self, gate: PromotionGate) -> None:
        assert gate.check_demotion("AAPL", rolling_sharpe_60d=float("-inf"),
                                   cluster_baseline_sharpe=0.80) is False

    def test_exactly_at_threshold_returns_false(self, gate: PromotionGate) -> None:
        # gap == threshold → not strictly greater → KEEP.
        # Use values whose subtraction is exact in IEEE-754: 0.5 - 0.2 = 0.3.
        assert gate.check_demotion("AAPL", rolling_sharpe_60d=0.2,
                                   cluster_baseline_sharpe=0.5) is False


# ---------------------------------------------------------------------------
# TestLogDecision
# ---------------------------------------------------------------------------

class TestLogDecision:
    def test_first_decision_creates_log(
        self, gate: PromotionGate,
    ) -> None:
        d = PromotionDecision(
            ticker="AAPL", decision=DECISION_PROMOTE,
            reasons=["all good"], metrics={"x": 1.0},
        )
        gate.log_decision(d)
        assert gate.log_path.exists()
        with open(gate.log_path) as f:
            doc = yaml.safe_load(f)
        assert len(doc["decisions"]) == 1
        assert doc["decisions"][0]["ticker"] == "AAPL"
        assert doc["decisions"][0]["decision"] == DECISION_PROMOTE

    def test_multiple_decisions_append(self, gate: PromotionGate) -> None:
        for t in ["AAPL", "MSFT", "GOOGL"]:
            gate.log_decision(
                PromotionDecision(ticker=t, decision=DECISION_KEEP_CLUSTER)
            )
        with open(gate.log_path) as f:
            doc = yaml.safe_load(f)
        tickers = [e["ticker"] for e in doc["decisions"]]
        assert tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_log_is_valid_yaml_roundtrip(
        self, gate: PromotionGate,
    ) -> None:
        d = PromotionDecision(
            ticker="AAPL", decision=DECISION_PROMOTE,
            reasons=["r1", "r2"],
            metrics={"a": 1.5, "b": 2},
        )
        gate.log_decision(d)
        with open(gate.log_path) as f:
            doc = yaml.safe_load(f)
        entry = doc["decisions"][0]
        assert entry["reasons"] == ["r1", "r2"]
        # Numeric metrics serialise cleanly.
        assert entry["metrics"]["a"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# TestResolveParamsPath
# ---------------------------------------------------------------------------

class TestResolveParamsPath:
    def _setup_cluster_dir(self, tmp_path: Path) -> Path:
        cluster_dir = tmp_path / "cluster_params"
        cluster_dir.mkdir()
        (cluster_dir / "cluster_default.yaml").write_text("x: 1\n")
        (cluster_dir / "cluster_0.yaml").write_text("x: 0\n")
        (cluster_dir / "cluster_1.yaml").write_text("x: 1\n")
        return cluster_dir

    def test_override_takes_precedence(
        self, gate: PromotionGate, strong_individual_wfo: WFOResult,
        tmp_path: Path,
    ) -> None:
        cluster_dir = self._setup_cluster_dir(tmp_path)
        gate.promote(
            ticker="AAPL", individual_wfo=strong_individual_wfo,
            cluster_id=0, cluster_weights={},
        )
        path = gate.resolve_params_path(
            "AAPL", cluster_id=0, cluster_params_dir=str(cluster_dir),
        )
        assert path is not None
        assert "stock_overrides" in path
        assert path.endswith("AAPL.yaml")

    def test_falls_back_to_cluster_id(
        self, gate: PromotionGate, tmp_path: Path,
    ) -> None:
        cluster_dir = self._setup_cluster_dir(tmp_path)
        # No override for MSFT.
        path = gate.resolve_params_path(
            "MSFT", cluster_id=1, cluster_params_dir=str(cluster_dir),
        )
        assert path is not None
        assert path.endswith("cluster_1.yaml")

    def test_falls_back_to_default(
        self, gate: PromotionGate, tmp_path: Path,
    ) -> None:
        cluster_dir = self._setup_cluster_dir(tmp_path)
        path = gate.resolve_params_path(
            "MSFT", cluster_id=None, cluster_params_dir=str(cluster_dir),
        )
        assert path is not None
        assert path.endswith("cluster_default.yaml")

    def test_returns_none_when_nothing_exists(
        self, gate: PromotionGate, tmp_path: Path,
    ) -> None:
        empty = tmp_path / "empty_cluster_dir"
        empty.mkdir()
        path = gate.resolve_params_path(
            "MSFT", cluster_id=None, cluster_params_dir=str(empty),
        )
        assert path is None

    def test_missing_cluster_id_falls_through_to_default(
        self, gate: PromotionGate, tmp_path: Path,
    ) -> None:
        cluster_dir = self._setup_cluster_dir(tmp_path)
        # cluster_id=99 doesn't exist → fall through to default.
        path = gate.resolve_params_path(
            "MSFT", cluster_id=99, cluster_params_dir=str(cluster_dir),
        )
        assert path is not None
        assert path.endswith("cluster_default.yaml")


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_evaluate_promote_log_resolve_cycle(
        self, gate: PromotionGate, cluster_wfo: WFOResult,
        strong_individual_wfo: WFOResult, tmp_path: Path,
    ) -> None:
        # Full workflow a caller would perform.
        decision = gate.evaluate(
            "AAPL", cluster_wfo, strong_individual_wfo, n_bars_available=3000
        )
        assert decision.decision == DECISION_PROMOTE

        gate.promote(
            ticker="AAPL",
            individual_wfo=strong_individual_wfo,
            cluster_id=0,
            cluster_weights={"TRENDING_UP": {"rsi": 1.0}},
            decision=decision,
        )
        gate.log_decision(decision)

        assert gate.is_promoted("AAPL")
        resolved = gate.resolve_params_path(
            "AAPL", cluster_id=0,
            cluster_params_dir=str(tmp_path / "cluster_params"),
        )
        assert resolved is not None and "AAPL.yaml" in resolved

        # Simulate demotion → log again → verify override is gone.
        gate.demote("AAPL")
        demote_decision = PromotionDecision(
            ticker="AAPL", decision="DEMOTE",
            reasons=["rolling 60d Sharpe dropped 0.35 below cluster"],
            metrics={"gap": 0.35},
        )
        gate.log_decision(demote_decision)

        assert not gate.is_promoted("AAPL")
        with open(gate.log_path) as f:
            doc = yaml.safe_load(f)
        assert len(doc["decisions"]) == 2
        assert doc["decisions"][-1]["decision"] == "DEMOTE"
