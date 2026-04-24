---
name: risk-manager
description: "Use this skill when implementing or modifying the RiskManager (Phase 4.3). Triggers on: 'implement RiskManager', 'ATR position sizing', 'Kelly fraction', 'portfolio-level risk limits', 'max position size', 'sector exposure limit', 'drawdown kill switch', 'evaluate exits', 'stop-loss placement', 'take-profit placement', or any task involving position sizing, portfolio constraints, or the daily risk evaluation loop. Do NOT use for individual EventFilter plugins (use event-filter) or for order submission to Alpaca (use order-management)."
---

# Risk Manager Skill

This skill guides Claude in implementing `RiskManager` — the component that translates approved trade signals into sized, risk-bounded orders and monitors open positions for exit conditions. It sits between the signal cascade and order execution.

## When to Use This Skill

- Implementing `src/risk/risk_manager.py`
- Adding ATR-based position sizing with Kelly fraction
- Implementing portfolio-level constraints (max position %, sector exposure)
- Wiring daily/total drawdown kill switches
- Connecting `EventFilter` instances to the exit evaluation loop
- Populating `TradeSignal.stop_loss_pct` and `take_profit_pct`

Do NOT use this skill for:
- Implementing individual `EventFilter` plugins → use `event-filter`
- Alpaca order submission → use `order-management`
- ATR-based trade exit filters → use `event-filter` (`AtrStopFilter`)

## Architecture Context

`RiskManager` receives a fully validated `TradeSignal` (quant + ML + LLM approved) and:
1. Computes position size using ATR-based sizing + Kelly fraction + regime and confidence scaling.
2. Checks portfolio-level constraints (max position, sector exposure, kill switches).
3. Populates `stop_loss_pct` and `take_profit_pct` on the signal.
4. After each daily bar: evaluates all open positions against `EventFilter` instances plus a confidence-based soft exit.

```
TradeSignal (llm_approved=True)
        │
        ▼
 RiskManager.size_position()    → bet_size, stop_loss_pct, take_profit_pct
        │
 RiskManager.check_portfolio()  → passes or rejects on portfolio constraints
        │
        ▼
 OrderManager.submit()
        │
 [next day]
        │
 RiskManager.evaluate_exits()   → EventFilters + confidence exit
```

## Step 1: Implement RiskManager

```python
# src/risk/risk_manager.py

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import logging
from src.models.trade_signal import TradeSignal, RegimeType
from src.plugins.base import EventFilter, PositionContext, BarContext
from src.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)

@dataclass
class Portfolio:
    """Snapshot of current portfolio state for constraint checking."""
    equity: float                          # total account value
    open_positions: dict[str, float]       # ticker → position value
    sector_exposure: dict[str, float]      # sector → total value
    daily_pnl: float                       # realized + unrealized today
    peak_equity: float                     # rolling peak for drawdown calc
    total_drawdown_pct: float              # (peak - current) / peak

    @property
    def total_exposure_pct(self) -> float:
        return sum(self.open_positions.values()) / self.equity if self.equity > 0 else 0.0


class RiskManager:
    """Sizes positions, enforces portfolio constraints, and evaluates exits.

    Configuration is loaded from config/settings.yaml::risk block:
        max_position_pct: 0.10      # max single position as % of equity
        max_sector_pct: 0.30        # max sector exposure as % of equity
        kelly_fraction: 0.25        # fractional Kelly (full Kelly is too aggressive)
        atr_risk_pct: 0.02          # % of equity to risk per position (1 ATR move)
        daily_drawdown_kill: 0.05   # halt trading if daily loss > 5%
        total_drawdown_kill: 0.15   # halt trading if total drawdown > 15%
        min_confidence_exit: 0.30   # exit position if confidence drops below this
    """

    def __init__(
        self,
        registry: PluginRegistry,
        max_position_pct: float = 0.10,
        max_sector_pct: float = 0.30,
        kelly_fraction: float = 0.25,
        atr_risk_pct: float = 0.02,
        daily_drawdown_kill: float = 0.05,
        total_drawdown_kill: float = 0.15,
        min_confidence_exit: float = 0.30,
    ):
        self.registry = registry
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.kelly_fraction = kelly_fraction
        self.atr_risk_pct = atr_risk_pct
        self.daily_drawdown_kill = daily_drawdown_kill
        self.total_drawdown_kill = total_drawdown_kill
        self.min_confidence_exit = min_confidence_exit

    # -------------------------------------------------------------------------
    # Entry sizing
    # -------------------------------------------------------------------------

    def size_position(
        self,
        signal: TradeSignal,
        atr_14: float,
        current_price: float,
        portfolio: Portfolio,
    ) -> TradeSignal:
        """Compute bet_size, stop_loss_pct, take_profit_pct and attach to signal.

        Sizing approach:
        1. ATR-based risk: risk atr_risk_pct × equity on one ATR move.
        2. Scale by signal confidence and meta-model bet_size.
        3. Scale down in VOLATILE regime (0.5×).
        4. Apply Kelly fraction cap.
        5. Cap at max_position_pct of equity.

        Returns the signal with all risk fields populated (or bet_size=0 if rejected).
        """
        if atr_14 <= 0 or current_price <= 0:
            signal.bet_size = 0.0
            return signal

        # Base size: risk atr_risk_pct * equity on a 1-ATR move
        dollar_risk = portfolio.equity * self.atr_risk_pct
        shares_from_atr = dollar_risk / atr_14
        position_value = shares_from_atr * current_price

        # Confidence scaling (signal.confidence from QuantEngine, already in [0,1])
        position_value *= signal.confidence

        # Meta-model bet_size scaling (in [0,1] from CalibratedClassifierCV)
        if signal.bet_size is not None:
            position_value *= signal.bet_size

        # Regime adjustment
        if signal.regime == RegimeType.VOLATILE:
            position_value *= 0.5

        # Kelly fraction cap
        position_value *= self.kelly_fraction

        # Hard cap at max_position_pct
        max_value = portfolio.equity * self.max_position_pct
        position_value = min(position_value, max_value)

        # Convert to fraction of equity
        signal.bet_size = position_value / portfolio.equity

        # ATR-based stop and take-profit
        stop_distance_pct = (2.0 * atr_14) / current_price
        tp_distance_pct = (4.0 * atr_14) / current_price
        signal.stop_loss_pct = stop_distance_pct
        signal.take_profit_pct = tp_distance_pct

        logger.debug(
            "Sized %s: bet_size=%.3f, stop=%.2f%%, tp=%.2f%%",
            signal.ticker, signal.bet_size, stop_distance_pct * 100, tp_distance_pct * 100,
        )
        return signal

    # -------------------------------------------------------------------------
    # Portfolio constraint checks
    # -------------------------------------------------------------------------

    def check_portfolio_constraints(
        self,
        signal: TradeSignal,
        sector: str,
        portfolio: Portfolio,
    ) -> tuple[bool, Optional[str]]:
        """Return (passes, rejection_reason). Reject if any constraint is violated.

        Checks (in order):
        1. Kill switches (daily and total drawdown)
        2. Max single position size
        3. Max sector exposure
        """
        # Kill switch: daily drawdown
        if portfolio.daily_pnl / portfolio.equity < -self.daily_drawdown_kill:
            return False, f"daily_drawdown_kill ({portfolio.daily_pnl / portfolio.equity:.1%})"

        # Kill switch: total drawdown
        if portfolio.total_drawdown_pct > self.total_drawdown_kill:
            return False, f"total_drawdown_kill ({portfolio.total_drawdown_pct:.1%})"

        # Max position check
        new_position_pct = signal.bet_size or 0.0
        if new_position_pct > self.max_position_pct:
            return False, f"max_position_pct ({new_position_pct:.1%} > {self.max_position_pct:.1%})"

        # Max sector check
        current_sector_pct = portfolio.sector_exposure.get(sector, 0.0) / portfolio.equity
        projected = current_sector_pct + new_position_pct
        if projected > self.max_sector_pct:
            return False, f"max_sector_pct ({projected:.1%} > {self.max_sector_pct:.1%})"

        return True, None

    # -------------------------------------------------------------------------
    # Exit evaluation
    # -------------------------------------------------------------------------

    def evaluate_exits(
        self,
        open_positions: list[PositionContext],
        bar: BarContext,
        context: dict,
        current_signals: dict[str, TradeSignal],
    ) -> list[tuple[PositionContext, str]]:
        """Evaluate all open positions for exit conditions.

        Runs EventFilters first (hard exits), then confidence-based soft exit.
        Returns list of (position, exit_reason) for positions that should close.
        """
        exits = []
        event_filters: list[EventFilter] = self.registry.get_all_event_filters()

        for position in open_positions:
            exit_reason = self._check_position_exits(
                position, bar, context, event_filters, current_signals
            )
            if exit_reason:
                exits.append((position, exit_reason))

        return exits

    def _check_position_exits(
        self,
        position: PositionContext,
        bar: BarContext,
        context: dict,
        event_filters: list[EventFilter],
        current_signals: dict[str, TradeSignal],
    ) -> Optional[str]:
        # 1. Hard exits from EventFilters (earnings, news shock, ATR stop)
        for f in event_filters:
            should, reason = f.should_exit(position, bar, context)
            if should:
                return reason

        # 2. Soft exit: current quant confidence dropped below threshold
        current_signal = current_signals.get(position.ticker)
        if current_signal and current_signal.confidence < self.min_confidence_exit:
            return f"low_confidence ({current_signal.confidence:.2f})"

        return None

    # -------------------------------------------------------------------------
    # Kill switch state
    # -------------------------------------------------------------------------

    def is_kill_switch_active(self, portfolio: Portfolio) -> bool:
        """Return True if trading should be halted today."""
        daily_loss_pct = -portfolio.daily_pnl / portfolio.equity if portfolio.equity > 0 else 0.0
        return (
            daily_loss_pct > self.daily_drawdown_kill
            or portfolio.total_drawdown_pct > self.total_drawdown_kill
        )
```

## Step 2: Load config from settings.yaml

Add a `risk:` block to `config/settings.yaml`:

```yaml
risk:
  max_position_pct: 0.10
  max_sector_pct: 0.30
  kelly_fraction: 0.25
  atr_risk_pct: 0.02
  daily_drawdown_kill: 0.05
  total_drawdown_kill: 0.15
  min_confidence_exit: 0.30
```

Load it in the orchestrator:

```python
import yaml
with open("config/settings.yaml") as f:
    cfg = yaml.safe_load(f)

risk_cfg = cfg.get("risk", {})
risk_manager = RiskManager(registry=registry, **risk_cfg)
```

## Sizing Math Reference

For a $100,000 account, AAPL at $150 with ATR(14) = $3, signal confidence = 0.75, meta bet_size = 0.6:

```
dollar_risk = $100,000 × 0.02 = $2,000
shares_from_atr = $2,000 / $3 = 667 shares
position_value = 667 × $150 = $100,050
× confidence 0.75 = $75,037
× bet_size 0.6 = $45,022
× kelly_fraction 0.25 = $11,256
cap at 10% equity = $10,000

bet_size = $10,000 / $100,000 = 0.10 (exactly at cap)
stop_loss_pct = (2 × $3) / $150 = 4.0%
take_profit_pct = (4 × $3) / $150 = 8.0%
```

## Unit Tests

```python
# tests/risk/test_risk_manager.py
import pytest
from src.risk.risk_manager import RiskManager, Portfolio
from src.models.trade_signal import TradeSignal, RegimeType
from src.plugins.registry import PluginRegistry

@pytest.fixture
def portfolio():
    return Portfolio(
        equity=100_000, open_positions={}, sector_exposure={},
        daily_pnl=0.0, peak_equity=100_000, total_drawdown_pct=0.0,
    )

@pytest.fixture
def signal():
    return TradeSignal(
        ticker="AAPL", direction=1, confidence=0.75,
        regime=RegimeType.TRENDING_UP, bet_size=0.6,
    )

@pytest.fixture
def rm():
    return RiskManager(registry=PluginRegistry())

def test_bet_size_capped_at_max_position(rm, signal, portfolio):
    # Very high ATR would produce oversized position — must be capped
    signal = rm.size_position(signal, atr_14=0.01, current_price=150.0, portfolio=portfolio)
    assert signal.bet_size <= rm.max_position_pct

def test_volatile_regime_halves_size(rm, signal, portfolio):
    signal.regime = RegimeType.VOLATILE
    result = rm.size_position(signal, atr_14=3.0, current_price=150.0, portfolio=portfolio)
    signal.regime = RegimeType.TRENDING_UP
    result_trending = rm.size_position(signal, atr_14=3.0, current_price=150.0, portfolio=portfolio)
    assert result.bet_size < result_trending.bet_size

def test_daily_drawdown_kill_switch(rm, signal, portfolio):
    portfolio.daily_pnl = -6_000  # 6% loss — exceeds 5% kill
    passes, reason = rm.check_portfolio_constraints(signal, "Technology", portfolio)
    assert not passes
    assert "daily_drawdown_kill" in reason

def test_sector_cap(rm, signal, portfolio):
    portfolio.sector_exposure["Technology"] = 25_000  # already 25%
    signal.bet_size = 0.10  # new position would be 35% → over 30% cap
    passes, reason = rm.check_portfolio_constraints(signal, "Technology", portfolio)
    assert not passes
    assert "max_sector_pct" in reason

def test_low_confidence_soft_exit(rm):
    from src.plugins.base import PositionContext, BarContext
    position = PositionContext(
        ticker="AAPL", entry_date=pd.Timestamp("2024-01-01"),
        entry_price=150.0, current_price=145.0,
        direction=1, shares=10, unrealized_pnl_pct=-0.033,
    )
    bar = BarContext(date=pd.Timestamp("2024-01-15"), open=145.0, high=146.0,
                    low=144.0, close=145.5, volume=50_000_000,
                    atr_14=3.0, sentiment_score=0.0, news_volume_ratio=1.0)
    low_confidence_signal = TradeSignal(
        ticker="AAPL", direction=1, confidence=0.20,  # below 0.30 threshold
        regime=RegimeType.RANGING, bet_size=0.0,
    )
    exits = rm.evaluate_exits([position], bar, {}, {"AAPL": low_confidence_signal})
    assert len(exits) == 1
    assert "low_confidence" in exits[0][1]
```

## Validation Checklist

- [ ] `size_position()` produces `bet_size <= max_position_pct` for all inputs
- [ ] VOLATILE regime position is half the TRENDING_UP size for identical inputs
- [ ] `stop_loss_pct` = 2 × ATR / price; `take_profit_pct` = 4 × ATR / price
- [ ] Daily drawdown kill switch halts new entries after 5% daily loss
- [ ] Total drawdown kill switch activates at 15% drawdown
- [ ] Sector cap prevents adding a position when sector is already at 30%
- [ ] `evaluate_exits()` checks EventFilters before the confidence-based soft exit
- [ ] `is_kill_switch_active()` returns True correctly in both kill-switch scenarios
