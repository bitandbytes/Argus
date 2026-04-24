---
name: order-management
description: "Use this skill when implementing the Alpaca OrderManager, the TradingPipeline orchestrator, or the daily run script (Phase 4.1 and 4.4). Triggers on: 'implement OrderManager', 'submit orders to Alpaca', 'paper trading setup', 'implement TradingPipeline', 'wire the pipeline orchestrator', 'run_daily.py', 'idempotent daily run', 'retry on API failure', or any task involving order submission, position tracking, or the end-to-end daily execution loop. Do NOT use for position sizing (use risk-manager) or for LLM signal validation (use llm-validator)."
---

# Order Management Skill

This skill guides Claude in implementing the Alpaca `OrderManager`, the `TradingPipeline` orchestrator, and the `run_daily.py` entry point. These are the execution-layer components that turn sized, validated `TradeSignal` objects into actual (paper) orders and coordinate the full daily pipeline run.

## When to Use This Skill

- Implementing `src/risk/order_manager.py` (Alpaca wrapper)
- Implementing `src/pipeline.py` (`TradingPipeline.run_daily()`)
- Writing or modifying `scripts/run_daily.py`
- Adding retry logic and idempotency guards
- Implementing position tracking and reconciliation
- Setting up paper trading with Alpaca

Do NOT use this skill for:
- Position sizing → use `risk-manager`
- LLM validation → use `llm-validator`
- Backtest runs → use `backtest-runner`

## Architecture Context

```
scripts/run_daily.py
    └── TradingPipeline.run_daily()
            ├── MarketDataProvider.fetch_batch()      # OHLCV
            ├── FinBERTEnricher.batch_enrich()        # sentiment
            ├── RegimeDetector.detect()               # regime
            ├── QuantEngine.generate_signal()         # direction + confidence
            ├── MetaLabelModel.evaluate()             # trade/no-trade + bet_size
            ├── LLMValidator.filter()                 # APPROVE/VETO
            ├── RiskManager.size_position()           # position sizing
            ├── RiskManager.check_portfolio_constraints()
            ├── OrderManager.submit_entry()           # Alpaca
            ├── RiskManager.evaluate_exits()          # open position checks
            └── OrderManager.submit_exit()            # Alpaca
```

The pipeline runs once per trading day, pre-market (default 08:30 ET). It must be idempotent: re-running on the same day must not duplicate orders.

## Step 1: Implement OrderManager

```python
# src/risk/order_manager.py

import logging
import time
from dataclasses import dataclass
from typing import Optional
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
from src.models.trade_signal import TradeSignal

logger = logging.getLogger(__name__)

@dataclass
class OrderResult:
    order_id: str
    ticker: str
    side: str          # "buy" or "sell"
    qty: float
    status: str        # "accepted", "filled", "rejected"
    error: Optional[str] = None


class OrderManager:
    """Wraps the Alpaca REST API for paper and live trading.

    Implements:
    - Market and limit order submission
    - Bracket orders (entry + stop-loss + take-profit)
    - Idempotency via client_order_id (prevents duplicate orders on re-run)
    - Exponential backoff retry for transient API failures
    - Position reconciliation against Alpaca's live position list

    Initialize with paper trading URL for Phase 4 validation:
        base_url = "https://paper-api.alpaca.markets"
    """

    MAX_RETRIES = 4
    BASE_BACKOFF = 2  # seconds

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
    ):
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")
        logger.info("OrderManager connected to %s", base_url)

    # -------------------------------------------------------------------------
    # Entry orders
    # -------------------------------------------------------------------------

    def submit_entry(
        self,
        signal: TradeSignal,
        portfolio_equity: float,
        order_type: str = "market",
    ) -> OrderResult:
        """Submit an entry order for a validated trade signal.

        Uses bracket orders to attach stop-loss and take-profit at submission.
        client_order_id = f"{ticker}_{date}_entry" ensures idempotency.
        """
        if not signal.bet_size or signal.bet_size <= 0:
            return OrderResult(
                order_id="", ticker=signal.ticker, side="", qty=0,
                status="skipped", error="bet_size is zero"
            )

        position_value = signal.bet_size * portfolio_equity
        # Alpaca requires integer qty for stocks; use notional for fractional
        notional = round(position_value, 2)

        side = "buy" if signal.direction == 1 else "sell"
        client_order_id = self._make_client_id(signal.ticker, "entry")

        # Check idempotency before submitting
        if self._order_exists(client_order_id):
            logger.info("Entry order for %s already submitted today — skipping", signal.ticker)
            return OrderResult(
                order_id=client_order_id, ticker=signal.ticker, side=side,
                qty=notional, status="already_submitted"
            )

        order_kwargs = {
            "symbol": signal.ticker,
            "notional": notional,
            "side": side,
            "type": order_type,
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }

        # Attach bracket legs if stop/TP are set
        if signal.stop_loss_pct and signal.take_profit_pct:
            current_price = self._get_latest_price(signal.ticker)
            if current_price:
                stop_price = round(
                    current_price * (1 - signal.direction * signal.stop_loss_pct), 2
                )
                tp_price = round(
                    current_price * (1 + signal.direction * signal.take_profit_pct), 2
                )
                order_kwargs.update({
                    "order_class": "bracket",
                    "stop_loss": {"stop_price": stop_price},
                    "take_profit": {"limit_price": tp_price},
                })

        return self._submit_with_retry(order_kwargs)

    # -------------------------------------------------------------------------
    # Exit orders
    # -------------------------------------------------------------------------

    def submit_exit(self, ticker: str, exit_reason: str) -> OrderResult:
        """Close an open position immediately (market order).

        client_order_id = f"{ticker}_{date}_exit_{reason}" for idempotency.
        """
        client_order_id = self._make_client_id(ticker, f"exit_{exit_reason[:20]}")
        if self._order_exists(client_order_id):
            logger.info("Exit for %s already submitted — skipping", ticker)
            return OrderResult(
                order_id=client_order_id, ticker=ticker, side="close",
                qty=0, status="already_submitted"
            )

        try:
            self.api.close_position(ticker)
            return OrderResult(
                order_id=client_order_id, ticker=ticker, side="sell",
                qty=0, status="accepted"
            )
        except APIError as e:
            logger.error("Failed to close %s: %s", ticker, e)
            return OrderResult(
                order_id="", ticker=ticker, side="sell", qty=0,
                status="rejected", error=str(e)
            )

    # -------------------------------------------------------------------------
    # Position queries
    # -------------------------------------------------------------------------

    def get_open_positions(self) -> dict[str, float]:
        """Return {ticker: market_value} for all open positions."""
        try:
            positions = self.api.list_positions()
            return {p.symbol: float(p.market_value) for p in positions}
        except APIError as e:
            logger.error("Failed to fetch positions: %s", e)
            return {}

    def get_account_equity(self) -> float:
        """Return current account equity."""
        try:
            account = self.api.get_account()
            return float(account.equity)
        except APIError as e:
            logger.error("Failed to fetch account: %s", e)
            return 0.0

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _submit_with_retry(self, order_kwargs: dict) -> OrderResult:
        ticker = order_kwargs["symbol"]
        for attempt in range(self.MAX_RETRIES):
            try:
                order = self.api.submit_order(**order_kwargs)
                return OrderResult(
                    order_id=order.id, ticker=ticker,
                    side=order_kwargs["side"],
                    qty=order_kwargs.get("notional", 0),
                    status=order.status,
                )
            except APIError as e:
                if attempt == self.MAX_RETRIES - 1:
                    logger.error("Order failed after %d retries: %s", self.MAX_RETRIES, e)
                    return OrderResult(
                        order_id="", ticker=ticker, side=order_kwargs["side"],
                        qty=0, status="rejected", error=str(e)
                    )
                backoff = self.BASE_BACKOFF ** (attempt + 1)
                logger.warning("Order attempt %d failed, retrying in %ds: %s", attempt + 1, backoff, e)
                time.sleep(backoff)

    def _order_exists(self, client_order_id: str) -> bool:
        """Check if an order with this client_order_id was already submitted today."""
        try:
            self.api.get_order_by_client_order_id(client_order_id)
            return True
        except APIError:
            return False

    def _make_client_id(self, ticker: str, suffix: str) -> str:
        from datetime import date
        return f"{ticker}_{date.today().isoformat()}_{suffix}"

    def _get_latest_price(self, ticker: str) -> Optional[float]:
        try:
            bars = self.api.get_latest_bar(ticker)
            return float(bars.c)
        except Exception:
            return None
```

## Step 2: Implement TradingPipeline

```python
# src/pipeline.py

import logging
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import yaml
from src.data.market_data import MarketDataProvider
from src.data.news_data import NewsDataProvider
from src.plugins.registry import PluginRegistry
from src.signals.regime_detector import RegimeDetector
from src.signals.quant_engine import QuantEngine
from src.signals.meta_label_model import MetaLabelModel
from src.plugins.filters.llm_validator import LLMValidator
from src.risk.risk_manager import RiskManager, Portfolio
from src.risk.order_manager import OrderManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    watchlist_path: str = "config/watchlist.yaml"
    plugins_path: str = "config/plugins.yaml"
    settings_path: str = "config/settings.yaml"
    mode: str = "paper"  # "paper" or "live"


class TradingPipeline:
    """Orchestrates the full daily trading pipeline.

    Initialization loads all components once. run_daily() executes the
    full sequence: ingest → features → signals → validate → size → execute → exits.

    Idempotency: each step uses date-keyed caches or client_order_ids,
    so re-running on the same day is safe and produces no duplicate orders.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._load_config()
        self._init_components()

    def _load_config(self):
        with open(self.config.settings_path) as f:
            self.settings = yaml.safe_load(f)
        with open(self.config.watchlist_path) as f:
            watchlist_cfg = yaml.safe_load(f)
        self.tickers: list[str] = watchlist_cfg.get("stocks", {})
        self.etf_tickers: set[str] = set(watchlist_cfg.get("etfs", []))
        self.ticker_sectors: dict[str, str] = {
            t: info.get("sector", "Unknown")
            for t, info in watchlist_cfg.get("stocks", {}).items()
        }

    def _init_components(self):
        self.registry = PluginRegistry()
        self.registry.discover_plugins(self.config.plugins_path)

        self.market_data = MarketDataProvider()
        self.news_provider = NewsDataProvider()
        self.regime_detector = RegimeDetector()
        self.quant_engine = QuantEngine(self.registry)
        self.meta_model = MetaLabelModel.load("data/models/meta_model/latest.pkl")
        self.llm_validator = LLMValidator()

        risk_cfg = self.settings.get("risk", {})
        self.risk_manager = RiskManager(registry=self.registry, **risk_cfg)

        alpaca_cfg = self.settings.get("alpaca", {})
        import os
        self.order_manager = OrderManager(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
            base_url=os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        )

    def run_daily(self) -> dict:
        """Execute the full daily pipeline. Returns a summary dict for logging/alerts.

        Steps:
        1. Pre-flight checks (kill switch, market hours)
        2. Data ingestion (OHLCV + news)
        3. Sentiment enrichment (FinBERT)
        4. Signal generation per ticker
        5. Entry orders for approved signals
        6. Exit evaluation for open positions
        7. Exit orders
        """
        today = pd.Timestamp.today().normalize()
        logger.info("=== Daily pipeline run: %s ===", today.date())

        # 1. Pre-flight
        equity = self.order_manager.get_account_equity()
        open_positions_map = self.order_manager.get_open_positions()
        portfolio = self._build_portfolio(equity, open_positions_map)

        if self.risk_manager.is_kill_switch_active(portfolio):
            logger.warning("Kill switch active — skipping all entries today")
            return {"status": "kill_switch", "entries": 0, "exits": 0}

        # 2. Data ingestion
        all_tickers = list(self.tickers.keys()) + list(self.etf_tickers)
        price_data = self.market_data.fetch_batch(all_tickers)

        # 3. Sentiment enrichment
        finbert = self.registry.get_enricher("finbert")
        sentiment_map = finbert.batch_enrich(all_tickers) if finbert else {}

        # 4. Signal generation + validation
        entry_results = []
        current_signals: dict[str, TradeSignal] = {}

        earnings_cal = self.market_data.get_earnings_calendar(all_tickers)

        for ticker in self.tickers:
            df = price_data.get(ticker)
            if df is None or df.empty:
                continue

            sentiment = sentiment_map.get(ticker, {}).get("sentiment_score", 0.0)
            regime = self.regime_detector.detect(df)
            signal = self.quant_engine.generate_signal(ticker, df, regime, sentiment)
            signal = self.meta_model.evaluate(signal, df, sentiment)

            current_signals[ticker] = signal

            if signal.bet_size and signal.bet_size > 0:
                context = {
                    "news_headlines": self.news_provider.get_headlines(ticker, days=7),
                    "earnings_calendar": earnings_cal,
                    "sector": self.ticker_sectors.get(ticker, "Unknown"),
                    "date": today,
                }
                signal = self.llm_validator.filter(signal, context)

            if signal.llm_approved and signal.bet_size and signal.bet_size > 0:
                signal = self.risk_manager.size_position(
                    signal, atr_14=self._get_atr(df), current_price=df["close"].iloc[-1],
                    portfolio=portfolio,
                )
                passes, reason = self.risk_manager.check_portfolio_constraints(
                    signal, self.ticker_sectors.get(ticker, "Unknown"), portfolio
                )
                if passes:
                    result = self.order_manager.submit_entry(signal, equity)
                    entry_results.append(result)
                    logger.info("Entry submitted: %s (status=%s)", ticker, result.status)
                else:
                    logger.info("Entry rejected for %s: %s", ticker, reason)

        # 5. Exit evaluation
        open_pos_contexts = self._build_position_contexts(open_positions_map, price_data)
        exit_results = []

        for position in open_pos_contexts:
            df = price_data.get(position.ticker)
            if df is None:
                continue
            bar = self._build_bar_context(position.ticker, df, sentiment_map)
            context = {
                "earnings_calendar": earnings_cal,
            }

        exits = self.risk_manager.evaluate_exits(
            open_pos_contexts, bar, context, current_signals
        )
        for position, reason in exits:
            result = self.order_manager.submit_exit(position.ticker, reason)
            exit_results.append(result)
            logger.info("Exit submitted: %s reason=%s (status=%s)", position.ticker, reason, result.status)

        summary = {
            "status": "ok",
            "date": str(today.date()),
            "entries_submitted": len(entry_results),
            "exits_submitted": len(exit_results),
            "kill_switch": False,
        }
        logger.info("Pipeline complete: %s", summary)
        return summary

    def _build_portfolio(self, equity: float, open_positions: dict[str, float]) -> Portfolio:
        from src.risk.risk_manager import Portfolio
        return Portfolio(
            equity=equity,
            open_positions=open_positions,
            sector_exposure={},  # TODO: enrich with sector data
            daily_pnl=0.0,       # TODO: pull from Alpaca account.unrealized_pl
            peak_equity=equity,  # TODO: track rolling peak in a local DB
            total_drawdown_pct=0.0,
        )

    def _get_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        import pandas_ta as ta
        atr = ta.atr(df["high"], df["low"], df["close"], length=period)
        return float(atr.iloc[-1]) if atr is not None and not atr.empty else 0.0

    def _build_position_contexts(self, open_positions: dict[str, float], price_data: dict) -> list:
        from src.plugins.base import PositionContext
        contexts = []
        for ticker, market_value in open_positions.items():
            df = price_data.get(ticker)
            if df is None:
                continue
            current_price = float(df["close"].iloc[-1])
            contexts.append(PositionContext(
                ticker=ticker,
                entry_date=pd.Timestamp.today(),  # TODO: load from trade log
                entry_price=current_price,         # TODO: load from trade log
                current_price=current_price,
                direction=1 if market_value > 0 else -1,
                shares=abs(market_value) / current_price,
                unrealized_pnl_pct=0.0,           # TODO: from Alpaca position.unrealized_plpc
            ))
        return contexts

    def _build_bar_context(self, ticker: str, df: pd.DataFrame, sentiment_map: dict):
        from src.plugins.base import BarContext
        row = df.iloc[-1]
        sent = sentiment_map.get(ticker, {})
        return BarContext(
            date=df.index[-1],
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            volume=float(row["volume"]),
            atr_14=self._get_atr(df),
            sentiment_score=sent.get("sentiment_score", 0.0),
            news_volume_ratio=sent.get("news_volume_ratio", 1.0),
        )
```

## Step 3: Write the CLI entry point

```python
# scripts/run_daily.py
"""Daily pipeline runner. Run pre-market, e.g. 08:30 ET.

Usage:
    python scripts/run_daily.py --mode paper
    python scripts/run_daily.py --mode live   # requires 3 months of paper trading first
"""

import argparse
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TradingPipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    args = parser.parse_args()

    config = PipelineConfig(mode=args.mode)
    pipeline = TradingPipeline(config)
    summary = pipeline.run_daily()

    print(f"Pipeline complete: {summary}")
    return 0 if summary["status"] in ("ok", "kill_switch") else 1

if __name__ == "__main__":
    sys.exit(main())
```

## Idempotency Rules

The pipeline must be safe to re-run on the same day without duplicate orders. Enforce idempotency at three levels:

| Level | Mechanism |
|---|---|
| **Entry orders** | `client_order_id = {ticker}_{date}_entry` — Alpaca rejects duplicates |
| **Exit orders** | `client_order_id = {ticker}_{date}_exit_{reason}` |
| **Feature computation** | FeatureStore upserts; already-computed features are not recomputed |

Before submitting any order, call `_order_exists(client_order_id)` and skip if True.

## Retry Policy

Network failures and transient Alpaca errors use exponential backoff:
- Attempt 1: immediate
- Attempt 2: wait 2s
- Attempt 3: wait 4s
- Attempt 4: wait 8s (final)

After 4 failures, log the error and return a `rejected` `OrderResult`. Never raise — a single ticker failure must not abort the whole pipeline.

## Paper Trading Validation Requirements

Before switching `ALPACA_BASE_URL` from paper to live:
- Minimum 3 months of consecutive daily paper trading runs without errors
- Positive total return on paper account
- LLM veto rate in expected range (10–30%)
- All exit reasons appearing in trade log (earnings, news, ATR, confidence)
- Drawdown kill switch tested manually (force a 5% daily loss in paper)

## Validation Checklist

- [ ] `OrderManager` connects to paper API without error
- [ ] `submit_entry()` with the same signal on the same day submits only one order
- [ ] `submit_exit()` uses `close_position()` (not a manual sell order)
- [ ] Retry logic activates on `APIError` and backs off correctly
- [ ] `TradingPipeline.run_daily()` completes without raising on missing price data
- [ ] Kill switch check happens before any order submission
- [ ] Re-running on same day produces no duplicate orders in Alpaca dashboard
- [ ] `scripts/run_daily.py --mode paper` exits with code 0 on success
