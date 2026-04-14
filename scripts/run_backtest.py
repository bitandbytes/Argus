"""
Backtest runner — Task 1.8.

Wires QuantEngine + RegimeDetector into PyBroker for historical simulation.
Signals are pre-computed via generate_series() (forward-only, no lookahead)
and looked up by date inside PyBroker's event loop.

Usage:
    python scripts/run_backtest.py --ticker AAPL --start 2020-01-01 --end 2024-12-31

    # Verify PyBroker import only
    python scripts/run_backtest.py --verify
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import mlflow
import numpy as np
import pandas as pd
import pybroker
from pybroker import ExecContext, FeeMode, Strategy, StrategyConfig
from pybroker.strategy import TestResult

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import MarketDataProvider
from src.models.trade_signal import TradeSignal
from src.plugins.registry import PluginRegistry
from src.signals.quant_engine import QuantEngine
from src.signals.regime_detector import RegimeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #
# CLI                                                                          #
# ──────────────────────────────────────────────────────────────────────────── #


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a PyBroker backtest of the Argus quant engine.")
    p.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL)")
    p.add_argument("--start", default="2020-01-01", help="Backtest window start (YYYY-MM-DD)")
    p.add_argument("--end", default="2024-12-31", help="Backtest window end (YYYY-MM-DD)")
    p.add_argument(
        "--warmup-start",
        default=None,
        help="HMM training window start (default: start - 730 days)",
    )
    p.add_argument("--initial-cash", type=float, default=100_000.0)
    p.add_argument(
        "--params-path",
        default="config/cluster_params/cluster_default.yaml",
        help="Regime weight YAML for QuantEngine",
    )
    p.add_argument(
        "--mlflow-tracking-uri",
        default="data/mlflow.db",
        help="SQLite DB path for MLflow (default: data/mlflow.db)",
    )
    p.add_argument("--mlflow-experiment", default="argus_backtests")
    p.add_argument(
        "--fee-amount",
        type=float,
        default=0.001,
        help="ORDER_PERCENT fee per trade — covers commission + slippage (default: 0.001 = 0.1%%)",
    )
    p.add_argument(
        "--entry-threshold",
        type=float,
        default=0.30,
        help="Minimum signal confidence required to open a position (default: 0.30)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Just verify PyBroker import and exit",
    )
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────── #
# Component loading                                                            #
# ──────────────────────────────────────────────────────────────────────────── #


def _load_components(
    params_path: str,
) -> tuple[PluginRegistry, QuantEngine, RegimeDetector]:
    """Load and initialise the plugin registry, QuantEngine, and RegimeDetector."""
    registry = PluginRegistry()
    registry.discover_plugins("config/plugins.yaml")

    quant_engine = QuantEngine(
        registry,
        settings_path="config/settings.yaml",
        params_path=params_path,
    )
    regime_detector = RegimeDetector()

    return registry, quant_engine, regime_detector


# ──────────────────────────────────────────────────────────────────────────── #
# Data preparation                                                             #
# ──────────────────────────────────────────────────────────────────────────── #


def _fetch_and_prep(
    ticker: str,
    warmup_start: str,
    start: str,
    end: str,
    mdp: MarketDataProvider,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch OHLCV and split into warmup / backtest / PyBroker DataFrames.

    Returns:
        Tuple of (warmup_df, full_df, backtest_df, pyb_df) where:
          - warmup_df: rows before backtest start (used to fit HMM)
          - full_df: all rows (warmup + backtest, used for signal pre-computation)
          - backtest_df: rows from backtest start onwards
          - pyb_df: backtest_df reformatted for PyBroker (date column, symbol column)
    """
    logger.info("Fetching OHLCV for %s from %s to %s", ticker, warmup_start, end)
    full_df = mdp.fetch_ohlcv(ticker, start=warmup_start, end=end)

    backtest_start = pd.Timestamp(start)
    warmup_df = full_df[full_df.index < backtest_start]
    backtest_df = full_df[full_df.index >= backtest_start]

    logger.info("Split: warmup=%d bars, backtest=%d bars", len(warmup_df), len(backtest_df))

    # PyBroker needs 'date' as a plain column (not the index) plus a 'symbol' column.
    pyb_df = backtest_df.reset_index()  # DatetimeIndex → 'date' column
    pyb_df["symbol"] = ticker
    pyb_df = pyb_df[["date", "symbol", "open", "high", "low", "close", "volume"]]

    return warmup_df, full_df, backtest_df, pyb_df


# ──────────────────────────────────────────────────────────────────────────── #
# Regime fitting                                                               #
# ──────────────────────────────────────────────────────────────────────────── #


def _fit_regime(
    detector: RegimeDetector,
    warmup_df: pd.DataFrame,
    ticker: str,
) -> RegimeDetector:
    """Fit HMM on warmup data only. Raises if warmup is too short."""
    logger.info("Fitting RegimeDetector on %d warmup bars for %s", len(warmup_df), ticker)
    detector.fit(warmup_df, ticker=ticker)
    logger.info("RegimeDetector fitted successfully")
    return detector


# ──────────────────────────────────────────────────────────────────────────── #
# Signal pre-computation                                                       #
# ──────────────────────────────────────────────────────────────────────────── #


def _precompute_signals(
    quant_engine: QuantEngine,
    regime_detector: RegimeDetector,
    full_df: pd.DataFrame,
    ticker: str,
) -> pd.Series:
    """Pre-compute all signals for the full period using forward-only slicing.

    Uses generate_series() which internally slices df.iloc[:i+1] per bar —
    strictly no lookahead bias.

    Returns:
        pd.Series of TradeSignal objects aligned to full_df.index.
    """
    logger.info("Computing regime series for %d bars...", len(full_df))
    regime_series = regime_detector.detect_series(full_df)

    logger.info("Pre-computing signals via generate_series() (this may take a minute)...")
    signals = quant_engine.generate_series(full_df, regime_series, ticker)

    # Normalise index to date-only (midnight) for reliable lookup by ctx.dt
    signals.index = signals.index.normalize()

    n_buy = sum(1 for s in signals if s.direction > 0 and s.confidence >= 0.30)
    n_sell = sum(1 for s in signals if s.direction < 0 and s.confidence >= 0.30)
    logger.info("Signals computed: %d buy candidates, %d sell candidates", n_buy, n_sell)

    return signals


# ──────────────────────────────────────────────────────────────────────────── #
# PyBroker strategy function                                                   #
# ──────────────────────────────────────────────────────────────────────────── #


def _build_strategy_fn(
    signals: pd.Series,
    entry_threshold: float = 0.30,
    warmup_bars: int = 50,
) -> Callable[[ExecContext], None]:
    """Build a PyBroker strategy function that looks up pre-computed signals.

    Args:
        signals: Pre-computed TradeSignal Series indexed by normalised dates.
        entry_threshold: Minimum confidence to open a position.
        warmup_bars: Bars to skip at the start (indicators need history).

    Returns:
        Callable compatible with Strategy.add_execution().
    """

    def strategy_fn(ctx: ExecContext) -> None:
        # ctx.bars is an int — number of bars completed for this symbol
        if ctx.bars < warmup_bars:
            return

        # ctx.dt is a datetime — the current bar's date
        ts = pd.Timestamp(ctx.dt).normalize()
        if ts not in signals.index:
            return

        signal: TradeSignal = signals[ts]
        has_long = ctx.long_pos() is not None

        if signal.direction > 0 and signal.confidence >= entry_threshold and not has_long:
            # Size to 95% of portfolio equity; leave buffer for fees
            shares = ctx.calc_target_shares(0.95)
            if shares > 0:
                ctx.buy_shares = shares

        elif has_long and signal.direction < 0 and signal.confidence >= entry_threshold:
            ctx.sell_all_shares()

    return strategy_fn


# ──────────────────────────────────────────────────────────────────────────── #
# Benchmark computation                                                        #
# ──────────────────────────────────────────────────────────────────────────── #


def _compute_benchmark(
    backtest_df: pd.DataFrame,
    initial_cash: float,
) -> dict:
    """Compute buy-and-hold benchmark metrics for the backtest period."""
    close = backtest_df["close"]
    first_close = float(close.iloc[0])
    last_close = float(close.iloc[-1])

    shares = initial_cash / first_close
    final_value = shares * last_close
    total_return_pct = (final_value / initial_cash - 1.0) * 100.0

    daily_returns = close.pct_change().dropna()
    bnh_sharpe = float(
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0
        else 0.0
    )

    rolling_max = close.cummax()
    drawdown = (close - rolling_max) / rolling_max
    bnh_max_dd_pct = float(drawdown.min()) * 100.0

    return {
        "bnh_total_return_pct": round(total_return_pct, 4),
        "bnh_sharpe": round(bnh_sharpe, 4),
        "bnh_max_drawdown_pct": round(bnh_max_dd_pct, 4),
        "bnh_final_value": round(final_value, 2),
    }


# ──────────────────────────────────────────────────────────────────────────── #
# PyBroker execution                                                           #
# ──────────────────────────────────────────────────────────────────────────── #


def _run_pybroker(
    pyb_df: pd.DataFrame,
    strategy_fn: Callable[[ExecContext], None],
    start: str,
    end: str,
    ticker: str,
    initial_cash: float,
    fee_amount: float,
) -> TestResult:
    """Configure and run the PyBroker backtest.

    Args:
        pyb_df: Backtest-period DataFrame with required PyBroker columns.
        strategy_fn: Per-bar execution callback built by _build_strategy_fn().
        start: Backtest start date string.
        end: Backtest end date string.
        ticker: Ticker symbol string.
        initial_cash: Starting portfolio cash.
        fee_amount: Fee as a fraction of order value (ORDER_PERCENT mode).

    Returns:
        PyBroker TestResult with portfolio, trades, and EvalMetrics.
    """
    config = StrategyConfig(
        initial_cash=initial_cash,
        fee_mode=FeeMode.ORDER_PERCENT,
        fee_amount=fee_amount,  # 0.1% per order covers commission + slippage
        buy_delay=1,  # fill on next bar's open — realistic execution
        sell_delay=1,
        exit_on_last_bar=True,  # close any open position at end of backtest
    )

    strategy = Strategy(
        data_source=pyb_df,
        start_date=start,
        end_date=end,
        config=config,
    )
    strategy.add_execution(strategy_fn, [ticker])

    logger.info("Running PyBroker backtest for %s (%s → %s)...", ticker, start, end)
    result = strategy.backtest()
    logger.info("Backtest complete — %d trades executed", len(result.trades))
    return result


# ──────────────────────────────────────────────────────────────────────────── #
# Artifact saving                                                              #
# ──────────────────────────────────────────────────────────────────────────── #


def _save_artifacts(
    result: TestResult,
    ticker: str,
    start: str,
    end: str,
    results_dir: str = "data/results",
) -> tuple[str, str]:
    """Save equity curve and trades to CSV files.

    Returns:
        Tuple of (equity_csv_path, trades_csv_path).
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    tag = f"{ticker}_{start}_{end}"

    equity_path = f"{results_dir}/equity_{tag}.csv"
    trades_path = f"{results_dir}/trades_{tag}.csv"

    result.portfolio.to_csv(equity_path)
    result.trades.to_csv(trades_path)

    logger.info("Artifacts saved: %s, %s", equity_path, trades_path)
    return equity_path, trades_path


# ──────────────────────────────────────────────────────────────────────────── #
# MLflow logging                                                               #
# ──────────────────────────────────────────────────────────────────────────── #


def _log_to_mlflow(
    result: TestResult,
    benchmark: dict,
    args: argparse.Namespace,
    equity_path: str,
    trades_path: str,
) -> str:
    """Log params, metrics, and artifacts to MLflow.

    Returns:
        MLflow run_id string.
    """
    db_abs = str(Path(args.mlflow_tracking_uri).resolve())
    mlflow.set_tracking_uri(f"sqlite:///{db_abs}")
    mlflow.set_experiment(args.mlflow_experiment)

    run_name = f"backtest_{args.ticker}_{args.start}_{args.end}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "ticker": args.ticker,
                "start": args.start,
                "end": args.end,
                "warmup_start": args.warmup_start,
                "initial_cash": args.initial_cash,
                "fee_mode": "ORDER_PERCENT",
                "fee_amount": args.fee_amount,
                "entry_threshold": args.entry_threshold,
                "params_path": args.params_path,
                "pybroker_version": pybroker.__version__,
            }
        )

        m = result.metrics
        strategy_metrics = {
            "sharpe_ratio": _safe_float(m.sharpe),
            "max_drawdown_pct": _safe_float(m.max_drawdown_pct),
            "win_rate": _safe_float(m.win_rate),
            "total_return_pct": _safe_float(m.total_return_pct),
            "profit_factor": _safe_float(m.profit_factor),
            "calmar": _safe_float(m.calmar),
            "trade_count": float(m.trade_count),
            "annual_return_pct": _safe_float(m.annual_return_pct),
            "annual_volatility_pct": _safe_float(m.annual_volatility_pct),
            "total_fees": _safe_float(m.total_fees),
            "sortino": _safe_float(m.sortino),
        }
        mlflow.log_metrics(strategy_metrics)
        mlflow.log_metrics(benchmark)

        # Comparison vs benchmark
        mlflow.log_metrics(
            {
                "sharpe_vs_bnh": _safe_float(m.sharpe) - benchmark["bnh_sharpe"],
                "return_vs_bnh_pct": _safe_float(m.total_return_pct)
                - benchmark["bnh_total_return_pct"],
                "max_dd_vs_bnh_pct": _safe_float(m.max_drawdown_pct)
                - benchmark["bnh_max_drawdown_pct"],
            }
        )

        mlflow.log_artifact(equity_path)
        mlflow.log_artifact(trades_path)

        run_id = run.info.run_id

    logger.info("MLflow run logged: %s (experiment: %s)", run_id, args.mlflow_experiment)
    return run_id


def _safe_float(val: object) -> float:
    """Convert metric values to float, replacing NaN/None with 0.0."""
    try:
        f = float(val)  # type: ignore[arg-type]
        return f if not (f != f) else 0.0  # nan check
    except (TypeError, ValueError):
        return 0.0


# ──────────────────────────────────────────────────────────────────────────── #
# Console summary                                                              #
# ──────────────────────────────────────────────────────────────────────────── #


def _print_summary(
    result: TestResult,
    benchmark: dict,
    run_id: Optional[str],
    ticker: str,
    start: str,
    end: str,
) -> None:
    m = result.metrics
    w = 46

    print()
    print("=" * w)
    print(f"  Backtest Results: {ticker}  {start} → {end}")
    print("=" * w)
    print(f"  {'Metric':<24} {'Strategy':>8}  {'Buy&Hold':>8}")
    print("-" * w)
    print(
        f"  {'Sharpe Ratio':<24} {_safe_float(m.sharpe):>8.3f}" f"  {benchmark['bnh_sharpe']:>8.3f}"
    )
    print(
        f"  {'Max Drawdown %':<24} {_safe_float(m.max_drawdown_pct):>8.2f}"
        f"  {benchmark['bnh_max_drawdown_pct']:>8.2f}"
    )
    print(
        f"  {'Total Return %':<24} {_safe_float(m.total_return_pct):>8.2f}"
        f"  {benchmark['bnh_total_return_pct']:>8.2f}"
    )
    print(f"  {'Annual Return %':<24} {_safe_float(m.annual_return_pct):>8.2f}" f"  {'N/A':>8}")
    print(
        f"  {'Annual Volatility %':<24} {_safe_float(m.annual_volatility_pct):>8.2f}"
        f"  {'N/A':>8}"
    )
    print(f"  {'Win Rate':<24} {_safe_float(m.win_rate):>8.3f}  {'N/A':>8}")
    print(f"  {'Profit Factor':<24} {_safe_float(m.profit_factor):>8.3f}  {'N/A':>8}")
    print(f"  {'Calmar Ratio':<24} {_safe_float(m.calmar):>8.3f}  {'N/A':>8}")
    print(f"  {'Sortino Ratio':<24} {_safe_float(m.sortino):>8.3f}  {'N/A':>8}")
    print(f"  {'Trade Count':<24} {int(m.trade_count):>8}  {'N/A':>8}")
    print(f"  {'Total Fees':<24} {_safe_float(m.total_fees):>8.2f}  {'N/A':>8}")
    print("-" * w)
    if run_id:
        print(f"  MLflow Run ID: {run_id[:8]}...")
    print("=" * w)
    print()

    # Warn if Sharpe is outside the expected range for Phase 1
    sharpe = _safe_float(m.sharpe)
    if not -1.0 <= sharpe <= 2.0:
        logger.warning(
            "Sharpe %.3f is outside the expected Phase 1 range [-1.0, +2.0] — "
            "check for lookahead bias or data issues.",
            sharpe,
        )


# ──────────────────────────────────────────────────────────────────────────── #
# Entrypoint                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #


def main() -> None:
    args = _parse_args()

    # ── Task 1.8 item 1: verify PyBroker import ──────────────────────────── #
    logger.info("PyBroker %s imported successfully", pybroker.__version__)
    if args.verify:
        print(f"PyBroker {pybroker.__version__} — import OK")
        return

    # Derive warmup start if not provided
    if args.warmup_start is None:
        warmup_dt = datetime.strptime(args.start, "%Y-%m-%d") - timedelta(days=730)
        args.warmup_start = warmup_dt.strftime("%Y-%m-%d")
        logger.info("warmup_start derived: %s", args.warmup_start)

    # ── Load components ───────────────────────────────────────────────────── #
    _, quant_engine, regime_detector = _load_components(args.params_path)
    mdp = MarketDataProvider()

    # ── Data preparation ──────────────────────────────────────────────────── #
    warmup_df, full_df, backtest_df, pyb_df = _fetch_and_prep(
        args.ticker, args.warmup_start, args.start, args.end, mdp
    )

    # ── HMM fitting ───────────────────────────────────────────────────────── #
    _fit_regime(regime_detector, warmup_df, args.ticker)

    # ── Signal pre-computation ────────────────────────────────────────────── #
    signals = _precompute_signals(quant_engine, regime_detector, full_df, args.ticker)

    # ── Benchmark ─────────────────────────────────────────────────────────── #
    benchmark = _compute_benchmark(backtest_df, args.initial_cash)

    # ── PyBroker backtest ─────────────────────────────────────────────────── #
    strategy_fn = _build_strategy_fn(signals, entry_threshold=args.entry_threshold)
    result = _run_pybroker(
        pyb_df,
        strategy_fn,
        args.start,
        args.end,
        args.ticker,
        args.initial_cash,
        args.fee_amount,
    )

    # ── Save artifacts ────────────────────────────────────────────────────── #
    equity_path, trades_path = _save_artifacts(result, args.ticker, args.start, args.end)

    # ── MLflow logging ────────────────────────────────────────────────────── #
    run_id: Optional[str] = None
    if not args.no_mlflow:
        run_id = _log_to_mlflow(result, benchmark, args, equity_path, trades_path)

    # ── Console summary ───────────────────────────────────────────────────── #
    _print_summary(result, benchmark, run_id, args.ticker, args.start, args.end)


if __name__ == "__main__":
    main()
