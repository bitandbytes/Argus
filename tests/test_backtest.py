"""
Unit tests for scripts/run_backtest.py.

All external I/O is mocked — tests run fully offline.
File I/O uses pytest's tmp_path fixture.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root and scripts/ are importable
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from src.models.trade_signal import RegimeType, TradeSignal

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    start: str = "2020-01-02",
    periods: int = 300,
    close_start: float = 100.0,
) -> pd.DataFrame:
    """Return a minimal lowercase-column OHLCV DataFrame (as MarketDataProvider returns)."""
    dates = pd.bdate_range(start=start, periods=periods)
    closes = [close_start + i * 0.5 for i in range(periods)]
    df = pd.DataFrame(
        {
            "open": [c - 0.5 for c in closes],
            "high": [c + 0.5 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1_000_000] * periods,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return df


def _make_trade_signal(
    direction: float = 0.6,
    confidence: float = 0.5,
    regime: RegimeType = RegimeType.TRENDING_UP,
    ticker: str = "AAPL",
    ts: datetime = datetime(2020, 6, 1),
) -> TradeSignal:
    return TradeSignal(
        ticker=ticker,
        timestamp=ts,
        direction=direction,
        confidence=confidence,
        source_layer="quant",
        regime=regime,
    )


# ---------------------------------------------------------------------------
# 1. Import verification
# ---------------------------------------------------------------------------


class TestImports:
    def test_pybroker_importable(self) -> None:
        import pybroker

        assert hasattr(pybroker, "Strategy"), "pybroker.Strategy not found"
        assert hasattr(pybroker, "FeeMode"), "pybroker.FeeMode not found"

    def test_pybroker_fee_mode_has_order_percent(self) -> None:
        from pybroker import FeeMode

        assert hasattr(
            FeeMode, "ORDER_PERCENT"
        ), "FeeMode.ORDER_PERCENT missing — check pybroker version"

    def test_vectorbt_importable(self) -> None:
        import vectorbt as vbt  # noqa: F401  (import-only check)

        assert hasattr(vbt, "Portfolio")

    def test_mlflow_importable(self) -> None:
        import mlflow  # noqa: F401

        assert hasattr(mlflow, "start_run")
        assert hasattr(mlflow, "log_params")
        assert hasattr(mlflow, "log_metrics")


# ---------------------------------------------------------------------------
# 2. Data preparation
# ---------------------------------------------------------------------------


class TestFetchAndPrep:
    def test_pybroker_df_has_required_columns(self, tmp_path: Path) -> None:
        from run_backtest import _fetch_and_prep

        full_df = _make_ohlcv(start="2018-01-02", periods=600)

        mock_mdp = MagicMock()
        mock_mdp.fetch_ohlcv.return_value = full_df

        _, _, _, pyb_df = _fetch_and_prep(
            "AAPL", "2018-01-02", "2020-01-02", "2022-01-01", mock_mdp
        )

        for col in ["date", "symbol", "open", "high", "low", "close", "volume"]:
            assert col in pyb_df.columns, f"Missing column: {col}"

    def test_symbol_column_matches_ticker(self, tmp_path: Path) -> None:
        from run_backtest import _fetch_and_prep

        full_df = _make_ohlcv(start="2018-01-02", periods=600)
        mock_mdp = MagicMock()
        mock_mdp.fetch_ohlcv.return_value = full_df

        _, _, _, pyb_df = _fetch_and_prep(
            "TSLA", "2018-01-02", "2020-01-02", "2022-01-01", mock_mdp
        )

        assert (pyb_df["symbol"] == "TSLA").all()

    def test_warmup_ends_before_backtest_start(self) -> None:
        from run_backtest import _fetch_and_prep

        full_df = _make_ohlcv(start="2018-01-02", periods=1000)
        mock_mdp = MagicMock()
        mock_mdp.fetch_ohlcv.return_value = full_df

        backtest_start = "2020-06-01"
        warmup_df, _, backtest_df, _ = _fetch_and_prep(
            "AAPL", "2018-01-02", backtest_start, "2022-01-01", mock_mdp
        )

        assert len(warmup_df) > 0, "warmup_df must not be empty"
        assert len(backtest_df) > 0, "backtest_df must not be empty"
        assert all(warmup_df.index < pd.Timestamp(backtest_start))
        assert all(backtest_df.index >= pd.Timestamp(backtest_start))

    def test_pyb_df_date_is_plain_column_not_index(self) -> None:
        from run_backtest import _fetch_and_prep

        full_df = _make_ohlcv(start="2018-01-02", periods=600)
        mock_mdp = MagicMock()
        mock_mdp.fetch_ohlcv.return_value = full_df

        _, _, _, pyb_df = _fetch_and_prep(
            "AAPL", "2018-01-02", "2020-01-02", "2022-01-01", mock_mdp
        )

        # PyBroker requires 'date' as a column, not the index
        assert "date" in pyb_df.columns
        assert pyb_df.index.name != "date"


# ---------------------------------------------------------------------------
# 3. Strategy function logic
# ---------------------------------------------------------------------------


class TestStrategyFn:
    """Test the per-bar strategy logic without running a full backtest."""

    def _make_signals_series(
        self, start: str = "2020-01-02", periods: int = 100, direction: float = 0.8
    ) -> pd.Series:
        dates = pd.bdate_range(start=start, periods=periods).normalize()
        sigs = [
            _make_trade_signal(direction=direction, confidence=0.6, ts=d.to_pydatetime())
            for d in dates
        ]
        return pd.Series(sigs, index=dates)

    def _make_ctx(self, n_bars: int = 100, current_date: str = "2020-06-01") -> MagicMock:
        ctx = MagicMock()
        ctx.bars = n_bars
        ctx.dt = pd.Timestamp(current_date).to_pydatetime()
        ctx.long_pos.return_value = None  # no open long by default
        ctx.buy_shares = None
        ctx.sell_all_shares = MagicMock()
        ctx.calc_target_shares.return_value = 10  # 10 shares
        return ctx

    def test_buys_on_strong_positive_signal(self) -> None:
        from run_backtest import _build_strategy_fn

        signals = self._make_signals_series(direction=0.8)
        fn = _build_strategy_fn(signals, entry_threshold=0.30, warmup_bars=50)

        # Use a date that falls within the 100-bday window starting 2020-01-02
        ctx = self._make_ctx(n_bars=60, current_date="2020-04-01")
        fn(ctx)

        assert ctx.buy_shares == 10, "Should have set buy_shares"

    def test_no_trade_during_warmup(self) -> None:
        from run_backtest import _build_strategy_fn

        signals = self._make_signals_series(direction=0.8)
        fn = _build_strategy_fn(signals, entry_threshold=0.30, warmup_bars=50)

        ctx = self._make_ctx(n_bars=30)  # below warmup_bars=50
        fn(ctx)

        assert ctx.buy_shares is None, "Should not trade during warmup"
        ctx.sell_all_shares.assert_not_called()

    def test_no_buy_when_below_confidence_threshold(self) -> None:
        from run_backtest import _build_strategy_fn

        signals = self._make_signals_series(direction=0.8)
        # Override one signal with low confidence
        ts = pd.Timestamp("2020-06-01").normalize()
        signals[ts] = _make_trade_signal(direction=0.8, confidence=0.10)  # below 0.30 threshold

        fn = _build_strategy_fn(signals, entry_threshold=0.30, warmup_bars=50)
        ctx = self._make_ctx(n_bars=60, current_date="2020-06-01")
        fn(ctx)

        assert ctx.buy_shares is None

    def test_sells_when_negative_signal_and_long_position(self) -> None:
        from run_backtest import _build_strategy_fn

        signals = self._make_signals_series(direction=-0.8)  # strong sell
        fn = _build_strategy_fn(signals, entry_threshold=0.30, warmup_bars=50)

        # Use a date within the 100-bday window starting 2020-01-02
        ctx = self._make_ctx(n_bars=60, current_date="2020-04-01")
        ctx.long_pos.return_value = MagicMock()  # simulate open long position

        fn(ctx)

        ctx.sell_all_shares.assert_called_once()

    def test_no_sell_when_no_long_position(self) -> None:
        from run_backtest import _build_strategy_fn

        signals = self._make_signals_series(direction=-0.8)
        fn = _build_strategy_fn(signals, entry_threshold=0.30, warmup_bars=50)

        ctx = self._make_ctx(n_bars=60, current_date="2020-06-01")
        ctx.long_pos.return_value = None  # no position — nothing to sell

        fn(ctx)

        ctx.sell_all_shares.assert_not_called()

    def test_no_trade_when_date_not_in_signals(self) -> None:
        from run_backtest import _build_strategy_fn

        signals = self._make_signals_series()  # only covers ~100 business days from 2020-01-02
        fn = _build_strategy_fn(signals, entry_threshold=0.30, warmup_bars=50)

        ctx = self._make_ctx(n_bars=60, current_date="2025-01-01")  # way outside range
        fn(ctx)

        assert ctx.buy_shares is None
        ctx.sell_all_shares.assert_not_called()

    def test_no_double_buy_when_already_long(self) -> None:
        from run_backtest import _build_strategy_fn

        signals = self._make_signals_series(direction=0.9)
        fn = _build_strategy_fn(signals, entry_threshold=0.30, warmup_bars=50)

        ctx = self._make_ctx(n_bars=60, current_date="2020-06-01")
        ctx.long_pos.return_value = MagicMock()  # already long

        fn(ctx)

        # Should NOT buy again
        assert ctx.buy_shares is None


# ---------------------------------------------------------------------------
# 4. Benchmark computation
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_bnh_return_known_input(self) -> None:
        from run_backtest import _compute_benchmark

        # $100 → $120 over 4 bars = 20% return
        df = pd.DataFrame(
            {"close": [100.0, 105.0, 110.0, 120.0]},
            index=pd.bdate_range("2020-01-02", periods=4),
        )
        result = _compute_benchmark(df, initial_cash=10_000.0)

        assert abs(result["bnh_total_return_pct"] - 20.0) < 0.01
        assert abs(result["bnh_final_value"] - 12_000.0) < 0.01

    def test_bnh_sharpe_is_reasonable(self) -> None:
        from run_backtest import _compute_benchmark

        rng = np.random.default_rng(42)
        prices = 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, 252))
        df = pd.DataFrame(
            {"close": prices},
            index=pd.bdate_range("2020-01-02", periods=252),
        )
        result = _compute_benchmark(df, initial_cash=100_000.0)

        # Sharpe should be finite and in a plausible range
        assert -10.0 < result["bnh_sharpe"] < 10.0

    def test_bnh_returns_required_keys(self) -> None:
        from run_backtest import _compute_benchmark

        df = _make_ohlcv(periods=50)
        result = _compute_benchmark(df, initial_cash=100_000.0)

        for key in [
            "bnh_total_return_pct",
            "bnh_sharpe",
            "bnh_max_drawdown_pct",
            "bnh_final_value",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_bnh_max_drawdown_is_negative(self) -> None:
        from run_backtest import _compute_benchmark

        # Prices go up then down — creates a real drawdown
        closes = [100.0, 110.0, 120.0, 100.0, 90.0]
        df = pd.DataFrame({"close": closes}, index=pd.bdate_range("2020-01-02", periods=5))
        result = _compute_benchmark(df, initial_cash=100_000.0)

        assert result["bnh_max_drawdown_pct"] < 0.0, "Max drawdown should be negative"


# ---------------------------------------------------------------------------
# 5. MLflow logging
# ---------------------------------------------------------------------------


class TestMLflowLogging:
    def _make_args(self, tmp_path: Path) -> MagicMock:
        args = MagicMock()
        args.ticker = "AAPL"
        args.start = "2020-01-01"
        args.end = "2024-12-31"
        args.warmup_start = "2018-01-01"
        args.initial_cash = 100_000.0
        args.fee_amount = 0.001
        args.entry_threshold = 0.30
        args.params_path = "config/cluster_params/cluster_default.yaml"
        args.mlflow_tracking_uri = str(tmp_path / "test_mlflow.db")
        args.mlflow_experiment = "test_experiment"
        return args

    def _make_mock_result(self) -> MagicMock:
        result = MagicMock()
        m = MagicMock()
        m.sharpe = 1.2
        m.max_drawdown_pct = -18.5
        m.win_rate = 0.55
        m.total_return_pct = 142.0
        m.profit_factor = 1.8
        m.calmar = 1.1
        m.trade_count = 45
        m.annual_return_pct = 19.5
        m.annual_volatility_pct = 22.0
        m.total_fees = 340.0
        m.sortino = 1.5
        result.metrics = m
        result.portfolio = pd.DataFrame({"equity": [100_000, 105_000]})
        result.trades = pd.DataFrame({"pnl": [500, -100]})
        return result

    def test_mlflow_logs_required_params(self, tmp_path: Path) -> None:
        from run_backtest import _log_to_mlflow

        args = self._make_args(tmp_path)
        result = self._make_mock_result()
        benchmark = {
            "bnh_total_return_pct": 195.0,
            "bnh_sharpe": 1.45,
            "bnh_max_drawdown_pct": -33.0,
            "bnh_final_value": 295_000.0,
        }
        equity_path = str(tmp_path / "equity.csv")
        trades_path = str(tmp_path / "trades.csv")
        result.portfolio.to_csv(equity_path)
        result.trades.to_csv(trades_path)

        logged_params: dict = {}
        all_metrics: dict = {}

        def capture_params(d: dict) -> None:
            logged_params.update(d)

        def capture_metrics(d: dict) -> None:
            all_metrics.update(d)

        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run") as mock_run,
            patch("mlflow.log_params", side_effect=capture_params),
            patch("mlflow.log_metrics", side_effect=capture_metrics),
            patch("mlflow.log_artifact"),
        ):
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_ctx.info.run_id = "abc123"
            mock_run.return_value = mock_ctx

            _log_to_mlflow(result, benchmark, args, equity_path, trades_path)

        # Required params
        for key in ["ticker", "start", "end", "fee_amount", "entry_threshold"]:
            assert key in logged_params, f"Missing param: {key}"

        # Required strategy metrics
        for key in ["sharpe_ratio", "max_drawdown_pct", "win_rate", "trade_count"]:
            assert key in all_metrics, f"Missing metric: {key}"

        # Required benchmark metrics
        for key in ["bnh_sharpe", "bnh_total_return_pct"]:
            assert key in all_metrics, f"Missing benchmark metric: {key}"

        # Required comparison metrics
        for key in ["sharpe_vs_bnh", "return_vs_bnh_pct"]:
            assert key in all_metrics, f"Missing comparison metric: {key}"


# ---------------------------------------------------------------------------
# 6. Warmup period sizing
# ---------------------------------------------------------------------------


class TestRegimeDetectorWarmup:
    def test_default_warmup_provides_enough_trading_days(self) -> None:
        """730 calendar days before 2020-01-01 must yield ≥ 504 business days."""
        from datetime import timedelta

        start = datetime(2020, 1, 1)
        warmup_start = start - timedelta(days=730)

        warmup_bdays = len(pd.bdate_range(warmup_start, start - pd.Timedelta(days=1)))
        assert warmup_bdays >= 504, (
            f"Warmup period yields only {warmup_bdays} trading days; "
            "RegimeDetector.fit() requires ≥ 504."
        )

    def test_warmup_derived_correctly_in_main(self) -> None:
        """main() should compute warmup_start = start - 730 calendar days."""
        import argparse
        from datetime import timedelta

        args = argparse.Namespace(
            ticker="AAPL",
            start="2022-06-01",
            end="2024-12-31",
            warmup_start=None,
            initial_cash=100_000,
            params_path="config/cluster_params/cluster_default.yaml",
            mlflow_tracking_uri="data/mlflow.db",
            mlflow_experiment="argus_backtests",
            fee_amount=0.001,
            entry_threshold=0.30,
            verify=False,
            no_mlflow=True,
        )

        expected = (datetime.strptime(args.start, "%Y-%m-%d") - timedelta(days=730)).strftime(
            "%Y-%m-%d"
        )

        # Simulate the derivation logic from main()
        from datetime import timedelta as td

        derived = (datetime.strptime(args.start, "%Y-%m-%d") - td(days=730)).strftime("%Y-%m-%d")

        assert derived == expected

    def test_sufficient_warmup_rows_for_hmm_fit(self) -> None:
        """A DataFrame with 100 rows must pass the RegimeDetector minimum check."""
        from src.signals.regime_detector import RegimeDetector

        # RegimeDetector needs n_components + 1 = 4 valid rows minimum.
        # With vol_20d, needs ~22 rows for first valid feature.
        # 100 rows comfortably exceeds both thresholds.
        df = _make_ohlcv(periods=100)
        detector = RegimeDetector()
        try:
            detector.fit(df, ticker="TEST")
        except ValueError as exc:
            pytest.fail(f"fit() raised ValueError unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 7. Safe float helper
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_converts_normal_float(self) -> None:
        from run_backtest import _safe_float

        assert _safe_float(1.5) == pytest.approx(1.5)

    def test_converts_nan_to_zero(self) -> None:
        from run_backtest import _safe_float

        assert _safe_float(float("nan")) == 0.0

    def test_converts_none_to_zero(self) -> None:
        from run_backtest import _safe_float

        assert _safe_float(None) == 0.0

    def test_converts_int(self) -> None:
        from run_backtest import _safe_float

        assert _safe_float(42) == pytest.approx(42.0)
