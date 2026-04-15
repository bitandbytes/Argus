"""Tests for triple_barrier_labels() — src/signals/triple_barrier.py."""

import pandas as pd
import pytest

from src.signals.triple_barrier import triple_barrier_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prices(values: list, start: str = "2020-01-01") -> pd.Series:
    """Build a close-price Series with a daily DatetimeIndex."""
    return pd.Series(
        [float(v) for v in values],
        index=pd.date_range(start, periods=len(values), freq="D"),
    )


def _signal(prices: pd.Series, idx: int, direction: float) -> tuple:
    """Return (signal_times, signal_directions) for a single signal at prices.index[idx]."""
    sig_time = pd.DatetimeIndex([prices.index[idx]])
    sig_dir = pd.Series([direction], index=sig_time)
    return sig_time, sig_dir


# ---------------------------------------------------------------------------
# Long signals
# ---------------------------------------------------------------------------

class TestLongSignals:
    def test_long_tp_hit(self) -> None:
        """Long trade: price rises to TP on day 2 → label=1, meta_label=1."""
        # entry=100, tp=104 (4%), sl=98 (2%)
        # day 1: 101 (below TP), day 2: 105 (above TP) → TP hit
        p = _prices([100, 101, 105, 90])
        sig_t, sig_d = _signal(p, 0, 1.0)
        df = triple_barrier_labels(p, sig_t, sig_d, tp_pct=0.04, sl_pct=0.02, max_holding_days=5)

        assert len(df) == 1
        row = df.iloc[0]
        assert row["label"] == 1
        assert row["meta_label"] == 1
        assert row["exit_price"] == 105.0
        assert row["tp_price"] == pytest.approx(104.0)
        assert row["sl_price"] == pytest.approx(98.0)

    def test_long_sl_hit(self) -> None:
        """Long trade: price falls to SL on day 2 → label=-1, meta_label=0."""
        # entry=100, tp=104, sl=98; day 1: 99, day 2: 97 → SL hit
        p = _prices([100, 99, 97, 110])
        sig_t, sig_d = _signal(p, 0, 1.0)
        df = triple_barrier_labels(p, sig_t, sig_d, tp_pct=0.04, sl_pct=0.02, max_holding_days=5)

        row = df.iloc[0]
        assert row["label"] == -1
        assert row["meta_label"] == 0
        assert row["exit_price"] == 97.0

    def test_long_timeout(self) -> None:
        """Long trade: price stays between barriers for all max_holding_days → label=0."""
        # entry=100, tp=110 (10%), sl=90 (10%); prices hover around 100
        p = _prices([100, 101, 100, 99, 101, 100, 102])
        sig_t, sig_d = _signal(p, 0, 1.0)
        df = triple_barrier_labels(p, sig_t, sig_d, tp_pct=0.10, sl_pct=0.10, max_holding_days=5)

        row = df.iloc[0]
        assert row["label"] == 0
        assert row["meta_label"] == 0

    def test_long_tp_hit_exact_touch(self) -> None:
        """TP is hit exactly on the boundary price → label=1."""
        # entry=100, tp=104; price on day 1 is exactly 104.0
        p = _prices([100, 104.0, 90])
        sig_t, sig_d = _signal(p, 0, 1.0)
        df = triple_barrier_labels(p, sig_t, sig_d, tp_pct=0.04, sl_pct=0.02)

        assert df.iloc[0]["label"] == 1

    def test_long_sl_hit_exact_touch(self) -> None:
        """SL is hit exactly on the boundary price → label=-1."""
        # entry=100, sl=98; price on day 1 is exactly 98.0
        p = _prices([100, 98.0, 110])
        sig_t, sig_d = _signal(p, 0, 1.0)
        df = triple_barrier_labels(p, sig_t, sig_d, tp_pct=0.04, sl_pct=0.02)

        assert df.iloc[0]["label"] == -1


# ---------------------------------------------------------------------------
# Short signals
# ---------------------------------------------------------------------------

class TestShortSignals:
    def test_short_tp_hit(self) -> None:
        """Short trade: price falls to TP on day 2 → label=1, meta_label=1."""
        # entry=100, tp=96 (4% below), sl=102 (2% above)
        # day 1: 99, day 2: 95 (hits tp=96) → TP hit
        p = _prices([100, 99, 95, 110])
        sig_t, sig_d = _signal(p, 0, -1.0)
        df = triple_barrier_labels(p, sig_t, sig_d, tp_pct=0.04, sl_pct=0.02, max_holding_days=5)

        row = df.iloc[0]
        assert row["label"] == 1
        assert row["meta_label"] == 1
        assert row["tp_price"] == pytest.approx(96.0)
        assert row["sl_price"] == pytest.approx(102.0)

    def test_short_sl_hit(self) -> None:
        """Short trade: price rises to SL on day 2 → label=-1, meta_label=0."""
        # entry=100, sl=102 (2% above); day 1: 101, day 2: 103 → SL hit
        p = _prices([100, 101, 103, 80])
        sig_t, sig_d = _signal(p, 0, -1.0)
        df = triple_barrier_labels(p, sig_t, sig_d, tp_pct=0.04, sl_pct=0.02, max_holding_days=5)

        row = df.iloc[0]
        assert row["label"] == -1
        assert row["meta_label"] == 0

    def test_short_timeout(self) -> None:
        """Short trade: price stays between barriers → label=0."""
        # entry=100, tp=90 (10% below), sl=110 (10% above); prices hover ~100
        p = _prices([100, 101, 99, 100, 101, 99, 100])
        sig_t, sig_d = _signal(p, 0, -1.0)
        df = triple_barrier_labels(p, sig_t, sig_d, tp_pct=0.10, sl_pct=0.10, max_holding_days=5)

        row = df.iloc[0]
        assert row["label"] == 0
        assert row["meta_label"] == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_signal_at_last_bar(self) -> None:
        """Signal on the final bar → no future prices, timeout (label=0)."""
        p = _prices([100, 102, 105])
        sig_t, sig_d = _signal(p, 2, 1.0)  # last bar
        df = triple_barrier_labels(p, sig_t, sig_d, tp_pct=0.04, sl_pct=0.02)

        assert len(df) == 1
        row = df.iloc[0]
        assert row["label"] == 0
        assert row["exit_price"] == row["entry_price"]

    def test_signal_at_second_to_last_bar(self) -> None:
        """Signal with only 1 future bar — exits at that bar regardless of barriers."""
        # entry=100, tp=104; only 1 future bar at 101 (doesn't hit tp=104) → timeout
        p = _prices([100, 101])
        sig_t, sig_d = _signal(p, 0, 1.0)
        df = triple_barrier_labels(p, sig_t, sig_d, tp_pct=0.04, sl_pct=0.02, max_holding_days=20)

        assert df.iloc[0]["label"] == 0

    def test_signal_time_not_in_index(self) -> None:
        """Signal time absent from price index → silently skipped, empty DataFrame."""
        p = _prices([100, 102, 105])
        # signal at a date not in the index
        sig_t = pd.DatetimeIndex(["2021-06-15"])
        sig_d = pd.Series([1.0], index=sig_t)
        df = triple_barrier_labels(p, sig_t, sig_d)

        assert len(df) == 0
        # Correct columns should still be present
        assert "label" in df.columns
        assert "meta_label" in df.columns

    def test_empty_signal_times(self) -> None:
        """Empty signal_times → empty DataFrame with correct columns."""
        p = _prices([100, 102, 105])
        sig_t = pd.DatetimeIndex([])
        sig_d = pd.Series([], dtype=float)
        df = triple_barrier_labels(p, sig_t, sig_d)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        for col in ["entry_price", "direction", "tp_price", "sl_price",
                    "exit_time", "exit_price", "label", "meta_label"]:
            assert col in df.columns

    def test_invalid_direction_skipped(self) -> None:
        """Direction=0 is invalid → row skipped, returns empty DataFrame."""
        p = _prices([100, 102, 105])
        sig_t, sig_d = _signal(p, 0, 0.0)
        df = triple_barrier_labels(p, sig_t, sig_d)

        assert len(df) == 0


# ---------------------------------------------------------------------------
# Multiple signals
# ---------------------------------------------------------------------------

class TestMultipleSignals:
    def test_multiple_signals_mixed_outcomes(self) -> None:
        """Three signals: TP, SL, and timeout — all three rows returned."""
        # Build a price series long enough for all signals and their horizons.
        # Signal 0 at bar 0 (long): bars 1-5 go to 106 → TP hit
        # Signal 1 at bar 10 (long): bars 11-15 go to 97 → SL hit
        # Signal 2 at bar 20 (long): bars 21-25 hover at 101 → timeout
        closes = [100.0] * 30
        closes[5] = 106.0   # bar 5: TP hit for signal at bar 0 (tp=104)
        closes[15] = 97.0   # bar 15: SL hit for signal at bar 10 (sl=98)
        # signal at bar 20: no barrier hit in next 5 bars (all ~100)

        p = _prices(closes)
        dates = p.index
        sig_t = pd.DatetimeIndex([dates[0], dates[10], dates[20]])
        sig_d = pd.Series([1.0, 1.0, 1.0], index=sig_t)

        df = triple_barrier_labels(
            p, sig_t, sig_d, tp_pct=0.04, sl_pct=0.02, max_holding_days=7
        )

        assert len(df) == 3
        assert df.iloc[0]["label"] == 1   # TP hit
        assert df.iloc[1]["label"] == -1  # SL hit
        assert df.iloc[2]["label"] == 0   # timeout

    def test_mixed_long_short_signals(self) -> None:
        """Long and short signals in same batch — each resolved independently."""
        # bar 0 (long, entry=100): bar 1=110 → TP hit (tp=104)
        # bar 5 (short, entry=100): bar 6=94 → TP hit (tp=96)
        closes = [100.0] * 15
        closes[1] = 110.0
        closes[6] = 94.0

        p = _prices(closes)
        dates = p.index
        sig_t = pd.DatetimeIndex([dates[0], dates[5]])
        sig_d = pd.Series([1.0, -1.0], index=sig_t)

        df = triple_barrier_labels(p, sig_t, sig_d, tp_pct=0.04, sl_pct=0.02)

        assert df.iloc[0]["direction"] == 1.0
        assert df.iloc[0]["label"] == 1
        assert df.iloc[1]["direction"] == -1.0
        assert df.iloc[1]["label"] == 1   # short TP hit (price dropped)

    def test_index_name_is_entry_time(self) -> None:
        """Output DataFrame index must be named 'entry_time'."""
        p = _prices([100, 102, 105])
        sig_t, sig_d = _signal(p, 0, 1.0)
        df = triple_barrier_labels(p, sig_t, sig_d)

        assert df.index.name == "entry_time"

    def test_barrier_price_columns_correct(self) -> None:
        """tp_price and sl_price columns store direction-relative values."""
        p = _prices([100, 102])
        # long: tp = 100 * 1.05 = 105, sl = 100 * 0.97 = 97
        sig_t, sig_d = _signal(p, 0, 1.0)
        df = triple_barrier_labels(p, sig_t, sig_d, tp_pct=0.05, sl_pct=0.03)

        assert df.iloc[0]["tp_price"] == pytest.approx(105.0)
        assert df.iloc[0]["sl_price"] == pytest.approx(97.0)

        # short: tp = 100 * 0.95 = 95, sl = 100 * 1.03 = 103
        sig_t2, sig_d2 = _signal(p, 0, -1.0)
        df2 = triple_barrier_labels(p, sig_t2, sig_d2, tp_pct=0.05, sl_pct=0.03)

        assert df2.iloc[0]["tp_price"] == pytest.approx(95.0)
        assert df2.iloc[0]["sl_price"] == pytest.approx(103.0)

    def test_vertical_barrier_respected(self) -> None:
        """Timeout exits at bar max_holding_days from entry, not beyond."""
        # tp/sl far away; 10 future bars with max_holding_days=3
        # expect exit at bar 3 (index 3 in future)
        p = _prices([100, 101, 102, 103, 200, 200])  # bar 4+ would hit TP
        sig_t, sig_d = _signal(p, 0, 1.0)
        df = triple_barrier_labels(
            p, sig_t, sig_d, tp_pct=0.50, sl_pct=0.50, max_holding_days=3
        )

        # With max_holding_days=3, only bars 1-3 are checked (prices 101,102,103)
        # tp=150, sl=50 → none hit → label=0, exit at bar 3 (price=103)
        row = df.iloc[0]
        assert row["label"] == 0
        assert row["exit_price"] == pytest.approx(103.0)
