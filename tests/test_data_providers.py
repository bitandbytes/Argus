"""
Unit tests for the Data Layer (src/data/).

All yfinance network calls are mocked — tests run offline.
File I/O uses pytest's ``tmp_path`` fixture (real temp directories, no mocking).
"""

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.feature_store import FeatureStore
from src.data.market_data import DataFetchError, MarketDataProvider
from src.data.news_data import NewsDataProvider
from src.models.trade_signal import FeatureVector, RegimeType


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(
    start: str = "2024-01-02",
    periods: int = 5,
    close_prices: list[float] | None = None,
) -> pd.DataFrame:
    """Return a minimal capitalised OHLCV DataFrame (as yfinance returns)."""
    dates = pd.date_range(start=start, periods=periods, freq="B")
    closes = close_prices if close_prices is not None else [100.0 + i for i in range(periods)]
    df = pd.DataFrame(
        {
            "Open": [c - 1 for c in closes],
            "High": [c + 1 for c in closes],
            "Low": [c - 2 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * periods,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture()
def mdp(tmp_path: Path) -> MarketDataProvider:
    """MarketDataProvider backed by a temporary cache directory."""
    return MarketDataProvider(cache_dir=str(tmp_path / "raw"), fetch_delay_sec=0.0)


@pytest.fixture()
def fs(tmp_path: Path) -> FeatureStore:
    """FeatureStore backed by a temporary feature directory."""
    return FeatureStore(feature_dir=str(tmp_path / "features"))


# ---------------------------------------------------------------------------
# MarketDataProvider — caching behaviour
# ---------------------------------------------------------------------------


class TestMarketDataProviderCaching:
    def test_fetch_ohlcv_uses_cache(self, mdp: MarketDataProvider) -> None:
        """Second fetch should return cached data without calling yfinance again."""
        mock_df = _make_ohlcv()
        # Patch is_cache_stale so the cached file is treated as fresh regardless
        # of the mock data's dates (which are in 2024 but the test runs in 2026+).
        with patch("yfinance.download", return_value=mock_df) as mock_dl, \
             patch.object(mdp, "is_cache_stale", return_value=False):
            mdp.fetch_ohlcv("AAPL", start="2024-01-01", end="2024-01-10")
            mdp.fetch_ohlcv("AAPL", start="2024-01-01", end="2024-01-10")
        mock_dl.assert_called_once()

    def test_fetch_ohlcv_force_refresh(self, mdp: MarketDataProvider) -> None:
        """force_refresh=True should call yfinance even when cache exists."""
        mock_df = _make_ohlcv()
        with patch("yfinance.download", return_value=mock_df) as mock_dl:
            mdp.fetch_ohlcv("AAPL", start="2024-01-01", end="2024-01-10")
            mdp.fetch_ohlcv("AAPL", start="2024-01-01", end="2024-01-10", force_refresh=True)
        assert mock_dl.call_count == 2

    def test_fetch_ohlcv_cache_persists_to_parquet(
        self, mdp: MarketDataProvider, tmp_path: Path
    ) -> None:
        """After a successful fetch the Parquet cache file must exist."""
        with patch("yfinance.download", return_value=_make_ohlcv()):
            mdp.fetch_ohlcv("MSFT", start="2024-01-01")
        assert (tmp_path / "raw" / "MSFT" / "daily.parquet").exists()


# ---------------------------------------------------------------------------
# MarketDataProvider — column normalisation
# ---------------------------------------------------------------------------


class TestMarketDataProviderNormalisation:
    def test_columns_are_lowercase(self, mdp: MarketDataProvider) -> None:
        """Returned DataFrame must have lowercase column names."""
        with patch("yfinance.download", return_value=_make_ohlcv()):
            df = mdp.fetch_ohlcv("AAPL", start="2024-01-01")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_index_named_date(self, mdp: MarketDataProvider) -> None:
        """Index must be a DatetimeIndex named 'date'."""
        with patch("yfinance.download", return_value=_make_ohlcv()):
            df = mdp.fetch_ohlcv("AAPL", start="2024-01-01")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "date"

    def test_multiindex_columns_flattened(self, mdp: MarketDataProvider) -> None:
        """MultiIndex columns (yfinance group_by behaviour) must be flattened."""
        raw = _make_ohlcv()
        raw.columns = pd.MultiIndex.from_tuples(
            [(c, "AAPL") for c in raw.columns], names=["Price", "Ticker"]
        )
        with patch("yfinance.download", return_value=raw):
            df = mdp.fetch_ohlcv("AAPL", start="2024-01-01")
        assert "close" in df.columns
        assert not isinstance(df.columns, pd.MultiIndex)

    def test_volume_is_integer(self, mdp: MarketDataProvider) -> None:
        """Volume column must be int64 (yfinance sometimes returns float)."""
        raw = _make_ohlcv()
        raw["Volume"] = raw["Volume"].astype(float)
        with patch("yfinance.download", return_value=raw):
            df = mdp.fetch_ohlcv("AAPL", start="2024-01-01")
        assert df["volume"].dtype == "int64"


# ---------------------------------------------------------------------------
# MarketDataProvider — error handling
# ---------------------------------------------------------------------------


class TestMarketDataProviderErrors:
    def test_empty_response_raises_when_no_cache(self, mdp: MarketDataProvider) -> None:
        """Empty yfinance response with no cache should raise DataFetchError."""
        with patch("yfinance.download", return_value=pd.DataFrame()):
            with pytest.raises(DataFetchError):
                mdp.fetch_ohlcv("FAKE", start="2024-01-01")

    def test_exception_raises_when_no_cache(self, mdp: MarketDataProvider) -> None:
        """yfinance exception with no cache should raise DataFetchError."""
        with patch("yfinance.download", side_effect=RuntimeError("network error")):
            with pytest.raises(DataFetchError):
                mdp.fetch_ohlcv("FAKE", start="2024-01-01")

    def test_exception_returns_stale_cache(self, mdp: MarketDataProvider) -> None:
        """yfinance exception when cache exists should return the stale cache."""
        # First fetch populates the cache
        with patch("yfinance.download", return_value=_make_ohlcv()):
            df_original = mdp.fetch_ohlcv("AAPL", start="2024-01-01")

        # Second fetch fails — should fall back to cache
        with patch("yfinance.download", side_effect=RuntimeError("network error")):
            df_fallback = mdp.fetch_ohlcv("AAPL", start="2024-01-01", force_refresh=True)

        assert len(df_fallback) == len(df_original)


# ---------------------------------------------------------------------------
# MarketDataProvider — data quality validation
# ---------------------------------------------------------------------------


class TestMarketDataProviderValidation:
    def test_nan_values_are_forward_filled(self, mdp: MarketDataProvider) -> None:
        """NaN close values must be forward-filled before caching."""
        raw = _make_ohlcv(periods=5)
        raw.loc[raw.index[2], "Close"] = float("nan")
        with patch("yfinance.download", return_value=raw):
            df = mdp.fetch_ohlcv("AAPL", start="2024-01-01")
        assert not df["close"].isna().any()

    def test_price_gap_warning_logged(
        self, mdp: MarketDataProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A >20% single-day price move should emit a WARNING."""
        closes = [100.0, 100.0, 200.0, 200.0, 200.0]  # 100% jump on day 3
        raw = _make_ohlcv(close_prices=closes)
        with patch("yfinance.download", return_value=raw):
            with caplog.at_level("WARNING"):
                mdp.fetch_ohlcv("AAPL", start="2024-01-01")
        assert any(">20%" in msg for msg in caplog.messages)

    def test_zero_volume_warning_logged(
        self, mdp: MarketDataProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Zero-volume rows should emit a WARNING."""
        raw = _make_ohlcv(periods=5)
        raw.loc[raw.index[1], "Volume"] = 0
        with patch("yfinance.download", return_value=raw):
            with caplog.at_level("WARNING"):
                mdp.fetch_ohlcv("AAPL", start="2024-01-01")
        assert any("zero volume" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# MarketDataProvider — batch fetching
# ---------------------------------------------------------------------------


class TestMarketDataProviderBatch:
    def test_batch_sleeps_between_tickers(self, mdp: MarketDataProvider) -> None:
        """fetch_batch must sleep between tickers (not before first, not after last)."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        with patch("yfinance.download", return_value=_make_ohlcv()):
            with patch("time.sleep") as mock_sleep:
                mdp.fetch_batch(tickers, start="2024-01-01")
        # Sleep called N-1 times (between pairs, not before first or after last)
        assert mock_sleep.call_count == len(tickers) - 1

    def test_batch_skips_failed_tickers(self, mdp: MarketDataProvider) -> None:
        """A failing ticker in fetch_batch must be skipped, not crash the whole batch."""

        def side_effect(tickers, **kwargs):
            if tickers == "FAKE":
                return pd.DataFrame()
            return _make_ohlcv()

        with patch("yfinance.download", side_effect=side_effect):
            results = mdp.fetch_batch(["AAPL", "FAKE", "MSFT"], start="2024-01-01")

        assert "AAPL" in results
        assert "MSFT" in results
        assert "FAKE" not in results

    def test_batch_returns_correct_count(self, mdp: MarketDataProvider) -> None:
        """fetch_batch should return one entry per successful ticker."""
        tickers = ["AAPL", "MSFT"]
        with patch("yfinance.download", return_value=_make_ohlcv()):
            results = mdp.fetch_batch(tickers, start="2024-01-01")
        assert len(results) == len(tickers)


# ---------------------------------------------------------------------------
# MarketDataProvider — staleness check
# ---------------------------------------------------------------------------


class TestMarketDataProviderStaleness:
    def test_no_cache_is_stale(self, mdp: MarketDataProvider) -> None:
        """Missing cache file must report as stale."""
        assert mdp.is_cache_stale("NOTEXIST") is True

    def test_fresh_cache_not_stale(self, mdp: MarketDataProvider, tmp_path: Path) -> None:
        """Cache with yesterday's date as the last row is not stale on a weekday."""
        yesterday = date.today() - timedelta(days=1)
        # Ensure yesterday is a weekday; if Monday, use Friday
        while yesterday.weekday() >= 5:
            yesterday -= timedelta(days=1)

        raw = _make_ohlcv(start=str(yesterday), periods=1)
        with patch("yfinance.download", return_value=raw):
            mdp.fetch_ohlcv("AAPL", start=str(yesterday))

        # Simulate being on the next weekday
        next_weekday = date.today()
        while next_weekday.weekday() >= 5:
            next_weekday += timedelta(days=1)

        with patch("src.data.market_data.datetime") as mock_dt:
            mock_dt.now.return_value.date.return_value = next_weekday
            mock_dt.side_effect = lambda *a, **kw: __import__("datetime").datetime(*a, **kw)
            result = mdp.is_cache_stale("AAPL")
        assert result is False

    def test_old_cache_is_stale(self, mdp: MarketDataProvider) -> None:
        """Cache with a date 5 days ago must be stale when checked on a weekday."""
        old_date = date(2024, 1, 2)  # Tuesday
        raw = _make_ohlcv(start=str(old_date), periods=1)
        with patch("yfinance.download", return_value=raw):
            mdp.fetch_ohlcv("AAPL", start=str(old_date))

        # Simulate checking on a weekday (Friday 2024-01-12) — 10 days after cache
        a_weekday = date(2024, 1, 12)  # Friday
        with patch("src.data.market_data.datetime") as mock_dt:
            mock_dt.now.return_value.date.return_value = a_weekday
            result = mdp.is_cache_stale("AAPL")
        assert result is True


# ---------------------------------------------------------------------------
# MarketDataProvider — earnings calendar
# ---------------------------------------------------------------------------


class TestEarningsCalendar:
    def test_returns_list_type(self, mdp: MarketDataProvider) -> None:
        """get_earnings_calendar must always return a list."""
        mock_ticker = MagicMock()
        mock_ticker.calendar = None
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = mdp.get_earnings_calendar("AAPL")
        assert isinstance(result, list)

    def test_returns_empty_on_error(self, mdp: MarketDataProvider) -> None:
        """Exceptions from yfinance must be swallowed and return empty list."""
        with patch("yfinance.Ticker", side_effect=RuntimeError("yfinance error")):
            result = mdp.get_earnings_calendar("AAPL")
        assert result == []

    def test_parses_dict_calendar(self, mdp: MarketDataProvider) -> None:
        """A valid dict calendar with 'Earnings Date' key should parse correctly."""
        ts = pd.Timestamp("2024-04-30")
        mock_ticker = MagicMock()
        mock_ticker.calendar = {"Earnings Date": [ts]}
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = mdp.get_earnings_calendar("AAPL")
        assert len(result) == 1
        assert result[0] == date(2024, 4, 30)


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------


class TestFeatureStoreSaveLoad:
    def test_save_and_load_roundtrip(self, fs: FeatureStore) -> None:
        """Saved data must be loadable and equal to the original."""
        df = pd.DataFrame(
            {"tech_rsi_14": [50.0, 55.0, 60.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        df.index.name = "date"
        fs.save_features("AAPL", df)
        loaded = fs.load_features("AAPL")
        pd.testing.assert_frame_equal(loaded, df)

    def test_load_with_start_filter(self, fs: FeatureStore) -> None:
        """Loading with start= must exclude rows before that date."""
        dates = pd.date_range("2024-01-02", periods=10, freq="B")
        df = pd.DataFrame({"val": range(10)}, index=dates)
        df.index.name = "date"
        fs.save_features("AAPL", df)
        loaded = fs.load_features("AAPL", start="2024-01-09")
        assert loaded.index[0] >= pd.Timestamp("2024-01-09")

    def test_load_with_end_filter(self, fs: FeatureStore) -> None:
        """Loading with end= must exclude rows after that date."""
        dates = pd.date_range("2024-01-02", periods=10, freq="B")
        df = pd.DataFrame({"val": range(10)}, index=dates)
        df.index.name = "date"
        fs.save_features("AAPL", df)
        loaded = fs.load_features("AAPL", end="2024-01-05")
        assert loaded.index[-1] <= pd.Timestamp("2024-01-05")

    def test_load_returns_empty_when_missing(self, fs: FeatureStore) -> None:
        """load_features on an unknown ticker returns an empty DataFrame, not None."""
        result = fs.load_features("NOTEXIST")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_save_appends_new_rows(self, fs: FeatureStore) -> None:
        """Saving non-overlapping batches should accumulate all rows."""
        df1 = pd.DataFrame(
            {"val": [1, 2]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        df1.index.name = "date"
        df2 = pd.DataFrame(
            {"val": [3, 4]},
            index=pd.to_datetime(["2024-01-04", "2024-01-05"]),
        )
        df2.index.name = "date"
        fs.save_features("AAPL", df1)
        fs.save_features("AAPL", df2)
        loaded = fs.load_features("AAPL")
        assert len(loaded) == 4

    def test_save_overwrites_same_dates(self, fs: FeatureStore) -> None:
        """A second save with the same dates should replace the old values."""
        df1 = pd.DataFrame({"val": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
        df1.index.name = "date"
        df2 = pd.DataFrame({"val": [99.0]}, index=pd.to_datetime(["2024-01-02"]))
        df2.index.name = "date"
        fs.save_features("AAPL", df1)
        fs.save_features("AAPL", df2)
        loaded = fs.load_features("AAPL")
        assert len(loaded) == 1
        assert loaded.iloc[0]["val"] == 99.0


class TestFeatureStoreExists:
    def test_exists_false_for_new_ticker(self, fs: FeatureStore) -> None:
        assert fs.exists("NOTEXIST") is False

    def test_exists_true_after_save(self, fs: FeatureStore) -> None:
        df = pd.DataFrame({"val": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
        df.index.name = "date"
        fs.save_features("AAPL", df)
        assert fs.exists("AAPL") is True


class TestFeatureStoreGetLatest:
    def test_returns_none_when_missing(self, fs: FeatureStore) -> None:
        assert fs.get_latest("NOTEXIST") is None

    def test_maps_tech_prefix_to_technical(self, fs: FeatureStore) -> None:
        df = pd.DataFrame(
            {"tech_rsi_14": [50.0]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        df.index.name = "date"
        fs.save_features("AAPL", df)
        fv = fs.get_latest("AAPL")
        assert isinstance(fv, FeatureVector)
        assert fv.technical["rsi_14"] == 50.0

    def test_maps_sent_prefix_to_sentiment(self, fs: FeatureStore) -> None:
        df = pd.DataFrame(
            {"sent_score": [0.8]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        df.index.name = "date"
        fs.save_features("AAPL", df)
        fv = fs.get_latest("AAPL")
        assert fv.sentiment["score"] == 0.8  # type: ignore[index]

    def test_maps_deriv_prefix_to_derived(self, fs: FeatureStore) -> None:
        df = pd.DataFrame(
            {"deriv_close_vs_sma20": [0.05]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        df.index.name = "date"
        fs.save_features("AAPL", df)
        fv = fs.get_latest("AAPL")
        assert fv.derived["close_vs_sma20"] == 0.05  # type: ignore[index]

    def test_maps_regime_column(self, fs: FeatureStore) -> None:
        df = pd.DataFrame(
            {"regime": ["TRENDING_UP"]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        df.index.name = "date"
        fs.save_features("AAPL", df)
        fv = fs.get_latest("AAPL")
        assert fv.regime == RegimeType.TRENDING_UP  # type: ignore[union-attr]

    def test_ticker_field_is_set(self, fs: FeatureStore) -> None:
        df = pd.DataFrame({"val": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
        df.index.name = "date"
        fs.save_features("AAPL", df)
        fv = fs.get_latest("AAPL")
        assert fv.ticker == "AAPL"  # type: ignore[union-attr]


class TestFeatureStoreUpdateSentiment:
    def test_update_creates_sentiment_column(self, fs: FeatureStore) -> None:
        """update_sentiment on a file without sentiment_score should add the column."""
        df = pd.DataFrame(
            {"tech_rsi_14": [50.0, 55.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        df.index.name = "date"
        fs.save_features("AAPL", df)

        scores = {date(2024, 1, 2): 0.3, date(2024, 1, 3): 0.6}
        fs.update_sentiment("AAPL", scores)

        loaded = fs.load_features("AAPL")
        assert "sentiment_score" in loaded.columns

    def test_update_sentiment_values(self, fs: FeatureStore) -> None:
        """Sentiment scores must match what was passed to update_sentiment."""
        fs.save_features(
            "AAPL",
            pd.DataFrame(
                {"tech_rsi_14": [50.0]},
                index=pd.to_datetime(["2024-01-02"]),
            ),
        )
        scores = {date(2024, 1, 2): 0.75}
        fs.update_sentiment("AAPL", scores)
        loaded = fs.load_features("AAPL")
        assert abs(loaded.loc["2024-01-02", "sentiment_score"] - 0.75) < 1e-9

    def test_update_on_empty_store_creates_file(self, fs: FeatureStore) -> None:
        """update_sentiment should create a new Parquet when none exists."""
        scores = {date(2024, 1, 2): 0.5}
        fs.update_sentiment("NEWT", scores)
        assert fs.exists("NEWT")


class TestFeatureStoreListTickers:
    def test_empty_store(self, fs: FeatureStore) -> None:
        assert fs.list_tickers() == []

    def test_lists_saved_tickers(self, fs: FeatureStore) -> None:
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            df = pd.DataFrame({"val": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
            df.index.name = "date"
            fs.save_features(ticker, df)
        assert fs.list_tickers() == ["AAPL", "GOOGL", "MSFT"]


# ---------------------------------------------------------------------------
# NewsDataProvider — stub behaviour
# ---------------------------------------------------------------------------


class TestNewsDataProvider:
    def test_get_headlines_returns_empty_list(self) -> None:
        ndp = NewsDataProvider()
        assert ndp.get_headlines("AAPL") == []

    def test_get_headlines_days_back_param_accepted(self) -> None:
        ndp = NewsDataProvider()
        assert ndp.get_headlines("AAPL", days_back=30) == []

    def test_get_macro_news_returns_empty_list(self) -> None:
        ndp = NewsDataProvider()
        assert ndp.get_macro_news() == []

    def test_get_macro_news_days_back_param_accepted(self) -> None:
        ndp = NewsDataProvider()
        assert ndp.get_macro_news(days_back=14) == []

    def test_init_with_api_key(self) -> None:
        ndp = NewsDataProvider(api_key="test-key-123")
        assert ndp._api_key == "test-key-123"
