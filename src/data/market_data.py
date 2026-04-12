"""
Market data provider wrapping yfinance with local Parquet caching.

Rate-limiting note: yfinance scrapes Yahoo Finance. Aggressive request patterns
can trigger IP blocks. This module deliberately throttles batch requests with a
configurable inter-ticker delay (default 1.0 s). Slow is better than blocked.
"""

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Raised when OHLCV data cannot be fetched and no usable cache exists."""


class MarketDataProvider:
    """
    Fetches and caches daily OHLCV data from yfinance.

    All downloaded data is cached as Parquet files under ``cache_dir``.
    Subsequent calls return the cached version unless the cache is stale or
    ``force_refresh=True`` is passed.

    Args:
        cache_dir: Root directory for raw OHLCV Parquet cache.
                   Each ticker gets its own sub-directory: ``{cache_dir}/{ticker}/daily.parquet``.
        fetch_delay_sec: Seconds to sleep between ticker fetches in ``fetch_batch``.
                         Keeps request rate well below Yahoo Finance's informal limits.
                         Must be ≥ 0.5 — values below 0.5 are clamped up.
    """

    _MIN_DELAY_SEC: float = 0.5

    def __init__(self, cache_dir: str = "data/raw", fetch_delay_sec: float = 1.0) -> None:
        self._cache_dir = Path(cache_dir)
        self._fetch_delay_sec = max(fetch_delay_sec, self._MIN_DELAY_SEC)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        ticker: str,
        start: str | date,
        end: str | date | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return daily OHLCV for *ticker* between *start* and *end*.

        Columns returned: ``open, high, low, close, volume`` (all lowercase).
        Index: ``DatetimeIndex`` named ``"date"``.

        The result is always cached locally. On subsequent calls the cache is
        returned as long as it is not stale (see ``is_cache_stale``).

        Args:
            ticker: Yahoo Finance ticker symbol (e.g. ``"AAPL"``).
            start: Inclusive start date. Accepts ``"YYYY-MM-DD"`` strings or
                   :class:`datetime.date` objects.
            end: Exclusive end date. Defaults to today if ``None``.
            force_refresh: Skip cache check and always re-download.

        Returns:
            DataFrame with columns ``open, high, low, close, volume``.

        Raises:
            DataFetchError: If yfinance returns no data and no cache is available.
        """
        cache_path = self._cache_path(ticker)

        if not force_refresh and cache_path.exists() and not self.is_cache_stale(ticker):
            logger.debug("Cache hit for %s — loading from %s", ticker, cache_path)
            return self._read_cache(cache_path)

        logger.info("Fetching OHLCV for %s from yfinance", ticker)
        try:
            raw = self._download(ticker, start, end)
        except Exception as exc:
            logger.error("yfinance error for %s: %s", ticker, exc)
            if cache_path.exists():
                logger.warning("Returning stale cache for %s after fetch failure", ticker)
                return self._read_cache(cache_path)
            raise DataFetchError(f"Cannot fetch {ticker} and no cache available.") from exc

        if raw.empty:
            logger.error("yfinance returned empty DataFrame for %s", ticker)
            if cache_path.exists():
                logger.warning("Returning stale cache for %s (empty response)", ticker)
                return self._read_cache(cache_path)
            raise DataFetchError(f"yfinance returned no data for {ticker} and no cache exists.")

        df = self._normalise(raw)
        df = self._validate(df, ticker)
        self._write_cache(df, cache_path)
        return df

    def fetch_batch(
        self,
        tickers: list[str],
        start: str | date,
        end: str | date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for multiple tickers sequentially with throttling.

        Tickers are fetched one at a time with ``fetch_delay_sec`` sleep between
        each request. Failed tickers are skipped and logged — they are NOT
        included in the returned dict.

        Args:
            tickers: List of ticker symbols.
            start: Inclusive start date.
            end: Exclusive end date. Defaults to today if ``None``.

        Returns:
            Mapping of ticker → OHLCV DataFrame for all successfully fetched tickers.
        """
        results: dict[str, pd.DataFrame] = {}
        n = len(tickers)
        for i, ticker in enumerate(tickers, start=1):
            logger.info("Fetching %s (%d/%d)", ticker, i, n)
            try:
                results[ticker] = self.fetch_ohlcv(ticker, start=start, end=end)
            except DataFetchError as exc:
                logger.error("Skipping %s in batch fetch: %s", ticker, exc)
            if i < n:
                time.sleep(self._fetch_delay_sec)
        return results

    def get_earnings_calendar(self, ticker: str) -> list[date]:
        """
        Return upcoming earnings dates for *ticker*.

        Phase 1 stub: uses the ``yf.Ticker.calendar`` attribute which is
        unreliable and often empty. Phase 3 will replace this with a
        dedicated Alpha Vantage / Finnhub call.

        Args:
            ticker: Yahoo Finance ticker symbol.

        Returns:
            List of upcoming earnings :class:`datetime.date` objects.
            Returns ``[]`` on any error — callers must treat this as optional data.
        """
        try:
            cal = yf.Ticker(ticker).calendar
            if cal is None or (isinstance(cal, dict) and not cal):
                return []
            # cal is a dict like {'Earnings Date': [Timestamp(...)], ...}
            if isinstance(cal, dict):
                raw_dates = cal.get("Earnings Date", [])
                if isinstance(raw_dates, pd.Timestamp):
                    raw_dates = [raw_dates]
                return [pd.Timestamp(d).date() for d in raw_dates if d is not None]
            # Sometimes yfinance returns a DataFrame
            if isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"]
                    if hasattr(val, "__iter__"):
                        return [pd.Timestamp(v).date() for v in val if v is not None]
                    return [pd.Timestamp(val).date()]
            return []
        except Exception as exc:
            logger.warning("get_earnings_calendar failed for %s: %s", ticker, exc)
            return []

    def is_cache_stale(self, ticker: str) -> bool:
        """
        Return ``True`` if the cached data for *ticker* is outdated.

        Staleness definition:
        - On weekdays: stale if the last cached date is before yesterday.
        - On weekends (Sat/Sun): never stale (markets are closed, no new data).

        Args:
            ticker: Ticker symbol to check.

        Returns:
            ``True`` if the cache should be refreshed, ``False`` otherwise.
            If the cache file does not exist, returns ``True``.
        """
        cache_path = self._cache_path(ticker)
        if not cache_path.exists():
            return True

        today = datetime.now().date()
        weekday = today.weekday()  # 0=Mon … 6=Sun
        if weekday >= 5:  # Saturday or Sunday — markets closed
            return False

        yesterday = today - timedelta(days=1)
        # On Monday, yesterday was Sunday — look back to Friday
        if yesterday.weekday() >= 5:
            yesterday = yesterday - timedelta(days=yesterday.weekday() - 4)

        try:
            df = pd.read_parquet(cache_path, columns=["close"])
            if df.empty:
                return True
            last_date = df.index[-1]
            if hasattr(last_date, "date"):
                last_date = last_date.date()
            return last_date < yesterday
        except Exception as exc:
            logger.warning("Could not read cache to check staleness for %s: %s", ticker, exc)
            return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> Path:
        return self._cache_dir / ticker / "daily.parquet"

    def _read_cache(self, path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        df.index.name = "date"
        return df

    def _write_cache(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=True)
        logger.debug("Cache written to %s (%d rows)", path, len(df))

    def _download(self, ticker: str, start: str | date, end: str | date | None) -> pd.DataFrame:
        """Call yfinance.download for a single ticker; return raw DataFrame."""
        kwargs: dict = {
            "tickers": ticker,
            "start": str(start),
            "auto_adjust": True,
            "progress": False,
            "actions": False,
        }
        if end is not None:
            kwargs["end"] = str(end)
        return yf.download(**kwargs)

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise a raw yfinance DataFrame to a flat, lowercase-column format.

        yfinance sometimes returns a MultiIndex when ``group_by='ticker'`` is
        set (not the case here, but defensive handling is cheap).
        """
        # Handle MultiIndex columns (e.g. ('Close', 'AAPL') → 'close')
        if isinstance(df.columns, pd.MultiIndex):
            # Take the first level (price type), drop ticker level
            df.columns = [col[0] for col in df.columns]

        df.columns = [str(c).lower().strip() for c in df.columns]

        # Keep only the columns we care about
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()

        # Ensure volume is integer where possible (yfinance returns float sometimes)
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).astype("int64")

        # Ensure float64 for price columns
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype("float64")

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df

    def _validate(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Run data quality checks. All issues are logged as warnings — never raised.

        Checks performed:
        1. NaN values: forward-fill OHLC; flag volume NaNs.
        2. Large single-day price moves (> 20 %): may indicate splits or bad data.
        3. Gaps > 5 consecutive business days in the date range.
        4. Zero-volume rows.

        Args:
            df: Normalised OHLCV DataFrame.
            ticker: Ticker symbol (for log messages).

        Returns:
            The (possibly forward-filled) DataFrame.
        """
        # 1. NaN check
        nan_count = int(df.isnull().sum().sum())
        if nan_count > 0:
            logger.warning(
                "%s: %d NaN values found — forward-filling OHLC columns", ticker, nan_count
            )
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col].ffill()

        # 2. Large price gaps (possible splits or data errors)
        if "close" in df.columns and len(df) > 1:
            pct_change = df["close"].pct_change().abs()
            gap_dates = df.index[pct_change > 0.20].tolist()
            if gap_dates:
                logger.warning(
                    "%s: %d date(s) with >20%% single-day price move: %s",
                    ticker,
                    len(gap_dates),
                    [str(d.date()) if hasattr(d, "date") else str(d) for d in gap_dates[:5]],
                )

        # 3. Missing business-day gaps
        if len(df) >= 2:
            first_date = df.index[0]
            last_date = df.index[-1]
            expected = pd.bdate_range(start=first_date, end=last_date)
            missing = expected.difference(df.index)
            if len(missing) > 0:
                # Find consecutive runs and report only those > 5
                missing_series = pd.Series(missing)
                gaps = (missing_series.diff() > pd.Timedelta("2 days")).cumsum()
                run_lengths = missing_series.groupby(gaps).count()
                long_gaps = run_lengths[run_lengths > 5]
                if not long_gaps.empty:
                    logger.warning(
                        "%s: %d missing business day(s) including %d gap(s) longer than 5 days",
                        ticker,
                        len(missing),
                        len(long_gaps),
                    )

        # 4. Zero volume
        if "volume" in df.columns:
            zero_vol = int((df["volume"] == 0).sum())
            if zero_vol > 0:
                logger.warning(
                    "%s: %d row(s) with zero volume (may be legitimate for illiquid instruments)",
                    ticker,
                    zero_vol,
                )

        return df
