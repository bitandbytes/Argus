"""
Feature store: persists and retrieves computed feature DataFrames as Parquet files.

Storage layout:
    data/features/{ticker}/daily.parquet

Each file holds all historical features for one ticker. New rows are appended;
same-date rows from a newer save overwrite the older version.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.trade_signal import FeatureVector, RegimeType

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Stores and retrieves per-ticker feature DataFrames.

    Features are written as Parquet files with a ``DatetimeIndex`` named ``"date"``.
    Column naming conventions (used by ``get_latest`` to reconstruct
    :class:`~src.models.trade_signal.FeatureVector`):

    - ``tech_*``  → :attr:`FeatureVector.technical`
    - ``sent_*``  → :attr:`FeatureVector.sentiment`
    - ``deriv_*`` → :attr:`FeatureVector.derived`
    - ``regime``  → :attr:`FeatureVector.regime`

    Args:
        feature_dir: Root directory for feature Parquet files.
    """

    def __init__(self, feature_dir: str = "data/features") -> None:
        self._feature_dir = Path(feature_dir)
        self._feature_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_features(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Persist feature rows for *ticker*.

        If a Parquet file already exists for this ticker, the new rows are
        merged with the existing data. Rows sharing a date with the incoming
        DataFrame take the new values (upsert semantics). The file is always
        sorted by date before writing.

        Args:
            ticker: Ticker symbol used as the storage key.
            df: DataFrame with a ``DatetimeIndex`` named ``"date"``.
                Any columns are accepted — callers determine the schema.
        """
        df = self._ensure_datetime_index(df)
        path = self._path(ticker)

        if path.exists():
            existing = pd.read_parquet(path)
            existing = self._ensure_datetime_index(existing)
            # Drop existing rows for dates that appear in the new data (upsert)
            existing = existing[~existing.index.isin(df.index)]
            merged = pd.concat([existing, df])
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            merged = df

        merged = merged.sort_index()
        merged.to_parquet(path, index=True)
        logger.debug("Saved %d feature rows for %s → %s", len(merged), ticker, path)

    def load_features(
        self,
        ticker: str,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """
        Load stored features for *ticker*, optionally filtered by date range.

        Args:
            ticker: Ticker symbol.
            start: Inclusive start date. Accepts ``date`` objects or ``"YYYY-MM-DD"`` strings.
            end: Inclusive end date. Accepts ``date`` objects or ``"YYYY-MM-DD"`` strings.

        Returns:
            DataFrame with a ``DatetimeIndex`` named ``"date"``.
            Returns an **empty** DataFrame (not ``None``) if no data exists.
        """
        path = self._path(ticker)
        if not path.exists():
            logger.debug("No feature store found for %s", ticker)
            return pd.DataFrame()

        df = pd.read_parquet(path)
        df = self._ensure_datetime_index(df)

        if start is not None:
            df = df.loc[str(start) :]
        if end is not None:
            df = df.loc[: str(end)]

        return df

    def update_sentiment(self, ticker: str, scores: dict[date, float]) -> None:
        """
        Upsert ``sentiment_score`` values into the existing feature store.

        Allows FinBERT to write sentiment independently of indicator computation.
        If no feature file exists for *ticker*, a new one is created with only
        the sentiment column.

        Args:
            ticker: Ticker symbol.
            scores: Mapping of ``{date: sentiment_score}`` where sentiment_score
                    is in ``[-1.0, +1.0]``.
        """
        if not scores:
            return

        sentiment_df = pd.DataFrame(
            {"sentiment_score": list(scores.values())},
            index=pd.to_datetime(list(scores.keys())),
        )
        sentiment_df.index.name = "date"
        self.save_features(ticker, sentiment_df)
        logger.debug("Updated sentiment_score for %s (%d dates)", ticker, len(scores))

    def get_latest(self, ticker: str) -> Optional[FeatureVector]:
        """
        Return the most recent feature row as a :class:`~src.models.trade_signal.FeatureVector`.

        Column naming conventions:
        - Columns prefixed ``tech_`` are stripped and placed in ``.technical``.
        - Columns prefixed ``sent_`` are stripped and placed in ``.sentiment``.
        - Columns prefixed ``deriv_`` are stripped and placed in ``.derived``.
        - A column named ``regime`` is parsed as :class:`~src.models.trade_signal.RegimeType`.

        Args:
            ticker: Ticker symbol.

        Returns:
            :class:`FeatureVector` or ``None`` if no data exists.
        """
        df = self.load_features(ticker)
        if df.empty:
            return None

        row = df.iloc[-1]
        timestamp = row.name  # DatetimeIndex value
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        technical: dict[str, float] = {}
        sentiment: dict[str, float] = {}
        derived: dict[str, float] = {}
        regime: Optional[RegimeType] = None

        for col, val in row.items():
            if col == "regime":
                try:
                    regime = RegimeType(val)
                except (ValueError, TypeError):
                    pass
            elif str(col).startswith("tech_"):
                key = str(col)[5:]  # strip "tech_"
                try:
                    technical[key] = float(val)
                except (TypeError, ValueError):
                    pass
            elif str(col).startswith("sent_"):
                key = str(col)[5:]  # strip "sent_"
                try:
                    sentiment[key] = float(val)
                except (TypeError, ValueError):
                    pass
            elif str(col).startswith("deriv_"):
                key = str(col)[6:]  # strip "deriv_"
                try:
                    derived[key] = float(val)
                except (TypeError, ValueError):
                    pass

        return FeatureVector(
            ticker=ticker,
            timestamp=timestamp,
            technical=technical,
            sentiment=sentiment,
            derived=derived,
            regime=regime,
        )

    def exists(self, ticker: str) -> bool:
        """
        Return ``True`` if a non-empty feature store exists for *ticker*.

        Args:
            ticker: Ticker symbol.

        Returns:
            ``True`` if data is available, ``False`` otherwise.
        """
        path = self._path(ticker)
        if not path.exists():
            return False
        try:
            df = pd.read_parquet(path, columns=[])
            return len(df) > 0
        except Exception:
            return False

    def list_tickers(self) -> list[str]:
        """
        Return all tickers that have feature data stored.

        Returns:
            Sorted list of ticker symbols with at least one Parquet file.
        """
        return sorted(
            p.parent.name for p in self._feature_dir.glob("*/daily.parquet") if p.exists()
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _path(self, ticker: str) -> Path:
        return self._feature_dir / ticker / "daily.parquet"

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce the index to DatetimeIndex and name it 'date'."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df
