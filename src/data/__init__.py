"""Data layer: market data fetching, feature persistence, and news stubs."""

from src.data.feature_store import FeatureStore
from src.data.market_data import DataFetchError, MarketDataProvider
from src.data.news_data import NewsDataProvider

__all__ = [
    "MarketDataProvider",
    "DataFetchError",
    "FeatureStore",
    "NewsDataProvider",
]
