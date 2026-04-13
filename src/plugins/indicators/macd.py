"""MACD (Moving Average Convergence Divergence) Indicator plugin."""

from typing import Any, Dict

import pandas as pd

from src.plugins.base import IndicatorPlugin, ParamSpec


class MACDIndicator(IndicatorPlugin):
    """
    MACD indicator using the histogram as the primary signal.

    MACD line = EMA(fast) - EMA(slow)
    Signal line = EMA(macd_line, signal_period)
    Histogram = MACD line - Signal line

    A positive and rising histogram is bullish; negative and falling is bearish.
    Normalization scales the histogram relative to recent price levels.

    Output columns: ``macd``, ``macd_signal``, ``macd_hist``.
    """

    name = "macd"
    category = "momentum"
    version = "1.0.0"
    output_column = "macd_hist"

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Compute MACD line, signal line, and histogram.

        Args:
            df: OHLCV DataFrame indexed by date.
            params: ``fast_period`` (int), ``slow_period`` (int),
                    ``signal_period`` (int).

        Returns:
            Copy of df with ``macd``, ``macd_signal``, ``macd_hist`` columns.
        """
        fast = int(params.get("fast_period", 12))
        slow = int(params.get("slow_period", 26))
        signal = int(params.get("signal_period", 9))

        df = df.copy()
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    def normalize(self, values: pd.Series) -> pd.Series:
        """Normalize MACD histogram to [-1, +1].

        +1: histogram is large and positive (strong bullish momentum)
        -1: histogram is large and negative (strong bearish momentum)
         0: histogram near zero (no momentum divergence)

        Uses a rolling 252-day standard deviation of the histogram to
        scale dynamically to price magnitude, then clips.

        Args:
            values: The ``macd_hist`` series.
        """
        rolling_std = values.rolling(window=252, min_periods=20).std()
        # Avoid division by zero on flat series
        rolling_std = rolling_std.replace(0, float("nan")).ffill().bfill()
        normalized = values / (2.0 * rolling_std)
        return normalized.clip(-1.0, 1.0)

    def get_tunable_params(self) -> Dict[str, ParamSpec]:
        """Return Optuna search space for MACD parameters."""
        return {
            "fast_period": ParamSpec(
                name="fast_period",
                type="int",
                low=5,
                high=19,
                default=12,
                description="Fast EMA period in days",
            ),
            "slow_period": ParamSpec(
                name="slow_period",
                type="int",
                low=20,
                high=50,
                default=26,
                description="Slow EMA period in days",
            ),
            "signal_period": ParamSpec(
                name="signal_period",
                type="int",
                low=5,
                high=15,
                default=9,
                description="Signal line EMA period in days",
            ),
        }

    def get_default_params(self) -> Dict[str, Any]:
        return {"fast_period": 12, "slow_period": 26, "signal_period": 9}
