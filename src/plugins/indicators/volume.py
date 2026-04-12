"""Volume Indicator plugin (OBV-based)."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.plugins.base import IndicatorPlugin, ParamSpec


class VolumeIndicator(IndicatorPlugin):
    """
    Volume indicator based on On-Balance Volume (OBV) momentum.

    OBV accumulates volume with the sign of each day's price change:
        OBV[t] = OBV[t-1] + volume[t]  if close[t] > close[t-1]
        OBV[t] = OBV[t-1] - volume[t]  if close[t] < close[t-1]
        OBV[t] = OBV[t-1]              if close[t] == close[t-1]

    The normalized signal compares the OBV's short EMA vs long EMA,
    expressed as a z-score of their percentage gap over a rolling window.
    Rising OBV (volume confirming price) = bullish; falling = bearish.

    Output columns: ``obv``, ``obv_ema_fast``, ``obv_ema_slow``, ``obv_signal``.
    """

    name = "volume"
    category = "volume"
    version = "1.0.0"

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Compute OBV and its fast/slow EMAs.

        Args:
            df: OHLCV DataFrame indexed by date.
            params: ``fast_period`` (int), ``slow_period`` (int).

        Returns:
            Copy of df with OBV-related columns added.
        """
        fast = int(params.get("fast_period", 10))
        slow = int(params.get("slow_period", 30))

        df = df.copy()
        direction = np.sign(df["close"].diff().fillna(0))
        df["obv"] = (direction * df["volume"]).cumsum()

        df["obv_ema_fast"] = df["obv"].ewm(span=fast, adjust=False).mean()
        df["obv_ema_slow"] = df["obv"].ewm(span=slow, adjust=False).mean()

        # Percentage gap between fast and slow OBV EMA
        # Positive = fast above slow (accumulation trend — bullish)
        slow_abs = df["obv_ema_slow"].abs().replace(0, float("nan"))
        df["obv_signal"] = (df["obv_ema_fast"] - df["obv_ema_slow"]) / slow_abs
        return df

    def normalize(self, values: pd.Series) -> pd.Series:
        """Normalize the OBV signal to [-1, +1].

        +1: strong accumulation (OBV fast EMA well above slow EMA — bullish)
        -1: strong distribution (OBV fast EMA well below slow EMA — bearish)
         0: OBV EMAs are equal (no volume trend)

        Scales the percentage gap by a rolling 252-day std, then clips.

        Args:
            values: The ``obv_signal`` series.
        """
        rolling_std = values.rolling(window=252, min_periods=20).std()
        rolling_std = rolling_std.replace(0, float("nan")).ffill().bfill()
        normalized = values / (2.0 * rolling_std)
        return normalized.clip(-1.0, 1.0)

    def get_tunable_params(self) -> Dict[str, ParamSpec]:
        """Return Optuna search space for Volume indicator parameters."""
        return {
            "fast_period": ParamSpec(
                name="fast_period",
                type="int",
                low=5,
                high=20,
                default=10,
                description="Fast OBV EMA period in days",
            ),
            "slow_period": ParamSpec(
                name="slow_period",
                type="int",
                low=20,
                high=60,
                default=30,
                description="Slow OBV EMA period in days",
            ),
        }

    def get_default_params(self) -> Dict[str, Any]:
        return {"fast_period": 10, "slow_period": 30}
