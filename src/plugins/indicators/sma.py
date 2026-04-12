"""SMA Crossover Indicator plugin."""

from typing import Any, Dict

import pandas as pd

from src.plugins.base import IndicatorPlugin, ParamSpec


class SMACrossoverIndicator(IndicatorPlugin):
    """
    Simple Moving Average crossover indicator.

    Measures the relative distance between a fast and slow SMA.
    A positive distance (fast > slow) is bullish; negative is bearish.

    Output column: ``sma_crossover`` — percentage gap between fast and slow SMA.
    """

    name = "sma_crossover"
    category = "trend"
    version = "1.0.0"

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Compute fast and slow SMAs and their percentage crossover gap.

        Args:
            df: OHLCV DataFrame indexed by date.
            params: Must contain ``fast_period`` and ``slow_period`` (ints).

        Returns:
            Copy of df with columns ``sma_fast``, ``sma_slow``, ``sma_crossover`` added.
        """
        fast = int(params.get("fast_period", 50))
        slow = int(params.get("slow_period", 200))

        df = df.copy()
        df["sma_fast"] = df["close"].rolling(window=fast, min_periods=fast).mean()
        df["sma_slow"] = df["close"].rolling(window=slow, min_periods=slow).mean()
        # Percentage gap: positive = fast above slow (bullish)
        df["sma_crossover"] = (df["sma_fast"] - df["sma_slow"]) / df["sma_slow"]
        return df

    def normalize(self, values: pd.Series) -> pd.Series:
        """Normalize the SMA crossover percentage gap to [-1, +1].

        +1: fast SMA is far above slow SMA (strong uptrend)
        -1: fast SMA is far below slow SMA (strong downtrend)
         0: fast and slow SMAs are equal (no trend)

        Scales by 0.05 (5% gap = full signal), then clips.
        """
        return (values / 0.05).clip(-1.0, 1.0)

    def get_tunable_params(self) -> Dict[str, ParamSpec]:
        """Return Optuna search space for SMA crossover parameters."""
        return {
            "fast_period": ParamSpec(
                name="fast_period",
                type="int",
                low=10,
                high=100,
                default=50,
                description="Fast SMA window in days",
            ),
            "slow_period": ParamSpec(
                name="slow_period",
                type="int",
                low=50,
                high=400,
                default=200,
                description="Slow SMA window in days",
            ),
        }

    def get_default_params(self) -> Dict[str, Any]:
        return {"fast_period": 50, "slow_period": 200}
