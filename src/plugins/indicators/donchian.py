"""Donchian Channel Indicator plugin."""

from typing import Any, Dict

import pandas as pd

from src.plugins.base import IndicatorPlugin, ParamSpec


class DonchianChannelIndicator(IndicatorPlugin):
    """
    Donchian Channel indicator measuring price position within N-period range.

    Upper channel = highest high over N periods (shifted 1 to avoid lookahead).
    Lower channel = lowest low over N periods (shifted 1 to avoid lookahead).
    Midline = (upper + lower) / 2.

    Position = (close - midline) / (upper - lower) * 2
    This gives a value in [-1, +1]: +1 = at upper channel, -1 = at lower channel.

    Output columns: ``dc_upper``, ``dc_lower``, ``dc_mid``, ``dc_position``.
    """

    name = "donchian"
    category = "trend"
    version = "1.0.0"

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Compute Donchian Channel and price position.

        Args:
            df: OHLCV DataFrame indexed by date.
            params: ``period`` (int, default 20).

        Returns:
            Copy of df with Donchian Channel columns added.
        """
        period = int(params.get("period", 20))
        df = df.copy()

        # Use shift(1) on the rolling window result to ensure we only see
        # data up to yesterday — the channel is defined by prior bars.
        df["dc_upper"] = df["high"].rolling(window=period, min_periods=period).max().shift(1)
        df["dc_lower"] = df["low"].rolling(window=period, min_periods=period).min().shift(1)
        df["dc_mid"] = (df["dc_upper"] + df["dc_lower"]) / 2.0

        channel_width = df["dc_upper"] - df["dc_lower"]
        # Avoid division by zero on flat price series
        df["dc_position"] = (
            (df["close"] - df["dc_mid"]) / channel_width.replace(0, float("nan")) * 2.0
        )
        return df

    def normalize(self, values: pd.Series) -> pd.Series:
        """Normalize Donchian channel position to [-1, +1].

        +1: price at the top of the channel (strong bullish momentum/breakout)
        -1: price at the bottom of the channel (strong bearish momentum/breakout)
         0: price at the midline (neutral)

        The dc_position is already in [-1, +1] by construction; just clip
        to handle rare cases where price exceeds the prior channel bounds.

        Args:
            values: The ``dc_position`` series.
        """
        return values.clip(-1.0, 1.0)

    def get_tunable_params(self) -> Dict[str, ParamSpec]:
        """Return Optuna search space for Donchian Channel parameters."""
        return {
            "period": ParamSpec(
                name="period",
                type="int",
                low=10,
                high=60,
                default=20,
                description="Lookback period for highest high / lowest low",
            ),
        }

    def get_default_params(self) -> Dict[str, Any]:
        return {"period": 20}
