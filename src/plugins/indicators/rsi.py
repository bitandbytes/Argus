"""RSI (Relative Strength Index) Indicator plugin."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.plugins.base import IndicatorPlugin, ParamSpec


class RSIIndicator(IndicatorPlugin):
    """
    Relative Strength Index indicator.

    Uses Wilder's smoothing method (exponential moving average with
    alpha = 1/period). RSI ranges from 0 to 100; values above 70 are
    traditionally overbought and below 30 are oversold.

    Output column: ``rsi`` — raw RSI value in [0, 100].
    """

    name = "rsi"
    category = "momentum"
    version = "1.0.0"
    output_column = "rsi"

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Compute RSI using Wilder's smoothing.

        Args:
            df: OHLCV DataFrame indexed by date.
            params: Must contain ``period`` (int, default 14).

        Returns:
            Copy of df with ``rsi`` column added (values in [0, 100]).
        """
        period = int(params.get("period", 14))
        df = df.copy()

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's smoothing: first value is simple average, then EWM
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        # When avg_loss is 0, RS is infinite → RSI = 100.
        # When avg_gain is 0, RS = 0 → RSI = 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
        # Replace inf (no losses) with a large value that gives RSI → 100
        rs = rs.replace([np.inf, -np.inf], 1e9)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        return df

    def normalize(self, values: pd.Series) -> pd.Series:
        """Normalize RSI to [-1, +1].

        +1: RSI = 100 (extremely overbought — bearish reversal warning) ... but
            in a momentum regime, high RSI = continuation. We use a mean-reversion
            convention here (consistent with Bollinger): near-overbought is bearish.

        Convention used:
            RSI 70+ → toward -1 (overbought, fading signal)
            RSI 50  →  0 (neutral)
            RSI 30- → toward +1 (oversold, buying opportunity)

        Formula: -(rsi - 50) / 50, clipped to [-1, +1].
        This makes oversold (+RSI low) = positive (bullish) signal.
        """
        return (-(values - 50.0) / 50.0).clip(-1.0, 1.0)

    def get_tunable_params(self) -> Dict[str, ParamSpec]:
        """Return Optuna search space for RSI parameters."""
        return {
            "period": ParamSpec(
                name="period",
                type="int",
                low=5,
                high=30,
                default=14,
                description="RSI lookback period in days",
            ),
        }

    def get_default_params(self) -> Dict[str, Any]:
        return {"period": 14}
