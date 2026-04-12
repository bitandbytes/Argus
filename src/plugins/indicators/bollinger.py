"""Bollinger Band Indicator plugin."""

from typing import Any, Dict

import pandas as pd

from src.plugins.base import IndicatorPlugin, ParamSpec


class BollingerBandIndicator(IndicatorPlugin):
    """
    Bollinger Band indicator based on %B (percent bandwidth position).

    %B = (close - lower_band) / (upper_band - lower_band)

    %B = 1.0 means price is at the upper band (overbought).
    %B = 0.0 means price is at the lower band (oversold).
    %B = 0.5 means price is at the midline (neutral).

    Output columns: ``bb_upper``, ``bb_lower``, ``bb_mid``, ``bb_pct_b``,
                    ``bb_width``.
    """

    name = "bollinger"
    category = "volatility"
    version = "1.0.0"

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Compute Bollinger Bands and %B.

        Args:
            df: OHLCV DataFrame indexed by date.
            params: ``period`` (int), ``num_std`` (float).

        Returns:
            Copy of df with Bollinger Band columns added.
        """
        period = int(params.get("period", 20))
        num_std = float(params.get("num_std", 2.0))

        df = df.copy()
        rolling = df["close"].rolling(window=period, min_periods=period)
        df["bb_mid"] = rolling.mean()
        std = rolling.std(ddof=1)
        df["bb_upper"] = df["bb_mid"] + num_std * std
        df["bb_lower"] = df["bb_mid"] - num_std * std

        band_width = df["bb_upper"] - df["bb_lower"]
        # Avoid division by zero on flat price series
        df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / band_width.replace(0, float("nan"))
        df["bb_width"] = band_width / df["bb_mid"]
        return df

    def normalize(self, values: pd.Series) -> pd.Series:
        """Normalize %B to [-1, +1] using mean-reversion convention.

        +1: price at or below lower band (oversold — bullish entry signal)
        -1: price at or above upper band (overbought — bearish signal)
         0: price at midline (neutral)

        Formula: -(2 * %B - 1), clipped to [-1, +1].
        This inverts %B so that low %B (near lower band) = positive signal.

        Args:
            values: The ``bb_pct_b`` series.
        """
        return (-(2.0 * values - 1.0)).clip(-1.0, 1.0)

    def get_tunable_params(self) -> Dict[str, ParamSpec]:
        """Return Optuna search space for Bollinger Band parameters."""
        return {
            "period": ParamSpec(
                name="period",
                type="int",
                low=10,
                high=50,
                default=20,
                description="Rolling window period in days",
            ),
            "num_std": ParamSpec(
                name="num_std",
                type="float",
                low=1.0,
                high=3.0,
                default=2.0,
                description="Number of standard deviations for band width",
            ),
        }

    def get_default_params(self) -> Dict[str, Any]:
        return {"period": 20, "num_std": 2.0}
