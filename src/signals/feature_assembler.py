"""Feature assembly for the ML meta-model (Task 2.2).

Converts a list of ``TradeSignal`` objects into a fixed-column DataFrame
suitable for training or running inference with ``MetaLabelModel``.
"""

from typing import List

import pandas as pd

from src.models.trade_signal import RegimeType, TradeSignal

# ---------------------------------------------------------------------------
# Fixed feature column layout — MUST NOT change once models are trained.
# ---------------------------------------------------------------------------

#: Ordered column list for the meta-model input matrix.
FEATURE_COLUMNS: List[str] = [
    # Quant indicator normalised scores ([-1, +1]), from QuantEngine._compute_scores()
    "sma_crossover",
    "macd",
    "rsi",
    "bollinger",
    "donchian",
    "volume",
    # Single sentiment score from FinBERT enricher (0.0 in Phase 1; wired in Phase 3)
    "sentiment",
    # Quant signal features — CRITICAL for meta-labeling.
    # These tell the meta-model "what did the quant engine predict?"
    "direction",    # composite score [-1, +1]
    "confidence",   # |direction| after MTF boost [0, 1]
    # Regime one-hot encoding (exactly one column equals 1.0 per row)
    "regime_trending_up",
    "regime_trending_down",
    "regime_ranging",
    "regime_volatile",
]

_REGIME_TO_COL = {
    RegimeType.TRENDING_UP: "regime_trending_up",
    RegimeType.TRENDING_DOWN: "regime_trending_down",
    RegimeType.RANGING: "regime_ranging",
    RegimeType.VOLATILE: "regime_volatile",
}


def build_feature_matrix(signals: List[TradeSignal]) -> pd.DataFrame:
    """
    Assemble the meta-model input matrix from a list of ``TradeSignal`` objects.

    Each signal contributes one row.  Missing indicator scores default to
    ``0.0`` so that neutral / early-warmup bars do not raise errors.

    Args:
        signals: List of :class:`~src.models.trade_signal.TradeSignal` objects
            produced by ``QuantEngine.generate_signal()`` or
            ``QuantEngine.generate_series()``.

    Returns:
        :class:`pandas.DataFrame` with :data:`FEATURE_COLUMNS` columns,
        indexed by signal timestamp (``index.name = "timestamp"``).
        Returns an empty DataFrame with correct columns when *signals* is empty.

    Example::

        signals = quant_engine.generate_series(df, regime_series, "AAPL")
        X = build_feature_matrix(list(signals))
        y = labels["meta_label"]           # from triple_barrier_labels()
        model.train(X, y)
    """
    if not signals:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    rows = []
    index = []

    for sig in signals:
        feat = sig.features  # dict populated by QuantEngine._compute_scores()

        # Regime one-hot: zero-initialise all four columns, then flip the right one.
        regime_vals = {col: 0.0 for col in _REGIME_TO_COL.values()}
        active_col = _REGIME_TO_COL.get(sig.regime)
        if active_col:
            regime_vals[active_col] = 1.0

        rows.append({
            "sma_crossover": float(feat.get("sma_crossover", 0.0)),
            "macd":           float(feat.get("macd",          0.0)),
            "rsi":            float(feat.get("rsi",           0.0)),
            "bollinger":      float(feat.get("bollinger",     0.0)),
            "donchian":       float(feat.get("donchian",      0.0)),
            "volume":         float(feat.get("volume",        0.0)),
            "sentiment":      float(feat.get("sentiment",     0.0)),
            "direction":      float(sig.direction),
            "confidence":     float(sig.confidence),
            **regime_vals,
        })
        index.append(sig.timestamp)

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(index))
    df.index.name = "timestamp"
    return df[FEATURE_COLUMNS]  # enforce fixed column order
