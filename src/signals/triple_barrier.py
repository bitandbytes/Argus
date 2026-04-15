"""Triple-barrier labeling per López de Prado AFML Chapter 3."""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def triple_barrier_labels(
    prices: pd.Series,
    signal_times: pd.DatetimeIndex,
    signal_directions: pd.Series,
    tp_pct: float = 0.04,
    sl_pct: float = 0.02,
    max_holding_days: int = 20,
) -> pd.DataFrame:
    """
    Apply triple-barrier labeling to a series of quant signals.

    For each signal, three barriers are set:
    - Take-profit (TP): favorable price target (entry × (1 ± tp_pct))
    - Stop-loss (SL):   unfavorable price target (entry × (1 ∓ sl_pct))
    - Vertical:         maximum holding period of ``max_holding_days`` bars

    The first barrier hit determines the label:
    - TP hit first → ``label = 1``  (trade was profitable)
    - SL hit first → ``label = -1`` (trade was a loss)
    - Timeout (neither hit) → ``label = 0``

    The binary meta-label is: ``meta_label = 1 if label == 1 else 0``.
    This is the training target for the XGBoost meta-model (Task 2.2):
    "Did this quant signal lead to a profitable trade?"

    Args:
        prices: Daily close prices with a ``DatetimeIndex``.
        signal_times: Timestamps when quant signals fired. Must be a
            ``DatetimeIndex`` (subset of ``prices.index`` or partially
            overlapping; missing entries are silently skipped).
        signal_directions: +1 (long) or -1 (short) for each signal time,
            indexed by ``signal_times``.
        tp_pct: Take-profit distance as a fraction of entry price
            (default ``0.04`` = 4 %).
        sl_pct: Stop-loss distance as a fraction of entry price
            (default ``0.02`` = 2 %).
        max_holding_days: Vertical barrier — maximum number of bars to
            hold the position (default ``20``).

    Returns:
        DataFrame indexed by ``entry_time`` (``DatetimeIndex``) with columns:

        ============  ======  ================================================
        entry_price   float   Close price at signal bar.
        direction     float   +1 (long) or -1 (short).
        tp_price      float   Favorable target price (direction-relative).
        sl_price      float   Unfavorable target price (direction-relative).
        exit_time     object  Timestamp of exit (barrier hit or last future bar).
        exit_price    float   Close price at exit.
        label         int     +1 (TP hit), -1 (SL hit), 0 (timeout).
        meta_label    int     1 (correct / profitable), 0 (wrong / timeout).
        ============  ======  ================================================

        Signals whose entry time is absent from ``prices.index`` or whose
        direction is not in ``{+1.0, -1.0}`` are silently skipped (logged
        at WARNING level). Returns an empty DataFrame with correct columns
        when no valid signals are provided.
    """
    _COLUMNS: List[str] = [
        "entry_price", "direction", "tp_price", "sl_price",
        "exit_time", "exit_price", "label", "meta_label",
    ]

    if len(signal_times) == 0:
        return pd.DataFrame(columns=_COLUMNS)

    valid_times: List[pd.Timestamp] = []
    records: List[dict] = []
    prices_index = prices.index

    for sig_time in signal_times:
        if sig_time not in prices_index:
            logger.warning("Signal time %s not in price index — skipping.", sig_time)
            continue

        direction = float(signal_directions[sig_time])
        if direction not in (1.0, -1.0):
            logger.warning(
                "Invalid direction %.4g at %s — skipping (expected +1 or -1).",
                direction, sig_time,
            )
            continue

        entry_price = float(prices[sig_time])

        # Barriers are always expressed relative to the trade direction so
        # that tp_price is the *favorable* target and sl_price is the
        # *unfavorable* target, regardless of long vs. short.
        if direction == 1.0:  # long
            tp_price = entry_price * (1.0 + tp_pct)
            sl_price = entry_price * (1.0 - sl_pct)
        else:  # short
            tp_price = entry_price * (1.0 - tp_pct)
            sl_price = entry_price * (1.0 + sl_pct)

        # Slice forward prices using positional indexing to avoid label-based
        # slice ambiguity and handle signals near the end of the series.
        loc = prices_index.get_loc(sig_time)
        future_slice = prices.iloc[loc + 1 : loc + 1 + max_holding_days]

        label: int = 0
        if len(future_slice) > 0:
            exit_time = future_slice.index[-1]
            exit_price = float(future_slice.iloc[-1])
        else:
            exit_time = sig_time
            exit_price = entry_price

        for bar_time, bar_price in future_slice.items():
            bar_price_f = float(bar_price)
            if direction == 1.0:  # long: TP above, SL below
                if bar_price_f >= tp_price:
                    label = 1
                    exit_time = bar_time
                    exit_price = bar_price_f
                    break
                if bar_price_f <= sl_price:
                    label = -1
                    exit_time = bar_time
                    exit_price = bar_price_f
                    break
            else:  # short: TP below, SL above
                if bar_price_f <= tp_price:
                    label = 1
                    exit_time = bar_time
                    exit_price = bar_price_f
                    break
                if bar_price_f >= sl_price:
                    label = -1
                    exit_time = bar_time
                    exit_price = bar_price_f
                    break

        meta_label: int = 1 if label == 1 else 0

        valid_times.append(sig_time)
        records.append({
            "entry_price": entry_price,
            "direction": direction,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "label": label,
            "meta_label": meta_label,
        })

    if not records:
        return pd.DataFrame(columns=_COLUMNS)

    df = pd.DataFrame(records, index=pd.DatetimeIndex(valid_times))
    df.index.name = "entry_time"
    return df
