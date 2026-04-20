# ADR-0012: Event-Driven Exit Layer Alongside Confidence-Based Exits

## Status
Proposed

## Context

Up to v1.1 of the architecture, the only documented exit rule in the
`QuantEngine` was confidence-based:

```
exit if confidence < exit_confidence_threshold   # default 0.20
```

This works for slow deterioration of the composite signal but does not
react to **discrete events that invalidate a thesis within hours**:

- **Earnings releases.** The price move around an earnings print is
  bimodal and not predicted by the technical / sentiment / fundamental
  features the composite signal uses. Holding into an earnings print with
  an open position gambles on the direction of the gap.
- **News shocks.** Downgrades, guidance cuts, fraud allegations, or
  regulatory announcements can flip sentiment and price within a single
  bar. The FinBERT sentiment channel picks up the event but the
  confidence-based exit is too slow to act on it.
- **Hard price moves.** A position that has already given back 2×ATR
  should be closed for risk-management reasons independent of whether
  the composite confidence is still above the exit threshold.

`TradeSignal` has `stop_loss_pct` and `take_profit_pct` fields but no
layer in the existing code path populates them, and the backtest harness
relies exclusively on the confidence exit. There is no mechanism to
record *why* a position was closed, which makes it difficult to attribute
drawdowns to the right cause (signal deterioration vs. earnings gap vs.
news shock vs. ATR stop).

## Decision

Introduce a dedicated **event-driven exit layer** that runs in parallel
with the confidence-based exit. Structured as:

1. **New plugin abstraction `EventFilter`** in `src/plugins/base.py`:
   ```
   should_exit(position, bar, context) -> (bool, Optional[str])
   ```
   The second return value is the reason string for logging and alerts.

2. **Three initial filters** implemented under `src/risk/event_filter.py`:
   - `EarningsBlackoutFilter` — exits a position if an earnings
     announcement falls within `[T - days_before, T + days_after]`
     (default `[T-2, T+1]`). Consumes the earnings calendar produced by
     `MarketDataProvider.get_earnings_calendar()`.
   - `NewsShockFilter` — exits if the same-bar FinBERT score satisfies
     `|sentiment_score| > sentiment_threshold` (default 0.75) **and**
     `news_volume_ratio > volume_ratio_threshold` (default 3.0). The
     conjunction prevents spurious exits on low-volume chatter.
   - `AtrStopFilter` — exits if price crosses the ATR-multiple stop
     (default `2.0 × ATR_14` below entry for longs, above for shorts) or
     take-profit level (default `4.0 × ATR_14`).

3. **Integration point.** `RiskManager.evaluate_exits(open_positions, bar)`
   (Phase 4) iterates enabled `EventFilter` instances. The exit decision
   is the OR of all triggers; the **first** firing filter supplies the
   `exit_reason`.

4. **TradeSignal extension.** A new `exit_reason: Optional[str]` field
   is added to `TradeSignal` so the MLflow trade log records the cause
   of each exit ("confidence_below_threshold", "earnings_blackout",
   "news_shock", "atr_stop", "take_profit"). This enables post-hoc
   attribution and ablation.

5. **Configuration.** A new top-level `exits:` block in
   `config/settings.yaml` (see §10 of the architecture doc) with per-filter
   thresholds and an `enabled: bool` toggle per filter. Disabling a filter
   via config (no code change) makes ablation studies trivial.

## Consequences

### Positive
- Closes a real risk-management gap: positions now exit before
  known-risky events (earnings) and in response to thesis-breaking news.
- `exit_reason` attribution unlocks ablation studies — we can measure the
  marginal drawdown reduction of each filter independently.
- The `EventFilter` abstraction is symmetric with the existing plugin
  types (ADR-0005), so future event-driven rules (e.g., macro-event
  blackouts, sector-rotation triggers) require zero core-code changes.
- ATR-based stops finally populate the `TradeSignal.stop_loss_pct` /
  `take_profit_pct` fields that have been dormant since Phase 1.

### Negative
- More exits → more trades → higher transaction-cost drag. The
  `NewsShockFilter` in particular is a "trade on noise" risk if the
  thresholds are too loose. Mitigated by requiring the sentiment-and-
  volume conjunction and by unit-testing against historical shock events.
- Early exits around earnings mean forgoing positive earnings surprises.
  This is an intentional trade-off: the bimodal distribution is
  unfavourable on a risk-adjusted basis for a multi-day holding period.
  Post-Phase-4 we may revisit this for names with strong pre-earnings
  fundamental / sentiment alignment.
- Adds configuration surface (three filter blocks, three `enabled`
  toggles, six thresholds). Defaults are chosen conservatively.

### Trade-offs / Alternatives Considered
- **Alternative A: fold event logic into the confidence score itself**
  (e.g., multiply confidence by 0.5 near earnings). Rejected because it
  obscures the cause of the exit and makes attribution impossible.
- **Alternative B: bake ATR stops into PyBroker's native stop-loss
  mechanism only, with no event filters.** Rejected because it handles
  only the price path and ignores earnings / news.
- **Alternative C: let the LLM validator veto ongoing positions on every
  bar.** Rejected because per-bar LLM calls would explode API cost and
  latency; the LLM validator (Layer 4) is designed as an entry gate only.

### Follow-ups
- Implementation tracked in TASKS.md §3.3.
- `RiskManager` that actually invokes these filters is tracked in
  TASKS.md §4.3.
- The earnings-calendar dependency is tracked in TASKS.md §3.3 as well
  (upgrading the existing Phase-1 stub in `MarketDataProvider`).
