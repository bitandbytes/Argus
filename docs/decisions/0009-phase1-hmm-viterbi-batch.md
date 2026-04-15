# ADR-0009: Phase 1 Regime Detector Uses Batch Viterbi in `detect_series()`

## Status
Accepted

## Context

`RegimeDetector.detect_series()` is used during backtesting to assign a regime
label to every bar in a historical DataFrame. Internally it calls
`self._hmm.predict(features)` (Viterbi) and `self._hmm.predict_proba(features)`
(forward-backward), both of which operate over the **entire observation
sequence at once**. This means that the regime assigned to bar *t* is informed
by observations at bars *t+1, t+2, …, T* — a form of in-sample smoothing
(lookahead bias within the backtest window).

The HMM model *parameters* (transition matrix, emission means and covariances)
are fitted only on warmup data that strictly precedes the backtest start date
(see `_fit_regime()` in `scripts/run_backtest.py`). Only the regime *path
labels* within the backtest window are affected by the full-sequence inference.

For the quant signal generator, `QuantEngine.generate_series()` already uses
strict forward-only slicing (`df.iloc[:i+1]` per bar), so lookahead is
confined to the regime classification step feeding into those signals.

An online (causal) alternative exists: the HMM forward algorithm produces
`P(s_t | o_1, …, o_t)` without referencing future observations. `hmmlearn`
does not expose this directly but it can be approximated by calling
`predict_proba()` on an incrementally expanding window, or by implementing
the forward pass manually. This is standard practice in production HMM systems.

## Decision

Accept the full-sequence batch Viterbi approximation for Phase 1. Document
the limitation prominently in `detect_series()` docstrings (already done in
`src/signals/regime_detector.py:50–53`) and in this ADR. Schedule replacement
with a causal online filter as part of Phase 3 (before paper trading begins).

Rationale for accepting the approximation in Phase 1:

- Phase 1 is a research checkpoint, not a production-grade backtest.
- The HMM parameters are not lookahead-contaminated — only the state-path
  smoothing is affected.
- Full-sequence Viterbi produces smoother, more coherent regime boundaries
  that are easier to analyse visually and are less prone to single-bar flicker.
- The magnitude of the bias is expected to be small: HMM state transitions
  are slow by design (high self-transition probability), so the difference
  between causal and non-causal inference is mainly visible at regime change
  points.
- The validation checkpoint (Task 1.9) requires only that the Sharpe ratio
  is "sensible (not random)" — not production-quality. A slight optimistic
  bias at regime boundaries does not invalidate the prototype.

## Consequences

### Positive
- Stable, coherent regime labels without the rapid state flicker common in
  incremental forward filtering.
- Easier to inspect and reason about during exploratory research.
- No additional implementation complexity in Phase 1.

### Negative
- Phase 1 backtest Sharpe ratios are marginally optimistic due to smoother
  regime boundaries. The magnitude is unquantified but expected to be small.
- Code that worked correctly in Phase 1 will need to be refactored for Phase 3
  before any paper-trading metrics can be trusted.

### Trade-offs / Notes
- **Phase 3 action item:** replace `detect_series()` internals with an
  incremental forward pass. The planned approach is to call
  `self._hmm.predict_proba()` on an expanding `df.iloc[:i+1]` window per bar,
  caching intermediate results to avoid O(N²) recomputation.
- All callers of `detect_series()` (`run_backtest.py`, `validate_phase1.py`)
  must be updated when the online filter is introduced — no interface change
  is needed (same signature), only the internal implementation changes.
- `RegimeDetector.detect()` (single-bar, live use) is **not affected** by this
  ADR: it uses only the tail window of the DataFrame and is already causal.
