# ADR-0011: Fundamentals as a First-Class Signal Pillar

## Status
Proposed

## Context

The stated goal of the system (see `README.md`, `CLAUDE.md`) is a **multi-model
positional trading tool** that combines **technicals, financials (fundamentals),
and news** to emit buy / sell / hold signals over a days-to-weeks horizon.

Up to v1.1 of the architecture document, only **technicals** (six
`IndicatorPlugin` instances) and **news sentiment** (FinBERT via
`DataEnricher`) were wired into the `QuantEngine` composite signal.
Fundamentals were listed in `§3.1` as an ingested data source (Alpha Vantage /
OpenBB) but did not feed any layer of the cascade. There was no
`FundamentalIndicatorPlugin` abstraction, no fundamentals-channel in the
composite score, and no fundamentals weight in
`config/cluster_params/cluster_default.yaml`. In effect, the third signal
pillar existed only on paper.

For a positional horizon (days to weeks), fundamentals-derived metrics —
valuation (P/E z-score, FCF yield), growth (YoY EPS), earnings surprises —
are meaningfully informative. Ignoring them leaves the system blind to a
signal category the goal statement explicitly calls out.

Two domain realities complicate a naive "add fundamentals with a global
weight" approach:

1. **Per-stock relevance varies.** Some equities (high-momentum tech, meme
   stocks, narrative-driven names) trade with a weak link to their
   fundamentals; others (dividend aristocrats, value names, defensive
   sectors) are strongly fundamentally anchored. A single global weight
   over-fits one cohort and under-fits the other.
2. **ETFs have no issuer-level fundamentals.** Pulling `OVERVIEW` / `EARNINGS`
   for a ticker like `ITA` or `4GLD.DE` returns no usable data. Forcing a
   non-zero fundamentals weight on an ETF would inject noise or an artefact
   of the forward-fill logic into the composite score.

## Decision

Promote fundamentals to a **first-class signal pillar** alongside technicals
and sentiment, with the following design:

1. **New plugin abstraction `FundamentalIndicatorPlugin`** in
   `src/plugins/base.py`, symmetric with `IndicatorPlugin` but taking both a
   fundamentals DataFrame and a price DataFrame. Initial implementations:
   `pe_zscore`, `fcf_yield`, `earnings_growth`, `earnings_surprise`.

2. **New composite channel `fundamental_score`** in the `QuantEngine`.
   Aggregated from the active `FundamentalIndicatorPlugin`s (mean of their
   normalised outputs), weighted per regime from
   `config/cluster_params/cluster_default.yaml`:
   - `TRENDING_UP: 0.15`
   - `TRENDING_DOWN: 0.10`
   - `RANGING: 0.20`
   - `VOLATILE: 0.05`

3. **Per-stock / per-cluster tunable weight.** The fundamentals weight is
   added to `BayesianTuner`'s search space (`[0.0, 0.30]`), so Optuna can
   discover the right weight per cluster / per promoted stock. A stock
   whose tuned weight converges near 0 is a fundamentally-decoupled name
   and is served by the tuner finding that automatically.

4. **Applicability multiplier `f_fund ∈ [0, 1]`** applied at runtime in the
   `QuantEngine`:
   - **ETFs** — tickers listed in `config/watchlist.yaml::etfs:` — force
     `f_fund = 0` unconditionally.
   - **Manual override** — `config/watchlist.yaml::stocks[*].fundamentals_weight_override`
     lets the operator set `f_fund = 0` (or any weight) for specific
     stocks without waiting for the tuner.
   - **Default** — `f_fund = 1` (tuned weight applies as-is).

5. **Weight re-normalisation.** When `f_fund = 0`, the remaining channel
   weights are re-normalised to sum to 1.0 so the composite score's
   dynamic range is preserved.

6. **Feature-store extensions.** A new `features/fundamentals/{ticker}/{fiscal_period}.parquet`
   partition; five new per-bar columns (`fund_pe_zscore`, `fund_fcf_yield`,
   `fund_earnings_growth`, `fund_earnings_surprise`, `fundamental_score`)
   in the existing `features/{ticker}/daily.parquet`. For ETFs, the
   `fund_*` columns are written as `NaN` and `fundamental_score` as `0.0`.

## Consequences

### Positive
- Closes the architecture-vs-goal gap: the system now uses all three
  signal pillars explicitly mentioned in the product goal.
- Tuner-driven per-cluster weighting addresses the "some stocks don't
  obey fundamentals" reality without hand-tuning per ticker.
- The ETF rule is robust and declarative (lives in `watchlist.yaml`), not
  scattered through code.
- `FundamentalIndicatorPlugin` is symmetric with `IndicatorPlugin`, so
  adding new fundamental signals (e.g., net-debt-to-EBITDA, ROIC) is a
  zero-core-code change — consistent with ADR-0005.

### Negative
- Adds a second daily data-fetch surface (Alpha Vantage fundamentals)
  with its own rate-limit and caching concerns (addressed in ADR-0013).
- Increases the search space for `BayesianTuner` by one continuous
  parameter per regime, mildly increasing overfitting risk. Mitigated by
  existing purged-CV / PBO checks (§12).
- The re-normalisation rule adds a small correctness surface: the
  implementation must ensure the sum of effective weights equals 1.0 in
  every regime and must be unit-tested for the ETF path.

### Trade-offs / Alternatives Considered
- **Alternative A: fundamentals as a `SignalFilter` rather than a channel.**
  Rejected because filters are binary veto gates; fundamentals are a
  graded signal that adds or subtracts confidence. A filter loses the
  nuance.
- **Alternative B: single global fundamentals weight, same for every stock.**
  Rejected because it violates the "per-stock adaptation" design
  principle and does not handle ETFs cleanly.
- **Alternative C: only use fundamentals in the ML meta-model as features.**
  Rejected because that hides the signal inside the precision filter and
  never influences the high-recall quant layer — defeating the purpose of
  making fundamentals a pillar of the composite signal.

### Follow-ups
- Implementation tracked in TASKS.md §3.2.
- Fundamentals provider rationale in ADR-0013.
