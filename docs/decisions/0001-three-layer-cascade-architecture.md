# ADR-0001: Three-Layer Cascade Architecture

## Status
Accepted

## Context

We need an architecture that combines classical quantitative analysis with modern AI/ML techniques to generate position trading signals. The naive approaches each have problems:

- **Pure quant strategies** are interpretable but produce many false positives in noisy markets.
- **Pure ML approaches** can overfit, are hard to debug, and don't leverage decades of well-understood quant theory.
- **Pure LLM approaches** are expensive (API costs), slow, and have unpredictable accuracy on numerical reasoning.
- **Naive ensembles** that average all three blur the signal — strong predictions get diluted by weak ones.

The user explicitly requested a system where the quant model runs first, with fallback to ML when confidence is low, and LLM context throughout.

## Decision

Adopt a **three-layer cascade architecture** with the following structure:

1. **Layer 1 — Sentiment context (always runs as features):** FinBERT processes news headlines in a batch job and produces sentiment scores for every ticker. These scores are added to the feature vector for both downstream layers.

2. **Layer 2 — Classical Quant Engine (primary signal generator):** A composite of normalized indicator signals (MA crossover, RSI, MACD, Bollinger, Donchian, volume) weighted by detected market regime. Produces a high-recall direction signal with confidence score. Cheap to run, interpretable.

3. **Layer 3 — ML Meta-Model (precision filter):** XGBoost trained via López de Prado's meta-labeling approach. Does NOT predict direction — instead, it decides whether to act on the quant signal and how much to bet. Improves precision by filtering false positives. Always runs after Layer 2.

4. **Layer 4 — LLM Validator (final gate):** OpenAI GPT-4o-mini analyzes recent news, earnings context, and macro risks. Only called when Layers 2+3 produce an actionable signal — minimizing API cost. Outputs APPROVE/VETO with reasoning.

## Consequences

**Positive:**
- Each layer has a clear, focused responsibility — easier to debug and improve independently.
- Quant layer's interpretability is preserved; meta-model adds ML benefits without replacing the quant logic.
- LLM cost is bounded: only ~5-15% of tickers generate signals on any given day, so API spend is minimized.
- Theoretically grounded in López de Prado's meta-labeling framework, which separates direction prediction from sizing.

**Negative:**
- More complex than a single-model approach — more moving parts to maintain.
- Requires careful interface design between layers to avoid coupling.
- Cascade ordering means a bug in an early layer affects everything downstream.

**Trade-offs accepted:**
- Slightly higher latency vs. parallel ensemble (acceptable for daily-bar position trading).
- Higher implementation complexity in exchange for better risk-adjusted returns and lower API costs.
