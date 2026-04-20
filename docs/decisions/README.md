# Architecture Decision Records (ADRs)

This directory contains records of significant architectural decisions made during the design and development of this project. Each ADR documents the context, the decision, and the consequences.

## Format

We use a lightweight format based on Michael Nygard's template:

```
# ADR-NNNN: Title

## Status
Proposed | Accepted | Deprecated | Superseded by ADR-XXXX

## Context
What is the issue we're addressing? Why does it need a decision?

## Decision
What did we decide?

## Consequences
What becomes easier? What becomes harder? What are the trade-offs?
```

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-three-layer-cascade-architecture.md) | Three-layer cascade architecture | Accepted |
| [0002](0002-finbert-local-sentiment.md) | FinBERT for local sentiment processing | Accepted |
| [0003](0003-meta-labeling-over-ensemble.md) | Meta-labeling over ensemble approach | Accepted |
| [0004](0004-hybrid-cluster-individual-tuning.md) | Hybrid cluster + individual tuning | Accepted |
| [0005](0005-plugin-architecture.md) | Plugin architecture for extensibility | Accepted |
| [0006](0006-llm-provider-openai.md) | LLM provider: OpenAI GPT-4o-mini | Accepted |
| [0007](0007-package-manager-pip.md) | Package manager: pip + requirements.txt | Accepted |
| [0008](0008-data-provider-yfinance.md) | Data provider: yfinance for prototype | Accepted |
| [0009](0009-phase1-hmm-viterbi-batch.md) | Phase 1 regime detector uses batch Viterbi in `detect_series()` | Accepted |
| [0010](0010-phase1-sentiment-stub.md) | Phase 1 sentiment features stubbed to zero | Accepted |
| [0011](0011-fundamentals-as-signal-pillar.md) | Fundamentals as a first-class signal pillar | Proposed |
| [0012](0012-event-driven-exit-layer.md) | Event-driven exit layer alongside confidence-based exits | Proposed |
| [0013](0013-alpha-vantage-fundamentals-and-news-provider.md) | Alpha Vantage as fundamentals + news provider (Finnhub fallback) | Proposed |

## Adding a New ADR

1. Copy an existing ADR as a template.
2. Number it sequentially (next available).
3. Use a short, descriptive kebab-case title.
4. Status starts as "Proposed", changes to "Accepted" after review.
5. Add an entry to the index above.
