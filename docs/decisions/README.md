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

## Adding a New ADR

1. Copy an existing ADR as a template.
2. Number it sequentially (next available).
3. Use a short, descriptive kebab-case title.
4. Status starts as "Proposed", changes to "Accepted" after review.
5. Add an entry to the index above.
