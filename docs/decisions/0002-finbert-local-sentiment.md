# ADR-0002: FinBERT for Local Sentiment Processing

## Status
Accepted

## Context

The pipeline needs sentiment analysis of financial news to enrich both the quant engine's composite signal and the ML meta-model's feature set. Options considered:

1. **Cloud LLM APIs** (GPT-4, Claude) for every news article — expensive at scale.
2. **Cloud sentiment APIs** (e.g., AlphaVantage NEWS_SENTIMENT) — pre-scored but limited rate.
3. **Local FinBERT** — open-source BERT-based model fine-tuned on financial text.
4. **Local FinGPT or Llama** — larger, more capable, but requires GPU.

For a watchlist of 10–100 tickers with potentially hundreds of headlines daily, running an LLM API on every headline would dominate operating costs. Sentiment is also a feature, not a final decision — high precision per-headline isn't critical; aggregate sentiment trends matter more.

## Decision

Use **FinBERT (ProsusAI/finbert)** as a local batch processor for all sentiment analysis. Run it on CPU as part of the daily pre-market data ingestion job. Outputs become features in the feature store.

Implement FinBERT as a `DataEnricher` plugin so it can be replaced with a more powerful model (e.g., self-hosted FinGPT in Phase 4) without changing pipeline code.

## Consequences

**Positive:**
- **Zero API costs** for sentiment — runs locally on any modern CPU.
- **Fast batch processing**: ~1500 headlines in ~60 seconds on an 8-core CPU.
- **Hardware-cheap**: 209 MB RAM in FP16, runs on any laptop. No GPU required.
- **Privacy**: News data never leaves the local machine.
- **Domain-specialized**: FinBERT is fine-tuned on financial text and outperforms general-purpose sentiment models on financial language.
- **Plugin abstraction** allows future swap to FinGPT/Llama with no pipeline changes.

**Negative:**
- FinBERT is older (2019) and less capable than modern LLMs on nuanced text.
- Cannot reason about complex multi-paragraph context (e.g., a full earnings transcript with caveats).
- Limited to sentence-level sentiment classification, not deep analysis.

**Mitigation:**
- For complex analysis (earnings calls, M&A announcements), the LLM Validator (Layer 4) provides deeper reasoning, but only when an actionable signal is being considered.
- This two-tier sentiment approach (cheap FinBERT for all, expensive LLM for actionable signals) reduces costs by ~90% compared to running an LLM on every ticker.
