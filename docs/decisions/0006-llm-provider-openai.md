# ADR-0006: LLM Provider — OpenAI GPT-4o-mini

## Status
Accepted

## Context

The Layer 4 LLM Validator needs a provider for deep analysis of news, earnings context, and macro risks. Options considered:

1. **Anthropic Claude API** — High quality, good for nuanced reasoning. Requires separate API key (Claude.ai subscription does NOT include API access).
2. **OpenAI GPT-4o-mini** — Very cheap (~$0.15/1M input tokens), high quality, structured output support.
3. **OpenAI GPT-4o** — Higher quality but ~10× more expensive than mini.
4. **Local Ollama (Llama, Mistral)** — Free but requires GPU for reasonable speed.
5. **Perplexity Sonar API** — Web-search-augmented, useful for current macro context.

Cost analysis for a 100-stock watchlist with ~10% signal rate:
- ~10 LLM calls per day, ~1500 input tokens each = 15K tokens/day
- GPT-4o-mini: ~$0.002/day, ~$0.06/month
- GPT-4o: ~$0.05/day, ~$1.50/month
- Claude Sonnet: ~$0.05/day, ~$1.50/month

Even GPT-4o-mini is more than sufficient for the validator's task: reading a structured prompt with news context and outputting an APPROVE/VETO decision with reasoning. The validator doesn't need to do complex math or code generation.

## Decision

Use **OpenAI GPT-4o-mini** for the Layer 4 LLM Validator in development and production.

Implement the LLM Validator as a `SignalFilter` plugin so the provider can be swapped via config without code changes. Phase 4 includes a self-hosted FinGPT alternative for users with GPU access.

## Consequences

**Positive:**
- **Lowest cost**: ~$0.06/month for the development watchlist; remains affordable at scale.
- **High quality**: GPT-4o-mini is sufficient for binary APPROVE/VETO decisions with structured reasoning.
- **Mature SDK**: `openai` Python package is well-maintained with structured output support (JSON mode).
- **Caching available**: Per-ticker per-day caching reduces redundant calls.
- **No subscription confusion**: Pay-as-you-go API key, separate from any consumer products.

**Negative:**
- Requires obtaining an OpenAI API key and adding payment method (no free tier for production).
- API costs scale with watchlist size and signal rate (still very cheap, but non-zero).
- Vendor lock-in to OpenAI's API format (mitigated by plugin abstraction).

**Future migration path:**
- Phase 4 self-hosted LLM (FinGPT/Llama) eliminates API costs entirely for users with GPU access.
- Plugin architecture means switching providers is a config change.
