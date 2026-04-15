# ADR-0010: Phase 1 Sentiment Features Stubbed to Zero

## Status
Accepted

## Context

The architecture (ADR-0001, §4.1) calls for FinBERT sentiment scores to feed
into the `QuantEngine` composite signal as a weighted input. The weight is
configured at 10% in `config/cluster_params/cluster_default.yaml` across all
four market regimes.

The `FinBERTEnricher` plugin (`src/plugins/enrichers/finbert.py`) is fully
implemented: it batches headlines through ProsusAI/finbert (109M params,
runs locally on CPU), caches results per headline via SHA-256 key, and
computes rolling features (`sentiment_score`, `sentiment_ma_5d`,
`sentiment_ma_20d`, `sentiment_momentum`, `news_volume_ratio`,
`negative_news_ratio`). The plugin is registered and enabled in
`config/plugins.yaml`.

However, `NewsDataProvider.get_headlines()` and `get_macro_news()`
(in `src/data/news_data.py`) both return `[]` in Phase 1. With no headlines
to score, `FinBERTEnricher.enrich()` produces all-zero sentiment features.
The `QuantEngine.generate_signal()` and `generate_series()` accept a
`sentiment_score: float = 0.0` parameter; the backtest orchestrator passes
`0.0` for every bar throughout Phase 1.

Obtaining historical news with timestamps precisely aligned to OHLCV bars
requires a paid API (Alpha Vantage Premium, Finnhub, Bloomberg). The free
tier of Alpha Vantage provides recent headlines only, with no historical
backfill, making backtesting with real sentiment data impossible without a
significant cost or data sourcing effort.

## Decision

Accept zero-valued sentiment for all Phase 1 backtests and the Phase 1
validation checkpoint (Task 1.9). The `sentiment` weight in
`cluster_default.yaml` remains at its configured value (10%) but contributes
`0.0 × 0.10 = 0.0` to every composite score.

The effect is that the Phase 1 composite signal is driven entirely by the
six technical indicators (SMA crossover, RSI, MACD, Bollinger, Donchian,
volume). The 10% sentiment weight effectively dampens the technical indicator
weights proportionally — the six indicators share the remaining 90%.

In Phase 3, `NewsDataProvider` will be implemented with Alpha Vantage or
Finnhub. The FinBERT enricher and its downstream feature pipeline require
no changes; only the data provider needs to return real headlines.

## Consequences

### Positive
- Zero additional cost for Phase 1: no news API subscription required.
- Phase 1 backtests are fully reproducible and deterministic
  (sentiment input is always exactly 0.0).
- Full FinBERT pipeline is implemented, tested, and ready for live data;
  no rework needed in Phase 3 beyond wiring the news provider.
- Backtest results reflect the pure quant+regime signal, making it easier
  to attribute performance to specific indicator families.

### Negative
- The sentiment weight (10%) is permanently neutral in Phase 1, reducing
  effective indicator weights by ~11% (e.g., a 35% weight becomes 31.5%).
- If sentiment is a genuinely informative signal, Phase 1 underestimates
  the system's ceiling Sharpe ratio.
- The zero-sentiment assumption cannot be validated against real sentiment
  data until Phase 3; Phase 1 cannot measure the incremental value of
  sentiment features.

### Trade-offs / Notes
- Alternative considered: renormalize `cluster_default.yaml` to redistribute
  the 10% sentiment weight among the six technical indicators for Phase 1
  only. This was rejected because it would introduce a special-case YAML and
  a code branch that adds complexity without improving Phase 1 research quality.
- Alternative considered: use a constant non-zero sentiment proxy (e.g., 0.5
  for all bars). This was rejected as it would inject a non-representative
  bullish bias into all signals.
- **Phase 3 action item:** implement `NewsDataProvider.get_headlines()` with
  Alpha Vantage or Finnhub. Re-run backtests with real sentiment to measure
  the incremental signal quality of FinBERT features. No `QuantEngine` changes
  are required.
