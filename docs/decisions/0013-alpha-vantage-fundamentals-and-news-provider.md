# ADR-0013: Alpha Vantage as Fundamentals + News Provider (Finnhub Fallback)

## Status
Proposed

## Context

ADR-0011 promotes fundamentals to a first-class signal pillar and ADR-0012
adds an earnings-blackout event filter; both require a concrete data
provider for fundamentals and earnings calendar. Separately, ADR-0010
stubs Phase 1 sentiment features to zero because no production news
provider was selected; Phase 3 now needs real headlines to activate the
already-built `FinBERTEnricher`.

We need a single decision that resolves three data-source questions at
once:
1. Which provider supplies **fundamental statements** (income statement,
   cash flow, balance sheet, company overview)?
2. Which provider supplies the **earnings calendar** (past and
   forthcoming announcement dates, consensus estimates, actuals)?
3. Which provider supplies **historical news headlines** with timestamps
   suitable for FinBERT sentiment scoring?

Available candidates:
- **Alpha Vantage** — free tier (25 req/day, 5 req/min) covers `OVERVIEW`,
  `INCOME_STATEMENT`, `CASH_FLOW`, `BALANCE_SHEET`, `EARNINGS`,
  `EARNINGS_CALENDAR`, `NEWS_SENTIMENT`. Covers all three needs from a
  single API key.
- **Finnhub** — free tier (60 req/min), broader news coverage (real-time
  + historical), weaker fundamentals coverage (limited history on free
  tier). Good as a backup.
- **FMP (Financial Modeling Prep)** — comprehensive fundamentals but
  tighter free-tier caps. Would be a third provider to maintain.
- **OpenBB** — aggregator that relies on downstream providers; adds a
  dependency surface without simplifying auth / rate-limits.
- **yfinance** — already in use for OHLCV; has `Ticker.financials`,
  `Ticker.cashflow`, `Ticker.info` but is unofficial and unreliable for
  production (per ADR-0008).

## Decision

Use **Alpha Vantage as the primary provider** for fundamentals, earnings
calendar, and news headlines. Use **Finnhub as a fallback** for news
coverage when Alpha Vantage rate-limits or misses a ticker. Use
**yfinance as a secondary fallback** for fundamentals only.

Concretely:

1. **`FundamentalsDataProvider`** (new, `src/data/fundamentals_data.py`)
   hits Alpha Vantage `OVERVIEW`, `INCOME_STATEMENT`, `CASH_FLOW`,
   `EARNINGS` with a monthly refresh cadence plus an on-demand refresh
   within `earnings_blackout_days + 1` of a scheduled earnings date.
   Falls back to `yfinance` `Ticker.financials` / `Ticker.cashflow` when
   Alpha Vantage returns a rate-limit error or empty response.

2. **`MarketDataProvider.get_earnings_calendar`** (upgrading the Phase 1
   stub) hits Alpha Vantage `EARNINGS_CALENDAR` daily. Cached in
   `features/events/earnings_calendar.parquet` with a 24-hour TTL.

3. **`NewsDataProvider`** (upgrading the Phase 1 stub in
   `src/data/news_data.py`) hits Alpha Vantage `NEWS_SENTIMENT` with a
   `(ticker, date)`-keyed on-disk cache. Falls back to Finnhub
   `/company-news` on rate-limit or empty-result errors. The FinBERT
   scoring layer is provider-agnostic — it only needs
   `List[{headline, ticker, timestamp}]`.

4. **Rate-limit strategy.** The free tier's 25 req/day is the binding
   constraint for the fundamentals + earnings path. Mitigations:
   - Monthly cache for fundamentals (each ticker fetched ~1×/month).
   - Daily cache for earnings calendar (1 req/day total, not per-ticker).
   - On-demand refresh for fundamentals bounded by the 25 req/day budget
     — prioritise tickers with an earnings announcement in the next 7
     days.
   - News headlines use Alpha Vantage's `NEWS_SENTIMENT` endpoint which
     is on a separate daily quota; Finnhub fallback covers the long tail.

5. **Secrets.** Both API keys live in env vars (`ALPHA_VANTAGE_API_KEY`,
   `FINNHUB_API_KEY`), referenced from `config/settings.yaml` under
   `news:`, `fundamentals:`. `.env.example` is updated accordingly.

6. **ETF handling.** `FundamentalsDataProvider.get_fundamentals(ticker)`
   returns an empty DataFrame for any ticker listed under
   `config/watchlist.yaml::etfs:` — no Alpha Vantage call is made, so no
   rate-limit budget is consumed for ETFs. The `QuantEngine` forces
   `f_fund = 0` regardless (per ADR-0011), so an empty DataFrame is a
   safe and cheap response.

## Consequences

### Positive
- One primary provider covers all three Phase 3 data needs, minimising
  credential management and SDK surface.
- Alpha Vantage `NEWS_SENTIMENT` includes pre-scored sentiment as a
  hint, but we still run FinBERT on the raw headlines for consistency
  with the existing Phase 1 sentiment pipeline. This means we can
  retroactively score *any* headline source without changing downstream
  code.
- Finnhub fallback limits the blast radius of a single-provider outage.
- The ETF short-circuit means ETFs never consume the free-tier
  fundamentals budget.

### Negative
- Free-tier limits (25 req/day fundamentals; 500 req/day news-sentiment)
  constrain the production watchlist size. For a 10-ticker watchlist
  with monthly fundamentals refresh, we consume ~10/30 ≈ 0.33 req/day
  fundamentals on average, leaving plenty of headroom. A 100-ticker
  watchlist would consume ~3.3 req/day — still well within limits. A
  500-ticker watchlist would be tight and may require the paid tier.
- Alpha Vantage data is reformatted per-endpoint; the provider must map
  Alpha Vantage's field names to a stable internal schema. Each
  `FundamentalIndicatorPlugin` expects the stable schema, not raw Alpha
  Vantage JSON.
- Coupling two pipelines (fundamentals, news) to one rate-limit bucket
  means a spike in one starves the other unless we split API keys.

### Trade-offs / Alternatives Considered
- **Alternative A: Finnhub for everything.** Rejected because Finnhub's
  fundamentals coverage on the free tier is weaker than Alpha Vantage's
  (shorter history, missing cash-flow fields on some tickers).
- **Alternative B: FMP for fundamentals, Finnhub for news.** Rejected
  because it adds a third credential and SDK surface for marginal
  benefit at the current watchlist size.
- **Alternative C: yfinance for fundamentals.** Considered only as the
  secondary fallback, not primary, because it is an unofficial scraper
  (same concerns as ADR-0008). Acceptable when Alpha Vantage is
  unavailable but not as the source of truth.

### Follow-ups
- Migration path: if the watchlist grows beyond the free-tier capacity,
  revisit this ADR and consider Alpha Vantage Premium (75 req/min,
  unlimited daily) or FMP Starter.
- Monitoring: track rate-limit-error rate in MLflow so we get an early
  signal before quotas are exhausted.
- Implementation tracked in TASKS.md §3.1 (news), §3.2 (fundamentals),
  §3.3 (earnings calendar).
