# ADR-0008: Data Provider — yfinance for Prototype

## Status
Accepted

## Context

The pipeline needs historical OHLCV daily data for backtesting and live signal generation. Options considered:

1. **yfinance** — Free, scraper-based, supports `period="max"` for full history.
2. **Polygon.io** — Paid (~$30/month for stocks), reliable API, good documentation.
3. **Alpaca Market Data** — Free for paper trading users, but limited historical depth.
4. **EOD Historical Data** — Paid (~$20/month), good European stock coverage.
5. **Alpha Vantage** — Free tier with rate limits, paid tier for production.

For Phase 1 prototyping, we need:
- Decades of daily history for major US stocks (yfinance provides this)
- Free access (we're not yet generating revenue from the system)
- Python library with simple API

For European ETFs (4GLD.DE, DFNS.DE), yfinance support is less reliable and may need verification.

## Decision

Use **yfinance** as the primary data provider for Phase 1 (prototyping) and Phase 2 (tuning). Wrap it in a `MarketDataProvider` interface so the underlying provider can be swapped without changing pipeline code.

For Phase 3 (production paper trading), evaluate migrating to a paid provider (Polygon.io or Alpaca) for reliability.

## Consequences

**Positive:**
- **Free**: Zero cost for full historical data.
- **Decades of history**: `period="max"` returns all available data, often back to the 1980s for major US stocks.
- **Simple API**: One function call to fetch OHLCV with proper date alignment.
- **Sufficient for daily-bar backtesting**: No need for paid intraday data.
- **Wide ticker support**: All major US exchanges, plus international exchanges with suffix notation (.DE, .L, etc.).

**Negative:**
- **Unofficial scraper**: yfinance scrapes Yahoo Finance, not an official API. Yahoo can change their layout and break the library.
- **Rate limiting**: Aggressive scraping can get IPs blocked.
- **Data quality**: Occasional gaps, missing dividends, or incorrect splits — not as clean as paid providers.
- **European ETF support is limited**: Tickers like 4GLD.DE and DFNS.DE may have incomplete or no data.
- **Intraday limitations**: 1-minute bars only available for the last 7 days, sub-daily bars limited to 60 days.

**Mitigation:**
- Wrap yfinance in `MarketDataProvider` abstraction so swapping providers is a single class change.
- Cache fetched data locally in Parquet files to minimize API calls and survive yfinance outages.
- Validate data quality on fetch: check for NaN values, suspicious price gaps, missing dates.
- For European ETFs, fall back to LSE listings (`.L` suffix) if Xetra (`.DE`) fails.
- Plan migration to Polygon.io or Alpaca for production (Phase 3) to improve reliability.
