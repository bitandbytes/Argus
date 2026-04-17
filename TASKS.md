# TASKS.md

Development task list organized by phase. Check off tasks as they are completed. Each task should be small enough to complete in a single session.

**Convention:**
- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[!]` Blocked (note reason)

---

## Phase 1 — Research & Prototyping (Weeks 1–6)

**Goal:** Working end-to-end prototype with quant signal generation, sentiment integration, and basic backtesting. No ML yet, no live trading.

### 1.1 Project Setup

- [x] Create Python virtual environment (`python -m venv .venv`)
- [x] Install base dependencies (`pip install -r requirements.txt`)
  - Note: pandas-ta pins numba==0.61.2; on Python 3.12 this is fine — do NOT add `numba>=0.65.0` to requirements.txt or pip resolution fails.
  - Note: torch>=2.6 fails on Windows 10 (WinError 1114, c10.dll init). Fix: install CPU wheel `pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/cpu` after the regular install. Documented in requirements.txt.
- [x] Verify FinBERT model downloads correctly via HuggingFace transformers
  - Model cached (109M params, labels: positive/negative/neutral). Sanity-check inference confirmed correct sentiment.
  - Fix applied: `use_safetensors=True` in `download_finbert.py` — transformers >=5.x blocks `torch.load` on torch <2.6 (CVE-2025-32434); safetensors bypasses this.
  - Fix applied: removed stale `hf.cached_download` reference in `verify_setup.py` — removed in huggingface_hub 1.x.
- [ ] Create `.env` from `.env.example` and add OpenAI API key
- [x] Set up MLflow locally (`mlflow ui` should serve on `localhost:5000`) — DB initialized at `data/mlflow.db`
- [x] Configure git hooks for black/ruff — `.pre-commit-config.yaml` created; run `pre-commit install`
- [x] Verify yfinance data fetch works for all tickers in `config/watchlist.yaml`
  - All 8 US stocks: PASS. 4GLD.DE (gold ETF): PASS. DFNS.DE (defense ETF): no data — replaced with `ITA` (iShares Aerospace & Defense ETF, US-listed, history from 2006).
- Note: VS Code shows "package not installed" hints — select `.venv` as the Python interpreter (Ctrl+Shift+P → "Python: Select Interpreter" → `.venv`).

### 1.2 Data Layer

- [x] Implement `MarketDataProvider` class wrapping yfinance with caching
  - `src/data/market_data.py` — fetch_ohlcv, fetch_batch (1.5s throttle), is_cache_stale
  - Cache: `data/raw/{ticker}/daily.parquet`; stale only on weekdays when last date < yesterday
  - Error handling: falls back to stale cache on yfinance failure; raises DataFetchError only when no cache exists
- [x] Implement `FeatureStore` using Parquet files in `data/features/`
  - `src/data/feature_store.py` — save_features (upsert), load_features (date filtering), get_latest, update_sentiment, exists, list_tickers
  - Column prefix conventions: `tech_`, `sent_`, `deriv_` map to FeatureVector fields
- [x] Add `get_earnings_calendar()` method to MarketDataProvider
  - Phase 1 stub: uses `yf.Ticker.calendar`; returns `[]` on any error
- [x] Create `NewsDataProvider` stub (Phase 1 may skip live news; use Alpha Vantage free tier later)
  - `src/data/news_data.py` — get_headlines and get_macro_news both return `[]`; interface ready for Alpha Vantage / Finnhub in Phase 3
- [x] Write unit tests for data providers (mock the yfinance calls)
  - `tests/test_data_providers.py` — 46 tests, all passing; yfinance fully mocked, file I/O uses tmp_path
- [x] Validate data quality: check for NaN values, missing dates, suspicious price gaps
  - MarketDataProvider._validate(): NaN forward-fill + warning, >20% price gap warning, missing business-day gap warning (>5 days), zero-volume warning

### 1.3 Plugin Foundation

- [x] Implement abstract base classes in `src/plugins/base.py`:
  - `IndicatorPlugin`
  - `SmoothingPlugin`
  - `DataEnricher`
  - `SignalFilter`
  - Supporting types: `SmoothResult`, `ParamSpec`
- [x] Implement `PluginRegistry` in `src/plugins/registry.py` with discovery logic
  - `discover_plugins(config_path)`, getters, `list_available()`, `get_filters_by_stage()`
- [x] Load plugin config from `config/plugins.yaml`
  - 6 indicators + finbert enricher defined; Kalman/LLM validator commented for Phase 4/3
- [x] Write tests for plugin registry: register, retrieve, list, error on missing
  - `tests/test_plugin_registry.py` — 14 tests; stubs defined inline; monkeypatched `_instantiate` for discover tests
- [x] Document the plugin contract in `.claude/skills/plugin-author/SKILL.md` (already drafted)

### 1.4 Indicator Plugins (each as separate plugin)

- [x] Implement `SMACrossoverIndicator` with `compute()` and `normalize()` methods
- [x] Implement `RSIIndicator`
- [x] Implement `MACDIndicator`
- [x] Implement `BollingerBandIndicator`
- [x] Implement `DonchianChannelIndicator`
- [x] Implement `VolumeIndicator`
- [x] Unit tests for each indicator with known input/output pairs
- [x] Verify each indicator's `get_tunable_params()` returns valid ParamSpec entries

### 1.5 FinBERT Sentiment Plugin

- [x] Implement `FinBERTEnricher` as a `DataEnricher` plugin
  - `src/plugins/enrichers/finbert.py` — enrich, batch_enrich, analyze_batch, analyze_batch_cached
  - Zero-arg instantiation for registry compatibility; `news_provider` injectable for testing
- [x] Cache model on first load to avoid re-downloading
  - HuggingFace auto-caches to `~/.cache/huggingface/` on first `from_pretrained` call
  - Per-headline inference cache in `data/models/finbert_cache/` (SHA256-keyed JSON files)
- [x] Implement batch processing for efficiency (process 64 headlines at a time)
  - `analyze_batch()` processes in chunks of `batch_size=64`; `max_length=128` truncation
- [x] Compute rolling features: `sentiment_ma_5d`, `sentiment_ma_20d`, `sentiment_momentum`
  - Also computes: `sentiment_score` (latest daily), `news_volume_ratio`, `negative_news_ratio`
  - Confidence-weighted aggregation per day before rolling windows
- [x] Test on sample headlines to verify sentiment scores are reasonable
  - `tests/plugins/enrichers/test_finbert_enricher.py` — 4 live inference tests in `TestLiveSampleInference`
  - Skipped automatically if model not cached; run `scripts/download_finbert.py` to enable
- [x] Decide on news source: skip historical news for Phase 1 backtests; use it live in Phase 3
  - Phase 1: `NewsDataProvider` stub returns `[]` → all sentiment features are 0.0 (intentional)
  - Phase 3: implement `NewsDataProvider` with Alpha Vantage / Finnhub; no changes to enricher needed
- [x] See `.claude/skills/finbert-integration/SKILL.md` for implementation guidance

### 1.6 Regime Detector

- [x] Implement `RegimeDetector` class with HMM (`hmmlearn`) + ADX dual approach
  - `src/signals/regime_detector.py` — `fit()`, `detect()`, `detect_series()`, `save()`, `load()`
  - Layer 1 of the cascade; not a plugin — instantiated directly before QuantEngine
- [x] Train HMM on log returns + realized volatility, n_components=3
  - `GaussianHMM(n_components=3, covariance_type="full")` trained on `[log_return_1d, realized_vol_20d]`
  - Uses last `lookback_days=504` rows; raises `ValueError` if insufficient data
- [x] Implement `detect()` returning `RegimeType` enum
  - Uses HMM forward-backward posteriors; returns `RegimeType` for the most recent row
  - Also implements `detect_series(df)` → `pd.Series[RegimeType]` for bulk/backtest use
- [x] Add ADX as the fast classifier (ADX > 25 = trending, ADX < 20 = ranging)
  - `_compute_adx()` uses `pandas_ta.adx(length=14)` — robust to column-name variation
- [x] Implement reconciliation logic combining HMM and ADX outputs
  - Priority: VOLATILE (uncertainty > 0.40) → RANGING (ADX < 20) → HMM direction (bull/bear/sideways)
  - All 5 reconciliation branches unit-tested directly in `TestReconciliation`
- [x] Persist trained HMM models in `data/models/hmm/` per cluster (or per stock if individual)
  - `fit()` auto-saves to `data/models/hmm/{ticker}.pkl` via joblib
  - `save(path)` / `classmethod load(path)` for explicit persistence
- [x] Test regime stability: regimes should not flicker daily
  - `TestRegimeStability::test_no_single_day_flicker_in_trending_series` verifies zero isolated flips
  - HMM transition matrix naturally penalises rapid state changes

### 1.7 Quant Engine

- [x] Implement `QuantEngine` class that uses plugin registry
  - `src/signals/quant_engine.py` — `generate_signal()`, `generate_series()`, `should_exit()`
  - Plugin-agnostic: iterates `IndicatorPlugin` instances from registry; never hardcodes names
- [x] Load active indicators from registry on init
  - `registry.get_all_indicators()` called in `__init__`; params loaded via `get_default_params()`
- [x] Apply regime-specific weights (initial defaults from architecture doc)
  - `config/cluster_params/cluster_default.yaml` — 4 regimes × 7 keys (6 indicators + sentiment)
  - Weights validated/normalized to sum to 1.0 on load; missing regimes filled with equal spread
- [x] Compute composite score = Σ(weight × normalized_score)
  - `_weighted_composite()` — sums weight × normalized score, clips to [-1, +1]
  - Added `output_column: str` class attribute to `IndicatorPlugin` base + all 6 plugins
- [x] Implement multi-timeframe confirmation (daily + weekly alignment check)
  - `_weekly_confirms()` — weekly SMA(4) vs SMA(10) via `resample("W-FRI")`; requires ≥20 weekly bars
  - `multi_timeframe_boost=1.15` from `config/settings.yaml`; confidence never exceeds 1.0
- [x] Generate `TradeSignal` dataclass with direction, confidence, regime
  - All required fields populated; `stop_loss_pct`, `take_profit_pct`, `bet_size`, `llm_approved` left None
- [x] Wire FinBERT sentiment as one of the inputs to the composite
  - `sentiment_score: float = 0.0` parameter on `generate_signal()`; weight from cluster_default.yaml
  - Phase 1: always 0.0 (stub). Phase 3: orchestrator passes real FinBERT score
- [x] See `.claude/skills/quant-engine-dev/SKILL.md` for design patterns

### 1.8 Initial Backtesting

- [x] Install PyBroker and verify import works
  - PyBroker 1.2.11 confirmed installed; `--verify` flag in run_backtest.py prints version and exits
- [x] Write `scripts/run_backtest.py` that runs the quant engine on historical data
  - Pre-computes signals via `generate_series()` (forward-only, no lookahead), feeds into PyBroker
  - HMM fitted on warmup window (default: start - 730 days) before backtest begins
- [x] Configure realistic transaction costs (0.05% commission, 0.05% slippage)
  - `FeeMode.ORDER_PERCENT` with `fee_amount=0.001` (0.1% per order, baked commission + slippage)
  - `buy_delay=1, sell_delay=1` — fills on next bar's open for realistic execution
- [x] Run backtest on AAPL 2020-2024, compute Sharpe, max drawdown, win rate
  - `python scripts/run_backtest.py --ticker AAPL --start 2020-01-01 --end 2024-12-31`
  - Metrics: sharpe, max_drawdown_pct, win_rate, total_return_pct, profit_factor, calmar, sortino
- [x] Visualize equity curve in a notebook
  - `notebooks/backtest_analysis.ipynb` — dual-panel equity + drawdown, monthly heatmap, trade P&L
- [x] Compare results against buy-and-hold benchmark
  - `_compute_benchmark()` computes B&H Sharpe, return, max drawdown; printed side-by-side
- [x] Log all backtest runs to MLflow
  - `sqlite:///data/mlflow.db`, experiment `argus_backtests`; params + metrics + CSV artifacts logged
- [ ] See `.claude/skills/backtest-runner/SKILL.md` for backtesting workflow

### 1.9 Phase 1 Validation Checkpoint

- [x] All tests pass (`pytest tests/`)
- [x] Backtest produces sensible (not random) results — Sharpe should be in `[-1.0, +2.0]` range
- [x] No lookahead bias detected (manual code review)
- [x] Pipeline can run end-to-end on at least one ticker
- [x] Document any deviations from the architecture doc as new ADRs
- [x] Demo run reviewed before proceeding to Phase 2

---

## Phase 2 — Meta-Labeling & Tuning (Weeks 7–12)

**Goal:** ML meta-model filters quant signals; per-stock tuning is automated; walk-forward validation prevents overfitting.

### 2.1 Triple-Barrier Labeling

- [x] Implement triple-barrier labeling per López de Prado AFML Chapter 3
- [x] Function signature: `triple_barrier_labels(prices, signals, tp_pct, sl_pct, max_holding_days)`
- [x] Returns labels: `+1` (TP hit), `-1` (SL hit), `0` (timeout)
- [x] Test on synthetic data with known outcomes
- [x] See `.claude/skills/ml-meta-labeler/SKILL.md` for full workflow

### 2.2 ML Meta-Model

- [x] Implement `MetaLabelModel` class using XGBoost
- [x] Build feature assembly: quant features + sentiment + regime (one-hot) + quant prediction
- [x] Train binary classifier: did the quant signal lead to a profitable trade?
- [x] Apply Platt scaling via `CalibratedClassifierCV`
- [x] Compute calibration curve and Brier score
- [x] Save trained model to `data/models/meta_model/` with versioning
- [x] Log training run to MLflow with all hyperparameters

### 2.3 Purged Cross-Validation

- [x] Implement `PurgedKFoldCV` per AFML Chapter 7 (or use `timeseriescv` library)
- [x] Embargo gap = max signal dependency horizon (e.g., 5 days for 5-day forward returns)
- [x] Use this CV split for all meta-model training
- [x] Verify it produces non-overlapping folds with proper purging

### 2.4 Stock Clustering

- [x] Implement `StockClusterer` class with K-Means and tslearn DTW options
- [x] Feature extraction per stock: Hurst exponent, mean ADX, lag-1 autocorr, volatility
- [x] Choose `k` via silhouette score (target k=4..8)
- [x] Persist cluster assignments to `config/cluster_assignments.yaml`
- [x] Test re-clustering: same input data should produce stable assignments

### 2.5 Bayesian Tuning with Optuna

- [x] Implement `BayesianTuner` wrapping Optuna
- [x] Define search space from each plugin's `get_tunable_params()`
- [x] Objective: maximize OOS Sharpe ratio
- [x] Use TPE sampler with 100 trials per cluster
- [x] Log every trial to MLflow
- [x] Implement parameter stability check across adjacent windows

### 2.6 Walk-Forward Optimization

- [x] Implement `WalkForwardOptimizer` with rolling 252/126 day windows
- [x] Optimize on in-sample, evaluate on out-of-sample
- [x] Concatenate all OOS results for aggregate metrics
- [x] Compute Probability of Backtest Overfitting (PBO) via CPCV
- [x] Output validated parameters to `config/cluster_params/cluster_{id}.yaml`

### 2.7 Promotion Gate

- [ ] Implement `PromotionGate` class with criteria from architecture doc
- [ ] Check: history ≥ 2500 bars, individual Sharpe > cluster × 1.20, param drift < 20%, PBO < 0.40
- [ ] Write promoted parameters to `config/stock_overrides/{ticker}.yaml`
- [ ] Implement demotion logic: revert if rolling 60-day Sharpe drops > 0.30 below cluster
- [ ] Log all promotion/demotion decisions to `config/promotion_log.yaml`

### 2.8 Phase 2 Validation Checkpoint

- [ ] Meta-model improves Sharpe ratio over raw quant signals (verified across multiple stocks)
- [ ] Walk-forward backtest shows positive OOS Sharpe on majority of clusters
- [ ] PBO < 0.40 for production parameter sets
- [ ] At least 2-3 stocks promoted to individual params (validates the promotion gate works)
- [ ] All anti-overfitting checks pass (see architecture doc Section 12)

---

## Phase 3 — LLM Integration & Production (Weeks 13–18)

**Goal:** LLM validation gate, risk management, paper trading, monitoring.

### 3.1 Paper Trading Setup (Free via Alpaca)

> **Note:** Alpaca offers free unlimited paper trading via their API. No paid service needed. Phase 3.1 keeps paper trading in Phase 3 because the free service is available.

- [ ] Create Alpaca account at alpaca.markets and get paper trading API keys
- [ ] Add Alpaca keys to `.env`
- [ ] Implement `OrderManager` class wrapping `alpaca-trade-api`
- [ ] Test order submission, status tracking, and position queries on paper account
- [ ] Implement order types: market, limit, stop-loss, take-profit
- [ ] Add rate limiting to respect Alpaca API limits

### 3.2 LLM Validator (OpenAI)

- [ ] Implement `LLMValidator` class as a `SignalFilter` plugin
- [ ] Use OpenAI GPT-4o-mini with structured JSON output
- [ ] Build context: recent news (7 days), upcoming earnings, sector context
- [ ] Cache responses per ticker per day to avoid redundant API calls
- [ ] Implement APPROVE/VETO decision logic
- [ ] Track API costs in MLflow (tokens used per call)
- [ ] Write tests with mocked OpenAI responses

### 3.3 Risk Manager

- [ ] Implement `RiskManager` class with position sizing logic
- [ ] ATR-based base size with confidence and Kelly-fraction scaling
- [ ] Regime adjustment (volatile = smaller size)
- [ ] Stop-loss placement using ATR multiplier
- [ ] Take-profit using ATR multiplier
- [ ] Portfolio-level checks: max position %, max sector exposure, kill switches
- [ ] Daily/total drawdown monitoring with kill switch activation

### 3.4 Pipeline Orchestrator

- [ ] Implement `TradingPipeline` class wiring all components
- [ ] `run_daily()` method: ingest → features → signals → validate → execute
- [ ] Error handling and retry logic for API failures
- [ ] Idempotency: re-running on the same day should not produce duplicate trades
- [ ] Logging: structured logs to `data/results/logs/`

### 3.5 Telegram Alerts

- [ ] Create Telegram bot via BotFather, get bot token
- [ ] Implement `AlertService` using `python-telegram-bot`
- [ ] Send alerts on: trade entry, trade exit, LLM veto, drawdown breach, daily summary
- [ ] Format messages with markdown for readability
- [ ] Test alert delivery before going live

### 3.6 Monitoring Dashboard

- [ ] Set up local Grafana (Docker) or use simple Streamlit dashboard
- [ ] Track key metrics: open positions, daily P&L, cumulative return, drawdown
- [ ] Visualize: regime distribution, signal generation rate, LLM veto rate
- [ ] Pull data from MLflow + local SQLite/Parquet logs

### 3.7 Scheduling

- [ ] Set up `cron` (Linux/Mac) or Task Scheduler (Windows) to run pipeline daily pre-market
- [ ] Alternative: use `APScheduler` or `Prefect` for in-process scheduling
- [ ] Verify timezone handling (US market hours regardless of local timezone)

### 3.8 Phase 3 Validation Checkpoint

- [ ] Paper trading runs daily for at least 5 consecutive days without errors
- [ ] LLM veto rate is in expected range (10-30% of signals)
- [ ] All alerts deliver successfully
- [ ] Risk limits enforced (no trades exceed max position %)
- [ ] Drawdown kill switch tested manually
- [ ] Begin formal 3-month paper trading validation period

---

## Phase 4 — Future Enhancements (Post-Production)

These are pluggable enhancements documented in `docs/architecture.md` Section 14. Implement as separate plugins per the priority order in the dependency map. Do not start until Phase 3 has been running successfully for at least 3 months.

- [ ] Kalman filter smoothing plugin (P0 — highest impact)
- [ ] Cross-asset correlation enricher (P1)
- [ ] Fractional differentiation for ML features (P1)
- [ ] Adaptive exit management using Kalman velocity (P2)
- [ ] Options flow data enricher (P2)
- [ ] Attention-based dynamic indicator weighting (P3)
- [ ] Self-hosted LLM (FinGPT/Llama) to eliminate API costs (P3)
- [ ] RL-based position sizing — DDPG/TD3 (P4 — needs 6+ months of trade data)

---

## Working Notes

Use this section for ad-hoc notes, blockers, or questions to discuss:

- (empty)
