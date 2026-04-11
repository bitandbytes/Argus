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

- [ ] Create Python virtual environment (`python -m venv .venv`)
- [ ] Install base dependencies (`pip install -r requirements.txt`)
- [ ] Verify FinBERT model downloads correctly via HuggingFace transformers
- [ ] Create `.env` from `.env.example` and add OpenAI API key
- [ ] Set up MLflow locally (`mlflow ui` should serve on `localhost:5000`)
- [ ] Configure git hooks for black/ruff (optional but recommended)
- [ ] Verify yfinance data fetch works for all tickers in `config/watchlist.yaml`

### 1.2 Data Layer

- [ ] Implement `MarketDataProvider` class wrapping yfinance with caching
- [ ] Implement `FeatureStore` using Parquet files in `data/features/`
- [ ] Add `get_earnings_calendar()` method to MarketDataProvider
- [ ] Create `NewsDataProvider` stub (Phase 1 may skip live news; use Alpha Vantage free tier later)
- [ ] Write unit tests for data providers (mock the yfinance calls)
- [ ] Validate data quality: check for NaN values, missing dates, suspicious price gaps

### 1.3 Plugin Foundation

- [ ] Implement abstract base classes in `src/plugins/base.py`:
  - `IndicatorPlugin`
  - `SmoothingPlugin`
  - `DataEnricher`
  - `SignalFilter`
  - Supporting types: `SmoothResult`, `ParamSpec`
- [ ] Implement `PluginRegistry` in `src/plugins/registry.py` with discovery logic
- [ ] Load plugin config from `config/plugins.yaml`
- [ ] Write tests for plugin registry: register, retrieve, list, error on missing
- [ ] Document the plugin contract in `.claude/skills/plugin-author/SKILL.md` (already drafted)

### 1.4 Indicator Plugins (each as separate plugin)

- [ ] Implement `SMACrossoverIndicator` with `compute()` and `normalize()` methods
- [ ] Implement `RSIIndicator`
- [ ] Implement `MACDIndicator`
- [ ] Implement `BollingerBandIndicator`
- [ ] Implement `DonchianChannelIndicator`
- [ ] Implement `VolumeIndicator`
- [ ] Unit tests for each indicator with known input/output pairs
- [ ] Verify each indicator's `get_tunable_params()` returns valid ParamSpec entries

### 1.5 FinBERT Sentiment Plugin

- [ ] Implement `FinBERTEnricher` as a `DataEnricher` plugin
- [ ] Cache model on first load to avoid re-downloading
- [ ] Implement batch processing for efficiency (process 64 headlines at a time)
- [ ] Compute rolling features: `sentiment_ma_5d`, `sentiment_ma_20d`, `sentiment_momentum`
- [ ] Test on sample headlines to verify sentiment scores are reasonable
- [ ] Decide on news source: skip historical news for Phase 1 backtests; use it live in Phase 3
- [ ] See `.claude/skills/finbert-integration/SKILL.md` for implementation guidance

### 1.6 Regime Detector

- [ ] Implement `RegimeDetector` class with HMM (`hmmlearn`) + ADX dual approach
- [ ] Train HMM on log returns + realized volatility, n_components=3
- [ ] Implement `detect()` returning `RegimeType` enum
- [ ] Add ADX as the fast classifier (ADX > 25 = trending, ADX < 20 = ranging)
- [ ] Implement reconciliation logic combining HMM and ADX outputs
- [ ] Persist trained HMM models in `data/models/hmm/` per cluster (or per stock if individual)
- [ ] Test regime stability: regimes should not flicker daily

### 1.7 Quant Engine

- [ ] Implement `QuantEngine` class that uses plugin registry
- [ ] Load active indicators from registry on init
- [ ] Apply regime-specific weights (initial defaults from architecture doc)
- [ ] Compute composite score = Σ(weight × normalized_score)
- [ ] Implement multi-timeframe confirmation (daily + weekly alignment check)
- [ ] Generate `TradeSignal` dataclass with direction, confidence, regime
- [ ] Wire FinBERT sentiment as one of the inputs to the composite
- [ ] See `.claude/skills/quant-engine-dev/SKILL.md` for design patterns

### 1.8 Initial Backtesting

- [ ] Install PyBroker and verify import works
- [ ] Write `scripts/run_backtest.py` that runs the quant engine on historical data
- [ ] Configure realistic transaction costs (0.05% commission, 0.05% slippage)
- [ ] Run backtest on AAPL 2020-2024, compute Sharpe, max drawdown, win rate
- [ ] Visualize equity curve in a notebook
- [ ] Compare results against buy-and-hold benchmark
- [ ] Log all backtest runs to MLflow
- [ ] See `.claude/skills/backtest-runner/SKILL.md` for backtesting workflow

### 1.9 Phase 1 Validation Checkpoint

- [ ] All tests pass (`pytest tests/`)
- [ ] Backtest produces sensible (not random) results — Sharpe should be in `[-1.0, +2.0]` range
- [ ] No lookahead bias detected (manual code review)
- [ ] Pipeline can run end-to-end on at least one ticker
- [ ] Document any deviations from the architecture doc as new ADRs
- [ ] Demo run reviewed before proceeding to Phase 2

---

## Phase 2 — Meta-Labeling & Tuning (Weeks 7–12)

**Goal:** ML meta-model filters quant signals; per-stock tuning is automated; walk-forward validation prevents overfitting.

### 2.1 Triple-Barrier Labeling

- [ ] Implement triple-barrier labeling per López de Prado AFML Chapter 3
- [ ] Function signature: `triple_barrier_labels(prices, signals, tp_pct, sl_pct, max_holding_days)`
- [ ] Returns labels: `+1` (TP hit), `-1` (SL hit), `0` (timeout)
- [ ] Test on synthetic data with known outcomes
- [ ] See `.claude/skills/ml-meta-labeler/SKILL.md` for full workflow

### 2.2 ML Meta-Model

- [ ] Implement `MetaLabelModel` class using XGBoost
- [ ] Build feature assembly: quant features + sentiment + regime (one-hot) + quant prediction
- [ ] Train binary classifier: did the quant signal lead to a profitable trade?
- [ ] Apply Platt scaling via `CalibratedClassifierCV`
- [ ] Compute calibration curve and Brier score
- [ ] Save trained model to `data/models/meta_model/` with versioning
- [ ] Log training run to MLflow with all hyperparameters

### 2.3 Purged Cross-Validation

- [ ] Implement `PurgedKFoldCV` per AFML Chapter 7 (or use `timeseriescv` library)
- [ ] Embargo gap = max signal dependency horizon (e.g., 5 days for 5-day forward returns)
- [ ] Use this CV split for all meta-model training
- [ ] Verify it produces non-overlapping folds with proper purging

### 2.4 Stock Clustering

- [ ] Implement `StockClusterer` class with K-Means and tslearn DTW options
- [ ] Feature extraction per stock: Hurst exponent, mean ADX, lag-1 autocorr, volatility
- [ ] Choose `k` via silhouette score (target k=4..8)
- [ ] Persist cluster assignments to `config/cluster_assignments.yaml`
- [ ] Test re-clustering: same input data should produce stable assignments

### 2.5 Bayesian Tuning with Optuna

- [ ] Implement `BayesianTuner` wrapping Optuna
- [ ] Define search space from each plugin's `get_tunable_params()`
- [ ] Objective: maximize OOS Sharpe ratio
- [ ] Use TPE sampler with 100 trials per cluster
- [ ] Log every trial to MLflow
- [ ] Implement parameter stability check across adjacent windows

### 2.6 Walk-Forward Optimization

- [ ] Implement `WalkForwardOptimizer` with rolling 252/126 day windows
- [ ] Optimize on in-sample, evaluate on out-of-sample
- [ ] Concatenate all OOS results for aggregate metrics
- [ ] Compute Probability of Backtest Overfitting (PBO) via CPCV
- [ ] Output validated parameters to `config/cluster_params/cluster_{id}.yaml`

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
