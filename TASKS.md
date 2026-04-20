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
  - `src/tuning/walk_forward.py` — step size = `out_of_sample_days` (non-overlapping OOS); enforces ≥3 windows; defaults pulled from `config/settings.yaml::tuning.{in_sample_days,out_of_sample_days}`.
- [x] Optimize on in-sample, evaluate on out-of-sample
  - `_run_window()` invokes `BayesianTuner.tune()` on the IS slice, then evaluates the top-k IS candidates on OOS for feeding PBO. Per-window stability check delegates to `BayesianTuner.stability_check` (20% drift threshold).
- [x] Concatenate all OOS results for aggregate metrics
  - `WFOResult.aggregate_oos_sharpe` = mean of per-window OOS Sharpes; full per-window series preserved in `WFOResult.windows[i].oos_sharpe`. CPCV (below) provides the stronger overfitting-aware aggregate called out by the architecture doc.
- [x] Compute Probability of Backtest Overfitting (PBO) via CPCV
  - `CombinatorialPurgedCV` (from Task 2.3) wired into `WalkForwardOptimizer.optimize()` and runs on the full dataset with the tuned `best_params`. Result populates `WFOResult.cpcv_pbo` (additive — coexists with the fast per-window `pbo`). CPCV failures are non-fatal and set `cpcv_pbo=None`. Knobs under `config/settings.yaml::tuning.cpcv` (`enabled`, `n_groups`, `n_test_groups`, `embargo_days`; defaults 10/2/5 → 45 paths).
- [x] Output validated parameters to `config/cluster_params/cluster_{id}.yaml`
  - `scripts/tune_clusters.py` — per-cluster WFO; aggregates OOS Sharpe, `pbo`, and `cpcv_pbo` across the cluster's tickers; writes `metadata` (incl. `cpcv_pbo`, `pbo_pass`, `pbo_source`, `is_stable`, `n_windows`) + `indicators.params` to YAML atomically. Promotion gate prefers `cpcv_pbo` when available, falls back to per-window `pbo`. QuantEngine already reads `indicators.params` via `_load_plugin_params`.

### 2.7 Promotion Gate

- [x] Implement `PromotionGate` class with criteria from architecture doc
  - `src/tuning/promotion_gate.py` — `PromotionDecision` dataclass + `evaluate/promote/demote/check_demotion/log_decision/resolve_params_path/is_promoted/list_promoted` methods. Thresholds loaded from `config/settings.yaml::tuning.promotion`. Orchestration wrapper in `scripts/tune_individual.py` runs WFO per ticker and calls the gate.
- [x] Check: history ≥ 2500 bars, individual Sharpe > cluster × 1.20, param drift < 20%, PBO < 0.40
  - All four criteria enforced in `evaluate()` (in order: history → Sharpe lift → stability → PBO). PBO prefers `WFOResult.cpcv_pbo` and falls back to per-window `pbo`. Cluster Sharpe ≤ 0 edge case requires an absolute lift ≥ 0.10 instead of a ratio.
- [x] Write promoted parameters to `config/stock_overrides/{ticker}.yaml`
  - Atomic temp-file + `shutil.move` write; carries cluster regime weights forward unchanged and embeds stock-specific `indicators.params` (via `BayesianTuner.unpack_params_static`). Metadata captures PBO/CPCV-PBO/stability/decision reasons for post-hoc auditing. `resolve_params_path()` gives downstream code the overrides → cluster → default fallback chain.
- [x] Implement demotion logic: revert if rolling 60-day Sharpe drops > 0.30 below cluster
  - `check_demotion()` (NaN/inf-safe predicate reading `tuning.promotion.demotion_sharpe_gap`) paired with `demote()` (deletes the override file). Phase 3's daily pipeline will invoke these once paper-trading fills supply the rolling live Sharpe.
- [x] Log all promotion/demotion decisions to `config/promotion_log.yaml`
  - `log_decision()` — read / modify / atomic-write append under the `decisions:` key, preserving every prior entry. `scripts/tune_individual.py` logs every evaluated ticker (promoted, kept, or skipped-for-history).

### 2.8 Phase 2 Validation Checkpoint

- [ ] Meta-model improves Sharpe ratio over raw quant signals (verified across multiple stocks)
- [ ] Walk-forward backtest shows positive OOS Sharpe on majority of clusters
- [ ] PBO < 0.40 for production parameter sets
- [ ] At least 2-3 stocks promoted to individual params (validates the promotion gate works)
- [ ] All anti-overfitting checks pass (see architecture doc Section 12)

---

## Phase 3 — Signal-Pillar Completion (Weeks 13–18)

**Goal:** Close the three gaps between the stated product goal (technicals + financials + news) and the current Phase-2 implementation: activate real news data for FinBERT, add fundamentals as a first-class signal pillar with per-stock tunable weight and ETF-aware zeroing, add event-driven exits, and refactor the feature assembler off its hardcoded column list.

**Why this becomes Phase 3 (review output, April 2026):** the v1.1 architecture lists fundamentals as an ingested data source but never feeds them into the `QuantEngine` composite. News is stubbed to zero (ADR-0010). Exit logic is confidence-only. `FeatureAssembler` hardcodes `FEATURE_COLUMNS`, blocking dynamic enrichment. See ADR-0011, ADR-0012, ADR-0013.

### 3.1 News Data Provider (Real Implementation)

Supersedes the Phase-1 stub in `src/data/news_data.py` and unblocks the already-built `FinBERTEnricher` (Phase 1.5).

- [ ] Implement `NewsDataProvider.get_headlines(ticker, start, end)` with Alpha Vantage `NEWS_SENTIMENT`
- [ ] Finnhub `/company-news` fallback on rate-limit or empty-result errors
- [ ] `(ticker, date)`-keyed on-disk cache under `data/raw/news/` with 12-hour TTL
- [ ] `get_macro_news()` equivalent via FRED + Alpha Vantage economic news (optional; stub acceptable if out of scope)
- [ ] Unit tests with recorded fixture responses for both providers (success + rate-limit error paths)
- [ ] Integration smoke test: call `FinBERTEnricher.enrich("AAPL", …)` end-to-end; assert non-zero sentiment features for a day with known news
- [ ] Update ADR-0010 status to "Superseded by ADR-0013" once merged
- [ ] Reference: ADR-0013

### 3.2 Fundamentals Data & Indicators (NEW SIGNAL PILLAR)

Adds fundamentals as a first-class signal pillar with per-stock / per-cluster tunable weight and hard ETF exclusion. See ADR-0011.

- [ ] Implement `FundamentalsDataProvider` in `src/data/fundamentals_data.py` (Alpha Vantage `OVERVIEW`, `INCOME_STATEMENT`, `CASH_FLOW`, `EARNINGS`; yfinance fallback)
- [ ] Cache raw statements in `data/features/fundamentals/{ticker}/{fiscal_period}.parquet`
- [ ] Monthly refresh + on-demand refresh within 7 days of a scheduled earnings date
- [ ] ETF short-circuit: return empty DataFrame for any ticker in `config/watchlist.yaml::etfs:` (no API call)
- [ ] Add `FundamentalIndicatorPlugin` abstract base to `src/plugins/base.py`
  - Contract: `compute(fundamentals_df, price_df) -> pd.Series`, `normalize(values) -> pd.Series`
  - Attribute `applies_to_etfs: bool = False`
- [ ] Register `FundamentalIndicatorPlugin` type in `src/plugins/registry.py`
- [ ] Implement initial fundamentals plugins under `src/plugins/fundamentals/`:
  - [ ] `pe_zscore.py` — trailing P/E vs sector 5-year z-score
  - [ ] `fcf_yield.py` — free cash flow / market cap
  - [ ] `earnings_growth.py` — YoY EPS growth from last 4 quarters
  - [ ] `earnings_surprise.py` — most recent actual EPS vs consensus estimate
- [ ] Wire `fundamental_score` aggregation channel into `QuantEngine._weighted_composite` (mean of enabled plugins' normalised outputs)
- [ ] Apply applicability multiplier `f_fund`:
  - [ ] ETFs force `f_fund = 0` (detect from `watchlist.yaml::etfs:`)
  - [ ] Optional per-stock `fundamentals_weight_override` in `watchlist.yaml`
  - [ ] Re-normalise remaining regime weights to sum to 1.0 when `f_fund = 0`
- [ ] Add `fundamentals_weight` per regime to `config/cluster_params/cluster_default.yaml` (TRENDING_UP: 0.15, TRENDING_DOWN: 0.10, RANGING: 0.20, VOLATILE: 0.05)
- [ ] Expose `fundamentals_weight` as a `ParamSpec` in `get_tunable_params()` so `BayesianTuner` can search it (`[0.0, 0.30]`)
- [ ] Unit tests: `FundamentalIndicatorPlugin` contract, ETF zero-weight path, re-normalisation math
- [ ] Reference: ADR-0011

### 3.3 Event-Driven Exits

Adds discrete event-based exits alongside the existing confidence-based exit. See ADR-0012. The `RiskManager` that consumes these filters is scheduled in Phase 4.3, but the filters themselves land here and are unit-testable in isolation.

- [ ] Replace Phase-1 stub `MarketDataProvider.get_earnings_calendar()` with real Alpha Vantage `EARNINGS_CALENDAR` client
- [ ] Cache earnings calendar in `data/features/events/earnings_calendar.parquet` with 24-hour TTL
- [ ] Add `EventFilter` abstract base to `src/plugins/base.py`
  - Contract: `should_exit(position, bar, context) -> (bool, Optional[str])`
- [ ] Register `EventFilter` type in `src/plugins/registry.py`
- [ ] Implement filters in `src/risk/event_filter.py`:
  - [ ] `EarningsBlackoutFilter` (default window: T-2 to T+1)
  - [ ] `NewsShockFilter` (default: `|sentiment| > 0.75` AND `news_volume_ratio > 3`)
  - [ ] `AtrStopFilter` (default: `2.0 × ATR_14` stop, `4.0 × ATR_14` take-profit)
- [ ] Add `exit_reason: Optional[str]` field to `TradeSignal` (`src/models/trade_signal.py`)
- [ ] Add `exits:` block to `config/settings.yaml` with per-filter thresholds + `enabled: bool` toggles
- [ ] Unit tests for each filter against synthetic earnings / news-shock / price-gap inputs
- [ ] Reference: ADR-0012

### 3.4 Dynamic Feature Schema

`src/signals/feature_assembler.py` currently hardcodes `FEATURE_COLUMNS` (~13 columns). Adding fundamentals or richer sentiment will silently drop features. Refactor to a schema registry built from active plugins.

- [ ] Remove hardcoded `FEATURE_COLUMNS` constant
- [ ] Build schema at runtime from active `IndicatorPlugin`, `DataEnricher`, and `FundamentalIndicatorPlugin` instances (each contributes its `output_column`s)
- [ ] Persist the training-time schema alongside each model artifact (`data/models/meta_model/v*_*.pkl.schema.json`)
- [ ] At inference time, assert assembled-feature schema matches training-time schema; raise with a clear error on mismatch
- [ ] Contract test: registering a new plugin propagates to assembled features without code changes to `FeatureAssembler`

### 3.5 Provider Choice ADR

- [ ] Publish ADR-0013 (already drafted — ensure index is updated)
- [ ] Document Alpha Vantage free-tier rate-limit strategy and fallback logic
- [ ] Add `ALPHA_VANTAGE_API_KEY` and `FINNHUB_API_KEY` to `.env.example`

### 3.6 Phase 3 Validation Checkpoint

- [ ] Feature matrix for a single sample stock (e.g., AAPL) shows **non-zero** technical, sentiment, AND fundamental columns on a recent bar
- [ ] Feature matrix for an ETF (e.g., `ITA`) shows `fundamental_score == 0.0` on every bar, regardless of fundamentals-provider behaviour
- [ ] Manual override test: setting `fundamentals_weight_override: 0.0` on a stock in `watchlist.yaml` reproduces the ETF behaviour
- [ ] Backtest A/B regression: `scripts/run_backtest.py` on AAPL 2020–2024 with fundamentals off vs on — fundamentals-on must not degrade Sharpe by more than 10% (overfitting guard)
- [ ] Event-filter ablation: disabling `EarningsBlackoutFilter` increases max drawdown on AAPL earnings dates
- [ ] All Phase 3 unit and integration tests pass
- [ ] ADRs 0011, 0012, 0013 reviewed and moved from Proposed to Accepted

---

## Phase 4 — LLM Integration & Production (Weeks 19–24)

**Goal:** LLM validation gate, risk management (consuming Phase-3 event filters), paper trading, monitoring.

> **Note (April 2026 renumber):** This phase was previously Phase 3. Its scope is unchanged; it is renumbered to make room for the Phase 3 signal-pillar completion work above. The `RiskManager` in 4.3 now integrates the `EventFilter` instances built in Phase 3.3.

### 4.1 Paper Trading Setup (Free via Alpaca)

> **Note:** Alpaca offers free unlimited paper trading via their API. No paid service needed.

- [ ] Create Alpaca account at alpaca.markets and get paper trading API keys
- [ ] Add Alpaca keys to `.env`
- [ ] Implement `OrderManager` class wrapping `alpaca-trade-api`
- [ ] Test order submission, status tracking, and position queries on paper account
- [ ] Implement order types: market, limit, stop-loss, take-profit
- [ ] Add rate limiting to respect Alpaca API limits

### 4.2 LLM Validator (OpenAI)

- [ ] Implement `LLMValidator` class as a `SignalFilter` plugin
- [ ] Use OpenAI GPT-4o-mini with structured JSON output
- [ ] Build context: recent news (7 days), upcoming earnings, sector context
- [ ] Cache responses per ticker per day to avoid redundant API calls
- [ ] Implement APPROVE/VETO decision logic
- [ ] Track API costs in MLflow (tokens used per call)
- [ ] Write tests with mocked OpenAI responses

### 4.3 Risk Manager

- [ ] Implement `RiskManager` class with position sizing logic
- [ ] ATR-based base size with confidence and Kelly-fraction scaling
- [ ] Regime adjustment (volatile = smaller size)
- [ ] Populate `TradeSignal.stop_loss_pct` and `take_profit_pct` using ATR multipliers
- [ ] `evaluate_exits(open_positions, bar)` — iterate Phase-3.3 `EventFilter` instances + confidence exit; set `TradeSignal.exit_reason` to the first triggered rule
- [ ] Portfolio-level checks: max position %, max sector exposure, kill switches
- [ ] Daily/total drawdown monitoring with kill switch activation

### 4.4 Pipeline Orchestrator

- [ ] Implement `TradingPipeline` class in `src/pipeline.py` wiring all components (Regime → FeatureAssembler → QuantEngine → MetaLabelModel → LLMValidator → RiskManager)
- [ ] `run_daily()` method: ingest → features → signals → validate → execute
- [ ] `scripts/run_daily.py` thin CLI wrapper around `TradingPipeline.run_daily`
- [ ] Error handling and retry logic for API failures
- [ ] Idempotency: re-running on the same day should not produce duplicate trades
- [ ] Logging: structured logs to `data/results/logs/`

### 4.5 Telegram Alerts

- [ ] Create Telegram bot via BotFather, get bot token
- [ ] Implement `AlertService` using `python-telegram-bot`
- [ ] Send alerts on: trade entry, trade exit (include `exit_reason`), LLM veto, drawdown breach, daily summary
- [ ] Format messages with markdown for readability
- [ ] Test alert delivery before going live

### 4.6 Monitoring Dashboard

- [ ] Set up local Grafana (Docker) or use simple Streamlit dashboard
- [ ] Track key metrics: open positions, daily P&L, cumulative return, drawdown, exit-reason distribution
- [ ] Visualize: regime distribution, signal generation rate, LLM veto rate
- [ ] Pull data from MLflow + local SQLite/Parquet logs

### 4.7 Scheduling

- [ ] Set up `cron` (Linux/Mac) or Task Scheduler (Windows) to run pipeline daily pre-market
- [ ] Alternative: use `APScheduler` or `Prefect` for in-process scheduling
- [ ] Verify timezone handling (US market hours regardless of local timezone)

### 4.8 Phase 4 Validation Checkpoint

- [ ] Paper trading runs daily for at least 5 consecutive days without errors
- [ ] LLM veto rate is in expected range (10-30% of signals)
- [ ] All alerts deliver successfully
- [ ] Risk limits enforced (no trades exceed max position %)
- [ ] Drawdown kill switch tested manually
- [ ] Event-exit attribution in trade log matches expected causes (earnings / news / ATR)
- [ ] Begin formal 3-month paper trading validation period

---

## Phase 5 — Future Enhancements (Post-Production)

> **Note (April 2026 renumber):** This was previously Phase 4. Renumbered only; contents and priority order are unchanged.

These are pluggable enhancements documented in `docs/architecture.md` Section 14. Implement as separate plugins per the priority order in the dependency map. Do not start until Phase 4 has been running successfully for at least 3 months.

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
