# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

This is a **multi-model position trading pipeline** for stocks. It uses a three-layer cascade architecture combining classical quantitative analysis, machine learning meta-labeling, and LLM-based sentiment validation to generate entry and exit signals for position trades (holding periods of days to weeks).

**Core design philosophy:**
- **Cascade over ensemble**: Quant engine produces high-recall direction signals; ML meta-model filters false positives; LLM validates only actionable signals (cost-efficient).
- **Per-stock adaptation**: Hybrid cluster-based defaults with individual promotion for stocks with sufficient history.
- **Plugin-first**: Every indicator, smoother, enricher, and signal filter is a plugin. Adding new components requires zero changes to core pipeline code.
- **Anti-overfitting by design**: Walk-forward optimization, purged cross-validation, Bayesian tuning.

Read `docs/architecture.md` for the full system architecture before making non-trivial changes.

## Architecture at a Glance

```
Data → FinBERT (batch sentiment) → Regime Detector → Quant Engine
                                                          ↓
                                                    ML Meta-Model
                                                          ↓ (only if trade=yes)
                                                    LLM Validator (OpenAI)
                                                          ↓
                                                    Risk Manager → Order Execution
```

**Three layers:**
1. **Quant Engine** (Layer 2): Composite of normalized indicator signals weighted by regime. Direction predictor with high recall.
2. **ML Meta-Model** (Layer 3): XGBoost trained via López de Prado's meta-labeling. Decides whether to act on quant signal + bet size.
3. **LLM Validator** (Layer 4): OpenAI GPT-4o-mini deep analysis of news/earnings context. Only called for actionable signals to minimize API cost.

**Sentiment** (Layer 1): FinBERT runs locally as a batch job over all tickers. Outputs become features for both quant and ML layers. Zero API cost.

## Tech Stack

- **Language**: Python 3.11+
- **Package management**: `pip` + `requirements.txt` (see ADR-0007)
- **Data**: yfinance (free, daily OHLCV) — see ADR-0008
- **Indicators**: pandas-ta
- **ML**: XGBoost, scikit-learn, Optuna (Bayesian tuning)
- **Sentiment**: HuggingFace transformers (FinBERT, runs on CPU)
- **Regime detection**: hmmlearn
- **Backtesting**: PyBroker (primary), VectorBT (parameter sweeps)
- **LLM**: OpenAI GPT-4o-mini (see ADR-0006)
- **Validation**: timeseriescv (purged k-fold)
- **Tracking**: MLflow
- **Broker**: Alpaca (paper trading is free)
- **Alerts**: Telegram bot

## Directory Structure

```
trading-pipeline/
├── CLAUDE.md                    # This file
├── TASKS.md                     # Phased development task list
├── README.md                    # Project intro
├── requirements.txt             # pip dependencies
├── .env.example                 # Template for API keys
├── config/
│   ├── settings.yaml            # Global config (thresholds, paths)
│   ├── watchlist.yaml           # Stocks to trade
│   └── plugins.yaml             # Active plugins config
├── docs/
│   ├── architecture.md          # Full architecture document
│   └── decisions/               # ADRs (architecture decision records)
├── .claude/
│   └── skills/                  # Custom skills for Claude Code
│       ├── plugin-author/       # How to write new plugins
│       ├── backtest-runner/     # How to run backtests
│       ├── quant-engine-dev/    # Quant engine development
│       ├── ml-meta-labeler/     # Meta-labeling workflow
│       └── finbert-integration/ # FinBERT setup and use
├── src/
│   ├── pipeline.py              # Main orchestrator
│   ├── plugins/
│   │   ├── base.py              # Abstract plugin interfaces
│   │   ├── registry.py          # Plugin discovery/registration
│   │   ├── indicators/          # IndicatorPlugin implementations
│   │   ├── smoothing/           # SmoothingPlugin implementations
│   │   ├── enrichers/           # DataEnricher implementations (FinBERT, etc.)
│   │   └── filters/             # SignalFilter implementations
│   ├── data/                    # Data providers, feature store
│   ├── signals/                 # Regime detector, quant engine, meta-model
│   ├── risk/                    # Position sizing, order management
│   ├── tuning/                  # Clustering, walk-forward, promotion gate
│   └── models/                  # Data models (TradeSignal, FeatureVector, etc.)
├── tests/                       # pytest tests
├── notebooks/                   # Research and exploration notebooks
├── scripts/                     # Entry points (run_daily.py, run_backtest.py, etc.)
└── data/                        # Local data storage (gitignored)
    ├── raw/                     # Downloaded OHLCV, news
    ├── features/                # Computed feature parquets
    ├── models/                  # Trained models (XGBoost, HMM, FinBERT cache)
    └── results/                 # Backtest results, logs
```

## Key Conventions

### Code style
- **Type hints required** on all function signatures.
- **Dataclasses** for data models (`TradeSignal`, `FeatureVector`, etc.) — no untyped dicts in interfaces.
- **Black** formatting (line length 100).
- **Ruff** for linting.
- **Docstrings** in Google style for all public methods.

### Plugin development
- Every indicator, smoother, enricher, and filter must implement one of the abstract base classes in `src/plugins/base.py`.
- Plugins are discovered via `config/plugins.yaml` — never hardcode plugin imports in core pipeline code.
- See the **plugin-author** skill in `.claude/skills/plugin-author/SKILL.md` for the full plugin contract and examples.

### Anti-lookahead bias (CRITICAL)
- All features must use `.shift(1)` when feeding into ML models to ensure they only see past data.
- Never use centered moving averages — always trailing.
- Walk-forward backtests must strictly enforce the day-by-day replay (PyBroker handles this).
- When in doubt, ask: "Could this code have known about today's close before today's close happened?"

### Testing
- pytest for all tests.
- Every plugin must have unit tests verifying its `compute()`/`normalize()`/`smooth()` methods.
- Integration tests for the full pipeline run on a known small dataset.
- Run tests: `pytest tests/`

### MLflow tracking
- Wrap all backtests and model training in `mlflow.start_run()`.
- Log parameters, metrics, and the trained model artifact.
- Local MLflow UI: `mlflow ui` then visit `http://localhost:5000`.

## Available Skills

When working on specific tasks, refer to these skills (located in `.claude/skills/`):

| Skill | When to use |
|-------|-------------|
| **plugin-author** | Creating any new plugin (indicator, smoother, enricher, filter) |
| **quant-engine-dev** | Modifying composite signal logic, regime weights, or signal generation |
| **ml-meta-labeler** | Working with triple-barrier labeling, training/calibrating XGBoost meta-model |
| **finbert-integration** | Setting up or using FinBERT for sentiment scoring |
| **backtest-runner** | Running backtests with PyBroker or VectorBT, walk-forward optimization |

## Architecture Decisions

All major architectural decisions are documented as ADRs in `docs/decisions/`. Read these before making changes that touch the affected areas:

- **ADR-0001**: Three-layer cascade architecture (cascade over ensemble)
- **ADR-0002**: FinBERT local sentiment processing (cost rationale)
- **ADR-0003**: Meta-labeling over ensemble (López de Prado approach)
- **ADR-0004**: Hybrid cluster + individual tuning (overfitting mitigation)
- **ADR-0005**: Plugin architecture for extensibility
- **ADR-0006**: LLM provider choice (OpenAI GPT-4o-mini)
- **ADR-0007**: Package manager choice (pip + requirements.txt)
- **ADR-0008**: Data provider choice (yfinance for prototype)

## Common Commands

```bash
# Setup (first time)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Then fill in API keys

# Run tests
pytest tests/ -v

# Run a backtest
python scripts/run_backtest.py --ticker AAPL --start 2020-01-01 --end 2024-12-31

# Tune cluster parameters
python scripts/tune_clusters.py --mode wfo

# Start MLflow UI
mlflow ui

# Daily pipeline run (paper trading mode)
python scripts/run_daily.py --mode paper
```

## Environment Variables

Required in `.env`:

```
OPENAI_API_KEY=sk-...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

## Important Constraints

- **Position trading timeframe only**: Holding periods of days to weeks, NOT day trading or HFT.
- **Daily bars are sufficient**: No intraday data dependencies. Pipeline runs once per day pre-market.
- **Free-tier first**: All Phase 1 components run with free data sources. Paid services are optional upgrades.
- **Paper trading before live**: Minimum 3 months of profitable paper trading required before any live capital deployment.
- **Anti-overfitting checks are non-negotiable**: PBO < 0.40, parameter stability < 20% drift, OOS Sharpe ≥ 0.5.

## When in Doubt

1. Check `docs/architecture.md` for the design rationale.
2. Check `docs/decisions/` for the relevant ADR.
3. Check `TASKS.md` to see which phase the work belongs to.
4. Check the relevant skill in `.claude/skills/`.
5. Ask the user for clarification rather than guessing on architectural matters.
