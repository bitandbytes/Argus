# Multi-Model Position Trading Pipeline

A Python-based algorithmic trading system for stock position trading (holding periods of days to weeks). Combines classical quantitative analysis, machine learning meta-labeling, and LLM-based sentiment validation in a three-layer cascade architecture.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI, Alpaca, and Telegram keys

# 4. Run tests
pytest tests/

# 5. Run a backtest
python scripts/run_backtest.py --ticker AAPL --start 2020-01-01 --end 2024-12-31
```

## Documentation

- **`CLAUDE.md`** — Project context for Claude Code
- **`TASKS.md`** — Phased development task list
- **`docs/architecture.md`** — Full system architecture
- **`docs/decisions/`** — Architecture Decision Records (ADRs)

## Architecture Overview

```
Data → FinBERT (sentiment) → Regime Detector → Quant Engine
                                                    ↓
                                              ML Meta-Model
                                                    ↓ (if trade=yes)
                                              LLM Validator (OpenAI)
                                                    ↓
                                              Risk Manager → Execution
```

See `docs/architecture.md` for the full design rationale.

## Status

🚧 **Phase 1 (Research & Prototyping)** — In progress

See `TASKS.md` for current development tasks.

## License

Private project. Not for distribution.
