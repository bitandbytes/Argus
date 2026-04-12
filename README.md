# Multi-Model Position Trading Pipeline

A Python-based algorithmic trading system for stock position trading (holding periods of days to weeks). Combines classical quantitative analysis, machine learning meta-labeling, and LLM-based sentiment validation in a three-layer cascade architecture.

## Requirements

- **Python 3.12** (hard requirement — see note below)
- Windows 10/11, Linux, or macOS

> **Python version note:** Python 3.12 is required. `pandas-ta` pins `numba==0.61.2` which is incompatible with Python 3.13+. On Windows 10, PyTorch ≥ 2.6 fails with WinError 1114 (c10.dll init failure); `torch==2.5.0+cpu` is the confirmed working version.

## Quick Start

```bash
# 1. Create virtual environment with Python 3.12
py -3.12 -m venv .venv            # Windows
python3.12 -m venv .venv          # Linux/Mac

# 2. Activate
.venv\Scripts\activate             # Windows
source .venv/bin/activate          # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install torch (CPU wheel — required on Windows 10)
pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/cpu

# 5. Verify setup
python scripts/verify_setup.py

# 6. Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI, Alpaca, and Telegram keys

# 7. Run tests
pytest tests/

# 8. Run a backtest
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
