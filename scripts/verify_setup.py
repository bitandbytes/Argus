"""
Setup verification script for the Argus trading pipeline.

Run this after completing the initial setup steps to confirm every
dependency and external connection is working correctly.

Usage:
    python scripts/verify_setup.py

Exit code 0 = all required checks passed.
Exit code 1 = one or more required checks failed (see output for details).

Warnings (marked [WARN]) are informational and do not affect the exit code.
They typically indicate optional API keys that are only needed in Phase 3.
"""

import os
import sys
import pathlib

# Ensure the project root is on the path so src/ imports work.
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
SKIP = "[SKIP]"

_failures: list[str] = []
_warnings: list[str] = []


def check(name: str, fn, *, required: bool = True) -> bool:
    """Run a single check, printing status and collecting failures."""
    try:
        fn()
        print(f"  {PASS} {name}")
        return True
    except Exception as e:
        tag = FAIL if required else WARN
        print(f"  {tag} {name}: {e}")
        if required:
            _failures.append(name)
        else:
            _warnings.append(name)
        return False


# ============================================================================
# Section 1: Core imports
# ============================================================================


def _check_imports() -> None:
    print("\n[1/6] Core package imports")

    check("python-dotenv", lambda: __import__("dotenv"))
    check("pyyaml", lambda: __import__("yaml"))
    check("pydantic", lambda: __import__("pydantic"))
    check("click", lambda: __import__("click"))
    check("pandas", lambda: __import__("pandas"))
    check("numpy", lambda: __import__("numpy"))
    check("pyarrow", lambda: __import__("pyarrow"))
    check("polars", lambda: __import__("polars"))
    check("yfinance", lambda: __import__("yfinance"))

    def _check_pandas_ta() -> None:
        import pandas_ta  # noqa: F401
        import numba  # noqa: F401
        import numba as nb

        # Verify numba JIT works (basic smoke test)
        @nb.njit
        def _add(a: float, b: float) -> float:
            return a + b

        assert _add(1.0, 2.0) == 3.0

    check("pandas-ta + numba (JIT smoke test)", _check_pandas_ta)
    check("transformers (HuggingFace)", lambda: __import__("transformers"))

    def _check_torch() -> None:
        try:
            import torch  # noqa: F401
        except OSError as e:
            if "1114" in str(e) or "shm.dll" in str(e) or "c10.dll" in str(e):
                raise RuntimeError(
                    "PyTorch DLL init failed — torch >=2.6 is incompatible with Windows 10 (WinError 1114).\n"
                    "    Fix: install the CPU wheel for torch 2.5.0 (confirmed working on Python 3.12 + Windows 10):\n"
                    '      pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/cpu\n'
                    "    Also ensure you are using Python 3.12 (required — see requirements.txt header)."
                ) from e
            raise

    check("torch", _check_torch)
    check("tokenizers", lambda: __import__("tokenizers"))
    check("hmmlearn", lambda: __import__("hmmlearn"))
    check("scikit-learn", lambda: __import__("sklearn"))
    check("xgboost", lambda: __import__("xgboost"))
    check("lightgbm", lambda: __import__("lightgbm"))
    check("optuna", lambda: __import__("optuna"))
    check("tslearn", lambda: __import__("tslearn"))
    check("lib-pybroker", lambda: __import__("pybroker"))
    check("vectorbt", lambda: __import__("vectorbt"))
    check("timeseriescv", lambda: __import__("timeseriescv"))
    check("mlflow", lambda: __import__("mlflow"))
    check("openai", lambda: __import__("openai"))
    check("alpaca-py", lambda: __import__("alpaca"))
    check("python-telegram-bot", lambda: __import__("telegram"))
    check("apscheduler", lambda: __import__("apscheduler"))
    check("pytest", lambda: __import__("pytest"))
    check("black", lambda: __import__("black"))
    check("ruff", lambda: __import__("ruff"))


# ============================================================================
# Section 2: Environment / .env file
# ============================================================================


def _check_env() -> None:
    print("\n[2/6] Environment variables (.env)")

    env_path = ROOT / ".env"
    check(
        ".env file exists",
        lambda: (
            (_ for _ in ()).throw(FileNotFoundError(f"Not found at {env_path}"))
            if not env_path.exists()
            else None
        ),
        required=False,
    )

    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)

    required_live_keys = {
        "OPENAI_API_KEY": "Phase 3 — LLM validator",
        "ALPACA_API_KEY": "Phase 3 — paper trading",
        "ALPACA_SECRET_KEY": "Phase 3 — paper trading",
        "TELEGRAM_BOT_TOKEN": "Phase 3 — alerts",
        "TELEGRAM_CHAT_ID": "Phase 3 — alerts",
    }
    for key, note in required_live_keys.items():
        val = os.getenv(key, "")
        placeholder = not val or "your" in val.lower() or val.endswith("...")
        check(
            f"{key} set ({note})",
            lambda v=val, p=placeholder: (
                (_ for _ in ()).throw(ValueError("Missing or placeholder value")) if p else None
            ),
            required=False,  # API keys are optional until Phase 3
        )


# ============================================================================
# Section 3: yfinance watchlist check
# ============================================================================


def _check_yfinance() -> None:
    print("\n[3/6] yfinance — watchlist ticker fetch")
    import yaml
    import yfinance as yf

    watchlist_path = ROOT / "config" / "watchlist.yaml"
    if not watchlist_path.exists():
        print(f"  {SKIP} config/watchlist.yaml not found — skipping ticker checks")
        return

    with open(watchlist_path) as f:
        wl = yaml.safe_load(f)

    all_tickers: list[dict] = []
    for section in ("stocks", "etfs"):
        all_tickers.extend(wl.get(section, []))

    etf_tickers = {e["ticker"] for e in wl.get("etfs", [])}

    for entry in all_tickers:
        ticker = entry["ticker"]
        is_etf = ticker in etf_tickers

        def _fetch(t: str = ticker) -> None:
            data = yf.Ticker(t).history(period="5d", auto_adjust=True)
            if data.empty:
                raise ValueError("returned empty DataFrame")

        # European ETFs are known uncertain; warn instead of fail
        check(
            f"yfinance fetch: {ticker}",
            _fetch,
            required=not is_etf,
        )


# ============================================================================
# Section 4: MLflow local setup
# ============================================================================


def _check_mlflow() -> None:
    print("\n[4/6] MLflow local setup")

    def _mlflow_write() -> None:
        import mlflow

        db_path = ROOT / "data" / "mlflow.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        # Creating an experiment writes to the DB
        exp_name = "__verify_setup_probe__"
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            client.create_experiment(exp_name)

    check("MLflow SQLite tracking URI writable", _mlflow_write)


# ============================================================================
# Section 5: FinBERT model cache check
# ============================================================================


def _check_finbert() -> None:
    print("\n[5/6] FinBERT model (HuggingFace cache)")

    def _finbert_cached() -> None:
        from transformers import AutoConfig

        model_name = "ProsusAI/finbert"
        try:
            # Try loading config offline-only; raises if not cached
            AutoConfig.from_pretrained(model_name, local_files_only=True)
        except Exception:
            raise RuntimeError(
                "FinBERT not in local cache. Run: python scripts/download_finbert.py"
            )

    check(
        "FinBERT cached locally",
        _finbert_cached,
        required=False,  # not required until Phase 1.5
    )


# ============================================================================
# Section 6: Project source imports
# ============================================================================


def _check_src() -> None:
    print("\n[6/6] Project source imports")

    check(
        "src.models.trade_signal (TradeSignal, FeatureVector)",
        lambda: __import__("src.models.trade_signal", fromlist=["TradeSignal"]),
    )
    check(
        "src.plugins.base (IndicatorPlugin, DataEnricher, etc.)",
        lambda: __import__("src.plugins.base", fromlist=["IndicatorPlugin"]),
    )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    print("=" * 60)
    print("  Argus Trading Pipeline — Setup Verification")
    print("=" * 60)

    _check_imports()
    _check_env()
    _check_yfinance()
    _check_mlflow()
    _check_finbert()
    _check_src()

    print("\n" + "=" * 60)
    if _failures:
        print(f"  RESULT: {len(_failures)} check(s) FAILED:")
        for name in _failures:
            print(f"    - {name}")
        if _warnings:
            print(f"\n  {len(_warnings)} warning(s) (non-blocking):")
            for name in _warnings:
                print(f"    ~ {name}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("  RESULT: All required checks passed!")
        if _warnings:
            print(f"  {len(_warnings)} warning(s) (non-blocking):")
            for name in _warnings:
                print(f"    ~ {name}")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
