---
name: backtest-runner
description: "Use this skill when running backtests, walk-forward optimization, or any historical strategy evaluation. Triggers on: 'run a backtest', 'backtest this strategy', 'walk-forward test', 'evaluate the strategy on historical data', 'check if there's lookahead bias', 'compute Sharpe ratio', or any task involving PyBroker, VectorBT, or historical strategy validation. Also use when computing Probability of Backtest Overfitting (PBO) or running purged cross-validation. Do NOT use for live/paper trading (use the order management workflow) or for plugin development (use plugin-author)."
---

# Backtest Runner Skill

This skill guides Claude in running backtests, walk-forward optimization, and validating strategies against historical data. The pipeline uses two backtesting libraries with distinct purposes.

## When to Use This Skill

Use this skill when:
- Running a backtest of the quant engine or full pipeline
- Performing walk-forward optimization with Optuna
- Computing strategy metrics (Sharpe, drawdown, win rate, profit factor)
- Validating against lookahead bias
- Comparing parameter sets or model versions
- Computing Probability of Backtest Overfitting (PBO)

## Two Backtesting Tools — When to Use Which

| Tool | Use For | Why |
|------|---------|-----|
| **PyBroker** | Full pipeline backtests, walk-forward optimization, ML meta-model evaluation | Event-driven, strict no-lookahead enforcement, native ML support |
| **VectorBT** | Parameter sweeps over a single indicator or simple rule | Vectorized, extremely fast for testing thousands of parameter combinations |

**Rule of thumb:** Use VectorBT for "what's the best RSI period for this stock?" and PyBroker for "how does the full multi-layer pipeline perform on this stock over 5 years?"

## PyBroker Workflow (Primary)

### Step 1: Set up the strategy

PyBroker uses an event-driven model where you define a strategy function that PyBroker calls for each historical bar. Here's the canonical pattern for our pipeline:

```python
import pybroker as pyb
from pybroker import Strategy, StrategyConfig
from src.signals.quant_engine import QuantEngine
from src.signals.regime_detector import RegimeDetector
from src.plugins.registry import PluginRegistry
import mlflow

def run_backtest(ticker: str, start: str, end: str):
    # Load plugins and engine
    registry = PluginRegistry()
    registry.discover_plugins("config/plugins.yaml")
    
    quant_engine = QuantEngine(registry, config_path="config/settings.yaml")
    regime_detector = RegimeDetector(config_path="config/settings.yaml")
    
    def strategy_fn(ctx):
        """Called by PyBroker for each historical bar."""
        # ctx.bars gives access to all data up to (and including) the current bar
        # CRITICAL: do not access future data — PyBroker enforces this
        
        if len(ctx.bars) < 200:  # Need warmup
            return
        
        df = ctx.bars
        regime = regime_detector.detect(df)
        signal = quant_engine.generate_signal(df, regime)
        
        if signal.confidence > 0.30:
            if signal.direction > 0 and not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.stop_loss_pct = 2.0
                ctx.take_profit_pct = 4.0
            elif signal.direction < 0 and ctx.long_pos():
                ctx.sell_all_shares()
    
    # Configure strategy with realistic costs
    config = StrategyConfig(
        initial_cash=100_000,
        fee_mode=pyb.FeeMode.PER_SHARE,
        fee_amount=0.005,  # $0.005 per share commission
    )
    
    strategy = Strategy(
        data_source=pyb.YFinance(),
        start_date=start,
        end_date=end,
        config=config,
    )
    strategy.add_execution(strategy_fn, [ticker])
    
    # Wrap in MLflow run
    with mlflow.start_run(run_name=f"backtest_{ticker}_{start}_{end}"):
        mlflow.log_params({
            "ticker": ticker,
            "start": start,
            "end": end,
            "initial_cash": 100_000,
        })
        
        result = strategy.backtest()
        
        mlflow.log_metrics({
            "sharpe_ratio": result.metrics.sharpe,
            "max_drawdown": result.metrics.max_drawdown,
            "total_return": result.metrics.total_return,
            "win_rate": result.metrics.win_rate,
            "profit_factor": result.metrics.profit_factor,
            "num_trades": len(result.trades),
        })
        
        return result
```

### Step 2: Run the backtest

Use `scripts/run_backtest.py` (create this file if it doesn't exist):

```bash
python scripts/run_backtest.py --ticker AAPL --start 2020-01-01 --end 2024-12-31
```

### Step 3: Inspect results in MLflow

```bash
mlflow ui
# Open http://localhost:5000 in browser
```

You can compare runs side-by-side, plot metrics, and download artifacts.

## Walk-Forward Optimization Workflow

Walk-forward optimization is the gold standard for parameter tuning without overfitting. The approach:

1. Split historical data into rolling windows (e.g., 252-day in-sample + 126-day out-of-sample).
2. Optimize parameters on each in-sample window using Optuna.
3. Evaluate the best parameters on the immediately following out-of-sample window.
4. Roll forward and repeat.
5. Concatenate all out-of-sample results to compute the true strategy performance.

```python
from src.tuning.walk_forward import WalkForwardOptimizer
from src.tuning.bayesian_tuner import BayesianTuner
import optuna

def run_wfo(ticker: str, in_sample_days: int = 252, oos_days: int = 126):
    optimizer = WalkForwardOptimizer(
        in_sample_days=in_sample_days,
        out_of_sample_days=oos_days,
        n_trials=100,
    )
    
    def objective(trial, train_data):
        # Suggest parameters from each plugin's tunable space
        params = {
            "rsi_period": trial.suggest_int("rsi_period", 7, 21),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 40),
            "fast_ma": trial.suggest_int("fast_ma", 5, 50),
            "slow_ma": trial.suggest_int("slow_ma", 20, 200),
        }
        if params["fast_ma"] >= params["slow_ma"]:
            return float("-inf")  # Invalid combination
        
        # Run backtest with these params on training data only
        result = run_backtest_with_params(ticker, train_data, params)
        return result.metrics.sharpe
    
    results = optimizer.optimize(ticker, objective)
    
    # Stability check
    if not optimizer.stability_check(results.param_history):
        print("WARNING: Parameters unstable across windows — possible overfitting")
    
    return results
```

## Anti-Lookahead Validation

Before trusting any backtest result, verify:

### Check 1: Code review for lookahead bias
- All `.rolling()` calls use `center=False` (the default)
- No use of `.shift(-N)` (negative shift looks at future)
- No features computed using data the model wouldn't have at the time

### Check 2: Walk-forward must show OOS degradation
If your in-sample Sharpe is 2.5 and your out-of-sample Sharpe is also 2.5, something is wrong. Real strategies almost always show some degradation. Suspicious zero-degradation is usually a sign of:
- Survivorship bias (only trading currently-listed stocks)
- Data leakage in feature engineering
- Hyperparameter selection on the test set

### Check 3: Compute Probability of Backtest Overfitting (PBO)

Use Combinatorial Purged Cross-Validation (CPCV) to compute PBO. If PBO > 0.50, the strategy is more likely to fail live than succeed:

```python
from src.tuning.purged_cv import CombinatorialPurgedCV

cpcv = CombinatorialPurgedCV(n_groups=10, n_test_groups=2, embargo_days=5)
results = cpcv.run(strategy_fn, data)
pbo = cpcv.compute_pbo(results)
print(f"PBO: {pbo:.2f}")
assert pbo < 0.40, "Strategy likely overfit — do not deploy"
```

## VectorBT for Parameter Sweeps

When you want to find the best parameters for a single indicator quickly, VectorBT is much faster:

```python
import vectorbt as vbt
import numpy as np

# Test all combinations of fast/slow MA periods
fast_periods = np.arange(5, 50, 5)
slow_periods = np.arange(20, 200, 10)

fast_ma = vbt.MA.run(close_prices, fast_periods, short_name="fast")
slow_ma = vbt.MA.run(close_prices, slow_periods, short_name="slow")

entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

portfolio = vbt.Portfolio.from_signals(
    close_prices, entries, exits, fees=0.001
)

# Heatmap of Sharpe ratios across all combinations
sharpe = portfolio.sharpe_ratio()
print(sharpe.unstack().idxmax())  # Best combination
```

## Critical Rules

### Rule 1: Always wrap backtests in MLflow
Every backtest run must be tracked. Never run a backtest without `mlflow.start_run()` because results without tracking are lost.

### Rule 2: Realistic costs
- **Commission**: At least 0.05% per trade (Alpaca, IBKR retail rate)
- **Slippage**: At least 0.05% for liquid stocks, more for illiquid
- **Spread**: For limit orders, account for the bid-ask spread

Backtests without realistic costs typically overstate Sharpe by 0.5–1.0.

### Rule 3: Mind the warmup period
Most indicators need historical data to compute (e.g., SMA-200 needs 200 bars). Skip the first N bars in your strategy logic where N = max(indicator_lookback). PyBroker handles this naturally because the strategy function only fires once enough bars exist.

### Rule 4: Survivorship bias
yfinance only returns data for stocks that still exist. This means a backtest of "S&P 500 stocks" is biased: stocks that went bankrupt are excluded. For Phase 1 we accept this bias for the development watchlist. For Phase 2+ tuning, document this limitation.

### Rule 5: Never train and test on the same data
This is obvious but easy to violate accidentally. Use walk-forward or purged k-fold CV. Never use a single train/test split — financial time series violate the i.i.d. assumption.

## Metrics to Always Report

| Metric | What it measures | Good value |
|--------|------------------|-----------|
| Sharpe Ratio | Risk-adjusted return | > 1.0 acceptable, > 1.5 good |
| Max Drawdown | Worst peak-to-trough loss | < 25% for position trading |
| Win Rate | % of winning trades | > 50% (but profit factor matters more) |
| Profit Factor | Gross wins / gross losses | > 1.5 |
| Number of Trades | Sample size for statistical significance | > 30 per OOS window |
| Calmar Ratio | Annual return / max drawdown | > 0.5 |

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Lookahead bias | Backtest Sharpe is suspiciously high (> 3) | Code review for `.shift(-N)` and `center=True` |
| Survivorship bias | Strategy works in backtest, fails live | Use point-in-time universes for Phase 2+ |
| Overfitting from too many params | Great in-sample, poor OOS | Reduce parameter count, use walk-forward |
| Missing transaction costs | Sharpe drops significantly when costs added | Always include realistic costs |
| Over-optimization on a single backtest | "Best Sharpe of 100 backtests" ≠ "Sharpe of best strategy" | Use Bonferroni correction or PBO |

## After Running a Backtest

1. Open MLflow UI and review the run.
2. Check if Sharpe is in the expected range for the strategy.
3. Verify the equity curve doesn't have any suspicious jumps.
4. Compare against buy-and-hold benchmark.
5. If parameters were tuned, run walk-forward to validate stability.
6. Document any anomalies in `data/results/notes.md`.
