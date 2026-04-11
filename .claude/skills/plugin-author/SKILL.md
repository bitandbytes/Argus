---
name: plugin-author
description: "Use this skill when creating any new plugin for the trading pipeline. This includes IndicatorPlugin (technical indicators like RSI, MACD), SmoothingPlugin (Kalman filters, exponential smoothers), DataEnricher (sentiment, cross-asset features, options flow), and SignalFilter (LLM validators, attention reweighting, RL position sizers). Triggers on: 'add a new indicator', 'create a plugin', 'implement a Kalman filter', 'add sentiment from X', 'add a signal filter', or any task that mentions extending the pipeline with a new component. Do NOT use for modifying core pipeline code (use quant-engine-dev instead) or for backtest workflows (use backtest-runner)."
---

# Plugin Author Skill

This skill guides Claude in creating new plugins for the trading pipeline. Every component that processes data (indicators, smoothers, enrichers, signal filters) is a plugin. The core pipeline never imports plugin implementations directly — plugins are discovered through `config/plugins.yaml`.

## When to Use This Skill

Use this skill whenever the task involves adding or modifying a plugin. Examples:
- "Add an Aroon indicator"
- "Implement a Kalman filter for price smoothing"
- "Add cross-asset correlation features"
- "Create an LLM validator using Anthropic's API"

## Plugin Types and Their Interfaces

There are exactly four plugin types. Choose the right one based on what the plugin produces:

| Plugin Type | What it produces | When to use |
|-------------|------------------|-------------|
| `IndicatorPlugin` | A normalized signal in `[-1, +1]` from price/volume data | Technical indicators (RSI, MACD, Bollinger, etc.) |
| `SmoothingPlugin` | A smoothed series + velocity + noise estimate | Pre-processing raw price data (Kalman, EMA, Holt-Winters) |
| `DataEnricher` | Additional features added to the FeatureVector | External data (sentiment, options flow, macro indicators) |
| `SignalFilter` | Modified or filtered TradeSignal | Post-processing (LLM validation, attention reweighting, RL sizing) |

## Step-by-Step Plugin Creation

### Step 1: Choose the plugin type and category

Read `src/plugins/base.py` to see the abstract base class for the plugin type you're implementing. All four ABCs are defined there with full method signatures and docstrings.

### Step 2: Create the plugin file

Place the file in the correct subdirectory:
- `src/plugins/indicators/` for `IndicatorPlugin`
- `src/plugins/smoothing/` for `SmoothingPlugin`
- `src/plugins/enrichers/` for `DataEnricher`
- `src/plugins/filters/` for `SignalFilter`

Use a descriptive snake_case filename matching the plugin's primary purpose (e.g., `kalman.py`, `aroon.py`, `options_flow.py`).

### Step 3: Implement the plugin class

Every plugin must implement ALL abstract methods of its base class. For an `IndicatorPlugin`, that means:

```python
from src.plugins.base import IndicatorPlugin, ParamSpec
import pandas as pd
import pandas_ta as ta

class AroonIndicator(IndicatorPlugin):
    name = "aroon"
    category = "trend"  # "trend" | "momentum" | "volatility" | "volume" | "filter"
    version = "1.0.0"

    def compute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Compute the indicator and add columns to the dataframe."""
        period = params.get("period", 14)
        aroon = ta.aroon(df["high"], df["low"], length=period)
        df = df.copy()
        df["aroon_up"] = aroon[f"AROONU_{period}"]
        df["aroon_down"] = aroon[f"AROOND_{period}"]
        df["aroon_osc"] = df["aroon_up"] - df["aroon_down"]
        return df

    def normalize(self, values: pd.Series) -> pd.Series:
        """Normalize the indicator output to [-1, +1]."""
        # Aroon oscillator is already in [-100, 100], scale to [-1, 1]
        return (values / 100.0).clip(-1, 1)

    def get_tunable_params(self) -> dict:
        """Return the parameter search space for Optuna tuning."""
        return {
            "period": ParamSpec(
                name="period",
                type="int",
                low=7,
                high=30,
                default=14,
                description="Aroon calculation period in days",
            ),
        }

    def get_default_params(self) -> dict:
        return {"period": 14}
```

### Step 4: Register the plugin in config

Add an entry to `config/plugins.yaml`:

```yaml
indicators:
  enabled:
    # ... existing entries ...
    - name: "aroon"
      class: "src.plugins.indicators.aroon.AroonIndicator"
      active: true
```

Use the **fully qualified class path** — the registry uses Python's import machinery to load it.

### Step 5: Write unit tests

Create a test file in `tests/plugins/` matching the plugin location:

```python
# tests/plugins/indicators/test_aroon.py
import pytest
import pandas as pd
import numpy as np
from src.plugins.indicators.aroon import AroonIndicator

@pytest.fixture
def sample_ohlcv():
    """Generate 100 days of synthetic OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=100)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100))
    return pd.DataFrame({
        "open": close + np.random.randn(100) * 0.1,
        "high": close + abs(np.random.randn(100)),
        "low": close - abs(np.random.randn(100)),
        "close": close,
        "volume": np.random.randint(1_000_000, 5_000_000, 100),
    }, index=dates)

def test_aroon_compute(sample_ohlcv):
    indicator = AroonIndicator()
    result = indicator.compute(sample_ohlcv, indicator.get_default_params())
    assert "aroon_up" in result.columns
    assert "aroon_down" in result.columns
    assert "aroon_osc" in result.columns
    # Aroon up/down should be in [0, 100]
    assert result["aroon_up"].dropna().between(0, 100).all()

def test_aroon_normalize(sample_ohlcv):
    indicator = AroonIndicator()
    result = indicator.compute(sample_ohlcv, indicator.get_default_params())
    normalized = indicator.normalize(result["aroon_osc"])
    # Normalized values must be in [-1, +1]
    assert normalized.dropna().between(-1, 1).all()

def test_aroon_tunable_params():
    indicator = AroonIndicator()
    params = indicator.get_tunable_params()
    assert "period" in params
    assert params["period"].low >= 1
    assert params["period"].high <= 100
```

### Step 6: Verify the plugin is discoverable

Run a quick smoke test:

```bash
python -c "
from src.plugins.registry import PluginRegistry
registry = PluginRegistry()
registry.discover_plugins('config/plugins.yaml')
print(registry.list_available())
"
```

The new plugin should appear in the list.

## Critical Rules

### Rule 1: Anti-lookahead bias
Indicators must NEVER use future data. This means:
- No centered moving averages (`pd.Series.rolling(window=N, center=True)` is BANNED)
- All `.rolling()` calls must use the default `center=False`
- When in doubt, ask: "Could this code know today's value before today's close happened?"

### Rule 2: Output type contracts
- `IndicatorPlugin.compute()` must return a `pd.DataFrame` (the input df with new columns added).
- `IndicatorPlugin.normalize()` must return a `pd.Series` with values in `[-1, +1]`.
- `SmoothingPlugin.smooth()` must return a `SmoothResult` dataclass with all fields populated.
- `DataEnricher.enrich()` must return a `Dict[str, float]`.
- `SignalFilter.filter()` must return a modified `TradeSignal` (or the original unchanged).

### Rule 3: Parameters via dict
All tunable parameters are passed via the `params: dict` argument. Never use class attributes for tunable values — they need to be discoverable by the Optuna tuner.

### Rule 4: Stateless computation
Plugins should be stateless. The `compute()` method must produce the same output for the same input — no hidden state between calls. This is critical for backtesting reproducibility.

Exception: trained ML models (like FinBERT) can hold state, but it must be loaded from disk in `__init__`, not modified during `compute()`.

### Rule 5: Document the normalization rationale
The `normalize()` method's logic should be obvious from the code. Document what `+1`, `0`, and `-1` mean in a docstring:

```python
def normalize(self, values: pd.Series) -> pd.Series:
    """Normalize Aroon oscillator to [-1, +1].
    
    +1: Strong uptrend (Aroon Up >> Aroon Down)
    -1: Strong downtrend (Aroon Down >> Aroon Up)
     0: No clear trend (Aroon Up ≈ Aroon Down)
    """
    return (values / 100.0).clip(-1, 1)
```

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Hardcoding parameter values inside `compute()` | Always read from `params` dict |
| Returning raw indicator values (not normalized) | The composite scorer expects `[-1, +1]` from `normalize()` |
| Importing the plugin in core pipeline code | Plugins are loaded by the registry — never `from src.plugins.indicators.X import Y` outside tests |
| Forgetting to add the plugin to `plugins.yaml` | The registry only loads what's in the config |
| Using `.rolling(center=True)` | This causes lookahead bias — use trailing windows only |
| Returning a `pd.Series` from `compute()` instead of a `pd.DataFrame` | The interface contract is DataFrame in, DataFrame out |

## After Creating the Plugin

1. Run the test suite: `pytest tests/plugins/indicators/test_aroon.py -v`
2. Verify the plugin loads: see Step 6 above.
3. Check that the plugin appears in MLflow runs after the next backtest.
4. Update `config/cluster_params/` if you want this indicator included in the default regime weights.
