---
name: quant-engine-dev
description: "Use this skill when modifying the QuantEngine, composite signal logic, regime-based indicator weights, or signal generation flow. Triggers on: 'modify the quant engine', 'change indicator weights', 'tune the composite signal', 'add multi-timeframe confirmation', 'adjust regime weights', or any task touching the core signal generation logic in src/signals/quant_engine.py. Do NOT use for adding new indicators (use plugin-author) or for ML meta-model work (use ml-meta-labeler)."
---

# Quant Engine Development Skill

This skill guides Claude in modifying the QuantEngine — the heart of the trading pipeline that combines plugin-based indicator signals into a directional trade signal. The QuantEngine is Layer 2 of the cascade architecture: it produces high-recall direction signals that the ML meta-model (Layer 3) then filters for precision.

## When to Use This Skill

Use this skill when:
- Modifying composite signal calculation logic
- Adjusting regime-dependent indicator weights
- Adding or changing multi-timeframe confirmation logic
- Implementing the signal generation flow
- Debugging unexpected signal output
- Tuning the entry/exit confidence thresholds

Do NOT use this skill for:
- Adding new indicators (that's the `plugin-author` skill)
- Changing the meta-labeling model (that's `ml-meta-labeler`)
- Backtesting (that's `backtest-runner`)

## Architectural Principles

### Principle 1: The QuantEngine never imports specific plugins
The engine iterates over `IndicatorPlugin` instances retrieved from the `PluginRegistry`. It must work with whatever indicators are registered in `config/plugins.yaml` — no hardcoded indicator names.

### Principle 2: Composite signal is always a weighted sum
The composite is computed as `Σ(weight_i × normalized_score_i)`, where weights depend on the current regime. The result must be in `[-1, +1]`. The confidence is `|composite|`.

### Principle 3: Weights sum to 1.0 per regime
Each regime's weight dictionary must sum to 1.0 (or be normalized to sum to 1.0). This ensures the composite stays in `[-1, +1]` when all indicators agree.

### Principle 4: Multi-timeframe confirmation only boosts confidence
When daily and weekly signals agree, multiply confidence by `multi_timeframe_boost` (default 1.15). Never let this push confidence above 1.0.

## QuantEngine Class Structure

```python
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import yaml
from src.plugins.registry import PluginRegistry
from src.plugins.base import IndicatorPlugin
from src.models.trade_signal import TradeSignal
from src.models.regime import RegimeType

class QuantEngine:
    """
    Layer 2 — Classical quant signal generator.
    
    Combines plugin-based indicator signals into a composite directional signal
    weighted by the current market regime. The engine is plugin-agnostic: it
    works with any IndicatorPlugin registered in the PluginRegistry.
    """
    
    def __init__(self, registry: PluginRegistry, config_path: str):
        self.registry = registry
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Load all active indicators from registry
        self.indicators: list[IndicatorPlugin] = registry.get_all_indicators()
        
        # Regime-dependent weights — loaded per stock from cluster_params or stock_overrides
        self.regime_weights: Dict[RegimeType, Dict[str, float]] = self._load_regime_weights()
        
        # Confidence thresholds
        self.entry_threshold = self.config["quant"]["entry_confidence_threshold"]
        self.exit_threshold = self.config["quant"]["exit_confidence_threshold"]
        self.mtf_boost = self.config["quant"]["multi_timeframe_boost"]
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        regime: RegimeType,
        ticker: str,
    ) -> TradeSignal:
        """
        Generate a trade signal for the given ticker at the latest bar.
        
        Args:
            df: OHLCV DataFrame, indexed by date, with all indicator columns already computed
            regime: Current market regime for this stock
            ticker: Stock ticker symbol
        
        Returns:
            TradeSignal with direction in [-1, +1] and confidence in [0, 1]
        """
        # Step 1: Compute normalized scores from each active indicator
        scores: Dict[str, float] = {}
        for indicator in self.indicators:
            params = self._get_params_for_indicator(ticker, indicator.name)
            df_with_indicator = indicator.compute(df, params)
            normalized = indicator.normalize(df_with_indicator[self._get_indicator_output_col(indicator)])
            scores[indicator.name] = normalized.iloc[-1]  # Latest value only
        
        # Step 2: Apply regime-specific weights
        weights = self.regime_weights[regime]
        composite = sum(
            scores[name] * weights.get(name, 0.0)
            for name in scores
        )
        composite = max(-1.0, min(1.0, composite))  # Clip to [-1, +1]
        
        # Step 3: Multi-timeframe confirmation
        confidence = abs(composite)
        if self._weekly_confirms(df, composite):
            confidence = min(1.0, confidence * self.mtf_boost)
        
        return TradeSignal(
            ticker=ticker,
            timestamp=df.index[-1],
            direction=composite,
            confidence=confidence,
            source_layer="quant",
            regime=regime,
            features=scores,
            stop_loss_pct=None,  # Set later by RiskManager
            take_profit_pct=None,
            bet_size=None,
            llm_approved=None,
        )
    
    def _weekly_confirms(self, df: pd.DataFrame, daily_composite: float) -> bool:
        """Check if weekly timeframe agrees with daily signal direction."""
        weekly = df["close"].resample("W").last()
        if len(weekly) < 20:
            return False
        weekly_sma_short = weekly.rolling(4).mean().iloc[-1]
        weekly_sma_long = weekly.rolling(10).mean().iloc[-1]
        weekly_direction = 1 if weekly_sma_short > weekly_sma_long else -1
        return (weekly_direction > 0 and daily_composite > 0) or \
               (weekly_direction < 0 and daily_composite < 0)
```

## Regime Weights Configuration

Regime weights live in `config/cluster_params/cluster_{id}.yaml` (cluster default) or `config/stock_overrides/{ticker}.yaml` (individual override). Format:

```yaml
indicators:
  weights:
    TRENDING_UP:
      sma_crossover: 0.35
      rsi: 0.15
      macd: 0.30
      bollinger: 0.10
      donchian: 0.05
      volume: 0.05
    TRENDING_DOWN:
      sma_crossover: 0.35
      rsi: 0.15
      macd: 0.30
      bollinger: 0.10
      donchian: 0.05
      volume: 0.05
    RANGING:
      sma_crossover: 0.10
      rsi: 0.30
      macd: 0.15
      bollinger: 0.30
      donchian: 0.05
      volume: 0.10
    VOLATILE:
      sma_crossover: 0.20
      rsi: 0.20
      macd: 0.20
      bollinger: 0.20
      donchian: 0.10
      volume: 0.10
```

**Validation**: Always verify weights sum to 1.0 (or normalize them) when loading. Use `assert abs(sum(weights.values()) - 1.0) < 1e-6`.

## Indicator Categories and Their Roles

When choosing weights for a regime, consider what each indicator category contributes:

| Category | Examples | Best in regime |
|----------|----------|----------------|
| **Trend** | SMA crossover, MACD, ADX | Trending (up or down) |
| **Momentum** | RSI, Stochastic, Williams %R | Both — but signals reverse meaning between regimes |
| **Volatility** | Bollinger Bands, ATR, Keltner | Ranging (mean reversion) |
| **Volume** | OBV, MFI, A/D Line | All regimes (confirmation only) |
| **Channel** | Donchian, Price Channel | Trending (breakouts) |

A balanced regime weight set has at least one indicator from trend, momentum, and volatility categories. Concentrating weight on one category increases the risk of false signals when that category misfires.

## Common Modifications

### Modification 1: Add multi-timeframe weekly confirmation
Already implemented above. The `_weekly_confirms()` method checks if weekly trend agrees with daily signal. When they agree, confidence gets a `multi_timeframe_boost` multiplier (default 1.15).

### Modification 2: Add a regime-specific exit signal
Mean-reversion regimes need different exit logic than trending regimes:

```python
def should_exit(self, df: pd.DataFrame, position_direction: int, regime: RegimeType) -> bool:
    """Determine if we should exit an open position."""
    current_signal = self.generate_signal(df, regime, "")
    
    if regime in (RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN):
        # In trending regimes, hold until trend reverses (signal flips direction)
        return (position_direction > 0 and current_signal.direction < -0.2) or \
               (position_direction < 0 and current_signal.direction > 0.2)
    else:
        # In ranging/volatile, exit on mean reversion (signal weakens past threshold)
        return current_signal.confidence < self.exit_threshold
```

### Modification 3: Sentiment as a quant input
The FinBERT sentiment score is a feature in the FeatureVector. Treat it as another input to the composite, with regime-dependent weight. Example regime weights with sentiment:

```yaml
TRENDING_UP:
  sma_crossover: 0.30
  macd: 0.25
  rsi: 0.10
  bollinger: 0.10
  donchian: 0.05
  volume: 0.10
  sentiment: 0.10
```

In `generate_signal()`, treat `sentiment_score` as if it were another indicator's normalized output.

### Modification 4: Adding signal caching
For backtests, the QuantEngine may be called many times with overlapping data. Add LRU caching for indicator computation if profiling shows it's a bottleneck. Use `functools.lru_cache` only on pure functions; never cache methods that depend on instance state.

## Critical Rules

### Rule 1: Composite must always be in [-1, +1]
After computing the weighted sum, clip it explicitly: `composite = max(-1.0, min(1.0, composite))`. Do not assume weights sum exactly to 1.0.

### Rule 2: Confidence is always in [0, 1]
Confidence is `|composite|` after clipping. The multi-timeframe boost can push it up but never above 1.0.

### Rule 3: Never use future data
The QuantEngine only sees historical data up to and including the current bar. The latest bar's indicator values use only past data because the indicators themselves are anti-lookahead (see `plugin-author` skill).

### Rule 4: Direction sign matters
Positive direction = long signal (buy), negative = short signal (sell). The Risk Manager and Order Manager rely on this convention. Never invert the sign in the QuantEngine.

### Rule 5: TradeSignal must be complete
Every field in `TradeSignal` must be populated except `stop_loss_pct`, `take_profit_pct`, `bet_size`, and `llm_approved` — those are filled in by downstream layers (RiskManager, LLMValidator).

## Testing the QuantEngine

Unit tests should cover:

```python
def test_generate_signal_returns_valid_range():
    engine = QuantEngine(registry, "config/settings.yaml")
    df = load_sample_ohlcv("AAPL")
    signal = engine.generate_signal(df, RegimeType.TRENDING_UP, "AAPL")
    
    assert -1.0 <= signal.direction <= 1.0
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.confidence == abs(signal.direction) * (1 + ...)  # MTF boost factor

def test_regime_weights_sum_to_one():
    engine = QuantEngine(registry, "config/settings.yaml")
    for regime, weights in engine.regime_weights.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-6

def test_no_lookahead_bias():
    engine = QuantEngine(registry, "config/settings.yaml")
    df = load_sample_ohlcv("AAPL")
    
    signal_full = engine.generate_signal(df, RegimeType.TRENDING_UP, "AAPL")
    signal_truncated = engine.generate_signal(df.iloc[:-1], RegimeType.TRENDING_UP, "AAPL")
    
    # The signal at time t should not change when we add data at time t+1
    df_extended = pd.concat([df.iloc[:-1], df.iloc[-1:]])
    signal_re_extended = engine.generate_signal(df_extended, RegimeType.TRENDING_UP, "AAPL")
    
    assert signal_full.direction == signal_re_extended.direction
```

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Weights don't sum to 1.0 | Composite occasionally exceeds [-1, +1] | Validate weights on load; normalize if needed |
| Hardcoding indicator names in engine | New indicators don't appear in composite | Iterate over `self.indicators`, never reference by name |
| Forgetting to clip composite | Confidence > 1.0 in some cases | Always `max(-1, min(1, composite))` after weighting |
| Multi-timeframe check too aggressive | Most signals get boosted | Verify the confirmation logic actually requires alignment |
| Sentiment treated as a binary feature | Composite swings unrealistically on news | FinBERT outputs are floats in [-1, +1] — use as continuous input |
