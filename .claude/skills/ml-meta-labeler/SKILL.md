---
name: ml-meta-labeler
description: "Use this skill when working with the ML meta-labeling layer (Layer 3 of the cascade). Triggers on: 'train the meta-model', 'implement triple-barrier labeling', 'meta-labeling', 'XGBoost calibration', 'Platt scaling', 'compute trade labels', 'label trades for ML training', or any task involving the precision-filtering ML model that decides whether to act on quant signals. Do NOT use for the quant engine itself (use quant-engine-dev) or for plugin development (use plugin-author)."
---

# ML Meta-Labeler Skill

This skill guides Claude in implementing and working with the ML meta-labeling layer — Layer 3 of the cascade architecture. The meta-model does NOT predict trade direction. Instead, it evaluates the quant engine's signals and decides which ones to act on, improving precision by filtering false positives.

## When to Use This Skill

Use this skill when:
- Implementing triple-barrier labeling for training data
- Training the XGBoost meta-model
- Calibrating model probabilities (Platt scaling)
- Evaluating meta-model performance with purged k-fold CV
- Adjusting confidence/uncertainty thresholds
- Debugging why the meta-model is over- or under-filtering

## The Meta-Labeling Concept (López de Prado, AFML Ch. 3)

Standard ML for trading tries to predict direction directly: "Will this stock go up tomorrow?" This couples two distinct decisions — trade direction and trade quality — into a single model.

Meta-labeling separates them:
1. **Primary model** (our QuantEngine) predicts direction with high recall.
2. **Meta-model** (our XGBoost classifier) takes the primary's signal as input and predicts: "Is this signal a true positive or a false positive?"

The result: better precision, better Sharpe, and a clean separation of concerns. The quant engine remains interpretable and the ML model has a focused, well-defined task.

## Step 1: Triple-Barrier Labeling

The meta-model needs binary labels: "did this quant signal lead to a profitable trade?" The triple-barrier method (AFML Ch. 3) generates these labels realistically.

For each quant signal at time `t`, we set three barriers:
- **Take-profit barrier**: price moves up by `tp_pct` (or `tp_pct × ATR`)
- **Stop-loss barrier**: price moves down by `sl_pct` (or `sl_pct × ATR`)
- **Vertical barrier**: maximum holding period of `max_holding_days`

The label is determined by which barrier is hit first:
- TP barrier hit first → label = `1` (profitable)
- SL barrier hit first → label = `-1` (loss)
- Time expires (vertical barrier) → label = `0` (neutral, no clear outcome)

For meta-labeling, we convert this to binary: `meta_label = 1 if (label == sign(quant_direction)) else 0`. The meta-model predicts whether the quant signal direction was correct.

```python
import pandas as pd
import numpy as np

def triple_barrier_labels(
    prices: pd.Series,
    signal_times: pd.DatetimeIndex,
    signal_directions: pd.Series,
    tp_pct: float = 0.04,
    sl_pct: float = 0.02,
    max_holding_days: int = 20,
) -> pd.DataFrame:
    """
    Apply triple-barrier labeling to a series of quant signals.
    
    Args:
        prices: Daily close prices, indexed by date
        signal_times: Timestamps when quant signals occurred
        signal_directions: +1 for long, -1 for short, indexed by signal_times
        tp_pct: Take-profit barrier as fraction of entry price (e.g., 0.04 = 4%)
        sl_pct: Stop-loss barrier as fraction of entry price
        max_holding_days: Vertical barrier — max trading days to hold
    
    Returns:
        DataFrame with columns: entry_price, tp_price, sl_price, exit_time, exit_price, 
                                 label (1=TP, -1=SL, 0=timeout), meta_label (1=correct, 0=wrong)
    """
    results = []
    
    for entry_time in signal_times:
        if entry_time not in prices.index:
            continue
        
        direction = signal_directions[entry_time]
        entry_price = prices[entry_time]
        
        if direction > 0:  # Long
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:  # Short
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)
        
        # Window of future prices to check
        future_prices = prices[entry_time:].iloc[1:max_holding_days + 1]
        
        label = 0  # Default: timeout
        exit_time = future_prices.index[-1] if len(future_prices) > 0 else entry_time
        exit_price = future_prices.iloc[-1] if len(future_prices) > 0 else entry_price
        
        for t, p in future_prices.items():
            if direction > 0:
                if p >= tp_price:
                    label = 1
                    exit_time = t
                    exit_price = p
                    break
                elif p <= sl_price:
                    label = -1
                    exit_time = t
                    exit_price = p
                    break
            else:  # Short
                if p <= tp_price:
                    label = 1
                    exit_time = t
                    exit_price = p
                    break
                elif p >= sl_price:
                    label = -1
                    exit_time = t
                    exit_price = p
                    break
        
        # Meta-label: was the quant signal direction correct?
        meta_label = 1 if (label == 1) else 0
        
        results.append({
            "entry_time": entry_time,
            "entry_price": entry_price,
            "direction": direction,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "label": label,
            "meta_label": meta_label,
        })
    
    return pd.DataFrame(results)
```

**Critical**: Use ATR-based barriers (not fixed percentages) for cross-stock generalization. A 2% stop on a $500 stock is a different risk than a 2% stop on a $20 stock when measured by daily volatility.

## Step 2: Feature Engineering for the Meta-Model

The meta-model receives a superset of the quant engine's features, plus the quant signal itself:

```python
def assemble_meta_features(
    quant_features: pd.DataFrame,  # All indicator outputs from the quant engine
    quant_signals: pd.DataFrame,    # Direction + confidence at each signal time
    sentiment_features: pd.DataFrame,  # FinBERT outputs
    regime_features: pd.DataFrame,  # One-hot encoded regime
) -> pd.DataFrame:
    """Build the feature matrix for meta-model training."""
    features = pd.concat([
        quant_features,
        quant_signals[["direction", "confidence"]],  # Critical: quant prediction is a feature
        sentiment_features,
        regime_features,
    ], axis=1)
    
    return features.dropna()
```

The fact that `quant_signals.direction` and `quant_signals.confidence` are features is essential — this is what makes it "meta"-labeling.

## Step 3: XGBoost Training with Purged CV

```python
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from src.tuning.purged_cv import PurgedKFoldCV
import mlflow
import mlflow.xgboost

def train_meta_model(
    X: pd.DataFrame,
    y: pd.Series,  # Binary meta-labels
    embargo_days: int = 5,
) -> CalibratedClassifierCV:
    """Train and calibrate the XGBoost meta-model."""
    
    # Use purged k-fold CV to prevent label leakage
    cv = PurgedKFoldCV(n_splits=5, embargo_days=embargo_days)
    
    base_model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.01,
        n_estimators=500,
        reg_lambda=7.0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )
    
    # Train with calibration (Platt scaling)
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",  # Platt scaling
        cv=cv,
    )
    
    with mlflow.start_run(run_name="meta_model_training"):
        mlflow.log_params(base_model.get_params())
        
        calibrated_model.fit(X, y)
        
        # Evaluate
        y_pred_proba = calibrated_model.predict_proba(X)[:, 1]
        from sklearn.metrics import f1_score, brier_score_loss
        f1 = f1_score(y, (y_pred_proba > 0.5).astype(int))
        brier = brier_score_loss(y, y_pred_proba)
        
        mlflow.log_metrics({
            "f1_score": f1,
            "brier_score": brier,
        })
        
        mlflow.xgboost.log_model(base_model, "meta_model")
    
    return calibrated_model
```

## Step 4: Calibration Verification

A well-calibrated model's predicted probabilities should match observed frequencies. If the model says "70% probability of success", actual success rate should be ~70%.

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_calibration_curve(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(mean_pred, fraction_pos, "o-", label="Meta-model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.title("Calibration Curve")
    plt.savefig("data/results/calibration_curve.png")
```

If the curve deviates significantly from the diagonal, the model is poorly calibrated and confidence-based position sizing will be unreliable.

## Step 5: Confidence Thresholding for Trade Decisions

In production, the meta-model's calibrated probability is used to decide:
1. **Should we trade?** Only if `proba > confidence_threshold` (e.g., 0.55)
2. **How much?** Bet size proportional to `(proba - 0.5) × 2` × Kelly fraction

```python
def should_trade(meta_proba: float, uncertainty: float) -> tuple[bool, float]:
    """
    Decision logic combining calibrated probability and uncertainty.
    
    Returns:
        (should_trade, bet_size_fraction)
    """
    confidence_threshold = 0.55
    uncertainty_threshold = 0.15
    
    if meta_proba < confidence_threshold:
        return False, 0.0
    
    if uncertainty > uncertainty_threshold:
        return False, 0.0
    
    # Bet size scales with how much above the threshold we are
    bet_size = (meta_proba - 0.5) * 2  # Maps [0.5, 1.0] -> [0.0, 1.0]
    bet_size = min(bet_size, 1.0)
    
    return True, bet_size
```

## Critical Rules

### Rule 1: Never train and test on overlapping data
Triple-barrier labels have label overlap (a label generated at time `t` depends on prices up to `t + max_holding_days`). Standard k-fold CV WILL leak information across folds. You MUST use purged k-fold CV with an embargo gap equal to `max_holding_days`.

### Rule 2: Class imbalance is the norm, not exception
Quant signals are designed for high recall, which means many are false positives. Expect meta-labels to be ~30-40% positive class (and 60-70% negative). Use `scale_pos_weight` in XGBoost or class weights to handle this.

### Rule 3: Calibrate, then threshold
Raw XGBoost output probabilities are not well-calibrated (they tend to be sigmoidal). Always wrap in `CalibratedClassifierCV` before using probabilities for trade decisions. Without calibration, "70% probability" might actually be ~85% true frequency, leading to over-confident position sizing.

### Rule 4: Use the F1-score (or Sharpe), not accuracy
Accuracy is misleading for imbalanced data. F1-score balances precision and recall. For trading, the ultimate metric is OOS Sharpe ratio of the filtered strategy — accuracy is just a proxy.

### Rule 5: Retrain monthly
The meta-model captures the relationship between quant features and trade success. This relationship drifts over time as market conditions change. Retrain monthly on the latest 252 days of signals. Compare new model OOS performance against the deployed model before promoting.

### Rule 6: Version every model
Save trained models to `data/models/meta_model/v{N}_{date}.pkl`. Log to MLflow with the version tag. Never overwrite a production model without keeping the previous version available for rollback.

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Standard k-fold leaks future info | Suspiciously high in-sample F1, poor live | Use PurgedKFoldCV with embargo |
| No calibration | Bet sizes are too aggressive | Wrap in CalibratedClassifierCV |
| Over-filtering signals | Very few trades, missed opportunities | Lower confidence threshold from 0.55 to 0.50 |
| Under-filtering | Too many false positive trades | Raise confidence threshold to 0.60 |
| Forgetting quant features in meta-features | Meta-model has nothing to learn from | Always include quant_direction and quant_confidence as features |
| Training on too little data | High variance in OOS metrics | Need at least 200-500 historical signals per training run |

## Workflow Checklist

When training a new meta-model version:
1. [ ] Generate quant signals over the latest 252 days
2. [ ] Apply triple-barrier labeling to compute meta-labels
3. [ ] Assemble feature matrix (quant features + sentiment + regime + quant prediction)
4. [ ] Verify class balance (should be 30-50% positive)
5. [ ] Train XGBoost with PurgedKFoldCV (embargo = max_holding_days)
6. [ ] Apply Platt calibration
7. [ ] Plot calibration curve and verify it tracks the diagonal
8. [ ] Compute F1, Brier score, and OOS Sharpe ratio of the filtered strategy
9. [ ] Compare against currently deployed model
10. [ ] If new model is better, log to MLflow and update production reference
11. [ ] Otherwise, keep current model and log the failed retraining attempt
