# ADR-0003: Meta-Labeling Over Ensemble Approach

## Status
Accepted

## Context

When combining multiple predictive models (classical quant + ML), there are two main paradigms:

1. **Ensemble averaging**: Run multiple models in parallel and combine their predictions (voting, averaging, stacking). Each model independently predicts direction.
2. **Cascade/Meta-labeling** (López de Prado, *Advances in Financial Machine Learning* Ch. 3): A primary model predicts direction with high recall. A secondary "meta-model" then predicts whether to act on that primary signal — purely a precision filter.

The user originally suggested a fallback approach: run the ML model only when the quant model's confidence was low. Research showed this is close to (but not exactly) meta-labeling, and meta-labeling is the more principled formulation.

Key research findings:
- Meta-labeling improves F1-score by filtering false positives without losing true positives.
- Position sizing benefits enormously when separated from direction prediction (different decisions need different optimization objectives).
- Pure stacked ensembles can blur strong signals with weak ones in noisy financial data.

## Decision

Use **meta-labeling as the primary architecture pattern** for combining the quant engine (Layer 2) with the ML model (Layer 3):

- The quant engine predicts trade direction with high recall (confidence threshold tuned for recall).
- The ML meta-model receives the quant features + quant prediction as input, and outputs `P(quant_signal_is_correct)`.
- A trade is taken only if the meta-model's calibrated probability exceeds threshold AND uncertainty is low.
- Position size is derived from the meta-model's probability (Kelly fraction × confidence).

The ML model is **always-on**, not a fallback — but it never independently generates direction signals. Its only job is to filter and size the quant layer's signals.

## Consequences

**Positive:**
- **Theoretically grounded**: Established framework with empirical support (López de Prado, mlfinlab community).
- **Better precision**: Meta-model filters out false positives that pure quant signals would have triggered.
- **Interpretable**: The quant signal can be explained independently; the meta-model's role is clear (gatekeeper).
- **Information advantage**: A non-linear ML meta-model can exploit patterns that the linear-thinking quant rules miss.
- **Position sizing**: Meta-model probability directly maps to bet size, eliminating the need for a separate sizing model.

**Negative:**
- Requires careful label generation (triple-barrier method) which has its own complexity.
- Two models to train and maintain instead of one.
- Risk of over-filtering: aggressive meta-model thresholds can suppress too many trades.

**Trade-offs accepted:**
- Higher implementation complexity in exchange for higher Sharpe ratio and lower drawdowns.
- Meta-model training requires sufficient historical signals (~hundreds of trades minimum), which is fine for daily bars over 5+ years.
