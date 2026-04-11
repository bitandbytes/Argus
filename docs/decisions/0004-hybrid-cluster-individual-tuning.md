# ADR-0004: Hybrid Cluster + Individual Tuning

## Status
Accepted

## Context

Each stock has unique behavior: AAPL responds to momentum signals that fail on utility stocks, and parameters optimal for 2023 may fail in 2025. We need to tune indicator and model parameters on a per-stock basis.

But naive per-stock tuning is dangerous. Bailey & López de Prado showed that with 5 years of daily data, testing more than ~45 parameter combinations can produce a backtest Sharpe ratio of 1.0+ purely by chance. With 10–12 tunable parameters and 100 Optuna trials, individual stock tuning has high overfitting risk for stocks with limited history.

The user originally framed this as "each stock is unique" and asked whether clustering or individual tuning is better. Research shows:
- Cluster-based tuning (López de Prado, ScienceDirect 2023 study on Russell 3000) outperforms universal models AND prevents overfitting.
- Pooled cluster data gives 5–10× more training observations per parameter set.
- BUT stocks with deep history (10+ years, 2500+ daily bars) can support individual tuning if validated rigorously.

## Decision

Use a **hybrid approach** with cluster-based defaults and individual promotion:

1. **Cluster-based tuning is the safe baseline.** All stocks begin with parameters optimized across their behavioral cluster (K-Means or DTW on Hurst exponent, ADX, autocorrelation, volatility features). Pooled data prevents overfitting.

2. **Per-stock promotion when statistically justified.** Stocks meeting strict criteria are promoted to individual parameters:
   - History ≥ 2500 daily bars (~10 years)
   - Individual OOS Sharpe > Cluster OOS Sharpe × 1.20 (significant improvement, not marginal)
   - Parameter stability < 20% drift between adjacent walk-forward windows
   - PBO (Probability of Backtest Overfitting) < 0.40
   - Minimum 30 trades per OOS window

3. **Automatic demotion on degradation.** If a promoted stock's rolling 60-day live Sharpe drops more than 0.30 below the cluster baseline, it reverts to cluster parameters.

Parameter loading priority: `stock_overrides/{ticker}.yaml` → `cluster_params/cluster_{id}.yaml`.

## Consequences

**Positive:**
- **Honors the "each stock is unique" insight** without paying the overfitting tax for stocks with insufficient history.
- **Statistically sound**: Cluster pooling provides enough data to validate individual parameters with confidence.
- **Self-correcting**: Demotion logic prevents stale individual parameters from degrading over time.
- **Expected distribution**: ~25% of liquid large-caps with deep history get individual params; ~75% use cluster defaults.

**Negative:**
- More complex than pure clustering or pure individual tuning.
- Requires maintaining two parameter sources (cluster + override) and a promotion log.
- Promotion gate logic must be carefully tested to avoid both over-promotion and under-promotion.

**Trade-offs accepted:**
- Implementation complexity in exchange for both robustness AND adaptation to individual stock behavior.
