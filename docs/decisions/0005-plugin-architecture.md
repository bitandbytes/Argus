# ADR-0005: Plugin Architecture for Extensibility

## Status
Accepted

## Context

The architecture document identifies 8 future enhancements (Phase 4): Kalman filter smoothing, fractional differentiation, attention-based weighting, cross-asset correlation, options flow signals, RL position sizing, adaptive exits, self-hosted LLM. Each is independent and can be developed at any time.

If these enhancements were hardcoded into the core pipeline, adding any one of them would require:
- Modifying `QuantEngine` source code
- Modifying the feature pipeline
- Updating tests for unrelated components
- Potential merge conflicts with parallel work
- Risk of breaking existing functionality

This is exactly the situation a plugin architecture is designed to solve.

## Decision

Implement a **plugin architecture with four abstract base classes** in `src/plugins/base.py`:

1. **`IndicatorPlugin`** — Computes a technical indicator and normalizes it to `[-1, +1]`. Examples: RSI, MACD, MA crossover, fractional differentiation.

2. **`SmoothingPlugin`** — Pre-processes price/volume series before indicator calculation. Returns a `SmoothResult` with smoothed series, velocity, noise estimate, and confidence. Examples: Kalman filter, Unscented Kalman, exponential smoothing.

3. **`DataEnricher`** — Adds external context features to the feature vector. Examples: FinBERT sentiment, cross-asset correlation, options flow, macro indicators.

4. **`SignalFilter`** — Post-processes signals at one of three stages: `pre_quant`, `post_quant`, `post_meta`. Examples: LLM validator, attention-based reweighter, RL position sizer.

A `PluginRegistry` class discovers plugins at startup from `config/plugins.yaml`. The core pipeline (`QuantEngine`, `MetaLabelModel`, etc.) iterates over registered plugins by interface type, never importing specific implementations.

Adding a new plugin requires:
1. Writing one class file implementing the relevant interface.
2. Adding one line to `config/plugins.yaml`.
3. Writing unit tests for the plugin.

Zero changes to core pipeline code.

## Consequences

**Positive:**
- **Extensibility without modification**: Phase 4 enhancements can be added by writing isolated plugin files.
- **Testability**: Each plugin is independently testable with simple unit tests against its interface.
- **Configuration-driven**: Toggling plugins on/off is a config change, not a code change.
- **Parallel development**: Multiple developers can work on different plugins without merge conflicts.
- **Experimentation**: Easy to A/B test new indicators or smoothers by toggling them in config.
- **Versioning**: Each plugin can declare its own version for tracking changes.

**Negative:**
- **Indirection cost**: Plugin discovery adds a layer of indirection compared to direct imports.
- **Interface stability**: Once published, the abstract base class signatures must remain stable or all plugins break.
- **Discovery overhead**: Startup time includes plugin discovery (negligible for daily pipeline; would matter for HFT).

**Mitigation:**
- Define abstract interfaces carefully upfront based on the 8 known Phase 4 enhancements to minimize the need for breaking changes.
- Use Python's `abc.ABC` to enforce interface compliance at class definition time.
- Provide a `plugin-author` skill in `.claude/skills/` to make new plugin development straightforward.
