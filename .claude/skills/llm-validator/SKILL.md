---
name: llm-validator
description: "Use this skill when implementing or modifying the LLM Validator (Phase 4.2). Triggers on: 'implement LLMValidator', 'add GPT-4o-mini validation', 'OpenAI structured output for trading', 'LLM APPROVE/VETO gate', 'cache OpenAI responses', 'track token costs in MLflow', 'wire LLM validation into the pipeline', or any task involving the OpenAI API as a final signal filter before order execution. Do NOT use for FinBERT sentiment (use finbert-integration) or for general SignalFilter plugin creation (use plugin-author)."
---

# LLM Validator Skill

This skill guides Claude in implementing the `LLMValidator` — Layer 4 of the cascade. The validator is the most expensive component: each call costs real API money, so it runs only for signals that passed the ML meta-model filter. Its job is deep contextual analysis of news, upcoming earnings, and macro risk that the statistical layers cannot capture.

## When to Use This Skill

- Implementing `src/plugins/filters/llm_validator.py`
- Designing the OpenAI prompt and structured JSON schema
- Adding per-ticker-per-day response caching
- Tracking token costs in MLflow
- Writing tests with mocked OpenAI responses
- Tuning the APPROVE/VETO threshold

Do NOT use this skill for:
- FinBERT sentiment enrichment → use `finbert-integration`
- Generic `SignalFilter` plugins → use `plugin-author`
- Risk sizing or exit decisions → use `risk-manager`

## Architecture Context

The validator is a `SignalFilter` plugin instantiated as the last step before `RiskManager`:

```
QuantEngine (direction + confidence)
    → MetaLabelModel (trade/no-trade + bet_size)
        → LLMValidator (APPROVE / VETO + reasoning)  ← this skill
            → RiskManager (position sizing + execution)
```

Cost guardrail: the orchestrator only calls `LLMValidator.filter()` when `TradeSignal.bet_size > 0` (meta-model approved). Typical veto rate is 10–30%, so ~70–90% of meta-model approvals proceed to execution.

## Step 1: Implement LLMValidator as a SignalFilter

```python
# src/plugins/filters/llm_validator.py

import json
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import mlflow
from openai import OpenAI
from src.plugins.base import SignalFilter
from src.models.trade_signal import TradeSignal

logger = logging.getLogger(__name__)

@dataclass
class LLMDecision:
    decision: str          # "APPROVE" or "VETO"
    confidence: float      # 0.0–1.0 (LLM's self-assessed confidence)
    reasoning: str         # short plain-text explanation
    risk_flags: list[str]  # e.g. ["upcoming_earnings", "sector_headwinds"]
    tokens_used: int


class LLMValidator(SignalFilter):
    """OpenAI GPT-4o-mini validation gate for actionable trade signals.

    Implements SignalFilter. Returns the signal unchanged (approved) or
    with bet_size=0 and llm_approved=False (vetoed).

    Only called for signals where meta-model bet_size > 0.
    Cached per (ticker, date) to avoid redundant API calls.
    Token costs are logged to the active MLflow run.
    """

    name = "llm_validator"
    version = "1.0.0"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        cache_dir: str = "data/models/llm_cache",
        veto_threshold: float = 0.60,  # LLM confidence below which we veto
        max_context_days: int = 7,
    ):
        self.client = OpenAI()  # reads OPENAI_API_KEY from env
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.veto_threshold = veto_threshold
        self.max_context_days = max_context_days

    def filter(self, signal: TradeSignal, context: dict) -> TradeSignal:
        """Validate the signal. Returns signal unchanged (approve) or with bet_size=0 (veto).

        Args:
            signal: The TradeSignal from MetaLabelModel.
            context: Must contain:
                - "news_headlines": list[dict] with keys text, date, source
                - "earnings_calendar": dict[str, list[Timestamp]]
                - "sector": str (e.g. "Technology")
                - "date": pd.Timestamp (today's bar date)
        """
        if signal.bet_size is None or signal.bet_size <= 0:
            return signal  # meta-model already rejected — skip LLM call

        decision = self._get_decision(signal, context)
        self._log_cost(signal.ticker, decision.tokens_used)

        signal.llm_approved = decision.decision == "APPROVE"
        if not signal.llm_approved:
            signal.bet_size = 0.0
            signal.exit_reason = f"llm_veto: {decision.reasoning[:80]}"
            logger.info("LLM vetoed %s: %s", signal.ticker, decision.reasoning)
        else:
            logger.info("LLM approved %s (confidence %.2f)", signal.ticker, decision.confidence)

        return signal

    def _get_decision(self, signal: TradeSignal, context: dict) -> LLMDecision:
        """Return cached decision if available, otherwise call the API."""
        cache_key = self._cache_key(signal, context)
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            with open(cache_path) as f:
                raw = json.load(f)
            return LLMDecision(**raw)

        decision = self._call_api(signal, context)

        with open(cache_path, "w") as f:
            json.dump(decision.__dict__, f, indent=2)

        return decision

    def _call_api(self, signal: TradeSignal, context: dict) -> LLMDecision:
        """Build the prompt and call GPT-4o-mini with structured JSON output."""
        prompt = self._build_prompt(signal, context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior equity analyst reviewing a quantitative trade signal. "
                        "Your job is to identify qualitative risks that the statistical model "
                        "cannot detect. Respond ONLY in the JSON schema provided."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # deterministic
            max_tokens=300,
        )

        tokens = response.usage.total_tokens
        raw = json.loads(response.choices[0].message.content)

        return LLMDecision(
            decision=raw.get("decision", "VETO"),
            confidence=float(raw.get("confidence", 0.0)),
            reasoning=raw.get("reasoning", ""),
            risk_flags=raw.get("risk_flags", []),
            tokens_used=tokens,
        )

    def _build_prompt(self, signal: TradeSignal, context: dict) -> str:
        direction_str = "LONG (buy)" if signal.direction == 1 else "SHORT (sell)"
        headlines = context.get("news_headlines", [])
        recent = [h["text"] for h in headlines[-10:]]  # last 10 headlines

        earnings = context.get("earnings_calendar", {}).get(signal.ticker, [])
        earnings_str = ", ".join(str(d.date()) for d in earnings[:3]) if earnings else "None in 30 days"

        return f"""
Quantitative signal summary:
- Ticker: {signal.ticker}
- Direction: {direction_str}
- Quant confidence: {signal.confidence:.2f}
- ML meta-model bet size: {signal.bet_size:.2f}
- Regime: {signal.regime}
- Sector: {context.get("sector", "Unknown")}

Recent news headlines (last {len(recent)}):
{chr(10).join(f"- {h}" for h in recent)}

Upcoming earnings dates: {earnings_str}

Respond with JSON in this EXACT schema:
{{
  "decision": "APPROVE" or "VETO",
  "confidence": 0.0 to 1.0,
  "reasoning": "one or two sentences explaining the decision",
  "risk_flags": ["list", "of", "risk", "tags"]
}}

VETO the signal if any of these are true:
1. Earnings within 3 days that could gap against the position
2. Recent news shows fundamental business deterioration
3. Major macro headwind directly affecting this sector
4. Regulatory or legal overhang that the price hasn't reflected

APPROVE if none of the above apply and the news context supports the technical direction.
"""

    def _cache_key(self, signal: TradeSignal, context: dict) -> str:
        """Deterministic key from (ticker, date, direction)."""
        date_str = str(context.get("date", "unknown"))
        raw = f"{signal.ticker}|{date_str}|{signal.direction}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _log_cost(self, ticker: str, tokens: int) -> None:
        """Log token usage to the active MLflow run (if any)."""
        # GPT-4o-mini pricing: $0.15/1M input + $0.60/1M output (approximate)
        approx_cost_usd = tokens * 0.000_000_375  # rough blended rate
        try:
            mlflow.log_metrics({
                "llm_tokens_used": tokens,
                "llm_cost_usd": approx_cost_usd,
            })
        except Exception:
            pass  # not in an MLflow run — no-op
```

## Step 2: Register in config/plugins.yaml

```yaml
filters:
  enabled:
    - name: "llm_validator"
      class: "src.plugins.filters.llm_validator.LLMValidator"
      active: true
      config:
        model: "gpt-4o-mini"
        veto_threshold: 0.60
        max_context_days: 7
```

## Step 3: Wire into the pipeline orchestrator

```python
# src/pipeline.py — inside run_daily()
from src.plugins.filters.llm_validator import LLMValidator

for ticker in tickers_with_actionable_signals:
    signal = quant_engine.generate_signal(ticker, ...)
    signal = meta_model.evaluate(signal)

    if signal.bet_size and signal.bet_size > 0:
        context = {
            "news_headlines": news_provider.get_headlines(ticker, days=7),
            "earnings_calendar": market_data.get_earnings_calendar([ticker]),
            "sector": watchlist[ticker].get("sector", "Unknown"),
            "date": today,
        }
        signal = llm_validator.filter(signal, context)

    risk_manager.process(signal)
```

## Prompt Engineering Guidelines

The prompt must be **specific and bounded**. Vague prompts cause inconsistent decisions.

| Do | Don't |
|---|---|
| List concrete VETO conditions | Ask the LLM to "assess the trade" |
| Request JSON with a fixed schema | Use open-ended text output |
| Keep prompt under 800 tokens | Include full article text |
| Use `temperature=0.1` for consistency | Default temperature (0.7) causes variance |
| Include only the last 10 headlines | Dump all 30-day headlines |

**Risk flags vocabulary** (keep consistent across calls for analysis):
- `upcoming_earnings` — earnings within 3 trading days
- `negative_news_momentum` — more negative than positive headlines this week
- `regulatory_risk` — SEC, DOJ, CFPB mentions
- `sector_headwind` — macro force affecting the whole sector
- `fundamental_deterioration` — revenue / margin compression news
- `management_instability` — CEO departure, proxy fight

## Caching Strategy

The cache key is `SHA256(ticker + "|" + date + "|" + direction)[:16]`. This means:
- Re-running the pipeline on the same day for the same signal hits cache (free).
- A different date always calls the API (news context changes daily).
- Flipping direction (LONG → SHORT) calls the API (different risk profile).

Cache files are stored as `data/models/llm_cache/{key}.json`. They are safe to keep indefinitely — the date is encoded in the key, so stale cache entries for past dates are never re-read (future dates produce different keys).

## Cost Control

Typical costs per ticker per day (GPT-4o-mini):
- Prompt tokens: ~250–400
- Completion tokens: ~80–120
- Total: ~400–500 tokens ≈ $0.0002/call

For a 50-stock watchlist with a 20% signal rate and 80% meta-model pass-through:
- ~8 LLM calls/day → $0.002/day → **< $1/year**

If costs spike, check:
1. `max_tokens` cap in the API call (keep ≤ 300)
2. Cache hit rate — confirm `data/models/llm_cache/` is not being cleared
3. Signal rate — if QuantEngine produces too many signals, tighten confidence threshold

## Testing with Mocked OpenAI

```python
# tests/plugins/filters/test_llm_validator.py
import pytest
from unittest.mock import patch, MagicMock
from src.plugins.filters.llm_validator import LLMValidator
from src.models.trade_signal import TradeSignal, RegimeType
import pandas as pd

@pytest.fixture
def sample_signal():
    return TradeSignal(
        ticker="AAPL",
        direction=1,
        confidence=0.75,
        regime=RegimeType.TRENDING_UP,
        bet_size=0.5,
        llm_approved=None,
    )

@pytest.fixture
def sample_context():
    return {
        "news_headlines": [{"text": "Apple reports record iPhone sales", "date": "2024-01-15"}],
        "earnings_calendar": {"AAPL": []},
        "sector": "Technology",
        "date": pd.Timestamp("2024-01-15"),
    }

def test_approve_signal(sample_signal, sample_context, tmp_path):
    validator = LLMValidator(cache_dir=str(tmp_path))

    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"decision": "APPROVE", "confidence": 0.85, "reasoning": "Strong momentum, no near-term risks.", "risk_flags": []}'
    mock_response.usage.total_tokens = 400

    with patch.object(validator.client.chat.completions, "create", return_value=mock_response):
        result = validator.filter(sample_signal, sample_context)

    assert result.llm_approved is True
    assert result.bet_size == 0.5

def test_veto_signal(sample_signal, sample_context, tmp_path):
    validator = LLMValidator(cache_dir=str(tmp_path))

    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"decision": "VETO", "confidence": 0.3, "reasoning": "Earnings in 2 days.", "risk_flags": ["upcoming_earnings"]}'
    mock_response.usage.total_tokens = 350

    with patch.object(validator.client.chat.completions, "create", return_value=mock_response):
        result = validator.filter(sample_signal, sample_context)

    assert result.llm_approved is False
    assert result.bet_size == 0.0

def test_cache_prevents_second_api_call(sample_signal, sample_context, tmp_path):
    validator = LLMValidator(cache_dir=str(tmp_path))

    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"decision": "APPROVE", "confidence": 0.80, "reasoning": "OK.", "risk_flags": []}'
    mock_response.usage.total_tokens = 400

    with patch.object(validator.client.chat.completions, "create", return_value=mock_response) as mock_api:
        validator.filter(sample_signal, sample_context)
        validator.filter(sample_signal, sample_context)  # second call

    # API should only be called once; second call hits cache
    assert mock_api.call_count == 1

def test_skips_if_no_bet_size(sample_signal, sample_context, tmp_path):
    sample_signal.bet_size = 0.0
    validator = LLMValidator(cache_dir=str(tmp_path))

    with patch.object(validator.client.chat.completions, "create") as mock_api:
        validator.filter(sample_signal, sample_context)

    mock_api.assert_not_called()
```

## Validation Checklist

- [ ] API call uses `response_format={"type": "json_object"}` (prevents free-text output)
- [ ] `temperature=0.1` is set (consistency over creativity)
- [ ] Cache files appear in `data/models/llm_cache/` after first run
- [ ] Second call for same (ticker, date, direction) does NOT hit the API
- [ ] Vetoed signal has `bet_size=0.0` and `llm_approved=False`
- [ ] Token count is logged to MLflow (test with `mlflow.start_run()`)
- [ ] `max_tokens=300` cap is enforced in the API call
- [ ] Tests with mocked OpenAI cover APPROVE, VETO, and cache-hit paths
