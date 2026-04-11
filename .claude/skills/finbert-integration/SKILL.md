---
name: finbert-integration
description: "Use this skill when setting up, configuring, or working with FinBERT for financial sentiment analysis. Triggers on: 'set up FinBERT', 'sentiment analysis', 'FinBERT batch processing', 'compute sentiment features', 'add news to features', 'FinBERT enricher', or any task involving the local sentiment scoring layer (Layer 1). Do NOT use for the LLM Validator (that's a separate skill area) or for creating other types of plugins (use plugin-author)."
---

# FinBERT Integration Skill

This skill guides Claude in working with FinBERT — the local sentiment analysis model that forms Layer 1 of the cascade architecture. FinBERT runs as a `DataEnricher` plugin, processing news headlines in batch and adding sentiment features to the feature store.

## When to Use This Skill

Use this skill when:
- Setting up FinBERT for the first time
- Implementing the `FinBERTEnricher` plugin
- Computing rolling sentiment features
- Debugging slow inference or memory issues
- Switching between CPU and GPU inference
- Caching sentiment results
- Choosing news data sources

## Why FinBERT (Not GPT-4)

FinBERT processes thousands of headlines per minute on CPU at zero API cost. The architecture uses a two-tier sentiment approach:

1. **FinBERT (Layer 1)**: Processes ALL news for ALL tickers daily as a feature. Cheap, fast, runs locally.
2. **GPT-4o-mini (Layer 4)**: Deep analysis of news context, but only for tickers with actionable signals (~5-15% of universe).

This split reduces sentiment-related API costs by ~90% while still leveraging LLM reasoning for high-stakes decisions.

## Hardware Requirements

FinBERT is a BERT-base model with ~110M parameters:

| Configuration | RAM | Speed (8-core CPU) | Notes |
|---------------|-----|---------------------|-------|
| FP32 (default) | ~440 MB | ~25 headlines/sec | Works on any laptop |
| FP16 | ~210 MB | ~30 headlines/sec | Slightly faster, marginal accuracy loss |
| INT4 quantized | ~52 MB | ~40 headlines/sec | Significant speedup, minor accuracy hit |
| GPU (any) | ~440 MB VRAM | ~500+ headlines/sec | Near-instant for typical batch sizes |

For a watchlist of 100 tickers with ~10 headlines each per day, batch processing takes ~40 seconds on CPU. No GPU needed.

## Step 1: Install Dependencies

```bash
pip install transformers torch
```

The model will download automatically on first use (~440 MB) and cache in `~/.cache/huggingface/`.

## Step 2: Implement the FinBERTEnricher Plugin

```python
# src/plugins/enrichers/finbert.py

from typing import Dict, List
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from src.plugins.base import DataEnricher
from src.models.feature_vector import FeatureVector

class FinBERTEnricher(DataEnricher):
    """
    Local sentiment analysis using FinBERT (ProsusAI/finbert).
    
    Runs in batch mode on CPU. Processes thousands of headlines per minute.
    Outputs sentiment scores in [-1, +1] where:
        +1 = strongly positive
        -1 = strongly negative
         0 = neutral
    """
    
    name = "finbert"
    data_type = "sentiment"
    version = "1.0.0"
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "cpu", batch_size: int = 64):
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        # Load model and tokenizer (downloads on first use, cached afterward)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # FinBERT label order: [positive, negative, neutral]
        self.label_map = {0: "positive", 1: "negative", 2: "neutral"}
    
    def analyze_batch(self, headlines: List[str]) -> List[Dict[str, float]]:
        """
        Process a batch of headlines and return sentiment probabilities.
        
        Args:
            headlines: List of headline strings
        
        Returns:
            List of dicts with keys: positive, negative, neutral, score, confidence
        """
        if not headlines:
            return []
        
        results = []
        
        # Process in batches to avoid OOM
        for i in range(0, len(headlines), self.batch_size):
            batch = headlines[i : i + self.batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            for j, p in enumerate(probs.cpu().numpy()):
                pos, neg, neu = p[0], p[1], p[2]
                # Score in [-1, +1]: P(positive) - P(negative)
                score = float(pos - neg)
                # Confidence: max probability across all classes
                confidence = float(max(pos, neg, neu))
                
                results.append({
                    "headline": batch[j],
                    "positive": float(pos),
                    "negative": float(neg),
                    "neutral": float(neu),
                    "score": score,
                    "confidence": confidence,
                })
        
        return results
    
    def enrich(self, ticker: str, features: FeatureVector) -> Dict[str, float]:
        """
        Compute sentiment features for a ticker. This is the main interface
        called by the pipeline.
        """
        # Load headlines for this ticker (from your news provider)
        headlines = self._load_recent_headlines(ticker, days_back=30)
        
        if not headlines:
            return self._empty_features()
        
        # Score each headline
        scored = self.analyze_batch([h["text"] for h in headlines])
        
        # Aggregate into time-bucketed features
        scores_df = pd.DataFrame(scored)
        scores_df["timestamp"] = [h["timestamp"] for h in headlines]
        scores_df = scores_df.set_index("timestamp").sort_index()
        
        return self._compute_rolling_features(scores_df)
    
    def batch_enrich(self, tickers: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Process all tickers in a single pass — more efficient than calling
        enrich() in a loop because we batch all headlines together.
        """
        # Collect all headlines from all tickers
        all_headlines = []
        ticker_indices = {}  # Maps ticker -> (start_idx, end_idx) in all_headlines
        
        for ticker in tickers:
            headlines = self._load_recent_headlines(ticker, days_back=30)
            start = len(all_headlines)
            all_headlines.extend([h["text"] for h in headlines])
            ticker_indices[ticker] = (start, len(all_headlines), headlines)
        
        # One big batched inference
        scored = self.analyze_batch(all_headlines)
        
        # Split results back per ticker
        results = {}
        for ticker, (start, end, raw_headlines) in ticker_indices.items():
            ticker_scores = scored[start:end]
            scores_df = pd.DataFrame(ticker_scores)
            scores_df["timestamp"] = [h["timestamp"] for h in raw_headlines]
            scores_df = scores_df.set_index("timestamp").sort_index()
            results[ticker] = self._compute_rolling_features(scores_df)
        
        return results
    
    def _compute_rolling_features(self, scores_df: pd.DataFrame) -> Dict[str, float]:
        """Compute rolling aggregations from per-headline scores."""
        if scores_df.empty:
            return self._empty_features()
        
        # Daily aggregation: weighted average by confidence
        daily = scores_df.resample("D").apply(
            lambda x: (x["score"] * x["confidence"]).sum() / x["confidence"].sum()
            if x["confidence"].sum() > 0 else 0.0
        )
        
        latest = daily.iloc[-1] if len(daily) > 0 else 0.0
        ma_5d = daily.tail(5).mean() if len(daily) >= 5 else latest
        ma_20d = daily.tail(20).mean() if len(daily) >= 20 else latest
        momentum = ma_5d - ma_20d
        
        # News volume features
        news_count_today = len(scores_df[scores_df.index.date == pd.Timestamp.now().date()])
        news_count_avg = len(scores_df) / 30  # 30-day average
        news_volume_ratio = news_count_today / news_count_avg if news_count_avg > 0 else 0
        
        # Negative news ratio (heavily weighted negative outcomes)
        negative_ratio = (scores_df["score"] < -0.3).sum() / len(scores_df)
        
        return {
            "sentiment_score": float(latest),
            "sentiment_ma_5d": float(ma_5d),
            "sentiment_ma_20d": float(ma_20d),
            "sentiment_momentum": float(momentum),
            "news_volume_ratio": float(news_volume_ratio),
            "negative_news_ratio": float(negative_ratio),
        }
    
    def _empty_features(self) -> Dict[str, float]:
        """Return zero features when no headlines are available."""
        return {
            "sentiment_score": 0.0,
            "sentiment_ma_5d": 0.0,
            "sentiment_ma_20d": 0.0,
            "sentiment_momentum": 0.0,
            "news_volume_ratio": 0.0,
            "negative_news_ratio": 0.0,
        }
    
    def _load_recent_headlines(self, ticker: str, days_back: int) -> List[Dict]:
        """Load headlines from the news data store. Implementation depends on data source."""
        # TODO: Wire up to NewsDataProvider in src/data/news_data.py
        raise NotImplementedError("Implement based on chosen news provider")
```

## Step 3: News Data Sources

The plugin needs a news source. Options ranked by suitability for Phase 1:

| Source | Cost | Historical Data | Quality | Notes |
|--------|------|----------------|---------|-------|
| **Alpha Vantage NEWS_SENTIMENT** | Free tier (5 calls/min) | ~2 years | Good | Best Phase 1 option — pre-tagged with tickers |
| **Finnhub** | Free tier (60 calls/min) | ~1 year | Good | Per-ticker news endpoint |
| **NewsAPI.org** | Free tier (100/day dev only) | ~1 month | Mixed | Not suitable for production |
| **Polygon.io** | Paid ($30/mo) | Years | Excellent | Use for Phase 3+ |
| **Web scraping** | Free | Variable | Risky | Maintenance burden, fragile |

**Recommendation for Phase 1**: Skip historical news for backtesting. Run the FinBERT enricher live from Phase 3 onward, accumulating sentiment history forward. Backtest the quant + ML layers on price data only, then validate the sentiment value-add in paper trading.

## Step 4: Caching Strategy

FinBERT inference is fast but not free. Cache results to avoid re-processing the same headlines:

```python
import hashlib
from pathlib import Path

class FinBERTEnricher(DataEnricher):
    def __init__(self, ..., cache_dir: str = "data/models/finbert_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_key(self, headline: str) -> str:
        """Stable hash of headline text."""
        return hashlib.sha256(headline.encode()).hexdigest()[:16]
    
    def analyze_batch_cached(self, headlines: List[str]) -> List[Dict[str, float]]:
        """Check cache before running inference."""
        results = []
        to_process = []
        to_process_indices = []
        
        for i, h in enumerate(headlines):
            cache_path = self.cache_dir / f"{self._cache_key(h)}.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    results.append(json.load(f))
            else:
                results.append(None)
                to_process.append(h)
                to_process_indices.append(i)
        
        if to_process:
            new_results = self.analyze_batch(to_process)
            for idx, result in zip(to_process_indices, new_results):
                results[idx] = result
                # Persist to cache
                cache_path = self.cache_dir / f"{self._cache_key(headlines[idx])}.json"
                with open(cache_path, "w") as f:
                    json.dump(result, f)
        
        return results
```

This makes incremental daily runs essentially free — only new headlines hit the model.

## Critical Rules

### Rule 1: FinBERT outputs three classes, not two
The model outputs probabilities for `[positive, negative, neutral]`. The score is `P(pos) - P(neg)`, NOT `2 * P(pos) - 1`. The neutral class matters and shouldn't be ignored.

### Rule 2: Confidence-weighted aggregation
When aggregating multiple headlines, weight by confidence (max class probability). A headline that scores `(0.4, 0.4, 0.2)` is essentially noise — it shouldn't contribute equally to the daily sentiment with one that scores `(0.9, 0.05, 0.05)`.

### Rule 3: Truncate to 128 tokens
FinBERT was trained on sentences, not paragraphs. Headlines are usually < 30 tokens, but full articles can exceed BERT's 512 limit. Truncate to 128 tokens max for headlines, or split articles into sentences first.

### Rule 4: Model evaluation mode
Always call `model.eval()` and use `torch.no_grad()` for inference. Otherwise PyTorch tracks gradients (slower) and dropout layers stay active (incorrect output).

### Rule 5: Batch for throughput
Single-headline inference is slow because of fixed overhead. Always batch (64-128 headlines per call) for production. The provided implementation handles this.

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Calling on every headline individually | Slow | Use `analyze_batch()` with batch_size=64 |
| Forgetting `.eval()` mode | Dropout active, inconsistent output | Always `model.eval()` after loading |
| Treating neutral as zero | Lose information about uncertain headlines | Use confidence-weighted aggregation |
| Re-loading model on every call | Massive slowdown | Load once in `__init__`, reuse for lifetime of plugin |
| Long article truncation losing context | Sentiment doesn't match article tone | Split long articles into sentences, score each |
| No cache → re-processing same headlines | Wasted compute | Implement cache (see Step 4) |

## Validation Checklist

After implementing FinBERTEnricher:
- [ ] Model loads without error on first use
- [ ] Sample inference: `analyze_batch(["Apple beats earnings expectations"])` returns positive score
- [ ] Sample inference: `analyze_batch(["Massive layoffs at Google"])` returns negative score
- [ ] Batch processing of 100 headlines completes in < 5 seconds on CPU
- [ ] Cache directory is created and JSON files appear after first run
- [ ] Plugin is registered in `config/plugins.yaml`
- [ ] Pipeline integration test passes (mock news source)

## Phase 1 vs Production

**Phase 1 (current):** Implement FinBERTEnricher with a stub `_load_recent_headlines()` that returns empty list. The plugin works but produces zero features. This is fine — the quant engine will function without sentiment.

**Phase 3:** Implement real news loading via Alpha Vantage NEWS_SENTIMENT API or another source. Begin live sentiment scoring as part of the daily pipeline run.

**Phase 4 (future):** Replace FinBERT with FinGPT (LoRA-tuned Llama) for higher accuracy, or with a self-hosted Mistral for richer analysis. The plugin abstraction means this is a config change, not a code rewrite.
