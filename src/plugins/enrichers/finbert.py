"""FinBERT Sentiment Enricher plugin (DataEnricher)."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.news_data import NewsDataProvider
from src.plugins.base import DataEnricher

logger = logging.getLogger(__name__)


class FinBERTEnricher(DataEnricher):
    """
    Local financial sentiment analysis using FinBERT (ProsusAI/finbert).

    Runs batch inference on CPU. Outputs sentiment scores in [-1, +1]:
        +1 = strongly positive
        -1 = strongly negative
         0 = neutral / ambiguous

    **Phase 1 note:** ``NewsDataProvider`` is a stub returning ``[]``, so all
    features are zero. Wire up a real news provider in Phase 3 (Alpha Vantage /
    Finnhub). The plugin is already registered in ``config/plugins.yaml`` and
    the registry will instantiate it with zero args.

    **Caching:** Inference results are persisted as SHA256-keyed JSON files in
    ``data/models/finbert_cache/``. Incremental daily runs only process new
    headlines.

    **Model:** ProsusAI/finbert — BERT-base fine-tuned on financial text.
    Labels are output in order ``[positive, negative, neutral]`` (label index 0,
    1, 2 respectively). Score = P(positive) − P(negative).
    """

    name = "finbert"
    data_type = "sentiment"
    version = "1.0.0"

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "cpu",
        batch_size: int = 64,
        cache_dir: str = "data/models/finbert_cache",
        news_provider: Optional[NewsDataProvider] = None,
    ) -> None:
        """
        Load FinBERT model and prepare for inference.

        Args:
            model_name: HuggingFace model identifier. Default: ProsusAI/finbert.
            device: Torch device string (``"cpu"`` or ``"cuda"``). Default: ``"cpu"``.
            batch_size: Number of headlines per tokenizer/model call. Default: 64.
            cache_dir: Directory for per-headline JSON inference cache.
            news_provider: ``NewsDataProvider`` instance. Instantiated without
                args if not provided (Phase 1 stub, safe to call with zero args).
        """
        self._model_name = model_name
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # FinBERT label order: [0: positive, 1: negative, 2: neutral]
        # Confirmed during task 1.1 sanity-check inference.
        self._news_provider = news_provider or NewsDataProvider()

        # Model and tokenizer are loaded lazily on first inference call.
        # This allows the registry to instantiate FinBERTEnricher without
        # network access (Phase 1: NewsDataProvider returns [] so inference
        # is never triggered; model download only happens when enrich() is
        # called with real headlines in Phase 3).
        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------ #
    # Core inference                                                        #
    # ------------------------------------------------------------------ #

    def _ensure_model_loaded(self) -> None:
        """Load tokenizer and model on first use (lazy initialisation)."""
        if self._tokenizer is not None:
            return
        logger.info("Loading FinBERT model '%s' on %s …", self._model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        # use_safetensors=True avoids torch.load CVE-2025-32434 (same fix as
        # scripts/download_finbert.py).
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name, use_safetensors=True
        )
        self._model.to(self._device)
        self._model.eval()
        logger.info("FinBERT ready.")

    def analyze_batch(self, headlines: List[str]) -> List[Dict[str, float]]:
        """
        Run FinBERT inference on a list of headlines (no caching).

        Args:
            headlines: Raw headline strings. May be empty.

        Returns:
            List of dicts, one per headline, with keys:
            ``headline``, ``positive``, ``negative``, ``neutral``,
            ``score`` (= P(pos) − P(neg), range [−1, +1]),
            ``confidence`` (= max class probability).
        """
        if not headlines:
            return []

        self._ensure_model_loaded()
        results: List[Dict[str, float]] = []

        for i in range(0, len(headlines), self._batch_size):
            batch = headlines[i : i + self._batch_size]

            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            for j, p in enumerate(probs.cpu().numpy()):
                pos, neg, neu = float(p[0]), float(p[1]), float(p[2])
                results.append(
                    {
                        "headline": batch[j],
                        "positive": pos,
                        "negative": neg,
                        "neutral": neu,
                        "score": pos - neg,
                        "confidence": max(pos, neg, neu),
                    }
                )

        return results

    def analyze_batch_cached(self, headlines: List[str]) -> List[Dict[str, float]]:
        """
        Run FinBERT inference with per-headline JSON caching.

        Already-seen headlines are loaded from disk. Only unseen headlines hit
        the model, making incremental daily runs essentially free.

        Args:
            headlines: Raw headline strings.

        Returns:
            Same structure as :meth:`analyze_batch`.
        """
        if not headlines:
            return []

        results: List[Optional[Dict]] = [None] * len(headlines)
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []

        for i, h in enumerate(headlines):
            cache_path = self._cache_dir / f"{self._cache_key(h)}.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    results[i] = json.load(f)
            else:
                uncached_texts.append(h)
                uncached_indices.append(i)

        if uncached_texts:
            new_results = self.analyze_batch(uncached_texts)
            for list_pos, (original_idx, result) in enumerate(zip(uncached_indices, new_results)):
                results[original_idx] = result
                cache_path = self._cache_dir / f"{self._cache_key(headlines[original_idx])}.json"
                with open(cache_path, "w") as f:
                    json.dump(result, f)

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # DataEnricher interface                                               #
    # ------------------------------------------------------------------ #

    def enrich(self, ticker: str, features: Any) -> Dict[str, float]:
        """
        Compute sentiment features for a single ticker.

        Calls ``NewsDataProvider.get_headlines()``. In Phase 1 this returns
        ``[]``, so the output is :meth:`_empty_features` (all zeros). Phase 3
        wires up a real news source.

        Args:
            ticker: Stock ticker symbol.
            features: Existing ``FeatureVector`` (unused in Phase 1).

        Returns:
            Dict with keys: ``sentiment_score``, ``sentiment_ma_5d``,
            ``sentiment_ma_20d``, ``sentiment_momentum``,
            ``news_volume_ratio``, ``negative_news_ratio``.
        """
        headlines = self._news_provider.get_headlines(ticker, days_back=30)

        if not headlines:
            return self._empty_features()

        scored = self.analyze_batch_cached([h["headline"] for h in headlines])

        scores_df = pd.DataFrame(scored)
        scores_df.index = pd.to_datetime([h["published_at"] for h in headlines])
        scores_df.index.name = "timestamp"
        scores_df = scores_df.sort_index()

        return self._compute_rolling_features(scores_df, total_days=30)

    def batch_enrich(self, tickers: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute sentiment features for all tickers in a single model pass.

        More efficient than calling :meth:`enrich` per ticker because all
        headlines from all tickers are batched into one
        :meth:`analyze_batch_cached` call.

        Args:
            tickers: List of ticker symbols.

        Returns:
            Dict mapping ticker → sentiment feature dict.
        """
        all_raw_headlines: List[str] = []
        all_timestamps: List[Any] = []
        slices: Dict[str, tuple] = {}

        for ticker in tickers:
            raw = self._news_provider.get_headlines(ticker, days_back=30)
            start = len(all_raw_headlines)
            all_raw_headlines.extend([h["headline"] for h in raw])
            all_timestamps.extend([h["published_at"] for h in raw])
            slices[ticker] = (start, len(all_raw_headlines), len(raw))

        all_scored = self.analyze_batch_cached(all_raw_headlines)

        results: Dict[str, Dict[str, float]] = {}
        for ticker in tickers:
            start, end, count = slices[ticker]
            if count == 0:
                results[ticker] = self._empty_features()
                continue

            scores_df = pd.DataFrame(all_scored[start:end])
            scores_df.index = pd.to_datetime(all_timestamps[start:end])
            scores_df.index.name = "timestamp"
            scores_df = scores_df.sort_index()

            results[ticker] = self._compute_rolling_features(scores_df, total_days=30)

        return results

    # ------------------------------------------------------------------ #
    # Feature computation                                                  #
    # ------------------------------------------------------------------ #

    def _compute_rolling_features(
        self, scores_df: pd.DataFrame, total_days: int = 30
    ) -> Dict[str, float]:
        """
        Aggregate per-headline scores into time-bucketed sentiment features.

        Args:
            scores_df: DataFrame with columns ``score`` and ``confidence``,
                indexed by publication timestamp (DatetimeIndex).
            total_days: Lookback window used for news-volume normalisation.

        Returns:
            Dict with 6 sentiment feature keys.
        """
        if scores_df.empty:
            return self._empty_features()

        def _conf_weighted_mean(group: pd.DataFrame) -> float:
            total_conf = float(group["confidence"].sum())
            if total_conf <= 0.0:
                return 0.0
            return float((group["score"] * group["confidence"]).sum() / total_conf)

        daily: pd.Series = scores_df.resample("D").apply(_conf_weighted_mean)

        n = len(daily)
        latest = float(daily.iloc[-1]) if n >= 1 else 0.0
        ma_5d = float(daily.tail(5).mean()) if n >= 1 else latest
        ma_20d = float(daily.tail(20).mean()) if n >= 1 else latest
        momentum = ma_5d - ma_20d

        # News volume: today's headline count vs 30-day average
        today_date = pd.Timestamp.today().date()
        today_count = int(sum(pd.Timestamp(ts).date() == today_date for ts in scores_df.index))
        avg_daily = len(scores_df) / max(total_days, 1)
        news_volume_ratio = today_count / avg_daily if avg_daily > 0 else 0.0

        negative_news_ratio = float((scores_df["score"] < -0.3).sum() / len(scores_df))

        return {
            "sentiment_score": latest,
            "sentiment_ma_5d": ma_5d,
            "sentiment_ma_20d": ma_20d,
            "sentiment_momentum": momentum,
            "news_volume_ratio": news_volume_ratio,
            "negative_news_ratio": negative_news_ratio,
        }

    def _empty_features(self) -> Dict[str, float]:
        """Return zero-valued features when no headlines are available."""
        return {
            "sentiment_score": 0.0,
            "sentiment_ma_5d": 0.0,
            "sentiment_ma_20d": 0.0,
            "sentiment_momentum": 0.0,
            "news_volume_ratio": 0.0,
            "negative_news_ratio": 0.0,
        }

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def _cache_key(self, headline: str) -> str:
        """Return a 16-char SHA256 hex prefix for a headline string."""
        return hashlib.sha256(headline.encode()).hexdigest()[:16]
