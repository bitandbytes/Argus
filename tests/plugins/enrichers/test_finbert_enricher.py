"""
Unit tests for FinBERTEnricher.

All model-dependent tests mock AutoTokenizer and
AutoModelForSequenceClassification so they run in ~1 second without requiring
the FinBERT model to be cached locally.

The final class, TestLiveSampleInference, is skipped unless the model is
found in the HuggingFace local cache (~/.cache/huggingface/).
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.plugins.enrichers.finbert import FinBERTEnricher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_KEYS = frozenset(
    {
        "sentiment_score",
        "sentiment_ma_5d",
        "sentiment_ma_20d",
        "sentiment_momentum",
        "news_volume_ratio",
        "negative_news_ratio",
    }
)


def _make_model_mock(logits: torch.Tensor) -> MagicMock:
    """Return a MagicMock that mimics a HuggingFace model returning given logits."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    def _call(**kwargs):
        batch_size = kwargs.get("input_ids", torch.zeros(1)).shape[0]
        out = MagicMock()
        out.logits = logits.repeat(batch_size, 1)
        return out

    mock_model.side_effect = _call
    return mock_model


def _make_tokenizer_mock() -> MagicMock:
    """Return a MagicMock tokenizer whose output can be used as **inputs."""
    mock_tok = MagicMock()

    def _call(texts, **kwargs):
        b = len(texts) if isinstance(texts, list) else 1
        result = MagicMock()
        result.to.return_value = {
            "input_ids": torch.zeros(b, 5, dtype=torch.long),
            "attention_mask": torch.ones(b, 5, dtype=torch.long),
        }
        return result

    mock_tok.side_effect = _call
    return mock_tok


# Positive-leaning logits: pos=2.0, neg=-1.0, neu=0.5
# After softmax: pos≈0.786, neg≈0.039, neu≈0.175 → score≈+0.747
_POS_LOGITS = torch.tensor([[2.0, -1.0, 0.5]])

# Negative-leaning logits: pos=-1.0, neg=2.0, neu=0.5
# After softmax: pos≈0.039, neg≈0.786, neu≈0.175 → score≈−0.747
_NEG_LOGITS = torch.tensor([[-1.0, 2.0, 0.5]])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def enricher(tmp_path):
    """FinBERTEnricher with mocked model and tokenizer, temp cache dir."""
    with patch(
        "src.plugins.enrichers.finbert.AutoTokenizer.from_pretrained",
        return_value=_make_tokenizer_mock(),
    ), patch(
        "src.plugins.enrichers.finbert.AutoModelForSequenceClassification.from_pretrained",
        return_value=_make_model_mock(_POS_LOGITS),
    ):
        yield FinBERTEnricher(cache_dir=str(tmp_path / "cache"))


@pytest.fixture()
def neg_enricher(tmp_path):
    """FinBERTEnricher returning negative-leaning logits, temp cache dir."""
    with patch(
        "src.plugins.enrichers.finbert.AutoTokenizer.from_pretrained",
        return_value=_make_tokenizer_mock(),
    ), patch(
        "src.plugins.enrichers.finbert.AutoModelForSequenceClassification.from_pretrained",
        return_value=_make_model_mock(_NEG_LOGITS),
    ):
        yield FinBERTEnricher(cache_dir=str(tmp_path / "neg_cache"))


# ---------------------------------------------------------------------------
# 1. Plugin metadata
# ---------------------------------------------------------------------------


class TestPluginMetadata:
    def test_name(self, enricher: FinBERTEnricher) -> None:
        assert enricher.name == "finbert"

    def test_data_type(self, enricher: FinBERTEnricher) -> None:
        assert enricher.data_type == "sentiment"

    def test_version(self, enricher: FinBERTEnricher) -> None:
        assert enricher.version == "1.0.0"


# ---------------------------------------------------------------------------
# 2. analyze_batch — core inference logic
# ---------------------------------------------------------------------------


class TestAnalyzeBatch:
    def test_empty_input_returns_empty_list(self, enricher: FinBERTEnricher) -> None:
        assert enricher.analyze_batch([]) == []

    def test_single_headline_returns_one_result(self, enricher: FinBERTEnricher) -> None:
        results = enricher.analyze_batch(["Apple beats earnings"])
        assert len(results) == 1

    def test_result_has_all_expected_keys(self, enricher: FinBERTEnricher) -> None:
        result = enricher.analyze_batch(["Test headline"])[0]
        assert set(result.keys()) == {
            "headline",
            "positive",
            "negative",
            "neutral",
            "score",
            "confidence",
        }

    def test_headline_preserved_in_result(self, enricher: FinBERTEnricher) -> None:
        h = "Apple beats earnings"
        result = enricher.analyze_batch([h])[0]
        assert result["headline"] == h

    def test_score_equals_positive_minus_negative(self, enricher: FinBERTEnricher) -> None:
        result = enricher.analyze_batch(["Test"])[0]
        assert abs(result["score"] - (result["positive"] - result["negative"])) < 1e-6

    def test_confidence_equals_max_class_probability(self, enricher: FinBERTEnricher) -> None:
        result = enricher.analyze_batch(["Test"])[0]
        expected_confidence = max(result["positive"], result["negative"], result["neutral"])
        assert abs(result["confidence"] - expected_confidence) < 1e-6

    def test_probabilities_sum_to_one(self, enricher: FinBERTEnricher) -> None:
        result = enricher.analyze_batch(["Test"])[0]
        total = result["positive"] + result["negative"] + result["neutral"]
        assert abs(total - 1.0) < 1e-5

    def test_positive_logits_give_positive_score(self, enricher: FinBERTEnricher) -> None:
        result = enricher.analyze_batch(["Good news"])[0]
        assert result["score"] > 0

    def test_negative_logits_give_negative_score(self, neg_enricher: FinBERTEnricher) -> None:
        result = neg_enricher.analyze_batch(["Bad news"])[0]
        assert result["score"] < 0

    def test_multiple_headlines_correct_count(self, enricher: FinBERTEnricher) -> None:
        headlines = [f"headline {i}" for i in range(5)]
        results = enricher.analyze_batch(headlines)
        assert len(results) == 5

    def test_chunking_70_headlines_calls_model_twice(self, enricher: FinBERTEnricher) -> None:
        """70 headlines with batch_size=64 must trigger exactly 2 model calls."""
        headlines = [f"headline {i}" for i in range(70)]
        enricher._model.reset_mock()
        enricher.analyze_batch(headlines)
        assert enricher._model.call_count == 2  # ceil(70 / 64) = 2

    def test_exact_batch_size_headlines_calls_model_once(self, enricher: FinBERTEnricher) -> None:
        headlines = [f"headline {i}" for i in range(64)]
        enricher._model.reset_mock()
        enricher.analyze_batch(headlines)
        assert enricher._model.call_count == 1


# ---------------------------------------------------------------------------
# 3. Caching behaviour
# ---------------------------------------------------------------------------


class TestCachingBehavior:
    def test_cache_miss_creates_json_file(self, enricher: FinBERTEnricher) -> None:
        headline = "Apple reports record profits"
        enricher.analyze_batch_cached([headline])
        cache_file = enricher._cache_dir / f"{enricher._cache_key(headline)}.json"
        assert cache_file.exists()

    def test_cached_json_has_expected_keys(self, enricher: FinBERTEnricher) -> None:
        headline = "Apple reports record profits"
        enricher.analyze_batch_cached([headline])
        cache_file = enricher._cache_dir / f"{enricher._cache_key(headline)}.json"
        with open(cache_file) as f:
            data = json.load(f)
        assert "score" in data
        assert "confidence" in data

    def test_cache_hit_skips_model_call(self, enricher: FinBERTEnricher) -> None:
        headline = "Apple reports record profits"
        enricher.analyze_batch_cached([headline])  # populates cache
        enricher._model.reset_mock()
        enricher.analyze_batch_cached([headline])  # should hit cache
        assert enricher._model.call_count == 0

    def test_cache_key_is_deterministic(self, enricher: FinBERTEnricher) -> None:
        h = "Apple reports record profits"
        assert enricher._cache_key(h) == enricher._cache_key(h)

    def test_different_headlines_have_different_cache_keys(self, enricher: FinBERTEnricher) -> None:
        assert enricher._cache_key("Apple up") != enricher._cache_key("Apple down")

    def test_partial_cache_only_runs_uncached_headlines(self, enricher: FinBERTEnricher) -> None:
        """With 3 headlines, 1 cached and 2 new, model should be called once."""
        h1 = "Cached headline"
        h2, h3 = "New headline A", "New headline B"

        # Pre-populate cache for h1 only
        enricher.analyze_batch_cached([h1])
        enricher._model.reset_mock()

        # Now run with all 3 — only 2 should hit the model (in one batch call)
        results = enricher.analyze_batch_cached([h1, h2, h3])
        assert len(results) == 3
        assert enricher._model.call_count == 1  # one batch for h2 + h3

    def test_empty_input_returns_empty_list(self, enricher: FinBERTEnricher) -> None:
        assert enricher.analyze_batch_cached([]) == []


# ---------------------------------------------------------------------------
# 4. _compute_rolling_features
# ---------------------------------------------------------------------------


class TestComputeRollingFeatures:
    def _make_scores_df(self, n: int, score_fn=None) -> pd.DataFrame:
        """Build a synthetic scored DataFrame with n daily rows."""
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        scores = [score_fn(i) if score_fn else 0.5 for i in range(n)]
        df = pd.DataFrame(
            {"score": scores, "confidence": [0.9] * n},
            index=dates,
        )
        df.index.name = "timestamp"
        return df

    def test_empty_df_returns_zero_features(self, enricher: FinBERTEnricher) -> None:
        df = pd.DataFrame(
            columns=["score", "confidence"],
            index=pd.DatetimeIndex([], name="timestamp"),
        )
        result = enricher._compute_rolling_features(df)
        assert result == enricher._empty_features()

    def test_all_six_keys_present(self, enricher: FinBERTEnricher) -> None:
        df = self._make_scores_df(25)
        result = enricher._compute_rolling_features(df)
        assert set(result.keys()) == EXPECTED_KEYS

    def test_all_values_are_finite(self, enricher: FinBERTEnricher) -> None:
        df = self._make_scores_df(25)
        result = enricher._compute_rolling_features(df)
        for k, v in result.items():
            assert np.isfinite(v), f"Feature '{k}' is not finite: {v}"

    def test_ma5d_differs_from_ma20d_when_sentiment_shifts(self, enricher: FinBERTEnricher) -> None:
        """First 10 days negative, last 15 positive — ma5d ≠ ma20d."""
        df = self._make_scores_df(25, score_fn=lambda i: -0.5 if i < 10 else 0.8)
        result = enricher._compute_rolling_features(df)
        assert result["sentiment_ma_5d"] != result["sentiment_ma_20d"]

    def test_momentum_equals_ma5d_minus_ma20d(self, enricher: FinBERTEnricher) -> None:
        df = self._make_scores_df(25, score_fn=lambda i: -0.5 if i < 10 else 0.8)
        result = enricher._compute_rolling_features(df)
        expected = result["sentiment_ma_5d"] - result["sentiment_ma_20d"]
        assert abs(result["sentiment_momentum"] - expected) < 1e-9

    def test_negative_news_ratio_correct(self, enricher: FinBERTEnricher) -> None:
        """2 of 4 headlines have score < -0.3 → ratio = 0.5."""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {"score": [-0.5, -0.4, 0.1, 0.6], "confidence": [0.9] * 4},
            index=dates,
        )
        df.index.name = "timestamp"
        result = enricher._compute_rolling_features(df)
        assert abs(result["negative_news_ratio"] - 0.5) < 1e-6

    def test_negative_news_ratio_zero_when_all_positive(self, enricher: FinBERTEnricher) -> None:
        df = self._make_scores_df(10, score_fn=lambda _: 0.9)
        result = enricher._compute_rolling_features(df)
        assert result["negative_news_ratio"] == 0.0

    def test_news_volume_ratio_is_nonnegative(self, enricher: FinBERTEnricher) -> None:
        df = self._make_scores_df(15)
        result = enricher._compute_rolling_features(df)
        assert result["news_volume_ratio"] >= 0.0

    def test_single_day_of_data_does_not_raise(self, enricher: FinBERTEnricher) -> None:
        df = self._make_scores_df(1)
        result = enricher._compute_rolling_features(df)
        assert set(result.keys()) == EXPECTED_KEYS

    def test_confidence_weighted_mean_applied(self, enricher: FinBERTEnricher) -> None:
        """High-confidence positive headline dominates low-confidence negative."""
        dates = pd.date_range("2024-06-01", periods=1, freq="D")
        # Two headlines on the same day: one very positive + confident,
        # one negative but low confidence.
        df = pd.DataFrame(
            {"score": [0.9, -0.8], "confidence": [0.95, 0.10]},
            index=[dates[0], dates[0]],
        )
        df.index.name = "timestamp"
        result = enricher._compute_rolling_features(df)
        # Weighted: (0.9*0.95 + (-0.8)*0.10) / (0.95+0.10) = 0.775/1.05 ≈ 0.738
        assert result["sentiment_score"] > 0


# ---------------------------------------------------------------------------
# 5. enrich() — single-ticker interface
# ---------------------------------------------------------------------------


class TestEnrichOutput:
    def test_returns_all_six_keys(self, enricher: FinBERTEnricher) -> None:
        result = enricher.enrich("AAPL", None)
        assert set(result.keys()) == EXPECTED_KEYS

    def test_phase1_stub_returns_zero_features(self, enricher: FinBERTEnricher) -> None:
        """Phase 1: NewsDataProvider returns [] → all zeros."""
        result = enricher.enrich("AAPL", None)
        assert result == enricher._empty_features()

    def test_all_values_are_floats(self, enricher: FinBERTEnricher) -> None:
        result = enricher.enrich("AAPL", None)
        for k, v in result.items():
            assert isinstance(v, float), f"Feature '{k}' is {type(v)}, expected float"

    def test_works_with_non_none_features_arg(self, enricher: FinBERTEnricher) -> None:
        """features parameter is unused in Phase 1 — any value should work."""
        result = enricher.enrich("MSFT", object())
        assert set(result.keys()) == EXPECTED_KEYS

    def test_enrich_with_injected_news(self, tmp_path) -> None:
        """Inject a mock NewsDataProvider to exercise non-empty headline path."""
        mock_provider = MagicMock()
        mock_provider.get_headlines.return_value = [
            {
                "headline": "Apple beats earnings",
                "published_at": pd.Timestamp("2024-06-15 10:00:00"),
            }
        ]

        with patch(
            "src.plugins.enrichers.finbert.AutoTokenizer.from_pretrained",
            return_value=_make_tokenizer_mock(),
        ), patch(
            "src.plugins.enrichers.finbert.AutoModelForSequenceClassification.from_pretrained",
            return_value=_make_model_mock(_POS_LOGITS),
        ):
            enricher = FinBERTEnricher(
                cache_dir=str(tmp_path / "cache"),
                news_provider=mock_provider,
            )

        result = enricher.enrich("AAPL", None)
        assert set(result.keys()) == EXPECTED_KEYS
        # With mocked positive logits, score should be > 0
        assert result["sentiment_score"] > 0


# ---------------------------------------------------------------------------
# 6. batch_enrich() — multi-ticker interface
# ---------------------------------------------------------------------------


class TestBatchEnrich:
    def test_returns_all_requested_tickers(self, enricher: FinBERTEnricher) -> None:
        tickers = ["AAPL", "MSFT", "GOOG"]
        result = enricher.batch_enrich(tickers)
        assert set(result.keys()) == set(tickers)

    def test_each_ticker_has_all_feature_keys(self, enricher: FinBERTEnricher) -> None:
        for ticker_features in enricher.batch_enrich(["AAPL", "MSFT"]).values():
            assert set(ticker_features.keys()) == EXPECTED_KEYS

    def test_empty_tickers_list(self, enricher: FinBERTEnricher) -> None:
        result = enricher.batch_enrich([])
        assert result == {}

    def test_single_ticker(self, enricher: FinBERTEnricher) -> None:
        result = enricher.batch_enrich(["AAPL"])
        assert set(result.keys()) == {"AAPL"}
        assert set(result["AAPL"].keys()) == EXPECTED_KEYS

    def test_phase1_stub_returns_zeros_for_all_tickers(self, enricher: FinBERTEnricher) -> None:
        result = enricher.batch_enrich(["AAPL", "MSFT"])
        for features in result.values():
            assert all(v == 0.0 for v in features.values())

    def test_batch_enrich_with_injected_news(self, tmp_path) -> None:
        """batch_enrich collects all headlines into one model call."""
        mock_provider = MagicMock()
        headlines_per_ticker = {
            "AAPL": [
                {
                    "headline": f"AAPL headline {i}",
                    "published_at": pd.Timestamp(f"2024-06-{15+i} 10:00"),
                }
                for i in range(3)
            ],
            "MSFT": [
                {
                    "headline": f"MSFT headline {i}",
                    "published_at": pd.Timestamp(f"2024-06-{15+i} 11:00"),
                }
                for i in range(2)
            ],
        }
        mock_provider.get_headlines.side_effect = lambda ticker, **kw: headlines_per_ticker.get(
            ticker, []
        )

        mock_model = _make_model_mock(_POS_LOGITS)
        with patch(
            "src.plugins.enrichers.finbert.AutoTokenizer.from_pretrained",
            return_value=_make_tokenizer_mock(),
        ), patch(
            "src.plugins.enrichers.finbert.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ):
            enricher = FinBERTEnricher(
                cache_dir=str(tmp_path / "cache"),
                news_provider=mock_provider,
            )

        result = enricher.batch_enrich(["AAPL", "MSFT"])
        assert set(result.keys()) == {"AAPL", "MSFT"}
        assert result["AAPL"]["sentiment_score"] > 0
        assert result["MSFT"]["sentiment_score"] > 0


# ---------------------------------------------------------------------------
# 7. Live sample inference (skipped unless model is locally cached)
# ---------------------------------------------------------------------------


def _finbert_cached() -> bool:
    """Return True if ProsusAI/finbert is available in the local HF cache."""
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained("ProsusAI/finbert", local_files_only=True)
        return True
    except Exception:
        return False


_SKIP_LIVE = pytest.mark.skipif(
    not _finbert_cached(),
    reason="ProsusAI/finbert not in local HuggingFace cache — run scripts/download_finbert.py",
)


class TestLiveSampleInference:
    """
    Runs real FinBERT inference on known-sentiment headlines.

    These tests validate the model's output direction and are the
    'sanity-check inference' step documented in TASKS.md 1.5.
    Skipped automatically when the model is not locally cached.
    """

    @_SKIP_LIVE
    def test_positive_headline_has_positive_score(self) -> None:
        e = FinBERTEnricher()
        results = e.analyze_batch(["Apple beats earnings expectations by a wide margin"])
        assert (
            results[0]["score"] > 0
        ), f"Expected positive score for positive headline, got {results[0]['score']:.4f}"

    @_SKIP_LIVE
    def test_negative_headline_has_negative_score(self) -> None:
        e = FinBERTEnricher()
        results = e.analyze_batch(["Massive layoffs announced as company faces bankruptcy"])
        assert (
            results[0]["score"] < 0
        ), f"Expected negative score for negative headline, got {results[0]['score']:.4f}"

    @_SKIP_LIVE
    def test_probabilities_sum_to_one_live(self) -> None:
        e = FinBERTEnricher()
        result = e.analyze_batch(["The company reported mixed quarterly results"])[0]
        total = result["positive"] + result["negative"] + result["neutral"]
        assert abs(total - 1.0) < 1e-5

    @_SKIP_LIVE
    def test_batch_of_100_headlines_completes(self) -> None:
        """100 headlines should complete in a reasonable time (no assertion on speed)."""
        e = FinBERTEnricher()
        headlines = [f"Company {i} reports quarterly earnings" for i in range(100)]
        results = e.analyze_batch(headlines)
        assert len(results) == 100
        assert all("score" in r for r in results)
