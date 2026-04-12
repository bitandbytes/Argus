"""
Download and cache the FinBERT model from HuggingFace.

This script downloads ProsusAI/finbert (~440 MB) to the HuggingFace local
cache (~/.cache/huggingface/). Run it once during setup; subsequent imports
in the pipeline will load from cache with no network access required.

Usage:
    python scripts/download_finbert.py

The FinBERT model is used by src/plugins/enrichers/finbert.py (Task 1.5).
"""

import sys
import pathlib
import time

MODEL_NAME = "ProsusAI/finbert"


def download_finbert() -> None:
    print(f"Downloading FinBERT model: {MODEL_NAME}")
    print("This is a one-time download (~440 MB). Please wait...\n")

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("[ERROR] transformers not installed. Run: pip install transformers torch")
        sys.exit(1)

    start = time.time()

    print("  Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  Tokenizer ready. Vocab size: {tokenizer.vocab_size}")

    print("  Downloading model weights...")
    # use_safetensors=True: transformers >=5.x blocks torch.load on torch <2.6 (CVE-2025-32434).
    # ProsusAI/finbert ships safetensors weights, so this bypasses the version check cleanly.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, use_safetensors=True)

    elapsed = time.time() - start
    num_params = sum(p.numel() for p in model.parameters())

    print(f"\n  Model parameters: {num_params:,}")
    print(f"  Labels: {list(model.config.id2label.values())}")
    print(f"  Download completed in {elapsed:.1f}s")

    # Quick sanity-check inference
    print("\n  Running sanity-check inference...")
    import torch

    inputs = tokenizer(
        ["Revenue grew 15% year-over-year.", "The company reported a net loss."],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    labels = list(model.config.id2label.values())

    print("  Sample results:")
    sentences = [
        "Revenue grew 15% year-over-year.",
        "The company reported a net loss.",
    ]
    for sent, prob in zip(sentences, probs):
        top_idx = prob.argmax().item()
        print(f"    '{sent[:50]}' -> {labels[top_idx]} ({prob[top_idx]:.2%})")

    print("\n[OK] FinBERT cached at ~/.cache/huggingface/")
    print("     Run 'python scripts/verify_setup.py' to confirm.")


if __name__ == "__main__":
    # Ensure project root is on path
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    download_finbert()
