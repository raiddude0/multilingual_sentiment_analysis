"""Inference helpers for the saved sentiment model."""

import os
from functools import lru_cache
from pathlib import Path

import torch
from transformers import pipeline

from .config import MODEL_DIR


def resolve_model_path() -> str:
    """Return an explicitly configured Hub/local model or the local default."""
    configured = os.environ.get("SENTIMENT_MODEL_PATH")
    if configured:
        model_path = Path(configured).expanduser()
        if model_path.is_dir():
            return str(model_path)
        if not model_path.is_absolute() and "/" in configured:
            return configured  # Hugging Face model ID, e.g. username/model-name
        raise FileNotFoundError(f"Configured SENTIMENT_MODEL_PATH does not exist: {model_path}")

    model_path = MODEL_DIR
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"No fine-tuned model found at {model_path}. Train one with `python -m multilingual_sentiment_analysis.train` "
            "or set SENTIMENT_MODEL_PATH to a compatible model directory."
        )
    return str(model_path)


@lru_cache(maxsize=1)
def get_classifier():
    use_gpu = os.environ.get("USE_GPU", "false").lower() == "true"
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("USE_GPU=true but CUDA is not available.")
    return pipeline(
        "sentiment-analysis", model=resolve_model_path(), tokenizer=resolve_model_path(),
        device=0 if use_gpu else -1,
    )


def predict(text: str) -> dict[str, float | str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input must be a non-empty string.")
    result = get_classifier()(text.strip())[0]
    return {"label": result["label"], "confidence": float(result["score"])}


def predict_batch(texts: list[str]) -> list[dict[str, float | str]]:
    if not isinstance(texts, list) or not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Input must be a non-empty list of non-empty strings.")
    return [
        {"label": result["label"], "confidence": float(result["score"])}
        for result in get_classifier()([text.strip() for text in texts])
    ]


if __name__ == "__main__":
    for sample in ("The flight was delayed three hours.", "Merci, le vol était très agréable !", "الخدمة كانت ممتازة"):
        print(f"{sample}\n  -> {predict(sample)}\n")
