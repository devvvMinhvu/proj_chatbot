"""Runtime inference for article category classification.

Two models predict an article's category (economy, entertainment, news, sports,
tech):

* a fine-tuned PhoBERT (``python -m nlp.train_phobert_classifier``), used when
  ``models/phobert_topic_classifier`` exists;
* TF-IDF + logistic regression (``python -m nlp.train_text_classifier``), the
  fallback and the baseline it is compared against.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib

from config import PHOBERT_TOPIC_MODEL_DIR, TEXT_CLASSIFIER_MODEL_PATH

#: PhoBERT reads at most this many subword tokens; only the beginning of the body fits.
PHOBERT_MAX_LENGTH = 256
#: Characters of body text that are segmented; more could not fit in the input anyway.
_PHOBERT_BODY_CHARS = 2500

_model: Any = None


def format_article_text(title: str = "", sapo: str = "", body: str = "") -> str:
    """Format article fields consistently for category training and inference."""
    return (
        f"Tiêu đề: {(title or '').strip()}\n"
        f"Tóm tắt: {(sapo or '').strip()}\n"
        f"Nội dung: {(body or '').strip()}"
    ).strip()


def phobert_input_text(title: str = "", sapo: str = "", body: str = "") -> str:
    """Title, sapo and body for PhoBERT, word-segmented with pyvi as PhoBERT expects.

    The crawler repeats the sapo as the first body paragraph; that copy is removed
    so the limited input holds more of the article.
    """
    from evaluation.datasets import strip_reference_from_document
    from pyvi import ViTokenizer

    title, sapo, body = (title or "").strip(), (sapo or "").strip(), (body or "").strip()
    if sapo and body:
        body, _ = strip_reference_from_document(body, sapo)
    raw = ". ".join(part for part in (title, sapo, body[:_PHOBERT_BODY_CHARS]) if part)
    return ViTokenizer.tokenize(" ".join(raw.split()))


@lru_cache(maxsize=1)
def _load_phobert():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(PHOBERT_TOPIC_MODEL_DIR), use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(PHOBERT_TOPIC_MODEL_DIR))
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _predict_phobert(title: str, sapo: str, body: str) -> dict[str, Any]:
    import torch

    tokenizer, model, device = _load_phobert()
    encoded = tokenizer(
        phobert_input_text(title, sapo, body),
        truncation=True,
        max_length=PHOBERT_MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits[0].float(), dim=-1).cpu().tolist()
    labels = [model.config.id2label[i] for i in range(len(probabilities))]
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "category": labels[best],
        "confidence": float(probabilities[best]),
        "probabilities": dict(zip(labels, map(float, probabilities))),
        "model": "phobert",
    }


def load_model():
    """Load and cache the trained article category classifier."""
    global _model
    if _model is None:
        if not TEXT_CLASSIFIER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình: {TEXT_CLASSIFIER_MODEL_PATH}. "
                "Hãy chạy 'python -m nlp.train_text_classifier' trước."
            )
        _model = joblib.load(TEXT_CLASSIFIER_MODEL_PATH)
    return _model


def predict_category(
    title: str = "",
    sapo: str = "",
    body: str = "",
    backend: str = "auto",
) -> dict[str, Any]:
    """Classify a news article and return predicted category with class probabilities.

    ``backend`` is "phobert", "tfidf" or "auto" (PhoBERT when its model exists).
    """
    if not ((title or "").strip() or (sapo or "").strip() or (body or "").strip()):
        raise ValueError("Bài báo không có nội dung để phân loại.")
    if backend not in {"auto", "phobert", "tfidf"}:
        raise ValueError(f"Unknown backend: {backend}")
    if backend == "phobert" or (backend == "auto" and (PHOBERT_TOPIC_MODEL_DIR / "config.json").exists()):
        return _predict_phobert(title, sapo, body)

    model = load_model()
    text = format_article_text(title=title, sapo=sapo, body=body)

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    class_labels = model.named_steps["classifier"].classes_

    probability_by_class = {
        label: float(probability)
        for label, probability in zip(class_labels, probabilities)
    }

    return {
        "category": prediction,
        "confidence": float(max(probabilities)),
        "probabilities": probability_by_class,
        "model": "tfidf",
    }


if __name__ == "__main__":
    from nlp.train_text_classifier import main

    main()
