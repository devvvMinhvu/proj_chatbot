"""Runtime inference for article category classification."""

from __future__ import annotations

from typing import Any

import joblib

from config import TEXT_CLASSIFIER_MODEL_PATH

_model: Any = None


def format_article_text(title: str = "", sapo: str = "", body: str = "") -> str:
    """Format article fields consistently for category training and inference."""
    return (
        f"Tiêu đề: {(title or '').strip()}\n"
        f"Tóm tắt: {(sapo or '').strip()}\n"
        f"Nội dung: {(body or '').strip()}"
    ).strip()


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


def predict_category(title: str = "", sapo: str = "", body: str = "") -> dict[str, Any]:
    """Classify a news article and return predicted category with class probabilities."""
    if not ((title or "").strip() or (sapo or "").strip() or (body or "").strip()):
        raise ValueError("Bài báo không có nội dung để phân loại.")

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
    }


if __name__ == "__main__":
    from nlp.train_text_classifier import main

    main()
