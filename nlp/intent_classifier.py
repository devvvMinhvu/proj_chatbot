"""Runtime inference for user intent classification."""

from __future__ import annotations

import joblib

from config import INTENT_MODEL_PATH

_model = None


def load_model():
    """Nạp mô hình đã huấn luyện một lần."""
    global _model

    if _model is None:
        if not INTENT_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình: {INTENT_MODEL_PATH}. "
                "Hãy chạy python -m nlp.train_intent_classifier"
            )
        _model = joblib.load(INTENT_MODEL_PATH)

    return _model


def predict_intent(text: str) -> str:
    """Dự đoán ý định người dùng bằng TF-IDF + Naive Bayes."""
    if not text or not text.strip():
        return "unknown"

    model = load_model()
    return str(model.predict([text.strip()])[0])


def predict_intent_with_confidence(text: str) -> dict:
    """Trả về ý định dự đoán và xác suất do mô hình cung cấp."""
    if not text or not text.strip():
        return {
            "intent": "unknown",
            "confidence": 0.0,
        }

    model = load_model()
    probabilities = model.predict_proba([text.strip()])[0]
    best_index = probabilities.argmax()

    return {
        "intent": str(model.classes_[best_index]),
        "confidence": float(probabilities[best_index]),
    }
