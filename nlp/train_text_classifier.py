"""Training pipeline for the news article category classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from config import NEWS_JSONL_PATH, TEXT_CLASSIFIER_MODEL_PATH
from nlp.text_classifier import format_article_text

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(data_path: Path = NEWS_JSONL_PATH) -> tuple[list[str], list[str]]:
    """Read labeled article records from JSONL for topic classification."""
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {data_path}")

    texts: list[str] = []
    labels: list[str] = []

    with data_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Lỗi JSON tại dòng {line_number}: {exc}") from exc

            category = (item.get("category") or "").strip()
            title = (item.get("title") or "").strip()
            sapo = (item.get("sapo") or "").strip()
            body = (item.get("body") or "").strip()

            if not category or not (title or sapo or body):
                continue

            texts.append(format_article_text(title=title, sapo=sapo, body=body))
            labels.append(category)

    if not texts:
        raise RuntimeError("Không có dữ liệu hợp lệ để huấn luyện.")

    return texts, labels


def create_classifier() -> Pipeline:
    """Combine word and character TF-IDF n-grams with Logistic Regression."""
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True,
    )
    features = FeatureUnion(
        [
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_classifier(
    data_path: Path = NEWS_JSONL_PATH,
    model_path: Path = TEXT_CLASSIFIER_MODEL_PATH,
) -> None:
    """Evaluate on an 80/20 stratified holdout, then refit on all data and persist."""
    texts, labels = load_dataset(data_path=data_path)

    print("========== DỮ LIỆU PHÂN LOẠI ==========")
    print("Tổng số bài:", len(texts))

    categories = sorted(set(labels))
    print("Các nhãn:", categories)
    for category in categories:
        print(f"{category}: {sum(label == category for label in labels)} bài")

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print("\n========== HUẤN LUYỆN ==========")
    print("Train:", len(X_train))
    print("Test:", len(X_test))

    model = create_classifier()
    print("Đang huấn luyện Logistic Regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n========== KẾT QUẢ ĐÁNH GIÁ ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=categories,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print("Thứ tự nhãn:", categories)
    print(confusion_matrix(y_test, predictions, labels=categories))

    print("\nĐang huấn luyện lại trên toàn bộ dữ liệu...")
    model.fit(texts, labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print("\n========== HOÀN THÀNH ==========")
    print("Đã lưu mô hình:", model_path)
    print("Mô hình dùng để dự đoán:", categories)


def main() -> None:
    train_classifier()


if __name__ == "__main__":
    main()
