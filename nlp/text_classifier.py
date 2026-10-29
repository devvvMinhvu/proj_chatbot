import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================
# 1. CAU HINH DU AN
# ============================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "news_15k.jsonl"
MODEL_PATH = ROOT / "data" / "text_classifier_15k.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ============================================
# 2. DOC DU LIEU JSONL
# ============================================

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu: {DATA_PATH}"
        )

    texts = []
    labels = []

    with DATA_PATH.open(
        "r",
        encoding="utf-8-sig"
    ) as file:

        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Lỗi JSON tại dòng {line_number}: {exc}"
                ) from exc

            category = (item.get("category") or "").strip()

            # Ghép các trường văn bản làm đầu vào phân loại.
            title = (item.get("title") or "").strip()
            sapo = (item.get("sapo") or "").strip()
            body = (item.get("body") or "").strip()

            text = (
                f"Tiêu đề: {title}\n"
                f"Tóm tắt: {sapo}\n"
                f"Nội dung: {body}"
            ).strip()

            if not category:
                continue

            if not (title or sapo or body):
                continue

            texts.append(text)
            labels.append(category)

    if not texts:
        raise RuntimeError("Không có dữ liệu hợp lệ để huấn luyện.")

    return texts, labels


# ============================================
# 3. KHOI TAO MO HINH PHAN LOAI
# ============================================

def create_classifier():
    """
    Kết hợp TF-IDF word n-grams và character n-grams.

    Word n-grams học các cụm từ xuất hiện trong bài báo.
    Character n-grams hỗ trợ xử lý đặc điểm chữ tiếng Việt.
    """

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

    features = FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer),
    ])

    classifier = Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    return classifier


# ============================================
# 4. HUAN LUYEN VA DANH GIA
# ============================================

def train_classifier():
    texts, labels = load_dataset()

    print("========== DU LIEU PHAN LOAI ==========")
    print("Tổng số bài:", len(texts))

    categories = sorted(set(labels))
    print("Các nhãn:", categories)

    for category in categories:
        print(
            f"{category}: "
            f"{sum(label == category for label in labels)} bài"
        )

    # Chia train/test theo tỷ lệ 80/20.
    # stratify giúp duy trì tỷ lệ từng lớp.
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

    # Đánh giá trên tập test chưa dùng khi huấn luyện.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== KET QUA DANH GIA ==========")
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
    print(confusion_matrix(
        y_test,
        predictions,
        labels=categories,
    ))

    # Sau khi đánh giá, huấn luyện lại trên toàn bộ dữ liệu
    # để tạo mô hình sử dụng trong hệ thống.
    print("\nĐang huấn luyện lại trên toàn bộ dữ liệu...")
    model.fit(texts, labels)

    joblib.dump(model, MODEL_PATH)

    print("\n========== HOAN THANH ==========")
    print("Đã lưu mô hình:", MODEL_PATH)
    print("Mô hình dùng để dự đoán:", categories)


# ============================================
# 5. HAM DU DOAN CHO CAC MODULE KHAC
# ============================================

def predict_category(title="", sapo="", body=""):
    """
    Phân loại một bài báo mới.
    Trả về nhãn dự đoán và xác suất tham khảo.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chưa có mô hình. Hãy chạy "
            "'python -m nlp.text_classifier' trước."
        )

    model = joblib.load(MODEL_PATH)

    text = (
        f"Tiêu đề: {title}\n"
        f"Tóm tắt: {sapo}\n"
        f"Nội dung: {body}"
    ).strip()

    if not (title.strip() or sapo.strip() or body.strip()):
        raise ValueError("Bài báo không có nội dung để phân loại.")

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]
    class_labels = model.named_steps["classifier"].classes_

    confidence = float(max(probabilities))

    probability_by_class = {
        label: float(probability)
        for label, probability in zip(
            class_labels,
            probabilities,
        )
    }

    return {
        "category": prediction,
        "confidence": confidence,
        "probabilities": probability_by_class,
    }


if __name__ == "__main__":
    train_classifier()