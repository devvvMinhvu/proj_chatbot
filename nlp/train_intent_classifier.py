
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


# =========================================
# 1. DUONG DAN
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "intent_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "intent_classifier.joblib"


# =========================================
# 2. DOC VA KIEM TRA DU LIEU
# =========================================

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    required_columns = {"text", "intent"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV phải có hai cột: text và intent."
        )

    df = df.dropna(subset=["text", "intent"]).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()

    df = df[
        (df["text"] != "") &
        (df["intent"] != "")
    ]

    # Loại bỏ mẫu trùng hoàn toàn.
    df = df.drop_duplicates(
        subset=["text", "intent"]
    )

    if df["intent"].nunique() < 2:
        raise ValueError(
            "Cần ít nhất hai nhãn ý định để huấn luyện."
        )

    counts = df["intent"].value_counts()

    if counts.min() < 2:
        raise ValueError(
            "Mỗi nhãn cần ít nhất hai mẫu để chia dữ liệu."
        )

    return df


# =========================================
# 3. TAO PIPELINE TF-IDF + NAIVE BAYES
# =========================================

def create_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "naive_bayes",
            MultinomialNB(alpha=0.3)
        )
    ])


# =========================================
# 4. HUAN LUYEN VA DANH GIA
# =========================================

def main():
    df = load_dataset()

    print("Số mẫu:", len(df))
    print("\nSố mẫu mỗi ý định:")
    print(df["intent"].value_counts().sort_index())

    X = df["text"]
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Huấn luyện mô hình trên tập huấn luyện.
    evaluation_model = create_model()
    evaluation_model.fit(X_train, y_train)

    # Đánh giá trên dữ liệu chưa dùng để huấn luyện.
    y_pred = evaluation_model.predict(X_test)

    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)

    print(
        f"Accuracy: {accuracy_score(y_test, y_pred):.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=sorted(y.unique()),
            zero_division=0
        )
    )

    print("\nMột số dự đoán trên tập kiểm thử:")

    for text, actual, predicted in zip(
        X_test.tolist()[:10],
        y_test.tolist()[:10],
        y_pred[:10]
    ):
        status = "ĐÚNG" if actual == predicted else "SAI"

        print(f"\nCâu: {text}")
        print(f"Nhãn thật: {actual}")
        print(f"Dự đoán:   {predicted} [{status}]")

    # Huấn luyện lại trên toàn bộ dữ liệu
    # để tạo mô hình sử dụng sau này.
    final_model = create_model()
    final_model.fit(X, y)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(final_model, MODEL_PATH)

    print("\n" + "=" * 60)
    print("ĐÃ LƯU MÔ HÌNH")
    print(MODEL_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()
