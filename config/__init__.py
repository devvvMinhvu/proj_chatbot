"""Centralized paths and runtime settings for proj_chatbot."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

NEWS_JSONL_PATH = DATA_DIR / "news.jsonl"
NEWS_DB_PATH = DATA_DIR / "news.db"
DATABASE_URL = f"sqlite:///{NEWS_DB_PATH.as_posix()}"

VECTOR_DB_DIR = DATA_DIR / "vector_db"
FAISS_INDEX_PATH = VECTOR_DB_DIR / "news.index"
CHUNK_METADATA_PATH = VECTOR_DB_DIR / "chunk_metadata.json"
TTS_CACHE_DIR = DATA_DIR / "tts_cache"
PIPER_MODELS_DIR = MODELS_DIR / "piper"

INTENT_DATASET_PATH = DATA_DIR / "intent_dataset.csv"
INTENT_MODEL_PATH = DATA_DIR / "intent_classifier.joblib"
TEXT_CLASSIFIER_MODEL_PATH = DATA_DIR / "text_classifier.joblib"

PHOBERT_MODEL_DIR = MODELS_DIR / "phobert"
VIT5_MODEL_DIR = MODELS_DIR / "vit5_summarization"

DEFAULT_LLM_MODEL = "qwen2.5:1.5b"

#: Display labels for the database's publisher and category values. The turn
#: parser recognises a publisher or category by its value or its label.
SOURCE_DISPLAY = {
    "vnexpress": "VnExpress",
    "tuoitre": "Tuổi Trẻ",
    "tienphong": "Tiền Phong",
}
CATEGORY_DISPLAY = {
    "news": "Thời sự",
    "economy": "Kinh tế",
    "tech": "Công nghệ",
    "entertainment": "Giải trí",
    "sports": "Thể thao",
}
