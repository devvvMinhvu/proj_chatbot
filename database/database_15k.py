from pathlib import Path

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import DateTime

# ==================================================
# 1. ĐƯỜNG DẪN DATABASE
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "news_15k.db"

DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"


# ==================================================
# 2. KẾT NỐI SQLITE
# ==================================================

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()


# ==================================================
# 3. BẢNG ARTICLE
# ==================================================

class Article(Base):
    __tablename__ = "articles"

    # ID gốc từ JSONL, ví dụ: vne_news_0001
    id = Column(String(100), primary_key=True)

    source = Column(String(100), nullable=True)
    category = Column(String(100), nullable=True)
    split = Column(String(50), nullable=True)

    title = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    body = Column(Text, nullable=True)
    sapo = Column(Text, nullable=True)
    published_at = Column(DateTime, nullable=True)
    __table_args__ = (
        Index("idx_articles_15k_category", "category"),
        Index("idx_articles_15k_source", "source"),
    )


# ==================================================
# 4. KHỞI TẠO DATABASE VÀ BẢNG
# ==================================================

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database:", DB_PATH)
    print("Đã kiểm tra/tạo bảng articles.")


if __name__ == "__main__":
    init_db()