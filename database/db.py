"""SQLAlchemy engine, Article ORM model, and automatic schema initialization."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Column, DateTime, Index, String, Text, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATA_DIR, DATABASE_URL, NEWS_DB_PATH

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


class Article(Base):
    """News article stored in SQLite and indexed into FAISS."""

    __tablename__ = "articles"

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
        Index("idx_articles_category", "category"),
        Index("idx_articles_source", "source"),
        Index("idx_articles_published_at", "published_at"),
    )


def init_db() -> None:
    """Create tables if needed and migrate legacy schemas in-place."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    existing_columns = {column["name"] for column in inspector.get_columns("articles")}
    existing_indexes = {index["name"] for index in inspector.get_indexes("articles")}

    with engine.begin() as connection:
        if "published_at" not in existing_columns:
            connection.execute(text("ALTER TABLE articles ADD COLUMN published_at DATETIME"))

        for legacy_index in ("idx_articles_15k_category", "idx_articles_15k_source"):
            if legacy_index in existing_indexes:
                connection.execute(text(f"DROP INDEX IF EXISTS {legacy_index}"))

        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_articles_category ON articles (category)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles (source)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles (published_at)"))


@lru_cache(maxsize=1)
def ensure_db_initialized() -> None:
    """Run init_db() at most once per process before database queries."""
    init_db()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized and migrated: {NEWS_DB_PATH}")
