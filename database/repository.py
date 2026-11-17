"""Database query helpers for retrieving articles and publication listings."""

from __future__ import annotations

from typing import Any

from database.db import Article, SessionLocal, ensure_db_initialized


def get_article_content(
    article_id: str | None = None,
    url: str | None = None,
) -> dict[str, str] | None:
    """Fetch article title, sapo, and full body from SQLite; prefer ID, then URL."""
    ensure_db_initialized()
    db = SessionLocal()
    try:
        article = None

        if article_id:
            article = db.query(Article).filter(Article.id == str(article_id)).first()

        if article is None and url:
            article = db.query(Article).filter(Article.url == url).first()

        if article is None:
            return None

        return {
            "title": article.title or "",
            "sapo": article.sapo or "",
            "body": article.body or "",
        }
    finally:
        db.close()


def get_latest_articles(limit: int = 10) -> tuple[list[dict[str, Any]], int, int]:
    """Return latest dated articles ordered by published_at DESC along with coverage counts."""
    ensure_db_initialized()
    db = SessionLocal()
    try:
        total_count = db.query(Article).count()
        dated_query = db.query(Article).filter(Article.published_at.isnot(None))
        dated_count = dated_query.count()
        rows = dated_query.order_by(Article.published_at.desc()).limit(limit).all()

        articles: list[dict[str, Any]] = []
        for article in rows:
            published_at = article.published_at
            if published_at is not None and hasattr(published_at, "strftime"):
                published_text = published_at.strftime("%d/%m/%Y %H:%M")
                published_iso = published_at.isoformat(sep=" ")
            else:
                published_text = str(published_at or "")
                published_iso = published_text

            articles.append(
                {
                    "article_id": getattr(article, "id", None),
                    "title": getattr(article, "title", None) or "Không có tiêu đề",
                    "source": getattr(article, "source", None) or "Không rõ nguồn",
                    "url": getattr(article, "url", None) or "",
                    "published_at": published_iso,
                    "published_at_display": published_text,
                    "similarity": None,
                }
            )

        return articles, total_count, dated_count
    finally:
        db.close()


def get_article_count() -> int:
    """Return total number of articles stored in SQLite."""
    ensure_db_initialized()
    db = SessionLocal()
    try:
        return db.query(Article).count()
    finally:
        db.close()
