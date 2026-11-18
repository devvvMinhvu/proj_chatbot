"""Database query helpers for retrieving articles, topic searches, and publication listings."""

from __future__ import annotations

from typing import Any, Callable

from sqlalchemy import or_

from database.db import Article, SessionLocal, ensure_db_initialized


def _format_published_fields(published_at: Any) -> tuple[str, str]:
    if published_at is not None and hasattr(published_at, "strftime"):
        return published_at.strftime("%d/%m/%Y %H:%M"), published_at.isoformat(sep=" ")
    text = str(published_at or "")
    return text, text


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
            published_text, published_iso = _format_published_fields(article.published_at)
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


def search_latest_topic_articles(
    terms: list[str],
    limit: int = 10,
    session_factory: Any = SessionLocal,
    article_model: Any = Article,
) -> tuple[list[dict[str, Any]], int, int]:
    """Find articles containing all topic terms ordered by published_at DESC."""
    if not terms:
        return [], 0, 0

    if session_factory is SessionLocal:
        ensure_db_initialized()

    db = session_factory()
    try:
        query = db.query(article_model)

        for term in terms:
            pattern = f"%{term}%"
            query = query.filter(
                or_(
                    article_model.title.ilike(pattern),
                    article_model.sapo.ilike(pattern),
                    article_model.body.ilike(pattern),
                )
            )

        total_count = query.count()
        dated_query = query.filter(article_model.published_at.isnot(None))
        dated_count = dated_query.count()
        rows = dated_query.order_by(article_model.published_at.desc()).limit(limit).all()

        articles: list[dict[str, Any]] = []
        for article in rows:
            published_display, published_iso = _format_published_fields(article.published_at)
            articles.append(
                {
                    "article_id": getattr(article, "id", None),
                    "title": getattr(article, "title", None) or "Không có tiêu đề",
                    "source": getattr(article, "source", None) or "Không rõ nguồn",
                    "url": getattr(article, "url", None) or "",
                    "category": getattr(article, "category", None) or "",
                    "published_at": published_iso,
                    "published_at_display": published_display,
                    "similarity": None,
                }
            )

        return articles, total_count, dated_count
    finally:
        db.close()


def search_topic_articles(
    accepted_values: set[str],
    normalize_fn: Callable[[str], str],
    limit: int = 10,
    session_factory: Any = SessionLocal,
    article_model: Any = Article,
) -> tuple[list[dict[str, Any]], int, bool, list[str]]:
    """Filter articles by SQLite category column ordered by published_at DESC."""
    if session_factory is SessionLocal:
        ensure_db_initialized()

    db = session_factory()
    try:
        category_rows = (
            db.query(article_model.category)
            .filter(article_model.category.isnot(None))
            .distinct()
            .all()
        )
        available_categories = [row[0] for row in category_rows if row and row[0]]
        matched_categories = [
            value for value in available_categories if normalize_fn(value) in accepted_values
        ]

        if not matched_categories:
            return [], 0, False, available_categories

        query = db.query(article_model).filter(article_model.category.in_(matched_categories))
        total_count = query.count()
        rows = query.order_by(article_model.published_at.desc()).limit(limit).all()

        articles: list[dict[str, Any]] = []
        for article in rows:
            published_text, _ = _format_published_fields(getattr(article, "published_at", None))
            articles.append(
                {
                    "article_id": getattr(article, "id", None),
                    "title": getattr(article, "title", None) or "Không có tiêu đề",
                    "source": getattr(article, "source", None) or "Không rõ nguồn",
                    "url": getattr(article, "url", None) or "",
                    "category": getattr(article, "category", None) or "",
                    "published_at": published_text,
                    "similarity": None,
                }
            )

        return articles, total_count, True, available_categories
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
