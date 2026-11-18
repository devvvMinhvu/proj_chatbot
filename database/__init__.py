"""SQLite database package for news articles."""

from __future__ import annotations

from typing import Any

__all__ = [
    "Article",
    "Base",
    "SessionLocal",
    "engine",
    "ensure_db_initialized",
    "get_article_content",
    "get_article_count",
    "get_latest_articles",
    "init_db",
    "rebuild_fts_index",
    "search_fts_articles",
    "search_latest_topic_articles",
    "search_topic_articles",
]

_DB_EXPORTS = {
    "Article",
    "Base",
    "SessionLocal",
    "engine",
    "ensure_db_initialized",
    "init_db",
    "rebuild_fts_index",
}
_REPO_EXPORTS = {
    "get_article_content",
    "get_article_count",
    "get_latest_articles",
    "search_fts_articles",
    "search_latest_topic_articles",
    "search_topic_articles",
}


def __getattr__(name: str) -> Any:
    if name in _DB_EXPORTS:
        from database import db

        return getattr(db, name)
    if name in _REPO_EXPORTS:
        from database import repository

        return getattr(repository, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
