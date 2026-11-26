"""Database query helpers for retrieving articles, topic searches, and FTS5 search."""

from __future__ import annotations

from datetime import date, datetime, time
import re
from typing import Any, Callable

from sqlalchemy import func, or_, text

from database.db import Article, SessionLocal, ensure_db_initialized


def _format_published_fields(published_at: Any) -> tuple[str, str]:
    if isinstance(published_at, str) and published_at.strip():
        try:
            published_at = datetime.fromisoformat(published_at.strip())
        except ValueError:
            pass
    if published_at is not None and hasattr(published_at, "strftime"):
        return (
            published_at.strftime("%d/%m/%Y %H:%M"),
            published_at.isoformat(sep=" ", timespec="seconds"),
        )
    text_value = str(published_at or "")
    return text_value, text_value


def _article_to_dict(
    article: Any,
    *,
    include_body: bool = False,
    similarity: float | None = None,
) -> dict[str, Any]:
    published_display, published_iso = _format_published_fields(
        getattr(article, "published_at", None)
    )
    record: dict[str, Any] = {
        "article_id": getattr(article, "id", None),
        "title": getattr(article, "title", None) or "Không có tiêu đề",
        "sapo": getattr(article, "sapo", None) or "",
        "source": getattr(article, "source", None) or "Không rõ nguồn",
        "category": getattr(article, "category", None) or "",
        "url": getattr(article, "url", None) or "",
        "published_at": published_iso,
        "published_at_display": published_display,
        "similarity": similarity,
    }
    if include_body:
        record["body"] = getattr(article, "body", None) or ""
    return record


def _sanitize_fts_tokens(terms: list[str]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for term in terms:
        for piece in re.findall(r"[0-9a-zA-ZÀ-ỹĐđ]+", str(term or "")):
            lowered = piece.casefold()
            if lowered and lowered not in seen:
                seen.add(lowered)
                tokens.append(piece)
    return tokens


def get_article_content(
    article_id: str | None = None,
    url: str | None = None,
) -> dict[str, Any] | None:
    """Fetch article metadata, sapo, and full body from SQLite; prefer ID, then URL."""
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

        return _article_to_dict(article, include_body=True)
    finally:
        db.close()


def get_latest_articles(
    limit: int = 10,
    source: str | None = None,
    category: str | None = None,
    published_from: date | None = None,
    published_to: date | None = None,
    exclude_sources: tuple[str, ...] | list[str] = (),
) -> tuple[list[dict[str, Any]], int, int]:
    """Return latest dated articles ordered by published_at DESC along with coverage counts.

    ``published_from`` / ``published_to`` (inclusive dates) limit the publication date;
    ``exclude_sources`` leaves those publishers out.
    """
    ensure_db_initialized()
    db = SessionLocal()
    try:
        query = db.query(Article)
        if source:
            query = query.filter(Article.source == source)
        if exclude_sources:
            query = query.filter(Article.source.notin_(list(exclude_sources)))
        if category:
            query = query.filter(Article.category == category)
        if published_from is not None:
            query = query.filter(Article.published_at >= datetime.combine(published_from, time.min))
        if published_to is not None:
            query = query.filter(Article.published_at <= datetime.combine(published_to, time.max))

        total_count = query.count()
        dated_query = query.filter(Article.published_at.isnot(None))
        dated_count = dated_query.count()
        rows = dated_query.order_by(Article.published_at.desc()).limit(limit).all()

        return [_article_to_dict(article) for article in rows], total_count, dated_count
    finally:
        db.close()


def get_archive_date_range() -> tuple[date | None, date | None]:
    """Oldest and newest publication dates in the database."""
    ensure_db_initialized()
    db = SessionLocal()
    try:
        oldest, newest = db.query(func.min(Article.published_at), func.max(Article.published_at)).one()
        return (oldest.date() if oldest else None), (newest.date() if newest else None)
    finally:
        db.close()


def _exclusion_filters(exclude_sources, params: dict[str, Any]) -> list[str]:
    """SQL conditions leaving out publishers; their names are bound as parameters."""
    names = []
    for index, value in enumerate(exclude_sources or ()):
        params[f"exclude_source_{index}"] = value
        names.append(f":exclude_source_{index}")
    return [f"a.source NOT IN ({', '.join(names)})"] if names else []


def _date_filters(published_from: date | None, published_to: date | None, params: dict[str, Any]) -> list[str]:
    filters = []
    if published_from is not None:
        params["published_from"] = datetime.combine(published_from, time.min).isoformat(sep=" ")
        filters.append("a.published_at >= :published_from")
    if published_to is not None:
        params["published_to"] = datetime.combine(published_to, time.max).isoformat(sep=" ")
        filters.append("a.published_at <= :published_to")
    return filters


def search_fts_articles(
    terms: list[str],
    limit: int = 20,
    source: str | None = None,
    category: str | None = None,
    sort_by_date: bool = False,
    include_body: bool = True,
    exclude_sources: tuple[str, ...] | list[str] = (),
) -> list[dict[str, Any]]:
    """Search articles via SQLite FTS5 with phrase priority and BM25 ranking."""
    tokens = _sanitize_fts_tokens(terms)
    if not tokens or limit <= 0:
        return []

    ensure_db_initialized()
    db = SessionLocal()
    try:
        filters = ["articles_fts MATCH :fts_query"]
        base_params: dict[str, Any] = {}
        if source:
            filters.append("a.source = :source")
            base_params["source"] = source
        if category:
            filters.append("a.category = :category")
            base_params["category"] = category
        filters.extend(_exclusion_filters(exclude_sources, base_params))

        where_clause = " AND ".join(filters)
        if sort_by_date:
            order_clause = "a.published_at IS NULL ASC, a.published_at DESC"
        else:
            # In SQLite FTS5, bm25() returns negative values where more negative = better match.
            order_clause = "bm25(articles_fts, 0.0, 5.0, 2.5, 1.0) ASC, a.published_at DESC"

        fts_queries: list[tuple[str, float]] = []
        if len(tokens) >= 2:
            fts_queries.append(('"' + " ".join(tokens) + '"', 0.90))
        fts_queries.append((" AND ".join(f'"{tok}"' for tok in tokens), 0.78))
        if len(tokens) >= 3:
            fts_queries.append((" OR ".join(f'"{tok}"' for tok in tokens), 0.62))

        collected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for tier_index, (fts_query, base_score) in enumerate(fts_queries):
            if len(collected) >= limit:
                break
            # Only fall back to OR when phrase + AND returned fewer than 3 matches.
            if tier_index == len(fts_queries) - 1 and len(tokens) >= 3 and len(collected) >= 3:
                break

            sql = text(
                f"""
                SELECT
                    a.id, a.source, a.category, a.title, a.url, a.sapo, a.body, a.published_at,
                    bm25(articles_fts, 0.0, 5.0, 2.5, 1.0) AS rank_score
                FROM articles_fts
                JOIN articles AS a ON a.id = articles_fts.id
                WHERE {where_clause}
                ORDER BY {order_clause}
                LIMIT :limit
                """
            )
            params = {**base_params, "fts_query": fts_query, "limit": limit * 2}
            rows = db.execute(sql, params).mappings().all()

            for row in rows:
                article_id = str(row["id"])
                if article_id in seen_ids:
                    continue
                seen_ids.add(article_id)

                published_display, published_iso = _format_published_fields(row["published_at"])
                raw_bm25 = float(row["rank_score"] or 0.0)
                # Normalize negative BM25 magnitude gently into [base_score - 0.05, base_score + 0.08].
                boost = min(0.08, max(0.0, (-raw_bm25) / 100.0))
                record: dict[str, Any] = {
                    "article_id": article_id,
                    "title": row["title"] or "Không có tiêu đề",
                    "sapo": row["sapo"] or "",
                    "source": row["source"] or "Không rõ nguồn",
                    "category": row["category"] or "",
                    "url": row["url"] or "",
                    "published_at": published_iso,
                    "published_at_display": published_display,
                    "similarity": round(base_score + boost, 4),
                    "fts_tier": tier_index,
                }
                if include_body:
                    record["body"] = row["body"] or ""
                collected.append(record)
                if len(collected) >= limit:
                    break

        return collected
    finally:
        db.close()


def _query_terms_match(
    query: Any,
    article_model: Any,
    terms: list[str],
    *,
    contiguous_phrase: bool,
) -> Any:
    if contiguous_phrase and len(terms) >= 2:
        phrase_pattern = f"%{' '.join(terms)}%"
        return query.filter(
            or_(
                article_model.title.ilike(phrase_pattern),
                article_model.sapo.ilike(phrase_pattern),
                article_model.body.ilike(phrase_pattern),
            )
        )

    for term in terms:
        pattern = f"%{term}%"
        query = query.filter(
            or_(
                article_model.title.ilike(pattern),
                article_model.sapo.ilike(pattern),
                article_model.body.ilike(pattern),
            )
        )
    return query


def search_latest_topic_articles(
    terms: list[str],
    limit: int = 10,
    source: str | None = None,
    category: str | None = None,
    session_factory: Any = SessionLocal,
    article_model: Any = Article,
    published_from: date | None = None,
    published_to: date | None = None,
    exclude_sources: tuple[str, ...] | list[str] = (),
) -> tuple[list[dict[str, Any]], int, int]:
    """Find articles matching topic terms ordered by published_at DESC, preferring contiguous phrases.

    The date range and the excluded publishers are applied in the query, so a
    dated request is not limited to the newest matches.
    """
    if not terms:
        return [], 0, 0

    use_fts = session_factory is SessionLocal
    if use_fts:
        ensure_db_initialized()

    # On the live SQLite DB, try FTS5 contiguous phrase search first so unaccented or
    # accented multi-word topics (e.g. "giá vàng") do not match scattered syllables.
    tokens = _sanitize_fts_tokens(terms)
    if use_fts and len(tokens) >= 2:
        db = session_factory()
        try:
            filters = ["articles_fts MATCH :fts_query"]
            params: dict[str, Any] = {"fts_query": '"' + " ".join(tokens) + '"'}
            if source:
                filters.append("a.source = :source")
                params["source"] = source
            if category:
                filters.append("a.category = :category")
                params["category"] = category
            filters.extend(_exclusion_filters(exclude_sources, params))
            filters.extend(_date_filters(published_from, published_to, params))

            where_clause = " AND ".join(filters)
            total_phrase = (
                db.execute(
                    text(
                        f"SELECT COUNT(*) FROM articles_fts JOIN articles a ON a.id = articles_fts.id WHERE {where_clause}"
                    ),
                    params,
                ).scalar()
                or 0
            )
            if total_phrase > 0:
                dated_phrase = (
                    db.execute(
                        text(
                            f"SELECT COUNT(*) FROM articles_fts JOIN articles a ON a.id = articles_fts.id "
                            f"WHERE {where_clause} AND a.published_at IS NOT NULL"
                        ),
                        params,
                    ).scalar()
                    or 0
                )
                rows = db.execute(
                    text(
                        f"SELECT a.id, a.source, a.category, a.title, a.url, a.sapo, a.published_at "
                        f"FROM articles_fts JOIN articles a ON a.id = articles_fts.id "
                        f"WHERE {where_clause} AND a.published_at IS NOT NULL "
                        f"ORDER BY a.published_at DESC LIMIT :limit"
                    ),
                    {**params, "limit": limit},
                ).mappings().all()

                articles = []
                for row in rows:
                    published_display, published_iso = _format_published_fields(row["published_at"])
                    articles.append(
                        {
                            "article_id": row["id"],
                            "title": row["title"] or "Không có tiêu đề",
                            "sapo": row["sapo"] or "",
                            "source": row["source"] or "Không rõ nguồn",
                            "category": row["category"] or "",
                            "url": row["url"] or "",
                            "published_at": published_iso,
                            "published_at_display": published_display,
                            "similarity": None,
                        }
                    )
                return articles, int(total_phrase), int(dated_phrase)
        finally:
            db.close()

    db = session_factory()
    try:
        base_query = db.query(article_model)
        if source and hasattr(article_model, "source"):
            base_query = base_query.filter(article_model.source == source)
        if category and hasattr(article_model, "category"):
            base_query = base_query.filter(article_model.category == category)
        if exclude_sources and hasattr(article_model, "source"):
            base_query = base_query.filter(article_model.source.notin_(list(exclude_sources)))
        if published_from is not None:
            base_query = base_query.filter(
                article_model.published_at >= datetime.combine(published_from, time.min)
            )
        if published_to is not None:
            base_query = base_query.filter(
                article_model.published_at <= datetime.combine(published_to, time.max)
            )

        active_query = None
        if len(terms) >= 2:
            phrase_query = _query_terms_match(base_query, article_model, terms, contiguous_phrase=True)
            if phrase_query.count() > 0:
                active_query = phrase_query

        if active_query is None:
            active_query = _query_terms_match(base_query, article_model, terms, contiguous_phrase=False)

        total_count = active_query.count()
        dated_query = active_query.filter(article_model.published_at.isnot(None))
        dated_count = dated_query.count()
        rows = dated_query.order_by(article_model.published_at.desc()).limit(limit).all()

        return [_article_to_dict(article) for article in rows], total_count, dated_count
    finally:
        db.close()


def search_topic_articles(
    accepted_values: set[str],
    normalize_fn: Callable[[str], str],
    limit: int = 10,
    source: str | None = None,
    session_factory: Any = SessionLocal,
    article_model: Any = Article,
    exclude_sources: tuple[str, ...] | list[str] = (),
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
        if source and hasattr(article_model, "source"):
            query = query.filter(article_model.source == source)
        if exclude_sources and hasattr(article_model, "source"):
            query = query.filter(article_model.source.notin_(list(exclude_sources)))

        total_count = query.count()
        rows = query.order_by(article_model.published_at.desc()).limit(limit).all()

        return [_article_to_dict(article) for article in rows], total_count, True, available_categories
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
