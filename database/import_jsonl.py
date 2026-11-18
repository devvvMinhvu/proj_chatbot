"""Batch JSONL-to-SQLite importer supporting published_at and backfill updates."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from config import NEWS_DB_PATH, NEWS_JSONL_PATH
from crawler.config import TIMEZONE
from crawler.extractor import parse_published_at
from database.db import Article, SessionLocal, init_db


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _parse_datetime(value: Any) -> datetime | None:
    """Parse an ISO-8601 or publisher timestamp into a naive Vietnam-time datetime."""
    if not value:
        return None
    normalized = parse_published_at(str(value))
    if not normalized:
        return None
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(TIMEZONE).replace(tzinfo=None)
    return parsed


def _save_batch(db, records: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Insert new articles and fill missing body/published_at on existing rows."""
    ids = [record["id"] for record in records]
    existing_articles = {
        article.id: article
        for article in db.scalars(select(Article).where(Article.id.in_(ids))).all()
    }

    new_records: list[dict[str, Any]] = []
    updated_count = 0
    unchanged_count = 0

    for record in records:
        existing = existing_articles.get(record["id"])
        if existing is None:
            new_records.append(record)
            continue

        changed = False
        if (not existing.body or not existing.body.strip()) and record["body"]:
            existing.body = record["body"]
            changed = True
        if existing.published_at is None and record["published_at"] is not None:
            existing.published_at = record["published_at"]
            changed = True

        if changed:
            updated_count += 1
        else:
            unchanged_count += 1

    if new_records or updated_count:
        try:
            if new_records:
                db.add_all([Article(**record) for record in new_records])
            db.commit()
        except Exception:
            db.rollback()
            raise

    return len(new_records), updated_count, unchanged_count


def import_jsonl(
    jsonl_path: Path = NEWS_JSONL_PATH,
    batch_size: int = 500,
) -> dict[str, int]:
    """Import articles from a JSONL file into SQLite in batches."""
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file JSONL: {jsonl_path}")

    init_db()

    processed = 0
    inserted = 0
    updated = 0
    skipped_existing = 0
    skipped_duplicate = 0
    invalid = 0

    batch: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    db = SessionLocal()

    try:
        with jsonl_path.open("r", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    invalid += 1
                    print(f"Lỗi JSON tại dòng {line_number}: {exc}")
                    continue

                if not isinstance(item, dict):
                    invalid += 1
                    print(f"Bỏ qua dòng {line_number}: dữ liệu không phải JSON object.")
                    continue

                article_id = _to_text(item.get("id"))
                if not article_id:
                    invalid += 1
                    print(f"Bỏ qua dòng {line_number}: thiếu id.")
                    continue

                processed += 1
                if article_id in seen_ids:
                    skipped_duplicate += 1
                    continue
                seen_ids.add(article_id)

                batch.append(
                    {
                        "id": article_id,
                        "source": _to_text(item.get("source")),
                        "category": _to_text(item.get("category")),
                        "split": _to_text(item.get("split")),
                        "title": _to_text(item.get("title")),
                        "url": _to_text(item.get("url")),
                        "body": _to_text(item.get("body")),
                        "sapo": _to_text(item.get("sapo")),
                        "published_at": _parse_datetime(item.get("published_at")),
                    }
                )

                if len(batch) >= batch_size:
                    inserted_now, updated_now, unchanged_now = _save_batch(db, batch)
                    inserted += inserted_now
                    updated += updated_now
                    skipped_existing += unchanged_now
                    batch.clear()
                    print(f"Đã xử lý {processed} bản ghi; thêm mới {inserted}, cập nhật {updated}.")

        if batch:
            inserted_now, updated_now, unchanged_now = _save_batch(db, batch)
            inserted += inserted_now
            updated += updated_now
            skipped_existing += unchanged_now

        total_articles = db.scalar(select(func.count()).select_from(Article)) or 0
        dated_articles = (
            db.scalar(select(func.count()).select_from(Article).where(Article.published_at.isnot(None))) or 0
        )

        print("\n========== KẾT QUẢ NHẬP JSONL ==========")
        print("Số bản ghi hợp lệ đã đọc:", processed)
        print("Số bài mới thêm:", inserted)
        print("Số bài được cập nhật:", updated)
        print("Số bài đã tồn tại không thay đổi:", skipped_existing)
        print("Số ID trùng trong JSONL bị bỏ qua:", skipped_duplicate)
        print("Số dòng lỗi hoặc thiếu ID:", invalid)
        print("Tổng số bài trong database:", total_articles)
        print("Số bài có ngày đăng:", dated_articles)
        print("Database:", NEWS_DB_PATH)

        return {
            "processed": processed,
            "inserted": inserted,
            "updated": updated,
            "skipped_existing": skipped_existing,
            "skipped_duplicate": skipped_duplicate,
            "invalid": invalid,
            "total_articles": total_articles,
            "dated_articles": dated_articles,
        }
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Nhập dữ liệu bài báo từ JSONL vào SQLite.")
    parser.add_argument(
        "--input",
        type=Path,
        default=NEWS_JSONL_PATH,
        help=f"Đường dẫn file JSONL đầu vào (mặc định: {NEWS_JSONL_PATH}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Số bản ghi mỗi lô ghi vào SQLite (mặc định: 500).",
    )
    args = parser.parse_args()
    import_jsonl(jsonl_path=args.input, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
