import json
from pathlib import Path

from sqlalchemy import select, func

from database.database_15k import (
    Article,
    SessionLocal,
    init_db,
)


# ==================================================
# 1. ĐƯỜNG DẪN FILE JSONL
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JSONL_PATH = PROJECT_ROOT / "data" / "news_15k.jsonl"

BATCH_SIZE = 500


# ==================================================
# 2. CHUẨN HÓA GIÁ TRỊ
# ==================================================

def to_text(value):
    """Chuyển giá trị JSON thành chuỗi an toàn."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


# ==================================================
# 3. NHẬP DỮ LIỆU THEO TỪNG LÔ
# ==================================================

def import_jsonl():
    if not JSONL_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file JSONL: {JSONL_PATH}"
        )

    # Tạo database và bảng nếu chưa tồn tại.
    init_db()

    processed = 0
    inserted = 0
    skipped_duplicate = 0
    invalid = 0

    batch = []
    batch_ids = set()

    db = SessionLocal()

    try:
        with JSONL_PATH.open(
            "r",
            encoding="utf-8-sig"
        ) as file:

            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                # Đọc từng dòng JSON.
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    invalid += 1
                    print(
                        f"Lỗi JSON tại dòng {line_number}: {exc}"
                    )
                    continue

                if not isinstance(item, dict):
                    invalid += 1
                    print(
                        f"Bỏ qua dòng {line_number}: "
                        "dữ liệu không phải JSON object."
                    )
                    continue

                article_id = to_text(item.get("id"))

                if not article_id:
                    invalid += 1
                    print(
                        f"Bỏ qua dòng {line_number}: thiếu id."
                    )
                    continue

                processed += 1

                # Tránh trùng ID trong cùng một lô.
                if article_id in batch_ids:
                    skipped_duplicate += 1
                    continue

                batch_ids.add(article_id)

                batch.append({
                    "id": article_id,
                    "source": to_text(item.get("source")),
                    "category": to_text(item.get("category")),
                    "split": to_text(item.get("split")),
                    "title": to_text(item.get("title")),
                    "url": to_text(item.get("url")),
                    "body": to_text(item.get("body")),
                    "sapo": to_text(item.get("sapo")),
                })

                if len(batch) >= BATCH_SIZE:
                    inserted_now, duplicates_now = (
                        save_batch(db, batch)
                    )

                    inserted += inserted_now
                    skipped_duplicate += duplicates_now

                    batch.clear()
                    batch_ids.clear()

                    print(
                        f"Đã xử lý {processed} bản ghi; "
                        f"thêm mới {inserted} bài."
                    )

        # Lưu lô cuối cùng còn lại.
        if batch:
            inserted_now, duplicates_now = save_batch(
                db,
                batch,
            )

            inserted += inserted_now
            skipped_duplicate += duplicates_now

        total_articles = db.scalar(
            select(func.count()).select_from(Article)
        )

        missing_title = db.scalar(
            select(func.count())
            .select_from(Article)
            .where(
                (Article.title.is_(None))
                | (Article.title == "")
            )
        )

        missing_body = db.scalar(
            select(func.count())
            .select_from(Article)
            .where(
                (Article.body.is_(None))
                | (Article.body == "")
            )
        )

        print("\n========== KẾT QUẢ NHẬP JSONL ==========")
        print("Số bản ghi hợp lệ đã đọc:", processed)
        print("Số bài mới thêm:", inserted)
        print("Số ID trùng bị bỏ qua:", skipped_duplicate)
        print("Số dòng lỗi hoặc thiếu ID:", invalid)
        print("Tổng số bài trong database:", total_articles)
        print("Bài thiếu title:", missing_title)
        print("Bài thiếu body:", missing_body)
        print("Database:", PROJECT_ROOT / "data" / "news_15k.db")

    finally:
        db.close()


# ==================================================
# 4. LƯU MỘT LÔ
# ==================================================

def save_batch(db, records):
    """Chỉ thêm ID chưa có; không ghi đè bài đã nhập trước đó."""

    ids = [record["id"] for record in records]

    existing_ids = set(
        db.scalars(
            select(Article.id).where(Article.id.in_(ids))
        ).all()
    )

    new_records = [
        record
        for record in records
        if record["id"] not in existing_ids
    ]

    duplicate_count = len(records) - len(new_records)

    if new_records:
        try:
            db.add_all([
                Article(**record)
                for record in new_records
            ])
            db.commit()

        except Exception:
            db.rollback()
            raise

    return len(new_records), duplicate_count


if __name__ == "__main__":
    import_jsonl()