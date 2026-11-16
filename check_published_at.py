
"""
KIỂM TRA TRƯỜNG published_at TRONG NEWS_15K.DB

- Chỉ đọc database, không chỉnh sửa dữ liệu.
- Chạy từ thư mục gốc proj_chatbot:
    python check_published_at.py

- Nếu database nằm ở vị trí khác:
    python check_published_at.py --db data/news_15k.db
"""

import argparse
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra published_at trong news_15k.db"
    )

    parser.add_argument(
        "--db",
        default="data/news_15k.db",
        help="Đường dẫn đến database"
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Số bài thiếu ngày muốn hiển thị"
    )

    args = parser.parse_args()

    db_path = Path(args.db)

    if not db_path.is_file():
        raise SystemExit(
            f"Không tìm thấy database: {db_path.resolve()}"
        )

    # Mở database ở chế độ chỉ đọc
    con = sqlite3.connect(
        f"file:{db_path.resolve().as_posix()}?mode=ro",
        uri=True
    )

    con.row_factory = sqlite3.Row

    try:
        # Kiểm tra bảng articles
        tables = {
            row["name"]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

        if "articles" not in tables:
            raise SystemExit("Không tìm thấy bảng articles.")

        columns = {
            row["name"]
            for row in con.execute(
                "PRAGMA table_info(articles)"
            )
        }

        required = {
            "id", "source", "category", "published_at"
        }

        missing_columns = required - columns

        if missing_columns:
            raise SystemExit(
                f"Thiếu cột: {missing_columns}"
            )

        # 1. Tổng quan
        total = con.execute(
            "SELECT COUNT(*) FROM articles"
        ).fetchone()[0]

        present = con.execute("""
            SELECT COUNT(*)
            FROM articles
            WHERE published_at IS NOT NULL
            AND TRIM(published_at) <> ''
        """).fetchone()[0]

        missing = total - present

        print("=" * 65)
        print("KIỂM TRA PUBLISHED_AT - NEWS_15K")
        print("=" * 65)

        print(f"Tổng số bài: {total:,}")

        if total > 0:
            print(
                f"Có ngày xuất bản: {present:,} "
                f"({present / total * 100:.2f}%)"
            )

            print(
                f"Thiếu ngày xuất bản: {missing:,} "
                f"({missing / total * 100:.2f}%)"
            )

        # 2. Theo nguồn báo
        print("\n[1] THỐNG KÊ THEO NGUỒN")

        rows = con.execute("""
            SELECT
                source,
                COUNT(*) AS total,
                SUM(
                    CASE
                        WHEN published_at IS NOT NULL
                        AND TRIM(published_at) <> ''
                        THEN 1 ELSE 0
                    END
                ) AS present
            FROM articles
            GROUP BY source
            ORDER BY source
        """).fetchall()

        for row in rows:
            absent = row["total"] - row["present"]

            print(
                f"{row['source']}: "
                f"Tổng={row['total']}, "
                f"Có ngày={row['present']}, "
                f"Thiếu={absent}"
            )

        # 3. Theo danh mục
        print("\n[2] THỐNG KÊ THEO DANH MỤC")

        rows = con.execute("""
            SELECT
                category,
                COUNT(*) AS total,
                SUM(
                    CASE
                        WHEN published_at IS NOT NULL
                        AND TRIM(published_at) <> ''
                        THEN 1 ELSE 0
                    END
                ) AS present
            FROM articles
            GROUP BY category
            ORDER BY category
        """).fetchall()

        for row in rows:
            absent = row["total"] - row["present"]

            print(
                f"{row['category']}: "
                f"Tổng={row['total']}, "
                f"Có ngày={row['present']}, "
                f"Thiếu={absent}"
            )

        # 4. Phân bố theo năm
        print("\n[3] PHÂN BỐ THEO NĂM")

        rows = con.execute("""
            SELECT
                SUBSTR(TRIM(published_at), 1, 4) AS year,
                COUNT(*) AS total
            FROM articles
            WHERE published_at IS NOT NULL
            AND TRIM(published_at) <> ''
            GROUP BY year
            ORDER BY year
        """).fetchall()

        for row in rows:
            print(f"Năm {row['year']}: {row['total']} bài")

        # 5. Khoảng ngày
        print("\n[4] KHOẢNG NGÀY XUẤT BẢN")

        row = con.execute("""
            SELECT
                MIN(published_at) AS earliest,
                MAX(published_at) AS latest
            FROM articles
            WHERE published_at IS NOT NULL
            AND TRIM(published_at) <> ''
        """).fetchone()

        print(f"Sớm nhất: {row['earliest']}")
        print(f"Muộn nhất: {row['latest']}")

        # 6. Số bài thiếu ngày theo nguồn và danh mục
        print("\n[5] BÀI THIẾU NGÀY THEO NGUỒN × DANH MỤC")

        rows = con.execute("""
            SELECT
                source,
                category,
                COUNT(*) AS missing
            FROM articles
            WHERE published_at IS NULL
            OR TRIM(published_at) = ''
            GROUP BY source, category
            ORDER BY source, category
        """).fetchall()

        if rows:
            for row in rows:
                print(
                    f"{row['source']} / {row['category']}: "
                    f"{row['missing']} bài"
                )
        else:
            print("Không còn bài thiếu ngày.")

        # 7. Ví dụ bài thiếu ngày
        print(
            f"\n[6] DANH SÁCH BÀI THIẾU NGÀY "
            f"(tối đa {args.examples} bài)"
        )

        rows = con.execute("""
            SELECT
                id,
                source,
                category,
                title,
                url
            FROM articles
            WHERE published_at IS NULL
            OR TRIM(published_at) = ''
            ORDER BY source, category, id
            LIMIT ?
        """, (max(args.examples, 0),)).fetchall()

        if rows:
            for index, row in enumerate(rows, 1):
                print(f"\n{index}. ID: {row['id']}")
                print(
                    f"   Nguồn: "
                    f"{row['source']} / {row['category']}"
                )
                print(f"   Tiêu đề: {row['title']}")
                print(f"   URL: {row['url']}")
        else:
            print("Không có bài nào thiếu ngày.")

        # 8. Kiểm tra split
        print("\n[7] KIỂM TRA SPLIT")

        if "split" in columns:
            rows = con.execute("""
                SELECT
                    COALESCE(
                        NULLIF(TRIM(split), ''),
                        '[NULL/BLANK]'
                    ) AS split_name,
                    COUNT(*) AS total
                FROM articles
                GROUP BY split_name
                ORDER BY split_name
            """).fetchall()

            for row in rows:
                print(
                    f"{row['split_name']}: {row['total']} bài"
                )
        else:
            print("Database không có cột split.")

        print("\n" + "=" * 65)
        print("HOÀN TẤT KIỂM TRA.")
        print("DATABASE ĐƯỢC MỞ Ở CHẾ ĐỘ READ-ONLY.")
        print("=" * 65)

    finally:
        con.close()


if __name__ == "__main__":
    main()
