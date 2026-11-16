
import time
import requests

from sqlalchemy import func

from database.database_15k import SessionLocal, Article
from database.backfill_published_at_by_source import (
    fetch_published_at,
    USER_AGENT,
)

SOURCES = ["tienphong", "tuoitre", "vnexpress"]

CATEGORIES = [
    "news",
    "economy",
    "tech",
    "entertainment",
    "sports",
]

SAMPLE_PER_GROUP = 2

db = SessionLocal()
http = requests.Session()

http.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi,en-US;q=0.8,en;q=0.6",
})

success = 0
failed = 0

try:
    print("KIEM TRA NGAY XUAT BAN - MAU NGAU NHIEN")
    print("=" * 70)

    for source in SOURCES:
        for category in CATEGORIES:

            articles = (
                db.query(Article)
                .filter(
                    Article.source == source,
                    Article.category == category,
                    Article.published_at.is_(None),
                    Article.url.isnot(None),
                    Article.url != "",
                )
                .order_by(func.random())
                .limit(SAMPLE_PER_GROUP)
                .all()
            )

            print(f"\n[{source.upper()} - {category.upper()}]")
            print(f"So bai lay mau: {len(articles)}")

            for article in articles:
                print("\nID:", article.id)
                print("Tieu de:", article.title)
                print("URL:", article.url)

                date, method, status = fetch_published_at(
                    article.url, http
                )

                print("Ket qua:", status)

                if date:
                    success += 1
                    print("Ngay xuat ban:", date)
                    print("Metadata:", method)
                else:
                    failed += 1

                time.sleep(1)

    print("\n" + "=" * 70)
    print("TONG KET")
    print("Lay duoc ngay:", success)
    print("Khong lay duoc:", failed)
    print("Tong so bai:", success + failed)
    print("Ty le thanh cong:",
          round(success / max(success + failed, 1) * 100, 2), "%")

    print("\nDRY RUN: Khong ghi thay doi vao database.")

finally:
    http.close()
    db.close()
