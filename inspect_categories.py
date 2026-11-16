
import sqlite3
from pathlib import Path

DB_PATH = Path("data/news_15k.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

columns = [
    row[1]
    for row in cursor.execute(
        "PRAGMA table_info(articles)"
    ).fetchall()
]

lookup = {col.lower(): col for col in columns}

candidates = [
    "title", "sapo", "description",
    "summary", "source", "published_at", "body"
]

selected = ["category"] + [
    lookup[name]
    for name in candidates
    if name in lookup
]

query = f"""
    SELECT {', '.join('"' + col + '"' for col in selected)}
    FROM articles
    WHERE category = ?
    ORDER BY RANDOM()
    LIMIT 5
"""

categories = [
    "news",
    "economy",
    "tech",
    "entertainment",
    "sports"
]

for category in categories:
    print(f"\n{'=' * 60}")
    print(f"CHUYÊN MỤC: {category.upper()}")
    print("=" * 60)

    rows = cursor.execute(query, (category,)).fetchall()

    for i, row in enumerate(rows, 1):
        print(f"\n--- BÀI {i} ---")

        for col, value in zip(selected, row):
            if value is not None:
                print(f"{col}: {str(value)[:500]}")

conn.close()
