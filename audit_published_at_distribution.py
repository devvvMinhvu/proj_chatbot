
import sqlite3

conn = sqlite3.connect("data/news_15k.db")
conn.row_factory = sqlite3.Row

print("===== PHAN BO ID CO PUBLISHED_AT =====")

rows = conn.execute("""
    SELECT source, category,
           MIN(id) AS min_id,
           MAX(id) AS max_id,
           COUNT(*) AS total
    FROM articles
    WHERE published_at IS NOT NULL
      AND TRIM(CAST(published_at AS TEXT)) != ''
    GROUP BY source, category
    ORDER BY source, category
""").fetchall()

for row in rows:
    print(dict(row))

print("\n===== PHAN BO ID THIEU PUBLISHED_AT =====")

rows = conn.execute("""
    SELECT source, category,
           MIN(id) AS min_id,
           MAX(id) AS max_id,
           COUNT(*) AS total
    FROM articles
    WHERE published_at IS NULL
       OR TRIM(CAST(published_at AS TEXT)) = ''
    GROUP BY source, category
    ORDER BY source, category
""").fetchall()

for row in rows:
    print(dict(row))
