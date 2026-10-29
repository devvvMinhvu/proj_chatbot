from database.database_get import SessionLocal, Article
from collections import Counter


db = SessionLocal()

articles = db.query(Article).all()

print("========== KIỂM TRA DATASET ==========")
print("Tổng số bài:", len(articles))


# ======================================
# 1. Tìm bài bị thiếu content
# ======================================

print("\n========== BÀI THIẾU CONTENT ==========")

missing_content = []

for article in articles:
    if not article.content or not article.content.strip():
        missing_content.append(article)

if len(missing_content) == 0:
    print("Không có bài nào thiếu content.")
else:
    for article in missing_content:
        print("\nID:", article.id)
        print("Title:", article.title)
        print("Category:", article.category)
        print("Source:", article.source)
        print("URL:", article.url)

    print("\nSố bài thiếu content:", len(missing_content))


# ======================================
# 2. Chuẩn hóa source để thống kê
# ======================================

print("\n========== SOURCE ==========")

normalized_sources = []

for article in articles:

    source = article.source

    if source:
        source = source.strip().lower()

        if source == "báo nhân dân":
            source = "Báo Nhân Dân"

    normalized_sources.append(source)


source_counter = Counter(normalized_sources)

for source, count in source_counter.items():
    print(source, ":", count)


# ======================================
# 3. Category
# ======================================

print("\n========== CATEGORY ==========")

category_counter = Counter(
    article.category.strip()
    for article in articles
    if article.category
)

for category, count in category_counter.items():
    print(category, ":", count)


# ======================================
# 4. Kiểm tra dữ liệu cơ bản
# ======================================

print("\n========== DỮ LIỆU THIẾU ==========")

missing_title = 0
missing_content_count = 0
missing_category = 0

for article in articles:

    if not article.title or not article.title.strip():
        missing_title += 1

    if not article.content or not article.content.strip():
        missing_content_count += 1

    if not article.category or not article.category.strip():
        missing_category += 1


print("Thiếu title:", missing_title)
print("Thiếu content:", missing_content_count)
print("Thiếu category:", missing_category)


db.close()