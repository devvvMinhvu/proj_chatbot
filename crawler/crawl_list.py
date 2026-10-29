import time
import requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit, urlunsplit

from database.database_get import SessionLocal, Article
from crawler.nhandan import crawler_article, save_article

from crawler.article_quality import is_non_text_article
# ==========================================
# CẤU HÌNH
# ==========================================

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# Mốc kiểm thử ban đầu, chưa phải mục tiêu cuối cùng
TARGET_PER_TOPIC = 1000

# Trang HTML đầu tiên và các trang API tiếp theo
MAX_PAGES = 100

# Giới hạn số URL thu thập cho mỗi chủ đề
MAX_URLS_PER_TOPIC = 3000
REQUEST_DELAY = 0.3


# ==========================================
# NGUỒN NHÂN DÂN THEO CHỦ ĐỀ
# ==========================================

TOPIC_SOURCES = {
    "Thời sự": {
        "url": "https://nhandan.vn/chinhtri/",
        "zone_id": 1171
    },
    "Kinh tế": {
        "url": "https://nhandan.vn/kinhte/",
        "zone_id": 1185
    },
    "Công nghệ": {
        "url": "https://nhandan.vn/khoahoc-congnghe/",
        "zone_id": 1292
    },
    "Giải trí": {
        "url": "https://nhandan.vn/vanhoa/",
        "zone_id": 1251
    },
    "Thể thao": {
        "url": "https://nhandan.vn/thethao/",
        "zone_id": 1224
    }
}


# ==========================================
# CHUẨN HÓA URL BÀI BÁO
# ==========================================

def normalize_article_url(href, base_url):
    if not href:
        return None

    full_url = urljoin(base_url, href)

    # Bỏ fragment và query string nếu có
    parsed = urlsplit(full_url)
    full_url = urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        "",
        ""
    ))

    # Chỉ chấp nhận bài từ Nhân Dân
    if parsed.netloc not in ("nhandan.vn", "www.nhandan.vn"):
        return None

    # URL bài viết Nhân Dân thường có hậu tố -post
    if "-post" not in parsed.path:
        return None

    return full_url


# ==========================================
# LẤY URL TỪ TRANG HTML VÀ API PHÂN TRANG
# ==========================================

def get_article_urls(
    category_url,
    zone_id,
    max_pages=MAX_PAGES,
    max_urls=MAX_URLS_PER_TOPIC
):
    article_urls = []
    seen_urls = set()

    for page in range(1, max_pages + 1):
        api_url = (
            f"https://api.nhandan.vn/api/"
            f"morenews-zone-{zone_id}-{page}.html"
            f"?phrase=&page_size=30"
            f"&sz={zone_id}&st=zone"
        )

        print(f"Đang lấy trang API {page}, zone {zone_id}")

        try:
            response = requests.get(
                api_url,
                headers=HEADERS,
                timeout=15
            )
            response.raise_for_status()

            result = response.json()

            if result.get("error_code") not in (0, "0", None):
                print("API báo lỗi:", result.get("error_message"))
                break

            contents = result.get("data", {}).get("contents", [])

            if not contents:
                print("API không còn bài.")
                break

            matched_count = 0
            new_count = 0

            for item in contents:
                zone = item.get("zone") or {}

                item_zone_id = zone.get("zone_id")
                parent_id = zone.get("parent_id")

                # Chỉ lấy chuyên mục gốc hoặc chuyên mục con
                # trực tiếp thuộc zone đang thu thập.
                if (
                    str(item_zone_id) != str(zone_id)
                    and str(parent_id) != str(zone_id)
                ):
                    continue

                matched_count += 1

                # Lấy tiêu đề và URL
                title = (item.get("title") or "").strip()
                href = item.get("url") or ""

                # Loại bài ảnh, video, infographic...
                if is_non_text_article(
                    title,
                    href,
                    zone.get("name", "")
                ):
                    continue

                # Chuẩn hóa URL
                url = normalize_article_url(
                    href,
                    category_url
                )

                # Bỏ URL không hợp lệ hoặc bị trùng
                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)
                article_urls.append(url)
                new_count += 1

                if len(article_urls) >= max_urls:
                    print("Đã đạt giới hạn URL:", max_urls)
                    return article_urls

            print(
                f"Trang {page}: {len(contents)} bài API, "
                f"{matched_count} bài phù hợp zone, "
                f"{new_count} URL mới"
            )

            # Không dừng chỉ vì một trang không có URL phù hợp:
            # các trang sau vẫn có thể chứa bài đúng chủ đề.
            time.sleep(REQUEST_DELAY)

        except (requests.RequestException, ValueError) as e:
            print(f"Lỗi lấy trang API {page}: {e}")
            break

    return article_urls


# ==========================================
# THỐNG KÊ CƠ SỞ DỮ LIỆU
# ==========================================

def get_topic_count(topic):
    db = SessionLocal()

    try:
        return db.query(Article).filter(
            Article.research_topic == topic
        ).count()
    finally:
        db.close()


def article_exists(url):
    db = SessionLocal()

    try:
        return db.query(Article).filter(
            Article.url == url
        ).first() is not None
    finally:
        db.close()


# ==========================================
# CRAWL MỘT CHỦ ĐỀ
# ==========================================

def crawl_topic(topic, source):
    category_url = source["url"]
    zone_id = source["zone_id"]

    current_count = get_topic_count(topic)

    print("\n" + "=" * 55)
    print("CHỦ ĐỀ:", topic)
    print("Zone ID:", zone_id)
    print("Đã có:", current_count)
    print("Mục tiêu kiểm thử:", TARGET_PER_TOPIC)
    print("=" * 55)

    if current_count >= TARGET_PER_TOPIC:
        print("Đã đạt mục tiêu kiểm thử, bỏ qua.")
        return

    urls = get_article_urls(
        category_url=category_url,
        zone_id=zone_id,
        max_pages=MAX_PAGES,
        max_urls=MAX_URLS_PER_TOPIC
    )

    print("Tổng URL thu được:", len(urls))

    for index, url in enumerate(urls, start=1):

        # Kiểm tra lại số liệu mới nhất
        current_count = get_topic_count(topic)

        if current_count >= TARGET_PER_TOPIC:
            print("Đã đủ số bài kiểm thử.")
            break

        # Bỏ qua URL đã có trong database
        if article_exists(url):
            continue

        print(f"\n[{index}/{len(urls)}] Đang crawl:")
        print(url)

        try:
            article = crawler_article(
                url,
                research_topic=topic
            )

            title = article.get("title", "").strip()
            content = article.get("content", "").strip()

            if not title:
                print("Bỏ qua: bài báo không có tiêu đề.")
                continue

            if not content:
                print("Bỏ qua: bài báo không có nội dung.")
                continue

            # Hàm save_article sẽ kiểm tra URL trùng
            # và lưu bài bằng Session riêng.
            save_article(article)

            # Xác nhận bài thực sự đã xuất hiện trong database
            if article_exists(url):
                current_count = get_topic_count(topic)

                print(
                    f"Tiến độ {topic}: "
                    f"{current_count}/{TARGET_PER_TOPIC}"
                )
            else:
                print("Chưa xác nhận được bài đã lưu thành công.")

        except requests.RequestException as e:
            print("Lỗi tải bài báo:", e)

        except Exception as e:
            print("Lỗi xử lý bài báo:", e)

        time.sleep(REQUEST_DELAY)

    print(
        f"\nKết thúc chủ đề {topic}: "
        f"{get_topic_count(topic)} bài có nhãn này trong database."
    )


# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================

def main():
    try:
        for topic, source in TOPIC_SOURCES.items():
            crawl_topic(topic, source)

    except KeyboardInterrupt:
        print("\nĐã dừng theo yêu cầu.")

    finally:
        print("\n===== TỔNG KẾT =====")

        total = 0

        for topic in TOPIC_SOURCES:
            count = get_topic_count(topic)
            total += count
            print(f"{topic}: {count} bài")

        print("Tổng số bản ghi theo 5 nhãn:", total)
        print("Chương trình kết thúc.")


if __name__ == "__main__":
    main()