import requests
from bs4 import BeautifulSoup
import re
from database.database_get import SessionLocal , Article
from processing.text_clean import clean_text
from crawler.article_quality import (
    normalize_article_url,
    is_non_text_article,
    missing_required_fields,
    has_paragraph_content,
)
from urllib.parse import urlsplit
def crawler_article(url, research_topic=None):

    response = requests.get(
    url,
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=15
)
    response.raise_for_status() 

    soup = BeautifulSoup(response.text, "html.parser")

    # =========================
    # TIÊU ĐỀ
    # =========================

    title_tag = soup.find("h1")

    if title_tag:
        title = clean_text(
            title_tag.get_text(" ", strip=True)
        )
    else:
        title = ""


    # =========================
    # SAPO
    # =========================

    sapo_tag = soup.find(
        "div",
        class_="article__sapo"
    )

    if sapo_tag:
        sapo = clean_text(
            sapo_tag.get_text(" ", strip=True)
        )
    else:
        sapo = ""


    
# =========================
# NỘI DUNG
# =========================

    contentbox = soup.find(
        "div",
        class_="article__body"
    )

    content_part = []

    # Nếu không tìm thấy vùng nội dung, dùng danh sách rỗng
    paragraphs = (
        contentbox.find_all("p")
        if contentbox is not None
        else []
    )

    for paragraph in paragraphs:
    # Xử lý các liên kết được chèn giữa câu
        for link in paragraph.select("a.cms-relate"):
            link_text = link.get_text(" ", strip=True)
            link.replace_with(f" {link_text} ")

        text = paragraph.get_text(" ", strip=True)

    # Chuẩn hóa khoảng trắng liên tiếp
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            content_part.append(text)

    raw_content = "\n".join(content_part)
    cleaned_content = clean_text(raw_content)




    # =========================
    # NGÀY ĐĂNG
    # =========================

    data_tag = soup.find(
        "div",
        class_="article__meta"
    )

    if data_tag:

        metatext = data_tag.get_text(
            " ",
            strip=True
        )

        match = re.search(
            r"\d{2}/\d{2}/\d{4}\s*-\s*\d{2}:\d{2}",
            metatext
        )

        if match:
            published_at = match.group()
        else:
            published_at = ""

    else:
        published_at = ""


    # =========================
    # TÁC GIẢ
    # =========================

    author_tag = soup.find(
        "div",
        class_="article__author"
    )

    if author_tag:
        author = clean_text(
            author_tag.get_text(" ", strip=True)
        )
    else:
        author = ""


    # =========================
    # CATEGORY
    # =========================

    category = ""

    article_box = soup.find(
        "div",
        class_="article"
    )

    if article_box:

        links = article_box.find_all("a")

        for link in links:

            text = link.get_text(
                " ",
                strip=True
            )

            href = link.get("href")

            if (
                href
                and href.startswith("https://nhandan.vn/")
                and text
            ):

                if (
                    "/tu-khoa/" not in href
                    and "/chu-de/" not in href
                ):

                    category = text
                    break


    # =========================
    # TẠO ARTICLE
    # =========================

    article = {
    "title": title,
    "sapo": sapo,
    "content": cleaned_content,
    "category": category,
    "research_topic": research_topic,
    "author": author,
    "published_at": published_at,
    "url": url
}

    return article

def save_article(article):
    title = (article.get("title") or "").strip()
    content = (article.get("content") or "").strip()
    url = normalize_article_url(article.get("url"))

    # Chuẩn hóa những trường văn bản còn lại.
    article_to_save = {
        "title": title,
        "sapo": (article.get("sapo") or "").strip(),
        "content": content,
        "category": (article.get("category") or "").strip(),
        "research_topic": (
            article.get("research_topic") or ""
        ).strip(),
        "author": (article.get("author") or "").strip(),
        "published_at": (
            article.get("published_at") or ""
        ).strip(),
        "source": "Báo Nhân Dân",
        "url": url,
    }
    # Chỉ nhận URL thuộc Báo Nhân Dân
    if urlsplit(url).hostname != "nhandan.vn":
        print("Bỏ qua: URL không thuộc Báo Nhân Dân.")
        return False

# Chỉ nhận 5 nhãn đã thống nhất
    valid_topics = {
        "Thời sự",
        "Kinh tế",
        "Công nghệ",
        "Giải trí",
        "Thể thao",
    }

    if article_to_save["research_topic"] not in valid_topics:
        print("Bỏ qua: nhãn nghiên cứu không hợp lệ.")
        return False
    if is_non_text_article(
        title,
        url,
        article_to_save["category"],
    ):
        print("Bỏ qua: bài ảnh/video/infographic hoặc media khác.")
        return False

    missing = missing_required_fields(
        article_to_save,
        include_source=True,
    )

    if missing:
        print("Bỏ qua: thiếu trường bắt buộc:", ", ".join(missing))
        return False

    if not has_paragraph_content(content):
        print("Bỏ qua: nội dung không đủ tiêu chí văn bản.")
        return False

    db = SessionLocal()

    try:
        existing = db.query(Article).filter(
            Article.url == url
        ).first()

        if existing:
            print("Bỏ qua: URL đã tồn tại.")
            return False

        new_article = Article(
            title=article_to_save["title"],
            sapo=article_to_save["sapo"],
            content=article_to_save["content"],
            category=article_to_save["category"],
            research_topic=article_to_save["research_topic"],
            author=article_to_save["author"],
            published_at=article_to_save["published_at"],
            source=article_to_save["source"],
            url=article_to_save["url"],
        )

        db.add(new_article)
        db.commit()

        print("Đã lưu bài báo hợp lệ!")
        return True

    except Exception as e:
        db.rollback()
        print("Lỗi khi lưu bài báo:", e)
        return False

    finally:
        db.close()


if __name__ == "__main__":
    test_url = (
        "https://nhandan.vn/"
        "khai-mo-khong-gian-phat-trien-moi-post991453.html"
    )

    # Chỉ crawl để kiểm tra, không lưu vào database
    article = crawler_article(
        test_url,
        research_topic="Kinh tế"
    )

    db = SessionLocal()

    try:
        normalized_url = normalize_article_url(test_url)

        saved_article = (
            db.query(Article)
            .filter(Article.url == normalized_url)
            .first()
        )

        print("\n===== SO SÁNH VỚI DATABASE =====")

        if saved_article:
            old_content = saved_article.content or ""
            new_content = article["content"] or ""

            print("Đã tìm thấy bài trong database.")
            print("Độ dài nội dung đang lưu:", len(old_content))
            print("Độ dài nội dung crawl mới:", len(new_content))

            print("\nNội dung đang lưu:")
            print(repr(old_content[:500]))

            print("\nNội dung crawl mới:")
            print(repr(new_content[:500]))

        else:
            print("Database chưa có bài báo này.")

    finally:
        db.close()

    print("\n===== KẾT QUẢ CUỐI CÙNG =====")
    print("Tiêu đề:", article["title"])
    print("Nội dung:", repr(article["content"][:1000]))
   