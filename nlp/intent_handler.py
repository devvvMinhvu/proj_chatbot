"""Bo dieu phoi y dinh cho tro ly doc bao.

- greeting/introduction: tra loi truc tiep.
- semantic_search: PhoBERT + FAISS, khong goi Qwen.
- latest_topic_search: tim theo tu khoa chu de trong SQLite, sap xep published_at moi nhat.
- summarization: PhoBERT + FAISS -> SQLite -> ViT5, khong goi Qwen.
- article_detail: PhoBERT + FAISS -> SQLite, khong goi Qwen.
- question_answering: dung pipeline RAG + Qwen hien co.
- latest_news: truy van SQLite theo published_at; neu con bai thieu ngay dang, canh bao ve do phu du lieu.

Cac ham dieu phoi chinh tra ve (answer, sources) cho app.py.
"""

from functools import lru_cache
import re
import unicodedata
from typing import Any

from sqlalchemy import or_

from database.database_15k import SessionLocal, Article
from nlp.intent_classifier import predict_intent
# Nạp pipeline nặng theo nhu cầu, tránh import PyTorch ngay khi mở Streamlit.
# Các intent truy vấn SQLite (topic_search/latest_topic_search/latest_news) không cần
# PhoBERT/FAISS/Qwen nên vẫn có thể chạy ngay cả khi máy đang thiếu bộ nhớ.
def ask(*args, **kwargs):
    from rag.pipeline_15k import ask as pipeline_ask
    return pipeline_ask(*args, **kwargs)


def retrieve(*args, **kwargs):
    from rag.pipeline_15k import retrieve as pipeline_retrieve
    return pipeline_retrieve(*args, **kwargs)


def group_chunks_by_article(*args, **kwargs):
    from rag.pipeline_15k import group_chunks_by_article as pipeline_group
    return pipeline_group(*args, **kwargs)



INTRODUCTION_RESPONSE = """Tôi là Trợ lý đọc báo thông minh, hỗ trợ bạn khai thác
thông tin từ các bài báo tiếng Việt.

Các chức năng chính:
- Tìm bài báo theo chuyên mục bằng dữ liệu category trong SQLite.
- Tìm kiếm bài báo theo nội dung bằng PhoBERT và FAISS.
- Tìm bài báo mới nhất theo chủ đề/từ khóa người dùng yêu cầu.
- Tóm tắt bài báo bằng mô hình ViT5.
- Hỏi đáp dựa trên bài báo bằng RAG và Qwen.
- Xem thông tin nguồn và toàn văn bài báo.
"""


# Các tên chủ đề có thể người dùng nhắc đến.
# Mỗi chủ đề được đối chiếu với category thực tế đang có trong SQLite.
TOPIC_QUERY_ALIASES = {
    "Thời sự": (
        "thời sự", "news",
    ),
    "Kinh tế": (
        "kinh tế", "economy", "economic",
    ),
    "Công nghệ": (
        "công nghệ", "tech", "technology", "khoa học công nghệ", "khoa học và công nghệ",
    ),
    "Giải trí": (
        "giải trí", "entertainment", "điện ảnh", "âm nhạc", "showbiz",
    ),
    "Thể thao": (
        "thể thao", "sports",
    ),
    "Giáo dục": (
        "giáo dục", "education",
    ),
    "Y tế": (
        "y tế", "sức khỏe", "health", "medical",
    ),
    "Môi trường": (
        "môi trường", "environment",
    ),
    "Du lịch": (
        "du lịch", "travel", "tourism",
    ),
    "Pháp luật": (
        "pháp luật", "law", "legal",
    ),
    "Chính trị": (
        "chính trị", "politics", "political",
    ),
    "Xã hội": (
        "xã hội", "society",
    ),
    "Thế giới": (
        "thế giới", "world", "international", "quốc tế",
    ),
    "Văn hóa": (
        "văn hóa", "culture",
    ),
}

# Những giá trị category có thể đang được lưu trong database.
# Bao gồm cả nhãn tiếng Anh của bộ phân loại bài báo và nhãn tiếng Việt.
CATEGORY_VALUE_ALIASES = {
    "Thời sự": ("news", "category news", "thoi su"),
    "Kinh tế": ("economy", "category economy", "kinh te"),
    "Công nghệ": ("tech", "technology", "category tech", "cong nghe",
                  "khoa hoc cong nghe"),
    "Giải trí": ("entertainment", "category entertainment", "giai tri"),
    "Thể thao": ("sports", "category sports", "the thao"),
    "Giáo dục": ("education", "category education", "giao duc"),
    "Y tế": ("health", "medical", "category health", "y te"),
    "Môi trường": ("environment", "category environment", "moi truong"),
    "Du lịch": ("travel", "tourism", "category travel", "du lich"),
    "Pháp luật": ("law", "legal", "category law", "phap luat"),
    "Chính trị": ("politics", "political", "category politics", "chinh tri"),
    "Xã hội": ("society", "category society", "xa hoi"),
    "Thế giới": ("world", "international", "category world", "the gioi"),
    "Văn hóa": ("culture", "category culture", "van hoa"),
}


def _normalize_label(value: str) -> str:
    """Chuẩn hóa chữ hoa/thường, dấu tiếng Việt và dấu phân cách."""
    value = str(value or "").casefold().replace("đ", "d")
    value = unicodedata.normalize("NFD", value)
    value = "".join(
        char for char in value
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _extract_topic(question: str) -> str | None:
    """Nhận diện một chủ đề đã được hỗ trợ trong câu người dùng."""
    normalized_question = f" {_normalize_label(question)} "
    matched_topics = set()

    for topic, aliases in TOPIC_QUERY_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_label(alias)
            if normalized_alias and f" {normalized_alias} " in normalized_question:
                matched_topics.add(topic)

    # Không tự ý chọn một chủ đề khi câu hỏi chứa nhiều chủ đề khác nhau.
    if len(matched_topics) != 1:
        return None

    return next(iter(matched_topics))


def _has_topic_search_cue(question: str) -> bool:
    """Nhận biết cách diễn đạt yêu cầu lọc theo chuyên mục."""
    normalized_question = f" {_normalize_label(question)} "
    cues = (
        "chuyen muc",
        "danh muc",
        "theo linh vuc",
        "thuoc linh vuc",
        "theo chu de",
        "thuoc chu de",
        "theo chuyen muc",
        "theo danh muc",
        "trong muc",
        "thuoc muc",
        "o muc",
    )
    return any(f" {cue} " in normalized_question for cue in cues)


LATEST_CUES = (
    "moi nhat",
    "gan day",
    "moi day",
    "hom nay",
    "vua dang",
    "vua cap nhat",
    "moi cap nhat",
    "cap nhat moi nhat",
    "cap nhat gan day",
    "tin moi",
    "bai moi",
)

# Các từ chỉ hành động tìm kiếm, cấu trúc câu và thời gian; không phải chủ đề.
LATEST_QUERY_STOPWORDS = {
    "tim", "kiem", "tra", "cuu", "hay", "giup", "cho", "toi",
    "minh", "ban", "xin", "vui", "long", "xem", "hien",
    "liet", "ke", "dua", "toi", "bai", "bao", "viet", "tin",
    "tuc", "ve", "lien", "quan", "den", "moi", "nhat", "gan",
    "day", "hom", "nay", "vua", "dang", "cap", "nhat", "cac",
    "nhung", "mot", "so", "tat", "ca", "trong", "theo", "noi",
    "dung", "chu", "de", "article", "articles", "news", "latest",
    "newest", "recent", "recently", "today", "about", "find", "search",
    "show", "me", "the", "for", "of",
}


def _has_latest_cue(question: str) -> bool:
    """Nhận biết yêu cầu ưu tiên bài báo mới nhất/gần đây."""
    normalized_question = f" {_normalize_label(question)} "
    return any(f" {cue} " in normalized_question for cue in LATEST_CUES)


def _extract_latest_topic_terms(question: str) -> list[str]:
    """Tách từ khóa nội dung, loại bỏ phần câu lệnh và cụm chỉ thời gian."""
    cleaned_question = re.sub(
        r"\bhiển\s+thị\b", " ", question or "", flags=re.IGNORECASE
    )
    raw_tokens = re.findall(
        r"[a-zA-ZÀ-ỹĐđ0-9]+", cleaned_question, flags=re.UNICODE
    )
    terms = []
    seen = set()

    for token in raw_tokens:
        normalized = _normalize_label(token)
        if not normalized or normalized in LATEST_QUERY_STOPWORDS:
            continue
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(token.casefold().strip())

    return terms


def _search_latest_topic_articles(
    terms: list[str],
    limit: int = 10,
):
    """Tìm bài chứa tất cả từ khóa chủ đề và sắp xếp theo ngày đăng giảm dần.

    Mỗi từ khóa có thể xuất hiện trong tiêu đề, sapo hoặc toàn văn.
    Chỉ các bài có published_at mới được đưa vào danh sách mới nhất.
    Trả về (articles, total_match_count, dated_match_count).
    """
    if not terms:
        return [], 0, 0

    db = SessionLocal()
    try:
        query = db.query(Article)

        # AND giữa các từ khóa; OR giữa tiêu đề, sapo và toàn văn.
        # Như vậy "giá vàng" cần có cả hai từ trong cùng bài báo.
        for term in terms:
            pattern = f"%{term}%"
            query = query.filter(
                or_(
                    Article.title.ilike(pattern),
                    Article.sapo.ilike(pattern),
                    Article.body.ilike(pattern),
                )
            )

        total_count = query.count()
        dated_query = query.filter(Article.published_at.isnot(None))
        dated_count = dated_query.count()
        rows = (
            dated_query
            .order_by(Article.published_at.desc())
            .limit(limit)
            .all()
        )

        articles = []
        for article in rows:
            published_at = article.published_at
            if published_at is not None and hasattr(published_at, "strftime"):
                published_display = published_at.strftime("%d/%m/%Y %H:%M")
                published_iso = published_at.isoformat(sep=" ")
            else:
                published_display = str(published_at or "")
                published_iso = published_display

            articles.append({
                "article_id": getattr(article, "id", None),
                "title": getattr(article, "title", None) or "Không có tiêu đề",
                "source": getattr(article, "source", None) or "Không rõ nguồn",
                "url": getattr(article, "url", None) or "",
                "category": getattr(article, "category", None) or "",
                "published_at": published_iso,
                "published_at_display": published_display,
                "similarity": None,
            })

        return articles, total_count, dated_count
    finally:
        db.close()


def _format_latest_topic_search(
    terms: list[str],
    articles: list[dict[str, Any]],
    total_count: int,
    dated_count: int,
) -> str:
    """Hiển thị bài theo chủ đề, đã sắp xếp mới nhất trước."""
    topic_text = " ".join(terms)

    if total_count == 0:
        return (
            f"Tôi chưa tìm thấy bài báo chứa các từ khóa **{topic_text}**. "
            "Bạn hãy thử rút gọn hoặc diễn đạt lại chủ đề."
        )

    if dated_count == 0:
        return (
            f"Tôi tìm thấy {total_count} bài báo có nội dung khớp **{topic_text}**, "
            "nhưng các bài này chưa có ngày đăng trong database nên chưa thể "
            "xác định bài nào mới nhất."
        )

    lines = [
        f"## Bài báo mới nhất về: {topic_text}",
        f"Tìm thấy {total_count} bài khớp từ khóa; có ngày đăng ở {dated_count} bài.",
        f"Hiển thị {len(articles)} bài mới nhất:",
    ]

    for index, article in enumerate(articles, start=1):
        title = article.get("title") or "Không có tiêu đề"
        newspaper = article.get("source") or "Không rõ nguồn"
        published = article.get("published_at_display") or article.get("published_at") or "Không rõ ngày đăng"
        url = article.get("url") or ""
        lines.append(f"{index}. **{title}** — {newspaper}")
        lines.append(f"   Ngày đăng: {published}")
        if url.startswith(("https://", "http://")):
            lines.append(f"   {url}")

    missing_date_count = total_count - dated_count
    if missing_date_count > 0:
        lines.append(
            f"\n**Lưu ý:** {missing_date_count} bài khớp từ khóa chưa có ngày đăng, "
            "nên không thể xếp hạng mới nhất."
        )
    if dated_count > len(articles):
        lines.append(
            f"\nĐang hiển thị {len(articles)}/{dated_count} bài có ngày đăng."
        )

    return "\n\n".join(lines)



def _search_topic_articles(topic: str, limit: int = 10):
    """Lọc bài báo theo category trong SQLite, không gọi FAISS hoặc Qwen."""
    db = SessionLocal()
    try:
        category_rows = (
            db.query(Article.category)
            .filter(Article.category.isnot(None))
            .distinct()
            .all()
        )
        available_categories = [
            row[0] for row in category_rows if row and row[0]
        ]

        accepted_values = {
            _normalize_label(value)
            for value in CATEGORY_VALUE_ALIASES.get(topic, ())
        }
        matched_categories = [
            value for value in available_categories
            if _normalize_label(value) in accepted_values
        ]

        if not matched_categories:
            return [], 0, False, available_categories

        query = db.query(Article).filter(
            Article.category.in_(matched_categories)
        )
        total_count = query.count()

        rows = (
            query
            .order_by(Article.published_at.desc())
            .limit(limit)
            .all()
        )

        articles = []
        for article in rows:
            published_at = getattr(article, "published_at", None)
            if published_at is not None and hasattr(published_at, "strftime"):
                published_text = published_at.strftime("%d/%m/%Y %H:%M")
            else:
                published_text = str(published_at or "")

            articles.append({
                "article_id": getattr(article, "id", None),
                "title": getattr(article, "title", None) or "Không có tiêu đề",
                "source": getattr(article, "source", None) or "Không rõ nguồn",
                "url": getattr(article, "url", None) or "",
                "category": getattr(article, "category", None) or "",
                "published_at": published_text,
                "similarity": None,
            })

        return articles, total_count, True, available_categories
    finally:
        db.close()


def _format_topic_search(
    topic: str,
    articles: list[dict[str, Any]],
    total_count: int,
    category_found: bool,
    available_categories: list[str],
) -> str:
    """Trình bày kết quả lọc theo chuyên mục và nêu rõ khi thiếu dữ liệu."""
    if not category_found:
        if available_categories:
            available_text = ", ".join(
                str(value) for value in sorted(set(available_categories))
            )
            return (
                f"Tôi chưa tìm thấy chuyên mục **{topic}** trong dữ liệu hiện tại. "
                f"Các giá trị category đang có trong database: {available_text}."
            )
        return "Database hiện chưa có giá trị category để tìm bài báo theo chủ đề."

    if not articles:
        return f"Hiện chưa có bài báo thuộc chuyên mục **{topic}**."

    lines = [
        f"## Bài báo thuộc chuyên mục: {topic}",
        f"Tìm thấy {total_count} bài báo. Hiển thị {len(articles)} bài:",
    ]

    for index, article in enumerate(articles, start=1):
        title = article.get("title") or "Không có tiêu đề"
        newspaper = article.get("source") or "Không rõ nguồn"
        published = article.get("published_at") or ""
        url = article.get("url") or ""

        lines.append(f"{index}. **{title}** — {newspaper}")
        if published:
            lines.append(f"   Ngày đăng: {published}")
        if url.startswith(("https://", "http://")):
            lines.append(f"   {url}")

    if total_count > len(articles):
        lines.append(
            f"\nĐang hiển thị {len(articles)}/{total_count} bài báo phù hợp."
        )

    return "\n\n".join(lines)



@lru_cache(maxsize=1)
def _load_summarizer():
    """Nap ViT5 khi can, tranh nap mo hinh neu khong tom tat."""
    from processing.summerizer import summarize
    return summarize


def get_article_content(
    article_id: str | None = None,
    url: str | None = None,
) -> dict[str, str] | None:
    """Lay tieu de, sapo va toan van tu SQLite; uu tien ID, sau do URL."""
    db = SessionLocal()
    try:
        article = None

        if article_id:
            article = (
                db.query(Article)
                .filter(Article.id == str(article_id))
                .first()
            )

        if article is None and url:
            article = (
                db.query(Article)
                .filter(Article.url == url)
                .first()
            )

        if article is None:
            return None

        return {
            "title": article.title or "",
            "sapo": article.sapo or "",
            "body": article.body or "",
        }
    finally:
        db.close()


def _similarity(source: dict[str, Any]) -> float:
    try:
        return float(source.get("similarity") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rank_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(sources, key=_similarity, reverse=True)


def _search_sources(question: str) -> list[dict[str, Any]]:
    """Truy xuat bang PhoBERT + FAISS, gom chunk thanh bai bao."""
    chunks = retrieve(
        question,
        candidate_k=50,
        top_k=10,
        max_chunks_per_article=2,
    )
    articles = group_chunks_by_article(chunks)

    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for article in articles:
        source = {
            "article_id": article.get("article_id"),
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "url": article.get("url", ""),
            "similarity": article.get("similarity", 0.0),
        }
        # Gop chunk co cung URL/ID thanh mot nguon duy nhat.
        identity = (
            str(source.get("article_id") or ""),
            str(source.get("url") or ""),
        )
        if identity == ("", ""):
            identity = (str(source.get("title") or ""), "")
        if identity not in seen:
            seen.add(identity)
            sources.append(source)

    return _rank_sources(sources)


def _find_source_with_body(sources: list[dict[str, Any]]):
    """Tim nguon co toan van trong SQLite, uu tien diem tuong dong cao."""
    for source in _rank_sources(sources):
        content = get_article_content(
            article_id=source.get("article_id"),
            url=source.get("url"),
        )
        if content and content.get("body", "").strip():
            return source, content
    return None, None


def _format_search_results(sources: list[dict[str, Any]]) -> str:
    """Tao danh sach ket qua tim kiem, khong goi LLM."""
    ranked = _rank_sources(sources)
    if not ranked:
        return "Tôi chưa tìm thấy bài báo phù hợp. Bạn hãy thử diễn đạt lại chủ đề tìm kiếm."

    lines = [f"Tôi tìm thấy {len(ranked)} bài báo liên quan:"]
    for i, source in enumerate(ranked, start=1):
        title = source.get("title") or "Không có tiêu đề"
        newspaper = source.get("source") or "Không rõ nguồn"
        url = source.get("url") or ""
        lines.append(f"{i}. **{title}** — {newspaper}")
        if url.startswith(("https://", "http://")):
            lines.append(f"   {url}")
    return "\n\n".join(lines)


def _empty_retrieval_message(intent: str) -> str:
    if intent == "summarization":
        return (
            "Tôi chưa tìm được bài báo phù hợp để tóm tắt. "
            "Bạn hãy thử nêu rõ tiêu đề hoặc chủ đề của bài báo."
        )
    if intent == "article_detail":
        return (
            "Tôi chưa tìm được toàn văn bài báo phù hợp. "
            "Bạn hãy nêu rõ hơn tiêu đề hoặc chủ đề bài báo."
        )
    return "Tôi chưa tìm thấy bài báo phù hợp. Bạn hãy thử diễn đạt lại yêu cầu."



def _get_latest_news(limit: int = 10):
    """Lay cac bai co ngay dang, moi nhat theo published_at desc.

    Tra ve (articles, total_count, dated_count). Chi dung ngay dang da luu;
    khong tu suy doan ngay cho cac bai published_at dang NULL.
    """
    db = SessionLocal()
    try:
        total_count = db.query(Article).count()
        dated_query = db.query(Article).filter(Article.published_at.isnot(None))
        dated_count = dated_query.count()
        rows = (
            dated_query
            .order_by(Article.published_at.desc())
            .limit(limit)
            .all()
        )

        articles: list[dict[str, Any]] = []
        for article in rows:
            published_at = article.published_at
            if published_at is not None and hasattr(published_at, "strftime"):
                published_text = published_at.strftime("%d/%m/%Y %H:%M")
                published_iso = published_at.isoformat(sep=" ")
            else:
                published_text = str(published_at or "")
                published_iso = published_text

            articles.append({
                "article_id": getattr(article, "id", None),
                "title": getattr(article, "title", None) or "Không có tiêu đề",
                "source": getattr(article, "source", None) or "Không rõ nguồn",
                "url": getattr(article, "url", None) or "",
                "published_at": published_iso,
                "published_at_display": published_text,
                "similarity": None,
            })

        return articles, total_count, dated_count
    finally:
        db.close()


def _format_latest_news(articles, total_count: int, dated_count: int) -> str:
    """Dinh dang danh sach tin moi nhat va neu ro do phu du lieu ngay dang."""
    if not articles:
        if dated_count == 0:
            return (
                "Hiện chưa có bài báo nào có ngày đăng trong cơ sở dữ liệu. "
                "Tôi chưa thể tạo danh sách tin mới nhất."
            )
        return "Tôi chưa tìm được bài báo có ngày đăng để hiển thị."

    lines = [f"## {len(articles)} bài báo mới nhất theo ngày đăng đã lưu"]
    for i, article in enumerate(articles, start=1):
        title = article.get("title") or "Không có tiêu đề"
        newspaper = article.get("source") or "Không rõ nguồn"
        published = article.get("published_at_display") or article.get("published_at") or "Không rõ ngày đăng"
        url = article.get("url") or ""
        lines.append(f"{i}. **{title}** — {newspaper}")
        lines.append(f"   Ngày đăng theo dữ liệu lưu: {published}")
        if url.startswith(("https://", "http://")):
            lines.append(f"   {url}")

    if dated_count < total_count:
        lines.append(
            f"\n**Lưu ý:** hiện mới có ngày đăng cho {dated_count:,}/{total_count:,} bài báo. "
            "Danh sách trên là các bài mới nhất trong số bài đã có ngày đăng, "
            "chưa thể bảo đảm là mới nhất trong toàn bộ dữ liệu."
        )
    return "\n\n".join(lines)

def handle_intent(question: str) -> tuple[str, list[dict[str, Any]]]:
    """Du doan y dinh va dieu phoi toi chuc nang phu hop."""
    if not question or not question.strip():
        return "Bạn hãy nhập yêu cầu cần xử lý.", []

    intent = predict_intent(question)

    # Yêu cầu ghi rõ chuyên mục được ưu tiên trước; nhánh topic_search
    # cũng sắp xếp theo published_at nên có thể xử lý cả "chuyên mục X mới nhất".
    detected_topic = _extract_topic(question)
    if detected_topic and _has_topic_search_cue(question):
        intent = "topic_search"
    # Nếu có dấu hiệu thời gian và vẫn còn từ khóa nội dung, đây là tìm
    # bài mới nhất theo một chủ đề cụ thể, không phải tin mới nhất nói chung.
    elif _has_latest_cue(question):
        intent = (
            "latest_topic_search"
            if _extract_latest_topic_terms(question)
            else "latest_news"
        )
    elif intent == "latest_topic_search":
        # Không có mốc thời gian thì đây là semantic_search thông thường.
        intent = "semantic_search"

    print(f"[INTENT] {intent} | {question}")

    if intent == "greeting":
        return (
            "Xin chào! Tôi là Trợ lý đọc báo thông minh. "
            "Tôi có thể giúp bạn tìm kiếm, tóm tắt và hỏi đáp "
            "về các bài báo tiếng Việt."
        ), []

    if intent == "introduction":
        return INTRODUCTION_RESPONSE, []

    if intent == "latest_news":
        articles, total_count, dated_count = _get_latest_news(limit=10)
        answer = _format_latest_news(articles, total_count, dated_count)
        return answer, articles

    if intent == "latest_topic_search":
        terms = _extract_latest_topic_terms(question)
        if not terms:
            # An toàn: không có chủ đề cụ thể thì trở về danh sách tin mới nhất.
            articles, total_count, dated_count = _get_latest_news(limit=10)
            return _format_latest_news(articles, total_count, dated_count), articles

        articles, total_count, dated_count = _search_latest_topic_articles(
            terms,
            limit=10,
        )
        answer = _format_latest_topic_search(
            terms,
            articles,
            total_count,
            dated_count,
        )
        return answer, articles

    # Tim bai theo chuyen muc: loc truc tiep category trong SQLite.
    # Nhanh nay khong goi embedding, FAISS hoac Qwen.
    if intent == "topic_search":
        topic = _extract_topic(question)
        if topic is None:
            return (
                "Bạn hãy nêu rõ một chuyên mục, ví dụ: kinh tế, công nghệ, "
                "thể thao, giải trí, giáo dục hoặc y tế."
            ), []

        articles, total_count, category_found, available_categories = (
            _search_topic_articles(topic, limit=10)
        )
        answer = _format_topic_search(
            topic,
            articles,
            total_count,
            category_found,
            available_categories,
        )
        return answer, articles

    # Truy xuat truc tiep cho tim kiem, tom tat va xem chi tiet.
    # Cac nhanh nay KHONG goi ask()/Qwen.
    if intent in {"semantic_search", "summarization", "article_detail"}:
        sources = _search_sources(question)

        if intent == "semantic_search":
            return _format_search_results(sources), sources

        source, content = _find_source_with_body(sources)
        if content is None:
            return _empty_retrieval_message(intent), sources

        title = content.get("title") or source.get("title") or "Bài báo"
        selected_source = dict(source)
        selected_source["title"] = title

        if intent == "article_detail":
            body = content.get("body", "").strip()
            if not body:
                return _empty_retrieval_message(intent), []
            return f"## {title}\n\n{body}", [selected_source]

        # summarization: chi nap ViT5 neu da tim thay bai co body.
        summary = _load_summarizer()(content["body"])
        if not summary or not summary.strip():
            return "Mô hình ViT5 chưa tạo được bản tóm tắt.", [selected_source]
        return f"## Tóm tắt: {title}\n\n{summary.strip()}", [selected_source]

    # Chi hoi dap can goi pipeline RAG + Qwen.
    if intent == "question_answering":
        result = ask(question)
        return (
            result.get("answer", "Chưa tạo được câu trả lời."),
            result.get("sources", []) or [],
        )

    # Nhan khong duoc biet: fallback an toan sang RAG hien co.
    result = ask(question)
    return (
        result.get("answer", "Chưa tạo được câu trả lời."),
        result.get("sources", []) or [],
    )
