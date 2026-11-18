"""Bộ điều phối ý định cho trợ lý đọc báo.

- greeting/introduction: trả lời trực tiếp.
- topic_search: lọc bài báo theo chuyên mục (category) trong SQLite.
- latest_topic_search: tìm theo từ khóa chủ đề trong SQLite, sắp xếp published_at mới nhất.
- latest_news: truy vấn SQLite theo published_at; nếu còn bài thiếu ngày đăng, cảnh báo độ phủ dữ liệu.
- semantic_search: Hybrid SQLite FTS5 + PhoBERT/FAISS, không gọi Qwen.
- summarization: Hybrid retrieval (hoặc bài đang chọn) -> SQLite -> ViT5, không gọi Qwen.
- article_detail: Hybrid retrieval (hoặc bài đang chọn) -> SQLite, không gọi Qwen.
- question_answering: Hybrid retrieval (hoặc bài đang ghim) -> Qwen RAG.

Các hàm điều phối chính trả về (answer, sources) cho giao diện Streamlit.
"""

from __future__ import annotations

from functools import lru_cache
import re
import unicodedata
from typing import Any

from database.db import Article, SessionLocal
from database.repository import (
    get_article_content,
    get_latest_articles,
    search_latest_topic_articles,
    search_topic_articles,
)
from nlp.intent_classifier import predict_intent


# Nạp pipeline nặng theo nhu cầu, tránh import PyTorch ngay khi mở Streamlit.
def ask(*args, **kwargs):
    from rag.pipeline import ask as pipeline_ask

    return pipeline_ask(*args, **kwargs)


def retrieve(*args, **kwargs):
    from rag.pipeline import retrieve as pipeline_retrieve

    return pipeline_retrieve(*args, **kwargs)


def group_chunks_by_article(*args, **kwargs):
    from rag.pipeline import group_chunks_by_article as pipeline_group

    return pipeline_group(*args, **kwargs)


def generate_answer(*args, **kwargs):
    from rag.pipeline import generate_answer as pipeline_generate_answer

    return pipeline_generate_answer(*args, **kwargs)


INTRODUCTION_RESPONSE = """Tôi là Trợ lý đọc báo thông minh, hỗ trợ bạn khai thác
thông tin từ các bài báo tiếng Việt.

Các chức năng chính:
- Tìm bài báo theo chuyên mục và nguồn báo (VnExpress, Tuổi Trẻ, Tiền Phong).
- Tìm kiếm kết hợp từ khóa chính xác (FTS5) và ngữ nghĩa (PhoBERT + FAISS).
- Tìm bài báo mới nhất theo chủ đề hoặc chuyên mục.
- Tóm tắt bài báo bằng mô hình ViT5.
- Hỏi đáp dựa trên bài báo bằng RAG và Qwen, hỗ trợ hỏi nối tiếp theo bài đang chọn.
"""


TOPIC_QUERY_ALIASES = {
    "Thời sự": ("thời sự", "news"),
    "Kinh tế": ("kinh tế", "economy", "economic"),
    "Công nghệ": ("công nghệ", "tech", "technology", "khoa học công nghệ", "khoa học và công nghệ"),
    "Giải trí": ("giải trí", "entertainment", "điện ảnh", "âm nhạc", "showbiz"),
    "Thể thao": ("thể thao", "sports"),
    "Giáo dục": ("giáo dục", "education"),
    "Y tế": ("y tế", "sức khỏe", "health", "medical"),
    "Môi trường": ("môi trường", "environment"),
    "Du lịch": ("du lịch", "travel", "tourism"),
    "Pháp luật": ("pháp luật", "law", "legal"),
    "Chính trị": ("chính trị", "politics", "political"),
    "Xã hội": ("xã hội", "society"),
    "Thế giới": ("thế giới", "world", "international", "quốc tế"),
    "Văn hóa": ("văn hóa", "culture"),
}

CATEGORY_VALUE_ALIASES = {
    "Thời sự": ("news", "category news", "thoi su"),
    "Kinh tế": ("economy", "category economy", "kinh te"),
    "Công nghệ": ("tech", "technology", "category tech", "cong nghe", "khoa hoc cong nghe"),
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

SOURCE_ALIASES = {
    "vnexpress": ("vnexpress", "vn express"),
    "tuoitre": ("tuoi tre", "tuoi tre online"),
    "tienphong": ("tien phong", "bao tien phong"),
}

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

LATEST_QUERY_STOPWORDS = {
    "tim", "kiem", "tra", "cuu", "hay", "giup", "cho", "toi",
    "minh", "ban", "xin", "vui", "long", "xem", "hien",
    "liet", "ke", "dua", "bai", "bao", "viet", "tin",
    "tuc", "ve", "lien", "quan", "den", "moi", "nhat", "gan",
    "day", "hom", "nay", "vua", "dang", "cap", "nhat", "cac",
    "nhung", "mot", "so", "tat", "ca", "trong", "theo", "noi",
    "dung", "chu", "de", "nhac", "toi", "de", "cap", "tren",
    "cua", "tu", "vnexpress", "tuoi", "tre", "tien", "phong",
    "article", "articles", "news", "latest", "newest", "recent",
    "recently", "today", "about", "find", "one", "mentioning",
    "search", "show", "me", "the", "for", "of",
}


def _normalize_label(value: str) -> str:
    """Chuẩn hóa chữ hoa/thường, dấu tiếng Việt và dấu phân cách."""
    value = str(value or "").casefold().replace("đ", "d")
    value = unicodedata.normalize("NFD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _extract_source(question: str) -> str | None:
    """Nhận diện nguồn báo (vnexpress, tuoitre, tienphong) được nhắc trong câu hỏi."""
    normalized = f" {_normalize_label(question)} "
    matched = []
    for source_slug, aliases in SOURCE_ALIASES.items():
        if any(f" {alias} " in normalized for alias in aliases):
            matched.append(source_slug)
    return matched[0] if len(matched) == 1 else None


def _strip_source_mentions(question: str) -> str:
    """Loại bỏ tên báo khỏi chuỗi truy vấn sau khi đã trích xuất bộ lọc source."""
    cleaned = re.sub(
        r"\b(?:trên|từ|của)?\s*(?:báo\s+)?(?:vn\s*express|tuổi\s+trẻ(?:\s+online)?|tuoi\s+tre|tiền\s+phong|tien\s+phong)\b",
        " ",
        question or "",
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split())


def _extract_topic(question: str) -> str | None:
    """Nhận diện một chủ đề đã được hỗ trợ trong câu người dùng."""
    normalized_question = f" {_normalize_label(question)} "
    matched_topics = set()

    for topic, aliases in TOPIC_QUERY_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_label(alias)
            if normalized_alias and f" {normalized_alias} " in normalized_question:
                matched_topics.add(topic)

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


def _has_search_cue(question: str) -> bool:
    """Nhận biết yêu cầu tìm kiếm bài báo theo thực thể hoặc từ khóa."""
    normalized_question = f" {_normalize_label(question)} "
    cues = (
        "tim bai",
        "tim mot bai",
        "tim cac bai",
        "tim tin",
        "tim kiem",
        "tra cuu",
        "nhac den",
        "nhac toi",
        "de cap",
        "co bai nao",
        "bai nao noi ve",
        "bai nao nhac",
        "find one",
        "mentioning",
        "find articles",
        "search for",
    )
    return any(f" {cue} " in normalized_question for cue in cues)


def _has_latest_cue(question: str) -> bool:
    """Nhận biết yêu cầu ưu tiên bài báo mới nhất/gần đây."""
    normalized_question = f" {_normalize_label(question)} "
    return any(f" {cue} " in normalized_question for cue in LATEST_CUES)


def _extract_latest_topic_terms(question: str) -> list[str]:
    """Tách từ khóa nội dung, loại bỏ phần câu lệnh và cụm chỉ thời gian."""
    cleaned_question = re.sub(r"\bhiển\s+thị\b", " ", question or "", flags=re.IGNORECASE)
    cleaned_question = _strip_source_mentions(cleaned_question)
    raw_tokens = re.findall(r"[a-zA-ZÀ-ỹĐđ0-9]+", cleaned_question, flags=re.UNICODE)
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


def _resolve_reference_article(
    question: str,
    context_sources: list[dict[str, Any]] | None,
    pinned_article: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, bool]:
    """Resolve ordinal ('bài số 1') or demonstrative ('bài này') references to a source item."""
    normalized = f" {_normalize_label(question)} "

    ordinal_map = {
        "bai dau": 0,
        "bai dau tien": 0,
        "bai thu nhat": 0,
        "bai thu hai": 1,
        "bai thu ba": 2,
        "bai thu tu": 3,
        "bai thu nam": 4,
    }
    if context_sources:
        for phrase, idx in ordinal_map.items():
            if f" {phrase} " in normalized and idx < len(context_sources):
                return context_sources[idx], True

        match = re.search(r"\bbai(?:\s+(?:bao|viet))?\s+(?:so\s+|thu\s+)?(\d{1,2})\b", normalized)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(context_sources):
                return context_sources[idx], True

        if " bai cuoi " in normalized or " bai cuoi cung " in normalized:
            return context_sources[-1], True

    if any(
        phrase in normalized
        for phrase in (" bai nay ", " bai do ", " bai vua roi ", " bai bao nay ", " bai bao do ")
    ):
        target = pinned_article or (context_sources[0] if context_sources else None)
        if target is not None:
            return target, True

    return None, False


def _get_latest_news(
    limit: int = 10,
    source: str | None = None,
    category: str | None = None,
):
    return get_latest_articles(limit=limit, source=source, category=category)


def _search_latest_topic_articles(
    terms: list[str],
    limit: int = 10,
    source: str | None = None,
    category: str | None = None,
):
    kwargs: dict[str, Any] = {
        "limit": limit,
        "session_factory": SessionLocal,
        "article_model": Article,
    }
    if source is not None:
        kwargs["source"] = source
    if category is not None:
        kwargs["category"] = category
    return search_latest_topic_articles(terms, **kwargs)


def _search_topic_articles(
    topic: str,
    limit: int = 10,
    source: str | None = None,
):
    accepted_values = {
        _normalize_label(value) for value in CATEGORY_VALUE_ALIASES.get(topic, ())
    }
    kwargs: dict[str, Any] = {
        "accepted_values": accepted_values,
        "normalize_fn": _normalize_label,
        "limit": limit,
        "session_factory": SessionLocal,
        "article_model": Article,
    }
    if source is not None:
        kwargs["source"] = source
    return search_topic_articles(**kwargs)


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
        published = (
            article.get("published_at_display")
            or article.get("published_at")
            or "Không rõ ngày đăng"
        )
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
        lines.append(f"\nĐang hiển thị {len(articles)}/{dated_count} bài có ngày đăng.")

    return "\n\n".join(lines)


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
            available_text = ", ".join(str(value) for value in sorted(set(available_categories)))
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
        published = article.get("published_at_display") or article.get("published_at") or ""
        url = article.get("url") or ""

        lines.append(f"{index}. **{title}** — {newspaper}")
        if published:
            lines.append(f"   Ngày đăng: {published}")
        if url.startswith(("https://", "http://")):
            lines.append(f"   {url}")

    if total_count > len(articles):
        lines.append(f"\nĐang hiển thị {len(articles)}/{total_count} bài báo phù hợp.")

    return "\n\n".join(lines)


@lru_cache(maxsize=1)
def _load_summarizer():
    """Nạp ViT5 khi cần, tránh nạp mô hình nếu không tóm tắt."""
    from processing.summarizer import summarize

    return summarize


def _similarity(source: dict[str, Any]) -> float:
    try:
        return float(source.get("similarity") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rank_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(sources, key=_similarity, reverse=True)


def _search_sources(
    question: str,
    source: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Truy xuất kết hợp FTS5 + PhoBERT/FAISS, gom chunk thành bài báo."""
    cleaned_question = _strip_source_mentions(question) if source else question
    kwargs: dict[str, Any] = {
        "candidate_k": 50,
        "top_k": 10,
        "max_chunks_per_article": 2,
    }
    if source is not None:
        kwargs["source"] = source
    if category is not None:
        kwargs["category"] = category

    chunks = retrieve(cleaned_question or question, **kwargs)
    articles = group_chunks_by_article(chunks)

    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for article in articles:
        item = {
            "article_id": article.get("article_id"),
            "title": article.get("title", ""),
            "sapo": article.get("sapo", ""),
            "source": article.get("source", ""),
            "category": article.get("category", ""),
            "url": article.get("url", ""),
            "published_at": article.get("published_at", ""),
            "published_at_display": article.get("published_at_display", ""),
            "similarity": article.get("similarity", 0.0),
        }
        identity = (
            str(item.get("article_id") or ""),
            str(item.get("url") or ""),
        )
        if identity == ("", ""):
            identity = (str(item.get("title") or ""), "")
        if identity not in seen:
            seen.add(identity)
            sources.append(item)

    return _rank_sources(sources)


def _find_source_with_body(sources: list[dict[str, Any]]):
    """Tìm nguồn có toàn văn trong SQLite, ưu tiên điểm tương đồng cao."""
    for source in _rank_sources(sources):
        content = get_article_content(
            article_id=source.get("article_id"),
            url=source.get("url"),
        )
        if content and content.get("body", "").strip():
            return source, content
    return None, None


def _format_search_results(sources: list[dict[str, Any]]) -> str:
    """Tạo danh sách kết quả tìm kiếm, không gọi LLM."""
    ranked = _rank_sources(sources)
    if not ranked:
        return (
            "Tôi chưa tìm thấy bài báo nào phù hợp trong dữ liệu hiện tại. "
            "Bạn hãy thử diễn đạt lại chủ đề hoặc kiểm tra từ khóa tìm kiếm."
        )

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


def _format_latest_news(articles: list[dict[str, Any]], total_count: int, dated_count: int) -> str:
    """Định dạng danh sách tin mới nhất và nêu rõ độ phủ dữ liệu ngày đăng."""
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


def _handle_target_article(
    question: str,
    target: dict[str, Any],
    intent: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Execute summarization, detail reading, or QA directly against one resolved article."""
    content = get_article_content(
        article_id=target.get("article_id"),
        url=target.get("url"),
    )
    if not content or not content.get("body", "").strip():
        return "Không tìm thấy toàn văn của bài báo được chọn trong database.", [target]

    selected = dict(target)
    selected["title"] = content.get("title") or target.get("title") or "Bài báo"
    selected.setdefault("sapo", content.get("sapo", ""))
    selected.setdefault("source", content.get("source", ""))
    selected.setdefault("category", content.get("category", ""))
    selected.setdefault("published_at_display", content.get("published_at_display", ""))

    normalized = f" {_normalize_label(question)} "
    wants_summary = intent == "summarization" or any(
        cue in normalized for cue in (" tom tat ", " y chinh ", " rut gon ", " tom luoc ")
    )
    wants_detail = intent == "article_detail" or any(
        cue in normalized
        for cue in (" toan van ", " chi tiet ", " doc bai ", " noi dung day du ", " xem bai ")
    )

    if wants_summary:
        summary = _load_summarizer()(content["body"])
        if not summary or not summary.strip():
            return "Mô hình ViT5 chưa tạo được bản tóm tắt.", [selected]
        return f"## Tóm tắt: {selected['title']}\n\n{summary.strip()}", [selected]

    if wants_detail:
        return f"## {selected['title']}\n\n{content['body'].strip()}", [selected]

    article_ctx = {
        "article_id": selected.get("article_id"),
        "title": selected["title"],
        "source": selected.get("source", ""),
        "url": selected.get("url", ""),
        "texts": [content["body"].strip()],
    }
    answer = generate_answer(question, [article_ctx])
    return answer, [selected]


def handle_intent(
    question: str,
    context_sources: list[dict[str, Any]] | None = None,
    pinned_article: dict[str, Any] | None = None,
    source_filter: str | None = None,
    category_filter: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Dự đoán ý định và điều phối tới chức năng phù hợp."""
    if not question or not question.strip():
        return "Bạn hãy nhập yêu cầu cần xử lý.", []

    effective_source = _extract_source(question) or source_filter
    intent = predict_intent(question)

    detected_topic = _extract_topic(question)
    if detected_topic and _has_topic_search_cue(question):
        intent = "topic_search"
    elif _has_latest_cue(question):
        intent = "latest_topic_search" if _extract_latest_topic_terms(question) else "latest_news"
    elif intent == "latest_topic_search":
        intent = "semantic_search"
    elif intent not in {"greeting", "introduction", "summarization", "article_detail"} and _has_search_cue(question):
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

    ref_article, explicit_ref = _resolve_reference_article(
        question,
        context_sources=context_sources,
        pinned_article=pinned_article,
    )
    if explicit_ref and ref_article is not None:
        return _handle_target_article(question, ref_article, intent)

    if (
        pinned_article is not None
        and intent not in {"latest_news", "topic_search", "latest_topic_search"}
        and not _has_search_cue(question)
    ):
        return _handle_target_article(question, pinned_article, intent)

    if intent == "latest_news":
        if effective_source is None and category_filter is None:
            articles, total_count, dated_count = _get_latest_news(limit=10)
        else:
            articles, total_count, dated_count = _get_latest_news(
                limit=10,
                source=effective_source,
                category=category_filter,
            )
        answer = _format_latest_news(articles, total_count, dated_count)
        return answer, articles

    if intent == "latest_topic_search":
        terms = _extract_latest_topic_terms(question)
        if not terms:
            if effective_source is None and category_filter is None:
                articles, total_count, dated_count = _get_latest_news(limit=10)
            else:
                articles, total_count, dated_count = _get_latest_news(
                    limit=10,
                    source=effective_source,
                    category=category_filter,
                )
            return _format_latest_news(articles, total_count, dated_count), articles

        if effective_source is None and category_filter is None:
            articles, total_count, dated_count = _search_latest_topic_articles(terms, limit=10)
        else:
            articles, total_count, dated_count = _search_latest_topic_articles(
                terms,
                limit=10,
                source=effective_source,
                category=category_filter,
            )
        answer = _format_latest_topic_search(terms, articles, total_count, dated_count)
        return answer, articles

    if intent == "topic_search":
        topic = _extract_topic(question)
        if topic is None:
            return (
                "Bạn hãy nêu rõ một chuyên mục, ví dụ: kinh tế, công nghệ, "
                "thể thao, giải trí, giáo dục hoặc y tế."
            ), []

        if effective_source is None:
            articles, total_count, category_found, available_categories = _search_topic_articles(
                topic,
                limit=10,
            )
        else:
            articles, total_count, category_found, available_categories = _search_topic_articles(
                topic,
                limit=10,
                source=effective_source,
            )
        answer = _format_topic_search(
            topic,
            articles,
            total_count,
            category_found,
            available_categories,
        )
        return answer, articles

    if intent in {"semantic_search", "summarization", "article_detail"}:
        if effective_source is None and category_filter is None:
            sources = _search_sources(question)
        else:
            search_kwargs: dict[str, Any] = {}
            if effective_source is not None:
                search_kwargs["source"] = effective_source
            if category_filter is not None:
                search_kwargs["category"] = category_filter
            sources = _search_sources(question, **search_kwargs)

        if intent == "semantic_search":
            return _format_search_results(sources), sources

        source_item, content = _find_source_with_body(sources)
        if content is None:
            return _empty_retrieval_message(intent), sources

        title = content.get("title") or source_item.get("title") or "Bài báo"
        selected_source = dict(source_item)
        selected_source["title"] = title

        if intent == "article_detail":
            body = content.get("body", "").strip()
            if not body:
                return _empty_retrieval_message(intent), []
            return f"## {title}\n\n{body}", [selected_source]

        summary = _load_summarizer()(content["body"])
        if not summary or not summary.strip():
            return "Mô hình ViT5 chưa tạo được bản tóm tắt.", [selected_source]
        return f"## Tóm tắt: {title}\n\n{summary.strip()}", [selected_source]

    clean_qa = _strip_source_mentions(question) if effective_source else question
    ask_kwargs: dict[str, Any] = {}
    if effective_source is not None:
        ask_kwargs["source"] = effective_source
    if category_filter is not None:
        ask_kwargs["category"] = category_filter
    result = ask(clean_qa or question, **ask_kwargs)
    return (
        result.get("answer", "Chưa tạo được câu trả lời."),
        result.get("sources", []) or [],
    )
