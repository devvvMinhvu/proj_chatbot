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


def stream_answer(*args, **kwargs):
    from rag.pipeline import stream_answer as pipeline_stream_answer

    return pipeline_stream_answer(*args, **kwargs)


def asks_for_financial_amount(question: str) -> bool:
    from rag.pipeline import asks_for_financial_amount as pipeline_asks_finance

    return pipeline_asks_finance(question)


def has_financial_evidence(question: str, articles: list[dict[str, Any]]) -> bool:
    from rag.pipeline import has_financial_evidence as pipeline_has_finance

    return pipeline_has_finance(question, articles)


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


_FOLLOWUP_STOPWORDS = {
    "tom", "tat", "rut", "gon", "luoc", "y", "chinh", "noi", "dung", "day", "du",
    "chi", "tiet", "toan", "van", "doc", "xem", "nghe", "hien", "thi",
    "bai", "bao", "viet", "tin", "tuc", "nay", "do", "vua", "roi", "moi", "nhat",
    "giup", "toi", "minh", "ban", "cho", "xin", "hay", "vui", "long", "di", "nhe",
    "nha", "voi", "ngan", "lai", "nao", "sao", "the", "la", "gi", "ve", "cua",
    "tu", "tren", "nhung", "cac", "cai", "them", "con", "nua", "khong",
}

_QA_GENERIC_WORDS = _FOLLOWUP_STOPWORDS | {
    "nguyen", "nhan", "ket", "qua", "su", "viec", "vu", "viec", "chi", "phi",
    "kinh", "phi", "ngan", "sach", "bao", "nhieu", "khi", "nao", "o", "dau",
    "vi", "sao", "tai", "sao", "nhu", "the", "nao", "ra", "sao", "cu", "the",
    "ai", "phat", "bieu", "noi", "gi", "dien", "ra", "xay", "ra", "thoi", "gian",
    "dia", "diem", "muc", "dich", "giai", "phap", "anh", "huong", "tac", "dong",
    "de", "cap", "nhac", "den", "chuyen", "gi", "co", "duoc", "khong", "phai",
    "da", "dang", "se", "trong", "va", "voi", "nhung", "mot", "cac", "do",
}


def _extract_content_tokens(question: str, *, include_qa_generic: bool = False) -> list[str]:
    """Extract normalized alphanumeric tokens excluding stopwords."""
    stopwords = _QA_GENERIC_WORDS if include_qa_generic else _FOLLOWUP_STOPWORDS
    tokens = re.findall(r"[a-z0-9]+", _normalize_label(question))
    return [tok for tok in tokens if len(tok) > 1 and tok not in stopwords]


def _wants_summary_or_detail(question: str, intent: str) -> tuple[bool, bool]:
    """Detect summary or full-detail intent from classifier output and Vietnamese cues."""
    normalized = f" {_normalize_label(question)} "
    wants_summary = intent == "summarization" or any(
        cue in normalized
        for cue in (" tom tat ", " y chinh ", " rut gon ", " tom luoc ", " noi dung chinh ")
    )
    wants_detail = intent == "article_detail" or any(
        cue in normalized
        for cue in (" toan van ", " chi tiet ", " doc bai ", " noi dung day du ", " xem bai ", " xem chi tiet ")
    )
    return wants_summary, wants_detail


def _resolve_reference_article(
    question: str,
    context_sources: list[dict[str, Any]] | None,
    pinned_article: dict[str, Any] | None,
    active_article: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, bool, str | None]:
    """Resolve ordinal, demonstrative, or in-list source references to a target article.

    Returns ``(target_article, matched_reference, error_message)``.
    """
    normalized = f" {_normalize_label(question)} "

    # Check explicit Vietnamese ordinal phrases (longest / unambiguous forms first).
    word_ordinals: tuple[tuple[str, int], ...] = (
        ("bai dau tien", 0),
        ("tin dau tien", 0),
        ("bai thu nhat", 0),
        ("bai thu hai", 1),
        ("bai thu ba", 2),
        ("bai thu tu", 3),
        ("bai thu nam", 4),
        ("bai thu sau", 5),
        ("bai thu bay", 6),
        ("bai thu tam", 7),
        ("bai thu chin", 8),
        ("bai thu muoi", 9),
        ("bai dau", 0),
        ("tin dau", 0),
        ("bai mot", 0),
        ("bai hai", 1),
        ("bai ba", 2),
        ("bai bon", 3),
        ("bai nam", 4),
        ("bai sau", 5),
        ("bai bay", 6),
        ("bai tam", 7),
        ("bai chin", 8),
        ("bai muoi", 9),
    )

    requested_idx: int | None = None
    explicit_ordinal_found = False

    match = re.search(
        r"\b(?:bai(?:\s+(?:bao|viet))?|tin|muc|ket\s+qua)\s+(?:so\s+|thu\s+)?(\d{1,2})\b",
        normalized,
    )
    if not match:
        match = re.match(
            r"^\s*(?:xem|doc|tom\s+tat|chon|mo)?\s*so\s+(\d{1,2})\s*$",
            normalized.strip(),
        )
    if match:
        requested_idx = int(match.group(1)) - 1
        explicit_ordinal_found = True
    else:
        for phrase, idx in word_ordinals:
            if f" {phrase} " not in normalized:
                continue
            # Avoid misclassifying "bài năm 2026" as ordinal 5.
            if phrase == "bai nam" and re.search(r"\bbai nam \d", normalized):
                continue
            requested_idx = idx
            explicit_ordinal_found = True
            break

    if explicit_ordinal_found and requested_idx is not None:
        if not context_sources:
            return (
                None,
                True,
                "Hiện chưa có danh sách bài báo nào trong cuộc trò chuyện. "
                "Bạn hãy tìm kiếm chủ đề hoặc xem tin mới nhất trước khi chọn số thứ tự bài.",
            )
        if requested_idx < 0 or requested_idx >= len(context_sources):
            return (
                None,
                True,
                f"Danh sách vừa rồi chỉ có {len(context_sources)} bài báo "
                f"(từ số 1 đến số {len(context_sources)}). Bạn hãy chọn số thứ tự trong khoảng này.",
            )
        return context_sources[requested_idx], True, None

    if re.search(
        r"\b(?:bai\s+cuoi(?:\s+cung)?|tin\s+cuoi\s+cung|(?:doc|tom\s+tat|xem)\s+tin\s+cuoi(?:\s|$))",
        normalized,
    ):
        if not context_sources:
            return None, True, "Hiện chưa có danh sách bài báo nào để chọn bài cuối."
        return context_sources[-1], True, None

    if any(
        phrase in normalized
        for phrase in (
            " bai nay ",
            " bai do ",
            " bai vua roi ",
            " bai bao nay ",
            " bai bao do ",
            " tin nay ",
            " tin do ",
        )
    ):
        target = pinned_article or active_article or (context_sources[0] if context_sources else None)
        if target is None:
            return (
                None,
                True,
                "Hiện chưa có bài báo nào đang được chọn. "
                "Bạn hãy tìm kiếm hoặc chọn một bài báo trước nhé.",
            )
        return target, True, None

    # Publisher selection inside existing search results (e.g., "tóm tắt bài trên Tuổi Trẻ").
    if context_sources:
        mentioned_source = _extract_source(question)
        if mentioned_source:
            wants_summary, wants_detail = _wants_summary_or_detail(question, "")
            without_source = _strip_source_mentions(question)
            if (wants_summary or wants_detail) and not _extract_content_tokens(without_source):
                for item in context_sources:
                    if (item.get("source") or "").casefold() == mentioned_source:
                        return item, True, None

    return None, False, None


def _is_bare_followup_request(question: str, intent: str) -> bool:
    """Detect short follow-up commands without an explicit topic or target."""
    wants_summary, wants_detail = _wants_summary_or_detail(question, intent)
    if not (wants_summary or wants_detail):
        return False
    return len(_extract_content_tokens(question)) == 0


def _article_has_overlap(
    question_tokens: list[str],
    article: dict[str, Any],
    content: dict[str, Any] | None = None,
) -> bool:
    """Check whether any specific query token overlaps with an active article."""
    searchable = " ".join(
        [
            str(article.get("title") or ""),
            str(article.get("sapo") or ""),
            str((content or {}).get("title") or ""),
            str((content or {}).get("sapo") or ""),
            str((content or {}).get("body") or ""),
        ]
    )
    folded_searchable = _normalize_label(searchable)
    article_words = set(re.findall(r"[a-z0-9]+", folded_searchable))
    return any(tok in article_words or tok in folded_searchable for tok in question_tokens)


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
    concise: bool = False,
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
        f"Tìm thấy {total_count} bài khớp từ khóa (có ngày đăng ở {dated_count} bài); "
        f"hiển thị {len(articles)} bài mới nhất.",
    ]

    if not concise:
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
    if dated_count > len(articles) and not concise:
        lines.append(f"\nĐang hiển thị {len(articles)}/{dated_count} bài có ngày đăng.")
    if concise:
        lines.append(
            '💡 *Nhấn **✨ Tóm tắt**, **📖 Đọc bài**, **💬 Hỏi bài này** trên thẻ bên dưới, hoặc gõ **"Tóm tắt bài số 1"**.*'
        )

    return "\n\n".join(lines)


def _format_topic_search(
    topic: str,
    articles: list[dict[str, Any]],
    total_count: int,
    category_found: bool,
    available_categories: list[str],
    concise: bool = False,
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

    if not concise:
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

    if total_count > len(articles) and not concise:
        lines.append(f"\nĐang hiển thị {len(articles)}/{total_count} bài báo phù hợp.")
    if concise:
        lines.append(
            '💡 *Nhấn **✨ Tóm tắt**, **📖 Đọc bài**, **💬 Hỏi bài này** trên thẻ bên dưới, hoặc gõ **"Tóm tắt bài số 1"**.*'
        )

    return "\n\n".join(lines)


@lru_cache(maxsize=1)
def _load_summarizer():
    """Nạp ViT5 khi cần, tránh nạp mô hình nếu không tóm tắt."""
    from processing.summarizer import summarize

    return summarize


def _summarize_with_optional_id(body: str, article_id: Any = None) -> str:
    """Invoke cached summarizer while staying compatible with 1-arg test mocks."""
    summarizer_fn = _load_summarizer()
    try:
        return summarizer_fn(body, article_id=article_id)
    except TypeError:
        return summarizer_fn(body)


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
    """Tìm nguồn có toàn văn (hoặc sapo dự phòng) trong SQLite, ưu tiên điểm cao."""
    ranked = _rank_sources(sources)
    sapo_fallback: tuple[dict[str, Any], dict[str, Any]] | None = None
    for source in ranked:
        content = get_article_content(
            article_id=source.get("article_id"),
            url=source.get("url"),
        )
        if content and (content.get("body") or "").strip():
            return source, content
        if sapo_fallback is None and content and (content.get("sapo") or "").strip():
            sapo_fallback = (source, content)
    if sapo_fallback is not None:
        return sapo_fallback
    return None, None


def _format_search_results(sources: list[dict[str, Any]], concise: bool = False) -> str:
    """Tạo danh sách kết quả tìm kiếm, không gọi LLM."""
    ranked = _rank_sources(sources)
    if not ranked:
        return (
            "Tôi chưa tìm thấy bài báo nào phù hợp trong dữ liệu hiện tại. "
            "Bạn hãy thử diễn đạt lại chủ đề hoặc kiểm tra từ khóa tìm kiếm."
        )

    if concise:
        return (
            f"## Tìm thấy {len(ranked)} bài báo liên quan\n\n"
            '💡 *Nhấn **✨ Tóm tắt**, **📖 Đọc bài**, **💬 Hỏi bài này** trên thẻ bên dưới, hoặc gõ **"Tóm tắt bài số 1"**.*'
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


def _format_latest_news(
    articles: list[dict[str, Any]],
    total_count: int,
    dated_count: int,
    concise: bool = False,
) -> str:
    """Định dạng danh sách tin mới nhất và nêu rõ độ phủ dữ liệu ngày đăng."""
    if not articles:
        if dated_count == 0:
            return (
                "Hiện chưa có bài báo nào có ngày đăng trong cơ sở dữ liệu. "
                "Tôi chưa thể tạo danh sách tin mới nhất."
            )
        return "Tôi chưa tìm được bài báo có ngày đăng để hiển thị."

    lines = [f"## {len(articles)} bài báo mới nhất theo ngày đăng đã lưu"]
    if not concise:
        for i, article in enumerate(articles, start=1):
            title = article.get("title") or "Không có tiêu đề"
            newspaper = article.get("source") or "Không rõ nguồn"
            published = (
                article.get("published_at_display")
                or article.get("published_at")
                or "Không rõ ngày đăng"
            )
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
    if concise:
        lines.append(
            '💡 *Nhấn **✨ Tóm tắt**, **📖 Đọc bài**, **💬 Hỏi bài này** trên thẻ bên dưới, hoặc gõ **"Tóm tắt bài số 1"**.*'
        )
    return "\n\n".join(lines)


def _articles_to_sources(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build UI-safe source dicts (without chunk text blobs) for retrieved RAG articles."""
    return [
        {
            "source_number": i,
            "article_id": article.get("article_id"),
            "title": article.get("title", ""),
            "sapo": article.get("sapo", ""),
            "source": article.get("source", ""),
            "category": article.get("category", ""),
            "url": article.get("url", ""),
            "published_at": article.get("published_at", ""),
            "published_at_display": article.get("published_at_display", ""),
            "similarity": article.get("similarity"),
        }
        for i, article in enumerate(articles, start=1)
    ]


def _handle_target_article(
    question: str,
    target: dict[str, Any],
    intent: str,
    *,
    preloaded_content: dict[str, Any] | None = None,
    stream_qa: bool = False,
) -> dict[str, Any]:
    """Execute summarization, detail reading, or QA directly against one resolved article."""
    content = preloaded_content or get_article_content(
        article_id=target.get("article_id"),
        url=target.get("url"),
    )
    raw_body = ((content or {}).get("body") or "").strip()
    raw_sapo = ((content or {}).get("sapo") or target.get("sapo") or "").strip()

    body_text = raw_body or raw_sapo
    sapo_only_note = ""
    if not raw_body and raw_sapo:
        sapo_only_note = "\n\n*(Lưu ý: bài báo chỉ có phần mở đầu trong cơ sở dữ liệu)*"

    if not body_text:
        return {
            "kind": "answer",
            "intent": intent,
            "answer": "Không tìm thấy toàn văn của bài báo được chọn trong database.",
            "sources": [target],
            "selected_single": True,
        }

    selected = dict(target)
    selected["title"] = (content or {}).get("title") or target.get("title") or "Bài báo"
    selected.setdefault("sapo", raw_sapo)
    selected.setdefault("source", (content or {}).get("source", ""))
    selected.setdefault("category", (content or {}).get("category", ""))
    selected.setdefault("published_at_display", (content or {}).get("published_at_display", ""))

    wants_summary, wants_detail = _wants_summary_or_detail(question, intent)

    if wants_summary:
        summary = _summarize_with_optional_id(body_text, article_id=selected.get("article_id"))
        if not summary or not summary.strip():
            answer = "Mô hình ViT5 chưa tạo được bản tóm tắt."
        else:
            answer = f"## Tóm tắt: {selected['title']}\n\n{summary.strip()}{sapo_only_note}"
        return {
            "kind": "answer",
            "intent": "summarization",
            "answer": answer,
            "sources": [selected],
            "selected_single": True,
        }

    if wants_detail:
        return {
            "kind": "answer",
            "intent": "article_detail",
            "answer": f"## {selected['title']}\n\n{body_text}{sapo_only_note}",
            "sources": [selected],
            "selected_single": True,
        }

    article_ctx = {
        "article_id": selected.get("article_id"),
        "title": selected["title"],
        "sapo": raw_sapo,
        "source": selected.get("source", ""),
        "url": selected.get("url", ""),
        "texts": [body_text],
    }
    if stream_qa:
        return {
            "kind": "qa_stream",
            "intent": "question_answering",
            "stream": stream_answer(question, [article_ctx]),
            "sources": [selected],
            "selected_single": True,
        }

    answer = generate_answer(question, [article_ctx])
    return {
        "kind": "answer",
        "intent": "question_answering",
        "answer": answer,
        "sources": [selected],
        "selected_single": True,
    }


def _execute_turn_plan(
    question: str,
    *,
    context_sources: list[dict[str, Any]] | None = None,
    pinned_article: dict[str, Any] | None = None,
    active_article: dict[str, Any] | None = None,
    source_filter: str | None = None,
    category_filter: str | None = None,
    concise_list: bool = False,
    stream_qa: bool = False,
) -> dict[str, Any]:
    """Resolve intent and return either an immediate answer or a QA token stream."""
    if not question or not question.strip():
        return {
            "kind": "answer",
            "intent": "empty",
            "answer": "Bạn hãy nhập yêu cầu cần xử lý.",
            "sources": [],
            "selected_single": False,
        }

    effective_source = _extract_source(question) or source_filter
    intent = predict_intent(question)

    detected_topic = _extract_topic(question)
    if detected_topic and _has_topic_search_cue(question):
        intent = "topic_search"
    elif _has_latest_cue(question):
        intent = "latest_topic_search" if _extract_latest_topic_terms(question) else "latest_news"
    elif intent == "latest_topic_search":
        intent = "semantic_search"
    elif (
        intent not in {"greeting", "introduction", "summarization", "article_detail"}
        and _has_search_cue(question)
    ):
        intent = "semantic_search"

    print(f"[INTENT] {intent} | {question}")

    if intent == "greeting":
        return {
            "kind": "answer",
            "intent": intent,
            "answer": (
                "Xin chào! Tôi là Trợ lý đọc báo thông minh. "
                "Tôi có thể giúp bạn tìm kiếm, tóm tắt và hỏi đáp "
                "về các bài báo tiếng Việt."
            ),
            "sources": [],
            "selected_single": False,
        }

    if intent == "introduction":
        return {
            "kind": "answer",
            "intent": intent,
            "answer": INTRODUCTION_RESPONSE,
            "sources": [],
            "selected_single": False,
        }

    # 1. Explicit ordinal, demonstrative, or in-list source reference.
    ref_article, matched_ref, ref_error = _resolve_reference_article(
        question,
        context_sources=context_sources,
        pinned_article=pinned_article,
        active_article=active_article,
    )
    if matched_ref:
        if ref_error:
            return {
                "kind": "answer",
                "intent": intent,
                "answer": ref_error,
                "sources": [],
                "selected_single": False,
            }
        if ref_article is not None:
            return _handle_target_article(
                question,
                ref_article,
                intent,
                stream_qa=stream_qa,
            )

    # 2. Bare follow-up without explicit target (e.g., "Tóm tắt giúp tôi", "Đọc chi tiết").
    if _is_bare_followup_request(question, intent):
        target = pinned_article or active_article or (context_sources[0] if context_sources else None)
        if target is None:
            return {
                "kind": "answer",
                "intent": intent,
                "answer": (
                    "Bạn muốn tóm tắt hoặc xem chi tiết bài báo nào? "
                    "Hãy tìm một chủ đề trước hoặc nhấn nút **✨ Tóm tắt** trên thẻ bài báo nhé."
                ),
                "sources": [],
                "selected_single": False,
            }
        return _handle_target_article(question, target, intent, stream_qa=stream_qa)

    # 3. Explicitly pinned article takes precedence for follow-ups.
    if (
        pinned_article is not None
        and intent not in {"latest_news", "topic_search", "latest_topic_search"}
        and not _has_search_cue(question)
    ):
        return _handle_target_article(question, pinned_article, intent, stream_qa=stream_qa)

    # 4. Implicit follow-up QA on the active article when user continues asking about it.
    if (
        active_article is not None
        and intent == "question_answering"
        and not _has_search_cue(question)
        and not _has_latest_cue(question)
        and not _has_topic_search_cue(question)
    ):
        explicit_src = _extract_source(question)
        active_src = (active_article.get("source") or "").casefold()
        if not explicit_src or explicit_src == active_src:
            specific_tokens = _extract_content_tokens(question, include_qa_generic=True)
            active_content = get_article_content(
                article_id=active_article.get("article_id"),
                url=active_article.get("url"),
            )
            if not specific_tokens or _article_has_overlap(
                specific_tokens, active_article, content=active_content
            ):
                return _handle_target_article(
                    question,
                    active_article,
                    intent,
                    preloaded_content=active_content,
                    stream_qa=stream_qa,
                )

    if intent == "latest_news":
        if effective_source is None and category_filter is None:
            articles, total_count, dated_count = _get_latest_news(limit=10)
        else:
            articles, total_count, dated_count = _get_latest_news(
                limit=10,
                source=effective_source,
                category=category_filter,
            )
        answer = _format_latest_news(articles, total_count, dated_count, concise=concise_list)
        return {
            "kind": "answer",
            "intent": intent,
            "answer": answer,
            "sources": articles,
            "selected_single": False,
        }

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
            return {
                "kind": "answer",
                "intent": "latest_news",
                "answer": _format_latest_news(
                    articles, total_count, dated_count, concise=concise_list
                ),
                "sources": articles,
                "selected_single": False,
            }

        if effective_source is None and category_filter is None:
            articles, total_count, dated_count = _search_latest_topic_articles(terms, limit=10)
        else:
            articles, total_count, dated_count = _search_latest_topic_articles(
                terms,
                limit=10,
                source=effective_source,
                category=category_filter,
            )
        answer = _format_latest_topic_search(
            terms, articles, total_count, dated_count, concise=concise_list
        )
        return {
            "kind": "answer",
            "intent": intent,
            "answer": answer,
            "sources": articles,
            "selected_single": False,
        }

    if intent == "topic_search":
        topic = _extract_topic(question)
        if topic is None:
            return {
                "kind": "answer",
                "intent": intent,
                "answer": (
                    "Bạn hãy nêu rõ một chuyên mục, ví dụ: kinh tế, công nghệ, "
                    "thể thao, giải trí, giáo dục hoặc y tế."
                ),
                "sources": [],
                "selected_single": False,
            }

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
            concise=concise_list,
        )
        return {
            "kind": "answer",
            "intent": intent,
            "answer": answer,
            "sources": articles,
            "selected_single": False,
        }

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
            return {
                "kind": "answer",
                "intent": intent,
                "answer": _format_search_results(sources, concise=concise_list),
                "sources": sources,
                "selected_single": False,
            }

        source_item, content = _find_source_with_body(sources)
        if content is None:
            return {
                "kind": "answer",
                "intent": intent,
                "answer": _empty_retrieval_message(intent),
                "sources": sources,
                "selected_single": False,
            }

        title = content.get("title") or source_item.get("title") or "Bài báo"
        selected_source = dict(source_item)
        selected_source["title"] = title
        raw_body = (content.get("body") or "").strip()
        raw_sapo = (content.get("sapo") or source_item.get("sapo") or "").strip()
        body_text = raw_body or raw_sapo
        sapo_only_note = (
            "\n\n*(Lưu ý: bài báo chỉ có phần mở đầu trong cơ sở dữ liệu)*"
            if (not raw_body and raw_sapo)
            else ""
        )

        if intent == "article_detail":
            if not body_text:
                return {
                    "kind": "answer",
                    "intent": intent,
                    "answer": _empty_retrieval_message(intent),
                    "sources": [],
                    "selected_single": False,
                }
            return {
                "kind": "answer",
                "intent": intent,
                "answer": f"## {title}\n\n{body_text}{sapo_only_note}",
                "sources": [selected_source],
                "selected_single": True,
            }

        summary = _summarize_with_optional_id(
            body_text,
            article_id=selected_source.get("article_id"),
        )
        if not summary or not summary.strip():
            answer = "Mô hình ViT5 chưa tạo được bản tóm tắt."
        else:
            answer = f"## Tóm tắt: {title}\n\n{summary.strip()}{sapo_only_note}"
        return {
            "kind": "answer",
            "intent": intent,
            "answer": answer,
            "sources": [selected_source],
            "selected_single": True,
        }

    # Question Answering (Hybrid RAG + Qwen)
    clean_qa = _strip_source_mentions(question) if effective_source else question
    effective_qa = clean_qa or question

    if not stream_qa:
        ask_kwargs: dict[str, Any] = {}
        if effective_source is not None:
            ask_kwargs["source"] = effective_source
        if category_filter is not None:
            ask_kwargs["category"] = category_filter
        result = ask(effective_qa, **ask_kwargs)
        return {
            "kind": "answer",
            "intent": "question_answering",
            "answer": result.get("answer", "Chưa tạo được câu trả lời."),
            "sources": result.get("sources", []) or [],
            "selected_single": False,
        }

    retrieve_kwargs: dict[str, Any] = {}
    if effective_source is not None:
        retrieve_kwargs["source"] = effective_source
    if category_filter is not None:
        retrieve_kwargs["category"] = category_filter
    chunks = retrieve(effective_qa, **retrieve_kwargs)
    if not chunks:
        return {
            "kind": "answer",
            "intent": "question_answering",
            "answer": "Tôi chưa tìm thấy bài báo phù hợp để trả lời câu hỏi này.",
            "sources": [],
            "selected_single": False,
        }

    articles = group_chunks_by_article(chunks)
    sources = _articles_to_sources(articles)
    if asks_for_financial_amount(effective_qa) and not has_financial_evidence(effective_qa, articles):
        return {
            "kind": "answer",
            "intent": "question_answering",
            "answer": (
                "Tôi chưa tìm thấy bằng chứng đủ rõ trong "
                "các bài báo liên quan để xác định kinh phí "
                "của dự án. Tôi không thể kết luận số tiền "
                "từ dữ liệu hiện có."
            ),
            "sources": sources,
            "selected_single": False,
        }

    return {
        "kind": "qa_stream",
        "intent": "question_answering",
        "stream": stream_answer(effective_qa, articles),
        "sources": sources,
        "selected_single": False,
    }


def prepare_turn_response(
    question: str,
    context_sources: list[dict[str, Any]] | None = None,
    pinned_article: dict[str, Any] | None = None,
    active_article: dict[str, Any] | None = None,
    source_filter: str | None = None,
    category_filter: str | None = None,
    concise_list: bool = True,
) -> dict[str, Any]:
    """UI-oriented entry point returning either a complete answer or a live Qwen token stream."""
    return _execute_turn_plan(
        question,
        context_sources=context_sources,
        pinned_article=pinned_article,
        active_article=active_article,
        source_filter=source_filter,
        category_filter=category_filter,
        concise_list=concise_list,
        stream_qa=True,
    )


def handle_intent(
    question: str,
    context_sources: list[dict[str, Any]] | None = None,
    pinned_article: dict[str, Any] | None = None,
    source_filter: str | None = None,
    category_filter: str | None = None,
    active_article: dict[str, Any] | None = None,
    concise_list: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Dự đoán ý định và điều phối tới chức năng phù hợp."""
    outcome = _execute_turn_plan(
        question,
        context_sources=context_sources,
        pinned_article=pinned_article,
        active_article=active_article,
        source_filter=source_filter,
        category_filter=category_filter,
        concise_list=concise_list,
        stream_qa=False,
    )
    return outcome["answer"], outcome["sources"]
