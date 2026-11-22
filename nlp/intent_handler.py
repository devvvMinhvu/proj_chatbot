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

from dataclasses import replace
from datetime import date
from functools import lru_cache
import re
import unicodedata
from typing import Any

from config import CATEGORY_DISPLAY
from database.db import Article, SessionLocal
from database.repository import (
    get_archive_date_range,
    get_article_content,
    get_latest_articles,
    search_latest_topic_articles,
    search_topic_articles,
)
from nlp.conversation_state import ConversationState, ListRequest, TurnRecord
from nlp.intent_classifier import predict_intent
from nlp.keyword_stats import conversation_words
from nlp.turn_parser import ParsedTurn, describe as describe_turn, parse_turn
from processing.text_matching import matched_keywords


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


def _order_by_date(articles):
    from rag.pipeline import order_by_date

    return order_by_date(articles)


def stream_answer(*args, **kwargs):
    from rag.pipeline import stream_answer as pipeline_stream_answer

    return pipeline_stream_answer(*args, **kwargs)


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
        "co bai viet nao",
        "co tin nao",
        "co noi dung nao",
        "bai nao noi ve",
        "bai nao nhac",
        "find one",
        "mentioning",
        "find articles",
        "search for",
    )
    return any(f" {cue} " in normalized_question for cue in cues)


#: Folded question phrases that ask for a fact from the news rather than a list of articles.
QUESTION_CUES = (
    "vi sao", "tai sao", "the nao", "ra sao", "bao nhieu", "khi nao", "bao gio",
    "o dau", "la gi", "la ai", "ai la", "nguyen nhan", "co phai",
)

#: Classifier intents the question rule may override (category browsing is left alone).
_SEARCH_INTENTS = {"semantic_search", "latest_topic_search", "unknown"}


def _has_question_cue(question: str) -> bool:
    """Nhận biết câu hỏi cần trả lời (vì sao, bao nhiêu, ai…, hoặc kết thúc bằng '?')."""
    normalized_question = f" {_normalize_label(question)} "
    if (question or "").strip().endswith("?") or normalized_question.startswith(" ai "):
        return True
    return any(f" {cue} " in normalized_question for cue in QUESTION_CUES)


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


def _article_covers_question(
    question: str,
    question_tokens: list[str],
    article: dict[str, Any],
    content: dict[str, Any] | None = None,
) -> bool:
    """Check whether every specific word of the question appears in the active article.

    Words are matched as whole words and accent-aware ("vàng" is not "vang"), so a new
    question such as "Vì sao giá vàng tăng?" is not answered from an article about fuel
    prices just because both mention "giá" and "tăng".
    """
    searchable = " ".join(
        [
            str(article.get("title") or ""),
            str(article.get("sapo") or ""),
            str((content or {}).get("title") or ""),
            str((content or {}).get("sapo") or ""),
            str((content or {}).get("body") or ""),
        ]
    )
    wanted = set(question_tokens)
    words = [
        word
        for word in re.findall(r"[0-9a-zA-ZÀ-ỹĐđ]+", question or "")
        if _normalize_label(word) in wanted
    ]
    distinct = {word.casefold() for word in words}
    return len(matched_keywords(sorted(distinct), searchable)) == len(distinct)


def _get_latest_news(limit: int = 10, **filters: Any):
    """Latest articles; ``filters`` are source, category, published_from, published_to."""
    return get_latest_articles(limit=limit, **filters)


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
    category: str,
    limit: int = 10,
    source: str | None = None,
):
    """Articles of one category, given as its database value (e.g. "sports")."""
    kwargs: dict[str, Any] = {
        "accepted_values": {_normalize_label(category)},
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


def _outcome(
    intent: str,
    answer: str,
    sources: list[dict[str, Any]] | None = None,
    *,
    selected_single: bool = False,
    is_list: bool = False,
) -> dict[str, Any]:
    return {
        "kind": "answer",
        "intent": intent,
        "answer": answer,
        "sources": sources or [],
        "selected_single": selected_single,
        "is_list": is_list,
    }


def _with_note(outcome: dict[str, Any], note: str) -> dict[str, Any]:
    """Prefix a short italic note to an answer or a streamed answer."""
    if not note:
        return outcome
    if outcome.get("kind") == "qa_stream":
        inner = outcome["stream"]

        def noted_stream():
            yield note
            yield from inner

        outcome["stream"] = noted_stream()
    else:
        outcome["answer"] = note + str(outcome.get("answer") or "")
    return outcome


def _article_context(target: dict[str, Any], content: dict[str, Any] | None) -> dict[str, Any] | None:
    """Article in the form the RAG prompt expects, with its publication details."""
    content = content or {}
    body = (content.get("body") or "").strip() or (content.get("sapo") or target.get("sapo") or "").strip()
    if not body:
        return None
    return {
        "article_id": target.get("article_id") or content.get("article_id"),
        "title": content.get("title") or target.get("title") or "Bài báo",
        "sapo": (content.get("sapo") or target.get("sapo") or "").strip(),
        "source": content.get("source") or target.get("source", ""),
        "category": content.get("category") or target.get("category", ""),
        "url": content.get("url") or target.get("url", ""),
        "published_at": content.get("published_at") or target.get("published_at", ""),
        "published_at_display": content.get("published_at_display")
        or target.get("published_at_display", ""),
        "texts": [body],
    }


def _limit_sentences(text: str, max_sentences: int | None) -> str:
    if not max_sentences:
        return text
    from processing.sentences import split_sentences

    sentences = split_sentences(text)
    return " ".join(sentences[:max_sentences]) if sentences else text


def _checked_summary(summary: str | None, body: str, max_sentences: int | None) -> tuple[str, str]:
    """Keep summary sentences whose names and numbers are in the article; respect the length.

    If no sentence survives, the article's opening sentences are used instead.
    Returns (summary, note for the reader).
    """
    from processing.fact_check import keep_supported_sentences

    kept, dropped = keep_supported_sentences((summary or "").strip(), body)
    note = ""
    if dropped and kept:
        note = f"\n\n*(Đã lược {dropped} câu có tên hoặc số liệu không có trong bài.)*"
    if not kept.strip() and (summary or "").strip():
        from processing.summarizers.extractive import LeadSummarizer

        kept = LeadSummarizer(num_sentences=max_sentences or 2).summarize(body)
        note = "\n\n*(Bản tóm tắt tự động có thông tin không khớp với bài nên đã dùng các câu mở đầu của bài.)*"
    return _limit_sentences(kept.strip(), max_sentences), note


def _handle_target_article(
    question: str,
    target: dict[str, Any],
    intent: str,
    *,
    action: str | None = None,
    max_sentences: int | None = None,
    preloaded_content: dict[str, Any] | None = None,
    stream_qa: bool = False,
) -> dict[str, Any]:
    """Summarize, read or answer a question about one resolved article.

    ``action`` is "summary", "detail" or "qa"; without it the request wording decides.
    """
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
        return _outcome(
            intent,
            "Không tìm thấy toàn văn của bài báo được chọn trong database.",
            [target],
            selected_single=True,
        )

    selected = dict(target)
    selected["title"] = (content or {}).get("title") or target.get("title") or "Bài báo"
    selected.setdefault("sapo", raw_sapo)
    selected.setdefault("source", (content or {}).get("source", ""))
    selected.setdefault("category", (content or {}).get("category", ""))
    selected.setdefault("published_at_display", (content or {}).get("published_at_display", ""))

    if action is None:
        wants_summary, wants_detail = _wants_summary_or_detail(question, intent)
        action = "summary" if wants_summary else "detail" if wants_detail else "qa"

    if action == "summary":
        summary, check_note = _checked_summary(
            _summarize_with_optional_id(body_text, article_id=selected.get("article_id")),
            body_text,
            max_sentences,
        )
        if not summary:
            answer = "Mô hình ViT5 chưa tạo được bản tóm tắt."
        else:
            answer = f"## Tóm tắt: {selected['title']}\n\n{summary}{check_note}{sapo_only_note}"
        return _outcome("summarization", answer, [selected], selected_single=True)

    if action == "detail":
        return _outcome(
            "article_detail",
            f"## {selected['title']}\n\n{body_text}{sapo_only_note}",
            [selected],
            selected_single=True,
        )

    article_ctx = _article_context(selected, content) or {
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
            "is_list": False,
        }
    answer = generate_answer(question, [article_ctx])
    return _outcome("question_answering", answer, [selected], selected_single=True)


def _answer_from_articles(
    question: str,
    contexts: list[dict[str, Any]],
    *,
    stream_qa: bool,
) -> dict[str, Any]:
    """Grounded answer over several articles (comparisons, follow-up questions)."""
    contexts = _order_by_date(contexts)
    sources = _articles_to_sources(contexts)
    if stream_qa:
        return {
            "kind": "qa_stream",
            "intent": "question_answering",
            "stream": stream_answer(question, contexts),
            "sources": sources,
            "selected_single": False,
            "is_list": False,
        }
    return _outcome("question_answering", generate_answer(question, contexts), sources)


# --- Resolving which article a message means -------------------------------------


_NO_LIST_MESSAGE = (
    "Hiện chưa có danh sách bài báo nào trong cuộc trò chuyện. "
    "Bạn hãy tìm kiếm chủ đề hoặc xem tin mới nhất trước khi chọn số thứ tự bài."
)


def _resolve_targets(
    parsed: ParsedTurn,
    state: ConversationState,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Articles the message points to: (targets, note for the reply, error message)."""
    items = state.items
    if parsed.item_refs:
        if not items:
            return [], "", _NO_LIST_MESSAGE
        out_of_range = [ref for ref in parsed.item_refs if not 1 <= ref <= len(items)]
        if out_of_range:
            return [], "", (
                f"Danh sách vừa rồi chỉ có {len(items)} bài báo "
                f"(từ số 1 đến số {len(items)}). Bạn hãy chọn số thứ tự trong khoảng này."
            )
        note = ""
        if state.empty_request is not None:
            older = state.list_request.describe() if state.list_request else "danh sách trước"
            note = (
                f"*(Lần tìm gần nhất \"{state.empty_request.describe()}\" không có kết quả; "
                f"số thứ tự theo danh sách trước: {older}.)*\n\n"
            )
        return [items[ref - 1] for ref in parsed.item_refs], note, None

    if parsed.relative is None:
        return [], "", None

    current = state.current_article()
    if parsed.relative == "current":
        if current is not None:
            return [current], "", None
        last_was_answer = state.last is not None and state.last.intent == "question_answering"
        if items and not last_was_answer:
            return [items[0]], "", None
        return [], "", None  # the caller decides: follow-up question or "nothing selected"

    if not items:
        return [], "", _NO_LIST_MESSAGE
    if parsed.relative == "first":
        return [items[0]], "", None
    if parsed.relative == "last":
        return [items[-1]], "", None
    position = state.index_of(current)
    if position is None:
        if parsed.relative == "next":
            return [items[0]], "", None
        return [], "", "Bạn chưa mở bài nào trong danh sách nên chưa có bài trước đó."
    step = 1 if parsed.relative == "next" else -1
    new_position = position + step
    if not 0 <= new_position < len(items):
        edge = "cuối" if step > 0 else "đầu"
        return [], "", f"Đây đã là bài {edge} của danh sách ({position + 1}/{len(items)})."
    return [items[new_position]], "", None


def _targets_cover_topic(targets: list[dict[str, Any]], parsed: ParsedTurn) -> bool:
    """A number like "lần thứ 8" only points to an item when that item is about the question."""
    if not parsed.topic_nouns:
        return True
    for target in targets:
        content = get_article_content(article_id=target.get("article_id"), url=target.get("url")) or {}
        text = " ".join(
            str(value or "")
            for value in (target.get("title"), target.get("sapo"), content.get("title"), content.get("body"))
        )
        if matched_keywords(parsed.topic_nouns, text, exact=parsed.exact_accents):
            return True
    return False


def _target_action(question: str, parsed: ParsedTurn, intent: str) -> str:
    wants_summary, wants_detail = _wants_summary_or_detail(question, "")
    if wants_summary:
        return "summary"
    if wants_detail:
        return "detail"
    if parsed.max_sentences:
        return "summary"
    if parsed.is_question:
        return "qa"
    if intent == "summarization":
        return "summary"
    return "detail"


# --- Lists -------------------------------------------------------------------------


def _date_note(request: ListRequest) -> tuple[ListRequest, str]:
    """Drop a date filter that lies outside the archive, with a note saying so."""
    if request.date_from is None:
        return request, ""
    oldest, newest = get_archive_date_range()
    if newest is not None and request.date_from > newest:
        note = (
            f"*(Dữ liệu chỉ có bài đến ngày {newest:%d/%m/%Y}, chưa có bài ngày "
            f"{request.date_from:%d/%m/%Y}. Dưới đây là các bài mới nhất hiện có.)*\n\n"
        )
        return replace(request, date_from=None, date_to=None), note
    if oldest is not None and request.date_to is not None and request.date_to < oldest:
        note = f"*(Dữ liệu bắt đầu từ ngày {oldest:%d/%m/%Y}.)*\n\n"
        return replace(request, date_from=None, date_to=None), note
    return request, ""


def _in_date_range(article: dict[str, Any], request: ListRequest) -> bool:
    if request.date_from is None:
        return True
    raw = str(article.get("published_at") or "")[:10]
    try:
        published = date.fromisoformat(raw)
    except ValueError:
        return False
    return request.date_from <= published <= (request.date_to or request.date_from)


def _run_list_request(request: ListRequest, *, concise_list: bool) -> dict[str, Any]:
    """Produce a list of articles for a latest / topic / category / keyword request."""
    request, note = _date_note(request)
    limit = request.count or 10
    words = request.topic_words

    if request.action == "latest" or (request.action in {"latest_topic", "search"} and not words):
        latest_kwargs: dict[str, Any] = {}
        for key, value in (
            ("source", request.source),
            ("category", request.category),
            ("published_from", request.date_from),
            ("published_to", request.date_to),
        ):
            if value is not None:
                latest_kwargs[key] = value
        articles, total_count, dated_count = _get_latest_news(limit=limit, **latest_kwargs)
        answer = _format_latest_news(articles, total_count, dated_count, concise=concise_list)
        return _with_note(_outcome("latest_news", answer, articles, is_list=True), note)

    if request.action == "latest_topic":
        topic_kwargs: dict[str, Any] = {}
        if request.source is not None:
            topic_kwargs["source"] = request.source
        if request.category is not None:
            topic_kwargs["category"] = request.category
        articles, total_count, dated_count = _search_latest_topic_articles(
            words,
            limit=limit if request.date_from is None else max(limit, 50),
            **topic_kwargs,
        )
        if request.date_from is not None:
            articles = [a for a in articles if _in_date_range(a, request)][:limit]
            dated_count = total_count = len(articles)
        answer = _format_latest_topic_search(words, articles, total_count, dated_count, concise=concise_list)
        return _with_note(_outcome("latest_topic_search", answer, articles, is_list=True), note)

    if request.action == "category" and not words and request.category:
        category_kwargs: dict[str, Any] = {"source": request.source} if request.source else {}
        articles, total_count, category_found, available = _search_topic_articles(
            request.category, limit=limit, **category_kwargs
        )
        label = CATEGORY_DISPLAY.get(request.category, request.category)
        answer = _format_topic_search(label, articles, total_count, category_found, available, concise=concise_list)
        return _with_note(_outcome("topic_search", answer, articles, is_list=True), note)

    query = request.question or " ".join(words)
    search_kwargs: dict[str, Any] = {}
    if request.source is not None:
        search_kwargs["source"] = request.source
    if request.category is not None:
        search_kwargs["category"] = request.category
    sources = _search_sources(query, **search_kwargs)
    sources = [s for s in sources if _in_date_range(s, request)][:limit]
    return _with_note(
        _outcome("semantic_search", _format_search_results(sources, concise=concise_list), sources, is_list=True),
        note,
    )


def _narrow_current_list(
    parsed: ParsedTurn,
    state: ConversationState,
    *,
    concise_list: bool,
) -> dict[str, Any]:
    """"Trong những bài này, chỉ lấy bài về X": keep the listed articles about X."""
    kept = []
    for item in state.items:
        content = get_article_content(article_id=item.get("article_id"), url=item.get("url")) or {}
        text = " ".join(str(v or "") for v in (item.get("title"), item.get("sapo"), content.get("body")))
        if matched_keywords(parsed.topic_nouns, text, exact=parsed.exact_accents):
            kept.append(item)
    words = " ".join(parsed.topic_nouns)
    if kept:
        answer = (
            f"## {len(kept)}/{len(state.items)} bài trong danh sách vừa rồi nhắc đến {words}\n\n"
            '💡 *Nhấn **✨ Tóm tắt**, **📖 Đọc bài**, **💬 Hỏi bài này** trên thẻ bên dưới, hoặc gõ **"Tóm tắt bài số 1"**.*'
        )
        if not concise_list:
            answer = _format_search_results(kept, concise=False)
        return _outcome("semantic_search", answer, kept, is_list=True)
    base = state.list_request or ListRequest(action="search")
    widened = replace(base, action="search", topic_words=list(parsed.topic_nouns), question=words)
    outcome = _run_list_request(widened, concise_list=concise_list)
    return _with_note(outcome, f"*(Danh sách vừa rồi không có bài nào nhắc đến {words}; tìm thêm trong dữ liệu.)*\n\n")


# --- Questions ---------------------------------------------------------------------


def _follow_up_answer(
    question: str,
    parsed: ParsedTurn,
    state: ConversationState,
    *,
    stream_qa: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Answer a question that points back to the previous turn ("dự án này ở tỉnh nào?")."""
    last = state.last
    previous_words = list(last.topic_words) if last else []
    own_words = list(parsed.topic_words)
    note = f"*(Hiểu là câu hỏi tiếp theo về: {' '.join(previous_words)})*\n\n" if previous_words else ""
    llm_question = f"{question}\n(Câu hỏi trước đó: {last.question})" if last else question

    contexts: list[dict[str, Any]] = []
    for source in (last.sources if last else [])[:5]:
        content = get_article_content(article_id=source.get("article_id"), url=source.get("url"))
        ctx = _article_context(source, content)
        if ctx is None:
            continue
        text = " ".join([ctx["title"], ctx["sapo"], *ctx["texts"]])
        # "dự án này" names the previous subject explicitly: answer from its articles,
        # even when the question words differ ("kinh phí" vs "tài trợ"). Without such a
        # pointer, the earlier articles must contain the question's own words.
        explicit_pointer = parsed.relative == "current"
        covered = len(matched_keywords(own_words, text, exact=parsed.exact_accents)) == len(set(own_words))
        if explicit_pointer or not own_words or covered:
            contexts.append(ctx)
    if contexts:
        outcome = _answer_from_articles(llm_question, contexts, stream_qa=stream_qa)
        return _with_note(outcome, note), previous_words + own_words

    expanded = " ".join(previous_words + own_words) or question
    chunks = retrieve(expanded)
    if not chunks:
        return _outcome("question_answering", "Tôi chưa tìm thấy bài báo phù hợp để trả lời câu hỏi này."), own_words
    articles = group_chunks_by_article(chunks)
    return _with_note(_answer_from_articles(llm_question, articles, stream_qa=stream_qa), note), previous_words + own_words


def _question_answer(
    question: str,
    parsed: ParsedTurn,
    *,
    source: str | None,
    category: str | None,
    stream_qa: bool,
) -> dict[str, Any]:
    clean_qa = _strip_source_mentions(question) if source else question
    effective_qa = clean_qa or question
    if not stream_qa:
        ask_kwargs: dict[str, Any] = {}
        if source is not None:
            ask_kwargs["source"] = source
        if category is not None:
            ask_kwargs["category"] = category
        result = ask(effective_qa, **ask_kwargs)
        return _outcome(
            "question_answering",
            result.get("answer", "Chưa tạo được câu trả lời."),
            result.get("sources", []) or [],
        )

    retrieve_kwargs: dict[str, Any] = {}
    if source is not None:
        retrieve_kwargs["source"] = source
    if category is not None:
        retrieve_kwargs["category"] = category
    chunks = retrieve(effective_qa, **retrieve_kwargs)
    if not chunks:
        return _outcome("question_answering", "Tôi chưa tìm thấy bài báo phù hợp để trả lời câu hỏi này.")
    articles = group_chunks_by_article(chunks)
    return _answer_from_articles(effective_qa, articles, stream_qa=True)


# --- The turn ------------------------------------------------------------------------


def _decide(
    question: str,
    parsed: ParsedTurn,
    state: ConversationState,
    *,
    source_filter: str | None,
    category_filter: str | None,
    concise_list: bool,
    stream_qa: bool,
) -> tuple[dict[str, Any], TurnRecord]:
    source = parsed.source or source_filter
    category = parsed.category or category_filter
    record = TurnRecord(
        question=question,
        intent="",
        topic_words=list(parsed.topic_words),
        max_sentences=parsed.max_sentences,
    )
    intent = predict_intent(question)

    # Small talk: trusted only when the message has no topic of its own.
    small_talk_words = conversation_words()
    if intent in {"greeting", "introduction"} and all(
        word in small_talk_words for word in parsed.topic_words
    ) and not (parsed.item_refs or parsed.relative):
        if intent == "greeting":
            answer = (
                "Xin chào! Tôi là Trợ lý đọc báo thông minh. "
                "Tôi có thể giúp bạn tìm kiếm, tóm tắt và hỏi đáp "
                "về các bài báo tiếng Việt."
            )
        else:
            answer = INTRODUCTION_RESPONSE
        return _outcome(intent, answer), record

    # 1. A specific article: number, position or "bài này".
    targets, note, error = _resolve_targets(parsed, state)
    if error:
        return _outcome(intent, error), record
    if targets and parsed.item_refs and not _targets_cover_topic(targets, parsed):
        targets, note = [], ""  # "lần thứ 8" in a question about something else
    if targets:
        action = _target_action(question, parsed, intent)
        if action == "qa" and len(targets) > 1:
            contexts = [
                ctx
                for target in targets
                if (ctx := _article_context(target, get_article_content(
                    article_id=target.get("article_id"), url=target.get("url")
                ))) is not None
            ]
            return _with_note(_answer_from_articles(question, contexts, stream_qa=stream_qa), note), record
        outcome = _handle_target_article(
            question,
            targets[-1],  # in "không phải bài 1, ý tôi là bài 2" the last number wins
            intent,
            action=action,
            max_sentences=parsed.max_sentences,
            stream_qa=stream_qa,
        )
        return _with_note(outcome, note), record
    if parsed.relative == "current" and not parsed.is_question and state.current_article() is None and not state.items:
        return _outcome(
            intent,
            "Hiện chưa có bài báo nào đang được chọn. Bạn hãy tìm kiếm hoặc chọn một bài báo trước nhé.",
        ), record
    wants_summary, wants_detail = _wants_summary_or_detail(question, intent)
    if (wants_summary or wants_detail) and not parsed.topic_nouns and not parsed.is_question:
        # "Tóm tắt giúp tôi": the open article, else the first in the list.
        target = state.current_article() or (state.items[0] if state.items else None)
        if target is not None:
            return _handle_target_article(
                question, target, intent,
                action="summary" if wants_summary else "detail",
                max_sentences=parsed.max_sentences, stream_qa=stream_qa,
            ), record
        return _outcome(
            intent,
            "Bạn muốn tóm tắt hoặc xem chi tiết bài báo nào? "
            "Hãy tìm một chủ đề trước hoặc nhấn nút **✨ Tóm tắt** trên thẻ bài báo nhé.",
        ), record
    if parsed.relative == "current" and not parsed.is_question and state.last is None:
        return _outcome(
            intent,
            "Hiện chưa có bài báo nào đang được chọn. Bạn hãy tìm kiếm hoặc chọn một bài báo trước nhé.",
        ), record

    # 2. A refinement of the previous turn: new filters, count or length, no new subject.
    last = state.last
    asks_for_new_list = _has_latest_cue(question) or _has_search_cue(question) or _has_topic_search_cue(question)
    if (
        last is not None
        and not parsed.topic_nouns
        and not asks_for_new_list
        and (parsed.has_slots or parsed.refers_to_list)
    ):
        if parsed.max_sentences and last.target is not None and last.intent == "summarization":
            outcome = _handle_target_article(
                question, last.target, "summarization", action="summary",
                max_sentences=parsed.max_sentences, stream_qa=stream_qa,
            )
            return outcome, record
        base = state.list_request or last.request
        if base is not None and (parsed.source or parsed.category or parsed.count or parsed.date_from):
            merged = replace(
                base,
                source=parsed.source or base.source,
                category=parsed.category or base.category,
                count=parsed.count or base.count,
                date_from=parsed.date_from or base.date_from,
                date_to=parsed.date_to or base.date_to,
            )
            record.request = merged
            record.topic_words = list(merged.topic_words)
            outcome = _run_list_request(merged, concise_list=concise_list)
            return _with_note(outcome, f"*(Lọc lại: {merged.describe()})*\n\n"), record

    # 3. Narrow the current list to a subject: "trong những bài này, chỉ lấy bài về Arsenal".
    if parsed.refers_to_list and parsed.topic_nouns and state.items:
        return _narrow_current_list(parsed, state, concise_list=concise_list), record

    # 4. A question about the news.
    current = state.current_article()
    asks = parsed.is_question and not _has_search_cue(question) and not _has_latest_cue(question)
    if asks and not (parsed.category and _has_topic_search_cue(question)):
        if current is not None and not _has_search_cue(question):
            specific_tokens = _extract_content_tokens(question, include_qa_generic=True)
            active_content = get_article_content(article_id=current.get("article_id"), url=current.get("url"))
            if not specific_tokens or _article_covers_question(question, specific_tokens, current, content=active_content):
                outcome = _handle_target_article(
                    question, current, "question_answering", action="qa",
                    preloaded_content=active_content, stream_qa=stream_qa,
                )
                return outcome, record
        points_back = parsed.relative == "current" or not parsed.topic_nouns
        if points_back and last is not None and last.topic_words:
            outcome, record.topic_words = _follow_up_answer(question, parsed, state, stream_qa=stream_qa)
            return outcome, record
        return _question_answer(question, parsed, source=source, category=category, stream_qa=stream_qa), record

    # 5. A list: category, latest news, latest on a topic, or a keyword search.
    request = ListRequest(
        action="search",
        topic_words=list(parsed.topic_words),
        source=source,
        category=category,
        count=parsed.count,
        date_from=parsed.date_from,
        date_to=parsed.date_to,
        question=question,
    )
    if _has_topic_search_cue(question) or (intent == "topic_search" and parsed.category):
        if parsed.category is None and not parsed.topic_words:
            labels = ", ".join(label.lower() for label in CATEGORY_DISPLAY.values())
            return _outcome(intent, f"Bạn hãy nêu rõ một chuyên mục, ví dụ: {labels}."), record
        if parsed.category is None:
            # A category the data does not have: search its words across all articles.
            record.request = replace(request, action="search")
            outcome = _run_list_request(record.request, concise_list=concise_list)
            words = " ".join(parsed.topic_words)
            return _with_note(outcome, f"*(Dữ liệu không có chuyên mục \"{words}\"; đây là các bài nhắc đến {words}.)*\n\n"), record
        record.request = replace(request, action="category")
    elif _has_latest_cue(question) or parsed.date_from is not None or (
        intent in {"latest_news", "latest_topic_search"} and not _has_search_cue(question)
    ):
        record.request = replace(request, action="latest_topic" if parsed.topic_words else "latest")
    elif intent in {"summarization", "article_detail"} and not _has_search_cue(question):
        return _search_then_open(question, intent, source=source, category=category), record
    else:
        record.request = request
    return _run_list_request(record.request, concise_list=concise_list), record


def _search_then_open(question: str, intent: str, *, source: str | None, category: str | None) -> dict[str, Any]:
    """"Tóm tắt bài về X": find the best article about X, then summarize or show it."""
    search_kwargs: dict[str, Any] = {}
    if source is not None:
        search_kwargs["source"] = source
    if category is not None:
        search_kwargs["category"] = category
    sources = _search_sources(question, **search_kwargs)
    source_item, content = _find_source_with_body(sources)
    if content is None:
        return _outcome(intent, _empty_retrieval_message(intent), sources)
    return _handle_target_article(
        question,
        source_item,
        intent,
        action="summary" if intent == "summarization" else "detail",
        preloaded_content=content,
    )


def respond(
    question: str,
    state: ConversationState,
    *,
    source_filter: str | None = None,
    category_filter: str | None = None,
    concise_list: bool = True,
    stream_qa: bool = True,
    today: date | None = None,
) -> dict[str, Any]:
    """Answer one message and update ``state``.

    Returns the outcome dict used by the app: ``kind`` ("answer" or "qa_stream"),
    ``intent``, ``answer`` or ``stream``, ``sources``, ``selected_single``.
    """
    if not question or not question.strip():
        return _outcome("empty", "Bạn hãy nhập yêu cầu cần xử lý.")
    parsed = parse_turn(question, today)
    outcome, record = _decide(
        question,
        parsed,
        state,
        source_filter=source_filter,
        category_filter=category_filter,
        concise_list=concise_list,
        stream_qa=stream_qa,
    )
    record.intent = outcome.get("intent", "")
    print(f"[INTENT] {record.intent} | {question} | {describe_turn(parsed)}")
    state.record(record, outcome)
    return outcome


def respond_on_article(
    question: str,
    article: dict[str, Any],
    action: str,
    state: ConversationState,
    *,
    stream_qa: bool = True,
) -> dict[str, Any]:
    """A button on an article card: "summarize", "detail" or "qa" for that article."""
    mapping = {
        "summarize": ("summary", "summarization"),
        "detail": ("detail", "article_detail"),
        "qa": ("qa", "question_answering"),
    }
    handler_action, intent = mapping.get(action, mapping["summarize"])
    outcome = _handle_target_article(question, article, intent, action=handler_action, stream_qa=stream_qa)
    parsed = parse_turn(question)
    record = TurnRecord(
        question=question,
        intent=outcome.get("intent", intent),
        topic_words=list(parsed.topic_words),
        max_sentences=parsed.max_sentences,
    )
    state.record(record, outcome)
    return outcome


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
    state: ConversationState | None = None,
) -> dict[str, Any]:
    """Compatibility entry point taking the older separate context values."""
    if state is None:
        state = ConversationState.from_legacy(context_sources, pinned_article, active_article)
    return respond(
        question,
        state,
        source_filter=source_filter,
        category_filter=category_filter,
        concise_list=concise_list,
        stream_qa=stream_qa,
    )


def prepare_turn_response(
    question: str,
    context_sources: list[dict[str, Any]] | None = None,
    pinned_article: dict[str, Any] | None = None,
    active_article: dict[str, Any] | None = None,
    source_filter: str | None = None,
    category_filter: str | None = None,
    concise_list: bool = True,
    state: ConversationState | None = None,
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
        state=state,
    )


def handle_intent(
    question: str,
    context_sources: list[dict[str, Any]] | None = None,
    pinned_article: dict[str, Any] | None = None,
    source_filter: str | None = None,
    category_filter: str | None = None,
    active_article: dict[str, Any] | None = None,
    concise_list: bool = False,
    state: ConversationState | None = None,
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
        state=state,
    )
    return outcome["answer"], outcome["sources"]
