"""Bộ điều phối ý định cho trợ lý đọc báo.

- greeting/introduction: trả lời trực tiếp.
- semantic_search: PhoBERT + FAISS, không gọi Qwen.
- summarization: PhoBERT + FAISS -> SQLite -> ViT5, không gọi Qwen.
- article_detail: PhoBERT + FAISS -> SQLite, không gọi Qwen.
- question_answering: dùng pipeline RAG + Qwen.
- latest_news: truy vấn SQLite theo published_at; nếu còn bài thiếu ngày đăng, cảnh báo độ phủ dữ liệu.

Các hàm điều phối chính trả về (answer, sources) cho giao diện Streamlit.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from database.repository import get_article_content, get_latest_articles
from nlp.intent_classifier import predict_intent
from rag.pipeline import ask, group_chunks_by_article, retrieve

INTRODUCTION_RESPONSE = """Tôi là Trợ lý đọc báo thông minh, hỗ trợ bạn khai thác
thông tin từ các bài báo tiếng Việt.

Các chức năng chính:
- Tìm kiếm bài báo theo nội dung bằng PhoBERT và FAISS.
- Tóm tắt bài báo bằng mô hình ViT5.
- Hỏi đáp dựa trên bài báo bằng RAG và Qwen.
- Xem thông tin nguồn và toàn văn bài báo.
"""


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


def _search_sources(question: str) -> list[dict[str, Any]]:
    """Truy xuất bằng PhoBERT + FAISS, gom chunk thành bài báo."""
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


def handle_intent(question: str) -> tuple[str, list[dict[str, Any]]]:
    """Dự đoán ý định và điều phối tới chức năng phù hợp."""
    if not question or not question.strip():
        return "Bạn hãy nhập yêu cầu cần xử lý.", []

    intent = predict_intent(question)
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
        articles, total_count, dated_count = get_latest_articles(limit=10)
        answer = _format_latest_news(articles, total_count, dated_count)
        return answer, articles

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

        summary = _load_summarizer()(content["body"])
        if not summary or not summary.strip():
            return "Mô hình ViT5 chưa tạo được bản tóm tắt.", [selected_source]
        return f"## Tóm tắt: {title}\n\n{summary.strip()}", [selected_source]

    result = ask(question)
    return (
        result.get("answer", "Chưa tạo được câu trả lời."),
        result.get("sources", []) or [],
    )
