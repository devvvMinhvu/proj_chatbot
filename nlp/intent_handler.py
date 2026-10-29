"""Dieu phoi y dinh cho tro ly doc bao.

Module nay tach logic xu ly intent khoi giao dien Streamlit.
Cac ham tra ve (answer, sources) de app.py hien thi ket qua.
"""

from functools import lru_cache
from typing import Any

from database.database_15k import SessionLocal, Article
from nlp.intent_classifier import predict_intent
from rag.pipeline_15k import ask


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
    """Nap ViT5 khi can, tranh nap mo hinh neu khong tom tat."""
    from processing.summerizer import summarize
    return summarize


def get_article_content(article_id: str | None = None,
                        url: str | None = None) -> dict[str, str] | None:
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


def _find_source_with_body(sources: list[dict[str, Any]]):
    """Tra nguon co toan van trong database theo diem tuong dong."""
    for source in _rank_sources(sources):
        content = get_article_content(
            article_id=source.get("article_id"),
            url=source.get("url"),
        )
        if content and content.get("body", "").strip():
            return source, content
    return None, None


def _format_search_results(sources: list[dict[str, Any]]) -> str:
    """Tao danh sach ket qua tim kiem tu cac nguon RAG tra ve."""
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


def handle_intent(question: str) -> tuple[str, list[dict[str, Any]]]:
    """Du doan intent va dieu phoi sang chuc nang tuong ung.

    Returns:
        (answer, sources): ket qua text va danh sach nguon cho app.py.
    """
    if not question or not question.strip():
        return "Bạn hãy nhập yêu cầu cần xử lý.", []

    intent = predict_intent(question)
    print(f"[INTENT] {intent} | {question}")

    # 1. Chao hoi: khong goi RAG.
    if intent == "greeting":
        return (
            "Xin chào! Tôi là Trợ lý đọc báo thông minh. "
            "Tôi có thể giúp bạn tìm kiếm, tóm tắt và hỏi đáp "
            "về các bài báo tiếng Việt."
        ), []

    # 2. Gioi thieu: khong goi RAG.
    if intent == "introduction":
        return INTRODUCTION_RESPONSE, []

    # Chuc nang tin moi chua duoc ket noi voi truong ngay dang da xac minh.
    # Tra loi minh bach thay vi dung RAG de doan tin nao la moi nhat.
    if intent == "latest_news":
        return (
            "Chức năng xem tin mới nhất chưa được kết nối với trường ngày đăng "
            "đã xác minh trong cơ sở dữ liệu. Hiện tôi chưa thể sắp xếp tin "
            "theo thời gian một cách đáng tin cậy."
        ), []

    # Cac y dinh can tim bai bao truoc tien dung pipeline retrieval/RAG hien co.
    result = ask(question)
    answer = result.get("answer", "Chưa tạo được câu trả lời.")
    sources = result.get("sources", []) or []

    # 3. Tom tat: dung retrieval de tim bai, ViT5 de sinh tom tat.
    if intent == "summarization":
        source, content = _find_source_with_body(sources)
        if content is None:
            return (
                "Tôi tìm được nguồn liên quan nhưng chưa lấy được toàn văn "
                "trong cơ sở dữ liệu để tóm tắt. Bạn thử nêu rõ tiêu đề bài báo "
                "hoặc kiểm tra trường `body` trong database."
            ), []

        summary = _load_summarizer()(content["body"])
        if not summary or not summary.strip():
            return "Mô hình chưa tạo được bản tóm tắt cho bài báo này.", []

        title = content.get("title") or source.get("title") or "Bài báo"
        selected_source = dict(source)
        selected_source["title"] = title
        return f"## Tóm tắt: {title}\n\n{summary.strip()}", [selected_source]

    # 4. Chi tiet: tim bai phu hop roi tra ve toan van.
    if intent == "article_detail":
        source, content = _find_source_with_body(sources)
        if content is None:
            return (
                "Tôi chưa tìm được toàn văn phù hợp trong cơ sở dữ liệu. "
                "Bạn hãy nêu rõ hơn tiêu đề hoặc chủ đề bài báo."
            ), []

        title = content.get("title") or source.get("title") or "Bài báo"
        selected_source = dict(source)
        selected_source["title"] = title
        body = content["body"].strip()
        return f"## {title}\n\n{body}", [selected_source]

    # 5. Tim kiem: hien thi danh sach bai bao, khong dua ra mot cau tra loi tong hop.
    if intent == "semantic_search":
        return _format_search_results(sources), _rank_sources(sources)

    # 6. Hoi dap: giu nguyen cau tra loi RAG/Qwen va cac nguon.
    if intent == "question_answering":
        return answer, sources

    # Du phong cho nhan la moi hoac khong mong doi.
    return answer, sources
