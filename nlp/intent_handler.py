"""Bo dieu phoi y dinh cho tro ly doc bao.

- greeting/introduction: tra loi truc tiep.
- semantic_search: PhoBERT + FAISS, khong goi Qwen.
- summarization: PhoBERT + FAISS -> SQLite -> ViT5, khong goi Qwen.
- article_detail: PhoBERT + FAISS -> SQLite, khong goi Qwen.
- question_answering: dung pipeline RAG + Qwen hien co.
- latest_news: truy van SQLite theo published_at; neu con bai thieu ngay dang, canh bao ve do phu du lieu.

Cac ham dieu phoi chinh tra ve (answer, sources) cho app.py.
"""

from functools import lru_cache
from typing import Any

from database.database_15k import SessionLocal, Article
from nlp.intent_classifier import predict_intent
from rag.pipeline_15k import ask, retrieve, group_chunks_by_article


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
