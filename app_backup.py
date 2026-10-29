import streamlit as st
from database.database_15k import SessionLocal, Article
from rag.pipeline_15k import ask, load_vector_database
import uuid
from nlp.text_classifier import predict_category
from nlp.intent_classifier import predict_intent
# =========================================
# 1. CAU HINH GIAO DIEN
# =========================================
def get_article_content(article_id=None, url=None):
    """
    Lấy toàn văn bài báo từ news_15k.db.
    Ưu tiên ID gốc, nếu không có thì tìm bằng URL.
    """
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

def render_sources(sources, message_key,expanded=False):
    if not sources:
        return

    st.markdown("### Nguồn bài báo")

    for i, source in enumerate(sources, start=1):
        title = source.get("title") or "Không có tiêu đề"
        url = source.get("url") or ""
        newspaper = source.get("source") or "Không rõ nguồn"
        article_id = source.get("article_id")

        # Khóa duy nhất cho từng nguồn trong từng tin nhắn
        view_key = f"view_article_{message_key}_{i}"
        content_key = f"article_content_{message_key}_{i}"

        with st.expander(f"[Nguồn {i}] {title}",expanded=expanded):
            st.caption(f"Báo: {newspaper}")

            if url.startswith(("https://", "http://")):
                st.link_button(
                    "🌐 Mở bài báo gốc",
                    url,
                    key=f"open_article_{message_key}_{i}",
                )

            similarity = source.get("similarity")

            if similarity is not None:
                try:
                    st.caption(
                        f"Điểm tương đồng: {float(similarity):.4f}"
                    )
                except (TypeError, ValueError):
                    pass

            if st.button(
                "📖 Xem nội dung bài báo",
                key=view_key,
            ):
                content = get_article_content(
                    article_id=article_id,
                    url=url,
                )

                if content is None:
                    st.session_state.pop(content_key, None)
                    st.warning(
                        "Không tìm thấy toàn văn bài báo trong database."
                    )
                else:
                    st.session_state[content_key] = content

            content = st.session_state.get(content_key)

            if content:
                st.divider()

                if content.get("title"):
                    st.markdown(f"**{content['title']}**")

                if content.get("sapo"):
                    st.markdown("**Tóm tắt bài báo**")
                    st.write(content["sapo"])

                if content.get("body"):
                    st.markdown("**Toàn văn bài báo**")
                    st.text(content["body"])
                else:
                    st.warning("Bài báo chưa có nội dung toàn văn.")

st.set_page_config(
    page_title="Trợ lý đọc báo thông minh",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Trợ lý đọc báo thông minh")
st.caption(
    "Hệ thống hỏi đáp báo chí ứng dụng NLP và RAG"
)


# =========================================
# 2. NAP VECTOR DATABASE
# =========================================

@st.cache_resource
def initialize_system():
    index, metadata = load_vector_database()
    return index.ntotal, len(metadata)


try:
    total_vectors, total_metadata = initialize_system()

    if total_vectors != total_metadata:
        st.error("Số vector không khớp số metadata.")
        st.stop()

except Exception as error:
    st.error(f"Không thể khởi tạo hệ thống: {error}")
    st.stop()


# =========================================
# 3. THANH BEN
# =========================================

with st.sidebar:
    st.header("Thông tin hệ thống")

    st.write("**Embedding:** PhoBERT")
    st.write("**Vector store:** FAISS")
    st.write("**LLM:** Qwen2.5-1.5B")
    st.write("**Dữ liệu:** 15.000 bài báo")

    st.metric("Tổng số vector", f"{total_vectors:,}")

    if st.button("Xóa lịch sử hội thoại"):
        st.session_state.messages = []
        st.rerun()

# =========================================
# 4. LICH SU HOI THOAI
# =========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if "id" not in message:
        message["id"] = str(uuid.uuid4())
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message_key=message["id"],
            )
            # sources = message.get("sources", [])
            # if sources:
            #     with st.expander(
            #         f"Nguồn bài báo ({len(sources)})"
            #     ):
            #         for source in sources:
            #             title = source.get(
            #                 "title",
            #                 "Không có tiêu đề",
            #             )

            #             url = source.get("url", "")
            #             newspaper = source.get(
            #                 "source",
            #                 "Không rõ nguồn",
            #             )

            #             st.markdown(f"**{title}**")
            #             st.caption(f"Báo: {newspaper}")

            #             if url:
            #                 st.markdown(f"[Đọc bài báo]({url})")

            #             similarity = source.get("similarity")

            #             if similarity is not None:
            #                 st.caption(
            #                     f"Điểm tương đồng: {similarity:.4f}"
            #                 )

            #             st.divider()


# =========================================
# 5. BO DIEU PHOI Y DINH
# =========================================

INTRODUCTION_RESPONSE = """Tôi là Trợ lý đọc báo thông minh, hỗ trợ bạn khai thác
thông tin từ các bài báo tiếng Việt.

Các chức năng chính:
- Tìm kiếm bài báo theo nội dung bằng PhoBERT và FAISS.
- Tóm tắt bài báo bằng mô hình ViT5.
- Hỏi đáp dựa trên bài báo bằng RAG và Qwen.
- Xem thông tin nguồn và toàn văn bài báo.
"""


def process_question(question: str):
    """Điều phối yêu cầu theo ý định dự đoán.

    Hiện tại greeting và introduction được xử lý trực tiếp.
    Những ý định khác tạm thời tiếp tục đi qua pipeline RAG hiện có.
    """
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

    # Các ý định còn lại tạm thời sử dụng RAG như luồng cũ.
    result = ask(question)
    return result.get("answer", "Chưa tạo được câu trả lời."), result.get("sources", [])


# =========================================
# 6. O HOI DAP
# =========================================

question = st.chat_input("Nhập câu hỏi về các bài báo...")

if question:
    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": question,
    }
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang xác định ý định và xử lý yêu cầu..."):
            try:
                answer, sources = process_question(question)

                assistant_message = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }

                # Chỉ lưu câu trả lời một lần.
                st.session_state.messages.append(assistant_message)
                st.markdown(answer)

                render_sources(
                    sources,
                    message_key=assistant_message["id"],
                    expanded=True,
                )

            except Exception as error:
                error_message = (
                    "Đã xảy ra lỗi khi xử lý yêu cầu. "
                    "Bạn hãy kiểm tra terminal để xem chi tiết."
                )
                st.error(error_message)
                print("Lỗi xử lý yêu cầu:", repr(error))

                st.session_state.messages.append({
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                })
