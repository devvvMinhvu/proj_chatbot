import uuid
import streamlit as st

from rag.pipeline_15k import load_vector_database
from nlp.intent_handler import get_article_content, handle_intent
from voice.speech_to_text import transcribe_audio, SpeechRecognitionError
from voice.text_to_speech import (
    VOICE_OPTIONS,
    synthesize_text_chunks,
    TextToSpeechError,
)


# =========================================
# 1. CAU HINH VA CAC HAM HO TRO
# =========================================

st.set_page_config(
    page_title="Trợ lý đọc báo thông minh",
    page_icon="📰",
    layout="wide",
)


@st.cache_resource
def initialize_system():
    index, metadata = load_vector_database()
    return index.ntotal, len(metadata)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_speech_chunks(text: str, voice: str):
    """Cache audio so the same article/voice does not need synthesis again."""
    return synthesize_text_chunks(text, voice=voice)


def render_sources(sources, message_key, expanded=False):
    if not sources:
        return

    st.markdown("### Nguồn bài báo")

    for i, source in enumerate(sources, start=1):
        title = source.get("title") or "Không có tiêu đề"
        url = source.get("url") or ""
        newspaper = source.get("source") or "Không rõ nguồn"
        article_id = source.get("article_id")

        view_key = f"view_article_{message_key}_{i}"
        content_key = f"article_content_{message_key}_{i}"

        with st.expander(f"[Nguồn {i}] {title}", expanded=expanded):
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
                    st.caption(f"Điểm tương đồng: {float(similarity):.4f}")
                except (TypeError, ValueError):
                    pass

            if st.button("📖 Xem nội dung bài báo", key=view_key):
                content = get_article_content(
                    article_id=article_id,
                    url=url,
                )
                if content is None:
                    st.session_state.pop(content_key, None)
                    st.warning("Không tìm thấy toàn văn bài báo trong database.")
                else:
                    st.session_state[content_key] = content

            content = st.session_state.get(content_key)
            if not content:
                continue

            st.divider()
            if content.get("title"):
                st.markdown(f"**{content['title']}**")

            if content.get("sapo"):
                st.markdown("**Tóm tắt bài báo**")
                st.write(content["sapo"])

            body = (content.get("body") or "").strip()
            if not body:
                st.warning("Bài báo chưa có nội dung toàn văn.")
                continue

            st.markdown("**Toàn văn bài báo**")
            st.text(body)

            # Đọc báo: Edge TTS tạo MP3, trình duyệt phát qua st.audio().
            # Dùng container thay vì expander lồng nhau (Streamlit không hỗ trợ
            # nested expanders).
            with st.container(border=True):
                st.markdown("**🔊 Đọc bài báo thành tiếng**")
                voice_label = st.selectbox(
                    "Chọn giọng đọc",
                    options=list(VOICE_OPTIONS.keys()),
                    index=0,
                    key=f"tts_voice_{message_key}_{i}",
                )
                voice_id = VOICE_OPTIONS[voice_label]
                audio_key = f"tts_audio_{message_key}_{i}_{voice_id}"

                if st.button(
                    "Tạo giọng đọc",
                    key=f"tts_generate_{message_key}_{i}_{voice_id}",
                ):
                    try:
                        with st.spinner("Đang tạo âm thanh bài báo..."):
                            audio_chunks = cached_speech_chunks(body, voice_id)
                        st.session_state[audio_key] = audio_chunks
                    except TextToSpeechError as error:
                        st.error(str(error))
                    except Exception as error:
                        st.error("Không thể tạo giọng đọc. Hãy kiểm tra terminal.")
                        print("Lỗi TTS:", repr(error))

                audio_chunks = st.session_state.get(audio_key, [])
                if audio_chunks:
                    st.success("Đã tạo giọng đọc. Phát lần lượt các phần bên dưới.")
                    if len(audio_chunks) > 1:
                        st.caption(
                            f"Bài báo dài, được chia thành {len(audio_chunks)} phần "
                            "để tránh giới hạn độ dài khi tổng hợp giọng nói."
                        )
                    for part_number, audio_bytes in enumerate(audio_chunks, start=1):
                        if len(audio_chunks) > 1:
                            st.caption(f"Phần {part_number}/{len(audio_chunks)}")
                        st.audio(audio_bytes, format="audio/mp3")


# =========================================
# 2. KHOI TAO HE THONG
# =========================================

st.title("📰 Trợ lý đọc báo thông minh")
st.caption("Hệ thống hỏi đáp báo chí ứng dụng NLP và RAG")

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

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Thông tin hệ thống")
    st.write("**Embedding:** PhoBERT")
    st.write("**Vector store:** FAISS")
    st.write("**LLM:** Qwen2.5-1.5B")
    st.write("**Tóm tắt:** ViT5")
    st.write("**Đọc báo:** Edge TTS")
    st.metric("Tổng số vector", f"{total_vectors:,}")

    if st.button("Xóa lịch sử hội thoại"):
        st.session_state.messages = []
        st.rerun()


# =========================================
# 4. HIEN THI LICH SU HOI THOAI
# =========================================

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


# =========================================
# 5. NHAN GIONG NOI VA O CHAT
# =========================================

# st.audio_input ghi âm từ micro của TRÌNH DUYỆT người dùng.
# Không dùng sr.Microphone(), vì Streamlit chạy Python trên máy chủ.
if "voice_input_version" not in st.session_state:
    st.session_state.voice_input_version = 0

recorded_audio = None
if hasattr(st, "audio_input"):
    recorded_audio = st.audio_input(
        "🎙️ Ghi âm câu hỏi bằng tiếng Việt",
        key=f"voice_input_{st.session_state.voice_input_version}",
        help="Nhấn ghi âm, nói yêu cầu, rồi dừng để gửi nội dung.",
    )
else:
    st.warning(
        "Phiên bản Streamlit hiện tại chưa hỗ trợ st.audio_input(). "
        "Hãy nâng cấp bằng: python -m pip install -U streamlit"
    )

typed_question = st.chat_input("Hoặc nhập câu hỏi về các bài báo...")
question = typed_question

if recorded_audio is not None:
    # Mỗi bản ghi được dùng đúng một lần; tăng version để tạo widget trống mới.
    st.session_state.voice_input_version += 1

    if typed_question:
        st.info("Bạn đã nhập cả văn bản và ghi âm. Hệ thống ưu tiên câu nhập bằng văn bản.")
    else:
        try:
            with st.spinner("Đang nhận dạng giọng nói tiếng Việt..."):
                question = transcribe_audio(recorded_audio.getvalue())
            st.caption(f"Nội dung nhận dạng: {question}")
        except SpeechRecognitionError as error:
            st.warning(str(error))
            question = None
        except Exception as error:
            st.error("Không thể nhận dạng bản ghi âm. Hãy kiểm tra terminal.")
            print("Lỗi speech-to-text:", repr(error))
            question = None


# =========================================
# 6. XU LY YEU CAU BANG BO DIEU PHOI INTENT
# =========================================

if question and question.strip():
    question = question.strip()

    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": question,
    }
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang xác định ý định, tìm bài báo và xử lý yêu cầu..."):
            try:
                answer, sources = handle_intent(question)

                assistant_message = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
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
