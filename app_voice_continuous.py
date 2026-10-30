from __future__ import annotations

import base64
import json
import queue
import uuid

import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from nlp.intent_handler import get_article_content, handle_intent
from rag.pipeline_15k import load_vector_database
from voice.speech_to_text import (
    ContinuousSpeechCollector,
    SpeechRecognitionError,
    transcribe_audio,
)
from voice.text_to_speech import (
    VOICE_OPTIONS,
    TextToSpeechError,
    synthesize_text_chunks,
)


# =========================================================
# 1. CAU HINH
# =========================================================
st.set_page_config(
    page_title="Trợ lý đọc báo thông minh",
    page_icon="📰",
    layout="wide",
)


@st.cache_resource
def initialize_system():
    index, metadata = load_vector_database()
    return index.ntotal, len(metadata)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_tts(text: str, voice: str):
    # Tạo MP3 cho câu trả lời bất kỳ, không phụ thuộc nội dung có phải bài báo.
    return synthesize_text_chunks(text, voice=voice)


def render_audio_player(audio_chunks: list[bytes], key: str, autoplay: bool = False):
    """Render one player; automatically continues through all TTS chunks."""
    if not audio_chunks:
        return

    encoded = [
        base64.b64encode(part).decode("ascii")
        for part in audio_chunks
        if part
    ]
    if not encoded:
        return

    player_id = "tts_" + re_safe_id(key)
    payload = json.dumps(encoded)
    auto = "true" if autoplay else "false"

    html = f"""
    <div style="font-family: sans-serif; width:100%;">
      <audio id="{player_id}" controls style="width:100%;"></audio>
      <div id="{player_id}_status"
           style="font-size:12px;color:#777;margin-top:4px;">
        {"Đang thử phát tự động. Nếu trình duyệt chặn, hãy nhấn nút Play." if autoplay
         else "Nhấn Play để nghe lại phản hồi."}
      </div>
    </div>
    <script>
    (() => {{
      const chunks = {payload};
      const player = document.getElementById("{player_id}");
      const status = document.getElementById("{player_id}_status");
      let index = 0;
      function playPart(i) {{
        if (i >= chunks.length) {{
          status.textContent = "Đã đọc xong phản hồi.";
          return;
        }}
        index = i;
        player.src = "data:audio/mp3;base64," + chunks[index];
        player.load();
        const p = player.play();
        if (p && p.catch) {{
          p.catch(() => {{
            status.textContent = "Trình duyệt chặn tự phát. Nhấn Play để nghe; các phần sau sẽ tự nối tiếp.";
          }});
        }}
      }}
      player.addEventListener("ended", () => {{
        if (index + 1 < chunks.length) playPart(index + 1);
        else status.textContent = "Đã đọc xong phản hồi.";
      }});
      if ({auto}) playPart(0);
      else player.src = "data:audio/mp3;base64," + chunks[0];
    }})();
    </script>
    """
    components.html(html, height=78, scrolling=False)


def re_safe_id(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum())[:36] or uuid.uuid4().hex


def process_question(question: str, source: str = "text"):
    """Run the same chatbot pipeline for typed and spoken questions."""
    question = (question or "").strip()
    if not question:
        return None

    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": question,
        "input_source": source,
    }
    st.session_state.messages.append(user_message)

    try:
        with st.spinner("Đang xử lý câu hỏi..."):
            answer, sources = handle_intent(question)
        answer = (answer or "Hệ thống chưa tạo được câu trả lời.").strip()
        audio_chunks = []
        tts_error = None
        try:
            with st.spinner("Đang tạo giọng đọc cho phản hồi..."):
                audio_chunks = cached_tts(answer, st.session_state.selected_voice)
        except Exception as exc:
            tts_error = str(exc)
            print("Lỗi TTS:", repr(exc))

        assistant_message = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": answer,
            "sources": sources or [],
            "audio_chunks": audio_chunks,
            "tts_error": tts_error,
            "autoplay_pending": bool(audio_chunks),
        }
    except Exception as exc:
        error_text = f"Đã xảy ra lỗi khi xử lý yêu cầu: {exc}"
        audio_chunks = []
        try:
            audio_chunks = cached_tts(error_text, st.session_state.selected_voice)
        except Exception:
            pass
        assistant_message = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": error_text,
            "sources": [],
            "audio_chunks": audio_chunks,
            "tts_error": None,
            "autoplay_pending": bool(audio_chunks),
        }

    st.session_state.messages.append(assistant_message)
    return assistant_message


def render_sources(sources, message_key):
    if not sources:
        return

    st.markdown("#### Nguồn bài báo")
    for i, source in enumerate(sources, start=1):
        title = source.get("title") or "Không có tiêu đề"
        url = source.get("url") or ""
        article_id = source.get("article_id")
        content_key = f"article_content_{message_key}_{i}"

        with st.expander(f"[Nguồn {i}] {title}"):
            st.caption(f"Báo: {source.get('source') or 'Không rõ nguồn'}")
            if url.startswith(("http://", "https://")):
                st.link_button("🌐 Mở bài báo gốc", url)

            similarity = source.get("similarity")
            if similarity is not None:
                try:
                    st.caption(f"Điểm tương đồng: {float(similarity):.4f}")
                except (TypeError, ValueError):
                    pass

            if st.button("📖 Xem nội dung bài báo", key=f"view_{message_key}_{i}"):
                content = get_article_content(article_id=article_id, url=url)
                if content is None:
                    st.session_state.pop(content_key, None)
                    st.warning("Không tìm thấy toàn văn bài báo trong database.")
                else:
                    st.session_state[content_key] = content

            content = st.session_state.get(content_key)
            if content:
                if content.get("title"):
                    st.markdown(f"**{content['title']}**")
                if content.get("sapo"):
                    st.write(content["sapo"])
                body = (content.get("body") or "").strip()
                if body:
                    st.text(body)
                    article_audio_key = f"article_audio_{message_key}_{i}"
                    if st.button("🔊 Tạo giọng đọc toàn văn", key=f"article_tts_{message_key}_{i}"):
                        try:
                            st.session_state[article_audio_key] = cached_tts(
                                body, st.session_state.selected_voice
                            )
                        except TextToSpeechError as exc:
                            st.error(str(exc))
                    if st.session_state.get(article_audio_key):
                        render_audio_player(
                            st.session_state[article_audio_key],
                            article_audio_key,
                            autoplay=False,
                        )


# =========================================================
# 2. KHOI TAO HE THONG
# =========================================================
st.title("📰 Trợ lý đọc báo thông minh")
st.caption(
    "Bạn có thể nhập câu hỏi hoặc bật micro để nói nhiều lượt. "
    "Bot sẽ tự đọc câu trả lời hiển thị trên giao diện."
)

try:
    total_vectors, total_metadata = initialize_system()
    if total_vectors != total_metadata:
        st.error("Số vector không khớp số metadata.")
        st.stop()
except Exception as exc:
    st.error(f"Không thể khởi tạo hệ thống: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_queue" not in st.session_state:
    st.session_state.voice_queue = queue.Queue(maxsize=8)
if "voice_collector" not in st.session_state:
    st.session_state.voice_collector = ContinuousSpeechCollector(
        output_queue=st.session_state.voice_queue,
        silence_seconds=1.15,
        energy_threshold=350,
        min_speech_seconds=0.30,
        max_utterance_seconds=20,
    )
if "voice_notice" not in st.session_state:
    st.session_state.voice_notice = None
if "selected_voice" not in st.session_state:
    st.session_state.selected_voice = list(VOICE_OPTIONS.values())[0]


# =========================================================
# 3. SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Cấu hình")
    voice_label = st.selectbox(
        "Giọng đọc phản hồi",
        options=list(VOICE_OPTIONS.keys()),
        index=list(VOICE_OPTIONS.values()).index(st.session_state.selected_voice),
    )
    st.session_state.selected_voice = VOICE_OPTIONS[voice_label]

    st.metric("Tổng số vector", f"{total_vectors:,}")
    st.caption("Embedding: PhoBERT | Vector DB: FAISS")
    st.caption("LLM: Qwen2.5-1.5B | Tóm tắt: ViT5")

    if st.button("Xóa lịch sử hội thoại"):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# 4. HIEN THI HOI THOAI + TU DONG DOC CAC PHAN HOI
# =========================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            audio_chunks = message.get("audio_chunks") or []
            autoplay = bool(message.get("autoplay_pending", False))
            if autoplay:
                message["autoplay_pending"] = False
            if audio_chunks:
                render_audio_player(
                    audio_chunks,
                    key=f"reply_{message['id']}",
                    autoplay=autoplay,
                )
            elif message.get("tts_error"):
                st.caption(f"Chưa tạo được giọng đọc: {message['tts_error']}")

            render_sources(message.get("sources", []), message["id"])


# =========================================================
# 5. NHAP VAN BAN
# =========================================================
typed_question = st.chat_input("Nhập câu hỏi hoặc yêu cầu...")
if typed_question:
    process_question(typed_question, source="text")
    st.rerun()


# =========================================================
# 6. MICRO NGHE LIEN TUC
# =========================================================
st.divider()
st.subheader("🎙️ Nói chuyện với trợ lý")
st.write(
    "Nhấn **START** một lần để bật micro. Nói một câu rồi dừng khoảng "
    "1,15 giây; hệ thống tự nhận diện và trả lời. Micro tiếp tục nghe "
    "cho đến khi bạn nhấn **STOP**."
)

energy_threshold = st.slider(
    "Độ nhạy micro (giảm ngưỡng nếu nói nhỏ)",
    min_value=100,
    max_value=1600,
    value=350,
    step=50,
    help="Nếu không nhận giọng nói, thử giảm xuống 150–250. Nếu tiếng ồn bị nhận thành lời, tăng ngưỡng.",
)
silence_seconds = st.slider(
    "Khoảng lặng để chốt câu",
    min_value=0.6,
    max_value=2.5,
    value=1.15,
    step=0.05,
)
collector = st.session_state.voice_collector
collector.energy_threshold = energy_threshold
collector.silence_seconds = silence_seconds

ctx = webrtc_streamer(
    key="continuous-vietnamese-microphone",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=collector,
    media_stream_constraints={
        "audio": {
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
        },
        "video": False,
    },
    async_processing=True,
)

if ctx.state.playing:
    st.success("🟢 Micro đang hoạt động. Bạn có thể nói nhiều câu liên tiếp.")
else:
    st.info("Micro đang tắt. Nhấn START và cho phép trình duyệt sử dụng micro.")

if st.session_state.voice_notice:
    st.warning(st.session_state.voice_notice)
    st.session_state.voice_notice = None


# Fragment polls the audio queue without blocking Streamlit's main script.
# When it gets an utterance, it runs the same handler as text input and asks
# Streamlit for a full rerun so the new user/assistant turns appear in history.
@st.fragment(run_every="500ms")
def poll_continuous_voice():
    if not ctx.state.playing:
        return

    audio_queue = st.session_state.voice_queue
    try:
        wav_bytes = audio_queue.get_nowait()
    except queue.Empty:
        return

    try:
        with st.spinner("Đã phát hiện bạn ngừng nói. Đang nhận diện..."):
            recognized_text = transcribe_audio(wav_bytes, language="vi-VN")
        if recognized_text:
            st.session_state.voice_notice = f"Đã nhận diện: {recognized_text}"
            process_question(recognized_text, source="voice")
            st.rerun(scope="app")
    except SpeechRecognitionError as exc:
        st.session_state.voice_notice = str(exc)
        st.rerun(scope="app")
    except Exception as exc:
        st.session_state.voice_notice = f"Lỗi Voice: {exc}"
        print("Voice pipeline error:", repr(exc))
        st.rerun(scope="app")


if ctx.state.playing:
    poll_continuous_voice()
