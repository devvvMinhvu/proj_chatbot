"""Unified Streamlit interface for the Vietnamese News Assistant."""

from __future__ import annotations

import base64
import io
import queue
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import av
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from database.repository import get_article_content, get_article_count, get_latest_articles
from nlp.intent_handler import _format_latest_news, handle_intent
from voice.reading_text import build_reading_text
from voice.speech_to_text import SpeechRecognitionError, transcribe_audio
from voice.text_to_speech import (
    DEFAULT_VOICE,
    FALLBACK_VOICE,
    VOICE_OPTIONS,
    TextToSpeechError,
    synthesize_text_chunks,
)
from voice.tts_background import (
    cancel_tts_task,
    poll_tts_events,
    release_tts_task,
    submit_tts_task,
)

# ============================================================
# 1. CẤU HÌNH TRANG & HẰNG SỐ
# ============================================================
st.set_page_config(
    page_title="Trợ lý đọc báo thông minh",
    page_icon="📰",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"] h1:first-child,
    [data-testid="stChatMessageContent"] h2:first-child,
    [data-testid="stChatMessageContent"] h3:first-child,
    [data-testid="stChatMessageContent"] h4:first-child,
    [data-testid="stChatMessageContent"] p:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    [data-testid="stChatMessageContent"] h2:first-child,
    [data-testid="stChatMessageContent"] h3:first-child {
        font-size: 1.25rem !important;
        line-height: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SOURCE_LABELS = {
    "Tất cả báo": None,
    "VnExpress": "vnexpress",
    "Tuổi Trẻ": "tuoitre",
    "Tiền Phong": "tienphong",
}
SOURCE_DISPLAY = {
    "vnexpress": "VnExpress",
    "tuoitre": "Tuổi Trẻ",
    "tienphong": "Tiền Phong",
}
CATEGORY_LABELS = {
    "Tất cả chuyên mục": None,
    "Thời sự": "news",
    "Kinh tế": "economy",
    "Công nghệ": "tech",
    "Giải trí": "entertainment",
    "Thể thao": "sports",
}
CATEGORY_DISPLAY = {
    "news": "Thời sự",
    "economy": "Kinh tế",
    "tech": "Công nghệ",
    "entertainment": "Giải trí",
    "sports": "Thể thao",
}

MODES = (
    "⌨️ Văn bản",
    "🎙️ Nhấn để nói",
    "🗣️ Hội thoại rảnh tay",
)

SAMPLE_RATE = 16000
VAD_RMS_THRESHOLD = 450
END_SILENCE_SECONDS = 0.85
MIN_SPEECH_SECONDS = 0.35
MAX_UTTERANCE_SECONDS = 15.0
PRE_ROLL_FRAMES = 8
VOICE_POLL_INTERVAL = 0.25


@st.cache_data
def initialize_system() -> int:
    """Đếm bài báo trong SQLite; không nạp PhoBERT/FAISS khi vừa mở trang."""
    return int(get_article_count())


@st.cache_data(show_spinner=False, max_entries=32)
def cached_article_summary(article_id: str, body: str) -> str:
    """Cache bản tóm tắt ViT5 theo article_id để không sinh lại khi mở lại hộp thoại."""
    from processing.summarizer import summarize

    return summarize(body)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_article_audio(text: str, voice: str) -> list[bytes]:
    """Cache MP3 cho đọc bài báo trong hộp thoại."""
    return synthesize_text_chunks(text, voice=voice)


@st.cache_resource
def get_tts_audio_queue_component():
    """Load the browser-side player that queues MP3 segments sequentially."""
    component_dir = Path(__file__).resolve().parent / "voice" / "audio_queue_component"
    if not (component_dir / "index.html").is_file():
        raise FileNotFoundError(
            f"Không tìm thấy audio queue component: {component_dir / 'index.html'}"
        )
    return components.declare_component(
        "news_assistant_tts_audio_queue",
        path=str(component_dir),
    )


# ============================================================
# 2. HỘP THOẠI ĐỌC & TÓM TẮT BÀI BÁO
# ============================================================
@st.dialog("📖 Chi tiết bài báo", width="large")
def show_article_dialog(source_item: dict[str, Any], mode: str = "read") -> None:
    content = get_article_content(
        article_id=source_item.get("article_id"),
        url=source_item.get("url"),
    )
    if content is None:
        st.warning("Không tìm thấy toàn văn bài báo trong cơ sở dữ liệu.")
        return

    title = content.get("title") or source_item.get("title") or "Không có tiêu đề"
    src_slug = content.get("source") or source_item.get("source") or ""
    cat_slug = content.get("category") or source_item.get("category") or ""
    published = (
        content.get("published_at_display")
        or source_item.get("published_at_display")
        or content.get("published_at")
        or source_item.get("published_at")
        or ""
    )
    url = content.get("url") or source_item.get("url") or ""

    st.subheader(title)

    meta_parts = [f"**Báo:** {SOURCE_DISPLAY.get(src_slug, src_slug or 'Không rõ')}"]
    if cat_slug:
        meta_parts.append(f"**Chuyên mục:** {CATEGORY_DISPLAY.get(cat_slug, cat_slug)}")
    if published:
        meta_parts.append(f"**Ngày đăng:** {published}")
    st.caption("  |  ".join(meta_parts))

    if url.startswith(("https://", "http://")):
        st.link_button("🌐 Mở bài báo gốc", url)

    sapo = (content.get("sapo") or "").strip()
    body = (content.get("body") or "").strip()

    if mode == "summarize":
        if sapo:
            st.markdown(f"**Sapo:** {sapo}")
        if not body:
            st.warning("Bài báo chưa có toàn văn để tóm tắt.")
            return
        with st.spinner("Đang tóm tắt bằng mô hình ViT5..."):
            summary = cached_article_summary(
                str(content.get("article_id") or url or title),
                body,
            )
        if summary and summary.strip():
            st.info(f"**Tóm tắt (ViT5):**\n\n{summary.strip()}")
        else:
            st.warning("Mô hình ViT5 chưa tạo được bản tóm tắt.")

        with st.expander("📄 Xem toàn văn bài báo"):
            for paragraph in body.split("\n\n"):
                if paragraph.strip():
                    st.write(paragraph.strip())
    else:
        if sapo:
            st.info(sapo)
        if body:
            for paragraph in body.split("\n\n"):
                if paragraph.strip():
                    st.write(paragraph.strip())
        else:
            st.warning("Bài báo chưa có nội dung toàn văn.")

    if body:
        st.divider()
        audio_key = f"dialog_tts_{content.get('article_id') or url}_{st.session_state.selected_voice}"
        if st.button("🔊 Nghe toàn văn bài báo", key=f"btn_{audio_key}"):
            try:
                with st.spinner("Đang tạo giọng đọc..."):
                    st.session_state[audio_key] = cached_article_audio(
                        body,
                        st.session_state.selected_voice,
                    )
            except TextToSpeechError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Không thể tạo giọng đọc: {exc}")

        chunks = st.session_state.get(audio_key, [])
        for idx, audio_bytes in enumerate(chunks, start=1):
            if len(chunks) > 1:
                st.caption(f"Phần {idx}/{len(chunks)}")
            st.audio(audio_bytes, format="audio/mp3")


# ============================================================
# 3. HIỂN THỊ THẺ BÀI BÁO & TRẠNG THÁI
# ============================================================
def _render_single_source_card(
    source: dict[str, Any],
    index: int,
    message_key: str,
) -> None:
    title = source.get("title") or "Không có tiêu đề"
    src_slug = source.get("source") or ""
    cat_slug = source.get("category") or ""
    published = source.get("published_at_display") or source.get("published_at") or ""
    url = source.get("url") or ""
    sapo = (source.get("sapo") or "").strip()
    similarity = source.get("similarity")

    with st.container(border=True):
        st.markdown(f"**{index}. {title}**")

        meta_bits = [f"Báo: {SOURCE_DISPLAY.get(src_slug, src_slug or 'Không rõ')}"]
        if cat_slug:
            meta_bits.append(f"Chuyên mục: {CATEGORY_DISPLAY.get(cat_slug, cat_slug)}")
        if published:
            meta_bits.append(f"Ngày đăng: {published}")
        if similarity is not None:
            try:
                meta_bits.append(f"Độ phù hợp: {float(similarity):.2f}")
            except (TypeError, ValueError):
                pass
        st.caption("  •  ".join(meta_bits))

        if sapo:
            preview = sapo if len(sapo) <= 200 else sapo[:197].rstrip() + "..."
            st.write(preview)

        col1, col2, col3, col4 = st.columns([1.15, 1.05, 1.25, 1.15])
        with col1:
            if st.button("📖 Đọc bài", key=f"read_{message_key}_{index}", use_container_width=True):
                show_article_dialog(source, mode="read")
        with col2:
            if st.button("✨ Tóm tắt", key=f"sum_{message_key}_{index}", use_container_width=True):
                show_article_dialog(source, mode="summarize")
        with col3:
            if st.button("💬 Hỏi bài này", key=f"pin_{message_key}_{index}", use_container_width=True):
                st.session_state.pinned_article = source
                st.rerun()
        with col4:
            if url.startswith(("https://", "http://")):
                st.link_button("🌐 Bài gốc", url, use_container_width=True)


def render_sources(sources: list[dict[str, Any]], message_key: str) -> None:
    if not sources:
        return

    st.markdown("#### Nguồn bài báo liên quan")
    primary = sources[:5]
    extra = sources[5:]

    for idx, source in enumerate(primary, start=1):
        _render_single_source_card(source, idx, message_key)

    if extra:
        with st.expander(f"Xem thêm {len(extra)} bài báo khác"):
            for idx, source in enumerate(extra, start=6):
                _render_single_source_card(source, idx, message_key)


def _get_last_sources() -> list[dict[str, Any]] | None:
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "assistant" and msg.get("sources"):
            return msg["sources"]
    return None


# ============================================================
# 4. TIỆN ÍCH ÂM THANH & XỬ LÝ HỘI THOẠI
# ============================================================
def make_wav_bytes(samples: np.ndarray) -> bytes:
    samples = np.asarray(samples, dtype=np.int16).reshape(-1)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())
    return buffer.getvalue()


def reset_voice_buffer() -> None:
    st.session_state.voice_chunks = []
    st.session_state.voice_pre_roll = []
    st.session_state.voice_is_speaking = False
    st.session_state.voice_silence_seconds = 0.0
    st.session_state.voice_speech_seconds = 0.0
    st.session_state.voice_total_seconds = 0.0


def cancel_active_tts() -> None:
    for msg in st.session_state.messages:
        if msg.get("role") != "assistant":
            continue
        msg["tts_autoplay"] = False
        task_id = msg.get("tts_task_id")
        if task_id and msg.get("tts_status") in {
            "queued", "generating", "playing", "fallback_playing"
        }:
            cancel_tts_task(task_id)
            msg["tts_status"] = "cancelled"
        msg.pop("tts_segments", None)
        msg.pop("tts_combined_audio", None)
        msg.pop("tts_audio_chunks", None)


def append_turn(
    question: str,
    *,
    speak_response: bool = False,
    background_tts: bool = False,
) -> None:
    """Process one user turn through handle_intent and attach optional TTS playback."""
    question = (question or "").strip()
    if not question:
        return

    cancel_active_tts()

    last_sources = _get_last_sources()
    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": question,
    }
    st.session_state.messages.append(user_message)

    try:
        answer, sources = handle_intent(
            question,
            context_sources=last_sources,
            pinned_article=st.session_state.pinned_article,
            source_filter=SOURCE_LABELS.get(st.session_state.filter_source_label),
            category_filter=CATEGORY_LABELS.get(st.session_state.filter_category_label),
        )
    except Exception as error:
        print("Lỗi xử lý yêu cầu:", repr(error))
        if getattr(error, "winerror", None) == 1455:
            answer = (
                "Windows chưa đủ bộ nhớ ảo để nạp PyTorch/PhoBERT. "
                "Hãy tăng Paging file trong Windows hoặc dùng các tính năng tra cứu SQLite."
            )
        else:
            answer = (
                "Đã xảy ra lỗi khi xử lý yêu cầu. "
                "Bạn hãy kiểm tra terminal để xem chi tiết."
            )
        sources = []

    assistant_message: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": answer,
        "sources": sources or [],
        "tts_autoplay": speak_response,
    }

    if speak_response:
        speech_text = build_reading_text(answer)
        if speech_text:
            if background_tts:
                task_id = str(uuid.uuid4())
                assistant_message.update(
                    {
                        "tts_status": "queued",
                        "tts_segments": [],
                        "tts_task_id": task_id,
                        "tts_progress": {"completed_chunks": 0, "total_chunks": 0},
                    }
                )
                try:
                    submit_tts_task(
                        task_id,
                        speech_text,
                        voice=st.session_state.selected_voice or DEFAULT_VOICE,
                        fallback_voice=FALLBACK_VOICE,
                        max_chunk_chars=420,
                    )
                except Exception as exc:
                    assistant_message["tts_status"] = "failed"
                    assistant_message["tts_error"] = str(exc)
                    assistant_message["tts_autoplay"] = False
            else:
                try:
                    assistant_message["tts_audio_chunks"] = synthesize_text_chunks(
                        speech_text,
                        voice=st.session_state.selected_voice or DEFAULT_VOICE,
                    )
                except Exception as exc:
                    assistant_message["tts_error"] = str(exc)

    st.session_state.messages.append(assistant_message)


def browse_filtered_news() -> None:
    """Show latest articles matching the sidebar source/category filters without calling LLM."""
    src_label = st.session_state.filter_source_label
    cat_label = st.session_state.filter_category_label
    src_slug = SOURCE_LABELS.get(src_label)
    cat_slug = CATEGORY_LABELS.get(cat_label)

    articles, total_count, dated_count = get_latest_articles(
        limit=10,
        source=src_slug,
        category=cat_slug,
    )
    heading_parts = []
    if cat_slug:
        heading_parts.append(f"chuyên mục **{cat_label}**")
    if src_slug:
        heading_parts.append(f"báo **{src_label}**")
    prompt_label = (
        "Xem tin mới nhất thuộc " + ", ".join(heading_parts)
        if heading_parts
        else "Xem 10 tin mới nhất hôm nay"
    )

    answer = _format_latest_news(articles, total_count, dated_count)
    st.session_state.messages.append(
        {"id": str(uuid.uuid4()), "role": "user", "content": prompt_label}
    )
    st.session_state.messages.append(
        {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": answer,
            "sources": articles,
        }
    )


# ============================================================
# 5. KHỞI TẠO SESSION STATE & KIỂM TRA DB
# ============================================================
state_defaults: dict[str, Any] = {
    "messages": [],
    "pinned_article": None,
    "interaction_mode": MODES[0],
    "selected_voice": DEFAULT_VOICE,
    "auto_read_push_to_talk": True,
    "filter_category_label": "Tất cả chuyên mục",
    "filter_source_label": "Tất cả báo",
    "voice_input_version": 0,
    "voice_chunks": [],
    "voice_pre_roll": [],
    "voice_is_speaking": False,
    "voice_silence_seconds": 0.0,
    "voice_speech_seconds": 0.0,
    "voice_total_seconds": 0.0,
    "voice_mute_until": 0.0,
}

for key, value in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

try:
    total_articles = initialize_system()
except Exception as error:
    st.error(f"Không thể kết nối cơ sở dữ liệu bài báo: {error}")
    st.stop()


# ============================================================
# 6. GIAO DIỆN CHÍNH & THANH BÊN
# ============================================================
st.title("📰 Trợ lý đọc báo thông minh")
st.caption("Tìm kiếm, tóm tắt và hỏi đáp tin tức từ VnExpress, Tuổi Trẻ và Tiền Phong")

with st.sidebar:
    st.header("🎛️ Điều khiển")

    selected_mode = st.radio(
        "Chế độ tương tác",
        options=MODES,
        key="interaction_mode",
    )

    if selected_mode != MODES[0]:
        voice_label = st.selectbox(
            "Giọng đọc phản hồi",
            options=list(VOICE_OPTIONS.keys()),
            index=list(VOICE_OPTIONS.values()).index(st.session_state.selected_voice),
        )
        st.session_state.selected_voice = VOICE_OPTIONS[voice_label]

        if selected_mode == MODES[1]:
            st.checkbox("Tự động đọc câu trả lời", key="auto_read_push_to_talk")

    st.divider()
    st.subheader("🧭 Khám phá theo bộ lọc")
    st.selectbox(
        "Chuyên mục",
        options=list(CATEGORY_LABELS.keys()),
        key="filter_category_label",
    )
    st.selectbox(
        "Nguồn báo",
        options=list(SOURCE_LABELS.keys()),
        key="filter_source_label",
    )
    if st.button("📰 Xem tin mới theo bộ lọc", use_container_width=True):
        browse_filtered_news()
        st.rerun()

    st.divider()
    if st.button("🗑️ Xóa lịch sử hội thoại", use_container_width=True):
        cancel_active_tts()
        st.session_state.messages = []
        st.session_state.pinned_article = None
        reset_voice_buffer()
        st.rerun()

    with st.expander("ℹ️ Thông tin hệ thống"):
        st.metric("Tổng số bài báo", f"{total_articles:,}")
        st.caption("Tìm kiếm: SQLite FTS5 + PhoBERT + FAISS")
        st.caption("Tóm tắt: ViT5 | Hỏi đáp: Qwen2.5-1.5B")
        st.caption("Giọng nói: Google STT + Edge TTS")


# Vị trí cố định cho fragment microphone khi dùng chế độ rảnh tay.
voice_fragment_slot = st.container()

# ============================================================
# 7. MÀN HÌNH CHÀO KHI CHƯA CÓ HỘI THOẠI
# ============================================================
if not st.session_state.messages:
    st.markdown("### 💡 Gợi ý nhanh để bắt đầu")
    st.write("Chọn một chủ đề hoặc câu hỏi mẫu bên dưới, hoặc nhập yêu cầu của bạn ở khung chat.")

    starter_cols = st.columns(2)
    starter_prompts = [
        ("🔥 10 tin mới nhất hôm nay", "Tin tức mới nhất hôm nay"),
        ("⚽ Tìm bài viết nhắc đến Mbappe", "Tìm các bài báo nhắc đến Mbappe"),
        ("📈 Tin mới nhất về giá vàng", "Tìm bài báo về giá vàng mới nhất"),
        ("🏫 Hỏi đáp RAG: Công trình vệ sinh Sơn La", "Dự án vệ sinh học đường tại Sơn La đã xây mới bao nhiêu công trình?"),
    ]
    for idx, (label, prompt) in enumerate(starter_prompts):
        with starter_cols[idx % 2]:
            if st.button(label, key=f"starter_{idx}", use_container_width=True):
                append_turn(
                    prompt,
                    speak_response=(selected_mode == MODES[2]),
                    background_tts=(selected_mode == MODES[2]),
                )
                st.rerun()

    st.markdown("#### 🗂️ Duyệt nhanh theo chuyên mục")
    cat_buttons = [
        ("🏛️ Thời sự", "Liệt kê bài báo thuộc chuyên mục thời sự"),
        ("📈 Kinh tế", "Liệt kê bài báo thuộc chuyên mục kinh tế"),
        ("💻 Công nghệ", "Liệt kê bài báo thuộc chuyên mục công nghệ"),
        ("🎬 Giải trí", "Liệt kê bài báo thuộc chuyên mục giải trí"),
        ("⚽ Thể thao", "Liệt kê bài báo thuộc chuyên mục thể thao"),
    ]
    cat_cols = st.columns(len(cat_buttons))
    for col, (cat_title, cat_prompt) in zip(cat_cols, cat_buttons):
        with col:
            if st.button(cat_title, key=f"quick_cat_{cat_title}", use_container_width=True):
                append_turn(
                    cat_prompt,
                    speak_response=(selected_mode == MODES[2]),
                    background_tts=(selected_mode == MODES[2]),
                )
                st.rerun()


# ============================================================
# 8. LỊCH SỬ HỘI THOẠI
# ============================================================
for message in st.session_state.messages:
    if "id" not in message:
        message["id"] = str(uuid.uuid4())

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            # Sync audio chunks (Push-to-Talk mode)
            sync_chunks = message.get("tts_audio_chunks") or []
            for part_idx, audio_bytes in enumerate(sync_chunks, start=1):
                if len(sync_chunks) > 1:
                    st.caption(f"Phần {part_idx}/{len(sync_chunks)}")
                st.audio(
                    audio_bytes,
                    format="audio/mp3",
                    autoplay=(part_idx == 1 and bool(message.get("tts_autoplay", False))),
                )
            if sync_chunks and message.get("tts_autoplay"):
                message["tts_autoplay"] = False

            # Background streaming audio queue (Hands-free mode)
            task_id = message.get("tts_task_id")
            audio_segments = message.get("tts_segments", [])
            tts_status = message.get("tts_status")

            if audio_segments and tts_status != "cancelled" and not message.get("tts_component_failed"):
                try:
                    audio_queue_component = get_tts_audio_queue_component()
                    component_value = audio_queue_component(
                        task_id=task_id or message["id"],
                        segments=[
                            {"id": item["id"], "audio_b64": item["audio_b64"]}
                            for item in audio_segments
                        ],
                        is_complete=tts_status in {"playing", "finished", "partial"},
                        autoplay=tts_status != "finished",
                        key=f"tts_queue_{message['id']}",
                        default=None,
                    )
                    if (
                        isinstance(component_value, dict)
                        and component_value.get("task_id") == task_id
                        and component_value.get("status") == "finished"
                        and tts_status != "finished"
                    ):
                        message["tts_status"] = "finished"
                        st.session_state.voice_mute_until = 0.0
                except Exception as exc:
                    print("Không thể hiển thị audio queue:", repr(exc))
                    message["tts_component_failed"] = True
                    if message.get("tts_combined_audio"):
                        message["tts_status"] = "fallback_playing"

            tts_status = message.get("tts_status")
            if message.get("tts_component_failed") and message.get("tts_combined_audio"):
                st.audio(message["tts_combined_audio"], format="audio/mpeg", autoplay=True)

            if tts_status in {"queued", "generating"}:
                progress = message.get("tts_progress") or {}
                completed = int(progress.get("completed_chunks", 0))
                total = int(progress.get("total_chunks", 0))
                st.caption(
                    f"Đang chuẩn bị giọng đọc: {completed}/{total} đoạn..."
                    if total
                    else "Đang chuẩn bị giọng đọc..."
                )
            elif tts_status == "playing":
                st.caption("🔊 Đang đọc câu trả lời (micro tạm dừng).")
            elif tts_status == "finished":
                st.caption("Đã đọc xong câu trả lời.")

            if message.get("tts_error"):
                st.caption(f"Chi tiết TTS: {message['tts_error']}")
            if message.get("tts_warning"):
                st.warning(message["tts_warning"])

            render_sources(message.get("sources", []), message_key=message["id"])


# ============================================================
# 9. BANNER BÀI ĐANG GHIM & KHUNG NHẬP LIỆU
# ============================================================
if st.session_state.pinned_article is not None:
    pinned_title = st.session_state.pinned_article.get("title") or "Bài báo đang chọn"
    pin_col1, pin_col2 = st.columns([5, 1])
    with pin_col1:
        st.info(f"📌 Đang hỏi nối tiếp về bài: **{pinned_title}**")
    with pin_col2:
        if st.button("✕ Bỏ ghim", use_container_width=True):
            st.session_state.pinned_article = None
            st.rerun()

recorded_audio = None
if selected_mode == MODES[1]:
    recorded_audio = st.audio_input(
        "🎙️ Ghi âm câu hỏi bằng tiếng Việt",
        key=f"voice_input_{st.session_state.voice_input_version}",
        help="Nhấn ghi âm, nói câu hỏi, rồi nhấn dừng để gửi.",
    )

voice_ctx = None
if selected_mode == MODES[2]:
    mic_info_col, mic_ctrl_col = st.columns([4, 2], vertical_alignment="center")
    with mic_info_col:
        st.caption(
            "🗣️ **Hội thoại rảnh tay:** Nhấn **Bật mic** rồi nói; hệ thống tự nhận diện sau khoảng lặng và đọc câu trả lời."
        )
    with mic_ctrl_col:
        voice_ctx = webrtc_streamer(
            key="continuous_voice_chat",
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints={"video": False, "audio": True},
            audio_receiver_size=1024,
            translations={
                "start": "🎙️ Bật mic",
                "stop": "⏹️ Tắt mic",
                "select_device": "Chọn mic",
            },
        )

typed_question = st.chat_input(
    "Nhập câu hỏi, tên nhân vật, chủ đề hoặc 'tóm tắt bài số 1'...",
    key="main_chat_input",
)

if recorded_audio is not None:
    st.session_state.voice_input_version += 1
    if typed_question:
        append_turn(
            typed_question,
            speak_response=bool(st.session_state.auto_read_push_to_talk),
            background_tts=False,
        )
        st.rerun()
    else:
        try:
            with st.spinner("Đang nhận dạng giọng nói tiếng Việt..."):
                recognized = transcribe_audio(recorded_audio.getvalue())
            append_turn(
                recognized,
                speak_response=bool(st.session_state.auto_read_push_to_talk),
                background_tts=False,
            )
            st.rerun()
        except SpeechRecognitionError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Không thể nhận dạng bản ghi âm: {exc}")
elif typed_question:
    append_turn(
        typed_question,
        speak_response=(selected_mode == MODES[2]),
        background_tts=(selected_mode == MODES[2]),
    )
    st.rerun()


# ============================================================
# 10. LUỒNG MICROPHONE LIÊN TỤC (KHI Ở CHẾ ĐỘ RẢNH TAY)
# ============================================================
def sync_tts_background_events() -> bool:
    changed = False
    now = time.monotonic()

    for message in st.session_state.messages:
        if (
            message.get("role") == "assistant"
            and message.get("tts_status") == "fallback_playing"
            and now >= st.session_state.voice_mute_until
        ):
            message["tts_status"] = "finished"
            changed = True

        if message.get("role") != "assistant":
            continue

        task_id = message.get("tts_task_id")
        status = message.get("tts_status")
        if not task_id or status not in {"queued", "generating", "playing", "cancelled"}:
            continue

        for event in poll_tts_events(task_id):
            event_type = event.get("type")
            if event_type == "started":
                if message.get("tts_status") != "cancelled":
                    message["tts_status"] = "generating"
                    changed = True
            elif event_type == "progress":
                if message.get("tts_status") != "cancelled":
                    message["tts_progress"] = {
                        "completed_chunks": event.get("completed_chunks", 0),
                        "total_chunks": event.get("total_chunks", 0),
                    }
                    message["tts_status"] = "generating"
                    changed = True
            elif event_type == "audio_segment":
                if message.get("tts_status") != "cancelled":
                    segment_bytes = event.get("audio_bytes", b"")
                    if segment_bytes:
                        if not message.get("tts_first_segment_at"):
                            message["tts_first_segment_at"] = time.monotonic()
                        segment_index = len(message.setdefault("tts_segments", [])) + 1
                        message["tts_segments"].append(
                            {
                                "id": f"{task_id}-{segment_index}",
                                "audio_b64": base64.b64encode(segment_bytes).decode("ascii"),
                                "duration_seconds": float(event.get("duration_seconds", 0.0)),
                            }
                        )
                        message["tts_status"] = "generating"
                        changed = True
            elif event_type == "completed":
                result = event.get("result")
                if message.get("tts_status") != "cancelled" and result is not None:
                    if result.audio_bytes:
                        message["tts_combined_audio"] = result.audio_bytes
                    if message.get("tts_segments"):
                        message["tts_status"] = (
                            "fallback_playing"
                            if message.get("tts_component_failed")
                            else "playing"
                        )
                        first_at = float(message.get("tts_first_segment_at", time.monotonic()))
                        elapsed = max(0.0, time.monotonic() - first_at)
                        remaining = max(0.0, result.duration_seconds - elapsed)
                        st.session_state.voice_mute_until = time.monotonic() + remaining + 1.5
                    else:
                        message["tts_status"] = "failed"
                        message["tts_error"] = "Edge TTS không tạo được âm thanh."
                    if result.failed_segments:
                        message["tts_warning"] = (
                            "Một số phần không tạo được âm thanh: "
                            + "; ".join(result.failed_segments[:4])
                        )
                    changed = True
                release_tts_task(task_id)
            elif event_type == "failed":
                if message.get("tts_status") != "cancelled":
                    message["tts_status"] = "playing" if message.get("tts_segments") else "failed"
                    message["tts_error"] = event.get("error", "Lỗi TTS không xác định.")
                    changed = True
                release_tts_task(task_id)
            elif event_type == "cancelled":
                release_tts_task(task_id)

    return changed


if selected_mode == MODES[2] and voice_ctx is not None:
    tts_tasks_pending = any(
        msg.get("role") == "assistant"
        and msg.get("tts_status") in {"queued", "generating", "playing", "fallback_playing"}
        for msg in st.session_state.messages
    )
    voice_poll_interval = (
        VOICE_POLL_INTERVAL if (voice_ctx.state.playing or tts_tasks_pending) else None
    )

    @st.fragment(run_every=voice_poll_interval)
    def poll_microphone():
        if sync_tts_background_events():
            st.rerun()

        tts_active = any(
            msg.get("role") == "assistant"
            and msg.get("tts_status") in {"queued", "generating", "playing"}
            for msg in st.session_state.messages
        )
        if tts_active:
            if voice_ctx.state.playing and voice_ctx.audio_receiver is not None:
                try:
                    voice_ctx.audio_receiver.get_frames(timeout=0.01)
                except Exception:
                    pass
            reset_voice_buffer()
            return

        if not voice_ctx.state.playing or voice_ctx.audio_receiver is None:
            return

        try:
            audio_frames = voice_ctx.audio_receiver.get_frames(timeout=0.05)
        except queue.Empty:
            return
        except Exception:
            return

        if time.monotonic() < st.session_state.voice_mute_until:
            reset_voice_buffer()
            return

        if not audio_frames:
            return

        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        completed_wav = None

        for input_frame in audio_frames:
            try:
                output_frames = resampler.resample(input_frame)
            except Exception:
                continue

            for output_frame in output_frames:
                samples = output_frame.to_ndarray().reshape(-1).astype(np.int16)
                if samples.size == 0:
                    continue

                frame_seconds = samples.size / SAMPLE_RATE
                samples_float = samples.astype(np.float32)
                rms = float(np.sqrt(np.mean(samples_float * samples_float)))
                is_speech = rms >= VAD_RMS_THRESHOLD

                if not st.session_state.voice_is_speaking:
                    if not is_speech:
                        pre_roll = st.session_state.voice_pre_roll
                        pre_roll.append(samples.copy())
                        if len(pre_roll) > PRE_ROLL_FRAMES:
                            pre_roll.pop(0)
                        continue

                    pre_roll = st.session_state.voice_pre_roll
                    st.session_state.voice_is_speaking = True
                    st.session_state.voice_chunks = list(pre_roll) + [samples.copy()]
                    st.session_state.voice_total_seconds = (
                        sum(len(part) for part in pre_roll) / SAMPLE_RATE + frame_seconds
                    )
                    st.session_state.voice_speech_seconds = frame_seconds
                    st.session_state.voice_silence_seconds = 0.0
                    st.session_state.voice_pre_roll = []
                    continue

                st.session_state.voice_chunks.append(samples.copy())
                st.session_state.voice_total_seconds += frame_seconds
                if is_speech:
                    st.session_state.voice_speech_seconds += frame_seconds
                    st.session_state.voice_silence_seconds = 0.0
                else:
                    st.session_state.voice_silence_seconds += frame_seconds

                if (
                    st.session_state.voice_silence_seconds >= END_SILENCE_SECONDS
                    or st.session_state.voice_total_seconds >= MAX_UTTERANCE_SECONDS
                ):
                    speech_seconds = st.session_state.voice_speech_seconds
                    chunks = st.session_state.voice_chunks
                    reset_voice_buffer()
                    if speech_seconds >= MIN_SPEECH_SECONDS and chunks:
                        completed_wav = make_wav_bytes(np.concatenate(chunks))
                    break

            if completed_wav is not None:
                break

        if completed_wav is None:
            return

        try:
            recognized_text = transcribe_audio(completed_wav).strip()
        except SpeechRecognitionError as exc:
            st.toast(str(exc), icon="⚠️")
            return
        except Exception:
            st.toast("Không nhận dạng được giọng nói. Hãy thử lại.", icon="⚠️")
            return

        if recognized_text:
            append_turn(recognized_text, speak_response=True, background_tts=True)
            st.rerun()

    with voice_fragment_slot:
        poll_microphone()
