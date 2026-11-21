"""Unified, low-latency Streamlit interface for the Vietnamese News Assistant."""

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
from nlp.intent_handler import (
    _format_latest_news,
    _handle_target_article,
    prepare_turn_response,
)
from voice.reading_text import build_reading_text
from voice.speech_to_text import SpeechRecognitionError, transcribe_audio
from voice.text_to_speech import DEFAULT_VOICE, FALLBACK_VOICE, VOICE_OPTIONS
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
TTS_POLL_INTERVAL = 0.35
TTS_ACTIVE_STATES = {"queued", "generating", "playing", "fallback_playing"}


@st.cache_data
def initialize_system() -> int:
    """Đếm bài báo trong SQLite; không nạp PhoBERT/FAISS khi vừa mở trang."""
    return int(get_article_count())


@st.cache_data(show_spinner=False, max_entries=64)
def cached_article_summary(article_id: str, body: str) -> str:
    """Cache bản tóm tắt ViT5 theo article_id để dùng chung giữa chat và modal."""
    from processing.summarizer import summarize

    return summarize(body, article_id=article_id)


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
# 2. HỘP THOẠI ĐỌC BÀI BÁO & STREAMING TTS TRONG MODAL
# ============================================================
def _sync_tts_state_container(container: dict[str, Any]) -> tuple[bool, bool]:
    """Drain background TTS events into a message or dialog dict.

    Returns ``(changed, reached_terminal)``.
    """
    changed = False
    reached_terminal = False
    now = time.monotonic()

    if (
        container.get("tts_status") == "fallback_playing"
        and now >= float(st.session_state.get("voice_mute_until", 0.0))
    ):
        container["tts_status"] = "finished"
        return True, True

    task_id = container.get("tts_task_id")
    status = container.get("tts_status")
    if not task_id or status not in {"queued", "generating", "playing", "cancelled"}:
        return changed, reached_terminal

    for event in poll_tts_events(task_id):
        event_type = event.get("type")
        if event_type == "started":
            if container.get("tts_status") != "cancelled":
                container["tts_status"] = "generating"
                changed = True
        elif event_type == "progress":
            if container.get("tts_status") != "cancelled":
                container["tts_progress"] = {
                    "completed_chunks": event.get("completed_chunks", 0),
                    "total_chunks": event.get("total_chunks", 0),
                }
                container["tts_status"] = "generating"
                changed = True
        elif event_type == "audio_segment":
            if container.get("tts_status") != "cancelled":
                segment_bytes = event.get("audio_bytes", b"")
                if segment_bytes:
                    if not container.get("tts_first_segment_at"):
                        container["tts_first_segment_at"] = time.monotonic()
                    segment_index = len(container.setdefault("tts_segments", [])) + 1
                    container["tts_segments"].append(
                        {
                            "id": f"{task_id}-{segment_index}",
                            "audio_b64": base64.b64encode(segment_bytes).decode("ascii"),
                            "duration_seconds": float(event.get("duration_seconds", 0.0)),
                            "mime_type": event.get("mime_type", "audio/wav"),
                        }
                    )
                    container["tts_status"] = "generating"
                    changed = True
        elif event_type == "completed":
            result = event.get("result")
            if container.get("tts_status") != "cancelled" and result is not None:
                if result.audio_bytes:
                    container["tts_combined_audio"] = result.audio_bytes
                if container.get("tts_segments"):
                    container["tts_status"] = (
                        "fallback_playing"
                        if container.get("tts_component_failed")
                        else "playing"
                    )
                    first_at = float(container.get("tts_first_segment_at", time.monotonic()))
                    elapsed = max(0.0, time.monotonic() - first_at)
                    remaining = max(0.0, result.duration_seconds - elapsed)
                    st.session_state.voice_mute_until = time.monotonic() + remaining + 1.5
                else:
                    container["tts_status"] = "failed"
                    container["tts_error"] = "Piper TTS không tạo được âm thanh."
                    reached_terminal = True
                if result.failed_segments:
                    container["tts_warning"] = (
                        "Một số phần không tạo được âm thanh: "
                        + "; ".join(result.failed_segments[:4])
                    )
                changed = True
            release_tts_task(task_id)
        elif event_type == "failed":
            if container.get("tts_status") != "cancelled":
                has_segments = bool(container.get("tts_segments"))
                container["tts_status"] = "playing" if has_segments else "failed"
                container["tts_error"] = event.get("error", "Lỗi TTS không xác định.")
                if not has_segments:
                    reached_terminal = True
                changed = True
            release_tts_task(task_id)
        elif event_type == "cancelled":
            release_tts_task(task_id)
            reached_terminal = True

    return changed, reached_terminal


def _render_streaming_tts_widget(container: dict[str, Any], *, key_prefix: str) -> bool:
    """Render the sequential browser audio queue widget for a message/dialog dict.

    Returns ``True`` when playback transitions into a terminal state.
    """
    reached_terminal = False
    task_id = container.get("tts_task_id")
    audio_segments = container.get("tts_segments", [])
    tts_status = container.get("tts_status")

    if audio_segments and tts_status != "cancelled" and not container.get("tts_component_failed"):
        try:
            audio_queue_component = get_tts_audio_queue_component()
            component_value = audio_queue_component(
                task_id=task_id or key_prefix,
                segments=[
                    {
                        "id": item["id"],
                        "audio_b64": item["audio_b64"],
                        "mime_type": item.get("mime_type", "audio/wav"),
                    }
                    for item in audio_segments
                ],
                is_complete=tts_status in {"playing", "finished", "partial"},
                autoplay=tts_status != "finished",
                key=f"{key_prefix}_{task_id or 'idle'}",
                default=None,
            )
            if isinstance(component_value, dict) and component_value.get("task_id") == task_id:
                reported = component_value.get("status")
                if reported == "finished" and tts_status != "finished":
                    container["tts_status"] = "finished"
                    st.session_state.voice_mute_until = 0.0
                    reached_terminal = True
                elif reported == "stopped" and tts_status != "cancelled":
                    if task_id:
                        cancel_tts_task(task_id)
                    container["tts_status"] = "cancelled"
                    container["tts_autoplay"] = False
                    st.session_state.voice_mute_until = 0.0
                    reached_terminal = True
        except Exception as exc:
            print("Không thể hiển thị audio queue:", repr(exc))
            container["tts_component_failed"] = True
            if container.get("tts_combined_audio"):
                container["tts_status"] = "fallback_playing"

    tts_status = container.get("tts_status")
    if container.get("tts_component_failed") and container.get("tts_combined_audio"):
        st.audio(container["tts_combined_audio"], format="audio/wav", autoplay=True)

    if tts_status in {"queued", "generating"}:
        progress = container.get("tts_progress") or {}
        completed = int(progress.get("completed_chunks", 0))
        total = int(progress.get("total_chunks", 0))
        st.caption(
            f"🔊 Đang phát trực tiếp ({completed}/{total} đoạn)..."
            if total
            else "🔊 Đang tổng hợp đoạn đầu tiên..."
        )
    elif tts_status == "playing":
        st.caption("🔊 Đang đọc câu trả lời (nhấn ⏹ Dừng đọc nếu muốn ngắt).")
    elif tts_status == "finished":
        st.caption("Đã đọc xong.")
    elif tts_status == "cancelled":
        st.caption("⏹️ Đã dừng đọc.")

    if container.get("tts_error"):
        st.caption(f"Chi tiết TTS: {container['tts_error']}")
    if container.get("tts_warning"):
        st.warning(container["tts_warning"])

    return reached_terminal


def _start_tts_for_container(
    container: dict[str, Any],
    speech_text: str,
    *,
    voice: str,
) -> None:
    """Attach and launch a non-blocking streaming TTS background task."""
    if not speech_text or not speech_text.strip():
        return
    task_id = str(uuid.uuid4())
    container.update(
        {
            "tts_autoplay": True,
            "tts_status": "queued",
            "tts_segments": [],
            "tts_task_id": task_id,
            "tts_progress": {"completed_chunks": 0, "total_chunks": 0},
        }
    )
    container.pop("tts_error", None)
    container.pop("tts_warning", None)
    container.pop("tts_combined_audio", None)
    container.pop("tts_component_failed", None)
    container.pop("tts_first_segment_at", None)
    try:
        submit_tts_task(
            task_id,
            speech_text,
            voice=voice or DEFAULT_VOICE,
            fallback_voice=FALLBACK_VOICE,
            max_chunk_chars=380,
        )
    except Exception as exc:
        container["tts_status"] = "failed"
        container["tts_error"] = str(exc)
        container["tts_autoplay"] = False


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

    action_col1, action_col2 = st.columns([1.2, 1.5])
    with action_col1:
        if url.startswith(("https://", "http://")):
            st.link_button("🌐 Mở bài báo gốc", url, use_container_width=True)
    with action_col2:
        if st.button("💬 Ghim vào chat để hỏi tiếp", key="dlg_pin_article", use_container_width=True):
            st.session_state.pinned_article = dict(source_item)
            st.session_state.active_article = dict(source_item)
            st.session_state.toast_notice = f"📌 Đã ghim: {title[:60]}"
            st.rerun()

    sapo = (content.get("sapo") or "").strip()
    body = (content.get("body") or "").strip()

    if mode == "summarize":
        target_text = body or sapo
        if sapo and body:
            st.markdown(f"**Sapo:** {sapo}")
        if not target_text:
            st.warning("Bài báo chưa có nội dung để tóm tắt.")
            return
        with st.spinner("Đang tóm tắt bằng mô hình ViT5..."):
            summary = cached_article_summary(
                str(content.get("article_id") or url or title),
                target_text,
            )
        if summary and summary.strip():
            st.info(f"**Tóm tắt (ViT5):**\n\n{summary.strip()}")
        else:
            st.warning("Mô hình ViT5 chưa tạo được bản tóm tắt.")

        if body:
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

    speakable_text = body or sapo
    if speakable_text:
        st.divider()
        article_key = str(content.get("article_id") or url or title)
        dialog_tts = st.session_state.get("dialog_tts")
        if not isinstance(dialog_tts, dict) or dialog_tts.get("article_key") != article_key:
            dialog_tts = {"article_key": article_key}
            st.session_state.dialog_tts = dialog_tts

        btn_col1, btn_col2 = st.columns([1.4, 1.0])
        with btn_col1:
            if st.button("🔊 Nghe bài báo (phát trực tiếp)", key=f"dlg_tts_{article_key}"):
                cancel_active_tts()
                dialog_tts = {"article_key": article_key}
                st.session_state.dialog_tts = dialog_tts
                speech_script = build_reading_text(f"{title}. {speakable_text}")
                _start_tts_for_container(
                    dialog_tts,
                    speech_script,
                    voice=st.session_state.selected_voice or DEFAULT_VOICE,
                )
        with btn_col2:
            if dialog_tts.get("tts_status") in TTS_ACTIVE_STATES:
                if st.button("⏹️ Dừng đọc", key=f"dlg_stop_{article_key}"):
                    if dialog_tts.get("tts_task_id"):
                        cancel_tts_task(dialog_tts["tts_task_id"])
                    dialog_tts["tts_status"] = "cancelled"

        poll_interval = (
            TTS_POLL_INTERVAL if dialog_tts.get("tts_status") in TTS_ACTIVE_STATES else None
        )

        @st.fragment(run_every=poll_interval)
        def _render_dialog_tts():
            current = st.session_state.get("dialog_tts") or {}
            if current.get("article_key") != article_key or not current.get("tts_task_id"):
                return
            _sync_tts_state_container(current)
            _render_streaming_tts_widget(current, key_prefix="dlg_tts_q")

        _render_dialog_tts()


# ============================================================
# 3. HIỂN THỊ THẺ BÀI BÁO & TRẠNG THÁI
# ============================================================
def _short_title(title: str, limit: int = 58) -> str:
    clean = " ".join((title or "Bài báo").split())
    return clean if len(clean) <= limit else clean[: limit - 3].rstrip() + "..."


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
                st.session_state.active_article = source
                show_article_dialog(source, mode="read")
        with col2:
            if st.button("✨ Tóm tắt", key=f"sum_{message_key}_{index}", use_container_width=True):
                st.session_state.active_article = source
                queue_user_turn(
                    f"Tóm tắt bài số {index}: {_short_title(title)}",
                    speak_response=(st.session_state.interaction_mode == MODES[2]),
                    target_article=source,
                    target_action="summarize",
                )
                st.rerun()
        with col3:
            if st.button("💬 Hỏi bài này", key=f"pin_{message_key}_{index}", use_container_width=True):
                st.session_state.pinned_article = source
                st.session_state.active_article = source
                st.session_state.toast_notice = (
                    f"📌 Đã ghim: {_short_title(title, 52)}. Hãy chọn gợi ý bên dưới hoặc nhập câu hỏi!"
                )
                st.rerun()
        with col4:
            if url.startswith(("https://", "http://")):
                st.link_button("🌐 Bài gốc", url, use_container_width=True)


def render_sources(
    sources: list[dict[str, Any]],
    message_key: str,
    *,
    is_latest_source_message: bool = True,
) -> None:
    if not sources:
        return

    is_expanded = (
        is_latest_source_message
        or st.session_state.get("expanded_history_message_id") == message_key
    )
    if not is_expanded:
        with st.expander(f"📚 Xem lại {len(sources)} nguồn bài báo từ lượt này"):
            for idx, source in enumerate(sources, start=1):
                title = source.get("title") or "Không có tiêu đề"
                url = source.get("url") or ""
                src_slug = source.get("source") or ""
                src_label = SOURCE_DISPLAY.get(src_slug, src_slug)
                if url.startswith(("https://", "http://")):
                    st.markdown(f"{idx}. [{title}]({url}) — *{src_label}*")
                else:
                    st.markdown(f"{idx}. **{title}** — *{src_label}*")
            if st.button(
                "🔄 Hiện nút thao tác cho danh sách này",
                key=f"expand_hist_{message_key}",
            ):
                st.session_state.expanded_history_message_id = message_key
                if len(sources) > 1:
                    st.session_state.last_search_sources = sources
                else:
                    st.session_state.active_article = sources[0]
                st.rerun()
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


def _get_message_by_id(message_id: str) -> dict[str, Any] | None:
    for msg in st.session_state.messages:
        if msg.get("id") == message_id:
            return msg
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
    """Stop any running chat or dialog TTS tasks cleanly and reset old message players."""
    for msg in st.session_state.messages:
        if msg.get("role") != "assistant":
            continue
        msg["tts_autoplay"] = False
        task_id = msg.get("tts_task_id")
        if task_id and msg.get("tts_status") in TTS_ACTIVE_STATES:
            cancel_tts_task(task_id)
        for tts_key in (
            "tts_task_id",
            "tts_status",
            "tts_progress",
            "tts_segments",
            "tts_combined_audio",
            "tts_first_segment_at",
            "tts_component_failed",
            "tts_error",
            "tts_warning",
        ):
            msg.pop(tts_key, None)

    dialog_tts = st.session_state.get("dialog_tts")
    if isinstance(dialog_tts, dict):
        dlg_task = dialog_tts.get("tts_task_id")
        if dlg_task and dialog_tts.get("tts_status") in TTS_ACTIVE_STATES:
            cancel_tts_task(dlg_task)
            dialog_tts["tts_status"] = "cancelled"
        dialog_tts.pop("tts_segments", None)
        dialog_tts.pop("tts_combined_audio", None)

    st.session_state.voice_mute_until = 0.0


def queue_user_turn(
    question: str,
    *,
    speak_response: bool = False,
    target_article: dict[str, Any] | None = None,
    target_action: str | None = None,
) -> None:
    """Append the user bubble immediately and schedule live execution on rerun."""
    clean_question = (question or "").strip()
    if not clean_question:
        return

    cancel_active_tts()
    st.session_state.messages.append(
        {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": clean_question,
        }
    )
    st.session_state.pending_turn = {
        "question": clean_question,
        "speak_response": bool(speak_response),
        "target_article": target_article,
        "target_action": target_action,
    }


def _execute_pending_turn(pending: dict[str, Any]) -> None:
    """Run one queued turn inside a live assistant bubble, streaming Qwen answers when applicable."""
    question = (pending.get("question") or "").strip()
    speak_response = bool(pending.get("speak_response", False))
    target_article = pending.get("target_article")
    target_action = pending.get("target_action")

    source_filter = SOURCE_LABELS.get(st.session_state.filter_source_label)
    category_filter = CATEGORY_LABELS.get(st.session_state.filter_category_label)

    with st.chat_message("assistant"):
        try:
            with st.spinner("⏳ Đang tìm kiếm và tổng hợp thông tin..."):
                if target_article is not None:
                    action_to_intent = {
                        "summarize": "summarization",
                        "detail": "article_detail",
                        "qa": "question_answering",
                    }
                    outcome = _handle_target_article(
                        question,
                        target_article,
                        action_to_intent.get(str(target_action), "summarization"),
                        stream_qa=True,
                    )
                else:
                    outcome = prepare_turn_response(
                        question,
                        context_sources=st.session_state.last_search_sources,
                        pinned_article=st.session_state.pinned_article,
                        active_article=st.session_state.active_article,
                        source_filter=source_filter,
                        category_filter=category_filter,
                        concise_list=True,
                    )

            if outcome.get("kind") == "qa_stream":
                streamed_text = st.write_stream(outcome["stream"])
                answer = (
                    streamed_text
                    if isinstance(streamed_text, str)
                    else "".join(str(part) for part in streamed_text)
                ).strip() or "Chưa tạo được câu trả lời."
            else:
                answer = str(outcome.get("answer") or "Chưa tạo được câu trả lời.")
                st.markdown(answer)

            sources = outcome.get("sources") or []
            selected_single = bool(outcome.get("selected_single", False))
            intent_name = outcome.get("intent", "")
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
            st.markdown(answer)
            sources = []
            selected_single = False
            intent_name = "error"

    if selected_single and sources:
        st.session_state.active_article = sources[0]
    elif sources:
        st.session_state.last_search_sources = sources
        st.session_state.active_article = None

    assistant_message: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "intent": intent_name,
        "tts_autoplay": speak_response,
    }

    if speak_response:
        speech_text = build_reading_text(answer, sources=sources)
        if speech_text:
            _start_tts_for_container(
                assistant_message,
                speech_text,
                voice=st.session_state.selected_voice or DEFAULT_VOICE,
            )

    st.session_state.messages.append(assistant_message)
    st.session_state.expanded_history_message_id = None
    st.session_state.pending_turn = None


def browse_filtered_news(*, speak_response: bool = False) -> None:
    """Show latest articles matching the sidebar source/category filters without calling LLM."""
    cancel_active_tts()
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

    answer = _format_latest_news(articles, total_count, dated_count, concise=True)
    if articles:
        st.session_state.last_search_sources = articles
        st.session_state.active_article = None

    st.session_state.messages.append(
        {"id": str(uuid.uuid4()), "role": "user", "content": prompt_label}
    )
    assistant_message: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": answer,
        "sources": articles,
        "intent": "latest_news",
        "tts_autoplay": speak_response,
    }
    if speak_response:
        speech_text = build_reading_text(answer, sources=articles)
        if speech_text:
            _start_tts_for_container(
                assistant_message,
                speech_text,
                voice=st.session_state.selected_voice or DEFAULT_VOICE,
            )
    st.session_state.messages.append(assistant_message)


def _render_message_audio_controls(message: dict[str, Any]) -> None:
    """Render an on-demand listen/replay button and an isolated streaming TTS fragment."""
    msg_id = message["id"]
    intent_name = message.get("intent", "")
    tts_status = message.get("tts_status")

    is_tts_active = tts_status in TTS_ACTIVE_STATES
    if intent_name not in {"empty", "error"} and not is_tts_active:
        btn_label = (
            "🔊 Nghe lại"
            if tts_status in {"finished", "cancelled", "failed"}
            else "🔊 Nghe câu trả lời"
        )
        if st.button(btn_label, key=f"speak_{msg_id}"):
            cancel_active_tts()
            speech_text = build_reading_text(
                message.get("content", ""),
                sources=message.get("sources"),
            )
            if speech_text:
                _start_tts_for_container(
                    message,
                    speech_text,
                    voice=st.session_state.selected_voice or DEFAULT_VOICE,
                )
            st.rerun()

    if not message.get("tts_task_id"):
        return

    poll_interval = TTS_POLL_INTERVAL if tts_status in TTS_ACTIVE_STATES else None

    @st.fragment(run_every=poll_interval)
    def _tts_fragment() -> None:
        current_msg = _get_message_by_id(msg_id)
        if current_msg is None:
            return
        changed, reached_terminal_sync = _sync_tts_state_container(current_msg)
        reached_terminal_ui = _render_streaming_tts_widget(
            current_msg, key_prefix=f"tts_queue_{msg_id}"
        )
        if reached_terminal_sync or reached_terminal_ui:
            st.rerun()
        elif changed and current_msg.get("tts_status") == "failed":
            st.rerun()

    _tts_fragment()


# ============================================================
# 5. KHỞI TẠO SESSION STATE & KIỂM TRA DB
# ============================================================
state_defaults: dict[str, Any] = {
    "messages": [],
    "last_search_sources": [],
    "active_article": None,
    "pinned_article": None,
    "pending_turn": None,
    "expanded_history_message_id": None,
    "toast_notice": None,
    "dialog_tts": None,
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

if st.session_state.toast_notice:
    st.toast(st.session_state.toast_notice)
    st.session_state.toast_notice = None


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

    if st.session_state.selected_voice not in VOICE_OPTIONS.values():
        st.session_state.selected_voice = DEFAULT_VOICE

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
        browse_filtered_news(speak_response=(selected_mode == MODES[2]))
        st.rerun()

    st.divider()
    if st.button("🗑️ Xóa lịch sử hội thoại", use_container_width=True):
        cancel_active_tts()
        st.session_state.messages = []
        st.session_state.last_search_sources = []
        st.session_state.active_article = None
        st.session_state.pinned_article = None
        st.session_state.pending_turn = None
        st.session_state.expanded_history_message_id = None
        st.session_state.dialog_tts = None
        reset_voice_buffer()
        st.rerun()

    with st.expander("ℹ️ Thông tin hệ thống"):
        st.metric("Tổng số bài báo", f"{total_articles:,}")
        st.caption("Tìm kiếm: SQLite FTS5 + PhoBERT + FAISS")
        st.caption("Tóm tắt: ViT5 (batched) | Hỏi đáp: Qwen2.5-1.5B (stream)")
        st.caption("Giọng nói: Google STT + Piper TTS cục bộ (offline)")


# Vị trí cố định cho fragment microphone khi dùng chế độ rảnh tay.
voice_fragment_slot = st.container()


# ============================================================
# 7. MÀN HÌNH CHÀO KHI CHƯA CÓ HỘI THOẠI
# ============================================================
if not st.session_state.messages and st.session_state.pending_turn is None:
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
                queue_user_turn(prompt, speak_response=(selected_mode == MODES[2]))
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
                queue_user_turn(cat_prompt, speak_response=(selected_mode == MODES[2]))
                st.rerun()


# ============================================================
# 8. LỊCH SỬ HỘI THOẠI & XỬ LÝ LƯỢT ĐANG CHỜ
# ============================================================
latest_source_msg_id: str | None = None
for msg in reversed(st.session_state.messages):
    if msg.get("role") == "assistant" and msg.get("sources"):
        latest_source_msg_id = msg.get("id")
        break

for message in st.session_state.messages:
    if "id" not in message:
        message["id"] = str(uuid.uuid4())

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            _render_message_audio_controls(message)
            render_sources(
                message.get("sources", []),
                message_key=message["id"],
                is_latest_source_message=(message["id"] == latest_source_msg_id),
            )

if st.session_state.pending_turn is not None:
    _execute_pending_turn(st.session_state.pending_turn)
    st.rerun()


# ============================================================
# 9. BANNER BÀI ĐANG GHIM, GỢI Ý NỐI TIẾP & KHUNG NHẬP LIỆU
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

# Contextual quick follow-up chips right above the input bar.
focus_article = st.session_state.pinned_article or st.session_state.active_article
search_list = st.session_state.last_search_sources
default_speak = selected_mode == MODES[2]

if focus_article is not None:
    focus_title = _short_title(focus_article.get("title") or "bài báo này", 42)
    st.caption(f"💬 Gợi ý hỏi nối tiếp cho: **{focus_title}**")
    chip_cols = st.columns(4)
    with chip_cols[0]:
        if st.button("✨ Tóm tắt bài này", key="chip_focus_sum", use_container_width=True):
            queue_user_turn(
                "Tóm tắt bài này",
                speak_response=default_speak,
                target_article=focus_article,
                target_action="summarize",
            )
            st.rerun()
    with chip_cols[1]:
        if st.button("📖 Đọc toàn văn", key="chip_focus_detail", use_container_width=True):
            queue_user_turn(
                "Xem toàn văn bài này",
                speak_response=default_speak,
                target_article=focus_article,
                target_action="detail",
            )
            st.rerun()
    with chip_cols[2]:
        if st.button("❓ Điểm chính là gì?", key="chip_focus_qa", use_container_width=True):
            queue_user_turn(
                "Ý chính và các thông tin quan trọng nhất trong bài này là gì?",
                speak_response=default_speak,
                target_article=focus_article,
                target_action="qa",
            )
            st.rerun()
    with chip_cols[3]:
        if len(search_list) > 1:
            if st.button("🔙 Về danh sách", key="chip_back_to_list", use_container_width=True):
                st.session_state.active_article = None
                st.session_state.pinned_article = None
                st.rerun()
        elif st.session_state.pinned_article is not None:
            if st.button("✕ Bỏ ghim", key="chip_unpin", use_container_width=True):
                st.session_state.pinned_article = None
                st.rerun()
elif search_list:
    st.caption("💬 Chọn nhanh bài báo trong danh sách vừa tìm:")
    chip_specs: list[tuple[str, str, dict[str, Any], str]] = [
        ("✨ Tóm tắt bài 1", f"Tóm tắt bài số 1: {_short_title(search_list[0].get('title', ''))}", search_list[0], "summarize"),
    ]
    if len(search_list) >= 2:
        chip_specs.append(
            ("✨ Tóm tắt bài 2", f"Tóm tắt bài số 2: {_short_title(search_list[1].get('title', ''))}", search_list[1], "summarize")
        )
    if len(search_list) >= 3:
        chip_specs.append(
            ("✨ Tóm tắt bài 3", f"Tóm tắt bài số 3: {_short_title(search_list[2].get('title', ''))}", search_list[2], "summarize")
        )
    chip_specs.append(
        ("📖 Đọc bài 1", f"Đọc toàn văn bài số 1: {_short_title(search_list[0].get('title', ''))}", search_list[0], "detail")
    )
    chip_cols = st.columns(len(chip_specs))
    for idx, (col, (chip_label, chip_prompt, chip_art, chip_action)) in enumerate(zip(chip_cols, chip_specs)):
        with col:
            if st.button(chip_label, key=f"chip_list_{idx}", use_container_width=True):
                st.session_state.active_article = chip_art
                queue_user_turn(
                    chip_prompt,
                    speak_response=default_speak,
                    target_article=chip_art,
                    target_action=chip_action,
                )
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
        queue_user_turn(
            typed_question,
            speak_response=bool(st.session_state.auto_read_push_to_talk),
        )
        st.rerun()
    else:
        try:
            with st.spinner("Đang nhận dạng giọng nói tiếng Việt..."):
                recognized = transcribe_audio(recorded_audio.getvalue())
            queue_user_turn(
                recognized,
                speak_response=bool(st.session_state.auto_read_push_to_talk),
            )
            st.rerun()
        except SpeechRecognitionError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Không thể nhận dạng bản ghi âm: {exc}")
elif typed_question:
    queue_user_turn(
        typed_question,
        speak_response=(selected_mode == MODES[2]),
    )
    st.rerun()


# ============================================================
# 10. LUỒNG MICROPHONE LIÊN TỤC (KHI Ở CHẾ ĐỘ RẢNH TAY)
# ============================================================
if selected_mode == MODES[2] and voice_ctx is not None:
    voice_poll_interval = VOICE_POLL_INTERVAL if voice_ctx.state.playing else None

    @st.fragment(run_every=voice_poll_interval)
    def poll_microphone() -> None:
        tts_active = any(
            msg.get("role") == "assistant" and msg.get("tts_status") in TTS_ACTIVE_STATES
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
        except (queue.Empty, Exception):
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
            queue_user_turn(recognized_text, speak_response=True)
            st.rerun()

    with voice_fragment_slot:
        poll_microphone()
