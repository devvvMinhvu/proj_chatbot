import base64
import io
from pathlib import Path
import queue
import re
import time
import unicodedata
import uuid
import wave

import av
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from database.database_15k import Article, SessionLocal
from nlp.intent_handler import get_article_content, handle_intent
from voice.speech_to_text import transcribe_wav, SpeechToTextError
from voice.tts_background import (
    submit_tts_task,
    poll_tts_events,
    release_tts_task,
    cancel_tts_task,
)
from voice.reading_text import build_reading_text


# ============================================================
# 1. CẤU HÌNH
# ============================================================
st.set_page_config(
    page_title="Trợ lý đọc báo thông minh",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Trợ lý đọc báo thông minh")
st.caption("Hệ thống hỏi đáp báo chí ứng dụng NLP và RAG — Voice v8 ổn định")

SAMPLE_RATE = 16000
VAD_RMS_THRESHOLD = 450       # Tăng nếu mic nhận tiếng ồn; giảm nếu giọng nói quá nhỏ.
END_SILENCE_SECONDS = 0.85   # Khoảng lặng để xác định người dùng đã nói xong.
MIN_SPEECH_SECONDS = 0.35
MAX_UTTERANCE_SECONDS = 15.0
PRE_ROLL_FRAMES = 8  # Giữ khoảng 120–200 ms âm thanh trước khi VAD nhận ra tiếng nói.
VOICE_POLL_INTERVAL = 0.25


@st.cache_resource
def get_tts_audio_queue_component():
    """Load the browser-side player that queues MP3 segments sequentially."""
    component_dir = Path(__file__).resolve().parent / "voice" / "audio_queue_component"
    if not (component_dir / "index.html").is_file():
        raise FileNotFoundError(
            f"Không tìm thấy audio queue component: {component_dir / 'index.html'}"
        )
    return components.declare_component(
        "news_assistant_tts_audio_queue_v8",
        path=str(component_dir),
    )


# ============================================================
# 2. HÀM TIỆN ÍCH ÂM THANH
# ============================================================
def make_wav_bytes(samples: np.ndarray) -> bytes:
    """Đóng gói mảng PCM mono 16-bit, 16 kHz thành WAV bytes."""
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
                content = get_article_content(article_id=article_id, url=url)

                if content is None:
                    st.session_state.pop(content_key, None)
                    st.warning("Không tìm thấy toàn văn bài báo trong database.")
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


def normalize_text(text: str) -> str:
    """Chuẩn hóa tiếng Việt để nhận diện lời chào/giới thiệu đơn giản."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").strip()


def get_local_response(question: str):
    """Xử lý lời chào và yêu cầu giới thiệu về chính trợ lý."""
    normalized = normalize_text(question)

    greetings = {
        "chao", "chao ban", "xin chao", "xin chao ban", "hello", "hi",
        "hey", "good morning", "good evening",
    }
    if normalized in greetings or normalized.startswith("xin chao "):
        return (
            "Xin chào! Mình là Trợ lý đọc báo thông minh. "
            "Bạn có thể hỏi mình về tin tức, tìm bài báo hoặc yêu cầu tóm tắt nội dung."
        ), []

    intro_phrases = (
        "gioi thieu", "gioi thieu ve ban", "gioi thieu ban",
        "gioi thieu he thong", "gioi thieu ve he thong",
        "gioi thieu tro ly", "ban la ai", "tro ly nay la gi",
        "he thong nay la gi",
    )
    if normalized in intro_phrases or any(
        phrase in normalized for phrase in (
            "gioi thieu ve ban", "gioi thieu ve he thong",
            "gioi thieu tro ly doc bao", "ban la ai",
        )
    ):
        return (
            "Mình là Trợ lý đọc báo thông minh, một hệ thống hỗ trợ tìm kiếm và hỏi đáp "
            "trên dữ liệu bài báo. Bạn có thể yêu cầu tìm tin theo chủ đề, xem nguồn bài báo, "
            "đọc nội dung hoặc đặt câu hỏi về các bài báo. Hệ thống sử dụng PhoBERT để biểu diễn "
            "văn bản, FAISS để tìm kiếm ngữ nghĩa và Qwen để hỗ trợ tạo câu trả lời. "
            "Bạn có thể nhập câu hỏi bằng bàn phím hoặc nói trực tiếp qua microphone."
        ), []

    return None


def speech_friendly_text(text: str) -> str:
    """Tạo văn bản đọc riêng; không chỉnh sửa nội dung hiển thị trong chat."""
    return build_reading_text(text)


def append_user_and_assistant(question: str) -> None:
    """Process a chat turn, render its text immediately, then synthesize in background."""
    for old_message in st.session_state.messages:
        if old_message.get("role") != "assistant":
            continue

        old_message["tts_autoplay"] = False
        old_task_id = old_message.get("tts_task_id")
        if old_task_id and old_message.get("tts_status") in {
            "queued", "generating", "playing", "fallback_playing"
        }:
            cancel_tts_task(old_task_id)
            old_message["tts_status"] = "cancelled"
        old_message.pop("tts_segments", None)
        old_message.pop("tts_combined_audio", None)

    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": question,
    }
    st.session_state.messages.append(user_message)

    try:
        local_response = get_local_response(question)
        if local_response is not None:
            answer, sources = local_response
        else:
            answer, sources = handle_intent(question)
    except Exception as error:
        print("Lỗi xử lý yêu cầu:", repr(error))
        if getattr(error, "winerror", None) == 1455:
            answer = (
                "Windows chưa đủ bộ nhớ ảo để nạp PyTorch/PhoBERT. "
                "Hãy tăng Paging file trong Windows, kiểm tra dung lượng ổ C "
                "và thử lại. Các chức năng tra cứu trực tiếp SQLite vẫn có thể hoạt động."
            )
        else:
            answer = (
                "Đã xảy ra lỗi khi xử lý yêu cầu. "
                "Bạn hãy kiểm tra terminal để xem chi tiết."
            )
        sources = []

    assistant_message = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "tts_autoplay": True,
        "tts_status": "queued",
        "tts_segments": [],
        "tts_task_id": str(uuid.uuid4()),
        "tts_progress": {"completed_chunks": 0, "total_chunks": 0},
    }
    st.session_state.messages.append(assistant_message)

    speech_text = speech_friendly_text(answer)
    if not speech_text:
        assistant_message["tts_status"] = "skipped"
        assistant_message["tts_autoplay"] = False
        return

    print(
        f"[TTS SCRIPT] Văn bản hiển thị: {len(answer)} ký tự; "
        f"văn bản đọc: {len(speech_text)} ký tự."
    )

    try:
        submit_tts_task(
            assistant_message["tts_task_id"],
            speech_text,
            voice="vi-VN-HoaiMyNeural",
            fallback_voice="vi-VN-NamMinhNeural",
            max_chunk_chars=420,
        )
    except Exception as error:
        print("Không thể khởi chạy tác vụ TTS nền:", repr(error))
        assistant_message["tts_status"] = "failed"
        assistant_message["tts_error"] = str(error)
        assistant_message["tts_autoplay"] = False


# ============================================================
# 3. KHỞI TẠO SESSION STATE
# ============================================================
state_defaults = {
    "messages": [],
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


# ============================================================
# 4. NẠP VECTOR DATABASE
# ============================================================
@st.cache_data
def initialize_system():
    """Đếm bài báo trong SQLite; không nạp PhoBERT/FAISS khi mở ứng dụng."""
    db = SessionLocal()
    try:
        return int(db.query(Article).count())
    finally:
        db.close()


try:
    total_articles = initialize_system()
except Exception as error:
    st.error(f"Không thể kết nối cơ sở dữ liệu bài báo: {error}")
    st.stop()


# ============================================================
# 5. THANH BÊN
# ============================================================
with st.sidebar:
    st.header("Thông tin hệ thống")
    st.write("**Embedding:** PhoBERT (nạp khi cần)")
    st.write("**Vector store:** FAISS (nạp khi cần)")
    st.write("**LLM:** Qwen2.5-1.5B")
    st.write("**Dữ liệu:** SQLite")
    st.metric("Số bài báo trong database", f"{total_articles:,}")
    st.caption(
        "Các chức năng tìm theo chuyên mục/chủ đề và tin mới nhất truy vấn SQLite. "
        "PhoBERT/FAISS chỉ được nạp khi cần tìm kiếm ngữ nghĩa, tóm tắt hoặc RAG."
    )

    if st.button("Xóa lịch sử hội thoại"):
        for old_message in st.session_state.messages:
            old_task_id = old_message.get("tts_task_id")
            if old_task_id:
                cancel_tts_task(old_task_id)
                release_tts_task(old_task_id)
        st.session_state.messages = []
        reset_voice_buffer()
        st.rerun()


# ============================================================
# 5.1. VỊ TRÍ CỐ ĐỊNH CHO FRAGMENT MICROPHONE
# ============================================================
# Streamlit tạo fragment ID dựa một phần vào vị trí (Delta path) của nó.
# Không đặt lời gọi poll_microphone() sau lịch sử chat có độ dài thay đổi,
# nếu không ID có thể đổi sau mỗi lần thêm tin nhắn và timer gọi ID cũ.
# Container này được tạo ở vị trí cố định trước khi render lịch sử chat.
voice_fragment_slot = st.container()


# ============================================================
# 6. HIỂN THỊ LỊCH SỬ HỘI THOẠI
# ============================================================
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
                        message["tts_playback_finished"] = True
                        st.session_state.voice_mute_until = 0.0
                except Exception as error:
                    print("Không thể hiển thị audio queue:", repr(error))
                    message["tts_component_failed"] = True
                    combined_audio = message.get("tts_combined_audio")
                    if combined_audio:
                        message["tts_status"] = "fallback_playing"
                    else:
                        st.warning("Không tải được trình phát audio. Nội dung văn bản vẫn khả dụng.")

            tts_status = message.get("tts_status")
            if message.get("tts_component_failed") and message.get("tts_combined_audio"):
                st.audio(
                    message["tts_combined_audio"],
                    format="audio/mpeg",
                    autoplay=True,
                )

            if tts_status in {"queued", "generating"}:
                progress = message.get("tts_progress") or {}
                completed = int(progress.get("completed_chunks", 0))
                total = int(progress.get("total_chunks", 0))
                if total:
                    st.caption(f"Đang chuẩn bị giọng đọc: {completed}/{total} đoạn đã xử lý...")
                else:
                    st.caption("Đang chuẩn bị giọng đọc...")
            elif tts_status == "playing":
                st.caption("Đang đọc câu trả lời. Microphone tạm dừng để tránh thu lại tiếng loa.")
            elif tts_status == "fallback_playing":
                st.caption("Đang phát bằng trình phát âm thanh dự phòng.")
            elif tts_status == "finished":
                st.caption("Đã đọc xong câu trả lời.")
            elif tts_status in {"failed", "skipped"} and message.get("tts_error"):
                st.caption("Chưa tạo được giọng nói cho câu trả lời này.")

            if message.get("tts_error"):
                st.caption(f"Chi tiết TTS: {message['tts_error']}")
            if message.get("tts_warning"):
                st.warning(message["tts_warning"])


# ============================================================
# 7. Ô CHAT + NÚT MICROPHONE CẠNH Ô NHẬP
# ============================================================
chat_col, mic_col = st.columns([13, 2], vertical_alignment="bottom")

with chat_col:
    question = st.chat_input(
        "Nhập câu hỏi về tin tức hoặc dùng microphone...",
        key="main_chat_input",
    )

with mic_col:
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

if question:
    append_user_and_assistant(question)
    st.rerun()


# ============================================================
# 8. XỬ LÝ LUỒNG MICROPHONE LIÊN TỤC
# ============================================================
tts_tasks_pending = any(
    message.get("role") == "assistant"
    and message.get("tts_status") in {"queued", "generating", "playing", "fallback_playing"}
    for message in st.session_state.messages
)
voice_poll_interval = (
    VOICE_POLL_INTERVAL
    if (voice_ctx.state.playing or tts_tasks_pending)
    else None
)


def sync_tts_background_events() -> bool:
    """Move worker events into Streamlit session state; return whether UI changed."""
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

        events = poll_tts_events(task_id)
        for event in events:
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
                        "api_calls": event.get("api_calls", 0),
                        "cache_hits": event.get("cache_hits", 0),
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
                        message["tts_segments"].append({
                            "id": f"{task_id}-{segment_index}",
                            "audio_b64": base64.b64encode(segment_bytes).decode("ascii"),
                            "duration_seconds": float(event.get("duration_seconds", 0.0)),
                        })
                        message["tts_status"] = "generating"
                        changed = True

            elif event_type == "completed":
                result = event.get("result")
                if message.get("tts_status") != "cancelled" and result is not None:
                    message["tts_duration_seconds"] = float(result.duration_seconds)
                    message["tts_api_calls"] = int(result.api_calls)
                    message["tts_cache_hits"] = int(result.cache_hits)
                    if result.audio_bytes:
                        message["tts_combined_audio"] = result.audio_bytes
                    if message.get("tts_segments"):
                        message["tts_status"] = (
                            "fallback_playing"
                            if message.get("tts_component_failed")
                            else "playing"
                        )
                    else:
                        message["tts_status"] = "failed"
                    if result.failed_segments:
                        preview = "; ".join(result.failed_segments[:4])
                        message["tts_warning"] = (
                            "Một số phần không tạo được âm thanh; các phần thành công vẫn được phát. "
                            f"Phần lỗi: {preview}"
                        )
                    if message.get("tts_segments"):
                        first_at = float(message.get("tts_first_segment_at", time.monotonic()))
                        elapsed = max(0.0, time.monotonic() - first_at)
                        remaining = max(0.0, result.duration_seconds - elapsed)
                        st.session_state.voice_mute_until = time.monotonic() + remaining + 1.5
                    else:
                        message["tts_error"] = (
                            "Edge TTS không tạo được đoạn âm thanh nào. "
                            "Câu trả lời văn bản vẫn được giữ."
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


@st.fragment(run_every=voice_poll_interval)
def poll_microphone():
    if sync_tts_background_events():
        st.rerun()

    # Pause microphone capture while TTS is being generated or played. Drain
    # WebRTC frames so the receiver queue does not grow without bound.
    tts_active = any(
        message.get("role") == "assistant"
        and message.get("tts_status") in {"queued", "generating", "playing"}
        for message in st.session_state.messages
    )

    if tts_active:
        if voice_ctx.state.playing and voice_ctx.audio_receiver is not None:
            try:
                voice_ctx.audio_receiver.get_frames(timeout=0.01)
            except Exception:
                pass
        reset_voice_buffer()
        return

    if not voice_ctx.state.playing:
        return

    receiver = voice_ctx.audio_receiver
    if receiver is None:
        return

    try:
        audio_frames = receiver.get_frames(timeout=0.05)
    except queue.Empty:
        return
    except Exception as error:
        print("Không đọc được luồng microphone:", repr(error))
        return

    # Vẫn lấy frame khỏi hàng đợi nhưng bỏ qua trong lúc chatbot đang đọc.
    if time.monotonic() < st.session_state.voice_mute_until:
        reset_voice_buffer()
        return

    if not audio_frames:
        return

    resampler = av.AudioResampler(
        format="s16",
        layout="mono",
        rate=SAMPLE_RATE,
    )

    completed_wav = None

    for input_frame in audio_frames:
        try:
            output_frames = resampler.resample(input_frame)
        except Exception as error:
            print("Lỗi chuyển đổi frame âm thanh:", repr(error))
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
                    # Giữ vài frame im lặng trước tiếng nói để tránh cắt mất âm đầu.
                    pre_roll = st.session_state.voice_pre_roll
                    pre_roll.append(samples.copy())
                    if len(pre_roll) > PRE_ROLL_FRAMES:
                        pre_roll.pop(0)
                    continue

                pre_roll = st.session_state.voice_pre_roll
                st.session_state.voice_is_speaking = True
                st.session_state.voice_chunks = list(pre_roll) + [samples.copy()]
                st.session_state.voice_total_seconds = (
                    sum(len(part) for part in pre_roll) / SAMPLE_RATE
                    + frame_seconds
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

            silence_finished = (
                st.session_state.voice_silence_seconds >= END_SILENCE_SECONDS
            )
            utterance_too_long = (
                st.session_state.voice_total_seconds >= MAX_UTTERANCE_SECONDS
            )

            if silence_finished or utterance_too_long:
                speech_seconds = st.session_state.voice_speech_seconds
                chunks = st.session_state.voice_chunks
                reset_voice_buffer()

                if speech_seconds >= MIN_SPEECH_SECONDS and chunks:
                    completed_samples = np.concatenate(chunks)
                    completed_wav = make_wav_bytes(completed_samples)
                break

        if completed_wav is not None:
            break

    if completed_wav is None:
        return

    try:
        recognized_text = transcribe_wav(completed_wav).strip()
    except SpeechToTextError as error:
        st.toast(str(error), icon="⚠️")
        return
    except Exception as error:
        print("Lỗi STT:", repr(error))
        st.toast("Không nhận dạng được giọng nói. Hãy thử lại.", icon="⚠️")
        return

    if not recognized_text:
        return

    append_user_and_assistant(recognized_text)
    st.rerun()


# Poll cả tác vụ TTS nền và microphone trong container có vị trí cố định.
with voice_fragment_slot:
    poll_microphone()
