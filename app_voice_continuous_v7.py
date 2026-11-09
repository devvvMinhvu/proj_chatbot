import asyncio
import io
import queue
import re
import time
import unicodedata
import uuid
import wave

import av
import numpy as np
import streamlit as st
from mutagen.mp3 import MP3
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from rag.pipeline_15k import load_vector_database
from nlp.intent_handler import get_article_content, handle_intent
from voice.speech_to_text import transcribe_wav, SpeechToTextError
from voice.tts_adapter import synthesize_text
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
st.caption("Hệ thống hỏi đáp báo chí ứng dụng NLP và RAG — Voice v7 — separate spoken script + adaptive TTS")

SAMPLE_RATE = 16000
VAD_RMS_THRESHOLD = 450       # Tăng nếu mic nhận tiếng ồn; giảm nếu giọng nói quá nhỏ.
END_SILENCE_SECONDS = 0.85   # Khoảng lặng để xác định người dùng đã nói xong.
MIN_SPEECH_SECONDS = 0.35
MAX_UTTERANCE_SECONDS = 15.0
VOICE_POLL_INTERVAL = 0.25


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


def get_mp3_duration(audio_bytes: bytes, text: str) -> float:
    """Đọc thời lượng MP3; dùng ước lượng dự phòng nếu metadata không đọc được."""
    try:
        return float(MP3(io.BytesIO(audio_bytes)).info.length)
    except Exception:
        # Ước lượng bảo thủ: khoảng 2.5 từ/giây.
        return max(2.0, len(text.split()) / 2.5)


def reset_voice_buffer() -> None:
    st.session_state.voice_chunks = []
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


def split_text_for_tts(text: str, max_chars: int = 450) -> list[str]:
    """Chia văn bản dài thành các đoạn ngắn, ưu tiên ranh giới câu và dấu câu."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentence_parts = re.split(r"(?<=[.!?])\s+", text)
    units = []
    for sentence in sentence_parts:
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue

        subparts = re.split(r"(?<=[,;:])\s+", sentence)
        for subpart in subparts:
            if len(subpart) <= max_chars:
                units.append(subpart)
                continue

            words = subpart.split()
            current = ""
            for word in words:
                if len(word) > max_chars:
                    if current:
                        units.append(current)
                        current = ""
                    for start in range(0, len(word), max_chars):
                        units.append(word[start:start + max_chars])
                    continue

                candidate = f"{current} {word}".strip()
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        units.append(current)
                    current = word

            if current:
                units.append(current)

    chunks = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = unit
    if current:
        chunks.append(current)

    return chunks


async def text_to_speech_in_chunks(text: str) -> tuple[bytes, float, list[str]]:
    """Tạo audio thích ứng; nếu một cụm lỗi thì tách nhỏ thay vì bỏ toàn bộ."""
    result = await synthesize_text(
        text,
        voice="vi-VN-HoaiMyNeural",
        fallback_voice="vi-VN-NamMinhNeural",
        max_chunk_chars=420,
    )
    return result.audio_bytes, result.duration_seconds, result.failed_segments


def append_user_and_assistant(question: str) -> None:
    """Xử lý một lượt chat và gắn audio trực tiếp vào tin nhắn assistant."""
    # Lượt trước được xem là kết thúc khi người dùng gửi câu hỏi mới.
    # Xóa audio cũ để Session State không giữ quá nhiều MP3.
    for old_message in st.session_state.messages:
        if old_message.get("role") == "assistant":
            old_message["tts_autoplay"] = False
            old_message.pop("tts_audio", None)

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
        "tts_autoplay": False,
    }

    speech_text = speech_friendly_text(answer)
    if speech_text:
        print(
            f"[TTS SCRIPT] Văn bản hiển thị: {len(answer)} ký tự; "
            f"văn bản đọc: {len(speech_text)} ký tự."
        )
        try:
            # Chỉ tổng hợp văn bản đọc riêng; answer gốc vẫn giữ nguyên trong chat.
            audio_bytes, playback_seconds, failed_segments = asyncio.run(
                text_to_speech_in_chunks(speech_text)
            )

            if audio_bytes:
                assistant_message["tts_audio"] = audio_bytes
                assistant_message["tts_autoplay"] = True
                st.session_state.voice_mute_until = (
                    time.monotonic() + playback_seconds + 2.0
                )

                if failed_segments:
                    # Giữ nguyên câu trả lời gốc. Chỉ cảnh báo phần chưa đọc được;
                    # các đoạn MP3 thành công vẫn được phát bình thường.
                    failed_preview = "; ".join(failed_segments[:5])
                    assistant_message["tts_warning"] = (
                        "Âm thanh chỉ đọc được một phần. Các cụm chưa tổng hợp được: "
                        f"{failed_preview}"
                    )
            else:
                assistant_message["tts_error"] = (
                    "TTS chưa tạo được âm thanh. Câu trả lời văn bản vẫn được giữ."
                )
                if failed_segments:
                    assistant_message["tts_warning"] = (
                        "Các cụm chưa đọc được: " + "; ".join(failed_segments[:5])
                    )
        except Exception as error:
            print("Lỗi tạo giọng nói:", repr(error))
            assistant_message["tts_error"] = str(error)

    st.session_state.messages.append(assistant_message)


# ============================================================
# 3. KHỞI TẠO SESSION STATE
# ============================================================
state_defaults = {
    "messages": [],
    "voice_chunks": [],
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


# ============================================================
# 5. THANH BÊN
# ============================================================
with st.sidebar:
    st.header("Thông tin hệ thống")
    st.write("**Embedding:** PhoBERT")
    st.write("**Vector store:** FAISS")
    st.write("**LLM:** Qwen2.5-1.5B")
    st.write("**Dữ liệu:** 15.000 bài báo")
    st.metric("Tổng số vector", f"{total_vectors:,}")

    if st.button("Xóa lịch sử hội thoại"):
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

            # Player lấy bytes từ message nên không biến mất sau một lần full-app rerun.
            audio_bytes = message.get("tts_audio")
            if audio_bytes:
                st.audio(
                    audio_bytes,
                    format="audio/mpeg",
                    autoplay=bool(message.get("tts_autoplay", False)),
                )

            if message.get("tts_error"):
                st.caption("Chưa phát được giọng nói cho câu trả lời này.")
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
voice_poll_interval = VOICE_POLL_INTERVAL if voice_ctx.state.playing else None


@st.fragment(run_every=voice_poll_interval)
def poll_microphone():
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

            if is_speech and not st.session_state.voice_is_speaking:
                st.session_state.voice_is_speaking = True
                st.session_state.voice_chunks = []
                st.session_state.voice_silence_seconds = 0.0
                st.session_state.voice_speech_seconds = 0.0
                st.session_state.voice_total_seconds = 0.0

            if not st.session_state.voice_is_speaking:
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


# Gọi fragment trong container có vị trí cố định, không phụ thuộc số tin nhắn
# hoặc số nguồn bài báo đang hiển thị trong khung chat.
with voice_fragment_slot:
    poll_microphone()
