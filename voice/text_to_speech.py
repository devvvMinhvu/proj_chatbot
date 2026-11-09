import asyncio
import edge_tts
from edge_tts.exceptions import NoAudioReceived


# Giữ giọng nữ làm mặc định.
DEFAULT_VOICE = "vi-VN-HoaiMyNeural"

# Giọng nam chỉ được dùng dự phòng nếu Edge TTS không trả audio cho Hoài My.
FALLBACK_VOICE = "vi-VN-NamMinhNeural"

# Hạn chế số lần gọi để tránh gửi quá nhiều yêu cầu liên tiếp lên dịch vụ.
PRIMARY_RETRY_DELAY = 4.0
FALLBACK_RETRY_DELAY = 3.0


async def _synthesize_once(
    text: str,
    voice: str,
    rate: str,
    volume: str,
) -> bytes:
    """Tạo một lượt audio MP3 bằng Edge TTS."""
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        volume=volume,
    )

    audio_data = bytearray()

    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            audio_data.extend(chunk.get("data", b""))

    if not audio_data:
        raise NoAudioReceived(
            "Edge TTS kết thúc nhưng không trả về dữ liệu âm thanh."
        )

    return bytes(audio_data)


async def text_to_speech(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    volume: str = "+0%",
) -> bytes:
    """
    Chuyển văn bản thành MP3.

    Luồng xử lý:
    1. Thử giọng được yêu cầu.
    2. Nếu nhận NoAudioReceived, chờ rồi thử lại một lần.
    3. Nếu giọng mặc định Hoài My vẫn lỗi, thử một lần bằng Nam Minh.

    Hàm giữ nguyên giao diện gọi hiện tại: await text_to_speech(text).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Văn bản đầu vào không được để trống.")

    try:
        return await _synthesize_once(text, voice, rate, volume)
    except NoAudioReceived as first_error:
        print(
            f"[TTS] Giọng {voice} không trả audio ở lần 1. "
            f"Thử lại sau {PRIMARY_RETRY_DELAY:.0f} giây."
        )
        await asyncio.sleep(PRIMARY_RETRY_DELAY)

        try:
            return await _synthesize_once(text, voice, rate, volume)
        except NoAudioReceived as second_error:
            # Chỉ tự chuyển giọng khi đang sử dụng giọng nữ mặc định.
            if voice != DEFAULT_VOICE:
                raise RuntimeError(
                    f"Edge TTS không trả audio sau 2 lần thử với giọng {voice}."
                ) from second_error

            print(
                f"[TTS] Giọng Hoài My tiếp tục lỗi. "
                f"Chuyển sang giọng dự phòng {FALLBACK_VOICE}."
            )
            await asyncio.sleep(FALLBACK_RETRY_DELAY)

            try:
                audio = await _synthesize_once(
                    text,
                    FALLBACK_VOICE,
                    rate,
                    volume,
                )
                print(f"[TTS] Đã tạo audio bằng giọng dự phòng {FALLBACK_VOICE}.")
                return audio
            except NoAudioReceived as fallback_error:
                raise RuntimeError(
                    "Edge TTS không trả audio cho cả hai giọng. "
                    f"Giọng chính {voice}: {second_error!r}; "
                    f"giọng dự phòng {FALLBACK_VOICE}: {fallback_error!r}. "
                    "Có thể dịch vụ Edge TTS hoặc kết nối mạng đang gặp sự cố."
                ) from fallback_error
