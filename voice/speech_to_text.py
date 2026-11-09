
import io
import speech_recognition as sr


class SpeechToTextError(Exception):
    """Lỗi trong quá trình chuyển giọng nói thành văn bản."""
    pass


def transcribe_wav(
    audio_bytes: bytes,
    language: str = "vi-VN"
) -> str:
    """
    Chuyển dữ liệu âm thanh WAV thành văn bản.

    Args:
        audio_bytes: Dữ liệu âm thanh dạng bytes.
        language: Mã ngôn ngữ nhận dạng.

    Returns:
        Văn bản được nhận dạng.
    """

    if not audio_bytes:
        raise SpeechToTextError("Không có dữ liệu âm thanh.")

    recognizer = sr.Recognizer()

    # Bước 1: Đọc dữ liệu WAV
    try:
        audio_file = io.BytesIO(audio_bytes)

        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)

    except (ValueError, OSError) as error:
        raise SpeechToTextError(
            "Dữ liệu âm thanh không hợp lệ. Cần file WAV."
        ) from error

    # Bước 2: Gửi âm thanh đi nhận dạng
    try:
        text = recognizer.recognize_google(
            audio_data,
            language=language
        )

        text = text.strip()

        if not text:
            raise SpeechToTextError(
                "Không nhận được nội dung văn bản."
            )

        return text

    except sr.UnknownValueError:
        raise SpeechToTextError(
            "Không nhận dạng được giọng nói. "
            "Hãy nói rõ hơn và thử lại."
        ) from None

    except sr.RequestError as error:
        raise SpeechToTextError(
            f"Không thể kết nối dịch vụ nhận dạng: {error}"
        ) from error
