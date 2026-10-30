"""Vietnamese Text-to-Speech using Edge TTS."""
from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

DEFAULT_VOICE = "vi-VN-HoaiMyNeural"
VOICE_OPTIONS = {
    "Nữ - Hoài My": "vi-VN-HoaiMyNeural",
    "Nam - Nam Minh": "vi-VN-NamMinhNeural",
}


class TextToSpeechError(RuntimeError):
    """Raised when text cannot be converted to audio."""


def clean_text_for_speech(text: str) -> str:
    """Remove common Markdown decoration while keeping the spoken wording."""
    text = (text or "").strip()
    # Keep the text inside fenced code blocks; remove only the fence/language tag.
    text = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```", r"\1", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[*_~]{1,3}", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text_for_tts(text: str, max_chars: int = 2500) -> list[str]:
    """Split long text at sentence/word boundaries."""
    if max_chars < 200:
        raise ValueError("max_chars nên từ 200 ký tự trở lên.")

    text = clean_text_for_speech(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?…])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            piece, sentence = sentence[:cut].strip(), sentence[cut:].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece)

        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks


async def _synthesize_one(text: str, voice: str, rate: str) -> bytes:
    try:
        import edge_tts
    except ImportError as exc:
        raise TextToSpeechError(
            "Thiếu edge-tts. Chạy: python -m pip install edge-tts"
        ) from exc

    audio_parts: list[bytes] = []
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        async for item in communicate.stream():
            if item.get("type") == "audio":
                audio_parts.append(item["data"])
    except Exception as exc:
        raise TextToSpeechError(
            "Không tạo được giọng đọc. Kiểm tra Internet hoặc thử lại."
        ) from exc

    audio = b"".join(audio_parts)
    if not audio:
        raise TextToSpeechError("Dịch vụ TTS không trả về dữ liệu âm thanh.")
    return audio


async def _synthesize_all(chunks: list[str], voice: str, rate: str) -> list[bytes]:
    result = []
    for chunk in chunks:
        result.append(await _synthesize_one(chunk, voice, rate))
    return result


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def synthesize_text_chunks(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    max_chars: int = 2500,
) -> list[bytes]:
    """Return MP3 parts for any UI/chat response or article text."""
    chunks = split_text_for_tts(text, max_chars=max_chars)
    return _run_async(_synthesize_all(chunks, voice, rate)) if chunks else []
