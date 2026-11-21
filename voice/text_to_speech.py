"""Vietnamese Text-to-Speech helpers backed by local offline Piper TTS."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

from voice.tts_adapter import (
    DEFAULT_VOICE,
    FALLBACK_VOICE,
    MAX_CHUNK_CHARS,
    VOICE_OPTIONS,
    synthesize_text,
)


class TextToSpeechError(RuntimeError):
    """Raised when text cannot be converted to audio."""


def clean_text_for_speech(text: str) -> str:
    """Remove common Markdown decoration while keeping the spoken wording."""
    text = (text or "").strip()
    text = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```", r"\1", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[*_~]{1,3}", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text_for_tts(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split long text at sentence/word boundaries."""
    from voice.tts_adapter import _split_long_text

    if max_chars < 40:
        raise ValueError("max_chars nên từ 40 ký tự trở lên.")
    cleaned = clean_text_for_speech(text)
    if not cleaned:
        return []
    return _split_long_text(cleaned, max_chars=max_chars)


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
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[bytes]:
    """Return WAV segments synthesized locally in memory via Piper TTS."""
    cleaned = clean_text_for_speech(text)
    if not cleaned:
        return []

    segments: list[bytes] = []
    result = _run_async(
        synthesize_text(
            cleaned,
            voice=voice,
            rate=rate,
            max_chunk_chars=max_chars,
            segment_callback=lambda audio_bytes, _meta: segments.append(audio_bytes),
        )
    )
    if not segments and not result.audio_bytes:
        raise TextToSpeechError("Không thể tạo giọng đọc cục bộ bằng Piper TTS.")
    return segments


async def text_to_speech(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    volume: str = "+0%",
) -> bytes:
    """Synthesize text to combined in-memory WAV bytes using local Piper TTS."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Văn bản đầu vào không được để trống.")

    result = await synthesize_text(
        clean_text_for_speech(text) or text.strip(),
        voice=voice,
        rate=rate,
        volume=volume,
    )
    if not result.audio_bytes:
        raise TextToSpeechError("Piper TTS không tạo được dữ liệu âm thanh WAV.")
    return result.audio_bytes
