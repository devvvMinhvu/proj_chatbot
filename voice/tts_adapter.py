"""Fast offline Vietnamese Text-to-Speech adapter powered by Piper ONNX.

Synthesizes WAV audio purely in memory (zero disk writes for audio) with a
bounded in-memory LRU cache, fast first-sentence chunking, and fallback voice
recovery.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import io
from pathlib import Path
import re
import threading
from typing import Any, Callable, Optional
import wave

import numpy as np

from config import PIPER_MODELS_DIR

DEFAULT_VOICE = "vi_VN-vais1000-medium"
FALLBACK_VOICE = "vi_VN-25hours_single-low"
VOICE_OPTIONS = {
    "Nữ - VAIS1000 (Chất lượng cao)": "vi_VN-vais1000-medium",
    "Nữ - 25Hours (Nhanh)": "vi_VN-25hours_single-low",
}
DEFAULT_RATE = "+0%"
DEFAULT_VOLUME = "+0%"

FIRST_CHUNK_MAX_CHARS = 160
MAX_CHUNK_CHARS = 380
AUDIO_MIME_TYPE = "audio/wav"
MEMORY_CACHE_MAX_ENTRIES = 64
CACHE_VERSION = "piper-tts-v1"

PIPER_DIR = PIPER_MODELS_DIR

_MEMORY_CACHE: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
_MEMORY_CACHE_LOCK = threading.Lock()
_VOICE_LOAD_LOCK = threading.Lock()
_PIPER_SYNTH_LOCK = threading.Lock()


class TTSTaskCancelled(RuntimeError):
    """Raised when a newer user turn cancels an unfinished TTS task."""


@dataclass
class TTSResult:
    audio_bytes: bytes
    duration_seconds: float
    failed_segments: list[str] = field(default_factory=list)
    cache_hits: int = 0
    api_calls: int = 0
    fallback_segments: int = 0
    mime_type: str = AUDIO_MIME_TYPE

    @property
    def succeeded(self) -> bool:
        return bool(self.audio_bytes)

    @property
    def partial(self) -> bool:
        return bool(self.audio_bytes) and bool(self.failed_segments)


def _cache_key(text: str, voice: str, rate: str, volume: str) -> str:
    payload = "\0".join((CACHE_VERSION, voice, rate, volume, text))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_memory_cache(
    text: str,
    voice: str,
    rate: str,
    volume: str,
) -> Optional[tuple[bytes, float]]:
    key = _cache_key(text, voice, rate, volume)
    with _MEMORY_CACHE_LOCK:
        entry = _MEMORY_CACHE.get(key)
        if entry is None:
            return None
        _MEMORY_CACHE.move_to_end(key)
        return entry


def _write_memory_cache(
    text: str,
    voice: str,
    rate: str,
    volume: str,
    data: bytes,
    duration_seconds: float,
) -> None:
    if not data or MEMORY_CACHE_MAX_ENTRIES <= 0:
        return
    key = _cache_key(text, voice, rate, volume)
    with _MEMORY_CACHE_LOCK:
        _MEMORY_CACHE[key] = (data, duration_seconds)
        _MEMORY_CACHE.move_to_end(key)
        while len(_MEMORY_CACHE) > MEMORY_CACHE_MAX_ENTRIES:
            _MEMORY_CACHE.popitem(last=False)


def clear_memory_cache() -> None:
    """Clear the bounded in-memory TTS audio cache."""
    with _MEMORY_CACHE_LOCK:
        _MEMORY_CACHE.clear()


def _rate_to_length_scale(rate: str) -> float:
    """Convert percentage rate strings like '+15%' or '-10%' to Piper length_scale."""
    match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%\s*$", str(rate or ""))
    if not match:
        return 1.0
    pct = float(match.group(1))
    scale = 1.0 / max(0.25, 1.0 + (pct / 100.0))
    return max(0.5, min(2.0, scale))


def _volume_to_scale(volume: str) -> float:
    """Convert percentage volume strings like '+0%' to Piper volume float."""
    match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%\s*$", str(volume or ""))
    if not match:
        return 1.0
    pct = float(match.group(1))
    return max(0.1, min(2.0, 1.0 + (pct / 100.0)))


@lru_cache(maxsize=2)
def _load_piper_voice(voice_name: str):
    """Load a local Piper ONNX voice model lazily on CPU (auto-downloading if absent)."""
    from piper import PiperVoice
    from piper.download_voices import download_voice

    voice_name = (voice_name or DEFAULT_VOICE).strip() or DEFAULT_VOICE
    model_dir = Path(PIPER_DIR)
    model_path = model_dir / f"{voice_name}.onnx"
    config_path = model_dir / f"{voice_name}.onnx.json"

    with _VOICE_LOAD_LOCK:
        if not (model_path.is_file() and config_path.is_file()):
            model_dir.mkdir(parents=True, exist_ok=True)
            download_voice(voice_name, model_dir)
        if not (model_path.is_file() and config_path.is_file()):
            raise FileNotFoundError(
                f"Không tìm thấy mô hình Piper '{voice_name}' tại {model_dir}"
            )
        return PiperVoice.load(
            str(model_path),
            config_path=str(config_path),
            use_cuda=False,
        )


def _wav_duration_seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
        nframes = reader.getnframes()
        framerate = reader.getframerate()
    if nframes <= 0 or framerate <= 0:
        return 0.0
    return float(nframes) / float(framerate)


def _synthesize_wav_sync(
    text: str,
    voice: str,
    rate: str = DEFAULT_RATE,
    volume: str = DEFAULT_VOLUME,
) -> tuple[bytes, float]:
    """Synthesize one text phrase in RAM as WAV bytes and compute exact duration."""
    from piper import SynthesisConfig

    piper_voice = _load_piper_voice(voice)
    syn_config = SynthesisConfig(
        length_scale=_rate_to_length_scale(rate),
        volume=_volume_to_scale(volume),
    )

    buffer = io.BytesIO()
    with _PIPER_SYNTH_LOCK:
        with wave.open(buffer, "wb") as wav_file:
            piper_voice.synthesize_wav(text, wav_file, syn_config=syn_config)

    wav_bytes = buffer.getvalue()
    duration = _wav_duration_seconds(wav_bytes)
    if duration <= 0.0:
        raise RuntimeError(f"Piper TTS ({voice}) trả về dữ liệu WAV rỗng.")
    return wav_bytes, duration


def _concatenate_wav_bytes(wav_parts: list[bytes]) -> bytes:
    """Merge multiple WAV chunks into a single valid WAV stream under one RIFF header."""
    valid_parts = [part for part in wav_parts if part]
    if not valid_parts:
        return b""
    if len(valid_parts) == 1:
        return valid_parts[0]

    base_params = None
    pcm_segments: list[bytes] = []

    for part in valid_parts:
        with wave.open(io.BytesIO(part), "rb") as reader:
            channels = reader.getnchannels()
            sampwidth = reader.getsampwidth()
            framerate = reader.getframerate()
            frames = reader.readframes(reader.getnframes())

        if not frames:
            continue

        if base_params is None:
            base_params = (channels, sampwidth, framerate)
            pcm_segments.append(frames)
            continue

        base_channels, base_sampwidth, base_framerate = base_params
        if (
            channels == base_channels
            and sampwidth == base_sampwidth
            and framerate == base_framerate
        ):
            pcm_segments.append(frames)
            continue

        # Safely adapt differing fallback sample formats/rates into the first chunk's format.
        if sampwidth == 2 and base_sampwidth == 2:
            arr = np.frombuffer(frames, dtype="<i2").astype(np.float32)
            if channels > 1:
                arr = arr.reshape(-1, channels).mean(axis=1)
            if framerate != base_framerate and arr.size > 1:
                new_len = max(1, round(arr.size * base_framerate / framerate))
                old_pos = np.arange(arr.size, dtype=np.float32)
                new_pos = np.linspace(0, arr.size - 1, new_len, dtype=np.float32)
                arr = np.interp(new_pos, old_pos, arr)
            if base_channels > 1:
                arr = np.repeat(arr[:, None], base_channels, axis=1).reshape(-1)
            pcm_segments.append(np.clip(arr, -32768, 32767).astype("<i2").tobytes())

    if base_params is None or not pcm_segments:
        return b""

    out_buffer = io.BytesIO()
    with wave.open(out_buffer, "wb") as writer:
        writer.setnchannels(base_params[0])
        writer.setsampwidth(base_params[1])
        writer.setframerate(base_params[2])
        writer.writeframes(b"".join(pcm_segments))
    return out_buffer.getvalue()


def _split_long_sentence_for_limit(sentence: str, first_limit: int, rest_limit: int) -> list[str]:
    """Split a single long sentence into <=limit parts, keeping part 1 compact when needed."""
    parts: list[str] = []
    remaining = sentence.strip()
    while remaining:
        limit = first_limit if not parts else rest_limit
        if len(remaining) <= limit:
            parts.append(remaining)
            break

        punct_cut = -1
        for punct in ("; ", ": ", ", "):
            pos = remaining.rfind(punct, 45, limit)
            if pos > punct_cut:
                punct_cut = pos + 1

        if punct_cut >= 45:
            piece = remaining[:punct_cut].strip()
            remaining = remaining[punct_cut:].strip()
        else:
            cut = remaining.rfind(" ", 0, limit + 1)
            if cut <= 0:
                cut = limit
            piece = remaining[:cut].strip()
            remaining = remaining[cut:].strip()

        if piece:
            parts.append(piece)
    return parts


def _split_long_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    first_chunk_max_chars: Optional[int] = None,
) -> list[str]:
    """Split at sentence/word boundaries, keeping chunk 1 short for fast Time-To-First-Audio."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    effective_first = (
        min(max_chars, first_chunk_max_chars)
        if first_chunk_max_chars is not None and first_chunk_max_chars >= 40
        else None
    )
    first_limit = effective_first or max_chars
    if len(normalized) <= first_limit:
        return [normalized]

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        cur_first_limit = first_limit if not chunks else max_chars
        if len(sentence) > cur_first_limit:
            parts = _split_long_sentence_for_limit(sentence, cur_first_limit, max_chars)
        else:
            parts = [sentence]

        for part in parts:
            limit = first_limit if (not chunks and effective_first is not None) else max_chars
            if not chunks and effective_first is not None and current:
                combined = f"{current} {part}".strip()
                if len(current) >= 45 or len(combined) > effective_first:
                    chunks.append(current)
                    current = part
                    continue
                current = combined
                continue

            candidate = f"{current} {part}".strip()
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part

    if current:
        chunks.append(current)
    return chunks


async def synthesize_text(
    text: str,
    *,
    voice: str = DEFAULT_VOICE,
    fallback_voice: Optional[str] = FALLBACK_VOICE,
    rate: str = DEFAULT_RATE,
    volume: str = DEFAULT_VOLUME,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    segment_callback: Optional[Callable[[bytes, dict[str, Any]], None]] = None,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> TTSResult:
    """Synthesize text with local Piper ONNX and emit each WAV chunk immediately."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Văn bản đầu vào không được để trống.")
    if max_chunk_chars < 40:
        raise ValueError("max_chunk_chars phải từ 40 ký tự trở lên.")

    first_limit = (
        min(max_chunk_chars, FIRST_CHUNK_MAX_CHARS)
        if len(text.strip()) > FIRST_CHUNK_MAX_CHARS
        else None
    )
    chunks = [
        chunk
        for chunk in _split_long_text(
            text,
            max_chars=max_chunk_chars,
            first_chunk_max_chars=first_limit,
        )
        if re.search(r"[0-9a-zA-ZÀ-ỹĐđ]", chunk)
    ]

    audio_parts: list[bytes] = []
    failed_segments: list[str] = []
    cache_hits = 0
    api_calls = 0
    fallback_segments = 0
    total_duration = 0.0

    for chunk_index, chunk in enumerate(chunks, start=1):
        if cancel_event is not None and cancel_event.is_set():
            failed_segments.extend(chunks[chunk_index - 1 :])
            break

        piece_audio: Optional[bytes] = None
        piece_duration = 0.0

        cached = _read_memory_cache(chunk, voice, rate, volume)
        if cached is not None:
            piece_audio, piece_duration = cached
            cache_hits += 1
        else:
            try:
                piece_audio, piece_duration = _synthesize_wav_sync(
                    chunk,
                    voice,
                    rate,
                    volume,
                )
                api_calls += 1
                _write_memory_cache(chunk, voice, rate, volume, piece_audio, piece_duration)
            except Exception as primary_error:
                print(f"[PIPER TTS] Lỗi giọng {voice} cho cụm {chunk[:60]!r}: {primary_error!r}")
                if fallback_voice and fallback_voice != voice:
                    fb_cached = _read_memory_cache(chunk, fallback_voice, rate, volume)
                    if fb_cached is not None:
                        piece_audio, piece_duration = fb_cached
                        cache_hits += 1
                        fallback_segments += 1
                    else:
                        try:
                            piece_audio, piece_duration = _synthesize_wav_sync(
                                chunk,
                                fallback_voice,
                                rate,
                                volume,
                            )
                            api_calls += 1
                            fallback_segments += 1
                            _write_memory_cache(
                                chunk,
                                fallback_voice,
                                rate,
                                volume,
                                piece_audio,
                                piece_duration,
                            )
                        except Exception as fb_error:
                            print(
                                f"[PIPER TTS] Giọng dự phòng {fallback_voice} lỗi cho {chunk[:60]!r}: {fb_error!r}"
                            )
        await asyncio.sleep(0)

        if piece_audio is not None:
            audio_parts.append(piece_audio)
            total_duration += piece_duration
            if segment_callback is not None:
                try:
                    segment_callback(
                        piece_audio,
                        {
                            "chunk_index": chunk_index,
                            "chunk_count": len(chunks),
                            "piece_index": 1,
                            "piece_count": 1,
                            "duration_seconds": piece_duration,
                            "mime_type": AUDIO_MIME_TYPE,
                        },
                    )
                except Exception as callback_error:
                    print(f"[PIPER TTS] Segment callback lỗi: {callback_error!r}")
        else:
            failed_segments.append(chunk)

        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "completed_chunks": chunk_index,
                        "total_chunks": len(chunks),
                        "api_calls": api_calls,
                        "cache_hits": cache_hits,
                        "failed_segments": len(failed_segments),
                    }
                )
            except Exception as callback_error:
                print(f"[PIPER TTS] Progress callback lỗi: {callback_error!r}")

    combined_audio = _concatenate_wav_bytes(audio_parts)
    if combined_audio:
        print(
            f"[PIPER TTS] Hoàn tất: {len(audio_parts)} đoạn WAV ({total_duration:.2f}s audio), "
            f"{api_calls} lượt tổng hợp, {cache_hits} RAM cache hit, "
            f"{fallback_segments} fallback, {len(failed_segments)} lỗi."
        )
    else:
        print(
            f"[PIPER TTS] Không tạo được audio: {api_calls} lượt tổng hợp, "
            f"{len(failed_segments)} đoạn thất bại."
        )

    return TTSResult(
        audio_bytes=combined_audio,
        duration_seconds=total_duration,
        failed_segments=failed_segments,
        cache_hits=cache_hits,
        api_calls=api_calls,
        fallback_segments=fallback_segments,
        mime_type=AUDIO_MIME_TYPE,
    )
