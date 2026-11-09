"""Adaptive, cached TTS adapter for the Vietnamese news-reading assistant.

The original assistant text is never modified. The adapter:
- spaces Edge TTS requests to reduce connection bursts;
- caches successful MP3 fragments on disk;
- retries NoAudioReceived once with a delay;
- splits a rejected phrase at word boundaries and synthesizes the pieces;
- uses the backup voice only for a final atomic fragment;
- enforces per-answer request/failure limits so an outage cannot create
  an unbounded number of WebSocket requests.

Public API:
    result = await synthesize_text(text)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import edge_tts
from edge_tts.exceptions import NoAudioReceived
from mutagen.mp3 import MP3


DEFAULT_VOICE = "vi-VN-HoaiMyNeural"
FALLBACK_VOICE = "vi-VN-NamMinhNeural"
DEFAULT_RATE = "+0%"
DEFAULT_VOLUME = "+0%"

# Process-wide request spacing. This is a conservative throttle, not an
# asserted Microsoft rate limit.
MIN_SECONDS_BETWEEN_REQUESTS = 2.5
RETRY_DELAY_SECONDS = 4.0
MAX_CHUNK_CHARS = 420
MAX_SPLIT_DEPTH = 8
MAX_API_CALLS_PER_TURN = 24
# Open a circuit per voice (not globally). This lets Nam Minh continue working
# when Hoai My repeatedly returns no audio, and vice versa.
MAX_CONSECUTIVE_NO_AUDIO_FAILURES_PER_VOICE = 3

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _PROJECT_ROOT / "data" / "tts_cache"
CACHE_VERSION = "tts-adapter-v3"

_schedule_lock = threading.Lock()
_next_request_at = 0.0


class TTSRequestLimitReached(RuntimeError):
    """Raised internally when a single answer reaches its request budget."""


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

    @property
    def succeeded(self) -> bool:
        return bool(self.audio_bytes)

    @property
    def partial(self) -> bool:
        return bool(self.audio_bytes) and bool(self.failed_segments)


@dataclass
class _RunStats:
    cache_hits: int = 0
    api_calls: int = 0
    fallback_segments: int = 0
    no_audio_failures_by_voice: dict[str, int] = field(default_factory=dict)
    cancel_event: Optional[threading.Event] = None

    def record_no_audio(self, voice: str) -> None:
        self.no_audio_failures_by_voice[voice] = (
            self.no_audio_failures_by_voice.get(voice, 0) + 1
        )

    def record_success(self, voice: str) -> None:
        self.no_audio_failures_by_voice[voice] = 0

    def voice_circuit_open(self, voice: str) -> bool:
        return (
            self.no_audio_failures_by_voice.get(voice, 0)
            >= MAX_CONSECUTIVE_NO_AUDIO_FAILURES_PER_VOICE
        )


def _reserve_request_delay() -> float:
    """Reserve a request start slot shared by all Streamlit sessions in process."""
    global _next_request_at
    with _schedule_lock:
        now = time.monotonic()
        scheduled_at = max(now, _next_request_at)
        _next_request_at = scheduled_at + MIN_SECONDS_BETWEEN_REQUESTS
        return max(0.0, scheduled_at - now)


async def _throttle_request() -> None:
    delay = _reserve_request_delay()
    if delay > 0:
        await asyncio.sleep(delay)


def _cache_key(text: str, voice: str, rate: str, volume: str) -> str:
    payload = "\0".join((CACHE_VERSION, voice, rate, volume, text))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.mp3"


def _read_cache(text: str, voice: str, rate: str, volume: str) -> Optional[bytes]:
    path = _cache_path(_cache_key(text, voice, rate, volume))
    try:
        data = path.read_bytes()
        if len(data) < 1024:
            return None
        info = MP3(io.BytesIO(data)).info
        if not info.length or info.length <= 0:
            return None
        return data
    except Exception:
        return None


def _write_cache(text: str, voice: str, rate: str, volume: str, data: bytes) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(_cache_key(text, voice, rate, volume))
        temporary = path.with_name(f"{path.stem}.{threading.get_ident()}.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    except OSError as error:
        # Cache failure must never prevent voice playback.
        print(f"[TTS CACHE] Không lưu được cache: {error!r}")


def _duration_seconds(data: bytes, text: str) -> float:
    try:
        return float(MP3(io.BytesIO(data)).info.length)
    except Exception:
        return max(0.4, len(text.split()) / 2.5)


async def _request_audio(
    text: str,
    voice: str,
    rate: str,
    volume: str,
    stats: _RunStats,
) -> bytes:
    if stats.cancel_event is not None and stats.cancel_event.is_set():
        raise TTSTaskCancelled("Tác vụ TTS đã bị hủy bởi lượt hội thoại mới.")
    if stats.api_calls >= MAX_API_CALLS_PER_TURN:
        raise TTSRequestLimitReached(
            f"Đã đạt giới hạn {MAX_API_CALLS_PER_TURN} request TTS cho một câu trả lời."
        )
    await _throttle_request()
    if stats.cancel_event is not None and stats.cancel_event.is_set():
        raise TTSTaskCancelled("Tác vụ TTS đã bị hủy bởi lượt hội thoại mới.")
    stats.api_calls += 1

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        volume=volume,
    )

    audio = bytearray()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            audio.extend(chunk.get("data", b""))

    if len(audio) < 1024:
        raise NoAudioReceived("Edge TTS không trả về dữ liệu MP3 hợp lệ.")

    return bytes(audio)


def _find_split(text: str) -> Optional[tuple[str, str]]:
    """Split near 70% at a word boundary, preferring punctuation boundaries."""
    words = text.split()
    if len(words) < 2:
        return None

    ideal = max(1, min(len(words) - 1, round(len(words) * 0.72)))
    punctuation = (".", "!", "?", ";", ":", ",", "—", "–")
    candidates = [
        index for index in range(1, len(words))
        if words[index - 1].endswith(punctuation)
    ]
    split_index = min(candidates, key=lambda i: abs(i - ideal)) if candidates else ideal

    left = " ".join(words[:split_index]).strip()
    right = " ".join(words[split_index:]).strip()
    if not left or not right:
        return None
    return left, right


async def _synthesize_piece(
    text: str,
    *,
    primary_voice: str,
    fallback_voice: Optional[str],
    rate: str,
    volume: str,
    depth: int,
    stats: _RunStats,
) -> tuple[list[bytes], list[str]]:
    """Synthesize one phrase with per-voice retry, fallback, then safe splitting.

    Order of recovery:
      1. Use a valid primary-voice cache entry, if available.
      2. Try the primary voice at most twice for NoAudioReceived.
      3. Try the fallback voice once before splitting the phrase.
      4. If both voices return no audio, split at a word boundary and retry.

    Circuit breakers are tracked separately for each voice. A failing primary
    voice therefore cannot prevent a healthy fallback voice from being tried.
    """
    text = text.strip()
    if not text:
        return [], []
    if stats.cancel_event is not None and stats.cancel_event.is_set():
        raise TTSTaskCancelled("Tác vụ TTS đã bị hủy bởi lượt hội thoại mới.")

    # Cache is checked before circuit-breaker state so previously generated
    # audio remains available even during a temporary Edge TTS outage.
    cached = _read_cache(text, primary_voice, rate, volume)
    if cached is not None:
        stats.cache_hits += 1
        stats.record_success(primary_voice)
        return [cached], []

    primary_error: Optional[Exception] = None
    primary_retryable = False

    if stats.voice_circuit_open(primary_voice):
        print(
            f"[TTS] Tạm ngưng giọng chính {primary_voice}: "
            "đã chạm ngưỡng lỗi NoAudioReceived riêng của giọng này."
        )
    else:
        # Primary voice: one initial request plus one delayed retry.
        for attempt in range(2):
            if stats.voice_circuit_open(primary_voice):
                break
            try:
                if attempt:
                    print(
                        f"[TTS] Thử lại giọng {primary_voice} cho cụm "
                        f"{text[:55]!r} sau {RETRY_DELAY_SECONDS:.0f} giây."
                    )
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                audio = await _request_audio(
                    text, primary_voice, rate, volume, stats
                )
                _write_cache(text, primary_voice, rate, volume, audio)
                stats.record_success(primary_voice)
                return [audio], []
            except TTSTaskCancelled:
                raise
            except TTSRequestLimitReached as error:
                print(f"[TTS] Đạt giới hạn request: {error}")
                return [], [text]
            except NoAudioReceived as error:
                primary_error = error
                primary_retryable = True
                stats.record_no_audio(primary_voice)
            except Exception as error:
                # A transport/protocol error is not recursively split: doing so
                # would multiply requests while the network/service may be down.
                primary_error = error
                print(
                    f"[TTS] Lỗi kết nối/protocol với giọng {primary_voice}, "
                    f"cụm {text[:55]!r}: {error!r}"
                )
                break

    # IMPORTANT: Try the backup voice before splitting. The old global failure
    # counter could stop this request before fallback was even attempted.
    if fallback_voice and fallback_voice != primary_voice:
        fallback_cached = _read_cache(text, fallback_voice, rate, volume)
        if fallback_cached is not None:
            stats.cache_hits += 1
            stats.fallback_segments += 1
            stats.record_success(fallback_voice)
            return [fallback_cached], []

        if stats.voice_circuit_open(fallback_voice):
            print(
                f"[TTS] Tạm ngưng giọng dự phòng {fallback_voice}: "
                "đã chạm ngưỡng lỗi NoAudioReceived riêng của giọng này."
            )
        else:
            try:
                print(
                    f"[TTS] Chuyển sang giọng dự phòng {fallback_voice} "
                    f"cho cụm: {text[:70]!r}"
                )
                audio = await _request_audio(
                    text, fallback_voice, rate, volume, stats
                )
                _write_cache(text, fallback_voice, rate, volume, audio)
                stats.fallback_segments += 1
                stats.record_success(fallback_voice)
                return [audio], []
            except TTSTaskCancelled:
                raise
            except TTSRequestLimitReached as error:
                print(f"[TTS] Đạt giới hạn request khi thử fallback: {error}")
                return [], [text]
            except NoAudioReceived as fallback_error:
                stats.record_no_audio(fallback_voice)
                print(
                    f"[TTS] Giọng dự phòng {fallback_voice} cũng không trả audio "
                    f"cho cụm {text[:70]!r}: {fallback_error!r}"
                )
            except Exception as fallback_error:
                # One backup attempt only; do not fan out a connection error.
                print(
                    f"[TTS] Lỗi khi dùng giọng dự phòng {fallback_voice}: "
                    f"{fallback_error!r}"
                )

    # Split only NoAudioReceived cases. Splitting transport errors tends to
    # create many futile requests and makes the original failure harder to see.
    if primary_retryable and depth < MAX_SPLIT_DEPTH:
        split = _find_split(text)
        if split is not None:
            left, right = split
            print(f"[TTS] Tách cụm không đọc được thành: {left!r} | {right!r}")
            left_audio, left_failed = await _synthesize_piece(
                left,
                primary_voice=primary_voice,
                fallback_voice=fallback_voice,
                rate=rate,
                volume=volume,
                depth=depth + 1,
                stats=stats,
            )
            right_audio, right_failed = await _synthesize_piece(
                right,
                primary_voice=primary_voice,
                fallback_voice=fallback_voice,
                rate=rate,
                volume=volume,
                depth=depth + 1,
                stats=stats,
            )
            return left_audio + right_audio, left_failed + right_failed

    print(
        f"[TTS] Không tạo được audio cho cụm {text[:90]!r}; "
        f"lỗi giọng chính={primary_error!r}. Giữ cụm trong danh sách thất bại."
    )
    return [], [text]


def _split_long_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split at sentence/word boundaries without dropping original words."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            parts: list[str] = []
            part = ""
            for word in sentence.split():
                candidate = f"{part} {word}".strip()
                if len(candidate) <= max_chars:
                    part = candidate
                else:
                    if part:
                        parts.append(part)
                    part = word
            if part:
                parts.append(part)
        else:
            parts = [sentence]

        for part in parts:
            candidate = f"{current} {part}".strip()
            if len(candidate) <= max_chars:
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
    """Synthesize text sequentially and emit each completed MP3 piece immediately.

    `segment_callback` runs on the caller's thread/async loop after each input
    chunk has been synthesized. It is intended for a thread-safe queue, not UI
    calls. The returned audio_bytes remains available for diagnostics/cache
    workflows; streaming applications can play emitted segments progressively.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Văn bản đầu vào không được để trống.")
    if max_chunk_chars < 40:
        raise ValueError("max_chunk_chars phải từ 40 ký tự trở lên.")

    chunks = _split_long_text(text, max_chars=max_chunk_chars)
    stats = _RunStats(cancel_event=cancel_event)
    audio_parts: list[bytes] = []
    failed_segments: list[str] = []
    total_duration = 0.0

    for chunk_index, chunk in enumerate(chunks, start=1):
        if cancel_event is not None and cancel_event.is_set():
            failed_segments.extend(chunks[chunk_index - 1:])
            break

        print(
            f"[TTS] Đang xử lý đoạn {chunk_index}/{len(chunks)} "
            f"({len(chunk)} ký tự)."
        )
        try:
            pieces, failures = await _synthesize_piece(
                chunk,
                primary_voice=voice,
                fallback_voice=fallback_voice,
                rate=rate,
                volume=volume,
                depth=0,
                stats=stats,
            )
        except TTSTaskCancelled:
            failed_segments.extend(chunks[chunk_index - 1:])
            break

        for piece_index, piece_audio in enumerate(pieces, start=1):
            audio_parts.append(piece_audio)
            piece_duration = _duration_seconds(piece_audio, chunk)
            total_duration += piece_duration
            if segment_callback is not None:
                try:
                    segment_callback(
                        piece_audio,
                        {
                            "chunk_index": chunk_index,
                            "chunk_count": len(chunks),
                            "piece_index": piece_index,
                            "piece_count": len(pieces),
                            "duration_seconds": piece_duration,
                        },
                    )
                except Exception as callback_error:
                    print(f"[TTS] Segment callback lỗi: {callback_error!r}")

        failed_segments.extend(failures)

        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "completed_chunks": chunk_index,
                        "total_chunks": len(chunks),
                        "api_calls": stats.api_calls,
                        "cache_hits": stats.cache_hits,
                        "failed_segments": len(failed_segments),
                    }
                )
            except Exception as callback_error:
                print(f"[TTS] Progress callback lỗi: {callback_error!r}")

        # Stop only at the hard per-answer request budget. Voice-specific
        # circuit breakers are checked inside _synthesize_piece, allowing a
        # healthy fallback voice (or cached audio) to keep serving later chunks.
        if stats.api_calls >= MAX_API_CALLS_PER_TURN:
            remaining = chunks[chunk_index:]
            failed_segments.extend(remaining)
            print(
                f"[TTS] Đã đạt giới hạn {MAX_API_CALLS_PER_TURN} request; "
                f"giữ lại {len(remaining)} đoạn chưa xử lý."
            )
            break

    combined_audio = b"".join(audio_parts)
    if combined_audio:
        print(
            f"[TTS] Hoàn tất: {len(audio_parts)} mảnh audio, "
            f"{stats.api_calls} request, {stats.cache_hits} cache hit, "
            f"{stats.fallback_segments} mảnh dùng giọng dự phòng, "
            f"{len(failed_segments)} mảnh chưa đọc được."
        )
    else:
        print(
            f"[TTS] Không tạo được audio: {stats.api_calls} request, "
            f"{len(failed_segments)} mảnh thất bại."
        )

    return TTSResult(
        audio_bytes=combined_audio,
        duration_seconds=total_duration,
        failed_segments=failed_segments,
        cache_hits=stats.cache_hits,
        api_calls=stats.api_calls,
        fallback_segments=stats.fallback_segments,
    )
