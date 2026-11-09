"""Thread-backed TTS tasks for Streamlit.

The Streamlit script submits a task and immediately renders the assistant text.
This module synthesizes in a daemon thread and publishes thread-safe queue
messages as each MP3 segment becomes available. It deliberately never touches
Streamlit APIs or session state from the worker thread.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from voice.tts_adapter import (
    FALLBACK_VOICE,
    DEFAULT_VOICE,
    TTSResult,
    TTSTaskCancelled,
    synthesize_text,
)


@dataclass
class _TaskState:
    events: queue.Queue = field(default_factory=queue.Queue)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None


_TASKS: dict[str, _TaskState] = {}
_TASKS_LOCK = threading.Lock()
# Serialize separate user turns within this Python process to avoid parallel
# bursts of Edge TTS WebSocket connections.
_SYNTHESIS_LOCK = threading.Lock()


def submit_tts_task(
    task_id: str,
    text: str,
    *,
    voice: str = DEFAULT_VOICE,
    fallback_voice: Optional[str] = FALLBACK_VOICE,
    rate: str = "+0%",
    volume: str = "+0%",
    max_chunk_chars: int = 420,
) -> None:
    """Start one TTS task in the background. Raises if task_id already exists."""
    if not task_id or not isinstance(text, str) or not text.strip():
        raise ValueError("task_id và text không được để trống.")

    state = _TaskState()
    with _TASKS_LOCK:
        if task_id in _TASKS:
            raise ValueError(f"Tác vụ TTS đã tồn tại: {task_id}")
        _TASKS[task_id] = state

    def publish_progress(progress: dict[str, Any]) -> None:
        state.events.put({"type": "progress", **progress})

    def publish_segment(audio_bytes: bytes, metadata: dict[str, Any]) -> None:
        if state.cancel_event.is_set():
            return
        state.events.put(
            {
                "type": "audio_segment",
                "audio_bytes": audio_bytes,
                **metadata,
            }
        )

    def worker() -> None:
        state.events.put({"type": "started"})
        try:
            with _SYNTHESIS_LOCK:
                if state.cancel_event.is_set():
                    raise TTSTaskCancelled("Tác vụ TTS đã bị hủy trước khi bắt đầu.")
                result: TTSResult = asyncio.run(
                    synthesize_text(
                        text,
                        voice=voice,
                        fallback_voice=fallback_voice,
                        rate=rate,
                        volume=volume,
                        max_chunk_chars=max_chunk_chars,
                        segment_callback=publish_segment,
                        progress_callback=publish_progress,
                        cancel_event=state.cancel_event,
                    )
                )
            state.events.put({"type": "completed", "result": result})
        except TTSTaskCancelled as error:
            state.events.put({"type": "cancelled", "error": str(error)})
        except Exception as error:
            state.events.put({"type": "failed", "error": repr(error)})

    thread = threading.Thread(
        target=worker,
        name=f"tts-{task_id[:8]}",
        daemon=True,
    )
    state.thread = thread
    thread.start()


def poll_tts_events(task_id: str, max_events: int = 100) -> list[dict[str, Any]]:
    """Drain currently available events for a task without blocking."""
    with _TASKS_LOCK:
        state = _TASKS.get(task_id)
    if state is None:
        return []

    items: list[dict[str, Any]] = []
    for _ in range(max_events):
        try:
            items.append(state.events.get_nowait())
        except queue.Empty:
            break
    return items


def cancel_tts_task(task_id: str) -> None:
    """Request cooperative cancellation at the next TTS request boundary."""
    with _TASKS_LOCK:
        state = _TASKS.get(task_id)
    if state is not None:
        state.cancel_event.set()


def release_tts_task(task_id: str) -> None:
    """Forget a completed task after the Streamlit session consumed its result."""
    with _TASKS_LOCK:
        _TASKS.pop(task_id, None)


def active_tts_task_count() -> int:
    with _TASKS_LOCK:
        return sum(
            1 for state in _TASKS.values()
            if state.thread is not None and state.thread.is_alive()
        )
