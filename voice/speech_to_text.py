"""Continuous Vietnamese speech capture and transcription.

The browser microphone is streamed to Streamlit through streamlit-webrtc.
This module detects speech/silence in incoming WebRTC audio frames, creates
one WAV utterance after a pause, and puts it into a thread-safe queue.
Google Speech Recognition then converts each WAV utterance to Vietnamese text.
"""
from __future__ import annotations

from collections import deque
from io import BytesIO
import queue
import time
import wave

import numpy as np


class SpeechRecognitionError(RuntimeError):
    """Raised when audio cannot be transcribed."""


def _to_bytes(audio) -> bytes:
    if isinstance(audio, (bytes, bytearray)):
        data = bytes(audio)
    elif hasattr(audio, "getvalue"):
        data = audio.getvalue()
    elif hasattr(audio, "read"):
        try:
            audio.seek(0)
        except (AttributeError, OSError):
            pass
        data = audio.read()
    else:
        raise TypeError("audio phải là bytes hoặc đối tượng file-like.")

    if not data:
        raise SpeechRecognitionError("Không nhận được dữ liệu âm thanh.")
    return bytes(data)


def pcm_to_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Wrap mono int16 PCM samples as a standard WAV file."""
    pcm = np.asarray(pcm, dtype="<i2").reshape(-1)
    if pcm.size == 0 or sample_rate <= 0:
        raise SpeechRecognitionError("Đoạn ghi âm rỗng hoặc sai định dạng.")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def transcribe_audio(
    audio: bytes | bytearray,
    language: str = "vi-VN",
    operation_timeout: int = 20,
) -> str:
    """Transcribe one WAV utterance recorded by the browser."""
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise SpeechRecognitionError(
            "Thiếu SpeechRecognition. Chạy: "
            "python -m pip install SpeechRecognition"
        ) from exc

    audio_bytes = _to_bytes(audio)
    recognizer = sr.Recognizer()
    recognizer.operation_timeout = operation_timeout

    try:
        with sr.AudioFile(BytesIO(audio_bytes)) as audio_file:
            audio_data = recognizer.record(audio_file)
    except (ValueError, OSError) as exc:
        raise SpeechRecognitionError(
            "Dữ liệu micro không phải WAV hợp lệ. Hãy kiểm tra micro và thử lại."
        ) from exc

    try:
        text = recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError as exc:
        raise SpeechRecognitionError(
            "Chưa nghe rõ câu nói. Hãy nói gần micro hơn hoặc giảm tiếng ồn."
        ) from exc
    except sr.RequestError as exc:
        raise SpeechRecognitionError(
            "Không kết nối được dịch vụ nhận dạng Google. Hãy kiểm tra Internet."
        ) from exc

    text = (text or "").strip()
    if not text:
        raise SpeechRecognitionError("Không nhận diện được nội dung giọng nói.")
    return text


class ContinuousSpeechCollector:
    """WebRTC audio callback that emits WAV utterances after a silence.

    `__call__` is invoked by streamlit-webrtc's media thread. It must never
    call Streamlit APIs; it only writes completed utterances to `output_queue`.
    """

    def __init__(
        self,
        output_queue: queue.Queue,
        silence_seconds: float = 1.15,
        energy_threshold: int = 350,
        min_speech_seconds: float = 0.30,
        max_utterance_seconds: float = 20.0,
        pre_roll_frames: int = 8,
    ):
        self.output_queue = output_queue
        self.silence_seconds = silence_seconds
        self.energy_threshold = energy_threshold
        self.min_speech_seconds = min_speech_seconds
        self.max_utterance_seconds = max_utterance_seconds
        self.pre_roll = deque(maxlen=pre_roll_frames)
        self.frames: list[np.ndarray] = []
        self.sample_rate: int | None = None
        self.speaking = False
        self.silence_started: float | None = None

    @staticmethod
    def _frame_to_mono_int16(frame) -> tuple[np.ndarray, int]:
        arr = frame.to_ndarray()
        sample_rate = int(getattr(frame, "sample_rate", 0) or 48000)

        if arr.ndim == 2:
            channels = len(getattr(frame.layout, "channels", [])) or 1
            is_planar = bool(getattr(frame.format, "is_planar", False))
            values = arr.astype(np.float32)

            if is_planar:
                arr = values.mean(axis=0)
            elif channels > 1:
                # Packed stereo: [L, R, L, R, ...] -> mono.
                arr = values.reshape(-1, channels).mean(axis=1)
            else:
                arr = values.reshape(-1)

        arr = np.asarray(arr).reshape(-1)
        if np.issubdtype(arr.dtype, np.floating):
            peak = float(np.max(np.abs(arr))) if arr.size else 0.0
            if peak <= 1.5:
                arr = arr * 32767.0
        arr = np.clip(arr, -32768, 32767).astype(np.int16)

        # Speech recognition is more predictable with 16 kHz mono WAV.
        if sample_rate != 16000 and arr.size:
            new_size = max(1, round(arr.size * 16000 / sample_rate))
            old_positions = np.arange(arr.size, dtype=np.float32)
            new_positions = np.linspace(
                0, max(arr.size - 1, 0), new_size, dtype=np.float32
            )
            arr = np.interp(new_positions, old_positions, arr).astype(np.int16)
            sample_rate = 16000

        return arr, sample_rate

    @staticmethod
    def _rms(pcm: np.ndarray) -> float:
        if pcm.size == 0:
            return 0.0
        values = pcm.astype(np.float32)
        return float(np.sqrt(np.mean(values * values)))

    def _reset(self) -> None:
        self.frames = []
        self.sample_rate = None
        self.speaking = False
        self.silence_started = None

    def _enqueue(self, pcm_frames: list[np.ndarray], sample_rate: int) -> None:
        if not pcm_frames or not sample_rate:
            return
        pcm = np.concatenate(pcm_frames)
        duration = len(pcm) / sample_rate
        if duration < self.min_speech_seconds:
            return

        wav_bytes = pcm_to_wav(pcm, sample_rate)
        try:
            self.output_queue.put_nowait(wav_bytes)
        except queue.Full:
            # Do not block the WebRTC callback. Discard the oldest pending
            # utterance so the newest spoken request can still be processed.
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(wav_bytes)
            except queue.Full:
                pass

    def __call__(self, frame):
        try:
            pcm, sample_rate = self._frame_to_mono_int16(frame)
            now = time.monotonic()
            volume = self._rms(pcm)

            if not self.speaking:
                self.pre_roll.append((pcm, sample_rate))
                if volume >= self.energy_threshold:
                    self.speaking = True
                    # Keep a short pre-roll so the first syllable is not cut.
                    recent = list(self.pre_roll)[:-1]
                    self.frames = [
                        samples for samples, rate in recent if rate == sample_rate
                    ]
                    self.sample_rate = sample_rate
                    self.frames.append(pcm)
                    self.silence_started = None
            else:
                if sample_rate != self.sample_rate:
                    # A sample-rate change is unusual; finalize the previous
                    # utterance before starting a new segment.
                    self._enqueue(self.frames, self.sample_rate or sample_rate)
                    self._reset()
                    self.pre_roll.append((pcm, sample_rate))
                else:
                    self.frames.append(pcm)
                    if volume >= self.energy_threshold:
                        self.silence_started = None
                    elif self.silence_started is None:
                        self.silence_started = now

                    frame_count = sum(len(part) for part in self.frames)
                    duration = frame_count / max(self.sample_rate or sample_rate, 1)
                    silence_done = (
                        self.silence_started is not None
                        and now - self.silence_started >= self.silence_seconds
                    )
                    max_length = duration >= self.max_utterance_seconds

                    if silence_done or max_length:
                        self._enqueue(self.frames, self.sample_rate or sample_rate)
                        self._reset()
        except Exception:
            # Never crash the WebRTC audio callback because of one bad frame.
            pass

        return frame
