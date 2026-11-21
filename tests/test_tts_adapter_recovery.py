"""Offline tests for in-memory Piper TTS caching, WAV merging, fallback, and cancellation."""

from __future__ import annotations

import asyncio
import io
import threading
import unittest
from unittest.mock import patch
import wave

from voice import tts_adapter as adapter


def _make_wav_bytes(duration_seconds: float = 0.25, framerate: int = 22050) -> bytes:
    nframes = max(1, round(duration_seconds * framerate))
    pcm = b"\x01\x00" * nframes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(pcm)
    return buf.getvalue()


class TestPiperTTSAdapter(unittest.TestCase):
    def setUp(self):
        adapter.clear_memory_cache()

    def tearDown(self):
        adapter.clear_memory_cache()

    def test_in_memory_cache_hit_and_zero_disk_writes(self):
        calls: list[str] = []
        sample_wav = _make_wav_bytes(0.4)

        def fake_synth(text, voice, rate, volume):
            calls.append(voice)
            return sample_wav, 0.4

        with patch.object(adapter, "_synthesize_wav_sync", side_effect=fake_synth):
            first = asyncio.run(adapter.synthesize_text("Giá vàng hôm nay tăng mạnh."))
            second = asyncio.run(adapter.synthesize_text("Giá vàng hôm nay tăng mạnh."))

        self.assertTrue(first.succeeded)
        self.assertEqual(first.api_calls, 1)
        self.assertEqual(first.cache_hits, 0)
        self.assertTrue(second.succeeded)
        self.assertEqual(second.api_calls, 0)
        self.assertEqual(second.cache_hits, 1)
        self.assertEqual(calls, [adapter.DEFAULT_VOICE])
        self.assertEqual(first.mime_type, "audio/wav")

    def test_primary_failure_falls_back_to_secondary_voice(self):
        calls: list[str] = []
        fallback_wav = _make_wav_bytes(0.5)

        def fake_synth(text, voice, rate, volume):
            calls.append(voice)
            if voice == adapter.DEFAULT_VOICE:
                raise RuntimeError("Primary model missing")
            return fallback_wav, 0.5

        with patch.object(adapter, "_synthesize_wav_sync", side_effect=fake_synth):
            result = asyncio.run(adapter.synthesize_text("Bản tin thể thao buổi sáng."))

        self.assertTrue(result.succeeded)
        self.assertEqual(calls, [adapter.DEFAULT_VOICE, adapter.FALLBACK_VOICE])
        self.assertEqual(result.fallback_segments, 1)
        self.assertEqual(result.failed_segments, [])

    def test_concatenate_wav_bytes_produces_valid_combined_wav(self):
        part1 = _make_wav_bytes(0.25, framerate=22050)
        part2 = _make_wav_bytes(0.35, framerate=22050)
        combined = adapter._concatenate_wav_bytes([part1, part2])

        with wave.open(io.BytesIO(combined), "rb") as reader:
            self.assertEqual(reader.getnchannels(), 1)
            self.assertEqual(reader.getsampwidth(), 2)
            self.assertEqual(reader.getframerate(), 22050)
            self.assertAlmostEqual(
                reader.getnframes() / reader.getframerate(),
                0.60,
                places=2,
            )

        # Cross-sample-rate merge (e.g. 22050 Hz primary + 16000 Hz fallback)
        part_16k = _make_wav_bytes(0.20, framerate=16000)
        mixed = adapter._concatenate_wav_bytes([part1, part_16k])
        with wave.open(io.BytesIO(mixed), "rb") as reader:
            self.assertEqual(reader.getframerate(), 22050)
            self.assertAlmostEqual(reader.getnframes() / reader.getframerate(), 0.45, places=2)

    def test_cancel_event_stops_remaining_chunks(self):
        calls: list[str] = []
        cancel_event = threading.Event()
        sample_wav = _make_wav_bytes(0.3)

        def fake_synth(text, voice, rate, volume):
            calls.append(text)
            return sample_wav, 0.3

        def on_segment(_audio_bytes, _meta):
            cancel_event.set()

        long_text = (
            "Đây là câu thứ nhất khá dài để tách thành phân đoạn mở đầu giúp trình duyệt bắt đầu phát âm thanh nhanh nhất có thể ngay lập tức. "
            "Đây là câu thứ hai dài hơn nữa và sẽ bị hủy bỏ ngay khi người dùng gửi câu hỏi mới trong lúc đang nghe."
        )
        with patch.object(adapter, "_synthesize_wav_sync", side_effect=fake_synth):
            result = asyncio.run(
                adapter.synthesize_text(
                    long_text,
                    segment_callback=on_segment,
                    cancel_event=cancel_event,
                )
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result.failed_segments), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
