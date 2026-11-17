"""Offline tests for v8 TTS retry/fallback/circuit-breaker behavior.

Run from the project root:
    python -m unittest discover -s tests -p "test_tts_adapter_recovery.py" -v

The Edge TTS network client is stubbed; no network requests are made.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

# Import-only Edge TTS stub. Tests replace _request_audio, so no synthesis
# client or external service is used.
try:
    import edge_tts  # noqa: F401
except ImportError:
    edge_stub = types.ModuleType("edge_tts")
    edge_stub.Communicate = object
    exceptions_stub = types.ModuleType("edge_tts.exceptions")

    class NoAudioReceived(Exception):
        pass

    exceptions_stub.NoAudioReceived = NoAudioReceived
    edge_stub.exceptions = exceptions_stub
    sys.modules["edge_tts"] = edge_stub
    sys.modules["edge_tts.exceptions"] = exceptions_stub

from voice import tts_adapter as adapter  # noqa: E402


class TestTTSAdapterRecovery(unittest.TestCase):
    def setUp(self):
        self.temp_cache = tempfile.TemporaryDirectory()
        self.cache_patcher = patch.object(
            adapter, "CACHE_DIR", adapter.Path(self.temp_cache.name)
        )
        self.cache_patcher.start()
        self.delay_patcher = patch.object(adapter, "RETRY_DELAY_SECONDS", 0.0)
        self.delay_patcher.start()

    def tearDown(self):
        self.delay_patcher.stop()
        self.cache_patcher.stop()
        self.temp_cache.cleanup()

    def test_primary_retries_once_then_fallback_is_used(self):
        calls: list[str] = []

        async def fake_request(text, voice, rate, volume, stats):
            stats.api_calls += 1
            calls.append(voice)
            if voice == adapter.DEFAULT_VOICE:
                raise adapter.NoAudioReceived("primary has no audio")
            return b"M" * 2048

        with patch.object(adapter, "_request_audio", side_effect=fake_request):
            result = asyncio.run(adapter.synthesize_text("Đây là câu kiểm tra."))

        self.assertTrue(result.audio_bytes)
        self.assertEqual(calls.count(adapter.DEFAULT_VOICE), 2)
        self.assertEqual(calls.count(adapter.FALLBACK_VOICE), 1)
        self.assertEqual(result.fallback_segments, 1)
        self.assertEqual(result.failed_segments, [])

    def test_primary_circuit_does_not_block_fallback_voice(self):
        calls: list[str] = []
        stats = adapter._RunStats(
            no_audio_failures_by_voice={adapter.DEFAULT_VOICE: 3}
        )

        async def fake_request(text, voice, rate, volume, stats_arg):
            stats_arg.api_calls += 1
            calls.append(voice)
            if voice == adapter.DEFAULT_VOICE:
                raise AssertionError("Primary circuit should be open")
            return b"M" * 2048

        with patch.object(adapter, "_request_audio", side_effect=fake_request):
            audio, failed = asyncio.run(
                adapter._synthesize_piece(
                    "Giá vàng hôm nay biến động.",
                    primary_voice=adapter.DEFAULT_VOICE,
                    fallback_voice=adapter.FALLBACK_VOICE,
                    rate=adapter.DEFAULT_RATE,
                    volume=adapter.DEFAULT_VOLUME,
                    depth=0,
                    stats=stats,
                )
            )

        self.assertEqual(len(audio), 1)
        self.assertEqual(failed, [])
        self.assertEqual(calls, [adapter.FALLBACK_VOICE])
        self.assertEqual(stats.fallback_segments, 1)

    def test_failed_long_phrase_splits_and_uses_fallback_for_parts(self):
        calls: list[tuple[str, str]] = []
        original_text = "Đoạn văn dài cần đọc."

        async def fake_request(text, voice, rate, volume, stats):
            stats.api_calls += 1
            calls.append((voice, text))
            if voice == adapter.DEFAULT_VOICE:
                raise adapter.NoAudioReceived("primary has no audio")
            # First fallback fails on the original phrase. Smaller pieces work.
            if text == original_text:
                raise adapter.NoAudioReceived("fallback failed for long phrase")
            return b"M" * 2048

        with patch.object(adapter, "_request_audio", side_effect=fake_request):
            result = asyncio.run(
                adapter.synthesize_text(original_text, max_chunk_chars=420)
            )

        self.assertTrue(result.audio_bytes)
        self.assertEqual(result.failed_segments, [])
        self.assertGreaterEqual(result.fallback_segments, 2)
        self.assertTrue(any(v == adapter.FALLBACK_VOICE for v, _ in calls))

    def test_both_open_voice_circuits_do_not_send_more_requests(self):
        stats = adapter._RunStats(
            no_audio_failures_by_voice={
                adapter.DEFAULT_VOICE: 3,
                adapter.FALLBACK_VOICE: 3,
            }
        )

        async def unexpected_request(*args, **kwargs):
            raise AssertionError("No network request should be made")

        with patch.object(adapter, "_request_audio", side_effect=unexpected_request):
            audio, failed = asyncio.run(
                adapter._synthesize_piece(
                    "Một cụm văn bản chưa có cache.",
                    primary_voice=adapter.DEFAULT_VOICE,
                    fallback_voice=adapter.FALLBACK_VOICE,
                    rate=adapter.DEFAULT_RATE,
                    volume=adapter.DEFAULT_VOLUME,
                    depth=0,
                    stats=stats,
                )
            )

        self.assertEqual(audio, [])
        self.assertEqual(failed, ["Một cụm văn bản chưa có cache."])


if __name__ == "__main__":
    unittest.main(verbosity=2)
