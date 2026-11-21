"""Vietnamese article summarization using a local ViT5 model."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import threading
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import VIT5_MODEL_DIR
from processing.chunking import split_text

#: Only the first chunks (400 words each, see processing.chunking) are summarized.
MAX_SUMMARY_CHUNKS = 2
#: ViT5 input length per chunk, in tokens; longer chunks are truncated.
MAX_INPUT_TOKENS = 512
#: Generation settings shared by the app and the benchmark's ``vit5_app`` system.
GENERATION_KWARGS: dict[str, Any] = {
    "max_new_tokens": 110,
    "min_new_tokens": 20,
    "num_beams": 2,
    "early_stopping": True,
}
_SUMMARY_CACHE_MAX_ENTRIES = 128
_SUMMARY_CACHE: OrderedDict[str, str] = OrderedDict()
_SUMMARY_CACHE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _get_summarizer_model() -> tuple[Any, Any, torch.device]:
    """Load ViT5 tokenizer and model lazily on first summarization request."""
    if not VIT5_MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Không tìm thấy mô hình ViT5 tại: {VIT5_MODEL_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        str(VIT5_MODEL_DIR),
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(VIT5_MODEL_DIR),
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    print(f"[SUMMARIZER] Device: {device}")
    return tokenizer, model, device


def _summary_cache_key(text: str, article_id: str | None = None) -> str:
    """Compute a deterministic cache key for normalized article text and optional ID."""
    norm_id = str(article_id).strip() if article_id else ""
    norm_text = " ".join(text.split())
    payload = f"{norm_id}\0{norm_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize_chunks_batch(chunks: list[str]) -> list[str]:
    """Summarize one or more text chunks in a single batched ViT5 pass."""
    cleaned_positions: list[tuple[int, str]] = [
        (idx, chunk.strip()) for idx, chunk in enumerate(chunks) if chunk and chunk.strip()
    ]
    if not cleaned_positions:
        return [""] * len(chunks)

    tokenizer, model, device = _get_summarizer_model()
    batch_texts = [text for _, text in cleaned_positions]
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, **GENERATION_KWARGS)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = [""] * len(chunks)
    for (pos, _), summary in zip(cleaned_positions, decoded):
        results[pos] = summary.strip()
    return results


def summarize_chunk(text: str) -> str:
    """Summarize one text chunk with ViT5."""
    if not text or not text.strip():
        return ""
    return summarize_chunks_batch([text])[0]


def summarize(text: str, article_id: str | None = None) -> str:
    """Summarize an article using a shared LRU cache and batched lead/body chunks."""
    if not text or not text.strip():
        return ""

    cache_key = _summary_cache_key(text, article_id=article_id)
    with _SUMMARY_CACHE_LOCK:
        cached = _SUMMARY_CACHE.get(cache_key)
        if cached is not None:
            _SUMMARY_CACHE.move_to_end(cache_key)
            return cached

    chunks = split_text(text)[:MAX_SUMMARY_CHUNKS]
    if not chunks:
        return ""

    summaries = [summary for summary in summarize_chunks_batch(chunks) if summary]
    result = " ".join(summaries).strip()

    if result:
        with _SUMMARY_CACHE_LOCK:
            _SUMMARY_CACHE[cache_key] = result
            _SUMMARY_CACHE.move_to_end(cache_key)
            while len(_SUMMARY_CACHE) > _SUMMARY_CACHE_MAX_ENTRIES:
                _SUMMARY_CACHE.popitem(last=False)

    return result
