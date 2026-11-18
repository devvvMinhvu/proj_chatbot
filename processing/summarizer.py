"""Vietnamese article summarization using a local ViT5 model."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import VIT5_MODEL_DIR
from processing.chunking import split_text


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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(VIT5_MODEL_DIR),
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    print(f"[SUMMARIZER] Device: {device}")
    return tokenizer, model, device


def summarize_chunk(text: str) -> str:
    """Summarize one text chunk with ViT5."""
    if not text or not text.strip():
        return ""

    tokenizer, model, device = _get_summarizer_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            min_new_tokens=20,
            num_beams=4,
            early_stopping=True,
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.strip()


def summarize(text: str) -> str:
    """Summarize an entire article, splitting into chunks when needed."""
    if not text or not text.strip():
        return ""

    chunks = split_text(text)
    if not chunks:
        return ""

    summaries = [summary for chunk in chunks if (summary := summarize_chunk(chunk))]
    return " ".join(summaries).strip()
