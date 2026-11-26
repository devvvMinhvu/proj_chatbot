"""Abstractive summarizers: the app's ViT5 pipeline and a generic seq2seq wrapper."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from processing.sentences import clean_generated_text
from processing.summarizers.base import Summarizer
from processing.summarizers.extractive import _empty_cuda_cache


class AppViT5Summarizer(Summarizer):
    """Exactly what the Streamlit app runs: ``processing.summarizer.summarize``.

    The article is cut into 400-word chunks (50-word overlap); only the first
    ``MAX_SUMMARY_CHUNKS`` chunks are summarized by ViT5 in one batch and the
    chunk summaries are joined. The settings reported by ``describe`` are read
    from ``processing.summarizer`` so they always match the app.
    """

    name = "vit5_app"
    kind = "abstractive"

    def summarize(self, text: str) -> str:
        from processing.summarizer import summarize

        return summarize(text)

    def describe(self) -> dict[str, Any]:
        from processing import summarizer as app_summarizer

        return {
            **super().describe(),
            "model": "models/vit5_summarization",
            "strategy": (
                f"first {app_summarizer.MAX_SUMMARY_CHUNKS} chunks of 400 words "
                "(overlap 50), summarized in one batch, summaries joined"
            ),
            "max_summary_chunks": app_summarizer.MAX_SUMMARY_CHUNKS,
            "max_input_tokens": app_summarizer.MAX_INPUT_TOKENS,
            "generation": dict(app_summarizer.GENERATION_KWARGS),
        }

    def close(self) -> None:
        from processing import summarizer as app_summarizer

        app_summarizer._get_summarizer_model.cache_clear()
        with app_summarizer._SUMMARY_CACHE_LOCK:
            app_summarizer._SUMMARY_CACHE.clear()
        _empty_cuda_cache()


def _collapse_whitespace(text: str) -> str:
    """Whitespace handling recommended by the mT5 XL-Sum model card."""
    return re.sub(r"\s+", " ", re.sub(r"\n+", " ", text.strip()))


class Seq2SeqSummarizer(Summarizer):
    """Single-pass summarization with any Hugging Face encoder-decoder model.

    Works for local directories (e.g. ``models/vit5_summarization``) and Hub IDs
    (e.g. ``csebuetnlp/mT5_multilingual_XLSum``). With ``local_files_only=True``
    (the default) nothing is downloaded: the model must already be on disk or in
    the Hugging Face cache.
    """

    kind = "abstractive"

    def __init__(
        self,
        model_name_or_path: str | Path,
        name: str | None = None,
        max_input_tokens: int = 1024,
        max_new_tokens: int = 120,
        min_new_tokens: int = 20,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 0,
        input_prefix: str = "",
        use_fast_tokenizer: bool = False,
        local_files_only: bool = True,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.name = name or f"seq2seq:{Path(self.model_name_or_path).name}"
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.input_prefix = input_prefix
        self.use_fast_tokenizer = use_fast_tokenizer
        self.local_files_only = local_files_only
        self._loaded: tuple[Any, Any, Any] | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                local_files_only=self.local_files_only,
                use_fast=self.use_fast_tokenizer,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path,
                local_files_only=self.local_files_only,
            )
            model.to(device)
            model.eval()
            self._loaded = (tokenizer, model, device)
        return self._loaded

    def summarize(self, text: str) -> str:
        text = _collapse_whitespace(str(text or ""))
        if not text:
            return ""
        import torch

        tokenizer, model, device = self._load()
        inputs = tokenizer(
            self.input_prefix + text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "num_beams": self.num_beams,
            "early_stopping": True,
        }
        if self.no_repeat_ngram_size:
            generate_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generate_kwargs)
        return clean_generated_text(tokenizer.decode(outputs[0], skip_special_tokens=True))

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "model": self.model_name_or_path,
            "strategy": f"single pass, input truncated to {self.max_input_tokens} tokens",
            "generation": {
                "num_beams": self.num_beams,
                "min_new_tokens": self.min_new_tokens,
                "max_new_tokens": self.max_new_tokens,
                "no_repeat_ngram_size": self.no_repeat_ngram_size,
            },
        }

    def close(self) -> None:
        self._loaded = None
        _empty_cuda_cache()
