"""Registry of summarization systems compared in the benchmark.

Built-in names:
  lead          Lead-k baseline (first k sentences)
  textrank      TextRank (Mihalcea & Tarau, 2004)
  phobert       PhoBERT centroid + MMR (BERT-based extractive)
  vit5_app      The app's ViT5 pipeline (chunk, summarize, concatenate)
  vit5          ViT5 single pass over the first 1024 tokens
  mt5_xlsum     mT5-base fine-tuned on XL-Sum (T5 family; Hub model, must be downloaded)

Any other encoder-decoder model (e.g. a BARTpho summarization checkpoint) can
be used as ``seq2seq:<local path or Hub ID>``.
"""

from __future__ import annotations

from config import VIT5_MODEL_DIR
from processing.summarizers.abstractive import AppViT5Summarizer, Seq2SeqSummarizer
from processing.summarizers.base import Summarizer
from processing.summarizers.extractive import (
    LeadSummarizer,
    PhoBERTCentroidSummarizer,
    TextRankSummarizer,
)

MT5_XLSUM_ID = "csebuetnlp/mT5_multilingual_XLSum"

AVAILABLE_SYSTEMS = ("lead", "textrank", "phobert", "vit5_app", "vit5", "mt5_xlsum")


def build_summarizer(name: str, num_sentences: int = 3) -> Summarizer:
    """Create a summarizer by registry name; ``num_sentences`` applies to extractive ones."""
    key = name.strip()
    if key == "lead":
        return LeadSummarizer(num_sentences=num_sentences)
    if key == "textrank":
        return TextRankSummarizer(num_sentences=num_sentences)
    if key == "phobert":
        return PhoBERTCentroidSummarizer(num_sentences=num_sentences)
    if key == "vit5_app":
        return AppViT5Summarizer()
    if key == "vit5":
        return Seq2SeqSummarizer(VIT5_MODEL_DIR, name="vit5", max_input_tokens=1024)
    if key == "mt5_xlsum":
        # Settings from the model card: 512 input tokens, beam 4, max 84, no_repeat_ngram 2.
        return Seq2SeqSummarizer(
            MT5_XLSUM_ID,
            name="mt5_xlsum",
            max_input_tokens=512,
            max_new_tokens=84,
            min_new_tokens=0,
            num_beams=4,
            no_repeat_ngram_size=2,
        )
    if key.startswith("seq2seq:"):
        return Seq2SeqSummarizer(key.split(":", 1)[1], name=key)
    raise ValueError(
        f"Unknown summarizer '{name}'. Choose from {', '.join(AVAILABLE_SYSTEMS)} "
        "or seq2seq:<model path or Hub ID>."
    )


__all__ = ["AVAILABLE_SYSTEMS", "Summarizer", "build_summarizer"]
