"""Datasets for the summarization benchmark.

Two sources are supported:

* ``news``  - the project's crawled ``data/news.jsonl``. The human-written sapo
  (lead) is the reference summary and the body is the input. Crawled bodies
  start with the sapo (about 99.9% of articles), so the paragraph that repeats the
  sapo is removed from the input; otherwise the reference leaks into the input
  and scores are inflated.
* ``jsonl`` - any gold dataset converted to one JSON object per line with
  ``document`` and ``summary`` fields (``id`` optional), e.g. the VietNews/VNDS
  test split. Word-segmentation underscores ("Hà_Nội") are turned into spaces.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator

from evaluation.metrics import tokenize

#: A paragraph containing at least this share of the reference's tokens is a copy of it.
LEAK_CONTAINMENT_THRESHOLD = 0.6
#: Only the first paragraphs are checked; the crawled sapo copy is almost always paragraph 0.
LEAK_SEARCH_PARAGRAPHS = 5


@dataclass
class SummarizationExample:
    id: str
    document: str
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)


def containment(reference: str, paragraph: str) -> float:
    """Share of reference tokens (with multiplicity) that also occur in ``paragraph``."""
    ref_counts = Counter(tokenize(reference))
    total = sum(ref_counts.values())
    if total == 0:
        return 0.0
    overlap = ref_counts & Counter(tokenize(paragraph))
    return sum(overlap.values()) / total


def strip_reference_from_document(
    document: str,
    reference: str,
    threshold: float = LEAK_CONTAINMENT_THRESHOLD,
    search_paragraphs: int = LEAK_SEARCH_PARAGRAPHS,
) -> tuple[str, int]:
    """Remove leading paragraphs that repeat the reference; return (document, removed_count)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document or "") if p.strip()]
    kept: list[str] = []
    removed = 0
    for index, paragraph in enumerate(paragraphs):
        if index < search_paragraphs and containment(reference, paragraph) >= threshold:
            removed += 1
            continue
        kept.append(paragraph)
    return "\n\n".join(kept), removed


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_number}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _sample(examples: list[SummarizationExample], limit: int | None, seed: int):
    random.Random(seed).shuffle(examples)
    return examples if limit is None or limit <= 0 else examples[:limit]


def load_news_examples(
    path: Path,
    limit: int | None = 200,
    seed: int = 42,
    min_words: int = 150,
    max_words: int = 1500,
    strip_leak: bool = True,
) -> tuple[list[SummarizationExample], dict[str, Any]]:
    """Load crawled articles as (body -> sapo) pairs, sampled reproducibly.

    The length filter is applied to the body *after* leak removal.
    """
    examples: list[SummarizationExample] = []
    stats = Counter()
    for row in _iter_jsonl(path):
        stats["rows"] += 1
        body = str(row.get("body") or "").strip()
        sapo = str(row.get("sapo") or "").strip()
        if not body or not sapo:
            stats["skipped_missing_fields"] += 1
            continue
        removed = 0
        if strip_leak:
            body, removed = strip_reference_from_document(body, sapo)
        stats["leak_paragraphs_removed"] += removed
        stats["articles_with_leak"] += bool(removed)
        word_count = len(body.split())
        if not min_words <= word_count <= max_words:
            stats["skipped_length"] += 1
            continue
        examples.append(
            SummarizationExample(
                id=str(row.get("id") or stats["rows"]),
                document=body,
                reference=sapo,
                metadata={
                    "source": row.get("source", ""),
                    "category": row.get("category", ""),
                    "title": row.get("title", ""),
                    "document_words": word_count,
                    "leak_paragraphs_removed": removed,
                },
            )
        )
    stats["eligible"] = len(examples)
    sampled = _sample(examples, limit, seed)
    stats["sampled"] = len(sampled)
    return sampled, dict(stats)


def _desegment(text: str) -> str:
    return re.sub(r"(?<=\w)_(?=\w)", " ", str(text or ""))


def load_jsonl_examples(
    path: Path,
    limit: int | None = None,
    seed: int = 42,
) -> tuple[list[SummarizationExample], dict[str, Any]]:
    """Load a gold dataset with ``document`` and ``summary`` fields per JSON line."""
    examples: list[SummarizationExample] = []
    stats = Counter()
    for row in _iter_jsonl(path):
        stats["rows"] += 1
        document = _desegment(row.get("document") or "").strip()
        summary = _desegment(row.get("summary") or "").strip()
        if not document or not summary:
            stats["skipped_missing_fields"] += 1
            continue
        examples.append(
            SummarizationExample(
                id=str(row.get("id") or stats["rows"]),
                document=document,
                reference=summary,
                metadata={"document_words": len(document.split())},
            )
        )
    stats["eligible"] = len(examples)
    sampled = _sample(examples, limit, seed)
    stats["sampled"] = len(sampled)
    return sampled, dict(stats)
