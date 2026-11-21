"""Rule-based Vietnamese sentence splitting for extractive summarization."""

from __future__ import annotations

import re

# Sentence-final punctuation, optional closing quotes/brackets, then whitespace.
_BOUNDARY_RE = re.compile(r"[.!?…]+[\"'”’)\]]*\s+")
_OPENING_CHARS = "\"'“‘([-–—"
_MIN_SENTENCE_WORDS = 3


def _starts_new_sentence(text: str) -> bool:
    """A boundary is real when the next visible character is uppercase or a digit."""
    stripped = text.lstrip(_OPENING_CHARS + " ")
    return bool(stripped) and (stripped[0].isupper() or stripped[0].isdigit())


def split_paragraph(paragraph: str) -> list[str]:
    """Split one paragraph into sentences without breaking decimals like '3.5' or 'TP.HCM'."""
    paragraph = " ".join(paragraph.split())
    if not paragraph:
        return []

    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(paragraph):
        if not _starts_new_sentence(paragraph[match.end() :]):
            continue
        sentence = paragraph[start : match.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()

    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_sentences(text: str) -> list[str]:
    """Split an article into sentences; paragraph breaks are always boundaries.

    Very short fragments (fewer than three words, e.g. bylines or photo credits)
    are merged into the preceding sentence of the same paragraph.
    """
    result: list[str] = []
    for paragraph in re.split(r"\n+", str(text or "")):
        merged: list[str] = []
        for sentence in split_paragraph(paragraph):
            if merged and len(sentence.split()) < _MIN_SENTENCE_WORDS:
                merged[-1] = f"{merged[-1]} {sentence}"
            else:
                merged.append(sentence)
        result.extend(merged)
    return result
