"""Accent-aware Vietnamese keyword matching shared by retrieval and intent handling.

Vietnamese words that differ only in accents mean different things ("giá" price,
"Gia" in Gia Lai; "tăng" rise, "tặng" give). Accented query words therefore match
only with the same accents, while words typed without accents and foreign names
(e.g. "Mbappé" vs "Mbappe") still match loosely.
"""

from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[0-9a-zA-ZÀ-ỹĐđ]+")

#: A Vietnamese syllable without diacritics: optional onset, a vowel nucleus, optional coda.
_VIETNAMESE_SYLLABLE = re.compile(
    r"(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx])?"
    r"[aeiouy]{1,3}"
    r"(?:ch|ng|nh|[cmnpt])?"
)


def fold_text(text: str) -> str:
    """Lowercase text and strip Vietnamese diacritics ("Đà Nẵng" -> "da nang")."""
    folded = str(text or "").casefold().replace("đ", "d")
    folded = unicodedata.normalize("NFD", folded)
    return "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")


def accented_lower(text: str) -> str:
    """Lowercase text in NFC form, keeping accents."""
    return unicodedata.normalize("NFC", str(text or "")).casefold()


def is_vietnamese_syllable(token: str) -> bool:
    """True when a (diacritic-free or accented) token can be a single Vietnamese syllable."""
    return bool(_VIETNAMESE_SYLLABLE.fullmatch(fold_text(token)))


def needs_exact_accents(keyword: str) -> bool:
    """True for accented Vietnamese words, which must match with their accents."""
    lowered = accented_lower(keyword)
    return fold_text(lowered) != lowered and is_vietnamese_syllable(lowered)


def matched_keywords(keywords: list[str], text: str, exact: bool | None = None) -> set[str]:
    """Return the keywords (lowercased) that occur in ``text`` as whole words, accent-aware.

    Multi-word keywords ("vệ sinh") must appear as a phrase. ``exact`` says whether
    the user's message was typed with accents: then every word, including ones
    without accent marks such as "sau", must match exactly. When it is None each
    keyword decides for itself (accented Vietnamese words exact, others loose).
    """
    accented_words = {tok for tok in _WORD.findall(accented_lower(text)) if len(tok) > 1}
    folded_words = {fold_text(tok) for tok in accented_words}
    matched: set[str] = set()
    for keyword in keywords:
        lowered = accented_lower(keyword).replace("_", " ")
        parts = _WORD.findall(lowered)
        if len(parts) > 1:
            if contains_phrase(parts, text):
                matched.add(lowered)
            continue
        keep_accents = needs_exact_accents(lowered) if exact is None else (
            exact and is_vietnamese_syllable(lowered)
        )
        if keep_accents:
            if lowered in accented_words:
                matched.add(lowered)
        elif fold_text(lowered) in folded_words:
            matched.add(lowered)
    return matched


def contains_phrase(phrase_words: list[str], text: str) -> bool:
    """True when the words appear consecutively in ``text``; accents count if the query has them."""
    if len(phrase_words) < 2:
        return False
    phrase = " ".join(accented_lower(word) for word in phrase_words)
    haystack = " ".join(_WORD.findall(accented_lower(text)))
    if fold_text(phrase) == phrase:
        return f" {phrase} " in f" {fold_text(haystack)} "
    return f" {phrase} " in f" {haystack} "
