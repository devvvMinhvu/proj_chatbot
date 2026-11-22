"""Read one user message into structured slots, independent of its intent.

The parser uses grammar and data instead of phrase lists:

* Word classes come from the pyvi tagger. A number after a common noun points
  to an item ("bài 2", "kết quả thứ hai"); a number before a noun is a quantity
  ("3 tin"); a number before "câu" is a summary length ("một câu"); a number
  after a proper noun belongs to a name ("Quả Bóng Vàng 2026").
* Publisher and category names come from the database values and their display
  labels in ``config``.
* Topic words are what is left once slots, function words and command words
  (both learned in ``nlp.keyword_stats``) are removed.
* Dates are found with ``dateparser``; bare numbers are never dates.

Whether a message was typed with accents is decided once for the whole message,
so "sau" (after) and "sáu" (six) never match each other in accented text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
import calendar
import re
from typing import Any, Callable

from config import CATEGORY_DISPLAY, SOURCE_DISPLAY
from nlp import lexicon
from nlp.keyword_stats import get_keyword_stats
from processing.text_matching import (
    accented_lower,
    contains_phrase,
    fold_text,
    is_vietnamese_syllable,
)

_WORD = re.compile(r"[0-9a-zA-ZÀ-ỹĐđ]+")
#: Word classes that can carry a topic: nouns, verbs, adjectives (pyvi tag set).
_CONTENT_TAGS = {"N", "Np", "Nc", "Nu", "Ny", "Nb", "V", "Vb", "Vy", "A", "Ab", "X"}
_DATE_SEPARATOR = re.compile(r"\d{1,2}\s*[/\-.]\s*\d{1,2}")


@dataclass
class ParsedTurn:
    text: str
    exact_accents: bool
    words: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    item_refs: list[int] = field(default_factory=list)
    relative: str | None = None
    refers_to_list: bool = False
    count: int | None = None
    max_sentences: int | None = None
    date_from: date | None = None
    date_to: date | None = None
    date_text: str = ""
    source: str | None = None
    category: str | None = None
    topic_words: list[str] = field(default_factory=list)
    #: Topic words that are nouns or names: a message with none brings no new subject.
    topic_nouns: list[str] = field(default_factory=list)
    entity_words: list[str] = field(default_factory=list)
    is_question: bool = False

    @property
    def has_slots(self) -> bool:
        """True when the message sets a filter, count, length or date."""
        return any(
            value is not None
            for value in (self.source, self.category, self.count, self.max_sentences, self.date_from)
        )

    @property
    def search_text(self) -> str:
        return " ".join(self.topic_words)


# --- Vocabulary from the database ----------------------------------------------


@lru_cache(maxsize=1)
def label_vocabulary() -> dict[str, dict[str, tuple[str, ...]]]:
    """Publisher and category names: each database value with its display label."""
    return {
        "source": {slug: (slug, label) for slug, label in SOURCE_DISPLAY.items()},
        "category": {slug: (slug, label) for slug, label in CATEGORY_DISPLAY.items()},
    }


def _find_label(text: str, kind: str) -> tuple[str | None, list[str]]:
    matches: list[tuple[str, str]] = []
    for slug, names in label_vocabulary()[kind].items():
        for name in names:
            words = _WORD.findall(name)
            if contains_phrase(words, text) or (len(words) == 1 and _contains_word(text, name)):
                matches.append((slug, name))
                break
    slugs = {slug for slug, _ in matches}
    if len(slugs) != 1:
        return None, []
    return matches[0][0], [name for _, name in matches]


def _contains_word(text: str, word: str) -> bool:
    lowered = accented_lower(text)
    target = accented_lower(word)
    if fold_text(target) == target:
        return target in {fold_text(tok) for tok in _WORD.findall(lowered)}
    return target in _WORD.findall(lowered)


# --- Dates -----------------------------------------------------------------------


def _date_range(matched: str, value: datetime) -> tuple[date, date]:
    folded = fold_text(matched)
    if "tuan" in folded:
        start = value.date() - timedelta(days=value.weekday())
        return start, start + timedelta(days=6)
    if "thang" in folded and not _DATE_SEPARATOR.search(matched):
        last = calendar.monthrange(value.year, value.month)[1]
        return date(value.year, value.month, 1), date(value.year, value.month, last)
    if re.fullmatch(r"\s*(?:nam|năm)\s+\d{4}\s*", matched, flags=re.IGNORECASE):
        return date(value.year, 1, 1), date(value.year, 12, 31)
    return value.date(), value.date()


def find_dates(
    text: str,
    today: date | None = None,
    after_noun: "Callable[[str], bool] | None" = None,
) -> tuple[date | None, date | None, list[str]]:
    """Date expressions in ``text``: the first one as a (from, to) range, and all matched words.

    ``after_noun(matched)`` tells whether the match directly follows a common noun;
    then a weekday-shaped match such as "thứ hai" is an item position ("tin thứ
    hai"), not Monday.
    """
    from dateparser.search import search_dates

    base = datetime.combine(today or date.today(), datetime.min.time())
    settings = {"RELATIVE_BASE": base, "PREFER_DATES_FROM": "past", "DATE_ORDER": "DMY"}
    found: list[tuple[str, datetime]] = []
    remaining = text
    for _ in range(3):  # dateparser can miss a second expression; search what is left
        try:
            batch = search_dates(remaining, languages=["vi"], settings=settings) or []
        except Exception:  # dateparser raises on some odd inputs; dates are optional
            batch = []
        if not batch:
            break
        found.extend(batch)
        for matched, _ in batch:
            remaining = remaining.replace(matched, " ")
    first: tuple[date, date] | None = None
    matched_texts: list[str] = []
    for matched, value in found:
        stripped = matched.strip()
        if re.fullmatch(r"[\d\s]+", stripped):
            continue  # a bare number ("2026", "3") is not a date
        if re.fullmatch(r"thu\s+\S+", fold_text(stripped)) and after_noun and after_noun(stripped):
            continue  # "tin thứ hai" is the second item, not Monday
        matched_texts.append(stripped)
        if first is None:
            first = _date_range(stripped, value)
    if first is None:
        return None, None, matched_texts
    return first[0], first[1], matched_texts


# --- Tokens --------------------------------------------------------------------


def tag_words(text: str) -> tuple[list[str], list[str]]:
    from pyvi import ViPosTagger, ViTokenizer

    words, tags = ViPosTagger.postagging(ViTokenizer.tokenize(text))
    return [w.replace("_", " ") for w in words], [str(t) for t in tags]


def _is_common_noun(tag: str) -> bool:
    return tag in {"N", "Nc", "Nu", "Ny", "Nb"}


def _split_ordinal(word: str, exact: bool) -> tuple[bool, list[str]]:
    """Split a compound like "thứ tư" into (has ordinal marker, number words)."""
    parts = word.split()
    if len(parts) > 1 and lexicon.is_one_of(parts[0], lexicon.ORDINAL_MARKERS, exact):
        return True, parts[1:]
    return False, parts


def _number_at(words: list[str], tags: list[str], index: int, exact: bool) -> tuple[int | None, int, bool]:
    """Number starting at ``index``: (value, index after it, starts with an ordinal marker)."""
    has_marker, parts = _split_ordinal(words[index], exact)
    if has_marker:
        value = lexicon.parse_number(parts, exact)
        return value, index + 1, True
    if tags[index] != "M" and not words[index].isdigit():
        return None, index, False
    end = index
    collected: list[str] = []
    while end < len(words) and (tags[end] == "M" or words[end].isdigit()):
        collected.extend(words[end].split())
        end += 1
    return lexicon.parse_number(collected, exact), end, False


def parse_turn(text: str, today: date | None = None) -> ParsedTurn:
    exact = lexicon.has_accents(text)
    parsed = ParsedTurn(text=text, exact_accents=exact)

    raw_words, raw_tags = tag_words(text)

    def after_noun(matched: str) -> bool:
        first_word = matched.split()[0]
        for position, word in enumerate(raw_words):
            if position and same_start(word, first_word):
                return _is_common_noun(raw_tags[position - 1])
        return False

    parsed.date_from, parsed.date_to, date_texts = find_dates(text, today, after_noun)
    parsed.date_text = ", ".join(date_texts)
    working = text
    for matched in date_texts:
        working = working.replace(matched, " ")

    parsed.source, source_names = _find_label(working, "source")
    parsed.category, category_names = _find_label(working, "category")
    label_words = {
        accented_lower(word) for name in source_names + category_names for word in _WORD.findall(name)
    } | {accented_lower(name) for name in source_names + category_names}

    words, tags = tag_words(working)
    parsed.words, parsed.tags = words, tags
    consumed: set[int] = set()

    # Numbers: item references, quantities and summary lengths.
    index = 0
    while index < len(words):
        value, after, marked = _number_at(words, tags, index, exact)
        if value is None:
            index += 1
            continue
        start = index
        previous = start - 1
        if not marked and previous >= 0 and lexicon.is_one_of(words[previous], lexicon.ORDINAL_MARKERS, exact):
            marked = True
            previous -= 1
        next_word = words[after] if after < len(words) else ""
        next_tag = tags[after] if after < len(words) else ""
        previous_tag = tags[previous] if previous >= 0 else ""
        if next_word and lexicon.is_one_of(next_word, lexicon.SENTENCE_UNITS, exact):
            parsed.max_sentences = value
            consumed.update(range(start, after + 1))
        elif previous_tag == "Np":
            pass  # part of a name: "Quả Bóng Vàng 2026"
        elif marked or _is_common_noun(previous_tag):
            parsed.item_refs.append(value)
            consumed.update(range(max(previous, 0), after))
        elif _is_common_noun(next_tag) or next_tag == "Np":
            parsed.count = value
            consumed.update(range(start, after + 1))
        index = max(after, index + 1)

    # Relative positions and demonstratives after a noun ("bài kế tiếp", "dự án này").
    for position in range(1, len(words)):
        if not _is_common_noun(tags[position - 1]):
            continue
        for name, options in lexicon.RELATIVE_WORDS.items():
            matched_len = _match_phrase(words, position, options, exact)
            if not matched_len:
                continue
            plural = position >= 2 and lexicon.is_one_of(words[position - 2], lexicon.PLURAL_MARKERS, exact)
            if name == "current" and plural:
                parsed.refers_to_list = True
            elif parsed.relative is None:
                parsed.relative = name
            consumed.update(range(position, position + matched_len))
            break

    # Questions.
    stripped = text.strip()
    last_word = _WORD.findall(stripped)[-1] if _WORD.findall(stripped) else ""
    parsed.is_question = (
        stripped.endswith("?")
        or any(_match_phrase(words, i, lexicon.QUESTION_WORDS, exact) for i in range(len(words)))
        or (last_word and lexicon.is_one_of(last_word, lexicon.FINAL_QUESTION_PARTICLES, exact))
    )

    # Topic words: everything left that is not a slot, a function word or a command word.
    stats = get_keyword_stats()
    seen: set[str] = set()
    for position, (word, tag) in enumerate(zip(words, tags)):
        if position in consumed or tag not in _CONTENT_TAGS or not _WORD.search(word):
            continue
        lowered = accented_lower(word)
        if lowered in label_words or lowered in seen:
            continue
        if any(lexicon.is_one_of(word, group, exact) for group in (
            lexicon.QUESTION_WORDS, lexicon.PLURAL_MARKERS, lexicon.ORDINAL_MARKERS,
            *lexicon.RELATIVE_WORDS.values(),
        )):
            continue
        sentence_start = position == 0 or tags[position - 1] == "F"
        is_name = tag == "Np" or (not sentence_start and word[:1].isupper()) or any(
            not is_vietnamese_syllable(part) for part in lowered.split()
        )
        if is_name and not lowered.isdigit():
            parsed.entity_words.append(lowered)
        elif not stats.is_topic_word(lowered):
            continue
        if not is_name and _is_date_word(lowered, today):
            continue  # a date word the first date search missed ("hôm nay" next to a date)
        seen.add(lowered)
        parsed.topic_words.append(lowered)
        if tag.startswith("N") or is_name:
            parsed.topic_nouns.append(lowered)
    if parsed.date_from is None:
        for word in list(parsed.topic_words):
            start, end, matched = find_dates(word, today)
            if start is not None:
                parsed.date_from, parsed.date_to, parsed.date_text = start, end, ", ".join(matched)
                break
    return parsed


def _is_date_word(word: str, today: date | None) -> bool:
    start, _, matched = find_dates(word, today)
    return start is not None and bool(matched) and accented_lower(matched[0]) == word


def same_start(token: str, word: str) -> bool:
    return accented_lower(token).split()[0] == accented_lower(word) if token.split() else False


def _match_phrase(words: list[str], position: int, options, exact: bool) -> int:
    """Length (in tokens) of the longest option that starts at ``position``."""
    best = 0
    for option in options:
        option_parts = option.split()
        span: list[str] = []
        length = 0
        while position + length < len(words) and len(span) < len(option_parts):
            span.extend(words[position + length].split())
            length += 1
        if len(span) == len(option_parts) and all(
            lexicon.same_word(a, b, exact) for a, b in zip(span, option_parts)
        ):
            best = max(best, length)
    return best


def describe(parsed: ParsedTurn) -> dict[str, Any]:
    """Compact view of the slots, for logs and tests."""
    return {
        key: value
        for key, value in {
            "item_refs": parsed.item_refs,
            "relative": parsed.relative,
            "refers_to_list": parsed.refers_to_list,
            "count": parsed.count,
            "max_sentences": parsed.max_sentences,
            "date": (parsed.date_from, parsed.date_to) if parsed.date_from else None,
            "source": parsed.source,
            "category": parsed.category,
            "topic_words": parsed.topic_words,
            "is_question": parsed.is_question,
        }.items()
        if value not in (None, [], False)
    }
