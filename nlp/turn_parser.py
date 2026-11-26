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
from nlp.keyword_stats import get_keyword_stats, item_nouns
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
    #: The item numbers follow a list word ("bài 1 và bài 2"), not another noun ("lần thứ 8").
    refs_named: bool = False
    #: Item numbers in a negated or "instead of" clause ("không phải bài 2").
    excluded_refs: list[int] = field(default_factory=list)
    relative: str | None = None
    refers_to_list: bool = False
    #: A demonstrative after a noun that is not a list item ("dự án này", "khoảng đó"):
    #: the message points back to the previous subject, not to an article.
    refers_back: bool = False
    #: "ba bài khác": items of the same list that were not shown yet.
    wants_other: bool = False
    count: int | None = None
    max_sentences: int | None = None
    date_from: date | None = None
    date_to: date | None = None
    date_text: str = ""
    source: str | None = None
    exclude_sources: list[str] = field(default_factory=list)
    #: "tìm trên tất cả các báo": lift any publisher filter.
    all_sources: bool = False
    category: str | None = None
    topic_words: list[str] = field(default_factory=list)
    #: Topic words that are nouns or names: a message with none brings no new subject.
    topic_nouns: list[str] = field(default_factory=list)
    entity_words: list[str] = field(default_factory=list)
    #: Topic words that are verbs ("chia tay", "tham dự"): an article may say them differently.
    predicate_words: list[str] = field(default_factory=list)
    #: Nouns right before a question word ("món gì", "tập mấy", "giải nào"): they say what
    #: kind of answer is wanted, not what the question is about.
    frame_words: list[str] = field(default_factory=list)
    is_question: bool = False
    #: A yes-or-no question ("Đúng hay sai: …", "Có phải … không?"): its details are claims.
    yes_no: bool = False

    @property
    def has_slots(self) -> bool:
        """True when the message sets a filter, count, length or date."""
        return self.all_sources or bool(self.exclude_sources) or any(
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


def _label_matches(text: str, kind: str) -> list[tuple[str, str]]:
    """(slug, name as written in the vocabulary) for every label ``text`` mentions."""
    matches: list[tuple[str, str]] = []
    for slug, names in label_vocabulary()[kind].items():
        for name in names:
            words = _WORD.findall(name)
            if contains_phrase(words, text) or (len(words) == 1 and _contains_word(text, name)):
                matches.append((slug, name))
                break
    return matches


def _find_label(text: str, kind: str) -> tuple[str | None, list[str]]:
    matches = _label_matches(text, kind)
    slugs = {slug for slug, _ in matches}
    if len(slugs) != 1:
        return None, []
    return matches[0][0], [name for _, name in matches]


def _negated_positions(words: list[str], tags: list[str], exact: bool, is_question: bool) -> list[bool]:
    """For each token: does a negation or exclusion earlier in its clause apply to it?

    A clause ends at punctuation or a contrast word ("mà", "nhưng"). "không" and
    "chưa" negate only in statements, since they end questions ("… không?").
    """
    negators = [*lexicon.NEGATORS, *lexicon.EXCLUSION_WORDS]
    if not is_question:
        negators += list(lexicon.STATEMENT_NEGATORS)
    flags: list[bool] = []
    negated = False
    index = 0
    while index < len(words):
        length = _match_phrase(words, index, negators, exact)
        if length:
            negated = True
            flags.extend([False] * length)
            index += length
            continue
        if tags[index] == "F" or lexicon.is_one_of(words[index], lexicon.CONTRAST_WORDS, exact):
            negated = False
        flags.append(negated)
        index += 1
    return flags


def _name_positions(words: list[str], name: str, exact: bool) -> list[int]:
    target = [accented_lower(part) for part in _WORD.findall(name)]
    return [i for i in range(len(words)) if _match_phrase(words, i, (" ".join(target),), exact)]


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
    """Date expressions in ``text`` as one (from, to) range, and all matched words.

    Several dates make a range from the earliest to the latest ("từ 21/10 đến
    23/10"). ``after_noun(matched)`` tells whether the match directly follows a
    common noun; then a weekday-shaped match such as "thứ hai" is an item
    position ("tin thứ hai"), not Monday. A match with digits needs a date
    separator or a date unit, so "bài số 9 trong" is not a date.
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
    exact = lexicon.has_accents(text)
    ranges: list[tuple[date, date]] = []
    matched_texts: list[str] = []
    for matched, value in found:
        stripped = matched.strip()
        if re.fullmatch(r"[\d\s]+", stripped):
            continue  # a bare number ("2026", "3") is not a date
        if re.search(r"\d", stripped) and not _DATE_SEPARATOR.search(stripped) and not any(
            lexicon.is_one_of(word, lexicon.DATE_UNITS, exact) for word in _WORD.findall(stripped)
        ):
            continue  # "9 trong": a number next to a word, not a date
        if fold_text(stripped).startswith("thu ") and after_noun and after_noun(stripped):
            continue  # "tin thứ hai" is the second item, not Monday
        matched_texts.append(stripped)
        ranges.append(_date_range(stripped, value))
    if not ranges:
        return None, None, matched_texts
    return min(start for start, _ in ranges), max(end for _, end in ranges), matched_texts


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
        """Does the matched date text start right after a common noun ("bài thứ ba")?"""
        target = [accented_lower(part) for part in _WORD.findall(matched)][:2]
        for position in range(1, len(raw_words)):
            if _match_phrase(raw_words, position, (" ".join(target),), exact) or (
                len(target) == 2 and same_start(raw_words[position], target[0])
                and accented_lower(raw_words[position]) == " ".join(target)
            ):
                if _is_common_noun(raw_tags[position - 1]):
                    return True
        return False

    parsed.date_from, parsed.date_to, date_texts = find_dates(text, today, after_noun)
    parsed.date_text = ", ".join(date_texts)
    working = text
    for matched in date_texts:
        working = working.replace(matched, " ")

    words, tags = tag_words(working)
    parsed.words, parsed.tags = words, tags
    consumed: set[int] = set()

    # Questions (needed first: "không" negates only in statements).
    stripped = text.strip()
    found_words = _WORD.findall(stripped)
    last_word = found_words[-1] if found_words else ""
    parsed.is_question = bool(
        stripped.endswith("?")
        or any(_match_phrase(words, i, lexicon.QUESTION_WORDS, exact) for i in range(len(words)))
        or (last_word and lexicon.is_one_of(last_word, lexicon.FINAL_QUESTION_PARTICLES, exact))
    )
    negated = _negated_positions(words, tags, exact, parsed.is_question)

    # Yes-or-no questions: a verification phrase, or a final "không/chưa" with no
    # question word ("Bài 2 có nhắc đến VnExpress không?"). The phrase is not a topic.
    verifying = False
    asks_wh = False
    for position in range(len(words)):
        length = _match_phrase(words, position, lexicon.VERIFICATION_PHRASES, exact)
        if length:
            verifying = True
            consumed.update(range(position, position + length))
        elif _match_phrase(words, position, lexicon.QUESTION_WORDS, exact):
            asks_wh = True
    ends_with_particle = bool(last_word) and lexicon.is_one_of(last_word, ("không", "chưa"), exact)
    parsed.yes_no = parsed.is_question and (verifying or (ends_with_particle and not asks_wh))

    # Publishers: a name in a negated clause is left out ("trừ VnExpress").
    included: list[str] = []
    source_names: list[str] = []
    for slug, name in _label_matches(working, "source"):
        source_names.append(name)
        positions = _name_positions(words, name, exact)
        if positions and all(negated[p] for p in positions):
            parsed.exclude_sources.append(slug)
        else:
            included.append(slug)
    parsed.source = included[0] if len(included) == 1 else None
    parsed.category, category_names = _find_label(working, "category")
    label_words = {
        accented_lower(word) for name in source_names + category_names for word in _WORD.findall(name)
    } | {accented_lower(name) for name in source_names + category_names}

    # Numbers: item references, quantities and summary lengths.
    count_noun: int | None = None
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
        elif not marked and _is_common_noun(previous_tag) and _is_common_noun(next_tag):
            # Between two nouns the number counts the second: "danh sách 10 tin".
            parsed.count = value
            count_noun = after
            consumed.update(range(start, after + 1))
        elif marked or _is_common_noun(previous_tag):
            (parsed.excluded_refs if negated[start] else parsed.item_refs).append(value)
            if previous >= 0:
                noun = words[previous]
                named = lexicon.is_one_of(noun, item_nouns(), exact) or lexicon.is_one_of(
                    noun.split()[0], item_nouns(), exact
                )
                parsed.refs_named = (parsed.refs_named or len(parsed.item_refs) <= 1) and named
            consumed.update(range(max(previous, 0), after))
        elif _is_common_noun(next_tag) or next_tag == "Np":
            parsed.count = value
            count_noun = after
            consumed.update(range(start, after + 1))
        index = max(after, index + 1)

    # Positions and demonstratives after a noun. After a list item ("bài kế tiếp",
    # "tin này", "bài vừa đọc") they select an article; after any other noun
    # ("dự án này", "khoảng đó") they point back to the previous subject.
    items = item_nouns()
    for position in range(1, len(words)):
        noun_at = position - 1
        if not _is_common_noun(tags[noun_at]):
            continue
        noun = words[noun_at]
        is_item = not items or any(
            lexicon.is_one_of(form, items, exact) for form in (noun, noun.split()[0])
        )
        if is_item and lexicon.is_one_of(words[position], lexicon.OTHER_WORDS, exact):
            parsed.wants_other = True  # "ba bài khác"
            consumed.add(position)
            continue
        name, matched_len = None, 0
        if (
            lexicon.is_one_of(words[position], lexicon.RECENT_MARKERS, exact)
            and position + 1 < len(words)
            and tags[position + 1].startswith("V")
        ):
            name, matched_len = "current", 2  # "bài vừa đọc"
        else:
            for option_name, options in lexicon.RELATIVE_WORDS.items():
                matched_len = _match_phrase(words, position, options, exact)
                if matched_len:
                    name = option_name
                    break
        if name is None:
            continue
        consumed.update(range(position, position + matched_len))
        if not is_item:
            if name == "current":
                parsed.refers_back = True
            continue
        plural = noun_at >= 1 and (
            lexicon.is_one_of(words[noun_at - 1], lexicon.PLURAL_MARKERS, exact) or count_noun == noun_at
        )
        if name == "current" and plural:
            parsed.refers_to_list = True
            if count_noun == noun_at:
                parsed.count = None  # "hai bài vừa nêu" names the shown list, not a count
        elif parsed.relative is None:
            parsed.relative = name

    # "tất cả các báo", "mọi nguồn": all publishers. Words of a question phrase are
    # not a publisher noun: unaccented "moi bao nhieu" is "mới bao nhiêu", not "mọi báo".
    question_tokens: set[int] = set()
    for position in range(len(words)):
        length = _match_phrase(words, position, lexicon.QUESTION_WORDS, exact)
        question_tokens.update(range(position, position + length))
    for position in range(len(words)):
        length = _match_phrase(words, position, lexicon.ALL_WORDS, exact)
        if not length:
            continue
        nxt = position + length
        if nxt < len(words) and lexicon.is_one_of(words[nxt], lexicon.PLURAL_MARKERS, exact):
            nxt += 1
        if nxt < len(words) and nxt not in question_tokens and _match_phrase(words, nxt, lexicon.PUBLISHER_NOUNS, exact):
            parsed.all_sources = True
            consumed.update(range(position, nxt + 1))

    # An article picked by number needs no date ("Bây giờ mở bài số 9").
    if parsed.item_refs:
        parsed.date_from = parsed.date_to = None

    # Topic words: everything left that is not a slot, a function word or a command word.
    stats = get_keyword_stats()
    seen: set[str] = set()
    grammar = (
        lexicon.QUESTION_WORDS, lexicon.PLURAL_MARKERS, lexicon.ORDINAL_MARKERS,
        lexicon.NEGATORS, lexicon.STATEMENT_NEGATORS, lexicon.EXCLUSION_WORDS,
        lexicon.CONTRAST_WORDS, lexicon.RECENT_MARKERS, lexicon.IMPERATIVE_WORDS, *lexicon.RELATIVE_WORDS.values(),
    )
    for position, (word, tag) in enumerate(zip(words, tags)):
        if position in consumed or tag not in _CONTENT_TAGS or not _WORD.search(word):
            continue
        lowered = accented_lower(word)
        if lowered in label_words or lowered in seen:
            continue
        if any(lexicon.is_one_of(word, group, exact) for group in grammar):
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
        elif tag.startswith("V"):
            parsed.predicate_words.append(lowered)
        # A noun or verb right before a question word frames it ("món gì", "tổ chức nào").
        if tag[:1] in {"N", "V"} and not is_name and position + 1 < len(words) and _match_phrase(
            words, position + 1, lexicon.QUESTION_WORDS, exact
        ):
            parsed.frame_words.append(lowered)
    if parsed.date_from is None and not parsed.item_refs:
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
            "excluded_refs": parsed.excluded_refs,
            "relative": parsed.relative,
            "refers_to_list": parsed.refers_to_list,
            "refers_back": parsed.refers_back,
            "wants_other": parsed.wants_other,
            "count": parsed.count,
            "max_sentences": parsed.max_sentences,
            "date": (parsed.date_from, parsed.date_to) if parsed.date_from else None,
            "source": parsed.source,
            "exclude_sources": parsed.exclude_sources,
            "all_sources": parsed.all_sources,
            "category": parsed.category,
            "topic_words": parsed.topic_words,
            "is_question": parsed.is_question,
            "yes_no": parsed.yes_no,
        }.items()
        if value not in (None, [], False)
    }
