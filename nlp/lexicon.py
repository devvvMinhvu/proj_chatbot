"""Closed-class Vietnamese words the turn parser needs.

Everything that can be learned from data is learned elsewhere (keyword
statistics, publisher and category names from the database). What remains here
is grammar that a parser without a trained model cannot infer: number words,
the ordinal marker, a few direction words, demonstratives and question words.
Words are matched accent-aware (see ``processing.text_matching``), so "sau"
(after) and "sáu" (six) stay distinct when typed with accents.
"""

from __future__ import annotations

from processing.text_matching import accented_lower, fold_text, needs_exact_accents

# --- Numbers --------------------------------------------------------------------

DIGIT_WORDS = {
    "không": 0, "linh": 0, "lẻ": 0,
    "một": 1, "mốt": 1, "nhất": 1,
    "hai": 2, "nhì": 2,
    "ba": 3,
    "bốn": 4, "tư": 4,
    "năm": 5, "lăm": 5,
    "sáu": 6,
    "bảy": 7, "bẩy": 7,
    "tám": 8,
    "chín": 9,
}
TEN_WORDS = {"mười": 10}
TENS_MARKERS = {"mươi"}
HUNDRED_WORDS = {"trăm": 100}

#: Words that turn the following number into a position ("thứ hai", "số 3").
ORDINAL_MARKERS = ("thứ", "số")

# --- Positions relative to the open article ------------------------------------

RELATIVE_WORDS = {
    "next": ("kế tiếp", "tiếp theo", "kế", "sau"),
    "previous": ("trước", "trước đó"),
    "first": ("đầu tiên", "đầu"),
    "last": ("cuối", "cuối cùng"),
    "current": ("này", "đó", "ấy", "kia", "vừa rồi", "vừa xong", "trên"),
}
PLURAL_MARKERS = ("những", "các")

# --- Questions and summary length ----------------------------------------------

QUESTION_WORDS = (
    "ai", "gì", "nào", "đâu", "sao", "mấy", "bao nhiêu", "bao giờ", "khi nào",
    "thế nào", "như thế nào", "ra sao", "vì sao", "tại sao", "làm sao", "có phải",
)
#: Question particles that only ask when they end the sentence ("… không?").
FINAL_QUESTION_PARTICLES = ("không", "chưa", "hả", "à")
#: Units that make a number a summary length ("một câu").
SENTENCE_UNITS = ("câu",)


def has_accents(text: str) -> bool:
    """True when any letter of ``text`` carries a Vietnamese accent or diacritic."""
    lowered = accented_lower(text)
    return fold_text(lowered) != lowered


def same_word(typed: str, reference: str, exact: bool | None = None) -> bool:
    """Accent-aware equality.

    ``exact`` should say whether the user's message was typed with accents; then
    words must match exactly, so "sau" (after) never equals "sáu" (six). In an
    unaccented message words match in accent-free form. When ``exact`` is None it
    is inferred from the word alone.
    """
    typed_lower = accented_lower(typed).replace("_", " ").strip()
    reference_lower = accented_lower(reference)
    if exact is None:
        exact = has_accents(typed_lower) or needs_exact_accents(typed_lower)
    if exact:
        return typed_lower == reference_lower
    return fold_text(typed_lower) == fold_text(reference_lower)


def is_one_of(typed: str, words, exact: bool | None = None) -> bool:
    return any(same_word(typed, word, exact) for word in words)


def number_word_value(word: str, exact: bool | None = None) -> int | None:
    for table in (DIGIT_WORDS, TEN_WORDS, HUNDRED_WORDS):
        for key, value in table.items():
            if same_word(word, key, exact):
                return value
    return None


def parse_number(words: list[str], exact: bool | None = None) -> int | None:
    """Value of a Vietnamese number written in digits or words.

    Handles "12", "mười hai", "hai mươi mốt", "một trăm linh năm" and ordinal
    forms after "thứ" ("nhất", "nhì", "tư"). Returns None when the words are
    not a number.
    """
    if not words:
        return None
    if len(words) == 1 and words[0].isdigit():
        return int(words[0])
    total = 0
    current = 0
    seen = False
    for word in words:
        if word.isdigit():
            current = current * 10 + int(word) if seen and current < 10 else int(word)
            seen = True
            continue
        if is_one_of(word, TENS_MARKERS, exact):
            current = (current or 1) * 10
            seen = True
            continue
        value = number_word_value(word, exact)
        if value is None:
            return None
        seen = True
        if value == 100:
            total += (current or 1) * 100
            current = 0
        elif value == 10:
            current += 10
        else:
            current += value
    return total + current if seen else None
