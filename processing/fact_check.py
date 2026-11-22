"""Check generated text against its source: numbers and names must appear there.

Used for Qwen answers (each sentence is checked before it is shown) and ViT5
summaries (unsupported sentences are dropped). It is a lexical check: it
catches invented numbers and names ("Chicago", "David Schultz", "3,5 triệu
tỷ" taken from another article) but not real words reused with a wrong
meaning ("tiền đạo 11 tuổi" from "gia nhập ở tuổi 11").
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from processing.text_matching import accented_lower, fold_text, is_vietnamese_syllable

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_CITATION = re.compile(r"\[\s*Nguồn\s*\d+(?:\s*[,;]\s*(?:Nguồn\s*)?\d+)*\s*\]", re.IGNORECASE)
_WORD = re.compile(r"[0-9a-zA-ZÀ-ỹĐđ]+")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
_DENIAL = re.compile(r"(?<![\wÀ-ỹ])(không|chưa|chẳng|sai)(?![\wÀ-ỹ])", re.IGNORECASE)
#: A sentence that is only a verdict ("Đúng.", "Không.") says nothing checkable on its own.
_VERDICT = re.compile(r"(?:đúng|sai|có|không|phải|chưa)(?:\s+(?:vậy|rồi|ạ))?", re.IGNORECASE)


def strip_citations(text: str) -> str:
    return _CITATION.sub(" ", text or "")


def split_glued_words(text: str) -> str:
    """Put a space where a lowercase letter is followed by a capital: "Sơn LaSáng" -> "Sơn La Sáng".

    The crawler glued each article's dateline to its first sentence, and the model
    copies that ("Sơn LaĐã"). Both the source and the generated text are split
    the same way, so names like "VnExpress" still match ("Vn Express" on both sides).
    """
    chars = list(text or "")
    out: list[str] = []
    for index, char in enumerate(chars):
        if index and char.isupper() and chars[index - 1].islower():
            out.append(" ")
        out.append(char)
    return "".join(out)


def _number_forms(number: str) -> set[str]:
    """Spellings that count as the same number: "43.000", "43,000", "43000"; "3,5" and "3.5"."""
    forms = {number, number.replace(",", "."), number.replace(".", ",")}
    plain = re.sub(r"[.,]", "", number)
    forms.add(plain)
    if plain.isdigit():
        forms.add(str(int(plain)))  # "07" and "7" are the same number
    return forms


def numbers_in(text: str) -> set[str]:
    return set(_NUMBER.findall(strip_citations(text)))


def _source_numbers(source: str) -> set[str]:
    found: set[str] = set()
    for number in numbers_in(source):
        found |= _number_forms(number)
    return found


def names_in(text: str) -> set[str]:
    """Proper names: capitalised words inside a sentence, and words that are not Vietnamese.

    Sentence-initial capitals are ignored. Returns lowercased, accent-free names.
    """
    names: set[str] = set()
    for sentence in _SENTENCE_END.split(split_glued_words(strip_citations(text))):
        words = _WORD.findall(sentence)
        for position, word in enumerate(words):
            if word.isdigit():
                continue
            capitalised = position > 0 and word[:1].isupper()
            foreign = not is_vietnamese_syllable(word.lower()) and not word.isupper()
            if capitalised or foreign:
                names.add(fold_text(word))
    return names


@dataclass
class FactCheck:
    unsupported_numbers: list[str]
    unsupported_names: list[str]

    @property
    def supported(self) -> bool:
        return not (self.unsupported_numbers or self.unsupported_names)


def denies(text: str) -> bool:
    """Does the sentence negate something ("không", "chưa", "sai")?"""
    return bool(_DENIAL.search(strip_citations(text)))


def is_verdict(text: str) -> bool:
    """"Đúng.", "Không ạ.", "Có [Nguồn 1]." — a yes or no with nothing else."""
    plain = re.sub(r"[\s.,!?…:;]+", " ", strip_citations(text)).strip()
    return bool(plain) and bool(_VERDICT.fullmatch(plain))


def check(text: str, source: str, allowed: str = "") -> FactCheck:
    """Numbers and names of ``text`` that appear neither in ``source`` nor in ``allowed``.

    ``allowed`` is usually the user's question. Callers pass it only for sentences
    that deny something: "Bài không nhắc đến Huế" may repeat the user's "Huế",
    but "Tuần lễ Số diễn ra ở Huế" must find "Huế" in the source, otherwise a
    false premise in the question would pass as a fact.
    """
    known_numbers = _source_numbers(source) | _source_numbers(allowed)
    missing_numbers = [n for n in sorted(numbers_in(text)) if not (_number_forms(n) & known_numbers)]
    known_words = {
        fold_text(w) for w in _WORD.findall(accented_lower(split_glued_words(f"{source} {allowed}")))
    }
    missing_names = [name for name in sorted(names_in(text)) if name.lower() not in known_words]
    return FactCheck(missing_numbers, missing_names)


def split_sentences_keep(text: str) -> list[str]:
    return [part for part in _SENTENCE_END.split(text or "") if part.strip()]


def keep_supported_sentences(summary: str, source: str) -> tuple[str, int]:
    """Drop summary sentences with a number or name the source does not contain."""
    kept: list[str] = []
    dropped = 0
    for sentence in split_sentences_keep(summary):
        if check(sentence, source).supported:
            kept.append(sentence.strip())
        else:
            dropped += 1
    return " ".join(kept), dropped


def fact_support(text: str, source: str) -> float | None:
    """Share of the numbers and names in ``text`` that the source contains (None if it has none)."""
    numbers = numbers_in(text)
    names = names_in(text)
    total = len(numbers) + len(names)
    if total == 0:
        return None
    result = check(text, source)
    missing = len(result.unsupported_numbers) + len(result.unsupported_names)
    return (total - missing) / total
