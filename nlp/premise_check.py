"""Check the claim inside a yes-or-no question against an article before answering.

"Đúng hay sai: Tuần lễ Số được tổ chức ở Huế cho 700 đại biểu?" carries its own
details. The language model tends to agree with them, and the answer check
(``processing.fact_check``) lets an answer repeat what the user wrote. So the
names and numbers of the claim are looked up in the article first: when some
are missing, the reply says so and quotes the article's sentences on the same
point ("… diễn ra từ ngày 28 đến 30/10 tại Quảng Ninh", "… dự kiến quy tụ 500
đại biểu"), without asking the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from nlp import lexicon
from nlp.turn_parser import ParsedTurn
from processing.fact_check import check, names_in, numbers_in, split_glued_words
from processing.text_matching import accented_lower, fold_text, matched_keywords

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
_WORD = re.compile(r"[0-9a-zA-ZÀ-ỹĐđ]+")
#: Prepositions of place: a claim "ở Huế" is answered by a sentence with "tại Quảng Ninh".
_PLACE_WORDS = ("ở", "tại")
#: Verbs too common to tie a sentence to the claim.
_LIGHT_VERBS = ("được", "là", "có", "bị", "cho", "làm")


@dataclass
class PremiseCheck:
    missing_names: list[str] = field(default_factory=list)  # as the user wrote them
    missing_numbers: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)  # article sentences on the same points

    @property
    def holds(self) -> bool:
        return not (self.missing_names or self.missing_numbers)


def claim_parts(question: str) -> tuple[list[str], list[str]]:
    """Names (as written) and numbers of the question's claim."""
    folded = names_in(question)
    names: list[str] = []
    for word in _WORD.findall(question):
        parts = _WORD.findall(split_glued_words(word))  # "VnExpress" is checked as "Vn Express"
        if word not in names and not word.isdigit() and any(fold_text(accented_lower(p)) in folded for p in parts):
            names.append(word)
    return names, sorted(numbers_in(question), key=question.find)


def _sentences(text: str) -> list[str]:
    """The text's sentences, each once (the crawler sometimes stored a paragraph twice)."""
    seen: list[str] = []
    for part in _SENTENCE_END.split(split_glued_words(text or "")):
        part = part.strip()
        if part and part not in seen:
            seen.append(part)
    return seen


def _evidence(parsed: ParsedTurn, sentences: list[str], missing_names: list[str], missing_numbers: list[str]) -> list[str]:
    """Up to two article sentences about the same points as the claim, in article order."""
    words, tags = parsed.words, parsed.tags
    missing = {accented_lower(w) for w in missing_names + missing_numbers}
    keywords = [
        accented_lower(word)
        for word, tag in zip(words, tags)
        if tag[:1] in {"N", "V", "A"} and accented_lower(word) not in missing
        and not lexicon.is_one_of(word, _LIGHT_VERBS + lexicon.QUESTION_NOUNS + lexicon.VERIFICATION_PHRASES)
        and not word.isdigit()
    ]
    units = []  # "700 đại biểu" -> "đại biểu"
    places = False  # "ở Huế"
    for index, word in enumerate(words):
        if word in missing_numbers and index + 1 < len(words) and tags[index + 1].startswith("N"):
            units.append(accented_lower(words[index + 1]))
        if word in missing_names and index and lexicon.is_one_of(words[index - 1], _PLACE_WORDS):
            places = True
    scored = []
    for position, sentence in enumerate(sentences):
        score = len(matched_keywords(keywords, sentence, exact=parsed.exact_accents))
        lowered = accented_lower(sentence)
        if any(re.search(r"\d[\d.,]*\s+" + re.escape(unit) + r"(?![\wÀ-ỹ])", lowered) for unit in units):
            score += 3
        if places and re.search(r"(?<![\wÀ-ỹ])(?:ở|tại)\s+[A-ZÀ-ỸĐ]", sentence):
            score += 3
        if score >= 2:
            scored.append((score, position, sentence))
    best = sorted(scored, key=lambda item: (-item[0], item[1]))[:2]
    return [sentence for _, _, sentence in sorted(best, key=lambda item: item[1])]


def check_premise(
    question: str, parsed: ParsedTurn, article_text: str, body: str | None = None
) -> PremiseCheck | None:
    """Missing names and numbers of a yes-or-no question's claim; None if there is no claim.

    ``article_text`` is everything known about the article (title, publisher, text);
    evidence sentences are taken from ``body`` when given.
    """
    if not parsed.yes_no:
        return None
    names, numbers = claim_parts(question)
    if not (names or numbers):
        return None
    # Names are matched with their accents when the user typed them: "Huế" is not "huệ".
    missing_names = [
        name for name in names
        if not matched_keywords([name], article_text, exact=parsed.exact_accents)
        and not matched_keywords([" ".join(_WORD.findall(split_glued_words(name)))], split_glued_words(article_text))
    ]
    unsupported = set(check(" ".join(numbers), article_text).unsupported_numbers)
    missing_numbers = [n for n in numbers if n in unsupported]
    premise = PremiseCheck(missing_names, missing_numbers)
    if not premise.holds:
        premise.evidence = _evidence(
            parsed, _sentences(article_text if body is None else body), missing_names, missing_numbers
        )
    return premise


#: "A hay B": a question that offers two answers.
_ALTERNATIVE_WORDS = ("hay", "hoặc", "hay là")
_COPULAS = ("là",)


def _option_words(words: list[str], tags: list[str]) -> list[str]:
    return [
        accented_lower(word) for word, tag in zip(words, tags)
        if (tag[:1] in {"N", "V", "A"} or tag == "M") and not lexicon.is_one_of(word, _LIGHT_VERBS)
    ]


def alternative_choice(question: str, parsed: ParsedTurn, body: str) -> tuple[str, str] | None:
    """For "A hay B?", the option the article states and the sentence that states it.

    "Đại biểu là con số dự kiến hay con số thực tế đã tham dự?": the words only
    option A has ("dự kiến") are in the article, a word only B has ("thực tế") is
    not, so the article answers A. None when both or neither option is there, or
    the question offers no choice ("đúng hay sai" asks for a verdict).
    """
    words, tags = parsed.words, parsed.tags
    splits = [
        i for i, word in enumerate(words)
        if lexicon.is_one_of(word, _ALTERNATIVE_WORDS, parsed.exact_accents)
        and not any(
            _phrase_covers(words, i, phrase, parsed.exact_accents)
            for phrase in lexicon.VERIFICATION_PHRASES + ("hay không",)
        )
    ]
    if len(splits) != 1:
        return None
    split = splits[0]
    # The left option is as long as the right one; what comes before it is shared
    # ("Sự kiện diễn ra ở | Quảng Ninh hay Hà Nội?"). A copula also ends the shared part.
    right_length = sum(1 for tag in tags[split + 1:] if tag != "F")
    copulas = [i + 1 for i in range(split) if lexicon.is_one_of(words[i], _COPULAS, parsed.exact_accents)]
    left_start = max([split - right_length, *copulas, 0])
    left = _option_words(words[left_start:split], tags[left_start:split])
    right = _option_words(words[split + 1:], tags[split + 1:])
    only_left = [w for w in left if w not in right]
    only_right = [w for w in right if w not in left]
    if not only_left or not only_right:
        return None
    sentences = _sentences(body)

    def found(option: list[str]) -> bool:
        return len(matched_keywords(option, body, exact=parsed.exact_accents)) == len(set(option))

    left_found, right_found = found(only_left), found(only_right)
    if left_found == right_found:
        return None
    chosen_words, chosen = (
        (only_left, words[left_start:split]) if left_found else (only_right, words[split + 1:])
    )
    keywords = chosen_words + [w for w in parsed.topic_words if w not in only_left + only_right]
    best = max(
        sentences,
        key=lambda sentence: (
            len(matched_keywords(chosen_words, sentence, exact=parsed.exact_accents)),
            len(matched_keywords(keywords, sentence, exact=parsed.exact_accents)),
        ),
        default="",
    )
    if not matched_keywords(chosen_words, best, exact=parsed.exact_accents):
        return None
    shown = [word for word in chosen if _WORD.search(word)]
    while shown and lexicon.is_one_of(shown[0], _LIGHT_VERBS):
        shown = shown[1:]  # "có 500" -> "500"
    return " ".join(shown), best


def _phrase_covers(words: list[str], index: int, phrase: str, exact: bool) -> bool:
    """Is token ``index`` part of ``phrase`` at some position?"""
    parts = phrase.split()
    for start in range(max(0, index - len(parts) + 1), index + 1):
        span = words[start:start + len(parts)]
        if len(span) == len(parts) and all(lexicon.same_word(a, b, exact) for a, b in zip(span, parts)):
            return True
    return False


def alternative_reply(option: str, sentence: str, *, label: str = "[Nguồn 1]") -> str:
    return f"Theo bài viết, đó là {option}:\n\n> {sentence} {label}"


def subject_in(parsed: ParsedTurn, article_text: str) -> bool:
    """Is the question's subject (its first noun or name) in the article?"""
    subjects = [w for w in parsed.topic_nouns if not lexicon.is_one_of(w, lexicon.QUESTION_NOUNS)]
    if not subjects:
        return False
    return bool(matched_keywords(subjects[:1], article_text, exact=parsed.exact_accents))


def premise_reply(premise: PremiseCheck, title: str, *, label: str = "[Nguồn 1]", found_any: bool = True) -> str:
    """The reply for a claim the article does not confirm."""
    gaps = []
    if premise.missing_names:
        gaps.append("không nhắc đến " + ", ".join(premise.missing_names))
    if premise.missing_numbers:
        gaps.append("không có con số " + ", ".join(premise.missing_numbers))
    opening = (
        f"Bài \"{title}\" không xác nhận điều này: bài {' và '.join(gaps)}."
        if found_any
        else f"Tôi không tìm thấy bài nào xác nhận điều này. Bài gần nhất, \"{title}\", {' và '.join(gaps)}."
    )
    if not premise.evidence:
        return opening
    quoted = "\n\n".join(f"> {sentence} {label}" for sentence in premise.evidence)
    return f"{opening} Bài viết ghi:\n\n{quoted}"
