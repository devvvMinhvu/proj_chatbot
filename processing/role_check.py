"""Who left and who arrived: check that an answer does not swap people's roles.

The number-and-name check (``processing.fact_check``) passes an answer that uses
the right names in the wrong places: "Long An chia tay HLV Huỳnh Ngọc San và bổ
nhiệm Victor Mansaray" when the article says the club replaced Trịnh Văn Hậu
with San, and Mansaray is a striker of the opposing club. This module reads the
role changes a text states with a small hand-written table of verbs ("thay",
"bổ nhiệm", "rời ghế", ...): each verb gives a role to the name right after it
(its object) or right before it (its subject). An answer sentence is rejected
when it gives a name the opposite role the source gives that name, or a role the
source gives only to someone else.

It covers changes of position (coaches, executives) and nothing else; other
relations ("who won", "who plays whom") are not checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

from processing.fact_check import split_glued_words, strip_citations
from processing.text_matching import accented_lower, fold_text, is_vietnamese_syllable

LEAVE, JOIN = "rời đi", "nhận vị trí"

#: verb -> (role of the name after it, role of the name before it).
ROLE_VERBS: dict[str, tuple[str | None, str | None]] = {
    "chia tay": (LEAVE, None),
    "sa thải": (LEAVE, None),
    "miễn nhiệm": (LEAVE, None),
    "thay": (LEAVE, None),  # the subject is usually the club or company ("Long An thay HLV Hậu")
    "thay thế": (LEAVE, JOIN),
    "kế nhiệm": (LEAVE, JOIN),
    "bổ nhiệm": (JOIN, None),
    "chọn": (JOIN, None),
    "tân": (JOIN, None),
    "tiếp quản": (None, JOIN),
    "nhận ghế": (None, JOIN),
    "ngồi ghế": (None, JOIN),
    "nhậm chức": (None, JOIN),
    "ra mắt": (None, JOIN),
    "rời": (None, LEAVE),
    "rời ghế": (None, LEAVE),
    "từ chức": (None, LEAVE),
    "mất ghế": (None, LEAVE),
    "thôi chức": (None, LEAVE),
}
#: "được bổ nhiệm", "bị sa thải": the name before the verb takes the object's role.
PASSIVE_MARKERS = ("được", "bị")
#: "thay HLV A bằng B": after these verbs, the name after "bằng" is the newcomer.
REPLACING_VERBS = ("thay", "thay thế")
INSTEAD_MARKERS = ("bằng",)
#: Words between a name and its verb that change nothing ("San đã được chọn").
AUXILIARIES = ("đã", "đang", "sẽ", "vừa", "mới", "chính thức", "cũng")
#: Titles between a verb and its name ("chia tay HLV Trịnh Văn Hậu", "bổ nhiệm ông Hiếu").
TITLES = (
    "ông", "bà", "anh", "chị", "hlv", "huấn luyện viên", "tổng giám đốc", "giám đốc",
    "chủ tịch", "ceo", "trợ lý", "thuyền trưởng",
)

_TOKEN = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:-[0-9A-Za-zÀ-ỹĐđ]+)*|[^\sA-Za-zÀ-ỹĐđ0-9]")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
_VERB_PARTS = sorted(((tuple(verb.split()), roles) for verb, roles in ROLE_VERBS.items()), key=lambda v: -len(v[0]))


@dataclass(frozen=True)
class RoleClaim:
    name: tuple[str, ...]  # folded name words
    role: str
    sentence: str


def _phrase_at(tokens: list[str], index: int, phrases) -> int:
    """Length of the longest phrase of ``phrases`` (lowercased words) starting at ``index``."""
    best = 0
    for phrase in phrases:
        parts = phrase.split() if isinstance(phrase, str) else list(phrase)
        if [accented_lower(t) for t in tokens[index:index + len(parts)]] == parts:
            best = max(best, len(parts))
    return best


def _is_name_word(tokens: list[str], index: int) -> bool:
    word = tokens[index]
    if not word[:1].isalpha() or not word[:1].isupper():
        return False
    if len(word) > 1 and word.isupper():
        return False  # an abbreviation such as HLV or CLB is a title, not a name
    if _phrase_at(tokens, index, TITLES):
        return False
    starts_sentence = index == 0 or not tokens[index - 1][:1].isalnum()
    if starts_sentence:
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        return not is_vietnamese_syllable(accented_lower(word)) or (
            following[:1].isupper() and following[:1].isalpha() and not following.isupper()
        )
    return True


def _name_from(tokens: list[str], index: int) -> tuple[str, ...]:
    """The name that starts at ``index`` after skipping titles, or ()."""
    for _ in range(2):
        length = _phrase_at(tokens, index, TITLES)
        if not length and index < len(tokens) and len(tokens[index]) > 1 and tokens[index].isupper():
            length = 1
        if not length:
            break
        index += length
    words = []
    while index < len(tokens) and _is_name_word(tokens, index):
        words.append(fold_text(accented_lower(tokens[index])))
        index += 1
    return tuple(words)


def _skip_auxiliaries(tokens: list[str], end: int) -> int:
    """Move ``end`` back over auxiliaries ("đã", "chính thức") that end there."""
    moved = True
    while moved and end > 0:
        moved = False
        for aux in AUXILIARIES:
            parts = aux.split()
            if end - len(parts) >= 0 and [accented_lower(t) for t in tokens[end - len(parts):end]] == parts:
                end -= len(parts)
                moved = True
                break
    return end


def _name_before(tokens: list[str], index: int) -> tuple[str, ...]:
    """The name that ends right before ``index``, skipping auxiliaries, or ()."""
    end = _skip_auxiliaries(tokens, index)
    start = end
    while start > 0 and _is_name_word(tokens, start - 1):
        start -= 1
    return tuple(fold_text(accented_lower(t)) for t in tokens[start:end])


def role_claims(text: str) -> list[RoleClaim]:
    """Role changes ``text`` states, one per (name, verb)."""
    return list(_role_claims(text))


@lru_cache(maxsize=16)  # the same sources are read once per answer sentence
def _role_claims(text: str) -> tuple[RoleClaim, ...]:
    claims: list[RoleClaim] = []
    for sentence in _SENTENCE_END.split(split_glued_words(strip_citations(text))):
        tokens = _TOKEN.findall(sentence)
        index = 0
        while index < len(tokens):
            matched = None
            inside_name = tokens[index][:1].isupper() and index > 0 and tokens[index - 1][:1].isalnum()
            for parts, roles in ([] if inside_name else _VERB_PARTS):  # "sân Tân An" is a name
                if _phrase_at(tokens, index, [parts]):
                    matched = (parts, roles)
                    break
            if matched is None:
                index += 1
                continue
            parts, (object_role, subject_role) = matched
            after = index + len(parts)
            marker = _skip_auxiliaries(tokens, index) - 1
            passive = marker >= 0 and accented_lower(tokens[marker]) in PASSIVE_MARKERS
            before = _name_before(tokens, marker if passive else index)
            if passive:
                if before and object_role:
                    claims.append(RoleClaim(before, object_role, sentence.strip()))
            else:
                if before and subject_role:
                    claims.append(RoleClaim(before, subject_role, sentence.strip()))
                following = _name_from(tokens, after)
                if following and object_role:
                    claims.append(RoleClaim(following, object_role, sentence.strip()))
                if " ".join(parts) in REPLACING_VERBS:
                    # "thay HLV Hậu bằng HLV San": San is the newcomer.
                    for position in range(after, min(after + 8, len(tokens))):
                        if tokens[position] in ".,;:!?":
                            break
                        if _phrase_at(tokens, position, INSTEAD_MARKERS):
                            instead = _name_from(tokens, position + 1)
                            if instead:
                                claims.append(RoleClaim(instead, JOIN, sentence.strip()))
                            break
            index = after
    return tuple(claims)


def _same_person(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """"San" and "Huỳnh Ngọc San" are the same person; "Long An" and "San" are not."""
    return bool(a and b) and (set(a) <= set(b) or set(b) <= set(a))


def role_conflicts(sentence: str, source: str) -> list[str]:
    """Names to which ``sentence`` gives a role that ``source`` contradicts.

    A name conflicts when the source gives it only the opposite role, or gives it
    no role while giving that role to someone else.
    """
    stated = role_claims(source)
    conflicts: list[str] = []
    for claim in role_claims(sentence):
        own = {c.role for c in stated if _same_person(c.name, claim.name)}
        if own:
            if claim.role not in own:
                conflicts.append(" ".join(claim.name))
            continue
        others = [c for c in stated if c.role == claim.role]
        if others:
            conflicts.append(" ".join(claim.name))
    return conflicts


def role_evidence(source: str, limit: int = 3) -> list[str]:
    """Sentences of ``source`` that state a role change, in order, without repeats."""
    seen: list[str] = []
    for claim in role_claims(source):
        if claim.sentence not in seen:
            seen.append(claim.sentence)
    return seen[:limit]
