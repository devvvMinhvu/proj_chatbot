"""Passage-first answers: pick the sentences that answer a question, then hold the answer to them.

A factual answer is built in three steps:

1. Each candidate article is split into numbered sentences (its passages). Index
   0 is the title, -1 a line with the publisher and publication time, and the
   text follows from 1, so a passage keeps the same identity
   (``article#index``) whichever search or turn brought the article up.
2. The sentences that match the question (its names, nouns, the noun it counts,
   the kind of answer it wants) are selected, a few per article. A sentence that
   continues the previous one ("Con số này …", "Công trình còn lại …") brings that
   one along. Only these passages are shown to the language model.
3. Every answer sentence must be supported by one passage, or one passage and
   the sentence before it, of the article it cites, not by words gathered from
   several articles. Support means: its numbers with the same counted noun
   ("25% nhân lực" is not "14,74% thị phần"), approximations kept ("gần 3 tỷ"),
   each number in the same clause as the name it is attached to ("ba lần giành
   Vua phá lưới" does not support "giành Quả Bóng Vàng 3 lần"), a joint total
   not given to one of its contributors, plans, offers and predictions not
   turned into facts ("dự kiến", "đề xuất", "dự đoán"), and roles not swapped.
   A sentence that only repeats the question supports nothing.

When the passages do not establish what was asked (a planned figure when the
actual one was asked, a count of something else), the reply says so and quotes
them, without the model. When the model's sentences all fail, the reply quotes
the passages instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import re
from typing import Any

from nlp import lexicon
from processing.fact_check import (
    denies,
    is_verdict,
    names_in,
    number_forms,
    split_glued_words,
    strip_citations,
)
from processing.role_check import ROLE_VERBS, role_claims, role_conflicts
from processing.text_matching import accented_lower, fold_text, is_vietnamese_syllable, matched_keywords

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[A-Za-zÀ-ỹĐđ]+|%|[^\sA-Za-zÀ-ỹĐđ\d]")
#: Clause breaks: punctuation followed by a space ("3,5 tỷ" and "11:07" stay whole) or a linking word.
_CLAUSE_BREAK = re.compile(r"[,;:](?=\s|$)|\s(?:trong đó|còn|trong khi|nhưng|song)\s", re.IGNORECASE)
_CITATION_NUMBERS = re.compile(r"\[\s*Nguồn\s*([\d\s,;Nguồn]+)\]", re.IGNORECASE)
#: Words after a number that are not what it counts ("ngày 28 đến 30/10").
_NOT_UNITS = {
    "đến", "và", "là", "của", "trong", "tại", "ở", "cho", "với", "khi", "sau", "trước", "so",
    "từ", "thì", "nhưng", "hay", "hoặc", "mà", "đã", "sẽ", "để", "theo", "vào", "trên", "dưới",
}
_TIME_UNITS = {"giờ", "phút", "ngày", "tháng", "năm", "h"}
#: Words that carry no information of their own in an answer ("Đúng, theo bài báo, …").
_FILLER = {
    "là", "có", "được", "đã", "đang", "sẽ", "của", "và", "cho", "theo", "bài", "báo", "viết",
    "đúng", "vậy", "phải", "thì", "này", "đó", "các", "những", "một", "rằng", "nguồn", "trong",
    "với", "về", "ở", "tại", "cả", "đều", "như", "nói", "nêu", "cho biết",
}
_TITLE_WORDS = {"hlv", "clb", "ubnd", "ông", "bà", "anh", "chị", "tp", "tphcm"}


# --- Articles as passages ----------------------------------------------------------


@dataclass(frozen=True)
class Passage:
    article: str  # stable article identity: its id, else its URL
    index: int  # position in the text it was read from: -1 publication details, 0 title, 1… text
    text: str

    @property
    def pid(self) -> str:
        """Stable identity: the article and a digest of the sentence itself.

        The position depends on what was read (the whole article or a retrieved
        chunk), so it is used only to find neighbouring sentences, never as identity.
        """
        return f"{self.article}#{sentence_digest(self.text)}"


def sentence_digest(text: str) -> str:
    canonical = re.sub(r"\s+", " ", accented_lower(split_glued_words(text or ""))).strip()
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def article_key(article: dict[str, Any]) -> str:
    return str(article.get("article_id") or article.get("url") or article.get("title") or "")


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_END.split(split_glued_words(text or "")) if part.strip()]


def article_passages(article: dict[str, Any]) -> list[Passage]:
    """Publication line, title and text sentences of an article, each once."""
    key = article_key(article)
    published = article.get("published_at_display") or str(article.get("published_at") or "")[:16]
    meta = f"Bài \"{article.get('title', '')}\" đăng trên báo {article.get('source', '')}"
    meta += f" lúc {published}." if published else "."
    passages = [Passage(key, -1, meta)]
    if article.get("title"):
        passages.append(Passage(key, 0, str(article["title"]).strip()))
    body = article.get("body")
    texts = [body] if body else list(article.get("texts") or [])
    sapo = (article.get("sapo") or "").strip()
    if sapo and not any(sapo[:60] in (t or "") for t in texts):
        texts.insert(0, sapo)
    seen: set[str] = set()
    index = 1
    for text in texts:
        for sentence in _split_sentences(text):
            if sentence in seen:
                continue  # the crawler sometimes stored a paragraph twice
            seen.add(sentence)
            passages.append(Passage(key, index, sentence))
            index += 1
    return passages


# --- What the question asks for ---------------------------------------------------------


@dataclass
class Need:
    keywords: dict[str, float] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    wants_number: bool = False
    wants_percent: bool = False
    wants_person: bool = False
    wants_place: bool = False
    wants_time: bool = False
    wants_publication: bool = False
    unit: str | None = None  # what the question counts ("đại biểu", "lần", "%", "tiền")
    yes_no: bool = False
    #: The outcome asked for: a kind of MODALITY_MARKERS ("expected" for "thực tế"), or
    #: "any" when a number question says "đã …" or "chính xác" (what actually happened).
    asks_outcome: str | None = None
    #: The counted noun came from "số" + noun; it may not be countable ("số điện thoại").
    unit_from_noun: bool = False
    entity_spans: list[str] = field(default_factory=list)  # the names as the user wrote them
    #: Numbers the question itself states ("ba danh hiệu", "21 bàn"), as sets of spellings:
    #: the answer has to come from the clause that states them.
    number_anchors: list[set[str]] = field(default_factory=list)
    exact: bool = True


def _has_phrase(text: str, phrases) -> bool:
    lowered = f" {accented_lower(text)} "
    return any(re.search(rf"(?<![\wÀ-ỹ]){re.escape(p)}(?![\wÀ-ỹ])", lowered) for p in phrases)


def question_need(question: str, parsed: Any | None = None, carried_words: list[str] | None = None) -> Need:
    """What a question asks for, from its parsed slots and question words.

    ``carried_words`` are the previous turn's subject words, for a follow-up
    ("Dự án này ở tỉnh nào?" after a question about the Sơn La school project).
    """
    if parsed is None:
        from nlp.turn_parser import parse_turn

        parsed = parse_turn(question)
    need = Need(exact=parsed.exact_accents)
    frames = set(parsed.frame_words)
    for word in parsed.topic_words:
        if lexicon.is_one_of(word, lexicon.REQUEST_WORDS, parsed.exact_accents):
            continue  # "trích câu nêu …", "trong bài": what to do, not what about
        if lexicon.is_one_of(word, lexicon.QUESTION_NOUNS, parsed.exact_accents) or word in frames:
            weight = 0.5
        elif word in parsed.entity_words:
            weight = 3.0
        elif word in parsed.topic_nouns:
            weight = 2.0
        else:
            weight = 1.0
        need.keywords[word] = max(weight, need.keywords.get(word, 0))
    for word in carried_words or []:
        if not lexicon.is_one_of(word, lexicon.REQUEST_WORDS + lexicon.QUESTION_NOUNS, parsed.exact_accents):
            need.keywords.setdefault(word, 1.0)
    need.entities = list(parsed.entity_words)
    list_numbers = {str(n) for n in [*parsed.item_refs, *parsed.excluded_refs]}
    for number in re.findall(r"\d+(?:[.,]\d+)*", question):
        if number not in list_numbers:  # "bài 8" picks an article; "21 bàn" is a fact
            need.keywords.setdefault(number, 4.0)  # a number the user quotes is the most specific anchor
    need.number_anchors = [
        q["forms"] for q in _quantities(question)
        if not q["temporal"] and not (q["forms"] & list_numbers)
    ]
    words, tags = parsed.words, parsed.tags
    for index, word in enumerate(words):
        if lexicon.is_one_of(word, lexicon.QUANTITY_WORDS, parsed.exact_accents):
            need.wants_number = True
            following = [
                w for w, t in zip(words[index + 1:index + 3], tags[index + 1:index + 3]) if t.startswith("N")
            ]
            if following and need.unit is None:
                need.unit = accented_lower(following[0])
    # "số" + a noun asks for that count: "số người", "số khách", "số vận động viên".
    # ("bài số 3" has a number after it; "con số" is one word.)
    for index, word in enumerate(words[:-1]):
        if accented_lower(word) == "số" and tags[index + 1].startswith("N"):
            need.wants_number = True
            if need.unit is None:
                need.unit = accented_lower(words[index + 1])
                need.unit_from_noun = True
            break
    if _has_phrase(question, lexicon.PERCENT_WORDS) or "%" in question:
        need.wants_number = need.wants_percent = True
        need.unit = "%"
        need.unit_from_noun = False
    if _has_phrase(question, lexicon.QUANTITY_NOUNS):
        need.wants_number = True
    if _has_phrase(question, lexicon.MONEY_WORDS) and (need.unit in (None, "tiền") or need.unit_from_noun):
        need.wants_number = True
        need.unit = "tiền"
        need.unit_from_noun = False
    need.yes_no = bool(getattr(parsed, "yes_no", False))
    if need.unit and need.unit != "%":
        need.keywords[need.unit] = max(1.5, need.keywords.get(need.unit, 0))
    need.wants_person = _has_phrase(question, lexicon.PERSON_QUESTION_WORDS)
    need.wants_place = _has_phrase(question, lexicon.PLACE_QUESTION_WORDS)
    need.wants_time = _has_phrase(question, lexicon.TIME_QUESTION_WORDS)
    need.wants_publication = _has_phrase(question, ("đăng", "báo nào", "nguồn nào", "trang nào"))
    for kind, markers in lexicon.OUTCOME_MARKERS.items():
        if _has_phrase(question, markers):
            need.asks_outcome = kind
            break
    # "Số khách đã có mặt là bao nhiêu?", "chính xác bao nhiêu người": what happened,
    # whatever the article planned, offered or predicted.
    if need.asks_outcome is None and need.wants_number and (
        _has_phrase(question, lexicon.ASPECT_MARKERS["done"]) or _has_phrase(question, lexicon.PRECISION_WORDS)
    ):
        need.asks_outcome = "any"
    need.entity_spans = _entity_spans(question, parsed.entity_words)
    return need


def _entity_spans(question: str, entities: list[str]) -> list[str]:
    """Adjacent name words joined as the user wrote them: "Quả", "Bóng Vàng" -> "Quả Bóng Vàng"."""
    tokens = _TOKEN.findall(question)
    wanted = {accented_lower(part) for entity in entities for part in entity.split()}
    folded = {fold_text(w) for w in wanted}
    spans: list[str] = []
    current: list[str] = []
    for token in tokens + [""]:
        lowered = accented_lower(token)
        if token and (lowered in wanted or (fold_text(lowered) == lowered and lowered in folded)):
            current.append(token)
            continue
        if current:
            spans.append(" ".join(current))
            current = []
    return spans


# --- Choosing passages -------------------------------------------------------------------


def _quantities(text: str) -> list[dict[str, Any]]:
    """Numbers of a text with what they count, their clause and whether they are approximate.

    Number words count too ("ba lần" is 3 lần), except words that are also
    ordinary words ("không", "một" before a name).
    """
    out: list[dict[str, Any]] = []
    clauses = _clauses(text)
    for clause_index, clause in enumerate(clauses):
        tokens = _TOKEN.findall(clause)
        index = 0
        while index < len(tokens):
            token = tokens[index]
            forms: set[str] = set()
            end = index + 1
            if token[0].isdigit():
                forms = number_forms(token)
            else:
                run = []
                while end - 1 < len(tokens) and _number_word(tokens[end - 1]):
                    run.append(tokens[end - 1])
                    end += 1
                end -= 1
                # A spelled-out number counts only before what it counts ("ba lần"),
                # not as an ordinal at the end of a phrase ("câu thứ hai.").
                # "trăm" in "phần trăm" multiplies nothing: a run needs a digit word.
                has_digit_word = any(
                    accented_lower(w) not in {"mươi", "trăm", "nghìn", "ngàn"} for w in run
                )
                if run and has_digit_word and end < len(tokens) and tokens[end][:1].isalpha():
                    value = lexicon.parse_number([accented_lower(w) for w in run])
                    if value:
                        forms = {str(value)}
                if not forms:
                    end = index + 1
            if not forms:
                index += 1
                continue
            after = tokens[end] if end < len(tokens) else ""
            before = tokens[index - 1] if index else ""
            temporal = after in {"/", ":", "-", "h"} or before in {"/", ":", "-"}
            # A year ("năm 2026", "Tuần lễ Số 2026") dates something; it does not count it.
            year = token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2100 and (
                accented_lower(before) in {"năm", "mùa"} or before[:1].isupper() or not after[:1].isalpha()
            )
            temporal = temporal or year
            unit = ""
            if after == "%":
                unit = "%"
            elif after and after[:1].isalpha() and accented_lower(after) not in _NOT_UNITS:
                unit = accented_lower(after)
            before_two = " ".join(accented_lower(t) for t in tokens[max(0, index - 2):index])
            approximate = lexicon.is_one_of(before, lexicon.APPROXIMATE_MARKERS) or any(
                before_two.endswith(marker) for marker in lexicon.APPROXIMATE_MARKERS if " " in marker
            )
            direction = next(
                (kind for kind, words in lexicon.APPROXIMATE_DIRECTIONS.items()
                 if lexicon.is_one_of(before, words) or any(before_two.endswith(w) for w in words if " " in w)),
                None,
            )
            out.append({
                "forms": forms,
                "unit": unit,
                "before": accented_lower(before),
                "after_unit": accented_lower(tokens[end + 1]) if unit == "%" and end + 1 < len(tokens) else "",
                "approximate": approximate,
                "direction": direction,
                "temporal": temporal,
                "clause": clause_index,
                "position": index,
            })
            index = max(end, index + 1)
    return out


def _number_word(token: str) -> bool:
    lowered = accented_lower(token)
    if lowered in {"không", "linh", "lẻ", "một", "năm", "tư", "nhất", "nhì"}:
        return False  # also "not", "a", "year", "fourth" …
    return lexicon.number_word_value(lowered, True) is not None or lowered in {"mươi", "trăm", "nghìn", "ngàn"}


def _subphrases(word: str) -> list[str]:
    """A keyword and its two-syllable parts: the tagger joins "tiền tài trợ" into one word."""
    parts = word.split()
    return [word] + [" ".join(parts[i:i + 2]) for i in range(len(parts) - 1)] if len(parts) > 2 else [word]


def _clauses(text: str) -> list[str]:
    return [part for part in _CLAUSE_BREAK.split(text or "") if part and part.strip()]


def _score(passage: Passage, need: Need, prior: set[str]) -> tuple[float, float]:
    """(score, keyword part of it) of a passage for a question."""
    text = passage.text
    keyword_score = sum(
        weight for word, weight in need.keywords.items() if matched_keywords([word], text, exact=need.exact)
    )
    if passage.index == -1:
        keyword_score = 3.0 if need.wants_publication else 0.0
    score = keyword_score
    quantities = _quantities(text) if need.wants_number else []
    if quantities:
        score += 1.0
        if need.unit and any(_same_unit(q, need.unit) for q in quantities):
            score += 3.0
    if need.wants_percent and any(
        q["unit"] == "%" and q["after_unit"] and matched_keywords(list(need.keywords), q["after_unit"], exact=need.exact)
        for q in quantities
    ):
        score += 2.0
    if need.wants_person and _name_groups(text):
        score += 1.0
    if need.wants_place and re.search(r"(?<![\wÀ-ỹ])(?:ở|tại)\s+[A-ZÀ-ỸĐ]", text):
        score += 1.5
    if need.wants_time and re.search(r"\d{1,2}/\d{1,2}|\bngày \d|\bnăm \d{4}|\d{1,2}h\d*|\d+ giờ|tháng \d", text):
        score += 1.5
    if need.number_anchors and any(
        anchor & q["forms"] for anchor in need.number_anchors for q in _quantities(text)
    ):
        score += 4.0  # "ba lần giành Vua phá lưới" for "ba danh hiệu Vua phá lưới"
    if passage.pid in prior:
        anchored = any(weight >= 3.0 for weight in need.keywords.values())
        score += 1.0 if anchored else 3.0
    return score, keyword_score


def _same_unit(quantity: dict[str, Any], unit: str) -> bool:
    if unit == "%":
        return quantity["unit"] == "%"
    if unit == "tiền":
        return quantity["unit"] in lexicon.MONEY_UNITS
    first = accented_lower(unit).split()[0]
    return quantity["unit"] == first or quantity["before"] == first.split()[-1]


def _breakdown_after(passage: Passage, by_index: dict[int, Passage]) -> list[Passage]:
    """Sentences after ``passage`` that break it down ("Trong đó, 10 … Công trình còn lại …")."""
    following: list[Passage] = []
    index = passage.index + 1
    while index in by_index and len(following) < 2:
        candidate = by_index[index]
        starts = accented_lower(candidate.text)
        breaks_down = any(starts.startswith(w) for w in lexicon.BREAKDOWN_STARTS)
        if not (breaks_down or (following and _continues(candidate))):
            break
        following.append(candidate)
        index += 1
    return following


def _continues(passage: Passage) -> bool:
    lowered = accented_lower(passage.text)
    return any(lowered.startswith(start + " ") for start in lexicon.CONTINUATION_STARTS) or any(
        word in lowered for word in lexicon.CONTINUATION_WORDS
    )


@dataclass
class Evidence:
    """The passages that may answer a question, per article, in the order given."""

    question: str
    need: Need
    articles: list[dict[str, Any]]  # [Nguồn i] is articles[i - 1]
    passages: list[list[Passage]]  # shown to the model, in article order
    sentences: list[list[Passage]]  # every passage of each article
    scores: dict[str, float] = field(default_factory=dict)
    positions: list[int] | None = None  # list numbers the user named, for comparisons
    weak: bool = False  # no passage matched the question's words
    #: Passages the reply actually used (filled as the answer is checked or quoted).
    cited: list[str] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return [p.pid for group in self.passages for p in group]

    def context(self) -> str:
        parts = []
        for label, (article, group) in enumerate(zip(self.articles, self.passages), start=1):
            published = article.get("published_at_display") or str(article.get("published_at") or "")[:16]
            heading = f"[Nguồn {label}] Tiêu đề: {article.get('title', '')} | Báo: {article.get('source', '')}"
            if published:
                heading += f" | Ngày đăng: {published}"
            if self.positions and label <= len(self.positions):
                heading += f" | Bài {self.positions[label - 1]} trong danh sách"
            lines = [f"- {p.text}" for p in group if p.index != -1]
            parts.append(heading + "\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def units(self, label: int) -> list[tuple[Passage, ...]]:
        """Passages (alone, or with the sentence before or after) that may support a sentence citing ``label``."""
        group = self.passages[label - 1]
        by_index = {p.index: p for p in self.sentences[label - 1]}
        units: list[tuple[Passage, ...]] = (
            [(by_index[-1],)] if -1 in by_index and self.need.wants_publication else []
        )
        for passage in group:
            units.append((passage,))
            previous = by_index.get(passage.index - 1)
            if previous is not None and passage.index > 1:
                units.append((previous, passage))
            following = by_index.get(passage.index + 1)
            if following is not None and passage.index >= 1:
                units.append((passage, following))
        return sorted(dict.fromkeys(units), key=len)

    def full_text(self, label: int) -> str:
        return " ".join(p.text for p in self.sentences[label - 1] if p.index != -1)

    def best(self, limit: int = 2, per_article: int = 1) -> list[tuple[int, Passage]]:
        """The highest-scoring passages, at most ``per_article`` from each article."""
        ranked: list[tuple[float, int, int, Passage]] = []
        for label, group in enumerate(self.passages, start=1):
            has_body = any(p.index >= 1 for p in group)
            for passage in group:
                if passage.index == -1 and not self.need.wants_publication:
                    continue
                if passage.index == 0 and has_body:
                    continue  # quote the article's sentences, not its headline
                score = self.scores.get(passage.pid, 0.0)
                if self.need.unit and any(_same_unit(q, self.need.unit) for q in _quantities(passage.text)):
                    score += 100.0  # a sentence that states the kind of number asked for
                ranked.append((score, -label, -passage.index, passage))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        chosen: list[tuple[int, Passage]] = []
        taken: dict[int, int] = {}
        for _, negative_label, _, passage in ranked:
            label = -negative_label
            if taken.get(label, 0) >= per_article:
                continue
            taken[label] = taken.get(label, 0) + 1
            chosen.append((label, passage))
            if len(chosen) >= limit:
                break
        return sorted(chosen, key=lambda item: (item[0], item[1].index))

    def quote_unit(self, label: int, unit: tuple[Passage, ...]) -> str:
        """The sentences that supported an answer, exactly, with their citation."""
        lines = []
        for passage in unit:
            if passage.index == -1 and not self.need.wants_publication:
                continue
            if passage.pid not in self.cited:
                self.cited.append(passage.pid)
            lines.append(f"> {passage.text} [Nguồn {label}]")
        return "\n\n".join(lines)

    def quote(self, chosen: list[tuple[int, Passage]]) -> str:
        """Quoted passages with their citations; a total keeps the sentences that break it down."""
        lines = []
        quoted: set[str] = set()
        for label, passage in chosen:
            by_index = {p.index: p for p in self.sentences[label - 1]}
            # "Con số này …" is quoted with the sentence it refers to.
            previous = by_index.get(passage.index - 1)
            lead = [previous] if previous is not None and passage.index > 1 and _continues(passage) else []
            for part in [*lead, passage, *_breakdown_after(passage, by_index)]:
                if part.pid in quoted:
                    continue
                quoted.add(part.pid)
                if part.pid not in self.cited:
                    self.cited.append(part.pid)
                prefix = ""
                if part is passage and self.positions and label <= len(self.positions):
                    prefix = f"**Bài {self.positions[label - 1]}:** "
                lines.append(f"> {prefix}{part.text} [Nguồn {label}]")
        return "\n\n".join(lines)

    def per_name(self, names: list[str] | None = None) -> list[tuple[int, Passage]]:
        """For each name the question asks about, the best sentence that has it.

        Sentences that give a number to that name alone are preferred over a joint
        total ("Công trình còn lại do BW Industrial tài trợ" over "Opella và BW
        Industrial bàn giao 11 công trình").
        """
        chosen: list[tuple[int, Passage]] = []
        for span in names if names is not None else self.need.entity_spans:
            key = " ".join(fold_text(accented_lower(w)) for w in span.split())
            best: tuple[float, int, Passage] | None = None
            for label, group in enumerate(self.passages, start=1):
                for passage in group:
                    if passage.index < 0 or not matched_keywords([span], passage.text, exact=self.need.exact):
                        continue
                    score = self.scores.get(passage.pid, 0.0)
                    if any(any(key in name or name in key for name in chain) for chain in _joint_names(passage.text)):
                        score *= 0.5
                    if self.need.unit and any(_same_unit(q, self.need.unit) for q in _quantities(passage.text)):
                        score += 2.0
                    if best is None or score > best[0]:
                        best = (score, label, passage)
            if best is not None and (best[1], best[2]) not in chosen:
                chosen.append((best[1], best[2]))
        return chosen

    def extractive_reply(self) -> str:
        """The passages themselves, when no generated sentence holds up.

        A question about several names gets a sentence for each of them.
        """
        if self.weak or not any(self.passages):
            return "Bài viết không có câu nào trả lời trực tiếp câu hỏi này."
        per_article = 1 if len(self.articles) > 1 else 2
        chosen = self.best(limit=max(2, len(self.articles)) if self.positions else 2, per_article=per_article)
        if len(self.need.entity_spans) > 1:
            named = self.per_name()
            if named:
                chosen = sorted(dict.fromkeys(named), key=lambda item: (item[0], item[1].index))
        return "Theo bài viết:\n\n" + self.quote(chosen)

    def missing_parts(self, shown: str) -> list[tuple[int, Passage]]:
        """Sentences for the names of a compound question that ``shown`` does not mention."""
        if len(self.need.entity_spans) < 2:
            return []
        missing = [span for span in self.need.entity_spans if not matched_keywords([span], shown, exact=self.need.exact)]
        return self.per_name(missing) if missing else []

    def partial_reply(self) -> str | None:
        """The reply when the passages do not establish the number that was asked for.

        "Có bao nhiêu đại biểu thực tế tham dự?" with only "dự kiến quy tụ 500 đại
        biểu": the planned figure is quoted and the actual one said to be missing.
        "Kane giành Quả Bóng Vàng mấy lần?" with only "ba lần giành Vua phá lưới":
        no passage counts what was asked, and the nearest passages are quoted.
        """
        need = self.need
        if self.weak or not (need.unit or need.asks_outcome):
            return None
        if need.unit and not need.asks_outcome and any(_same_unit(q, need.unit) for q in _quantities(self.question)):
            return None  # "Số tiền gần 3 tỷ đồng là cho cả dự án hay …?" quotes the figure itself
        if not (need.wants_number or need.asks_outcome):
            return None
        kinds = list(lexicon.MODALITY_MARKERS) if need.asks_outcome == "any" else [need.asks_outcome]
        planned: tuple[int, Passage, str, bool] | None = None
        for label, group in enumerate(self.passages, start=1):
            by_index = {p.index: p for p in self.sentences[label - 1]}
            # The article's sentences before its headline, so the quote is a sentence.
            for passage in sorted(group, key=lambda p: (p.index < 1, p.index)):
                if passage.index == -1:
                    continue
                for quantity in _quantities(passage.text):
                    # An outcome question may count with another noun ("bao nhiêu người
                    # đã chấp nhận" against "đề xuất với 60.000 nhân viên").
                    if need.unit and not need.asks_outcome and not _same_unit(quantity, need.unit):
                        continue
                    if quantity["temporal"]:
                        continue
                    clause = _clauses(passage.text)[quantity["clause"]]
                    if not self._entities_attached(clause, passage, by_index.get(passage.index - 1)):
                        continue
                    if need.asks_outcome:
                        # Only a number stated as an outcome answers "đã chấp nhận", "thực
                        # tế"; the company's 223.000 employees are not those who accepted.
                        if any(_has_phrase(clause, lexicon.OUTCOME_MARKERS[k]) for k in kinds) or (
                            need.asks_outcome == "any" and _has_phrase(clause, lexicon.ASPECT_MARKERS["done"])
                        ):
                            return None
                        kind = next((k for k in kinds if _has_phrase(clause, lexicon.MODALITY_MARKERS[k])), None)
                        counts_it = bool(need.unit) and _same_unit(quantity, need.unit)
                        # A plan that counts the asked noun beats one about something else
                        # ("đề xuất với 60.000 nhân viên" over "25% nhân lực có thể nghỉ việc").
                        if kind is not None and (planned is None or (counts_it and not planned[3])):
                            planned = (label, passage, kind, counts_it)
                        continue
                    return None  # established
        if planned is not None:
            label, passage, kind, _ = planned
            return lexicon.MODALITY_REPLIES[kind] + "\n\n" + self.quote([(label, passage)])
        if not need.unit:
            return None
        if need.asks_outcome:
            return None  # no number of either kind: the model may still answer
        if need.unit_from_noun and not any(
            _same_unit(q, need.unit) for sentences in self.sentences for p in sentences for q in _quantities(p.text)
        ):
            return None  # nothing here counts it ("số điện thoại"): not a count question
        chosen = self.best(limit=1, per_article=1)
        counted = self._best_counting(exclude={p.pid for _, p in chosen})
        if counted is not None:
            chosen = sorted(chosen + [counted], key=lambda item: (item[0], item[1].index))
        if not chosen:
            return "Bài viết không có con số trả lời câu hỏi này."
        about = f" về {', '.join(need.entity_spans)}" if need.entity_spans else ""
        return (
            f"Bài viết không có con số trả lời trực tiếp câu hỏi này{about}. Các câu liên quan:\n\n"
            + self.quote(chosen)
        )

    def _best_counting(self, exclude: set[str]) -> tuple[int, Passage] | None:
        """The highest-scoring sentence that counts the asked unit and names one of the question's names."""
        best: tuple[float, int, Passage] | None = None
        for label, sentences in enumerate(self.sentences, start=1):
            for passage in sentences:
                if passage.index < 1 or passage.pid in exclude:
                    continue
                if not any(_same_unit(q, self.need.unit) for q in _quantities(passage.text)):
                    continue
                if self.need.entities and not matched_keywords(self.need.entities, passage.text, exact=self.need.exact):
                    continue
                score = self.scores.get(passage.pid, 0.0)
                if best is None or score > best[0]:
                    best = (score, label, passage)
        return (best[1], best[2]) if best else None

    def _entities_attached(self, clause: str, passage: Passage, previous: Passage | None) -> bool:
        """Are the question's names in the number's clause (or, for "của hãng", the sentence before)?"""
        for entity in self.need.entities:
            if matched_keywords([entity], clause, exact=self.need.exact):
                continue
            if matched_keywords([entity], passage.text, exact=self.need.exact):
                # In the sentence but another clause: the number's clause must still be
                # about the question ("tổng tài trợ gần 3 tỷ" for "số tiền tài trợ"), not
                # about something else ("bốn lần cho mượn" for "giành Quả Bóng Vàng mấy lần").
                own_words = [
                    part
                    for w, weight in self.need.keywords.items()
                    if weight >= 1.0 and w not in self.need.entities and w != self.need.unit
                    and not w[:1].isdigit()
                    for part in _subphrases(w)
                ]
                if matched_keywords(own_words, clause, exact=self.need.exact):
                    continue
                return False
            refers_back = _continues(passage) or _has_phrase(clause, lexicon.COREFERENCE_WORDS)
            if not (refers_back and previous and matched_keywords([entity], previous.text, exact=self.need.exact)):
                return False
        return True


def select_evidence(
    question: str,
    articles: list[dict[str, Any]],
    *,
    parsed: Any | None = None,
    prior: list[str] | None = None,
    positions: list[int] | None = None,
    per_article: int = 4,
    limit: int = 10,
    keep_every_article: bool = False,
    carried_words: list[str] | None = None,
) -> Evidence:
    """The passages of ``articles`` that may answer ``question``.

    Articles without a matching passage are left out, unless
    ``keep_every_article`` (a comparison: each requested article is represented,
    by its opening when nothing else matches). ``prior`` are passage ids that
    answered the previous turn; they are preferred for a follow-up.
    """
    need = question_need(question, parsed, carried_words)
    prior_ids = set(prior or [])
    must_keep: set[str] = set()
    asks_role = bool(matched_keywords(list(ROLE_VERBS), question))
    # A question with no subject of its own ("So sánh hai bài", "Ý chính là gì?") is
    # answered from each article's opening.
    generic = not any(weight >= 2.0 for weight in need.keywords.values()) and not need.unit and not need.asks_outcome
    kept_articles: list[dict[str, Any]] = []
    kept_passages: list[list[Passage]] = []
    kept_sentences: list[list[Passage]] = []
    kept_positions: list[int] = []
    scores: dict[str, float] = {}
    any_match = False
    ranked_all: list[tuple[float, int, Passage]] = []
    per_article_ranked: list[list[tuple[float, Passage]]] = []
    passages_of: list[list[Passage]] = []
    for article in articles:
        passages = article_passages(article)
        by_index = {p.index: p for p in passages}
        ranked = []
        for passage in passages:
            score, keyword_score = _score(passage, need, prior_ids)
            previous = by_index.get(passage.index - 1)
            if previous is not None and passage.index > 1 and _continues(passage) and keyword_score >= 1.0:
                # "Con số này … của hãng" after "Microsoft cho biết …".
                score += 0.75 * sum(
                    need.keywords.get(e, 0.0) for e in need.entities
                    if matched_keywords([e], previous.text, exact=need.exact)
                    and not matched_keywords([e], passage.text, exact=need.exact)
                )
            scores[passage.pid] = score
            states_unit = bool(need.unit) and passage.index >= 1 and any(
                _same_unit(q, need.unit) for q in _quantities(passage.text)
            )
            # "Số người đã đến sau khi kết thúc?": the planned figure is what the article has.
            states_unit = states_unit or (
                bool(need.asks_outcome) and passage.index >= 1
                and any(not q["temporal"] for q in _quantities(passage.text))
                and any(
                    _has_phrase(passage.text, lexicon.MODALITY_MARKERS[kind])
                    for kind in (lexicon.MODALITY_MARKERS if need.asks_outcome == "any" else [need.asks_outcome])
                )
            )
            if states_unit:
                must_keep.add(passage.pid)  # it states the kind of figure asked for
            if asks_role and passage.index >= 1 and role_claims(passage.text):
                must_keep.add(passage.pid)  # "HLV nào mới được bổ nhiệm?": who took or left a post
            if keyword_score >= 1.0 or states_unit or (passage.pid in prior_ids and score > 0):
                ranked.append((score, passage))
        ranked.sort(key=lambda item: (-item[0], item[1].index))
        per_article_ranked.append(ranked)
        passages_of.append(passages)
        any_match = any_match or bool(ranked)
    for position, (article, ranked, passages) in enumerate(zip(articles, per_article_ranked, passages_of)):
        chosen: list[Passage] = []
        if ranked and not generic:
            top = ranked[0][0]
            chosen = [p for score, p in ranked[:per_article] if score >= top * 0.35]
            chosen += [p for _, p in ranked if p.pid in must_keep and p not in chosen][:2]
        elif keep_every_article or generic:
            chosen = [p for p in passages if p.index in (0, 1, 2)]
            chosen += [p for p in passages if p.pid in must_keep and p not in chosen][:2]
        if not chosen:
            continue
        by_index = {p.index: p for p in passages}
        with_context: list[Passage] = []
        for passage in chosen:
            previous = by_index.get(passage.index - 1)
            if previous is not None and passage.index > 1 and _continues(passage) and previous not in chosen:
                with_context.append(previous)
            with_context.append(passage)
            with_context.extend(_breakdown_after(passage, by_index))
        group = sorted(set(with_context), key=lambda p: p.index)
        kept_articles.append(article)
        kept_passages.append(group)
        kept_sentences.append(passages)
        if positions:
            kept_positions.append(positions[position] if position < len(positions) else position + 1)
        for passage in group:
            ranked_all.append((scores[passage.pid], len(kept_articles), passage))
    # Keep at most ``limit`` passages overall, best first, but every kept article keeps one.
    if sum(len(g) for g in kept_passages) > limit and not keep_every_article:
        keep = {p.pid for _, _, p in sorted(ranked_all, key=lambda item: -item[0])[:limit]}
        for i, group in enumerate(kept_passages):
            trimmed = [p for p in group if p.pid in keep] or group[:1]
            kept_passages[i] = trimmed
    return Evidence(
        question=question,
        need=need,
        articles=kept_articles,
        passages=kept_passages,
        sentences=kept_sentences,
        scores=scores,
        positions=kept_positions or None,
        weak=not any_match and not generic and not keep_every_article,
    )


# --- Checking an answer sentence against the passages -------------------------------------


def _name_groups(text: str) -> list[tuple[int, int, str]]:
    """(start, end, folded name) of each run of capitalised words that is not a title."""
    tokens = _TOKEN.findall(text)
    groups: list[tuple[int, int, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        starts_sentence = index == 0 or not tokens[index - 1][:1].isalnum()
        capital = token[:1].isupper() and token[:1].isalpha() and accented_lower(token) not in _TITLE_WORDS
        if capital and starts_sentence:
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            capital = not is_vietnamese_syllable(accented_lower(token)) or (
                following[:1].isupper() and following[:1].isalpha()
            )
        if not capital:
            index += 1
            continue
        end = index
        while end + 1 < len(tokens) and tokens[end + 1][:1].isupper() and tokens[end + 1][:1].isalpha():
            end += 1
        groups.append((index, end, " ".join(fold_text(accented_lower(t)) for t in tokens[index:end + 1])))
        index = end + 1
    return groups


def _joint_names(clause: str) -> list[set[str]]:
    """Names joined into one subject ("Quỹ Hy vọng cùng Opella Việt Nam và BW Industrial")."""
    tokens = _TOKEN.findall(clause)
    groups = _name_groups(clause)
    chains: list[set[str]] = []
    current: set[str] = set()
    last_end = None
    for start, end, name in groups:
        between = [accented_lower(t) for t in tokens[last_end + 1:start]] if last_end is not None else []
        joined = last_end is not None and len(between) <= 2 and any(w in lexicon.JOINING_WORDS for w in between) and all(
            not w[:1].isdigit() for w in between
        )
        if joined:
            current.add(name)
        else:
            if len(current) > 1:
                chains.append(current)
            current = {name}
        last_end = end
    if len(current) > 1:
        chains.append(current)
    return chains


def _folded_words(text: str) -> set[str]:
    return {fold_text(w) for w in re.findall(r"[0-9a-zA-ZÀ-ỹĐđ]+", accented_lower(split_glued_words(text)))}


def is_echo(sentence: str, question: str) -> bool:
    """Does an affirmative sentence only repeat the question ("Có, BW tài trợ cả 11 công trình")?"""
    if denies(sentence):
        return False
    asked = {accented_lower(w) for w in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", question)}
    said = [accented_lower(w) for w in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", strip_citations(sentence))]
    new = [w for w in said if w not in asked and w not in _FILLER]
    return bool(said) and not new


@dataclass
class Support:
    label: int
    unit: tuple[Passage, ...]


def _quantity_supported(answer_q: dict[str, Any], unit_qs: list[dict[str, Any]]) -> bool:
    for q in unit_qs:
        if not (answer_q["forms"] & q["forms"]):
            continue
        if q["approximate"] and not answer_q["approximate"]:
            continue  # "gần 3 tỷ" is not "3 tỷ"
        if q["approximate"] and q["direction"] != answer_q["direction"]:
            continue  # "hơn 25%" (more than) is not "khoảng 25%" (about)
        if answer_q["unit"] == "%":
            if q["unit"] != "%":
                continue
            if answer_q["after_unit"] and q["after_unit"] and answer_q["after_unit"] != q["after_unit"]:
                continue  # "25% nhân lực" is not "25% thị phần"
            return True
        if not answer_q["unit"] or answer_q["temporal"]:
            return True
        if answer_q["unit"] in _TIME_UNITS and q["temporal"]:
            return True
        if answer_q["unit"] in (q["unit"], q["before"]):
            return True
    return False


_MIN_COVERAGE = 0.6
_PART_BREAK = re.compile(r"[,;:](?=\s|$)|\s(?:và|còn|trong đó|trong khi|nhưng|song|rồi)\s", re.IGNORECASE)


_ROLE_WORDS = {word for verb in ROLE_VERBS for word in verb.split()}


def _content_words(text: str) -> list[str]:
    """Words that carry the claim; role-change verbs are left to the role check."""
    words = [accented_lower(w) for w in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", split_glued_words(strip_citations(text)))]
    return [
        w for w in words
        if w not in _FILLER and w not in _ROLE_WORDS and not lexicon.is_one_of(w, lexicon.QUESTION_WORDS)
    ]


def _word_parts(text: str) -> list[list[str]]:
    """The content words of each part of a sentence (split at clause breaks and "và")."""
    parts = [_content_words(part) for part in _PART_BREAK.split(text) if part and part.strip()]
    return [part for part in parts if len(part) >= 2]


def _coverage(words: list[str], scope: str) -> float:
    available = {accented_lower(w) for w in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", split_glued_words(scope))}
    folded = {fold_text(w) for w in available}
    hits = sum(1 for w in words if w in available or (fold_text(w) == w and w in folded))
    return hits / len(words) if words else 1.0


@lru_cache(maxsize=2048)
def _verbs(text: str) -> tuple[str, ...]:
    """Verbs of a short text (lowercased), from the part-of-speech tagger."""
    from nlp.turn_parser import tag_words

    words, tags = tag_words(text)
    return tuple(accented_lower(w) for w, t in zip(words, tags) if t.startswith("V"))


def _aspects(text: str) -> set[str]:
    return {kind for kind, words in lexicon.ASPECT_MARKERS.items() if _has_phrase(text, words)}


def _name_order(text: str, names: list[str]) -> list[str]:
    """``names`` (folded) in the order they first appear in ``text``."""
    tokens = [fold_text(accented_lower(t)) for t in _TOKEN.findall(split_glued_words(text))]
    found = []
    for name in names:
        parts = name.split()
        for i in range(len(tokens) - len(parts) + 1):
            if tokens[i:i + len(parts)] == parts:
                found.append((i, name))
                break
    return [name for _, name in sorted(found)]


def _part_stated(part: str, scope: str) -> bool:
    """Does one sentence (or continuing pair) state this part of an answer?"""
    words = _content_words(part)
    if len(words) >= 2 and _coverage(words, scope) < _MIN_COVERAGE:
        return False
    # Every verb, except the role-change verbs the role check reads.
    scope_words = {accented_lower(w) for w in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", split_glued_words(scope))}
    for verb in _verbs(part):
        if lexicon.is_one_of(verb, lexicon.LIGHT_VERBS) or all(w in _ROLE_WORDS for w in verb.split()):
            continue
        if not all(w in scope_words for w in verb.split()):
            return False  # "hoàn tất" where the article says "xem xét"
    # Completion status: "đã" against "đang", "sẽ", "chưa".
    said, stated = _aspects(part), _aspects(scope)
    if said and stated and not (said & stated):
        return False
    # Names around a relation keep their order. The names are those the article
    # capitalises, so "Tươi" counts even at the start of the answer.
    scope_names = [g[2] for g in _name_groups(scope)]
    in_part = _name_order(part, scope_names)
    if len(in_part) >= 2:
        in_scope = _name_order(scope, in_part)
        for first, second in zip(in_part, in_part[1:]):
            if first in in_scope and second in in_scope and in_scope.index(first) > in_scope.index(second):
                between = _between(part, first, second)
                if any(w not in lexicon.JOINING_WORDS and w not in _FILLER for w in between):
                    return False
    return True


def _between(text: str, first: str, second: str) -> list[str]:
    tokens = [fold_text(accented_lower(t)) for t in _TOKEN.findall(split_glued_words(text))]
    raw = [accented_lower(t) for t in _TOKEN.findall(split_glued_words(text))]

    def position(name: str) -> tuple[int, int]:
        parts = name.split()
        for i in range(len(tokens) - len(parts) + 1):
            if tokens[i:i + len(parts)] == parts:
                return i, i + len(parts)
        return -1, -1

    _, end = position(first)
    start, _ = position(second)
    return raw[end:start] if 0 <= end <= start else []


def _continues_text(text: str) -> bool:
    return _continues(Passage("", 1, text))


def _names_together(names: list[str], texts: list[str], joined: bool) -> bool:
    """Do the names (folded) appear in one clause, or anywhere in a continuing pair?"""
    if len(names) < 2:
        return True
    scopes = [" ".join(texts)] if joined else [c for text in texts for c in _clauses(text)]
    for scope in scopes:
        words = _folded_words(scope)
        if all(all(part in words for part in name.split()) for name in names):
            return True
    return False


def _unit_supports(
    sentence: str, question: str, texts: list[str], article_text: str, *,
    claim_names: list[str] | None = None,
    anchors: tuple[list[set[str]], list[str]] | None = None,
) -> bool:
    """Is ``sentence`` supported by one passage (or a passage and the one next to it)?

    Besides its names and numbers, each part of the sentence must be stated by one
    sentence (or a continuing pair): most of its words, every verb (role-change verbs
    are left to the role check), the same completion status ("đã hoàn tất" is not
    "đang xem xét") and the same order of the names around the relation ("Tươi đóng
    vai Trang Moon" is not "Trang Moon đóng vai Tươi"). When that cannot be
    confirmed the sentence is not supported, and the reply quotes the article.
    """
    plain = strip_citations(sentence)
    # A number and its name must come from the same sentence, unless the second
    # sentence continues the first ("Anh ghi 21 bàn …" after "… Kane …").
    scopes = [" ".join(texts)] if len(texts) == 2 and _continues_text(texts[1]) else list(texts)
    joined = " ".join(texts)
    # "Bài viết không nêu …" is decided from the passages, never taken from the model;
    # any other negative must be stated by the passage itself.
    if _has_phrase(plain, lexicon.ABSENCE_PHRASES):
        return False
    if denies(plain) and not any(denies(scope) for scope in scopes):
        return False
    allowed = ""
    known = _folded_words(joined)
    if any(name not in known for name in names_in(plain)):
        return False
    # Most of each part's words must come from one sentence (or a continuing pair):
    # "Trang Moon là tên nhân vật" has none of "tên nhân vật" in the sentence about Trang
    # Moon, and "Victor Mansaray thuộc CLB Long An" no sentence with both.
    for part_text in _PART_BREAK.split(plain):
        if not part_text or len(_content_words(part_text)) < 1:
            continue
        if not any(_part_stated(part_text, scope) for scope in scopes):
            return False
    # Names said together must be together in the passage: "Toản (Duy Hưng)" does not
    # make Duy Hưng the actor of Tươi. A short answer carries the question's names.
    joined_pair = len(texts) == 2 and _continues_text(texts[1])
    for part_text in _PART_BREAK.split(plain):
        if not part_text or not part_text.strip():
            continue
        names = [g[2] for g in _name_groups(part_text)]
        names += [n for n in (claim_names or []) if n not in names]
        if not _names_together(names, texts, joined_pair):
            return False
    unit_quantities = [q for text in texts for q in _quantities(text)]
    allowed_quantities = _quantities(allowed) if allowed else []
    answer_clauses = _clauses(plain)
    for clause in answer_clauses:
        clause_names = _name_groups(clause)
        clause_words = _folded_words(clause)
        for quantity in _quantities(clause):
            if not (_quantity_supported(quantity, unit_quantities) or _quantity_supported(quantity, allowed_quantities)):
                return False
            if allowed_quantities and _quantity_supported(quantity, allowed_quantities):
                continue
            # The name nearest the number must be in the sentence that has the number:
            # "Kane ba lần giành Vua phá lưới" does not support "Kane giành Quả Bóng Vàng 3 lần".
            if clause_names:
                position = quantity["position"]
                nearest = min(clause_names, key=lambda g: min(abs(g[0] - position), abs(g[1] - position)))[2]
                attached = False
                for scope in scopes:
                    if not _quantity_supported(quantity, _quantities(scope)):
                        continue
                    if not all(part in _folded_words(scope) for part in nearest.split()):
                        continue
                    # A joint total is not one contributor's: "Opella và BW Industrial
                    # bàn giao 11 công trình" does not support "BW Industrial tài trợ 11".
                    attached = not any(
                        any(nearest in name or name in nearest for name in chain)
                        and any(not all(part in clause_words for part in name.split()) for name in chain)
                        for passage_clause in _clauses(scope)
                        if _quantity_supported(quantity, _quantities(passage_clause))
                        for chain in _joint_names(passage_clause)
                    )
                    if attached:
                        break
                if not attached:
                    return False
    # Plans, offers and predictions stay what they are.
    for passage_clause in (c for text in texts for c in _clauses(text)):
        names = names_in("x " + passage_clause)
        numbers = {f for q in _quantities(passage_clause) for f in q["forms"]}
        answer_numbers = {f for q in _quantities(plain) for f in q["forms"]}
        shares = (numbers & answer_numbers) or (names & names_in(plain) and len(names & names_in(plain)) >= 2)
        if not shares:
            continue
        for kind, markers in lexicon.MODALITY_MARKERS.items():
            if _has_phrase(passage_clause, markers) and not _has_phrase(plain, markers):
                return False
    for kind, markers in lexicon.MODALITY_MARKERS.items():
        if _has_phrase(plain, markers) and not any(_has_phrase(text, markers) for text in texts):
            return False  # "Dự kiến … sẽ nghỉ việc" where the article states an offer
            if _has_phrase(plain, lexicon.OUTCOME_MARKERS[kind]) and _has_phrase(passage_clause, markers) and not _has_phrase(
                passage_clause, lexicon.OUTCOME_MARKERS[kind]
            ):
                return False
    if role_conflicts(plain, article_text):
        return False
    if anchors is not None and not _anchored(plain, texts, anchors):
        return False
    return True


def _anchored(plain: str, texts: list[str], anchors: tuple[list[set[str]], list[str]]) -> bool:
    """Is the answer stated by a sentence that also has the question's numbers and names?

    "Trong bài, ba danh hiệu Vua phá lưới thuộc giải đấu nào?" is answered by the
    sentence "Kane ba lần giành Vua phá lưới Ngoại hạng Anh", not by "giành Vua phá
    lưới World Cup 2018", which mentions the award without the three titles.
    """
    numbers, names = anchors
    if not numbers:
        return True
    joined = len(texts) == 2 and _continues_text(texts[1])
    scopes = [" ".join(texts)] if joined else list(texts)
    parts = [part for part in _PART_BREAK.split(plain) if part and _content_words(part)]
    for scope in scopes:
        if not all(_part_stated(part, scope) for part in parts):
            continue
        forms = [q["forms"] for q in _quantities(scope)]
        words = _folded_words(scope)
        if all(any(anchor & f for f in forms) for anchor in numbers) and all(
            all(part in words for part in name.split()) for name in names
        ):
            return True
    return False


def cited_labels(sentence: str) -> list[int]:
    labels: list[int] = []
    for match in _CITATION_NUMBERS.finditer(sentence):
        labels.extend(int(n) for n in re.findall(r"\d+", match.group(1)))
    return labels


def find_support(sentence: str, evidence: Evidence) -> list[Support]:
    """The passages that support ``sentence``: those of the cited articles first, then any.

    Returns one support per article that holds it; empty when none does. A
    sentence claiming the same of every article ("cả hai bài đều …") needs every
    cited article to hold it on its own.
    """
    plain = strip_citations(sentence).strip()
    if not plain or is_verdict(plain):
        return []
    if is_echo(plain, evidence.question):
        return []
    labels = [n for n in cited_labels(sentence) if 1 <= n <= len(evidence.articles)]
    order = labels + [n for n in range(1, len(evidence.articles) + 1) if n not in labels]
    # A short answer ("Duy Hưng") stands for the question's claim: its names come along.
    claim_names = None
    if len(_content_words(plain)) <= 4 and evidence.need.entity_spans:
        claim_names = [
            " ".join(fold_text(accented_lower(w)) for w in span.split()) for span in evidence.need.entity_spans
        ]
    # The question's own numbers and names bind the answer to the clause that has them.
    anchors = (
        evidence.need.number_anchors,
        [" ".join(fold_text(accented_lower(w)) for w in span.split()) for span in evidence.need.entity_spans],
    )
    found: list[Support] = []
    for label in order:
        article_text = evidence.full_text(label)
        for unit in evidence.units(label):
            if _unit_supports(
                plain, evidence.question, [p.text for p in unit], article_text, claim_names=claim_names,
                anchors=anchors,
            ):
                found.append(Support(label, unit))
                break
    if not found:
        return []
    if _has_phrase(plain, lexicon.EVERY_WORDS) and len(evidence.articles) > 1:
        supported = {s.label for s in found}
        needed = set(labels) if len(labels) > 1 else set(range(1, len(evidence.articles) + 1))
        if not needed <= supported:
            return []
    cited_found = [s for s in found if s.label in labels]
    return cited_found or found[:1]


def cite(sentence: str, supports: list[Support]) -> str:
    """The sentence with the citations of the passages that support it."""
    plain = strip_citations(sentence).rstrip()
    plain = re.sub(r"\s+([.!?…:])", r"\1", plain)
    tag = "[" + ", ".join(f"Nguồn {s.label}" for s in sorted(supports, key=lambda s: s.label)) + "]"
    trailing = re.search(r"[.!?…:]+$", plain)
    if trailing:
        return f"{plain[:trailing.start()].rstrip()} {tag}{trailing.group(0)}"
    return f"{plain} {tag}"


def supported_by_text(sentence: str, text: str) -> bool:
    """Is a summary sentence supported by ``text``, each clause by one sentence (or two in a row)?

    A summary may join facts from different parts of the article ("Kane, người
    từng bị Arsenal loại, … đoạt Quả Bóng Vàng sau 25 năm"), so each clause is
    checked on its own, with the same checks as an answer: numbers and names stay
    together ("Tiền đạo 11 tuổi Kane ghi 21 bàn" has no sentence with both numbers),
    and the clause's wording, verbs, status and name order must be in one sentence.
    """
    sentences = _split_sentences(text)
    units = [[s] for s in sentences] + [[a, b] for a, b in zip(sentences, sentences[1:])]
    clauses = [c for c in _clauses(strip_citations(sentence)) if re.search(r"[0-9A-Za-zÀ-ỹĐđ]", c)]
    return all(
        any(_unit_supports(clause, "", unit, text) for unit in units) for clause in clauses or [sentence]
    )
