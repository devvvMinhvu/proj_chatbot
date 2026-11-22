"""Corpus statistics that decide which query words carry a topic.

Two kinds of words are not topic words, and both are learned from data instead
of being listed by hand:

* Function words (là, đã, tại, những…): words that appear in a large share of
  the news articles. The cut-off is the knee of the document-frequency curve,
  found automatically.
* Command words (tìm, bài, xem, giúp…): words that are more common in user
  requests (``data/intent_dataset.csv``) than in news articles.

Words typed with accents are looked up with their accents; words typed without
accents are looked up in accent-free form, so "la" (from "là") counts as a
function word while an accented "lá" does not. Names are handled by callers:
capitalised words and words that cannot be Vietnamese syllables are always kept.

Build the cache once with ``python -m nlp.keyword_stats``; the app builds it on
first use if it is missing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import csv
import gzip
import json
import math
from pathlib import Path
import re
import sqlite3

from config import DATA_DIR, INTENT_DATASET_PATH, NEWS_DB_PATH
from processing.text_matching import accented_lower, fold_text

KEYWORD_STATS_PATH = DATA_DIR / "keyword_stats.json.gz"
_WORD = re.compile(r"[0-9a-zA-ZÀ-ỹĐđ]+")
#: Only the most frequent words can be function words; the knee is searched among them.
_KNEE_SEARCH_SIZE = 3000


def _words(text: str) -> list[str]:
    """Lowercased words; compound words from the segmenter are joined with spaces."""
    return [word.replace("_", " ") for word in segment(accented_lower(text))]


def segment(text: str) -> list[str]:
    """Vietnamese word segmentation (pyvi): "dự án vệ sinh" -> ["dự_án", "vệ_sinh"]."""
    from pyvi import ViTokenizer

    return [tok for tok in ViTokenizer.tokenize(str(text or "")).split() if _WORD.search(tok)]


@dataclass
class KeywordStats:
    documents: int
    accented_df: dict[str, int]
    folded_df: dict[str, int]
    function_df: int
    command_words: set[str] = field(default_factory=set)
    command_words_folded: set[str] = field(default_factory=set)

    def df(self, word: str) -> int:
        lowered = accented_lower(word)
        folded = fold_text(lowered)
        if folded == lowered:
            return self.folded_df.get(folded, 0)
        return self.accented_df.get(lowered, 0)

    def idf(self, word: str) -> float:
        return math.log((1 + self.documents) / (1 + self.df(word))) + 1.0

    def is_function_word(self, word: str) -> bool:
        return self.df(word) >= self.function_df

    def is_command_word(self, word: str) -> bool:
        lowered = accented_lower(word)
        folded = fold_text(lowered)
        if folded == lowered:
            return folded in self.command_words_folded
        return lowered in self.command_words

    def is_topic_word(self, word: str) -> bool:
        if len(word) < 2 and not word.isdigit():
            return False
        return not (self.is_function_word(word) or self.is_command_word(word))

    def to_json(self) -> dict:
        return {
            "documents": self.documents,
            "accented_df": self.accented_df,
            "folded_df": self.folded_df,
            "function_df": self.function_df,
            "command_words": sorted(self.command_words),
            "command_words_folded": sorted(self.command_words_folded),
        }

    @classmethod
    def from_json(cls, data: dict) -> "KeywordStats":
        return cls(
            documents=int(data["documents"]),
            accented_df={k: int(v) for k, v in data["accented_df"].items()},
            folded_df={k: int(v) for k, v in data["folded_df"].items()},
            function_df=int(data["function_df"]),
            command_words=set(data.get("command_words", [])),
            command_words_folded=set(data.get("command_words_folded", [])),
        )


def knee_threshold(document_frequencies: list[int]) -> int:
    """Document frequency at the knee of the rank/frequency curve (Kneedle method).

    Frequencies are sorted in descending order and both axes scaled to [0, 1];
    the knee is the point farthest below the straight line joining the first
    and last points. Words at or above that frequency are function words.
    """
    values = sorted((v for v in document_frequencies if v > 0), reverse=True)[:_KNEE_SEARCH_SIZE]
    if len(values) < 3:
        return max(values) + 1 if values else 1
    top, bottom = values[0], values[-1]
    span = max(top - bottom, 1)
    last = len(values) - 1
    best_index, best_gap = 0, -1.0
    for index, value in enumerate(values):
        x = index / last
        y = (value - bottom) / span
        gap = (1.0 - x) - y  # distance below the line y = 1 - x
        if gap > best_gap:
            best_index, best_gap = index, gap
    return values[best_index]


def command_words_from_requests(
    requests: list[str],
    documents: int,
    accented_df: dict[str, int],
) -> set[str]:
    """Words more common in user requests than in news articles (seen in 2+ requests)."""
    counts: Counter[str] = Counter()
    for text in requests:
        counts.update(set(_words(text)))
    total = max(len(requests), 1)
    words: set[str] = set()
    for word, count in counts.items():
        if count < 2:
            continue
        request_share = count / total
        article_share = accented_df.get(word, 0) / max(documents, 1)
        if request_share > article_share:
            words.add(word)
    return words


def build_stats(
    db_path: Path = NEWS_DB_PATH,
    requests_path: Path = INTENT_DATASET_PATH,
) -> KeywordStats:
    accented: Counter[str] = Counter()
    documents = 0
    with sqlite3.connect(str(db_path)) as connection:
        for title, sapo, body in connection.execute("SELECT title, sapo, body FROM articles"):
            documents += 1
            accented.update(set(_words(f"{title or ''} {sapo or ''} {body or ''}")))
    folded: Counter[str] = Counter()
    for word, count in accented.items():
        folded[fold_text(word)] += count
    # A folded form's DF can exceed the document count when several accented forms
    # share it; cap it so the share stays a probability.
    folded_df = {word: min(count, documents) for word, count in folded.items()}
    accented_df = dict(accented)

    requests: list[str] = []
    if requests_path.exists():
        with requests_path.open(encoding="utf-8") as handle:
            requests = [row["text"] for row in csv.DictReader(handle) if row.get("text")]
    command = command_words_from_requests(requests, documents, accented_df)

    return KeywordStats(
        documents=documents,
        accented_df=accented_df,
        folded_df=folded_df,
        function_df=knee_threshold(list(accented_df.values())),
        command_words=command,
        command_words_folded={fold_text(word) for word in command},
    )


def save_stats(stats: KeywordStats, path: Path = KEYWORD_STATS_PATH) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(stats.to_json(), handle, ensure_ascii=False)


@lru_cache(maxsize=1)
def get_keyword_stats() -> KeywordStats:
    """Load the cached statistics, building them from the database if needed."""
    if KEYWORD_STATS_PATH.exists():
        with gzip.open(KEYWORD_STATS_PATH, "rt", encoding="utf-8") as handle:
            return KeywordStats.from_json(json.load(handle))
    stats = build_stats()
    try:
        save_stats(stats)
    except OSError:
        pass
    return stats


@lru_cache(maxsize=1)
def conversation_words(path: Path = INTENT_DATASET_PATH) -> frozenset[str]:
    """Words of small talk, learned from the greeting and introduction examples.

    A word counts when it appears in at least two such examples ("bạn", "chào",
    "trợ lý"…). A greeting or introduction label is trusted only when every topic
    word of the message is one of these.
    """
    counts: Counter[str] = Counter()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("intent") or "").strip() in {"greeting", "introduction"}:
                    counts.update(set(_words(row.get("text") or "")))
    return frozenset(word for word, count in counts.items() if count >= 2)


if __name__ == "__main__":
    built = build_stats()
    save_stats(built)
    get_keyword_stats.cache_clear()
    print(f"Articles: {built.documents}, vocabulary: {len(built.accented_df)}")
    print(f"Function-word cut-off: document frequency >= {built.function_df}")
    print(f"Command words: {', '.join(sorted(built.command_words))}")
    print(f"Saved to {KEYWORD_STATS_PATH}")
