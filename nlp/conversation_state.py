"""What the conversation has shown and selected so far.

One object replaces the loose ``last_search_sources`` / ``active_article`` /
``pinned_article`` values. Its rules:

* The numbered list is the last list the user saw. An empty search does not
  replace it, but it is remembered so a later "bài số 1" can say which list it
  comes from.
* Pinning and opening are both selections with an order. "bài này" or "bài
  vừa rồi" means the most recent selection, whichever kind it was.
* The last turn keeps its topic words, target and sources, so a follow-up
  ("dự án này", "nói ngắn hơn") can reuse them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class ListRequest:
    """How a list was produced, so it can be re-run with changed filters."""

    action: str
    topic_words: list[str] = field(default_factory=list)
    source: str | None = None
    category: str | None = None
    count: int | None = None
    date_from: date | None = None
    date_to: date | None = None
    question: str = ""
    exclude_sources: list[str] = field(default_factory=list)
    #: Articles already shown for this request ("ba bài khác"); they are left out.
    skip: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = list(self.topic_words)
        if self.category:
            parts.append(f"chuyên mục {self.category}")
        if self.source:
            parts.append(f"báo {self.source}")
        if self.exclude_sources:
            parts.append("trừ báo " + ", ".join(self.exclude_sources))
        if self.date_from:
            if self.date_to and self.date_to != self.date_from:
                parts.append(f"từ {self.date_from:%d/%m/%Y} đến {self.date_to:%d/%m/%Y}")
            else:
                parts.append(f"ngày {self.date_from:%d/%m/%Y}")
        return ", ".join(parts) or self.question or "tin mới nhất"

    def same_filters(self, other: "ListRequest | None") -> bool:
        """Same kind of list with the same subject and filters (the count may differ)."""
        if other is None:
            return False
        keys = ("action", "topic_words", "source", "category", "date_from", "date_to")
        return all(getattr(self, key) == getattr(other, key) for key in keys) and sorted(
            self.exclude_sources
        ) == sorted(other.exclude_sources)


def article_key(article: dict[str, Any]) -> str:
    """Identity of an article across lists: its id, else its URL, else its title."""
    return str(article.get("article_id") or article.get("url") or article.get("title") or "")


@dataclass
class TurnRecord:
    question: str
    intent: str
    topic_words: list[str] = field(default_factory=list)
    target: dict[str, Any] | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    request: ListRequest | None = None
    max_sentences: int | None = None
    #: Passages ("article#index") that the answer was taken from; a follow-up prefers them.
    evidence: list[str] = field(default_factory=list)
    #: List numbers of the articles answered from, when the user named them (a comparison).
    positions: list[int] = field(default_factory=list)
    #: (list number, article) of the articles last compared; questions "trong hai bài vừa
    #: so sánh" stay within them until another comparison or list replaces them.
    compared: list[tuple[int, dict[str, Any]]] = field(default_factory=list)


def same_article(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if not a or not b:
        return False
    for key in ("article_id", "url"):
        if a.get(key) and a.get(key) == b.get(key):
            return True
    return bool(a.get("title")) and a.get("title") == b.get("title")


@dataclass
class ConversationState:
    items: list[dict[str, Any]] = field(default_factory=list)
    list_request: ListRequest | None = None
    #: Set when the most recent list request found nothing; ``items`` still holds the older list.
    empty_request: ListRequest | None = None
    selections: list[tuple[int, str, dict[str, Any]]] = field(default_factory=list)
    pinned: dict[str, Any] | None = None
    last: TurnRecord | None = None
    _sequence: int = 0

    # --- Selections ---------------------------------------------------------------

    def select(self, article: dict[str, Any], kind: str = "open") -> None:
        self._sequence += 1
        self.selections.append((self._sequence, kind, dict(article)))
        self.selections = self.selections[-20:]
        if kind == "pin":
            self.pinned = dict(article)

    def unpin(self) -> None:
        self.pinned = None
        self.selections = [entry for entry in self.selections if entry[1] != "pin"]

    def deselect(self) -> None:
        """Back to the list: no article is open or pinned."""
        self.pinned = None
        self.selections = []

    def current_article(self) -> dict[str, Any] | None:
        """The most recent selection, pinned or opened."""
        return self.selections[-1][2] if self.selections else None

    def clear(self) -> None:
        self.items = []
        self.list_request = None
        self.empty_request = None
        self.selections = []
        self.pinned = None
        self.last = None

    # --- Lists --------------------------------------------------------------------

    def index_of(self, article: dict[str, Any] | None) -> int | None:
        for index, item in enumerate(self.items):
            if same_article(item, article):
                return index
        return None

    def set_list(self, items: list[dict[str, Any]], request: ListRequest | None) -> None:
        if items:
            self.items = list(items)
            self.list_request = request
            self.empty_request = None
        else:
            self.empty_request = request

    def show_list(self, items: list[dict[str, Any]], request: ListRequest | None) -> None:
        """A new list on screen: opened articles are closed, pinned ones stay."""
        self.set_list(items, request)
        if items:
            self.selections = [entry for entry in self.selections if entry[1] == "pin"]

    # --- Turns --------------------------------------------------------------------

    def record(self, record: TurnRecord, outcome: dict[str, Any]) -> None:
        sources = outcome.get("sources") or []
        if outcome.get("selected_single") and sources:
            self.select(sources[0], "open")
            record.target = sources[0]
        elif outcome.get("keeps_list"):
            pass  # articles picked from the shown list ("so sánh bài 2 và bài 9"): it stays as it was
        elif outcome.get("is_list"):
            self.show_list(sources, record.request)
        elif sources:
            # A question answered from several articles: they become the numbered list.
            self.set_list(sources, record.request)
        record.sources = list(sources)
        # The same list as the outcome's: a streamed answer fills it as it is checked.
        evidence = outcome.get("evidence")
        record.evidence = evidence if isinstance(evidence, list) else []
        record.positions = list(outcome.get("list_positions") or [])
        # A turn without articles (a clarifying question) keeps the pair being compared.
        carried = list(self.last.compared) if self.last is not None and not sources else []
        record.compared = list(outcome.get("compared") or carried)
        self.last = record

    @classmethod
    def from_legacy(
        cls,
        context_sources: list[dict[str, Any]] | None = None,
        pinned_article: dict[str, Any] | None = None,
        active_article: dict[str, Any] | None = None,
    ) -> "ConversationState":
        """State from the older separate values; the active article counts as more recent."""
        state = cls(items=list(context_sources or []))
        if pinned_article:
            state.select(pinned_article, "pin")
        if active_article and not same_article(active_article, pinned_article):
            state.select(active_article, "open")
        return state
