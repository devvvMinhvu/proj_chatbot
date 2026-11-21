"""Common interface shared by every summarization system in the benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Summarizer(ABC):
    """A summarization system that maps an article body to a summary string."""

    #: Short identifier used in reports (e.g. "textrank", "vit5").
    name: str = "summarizer"
    #: "extractive" or "abstractive".
    kind: str = "extractive"

    @abstractmethod
    def summarize(self, text: str) -> str:
        """Return a summary of ``text`` (empty string for empty input)."""

    def describe(self) -> dict[str, Any]:
        """Configuration recorded in benchmark reports for reproducibility."""
        return {"name": self.name, "kind": self.kind}

    def close(self) -> None:
        """Release models/GPU memory. Called after the system has been benchmarked."""
