"""Extractive summarizers: Lead-k baseline, TextRank, and PhoBERT centroid + MMR."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from evaluation.metrics import tokenize
from processing.sentences import split_sentences
from processing.summarizers.base import Summarizer


def _join_in_document_order(sentences: list[str], selected: list[int]) -> str:
    return " ".join(sentences[i] for i in sorted(selected))


class LeadSummarizer(Summarizer):
    """Lead-k: the first k sentences. The standard strong baseline for news."""

    kind = "extractive"

    def __init__(self, num_sentences: int = 3) -> None:
        if num_sentences <= 0:
            raise ValueError("num_sentences must be positive")
        self.num_sentences = num_sentences
        self.name = f"lead{num_sentences}"

    def summarize(self, text: str) -> str:
        sentences = split_sentences(text)
        return " ".join(sentences[: self.num_sentences])

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "num_sentences": self.num_sentences}


def pagerank(
    weights: np.ndarray,
    damping: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> np.ndarray:
    """Weighted PageRank by power iteration; dangling nodes jump uniformly."""
    size = weights.shape[0]
    if size == 0:
        return np.empty(0)
    row_sums = weights.sum(axis=1, keepdims=True)
    transition = np.divide(
        weights,
        row_sums,
        out=np.full_like(weights, 1.0 / size, dtype=np.float64),
        where=row_sums > 0,
    )
    scores = np.full(size, 1.0 / size)
    for _ in range(max_iter):
        updated = (1 - damping) / size + damping * (transition.T @ scores)
        if np.abs(updated - scores).sum() < tol:
            return updated
        scores = updated
    return scores


def textrank_similarity(a: list[str], b: list[str]) -> float:
    """Sentence similarity from the original TextRank paper (Mihalcea & Tarau, 2004).

    sim(Si, Sj) = |{w : w in Si and w in Sj}| / (log|Si| + log|Sj|)
    """
    if len(a) < 2 or len(b) < 2:
        # log(1) = 0 would make the denominator vanish for one-token sentences.
        return 0.0
    overlap = len(set(a) & set(b))
    if overlap == 0:
        return 0.0
    return overlap / (math.log(len(a)) + math.log(len(b)))


class TextRankSummarizer(Summarizer):
    """Graph-based extractive summarization with the original TextRank similarity."""

    name = "textrank"
    kind = "extractive"

    def __init__(self, num_sentences: int = 3, damping: float = 0.85) -> None:
        if num_sentences <= 0:
            raise ValueError("num_sentences must be positive")
        self.num_sentences = num_sentences
        self.damping = damping

    def rank(self, sentences: list[str]) -> np.ndarray:
        tokens = [tokenize(sentence) for sentence in sentences]
        size = len(sentences)
        weights = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            for j in range(i + 1, size):
                weights[i, j] = weights[j, i] = textrank_similarity(tokens[i], tokens[j])
        return pagerank(weights, damping=self.damping)

    def summarize(self, text: str) -> str:
        sentences = split_sentences(text)
        if len(sentences) <= self.num_sentences:
            return " ".join(sentences)
        scores = self.rank(sentences)
        # Stable tie-break: earlier sentences win when scores are equal.
        order = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
        return _join_in_document_order(sentences, order[: self.num_sentences])

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "num_sentences": self.num_sentences,
            "damping": self.damping,
            "similarity": "Mihalcea & Tarau 2004 word overlap",
        }


def mmr_select(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    k: int,
    diversity: float = 0.3,
) -> list[int]:
    """Maximal Marginal Relevance over L2-normalized embeddings.

    score(i) = (1 - diversity) * relevance[i] - diversity * max_{j in selected} cos(i, j)
    """
    selected: list[int] = []
    candidates = list(range(len(relevance)))
    while candidates and len(selected) < k:
        if selected:
            redundancy = (embeddings[candidates] @ embeddings[selected].T).max(axis=1)
        else:
            redundancy = np.zeros(len(candidates))
        scores = (1 - diversity) * relevance[candidates] - diversity * redundancy
        best = candidates[int(np.argmax(scores))]
        selected.append(best)
        candidates.remove(best)
    return selected


def _load_word_segmenter() -> Callable[[str], str] | None:
    """Return a Vietnamese word segmenter if pyvi is installed (PhoBERT expects one)."""
    try:
        from pyvi import ViTokenizer  # type: ignore[import-not-found]
    except ImportError:
        return None
    return ViTokenizer.tokenize


class PhoBERTCentroidSummarizer(Summarizer):
    """BERT-based unsupervised extractive summarization with PhoBERT.

    Sentences are embedded with the project's PhoBERT encoder (mean pooling),
    scored by cosine similarity to the document centroid, and selected with MMR
    to avoid redundancy. Input is word-segmented with pyvi when it is installed,
    which is how PhoBERT was pre-trained.
    """

    name = "phobert_centroid"
    kind = "extractive"

    def __init__(
        self,
        num_sentences: int = 3,
        diversity: float = 0.3,
        embed: Callable[[list[str]], np.ndarray] | None = None,
        segmenter: Callable[[str], str] | None | bool = True,
    ) -> None:
        if num_sentences <= 0:
            raise ValueError("num_sentences must be positive")
        self.num_sentences = num_sentences
        self.diversity = diversity
        self._embed = embed
        if segmenter is True:
            self._segment = _load_word_segmenter()
        elif segmenter is False:
            self._segment = None
        else:
            self._segment = segmenter

    def _embed_sentences(self, sentences: list[str]) -> np.ndarray:
        if self._embed is None:
            from nlp.semantic_search import create_embeddings

            self._embed = lambda texts: create_embeddings(
                texts, batch_size=16, normalize=True, max_length=256
            )
        inputs = [self._segment(s) for s in sentences] if self._segment else sentences
        vectors = np.asarray(self._embed(inputs), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)

    def summarize(self, text: str) -> str:
        sentences = split_sentences(text)
        if len(sentences) <= self.num_sentences:
            return " ".join(sentences)
        embeddings = self._embed_sentences(sentences)
        centroid = embeddings.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        relevance = embeddings @ centroid
        selected = mmr_select(embeddings, relevance, self.num_sentences, self.diversity)
        return _join_in_document_order(sentences, selected)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "num_sentences": self.num_sentences,
            "mmr_diversity": self.diversity,
            "word_segmented": self._segment is not None,
            "encoder": "PhoBERT mean pooling (models/phobert)",
        }

    def close(self) -> None:
        try:
            from nlp.semantic_search import get_embedding_model
        except ImportError:
            return
        get_embedding_model.cache_clear()
        _empty_cuda_cache()


def _empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
