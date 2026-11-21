"""Summarization metrics for Vietnamese text: ROUGE-N, ROUGE-L, BLEU, bootstrap CIs.

All metrics work on *syllable* tokens (Vietnamese words are space-separated
syllables). Word-segmented corpora such as VietNews join syllables with "_";
those underscores are split back into syllables so segmented and unsegmented
texts are scored identically.

Note: the popular ``rouge_score`` package's default tokenizer drops every
non-ASCII character, which silently destroys Vietnamese text. That is why the
metrics are implemented here with a Unicode-aware tokenizer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)

#: Per-sample F1 scores compared between systems.
F1_METRICS = ("rouge1_f1", "rouge2_f1", "rougeL_f1")


def tokenize(text: str) -> list[str]:
    """Lowercase, NFC-normalize, and split text into alphanumeric syllable tokens."""
    normalized = unicodedata.normalize("NFC", str(text or "")).casefold()
    return _TOKEN_RE.findall(normalized)


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float


def _prf(overlap: float, hyp_total: int, ref_total: int) -> PRF:
    if overlap <= 0 or hyp_total <= 0 or ref_total <= 0:
        return PRF(0.0, 0.0, 0.0)
    precision = overlap / hyp_total
    recall = overlap / ref_total
    return PRF(precision, recall, 2 * precision * recall / (precision + recall))


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(reference: Sequence[str], hypothesis: Sequence[str], n: int) -> PRF:
    """ROUGE-N precision/recall/F1 with clipped n-gram counts."""
    ref_grams = _ngrams(reference, n)
    hyp_grams = _ngrams(hypothesis, n)
    overlap = sum((ref_grams & hyp_grams).values())
    return _prf(overlap, sum(hyp_grams.values()), sum(ref_grams.values()))


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence (O(len(a)*len(b)) time, O(len(b)) memory)."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference: Sequence[str], hypothesis: Sequence[str]) -> PRF:
    """ROUGE-L precision/recall/F1 from the LCS over the full token sequences."""
    return _prf(lcs_length(reference, hypothesis), len(hypothesis), len(reference))


def rouge_scores(reference: str, hypothesis: str) -> dict[str, float]:
    """Return ROUGE-1/2/L precision, recall, and F1 for one reference/hypothesis pair."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    scores: dict[str, float] = {}
    for name, prf in (
        ("rouge1", rouge_n(ref_tokens, hyp_tokens, 1)),
        ("rouge2", rouge_n(ref_tokens, hyp_tokens, 2)),
        ("rougeL", rouge_l(ref_tokens, hyp_tokens)),
    ):
        scores[f"{name}_p"] = prf.precision
        scores[f"{name}_r"] = prf.recall
        scores[f"{name}_f1"] = prf.f1
    return scores


def _bleu_from_stats(
    matches: Sequence[int],
    totals: Sequence[int],
    hyp_len: int,
    ref_len: int,
    smooth: bool,
) -> float:
    if hyp_len == 0:
        return 0.0
    log_precision = 0.0
    for n, (match, total) in enumerate(zip(matches, totals), start=1):
        if smooth and n > 1:
            # Lin & Och (2004) add-one smoothing for higher-order n-grams.
            match, total = match + 1, total + 1
        if total == 0 or match == 0:
            return 0.0
        log_precision += math.log(match / total)
    log_precision /= len(matches)
    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return brevity_penalty * math.exp(log_precision)


def _bleu_stats(reference: Sequence[str], hypothesis: Sequence[str], max_n: int):
    matches, totals = [], []
    for n in range(1, max_n + 1):
        ref_grams = _ngrams(reference, n)
        hyp_grams = _ngrams(hypothesis, n)
        matches.append(sum((ref_grams & hyp_grams).values()))
        totals.append(max(len(hypothesis) - n + 1, 0))
    return matches, totals


def sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Smoothed sentence-level BLEU (single reference), in [0, 1]."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    matches, totals = _bleu_stats(ref_tokens, hyp_tokens, max_n)
    return _bleu_from_stats(matches, totals, len(hyp_tokens), len(ref_tokens), smooth=True)


def corpus_bleu(references: Sequence[str], hypotheses: Sequence[str], max_n: int = 4) -> float:
    """Corpus-level BLEU-N (Papineni et al., 2002), single reference per sample, in [0, 1]."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    matches = [0] * max_n
    totals = [0] * max_n
    hyp_len = ref_len = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_tokens = tokenize(reference)
        hyp_tokens = tokenize(hypothesis)
        sample_matches, sample_totals = _bleu_stats(ref_tokens, hyp_tokens, max_n)
        matches = [a + b for a, b in zip(matches, sample_matches)]
        totals = [a + b for a, b in zip(totals, sample_totals)]
        hyp_len += len(hyp_tokens)
        ref_len += len(ref_tokens)
    return _bleu_from_stats(matches, totals, hyp_len, ref_len, smooth=False)


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of per-sample scores."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (0.0, 0.0)
    if array.size == 1 or n_resamples <= 0:
        mean = float(array.mean())
        return (mean, mean)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


@dataclass(frozen=True)
class PairedBootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    samples: int


def paired_bootstrap(
    system_scores: Sequence[float],
    baseline_scores: Sequence[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Paired bootstrap test (Koehn, 2004) for the mean difference system - baseline.

    Both sequences must hold scores for the same samples in the same order.
    Articles are resampled with replacement and the per-article differences are
    averaged. The two-sided p-value is twice the smaller tail mass of resampled
    means on either side of zero (capped at 1); the CI is the percentile interval.
    """
    a = np.asarray(system_scores, dtype=np.float64)
    b = np.asarray(baseline_scores, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired scores must have the same length")
    if a.size == 0:
        raise ValueError("paired scores must not be empty")
    diffs = a - b
    mean_diff = float(diffs.mean())
    if n_resamples <= 0 or a.size == 1:
        return PairedBootstrapResult(mean_diff, mean_diff, mean_diff, 1.0, int(a.size))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    means = diffs[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    tail = min(float((means <= 0).mean()), float((means >= 0).mean()))
    return PairedBootstrapResult(
        mean_diff=mean_diff,
        ci_low=float(np.quantile(means, alpha)),
        ci_high=float(np.quantile(means, 1 - alpha)),
        p_value=min(1.0, 2 * tail),
        samples=int(a.size),
    )
