"""Unit tests for summarization metrics, summarizers, datasets, and the benchmark CLI.

None of these tests load PhoBERT or ViT5; model-backed systems are exercised with
injected fake encoders so the suite runs without GPUs or model files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluation import significance, summarization_benchmark
from evaluation.datasets import (
    containment,
    load_jsonl_examples,
    load_news_examples,
    strip_reference_from_document,
)
from evaluation.metrics import (
    bootstrap_ci,
    corpus_bleu,
    paired_bootstrap,
    rouge_l,
    rouge_n,
    rouge_scores,
    sentence_bleu,
    tokenize,
)
from processing.sentences import split_sentences
from processing.summarizers import build_summarizer
from processing.summarizers.extractive import (
    LeadSummarizer,
    PhoBERTCentroidSummarizer,
    TextRankSummarizer,
    mmr_select,
    pagerank,
)


class TokenizeTests(unittest.TestCase):
    def test_keeps_vietnamese_diacritics_and_lowercases(self):
        self.assertEqual(tokenize("Hà Nội, ĐÀ NẴNG!"), ["hà", "nội", "đà", "nẵng"])

    def test_splits_word_segmentation_underscores(self):
        self.assertEqual(tokenize("Hà_Nội mưa"), tokenize("Hà Nội mưa"))

    def test_normalizes_unicode_composition(self):
        decomposed = "Việt"  # "Việt" written with combining marks
        self.assertEqual(tokenize(decomposed), tokenize("Việt"))


class RougeTests(unittest.TestCase):
    def test_rouge_n_hand_computed(self):
        ref, hyp = "a b c d".split(), "a b e".split()
        r1 = rouge_n(ref, hyp, 1)
        self.assertAlmostEqual(r1.precision, 2 / 3)
        self.assertAlmostEqual(r1.recall, 0.5)
        self.assertAlmostEqual(r1.f1, 4 / 7)
        r2 = rouge_n(ref, hyp, 2)
        self.assertAlmostEqual(r2.f1, 0.4)

    def test_rouge_l_hand_computed(self):
        self.assertAlmostEqual(rouge_l("a b c d".split(), "a c d e".split()).f1, 0.75)

    def test_identical_and_empty(self):
        scores = rouge_scores("Giá vàng tăng mạnh", "giá vàng tăng mạnh")
        self.assertAlmostEqual(scores["rouge1_f1"], 1.0)
        self.assertAlmostEqual(scores["rouge2_f1"], 1.0)
        self.assertAlmostEqual(scores["rougeL_f1"], 1.0)
        self.assertEqual(rouge_scores("giá vàng", "")["rouge1_f1"], 0.0)


class BleuTests(unittest.TestCase):
    def test_identical_is_one(self):
        self.assertAlmostEqual(corpus_bleu(["a b c d e"], ["a b c d e"]), 1.0)

    def test_brevity_penalty(self):
        # All n-gram precisions are 1; BP = exp(1 - 6/4).
        self.assertAlmostEqual(corpus_bleu(["a b c d e f"], ["a b c d"]), math.exp(-0.5))

    def test_no_four_gram_match_is_zero_unsmoothed(self):
        self.assertEqual(corpus_bleu(["a b c d"], ["a b x d"]), 0.0)
        self.assertGreater(sentence_bleu("a b c d", "a b x d"), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            corpus_bleu(["a"], [])


class BootstrapTests(unittest.TestCase):
    def test_interval_contains_mean_and_is_deterministic(self):
        values = [0.1, 0.4, 0.35, 0.8, 0.5, 0.2]
        low, high = bootstrap_ci(values, n_resamples=500, seed=7)
        self.assertLessEqual(low, np.mean(values))
        self.assertGreaterEqual(high, np.mean(values))
        self.assertEqual((low, high), bootstrap_ci(values, n_resamples=500, seed=7))


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_systems_are_not_different(self):
        scores = [0.2, 0.5, 0.3, 0.4]
        result = paired_bootstrap(scores, scores, n_resamples=200)
        self.assertEqual(result.mean_diff, 0.0)
        self.assertEqual(result.p_value, 1.0)
        self.assertEqual((result.ci_low, result.ci_high), (0.0, 0.0))

    def test_consistently_better_system_is_significant(self):
        rng = np.random.default_rng(0)
        baseline = rng.uniform(0.1, 0.4, size=100)
        better = baseline + 0.05 + rng.normal(0, 0.01, size=100)
        result = paired_bootstrap(better, baseline, n_resamples=2000)
        self.assertAlmostEqual(result.mean_diff, float(np.mean(better - baseline)))
        self.assertLess(result.p_value, 0.01)
        self.assertGreater(result.ci_low, 0.0)
        # Swapping the order flips the sign but keeps the p-value.
        swapped = paired_bootstrap(baseline, better, n_resamples=2000)
        self.assertAlmostEqual(swapped.mean_diff, -result.mean_diff)
        self.assertLess(swapped.ci_high, 0.0)

    def test_noisy_zero_mean_difference_is_not_significant(self):
        rng = np.random.default_rng(1)
        baseline = rng.uniform(0.1, 0.4, size=60)
        noisy = baseline + rng.normal(0, 0.05, size=60)
        noisy -= np.mean(noisy - baseline)  # exact zero mean difference
        self.assertGreater(paired_bootstrap(noisy, baseline, n_resamples=2000).p_value, 0.5)

    def test_rejects_unpaired_input(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([0.1, 0.2], [0.1])
        with self.assertRaises(ValueError):
            paired_bootstrap([], [])


def _prediction_rows(system: str, scores: list[float], offset: int = 0) -> list[dict]:
    return [
        {"system": system, "id": f"a{i + offset}", **{m: s for m in ("rouge1_f1", "rouge2_f1", "rougeL_f1")}}
        for i, s in enumerate(scores)
    ]


class SignificanceModuleTests(unittest.TestCase):
    def test_compare_systems_against_baseline(self):
        base = [0.2, 0.25, 0.3, 0.22, 0.28, 0.24] * 5
        rows = significance.group_rows(
            _prediction_rows("lead3", base)
            + _prediction_rows("vit5", [s + 0.1 for s in base])
            + _prediction_rows("same", base)
        )
        comparisons = significance.compare_systems(rows, "lead3", n_resamples=500)
        by_system = {c["system"]: c for c in comparisons}
        self.assertEqual(set(by_system), {"vit5", "same"})
        self.assertTrue(by_system["vit5"]["metrics"]["rougeL_f1"]["significant"])
        self.assertAlmostEqual(by_system["vit5"]["metrics"]["rouge1_f1"]["mean_diff"], 0.1)
        self.assertFalse(by_system["same"]["metrics"]["rouge1_f1"]["significant"])
        markdown = "\n".join(significance.comparison_markdown(comparisons, 500))
        self.assertIn("## Paired comparison vs `lead3`", markdown)
        self.assertIn("| vit5 | 30 | +10.00*", markdown)

    def test_only_shared_samples_are_compared(self):
        rows = significance.group_rows(
            _prediction_rows("lead3", [0.1, 0.2, 0.3]) + _prediction_rows("x", [0.2, 0.3], offset=1)
        )
        (entry,) = significance.compare_systems(rows, "lead3", n_resamples=100)
        self.assertEqual(entry["samples"], 2)
        self.assertEqual(entry["unmatched_samples"], 1)
        self.assertIn("Warning", "\n".join(significance.comparison_markdown([entry], 100)))

    def test_missing_baseline_raises(self):
        rows = significance.group_rows(_prediction_rows("vit5", [0.1]))
        with self.assertRaises(ValueError):
            significance.compare_systems(rows, "lead3")

    def test_default_baseline_prefers_lead(self):
        self.assertEqual(significance.default_baseline(["vit5", "lead2", "textrank"]), "lead2")
        self.assertEqual(significance.default_baseline(["vit5", "textrank"]), "vit5")

    def test_cli_combines_separate_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"
            first.write_text(
                "\n".join(json.dumps(r) for r in _prediction_rows("lead3", [0.2, 0.3, 0.25, 0.27])),
                encoding="utf-8",
            )
            second.write_text(
                "\n".join(json.dumps(r) for r in _prediction_rows("mt5", [0.3, 0.4, 0.35, 0.37])),
                encoding="utf-8",
            )
            output = Path(tmp) / "sig.md"
            code = significance.main(
                [str(first), str(second), "--bootstrap", "200", "--output", str(output)]
            )
            self.assertEqual(code, 0)
            self.assertIn("| mt5 | 4 |", output.read_text(encoding="utf-8"))


class SentenceSplitTests(unittest.TestCase):
    def test_splits_on_terminal_punctuation_before_uppercase(self):
        text = "Giá xăng tăng 3.5% trong tuần. Người dân lo lắng! Chính phủ có giải pháp không?"
        self.assertEqual(
            split_sentences(text),
            [
                "Giá xăng tăng 3.5% trong tuần.",
                "Người dân lo lắng!",
                "Chính phủ có giải pháp không?",
            ],
        )

    def test_does_not_split_abbreviations_or_lowercase_continuations(self):
        text = "Ông về TP.HCM hôm qua. sau đó ông đi tiếp về phía nam."
        self.assertEqual(len(split_sentences(text)), 1)

    def test_quotes_and_paragraphs(self):
        text = 'Ông nói: "Chúng tôi sẽ làm." Sau đó ông rời đi.\n\nĐoạn thứ hai bắt đầu ở đây.'
        self.assertEqual(
            split_sentences(text),
            ['Ông nói: "Chúng tôi sẽ làm."', "Sau đó ông rời đi.", "Đoạn thứ hai bắt đầu ở đây."],
        )

    def test_short_fragments_are_merged(self):
        self.assertEqual(
            split_sentences("Trận đấu kết thúc với tỷ số hòa. Ảnh: AP"),
            ["Trận đấu kết thúc với tỷ số hòa. Ảnh: AP"],
        )


ARTICLE = (
    "Bão số 3 đổ bộ vào miền Bắc gây mưa lớn. "
    "Mưa lớn do bão số 3 khiến nhiều tuyến đường ở miền Bắc ngập sâu. "
    "Một quán cà phê mới khai trương ở phố cổ. "
    "Chính quyền miền Bắc sơ tán người dân khỏi vùng bão số 3. "
    "Giá vé xem phim cuối tuần không thay đổi."
)


class ExtractiveSummarizerTests(unittest.TestCase):
    def test_lead(self):
        summary = LeadSummarizer(num_sentences=2).summarize(ARTICLE)
        self.assertEqual(summary, " ".join(split_sentences(ARTICLE)[:2]))

    def test_pagerank_is_a_distribution(self):
        weights = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        scores = pagerank(weights)
        self.assertAlmostEqual(scores.sum(), 1.0, places=6)
        self.assertEqual(int(np.argmax(scores)), 0)

    def test_pagerank_handles_isolated_nodes(self):
        scores = pagerank(np.zeros((3, 3)))
        np.testing.assert_allclose(scores, [1 / 3] * 3)

    def test_textrank_prefers_central_sentences_in_document_order(self):
        summary = TextRankSummarizer(num_sentences=3).summarize(ARTICLE)
        sentences = split_sentences(ARTICLE)
        chosen = [s for s in sentences if s in summary]
        self.assertEqual(len(chosen), 3)
        self.assertNotIn("quán cà phê", summary)
        self.assertNotIn("vé xem phim", summary)
        self.assertEqual(summary, " ".join(chosen))  # original order preserved

    def test_short_documents_are_returned_whole(self):
        text = "Chỉ có một câu duy nhất ở đây."
        self.assertEqual(TextRankSummarizer(num_sentences=3).summarize(text), text)

    def test_mmr_avoids_duplicates(self):
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        relevance = np.array([0.9, 0.9, 0.5])
        self.assertEqual(mmr_select(embeddings, relevance, k=2, diversity=0.5), [0, 2])

    def test_phobert_centroid_with_fake_encoder(self):
        def fake_embed(texts):
            lowered = [t.lower() for t in texts]
            # One axis per topic, so off-topic sentences are near-orthogonal to the storm ones.
            return np.array(
                [[t.count("bão"), t.count("phim"), t.count("cà phê")] for t in lowered],
                dtype=float,
            ) + 0.01

        # diversity=0 isolates centroid relevance (MMR is covered by test_mmr_avoids_duplicates).
        summarizer = PhoBERTCentroidSummarizer(
            num_sentences=2, diversity=0.0, embed=fake_embed, segmenter=False
        )
        summary = summarizer.summarize(ARTICLE)
        self.assertNotIn("quán cà phê", summary)
        self.assertNotIn("vé xem phim", summary)
        self.assertEqual(len([s for s in split_sentences(ARTICLE) if s in summary]), 2)
        self.assertFalse(summarizer.describe()["word_segmented"])

    def test_registry(self):
        self.assertEqual(build_summarizer("lead", num_sentences=2).name, "lead2")
        self.assertEqual(build_summarizer("seq2seq:some/model").name, "seq2seq:some/model")
        with self.assertRaises(ValueError):
            build_summarizer("does-not-exist")


SAPO = "Bão số 3 đổ bộ vào miền Bắc, gây mưa lớn và ngập sâu nhiều tuyến đường."
BODY = (
    SAPO + "\n\n"
    "Chiều qua, nước dâng cao khiến giao thông tê liệt. " * 20
    + "\n\nChính quyền đã sơ tán hàng nghìn hộ dân khỏi vùng nguy hiểm. " * 20
)


class DatasetTests(unittest.TestCase):
    def test_strip_reference_removes_copied_lead(self):
        stripped, removed = strip_reference_from_document(BODY, SAPO)
        self.assertEqual(removed, 1)
        self.assertNotIn("Bão số 3", stripped)
        self.assertLess(containment(SAPO, stripped.split("\n\n")[0]), 0.6)

    def test_strip_reference_keeps_unrelated_document(self):
        document = "Đoạn một nói về thể thao.\n\nĐoạn hai nói về kinh tế."
        self.assertEqual(strip_reference_from_document(document, SAPO), (document, 0))

    def _write_jsonl(self, directory: str, name: str, rows: list[dict]) -> Path:
        path = Path(directory) / name
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        return path

    def test_news_loader_strips_leak_and_filters_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_jsonl(
                tmp,
                "news.jsonl",
                [
                    {"id": "a", "sapo": SAPO, "body": BODY, "source": "vnexpress"},
                    {"id": "short", "sapo": SAPO, "body": SAPO + "\n\nNgắn."},
                    {"id": "nosapo", "sapo": "", "body": BODY},
                ],
            )
            examples, stats = load_news_examples(path, limit=None, min_words=50)
        self.assertEqual([e.id for e in examples], ["a"])
        self.assertNotIn(SAPO, examples[0].document)
        self.assertEqual(stats["articles_with_leak"], 2)
        self.assertEqual(stats["skipped_length"], 1)
        self.assertEqual(stats["skipped_missing_fields"], 1)

    def test_jsonl_loader_desegments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_jsonl(
                tmp, "gold.jsonl", [{"id": 1, "document": "Hà_Nội mưa to .", "summary": "Hà_Nội mưa"}]
            )
            examples, _ = load_jsonl_examples(path)
        self.assertEqual(examples[0].document, "Hà Nội mưa to .")
        self.assertEqual(examples[0].reference, "Hà Nội mưa")


class BenchmarkCliTests(unittest.TestCase):
    def test_end_to_end_with_extractive_systems(self):
        rows = [
            {"id": f"d{i}", "document": ARTICLE, "summary": "Bão số 3 gây mưa lớn ở miền Bắc."}
            for i in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "gold.jsonl"
            data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
            out = Path(tmp) / "out"
            code = summarization_benchmark.main(
                [
                    "--dataset", "jsonl",
                    "--input", str(data),
                    "--systems", "lead,textrank,does-not-exist",
                    "--num-sentences", "2",
                    "--bootstrap", "50",
                    "--length-budget", "5",
                    "--significance-resamples", "100",
                    "--output-dir", str(out),
                ]
            )
            self.assertEqual(code, 0)
            results = json.loads((out / "results.json").read_text(encoding="utf-8"))
            summary_md = (out / "summary.md").read_text(encoding="utf-8")
            predictions = (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(results["systems"]["lead"]["status"], "ok")
        self.assertEqual(results["systems"]["lead"]["samples"], 4)
        self.assertEqual(results["systems"]["does-not-exist"]["status"], "error")
        self.assertIn("| lead2 | extractive |", summary_md)
        self.assertIn("| textrank | extractive |", summary_md)
        self.assertEqual(len(predictions), 8)
        low, high = results["systems"]["textrank"]["ci95_rouge1_f1"]
        self.assertLessEqual(low, high)
        self.assertIn("Length-controlled (every output cut to its first 5 words)", summary_md)
        first = json.loads(predictions[0])
        self.assertEqual(len(first["budget_summary"].split()), 5)
        self.assertIn("mean_budget_rougeL_f1", results["systems"]["lead"])
        # Paired significance: textrank vs the default Lead-k baseline on all 4 samples.
        (comparison,) = results["comparisons"]
        self.assertEqual((comparison["system"], comparison["baseline"]), ("textrank", "lead2"))
        self.assertEqual(comparison["samples"], 4)
        self.assertIn("## Paired comparison vs `lead2`", summary_md)

    def test_registry_key_baseline_and_missing_baseline(self):
        rows = [{"id": "d1", "document": ARTICLE, "summary": "Bão số 3 gây mưa lớn."}]
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "gold.jsonl"
            data.write_text(json.dumps(rows[0], ensure_ascii=False), encoding="utf-8")
            common = ["--dataset", "jsonl", "--input", str(data), "--systems", "lead,textrank",
                      "--bootstrap", "10", "--significance-resamples", "10"]
            summarization_benchmark.main(common + ["--baseline", "lead", "--output-dir", f"{tmp}/a"])
            by_key = json.loads(Path(f"{tmp}/a/results.json").read_text(encoding="utf-8"))
            summarization_benchmark.main(common + ["--baseline", "vit5", "--output-dir", f"{tmp}/b"])
            missing = json.loads(Path(f"{tmp}/b/results.json").read_text(encoding="utf-8"))
        self.assertEqual(by_key["comparisons"][0]["baseline"], "lead3")
        self.assertNotIn("comparisons", missing)
        self.assertIn("vit5", missing["comparisons_note"])


if __name__ == "__main__":
    unittest.main()
