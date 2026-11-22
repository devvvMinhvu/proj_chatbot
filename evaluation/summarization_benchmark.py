"""Benchmark summarization systems (TextRank, PhoBERT, ViT5, mT5, ...) on the same data.

Examples:
  python -m evaluation.summarization_benchmark --limit 200 --length-budget 40
  python -m evaluation.summarization_benchmark --systems lead,textrank,phobert --limit 1000
  python -m evaluation.summarization_benchmark --dataset jsonl --input data/vietnews_test.jsonl

Outputs go to reports/summarization_benchmark/<run id>/:
  summary.md        tables ready for the report (ROUGE F1 with 95% bootstrap CI, BLEU, length,
                    speed, and a paired significance test of every system vs the baseline)
  summary.csv       the same numbers, machine-readable
  results.json      full config, dataset statistics, and aggregate metrics per system
  predictions.jsonl every generated summary with its per-sample scores
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

from config import NEWS_JSONL_PATH, PROJECT_ROOT, REPORTS_DIR
from evaluation.datasets import SummarizationExample, load_jsonl_examples, load_news_examples
from evaluation.metrics import F1_METRICS, bootstrap_ci, corpus_bleu, rouge_scores, sentence_bleu
from evaluation.significance import (
    compare_systems,
    comparison_markdown,
    default_baseline,
    group_rows,
)
from processing.fact_check import fact_support
from processing.summarizers import AVAILABLE_SYSTEMS, build_summarizer

DEFAULT_SYSTEMS = "lead,textrank,phobert,vit5_app,vit5"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_system(
    system_name: str,
    examples: list[SummarizationExample],
    num_sentences: int,
    bootstrap_resamples: int,
    seed: int,
    predictions_handle,
    length_budget: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one summarizer over all examples; return (aggregate result, per-sample rows).

    With ``length_budget`` > 0, every summary is additionally scored after being
    cut to its first ``length_budget`` words ("budget_*" metrics). ROUGE F1 is
    sensitive to output length, so this length-controlled score is the fairer
    comparison between systems that produce summaries of different lengths.
    """
    summarizer = build_summarizer(system_name, num_sentences=num_sentences)
    print(f"\n=== {summarizer.name} ({summarizer.kind}) ===", flush=True)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index, example in enumerate(examples, start=1):
            sample_started = time.perf_counter()
            summary = summarizer.summarize(example.document)
            latency = time.perf_counter() - sample_started
            scores = rouge_scores(example.reference, summary)
            scores["bleu"] = sentence_bleu(example.reference, summary)
            if length_budget > 0:
                budget_summary = " ".join(summary.split()[:length_budget])
                budget_scores = rouge_scores(example.reference, budget_summary)
                for key in F1_METRICS:
                    scores[f"budget_{key}"] = budget_scores[key]
                scores["budget_summary"] = budget_summary
            row = {
                "system": summarizer.name,
                "id": example.id,
                "summary": summary,
                "reference": example.reference,
                "summary_words": len(summary.split()),
                "document_words": len(example.document.split()),
                "latency_seconds": latency,
                # Share of the summary's numbers and names found in the article (None: it has none).
                "fact_support": fact_support(summary, example.document),
                **scores,
            }
            rows.append(row)
            predictions_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 25 == 0 or index == len(examples):
                elapsed = time.perf_counter() - started
                print(
                    f"  {index}/{len(examples)} | R-L {_mean([r['rougeL_f1'] for r in rows]):.4f}"
                    f" | {elapsed / index:.2f}s/article",
                    flush=True,
                )
    finally:
        description = summarizer.describe()
        summarizer.close()
    predictions_handle.flush()

    total_seconds = time.perf_counter() - started
    result: dict[str, Any] = {
        "status": "ok",
        "config": description,
        "samples": len(rows),
        "total_seconds": total_seconds,
        "mean_latency_seconds": _mean([r["latency_seconds"] for r in rows]),
        "mean_summary_words": _mean([r["summary_words"] for r in rows]),
        "mean_reference_words": _mean([len(e.reference.split()) for e in examples]),
        "mean_compression": _mean(
            [r["summary_words"] / r["document_words"] for r in rows if r["document_words"]]
        ),
        "corpus_bleu": corpus_bleu([e.reference for e in examples], [r["summary"] for r in rows]),
        "mean_sentence_bleu": _mean([r["bleu"] for r in rows]),
        "mean_fact_support": _mean([r["fact_support"] for r in rows if r["fact_support"] is not None]),
    }
    for metric in ("rouge1", "rouge2", "rougeL"):
        for part in ("p", "r", "f1"):
            key = f"{metric}_{part}"
            result[f"mean_{key}"] = _mean([r[key] for r in rows])
    ci_keys = list(F1_METRICS)
    if length_budget > 0:
        result["length_budget_words"] = length_budget
        result["corpus_bleu_budget"] = corpus_bleu(
            [e.reference for e in examples], [r["budget_summary"] for r in rows]
        )
        for key in F1_METRICS:
            result[f"mean_budget_{key}"] = _mean([r[f"budget_{key}"] for r in rows])
        ci_keys += [f"budget_{key}" for key in F1_METRICS]
    for key in ci_keys:
        low, high = bootstrap_ci([r[key] for r in rows], n_resamples=bootstrap_resamples, seed=seed)
        result[f"ci95_{key}"] = [low, high]
    return result, rows


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_summary_tables(output_dir: Path, report: dict[str, Any]) -> None:
    """Write summary.md and summary.csv from the aggregate report."""
    systems = report["systems"]
    dataset = report["dataset"]
    lines = [
        "# Summarization benchmark",
        "",
        f"- Run: `{report['run_id']}` (commit `{report['git_commit']}`)",
        f"- Dataset: `{dataset['name']}` from `{dataset['path']}`, "
        f"{dataset['stats'].get('sampled', 0)} samples (seed {report['seed']})",
        f"- Reference: {dataset['reference']}",
        "- Metrics: syllable-level ROUGE F1 (mean, 95% bootstrap CI) and corpus BLEU-4, x100",
    ]
    ok_results = {key: r for key, r in systems.items() if r.get("status") == "ok"}
    reference_words = next((r["mean_reference_words"] for r in ok_results.values()), None)
    if reference_words is not None:
        lines.append(f"- Mean reference length: {reference_words:.1f} words")

    def cell(result: dict[str, Any], key: str) -> str:
        low, high = result[f"ci95_{key}"]
        return f"{100 * result[f'mean_{key}']:.2f} ({100 * low:.1f}-{100 * high:.1f})"

    lines += [
        "",
        "## Full-length outputs",
        "",
        "| System | Type | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Facts in source | Words | s/article |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, result in systems.items():
        if result.get("status") != "ok":
            lines.append(f"| {key} | - | error: {result.get('error', '')} | | | | | | |")
            continue
        lines.append(
            f"| {result['config']['name']} | {result['config']['kind']} "
            f"| {cell(result, 'rouge1_f1')} | {cell(result, 'rouge2_f1')} "
            f"| {cell(result, 'rougeL_f1')} | {100 * result['corpus_bleu']:.2f} "
            f"| {100 * result.get('mean_fact_support', 0.0):.1f}% "
            f"| {result['mean_summary_words']:.1f} | {result['mean_latency_seconds']:.3f} |"
        )

    budgeted = {key: r for key, r in ok_results.items() if "length_budget_words" in r}
    if budgeted:
        budget = next(iter(budgeted.values()))["length_budget_words"]
        lines += [
            "",
            f"## Length-controlled (every output cut to its first {budget} words)",
            "",
            "| System | Type | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |",
            "|---|---|---|---|---|---|",
        ]
        for result in budgeted.values():
            lines.append(
                f"| {result['config']['name']} | {result['config']['kind']} "
                f"| {cell(result, 'budget_rouge1_f1')} | {cell(result, 'budget_rouge2_f1')} "
                f"| {cell(result, 'budget_rougeL_f1')} | {100 * result['corpus_bleu_budget']:.2f} |"
            )
    if report.get("comparisons"):
        lines += comparison_markdown(report["comparisons"], report["significance_resamples"])
    elif report.get("comparisons_note"):
        lines += ["", report["comparisons_note"]]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_rows = []
    for result in ok_results.values():
        ci_keys = [k.removeprefix("ci95_") for k in result if k.startswith("ci95_")]
        csv_rows.append(
            {
                "system": result["config"]["name"],
                "kind": result["config"]["kind"],
                "samples": result["samples"],
                **{k: result[k] for k in result if k.startswith("mean_")},
                **{f"{k}_ci_low": result[f"ci95_{k}"][0] for k in ci_keys},
                **{f"{k}_ci_high": result[f"ci95_{k}"][1] for k in ci_keys},
                "corpus_bleu": result["corpus_bleu"],
                "corpus_bleu_budget": result.get("corpus_bleu_budget", ""),
            }
        )

    if csv_rows:
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)


def update_comparisons(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    baseline: str | None,
    n_resamples: int,
    seed: int,
) -> None:
    """Recompute paired significance tests against the baseline over finished systems."""
    rows_by_system = group_rows(rows)
    if len(rows_by_system) < 2:
        return
    if baseline is None:
        resolved = default_baseline(rows_by_system)
    elif baseline in rows_by_system:
        resolved = baseline
    else:
        # Accept the registry key (e.g. "lead") as well as the report name ("lead3").
        result = report["systems"].get(baseline, {})
        resolved = result.get("config", {}).get("name") if result.get("status") == "ok" else None
        if resolved not in rows_by_system:
            report["comparisons_note"] = f"Baseline '{baseline}' has no results yet; comparisons skipped."
            return
    report.pop("comparisons_note", None)
    report["comparisons"] = compare_systems(rows_by_system, resolved, n_resamples=n_resamples, seed=seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("news", "jsonl"), default="news")
    parser.add_argument("--input", type=Path, default=None, help="Dataset path (default: data/news.jsonl).")
    parser.add_argument(
        "--systems",
        default=DEFAULT_SYSTEMS,
        help=f"Comma-separated systems from {', '.join(AVAILABLE_SYSTEMS)} or seq2seq:<model>.",
    )
    parser.add_argument("--limit", type=int, default=200, help="Number of sampled articles (0 = all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-sentences", type=int, default=3, help="Sentences per extractive summary.")
    parser.add_argument("--min-words", type=int, default=150, help="news: minimum body length.")
    parser.add_argument("--max-words", type=int, default=1500, help="news: maximum body length.")
    parser.add_argument(
        "--keep-leak",
        action="store_true",
        help="news: do NOT remove the sapo paragraph from the body (only for showing the leakage effect).",
    )
    parser.add_argument(
        "--length-budget",
        type=int,
        default=0,
        help="Also score every output cut to its first N words (length-controlled ROUGE). 0 = off.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for CIs.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="System every other system is tested against (registry key or report name; default: lead).",
    )
    parser.add_argument(
        "--significance-resamples",
        type=int,
        default=10000,
        help="Resamples for the paired bootstrap significance test.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.dataset == "news":
        path = args.input or NEWS_JSONL_PATH
        examples, stats = load_news_examples(
            path,
            limit=args.limit,
            seed=args.seed,
            min_words=args.min_words,
            max_words=args.max_words,
            strip_leak=not args.keep_leak,
        )
        reference = "article sapo (human-written lead); sapo paragraph removed from input" + (
            " -- DISABLED (--keep-leak)" if args.keep_leak else ""
        )
    else:
        if args.input is None:
            parser.error("--input is required for --dataset jsonl")
        path = args.input
        examples, stats = load_jsonl_examples(path, limit=args.limit, seed=args.seed)
        reference = "gold summary field"

    if not examples:
        print("No examples to evaluate.", file=sys.stderr)
        return 1

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or REPORTS_DIR / "summarization_benchmark" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    systems = [name.strip() for name in args.systems.split(",") if name.strip()]

    print(f"Dataset: {args.dataset} ({path}) | stats: {stats}")
    print(f"Systems: {systems} | output: {output_dir}")

    report: dict[str, Any] = {
        "run_id": run_id,
        "git_commit": _git_commit(),
        "seed": args.seed,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "dataset": {"name": args.dataset, "path": str(path), "reference": reference, "stats": stats},
        "systems": {},
        "significance_resamples": args.significance_resamples,
    }
    all_rows: list[dict[str, Any]] = []
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
        for system in systems:
            try:
                result, rows = evaluate_system(
                    system,
                    examples,
                    args.num_sentences,
                    args.bootstrap,
                    args.seed,
                    predictions,
                    length_budget=args.length_budget,
                )
                report["systems"][system] = result
                all_rows.extend(rows)
            except Exception as exc:  # keep benchmarking the remaining systems
                traceback.print_exc()
                report["systems"][system] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            update_comparisons(report, all_rows, args.baseline, args.significance_resamples, args.seed)
            # Persist after every system so a crash or interruption keeps finished results.
            (output_dir / "results.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            write_summary_tables(output_dir, report)

    print("\n" + (output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
