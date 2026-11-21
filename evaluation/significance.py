"""Paired significance tests between summarization systems.

Used automatically by ``evaluation.summarization_benchmark``. It can also combine
``predictions.jsonl`` files from separate runs, as long as they were made on the
same articles (same dataset, seed, limit, and filters):

  python -m evaluation.significance \
      reports/summarization_benchmark/news200/predictions.jsonl \
      reports/summarization_benchmark/news200_mt5/predictions.jsonl \
      --baseline lead3 --output reports/summarization_benchmark/news200_significance.md

Each system is compared with the baseline on the articles both have scored. No
correction for multiple comparisons is applied; with many systems, read p-values
near 0.05 with care (or apply Bonferroni: divide 0.05 by the number of tests).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

from evaluation.metrics import F1_METRICS, paired_bootstrap

SIGNIFICANCE_LEVEL = 0.05
Rows = dict[str, dict[str, dict[str, Any]]]  # system -> sample id -> prediction row


def group_rows(rows: Iterable[dict[str, Any]]) -> Rows:
    """Index prediction rows by system name and sample id."""
    grouped: Rows = defaultdict(dict)
    for row in rows:
        grouped[str(row["system"])][str(row["id"])] = row
    return dict(grouped)


def load_predictions(paths: Iterable[Path]) -> Rows:
    """Read one or more predictions.jsonl files into a single system/id index."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return group_rows(rows)


def compare_systems(
    rows_by_system: Rows,
    baseline: str,
    metrics: Iterable[str] = F1_METRICS,
    n_resamples: int = 10000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Paired bootstrap of every system against ``baseline`` on their shared samples."""
    if baseline not in rows_by_system:
        raise ValueError(
            f"Baseline '{baseline}' not found. Systems: {', '.join(sorted(rows_by_system))}"
        )
    metrics = tuple(metrics)
    baseline_rows = rows_by_system[baseline]
    comparisons: list[dict[str, Any]] = []
    for system, system_rows in rows_by_system.items():
        if system == baseline:
            continue
        shared = sorted(set(system_rows) & set(baseline_rows))
        if not shared:
            raise ValueError(f"'{system}' and '{baseline}' share no sample ids")
        entry: dict[str, Any] = {
            "system": system,
            "baseline": baseline,
            "samples": len(shared),
            "unmatched_samples": len(set(system_rows) ^ set(baseline_rows)),
            "metrics": {},
        }
        for metric in metrics:
            result = paired_bootstrap(
                [system_rows[i][metric] for i in shared],
                [baseline_rows[i][metric] for i in shared],
                n_resamples=n_resamples,
                seed=seed,
            )
            entry["metrics"][metric] = {
                "mean_diff": result.mean_diff,
                "ci95": [result.ci_low, result.ci_high],
                "p_value": result.p_value,
                "significant": result.p_value < SIGNIFICANCE_LEVEL,
            }
        comparisons.append(entry)
    return comparisons


def comparison_markdown(comparisons: list[dict[str, Any]], n_resamples: int) -> list[str]:
    """Render comparisons as a Markdown section (differences x100, ROUGE points)."""
    if not comparisons:
        return []
    baseline = comparisons[0]["baseline"]
    metrics = list(comparisons[0]["metrics"])
    header_names = {"rouge1_f1": "ROUGE-1", "rouge2_f1": "ROUGE-2", "rougeL_f1": "ROUGE-L"}
    lines = [
        "",
        f"## Paired comparison vs `{baseline}`",
        "",
        f"Difference in F1 points (system - {baseline}), 95% CI, two-sided p from a paired "
        f"bootstrap with {n_resamples} resamples. `*` = p < {SIGNIFICANCE_LEVEL}; "
        "no multiple-comparison correction.",
        "",
        "| System | n | " + " | ".join(header_names.get(m, m) for m in metrics) + " |",
        "|---|---|" + "---|" * len(metrics),
    ]
    for entry in comparisons:
        cells = []
        for metric in metrics:
            values = entry["metrics"][metric]
            low, high = values["ci95"]
            star = "*" if values["significant"] else ""
            cells.append(
                f"{100 * values['mean_diff']:+.2f}{star} ({100 * low:+.1f}, {100 * high:+.1f}; "
                f"p={values['p_value']:.3f})"
            )
        lines.append(f"| {entry['system']} | {entry['samples']} | " + " | ".join(cells) + " |")
    if any(entry["unmatched_samples"] for entry in comparisons):
        lines += [
            "",
            "Warning: some systems were scored on different samples; only shared samples were compared.",
        ]
    return lines


def default_baseline(systems: Iterable[str]) -> str:
    """Prefer a Lead-k baseline; otherwise the first system."""
    systems = list(systems)
    return next((s for s in systems if s.startswith("lead")), systems[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path, nargs="+", help="predictions.jsonl files")
    parser.add_argument("--baseline", default=None, help="System name to compare against (default: lead*).")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output file.")
    args = parser.parse_args(argv)

    rows = load_predictions(args.predictions)
    baseline = args.baseline or default_baseline(rows)
    comparisons = compare_systems(rows, baseline, n_resamples=args.bootstrap, seed=args.seed)
    text = "\n".join(comparison_markdown(comparisons, args.bootstrap)).strip() + "\n"
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
