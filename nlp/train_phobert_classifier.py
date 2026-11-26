"""Fine-tune PhoBERT to classify news articles by category, and compare it with TF-IDF.

The split is the same as ``nlp.train_text_classifier`` (stratified 80/20,
random_state 42, over ``data/news.jsonl``), so both models are scored on the
same test articles. 10% of the training part is held out for choosing the best
epoch; the test part is used once, at the end.

Input text: title, sapo and body (with the sapo paragraph that the crawler
repeats at the start of the body removed), word-segmented with pyvi as PhoBERT
expects, truncated to ``--max-length`` subword tokens.

On a 4 GB GPU the defaults freeze the word-embedding table and use gradient
checkpointing with mixed precision; disable them on a larger GPU.

    python -m nlp.train_phobert_classifier                 # train, evaluate, save
    python -m nlp.train_phobert_classifier --epochs 4 --batch-size 16 --no-freeze-embeddings

Outputs:
    models/phobert_topic_classifier/    model, tokenizer, label map, metrics
    reports/topic_classifier/<run>/     results.json, summary.md, predictions.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from config import NEWS_JSONL_PATH, PHOBERT_MODEL_DIR, PHOBERT_TOPIC_MODEL_DIR, REPORTS_DIR
from nlp.text_classifier import PHOBERT_MAX_LENGTH, phobert_input_text
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SHARE = 0.1


@dataclass
class Article:
    title: str
    sapo: str
    body: str
    category: str


def load_articles(path: Path = NEWS_JSONL_PATH) -> list[Article]:
    """Articles in the same order and with the same filter as the TF-IDF trainer."""
    articles: list[Article] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            category = (item.get("category") or "").strip()
            title = (item.get("title") or "").strip()
            sapo = (item.get("sapo") or "").strip()
            body = (item.get("body") or "").strip()
            if not category or not (title or sapo or body):
                continue
            articles.append(Article(title, sapo, body, category))
    return articles


def split_indices(labels: list[str]) -> tuple[list[int], list[int], list[int]]:
    """(train, validation, test) indices; test matches nlp.train_text_classifier exactly."""
    from sklearn.model_selection import train_test_split

    indices = list(range(len(labels)))
    train_full, test = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    train, validation = train_test_split(
        train_full,
        test_size=VALIDATION_SHARE,
        random_state=RANDOM_STATE,
        stratify=[labels[i] for i in train_full],
    )
    return train, validation, test


def model_text(article: Article) -> str:
    """The same input text as used at prediction time (nlp.text_classifier)."""
    return phobert_input_text(article.title, article.sapo, article.body)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pad(encodings: dict[str, list[list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    width = max(len(ids) for ids in encodings["input_ids"])
    return {
        "input_ids": [ids + [pad_id] * (width - len(ids)) for ids in encodings["input_ids"]],
        "attention_mask": [mask + [0] * (width - len(mask)) for mask in encodings["attention_mask"]],
    }


def predict_logits(model, tokenizer, texts: list[str], max_length: int, batch_size: int, device) -> np.ndarray:
    import torch

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**encoded).logits
            outputs.append(logits.float().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, model.config.num_labels))


def scores(gold: list[int], predicted: list[int], label_names: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    return {
        "accuracy": accuracy_score(gold, predicted),
        "macro_f1": f1_score(gold, predicted, average="macro"),
        "report": classification_report(
            gold, predicted, labels=list(range(len(label_names))), target_names=label_names,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(gold, predicted, labels=list(range(len(label_names)))).tolist(),
    }


def mcnemar(gold: list[int], first: list[int], second: list[int]) -> dict[str, Any]:
    """Exact McNemar test on the articles where exactly one of two classifiers is right."""
    from scipy.stats import binomtest

    only_first = sum(1 for g, a, b in zip(gold, first, second) if a == g and b != g)
    only_second = sum(1 for g, a, b in zip(gold, first, second) if b == g and a != g)
    discordant = only_first + only_second
    p_value = binomtest(only_first, discordant, 0.5).pvalue if discordant else 1.0
    return {"only_first_correct": only_first, "only_second_correct": only_second, "p_value": p_value}


def tfidf_baseline(articles: list[Article], train: list[int], test: list[int], label_ids: dict[str, int]) -> list[int]:
    """The current TF-IDF + logistic regression model, trained on the same training part."""
    from nlp.text_classifier import format_article_text
    from nlp.train_text_classifier import create_classifier

    def text(i: int) -> str:
        a = articles[i]
        return format_article_text(title=a.title, sapo=a.sapo, body=a.body)

    train_ids = sorted(set(train))
    model = create_classifier()
    model.fit([text(i) for i in train_ids], [articles[i].category for i in train_ids])
    return [label_ids[label] for label in model.predict([text(i) for i in test])]


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    articles = load_articles(args.data)
    if args.limit:
        # Smoke runs only: a stratified-enough slice of the data.
        articles = articles[:: max(1, len(articles) // args.limit)][: args.limit]
    label_names = sorted({a.category for a in articles})
    label_ids = {name: i for i, name in enumerate(label_names)}
    labels = [label_ids[a.category] for a in articles]
    train_idx, val_idx, test_idx = split_indices([a.category for a in articles])
    print(f"Articles: {len(articles)} | train {len(train_idx)}, validation {len(val_idx)}, test {len(test_idx)}")
    print(f"Labels: {label_names} | device: {device}")

    started = time.perf_counter()
    print("Segmenting text with pyvi ...", flush=True)
    texts = [model_text(a) for a in articles]
    print(f"  done in {time.perf_counter() - started:.0f}s", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), use_fast=False)
    encoded = tokenizer(texts, truncation=True, max_length=args.max_length)
    encodings = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    output_dir: Path = args.output
    if args.evaluate_only:
        # Score an already trained model; its training history is in its metrics.json.
        best: dict[str, Any] = {"macro_f1": None, "epoch": None}
        history: list[dict[str, Any]] = []
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(args.base_model),
            num_labels=len(label_names),
            id2label=dict(enumerate(label_names)),
            label2id=label_ids,
        )
        if args.freeze_embeddings:
            for parameter in model.roberta.embeddings.word_embeddings.parameters():
                parameter.requires_grad = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.to(device)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
        steps_per_epoch = math.ceil(len(train_idx) / (args.batch_size * args.accumulation))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(0.1 * steps_per_epoch * args.epochs), steps_per_epoch * args.epochs
        )
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

        best = {"macro_f1": -1.0, "epoch": 0}
        history = []
        rng = random.Random(args.seed)
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = list(train_idx)
            rng.shuffle(order)
            # Similar lengths in one batch waste less padding; batches stay shuffled.
            order = sorted(order, key=lambda i: len(encodings["input_ids"][i]))
            chunks = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
            rng.shuffle(chunks)
            epoch_started = time.perf_counter()
            running = 0.0
            optimizer.zero_grad()
            for step, chosen in enumerate(chunks, start=1):
                batch = _pad({k: [encodings[k][i] for i in chosen] for k in encodings}, tokenizer.pad_token_id)
                inputs = {k: torch.tensor(v, device=device) for k, v in batch.items()}
                targets = torch.tensor([labels[i] for i in chosen], device=device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = model(**inputs, labels=targets).loss / args.accumulation
                scaler.scale(loss).backward()
                running += loss.item() * args.accumulation
                if step % args.accumulation == 0 or step == len(chunks):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                if step % 200 == 0:
                    print(f"  epoch {epoch} step {step}/{len(chunks)} loss {running / step:.4f}", flush=True)

            val_logits = predict_logits(model, tokenizer, [texts[i] for i in val_idx], args.max_length, args.eval_batch_size, device)
            val_pred = val_logits.argmax(axis=1).tolist()
            val_scores = scores([labels[i] for i in val_idx], val_pred, label_names)
            record = {
                "epoch": epoch,
                "train_loss": running / max(len(chunks), 1),
                "val_accuracy": val_scores["accuracy"],
                "val_macro_f1": val_scores["macro_f1"],
                "seconds": time.perf_counter() - epoch_started,
            }
            history.append(record)
            print(
                f"Epoch {epoch}: loss {record['train_loss']:.4f} | val acc {record['val_accuracy']:.4f} "
                f"| val macro-F1 {record['val_macro_f1']:.4f} | {record['seconds']:.0f}s",
                flush=True,
            )
            if val_scores["macro_f1"] > best["macro_f1"]:
                best = {"macro_f1": val_scores["macro_f1"], "epoch": epoch}
                output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

    if best["epoch"] is not None:
        print(f"Best epoch: {best['epoch']} (validation macro-F1 {best['macro_f1']:.4f})")
    print("Evaluating on the test set ...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(output_dir)).to(device)
    test_texts = [texts[i] for i in test_idx]
    gold = [labels[i] for i in test_idx]
    inference_started = time.perf_counter()
    test_logits = predict_logits(model, tokenizer, test_texts, args.max_length, args.eval_batch_size, device)
    ms_per_article = 1000 * (time.perf_counter() - inference_started) / max(len(test_idx), 1)
    phobert_pred = test_logits.argmax(axis=1).tolist()

    print("Training the TF-IDF baseline on the same training part ...", flush=True)
    tfidf_pred = tfidf_baseline(articles, train_idx + val_idx, test_idx, label_ids)

    results = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "data": str(args.data),
        "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx), "random_state": RANDOM_STATE},
        "labels": label_names,
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "best_epoch": best["epoch"],
        "history": history,
        "phobert": {**scores(gold, phobert_pred, label_names), "ms_per_article": ms_per_article},
        "tfidf_logreg": scores(gold, tfidf_pred, label_names),
        "mcnemar_phobert_vs_tfidf": mcnemar(gold, phobert_pred, tfidf_pred),
    }
    metrics_path = output_dir / "metrics.json"
    if not args.evaluate_only or not metrics_path.exists():
        # An evaluate-only run keeps the training run's metrics (with its epoch history).
        metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(results, args, articles, test_idx, gold, phobert_pred, tfidf_pred, label_names)
    return results


def write_report(results, args, articles, test_idx, gold, phobert_pred, tfidf_pred, label_names) -> Path:
    run_dir = args.report_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["title", "gold", "phobert", "tfidf"])
        for i, g, p, t in zip(test_idx, gold, phobert_pred, tfidf_pred):
            writer.writerow([articles[i].title, label_names[g], label_names[p], label_names[t]])

    phobert, tfidf, test = results["phobert"], results["tfidf_logreg"], results["mcnemar_phobert_vs_tfidf"]
    lines = [
        "# Topic classification: PhoBERT vs TF-IDF",
        "",
        f"- Test set: {len(gold)} articles (stratified 20%, random_state {RANDOM_STATE}), same for both models",
        f"- PhoBERT: best epoch {results['best_epoch'] or 'see metrics.json of the training run'}, "
        f"max {args.max_length} tokens, "
        f"{phobert['ms_per_article']:.1f} ms/article",
        "",
        "| Model | Accuracy | Macro-F1 |",
        "|---|---|---|",
        f"| TF-IDF + logistic regression | {tfidf['accuracy']:.4f} | {tfidf['macro_f1']:.4f} |",
        f"| PhoBERT (fine-tuned) | {phobert['accuracy']:.4f} | {phobert['macro_f1']:.4f} |",
        "",
        f"McNemar (exact): only PhoBERT right {test['only_first_correct']}, only TF-IDF right "
        f"{test['only_second_correct']}, p = {test['p_value']:.4g}",
        "",
        "## Per-category F1",
        "",
        "| Category | TF-IDF | PhoBERT |",
        "|---|---|---|",
    ]
    for name in label_names:
        lines.append(f"| {name} | {tfidf['report'][name]['f1-score']:.4f} | {phobert['report'][name]['f1-score']:.4f} |")
    lines += ["", "## PhoBERT confusion matrix (rows: gold, columns: predicted)", "",
              "| | " + " | ".join(label_names) + " |", "|---" * (len(label_names) + 1) + "|"]
    for name, row in zip(label_names, phobert["confusion_matrix"]):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport: {run_dir}")
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT for news topic classification.")
    parser.add_argument("--data", type=Path, default=NEWS_JSONL_PATH)
    parser.add_argument("--base-model", type=Path, default=PHOBERT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=PHOBERT_TOPIC_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR / "topic_classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2, help="gradient accumulation steps")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=PHOBERT_MAX_LENGTH, help="PhoBERT accepts at most 256 tokens")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--limit", type=int, default=0, help="use only N articles (smoke test)")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="skip training; score the saved model in --output and the TF-IDF baseline on the test set",
    )
    parser.add_argument("--freeze-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
