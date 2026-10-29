"""Evaluate the current news-assistant models in proj_chatbot.

Expected layout (run this file from the project root or place it in the root):
  data/intent_dataset.csv
  data/intent_classifier.joblib
  data/news_15k.jsonl
  data/text_classifier_15k.joblib
  data/vector_db_15k/news.index
  data/vector_db_15k/chunk_metadata.json
  models/vit5_summarization/

The script evaluates classifiers on a fresh stratified holdout by cloning the
saved sklearn pipeline and fitting only on the training split. It does NOT
score a saved model against data it has already seen.

Retrieval and RAG are reported as operational smoke tests until a labelled
query/relevance/answer evaluation set is available. ViT5 is compared with the
article sapo as a proxy reference; sapo is not guaranteed to be a gold summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
COMPONENT_REPORT_DIR = REPORT_DIR / ".model_eval_parts"

COMPONENTS = ["intent", "category", "retrieval", "rag", "summarization"]


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def print_header(title: str) -> None:
    print("\n" + "=" * 76, flush=True)
    print(title, flush=True)
    print("=" * 76, flush=True)


def safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Bỏ qua JSONL lỗi ở dòng {line_no}: {exc}")
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def evaluate_classifier(
    *,
    name: str,
    data_texts: list[str],
    labels: list[str],
    model_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate a clone of a saved sklearn estimator on a stratified holdout."""
    import joblib
    from sklearn.base import clone
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.model_selection import train_test_split

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {model_path}")
    if len(data_texts) != len(labels):
        raise ValueError("Số văn bản và số nhãn không khớp.")
    if len(data_texts) < 2 or len(set(labels)) < 2:
        raise ValueError("Không đủ dữ liệu hoặc số nhãn để đánh giá.")

    counts = Counter(labels)
    if min(counts.values()) < 2:
        raise ValueError("Mỗi nhãn cần ít nhất 2 mẫu để chia tập phân tầng.")

    saved_model = joblib.load(model_path)
    if not hasattr(saved_model, "get_params"):
        raise TypeError(
            "Artifact không phải sklearn estimator có get_params(); "
            "không thể tạo bản sao để đánh giá holdout công bằng."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        data_texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model = clone(saved_model)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    class_labels = sorted(set(labels))

    report = classification_report(
        y_test,
        predictions,
        labels=class_labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=class_labels)
    result = {
        "name": name,
        "status": "ok",
        "model_path": str(model_path),
        "total_samples": len(data_texts),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_counts": dict(sorted(counts.items())),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            label: {
                metric: float(report[label][metric])
                for metric in ("precision", "recall", "f1-score", "support")
            }
            for label in class_labels
        },
        "confusion_matrix": {
            "labels": class_labels,
            "matrix": matrix.tolist(),
        },
        "test_predictions": [
            {
                "text": str(text),
                "actual": str(actual),
                "predicted": str(predicted),
                "correct": bool(actual == predicted),
            }
            for text, actual, predicted in zip(X_test, y_test, predictions)
        ],
        "evaluation_note": (
            "A fresh stratified holdout was evaluated using a clone of the saved estimator. "
            "Small datasets can produce unstable metrics."
        ),
    }

    print(f"Mô hình: {name}")
    print(f"Mẫu: {len(data_texts)} | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Accuracy:       {result['accuracy']:.4f}")
    print(f"Macro precision:{result['macro_precision']:.4f}")
    print(f"Macro recall:   {result['macro_recall']:.4f}")
    print(f"Macro F1:       {result['macro_f1']:.4f}")
    print(f"Weighted F1:    {result['weighted_f1']:.4f}")
    print("F1 theo nhãn:")
    for label, values in result["per_class"].items():
        print(f"  {label:24s} P={values['precision']:.3f} "
              f"R={values['recall']:.3f} F1={values['f1-score']:.3f} "
              f"(n={int(values['support'])})")
    print("Các lỗi trong tập kiểm thử:")
    mistakes = [row for row in result["test_predictions"] if not row["correct"]]
    if not mistakes:
        print("  Không có lỗi trên tập kiểm thử này.")
    else:
        for row in mistakes[:20]:
            print(f"  Thật={row['actual']} | Dự đoán={row['predicted']} | {row['text']}")
    return result


def component_intent() -> dict[str, Any]:
    print_header("1/5 - PHÂN LOẠI Ý ĐỊNH: TF-IDF + MULTINOMIAL NAIVE BAYES")
    dataset_path = DATA_DIR / "intent_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {dataset_path}")

    import pandas as pd

    df = pd.read_csv(dataset_path, encoding="utf-8")
    if not {"text", "intent"}.issubset(df.columns):
        raise ValueError("intent_dataset.csv phải có cột text,intent")
    df = df.dropna(subset=["text", "intent"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["intent"] != "")]
    df = df.drop_duplicates(subset=["text", "intent"])

    return evaluate_classifier(
        name="intent_classifier (TF-IDF + MultinomialNB)",
        data_texts=df["text"].tolist(),
        labels=df["intent"].tolist(),
        model_path=DATA_DIR / "intent_classifier.joblib",
    )


def component_category() -> dict[str, Any]:
    print_header("2/5 - PHÂN LOẠI CHỦ ĐỀ BÀI BÁO: TF-IDF + LOGISTIC REGRESSION")
    dataset_path = DATA_DIR / "news_15k.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {dataset_path}")

    rows = read_jsonl(dataset_path)
    texts: list[str] = []
    labels: list[str] = []
    for row in rows:
        label = safe_text(row.get("category"))
        # Match the expected text input as title + sapo + body.
        text = " ".join(
            part for part in [
                safe_text(row.get("title")),
                safe_text(row.get("sapo")),
                safe_text(row.get("body")),
            ] if part
        )
        if label and text:
            texts.append(text)
            labels.append(label)

    if not texts:
        raise ValueError("Không có bản ghi đủ title/sapo/body và category.")

    return evaluate_classifier(
        name="text_classifier_15k (topic classification)",
        data_texts=texts,
        labels=labels,
        model_path=DATA_DIR / "text_classifier_15k.joblib",
    )


def component_retrieval() -> dict[str, Any]:
    print_header("3/5 - TÌM KIẾM NGỮ NGHĨA: PHOBERT + FAISS")
    from rag.pipeline_15k import load_vector_database, retrieve
    import inspect

    start = time.perf_counter()
    index, metadata = load_vector_database()
    load_seconds = time.perf_counter() - start
    if index is None:
        raise RuntimeError("load_vector_database() trả về index rỗng.")

    queries = [
        "Dự án vệ sinh học đường tại Sơn La đã xây mới bao nhiêu công trình?",
        "Các bài báo về trí tuệ nhân tạo và công nghệ",
        "Hoạt động bảo vệ môi trường tại Việt Nam",
    ]
    rows = []
    signature = inspect.signature(retrieve)
    supported_kwargs: dict[str, Any] = {}
    if "candidate_k" in signature.parameters:
        supported_kwargs["candidate_k"] = 50
    if "top_k" in signature.parameters:
        supported_kwargs["top_k"] = 5
    if "max_chunks_per_article" in signature.parameters:
        supported_kwargs["max_chunks_per_article"] = 2

    for query in queries:
        started = time.perf_counter()
        output = retrieve(query, **supported_kwargs)
        elapsed = time.perf_counter() - started
        if isinstance(output, dict):
            results = output.get("results", output.get("chunks", output.get("articles", [])))
        elif isinstance(output, tuple):
            results = output[0] if output else []
        else:
            results = output
        try:
            count = len(results) if results is not None else 0
        except TypeError:
            count = 0
        nonempty = count > 0
        rows.append({
            "query": query,
            "result_count": count,
            "latency_seconds": round(elapsed, 4),
            "returned_results": nonempty,
        })
        print(f"Truy vấn: {query}")
        print(f"  Kết quả: {count} | Thời gian: {elapsed:.3f}s")

    result = {
        "name": "PhoBERT embeddings + FAISS retrieval",
        "status": "ok",
        "index_vectors": int(getattr(index, "ntotal", 0)),
        "metadata_count": len(metadata) if metadata is not None else None,
        "index_load_seconds": round(load_seconds, 4),
        "queries": rows,
        "nonempty_query_rate": sum(r["returned_results"] for r in rows) / len(rows),
        "evaluation_note": (
            "Operational smoke test only. Precision@k, Recall@k, MRR and nDCG require "
            "a labelled query-to-relevant-article dataset."
        ),
    }
    print(f"Vector: {result['index_vectors']:,} | Metadata: {result['metadata_count']:,}")
    print("Tỷ lệ truy vấn trả về kết quả:", f"{result['nonempty_query_rate']:.1%}")
    print("Lưu ý: chưa thể tính Precision@k/MRR nếu chưa có nhãn liên quan đúng.")
    return result


def component_rag() -> dict[str, Any]:
    print_header("4/5 - RAG + QWEN2.5-1.5B: KIỂM THỬ VẬN HÀNH")
    from rag.pipeline_15k import ask

    query = "Dự án vệ sinh học đường tại Sơn La đã xây mới bao nhiêu công trình?"
    started = time.perf_counter()
    output = ask(query)
    elapsed = time.perf_counter() - started

    if not isinstance(output, dict):
        raise TypeError("rag.pipeline_15k.ask() không trả về dict như dự kiến.")

    answer = safe_text(output.get("answer"))
    sources = output.get("sources") or []
    source_count = len(sources) if hasattr(sources, "__len__") else 0
    answer_has_expected_number = bool(re.search(r"\b11\b", answer))
    result = {
        "name": "RAG retrieval + Qwen2.5-1.5B generation",
        "status": "ok" if answer else "warning",
        "query": query,
        "latency_seconds": round(elapsed, 4),
        "answer_length_chars": len(answer),
        "source_count": source_count,
        "answer_preview": answer[:1200],
        "expected_fact_11_present": answer_has_expected_number,
        "evaluation_note": (
            "Single-question smoke test. The check for '11' is a project-specific heuristic, "
            "not a general answer-quality metric. For rigorous evaluation, create labelled QA "
            "items with expected answers and source documents."
        ),
    }
    print(f"Thời gian: {elapsed:.2f}s")
    print(f"Số nguồn: {source_count}")
    print(f"Có số liệu kiểm tra '11': {answer_has_expected_number}")
    print("Câu trả lời:")
    print(answer[:1200] or "(Rỗng)")
    return result


def tokenize_for_rouge(text: str) -> list[str]:
    # Vietnamese words are separated largely by spaces. This is a simple
    # whitespace-like tokenizer for a consistent proxy evaluation.
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def ngram_f1(reference: list[str], hypothesis: list[str], n: int) -> float:
    if not reference or not hypothesis:
        return 0.0
    ref_grams = Counter(tuple(reference[i:i + n]) for i in range(len(reference) - n + 1))
    hyp_grams = Counter(tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) - n + 1))
    overlap = sum((ref_grams & hyp_grams).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(hyp_grams.values()), 1)
    recall = overlap / max(sum(ref_grams.values()), 1)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def lcs_length(a: list[str], b: list[str]) -> int:
    # Memory-efficient LCS for ROUGE-L.
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


def rouge_l_f1(reference: list[str], hypothesis: list[str]) -> float:
    if not reference or not hypothesis:
        return 0.0
    common = lcs_length(reference, hypothesis)
    precision = common / len(hypothesis)
    recall = common / len(reference)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def component_summarization(sample_count: int) -> dict[str, Any]:
    print_header("5/5 - TÓM TẮT VĂN BẢN: VIT5 (ROUGE SO VỚI SAPO LÀM MỐC THAM KHẢO)")
    from processing.summerizer import summarize

    rows = read_jsonl(DATA_DIR / "news_15k.jsonl")
    eligible = [
        row for row in rows
        if safe_text(row.get("body"))
        and safe_text(row.get("sapo"))
        and 500 <= len(safe_text(row.get("body"))) <= 5000
    ]
    if not eligible:
        raise ValueError("Không tìm thấy bài báo có body và sapo phù hợp để đánh giá.")

    random.Random(42).shuffle(eligible)
    samples = eligible[:max(1, min(sample_count, len(eligible)))]
    metrics: list[dict[str, Any]] = []

    for i, row in enumerate(samples, start=1):
        title = safe_text(row.get("title")) or f"Bài báo {i}"
        body = safe_text(row.get("body"))
        reference_text = safe_text(row.get("sapo"))
        started = time.perf_counter()
        summary = summarize(body)
        elapsed = time.perf_counter() - started
        ref_tokens = tokenize_for_rouge(reference_text)
        hyp_tokens = tokenize_for_rouge(summary)
        scores = {
            "rouge1_f1": ngram_f1(ref_tokens, hyp_tokens, 1),
            "rouge2_f1": ngram_f1(ref_tokens, hyp_tokens, 2),
            "rougeL_f1": rouge_l_f1(ref_tokens, hyp_tokens),
        }
        metrics.append({
            "title": title,
            "reference_chars": len(reference_text),
            "summary_chars": len(summary),
            "latency_seconds": round(elapsed, 4),
            **scores,
            "summary_preview": summary[:500],
        })
        print(f"[{i}/{len(samples)}] {title}")
        print(f"  thời gian={elapsed:.2f}s | độ dài={len(summary)} ký tự | "
              f"R1={scores['rouge1_f1']:.3f} R2={scores['rouge2_f1']:.3f} "
              f"RL={scores['rougeL_f1']:.3f}")

    result = {
        "name": "ViT5 summarization",
        "status": "ok",
        "samples_evaluated": len(metrics),
        "reference_field": "sapo",
        "mean_latency_seconds": sum(m["latency_seconds"] for m in metrics) / len(metrics),
        "mean_rouge1_f1": sum(m["rouge1_f1"] for m in metrics) / len(metrics),
        "mean_rouge2_f1": sum(m["rouge2_f1"] for m in metrics) / len(metrics),
        "mean_rougeL_f1": sum(m["rougeL_f1"] for m in metrics) / len(metrics),
        "per_article": metrics,
        "evaluation_note": (
            "Sapo is used as a proxy reference, not a verified gold summary. ROUGE overlap is "
            "useful for comparison but does not fully measure factuality or readability."
        ),
    }
    print("\nTRUNG BÌNH:")
    print(f"  ROUGE-1 F1: {result['mean_rouge1_f1']:.4f}")
    print(f"  ROUGE-2 F1: {result['mean_rouge2_f1']:.4f}")
    print(f"  ROUGE-L F1: {result['mean_rougeL_f1']:.4f}")
    print(f"  Thời gian/bài: {result['mean_latency_seconds']:.2f}s")
    print("Lưu ý: sapo chỉ là mốc tham khảo, không chắc chắn là bản tóm tắt chuẩn.")
    return result


def run_component(component: str, summarization_samples: int) -> dict[str, Any]:
    functions = {
        "intent": component_intent,
        "category": component_category,
        "retrieval": component_retrieval,
        "rag": component_rag,
        "summarization": lambda: component_summarization(summarization_samples),
    }
    report_path = COMPONENT_REPORT_DIR / f"{component}.json"
    try:
        result = functions[component]()
        result["component"] = component
        result["status"] = result.get("status", "ok")
    except Exception as exc:
        traceback.print_exc()
        result = {
            "component": component,
            "name": component,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(f"\nLỖI khi đánh giá {component}: {type(exc).__name__}: {exc}", flush=True)
    write_json(report_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Đánh giá các mô hình đang có trong dự án trợ lý đọc báo."
    )
    parser.add_argument(
        "--component",
        choices=COMPONENTS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--summarization-samples",
        type=int,
        default=5,
        help="Số bài báo dùng thử cho ViT5 (mặc định: 5).",
    )
    parser.add_argument(
        "--only",
        choices=COMPONENTS,
        help="Chỉ đánh giá một thành phần để kiểm thử riêng.",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    COMPONENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if args.component:
        run_component(args.component, args.summarization_samples)
        return 0

    selected = [args.only] if args.only else COMPONENTS
    print_header("ĐÁNH GIÁ TỔNG HỢP CÁC MÔ HÌNH - TRỢ LÝ ĐỌC BÁO")
    print(f"Thư mục dự án: {ROOT}")
    print("Mỗi thành phần chạy trong tiến trình riêng để hạn chế xung đột bộ nhớ GPU.")
    print(f"Số bài tóm tắt ViT5: {args.summarization_samples}")

    for component in selected:
        print_header(f"BẮT ĐẦU THÀNH PHẦN: {component}")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--component",
            component,
            "--summarization-samples",
            str(args.summarization_samples),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if completed.returncode != 0:
            print(f"Tiến trình {component} kết thúc với mã {completed.returncode}.")

    results: dict[str, Any] = {}
    for component in selected:
        part_path = COMPONENT_REPORT_DIR / f"{component}.json"
        if part_path.exists():
            try:
                results[component] = json.loads(part_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                results[component] = {"status": "error", "error": "Không đọc được báo cáo con."}
        else:
            results[component] = {"status": "error", "error": "Không có báo cáo thành phần."}

    final_report = {
        "project": "Trợ lý đọc báo thông minh",
        "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(ROOT),
        "components": results,
        "global_notes": [
            "Intent/topic classification uses a fresh stratified holdout and a cloned saved sklearn estimator.",
            "Retrieval and RAG numbers are smoke tests unless a labelled relevance/QA dataset is added.",
            "Summarization ROUGE uses sapo as a proxy reference, not guaranteed ground truth.",
        ],
    }
    final_path = REPORT_DIR / "model_evaluation_report.json"
    write_json(final_path, final_report)

    print_header("TỔNG KẾT")
    for component, report in results.items():
        status = report.get("status", "unknown")
        if component in ("intent", "category") and status == "ok":
            print(f"{component:16s} status={status:8s} accuracy={report.get('accuracy', 0):.4f} "
                  f"macro-F1={report.get('macro_f1', 0):.4f}")
        elif component == "summarization" and status == "ok":
            print(f"{component:16s} status={status:8s} ROUGE-L={report.get('mean_rougeL_f1', 0):.4f} "
                  f"samples={report.get('samples_evaluated', 0)}")
        elif component == "retrieval" and status == "ok":
            print(f"{component:16s} status={status:8s} nonempty={report.get('nonempty_query_rate', 0):.1%}")
        elif component == "rag" and status == "ok":
            print(f"{component:16s} status={status:8s} sources={report.get('source_count', 0)} "
                  f"time={report.get('latency_seconds', 0):.2f}s")
        else:
            print(f"{component:16s} status={status}: {report.get('error', '')}")

    print(f"\nĐã ghi báo cáo JSON: {final_path}")
    print("Chú ý: kết quả của từng mô hình có thể bị lỗi độc lập; hãy xem log và file JSON để biết chi tiết.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
