
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from processing.chunking import split_text


# =========================================
# 1. CAU HINH MO HINH
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "vit5_summarization"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    use_fast=False
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    str(MODEL_PATH)
)

model.to(DEVICE)
model.eval()

print(f"[SUMMARIZER] Device: {DEVICE}")


# =========================================
# 2. TOM TAT MOT DOAN VAN BAN
# =========================================

def summarize_chunk(text: str) -> str:
    """Tóm tắt một đoạn văn bản bằng mô hình ViT5."""

    if not text or not text.strip():
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Chuyển dữ liệu đầu vào sang cùng thiết bị với mô hình.
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            min_new_tokens=20,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary.strip()


# =========================================
# 3. TOM TAT TOAN BO BAI BAO
# =========================================

def summarize(text: str) -> str:
    """Tóm tắt bài báo, kể cả khi bài báo được chia thành nhiều đoạn."""

    if not text or not text.strip():
        return ""

    chunks = split_text(text)

    if not chunks:
        return ""

    summaries = []

    for chunk in chunks:
        summary = summarize_chunk(chunk)

        if summary:
            summaries.append(summary)

    final_summary = " ".join(summaries)

    return final_summary.strip()
