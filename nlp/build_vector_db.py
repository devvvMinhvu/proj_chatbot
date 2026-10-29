
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from database.database_get import SessionLocal, Article

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "phobert"
OUTPUT_DIR = PROJECT_ROOT / "data" / "vector_db_v2"
INDEX_PATH = OUTPUT_DIR / "news.index"
METADATA_PATH = OUTPUT_DIR / "chunk_metadata.json"

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
BATCH_SIZE = 16
MAX_TOKEN_LENGTH = 256


def split_into_chunks(text: str, chunk_size: int = CHUNK_CHARS,
                      overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text by character length, preferring sentence boundaries."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size/overlap configuration")
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer ending a chunk after punctuation, but don't create tiny chunks.
            search_from = start + int(chunk_size * 0.65)
            candidates = [
                text.rfind(". ", search_from, end),
                text.rfind("! ", search_from, end),
                text.rfind("? ", search_from, end),
                text.rfind("; ", search_from, end),
            ]
            boundary = max(candidates)
            if boundary >= search_from:
                end = boundary + 1  # include the punctuation character

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        start = next_start

    return chunks


def mean_pool(last_hidden_state: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts: list[str], tokenizer: Any, model: Any,
                 device: torch.device) -> np.ndarray:
    """Create normalized float32 embeddings using PhoBERT mean pooling."""
    all_vectors: list[np.ndarray] = []
    model.eval()

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            output = model(**encoded)
            vectors = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            vectors = F.normalize(vectors, p=2, dim=1)

        all_vectors.append(vectors.cpu().numpy().astype("float32"))
        processed = min(start + len(batch), len(texts))
        if processed == len(texts) or processed % (BATCH_SIZE * 20) == 0:
            print(f"Đã embedding: {processed}/{len(texts)} chunks")

    return np.concatenate(all_vectors, axis=0)


def build_records() -> tuple[list[dict[str, Any]], list[str], int, int]:
    db = SessionLocal()
    records: list[dict[str, Any]] = []
    embedding_texts: list[str] = []
    article_count = 0
    skipped_count = 0

    try:
        articles = db.query(Article).order_by(Article.id.asc()).all()
        print(f"Số bài đọc từ SQLite: {len(articles)}")

        for article in articles:
            title = (article.title or "").strip()
            content = (article.content or "").strip()
            if not title or not content:
                skipped_count += 1
                continue

            article_count += 1
            chunks = split_into_chunks(content)
            for chunk_no, chunk_text in enumerate(chunks):
                vector_id = len(records)
                # Prefix the title to help retrieve chunks using article-specific queries.
                embedding_text = f"Tiêu đề: {title}\nNội dung: {chunk_text}"
                record = {
                    "vector_id": vector_id,
                    "article_id": int(article.id),
                    "chunk_id": chunk_no,
                    "title": title,
                    "sapo": (article.sapo or "").strip(),
                    "source": (article.source or "").strip(),
                    "category": (article.category or "").strip(),
                    "research_topic": (article.research_topic or "").strip(),
                    "author": (article.author or "").strip(),
                    "published_at": (article.published_at or "").strip(),
                    "url": (article.url or "").strip(),
                    "text": chunk_text,
                }
                records.append(record)
                embedding_texts.append(embedding_text)
    finally:
        db.close()

    return records, embedding_texts, article_count, skipped_count


def main() -> None:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy PhoBERT tại: {MODEL_DIR}\n"
            "Kiểm tra thư mục models/phobert trong project."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("========== XÂY DỰNG VECTOR DATABASE V2 ==========")
    print("Model:", MODEL_DIR)
    print("Thư mục đầu ra:", OUTPUT_DIR)
    print("Thiết bị:", "CUDA" if torch.cuda.is_available() else "CPU")
    print("Đọc dữ liệu SQLite và chia chunk...")

    records, texts, article_count, skipped_count = build_records()
    if not records:
        raise RuntimeError("Không có chunk nào để lập chỉ mục.")

    print("Số bài có đủ title/content:", article_count)
    print("Số bài bị bỏ qua do thiếu title/content:", skipped_count)
    print("Tổng số chunks:", len(records))
    print(f"Chunk: {CHUNK_CHARS} ký tự, overlap: {CHUNK_OVERLAP} ký tự")

    print("Đang nạp tokenizer và PhoBERT từ ổ đĩa cục bộ...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Đang tính embedding. Quá trình này có thể mất một lúc trên CPU...")
    vectors = encode_texts(texts, tokenizer, model, device)
    if vectors.shape[0] != len(records):
        raise RuntimeError(
            f"Số vectors ({vectors.shape[0]}) không khớp metadata ({len(records)})."
        )

    # Because vectors are L2-normalized, inner product is cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Write temporary files first, then rename within the versioned folder.
    index_tmp = OUTPUT_DIR / "news.index.tmp"
    metadata_tmp = OUTPUT_DIR / "chunk_metadata.json.tmp"
    faiss.write_index(index, str(index_tmp))
    with metadata_tmp.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    index_tmp.replace(INDEX_PATH)
    metadata_tmp.replace(METADATA_PATH)

    # Reload both files and check their row counts align before declaring success.
    check_index = faiss.read_index(str(INDEX_PATH))
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        check_metadata = json.load(file)

    if check_index.ntotal != len(check_metadata):
        raise RuntimeError(
            "Kiểm tra sau khi lưu thất bại: số vector khác số metadata."
        )

    print("\n========== HOÀN TẤT ==========")
    print("Số bài được lập chỉ mục:", article_count)
    print("Số chunks / vectors:", check_index.ntotal)
    print("Kích thước vector:", check_index.d)
    print("FAISS index:", INDEX_PATH)
    print("Metadata:", METADATA_PATH)
    print("news.db và vector store cũ không bị sửa.")


if __name__ == "__main__":
    main()
