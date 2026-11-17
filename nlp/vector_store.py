"""FAISS vector database builder, loader, and similarity search interface."""

from __future__ import annotations

import json
from typing import Any

import faiss
import numpy as np
import torch

from config import CHUNK_METADATA_PATH, FAISS_INDEX_PATH, VECTOR_DB_DIR
from database.db import Article, SessionLocal, ensure_db_initialized
from nlp.semantic_search import create_embedding, create_embeddings, get_embedding_model
from processing.chunking import split_text

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
BATCH_SIZE = 4
FLUSH_SIZE = 128
MAX_LENGTH = 256

_index: faiss.Index | None = None
_metadata: list[dict[str, Any]] | None = None


def build_vector_database() -> None:
    """Build and atomically persist the FAISS index and chunk metadata from SQLite."""
    ensure_db_initialized()
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    _, model, device = get_embedding_model()
    index = faiss.IndexFlatIP(model.config.hidden_size)
    metadata: list[dict[str, Any]] = []
    pending_texts: list[str] = []

    db = SessionLocal()
    article_count = 0
    skipped_count = 0

    try:
        total_articles = db.query(Article).count()

        print("===== XÂY DỰNG VECTOR DATABASE =====")
        print("Số bài trong DB:", total_articles)
        print("Thiết bị:", device)
        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        print("Batch size:", BATCH_SIZE)

        articles = db.query(Article).order_by(Article.id.asc()).yield_per(100)

        def flush_batch() -> None:
            if not pending_texts:
                return

            vectors = create_embeddings(
                pending_texts,
                batch_size=BATCH_SIZE,
                normalize=True,
                max_length=MAX_LENGTH,
            )
            if len(vectors) != len(pending_texts):
                raise RuntimeError("Số vector không khớp số chunks.")

            index.add(vectors)
            pending_texts.clear()
            print(f"Đã tạo embedding: {index.ntotal:,} chunks")

        for article in articles:
            title = (article.title or "").strip()
            sapo = (article.sapo or "").strip()
            body = (article.body or "").strip()

            if not body:
                skipped_count += 1
                continue

            chunks = split_text(body, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            valid_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

            if not valid_chunks:
                skipped_count += 1
                continue

            article_count += 1

            for chunk_id, chunk in enumerate(valid_chunks):
                embedding_text = (
                    f"Tiêu đề: {title}\n"
                    f"Tóm tắt: {sapo}\n"
                    f"Nội dung: {chunk}"
                )
                pending_texts.append(embedding_text)
                metadata.append(
                    {
                        "vector_id": len(metadata),
                        "article_id": article.id,
                        "chunk_id": chunk_id,
                        "title": title,
                        "sapo": sapo,
                        "source": article.source or "",
                        "category": article.category or "",
                        "split": article.split or "",
                        "url": article.url or "",
                        "text": chunk,
                    }
                )

                if len(pending_texts) >= FLUSH_SIZE:
                    flush_batch()

            if article_count % 500 == 0:
                print(f"Đã xử lý {article_count:,}/{total_articles:,} bài báo")

        flush_batch()
    finally:
        db.close()

    if index.ntotal != len(metadata):
        raise RuntimeError(f"FAISS có {index.ntotal} vector, metadata có {len(metadata)} bản ghi.")

    if index.ntotal == 0:
        raise RuntimeError("Không tạo được vector nào.")

    index_tmp = VECTOR_DB_DIR / "news.index.tmp"
    metadata_tmp = VECTOR_DB_DIR / "chunk_metadata.json.tmp"

    faiss.write_index(index, str(index_tmp))
    with metadata_tmp.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False)

    index_tmp.replace(FAISS_INDEX_PATH)
    metadata_tmp.replace(CHUNK_METADATA_PATH)

    saved_index, saved_metadata = load_vector_database(force_reload=True)

    print("\n===== HOÀN TẤT =====")
    print("Số bài đã xử lý:", article_count)
    print("Số bài bỏ qua:", skipped_count)
    print("Tổng vector:", saved_index.ntotal)
    print("Tổng metadata:", len(saved_metadata))
    print("Index:", FAISS_INDEX_PATH)
    print("Metadata:", CHUNK_METADATA_PATH)


def load_vector_database(force_reload: bool = False) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Load and cache the FAISS index and chunk metadata."""
    global _index, _metadata

    if _index is not None and _metadata is not None and not force_reload:
        return _index, _metadata

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy FAISS index: {FAISS_INDEX_PATH}")

    if not CHUNK_METADATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy metadata: {CHUNK_METADATA_PATH}")

    print("Đang nạp Vector Database...")

    loaded_index = faiss.read_index(str(FAISS_INDEX_PATH))
    with CHUNK_METADATA_PATH.open("r", encoding="utf-8") as file:
        loaded_metadata = json.load(file)

    if loaded_index.ntotal != len(loaded_metadata):
        raise RuntimeError(
            f"FAISS có {loaded_index.ntotal} vector nhưng metadata có {len(loaded_metadata)} bản ghi."
        )

    if loaded_index.ntotal == 0:
        raise RuntimeError("Vector Database đang rỗng.")

    _index = loaded_index
    _metadata = loaded_metadata

    print("Đã nạp thành công.")
    print("Tổng số vector:", _index.ntotal)
    print("Tổng số metadata:", len(_metadata))
    print("Kích thước vector:", _index.d)

    return _index, _metadata


def search_vectors(
    query: str,
    candidate_k: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Encode a query and search the FAISS index for nearest chunk candidates."""
    index, metadata = load_vector_database()

    if not query or not query.strip():
        raise ValueError("Câu hỏi không được để trống.")

    if index.ntotal == 0 or candidate_k <= 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64), metadata

    embedding = create_embedding(query, normalize=True)
    query_vector = np.asarray([embedding], dtype=np.float32)

    if query_vector.shape[1] != index.d:
        raise RuntimeError(f"Embedding có chiều {query_vector.shape[1]}, FAISS yêu cầu {index.d}.")

    faiss.normalize_L2(query_vector)
    actual_k = min(candidate_k, index.ntotal)
    scores, indices = index.search(query_vector, actual_k)
    return scores[0], indices[0], metadata


if __name__ == "__main__":
    build_vector_database()
