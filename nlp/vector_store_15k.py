import json
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from database.database_15k import SessionLocal, Article
from nlp.semantic_search import tokenizer, model, DEVICE
from processing.chunking import split_text


# =========================================
# 1. CAU HINH
# =========================================

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT / "data" / "vector_db_15k"
INDEX_PATH = OUTPUT_DIR / "news.index"
METADATA_PATH = OUTPUT_DIR / "chunk_metadata.json"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
BATCH_SIZE = 4
FLUSH_SIZE = 128
MAX_LENGTH = 256


# =========================================
# 2. TAO EMBEDDING
# =========================================

def encode_texts(texts):
    vectors = []

    model.eval()

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

            mask = inputs["attention_mask"].unsqueeze(-1)
            mask = mask.to(outputs.last_hidden_state.dtype)

            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)

            embeddings = summed / counts

            # Chuan hoa vector de dung cosine similarity
            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )

        vectors.append(
            embeddings.cpu().numpy().astype(np.float32)
        )

    if not vectors:
        return np.empty(
            (0, model.config.hidden_size),
            dtype=np.float32
        )

    return np.concatenate(vectors, axis=0)


# =========================================
# 3. XAY DUNG VECTOR DATABASE
# =========================================

def build_vector_database():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatIP(model.config.hidden_size)
    metadata = []
    pending_texts = []

    db = SessionLocal()
    article_count = 0
    skipped_count = 0

    try:
        total_articles = db.query(Article).count()

        print("===== XAY DUNG VECTOR DATABASE 15K =====")
        print("So bai trong DB:", total_articles)
        print("Thiet bi:", DEVICE)
        print("GPU:", torch.cuda.get_device_name(0)
              if torch.cuda.is_available() else "CPU")
        print("Batch size:", BATCH_SIZE)

        articles = (
            db.query(Article)
            .order_by(Article.id.asc())
            .yield_per(100)
        )

        def flush_batch():
            if not pending_texts:
                return

            vectors = encode_texts(pending_texts)

            if len(vectors) != len(pending_texts):
                raise RuntimeError(
                    "So vector khong khop so chunks."
                )

            index.add(vectors)
            pending_texts.clear()

            print(
                f"Da tao embedding: {index.ntotal:,} chunks"
            )

        for article in articles:
            title = (article.title or "").strip()
            sapo = (article.sapo or "").strip()
            body = (article.body or "").strip()

            if not body:
                skipped_count += 1
                continue

            chunks = split_text(
                body,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP
            )

            valid_chunks = [
                chunk.strip()
                for chunk in chunks
                if chunk and chunk.strip()
            ]

            if not valid_chunks:
                skipped_count += 1
                continue

            article_count += 1

            for chunk_id, chunk in enumerate(valid_chunks):
                # Dua tieu de va sapo vao embedding de bo sung ngu canh
                embedding_text = (
                    f"Tieu de: {title}\n"
                    f"Tom tat: {sapo}\n"
                    f"Noi dung: {chunk}"
                )

                pending_texts.append(embedding_text)

                metadata.append({
                    "vector_id": len(metadata),
                    "article_id": article.id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "sapo": sapo,
                    "source": article.source or "",
                    "category": article.category or "",
                    "split": article.split or "",
                    "url": article.url or "",
                    "text": chunk
                })

                if len(pending_texts) >= FLUSH_SIZE:
                    flush_batch()

            if article_count % 500 == 0:
                print(
                    f"Da xu ly {article_count:,}/"
                    f"{total_articles:,} bai bao"
                )

        # Xu ly cac chunk con lai
        flush_batch()

    finally:
        db.close()

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"FAISS co {index.ntotal} vector, "
            f"metadata co {len(metadata)} ban ghi."
        )

    if index.ntotal == 0:
        raise RuntimeError("Khong tao duoc vector nao.")

    # =====================================
    # 4. LUU FILE
    # =====================================

    index_tmp = OUTPUT_DIR / "news.index.tmp"
    metadata_tmp = OUTPUT_DIR / "chunk_metadata.json.tmp"

    faiss.write_index(index, str(index_tmp))

    with metadata_tmp.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False
        )

    index_tmp.replace(INDEX_PATH)
    metadata_tmp.replace(METADATA_PATH)

    # Doc lai de kiem tra
    saved_index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        saved_metadata = json.load(file)

    if saved_index.ntotal != len(saved_metadata):
        raise RuntimeError(
            "Kiem tra sau khi luu that bai."
        )

    print("\n===== HOAN THANH =====")
    print("So bai da xu ly:", article_count)
    print("So bai bo qua:", skipped_count)
    print("Tong vector:", saved_index.ntotal)
    print("Tong metadata:", len(saved_metadata))
    print("Index:", INDEX_PATH)
    print("Metadata:", METADATA_PATH)


if __name__ == "__main__":
    build_vector_database()