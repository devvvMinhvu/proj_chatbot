import json
from pathlib import Path

import faiss
import numpy as np

from database.database_get import SessionLocal, Article
from nlp.semantic_search import create_embedding, create_embeddings
from processing.chunking import split_text


# ==================================================
# 1. ĐƯỜNG DẪN VÀ CẤU HÌNH
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "vector_db_v2"
INDEX_PATH = DATA_DIR / "news.index"
METADATA_PATH = DATA_DIR / "chunk_metadata.json"

BATCH_SIZE = 4

# Khởi tạo index rỗng. Kích thước thật sẽ được xác định
# từ embedding khi xây dựng kho vector.
index = faiss.IndexFlatIP(768)

metadata_cache = None
index_loaded = False


# ==================================================
# 2. LẤY DỮ LIỆU TỪ SQLITE
# ==================================================

def get_article():
    db = SessionLocal()

    try:
        return (
            db.query(Article)
            .order_by(Article.id.asc())
            .all()
        )
    finally:
        db.close()


# ==================================================
# 3. XÂY DỰNG VECTOR DATABASE
# ==================================================

def build_embeddings():
    global index, metadata_cache, index_loaded

    articles = get_article()

    metadata = []
    embedding_texts = []

    print("Số bài đọc từ database:", len(articles))

    for article in articles:
        title = (article.title or "").strip()
        sapo = (article.sapo or "").strip()
        content = (article.content or "").strip()

        # Chỉ bỏ qua bài không có nội dung.
        if not content:
            print("Bỏ qua bài thiếu content:", article.id)
            continue

        # Chia phần nội dung bài báo thành các chunk.
        # Tạm giữ thông số cũ của bạn; cần xem chunking.py
        # để xác nhận chunk_size được tính theo ký tự hay từ.
        chunks = split_text(
            content,
            chunk_size=150,
            overlap=30
        )

        for chunk_id, chunk in enumerate(chunks):
            chunk = chunk.strip()

            if not chunk:
                continue

            # Nhắc lại tiêu đề và sapo giúp embedding có ngữ cảnh.
            embedding_text = (
                f"Tiêu đề: {title}\n"
                f"Tóm tắt: {sapo}\n"
                f"Nội dung: {chunk}"
            )

            embedding_texts.append(embedding_text)

            # Mỗi metadata tương ứng đúng một vector.
            metadata.append({
                "vector_id": len(metadata),
                "article_id": int(article.id),
                "chunk_id": chunk_id,
                "title": title,
                "sapo": sapo,
                "category": (article.category or "").strip(),
                "research_topic": (
                    article.research_topic or ""
                ).strip(),
                "author": (article.author or "").strip(),
                "published_at": (
                    article.published_at or ""
                ).strip(),
                "source": (article.source or "").strip(),
                "url": (article.url or "").strip(),
                "text": chunk
            })

        print(
            f"Đã xử lý bài {article.id}: "
            f"{len(chunks)} chunks"
        )

    if not embedding_texts:
        raise RuntimeError(
            "Không có chunk hợp lệ để xây dựng vector database."
        )

    print("Tổng số chunks:", len(embedding_texts))
    print("Đang tạo embedding theo batch...")

    # Tạo embedding theo batch thay vì gọi PhoBERT
    # riêng cho từng chunk.
    vectors = create_embeddings(
        embedding_texts,
        batch_size=BATCH_SIZE
    )

    if len(vectors) != len(metadata):
        raise RuntimeError(
            f"Số vector ({len(vectors)}) không khớp "
            f"số metadata ({len(metadata)})."
        )

    # Chuẩn hóa vector để Inner Product tương đương cosine similarity.
    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)

    # Tự lấy chiều vector từ kết quả embedding.
    new_index = faiss.IndexFlatIP(vectors.shape[1])
    new_index.add(vectors)

    if new_index.ntotal != len(metadata):
        raise RuntimeError(
            "Số vector trong FAISS không khớp metadata."
        )

    # Chỉ thay index trong bộ nhớ sau khi tạo thành công.
    index = new_index
    metadata_cache = metadata
    index_loaded = True

    print("Xây dựng vector database hoàn tất.")
    print("Số vector:", index.ntotal)
    print("Kích thước vector:", index.d)

    return metadata


# ==================================================
# 4. LƯU INDEX VÀ METADATA
# ==================================================

def save_index(metadata=None):
    global metadata_cache

    if metadata is None:
        metadata = metadata_cache

    if metadata is None:
        raise RuntimeError(
            "Chưa có metadata. Hãy chạy build_embeddings() trước."
        )

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"Không thể lưu: index có {index.ntotal} vector, "
            f"nhưng metadata có {len(metadata)} bản ghi."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Lưu file tạm trước để hạn chế ghi đè bằng file chưa hoàn chỉnh.
    index_tmp = DATA_DIR / "news.index.tmp"
    metadata_tmp = DATA_DIR / "chunk_metadata.json.tmp"

    faiss.write_index(index, str(index_tmp))

    with metadata_tmp.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    # Chỉ lưu vào thư mục version mới, không đụng file cũ.
    index_tmp.replace(INDEX_PATH)
    metadata_tmp.replace(METADATA_PATH)

    print("Đã lưu FAISS index:", INDEX_PATH)
    print("Đã lưu metadata:", METADATA_PATH)


# ==================================================
# 5. NẠP INDEX VÀ METADATA
# ==================================================

def load_index(force_reload=False):
    global index, metadata_cache, index_loaded

    if index_loaded and not force_reload:
        return metadata_cache

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy index: {INDEX_PATH}. "
            "Hãy xây dựng vector database trước."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy metadata: {METADATA_PATH}."
        )

    loaded_index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        loaded_metadata = json.load(file)

    if loaded_index.ntotal != len(loaded_metadata):
        raise RuntimeError(
            f"Index có {loaded_index.ntotal} vector nhưng metadata "
            f"có {len(loaded_metadata)} bản ghi."
        )

    index = loaded_index
    metadata_cache = loaded_metadata
    index_loaded = True

    return metadata_cache


# ==================================================
# 6. TÌM KIẾM NGỮ NGHĨA
# ==================================================

def search(text, k=3):
    results = semantic_search_raw(text, k=k)
    return (
        np.array([[r["similarity"] for r in results]], dtype=np.float32),
        np.array([[r["vector_id"] for r in results]], dtype=np.int64)
    )


def semantic_search_raw(query, k=20):
    metadata = load_index()

    if index.ntotal == 0 or k <= 0:
        return []

    query_embedding = create_embedding(query)

    query_vector = np.asarray(
        [query_embedding],
        dtype=np.float32
    )

    faiss.normalize_L2(query_vector)

    actual_k = min(k, index.ntotal)

    similarities, indices = index.search(
        query_vector,
        actual_k
    )

    results = []

    for similarity, vector_id in zip(
        similarities[0],
        indices[0]
    ):
        # FAISS có thể trả -1 nếu không có đủ kết quả.
        if vector_id < 0 or vector_id >= len(metadata):
            continue

        results.append({
            **metadata[vector_id],
            "similarity": float(similarity)
        })

    return results


def diversify_results(
    results,
    max_chunks_per_article=2,
    top_k=5
):
    selected = []
    article_count = {}

    for result in results:
        article_id = result["article_id"]

        current_count = article_count.get(article_id, 0)

        if current_count >= max_chunks_per_article:
            continue

        selected.append(result)
        article_count[article_id] = current_count + 1

        if len(selected) >= top_k:
            break

    return selected


def semantic_search(query, k=5):
    # Lấy nhiều chunk ứng viên trước, sau đó hạn chế
    # số chunk đến từ cùng một bài báo.
    candidates = semantic_search_raw(
        query,
        k=max(20, k)
    )

    results = diversify_results(
        candidates,
        max_chunks_per_article=2,
        top_k=k
    )

    print("========= KẾT QUẢ TRUY XUẤT =========")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("Article ID:", result["article_id"])
        print("Title:", result["title"])
        print("Chunk ID:", result["chunk_id"])
        print("Similarity:", result["similarity"])
        print("Text:", result["text"][:200])

    return results


# ==================================================
# 7. LẤY BÀI BÁO THEO ID
# ==================================================

def get_articles_by_ids(article_ids):
    if not article_ids:
        return []

    db = SessionLocal()

    try:
        articles = (
            db.query(Article)
            .filter(Article.id.in_(set(article_ids)))
            .all()
        )

        article_dict = {
            article.id: article
            for article in articles
        }

        # Giữ nguyên thứ tự và không nhân bản bài báo.
        ordered_articles = []
        seen = set()

        for article_id in article_ids:
            if article_id in article_dict and article_id not in seen:
                ordered_articles.append(article_dict[article_id])
                seen.add(article_id)

        return ordered_articles

    finally:
        db.close()


# ==================================================
# 8. CHẠY FILE ĐỂ TẠO VECTOR DATABASE
# ==================================================

if __name__ == "__main__":
    metadata = build_embeddings()
    save_index(metadata)

    # Kiểm tra lại bằng cách đọc hai file vừa lưu.
    loaded_metadata = load_index(force_reload=True)

    print("\n========== KIỂM TRA ==========")
    print("Tổng số vector:", index.ntotal)
    print("Tổng số metadata:", len(loaded_metadata))
    print("Thư mục lưu:", DATA_DIR)