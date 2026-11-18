# Trợ Lý Đọc Báo Thông Minh (`proj_chatbot`)

Hệ thống hỏi đáp, tìm kiếm kết hợp (SQLite FTS5 + PhoBERT/FAISS), phân loại và tóm tắt bài báo tiếng Việt từ **VnExpress**, **Tuổi Trẻ** và **Tiền Phong**, hỗ trợ giao diện hợp nhất (văn bản, nhấn để nói và hội thoại giọng nói rảnh tay) trên Streamlit.

## Cấu trúc thư mục

- `config/`: Đường dẫn tập trung và cấu hình hệ thống (`DATA_DIR`, `NEWS_JSONL_PATH`, `NEWS_DB_PATH`, `VECTOR_DB_DIR`, `TTS_CACHE_DIR`, đường dẫn các mô hình).
- `crawler/`: Thu thập và hậu xử lý bài báo tiếng Việt từ VnExpress, Tuổi Trẻ và Tiền Phong (`config.py`, `http_client.py`, `extractor.py`, `sources.py`, `pipeline.py`, `process.py`, `postprocess.py`).
- `database/`: Mô hình SQLAlchemy + FTS5 (`db.py`), truy vấn bài báo/chuyên mục/tìm kiếm toàn văn (`repository.py`) và nhập JSONL vào SQLite (`import_jsonl.py`).
- `processing/`: Chia đoạn văn bản (`chunking.py`) và tóm tắt bài báo bằng ViT5 (`summarizer.py`).
- `nlp/`:
  - `semantic_search.py`: Sinh vector embedding PhoBERT (lazy-load).
  - `vector_store.py`: Xây dựng, nạp và truy vấn FAISS index.
  - `intent_classifier.py` & `train_intent_classifier.py`: Phân loại ý định người dùng.
  - `text_classifier.py` & `train_text_classifier.py`: Phân loại chuyên mục bài báo.
  - `intent_handler.py`: Bộ điều phối ý định hỗ trợ lọc báo/chuyên mục và ngữ cảnh nhiều lượt.
- `rag/`: Truy xuất kết hợp SQLite FTS5 + FAISS và sinh câu trả lời Qwen2.5 (`pipeline.py`).
- `voice/`: Nhận dạng giọng nói (`speech_to_text.py`), chuẩn bị văn bản đọc (`reading_text.py`), tổng hợp giọng nói Edge TTS có cache và fallback (`text_to_speech.py`, `tts_adapter.py`, `tts_background.py`).
- `app.py`: Ứng dụng Streamlit hợp nhất (Văn bản / Nhấn để nói / Hội thoại rảnh tay, lọc chuyên mục & nguồn báo, hộp thoại đọc & tóm tắt bài báo).
- `evaluate_all_models.py`: Đánh giá các thành phần (intent, category, retrieval, RAG, summarization).
- `tests/`: Bộ kiểm thử đơn vị (`test_topic_search.py`, `test_tts_adapter_recovery.py`, `test_hybrid_search.py`).

## Quy trình dữ liệu & huấn luyện

1. **Thu thập dữ liệu bài báo (JSONL)**
   ```bash
   python -m crawler.process
   ```
   Bổ sung toàn văn bị thiếu (và tùy chọn bổ sung `published_at`):
   ```bash
   python -m crawler.postprocess
   python -m crawler.postprocess --fill-dates
   ```

2. **Nhập dữ liệu vào SQLite & lập chỉ mục FTS5**
   ```bash
   python -m database.import_jsonl
   ```

3. **Xây dựng Vector Database (FAISS + PhoBERT)**
   ```bash
   python -m nlp.vector_store
   ```

4. **Huấn luyện các bộ phân loại**
   ```bash
   python -m nlp.train_intent_classifier
   python -m nlp.train_text_classifier
   ```

5. **Kiểm thử & Đánh giá**
   ```bash
   python -m unittest discover -s tests -v
   python evaluate_all_models.py
   ```

## Chạy ứng dụng

```bash
streamlit run app.py
```
