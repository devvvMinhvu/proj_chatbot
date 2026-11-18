# Trợ Lý Đọc Báo Thông Minh (`proj_chatbot`)

Hệ thống hỏi đáp, tìm kiếm ngữ nghĩa, phân loại và tóm tắt bài báo tiếng Việt ứng dụng PhoBERT, FAISS, ViT5 và Qwen2.5 (qua Ollama), hỗ trợ cả giao diện văn bản và giọng nói trên Streamlit.

## Cấu trúc thư mục

- `config/`: Đường dẫn tập trung và cấu hình hệ thống (`DATA_DIR`, `NEWS_JSONL_PATH`, `NEWS_DB_PATH`, `VECTOR_DB_DIR`, đường dẫn các mô hình).
- `crawler/`: Thu thập và hậu xử lý bài báo tiếng Việt từ VnExpress, Tuổi Trẻ và Tiền Phong (`config.py`, `http_client.py`, `extractor.py`, `sources.py`, `pipeline.py`, `process.py`, `postprocess.py`).
- `database/`: Mô hình SQLAlchemy (`db.py`), truy vấn bài báo/tin mới nhất (`repository.py`) và nhập dữ liệu JSONL vào SQLite (`import_jsonl.py`).
- `processing/`: Chia đoạn văn bản (`chunking.py`) và tóm tắt bài báo bằng ViT5 (`summarizer.py`).
- `nlp/`:
  - `semantic_search.py`: Sinh vector embedding PhoBERT.
  - `vector_store.py`: Xây dựng, nạp và truy vấn FAISS index.
  - `intent_classifier.py` & `train_intent_classifier.py`: Phân loại ý định người dùng.
  - `text_classifier.py` & `train_text_classifier.py`: Phân loại chuyên mục bài báo.
  - `intent_handler.py`: Bộ điều phối ý định trả lời cho giao diện.
- `rag/`: Pipeline truy xuất kết hợp lọc từ khóa/địa danh và sinh câu trả lời bằng Qwen2.5 (`pipeline.py`).
- `voice/`: Nhận dạng giọng nói (`speech_to_text.py`) và tổng hợp giọng nói tiếng Việt qua Edge TTS (`text_to_speech.py`).
- `app.py`, `app_voice.py`, `app_voice_continuous.py`: Các ứng dụng Streamlit (văn bản, ghi âm từng lượt, micro liên tục).
- `evaluate_all_models.py`: Đánh giá các thành phần (intent, category, retrieval, RAG, summarization).

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

2. **Nhập dữ liệu vào SQLite**
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

5. **Đánh giá hệ thống**
   ```bash
   python evaluate_all_models.py
   ```

## Chạy ứng dụng

- Giao diện chat văn bản:
  ```bash
  streamlit run app.py
  ```
- Giao diện hỗ trợ ghi âm giọng nói:
  ```bash
  streamlit run app_voice.py
  ```
- Giao diện hội thoại giọng nói liên tục (WebRTC):
  ```bash
  streamlit run app_voice_continuous.py
  ```
