# Trợ Lý Đọc Báo Thông Minh (`proj_chatbot`)

Hệ thống hỏi đáp, tìm kiếm kết hợp (SQLite FTS5 + PhoBERT/FAISS), phân loại và tóm tắt bài báo tiếng Việt từ **VnExpress**, **Tuổi Trẻ** và **Tiền Phong**, hỗ trợ giao diện hợp nhất (văn bản, nhấn để nói và hội thoại giọng nói rảnh tay) trên Streamlit.

## Cấu trúc thư mục

- `config/`: Đường dẫn tập trung và cấu hình hệ thống (`DATA_DIR`, `NEWS_JSONL_PATH`, `NEWS_DB_PATH`, `VECTOR_DB_DIR`, `PIPER_MODELS_DIR`, đường dẫn các mô hình).
- `crawler/`: Thu thập và hậu xử lý bài báo tiếng Việt từ VnExpress, Tuổi Trẻ và Tiền Phong (`config.py`, `http_client.py`, `extractor.py`, `sources.py`, `pipeline.py`, `process.py`, `postprocess.py`).
- `database/`: Mô hình SQLAlchemy + FTS5 (`db.py`), truy vấn bài báo/chuyên mục/tìm kiếm toàn văn (`repository.py`) và nhập JSONL vào SQLite (`import_jsonl.py`).
- `processing/`: Chia đoạn văn bản (`chunking.py`), tách câu tiếng Việt (`sentences.py`), tóm tắt bài báo bằng ViT5 cho ứng dụng (`summarizer.py`) và các hệ thống tóm tắt dùng để so sánh (`summarizers/`: Lead-k, TextRank, PhoBERT centroid + MMR, ViT5, mT5).
- `evaluation/`: Độ đo ROUGE/BLEU tiếng Việt (`metrics.py`), nạp dữ liệu đánh giá (`datasets.py`) và benchmark so sánh mô hình tóm tắt (`summarization_benchmark.py`).
- `nlp/`:
  - `semantic_search.py`: Sinh vector embedding PhoBERT (lazy-load).
  - `vector_store.py`: Xây dựng, nạp và truy vấn FAISS index.
  - `intent_classifier.py` & `train_intent_classifier.py`: Phân loại ý định người dùng.
  - `text_classifier.py` & `train_text_classifier.py`: Phân loại chuyên mục bài báo.
  - `intent_handler.py`: Bộ điều phối ý định hỗ trợ lọc báo/chuyên mục và ngữ cảnh nhiều lượt.
- `rag/`: Truy xuất kết hợp SQLite FTS5 + FAISS và sinh câu trả lời Qwen2.5 (`pipeline.py`).
- `voice/`: Nhận dạng giọng nói (`speech_to_text.py`), chuẩn bị văn bản đọc (`reading_text.py`), tổng hợp giọng nói cục bộ Piper TTS không lưu đĩa với LRU cache trong RAM (`text_to_speech.py`, `tts_adapter.py`, `tts_background.py`, `models/piper/`).
- `app.py`: Ứng dụng Streamlit hợp nhất (Văn bản / Nhấn để nói / Hội thoại rảnh tay, lọc chuyên mục & nguồn báo, hộp thoại đọc & tóm tắt bài báo).
- `evaluate_all_models.py`: Đánh giá các thành phần (intent, category, retrieval, RAG, summarization).
- `tests/`: Bộ kiểm thử đơn vị (`test_topic_search.py`, `test_tts_adapter_recovery.py`, `test_hybrid_search.py`, `test_summarization_benchmark.py`).

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

6. **So sánh các mô hình tóm tắt (TextRank, BERT, T5, ViT5)**
   ```bash
   python -m evaluation.summarization_benchmark --limit 200 --length-budget 40
   ```
   - Hệ thống (`--systems`): `lead` (Lead-k), `textrank`, `phobert` (PhoBERT centroid + MMR),
     `vit5_app` (đúng pipeline ViT5 của ứng dụng: chia đoạn 400 từ, tóm tắt 2 đoạn đầu rồi ghép;
     cấu hình đọc trực tiếp từ `processing/summarizer.py`),
     `vit5` (ViT5 một lượt, 1024 token đầu), `mt5_xlsum` (mT5 XL-Sum, cần tải trước từ
     Hugging Face) hoặc `seq2seq:<đường dẫn/ID mô hình>` cho mô hình khác (ví dụ BARTpho).
   - Dữ liệu mặc định là `data/news.jsonl`, dùng sapo làm bản tóm tắt tham chiếu. Thân bài
     báo thu thập được lặp lại sapo ở đoạn đầu (~99,9% bài), nên đoạn này **được loại khỏi
     đầu vào** trước khi tóm tắt; nếu không, điểm ROUGE bị thổi phồng (`--keep-leak` chỉ dùng
     để minh họa hiệu ứng này).
   - Bộ dữ liệu chuẩn (ví dụ tập test VietNews/VNDS) cần chuyển sang JSONL, mỗi dòng có
     `document` và `summary`, rồi chạy với `--dataset jsonl --input <file>`; dấu `_` của văn
     bản đã tách từ được tự động đổi thành khoảng trắng.
   - ROUGE/BLEU tính ở mức âm tiết bằng tokenizer Unicode riêng (gói `rouge_score` mặc định
     xóa mọi ký tự không phải ASCII nên không dùng được cho tiếng Việt). `--length-budget N`
     chấm thêm bản cắt còn N từ đầu để so sánh công bằng giữa các hệ thống có độ dài khác nhau.
   - Kết quả nằm trong `reports/summarization_benchmark/<run>/`: `summary.md` (bảng kèm
     khoảng tin cậy 95% bootstrap), `summary.csv`, `results.json`, `predictions.jsonl`.
   - `summary.md` có thêm bảng **kiểm định ý nghĩa thống kê** (paired bootstrap, Koehn 2004):
     chênh lệch ROUGE của từng hệ thống so với baseline (`--baseline`, mặc định Lead-k), khoảng
     tin cậy 95% và p-value hai phía; `*` là p < 0,05 (chưa hiệu chỉnh so sánh bội).
   - Ghép kết quả từ nhiều lần chạy trên **cùng tập bài** (cùng dữ liệu, `--seed`, `--limit`):
     ```bash
     python -m evaluation.significance <run1>/predictions.jsonl <run2>/predictions.jsonl --baseline lead3 --output so_sanh.md
     ```

## Chạy ứng dụng

```bash
streamlit run app.py
```
