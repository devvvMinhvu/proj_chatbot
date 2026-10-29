from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# Xác định đường dẫn dựa trên vị trí file, không phụ thuộc thư mục chạy lệnh.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "phobert"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model = AutoModel.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model.to(DEVICE)
model.eval()


def create_embeddings(texts, batch_size=16):
    """
    Tạo embedding cho nhiều văn bản theo batch.
    Kết quả có dạng (số văn bản, kích thước vector).
    """
    if not texts:
        return np.empty(
            (0, model.config.hidden_size),
            dtype=np.float32
        )

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

            # Loại các token padding khỏi phép tính trung bình.
            mask = inputs["attention_mask"].unsqueeze(-1)
            mask = mask.to(outputs.last_hidden_state.dtype)

            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts

        all_embeddings.append(
            embeddings.cpu().numpy().astype(np.float32)
        )

        processed = min(start + len(batch), len(texts))
        print(f"Đã tạo embedding: {processed}/{len(texts)}")

    return np.concatenate(all_embeddings, axis=0)


def create_embedding(text):
    """Giữ hàm cũ để các file đang sử dụng không bị ảnh hưởng."""
    return create_embeddings([text])[0]