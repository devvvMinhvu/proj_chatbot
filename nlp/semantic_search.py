"""PhoBERT embedding utilities with lazy model initialization."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import PHOBERT_MODEL_DIR


@lru_cache(maxsize=1)
def get_embedding_model() -> tuple[Any, Any, torch.device]:
    """Load PhoBERT tokenizer and model once on first use."""
    if not PHOBERT_MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy PhoBERT tại: {PHOBERT_MODEL_DIR}\n"
            "Kiểm tra thư mục models/phobert trong project."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        str(PHOBERT_MODEL_DIR),
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(PHOBERT_MODEL_DIR),
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def create_embeddings(
    texts: list[str],
    batch_size: int = 16,
    normalize: bool = False,
    max_length: int = 256,
    show_progress: bool = False,
) -> np.ndarray:
    """Create batched PhoBERT mean-pooled embeddings for a list of texts."""
    tokenizer, model, device = get_embedding_model()

    if not texts:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)

    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

        if show_progress:
            processed = min(start + len(batch), len(texts))
            print(f"Đã tạo embedding: {processed}/{len(texts)}")

    return np.concatenate(all_embeddings, axis=0)


def create_embedding(text: str, normalize: bool = False) -> np.ndarray:
    """Create an embedding vector for a single query or document string."""
    return create_embeddings([text], normalize=normalize)[0]
