from __future__ import annotations

import time

import numpy as np

from app.config import (EMBEDDING_MODEL, EMBEDDING_PROVIDER,
                        LOCAL_EMBEDDING_MODEL, OPENAI_API_KEY)

_openai_client = None
_local_model = None


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _embed_local(texts: list[str]) -> np.ndarray:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)   # downloads once (~90 MB)
    vecs = _local_model.encode(texts, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def _embed_openai(texts: list[str], batch_size: int = 96) -> np.ndarray:
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing. Put it in your .env file.")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)

    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(5):
            try:
                resp = _openai_client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                vectors.extend(d.embedding for d in resp.data)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    return np.array(vectors, dtype=np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (len(texts), dim)."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    arr = _embed_local(texts) if EMBEDDING_PROVIDER == "local" else _embed_openai(texts)
    return _normalize(arr)