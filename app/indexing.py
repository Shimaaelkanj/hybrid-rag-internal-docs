from __future__ import annotations

import json
import pickle
import re
import time

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from app.chunking import Chunk
from app.config import (BM25_FILE, CHROMA_DIR, CHROMA_HOST, CHROMA_PORT, CHUNKS_FILE,
                        COLLECTION_NAME, DEDUP_THRESHOLD, DUPLICATES_FILE)


# ---------- Chroma ----------
def _client():
    """Chroma server when CHROMA_HOST is set (docker-compose), otherwise local files."""
    if CHROMA_HOST:
        return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    def open_collection():
        return _client().get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"})   # distance = 1 - cosine

    if not CHROMA_HOST:
        return open_collection()

    last = None                      # the Chroma container may still be starting: retry for ~30 s
    for _ in range(15):
        try:
            return open_collection()
        except Exception as e:       # noqa: BLE001
            last = e
            time.sleep(2)
    raise RuntimeError(f"Could not reach Chroma at {CHROMA_HOST}:{CHROMA_PORT}: {last}") from last


def reset_indexes():
    try:
        _client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    for f in (CHUNKS_FILE, BM25_FILE, DUPLICATES_FILE):
        f.unlink(missing_ok=True)


def add_document_chunks(chunks: list[Chunk], vectors: np.ndarray, collection,
                        threshold: float = DEDUP_THRESHOLD):
    """Dedup, then insert. Returns (accepted_chunks, duplicate_records)."""
    source = chunks[0].source
    collection.delete(where={"source": source})      # re-ingesting replaces old chunks
    existing = collection.count()

    accepted_idx: list[int] = []
    duplicates: list[dict] = []

    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        dup = None

        # 1) near-duplicate of a chunk accepted earlier in this same document
        if accepted_idx:
            sims = vectors[accepted_idx] @ vec
            j = int(np.argmax(sims))
            if sims[j] > threshold:
                dup = (chunks[accepted_idx[j]].id, float(sims[j]))

        # 2) near-duplicate of a chunk already in the store (other documents)
        if dup is None and existing > 0:
            res = collection.query(query_embeddings=[vec.tolist()], n_results=1,
                                   include=["distances"])
            if res["ids"][0]:
                sim = 1 - res["distances"][0][0]
                if sim > threshold:
                    dup = (res["ids"][0][0], float(sim))

        if dup:
            duplicates.append({"chunk_id": chunk.id, "source": chunk.source,
                               "duplicate_of": dup[0], "similarity": round(dup[1], 4),
                               "preview": chunk.text[:120]})
        else:
            accepted_idx.append(i)

    for start in range(0, len(accepted_idx), 500):
        batch = accepted_idx[start:start + 500]
        collection.add(
            ids=[chunks[i].id for i in batch],
            embeddings=vectors[batch].tolist(),
            documents=[chunks[i].text for i in batch],
            metadatas=[chunks[i].metadata() for i in batch],
        )
    return [chunks[i] for i in accepted_idx], duplicates


# ---------- chunk store (source of truth) ----------
def load_chunk_rows() -> list[dict]:
    if not CHUNKS_FILE.exists():
        return []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_chunk_rows(rows: list[dict]) -> None:
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def log_duplicates(dups: list[dict]) -> None:
    if not dups:
        return
    with DUPLICATES_FILE.open("a", encoding="utf-8") as f:
        for d in dups:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


# ---------- BM25 ----------
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rebuild_bm25(rows: list[dict]) -> None:
    if not rows:
        BM25_FILE.unlink(missing_ok=True)
        return
    bm25 = BM25Okapi([tokenize(r["text"]) for r in rows])
    with BM25_FILE.open("wb") as f:
        pickle.dump({"ids": [r["id"] for r in rows], "bm25": bm25}, f)


def load_bm25():
    with BM25_FILE.open("rb") as f:
        return pickle.load(f)


# ---------- sync check ----------
def verify_sync(collection) -> bool:
    rows = load_chunk_rows()
    store_ids = {r["id"] for r in rows}
    chroma_ids = set(collection.get(include=[])["ids"])
    bm25_ids = set(load_bm25()["ids"]) if BM25_FILE.exists() else set()
    return store_ids == chroma_ids == bm25_ids
