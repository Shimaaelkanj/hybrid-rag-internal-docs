
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import (DENSE_WEIGHT, RERANK_CANDIDATES, RERANK_TOP_N,
                        RERANKER_MODEL, RRF_K, TOP_K)
from app.embeddings import embed_texts
from app.indexing import get_collection, load_bm25, load_chunk_rows, tokenize
from app.config import BM25_FILE


@dataclass
class Hit:
    id: str
    text: str
    source: str
    heading: str
    page: int | None
    strategy: str
    dense_rank: int | None = None
    dense_score: float | None = None     # cosine similarity
    sparse_rank: int | None = None
    sparse_score: float | None = None    # BM25 score
    rrf_score: float | None = None
    rerank_score: float | None = None


def rrf_fuse(dense: list[Hit], sparse: list[Hit],
             dense_weight: float, sparse_weight: float, k: int = RRF_K) -> list[Hit]:
    """Weighted Reciprocal Rank Fusion: score = sum(weight / (k + rank)) over the lists."""
    merged: dict[str, Hit] = {}

    for h in dense:
        h.rrf_score = dense_weight / (k + h.dense_rank)
        merged[h.id] = h

    for h in sparse:
        contribution = sparse_weight / (k + h.sparse_rank)
        if h.id in merged:
            m = merged[h.id]
            m.sparse_rank, m.sparse_score = h.sparse_rank, h.sparse_score
            m.rrf_score += contribution
        else:
            h.rrf_score = contribution
            merged[h.id] = h

    return sorted(merged.values(), key=lambda h: h.rrf_score, reverse=True)


class Retriever:
    def __init__(self):
        self._reranker = None
        self.reload()

    def reload(self) -> None:
        """Re-read the indexes (call after ingesting new documents)."""
        self.collection = get_collection()
        self.rows = {r["id"]: r for r in load_chunk_rows()}
        if BM25_FILE.exists():
            data = load_bm25()
            self.bm25_ids, self.bm25 = data["ids"], data["bm25"]
        else:
            self.bm25_ids, self.bm25 = [], None

    def _hit(self, chunk_id: str) -> Hit | None:
        r = self.rows.get(chunk_id)
        if r is None:
            return None
        return Hit(id=r["id"], text=r["text"], source=r["source"], heading=r["heading"],
                   page=r["page"], strategy=r["strategy"])

    # ---- 1. dense ----
    def dense_search(self, query: str, k: int = TOP_K) -> list[Hit]:
        n = self.collection.count()
        if n == 0:
            return []
        qv = embed_texts([query])[0]
        res = self.collection.query(query_embeddings=[qv.tolist()],
                                    n_results=min(k, n), include=["distances"])
        hits = []
        for rank, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), 1):
            h = self._hit(cid)
            if h:
                h.dense_rank, h.dense_score = rank, 1 - dist   # cosine distance -> similarity
                hits.append(h)
        return hits

    # ---- 2. sparse ----
    def sparse_search(self, query: str, k: int = TOP_K) -> list[Hit]:
        tokens = tokenize(query)
        if self.bm25 is None or not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        hits, rank = [], 0
        for i in np.argsort(scores)[::-1][:k]:
            if scores[i] <= 0:          # no keyword overlap at all
                break
            h = self._hit(self.bm25_ids[i])
            if h:
                rank += 1
                h.sparse_rank, h.sparse_score = rank, float(scores[i])
                hits.append(h)
        return hits

    # ---- 4. reranker (cross-encoder) ----
    def rerank(self, query: str, hits: list[Hit], top_n: int = RERANK_TOP_N) -> list[Hit]:
        if not hits:
            return []
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_MODEL)
        scores = self._reranker.predict([(query, h.text) for h in hits])
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:top_n]

    # ---- full pipeline ----
    def search(self, query: str, mode: str = "hybrid", k: int = TOP_K,
               dense_weight: float = DENSE_WEIGHT, sparse_weight: float | None = None,
               rerank: bool = True, candidates: int = RERANK_CANDIDATES,
               top_n: int = RERANK_TOP_N) -> list[Hit]:
        if sparse_weight is None:
            sparse_weight = 1.0 - dense_weight

        if mode == "dense":
            hits = self.dense_search(query, k)
        elif mode == "sparse":
            hits = self.sparse_search(query, k)
        elif mode == "hybrid":
            fused = rrf_fuse(self.dense_search(query, k), self.sparse_search(query, k),
                             dense_weight, sparse_weight)
            hits = fused[:candidates]
        else:
            raise ValueError("mode must be dense, sparse or hybrid")

        return self.rerank(query, hits, top_n) if rerank else hits[:top_n]