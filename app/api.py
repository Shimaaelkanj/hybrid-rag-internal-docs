from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.chunking import STRATEGIES
from app.config import DENSE_WEIGHT, DUPLICATES_FILE, RAW_DIR, TOP_K
from app.generation import answer_question
from app.indexing import load_chunk_rows
from app.ingest import ingest_path
from app.loaders import LOADERS
from app.retrieval import Retriever

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

state: dict = {}
ingest_lock = threading.Lock()        # only one ingestion may write the indexes at a time


# ------------------------------------------------------------------ response / request models
class Source(BaseModel):
    ref: int = Field(description="The number used in the answer, e.g. 2 for [2].")
    source: str = Field(description="File name of the document.")
    heading: str = Field("", description="Section heading the chunk came from.")
    page: int | None = Field(None, description="Page number (PDF files only).")
    rerank_score: float | None = Field(None, description="Cross-encoder relevance score.")
    text: str = Field(description="The retrieved chunk text.")


class CitationCheck(BaseModel):
    claim: str
    citations: list[int]
    verdict: str = Field(description="supported | partial | unsupported | uncited | invalid_reference | gap")
    reason: str


class Confidence(BaseModel):
    retrieval: float = Field(description="How relevant the retrieved chunks were (0-1).")
    citation_coverage: float = Field(description="Share of claims with a verified citation (0-1).")
    completeness: float = Field(description="Whether every part of the question was answered (0-1).")
    composite: float = Field(description="Weighted combination of the three scores (0-1).")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000,
                          examples=["How many vacation days do employees get?"])
    verify: bool = Field(True, description="Check every citation with an LLM judge (slower, more trustworthy).")
    retrieval_mode: Literal["hybrid", "dense"] = Field("hybrid", description="Hybrid (dense + BM25) or dense-only retrieval.")
    dense_weight: float = Field(DENSE_WEIGHT, ge=0.0, le=1.0, description="Hybrid fusion weight for dense results (0.7 = 70% dense / 30% sparse).")
    rerank: bool = Field(True, description="Rerank candidates with the cross-encoder.")


class AskResponse(BaseModel):
    status: Literal["answered", "insufficient_context"]
    question: str
    answer: str = ""
    retrieval_mode: str
    retrieval_confidence: float
    confidence: Confidence | None = Field(None, description="Present when verify=true and an answer was generated.")
    sources: list[Source]
    citation_checks: list[CitationCheck] = []
    warnings: list[str] = []
    found: str = Field("", description="'I don't know' case: what the system did find.")
    not_found: str = Field("", description="'I don't know' case: what it could not find.")
    suggested_documents: list[str] = Field([], description="'I don't know' case: documents worth checking manually.")
    latency_ms: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, examples=["ERR-4021"])
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    k: int = Field(TOP_K, ge=1, le=50, description="Candidates fetched from each retriever.")
    dense_weight: float = Field(DENSE_WEIGHT, ge=0.0, le=1.0)
    rerank: bool = True
    top_n: int = Field(5, ge=1, le=20, description="Results returned.")


class SearchHit(BaseModel):
    id: str
    text: str
    source: str
    heading: str = ""
    page: int | None = None
    strategy: str
    dense_rank: int | None = None
    dense_score: float | None = Field(None, description="Cosine similarity.")
    sparse_rank: int | None = None
    sparse_score: float | None = Field(None, description="BM25 score.")
    rrf_score: float | None = Field(None, description="Weighted reciprocal-rank-fusion score.")
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchHit]
    latency_ms: int


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    characters: int
    strategies: list[str]
    sections: list[str]


class DocumentsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: list[DocumentInfo]


class IngestFileResult(BaseModel):
    source: str
    chunks_indexed: int
    duplicates_skipped: int


class IngestResponse(BaseModel):
    strategy: str
    files: list[IngestFileResult]
    total_chunks_in_index: int


# ------------------------------------------------------------------ app
@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = await asyncio.to_thread(Retriever)     # load the indexes once at startup
    yield
    state.clear()


app = FastAPI(
    title="Hybrid RAG API",
    version="1.0.0",
    description=("Question answering over internal documentation with hybrid (dense + BM25) retrieval, "
                 "cross-encoder reranking, grounded answers with verified citations and confidence scores."),
    lifespan=lifespan,
)
router = APIRouter(prefix="/v1")


def _count_duplicates() -> int:
    if not DUPLICATES_FILE.exists():
        return 0
    return sum(1 for line in DUPLICATES_FILE.read_text(encoding="utf-8").splitlines() if line.strip())


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "Hybrid RAG API", "docs": "/docs", "health": "/v1/health"}


@router.get("/health", summary="Service and index status")
async def health():
    r = state["retriever"]
    return {"status": "ok", "chunks": len(r.rows), "chroma_count": r.collection.count()}


@router.post("/ask", response_model=AskResponse, summary="Ask a question")
async def ask(req: AskRequest):
    """Returns a grounded answer with `[n]` citations, the cited sources, per-claim citation checks
    and confidence scores. When the documents do not contain enough evidence, `status` is
    `insufficient_context` and the response explains what was and was not found."""
    start = time.perf_counter()
    try:
        result = await asyncio.to_thread(
            answer_question, req.question, state["retriever"], req.verify,
            mode=req.retrieval_mode, rerank=req.rerank, dense_weight=req.dense_weight)
    except RuntimeError as e:                    # e.g. the LLM server is not reachable
        raise HTTPException(status_code=502, detail=str(e))
    return AskResponse(**asdict(result), retrieval_mode=req.retrieval_mode,
                       latency_ms=int((time.perf_counter() - start) * 1000))


@router.post("/search", response_model=SearchResponse, summary="Retrieval only (no LLM)")
async def search(req: SearchRequest):
    """Returns ranked chunks with the dense rank, BM25 rank, fusion score and rerank score."""
    start = time.perf_counter()
    retriever = state["retriever"]
    hits = await asyncio.to_thread(
        lambda: retriever.search(req.query, mode=req.mode, k=req.k, dense_weight=req.dense_weight,
                                 rerank=req.rerank, top_n=req.top_n))
    return SearchResponse(query=req.query, mode=req.mode, results=[asdict(h) for h in hits],
                          latency_ms=int((time.perf_counter() - start) * 1000))


@router.get("/documents", response_model=DocumentsResponse, summary="List indexed documents")
async def documents():
    docs: dict[str, dict] = {}
    for row in load_chunk_rows():
        d = docs.setdefault(row["source"], {"source": row["source"], "chunks": 0, "characters": 0,
                                            "strategies": set(), "sections": []})
        d["chunks"] += 1
        d["characters"] += len(row["text"])
        d["strategies"].add(row["strategy"])
        if row["heading"] and row["heading"] not in d["sections"]:
            d["sections"].append(row["heading"])
    items = [DocumentInfo(**{**d, "strategies": sorted(d["strategies"])}) for d in docs.values()]
    return DocumentsResponse(total_documents=len(items), total_chunks=sum(i.chunks for i in items),
                             documents=sorted(items, key=lambda i: i.source))


@router.post("/ingest", response_model=IngestResponse, summary="Upload documents for indexing")
async def ingest(files: list[UploadFile] = File(..., description="Markdown, text, HTML or PDF files."),
                 strategy: str = Form("recursive", description="fixed | recursive | semantic")):
    """Uploading a file whose name is already indexed replaces its chunks.
    Near-duplicate chunks (cosine similarity > 0.95) are skipped and counted."""
    if strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"strategy must be one of {list(STRATEGIES)}")

    saved: list[Path] = []
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in LOADERS:
            raise HTTPException(status_code=415,
                                detail=f"Unsupported file type: '{name}'. Allowed: {sorted(LOADERS)}")
        data = await f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"'{name}' is larger than 20 MB.")
        dest = RAW_DIR / name
        dest.write_bytes(data)
        saved.append(dest)

    def work() -> list[IngestFileResult]:
        with ingest_lock:
            results = []
            for path in saved:
                before = _count_duplicates()
                ingest_path(path, strategy)
                kept = sum(1 for r in load_chunk_rows() if r["source"] == path.name)
                results.append(IngestFileResult(source=path.name, chunks_indexed=kept,
                                                duplicates_skipped=_count_duplicates() - before))
            state["retriever"].reload()           # pick up the new indexes
            return results

    results = await asyncio.to_thread(work)
    return IngestResponse(strategy=strategy, files=results,
                          total_chunks_in_index=len(state["retriever"].rows))


app.include_router(router)
