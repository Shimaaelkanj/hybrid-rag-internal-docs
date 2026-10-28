
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# ---- Phase 3: generation ----
# Default = local Ollama. Any OpenAI-compatible provider works (see .env below).
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 180))

# Below this retrieval confidence (0-1) the system answers "I don't know"
MIN_RETRIEVAL_CONFIDENCE = float(os.getenv("MIN_RETRIEVAL_CONFIDENCE", 0.30))

# Composite confidence weights (sum to 1)
W_RETRIEVAL = 0.4
W_CITATION = 0.4
W_COMPLETENESS = 0.2
# ---- Phase 2: retrieval ----
TOP_K = int(os.getenv("TOP_K", 10))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.7))     # sparse weight = 1 - dense
RRF_K = 60                                               # standard RRF constant
RERANK_CANDIDATES = 20                                   # fused candidates sent to reranker
RERANK_TOP_N = 5                                         # final results kept
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # free, ~90 MB, downloads once
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR") or ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"        # source of truth for both indexes
BM25_FILE = INDEX_DIR / "bm25.pkl"
DUPLICATES_FILE = INDEX_DIR / "duplicates.jsonl"

for d in (RAW_DIR, PROCESSED_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")   # "local" (free) or "openai"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "internal_docs"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))        # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", 0.95))
SEMANTIC_BREAK_PERCENTILE = float(os.getenv("SEMANTIC_BREAK_PERCENTILE", 90))
SEMANTIC_MIN_CHARS = 300


# ---- Phase 4/5 additions ----
# The judge model can differ from the generator (a stronger judge gives more trustworthy scores).
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", LLM_BASE_URL)
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", LLM_API_KEY)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)
CACHE_FILE = ROOT / "eval" / ".llm_cache.jsonl"

# Chroma: leave CHROMA_HOST empty for local files; docker-compose sets it to use a Chroma server.
CHROMA_HOST = os.getenv("CHROMA_HOST", "")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
