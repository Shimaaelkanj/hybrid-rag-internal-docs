FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf_cache ANONYMIZED_TELEMETRY=False

# CPU-only PyTorch keeps the image much smaller
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# The Chroma client must match the Chroma server image version (set CHROMA_VERSION in .env)
ARG CHROMA_VERSION
RUN pip install "chromadb==${CHROMA_VERSION}"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the two local models into the image so the container never downloads them at runtime
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app ./app
COPY sample_docs ./sample_docs

EXPOSE 8000

# Seed the sample documents on first start (skipped when the index is already populated)
CMD ["sh", "-c", "if [ \"$SEED_ON_START\" = \"true\" ]; then python -m app.seed; fi; exec uvicorn app.api:app --host 0.0.0.0 --port 8000"]
