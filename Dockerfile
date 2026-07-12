# Golden-path image: CPU (default). GPU variant lives in Dockerfile.gpu.
# Dependencies are installed FROM uv.lock (uv sync --frozen) so the image
# matches the tested, committed lockfile exactly — no floor drift. The project
# runs from source on PYTHONPATH (README/docs are .dockerignore'd), so we install
# dependencies only (--no-install-project), not the project wheel.
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir uv

# Locked dependencies only (cached layer; independent of source changes).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Pre-download embedding models (CPU providers) to avoid first-request latency.
RUN python -c "\
from fastembed import TextEmbedding; \
models=['BAAI/bge-large-en-v1.5','BAAI/bge-base-en-v1.5','BAAI/bge-base-en','sentence-transformers/all-MiniLM-L6-v2']; \
[TextEmbedding(m, providers=['CPUExecutionProvider']) for m in models]; \
print('pre-downloaded', len(models), 'models')"

# Application source (runs via PYTHONPATH; the project wheel is not built).
COPY src ./src
ENV PYTHONPATH=/app/src
COPY bin/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Runtime defaults — override at `docker run`. No secrets are baked in;
# pass QDRANT_URL / QDRANT_API_KEY / MCP_ALLOWED_HOSTS at runtime.
ENV MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=10650 \
    FASTEMBED_CUDA=false \
    QDRANT_AUTO_CREATE_COLLECTIONS=true \
    QDRANT_ENABLE_QUANTIZATION=true \
    QDRANT_HNSW_EF_CONSTRUCT=200 \
    QDRANT_HNSW_M=16 \
    PYTHONUNBUFFERED=1

EXPOSE 10650
ENTRYPOINT ["/app/entrypoint.sh"]
