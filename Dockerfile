FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    HOME=/home/appuser \
    XDG_CACHE_HOME=/home/appuser/.cache

# Create non-root application user and data/cache directories first
RUN useradd -m -u 10001 -d /home/appuser appuser && \
    mkdir -p /app/chroma_cloud_data /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser

# Install curl and temporary build dependencies, install pinned requirements, then purge build-essential in one layer
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user BEFORE pre-caching ONNX model so it lands in /home/appuser/.cache/chroma (Fixes H11)
USER appuser

# Pre-cache open source embedding model (lightweight ONNX runtime) under appuser HOME
RUN python3 -c "import chromadb; from chromadb.utils import embedding_functions; embedding_functions.DefaultEmbeddingFunction()(['warmup'])"

# Copy application source code and web UI with appuser ownership
COPY --chown=appuser:appuser . .

# Standard Render / Cloud Port
EXPOSE 10000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health/live || exit 1

# Launch server
CMD ["python3", "app.py"]
