FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache open source embedding model (lightweight ONNX runtime)
RUN python3 -c "import chromadb; from chromadb.utils import embedding_functions; embedding_functions.DefaultEmbeddingFunction()(['warmup'])"

# Create non-root application user and data directories
RUN useradd -m -u 10001 appuser && \
    mkdir -p /app/chroma_cloud_data && \
    chown -R appuser:appuser /app

# Copy application source code and web UI
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Standard Render / Cloud Port
EXPOSE 10000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health/live || exit 1

# Launch server
CMD ["python3", "app.py"]
