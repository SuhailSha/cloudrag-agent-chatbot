FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     curl     && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache open source embedding model (lightweight ONNX runtime)
RUN python3 -c "import chromadb; from chromadb.utils import embedding_functions; embedding_functions.DefaultEmbeddingFunction()(['warmup'])"

# Copy application source code and web UI
COPY . .

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Launch server
CMD ["python3", "app.py"]
