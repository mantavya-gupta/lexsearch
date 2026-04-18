FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    PyMuPDF \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    qdrant-client \
    sentence-transformers \
    rank-bm25 \
    groq \
    cohere \
    pydantic

# Copy app code
COPY app/ ./app/
COPY data/ ./data/

# Create cache directory
RUN mkdir -p data/processed/.cache

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
