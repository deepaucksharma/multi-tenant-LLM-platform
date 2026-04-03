# Multi-Tenant LLM Platform — HF Spaces Deployment
#
# Runs the FastAPI inference server on port 7860 (HF Spaces convention).
# Uses HF Serverless Inference API for generation — no local GPU needed.
#
# Build:
#   docker build -t multi-tenant-llm-demo .
#
# Run locally:
#   docker run -p 7860:7860 -e HF_TOKEN=hf_xxx multi-tenant-llm-demo
#
# Deploy to HF Spaces:
#   Create a Docker Space at huggingface.co/new-space, link this repo.
#   Set HF_TOKEN in Space Settings → Repository secrets.

FROM python:3.11-slim

# System deps (chromadb needs sqlite, presidio needs spacy data path)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# Exclude heavy ML packages not needed for API-only + HF Inference backend
COPY requirements-spaces.txt ./
RUN pip install --no-cache-dir -r requirements-spaces.txt

# Copy application source
COPY inference/       inference/
COPY rag/             rag/
COPY tenant_data_pipeline/ tenant_data_pipeline/
COPY training/        training/
COPY monitoring/      monitoring/
COPY evaluation/      evaluation/

# Copy config files
COPY .env.example     .env.example

# Create required runtime directories
RUN mkdir -p logs data/chroma

# HF Spaces: port 7860, non-root user (uid 1000)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Default env — uses HF Inference API, no local model loading
ENV INFERENCE_BACKEND=hf_inference \
    API_PORT=7860 \
    DEMO_MODE=true \
    DEMO_TENANT=sis \
    SQLITE_DB=/app/logs/audit.db \
    CHROMA_PERSIST_DIR=/app/data/chroma

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

CMD ["python", "-m", "uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "7860"]
