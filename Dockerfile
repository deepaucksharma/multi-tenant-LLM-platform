# Multi-Tenant LLM Platform — HF Spaces Deployment
#
# Multi-stage build:
#   Stage 1 (builder): Node.js builds Next.js static export → web_app/out/
#   Stage 2 (runtime): Python FastAPI serves the API + the static UI at /ui
#
# Build:
#   docker build -t multi-tenant-llm-demo .
#
# Run locally:
#   docker run -p 7860:7860 --env-file .env multi-tenant-llm-demo
#
# Deploy to HF Spaces:
#   Create a Docker Space at huggingface.co/new-space, push this repo.
#   Set HF_TOKEN in Space Settings → Repository secrets.
#
# Access:
#   API:  http://localhost:7860/health
#   UI:   http://localhost:7860/ui

# ── Stage 1: Build Next.js static export ────────────────────────────────────
FROM node:20-slim AS builder

WORKDIR /app/web_app

COPY web_app/package*.json ./
RUN npm ci --prefer-offline

COPY web_app/ ./

# Static export — no server-side rendering, no rewrites
# API calls go to /api/* which the FastAPI middleware resolves
RUN NEXT_EXPORT=true npm run build


# ── Stage 2: Python runtime ──────────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (lightweight — generation is via HF Serverless Inference API)
COPY requirements-spaces.txt ./
RUN pip install --no-cache-dir -r requirements-spaces.txt

# Application source
COPY inference/       inference/
COPY rag/             rag/
COPY tenant_data_pipeline/ tenant_data_pipeline/
COPY training/        training/
COPY monitoring/      monitoring/
COPY evaluation/      evaluation/
COPY .env.example     .env.example

# Bring in the built Next.js static export from Stage 1
COPY --from=builder /app/web_app/out/ web_app/out/

# Runtime directories
RUN mkdir -p logs data/chroma

# HF Spaces non-root user (uid 1000)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Default env — HF Inference API backend, demo mode for public access
ENV INFERENCE_BACKEND=hf_inference \
    API_PORT=7860 \
    DEMO_MODE=true \
    DEMO_TENANT=sis \
    SQLITE_DB=/app/logs/audit.db \
    CHROMA_PERSIST_DIR=/app/data/chroma

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

CMD ["python", "-m", "uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "7860"]
