# Deploying to Hugging Face Spaces

This project ships a Docker-based HF Space that bundles the FastAPI inference backend and the Next.js chat UI into a single container — no local GPU required.

## What runs in the Space

| Component | Details |
|-----------|---------|
| FastAPI inference server | Port 7860, `INFERENCE_BACKEND=hf_inference` |
| HF Serverless Inference API | Generation routed to HF free tier (rate-limited) |
| Next.js chat UI | Static export served at `/ui` |
| ChromaDB RAG | CPU-based retrieval from bundled indexes |

## Quick deploy

1. **Create a Space** — go to [huggingface.co/new-space](https://huggingface.co/new-space), choose **Docker**, set it public.

2. **Link this repo** — in Space Settings → Repository, connect your GitHub fork of this project, or push directly:
   ```bash
   git remote add space https://huggingface.co/spaces/deepaucksharma/multi-tenant-llm-demo
   git push space main
   ```

3. **Set secrets** — in Space Settings → Repository secrets, add:
   | Secret | Value |
   |--------|-------|
   | `HF_TOKEN` | Your HF write token (for adapter hot-swap and API auth) |
   | `DEMO_MODE` | `true` |
   | `DEMO_TENANT` | `sis` (or `mfg`) |

4. HF auto-detects the `Dockerfile` and builds the image. Build takes ~5 minutes on first push.

## Access

| Path | What it serves |
|------|----------------|
| `/ui` | Next.js chat demo |
| `/health` | API health check |
| `/chat` | Chat endpoint (POST) |
| `/chat/stream` | Streaming chat (POST) |
| `/tenants` | List tenants |

The Next.js UI at `/ui` calls `/api/*` which the FastAPI middleware transparently resolves to the actual API routes.

## Build the image locally

```bash
# Build (requires Docker + Node.js for the multi-stage build)
make docker-build

# Run (reads HF_TOKEN and other vars from .env)
make docker-run

# Access UI: http://localhost:7860/ui
# Access API: http://localhost:7860/health
```

## Environment variables

Copy `.env.example` to `.env` and set at minimum:

```env
HF_TOKEN=hf_xxxxxxxxxxxx
INFERENCE_BACKEND=hf_inference
DEMO_MODE=true
DEMO_TENANT=sis
HF_INFERENCE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
# Point to your trained adapters once pushed:
# HF_INFERENCE_MODEL_SIS=deepaucksharma/multi-tenant-llm-sis-sft
# HF_INFERENCE_MODEL_MFG=deepaucksharma/multi-tenant-llm-mfg-sft
```

## Pushing trained adapters first

Before deploying, push your trained adapters so the Space can resolve them:

```bash
# Push adapter configs (fast, no weights)
make push-hub

# Then update your Space secrets to use the adapter repos:
# HF_INFERENCE_MODEL_SIS=deepaucksharma/multi-tenant-llm-sis-sft
```

## Rate limits

The free HF Inference API tier is rate-limited (~100 req/hr for larger models). The backend includes exponential backoff and clear error messages. For sustained load, use a dedicated Inference Endpoint.

## Updating the Space

Any push to the linked branch triggers a rebuild. For config-only changes (env vars / secrets), updates take effect immediately without a rebuild.
