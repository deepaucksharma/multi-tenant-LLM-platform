# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A **multi-tenant LLM platform POC** with two isolated tenants (SIS/Education, MFG/Manufacturing) sharing one base model (`Qwen/Qwen2.5-1.5B-Instruct`). Each tenant has its own QLoRA adapter, ChromaDB RAG index, and isolated data pipeline.

## Commands

```bash
# Environment
make setup                         # create venv + install requirements.txt
source venv/bin/activate

# Data pipeline (Phase 1)
make data                          # python -m tenant_data_pipeline.run_pipeline

# RAG index (Phase 2, requires Phase 1 chunks)
make index                         # python -m rag.build_index --force
python -m rag.build_index --list   # inspect collections

# Training (Phase 3, requires Phase 1 data)
make mlflow                        # start MLflow at localhost:5000 first
make train                         # SFT for both tenants
make dpo                           # DPO for SIS only

# Inference server (Phase 4, requires Phases 2+3)
make serve                         # uvicorn on port 8000

# Evaluation (requires running server)
make eval                          # full evaluation suite

# Clients
make web                           # Next.js on port 3000
make mobile                        # Flutter
make voice                         # Voice WebSocket on port 8001
make monitor                       # Streamlit dashboard on port 8502

# Tests
make test                          # pytest tests/
pytest tests/test_pipeline.py -k test_chunker   # single test
```

## Architecture

```
tenant_data_pipeline/   synthetic docs → ingest → PII redact → chunk → SFT/DPO datasets
rag/                    ChromaDB + BM25 hybrid retrieval, cross-encoder reranking, grounding
training/               QLoRA SFT + DPO via TRL, MLflow tracking
evaluation/             golden sets, hallucination, bias, red-team, compliance, benchmark
inference/              FastAPI + SSE streaming, adapter hot-swap, SQLite audit logging
web_app/                Next.js 14 + Tailwind, streaming chat with citations
mobile_app/             Flutter, provider state, SSE parsing
voice_agent/            faster-whisper STT + edge-tts TTS over WebSocket
monitoring/             Streamlit dashboard, psutil metrics, alert rules
mlops/                  MLflow experiment tracker, adapter registry, retrain checker
```

## Key design decisions

- **Tenant isolation**: separate raw data dirs, separate ChromaDB collections (`tenant_{id}_docs`), separate LoRA adapters per tenant
- **Adapter hot-swap**: `MultiTenantModelManager` in `inference/adapter_manager.py` loads all adapters at startup; `model.set_adapter(name)` switches without reloading the base model
- **Config single source of truth**: `tenant_data_pipeline/config.py` — all modules import `TENANTS` from here; never hardcode paths elsewhere
- **Hardware**: Ryzen 9 + 8GB VRAM. Use 4-bit NF4 quantization. If OOM during training: reduce `max_seq_length` to 256, reduce LoRA `r` to 8, set `target_modules=[q_proj, v_proj]`
- **Data flow**: `data/raw/{tenant}/` → `data/{tenant}/processed/` → `data/{tenant}/chunks/` → `data/{tenant}/sft/` and `data/{tenant}/dpo/`
- **Model artifacts**: `models/adapters/{tenant}/sft/`, `models/aligned/sis_dpo/`
- **Audit DB**: SQLite at `logs/audit.db`, tables: `requests`, `feedback`

## Reference documents

- `ref 1/` (files 1-8): high-level architecture guide (6 series)
- `ref 2/` (files 1-9): detailed implementation code (10 series)
  - `ref 2/1` → data pipeline, `ref 2/2` → RAG, `ref 2/3` → training
  - `ref 2/4` → evaluation, `ref 2/5` → inference, `ref 2/6` → web app
  - `ref 2/7` → mobile, `ref 2/8` → voice, `ref 2/9` → monitoring
