# Subsystems

This page exists to map major folders to system responsibilities.
It is the shortest path from "what does this area do?" to "which code should I open first?"
Read this if you understand the platform shape and now want a module-level guide.

## Ownership Map

| Folder | Responsibility | Start with | Main outputs |
| --- | --- | --- | --- |
| `inference/` | Central runtime API, auth, routing, canary, audit, backend abstraction | `inference/app.py` | Responses, feedback, stats, audit records |
| `rag/` | Retrieval, hybrid ranking, prompt assembly, grounding | `rag/rag_chain.py` | Context chunks, citations, grounding metadata |
| `tenant_data_pipeline/` | Document generation, ingest, redaction, chunking, dataset builds | `tenant_data_pipeline/run_pipeline.py` | Cleaned data and training datasets under `data/` |
| `training/` | Runtime-adaptive model loading, SFT, DPO, merge, environment checks | `training/train_all.py` | Tenant adapters and training metadata |
| `evaluation/` | Golden sets, hallucination checks, red-team, compliance, benchmark, judge, human-eval forms | `evaluation/run_all_evals.py` | Consolidated reports under `evaluation/reports/` |
| `monitoring/` | Metrics aggregation, alerts, dashboard UI | `monitoring/dashboard.py` | Dashboard views and alert state |
| `mlops/` | Registry, rollback, retrain recommendation | `mlops/registry.py` | Version records and model control decisions |
| `voice_agent/` | HTTP and WebSocket voice experience | `voice_agent/voice_server.py` | Speech-driven requests and audio responses |
| `web_app/` | Browser UI for chat, monitoring, and compare mode | `web_app/src/app/page.tsx` | Next.js frontend |
| `mobile_app/` | Flutter mobile client | `mobile_app/lib/main.dart` | Mobile chat surface |
| `tests/` | Lightweight unit and runtime-control-flow checks | `tests/test_basic.py` | CI confidence on core contracts |
| `scripts/` | Small operational helpers | `scripts/register_ollama_models.py` | Local utility workflows |

## Client Surfaces

| Surface | What it exposes | Important detail |
| --- | --- | --- |
| Web app | Chat, monitoring panel, tenant compare view | Best first UI for engineers exploring behavior |
| Mobile app | Chat plus settings | Thin client over the same backend |
| Voice app | Voice upload and WebSocket streaming | Adds STT and TTS around the same inference flow |
| Monitoring UI | Metrics, alerts, model table | Operator-oriented view over logs and registry data |

## Cross-Cutting Concerns

| Concern | Owned mainly by | Why it shows up elsewhere |
| --- | --- | --- |
| Tenant isolation | `inference/tenant_router.py` plus `tenant_data_pipeline/config.py` | Data, prompts, adapters, and RAG collections all depend on it |
| Model routing | `inference/` | Tied to adapters, registry state, and canary behavior |
| Runtime adaptability | `training/model_loader.py` | Training and local inference need different hardware fallbacks |
| Auditability | `inference/audit_logger.py` and `logs/` | Monitoring, feedback, retraining, and evaluation all consume it |
| Quality control | `evaluation/`, `monitoring/`, `mlops/` | The platform separates evaluation, alerting, and rollout decisions |

## What A New Engineer Usually Opens First

| Goal | Open |
| --- | --- |
| Understand the API surface | `inference/app.py` |
| Understand tenant definitions | `tenant_data_pipeline/config.py` and `inference/tenant_router.py` |
| Understand RAG | `rag/rag_chain.py` |
| Understand how data is produced | `tenant_data_pipeline/run_pipeline.py` |
| Understand training orchestration | `training/train_all.py` |
| Understand ops signals | `monitoring/dashboard.py`, `mlops/retrain_trigger.py`, `mlops/rollback.py` |

## Next Pages To Read

- [repo-map.md](repo-map.md)
- [operations.md](operations.md)
- [flows.md](flows.md)
