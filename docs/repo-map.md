# Repo Map

This page exists to help a newcomer decide where to click next.
It is a directory guide, not a restatement of every implementation detail.
Read this if you are about to browse the repo tree and want to avoid opening low-value files first.

## Read Order

1. `README.md`
2. `docs/architecture.md`
3. `docs/flows.md`
4. `inference/app.py`
5. `tenant_data_pipeline/config.py`
6. `rag/rag_chain.py`

## Top-Level Directory Guide

| Path | What it contains | Open early? | Reason |
| --- | --- | --- | --- |
| `docs/` | Project-level architecture and navigation docs | Yes | Best GitHub-first entry point |
| `inference/` | Main runtime API and tenant-aware serving | Yes | Center of the platform |
| `rag/` | Retrieval and grounding stack | Yes | Explains answer quality and citations |
| `tenant_data_pipeline/` | Source-data processing and dataset generation | Yes | Explains where tenant knowledge comes from |
| `training/` | Adapter training and runtime adaptability | Yes | Explains how model variants are produced |
| `evaluation/` | Quality, safety, and benchmark workflows | Yes | Explains how model behavior is judged |
| `monitoring/` | Metrics and alerting surfaces | Soon | Explains observability and operator view |
| `mlops/` | Registry, rollback, and retraining logic | Soon | Explains rollout and recovery decisions |
| `voice_agent/` | Voice-specific wrapper around inference | Soon | Useful after you understand the core API |
| `web_app/` | Next.js product surface | Soon | Best UI for exploring behavior |
| `mobile_app/` | Flutter product surface | Later | Thin client over the same backend |
| `tests/` | Unit and control-flow checks | Soon | Fastest way to see what the repo cares about |
| `scripts/` | Utility helpers | Later | Useful once you know the main workflows |
| `.github/` | CI workflow definitions | Later | Useful for pipeline and enforcement context |
| `data/` | Generated tenant datasets and indexes | Later | Output, not the best first reading |
| `models/` | Base model, adapters, merged outputs, registry | Later | Mostly artifacts and state |
| `logs/` | Audit and rollback records | Later | Runtime output |
| `mlruns/` | MLflow experiment artifacts | Later | Generated tracking data |
| `.kilo/` | Local planning artifacts | Ignore at first | Not part of the product runtime |
| `ref 1/`, `ref 2/` | Reference material | Ignore at first | Support material, not core code paths |

## What To Open For Common Questions

| Question | Best first file |
| --- | --- |
| How does a user request get served? | `inference/app.py` |
| Where are tenants defined? | `tenant_data_pipeline/config.py` |
| How is tenant routing enforced? | `inference/tenant_router.py` |
| How does RAG work here? | `rag/rag_chain.py` |
| How are adapters trained? | `training/train_all.py` and `training/sft_train.py` |
| How do we judge model quality? | `evaluation/run_all_evals.py` |
| How do we observe production behavior? | `monitoring/dashboard.py` |
| How do we promote or roll back models? | `mlops/registry.py` and `mlops/rollback.py` |

## Generated Vs Source

| Mostly source code | Mostly generated or stateful |
| --- | --- |
| `inference/` | `data/` |
| `rag/` | `models/` |
| `tenant_data_pipeline/` | `logs/` |
| `training/` | `mlruns/` |
| `evaluation/` | `evaluation/reports/` |
| `monitoring/` |  |

Training outputs are expected to accumulate in the project under `models/`, `evaluation/reports/`, and `mlruns/`; they are local artifacts, not auto-committed source files.

## Next Pages To Read

- [subsystems.md](subsystems.md)
- [operations.md](operations.md)
- [architecture.md](architecture.md)
