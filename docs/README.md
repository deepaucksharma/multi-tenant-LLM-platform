# Documentation Hub

This page exists to make the repository easy to browse in GitHub without reading the whole codebase first.
It organizes the project around system behavior, not around implementation trivia.
Read this if you are new to the repo and want to understand what the platform does before opening files.

## Start Here

1. Read [architecture.md](architecture.md) for the big picture.
2. Read [flows.md](flows.md) to see how requests, data, and models move through the system.
3. Read [repo-map.md](repo-map.md) when you are ready to open code.

## Start Here By Audience

| If you are... | Read first | Then read |
| --- | --- | --- |
| New engineer | [architecture.md](architecture.md) | [repo-map.md](repo-map.md) |
| Contributor changing behavior | [flows.md](flows.md) | [subsystems.md](subsystems.md) |
| Operator or reviewer | [operations.md](operations.md) | [flows.md](flows.md) |
| Product or demo reviewer | [architecture.md](architecture.md) | [flows.md](flows.md) |

## Documentation Map

| Page | Purpose | Open this when you need |
| --- | --- | --- |
| [architecture.md](architecture.md) | Flagship high-level architecture | One mental model for the whole system |
| [flows.md](flows.md) | Small diagrams for runtime and lifecycle flows | Request path, data path, training path, quality loop, voice path |
| [subsystems.md](subsystems.md) | Condensed module and ownership map | Which folder owns which concern |
| [operations.md](operations.md) | Commands, endpoints, artifacts, CI/CD | Running, inspecting, or operating the platform |
| [repo-map.md](repo-map.md) | High-signal directory guide | Where to start reading and what to ignore at first |

## System Surfaces

| Surface | Primary user | Backed by | What it is for |
| --- | --- | --- | --- |
| Inference API | All clients | `inference/` | Main chat, streaming, routing, feedback, stats, and canary runtime |
| Web app | Engineers, reviewers | `web_app/` | Browser chat, monitoring, and side-by-side tenant comparison |
| Mobile app | End users, demos | `mobile_app/` | Lightweight Flutter chat client |
| Voice app | End users, demos | `voice_agent/` | Speech input/output over HTTP and WebSocket |
| Monitoring UI | Operators | `monitoring/` | Dashboard for metrics, alerts, and model health |

## Platform At A Glance

| Area | What lives there | Main output |
| --- | --- | --- |
| Tenant data pipeline | Synthetic docs, ingest, PII redaction, chunking, dataset builders | Tenant datasets and cleaned chunks under `data/` |
| RAG | Retrieval, hybrid ranking, prompt construction, grounding | Context, citations, and grounded prompts |
| Training | SFT, DPO, adapter merge, runtime-adaptive model loading | Tenant adapters and training metadata |
| Inference | Auth, routing, canary, audit logging, backend abstraction | Tenant-aware responses and feedback records |
| Evaluation | Golden sets, hallucination, red-team, compliance, benchmark, judge flows | Reports under `evaluation/reports/` |
| MLOps | Registry, rollback, retrain recommendation | Version control and promotion decisions for models |

## Evaluation Context

| Observation | How to interpret it |
| --- | --- |
| Base-model golden-set score is very low, even `0%` | Usually expected before tenant fine-tuning because the rubric checks domain-specific required elements |
| The platform still runs end to end | This is a model-adaptation gap, not a platform-completeness gap |
| `make train` raises golden-set scores | Expected because it builds tenant-specific SFT adapters |
| DPO comes after SFT | It is the preference/alignment layer, not the first domain-adaptation step |

## Subsystem Index

| Subsystem | Primary folder | Why it matters first |
| --- | --- | --- |
| Inference runtime | `inference/` | It is the central API and routing layer |
| Retrieval and grounding | `rag/` | It explains why answers are cited and tenant-scoped |
| Tenant data pipeline | `tenant_data_pipeline/` | It explains where domain knowledge comes from |
| Training | `training/` | It explains how `base`, `sft`, and `dpo` variants appear |
| Evaluation and safety | `evaluation/` | It explains how quality is measured beyond unit tests |
| Monitoring and MLOps | `monitoring/`, `mlops/` | They explain production visibility and model decisions |
| Product surfaces | `web_app/`, `mobile_app/`, `voice_agent/` | They show how users reach the same backend |

## Supplemental Reference Material

| File | Use it for |
| --- | --- |
| [`../AMD_GPU_SETUP.md`](../AMD_GPU_SETUP.md) | Local ROCm and AMD setup details |
| [`../DEPLOYMENT_COMPLETE.md`](../DEPLOYMENT_COMPLETE.md) | Repository state and rollout summary |
| [`../CODE_REVIEW.md`](../CODE_REVIEW.md) | Review notes for the runtime-adaptive changes |

## Next Pages To Read

- [architecture.md](architecture.md)
- [flows.md](flows.md)
- [repo-map.md](repo-map.md)
