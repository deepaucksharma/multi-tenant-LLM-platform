# Operations

This page exists to collect the few commands, endpoints, and artifact locations that matter for running the platform.
It is intentionally compact: enough to operate the project without turning into a setup manual.
Read this if you need to start services, inspect health, or understand where outputs land.

## Runtime Modes

| Mode | How it is selected | Use it for | Important detail |
| --- | --- | --- | --- |
| Hugging Face inference | Default or `INFERENCE_BACKEND=huggingface` | Local adapter-based serving | Uses one base model plus hot-swapped tenant adapters |
| Ollama inference | `INFERENCE_BACKEND=ollama` or `make serve-ollama` | AMD-friendly local serving | Model names are resolved per tenant and model type |
| Runtime-adaptive training | `DEVICE` and `USE_4BIT` env vars | SFT and DPO across CUDA, ROCm, or CPU | Falls back when 4-bit or GPU support is unavailable |

## Key Commands

| Goal | Command | What it triggers |
| --- | --- | --- |
| Generate tenant data | `make data` | Full tenant data pipeline |
| Build retrieval indexes | `make index` | Chroma and retrieval-ready assets |
| Start inference API | `make serve` | FastAPI app on port `8000` |
| Start inference with Ollama | `make serve-ollama` | API with remote Ollama backend |
| Check Ollama backend | `make check-ollama` | Availability and resolved local models |
| Run training smoke test | `make train-smoke` | Minimal SFT path validation |
| Check training environment | `make check-train-env` | Hardware and runtime capability detection |
| Run evaluation suite | `make eval` | Consolidated evaluation pipeline |
| Start monitoring UI | `make monitor` | Monitoring app on port `8502` |
| Start web UI | `make web` | Next.js dev server |
| Start mobile client | `make mobile` | Flutter run |
| Start voice server | `make voice` | Voice API and browser UI |
| Run tests | `make test` | Pytest suite |

## Where Training Results Are Saved

Training artifacts are persisted inside the project workspace by default.
They are saved locally for reuse by inference and evaluation, but they are not automatically committed to git.

| Training action | Project-local outputs |
| --- | --- |
| `make train` | SFT adapters under `models/adapters/sis/sft` and `models/adapters/mfg/sft` |
| SFT training | Tokenizer files and `training_metadata.json` inside each tenant SFT adapter folder |
| SFT registration | Version entries in `models/registry.json` |
| Full training orchestration | Run summaries in `evaluation/reports/` |
| MLflow-backed tracking | Local logs under `mlruns/` and `mlruns/local_logs/` |
| DPO training | DPO adapters under `models/adapters/<tenant>/dpo` with their own metadata |

## Interpreting Golden-Set Results

The golden set is a domain-fit metric, not just a “did the API return text?” check.
It scores in-domain answers with keyword overlap, semantic similarity, and response adequacy, and it expects domain-specific required elements to appear in the answer.
That means a very low baseline, including `0%`, can be expected when the system is still using an unfine-tuned base route.

| Scenario | Expected interpretation |
| --- | --- |
| Base route scores near `0%` | The base model is answering generically instead of using SIS/MFG terminology |
| API, RAG, and monitoring still work | The platform is operational; the failing layer is model specialization, not system plumbing |
| `make train` improves results | Expected, because `make train` builds the tenant SFT adapters that inject domain language |
| DPO improves after that | Expected, because DPO is the next alignment stage after SFT |

Important repo-specific nuance:

- `make train` runs SFT for `sis` and `mfg`.
- DPO is a separate training phase, available through the training scripts and orchestrators.
- The configured training base model is `Qwen/Qwen2.5-1.5B-Instruct`.
- The Ollama fallback resolves to a plain base model such as `qwen2.5:1.5b` unless you register tenant-specific routes.

For this project, the intended story is:

1. Base-model golden-set performance can be poor because the model is not yet domain-adapted.
2. Post-SFT performance should rise materially, with `60-80%+` being the target range described by this project’s design.
3. DPO then improves alignment details such as response preference and safer refusal behavior.

## Endpoint Families

| Family | Representative routes | Purpose |
| --- | --- | --- |
| Health and info | `GET /health`, `GET /backend/status`, `GET /tenants` | Liveness, backend mode, tenant catalog |
| Tenant detail | `GET /tenants/{tenant_id}` | Tenant domain, route, adapter, collection, topics |
| Chat | `POST /chat`, `POST /chat/stream` | Synchronous and streaming inference |
| Feedback | `POST /feedback` | User quality signal capture |
| Monitoring stats | `GET /stats`, `GET /stats/recent`, `GET /model/stats` | Runtime and model observations |
| Canary | `POST /canary/configure`, `POST /canary/promote/{tenant_id}` | Traffic shaping and promotion |
| Registry and debug | `GET /registry`, `GET /rag/test` | Registry view and RAG inspection |
| Voice | `POST /voice/process`, `GET /voice/session/{session_id}`, `WS /voice/ws` | Speech-driven interaction |

## Artifact Locations

| Path | What appears there |
| --- | --- |
| `data/raw/` | Seed documents per tenant |
| `data/<tenant>/processed/` | Cleaned document copies |
| `data/<tenant>/chunks/` | Retrieval-ready chunk files |
| `data/<tenant>/sft/` | Supervised fine-tuning datasets |
| `data/<tenant>/dpo/` | Preference datasets |
| `data/chroma/` | Vector-store persistence |
| `models/adapters/` | Tenant-specific adapter outputs |
| `models/registry.json` | Registry state and active versions |
| `logs/audit.db` | Request, latency, grounding, and feedback history |
| `logs/rollbacks.jsonl` | Rollback events |
| `evaluation/reports/` | Pipeline, training, eval, and retrain reports |
| `mlruns/` | MLflow tracking artifacts |

## CI/CD Snapshot

| Stage | What it validates |
| --- | --- |
| Lint and unit tests | Basic correctness and import safety |
| Data pipeline | Tenant data generation and chunk output |
| Evaluation suite | Safety gates and report generation |
| Module tests | Monitoring, evaluation, and registry imports |
| Training smoke | CPU-friendly adapter training path |

## Operational Priorities

| If you need to know... | Open |
| --- | --- |
| Which backend is serving | `GET /backend/status` or `inference/model_backend.py` |
| Whether a tenant route is healthy | `GET /tenants/{tenant_id}` and `GET /health` |
| Whether quality is degrading | `monitoring/dashboard.py` and `evaluation/reports/` |
| Which model version is active | `models/registry.json` or `mlops/registry.py` |
| How rollback works | `mlops/rollback.py` |

## Next Pages To Read

- [flows.md](flows.md)
- [repo-map.md](repo-map.md)
- [subsystems.md](subsystems.md)
