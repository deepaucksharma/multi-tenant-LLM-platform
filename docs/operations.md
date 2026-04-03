# Operations

This page exists to collect the few commands, endpoints, and artifact locations that matter for running the platform.
It is intentionally compact: enough to operate the project without turning into a setup manual.
Read this if you need to start services, inspect health, or understand where outputs land.

## Runtime Modes

| Mode | How it is selected | Use it for | Important detail |
| --- | --- | --- | --- |
| HF Serverless Inference | `INFERENCE_BACKEND=hf_inference` or `make serve-hf-inference` | Zero-GPU demo / HF Spaces | Routes to HF Inference API; free tier rate-limited ~100 req/hr |
| Hugging Face local | `INFERENCE_BACKEND=hf` | Local adapter-based serving | Uses one base model plus hot-swapped tenant adapters |
| Ollama inference | `INFERENCE_BACKEND=ollama` or `make serve-ollama` | AMD-friendly local serving | Model names are resolved per tenant and model type |
| Auto | `INFERENCE_BACKEND=auto` (default) | Development | Tries Ollama → HF Inference API (if `HF_TOKEN` set) → HF local |
| Runtime-adaptive training | `DEVICE` and `USE_4BIT` env vars | SFT and DPO across CUDA, ROCm, or CPU | Falls back when 4-bit or GPU support is unavailable |

## Hugging Face Hub

All Hub operations require `HF_TOKEN` (write-scoped) in your `.env`.
Repos default to **public** (enables free HF Inference API serving). Set `HF_REPO_PRIVATE=true` to restrict.

| Goal | Command | Output |
| --- | --- | --- |
| Push adapter metadata | `make push-hub` | Config/tokenizer files only (no weights — safe, fast) |
| Push adapter + weights | `make push-hub-weights` | Adds `.safetensors` files (~200-600 MB per adapter) |
| Push merged full model | `make push-hub-merged` | Full model from `models/merged/` (~3 GB) |
| Push as private repos | `make push-hub-private` | Same as `push-hub` but creates private repos |
| Preview without uploading | `make push-hub-dry` | Dry-run, no network calls |
| Push SFT/DPO datasets | `make push-datasets` | JSONL files → dataset repos on Hub |
| Preview dataset push | `make push-datasets-dry` | Dry-run |
| Generate Colab notebook | `make generate-colab` | Writes `notebooks/train_on_colab.ipynb` |
| Build demo Docker image | `make docker-build` | Bundles API + Next.js UI for HF Spaces |
| Run demo image locally | `make docker-run` | Runs on port 7860 with `.env` vars |

## Hugging Face Jobs

This repo now includes a Hugging Face Jobs launcher that submits remote jobs against a GitHub snapshot of the repository.
Use it for ephemeral compute like tests, data generation, RAG indexing, evaluation, and training.
Use Spaces for the long-lived demo surface.

Important operational detail:

- Jobs run the remote git ref, not your uncommitted local edits.
- Jobs require paid Hugging Face credits.
- Training artifacts are ephemeral unless you push them to the Hub or mount a bucket volume.
- The launcher can mount an HF bucket with `--artifact-bucket owner/bucket` and copy selected outputs there.

| Goal | Command | Notes |
| --- | --- | --- |
| List job presets | `make hf-jobs-list` | Shows the repo-specific HF Jobs tasks and suites |
| View hardware flavors | `make hf-jobs-hardware` | Reads live hardware options from the Jobs API |
| Submit CI-style suite | `make hf-jobs-ci` | Runs tests, module checks, data pipeline, eval, training smoke, web build |
| Submit data/index/web build suite | `make hf-jobs-build` | Good for build verification without GPU training |
| Submit both SFT runs | `make hf-jobs-train` | Starts one GPU job per tenant |
| Submit both DPO runs | `make hf-jobs-align` | Starts one GPU job per tenant, each including SFT + DPO |
| Submit one preset | `make hf-job PRESET=sft-sis` | Generic escape hatch for any launcher preset |

See [huggingface-jobs.md](huggingface-jobs.md) for preset details, caveats, and examples.

Hub repo naming convention:
- Adapters: `deepaucksharma/multi-tenant-llm-{tenant}-{type}` (e.g. `…-sis-sft`)
- Merged: `deepaucksharma/multi-tenant-llm-{tenant}-{type}-merged`
- Datasets: `deepaucksharma/multi-tenant-llm-{tenant}-{type}-data`

HF Inference API model resolution order (most specific wins):
```
HF_INFERENCE_MODEL_{TENANT}_{TYPE}  →  HF_INFERENCE_MODEL_{TENANT}  →  HF_INFERENCE_MODEL  →  Qwen/Qwen2.5-1.5B-Instruct
```

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
