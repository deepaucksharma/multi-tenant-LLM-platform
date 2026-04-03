# Running This Repo On Hugging Face Jobs

This project already had a Hugging Face-facing deployment story through Spaces and Hub publishing.
It now also has a repo-native Hugging Face Jobs launcher so the heavier pipeline work can move off the local machine.

Use Jobs for:

- tests and import checks
- tenant data generation
- RAG index builds
- evaluation runs
- SFT and DPO training
- Next.js production builds

Use Spaces for:

- the persistent demo UI
- the long-lived inference surface
- public interactive access

## Why The Launcher Exists

Hugging Face Jobs execute on remote infrastructure, not against your live local workspace.
That means a reusable launcher has to answer three things consistently:

1. Which repo snapshot should the job run?
2. Which install/runtime steps belong to each task?
3. Where should artifacts go after the job ends?

The launcher solves that by:

- resolving the repo remote from `origin`
- defaulting to the current `HEAD` ref
- downloading a GitHub snapshot of that ref inside the job
- running a task-specific preset
- optionally copying selected outputs into a mounted HF bucket

If your worktree is dirty, the launcher prints a warning because those uncommitted edits are not part of the remote snapshot.

## Requirements

- A Hugging Face account with Jobs access and credits
- Local authentication via `hf auth login`
- A GitHub remote for this repo that the job can fetch

For live Jobs behavior and hardware details, see the official docs:

- https://huggingface.co/docs/hub/jobs-overview
- https://huggingface.co/docs/hub/jobs-configuration
- https://huggingface.co/docs/hub/jobs-popular-images

## Quick Commands

```bash
# Show available presets and suites
make hf-jobs-list

# View live hardware flavors
make hf-jobs-hardware

# Submit the CI-style remote suite
make hf-jobs-ci

# Submit one remote training job
make hf-job PRESET=sft-sis

# Preview the generated command without submitting
python scripts/hf_jobs.py submit sft-sis --dry-run
```

## Presets

| Preset | Default hardware | What it does |
| --- | --- | --- |
| `unit-tests` | `cpu-upgrade` | Installs the CPU job deps and runs `pytest tests/` |
| `module-tests` | `cpu-upgrade` | Mirrors the import-level monitoring/eval checks from CI |
| `data-pipeline` | `cpu-upgrade` | Runs the full tenant data pipeline and validates chunk counts |
| `rag-index` | `cpu-performance` | Regenerates tenant data and builds the Chroma indexes |
| `evaluation` | `cpu-performance` | Runs the consolidated evaluation suite |
| `training-smoke` | `cpu-performance` | Runs the adaptive CPU smoke path for SFT |
| `sft-sis` | `a10g-large` | Full SIS SFT run |
| `sft-mfg` | `a10g-large` | Full MFG SFT run |
| `dpo-sis` | `a10g-large` | Self-contained SIS SFT + DPO run |
| `dpo-mfg` | `a10g-large` | Self-contained MFG SFT + DPO run |
| `web-build` | `cpu-upgrade` | Runs `npm ci` and `npm run build` for the Next.js app |

## Suites

| Suite | Presets |
| --- | --- |
| `ci` | `unit-tests`, `module-tests`, `data-pipeline`, `evaluation`, `training-smoke`, `web-build` |
| `build-all` | `data-pipeline`, `rag-index`, `web-build` |
| `train-all` | `sft-sis`, `sft-mfg` |
| `align-all` | `dpo-sis`, `dpo-mfg` |

## Artifact Persistence

Jobs are ephemeral.
If you want outputs to survive after the container exits, use one of these patterns:

1. Push final model artifacts to the Hugging Face Hub after training.
2. Mount an HF bucket and let the launcher copy selected artifact paths there.

Example:

```bash
python scripts/hf_jobs.py submit sft-sis \
  --artifact-bucket deepaucksharma/multi-tenant-llm-artifacts
```

When a bucket is mounted, the launcher copies the preset’s key outputs into:

```text
/job-artifacts/<JOB_ID>/<preset-name>/
```

## Useful Follow-Up Commands

```bash
# List recent jobs
python scripts/hf_jobs.py ps

# Inspect one job
python scripts/hf_jobs.py inspect <job_id>

# Stream logs
python scripts/hf_jobs.py logs <job_id> --follow

# Cancel a run
python scripts/hf_jobs.py cancel <job_id>
```

## Practical Guidance

- Run `ci` on CPU first to move the current GitHub Actions-style checks onto Hugging Face.
- Use `train-all` once the data and evaluation path is stable.
- Treat Jobs as the compute plane and Spaces as the serving plane.
- For long training runs, prefer mounting a bucket or following training with a Hub push step so the outputs are not stranded in an ephemeral container.
