"""
Hugging Face Jobs launcher for this repository.

Runs repo presets on Hugging Face infrastructure by submitting Docker-based jobs
that download a GitHub snapshot of the repository at a specific ref.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import quote

from dotenv import load_dotenv
from huggingface_hub import HfApi, Volume, get_token

load_dotenv()

DEFAULT_WORKDIR = "/workspace/repo"
DEFAULT_ARTIFACT_MOUNT = "/job-artifacts"


@dataclass(frozen=True)
class JobPreset:
    """A reproducible Hugging Face Jobs task for this repo."""

    name: str
    description: str
    image: str
    flavor: str
    timeout: str
    install_commands: Sequence[str]
    run_command: str
    env: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)
    artifact_paths: Sequence[str] = field(default_factory=tuple)
    requires_hf_token: bool = False


def _sh(value: str) -> str:
    return shlex.quote(value)


def _dedent(script: str) -> str:
    return textwrap.dedent(script).strip()


PRESETS: Dict[str, JobPreset] = {
    "unit-tests": JobPreset(
        name="unit-tests",
        description="Run the pytest suite used by local development and CI.",
        image="python:3.11-slim",
        flavor="cpu-upgrade",
        timeout="45m",
        install_commands=(
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements-hf-jobs.txt",
            "python -m pip install pytest ruff",
            (
                "python -c \"import nltk; "
                "nltk.download('punkt', quiet=True); "
                "nltk.download('stopwords', quiet=True)\""
            ),
        ),
        run_command="python -m pytest tests/ -v --tb=short",
    ),
    "module-tests": JobPreset(
        name="module-tests",
        description="Mirror the import-level GitHub Actions checks on HF Jobs.",
        image="python:3.11-slim",
        flavor="cpu-upgrade",
        timeout="45m",
        install_commands=(
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements-hf-jobs.txt",
            (
                "python -c \"import nltk; "
                "nltk.download('punkt', quiet=True); "
                "nltk.download('stopwords', quiet=True)\""
            ),
        ),
        run_command=_dedent(
            """
            python - <<'PY'
            from monitoring.metrics_collector import MetricsCollector
            from monitoring.alerting import ALERT_RULES
            from mlops.retrain_trigger import check_retrain_needed
            from mlops.rollback import list_versions
            from mlops.registry import ModelRegistry
            from evaluation.eval_config import compute_keyword_overlap
            from evaluation.red_team import ADVERSARIAL_TESTS
            from evaluation.compliance_test import COMPLIANCE_TESTS

            mc = MetricsCollector(audit_db_path='/tmp/nonexistent.db')
            system = mc.collect_system_metrics()
            print(f'CPU={system.cpu_percent}%, MEM={system.memory_percent}%')
            print(f'Alert rules: {len(ALERT_RULES)}')
            _ = check_retrain_needed('sis')
            _ = list_versions('/tmp/test_registry.json', 'sis')

            reg = ModelRegistry('/tmp/test_registry.json')
            summary = reg.get_summary()
            print(f'Registry summary: {summary}')

            score = compute_keyword_overlap(
                'proof of residency and birth certificate',
                ['proof of residency', 'birth certificate'],
            )
            assert score == 1.0, f'Expected 1.0, got {score}'
            assert 'sis' in ADVERSARIAL_TESTS and 'mfg' in ADVERSARIAL_TESTS
            assert 'sis' in COMPLIANCE_TESTS and 'mfg' in COMPLIANCE_TESTS
            print('Module checks passed')
            PY
            """
        ),
    ),
    "data-pipeline": JobPreset(
        name="data-pipeline",
        description="Run the full tenant data pipeline and persist reports.",
        image="python:3.11-slim",
        flavor="cpu-upgrade",
        timeout="90m",
        install_commands=(
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements-hf-jobs.txt",
            (
                "python -c \"import nltk; "
                "nltk.download('punkt', quiet=True); "
                "nltk.download('stopwords', quiet=True)\""
            ),
        ),
        run_command=_dedent(
            """
            python -m tenant_data_pipeline.run_pipeline
            python - <<'PY'
            import json
            from pathlib import Path

            for tenant_id in ('sis', 'mfg'):
                chunks = json.loads(Path(f'data/{tenant_id}/chunks/chunks.json').read_text())
                print(f'{tenant_id} chunks: {len(chunks)}')
                if len(chunks) < 20:
                    raise SystemExit(f'{tenant_id} has too few chunks: {len(chunks)}')
            print('Chunk count check passed')
            PY
            """
        ),
        artifact_paths=("data/sis", "data/mfg", "evaluation/reports"),
    ),
    "rag-index": JobPreset(
        name="rag-index",
        description="Generate tenant data and build the Chroma-backed RAG indexes.",
        image="python:3.11-slim",
        flavor="cpu-performance",
        timeout="2h",
        install_commands=(
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements-hf-jobs.txt",
            (
                "python -c \"import nltk; "
                "nltk.download('punkt', quiet=True); "
                "nltk.download('stopwords', quiet=True)\""
            ),
        ),
        run_command=_dedent(
            """
            python -m tenant_data_pipeline.run_pipeline
            python -m rag.build_index --force
            """
        ),
        artifact_paths=("data/chroma", "evaluation/reports", "data/sis", "data/mfg"),
    ),
    "evaluation": JobPreset(
        name="evaluation",
        description="Run the consolidated evaluation suite on HF Jobs.",
        image="python:3.11-slim",
        flavor="cpu-performance",
        timeout="90m",
        install_commands=(
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements-hf-jobs.txt",
            (
                "python -c \"import nltk; "
                "nltk.download('punkt', quiet=True); "
                "nltk.download('stopwords', quiet=True)\""
            ),
        ),
        run_command=_dedent(
            """
            python -m tenant_data_pipeline.run_pipeline
            python evaluation/run_all_evals.py --model-version hf-jobs-baseline
            """
        ),
        artifact_paths=("evaluation/reports", "data/sis", "data/mfg"),
    ),
    "training-smoke": JobPreset(
        name="training-smoke",
        description="CPU-safe smoke test for the adaptive SFT path.",
        image="huggingface/trl",
        flavor="cpu-performance",
        timeout="2h",
        install_commands=(
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements.txt",
            (
                "python -c \"import nltk; "
                "nltk.download('punkt', quiet=True); "
                "nltk.download('stopwords', quiet=True)\""
            ),
        ),
        env={
            "DEVICE": "cpu",
            "USE_4BIT": "false",
            "SMOKE_TEST_BASE_MODEL": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "MLFLOW_TRACKING_URI": "file:./mlruns-ci",
        },
        run_command=_dedent(
            """
            python -m tenant_data_pipeline.run_pipeline
            python -m training.check_env
            python -m training.sft_train --tenant sis --smoke-test --max-steps 1
            python - <<'PY'
            import json
            from pathlib import Path

            metadata = json.loads(Path('models/adapters/sis/sft/training_metadata.json').read_text())
            assert metadata['tenant_id'] == 'sis'
            assert metadata['model_type'] == 'sft'
            print('Smoke-test artifacts validated')
            PY
            """
        ),
        artifact_paths=("models/adapters/sis/sft", "models/registry.json", "mlruns-ci", "evaluation/reports"),
    ),
    "sft-sis": JobPreset(
        name="sft-sis",
        description="Full SIS SFT training on a single GPU job.",
        image="huggingface/trl",
        flavor="a10g-large",
        timeout="8h",
        install_commands=(
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements.txt",
            (
                "python -c \"import nltk; "
                "nltk.download('punkt', quiet=True); "
                "nltk.download('stopwords', quiet=True)\""
            ),
        ),
        run_command=_dedent(
            """
            python -m tenant_data_pipeline.run_pipeline
            python -m training.sft_train --tenant sis
            """
        ),
        artifact_paths=("models/adapters/sis/sft", "models/registry.json", "mlruns", "evaluation/reports"),
    ),
    "sft-mfg": JobPreset(
        name="sft-mfg",
        description="Full MFG SFT training on a single GPU job.",
        image="huggingface/trl",
        flavor="a10g-large",
        timeout="8h",
        install_commands=(
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements.txt",
            (
                "python -c \"import nltk; "
                "nltk.download('punkt', quiet=True); "
                "nltk.download('stopwords', quiet=True)\""
            ),
        ),
        run_command=_dedent(
            """
            python -m tenant_data_pipeline.run_pipeline
            python -m training.sft_train --tenant mfg
            """
        ),
        artifact_paths=("models/adapters/mfg/sft", "models/registry.json", "mlruns", "evaluation/reports"),
    ),
    "dpo-sis": JobPreset(
        name="dpo-sis",
        description="Self-contained SIS SFT + DPO alignment run.",
        image="huggingface/trl",
        flavor="a10g-large",
        timeout="10h",
        install_commands=(
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements.txt",
            (
                "python -c \"import nltk; "
                "nltk.download('punkt', quiet=True); "
                "nltk.download('stopwords', quiet=True)\""
            ),
        ),
        run_command=_dedent(
            """
            python -m tenant_data_pipeline.run_pipeline
            python -m training.sft_train --tenant sis
            python -m training.dpo_train --tenant sis
            """
        ),
        artifact_paths=("models/adapters/sis/sft", "models/adapters/sis/dpo", "models/registry.json", "mlruns", "evaluation/reports"),
    ),
    "dpo-mfg": JobPreset(
        name="dpo-mfg",
        description="Self-contained MFG SFT + DPO alignment run.",
        image="huggingface/trl",
        flavor="a10g-large",
        timeout="10h",
        install_commands=(
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements.txt",
            (
                "python -c \"import nltk; "
                "nltk.download('punkt', quiet=True); "
                "nltk.download('stopwords', quiet=True)\""
            ),
        ),
        run_command=_dedent(
            """
            python -m tenant_data_pipeline.run_pipeline
            python -m training.sft_train --tenant mfg
            python -m training.dpo_train --tenant mfg
            """
        ),
        artifact_paths=("models/adapters/mfg/sft", "models/adapters/mfg/dpo", "models/registry.json", "mlruns", "evaluation/reports"),
    ),
    "web-build": JobPreset(
        name="web-build",
        description="Install the Next.js app dependencies and produce a production build.",
        image="node:20-bookworm",
        flavor="cpu-upgrade",
        timeout="45m",
        install_commands=(),
        run_command=_dedent(
            """
            cd web_app
            npm ci
            npm run build
            """
        ),
        artifact_paths=("web_app/.next",),
    ),
}

SUITES: Dict[str, Sequence[str]] = {
    "ci": (
        "unit-tests",
        "module-tests",
        "data-pipeline",
        "evaluation",
        "training-smoke",
        "web-build",
    ),
    "build-all": (
        "data-pipeline",
        "rag-index",
        "web-build",
    ),
    "train-all": (
        "sft-sis",
        "sft-mfg",
    ),
    "align-all": (
        "dpo-sis",
        "dpo-mfg",
    ),
}


@dataclass(frozen=True)
class RepoContext:
    repo_url: str
    ref: str
    archive_url: str
    slug: str
    dirty: bool


def _run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def working_tree_dirty() -> bool:
    return bool(_run_git("status", "--porcelain"))


def resolve_ref(explicit_ref: Optional[str] = None) -> str:
    if explicit_ref:
        return explicit_ref
    return _run_git("rev-parse", "HEAD")


def normalize_repo_url(repo_url: str) -> str:
    repo_url = repo_url.strip()
    if repo_url.startswith("git@github.com:"):
        repo_url = repo_url.replace("git@github.com:", "https://github.com/", 1)
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]
    return repo_url


def github_repo_slug(repo_url: str) -> str:
    repo_url = normalize_repo_url(repo_url)
    prefix = "https://github.com/"
    if not repo_url.startswith(prefix):
        raise ValueError(
            "This launcher currently supports GitHub remotes so Jobs can fetch a repo snapshot."
        )
    return repo_url[len(prefix):].strip("/")


def build_archive_url(repo_url: str, ref: str) -> str:
    slug = github_repo_slug(repo_url)
    return f"https://codeload.github.com/{slug}/tar.gz/{quote(ref, safe='')}"


def load_repo_context(repo_url: Optional[str], ref: Optional[str]) -> RepoContext:
    resolved_repo_url = normalize_repo_url(repo_url or _run_git("remote", "get-url", "origin"))
    resolved_ref = resolve_ref(ref)
    slug = github_repo_slug(resolved_repo_url)
    return RepoContext(
        repo_url=resolved_repo_url,
        ref=resolved_ref,
        archive_url=build_archive_url(resolved_repo_url, resolved_ref),
        slug=slug,
        dirty=working_tree_dirty(),
    )


def render_artifact_copy_block(preset: JobPreset, artifact_bucket: Optional[str]) -> str:
    if not artifact_bucket or not preset.artifact_paths:
        return ""

    copy_lines = [
        f'DEST_DIR="{DEFAULT_ARTIFACT_MOUNT}/${{JOB_ID:-manual}}/{preset.name}"',
        'mkdir -p "$DEST_DIR"',
    ]
    for artifact_path in preset.artifact_paths:
        copy_lines.extend(
            [
                f'if [ -e {_sh(artifact_path)} ]; then',
                f'  cp -R {_sh(artifact_path)} "$DEST_DIR"/',
                "fi",
            ]
        )

    return "\n".join(
        [
            f"if [ -d {DEFAULT_ARTIFACT_MOUNT} ]; then",
            *copy_lines,
            "fi",
        ]
    )


def build_job_script(
    preset: JobPreset,
    repo: RepoContext,
    env_overrides: Mapping[str, str],
    artifact_bucket: Optional[str],
) -> str:
    env = {
        "HF_JOB_PRESET": preset.name,
        "HF_JOB_REPO": repo.slug,
        "HF_JOB_REF": repo.ref,
        **preset.env,
        **env_overrides,
    }

    export_lines = [f"export {key}={_sh(value)}" for key, value in env.items()]
    script_lines: List[str] = [
        "set -euo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        (
            "if command -v apt-get >/dev/null 2>&1; then "
            "apt-get update >/dev/null && "
            "apt-get install -y --no-install-recommends ca-certificates curl git build-essential >/dev/null; "
            "fi"
        ),
        f"rm -rf {DEFAULT_WORKDIR}",
        f"mkdir -p {DEFAULT_WORKDIR}",
        f"curl -L {_sh(repo.archive_url)} | tar -xz --strip-components=1 -C {DEFAULT_WORKDIR}",
        f"cd {DEFAULT_WORKDIR}",
        *export_lines,
        *preset.install_commands,
        preset.run_command,
    ]

    artifact_block = render_artifact_copy_block(preset, artifact_bucket)
    if artifact_block:
        script_lines.append(artifact_block)

    return "\n".join(script_lines)


def parse_key_value_pairs(values: Sequence[str]) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE format, got: {item}")
        key, value = item.split("=", 1)
        pairs[key.strip()] = value
    return pairs


def expand_presets(selected: Sequence[str], suite_names: Sequence[str]) -> List[str]:
    names: List[str] = []
    for suite_name in suite_names:
        if suite_name not in SUITES:
            raise ValueError(f"Unknown suite: {suite_name}")
        names.extend(SUITES[suite_name])

    names.extend(selected)
    deduped: List[str] = []
    for name in names:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset: {name}")
        if name not in deduped:
            deduped.append(name)
    return deduped


def build_volumes(artifact_bucket: Optional[str]) -> List[Volume]:
    if not artifact_bucket:
        return []
    return [Volume(type="bucket", source=artifact_bucket, mount_path=DEFAULT_ARTIFACT_MOUNT)]


def build_secrets(
    preset: JobPreset,
    include_hf_token: bool,
) -> Dict[str, str]:
    secrets: Dict[str, str] = {}
    if include_hf_token or preset.requires_hf_token:
        token = get_token()
        if not token:
            raise RuntimeError(
                "No local Hugging Face token found. Run `hf auth login` first or disable token passing."
            )
        secrets["HF_TOKEN"] = token
    return secrets


def submit_jobs(args: argparse.Namespace) -> int:
    preset_names = expand_presets(args.presets, args.suite)
    repo = load_repo_context(args.repo_url, args.ref)
    env_overrides = parse_key_value_pairs(args.env)
    label_overrides = parse_key_value_pairs(args.label)
    api = HfApi(token=args.token)

    if repo.dirty:
        print(
            "Warning: the local worktree has uncommitted changes. HF Jobs will run the remote "
            f"snapshot for ref {repo.ref}, not your uncommitted files.",
            file=sys.stderr,
        )

    for preset_name in preset_names:
        preset = PRESETS[preset_name]
        script = build_job_script(
            preset=preset,
            repo=repo,
            env_overrides=env_overrides,
            artifact_bucket=args.artifact_bucket,
        )
        labels = {
            "project": "multi-tenant-llm",
            "preset": preset.name,
            "repo": repo.slug,
            **preset.labels,
            **label_overrides,
        }
        volumes = build_volumes(args.artifact_bucket)
        secrets = build_secrets(preset, include_hf_token=args.with_hf_token)

        request = {
            "image": args.image or preset.image,
            "command": ["/bin/bash", "-lc", script],
            "env": dict(env_overrides) or None,
            "secrets": secrets or None,
            "flavor": args.flavor or preset.flavor,
            "timeout": args.timeout or preset.timeout,
            "labels": labels,
            "volumes": volumes or None,
            "namespace": args.namespace,
            "token": args.token,
        }

        if args.dry_run:
            print(f"\n=== {preset.name} ===")
            print(f"image: {request['image']}")
            print(f"flavor: {request['flavor']}")
            print(f"timeout: {request['timeout']}")
            print(f"repo: {repo.repo_url}")
            print(f"ref: {repo.ref}")
            if args.artifact_bucket:
                print(f"artifact_bucket: {args.artifact_bucket}")
            print("--- script ---")
            print(script)
            continue

        job = api.run_job(**request)
        print(f"{preset.name}: {job.id}")
        print(f"  status: {job.status.stage}")
        print(f"  url: {job.url}")
        if getattr(job, "endpoint", None):
            print(f"  endpoint: {job.endpoint}")

    return 0


def list_presets(_: argparse.Namespace) -> int:
    print("Presets:")
    for preset in PRESETS.values():
        print(
            f"  {preset.name:15} {preset.flavor:14} {preset.timeout:>6}  {preset.description}"
        )
    print("\nSuites:")
    for suite_name, preset_names in SUITES.items():
        print(f"  {suite_name:15} {' '.join(preset_names)}")
    return 0


def show_hardware(args: argparse.Namespace) -> int:
    api = HfApi(token=args.token)
    for hardware in api.list_jobs_hardware(token=args.token):
        cost_per_hour = getattr(hardware, "cost_per_hour", None)
        cost_text = f"${cost_per_hour}/h" if cost_per_hour is not None else "n/a"
        print(f"{hardware.name:15} {cost_text:>10}  {hardware.pretty_name}")
    return 0


def list_jobs(args: argparse.Namespace) -> int:
    api = HfApi(token=args.token)
    for job in api.list_jobs(namespace=args.namespace, token=args.token):
        print(f"{job.id}  {job.status.stage:12}  {job.url}")
    return 0


def inspect_job(args: argparse.Namespace) -> int:
    api = HfApi(token=args.token)
    job = api.inspect_job(job_id=args.job_id, namespace=args.namespace, token=args.token)
    print(f"id: {job.id}")
    print(f"status: {job.status.stage}")
    print(f"url: {job.url}")
    print(f"image: {job.docker_image}")
    print(f"command: {job.command}")
    print(f"flavor: {job.flavor}")
    print(f"labels: {job.labels}")
    print(f"created_at: {job.created_at}")
    print(f"owner: {job.owner.name}")
    if getattr(job, "endpoint", None):
        print(f"endpoint: {job.endpoint}")
    return 0


def stream_logs(args: argparse.Namespace) -> int:
    api = HfApi(token=args.token)
    for line in api.fetch_job_logs(
        job_id=args.job_id,
        namespace=args.namespace,
        follow=args.follow,
        token=args.token,
    ):
        print(line, end="")
    return 0


def cancel_job(args: argparse.Namespace) -> int:
    api = HfApi(token=args.token)
    api.cancel_job(job_id=args.job_id, namespace=args.namespace, token=args.token)
    print(f"Canceled {args.job_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit and inspect Hugging Face Jobs for this repo."
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token override for the Jobs API client.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available presets and suites.")
    list_parser.set_defaults(func=list_presets)

    hardware_parser = subparsers.add_parser("hardware", help="List available HF Jobs hardware.")
    hardware_parser.set_defaults(func=show_hardware)

    ps_parser = subparsers.add_parser("ps", help="List jobs in the current namespace.")
    ps_parser.add_argument("--namespace", default=None)
    ps_parser.set_defaults(func=list_jobs)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a submitted job.")
    inspect_parser.add_argument("job_id")
    inspect_parser.add_argument("--namespace", default=None)
    inspect_parser.set_defaults(func=inspect_job)

    logs_parser = subparsers.add_parser("logs", help="Fetch job logs.")
    logs_parser.add_argument("job_id")
    logs_parser.add_argument("--namespace", default=None)
    logs_parser.add_argument("--follow", action="store_true")
    logs_parser.set_defaults(func=stream_logs)

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a running job.")
    cancel_parser.add_argument("job_id")
    cancel_parser.add_argument("--namespace", default=None)
    cancel_parser.set_defaults(func=cancel_job)

    submit_parser = subparsers.add_parser("submit", help="Submit one or more presets.")
    submit_parser.add_argument("presets", nargs="*", default=[])
    submit_parser.add_argument("--suite", action="append", default=[])
    submit_parser.add_argument("--repo-url", default=None)
    submit_parser.add_argument("--ref", default=None)
    submit_parser.add_argument("--namespace", default=None)
    submit_parser.add_argument("--flavor", default=None)
    submit_parser.add_argument("--timeout", default=None)
    submit_parser.add_argument("--image", default=None)
    submit_parser.add_argument("--env", action="append", default=[])
    submit_parser.add_argument("--label", action="append", default=[])
    submit_parser.add_argument(
        "--artifact-bucket",
        default=None,
        help="Optional HF bucket (owner/name) mounted to persist selected job artifacts.",
    )
    submit_parser.add_argument(
        "--with-hf-token",
        action="store_true",
        help="Inject the local HF token into the job as HF_TOKEN.",
    )
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.set_defaults(func=submit_jobs)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
