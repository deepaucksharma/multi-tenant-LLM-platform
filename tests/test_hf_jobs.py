"""Unit tests for the Hugging Face Jobs launcher."""
from mlops.hf_jobs import (
    PRESETS,
    RepoContext,
    SUITES,
    build_archive_url,
    build_job_script,
    expand_presets,
    normalize_repo_url,
)


def test_normalize_repo_url_supports_https_and_ssh():
    assert (
        normalize_repo_url("https://github.com/deepaucksharma/multi-tenant-LLM-platform.git")
        == "https://github.com/deepaucksharma/multi-tenant-LLM-platform"
    )
    assert (
        normalize_repo_url("git@github.com:deepaucksharma/multi-tenant-LLM-platform.git")
        == "https://github.com/deepaucksharma/multi-tenant-LLM-platform"
    )


def test_build_archive_url_uses_codeload_snapshot():
    url = build_archive_url(
        "https://github.com/deepaucksharma/multi-tenant-LLM-platform",
        "abc123def456",
    )
    assert url == (
        "https://codeload.github.com/deepaucksharma/"
        "multi-tenant-LLM-platform/tar.gz/abc123def456"
    )


def test_expand_presets_deduplicates_suite_members():
    names = expand_presets(["unit-tests"], ["ci"])
    assert names[0] == "unit-tests"
    assert names.count("unit-tests") == 1
    assert "training-smoke" in names


def test_build_job_script_embeds_repo_context_and_bucket_copy():
    repo = RepoContext(
        repo_url="https://github.com/deepaucksharma/multi-tenant-LLM-platform",
        ref="abc123def456",
        archive_url=(
            "https://codeload.github.com/deepaucksharma/"
            "multi-tenant-LLM-platform/tar.gz/abc123def456"
        ),
        slug="deepaucksharma/multi-tenant-LLM-platform",
        dirty=False,
    )
    script = build_job_script(
        preset=PRESETS["data-pipeline"],
        repo=repo,
        env_overrides={"EXTRA_FLAG": "1"},
        artifact_bucket="deepaucksharma/test-bucket",
    )

    assert "curl -L" in script
    assert repo.archive_url in script
    assert "python -m tenant_data_pipeline.run_pipeline" in script
    assert "export HF_JOB_REF=abc123def456" in script
    assert "export EXTRA_FLAG=1" in script
    assert "/job-artifacts/${JOB_ID:-manual}/data-pipeline" in script
    assert "cp -R 'data/sis'" in script or "cp -R data/sis" in script


def test_train_all_suite_maps_to_both_sft_jobs():
    assert SUITES["train-all"] == ("sft-sis", "sft-mfg")
