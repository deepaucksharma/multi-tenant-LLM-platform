.PHONY: setup setup-rocm install-rocm-torch mlflow data train train-smoke train-smoke-tiny dpo index serve serve-prod serve-ollama serve-hf-inference check-ollama register-ollama-models check-train-env check-model-ready push-hub push-hub-weights push-hub-dry push-hub-private push-hub-merged push-datasets push-datasets-dry generate-colab docker-build docker-run eval web mobile voice monitor test clean all smoke-data smoke-index smoke-train-sis smoke-train-mfg smoke-dpo smoke-inference smoke-eval smoke-pipeline

PYTHON ?= python3
PIP ?= pip3
VENV := venv

# ── Environment ────────────────────────────────────────────────────────────────
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/$(PIP) install --upgrade pip
	$(VENV)/bin/$(PIP) install -r requirements.txt
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"


# ── AMD ROCm GPU Setup (RX 6800 XT / RDNA2 / gfx1030) ──────────────────────────
setup-rocm:
	@echo "Running ROCm setup for AMD RX 6800 XT (gfx1030/RDNA2)..."
	bash setup_rocm.sh
	@echo "IMPORTANT: Restart WSL after: run 'wsl --shutdown' in PowerShell"

install-rocm-torch:
	$(PIP) uninstall -y torch torchvision torchaudio 2>/dev/null || true
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
	@echo "Verifying GPU..."
	$(PYTHON) -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# ── MLflow tracking server ──────────────────────────────────────────────────────
mlflow:
	mlflow server \
		--host 127.0.0.1 \
		--port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns

# ── Data pipeline ──────────────────────────────────────────────────────────────
data:
	$(PYTHON) -m tenant_data_pipeline.run_pipeline

# ── RAG index ──────────────────────────────────────────────────────────────────
index:
	$(PYTHON) -m rag.build_index --force

# ── Training ───────────────────────────────────────────────────────────────────
train:
	$(PYTHON) -m training.sft_train --tenant sis
	$(PYTHON) -m training.sft_train --tenant mfg

train-smoke:
	$(PYTHON) -m training.sft_train --tenant sis --smoke-test

train-smoke-tiny:
	SMOKE_TEST_BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 $(PYTHON) -m training.sft_train --tenant sis --smoke-test

# ── Comprehensive Smoke Pipeline (Both Tenants) ───────────────────────────────

## Generate synthetic data for both tenants
smoke-data:
	@echo "🔄 Running smoke data pipeline for both tenants..."
	$(PYTHON) -m tenant_data_pipeline.run_pipeline

## Build RAG indexes for both tenants
smoke-index:
	@echo "🔄 Building smoke RAG indexes for both tenants..."
	$(PYTHON) -m rag.build_index --force

## Train SFT adapter for SIS tenant (smoke mode, optimized)
smoke-train-sis:
	@echo "🔄 Training smoke SFT adapter for SIS..."
	DEVICE=cpu MLFLOW_TRACKING_URI=file:./mlruns-smoke \
	SMOKE_TEST_LOCAL_MODEL_PATH=./models/base/smoke-gpt2 \
	SMOKE_TEST_BASE_MODEL= \
	SMOKE_SEQ_LEN=256 \
	$(PYTHON) -m training.sft_train --tenant sis --smoke-test --max-steps 1

## Train SFT adapter for MFG tenant (smoke mode, optimized)
smoke-train-mfg:
	@echo "🔄 Training smoke SFT adapter for MFG..."
	DEVICE=cpu MLFLOW_TRACKING_URI=file:./mlruns-smoke \
	SMOKE_TEST_LOCAL_MODEL_PATH=./models/base/smoke-gpt2 \
	SMOKE_TEST_BASE_MODEL= \
	SMOKE_SEQ_LEN=64 SMOKE_N_LAYER=1 SMOKE_N_EMBD=32 \
	$(PYTHON) -m training.sft_train --tenant mfg --smoke-test --max-steps 1

## Train DPO alignment for SIS tenant (smoke mode, optimized)
smoke-dpo:
	@echo "🔄 Training smoke DPO alignment for SIS..."
	DEVICE=cpu MLFLOW_TRACKING_URI=file:./mlruns-smoke \
	SMOKE_TEST_LOCAL_MODEL_PATH=./models/base/smoke-gpt2 \
	SMOKE_TEST_BASE_MODEL= \
	SMOKE_SEQ_LEN=64 SMOKE_N_LAYER=1 SMOKE_N_EMBD=32 \
	$(PYTHON) -m training.dpo_train_simple --tenant sis --smoke-test --max-steps 1

## Quick offline smoke validation: SFT for both tenants + DPO for SIS (fastest path)
train-smoke-offline:
	@echo "🔄 Running fast offline smoke validation..."
	DEVICE=cpu MLFLOW_TRACKING_URI=file:./mlruns-smoke \
	SMOKE_TEST_LOCAL_MODEL_PATH=./models/base/smoke-gpt2 \
	SMOKE_TEST_BASE_MODEL= \
	SMOKE_SEQ_LEN=64 SMOKE_N_LAYER=1 SMOKE_N_EMBD=32 \
	$(PYTHON) -m training.sft_train --tenant sis --smoke-test --max-steps 1 && \
	DEVICE=cpu MLFLOW_TRACKING_URI=file:./mlruns-smoke \
	SMOKE_TEST_LOCAL_MODEL_PATH=./models/base/smoke-gpt2 \
	SMOKE_TEST_BASE_MODEL= \
	SMOKE_SEQ_LEN=64 SMOKE_N_LAYER=1 SMOKE_N_EMBD=32 \
	$(PYTHON) -m training.sft_train --tenant mfg --smoke-test --max-steps 1 && \
	DEVICE=cpu MLFLOW_TRACKING_URI=file:./mlruns-smoke \
	SMOKE_TEST_LOCAL_MODEL_PATH=./models/base/smoke-gpt2 \
	SMOKE_TEST_BASE_MODEL= \
	SMOKE_SEQ_LEN=64 SMOKE_N_LAYER=1 SMOKE_N_EMBD=32 \
	$(PYTHON) -m training.dpo_train_simple --tenant sis --smoke-test --max-steps 1 && \
	@echo "✅ All smoke training passed: SIS SFT, MFG SFT, SIS DPO"

## Test inference for both tenants (smoke mode)
smoke-inference:
	@echo "🔄 Testing smoke inference for both tenants..."
	$(PYTHON) -m inference.app --smoke-test
	pkill -f "inference.app"
	sleep 2

## Run evaluation suite for both tenants (smoke mode)
smoke-eval:
	@echo "🔄 Running smoke evaluation for both tenants..."
	$(PYTHON) -m evaluation.run_all_evals --tenant sis
	$(PYTHON) -m evaluation.run_all_evals --tenant mfg

## Complete smoke pipeline: data → index → train both → dpo → inference → eval
smoke-pipeline: smoke-data smoke-index smoke-train-sis smoke-train-mfg smoke-dpo smoke-inference smoke-eval
	@echo "🎉 Complete smoke pipeline validation finished!"
	@echo "✅ Both SIS and MFG tenants validated end-to-end"
	@echo "✅ Data pipeline, RAG indexing, SFT training, DPO alignment, inference, and evaluation all working"

dpo:
	$(PYTHON) -m training.dpo_train --tenant sis

# ── Inference server ───────────────────────────────────────────────────────────
serve:
	$(PYTHON) -m uvicorn inference.app:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	$(PYTHON) -m uvicorn inference.app:app --host 0.0.0.0 --port 8000 --workers 1

serve-ollama:
	INFERENCE_BACKEND=ollama $(PYTHON) -m uvicorn inference.app:app --host 0.0.0.0 --port 8000 --reload

serve-hf-inference:
	INFERENCE_BACKEND=hf_inference $(PYTHON) -m uvicorn inference.app:app --host 0.0.0.0 --port 8000 --reload

check-ollama:
	$(PYTHON) -c "from inference.model_backend import OllamaBackend; b = OllamaBackend(); print({'available': b.is_available(), 'base_url': b.base_url, 'models': b.list_models() if b.is_available() else []})"

register-ollama-models:
	$(PYTHON) scripts/register_ollama_models.py --tenant all --model-type sft

check-train-env:
	$(PYTHON) -m training.check_env

check-model-ready:
	$(PYTHON) -m training.check_model --tenant sis --smoke-test

# ── Hugging Face Hub ───────────────────────────────────────────────────────────
# Profile: https://huggingface.co/deepaucksharma
# Repos:   https://huggingface.co/deepaucksharma?search=multi-tenant-llm
# Requires HF_TOKEN in .env (write-scoped token from huggingface.co/settings/tokens)

## Push adapter config/metadata JSON only (no weight files — safe, fast)
push-hub:
	$(PYTHON) -m training.push_to_hub --all

## Push adapter configs AND .safetensors weights (~200-600 MB per adapter)
push-hub-weights:
	$(PYTHON) -m training.push_to_hub --all --weights

## Dry-run: show what would be pushed without uploading
push-hub-dry:
	$(PYTHON) -m training.push_to_hub --all --dry-run

## Push adapters as private repos
push-hub-private:
	$(PYTHON) -m training.push_to_hub --all --private

## Push merged full models from models/merged/ (~3 GB per model)
push-hub-merged:
	$(PYTHON) -m training.push_to_hub --all --merged --weights

# ── Dataset Hub publishing ─────────────────────────────────────────────────────

## Push SFT/DPO datasets to HF Hub (public dataset repos)
push-datasets:
	$(PYTHON) -m training.push_datasets --all

## Dry-run: show what would be uploaded
push-datasets-dry:
	$(PYTHON) -m training.push_datasets --all --dry-run

# ── Colab notebook ─────────────────────────────────────────────────────────────

## Generate the Google Colab training notebook
generate-colab:
	$(PYTHON) notebooks/generate_colab_notebook.py

# ── Docker (HF Spaces) ─────────────────────────────────────────────────────────

## Build Docker image for HF Spaces deployment
docker-build:
	docker build -t multi-tenant-llm-demo .

## Run Docker image locally (set HF_TOKEN in .env first)
docker-run:
	docker run --rm -p 7860:7860 --env-file .env multi-tenant-llm-demo

# ── Evaluation ─────────────────────────────────────────────────────────────────
eval:
	$(PYTHON) -m evaluation.run_all_evals

# ── Web UI (Next.js) ───────────────────────────────────────────────────────────
web:
	cd web_app && npm run dev

web-install:
	cd web_app && npm install

# ── Mobile (Flutter) ───────────────────────────────────────────────────────────
mobile:
	cd mobile_app && flutter run

mobile-install:
	cd mobile_app && flutter pub get

# ── Voice agent ────────────────────────────────────────────────────────────────
voice:
	$(PYTHON) -m voice_agent.voice_server

# ── Monitoring dashboard ───────────────────────────────────────────────────────
monitor:
	$(PYTHON) -m uvicorn monitoring.dashboard:app --host 0.0.0.0 --port 8502

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/

test-cov:
	$(PYTHON) -m pytest tests/ --cov --cov-report=html

# ── Full pipeline ──────────────────────────────────────────────────────────────
all: data index train serve

# ── Clean generated artifacts ─────────────────────────────────────────────────
clean:
	rm -rf data/raw/ data/sis/ data/mfg/ data/chroma/
	rm -rf models/adapters/ models/aligned/ models/merged/
	rm -rf logs/ mlruns/ mlflow.db
	rm -rf evaluation/reports/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned generated artifacts."

clean-all: clean
	rm -rf $(VENV) web_app/node_modules web_app/.next
