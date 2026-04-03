.PHONY: setup mlflow data train dpo index serve serve-prod serve-ollama check-ollama register-ollama-models check-train-env eval web mobile voice monitor test clean all

PYTHON ?= python3
PIP ?= pip3
VENV := venv

# ── Environment ────────────────────────────────────────────────────────────────
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/$(PIP) install --upgrade pip
	$(VENV)/bin/$(PIP) install -r requirements.txt
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

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
	$(PYTHON) training/sft_train.py --tenant sis
	$(PYTHON) training/sft_train.py --tenant mfg

dpo:
	$(PYTHON) training/dpo_train.py --tenant sis

# ── Inference server ───────────────────────────────────────────────────────────
serve:
	$(PYTHON) -m uvicorn inference.app:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	$(PYTHON) -m uvicorn inference.app:app --host 0.0.0.0 --port 8000 --workers 1

serve-ollama:
	INFERENCE_BACKEND=ollama $(PYTHON) -m uvicorn inference.app:app --host 0.0.0.0 --port 8000 --reload

check-ollama:
	$(PYTHON) -c "from inference.model_backend import OllamaBackend; b = OllamaBackend(); print({'available': b.is_available(), 'base_url': b.base_url, 'models': b.list_models() if b.is_available() else []})"

register-ollama-models:
	$(PYTHON) scripts/register_ollama_models.py --tenant all --model-type sft

check-train-env:
	$(PYTHON) -m training.check_env

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
