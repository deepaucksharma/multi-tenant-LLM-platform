.PHONY: setup mlflow data train dpo index serve eval web mobile voice monitor test clean all

PYTHON := python
PIP := pip
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
	uvicorn inference.app:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn inference.app:app --host 0.0.0.0 --port 8000 --workers 1

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
	uvicorn monitoring.dashboard:app --host 0.0.0.0 --port 8502

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/

test-cov:
	pytest tests/ --cov --cov-report=html

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
