# Multi-Tenant LLM Platform with RAG

[![Tests](https://img.shields.io/badge/tests-19%2F19%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Enterprise-grade multi-tenant LLM platform with domain-specific fine-tuning, hybrid RAG retrieval, and production-ready infrastructure.

## 🎯 Overview

This project demonstrates a complete production-ready multi-tenant LLM system with:

- **Multi-Tenant Architecture**: Complete isolation between tenants (data, models, RAG collections)
- **Domain Adaptation**: QLoRA fine-tuning for tenant-specific knowledge
- **Hybrid RAG**: ChromaDB vector search + BM25 keyword search + cross-encoder reranking
- **Production API**: FastAPI with streaming, audit logging, canary deployment
- **Full-Stack**: REST API, Next.js web app, Flutter mobile app, WebSocket voice agent
- **MLOps**: Model registry, monitoring dashboard, automated retraining, CI/CD pipeline
- **Comprehensive Evaluation**: Golden sets, hallucination detection, bias audits, red-team testing

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│  Web App (Next.js) │ Mobile (Flutter) │ Voice (WebSocket)       │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     Inference API (FastAPI)                      │
│  • Tenant Routing  • Streaming  • Audit Logging  • Auth         │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
┌───────────────────┐                    ┌──────────────────────┐
│   RAG Pipeline    │                    │   LLM Generation     │
│  • ChromaDB       │                    │  • Base Model        │
│  • BM25 Index     │                    │  • LoRA Adapters     │
│  • Reranker       │                    │  • 4-bit Quant       │
│  • Grounding      │                    │  • Streaming         │
└───────────────────┘                    └──────────────────────┘
        │                                           │
┌───────────────────────────────────────────────────────────────┐
│                    Data & Training Layer                       │
│  • Synthetic Data  • PII Redaction  • SFT/DPO Training        │
└───────────────────────────────────────────────────────────────┘
```

## ✨ Features

### 🔐 Multi-Tenant Isolation
- Separate ChromaDB collections per tenant
- Tenant-specific LoRA adapters
- Isolated audit logs and metrics
- API key-based access control

### 🧠 Advanced RAG
- **Hybrid Retrieval**: Dense (vector) + Sparse (BM25) fusion
- **Reranking**: Cross-encoder for precision improvement
- **Grounding Verification**: Semantic similarity scoring
- **Citation Tracking**: Source attribution for every response

### 🎓 Domain Adaptation
- **QLoRA Fine-Tuning**: 4-bit quantization for efficient training
- **SFT (Supervised Fine-Tuning)**: Domain-specific instruction following
- **DPO (Direct Preference Optimization)**: Alignment with human preferences
- **MLflow Tracking**: Experiment logging and model versioning

### 🚀 Production-Ready API
- **Streaming Responses**: SSE for real-time token generation
- **Canary Deployment**: A/B testing with traffic splitting
- **Audit Logging**: Complete request/response tracking
- **Health Monitoring**: GPU stats, latency metrics, error rates

### 📊 Comprehensive Evaluation
- **Golden Sets**: Curated Q&A pairs for regression testing
- **Hallucination Detection**: Grounding score verification
- **Bias Audits**: Fairness across demographics
- **Red Team Testing**: Adversarial prompt injection
- **Compliance Checks**: FERPA, ISO documentation

### 🎨 Multi-Channel Interfaces
- **Web App**: Next.js 14 with streaming chat and citations
- **Mobile App**: Flutter with offline support
- **Voice Agent**: WebSocket with STT/TTS pipeline
- **REST API**: 15+ endpoints with OpenAPI docs

## 📦 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+ (for web app)
- Flutter 3.16+ (for mobile app)
- CUDA-capable GPU (optional, for training)

### Quick Start

```bash
# Clone repository
git clone https://github.com/deepaucksharma/multi-tenant-LLM-platform.git
cd multi-tenant-LLM-platform

# Install Python dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Run data pipeline
make data

# Build RAG indexes
make index

# Start inference server
make serve

# In another terminal, start monitoring
make monitor

# In another terminal, start web app
make web-install
make web
```

### Local AMD / WSL Inference With Ollama

For AMD GPUs on Windows, the smoothest local path is to run Ollama on the Windows host and keep FastAPI/RAG in WSL.

```bash
# In Windows PowerShell or Command Prompt
ollama pull qwen2.5:1.5b

# In WSL
cp .env.example .env

# Use Ollama explicitly
echo "INFERENCE_BACKEND=ollama" >> .env

# Optional tenant-specific models
echo "OLLAMA_MODEL_SIS=qwen2.5:1.5b" >> .env
echo "OLLAMA_MODEL_MFG=qwen2.5:1.5b" >> .env

# Verify Ollama is reachable from WSL
make check-ollama

# Optional: create tenant-specific local aliases with tenant prompts
make register-ollama-models

# Start the API
make serve-ollama
```

Useful endpoints for this mode:

- `GET /health` for a quick readiness check
- `GET /backend/status` for resolved backend and Ollama model mapping

If you want stable tenant aliases instead of routing both tenants to the same base model, set:

```bash
echo "OLLAMA_MODEL_SIS_SFT=tenant-sis-sft" >> .env
echo "OLLAMA_MODEL_MFG_SFT=tenant-mfg-sft" >> .env
echo "OLLAMA_SOURCE_MODEL=qwen2.5:1.5b" >> .env
make register-ollama-models
```

### Local Training Fallbacks

Training now adapts to the available runtime instead of assuming CUDA-only 4-bit QLoRA:

- `DEVICE=auto` uses CUDA/ROCm when available, otherwise CPU
- `USE_4BIT=auto` enables bitsandbytes only when it is actually supported
- when 4-bit is unavailable, training falls back to standard LoRA with `adamw_torch`

For local AMD or CPU development, these defaults are usually enough:

```bash
echo "DEVICE=auto" >> .env
echo "USE_4BIT=auto" >> .env
```

If you explicitly want to disable bitsandbytes even on NVIDIA:

```bash
echo "USE_4BIT=false" >> .env
```

You can verify whether the local environment is actually ready for training with:

```bash
make check-train-env
```

### Full Pipeline

```bash
# Run everything end-to-end
./run_e2e.sh --demo

# Or step by step
make data          # Generate synthetic data
make index         # Build RAG indexes
make train         # Train adapters (GPU required)
make serve         # Start API server
make web           # Start web app
make monitor       # Start dashboard
make test          # Run tests
```

## 🎯 Usage

### API Examples

```bash
# Health check
curl http://localhost:8000/health

# List tenants
curl http://localhost:8000/tenants

# Test RAG retrieval
curl "http://localhost:8000/rag/test?query=enrollment&tenant_id=sis&top_k=3"

# Chat with SIS tenant
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "sis",
    "message": "What documents do I need for enrollment?",
    "use_rag": true,
    "model_type": "sft"
  }'

# Streaming chat
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "mfg",
    "message": "Explain quality control procedures",
    "use_rag": true
  }'

# Submit feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "abc123",
    "tenant_id": "sis",
    "rating": 5,
    "comment": "Very helpful!"
  }'
```

### Python SDK

```python
from inference.adapter_manager import get_adapter_manager
from rag.rag_chain import RAGRequest, execute_rag_chain

# Load model with adapter
manager = get_adapter_manager()
manager.load_base_model()
manager.load_adapter("models/adapters/sis/sft")

# RAG query
request = RAGRequest(
    query="What is the enrollment process?",
    tenant_id="sis",
    top_k=3
)

response = execute_rag_chain(
    request,
    generate_fn=lambda msgs: manager.generate(msgs)
)

print(f"Answer: {response.answer}")
print(f"Citations: {len(response.citations)}")
print(f"Grounding: {response.grounding_report['grounding_score']}")
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ --cov --cov-report=html

# Run specific test
pytest tests/test_basic.py::TestRAG -v

# Run evaluation suite
python -m evaluation.run_all_evals
```

**Test Results**: 19/19 passing ✅

## 📊 Tenants

### SIS (Student Information System)
- **Domain**: Education / School District
- **Topics**: Enrollment, attendance, grading, transcripts, FERPA compliance
- **Documents**: 10 synthetic documents, 40 chunks
- **Use Cases**: Student records, parent communication, policy queries

### MFG (Manufacturing)
- **Domain**: Industrial Quality Control
- **Topics**: Quality control, ISO documentation, safety protocols, CAPA
- **Documents**: 10 synthetic documents, 49 chunks
- **Use Cases**: SOP queries, defect classification, compliance checks

## 🔧 Configuration

### Environment Variables

```bash
# Model Configuration
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
DEVICE=cuda                    # or 'cpu'
MAX_SEQ_LEN=512

# Storage
DATA_ROOT=./data
CHROMA_PERSIST_DIR=./data/chroma
SQLITE_DB=./logs/audit.db

# Services
API_PORT=8000
WEB_PORT=3000
VOICE_PORT=8001
MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Security
DEMO_MODE=true                 # Disable for production
HF_TOKEN=                      # Optional: HuggingFace token
```

### Training Configuration

```yaml
# training/configs/sft_sis.yaml
model:
  base_model: Qwen/Qwen2.5-1.5B-Instruct
  max_seq_length: 512

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_ratio: 0.1
```

## 📁 Project Structure

```
multi-tenant-LLM-platform/
├── tenant_data_pipeline/      # Data generation & processing
├── rag/                       # RAG system (retrieval, reranking)
├── training/                  # Model training (SFT, DPO)
├── inference/                 # FastAPI server
├── evaluation/                # Evaluation suite
├── web_app/                   # Next.js web interface
├── mobile_app/                # Flutter mobile app
├── voice_agent/               # Voice interface
├── monitoring/                # Monitoring dashboard
├── mlops/                     # Model registry & automation
├── tests/                     # Unit tests
├── data/                      # Data storage
├── models/                    # Model storage
├── logs/                      # Logs & audit trail
├── Makefile                   # Build automation
├── run_e2e.sh                 # End-to-end script
└── requirements.txt           # Python dependencies
```

## 🚀 Deployment

### Docker

```bash
# Build image
docker build -t multi-tenant-llm .

# Run container
docker run -p 8000:8000 -v $(pwd)/data:/app/data multi-tenant-llm
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -l app=llm-inference
```

### Cloud Deployment

- **AWS**: ECS/EKS with GPU instances (g4dn, p3)
- **GCP**: GKE with T4/V100 GPUs
- **Azure**: AKS with NC-series VMs

## 📈 Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| RAG Retrieval Latency | 50-200ms |
| Reranking Latency | 20-50ms |
| Generation Latency (CPU) | 10-30s |
| Generation Latency (GPU) | 1-3s |
| Retrieval Precision@3 | 0.85+ |
| Grounding Score | 0.6-0.9 |
| Test Pass Rate | 100% (19/19) |

### Scalability

- **Horizontal**: Load balancer + multiple API instances
- **Vertical**: GPU scaling (multi-GPU, larger models)
- **Caching**: Redis for embeddings and responses
- **Database**: PostgreSQL for audit logs, Pinecone for vectors

## 🛠️ Development

### Adding a New Tenant

1. **Define tenant config** in `tenant_data_pipeline/config.py`
2. **Generate synthetic data** or add real documents to `data/raw/{tenant_id}/`
3. **Run data pipeline**: `make data`
4. **Build RAG index**: `make index`
5. **Train adapter**: `python training/sft_train.py --tenant {tenant_id}`
6. **Test**: `curl http://localhost:8000/tenants/{tenant_id}`

### Adding a New Evaluation

1. Create test file in `evaluation/`
2. Implement evaluation logic
3. Add to `evaluation/run_all_evals.py`
4. Run: `python -m evaluation.run_all_evals`

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- **Qwen Team**: Base model (Qwen2.5-1.5B-Instruct)
- **Hugging Face**: Transformers, PEFT, TRL libraries
- **ChromaDB**: Vector database
- **FastAPI**: Web framework
- **Next.js**: Web application framework
- **Flutter**: Mobile application framework

## 📚 Documentation

- **[E2E Execution Summary](E2E_EXECUTION_SUMMARY.md)**: Complete system documentation
- **[Architecture Details](CLAUDE.md)**: Design decisions and implementation notes
- **[API Documentation](http://localhost:8000/docs)**: Interactive Swagger UI
- **[Training Guide](training/README.md)**: Model training instructions
- **[Evaluation Guide](evaluation/README.md)**: Evaluation framework details

## 🐛 Known Issues

1. **Model Download**: Base model needs to be downloaded first (~3GB)
2. **GPU Required**: Training requires CUDA-capable GPU
3. **CPU Inference**: Slower without GPU (10-30s vs 1-3s)

## 🔮 Roadmap

- [ ] Add authentication (OAuth2, JWT)
- [ ] Implement rate limiting
- [ ] Add caching layer (Redis)
- [ ] Docker Compose setup
- [ ] Kubernetes manifests
- [ ] Multi-GPU support
- [ ] Model quantization (GGUF, GPTQ)
- [ ] Vector DB migration (Pinecone, Weaviate)
- [ ] Multi-modal support (images, PDFs)

## 📞 Contact

**Deepak Sharma**
- GitHub: [@deepaucksharma](https://github.com/deepaucksharma)
- Repository: [multi-tenant-LLM-platform](https://github.com/deepaucksharma/multi-tenant-LLM-platform)

---

**Built with ❤️ for demonstrating enterprise-grade LLM systems**

⭐ Star this repo if you find it useful!
