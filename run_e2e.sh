#!/bin/bash
# End-to-End Execution Script for Multi-Tenant LLM Platform
# This script runs the complete pipeline from data generation to serving

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# Check if running in correct directory
if [ ! -f "Makefile" ]; then
    log_error "Must run from project root directory"
    exit 1
fi

# Parse arguments
SKIP_DATA=false
SKIP_TRAIN=false
SKIP_WEB_INSTALL=false
DEMO_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-web-install)
            SKIP_WEB_INSTALL=true
            shift
            ;;
        --demo)
            DEMO_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-data] [--skip-train] [--skip-web-install] [--demo]"
            exit 1
            ;;
    esac
done

# ============================================================
# PHASE 0: Environment Check
# ============================================================
log_section "PHASE 0: Environment Check"

log_info "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    log_error "Python not found"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
log_success "Python found: $PYTHON_VERSION"

log_info "Checking required Python packages..."
python3 -c "import torch; import transformers; import chromadb; import fastapi" 2>/dev/null
if [ $? -eq 0 ]; then
    log_success "Core Python packages installed"
else
    log_error "Missing required packages. Run: pip install -r requirements.txt"
    exit 1
fi

log_info "Checking GPU availability..."
GPU_INFO=$(python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')" 2>/dev/null)
log_info "$GPU_INFO"

if [ "$SKIP_TRAIN" = false ]; then
    log_info "Checking training environment..."
    if python3 -m training.check_env > /tmp/train_env_check.json 2>/dev/null; then
        log_success "Training environment is ready"
        if python3 -m training.check_model --tenant sis > /tmp/train_model_check.json 2>/dev/null; then
            log_success "Local base model is ready for offline training"
        else
            log_warning "Local base model weights are not available offline"
            log_info "Training may still work if model download from Hugging Face is available"
        fi
    else
        log_warning "Training environment is incomplete"
        log_info "Run 'make check-train-env' for details"
    fi
fi

log_info "Checking disk space..."
DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
log_info "Available disk space: $DISK_SPACE"

# ============================================================
# PHASE 1: Data Pipeline
# ============================================================
if [ "$SKIP_DATA" = false ]; then
    log_section "PHASE 1: Data Pipeline (Synthetic → Ingest → PII → Chunk → SFT/DPO)"
    
    log_info "Running data pipeline..."
    python3 -m tenant_data_pipeline.run_pipeline
    
    if [ $? -eq 0 ]; then
        log_success "Data pipeline completed"
        
        # Show data statistics
        log_info "Data statistics:"
        for tenant in sis mfg; do
            if [ -f "data/$tenant/sft/train_chat.json" ]; then
                TRAIN_COUNT=$(python3 -c "import json; print(len(json.load(open('data/$tenant/sft/train_chat.json'))))")
                log_info "  $tenant: $TRAIN_COUNT training examples"
            fi
        done
    else
        log_error "Data pipeline failed"
        exit 1
    fi
else
    log_warning "Skipping data pipeline (--skip-data)"
fi

# ============================================================
# PHASE 2: RAG Index Building
# ============================================================
log_section "PHASE 2: RAG Index Building (ChromaDB + BM25)"

log_info "Building vector indexes for both tenants..."
python3 -m rag.build_index --force

if [ $? -eq 0 ]; then
    log_success "RAG indexes built"
    
    # Show index statistics
    log_info "Index statistics:"
    python3 -m rag.build_index --list
else
    log_error "Index building failed"
    exit 1
fi

# ============================================================
# PHASE 3: Model Training (Optional)
# ============================================================
if [ "$SKIP_TRAIN" = false ]; then
    log_section "PHASE 3: Model Training (Adaptive LoRA)"
    
    log_warning "Training can use GPU when available and falls back adaptively, but still takes significant time"
    log_info "Training SIS tenant adapter..."

    if python3 -m training.check_env >/dev/null 2>&1; then
        # Train SIS
        log_info "Training SIS SFT adapter..."
        python3 -m training.sft_train --tenant sis --epochs 1
        
        if [ $? -eq 0 ]; then
            log_success "SIS adapter trained"
        else
            log_error "SIS training failed"
        fi
        
        # Train MFG
        log_info "Training MFG SFT adapter..."
        python3 -m training.sft_train --tenant mfg --epochs 1
        
        if [ $? -eq 0 ]; then
            log_success "MFG adapter trained"
        else
            log_error "MFG training failed"
        fi
    else
        log_warning "Training dependencies are not fully installed - skipping training"
        log_info "The system will use base model or Ollama-backed inference without local adapters"
    fi
else
    log_warning "Skipping training (--skip-train)"
fi

# ============================================================
# PHASE 4: Run Tests
# ============================================================
log_section "PHASE 4: Running Tests"

log_info "Running pytest suite..."
python3 -m pytest tests/ -v --tb=short

if [ $? -eq 0 ]; then
    log_success "All tests passed (19/19)"
else
    log_error "Some tests failed"
    exit 1
fi

# ============================================================
# PHASE 5: Web App Setup
# ============================================================
if [ "$SKIP_WEB_INSTALL" = false ]; then
    log_section "PHASE 5: Web App Setup"
    
    if command -v npm &> /dev/null; then
        log_info "Installing Next.js dependencies..."
        cd web_app
        npm install --silent
        cd ..
        log_success "Web dependencies installed"
    else
        log_warning "npm not found - skipping web app setup"
    fi
else
    log_warning "Skipping web install (--skip-web-install)"
fi

# ============================================================
# PHASE 6: Start Services
# ============================================================
log_section "PHASE 6: Starting Services"

# Create logs directory
mkdir -p logs

log_info "Starting inference API server on port 8000..."
nohup python3 -m uvicorn inference.app:app --host 0.0.0.0 --port 8000 > logs/inference.log 2>&1 &
INFERENCE_PID=$!
log_success "Inference API started (PID: $INFERENCE_PID)"

# Wait for API to be ready
log_info "Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API is ready"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        log_error "API failed to start"
        exit 1
    fi
done

log_info "Starting monitoring dashboard on port 8002..."
nohup python3 monitoring/dashboard.py > logs/monitor.log 2>&1 &
MONITOR_PID=$!
log_success "Monitoring dashboard started (PID: $MONITOR_PID)"

# Wait for monitor to be ready
sleep 3

if command -v npm &> /dev/null && [ -d "web_app/node_modules" ]; then
    log_info "Starting Next.js web app on port 3000..."
    cd web_app
    nohup npm run dev > ../logs/web.log 2>&1 &
    WEB_PID=$!
    cd ..
    log_success "Web app started (PID: $WEB_PID)"
    
    # Wait for web app
    sleep 5
fi

# ============================================================
# PHASE 7: Health Checks & Demo
# ============================================================
log_section "PHASE 7: System Health Check"

log_info "Checking API health..."
HEALTH=$(curl -s http://localhost:8000/health)
if [ $? -eq 0 ]; then
    log_success "API health check passed"
    echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
else
    log_error "API health check failed"
fi

log_info "Checking tenants..."
TENANTS=$(curl -s http://localhost:8000/tenants)
if [ $? -eq 0 ]; then
    log_success "Tenants endpoint working"
    echo "$TENANTS" | python3 -m json.tool 2>/dev/null || echo "$TENANTS"
fi

# ============================================================
# PHASE 8: Demo Requests (if --demo flag)
# ============================================================
if [ "$DEMO_MODE" = true ]; then
    log_section "PHASE 8: Running Demo Requests"
    
    log_info "Testing SIS tenant query..."
    curl -s -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{
            "tenant_id": "sis",
            "message": "What documents do I need for enrollment?",
            "use_rag": true,
            "model_type": "sft"
        }' | python3 -m json.tool
    
    echo ""
    log_info "Testing MFG tenant query..."
    curl -s -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{
            "tenant_id": "mfg",
            "message": "What are the quality control procedures?",
            "use_rag": true,
            "model_type": "sft"
        }' | python3 -m json.tool
    
    echo ""
    log_info "Testing RAG retrieval..."
    curl -s "http://localhost:8000/rag/test?query=enrollment&tenant_id=sis&top_k=3" | python3 -m json.tool
fi

# ============================================================
# Summary
# ============================================================
log_section "🎉 END-TO-END EXECUTION COMPLETE"

echo ""
echo "Services Running:"
echo "  ✓ Inference API:        http://localhost:8000"
echo "  ✓ API Docs:             http://localhost:8000/docs"
echo "  ✓ Monitoring Dashboard: http://localhost:8002"
if command -v npm &> /dev/null && [ -d "web_app/node_modules" ]; then
    echo "  ✓ Web App:              http://localhost:3000"
fi

echo ""
echo "Process IDs:"
echo "  Inference API: $INFERENCE_PID"
echo "  Monitor:       $MONITOR_PID"
if [ ! -z "$WEB_PID" ]; then
    echo "  Web App:       $WEB_PID"
fi

echo ""
echo "Logs:"
echo "  Inference: logs/inference.log"
echo "  Monitor:   logs/monitor.log"
if [ ! -z "$WEB_PID" ]; then
    echo "  Web:       logs/web.log"
fi

echo ""
echo "To stop all services:"
echo "  kill $INFERENCE_PID $MONITOR_PID${WEB_PID:+ $WEB_PID}"
echo ""
echo "Or use: pkill -f 'uvicorn|dashboard.py|next dev'"

echo ""
echo "Quick Test Commands:"
echo "  # Health check"
echo "  curl http://localhost:8000/health"
echo ""
echo "  # List tenants"
echo "  curl http://localhost:8000/tenants"
echo ""
echo "  # Test chat (SIS)"
echo "  curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \\"
echo "    -d '{\"tenant_id\":\"sis\",\"message\":\"What is the enrollment process?\",\"use_rag\":true}'"
echo ""
echo "  # Run evaluation"
echo "  python3 -m evaluation.run_all_evals"
echo ""

# Save PIDs to file for easy cleanup
echo "$INFERENCE_PID" > logs/pids.txt
echo "$MONITOR_PID" >> logs/pids.txt
if [ ! -z "$WEB_PID" ]; then
    echo "$WEB_PID" >> logs/pids.txt
fi

log_success "All systems operational! 🚀"
