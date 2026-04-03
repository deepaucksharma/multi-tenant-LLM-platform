# Running Multi-Tenant LLM Platform on AMD GPU (WSL2)

## 🎯 Overview

This guide shows how to run the entire project on AMD GPU using WSL2 with ROCm support.

## 📋 Prerequisites

### System Requirements
- Windows 11 (build 22000+) or Windows 10 (build 19044+)
- WSL2 enabled
- AMD GPU (RDNA 2/3 or newer recommended)
- 16GB+ RAM
- 50GB+ free disk space

### Supported AMD GPUs
- **RDNA 3**: RX 7900 XTX, 7900 XT, 7800 XT, 7700 XT, 7600
- **RDNA 2**: RX 6950 XT, 6900 XT, 6800 XT, 6700 XT, 6600 XT
- **RDNA 1**: RX 5700 XT, 5600 XT (limited support)

## 🚀 Installation Steps

### Step 1: Install ROCm in WSL2

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install prerequisites
sudo apt install -y wget gnupg2

# Add ROCm repository (Ubuntu 22.04)
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.0.2 jammy main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update

# Install ROCm
sudo apt install -y rocm-hip-sdk rocm-libs

# Add user to render and video groups
sudo usermod -a -G render,video $USER

# Reboot WSL
exit
# Then in PowerShell: wsl --shutdown
# Restart WSL
```

### Step 2: Verify ROCm Installation

```bash
# Check ROCm version
/opt/rocm/bin/rocminfo

# Check GPU
/opt/rocm/bin/rocm-smi

# You should see your AMD GPU listed
```

### Step 3: Install PyTorch with ROCm Support

```bash
cd /home/deepak/src/ai-poc

# Uninstall CPU PyTorch
pip uninstall torch torchvision torchaudio -y

# Install ROCm PyTorch (ROCm 6.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify PyTorch sees GPU
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch version: 2.x.x+rocm6.0
ROCm available: True
GPU count: 1
GPU name: AMD Radeon RX 7900 XTX (or your GPU)
```

### Step 4: Update Environment Configuration

```bash
# Edit .env file
nano .env
```

Change:
```bash
DEVICE=cuda  # Keep as 'cuda' (PyTorch uses same API for ROCm)
```

Add ROCm environment variables:
```bash
# Add to .env or ~/.bashrc
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For RDNA 3 (adjust for your GPU)
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

**GFX Version by GPU:**
- RDNA 3 (RX 7000): `11.0.0`
- RDNA 2 (RX 6000): `10.3.0`
- RDNA 1 (RX 5000): `10.1.0`

### Step 5: Install Project Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install additional ROCm-specific packages
pip install bitsandbytes  # May need ROCm build
```

**Note:** If bitsandbytes doesn't work with ROCm, you can:
1. Skip 4-bit quantization (remove `load_in_4bit` from model loading)
2. Use 8-bit quantization instead
3. Build bitsandbytes from source with ROCm support

## 🏃 Running the Project

### Option 1: Full End-to-End (Recommended)

```bash
# Run everything with GPU support
./run_e2e.sh --demo

# This will:
# 1. Generate synthetic data
# 2. Build RAG indexes
# 3. Run tests
# 4. Start inference server (with GPU)
# 5. Start monitoring dashboard
# 6. Run demo queries
```

### Option 2: Step-by-Step Execution

```bash
# 1. Generate data (CPU task)
make data

# 2. Build RAG indexes (uses GPU for embeddings)
make index

# 3. Train adapters (GPU required)
make train

# 4. Start inference server (GPU for generation)
make serve

# 5. Start monitoring dashboard
make monitor

# 6. Start web app (optional)
make web-install
make web

# 7. Run tests
make test
```

### Option 3: Training Only

```bash
# Train SIS tenant adapter
python3 training/sft_train.py --tenant sis --epochs 3

# Train MFG tenant adapter
python3 training/sft_train.py --tenant mfg --epochs 3

# Train DPO (alignment)
python3 training/dpo_train.py --tenant sis --epochs 1
```

## 🔧 Troubleshooting

### Issue 1: ROCm Not Detected

```bash
# Check if GPU is visible
/opt/rocm/bin/rocm-smi

# If not visible, check WSL GPU support
# In PowerShell (as Admin):
wsl --update
wsl --shutdown

# Check Windows GPU driver is up to date
```

### Issue 2: PyTorch Not Using GPU

```bash
# Verify PyTorch installation
python3 -c "import torch; print(torch.version.hip)"

# Should show ROCm version like: 6.0.32831

# If None, reinstall PyTorch with ROCm
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

### Issue 3: Out of Memory (OOM)

```bash
# Reduce batch size in training configs
nano training/configs/sft_sis.yaml

# Change:
per_device_train_batch_size: 2  # Reduce from 4
gradient_accumulation_steps: 8   # Increase from 4

# Or use gradient checkpointing
gradient_checkpointing: true
```

### Issue 4: Slow Performance

```bash
# Check GPU utilization
watch -n 1 /opt/rocm/bin/rocm-smi

# If low utilization:
# 1. Increase batch size
# 2. Use mixed precision (fp16)
# 3. Disable gradient checkpointing if memory allows
```

### Issue 5: bitsandbytes Not Working

If 4-bit quantization fails, modify model loading:

```python
# Edit inference/adapter_manager.py
# Change from:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # Remove this
    ...
)

# To:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use fp16 instead
    device_map="auto",
)
```

## 📊 Performance Optimization

### 1. Enable Flash Attention (if supported)

```bash
pip install flash-attn --no-build-isolation
```

### 2. Use Optimized Kernels

```bash
# Set environment variables
export PYTORCH_ROCM_ARCH=gfx1100  # For RDNA 3
export ROCM_FORCE_ENABLE_FLASH_ATTENTION=1
```

### 3. Tune Memory Settings

```bash
# Add to .env
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=4
export AMD_DIRECT_DISPATCH=1
```

### 4. Monitor GPU Usage

```bash
# Real-time monitoring
watch -n 1 /opt/rocm/bin/rocm-smi

# Or use
/opt/rocm/bin/rocm-smi --showmeminfo vram --showuse
```

## 🎯 Expected Performance

### With AMD RX 7900 XTX (24GB VRAM)

| Task | Time | Notes |
|------|------|-------|
| Data Pipeline | 2-5 min | CPU-bound |
| RAG Index Build | 1-2 min | GPU for embeddings |
| SFT Training (3 epochs) | 15-30 min | Depends on data size |
| DPO Training (1 epoch) | 10-15 min | Smaller dataset |
| Inference (per request) | 1-3s | With RAG |
| Streaming (first token) | 200-500ms | TTFT |

### With AMD RX 6800 XT (16GB VRAM)

| Task | Time | Notes |
|------|------|-------|
| SFT Training | 20-40 min | Reduce batch size |
| Inference | 1.5-4s | Slightly slower |

## 🚀 Quick Start Commands

```bash
# 1. Install ROCm and PyTorch (one-time setup)
./scripts/setup_rocm.sh  # Create this script from Step 1-3

# 2. Verify GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# 3. Run everything
./run_e2e.sh --demo

# 4. Test inference
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "sis",
    "message": "What documents do I need for enrollment?",
    "use_rag": true,
    "model_type": "sft",
    "max_new_tokens": 200
  }'

# 5. Monitor GPU
watch -n 1 /opt/rocm/bin/rocm-smi
```

## 📝 Training Configuration for AMD GPU

Create optimized config for your GPU:

```yaml
# training/configs/sft_sis_amd.yaml
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
  per_device_train_batch_size: 4  # Adjust based on VRAM
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_ratio: 0.1
  fp16: true  # Use mixed precision
  gradient_checkpointing: true  # Save memory
  optim: adamw_torch  # Use PyTorch optimizer
  max_grad_norm: 1.0
  logging_steps: 10
  eval_steps: 50
  save_steps: 100
```

## 🔍 Debugging

### Check GPU Memory Usage

```bash
# During training
/opt/rocm/bin/rocm-smi --showmeminfo vram

# In Python
python3 -c "
import torch
print(f'Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB')
print(f'Reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB')
print(f'Max allocated: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB')
"
```

### Enable Verbose Logging

```bash
# Set environment variables
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export ROCM_VERBOSE=1
export HIP_VISIBLE_DEVICES=0
```

### Profile Performance

```bash
# Use ROCm profiler
/opt/rocm/bin/rocprof python3 training/sft_train.py --tenant sis

# Or PyTorch profiler
python3 -c "
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Your code here
    pass

print(prof.key_averages().table(sort_by='cuda_time_total'))
"
```

## 📚 Additional Resources

- **ROCm Documentation**: https://rocm.docs.amd.com/
- **PyTorch ROCm**: https://pytorch.org/get-started/locally/
- **WSL GPU Support**: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
- **AMD GPU Optimization**: https://github.com/ROCm/ROCm

## 🎉 Success Checklist

- [ ] ROCm installed and GPU detected
- [ ] PyTorch with ROCm support installed
- [ ] GPU visible in PyTorch (`torch.cuda.is_available() == True`)
- [ ] Data pipeline completed
- [ ] RAG indexes built
- [ ] Training runs without OOM
- [ ] Inference server using GPU
- [ ] All tests passing
- [ ] Monitoring dashboard shows GPU stats

## 💡 Tips

1. **Start Small**: Test with 1 epoch first to verify GPU works
2. **Monitor Memory**: Use `rocm-smi` to watch VRAM usage
3. **Adjust Batch Size**: If OOM, reduce batch size and increase gradient accumulation
4. **Use FP16**: Mixed precision training is faster and uses less memory
5. **Gradient Checkpointing**: Trades compute for memory (slower but fits larger models)

## 🆘 Getting Help

If you encounter issues:

1. Check ROCm version: `/opt/rocm/bin/rocminfo | grep "Name:"`
2. Check PyTorch version: `python3 -c "import torch; print(torch.__version__)"`
3. Check GPU: `/opt/rocm/bin/rocm-smi`
4. Check logs: `tail -f logs/inference.log`
5. Open issue on GitHub with error details

---

**Ready to run on AMD GPU!** 🚀

Start with: `./run_e2e.sh --demo`
