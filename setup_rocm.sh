#!/bin/bash
# Automated ROCm Setup for WSL2 with AMD GPU
# This script installs ROCm and configures PyTorch for AMD GPU

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "ROCm Setup for AMD GPU on WSL2"
echo "=========================================="
echo ""

# Check if running on WSL
if ! grep -qi microsoft /proc/version; then
    log_error "This script is designed for WSL2. Please run on WSL."
    exit 1
fi

log_success "Running on WSL2"

# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
log_info "Ubuntu version: $UBUNTU_VERSION"

# Step 1: Update system
log_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install prerequisites
log_info "Installing prerequisites..."
sudo apt install -y wget gnupg2 software-properties-common

# Step 3: Add ROCm repository
log_info "Adding ROCm repository..."
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

if [ "$UBUNTU_VERSION" == "22.04" ]; then
    ROCM_REPO="jammy"
elif [ "$UBUNTU_VERSION" == "20.04" ]; then
    ROCM_REPO="focal"
else
    log_warning "Ubuntu $UBUNTU_VERSION may not be officially supported. Using jammy repo."
    ROCM_REPO="jammy"
fi

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.0.2 $ROCM_REPO main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update

# Step 4: Install ROCm
log_info "Installing ROCm (this may take 10-15 minutes)..."
sudo apt install -y rocm-hip-sdk rocm-libs

# Step 5: Add user to groups
log_info "Adding user to render and video groups..."
sudo usermod -a -G render,video $USER

# Step 6: Set environment variables
log_info "Setting up environment variables..."

ENV_VARS="
# ROCm Environment Variables
export ROCM_PATH=/opt/rocm
export PATH=\$ROCM_PATH/bin:\$PATH
export LD_LIBRARY_PATH=\$ROCM_PATH/lib:\$LD_LIBRARY_PATH

# AMD GPU Optimization
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # Adjust for your GPU
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=4
export AMD_DIRECT_DISPATCH=1
export PYTORCH_ROCM_ARCH=gfx1100  # Adjust for your GPU
"

if ! grep -q "ROCM_PATH" ~/.bashrc; then
    echo "$ENV_VARS" >> ~/.bashrc
    log_success "Environment variables added to ~/.bashrc"
else
    log_info "Environment variables already in ~/.bashrc"
fi

# Source the variables for current session
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Step 7: Verify ROCm installation
log_info "Verifying ROCm installation..."
if [ -f "/opt/rocm/bin/rocminfo" ]; then
    log_success "ROCm installed successfully"
    /opt/rocm/bin/rocminfo | grep "Name:" | head -5
else
    log_error "ROCm installation failed"
    exit 1
fi

# Step 8: Check GPU
log_info "Checking for AMD GPU..."
if /opt/rocm/bin/rocm-smi &> /dev/null; then
    log_success "AMD GPU detected:"
    /opt/rocm/bin/rocm-smi --showproductname
else
    log_warning "No AMD GPU detected or driver issue"
    log_info "You may need to:"
    log_info "  1. Update Windows GPU driver"
    log_info "  2. Run 'wsl --update' in PowerShell"
    log_info "  3. Restart WSL with 'wsl --shutdown'"
fi

# Step 9: Install PyTorch with ROCm
log_info "Installing PyTorch with ROCm support..."

# Uninstall existing PyTorch
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install ROCm PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Step 10: Verify PyTorch
log_info "Verifying PyTorch installation..."
python3 << 'EOF'
import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: GPU not detected by PyTorch")
EOF

# Step 11: Install project dependencies
if [ -f "requirements.txt" ]; then
    log_info "Installing project dependencies..."
    pip install -r requirements.txt
    log_success "Dependencies installed"
fi

# Step 12: Update .env file
if [ -f ".env" ]; then
    log_info "Updating .env file..."
    sed -i 's/DEVICE=cpu/DEVICE=cuda/' .env
    log_success ".env updated to use GPU"
elif [ -f ".env.example" ]; then
    log_info "Creating .env from .env.example..."
    cp .env.example .env
    sed -i 's/DEVICE=cpu/DEVICE=cuda/' .env
    log_success ".env created"
fi

echo ""
echo "=========================================="
echo "ROCm Setup Complete!"
echo "=========================================="
echo ""
log_success "Next steps:"
echo "  1. Restart WSL: exit, then in PowerShell run 'wsl --shutdown'"
echo "  2. Start WSL again"
echo "  3. Verify GPU: /opt/rocm/bin/rocm-smi"
echo "  4. Test PyTorch: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "  5. Run project: ./run_e2e.sh --demo"
echo ""
log_warning "IMPORTANT: You must restart WSL for group changes to take effect!"
echo ""
