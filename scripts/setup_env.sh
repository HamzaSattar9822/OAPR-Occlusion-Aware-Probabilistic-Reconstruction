#!/bin/bash
# setup_env.sh — Create conda environment and install all dependencies

set -e

ENV_NAME="oapr_pose"
PYTHON_VERSION="3.10"

echo "============================================"
echo "  OAPR Pose Estimation — Environment Setup"
echo "============================================"

# Check conda
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create environment
echo "[1/5] Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# PyTorch with CUDA (adjust cuda version if needed — check with: nvidia-smi)
echo "[2/5] Installing PyTorch with CUDA 11.8..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Core requirements
echo "[3/5] Installing core requirements..."
pip install -r requirements.txt

# pycocotools needs C compiler
echo "[4/5] Installing pycocotools (COCO evaluation)..."
pip install pycocotools

# CrowdPose eval tools
echo "[5/5] Installing CrowdPose tools..."
pip install crowdposetools || echo "[WARN] crowdposetools not found on PyPI — will use local eval script"

echo ""
echo "✅ Environment '$ENV_NAME' ready."
echo ""
echo "Activate with:"
echo "   conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "   bash scripts/download_coco.sh"
echo "   bash scripts/download_crowdpose.sh"
