#!/bin/bash
# OAPR Project - Virtual Environment Setup & Installation

echo "=============================================================================="
echo "OAPR Framework - Environment Setup"
echo "=============================================================================="
echo ""

# Step 1: Navigate to project
echo "[1/4] Navigating to project directory..."
cd /Users/saad/Downloads/oapr_pose || exit 1
echo "✓ In: $(pwd)"
echo ""

# Step 2: Create virtual environment
echo "[2/4] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Step 3: Activate virtual environment
echo "[3/4] Activating virtual environment..."
source venv/bin/activate
echo "✓ Activated: $(which python3)"
echo ""

# Step 4: Install dependencies
echo "[4/4] Installing dependencies..."
echo "This may take 3-5 minutes... please wait"
echo ""

# Install core packages
pip3 install --quiet --upgrade pip
pip3 install --quiet opencv-python torch torchvision numpy scipy matplotlib pyyaml tqdm tensorboard einops pillow

echo ""
echo "✓ Installation complete!"
echo ""
echo "=============================================================================="
echo "VERIFICATION"
echo "=============================================================================="
python3 -c "import cv2; print('✓ cv2 (OpenCV) installed')" 2>/dev/null || echo "✗ cv2 not installed"
python3 -c "import torch; print('✓ torch installed')" 2>/dev/null || echo "✗ torch not installed"
python3 -c "import numpy; print('✓ numpy installed')" 2>/dev/null || echo "✗ numpy not installed"
python3 -c "import scipy; print('✓ scipy installed')" 2>/dev/null || echo "✗ scipy not installed"

echo ""
echo "=============================================================================="
echo "READY TO RUN!"
echo "=============================================================================="
echo ""
echo "Your environment is now ready. To run the tests:"
echo ""
echo "  python3 test_results_demo.py --epochs 5"
echo ""
echo "Or:"
echo ""
echo "  python3 test_remote_dataset.py --epochs 10 --num_samples 100"
echo ""
echo "=============================================================================="
