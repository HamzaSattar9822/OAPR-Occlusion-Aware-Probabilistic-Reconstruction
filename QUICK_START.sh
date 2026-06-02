#!/bin/bash
# QUICK_START.sh
# Fast setup and training guide for OAPR framework

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         OAPR Framework — Quick Start Guide                     ║"
echo "║   Occlusion-Aware Probabilistic Pose Reconstruction            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Environment Setup
# ─────────────────────────────────────────────────────────────────────

echo "Step 1: Environment Setup"
echo "──────────────────────────"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "✓ Virtual environment activated"

echo "Installing dependencies (this may take 5-10 minutes)..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Download Datasets
# ─────────────────────────────────────────────────────────────────────

echo ""
echo "Step 2: Download Datasets"
echo "─────────────────────────"
echo "Note: Download takes 1-2 hours. Use wget -c for resumable downloads."
echo ""
echo "Option A: Download via scripts"
echo "  bash scripts/download_coco.sh"
echo "  bash scripts/download_crowdpose.sh"
echo ""
echo "Option B: Manual wget commands"
echo "  mkdir -p data/coco"
echo "  cd data/coco"
echo "  wget -c http://images.cocodataset.org/zips/train2017.zip"
echo "  wget -c http://images.cocodataset.org/zips/val2017.zip"
echo "  unzip '*.zip'"
echo "  cd ../.."
echo ""
read -p "Have you downloaded the datasets? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please download datasets first, then re-run this script."
    exit 1
fi

echo "✓ Datasets available"

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Quick Validation
# ─────────────────────────────────────────────────────────────────────

echo ""
echo "Step 3: Validation"
echo "─────────────────"
echo "Running model forward pass test..."

python3 << 'EOF'
import torch
from src.models import build_oapr_framework

try:
    cfg = {
        'model': {
            'num_keypoints': 17,
            'seq_len': 7,
            'hidden_size': 256,
            'use_mamba': False,  # Use fallback on first run
        },
        'loss': {'type': 'cauchy_mixture'}
    }
    
    model = build_oapr_framework(cfg)
    video = torch.randn(2, 7, 17, 2)
    output = model(video)
    
    print("✓ Model validation passed")
    print(f"  Input shape: {video.shape}")
    print(f"  Keypoints: {output['keypoints'].shape}")
    print(f"  Confidence: {output['confidence'].shape}")
    print(f"  Occlusion mask: {output['occlusion_mask'].shape}")
except Exception as e:
    print(f"✗ Validation failed: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Choose Training Path
# ─────────────────────────────────────────────────────────────────────

echo ""
echo "Step 4: Training Options"
echo "───────────────────────"
echo ""
echo "Choose what to run:"
echo ""
echo "  1) M1 Baseline (HRNet, 1-2 hours)"
echo "     python train_baseline.py --config configs/baseline_hrnet.yaml"
echo ""
echo "  2) M2 Spatiotemporal (Mamba backbone, 36 hours)"
echo "     python train_oapr.py --config configs/m2_mamba_temporal.yaml"
echo ""
echo "  3) M3 Complete OAPR (Full framework, 42 hours)"
echo "     python train_oapr.py --config configs/m3_oapr_complete.yaml"
echo ""
echo "  4) Run Quick Test (2 epochs, 5 minutes)"
echo "     python train_oapr.py --config configs/m3_oapr_complete.yaml \\"
echo "         --override training.epochs=2 training.batch_size=4"
echo ""

# ─────────────────────────────────────────────────────────────────────
# STEP 5: Monitoring
# ─────────────────────────────────────────────────────────────────────

echo ""
echo "Step 5: Monitor Training"
echo "───────────────────────"
echo ""
echo "In another terminal, run:"
echo "  tensorboard --logdir logs/"
echo ""
echo "Then open: http://localhost:6006"
echo ""

# ─────────────────────────────────────────────────────────────────────
# STEP 6: Evaluation
# ─────────────────────────────────────────────────────────────────────

echo "Step 6: Evaluate (After Training)"
echo "─────────────────────────────────"
echo ""
echo "  python evaluate.py \\"
echo "    --config configs/m3_oapr_complete.yaml \\"
echo "    --checkpoint checkpoints/oapr_m3/best.pth \\"
echo "    --visualize --vis_dir outputs/visualizations"
echo ""

# ─────────────────────────────────────────────────────────────────────
# STEP 7: Documentation
# ─────────────────────────────────────────────────────────────────────

echo "Step 7: Documentation"
echo "───────────────────"
echo ""
echo "Read these files for more details:"
echo "  - README.md                 (Project overview & architecture)"
echo "  - IMPLEMENTATION_SUMMARY.md (Detailed implementation guide)"
echo "  - ABLATION_STUDIES.md       (Ablation study recipes)"
echo ""

# ─────────────────────────────────────────────────────────────────────
# STEP 8: Quick Reference
# ─────────────────────────────────────────────────────────────────────

echo "Quick Reference: Config Overrides"
echo "──────────────────────────────────"
echo ""
echo "# Change sequence length"
echo "  --override model.seq_len=5"
echo ""
echo "# Disable Mamba (use Transformer)"
echo "  --override model.use_mamba=false"
echo ""
echo "# Disable occlusion module"
echo "  --override model.reconstruct_occluded=false"
echo ""
echo "# Use different loss"
echo "  --override loss.type=laplace"
echo ""
echo "# Resume from checkpoint"
echo "  --resume checkpoints/oapr_m3/best.pth"
echo ""
echo "# Evaluate on CrowdPose instead of COCO"
echo "  --override dataset.name=crowdpose"
echo ""

# ─────────────────────────────────────────────────────────────────────

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Setup Complete! Ready to train OAPR framework.               ║"
echo "║                                                                ║"
echo "║  Next: Choose a training option (see Step 4 above)             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
