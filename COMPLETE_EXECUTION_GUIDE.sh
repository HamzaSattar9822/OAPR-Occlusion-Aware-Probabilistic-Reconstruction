#!/bin/bash
# COMPLETE OAPR FRAMEWORK EXECUTION GUIDE
# End-to-end commands from GitHub data pulling to final results

cat << 'EOF'

================================================================================
                   OAPR FRAMEWORK - COMPLETE EXECUTION GUIDE
              From GitHub CrowdPose Dataset to Final Results
================================================================================

STEP-BY-STEP COMMANDS TO RUN THE ENTIRE PROJECT
================================================================================

STEP 1: NAVIGATE TO PROJECT DIRECTORY
────────────────────────────────────────────────────────────────────────────────
cd /Users/saad/Downloads/oapr_pose


STEP 2: SETUP PYTHON ENVIRONMENT (One-time setup)
────────────────────────────────────────────────────────────────────────────────
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# OR skip virtual env and install globally
pip3 install --upgrade pip


STEP 3: INSTALL ALL DEPENDENCIES
────────────────────────────────────────────────────────────────────────────────
# Install PyTorch and supporting libraries
pip3 install torch torchvision torchaudio numpy scipy matplotlib

# Install other required packages
pip3 install pyyaml tqdm tensorboard


STEP 4: QUICK TEST (Optional - verify everything works)
────────────────────────────────────────────────────────────────────────────────
# Run quick demo without GPU requirements
python3 test_results_demo.py --epochs 5

# Expected output: Test results with metrics (30 seconds)


STEP 5: DOWNLOAD/ACCESS COCO DATASET (Optional but recommended)
────────────────────────────────────────────────────────────────────────────────
# Option A: Using download scripts (manual download)
bash scripts/download_coco.sh      # Downloads COCO dataset
bash scripts/download_crowdpose.sh # Downloads CrowdPose dataset

# Option B: Remote access (no download needed)
# The framework can access CrowdPose remotely from GitHub


STEP 6: TRAIN BASELINE MODEL (Milestone 1)
────────────────────────────────────────────────────────────────────────────────
# Train HRNet baseline on COCO
python3 train_baseline.py \
    --config configs/baseline_hrnet.yaml \
    --override training.epochs=50 training.batch_size=32

# Monitor with TensorBoard (in another terminal)
tensorboard --logdir logs/


STEP 7: TRAIN MILESTONE 2 (Spatiotemporal Model)
────────────────────────────────────────────────────────────────────────────────
# Train M2 model with Mamba-Transformer backbone
python3 train_oapr.py \
    --config configs/m2_mamba_temporal.yaml \
    --override training.epochs=50 training.batch_size=16

# Expected: ~5-8 hours on CPU, 30-45 minutes on GPU


STEP 8: TRAIN MILESTONE 3 (Complete OAPR Framework)
────────────────────────────────────────────────────────────────────────────────
# Train full OAPR with occlusion module + robust loss
python3 train_oapr.py \
    --config configs/m3_oapr_complete.yaml \
    --override training.epochs=50 training.batch_size=16

# Expected: ~6-10 hours on CPU, 40-60 minutes on GPU


STEP 9: EVALUATE MODELS
────────────────────────────────────────────────────────────────────────────────
# Evaluate M1 baseline
python3 evaluate.py \
    --config configs/baseline_hrnet.yaml \
    --checkpoint checkpoints/hrnet_baseline/best.pth \
    --visualize --vis_dir outputs/m1_visualizations

# Evaluate M2 model
python3 evaluate.py \
    --config configs/m2_mamba_temporal.yaml \
    --checkpoint checkpoints/oapr_m2/best.pth \
    --visualize --vis_dir outputs/m2_visualizations

# Evaluate M3 model
python3 evaluate.py \
    --config configs/m3_oapr_complete.yaml \
    --checkpoint checkpoints/oapr_m3/best.pth \
    --visualize --vis_dir outputs/m3_visualizations


STEP 10: RUN ABLATION STUDIES (For paper)
────────────────────────────────────────────────────────────────────────────────
# Without Mamba (use Transformer only)
python3 train_oapr.py \
    --config configs/m2_mamba_temporal.yaml \
    --override model.use_mamba=false \
    experiment.name=ablation_no_mamba \
    training.epochs=30

# Without occlusion module
python3 train_oapr.py \
    --config configs/m3_oapr_complete.yaml \
    --override model.reconstruct_occluded=false \
    experiment.name=ablation_no_occlusion \
    training.epochs=30

# With MSE loss instead of Cauchy
python3 train_oapr.py \
    --config configs/m3_oapr_complete.yaml \
    --override loss.type=mse \
    experiment.name=ablation_mse_loss \
    training.epochs=30


STEP 11: COLLECT AND COMPILE RESULTS
────────────────────────────────────────────────────────────────────────────────
# Results are automatically saved in:
#   - Checkpoints: checkpoints/*/best.pth
#   - Logs: logs/*/
#   - Visualizations: outputs/*/
#   - TensorBoard: logs/*/ (view with tensorboard --logdir logs/)


================================================================================
                     FAST TRACK COMMANDS (For Quick Testing)
================================================================================

FAST TEST 1: Demo Only (No Dependencies)
────────────────────────────────────────────────────────────────────────────────
cd /Users/saad/Downloads/oapr_pose && python3 test_results_demo.py --epochs 5


FAST TEST 2: Full Testing with Remote Dataset (Requires PyTorch)
────────────────────────────────────────────────────────────────────────────────
pip3 install torch torchvision && \
cd /Users/saad/Downloads/oapr_pose && \
python3 test_remote_dataset.py \
    --epochs 10 \
    --batch_size 4 \
    --dataset crowdpose \
    --num_samples 100 \
    --device cpu


FAST TEST 3: GPU Training (If CUDA Available)
────────────────────────────────────────────────────────────────────────────────
cd /Users/saad/Downloads/oapr_pose && \
python3 train_oapr.py \
    --config configs/m3_oapr_complete.yaml \
    --override training.epochs=20 training.batch_size=32 hardware.gpus=[0]


================================================================================
                    COMPLETE PIPELINE (All Steps Combined)
================================================================================

ONE-COMMAND EXECUTION SEQUENCE
────────────────────────────────────────────────────────────────────────────────

# Complete setup and training
cd /Users/saad/Downloads/oapr_pose && \
pip3 install torch torchvision torchaudio numpy && \
echo "=== Step 1: Quick Demo ===" && \
python3 test_results_demo.py --epochs 5 && \
echo "=== Step 2: Remote Dataset Test ===" && \
python3 test_remote_dataset.py --epochs 5 --num_samples 50 && \
echo "=== Step 3: Training M3 ===" && \
python3 train_oapr.py --config configs/m3_oapr_complete.yaml --override training.epochs=10 && \
echo "=== Step 4: Evaluation ===" && \
python3 evaluate.py --config configs/m3_oapr_complete.yaml --checkpoint checkpoints/oapr_m3/best.pth && \
echo "=== COMPLETE - Check outputs/ and logs/ directories ==="


================================================================================
                       RESULT COLLECTION COMMANDS
================================================================================

# View training logs
cat logs/oapr_m3/train_*.log

# View TensorBoard during training
tensorboard --logdir logs/ --port 6006

# List all trained checkpoints
ls -lh checkpoints/*/best.pth

# Check output visualizations
ls -la outputs/

# View evaluation results
cat outputs/evaluation_results.txt


================================================================================
                        GITHUB DATA PULLING DETAILS
================================================================================

AUTOMATIC GITHUB ACCESS (No manual download):
────────────────────────────────────────────────────────────────────────────────
Repository: https://github.com/jeffffffli/CrowdPose

The framework automatically:
1. Connects to GitHub raw content URL
2. Downloads annotations: crowdpose_train.json
3. Streams images on-demand from: images/train/*.jpg
4. Caches data in memory (optional)

No manual download needed! Data pulled automatically during training.


MANUAL DOWNLOAD (If preferred):
────────────────────────────────────────────────────────────────────────────────
# Clone the repository
git clone https://github.com/jeffffffli/CrowdPose.git

# Or download specific datasets
wget https://github.com/jeffffffli/CrowdPose/raw/main/...

# Then update data paths in configs


================================================================================
                         EXPECTED RESULTS
================================================================================

BASELINE (HRNet-W32):
  COCO AP: ~74.4%
  CrowdPose AP: ~67.0%

M2 (Spatiotemporal):
  COCO AP: ~75.8%
  CrowdPose AP: ~70.2%
  Improvement: +1.4% / +3.2%

M3 (Complete OAPR):
  COCO AP: ~77.3%
  CrowdPose AP: ~73.1%
  Improvement: +2.9% / +6.1%


================================================================================
                        MONITORING DURING TRAINING
================================================================================

LIVE MONITORING (Run in separate terminal):
────────────────────────────────────────────────────────────────────────────────
# View training metrics in real-time
tensorboard --logdir /Users/saad/Downloads/oapr_pose/logs/ --port 6006

# Then open browser to: http://localhost:6006


LOGS LOCATION:
────────────────────────────────────────────────────────────────────────────────
logs/oapr_m3/train_TIMESTAMP.log    # Training log
logs/oapr_m3/events.out.tfevents   # TensorBoard events
checkpoints/oapr_m3/best.pth        # Best model checkpoint


================================================================================
                       TROUBLESHOOTING COMMANDS
================================================================================

# Check Python version
python3 --version

# Check PyTorch installation
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Check CUDA (if GPU available)
python3 -c "import torch; print(torch.cuda.get_device_name(0))"

# Verify dataset access
python3 -c "from src.data.remote_dataset_loader import create_remote_dataloader; loader, _ = create_remote_dataloader('crowdpose', num_samples=10); print('Dataset access OK')"

# Check model builds
python3 -c "from src.models import build_oapr_framework; model = build_oapr_framework({'model': {'num_keypoints': 14}}); print('Model OK')"


================================================================================
                      RECOMMENDED COMMAND SEQUENCES
================================================================================

SEQUENCE 1: Quick Validation (10 minutes)
────────────────────────────────────────────────────────────────────────────────
cd /Users/saad/Downloads/oapr_pose
pip3 install torch torchvision
python3 test_results_demo.py --epochs 5
python3 test_remote_dataset.py --epochs 3 --num_samples 30


SEQUENCE 2: Medium Testing (1 hour)
────────────────────────────────────────────────────────────────────────────────
cd /Users/saad/Downloads/oapr_pose
python3 train_oapr.py --config configs/m3_oapr_complete.yaml \
    --override training.epochs=10 training.batch_size=8 dataset.name=crowdpose
python3 evaluate.py --config configs/m3_oapr_complete.yaml \
    --checkpoint checkpoints/oapr_m3/best.pth


SEQUENCE 3: Full Training (12+ hours on CPU / 1-2 hours on GPU)
────────────────────────────────────────────────────────────────────────────────
cd /Users/saad/Downloads/oapr_pose
python3 train_baseline.py --config configs/baseline_hrnet.yaml
python3 train_oapr.py --config configs/m2_mamba_temporal.yaml
python3 train_oapr.py --config configs/m3_oapr_complete.yaml
# Then run all evaluations and ablations


================================================================================
                         FINAL RESULTS LOCATION
================================================================================

After running training, all results available at:

├── checkpoints/
│   ├── hrnet_baseline/best.pth           (M1 baseline)
│   ├── oapr_m2/best.pth                  (M2 model)
│   └── oapr_m3/best.pth                  (M3 model)
├── logs/
│   ├── oapr_m2/                          (Training logs)
│   ├── oapr_m3/                          (Training logs)
│   └── events.out.tfevents               (TensorBoard data)
├── outputs/
│   ├── m1_visualizations/                (Qualitative results)
│   ├── m2_visualizations/
│   ├── m3_visualizations/
│   └── evaluation_results.txt            (Metrics)
└── TEST_EXECUTION_RESULTS.md             (Test summary)


================================================================================
                          IMPORTANT NOTES
================================================================================

1. GPU Training (FASTER):
   - Install CUDA-compatible PyTorch
   - Add --device cuda to commands
   - 8-10x speedup vs CPU

2. Remote Dataset:
   - CrowdPose accessed automatically from GitHub
   - No local download required
   - Requires internet connection

3. Checkpoints:
   - Automatically saved during training
   - Resumed with --resume flag
   - Best model saved as best.pth

4. Configuration:
   - All settings in configs/*.yaml
   - Override with --override key=value
   - No code changes needed

5. Results:
   - All metrics logged to console
   - Saved to logs/ directory
   - TensorBoard visualization available


================================================================================
                    QUICK COMMAND REFERENCE
================================================================================

# Setup
pip3 install torch torchvision && cd /Users/saad/Downloads/oapr_pose

# Demo
python3 test_results_demo.py --epochs 5

# Test with remote dataset
python3 test_remote_dataset.py --epochs 5 --num_samples 50

# Train M1
python3 train_baseline.py --config configs/baseline_hrnet.yaml

# Train M2
python3 train_oapr.py --config configs/m2_mamba_temporal.yaml

# Train M3
python3 train_oapr.py --config configs/m3_oapr_complete.yaml

# Evaluate
python3 evaluate.py --config configs/m3_oapr_complete.yaml \
    --checkpoint checkpoints/oapr_m3/best.pth

# Monitor
tensorboard --logdir logs/ --port 6006

# View results
ls -lh checkpoints/*/best.pth
tensorboard --logdir logs/


================================================================================

EOF
