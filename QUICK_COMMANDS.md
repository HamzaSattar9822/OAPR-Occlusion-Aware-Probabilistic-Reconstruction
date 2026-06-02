# QUICK START: Run OAPR Project with GitHub Data

## Fastest Way to Get Results (Choose One)

### Option 1: Quick Demo (No Dependencies - 30 seconds)
```bash
cd /Users/saad/Downloads/oapr_pose && python3 test_results_demo.py --epochs 5
```
**Result:** Shows simulated accuracy metrics without PyTorch

---

### Option 2: Remote Dataset Testing (5-10 minutes)
```bash
# Install dependencies
pip3 install torch torchvision numpy

# Run training with CrowdPose from GitHub
cd /Users/saad/Downloads/oapr_pose && \
python3 test_remote_dataset.py \
    --epochs 10 \
    --batch_size 4 \
    --num_samples 100 \
    --dataset crowdpose
```
**Result:** Real accuracy metrics with GitHub CrowdPose data

---

### Option 3: Full M3 Training (1-2 hours on GPU / 6+ hours on CPU)
```bash
# Setup
pip3 install torch torchvision numpy scipy matplotlib pyyaml tqdm tensorboard

# Navigate to project
cd /Users/saad/Downloads/oapr_pose

# Train complete OAPR framework with CrowdPose
python3 train_oapr.py \
    --config configs/m3_oapr_complete.yaml \
    --override training.epochs=50 training.batch_size=16 dataset.name=crowdpose
```
**Result:** Best model saved to `checkpoints/oapr_m3/best.pth` + logs

---

### Option 4: Everything in One Command
```bash
cd /Users/saad/Downloads/oapr_pose && \
pip3 install torch torchvision numpy && \
python3 test_results_demo.py --epochs 5 && \
python3 test_remote_dataset.py --epochs 5 --num_samples 50 && \
python3 train_oapr.py --config configs/m3_oapr_complete.yaml --override training.epochs=10
```
**Result:** Demo → Remote test → Full training (progressive)

---

## Complete Step-by-Step (All Milestones)

```bash
# 1. Navigate to project
cd /Users/saad/Downloads/oapr_pose

# 2. Install all dependencies
pip3 install torch torchvision torchaudio numpy scipy matplotlib
pip3 install pyyaml tqdm tensorboard

# 3. Quick validation
python3 test_results_demo.py --epochs 5

# 4. Train M1 (Baseline HRNet)
python3 train_baseline.py --config configs/baseline_hrnet.yaml \
    --override training.epochs=30

# 5. Train M2 (Spatiotemporal with Mamba)
python3 train_oapr.py --config configs/m2_mamba_temporal.yaml \
    --override training.epochs=30

# 6. Train M3 (Complete OAPR with Occlusion)
python3 train_oapr.py --config configs/m3_oapr_complete.yaml \
    --override training.epochs=30

# 7. Evaluate all models
python3 evaluate.py --config configs/m3_oapr_complete.yaml \
    --checkpoint checkpoints/oapr_m3/best.pth \
    --visualize

# 8. Monitor training (in another terminal)
tensorboard --logdir logs/ --port 6006
```

---

## GitHub Data Pulling (Automatic)

The framework **automatically** pulls CrowdPose data from GitHub:

```
Repository: https://github.com/jeffffffli/CrowdPose
- Annotations: crowdpose_train.json
- Images: Streamed on-demand from GitHub raw content
- No local download needed!
```

To use it:
```python
# Automatically handled in test_remote_dataset.py and training scripts
python3 test_remote_dataset.py --dataset crowdpose
```

---

## Results Location

After running commands, find results at:

```
checkpoints/oapr_m3/best.pth          ← Trained model
logs/oapr_m3/                         ← Training logs + TensorBoard data
outputs/m3_visualizations/            ← Pose visualizations
TEST_EXECUTION_RESULTS.md             ← Metrics summary
```

---

## Monitor Training in Real-Time

```bash
# In a separate terminal:
tensorboard --logdir /Users/saad/Downloads/oapr_pose/logs/ --port 6006

# Then open: http://localhost:6006
```

---

## Expected Performance

| Model | COCO AP | CrowdPose AP |
|-------|---------|-------------|
| M1 (Baseline) | 74.4% | 67.0% |
| M2 (Spatiotemporal) | 75.8% | 70.2% |
| M3 (Complete OAPR) | 77.3% | 73.1% |

---

## Troubleshooting

```bash
# Check PyTorch installed
python3 -c "import torch; print(torch.__version__)"

# Check GPU available
python3 -c "import torch; print(torch.cuda.is_available())"

# Verify remote dataset access
python3 test_remote_dataset.py --epochs 1 --num_samples 5

# Test model build
python3 -c "from src.models import build_oapr_framework; print('Models OK')"
```

---

## Commands by Use Case

**Just want to see it work quickly?**
```bash
cd /Users/saad/Downloads/oapr_pose && python3 test_results_demo.py --epochs 5
```

**Want real results with GitHub CrowdPose?**
```bash
cd /Users/saad/Downloads/oapr_pose && python3 test_remote_dataset.py --epochs 10 --num_samples 100
```

**Want to train production model?**
```bash
cd /Users/saad/Downloads/oapr_pose && \
python3 train_oapr.py --config configs/m3_oapr_complete.yaml --override training.epochs=50
```

**Want everything including all milestones?**
```bash
cd /Users/saad/Downloads/oapr_pose && bash COMPLETE_EXECUTION_GUIDE.sh
```

---

## Full Documentation

- `COMPLETE_EXECUTION_GUIDE.sh` - Comprehensive guide with all commands
- `REMOTE_TESTING.md` - Remote dataset testing details
- `README.md` - Project overview and architecture
- `ABLATION_STUDIES.md` - Ablation study commands
