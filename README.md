# OAPR: Occlusion-Aware Probabilistic Pose Reconstruction

A state-of-the-art framework for robust multi-human pose estimation under severe occlusion and crowded scenes.

**Key Innovation:** Combines spatiotemporal Mamba modeling, occlusion-aware reconstruction, and distribution-robust probabilistic learning into a unified end-to-end framework.

## Project Structure

```
oapr_pose/
├── configs/
│   ├── baseline_hrnet.yaml           # M1: HRNet baseline config
│   ├── m2_mamba_temporal.yaml        # M2: Spatiotemporal model config
│   └── m3_oapr_complete.yaml         # M3: Complete framework config
├── scripts/
│   ├── download_coco.sh
│   ├── download_crowdpose.sh
│   └── setup_env.sh
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── coco_dataset.py
│   │   ├── crowdpose_dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hrnet_baseline.py         # M1: HRNet-W32 baseline
│   │   ├── mamba_backbone.py         # M2: Hybrid Mamba-Transformer backbone
│   │   ├── occlusion_module.py       # M3: OAPR occlusion reconstruction module
│   │   ├── robust_loss.py            # M3: Cauchy/Laplace/mixture losses
│   │   └── oapr_framework.py         # M2+M3: Complete OAPR framework
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   └── checkpoint.py
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py
├── train_baseline.py                 # M1: HRNet baseline training
├── train_oapr.py                     # M2+M3: Unified OAPR training
├── evaluate.py                       # M1: Baseline evaluation
└── requirements.txt
```

## Milestones & Implementation Status

| Week | Deliverable | Status | Key Components |
|------|-------------|--------|-----------------|
| **1** | **Baseline + Dataset Setup** | COMPLETE | HRNet-W32 baseline, COCO/CrowdPose dataloaders, train/eval infrastructure |
| **2** | **Spatiotemporal Model** | COMPLETE | Hybrid Mamba-Transformer backbone, video clip processing (7-frame sequences), spatial transformer attention, temporal Mamba/fallback |
| **3** | **Occlusion Module + Robust Loss** | COMPLETE | Occlusion detector, pose reconstruction via spatial+temporal+instance context, Cauchy/Laplace/Mixture losses, uncertainty-aware outputs |
| **4** | **Experiments + Results** | IN PROGRESS | Training on COCO/CrowdPose, ablation studies |
| **5** | **Article Writing** | PENDING | Compile results and write IEEE paper |

## Core Innovations (M2 & M3)

### Milestone 2: Hybrid Spatiotemporal Backbone

**File:** `src/models/mamba_backbone.py`

- **Temporal Mamba:** Uses state-space model (Mamba/SSM) for long-range temporal dependencies
- **Spatial Transformer:** Per-frame attention for joint refinement
- **Instance Tokens:** Learnable embeddings for multi-person disambiguation
- **Fallback:** Auto-switches to Temporal Transformer if Mamba unavailable

```python
# Usage
from src.models import HybridMambaTransformer

backbone = HybridMambaTransformer(
    num_keypoints=17, seq_len=7, hidden_size=256,
    num_heads=8, num_layers=3, use_mamba=True
)
output = backbone(video_clip)  # (B, K, 3) -> x, y, confidence
```

### Milestone 3: Occlusion-Aware Reconstruction (OAPR Core)

**File:** `src/models/occlusion_module.py`

The primary novelty of the framework: instead of just predicting poses, we reconstruct missing joints using:

1. **Occlusion Detector:** Identifies low-confidence/occluded joints via uncertainty thresholds
2. **Spatial Context Encoder:** Graph-based reasoning over skeleton structure
3. **Temporal Context Encoder:** LSTM modeling of motion continuity
4. **Pose Reconstructor:** Fuses all contexts to recover occluded joints

```python
# Usage
from src.models import OcclusionAwarePoseReconstruction

oapr = OcclusionAwarePoseReconstruction(
    hidden_size=256, num_joints=17, seq_len=7
)
output = oapr(
    joint_features, predictions, confidence, joint_trajectories
)
# Returns: coordinates, confidence, occlusion_mask, occlusion_score
```

### Milestone 3: Robust Probabilistic Loss

**File:** `src/models/robust_loss.py`

Heavy-tailed distributions for robust regression under outliers and occlusion:

- **Cauchy Loss:** Most robust (unbounded tails)
- **Laplace Loss:** Moderate robustness
- **Cauchy Mixture:** Adaptive robustness (learnable scales)

Each joint outputs coordinates and uncertainty, enabling confidence-aware learning.

```python
# Usage
from src.models import ProbabilisticPoseLoss

loss_fn = ProbabilisticPoseLoss(loss_type='cauchy_mixture')
loss, loss_dict = loss_fn(
    predictions, targets, weights,
    uncertainties, occlusion_scores
)
```

## End-to-End Framework

**File:** `src/models/oapr_framework.py`

Unified model combining all components:

```python
from src.models import build_oapr_framework

model = build_oapr_framework(cfg)

# Forward pass
output = model(video_clip)
# Returns: keypoints, confidence, occlusion_mask, occlusion_score

# Training
loss, loss_dict = model.compute_loss(
    predictions, targets, weights,
    uncertainties, occlusion_scores
)
```

## Training

### Milestone 1: Baseline (HRNet)

```bash
python train_baseline.py --config configs/baseline_hrnet.yaml
```

### Milestone 2: Spatiotemporal Model

```bash
python train_oapr.py --config configs/m2_mamba_temporal.yaml
```

Configuration highlights:
- Video sequences: 7 frames
- Hybrid backbone with Mamba and Transformer
- Lower learning rate (5e-4) for temporal learning
- 150 epochs on COCO

### Milestone 3: Complete OAPR (Occlusion + Robust Loss)

```bash
python train_oapr.py --config configs/m3_oapr_complete.yaml
```

Configuration highlights:
- OAPR module enabled
- Cauchy mixture loss (distribution-robust)
- Occlusion simulation (random patch erasing)
- Temporal stability metrics
- CrowdPose evaluation

### Resume Training

```bash
python train_oapr.py --config configs/m3_oapr_complete.yaml \
    --resume checkpoints/best.pth
```

## Evaluation

```bash
python evaluate.py --config configs/baseline_hrnet.yaml \
    --checkpoint checkpoints/best.pth \
    --visualize --vis_dir outputs/visualizations
```

## Expected Results

### Baseline (HRNet-W32, M1)

| Dataset | AP | AP50 | AP75 |
|---------|----|----|------|
| COCO val2017 | 74.4 | 90.5 | 81.9 |
| CrowdPose test | 67.0 | 85.7 | 72.2 |

### With OAPR (M2+M3, expected improvements)

| Benchmark | Baseline -> OAPR | Notes |
|-----------|-----------------|-------|
| COCO (overall) | +0.5-1.5 percent AP | Marginal gains on uncrowded scenes |
| CrowdPose (crowded) | +2-3 percent AP | Significant gains on occlusion-heavy data |
| Temporal stability | +10-15 percent | Frame-to-frame consistency improvement |
| Occlusion robustness | +5-10 percent AP | Performance on self-occlusion subsets |

## Mandatory Client Requirements - Implementation Checklist

| Requirement | Status | File(s) |
|-------------|--------|---------|
| Hybrid Mamba + Transformer backbone | COMPLETE | `mamba_backbone.py` |
| Video-based spatiotemporal modeling (5-9 frames) | COMPLETE | `mamba_backbone.py`, `oapr_framework.py` |
| Instance-aware multi-person representation | COMPLETE | `occlusion_module.py`, `oapr_framework.py` |
| **Occlusion handling module** | COMPLETE | `occlusion_module.py` |
| Probabilistic/robust loss (Cauchy/Laplace) | COMPLETE | `robust_loss.py` |
| Uncertainty-aware outputs (confidence + coordinates) | COMPLETE | All models |
| Clean, modular code | COMPLETE | All new files |
| Training scripts with logging | COMPLETE | `train_oapr.py` |
| COCO + CrowdPose support | COMPLETE | `src/data/` |
| Ablation studies ready | COMPLETE | Configs support ablations |

## Architecture Diagram

```
Video Frames (B, T, 3, H, W)
    |
    v
[Optional: Pose Extractor if needed]
    |
    v
Video Clips (B, T, K, 2)
    |
    v
+-----------------------------------------------+
| MILESTONE 2: Hybrid Backbone                  |
+-----------------------------------------------+
| Temporal Mamba -> Long-range dependencies     |
| Spatial Transformer -> Joint refinement       |
| Instance Tokens -> Multi-person handling      |
+-----------------------------------------------+
    |
    v
Backbone Output (B, K, 3)
    |
    v
+-----------------------------------------------+
| MILESTONE 3: Occlusion Reconstruction (OAPR) |
+-----------------------------------------------+
| Detector -> Identify occluded joints          |
| Spatial Encoder -> Joint relationships        |
| Temporal Encoder -> Motion history            |
| Reconstructor -> Recover missing joints       |
+-----------------------------------------------+
    |
    v
Refined Coordinates (B, K, 2)
    |
    v
Robust Probabilistic Loss (Cauchy Mixture)
    |
    v
Backprop with Uncertainty Regularization
```

## Ablation Study Templates

All key ablations can be run via config overrides:

```bash
# M2: Without Mamba (Transformer only)
python train_oapr.py --config configs/m2_mamba_temporal.yaml \
    --override model.use_mamba=false

# M3: Without occlusion module
python train_oapr.py --config configs/m3_oapr_complete.yaml \
    --override model.reconstruct_occluded=false

# M3: Standard MSE loss (not robust)
python train_oapr.py --config configs/m3_oapr_complete.yaml \
    --override loss.type=mse
```

## Installation

```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies (updated with Mamba)
pip install -r requirements.txt

# Download datasets
bash scripts/download_coco.sh
bash scripts/download_crowdpose.sh
```

## Key Papers & References

1. **Mamba backbone:** Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (ICLR 2024)
2. **Robust loss:** Barron et al., "A General and Adaptive Robust Loss Function" (CVPR 2019)
3. **HRNet baseline:** Wang et al., "Deep High-Resolution Representation Learning for Visual Recognition" (TPAMI 2020)
4. **Pose estimation:** CMU-Pose, OpenPifPaf, SimpleBaseline

## Author Notes

This framework represents Milestones 2 and 3 of the OAPR project:

- **M2 (Week 2):** Spatiotemporal foundation with Mamba backbone
- **M3 (Week 3):** Occlusion reconstruction and probabilistic learning

All code is clean, modular, and well-documented for publication. Training configurations support full ablation studies required by reviewers.

**Next:** Run M3 training on benchmarks and compile results for the paper (M4, Week 4).
