# OAPR: Occlusion-Aware Probabilistic Pose Reconstruction

Single-image pose estimation with occlusion-aware GCN reconstruction.

**Architecture (article):** HRNet-W32 → Hybrid Mamba–Transformer joint encoder → confidence head → occlusion gate (τ) → GCN skeleton-graph reconstruction → β-blend fusion → Cauchy-mixture loss.

## Quick inference demo

```bash
python test.py --image path/to/person.jpg --weights path/to/best.pth
```

Optional: `--out outputs/demo_pose.png` and `--device cuda`.

`model.py` exposes `build_oapr` / `load_weights` / `describe_components`. Checkpoint mismatches load matching tensors and warn on the rest.

## Project Structure

```
oapr_pose/
├── model.py                          # Plug-and-play OAPR (importable)
├── test.py                           # Single-image pose overlay demo
├── configs/
│   ├── baseline_hrnet.yaml
│   └── m3_oapr_complete.yaml
├── gpu_jobs/                         # Colab GPU scripts (eval / ablation / train)
├── src/
│   ├── data/
│   ├── models/
│   │   ├── hrnet_baseline.py
│   │   ├── mamba_backbone.py         # Joint-sequence Mamba–Transformer
│   │   ├── occlusion_module.py       # Gate (τ) + GCN + β-blend
│   │   ├── robust_loss.py            # Cauchy / Laplace / mixture
│   │   └── oapr_framework.py
│   ├── utils/
│   └── evaluation/
├── train_baseline.py
├── train_oapr.py
├── evaluate.py
└── requirements.txt
```

## Forward pass

```
Person crop (B, 3, 256, 192)
    → HRNet-W32 heatmaps + features
    → DARK decode + sample per-joint features
    → Mamba over K=17 joints  (or attention fallback)
    → Transformer over joints
    → confidence head
    → occlusion gate (τ)
    → GCN reconstruction over skeleton graph
    → p* = (1−m)·p + m·(β·p_recon + (1−β)·p)
    → Cauchy-mixture loss (train)
```

```python
from model import build_oapr, load_weights, describe_components

model = build_oapr(pretrained=False)
load_weights(model, "checkpoints/best.pth")
print(describe_components(model))
out = model(images)  # images: (B, 3, 256, 192)
# out['keypoints'] (B, K, 2), out['confidence'] (B, K, 1)
```

## Training / evaluation

```bash
python train_baseline.py --config configs/baseline_hrnet.yaml
python train_oapr.py --config configs/m3_oapr_complete.yaml
python evaluate.py --config configs/m3_oapr_complete.yaml \
    --checkpoint checkpoints/best.pth
```

GPU jobs for Colab live under `gpu_jobs/` (see `COLAB_RUNNER.ipynb` and `DRIVE_CHECKLIST.md`).

## Ablations

```bash
# Without Mamba (joint attention fallback)
python train_oapr.py --config configs/m3_oapr_complete.yaml \
    --override model.use_mamba=false

# Without GCN reconstruction
python train_oapr.py --config configs/m3_oapr_complete.yaml \
    --override model.use_gcn=false
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash scripts/download_coco.sh
```

`mamba-ssm` is optional; without it the joint encoder uses attention fallback.

## References

1. Gu et al., Mamba (ICLR 2024)
2. Barron et al., General / adaptive robust loss (CVPR 2019)
3. Wang et al., HRNet (TPAMI 2020)
