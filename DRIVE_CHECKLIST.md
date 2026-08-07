# Google Drive checklist for OAPR GPU jobs (COLAB_RUNNER.ipynb)
#
# Mount point in Colab: /content/drive/MyDrive/
# Repo scripts expect these exact paths unless you edit the notebook cells.

## 1) COCO val (required for 01–05, and for val during 06)

```
/content/drive/MyDrive/coco/
├── images/
│   └── val2017/                          # all COCO val2017 JPGs
│       ├── 000000000139.jpg
│       └── ...
└── annotations/
    └── person_keypoints_val2017.json     # official COCO keypoints val annotations
```

Optional (only if you run training job 06 with the full train split):

```
/content/drive/MyDrive/coco/
├── images/
│   └── train2017/                        # COCO train2017 JPGs
└── annotations/
    └── person_keypoints_train2017.json
```

The notebook creates:

```
/content/oapr_pose/data/coco  →  symlink to /content/drive/MyDrive/coco
```

Config `configs/m3_oapr_complete.yaml` uses `dataset.root: ./data/coco`.

## 2) Checkpoint (required for inference jobs 01–05)

```
/content/drive/MyDrive/oapr_checkpoints/best.pth
```

- This is the weights file passed as `--checkpoint`.
- For job `06_train.py`, the same path is used as an optional **resume** file:
  - if the file exists → resume
  - if missing → train from HRNet ImageNet init (still valid)

After training, best weights are also written under results (see below).

## 3) Results directory (created automatically)

```
/content/drive/MyDrive/oapr_results/
├── 01_evaluate.json
├── 02_ablation.json
├── 03_sensitivity.json
├── 04_complexity.json
├── 05_residual_hist.png
├── 05_residual_hist_stats.json
└── 06_train/
    ├── best.pth
    ├── checkpoint_epochXXX.pth
    └── 06_train_summary.json
```

## Quick verify in Colab

```python
from pathlib import Path
assert Path('/content/drive/MyDrive/coco/images/val2017').is_dir()
assert Path('/content/drive/MyDrive/coco/annotations/person_keypoints_val2017.json').is_file()
assert Path('/content/drive/MyDrive/oapr_checkpoints/best.pth').is_file()
print('Drive layout OK')
```
