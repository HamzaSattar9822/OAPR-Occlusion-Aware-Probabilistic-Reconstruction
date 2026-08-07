"""
Shared helpers for gpu_jobs/* (Colab / GPU entrypoints only).

No fabricated metrics — callers compute from real model + COCO val.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Repo root = parent of gpu_jobs/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_dataset  # noqa: E402
from src.data.transforms import get_affine_transform, affine_transform  # noqa: E402
from src.models import build_pose_model  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        docs = list(yaml.safe_load_all(f))
    return docs[0]


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    if not overrides:
        return cfg
    for override in overrides:
        key_path, value = override.split("=", 1)
        keys = key_path.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        d[keys[-1]] = value
    return cfg


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_val_loader(cfg: dict, batch_size: int | None = None) -> tuple:
    dataset = build_dataset(cfg, split="val")
    bs = batch_size or cfg["training"].get("batch_size", 16)
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def load_model(cfg: dict, checkpoint: str, device: torch.device):
    model = build_pose_model(cfg).to(device)
    load_checkpoint(checkpoint, model, device=device)
    model.eval()
    return model


def transform_preds_to_image(coords_crop, centers, scales, image_size):
    """Crop-space (B,K,2) → original image coordinates."""
    B, K, _ = coords_crop.shape
    out = np.zeros_like(coords_crop, dtype=np.float32)
    for b in range(B):
        trans = get_affine_transform(centers[b], scales[b], 0, image_size, inv=True)
        for j in range(K):
            out[b, j] = affine_transform(coords_crop[b, j], trans)
    return out


@torch.no_grad()
def run_coco_eval(model, loader, dataset, cfg, device) -> dict:
    """
    Full OAPR / HRNet forward on val → COCO AP metrics.
    Returns whatever dataset.evaluate() returns (no hardcoded numbers).
    """
    image_size = cfg["dataset"]["image_size"]
    num_kps = cfg["model"]["num_keypoints"]
    all_preds = []

    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        centers = batch["center"].numpy()
        scales = batch["scale"].numpy()
        image_ids = batch["image_id"]

        output = model(images)
        if isinstance(output, dict) and "keypoints" in output:
            coords_crop = output["keypoints"].detach().cpu().numpy()
            conf = output["confidence"].detach().cpu().numpy()
            if conf.ndim == 3:
                conf = conf[..., 0]
            coords = transform_preds_to_image(coords_crop, centers, scales, image_size)
            maxvals = conf
        else:
            from src.models.hrnet_baseline import decode_heatmaps

            heatmaps_np = output.detach().cpu().numpy()
            coords, maxvals = decode_heatmaps(
                heatmaps_np,
                centers,
                scales,
                cfg["model"]["heatmap_size"],
                image_size,
                use_dark=cfg["evaluation"].get("post_processing", "dark") == "dark",
            )
            maxvals = maxvals[:, :, 0]

        B = images.size(0)
        for b in range(B):
            kps = np.zeros((num_kps, 3), dtype=np.float32)
            kps[:, :2] = coords[b]
            kps[:, 2] = maxvals[b]
            all_preds.append(
                {
                    "image_id": int(image_ids[b]),
                    "keypoints": kps,
                    "score": float(maxvals[b].mean()),
                }
            )

    return dataset.evaluate(all_preds)


def write_json(path: str, payload: dict | list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"[gpu_jobs] wrote {path}")
