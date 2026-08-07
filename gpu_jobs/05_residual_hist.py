#!/usr/bin/env python3
"""05_residual_hist.py — Collect |pred−GT| residuals on val; save histogram PNG.

Uses crop-space joints from the dataloader vs model crop-space keypoints.
Does not invent residual values.

Usage:
  python gpu_jobs/05_residual_hist.py --checkpoint ... --config ... --out hist.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    apply_overrides,
    build_val_loader,
    get_device,
    load_config,
    load_model,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="PNG path (JSON sidecar written too)")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--max-batches", type=int, default=0, help="0 = full val")
    return p.parse_args()


@torch.no_grad()
def collect_residuals(model, loader, device, max_batches=0) -> np.ndarray:
    residuals = []
    model.eval()
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        joints = batch["joints"].numpy()  # (B, K, 3) crop space
        weights = batch["target_weight"].numpy().reshape(joints.shape[0], -1)  # (B, K)

        output = model(images)
        if not (isinstance(output, dict) and "keypoints" in output):
            raise RuntimeError("05_residual_hist expects OAPR dict output with keypoints")
        pred = output["keypoints"].detach().cpu().numpy()  # (B, K, 2)
        gt = joints[:, :, :2]
        err = np.linalg.norm(pred - gt, axis=-1)  # (B, K)
        visible = weights > 0
        residuals.append(err[visible])
    if not residuals:
        return np.array([], dtype=np.float64)
    return np.concatenate(residuals, axis=0)


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    device = get_device()
    model = load_model(cfg, args.checkpoint, device)
    _, loader = build_val_loader(cfg)

    residuals = collect_residuals(model, loader, device, args.max_batches)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    if residuals.size == 0:
        ax.text(0.5, 0.5, "No visible joints collected", ha="center", va="center")
        stats = {"n": 0}
    else:
        ax.hist(residuals, bins=50, color="#2c7fb8", edgecolor="white", alpha=0.9)
        ax.set_xlabel("L2 residual (crop pixels)")
        ax.set_ylabel("Count")
        ax.set_title("OAPR prediction residuals on COCO val (visible joints)")
        stats = {
            "n": int(residuals.size),
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "median": float(np.median(residuals)),
            "p90": float(np.percentile(residuals, 90)),
            "p99": float(np.percentile(residuals, 99)),
        }
        ax.axvline(stats["mean"], color="red", linestyle="--", label=f"mean={stats['mean']:.2f}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[05_residual_hist] wrote {args.out}")

    sidecar = os.path.splitext(args.out)[0] + "_stats.json"
    payload = {
        "job": "05_residual_hist",
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "stats": stats,
        "png": args.out,
    }
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"[05_residual_hist] wrote {sidecar}")


if __name__ == "__main__":
    main()
