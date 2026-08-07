#!/usr/bin/env python3
"""04_complexity.py — Parameter count, GFLOPs, peak GPU memory.

Usage:
  python gpu_jobs/04_complexity.py --checkpoint ... --config ... --out ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    apply_overrides,
    get_device,
    load_config,
    load_model,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "total_params_M": round(total / 1e6, 4),
    }


def estimate_gflops(model, device, image_size, batch_size=1):
    """Try thop; if unavailable, return null (never invent a number)."""
    W, H = int(image_size[0]), int(image_size[1])
    dummy = torch.randn(batch_size, 3, H, W, device=device)
    try:
        from thop import profile  # type: ignore

        model.eval()
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        # thop MACs ≈ FLOPs/2 for conv; report both honestly
        return {
            "macs": float(macs),
            "gmacs": float(macs) / 1e9,
            "gflops_approx_2x_macs": float(macs) * 2 / 1e9,
            "thop_params": float(params),
            "note": "GFLOPs≈2×MACs (thop convention); see gflops_approx_2x_macs",
        }
    except Exception as e:
        return {
            "macs": None,
            "gflops_approx_2x_macs": None,
            "error": f"thop unavailable or failed: {e}",
            "note": "Install thop in Colab for GFLOPs; params still reported.",
        }


@torch.no_grad()
def peak_gpu_memory_mb(model, device, image_size, batch_size=1) -> dict:
    if device.type != "cuda":
        return {"peak_allocated_mb": None, "note": "CUDA unavailable"}
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    W, H = int(image_size[0]), int(image_size[1])
    dummy = torch.randn(batch_size, 3, H, W, device=device)
    _ = model(dummy)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {
        "peak_allocated_mb": float(peak),
        "batch_size": batch_size,
        "input_hw": [H, W],
    }


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    device = get_device()
    model = load_model(cfg, args.checkpoint, device)
    image_size = cfg["dataset"]["image_size"]

    params = count_params(model)
    gflops = estimate_gflops(model, device, image_size, args.batch_size)
    mem = peak_gpu_memory_mb(model, device, image_size, args.batch_size)

    payload = {
        "job": "04_complexity",
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "params": params,
        "compute": gflops,
        "memory": mem,
    }
    write_json(args.out, payload)


if __name__ == "__main__":
    main()
