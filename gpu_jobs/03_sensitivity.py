#!/usr/bin/env python3
"""03_sensitivity.py — Sweep occlusion threshold τ and confidence blend β.

τ = model.occlusion_threshold (confidence gate)
β = model.confidence_beta (heatmap vs backbone confidence blend)

Prints / saves AP for each value. No fabricated numbers.

Usage:
  python gpu_jobs/03_sensitivity.py --checkpoint ... --config ... --out ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    apply_overrides,
    build_val_loader,
    get_device,
    load_config,
    load_model,
    run_coco_eval,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument(
        "--tau",
        nargs="*",
        type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="Occlusion threshold τ values",
    )
    p.add_argument(
        "--beta",
        nargs="*",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Confidence blend β values",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    device = get_device()
    dataset, loader = build_val_loader(cfg)
    model = load_model(cfg, args.checkpoint, device)

    base_tau = float(getattr(model, "occlusion_threshold", 0.5))
    base_beta = float(getattr(model, "confidence_beta", 0.5))

    rows = []

    print("\n[03_sensitivity] --- sweep τ (β fixed at config default) ---")
    for tau in args.tau:
        if hasattr(model, "set_ablation"):
            model.set_ablation(occlusion_threshold=tau, confidence_beta=base_beta)
        metrics = run_coco_eval(model, loader, dataset, cfg, device)
        row = {
            "param": "tau",
            "value": float(tau),
            "beta_fixed": base_beta,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        rows.append(row)
        print(f"  τ={tau:.2f}  AP={row['metrics'].get('AP', float('nan')):.4f}")

    print("\n[03_sensitivity] --- sweep β (τ fixed at config default) ---")
    for beta in args.beta:
        if hasattr(model, "set_ablation"):
            model.set_ablation(occlusion_threshold=base_tau, confidence_beta=beta)
        metrics = run_coco_eval(model, loader, dataset, cfg, device)
        row = {
            "param": "beta",
            "value": float(beta),
            "tau_fixed": base_tau,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        rows.append(row)
        print(f"  β={beta:.2f}  AP={row['metrics'].get('AP', float('nan')):.4f}")

    # Restore defaults
    if hasattr(model, "set_ablation"):
        model.set_ablation(occlusion_threshold=base_tau, confidence_beta=base_beta)

    payload = {
        "job": "03_sensitivity",
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "rows": rows,
    }
    write_json(args.out, payload)


if __name__ == "__main__":
    main()
