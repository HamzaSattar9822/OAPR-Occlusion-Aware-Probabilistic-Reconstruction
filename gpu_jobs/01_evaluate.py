#!/usr/bin/env python3
"""01_evaluate.py — Full OAPR on COCO val; print/save AP/AP50/APH JSON.

Usage:
  python gpu_jobs/01_evaluate.py \\
    --checkpoint /path/to/best.pth \\
    --config configs/m3_oapr_complete.yaml \\
    --out /path/to/01_evaluate.json
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
from src.evaluation import print_metrics_table  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    device = get_device()
    print(f"[01_evaluate] device={device}")

    model = load_model(cfg, args.checkpoint, device)
    dataset, loader = build_val_loader(cfg)
    metrics = run_coco_eval(model, loader, dataset, cfg, device)

    print_metrics_table(metrics, cfg["dataset"]["name"].upper())
    payload = {
        "job": "01_evaluate",
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    write_json(args.out, payload)


if __name__ == "__main__":
    main()
