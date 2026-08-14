#!/usr/bin/env python3
"""02_ablation.py — Toggle OFF each paper component; one JSON row per config.

Ablations (each run is full-model minus ONE component):
  - full               : all components ON
  - no_gcn             : GCN skeleton-graph reconstruction OFF
  - no_mamba           : rebuild with use_mamba=False (joint attention fallback)
  - no_confidence_gate : occlusion confidence gate OFF (pass-through)

Usage:
  python gpu_jobs/02_ablation.py --checkpoint ... --config ... --out ...
"""

from __future__ import annotations

import argparse
import copy
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


ABLATIONS = (
    {"name": "full", "flags": {}},
    {"name": "no_gcn", "flags": {"use_gcn": False}},
    {"name": "no_mamba", "flags": {"use_mamba": False}, "rebuild": True},
    {"name": "no_confidence_gate", "flags": {"use_confidence_gate": False}},
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    device = get_device()
    dataset, loader = build_val_loader(cfg)

    rows = []
    for abl in ABLATIONS:
        print(f"\n[02_ablation] === {abl['name']} ===")
        cfg_run = copy.deepcopy(cfg)
        flags = abl["flags"]

        if abl.get("rebuild"):
            cfg_run.setdefault("model", {})["use_mamba"] = False
            model = load_model(cfg_run, args.checkpoint, device)
        else:
            model = load_model(cfg, args.checkpoint, device)
            if hasattr(model, "set_ablation"):
                model.set_ablation(**flags)

        metrics = run_coco_eval(model, loader, dataset, cfg_run, device)
        row = {
            "ablation": abl["name"],
            "flags": flags,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        rows.append(row)
        print(
            f"[02_ablation] {abl['name']}: "
            f"AP={row['metrics'].get('AP', float('nan')):.4f}"
        )

    payload = {
        "job": "02_ablation",
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "rows": rows,
    }
    write_json(args.out, payload)


if __name__ == "__main__":
    main()
