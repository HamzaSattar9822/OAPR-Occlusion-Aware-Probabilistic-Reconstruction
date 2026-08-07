#!/usr/bin/env python3
"""06_train.py — Retrain the fixed OAPR image→keypoint pipeline; save checkpoint.

Uses crop-space joints (not heatmaps) for ProbabilisticPoseLoss.
Writes checkpoints under --out (e.g. Drive oapr_results/06_train).

Usage:
  python gpu_jobs/06_train.py \\
    --checkpoint /path/to/optional_resume.pth \\
    --config configs/m3_oapr_complete.yaml \\
    --out /content/drive/MyDrive/oapr_results/06_train
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    apply_overrides,
    get_device,
    load_config,
    run_coco_eval,
)
from src.data import build_dataset  # noqa: E402
from src.models import build_pose_model  # noqa: E402
from src.utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Resume path if file exists; otherwise start from pretrained HRNet init",
    )
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="Output directory for checkpoints + log JSON")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch):
    model.train()
    use_amp = cfg["training"].get("amp", True) and device.type == "cuda"
    total_loss = 0.0
    n = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        # Coordinate GT in crop space (NOT heatmaps)
        joints = batch["joints"].to(device, non_blocking=True)  # (B, K, 3)
        targets = joints[:, :, :2]
        weights = batch["target_weight"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            output = model(images)
            pred = torch.cat([output["keypoints"], output["confidence"]], dim=-1)
            loss, _ = model.compute_loss(
                pred,
                targets,
                weights,
                output["confidence"],
                output.get("occlusion_score"),
            )

        scaler.scale(loss).backward()
        clip = cfg["training"].get("clip_grad_norm", 0)
        if clip and clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs
        if (i + 1) % cfg["logging"].get("print_freq", 50) == 0:
            print(
                f"  epoch {epoch} [{i+1}/{len(loader)}] "
                f"loss={total_loss / max(n, 1):.4f}"
            )
    return {
        "loss": total_loss / max(n, 1),
        "seconds": time.time() - t0,
    }


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"[06_train] device={device}  out={args.out}")

    train_ds = build_dataset(cfg, split="train")
    val_ds = build_dataset(cfg, split="val")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 2),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )

    model = build_pose_model(cfg).to(device)
    start_epoch = 0
    best_ap = 0.0
    if args.checkpoint and os.path.isfile(args.checkpoint):
        start_epoch, best_ap, _ = load_checkpoint(
            args.checkpoint, model, device=device
        )
        print(f"[06_train] resumed epoch={start_epoch} best_ap={best_ap:.4f}")
    else:
        print(f"[06_train] no resume file at {args.checkpoint}; training from init")

    opt_cfg = cfg["training"]["optimizer"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
    )
    sched_cfg = cfg["training"]["scheduler"]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=sched_cfg["milestones"],
        gamma=sched_cfg.get("gamma", 0.1),
    )
    scaler = GradScaler(enabled=cfg["training"].get("amp", True) and device.type == "cuda")

    history = []
    total_epochs = cfg["training"]["epochs"]
    eval_interval = cfg["evaluation"].get("interval", 5)

    for epoch in range(start_epoch, total_epochs):
        print(f"\n[06_train] Epoch {epoch + 1}/{total_epochs}")
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg, epoch + 1
        )
        scheduler.step()

        metrics = {}
        is_best = False
        if ((epoch + 1) % eval_interval == 0) or (epoch + 1 == total_epochs):
            metrics = run_coco_eval(model, val_loader, val_ds, cfg, device)
            ap = float(metrics.get("AP", 0.0))
            if ap > best_ap:
                best_ap = ap
                is_best = True
            print(f"[06_train] val AP={ap:.4f}  best={best_ap:.4f}")

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "cfg": cfg,
        }
        save_checkpoint(
            state,
            is_best=is_best,
            output_dir=args.out,
            filename=f"checkpoint_epoch{epoch + 1:03d}.pth",
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_stats,
                "metrics": {k: float(v) for k, v in metrics.items()} if metrics else {},
                "best_ap": float(best_ap),
            }
        )

    summary = {
        "job": "06_train",
        "checkpoint_resume": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "out_dir": args.out,
        "best_ap": float(best_ap),
        "best_ckpt": os.path.join(args.out, "best.pth"),
        "history": history,
    }
    summary_path = os.path.join(args.out, "06_train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in summary if k != "history"}, indent=2))
    print(f"[06_train] wrote {summary_path}")


if __name__ == "__main__":
    main()
