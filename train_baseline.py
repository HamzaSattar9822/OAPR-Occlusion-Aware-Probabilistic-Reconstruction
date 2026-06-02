# train_baseline.py
"""
Milestone 1 — HRNet Baseline Training Script
Trains HRNet-W32 on COCO or CrowdPose with pretrained ImageNet weights.

Usage:
    python train_baseline.py --config configs/baseline_hrnet.yaml
    python train_baseline.py --config configs/baseline_hrnet.yaml --resume checkpoints/hrnet_baseline/checkpoint.pth
    python train_baseline.py --config configs/baseline_hrnet.yaml --override dataset.name=crowdpose

All training details are logged to TensorBoard and a .log file.
"""

import os
import sys
import argparse
import random
import logging
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import build_dataset
from src.models import build_model, JointsMSELoss, decode_heatmaps
from src.utils import setup_logger, TBLogger, AverageMeter, save_checkpoint, load_checkpoint
from src.evaluation import accuracy_heatmap, print_metrics_table


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path, 'r') as f:
        # yaml.safe_load handles the '---' separator
        docs = list(yaml.safe_load_all(f))
    return docs[0]  # first document only


def apply_overrides(cfg, overrides):
    """Apply 'key.subkey=value' overrides to cfg dict."""
    for override in overrides:
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        # Try to cast to int/float/bool
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
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


# ─── Seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Training step ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, epoch, cfg, logger, tb_logger):
    model.train()

    batch_time = AverageMeter('batch_time')
    data_time  = AverageMeter('data_time')
    losses     = AverageMeter('loss')
    acc_meter  = AverageMeter('acc')

    print_freq = cfg['logging']['print_freq']
    use_amp    = cfg['training'].get('amp', True)

    end = time.time()

    for i, batch in enumerate(loader):
        data_time.update(time.time() - end)

        images        = batch['image'].to(device, non_blocking=True)
        targets       = batch['target'].to(device, non_blocking=True)
        target_weight = batch['target_weight'].to(device, non_blocking=True)

        # Forward
        with autocast(enabled=use_amp):
            output = model(images)
            loss   = criterion(output, targets, target_weight)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Gradient clipping
        clip = cfg['training'].get('clip_grad_norm', 0)
        if clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)

        scaler.step(optimizer)
        scaler.update()

        # Metrics
        losses.update(loss.item(), images.size(0))
        avg_acc, _, _, _ = accuracy_heatmap(
            output.detach().cpu().numpy(),
            targets.detach().cpu().numpy()
        )
        acc_meter.update(avg_acc, images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        global_step = epoch * len(loader) + i

        if i % print_freq == 0:
            eta_seconds = batch_time.avg * (len(loader) - i)
            eta = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            logger.info(
                f"Epoch [{epoch}][{i}/{len(loader)}]  "
                f"ETA: {eta}  "
                f"Loss: {losses.avg:.4f}  "
                f"Acc: {acc_meter.avg:.4f}  "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            tb_logger.add_scalars({
                'train/loss': losses.val,
                'train/acc':  acc_meter.val,
                'train/lr':   optimizer.param_groups[0]['lr'],
            }, global_step)

    return losses.avg, acc_meter.avg


# ─── Validation step ──────────────────────────────────────────────────────────

def validate(model, loader, criterion, device, cfg, logger, dataset):
    """
    Run validation and compute COCO/CrowdPose AP.
    Returns metrics dict.
    """
    model.eval()

    losses    = AverageMeter('val_loss')
    all_preds = []

    image_size   = cfg['dataset']['image_size']    # [W, H]
    heatmap_size = cfg['model']['heatmap_size']    # [W, H]
    use_dark     = cfg['evaluation'].get('post_processing', 'dark') == 'dark'
    use_flip     = cfg['evaluation'].get('flip_test', True)
    num_kps      = cfg['model']['num_keypoints']

    with torch.no_grad():
        for batch in loader:
            images        = batch['image'].to(device, non_blocking=True)
            targets       = batch['target'].to(device, non_blocking=True)
            target_weight = batch['target_weight'].to(device, non_blocking=True)
            centers       = batch['center'].numpy()
            scales        = batch['scale'].numpy()
            image_ids     = batch['image_id']

            output = model(images)

            # Test-time flip augmentation
            if use_flip:
                flip_pairs = cfg['dataset']['augmentation'].get('flip_pairs', [])
                flipped = torch.flip(images, dims=[3])
                flip_out = model(flipped)
                # Flip back
                flip_out = torch.flip(flip_out, dims=[3])
                # Swap paired keypoints
                for left, right in flip_pairs:
                    flip_out[:, left, :, :], flip_out[:, right, :, :] = \
                        flip_out[:, right, :, :].clone(), flip_out[:, left, :, :].clone()
                output = (output + flip_out) * 0.5

            loss = criterion(output, targets, target_weight)
            losses.update(loss.item(), images.size(0))

            # Decode heatmaps → image coords
            heatmaps_np = output.cpu().numpy()
            coords, maxvals = decode_heatmaps(
                heatmaps_np, centers, scales,
                heatmap_size, image_size,
                use_dark=use_dark
            )

            # Build predictions in COCO format
            B = images.size(0)
            for b in range(B):
                kps = np.zeros((num_kps, 3), dtype=np.float32)
                kps[:, :2] = coords[b]
                kps[:, 2]  = maxvals[b, :, 0]

                all_preds.append({
                    'image_id': int(image_ids[b]),
                    'keypoints': kps,
                    'score': float(maxvals[b].mean()),
                })

    logger.info(f"Validation loss: {losses.avg:.4f}  |  Running COCO eval...")

    # COCO / CrowdPose AP
    ap_metrics = dataset.evaluate(all_preds)
    ap_metrics['val_loss'] = losses.avg

    print_metrics_table(ap_metrics, cfg['dataset']['name'].upper())
    return ap_metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OAPR — Milestone 1 Baseline Training")
    parser.add_argument('--config',   default='configs/baseline_hrnet.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--resume',   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--override', nargs='*', default=[],
                        help='Override config values, e.g. dataset.name=crowdpose')
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    exp_name  = cfg['experiment']['name']
    output_dir = cfg['experiment']['output_dir']
    log_dir    = cfg['experiment']['log_dir']

    os.makedirs(output_dir, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────────────
    logger    = setup_logger(exp_name, log_dir)
    tb_logger = TBLogger(log_dir, enabled=cfg['logging'].get('use_tensorboard', True))
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Config: {cfg}")

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_seed(cfg['experiment'].get('seed', 42))

    # ── Device ────────────────────────────────────────────────────────────────
    gpu_ids = cfg['hardware'].get('gpus', [0])
    if torch.cuda.is_available() and gpu_ids:
        device = torch.device(f'cuda:{gpu_ids[0]}')
        logger.info(f"Using GPU: {gpu_ids}")
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available — training on CPU (slow!)")

    # ── Dataset & Loaders ─────────────────────────────────────────────────────
    logger.info("Building datasets...")
    train_dataset = build_dataset(cfg, split='train')
    val_dataset   = build_dataset(cfg, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training'].get('num_workers', 4),
        pin_memory=cfg['training'].get('pin_memory', True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training'].get('num_workers', 4),
        pin_memory=cfg['training'].get('pin_memory', True),
    )

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Building model...")
    model = build_model(cfg).to(device)

    # Multi-GPU (DataParallel for 1-2 GPUs; use DDP for larger setups)
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logger.info(f"DataParallel on GPUs: {gpu_ids}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = JointsMSELoss(
        use_target_weight=cfg['training']['loss'].get('use_target_weight', True)
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt_cfg = cfg['training']['optimizer']
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt_cfg['lr'],
        weight_decay=opt_cfg.get('weight_decay', 1e-4),
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    sched_cfg  = cfg['training']['scheduler']
    scheduler  = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=sched_cfg['milestones'],
        gamma=sched_cfg.get('gamma', 0.1),
    )

    # ── AMP Scaler ────────────────────────────────────────────────────────────
    scaler = GradScaler(enabled=cfg['training'].get('amp', True))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_ap     = 0.0

    if args.resume:
        start_epoch, best_ap, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler, device=device
        )

    # ── Training Loop ─────────────────────────────────────────────────────────
    total_epochs = cfg['training']['epochs']
    eval_interval = cfg['evaluation'].get('interval', 10)
    save_freq     = cfg['logging'].get('save_checkpoint_freq', 10)

    logger.info(f"Starting training: epochs {start_epoch} → {total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{total_epochs}  |  LR: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"{'='*60}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, cfg, logger, tb_logger
        )
        tb_logger.add_scalars({
            'epoch/train_loss': train_loss,
            'epoch/train_acc':  train_acc,
            'epoch/lr':         optimizer.param_groups[0]['lr'],
        }, epoch)

        scheduler.step()

        # Evaluate
        is_eval   = ((epoch + 1) % eval_interval == 0) or (epoch + 1 == total_epochs)
        is_save   = ((epoch + 1) % save_freq == 0)
        is_best   = False
        ap_metrics = {}

        if is_eval:
            ap_metrics = validate(
                model, val_loader, criterion,
                device, cfg, logger, val_dataset
            )
            tb_logger.log_metrics(ap_metrics, epoch)

            current_ap = ap_metrics.get('AP', 0.0)
            if current_ap > best_ap:
                best_ap = current_ap
                is_best = True
                logger.info(f"★ New best AP: {best_ap:.4f}")

        # Save checkpoint
        if is_save or is_best:
            state = {
                'epoch':                epoch,
                'model_state_dict':     (model.module.state_dict()
                                         if hasattr(model, 'module')
                                         else model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics':              ap_metrics,
                'cfg':                  cfg,
            }
            save_checkpoint(
                state, is_best, output_dir,
                filename=f'checkpoint_epoch{epoch+1:03d}.pth'
            )

    logger.info(f"\nTraining complete. Best AP: {best_ap:.4f}")
    logger.info(f"Best model: {os.path.join(output_dir, 'best.pth')}")
    tb_logger.close()


if __name__ == '__main__':
    main()
