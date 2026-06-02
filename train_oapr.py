"""
train_oapr.py — Unified training script for Milestones 2 & 3

Trains the complete OAPR framework:
- Milestone 2: Spatiotemporal Mamba-Transformer backbone
- Milestone 3: Occlusion-aware reconstruction + robust probabilistic loss

Usage:
    python train_oapr.py --config configs/m2_mamba_temporal.yaml
    python train_oapr.py --config configs/m3_oapr_complete.yaml --resume checkpoints/best.pth
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import build_dataset
from src.models import build_oapr_framework
from src.utils import setup_logger, TBLogger, AverageMeter, save_checkpoint, load_checkpoint
from src.evaluation import accuracy_heatmap, print_metrics_table


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path):
    """Load YAML config (supports multi-document)."""
    with open(path, 'r') as f:
        docs = list(yaml.safe_load_all(f))
    return docs[0]


def apply_overrides(cfg, overrides):
    """Apply 'key.subkey=value' overrides."""
    for override in overrides:
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        d = cfg
        for k in keys[:-1]:
            d = d[k]
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


# ─── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, epoch, cfg, logger, tb_logger):
    """Train one epoch with robust probabilistic loss."""
    model.train()
    
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    losses = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    
    print_freq = cfg['logging']['print_freq']
    use_amp = cfg['training'].get('amp', True)
    
    end = time.time()
    
    for i, batch in enumerate(loader):
        data_time.update(time.time() - end)
        
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        target_weight = batch['target_weight'].to(device, non_blocking=True)
        
        # Forward pass with AMP
        with autocast(enabled=use_amp):
            output = model(images)
            
            # Use robust probabilistic loss
            loss, loss_dict = model.compute_loss(
                output['keypoints'],
                targets,
                target_weight,
                output['confidence'],
                output.get('occlusion_score')
            )
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        clip = cfg['training'].get('clip_grad_norm', 0)
        if clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        losses.update(loss.item(), images.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        global_step = epoch * len(loader) + i
        
        if i % print_freq == 0:
            eta_seconds = batch_time.avg * (len(loader) - i)
            eta = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            
            log_msg = (
                f"Epoch [{epoch}][{i}/{len(loader)}]  "
                f"ETA: {eta}  "
                f"Loss: {losses.avg:.4f}  "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Log loss breakdown
            if 'total_loss' in loss_dict:
                log_msg += f"  [coord: {loss_dict.get('coord_loss', 0):.4f}, "
                log_msg += f"unc: {loss_dict.get('uncertainty_loss', 0):.4f}]"
            
            logger.info(log_msg)
            
            tb_logger.add_scalars({
                'train/loss': losses.val,
                'train/loss_breakdown': loss_dict,
                'train/lr': optimizer.param_groups[0]['lr'],
            }, global_step)
    
    return losses.avg


def validate(model, loader, criterion, device, cfg, logger, dataset):
    """Validation loop with occlusion-aware metrics."""
    model.eval()
    
    losses = AverageMeter('val_loss')
    all_preds = []
    all_occlusions = []
    
    num_kps = cfg['model']['num_keypoints']
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            target_weight = batch['target_weight'].to(device, non_blocking=True)
            centers = batch['center'].numpy()
            scales = batch['scale'].numpy()
            image_ids = batch['image_id']
            
            # Forward
            output = model(images)
            
            keypoints = output['keypoints']
            confidence = output['confidence']
            occlusion_mask = output['occlusion_mask']
            
            # Loss
            loss, _ = model.compute_loss(
                torch.cat([keypoints, confidence], dim=-1),
                targets,
                target_weight,
                confidence,
                output.get('occlusion_score')
            )
            losses.update(loss.item(), images.size(0))
            
            # Collect predictions
            B = images.size(0)
            for b in range(B):
                kps = torch.zeros((num_kps, 3), dtype=torch.float32)
                kps[:, :2] = keypoints[b]
                kps[:, 2] = confidence[b].squeeze(-1)
                
                all_preds.append({
                    'image_id': int(image_ids[b]),
                    'keypoints': kps.cpu().numpy(),
                    'score': float(confidence[b].mean()),
                })
                
                all_occlusions.append({
                    'image_id': int(image_ids[b]),
                    'occlusion_mask': occlusion_mask[b].cpu().numpy(),
                })
    
    logger.info(f"Validation loss: {losses.avg:.4f}")
    
    # Compute AP
    ap_metrics = dataset.evaluate(all_preds)
    ap_metrics['val_loss'] = losses.avg
    
    # Log occlusion statistics (optional)
    occlusion_rate = np.concatenate([o['occlusion_mask'] for o in all_occlusions]).mean()
    ap_metrics['occlusion_rate'] = occlusion_rate
    
    print_metrics_table(ap_metrics, cfg['dataset']['name'].upper())
    return ap_metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OAPR Training — Milestones 2 & 3"
    )
    parser.add_argument('--config', default='configs/m3_oapr_complete.yaml',
                       help='Path to config')
    parser.add_argument('--resume', default=None, help='Checkpoint to resume from')
    parser.add_argument('--override', nargs='*', default=[], 
                       help='Config overrides')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
    
    exp_name = cfg['experiment']['name']
    output_dir = cfg['experiment']['output_dir']
    log_dir = cfg['experiment']['log_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Logger
    logger = setup_logger(exp_name, log_dir)
    tb_logger = TBLogger(log_dir, enabled=cfg['logging'].get('use_tensorboard', True))
    
    logger.info(f"{'='*70}")
    logger.info(f"OAPR Framework Training")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Config: {args.config}")
    logger.info(f"{'='*70}")
    
    # Reproducibility
    set_seed(cfg['experiment'].get('seed', 42))
    
    # Device
    gpu_ids = cfg['hardware'].get('gpus', [0])
    if torch.cuda.is_available() and gpu_ids:
        device = torch.device(f'cuda:{gpu_ids[0]}')
        logger.info(f"Using GPU: {gpu_ids}")
    else:
        device = torch.device('cpu')
        logger.warning("CUDA unavailable—training on CPU (slow!)")
    
    # Dataset
    logger.info("Building datasets...")
    train_dataset = build_dataset(cfg, split='train')
    val_dataset = build_dataset(cfg, split='val')
    
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
    
    logger.info(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")
    
    # Model
    logger.info("Building OAPR framework...")
    model = build_oapr_framework(cfg).to(device)
    
    # Multi-GPU
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logger.info(f"DataParallel on GPUs: {gpu_ids}")
    
    # Optimizer
    opt_cfg = cfg['training']['optimizer']
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt_cfg['lr'],
        weight_decay=opt_cfg.get('weight_decay', 1e-4),
    )
    
    # Scheduler
    sched_cfg = cfg['training']['scheduler']
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=sched_cfg['milestones'],
        gamma=sched_cfg.get('gamma', 0.1),
    )
    
    # AMP
    scaler = GradScaler(enabled=cfg['training'].get('amp', True))
    
    # Resume
    start_epoch = 0
    best_ap = 0.0
    
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        start_epoch, best_ap, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler, device=device
        )
    
    # Training loop
    total_epochs = cfg['training']['epochs']
    eval_interval = cfg['evaluation'].get('interval', 5)
    save_freq = cfg['logging'].get('save_checkpoint_freq', 5)
    
    logger.info(f"Starting training: epochs {start_epoch} → {total_epochs}")
    logger.info(f"{'='*70}\n")
    
    for epoch in range(start_epoch, total_epochs):
        logger.info(f"Epoch {epoch+1}/{total_epochs} | "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, None, optimizer, scaler,
            device, epoch, cfg, logger, tb_logger
        )
        
        tb_logger.add_scalars({
            'epoch/train_loss': train_loss,
            'epoch/lr': optimizer.param_groups[0]['lr'],
        }, epoch)
        
        scheduler.step()
        
        # Evaluate
        is_eval = ((epoch + 1) % eval_interval == 0) or (epoch + 1 == total_epochs)
        is_save = ((epoch + 1) % save_freq == 0)
        is_best = False
        ap_metrics = {}
        
        if is_eval:
            ap_metrics = validate(
                model, val_loader, None,
                device, cfg, logger, val_dataset
            )
            tb_logger.log_metrics(ap_metrics, epoch)
            
            current_ap = ap_metrics.get('AP', 0.0)
            if current_ap > best_ap:
                best_ap = current_ap
                is_best = True
                logger.info(f"★ New best AP: {best_ap:.4f}")
        
        # Save
        if is_save or is_best:
            state = {
                'epoch': epoch,
                'model_state_dict': (model.module.state_dict()
                                    if hasattr(model, 'module')
                                    else model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': ap_metrics,
                'cfg': cfg,
            }
            save_checkpoint(
                state, is_best, output_dir,
                filename=f'checkpoint_epoch{epoch+1:03d}.pth'
            )
        
        logger.info(f"")
    
    logger.info(f"{'='*70}")
    logger.info(f"Training complete. Best AP: {best_ap:.4f}")
    logger.info(f"Best model: {os.path.join(output_dir, 'best.pth')}")
    tb_logger.close()


if __name__ == '__main__':
    main()
