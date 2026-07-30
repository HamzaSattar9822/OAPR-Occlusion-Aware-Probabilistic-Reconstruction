# src/utils/checkpoint.py
"""
Checkpoint save/load utilities.
Saves model, optimizer, scheduler state + config + metrics for full reproducibility.
"""

import os
import logging
import torch

logger = logging.getLogger(__name__)


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    """
    Save training checkpoint.

    Args:
        state: dict with keys: epoch, model_state_dict, optimizer_state_dict,
               scheduler_state_dict, metrics, cfg
        is_best: whether this is the best model so far (saves extra copy)
        output_dir: directory to save checkpoints
        filename: checkpoint filename
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")

    if is_best:
        best_path = os.path.join(output_dir, 'best.pth')
        torch.save(state, best_path)
        logger.info(f"New best model saved: {best_path}  "
                    f"(AP={state['metrics'].get('AP', 0):.4f})")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load checkpoint. Handles strict/non-strict loading gracefully.

    Args:
        path: path to .pth file
        model: model to load weights into
        optimizer: optional optimizer to restore state
        scheduler: optional scheduler to restore state
        device: torch device

    Returns:
        start_epoch: int
        best_ap: float
        metrics: dict of last recorded metrics
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Load model weights
    state_dict = ckpt['model_state_dict']
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.info("Model loaded (strict=True)")
    except RuntimeError as e:
        logger.warning(f"Strict load failed: {e}\n  Attempting non-strict load...")
        model.load_state_dict(state_dict, strict=False)
        logger.info("Model loaded (strict=False — some weights skipped)")

    start_epoch = ckpt.get('epoch', 0) + 1
    metrics     = ckpt.get('metrics', {})
    best_ap     = metrics.get('AP', 0.0)

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        logger.info("Optimizer state restored")

    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logger.info("Scheduler state restored")

    logger.info(f"Resuming from epoch {start_epoch}, best AP={best_ap:.4f}")
    return start_epoch, best_ap, metrics


def load_pretrained_backbone(model, pretrained_path, backbone_key='backbone'):
    """
    Load only backbone weights from a pretrained checkpoint.
    Useful when loading COCO-pretrained backbone for CrowdPose fine-tuning.

    Args:
        model: full model
        pretrained_path: path to pretrained .pth
        backbone_key: key prefix for backbone in state_dict
    """
    ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)

    backbone_state = {
        k[len(backbone_key)+1:]: v
        for k, v in state.items()
        if k.startswith(backbone_key + '.')
    }

    missing, unexpected = model.backbone.load_state_dict(
        backbone_state, strict=False
    )
    logger.info(f"Backbone loaded from {pretrained_path}")
    if missing:
        logger.info(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.info(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
