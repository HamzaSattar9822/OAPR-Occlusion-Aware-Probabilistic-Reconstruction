# src/utils/logger.py
"""
Logging setup with TensorBoard support.
Logs to both console and file. All experiments tracked under logs/.
"""

import os
import logging
import time
from pathlib import Path


def setup_logger(name, log_dir, level=logging.INFO):
    """
    Set up a logger that writes to console + file.

    Args:
        name: logger name (usually the experiment name)
        log_dir: directory to store log file
        level: logging level

    Returns:
        logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger


class TBLogger:
    """
    Thin TensorBoard wrapper. Falls back gracefully if TensorBoard unavailable.
    Tracks: loss, AP metrics, learning rate.
    """

    def __init__(self, log_dir, enabled=True):
        self.enabled = enabled
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
                print(f"TensorBoard logging to: {log_dir}")
                print(f"  View with: tensorboard --logdir {log_dir}")
            except ImportError:
                print("[WARN] TensorBoard not available. Skipping TB logging.")
                self.enabled = False

    def add_scalar(self, tag, value, step):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def add_scalars(self, tag_dict, step):
        for tag, value in tag_dict.items():
            self.add_scalar(tag, value, step)

    def log_metrics(self, metrics_dict, step, prefix='val'):
        """Log a dict of metrics (AP, loss, etc.)."""
        for k, v in metrics_dict.items():
            self.add_scalar(f'{prefix}/{k}', v, step)

    def close(self):
        if self.writer:
            self.writer.close()


class AverageMeter:
    """Running average meter for loss tracking during training."""

    def __init__(self, name=''):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'
