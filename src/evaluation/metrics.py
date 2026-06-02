# src/evaluation/metrics.py
"""
Evaluation utilities.
- COCO AP computation
- Per-joint accuracy (PCKh)
- Temporal stability metric (for future milestones)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_pckh(preds, gt_joints, gt_joints_vis, threshold=0.5):
    """
    PCKh metric: fraction of keypoints within threshold * head_size of GT.
    Standard metric for MPII-style evaluation.

    Args:
        preds: (B, K, 2) predicted joint coords in image space
        gt_joints: (B, K, 2) ground truth joints
        gt_joints_vis: (B, K) visibility flags (0/1)
        threshold: fraction of head size

    Returns:
        acc: per-joint accuracy array (K,)
        avg_acc: mean accuracy over valid joints
        cnt: number of evaluated joints
    """
    B, K = preds.shape[:2]

    # Head size: distance between head top and neck (or bounding box diagonal)
    # For COCO, approximate as max joint range / 10
    head_size = np.linalg.norm(
        gt_joints[:, 0, :] - gt_joints[:, 1, :], axis=1
    ).mean()

    dist_threshold = threshold * head_size

    acc = np.zeros(K, dtype=np.float32)
    cnt = np.zeros(K, dtype=np.int32)

    for k in range(K):
        vis = gt_joints_vis[:, k] > 0  # valid joints for this keypoint
        if vis.sum() == 0:
            continue
        dist = np.linalg.norm(
            preds[vis, k, :] - gt_joints[vis, k, :], axis=1
        )
        acc[k] = (dist < dist_threshold).mean()
        cnt[k] = vis.sum()

    valid = cnt > 0
    avg_acc = acc[valid].mean() if valid.any() else 0.0
    return acc, avg_acc, cnt.sum()


def accuracy_heatmap(output, target, hm_type='gaussian', threshold=0.5):
    """
    Compute per-keypoint accuracy on heatmaps using distance threshold.
    Used for quick training monitoring (not the final AP metric).

    Args:
        output: (B, K, H, W) predicted heatmaps
        target: (B, K, H, W) GT heatmaps
        threshold: normalized distance threshold (fraction of image diagonal)

    Returns:
        avg_acc: float
        acc: (K,) per-joint accuracy
        cnt: int total joints evaluated
        pred: predicted coordinates
    """
    B, K, H, W = output.shape

    pred, _ = _get_max_preds(output)  # (B, K, 2) in heatmap space
    gt,   _ = _get_max_preds(target)

    # Normalize to [0,1] based on heatmap size
    norm = np.array([H, W], dtype=np.float32)
    pred_norm = pred / norm
    gt_norm   = gt   / norm

    dist = np.linalg.norm(pred_norm - gt_norm, axis=2)  # (B, K)

    # Mask invisible joints (GT heatmap all-zero)
    target_vis = (target.max(axis=(2, 3)) > 0.01).astype(np.float32)  # (B, K)

    acc = np.zeros(K, dtype=np.float32)
    cnt = 0
    for k in range(K):
        vis = target_vis[:, k] > 0
        if vis.sum() == 0:
            acc[k] = -1
            continue
        acc[k] = (dist[vis, k] < threshold).mean()
        cnt += vis.sum()

    valid = acc >= 0
    avg_acc = acc[valid].mean() if valid.any() else 0.0
    return avg_acc, acc, cnt, pred


def _get_max_preds(batch_heatmaps):
    """Get argmax coords from heatmaps (numpy)."""
    B, K, H, W = batch_heatmaps.shape
    flat = batch_heatmaps.reshape(B, K, -1)
    idx = np.argmax(flat, axis=2)
    maxvals = np.amax(flat, axis=2)

    preds = np.zeros((B, K, 2), dtype=np.float32)
    preds[:, :, 0] = idx % W   # x
    preds[:, :, 1] = idx // W  # y

    preds = np.where(
        np.tile(maxvals[:, :, np.newaxis], (1, 1, 2)) > 0,
        preds, 0
    )
    return preds, maxvals[:, :, np.newaxis]


def print_metrics_table(metrics_dict, dataset_name='COCO'):
    """Pretty-print evaluation results in a table format."""
    border = "=" * 50
    print(border)
    print(f"  Evaluation Results — {dataset_name}")
    print(border)
    for k, v in metrics_dict.items():
        print(f"  {k:<10}: {v:.4f}")
    print(border)


def compute_temporal_stability(preds_seq):
    """
    Placeholder for temporal stability metric (Milestone 2+).
    Measures frame-to-frame jitter in joint predictions.

    Args:
        preds_seq: list of (K, 2) arrays, one per frame

    Returns:
        jitter: mean per-joint velocity (lower = more stable)
    """
    if len(preds_seq) < 2:
        return 0.0
    diffs = [
        np.linalg.norm(preds_seq[i+1] - preds_seq[i], axis=1)
        for i in range(len(preds_seq) - 1)
    ]
    return float(np.mean(diffs))
