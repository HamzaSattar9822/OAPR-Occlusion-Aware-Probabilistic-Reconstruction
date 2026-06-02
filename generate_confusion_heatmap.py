#!/usr/bin/env python3
"""
Run evaluation on COCO val and save confusion matrix + heatmap figures.

Outputs are written per batch size, e.g.:
    outputs/confusion_heatmaps/batch_16/

Usage:
    python generate_confusion_heatmap.py --batch-size 16
    python generate_confusion_heatmap.py --batch-sizes 4,8,16,32
    python generate_confusion_heatmap.py --checkpoint checkpoints/hrnet_baseline/best.pth
"""

import json
import os
import sys
import argparse

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch.utils.data import DataLoader, Subset
from src.data import build_dataset
from src.models import build_model, decode_heatmaps, JointsMSELoss
from src.evaluation.metrics import accuracy_heatmap, _get_max_preds
from src.utils import load_checkpoint

COCO_KEYPOINT_NAMES = [
    'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
    'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
    'L_wrist', 'R_wrist', 'L_hip', 'R_hip',
    'L_knee', 'R_knee', 'L_ankle', 'R_ankle',
]

OUTCOME_LABELS = ['Correct', 'Incorrect', 'Missed', 'False positive']


def load_config(path):
    with open(path, 'r') as f:
        docs = list(yaml.safe_load_all(f))
    return docs[0]


def classify_keypoint(dist_norm, gt_vis, pred_conf, dist_thresh=0.5, conf_thresh=0.1):
    """Classify one keypoint into an outcome index."""
    if gt_vis > 0:
        if dist_norm < dist_thresh:
            return 0  # Correct
        if pred_conf > conf_thresh:
            return 1  # Incorrect
        return 2  # Missed
    if pred_conf > conf_thresh:
        return 3  # False positive
    return -1  # ignore


def build_joint_swap_matrix(gt_hm, pred_hm, gt_vis_mask):
    """
    17x17: for visible GT joint i, which pred channel j is closest (argmin distance).
    Counts joint-index confusion (left/right swaps, etc.).
    """
    K = gt_hm.shape[1]
    gt_coords, _ = _get_max_preds(gt_hm)
    pred_coords, _ = _get_max_preds(pred_hm)
    H, W = gt_hm.shape[2], gt_hm.shape[3]
    norm = np.array([W, H], dtype=np.float32)

    confusion = np.zeros((K, K), dtype=np.int64)
    for i in range(K):
        if not gt_vis_mask[i]:
            continue
        dists = np.linalg.norm(
            (pred_coords[0] / norm) - (gt_coords[0, i:i+1] / norm), axis=1
        )
        j = int(np.argmin(dists))
        confusion[i, j] += 1
    return confusion


def plot_outcome_matrix(counts, out_dir):
    """17 x 4 keypoint detection outcome heatmap."""
    mat = counts.astype(float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat_pct = mat / row_sums * 100

    fig, ax = plt.subplots(figsize=(8, 10))
    if HAS_SEABORN:
        sns.heatmap(
            mat_pct, annot=True, fmt='.1f', cmap='YlGnBu',
            xticklabels=OUTCOME_LABELS, yticklabels=COCO_KEYPOINT_NAMES,
            ax=ax, cbar_kws={'label': '% of GT visible joints'},
        )
    else:
        im = ax.imshow(mat_pct, aspect='auto', cmap='YlGnBu')
        ax.set_xticks(range(len(OUTCOME_LABELS)))
        ax.set_xticklabels(OUTCOME_LABELS, rotation=30, ha='right')
        ax.set_yticks(range(len(COCO_KEYPOINT_NAMES)))
        ax.set_yticklabels(COCO_KEYPOINT_NAMES)
        plt.colorbar(im, ax=ax, label='% of GT visible joints')
    ax.set_title('Keypoint detection confusion (outcomes per joint)')
    ax.set_xlabel('Predicted outcome')
    ax.set_ylabel('Ground-truth joint')
    fig.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrix_keypoint_outcomes.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_swap_matrix(swap_counts, out_dir):
    """17x17 joint-index confusion (nearest pred channel per GT joint)."""
    mat = swap_counts.astype(float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat_pct = mat / row_sums * 100

    fig, ax = plt.subplots(figsize=(11, 9))
    if HAS_SEABORN:
        sns.heatmap(
            mat_pct, annot=False, cmap='Blues',
            xticklabels=COCO_KEYPOINT_NAMES, yticklabels=COCO_KEYPOINT_NAMES,
            ax=ax, cbar_kws={'label': '%'},
        )
    else:
        im = ax.imshow(mat_pct, cmap='Blues')
        ax.set_xticks(range(17))
        ax.set_xticklabels(COCO_KEYPOINT_NAMES, rotation=45, ha='right')
        ax.set_yticks(range(17))
        ax.set_yticklabels(COCO_KEYPOINT_NAMES)
        plt.colorbar(im, ax=ax)
    ax.set_title('Joint-index confusion (GT joint → nearest pred channel)')
    ax.set_xlabel('Predicted joint index')
    ax.set_ylabel('GT joint index')
    fig.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrix_joint_swap.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_accuracy_heatmap(per_joint_acc, out_dir):
    """Per-joint PCK-style accuracy as a heatmap strip."""
    acc = np.array(per_joint_acc, dtype=np.float32)
    acc = np.where(acc < 0, np.nan, acc)

    fig, ax = plt.subplots(figsize=(12, 2.5))
    data = acc.reshape(1, -1)
    if HAS_SEABORN:
        sns.heatmap(
            data, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
            xticklabels=COCO_KEYPOINT_NAMES, yticklabels=['PCK@0.5'],
            ax=ax, cbar_kws={'label': 'accuracy'},
        )
    else:
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(17))
        ax.set_xticklabels(COCO_KEYPOINT_NAMES, rotation=45, ha='right')
        plt.colorbar(im, ax=ax)
    ax.set_title('Per-keypoint heatmap accuracy (heatmap space, threshold=0.5)')
    fig.tight_layout()
    path = os.path.join(out_dir, 'heatmap_per_joint_accuracy.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sample_heatmaps(pred_hm, gt_hm, out_dir, tag='sample'):
    """Save mean predicted vs GT heatmap for one batch item."""
    pred_mean = pred_hm.mean(axis=0)
    gt_mean = gt_hm.mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gt_mean, cmap='hot')
    axes[0].set_title('GT heatmaps (mean)')
    axes[0].axis('off')
    axes[1].imshow(pred_mean, cmap='hot')
    axes[1].set_title('Predicted heatmaps (mean)')
    axes[1].axis('off')
    diff = np.abs(pred_mean - gt_mean)
    im = axes[2].imshow(diff, cmap='magma')
    axes[2].set_title('|Pred − GT|')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle(f'Model heatmap comparison ({tag})')
    fig.tight_layout()
    path = os.path.join(out_dir, f'heatmap_pred_vs_gt_{tag}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def output_dir_for_batch(batch_size, base_dir='outputs/confusion_heatmaps'):
    """Separate folder per batch size."""
    out_dir = os.path.join(base_dir, f'batch_{batch_size}')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


@torch.no_grad()
def run_eval(cfg, checkpoint=None, max_batches=30, batch_size=8, num_workers=0,
             out_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if out_dir is None:
        out_dir = output_dir_for_batch(batch_size)
    else:
        os.makedirs(out_dir, exist_ok=True)

    val_dataset = build_dataset(cfg, split='val')
    n = min(len(val_dataset), max_batches * batch_size)
    val_dataset = Subset(val_dataset, list(range(n)))

    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    model = build_model(cfg).to(device)
    model.eval()
    if checkpoint and os.path.isfile(checkpoint):
        load_checkpoint(checkpoint, model, device=device)
        print(f'Loaded checkpoint: {checkpoint}')
    else:
        print('No checkpoint — using ImageNet-pretrained HRNet weights.')

    criterion = JointsMSELoss(
        use_target_weight=cfg['training']['loss'].get('use_target_weight', True)
    )

    outcome_counts = np.zeros((17, 4), dtype=np.int64)
    swap_counts = np.zeros((17, 17), dtype=np.int64)
    acc_accum = np.zeros(17, dtype=np.float64)
    acc_weight = np.zeros(17, dtype=np.float64)
    total_loss = 0.0
    total_n = 0

    heatmap_size = cfg['model']['heatmap_size']
    image_size = cfg['dataset']['image_size']

    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        target_weight = batch['target_weight'].to(device)

        output = model(images)
        loss = criterion(output, targets, target_weight)
        total_loss += loss.item() * images.size(0)
        total_n += images.size(0)

        out_np = output.cpu().numpy()
        tgt_np = targets.cpu().numpy()
        tw_np = target_weight.cpu().numpy()

        avg_acc, per_acc, cnt, _ = accuracy_heatmap(out_np, tgt_np, threshold=0.5)
        for k in range(17):
            if per_acc[k] >= 0:
                acc_accum[k] += per_acc[k] * max(tw_np[:, k].sum(), 1)
                acc_weight[k] += max(tw_np[:, k].sum(), 1)

        pred_hm, pred_conf = _get_max_preds(out_np)
        gt_hm, _ = _get_max_preds(tgt_np)
        H, W = tgt_np.shape[2], tgt_np.shape[3]
        norm = np.array([W, H], dtype=np.float32)

        B = out_np.shape[0]
        for b in range(B):
            for k in range(17):
                gt_vis = float(tw_np[b, k, 0] > 0)
                dist = np.linalg.norm(pred_hm[b, k] / norm - gt_hm[b, k] / norm)
                conf = float(pred_conf[b, k, 0])
                idx = classify_keypoint(dist, gt_vis, conf)
                if idx >= 0:
                    outcome_counts[k, idx] += 1

            gt_vis_mask = (tw_np[b, :, 0] > 0)
            swap_counts += build_joint_swap_matrix(
                tgt_np[b:b+1], out_np[b:b+1], gt_vis_mask
            )

        if batch_idx == 0:
            plot_sample_heatmaps(out_np[0], tgt_np[0], out_dir, tag='batch0')

        if batch_idx + 1 >= max_batches:
            break

    per_joint_acc = np.where(
        acc_weight > 0, acc_accum / acc_weight, -1
    ).tolist()

    paths = [
        plot_outcome_matrix(outcome_counts, out_dir),
        plot_swap_matrix(swap_counts, out_dir),
        plot_accuracy_heatmap(per_joint_acc, out_dir),
    ]

    summary = {
        'batch_size': batch_size,
        'max_batches': max_batches,
        'batches_run': min(batch_idx + 1, max_batches),
        'samples': total_n,
        'avg_loss': total_loss / max(total_n, 1),
        'mean_pck': float(np.nanmean([a for a in per_joint_acc if a >= 0])),
        'per_joint_accuracy': per_joint_acc,
        'output_dir': out_dir,
        'figures': paths,
    }

    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({k: v for k, v in summary.items() if k != 'figures'}, f, indent=2)
    summary['summary_json'] = summary_path
    return summary


def parse_batch_sizes(args):
    """Resolve one or many batch sizes from CLI args."""
    if args.batch_sizes:
        sizes = [int(x.strip()) for x in args.batch_sizes.split(',') if x.strip()]
    else:
        sizes = [args.batch_size]
    return sorted(set(sizes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baseline_hrnet.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--max-batches', type=int, default=15,
                        help='Number of batches to evaluate per batch-size run')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Single batch size (used if --batch-sizes not set)')
    parser.add_argument('--batch-sizes', default=None,
                        help='Comma-separated batch sizes, e.g. 4,8,16,32')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output-base', default='outputs/confusion_heatmaps',
                        help='Base directory; each batch size gets batch_<N>/ subfolder')
    args = parser.parse_args()

    batch_sizes = parse_batch_sizes(args)
    cfg = load_config(args.config)
    cfg['training']['num_workers'] = args.num_workers

    print('=' * 60)
    print('OAPR — Confusion matrix & heatmap generation')
    print('=' * 60)
    print(f'Batch sizes: {batch_sizes}')
    print(f'Max batches per run: {args.max_batches}')
    print(f'Output base: {args.output_base}')
    print('=' * 60)

    all_summaries = []
    for bs in batch_sizes:
        cfg['training']['batch_size'] = bs
        out_dir = output_dir_for_batch(bs, args.output_base)

        print(f'\n>>> Batch size {bs} → {out_dir}')
        summary = run_eval(
            cfg,
            checkpoint=args.checkpoint,
            max_batches=args.max_batches,
            batch_size=bs,
            num_workers=args.num_workers,
            out_dir=out_dir,
        )
        all_summaries.append(summary)

        print(f"  Samples evaluated: {summary['samples']}")
        print(f"  Mean PCK (heatmap):  {summary['mean_pck']:.4f}")
        print(f"  Val loss:            {summary['avg_loss']:.4f}")
        print('  Saved figures:')
        for p in summary['figures']:
            print(f'    {p}')
        print(f"  Summary JSON:        {summary['summary_json']}")

    print('\n' + '=' * 60)
    print('All runs complete:')
    for s in all_summaries:
        print(f"  batch_{s['batch_size']}: {s['samples']} samples → {s['output_dir']}")
    print('=' * 60)


if __name__ == '__main__':
    main()
