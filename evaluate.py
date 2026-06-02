# evaluate.py
"""
Milestone 1 — Evaluation Script
Runs a trained model on COCO val / CrowdPose test and reports AP metrics.
Also generates qualitative visualizations of predictions.

Usage:
    python evaluate.py --config configs/baseline_hrnet.yaml \
                       --checkpoint checkpoints/hrnet_baseline/best.pth

    # Evaluate on CrowdPose:
    python evaluate.py --config configs/baseline_hrnet.yaml \
                       --checkpoint checkpoints/hrnet_baseline/best.pth \
                       --override dataset.name=crowdpose

    # Save visualizations:
    python evaluate.py --config configs/baseline_hrnet.yaml \
                       --checkpoint checkpoints/hrnet_baseline/best.pth \
                       --visualize --vis_dir outputs/visualizations
"""

import os
import sys
import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import build_dataset
from src.models import build_model, JointsMSELoss, decode_heatmaps
from src.utils import setup_logger, load_checkpoint
from src.evaluation import print_metrics_table


# COCO skeleton connections for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                      # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12),                            # torso
    (11, 12),                                    # hips
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

CROWDPOSE_SKELETON = [
    (0, 1),                # shoulders
    (0, 2), (2, 4),        # left arm
    (1, 3), (3, 5),        # right arm
    (6, 7),                # hips
    (0, 6), (1, 7),        # torso
    (6, 8), (8, 10),       # left leg
    (7, 9), (9, 11),       # right leg
    (12, 13),              # head-neck
]

JOINT_COLORS = [
    (255, 0, 0),    (255, 85, 0),   (255, 170, 0),  (255, 255, 0),
    (170, 255, 0),  (85, 255, 0),   (0, 255, 0),    (0, 255, 85),
    (0, 255, 170),  (0, 255, 255),  (0, 170, 255),  (0, 85, 255),
    (0, 0, 255),    (85, 0, 255),   (170, 0, 255),  (255, 0, 255),
    (255, 0, 170),
]


def draw_pose(image, keypoints, skeleton, threshold=0.3):
    """
    Draw predicted skeleton on image.

    Args:
        image: HxWx3 uint8 BGR numpy array
        keypoints: (K, 3) array [x, y, conf]
        skeleton: list of (i, j) bone pairs
        threshold: minimum confidence to draw a joint

    Returns:
        annotated image
    """
    vis = image.copy()

    # Draw bones
    for i, (a, b) in enumerate(skeleton):
        if a >= len(keypoints) or b >= len(keypoints):
            continue
        if keypoints[a, 2] < threshold or keypoints[b, 2] < threshold:
            continue
        xa, ya = int(keypoints[a, 0]), int(keypoints[a, 1])
        xb, yb = int(keypoints[b, 0]), int(keypoints[b, 1])
        color = JOINT_COLORS[i % len(JOINT_COLORS)]
        cv2.line(vis, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)

    # Draw joints
    for k, kp in enumerate(keypoints):
        if kp[2] < threshold:
            continue
        x, y = int(kp[0]), int(kp[1])
        color = JOINT_COLORS[k % len(JOINT_COLORS)]
        cv2.circle(vis, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def evaluate(model, loader, criterion, device, cfg, logger, dataset,
             visualize=False, vis_dir=None, max_vis=50):
    """
    Full evaluation loop.

    Args:
        model: trained model
        loader: validation DataLoader
        criterion: loss function
        device: torch device
        cfg: config dict
        logger: logger
        dataset: dataset object (for evaluate())
        visualize: whether to save qualitative results
        vis_dir: output directory for visualizations
        max_vis: maximum number of images to visualize

    Returns:
        metrics dict
    """
    model.eval()

    image_size   = cfg['dataset']['image_size']
    heatmap_size = cfg['model']['heatmap_size']
    use_dark     = cfg['evaluation'].get('post_processing', 'dark') == 'dark'
    use_flip     = cfg['evaluation'].get('flip_test', True)
    num_kps      = cfg['model']['num_keypoints']
    dataset_name = cfg['dataset']['name'].lower()

    skeleton = COCO_SKELETON if dataset_name == 'coco' else CROWDPOSE_SKELETON

    all_preds  = []
    total_loss = 0.0
    total_n    = 0
    vis_count  = 0

    if visualize and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        logger.info(f"Saving visualizations to: {vis_dir}")

    start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images        = batch['image'].to(device, non_blocking=True)
            targets       = batch['target'].to(device, non_blocking=True)
            target_weight = batch['target_weight'].to(device, non_blocking=True)
            centers       = batch['center'].numpy()
            scales        = batch['scale'].numpy()
            image_ids     = batch['image_id']

            output = model(images)

            # TTA flip
            if use_flip:
                flip_pairs = cfg['dataset']['augmentation'].get('flip_pairs', [])
                flipped    = torch.flip(images, dims=[3])
                flip_out   = model(flipped)
                flip_out   = torch.flip(flip_out, dims=[3])
                for left, right in flip_pairs:
                    flip_out[:, left, :, :], flip_out[:, right, :, :] = \
                        flip_out[:, right, :, :].clone(), flip_out[:, left, :, :].clone()
                output = (output + flip_out) * 0.5

            loss = criterion(output, targets, target_weight)
            total_loss += loss.item() * images.size(0)
            total_n    += images.size(0)

            heatmaps_np = output.cpu().numpy()
            coords, maxvals = decode_heatmaps(
                heatmaps_np, centers, scales,
                heatmap_size, image_size,
                use_dark=use_dark
            )

            B = images.size(0)
            for b in range(B):
                kps = np.zeros((num_kps, 3), dtype=np.float32)
                kps[:, :2] = coords[b]
                kps[:, 2]  = maxvals[b, :, 0]

                all_preds.append({
                    'image_id': int(image_ids[b]),
                    'keypoints': kps,
                    'score':     float(maxvals[b].mean()),
                })

                # Visualization
                if visualize and vis_count < max_vis:
                    _save_visualization(
                        images[b].cpu(), kps, skeleton,
                        vis_dir, vis_count, int(image_ids[b])
                    )
                    vis_count += 1

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start
                logger.info(f"  [{batch_idx+1}/{len(loader)}]  "
                            f"elapsed: {elapsed:.1f}s")

    avg_loss = total_loss / max(total_n, 1)
    logger.info(f"Evaluation loss: {avg_loss:.4f}")
    logger.info("Computing AP metrics...")

    metrics = dataset.evaluate(all_preds)
    metrics['val_loss'] = avg_loss

    print_metrics_table(metrics, dataset_name.upper())
    return metrics


def _save_visualization(image_tensor, keypoints, skeleton, vis_dir, idx, image_id):
    """Denormalize image tensor and draw predicted skeleton, then save."""
    import torchvision.transforms.functional as TF

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = image_tensor * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    vis = draw_pose(img_bgr, keypoints, skeleton, threshold=0.3)

    save_path = os.path.join(vis_dir, f'pred_{idx:04d}_imgid{image_id}.jpg')
    cv2.imwrite(save_path, vis)


def load_config(path):
    with open(path, 'r') as f:
        docs = list(yaml.safe_load_all(f))
    return docs[0]


def apply_overrides(cfg, overrides):
    for override in overrides:
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        if value.lower() == 'true':   value = True
        elif value.lower() == 'false': value = False
        else:
            try:    value = int(value)
            except ValueError:
                try:    value = float(value)
                except ValueError: pass
        d[keys[-1]] = value
    return cfg


def main():
    parser = argparse.ArgumentParser(description="OAPR — Evaluate baseline model")
    parser.add_argument('--config',     default='configs/baseline_hrnet.yaml')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--override',   nargs='*', default=[])
    parser.add_argument('--visualize',  action='store_true',
                        help='Save qualitative pose predictions')
    parser.add_argument('--vis_dir',    default='outputs/visualizations')
    parser.add_argument('--max_vis',    type=int, default=100,
                        help='Max images to visualize')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    exp_name = cfg['experiment']['name'] + '_eval'
    log_dir  = cfg['experiment']['log_dir']
    logger   = setup_logger(exp_name, log_dir)

    # Device
    gpu_ids = cfg['hardware'].get('gpus', [0])
    device  = torch.device(f'cuda:{gpu_ids[0]}') \
              if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Evaluating on: {device}")

    # Dataset
    val_dataset = build_dataset(cfg, split='val')
    val_loader  = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training'].get('num_workers', 4),
        pin_memory=True,
    )

    # Model
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    if len(gpu_ids) > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=gpu_ids)

    criterion = JointsMSELoss(
        use_target_weight=cfg['training']['loss'].get('use_target_weight', True)
    ).to(device)

    # Evaluate
    metrics = evaluate(
        model, val_loader, criterion, device, cfg, logger, val_dataset,
        visualize=args.visualize,
        vis_dir=args.vis_dir,
        max_vis=args.max_vis,
    )

    logger.info("\nFinal Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
