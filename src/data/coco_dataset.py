# src/data/coco_dataset.py
"""
COCO Keypoints dataset loader.
Handles person detection crops, keypoint loading, and the 70/30 split.
"""

import os
import copy
import logging
import json
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from .transforms import PoseTransform

logger = logging.getLogger(__name__)

# COCO 17 keypoints (0-indexed)
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

COCO_FLIP_PAIRS = [
    [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]
]


class COCODataset(Dataset):
    """
    COCO 2017 person keypoints dataset.

    Loads pre-detected person crops for top-down pose estimation.
    For training, uses the GT bounding boxes from annotations.
    For evaluation, COCO-eval API is used.

    Args:
        cfg: config dict
        split: 'train' or 'val'
    """

    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.is_train = (split == 'train')

        data_cfg = cfg['dataset']
        self.root = data_cfg['root']
        self.image_size = np.array(data_cfg['image_size'])      # [W, H]
        self.heatmap_size = np.array(cfg['model']['heatmap_size'])
        self.num_keypoints = cfg['model']['num_keypoints']
        self.sigma = cfg['model']['sigma']
        self.pixel_std = 200  # Standard scale normalization

        # Annotation file
        if self.is_train:
            ann_file = os.path.join(self.root, data_cfg['train_ann'])
        else:
            ann_file = os.path.join(self.root, data_cfg['val_ann'])

        logger.info(f"Loading COCO annotations from {ann_file}")
        self.coco = COCO(ann_file)
        self.image_set_index = self._load_image_set_index()
        self.db = self._get_db()

        # 70/30 split if requested and we're working with train annotations
        if self.is_train:
            split_ratio = data_cfg.get('train_split_ratio', 0.70)
            self.db = self._apply_split(self.db, split_ratio)

        # Drop annotations whose image files are missing (partial COCO downloads)
        before = len(self.db)
        self.db = [r for r in self.db if os.path.isfile(r['image_path'])]
        skipped = before - len(self.db)
        if skipped:
            logger.warning(
                f"Skipped {skipped} samples with missing image files "
                f"({len(self.db)} remaining)"
            )

        self.transform = PoseTransform(cfg, is_train=self.is_train)

        logger.info(f"COCODataset ({split}): {len(self.db)} samples loaded")

    def _load_image_set_index(self):
        """Return list of image IDs that contain at least one person."""
        cats = [cat['id'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        person_cat_id = self.coco.getCatIds(catNms=['person'])[0]
        image_ids = self.coco.getImgIds(catIds=[person_cat_id])
        return sorted(image_ids)

    def _get_db(self):
        """Build flat list of person instances with their keypoints and boxes."""
        gt_db = []

        for img_id in self.image_set_index:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            img_info = self.coco.loadImgs(img_id)[0]
            image_path = self._get_image_path(img_info['file_name'])

            valid_objs = []
            for ann in anns:
                # Skip if not person or no keypoints
                if ann['category_id'] != self.coco.getCatIds(catNms=['person'])[0]:
                    continue
                if 'keypoints' not in ann:
                    continue
                if max(ann['keypoints'][2::3]) == 0:
                    continue
                if ann.get('num_keypoints', 0) < 1:
                    continue

                # Bounding box
                x, y, w, h = ann['bbox']
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img_info['width'] - 1, x + max(0, w - 1))
                y2 = min(img_info['height'] - 1, y + max(0, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                center, scale = self._box_to_center_scale(
                    x1, y1, x2 - x1, y2 - y1
                )

                # Keypoints
                joints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
                joints_vis = np.zeros((self.num_keypoints, 3), dtype=np.float32)
                for kp_idx in range(self.num_keypoints):
                    kp = ann['keypoints'][kp_idx * 3:(kp_idx + 1) * 3]
                    joints[kp_idx, 0] = kp[0]
                    joints[kp_idx, 1] = kp[1]
                    vis = min(1, kp[2])  # 0=invisible, 1=labeled
                    joints_vis[kp_idx] = [vis, vis, 0]

                valid_objs.append({
                    'image_path': image_path,
                    'image_id': img_id,
                    'ann_id': ann['id'],
                    'center': center,
                    'scale': scale,
                    'joints': joints,
                    'joints_vis': joints_vis,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': ann.get('score', 1.0),
                })

            gt_db.extend(valid_objs)

        return gt_db

    def _apply_split(self, db, ratio):
        """Apply 70/30 train/test split deterministically."""
        random.seed(42)
        db_copy = db.copy()
        random.shuffle(db_copy)
        n = int(len(db_copy) * ratio)
        logger.info(f"Applying {ratio:.0%}/{1-ratio:.0%} split: "
                    f"{n} train / {len(db_copy)-n} held-out")
        return db_copy[:n]

    def _box_to_center_scale(self, x, y, w, h, aspect_ratio=None):
        """Convert bounding box to center + scale."""
        if aspect_ratio is None:
            aspect_ratio = self.image_size[0] / self.image_size[1]  # W/H

        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / self.pixel_std, h / self.pixel_std], dtype=np.float32)
        scale = scale * 1.25  # padding around box

        return center, scale

    def _get_image_path(self, file_name):
        """Resolve full image path."""
        if self.is_train:
            folder = 'train2017'
        else:
            folder = 'val2017'
        return os.path.join(self.root, 'images', folder, file_name)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_path = db_rec['image_path']
        image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {image_path} (rebuild dataset index)"
            )

        joints = db_rec['joints']
        joints_vis = db_rec['joints_vis']
        center = db_rec['center']
        scale = db_rec['scale']

        sample = self.transform(image, joints, joints_vis, center, scale)

        # Attach metadata needed for evaluation
        sample['image_id'] = db_rec['image_id']
        sample['ann_id'] = db_rec['ann_id']
        sample['bbox'] = torch.tensor(db_rec['bbox'], dtype=torch.float32)

        return sample

    def evaluate(self, preds, output_dir=None):
        """
        Evaluate predictions using COCO eval API.

        Args:
            preds: list of dicts with keys: image_id, keypoints, score, bbox
        Returns:
            dict with AP, AP50, AP75, APm, APl
        """
        from pycocotools.cocoeval import COCOeval
        import tempfile

        # Build results in COCO format
        results = []
        for p in preds:
            kps = p['keypoints'].flatten().tolist()
            results.append({
                'image_id': int(p['image_id']),
                'category_id': 1,
                'keypoints': kps,
                'score': float(p['score']),
            })

        if not results:
            logger.warning("No predictions to evaluate.")
            return {}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(results, f)
            tmp_path = f.name

        coco_dt = self.coco.loadRes(tmp_path)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        return {
            'AP':   stats[0],
            'AP50': stats[1],
            'APH':  stats[2],  # AP@OKS=0.75 (hard threshold)
            'AP75': stats[2],
            'APm':  stats[3],
            'APl':  stats[4],
            'AR':   stats[5],
        }
