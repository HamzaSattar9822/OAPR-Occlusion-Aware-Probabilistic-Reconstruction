# src/data/crowdpose_dataset.py
"""
CrowdPose dataset loader.
CrowdPose has 14 keypoints and includes a crowd_index per image
(useful for evaluating performance on easy/medium/hard crowd levels).
"""

import os
import copy
import logging
import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import PoseTransform

logger = logging.getLogger(__name__)

# CrowdPose 14 keypoints
CROWDPOSE_KEYPOINT_NAMES = [
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'head', 'neck'
]

CROWDPOSE_FLIP_PAIRS = [
    [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]
]


class CrowdPoseDataset(Dataset):
    """
    CrowdPose dataset for multi-person pose estimation.

    Inherits the same interface as COCODataset.
    CrowdPose JSON format follows a COCO-like structure.

    Args:
        cfg: config dict
        split: 'train', 'val', or 'test'
    """

    NUM_KEYPOINTS = 14
    PIXEL_STD = 200

    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.is_train = (split == 'train')

        data_cfg = cfg['dataset']
        self.root = data_cfg['root']
        self.image_size = np.array(data_cfg['image_size'])
        self.heatmap_size = np.array(cfg['model']['heatmap_size'])
        self.num_keypoints = self.NUM_KEYPOINTS
        self.sigma = cfg['model']['sigma']

        # Annotation file
        ann_map = {
            'train': 'annotations/crowdpose_train.json',
            'val':   'annotations/crowdpose_val.json',
            'test':  'annotations/crowdpose_test.json',
        }
        ann_file = os.path.join(self.root, ann_map[split])
        logger.info(f"Loading CrowdPose annotations from {ann_file}")

        with open(ann_file, 'r') as f:
            anno = json.load(f)

        self.images = {img['id']: img for img in anno['images']}
        self.db = self._build_db(anno['annotations'])

        if self.is_train:
            split_ratio = data_cfg.get('train_split_ratio', 0.70)
            self.db = self._apply_split(self.db, split_ratio)

        self.transform = PoseTransform(cfg, is_train=self.is_train)
        logger.info(f"CrowdPoseDataset ({split}): {len(self.db)} samples loaded")

    def _build_db(self, annotations):
        db = []
        for ann in annotations:
            if 'keypoints' not in ann:
                continue
            if max(ann['keypoints'][2::3]) == 0:
                continue

            img_info = self.images[ann['image_id']]
            image_path = os.path.join(self.root, 'images', img_info['file_name'])

            x, y, w, h = ann['bbox']
            center, scale = self._box_to_center_scale(x, y, w, h)

            joints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
            joints_vis = np.zeros((self.num_keypoints, 3), dtype=np.float32)
            for kp_idx in range(self.num_keypoints):
                kp = ann['keypoints'][kp_idx * 3:(kp_idx + 1) * 3]
                joints[kp_idx, 0] = kp[0]
                joints[kp_idx, 1] = kp[1]
                vis = min(1, kp[2])
                joints_vis[kp_idx] = [vis, vis, 0]

            db.append({
                'image_path': image_path,
                'image_id': ann['image_id'],
                'ann_id': ann['id'],
                'center': center,
                'scale': scale,
                'joints': joints,
                'joints_vis': joints_vis,
                'bbox': [x, y, w, h],
                'crowd_index': img_info.get('crowdIndex', 0.0),
            })
        return db

    def _apply_split(self, db, ratio):
        random.seed(42)
        db_copy = db.copy()
        random.shuffle(db_copy)
        n = int(len(db_copy) * ratio)
        return db_copy[:n]

    def _box_to_center_scale(self, x, y, w, h):
        aspect_ratio = self.image_size[0] / self.image_size[1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w / self.PIXEL_STD, h / self.PIXEL_STD], dtype=np.float32)
        scale = scale * 1.25
        return center, scale

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        image = cv2.imread(db_rec['image_path'],
                           cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise FileNotFoundError(f"Image not found: {db_rec['image_path']}")

        sample = self.transform(
            image,
            db_rec['joints'],
            db_rec['joints_vis'],
            db_rec['center'],
            db_rec['scale'],
        )

        sample['image_id'] = db_rec['image_id']
        sample['ann_id'] = db_rec['ann_id']
        sample['crowd_index'] = db_rec['crowd_index']
        sample['bbox'] = torch.tensor(db_rec['bbox'], dtype=torch.float32)
        return sample

    def evaluate(self, preds, output_dir=None):
        """
        Evaluate using CrowdPose evaluation protocol.
        Reports AP, AP50, AP75 and crowd-level breakdown.
        """
        try:
            from crowdposetools.coco import COCO as CrowdCOCO
            from crowdposetools.cocoeval import COCOeval as CrowdEval
            use_crowdpose_api = True
        except ImportError:
            logger.warning("crowdposetools not installed — falling back to simple AP")
            use_crowdpose_api = False

        results = []
        for p in preds:
            results.append({
                'image_id': int(p['image_id']),
                'category_id': 1,
                'keypoints': p['keypoints'].flatten().tolist(),
                'score': float(p['score']),
            })

        if not results:
            return {}

        if use_crowdpose_api:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(results, f)
                tmp_path = f.name

            ann_file = os.path.join(
                self.root,
                f'annotations/crowdpose_{self.split}.json'
            )
            coco_gt = CrowdCOCO(ann_file)
            coco_dt = coco_gt.loadRes(tmp_path)
            ev = CrowdEval(coco_gt, coco_dt, 'keypoints')
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            stats = ev.stats
            return {
                'AP':   stats[0],
                'AP50': stats[1],
                'AP75': stats[2],
                'AR':   stats[5],
            }
        else:
            logger.warning("Returning empty metrics — install crowdposetools for eval")
            return {}
