# src/data/transforms.py
"""
Augmentation pipeline for pose estimation.
Follows standard HRNet preprocessing conventions for COCO/CrowdPose.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T


# ─── Affine helpers ───────────────────────────────────────────────────────────

def get_affine_transform(center, scale, rot, output_size, shift=(0., 0.), inv=False):
    """Compute affine transform matrix for cropping a person instance."""
    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale])
    scale_tmp = scale * 200.0  # scale is normalized by 200px

    src_w = scale_tmp[0]
    dst_w, dst_h = output_size

    rot_rad = np.pi * rot / 180
    src_dir = _rotate_point(np.array([0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def affine_transform(pt, trans):
    """Apply affine transform to a 2D point."""
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
    new_pt = trans.dot(new_pt)
    return new_pt[:2]


def _rotate_point(pt, angle_rad):
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    return np.array([pt[0] * cs - pt[1] * sn,
                     pt[0] * sn + pt[1] * cs], dtype=np.float32)


def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


# ─── Heatmap generation ───────────────────────────────────────────────────────

def generate_target_heatmaps(joints, joints_vis, heatmap_size, image_size, sigma=2):
    """
    Generate Gaussian heatmap targets for each keypoint.

    Args:
        joints: (num_kps, 3) — x, y, visibility
        joints_vis: (num_kps, 3) — visibility flags
        heatmap_size: (W, H) tuple
        image_size: (W, H) input image crop size
        sigma: Gaussian sigma

    Returns:
        target: (num_kps, H, W) float32
        target_weight: (num_kps, 1) float32
    """
    num_joints = joints.shape[0]
    heatmap_w, heatmap_h = heatmap_size

    target = np.zeros((num_joints, heatmap_h, heatmap_w), dtype=np.float32)
    target_weight = np.ones((num_joints, 1), dtype=np.float32)

    tmp_size = sigma * 3
    feat_stride = np.array(image_size) / np.array(heatmap_size)

    for joint_id in range(num_joints):
        target_weight[joint_id] = joints_vis[joint_id, 0]

        if joints_vis[joint_id, 0] == 0:
            continue

        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

        ul = [mu_x - tmp_size, mu_y - tmp_size]
        br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]

        if ul[0] >= heatmap_w or ul[1] >= heatmap_h or br[0] < 0 or br[1] < 0:
            target_weight[joint_id] = 0
            continue

        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        g_x = max(0, -ul[0]), min(br[0], heatmap_w) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_h) - ul[1]
        img_x = max(0, ul[0]), min(br[0], heatmap_w)
        img_y = max(0, ul[1]), min(br[1], heatmap_h)

        # Convert to integers for slice indices
        g_x = (int(g_x[0]), int(g_x[1]))
        g_y = (int(g_y[0]), int(g_y[1]))
        img_x = (int(img_x[0]), int(img_x[1]))
        img_y = (int(img_y[0]), int(img_y[1]))

        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight


# ─── Image normalization ──────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_normalize_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Augmentation pipeline ────────────────────────────────────────────────────

class PoseTransform:
    """
    Full augmentation pipeline for a single person crop:
    - Random rotation
    - Random scale
    - Random flip
    - Color jitter
    - Affine warp + heatmap generation
    """

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.image_size = np.array(cfg['dataset']['image_size'])    # [W, H]
        self.heatmap_size = np.array(cfg['model']['heatmap_size'])  # [W, H]
        self.sigma = cfg['model']['sigma']
        self.num_keypoints = cfg['model']['num_keypoints']

        aug = cfg['dataset'].get('augmentation', {})
        self.flip = aug.get('flip', True)
        self.flip_pairs = aug.get('flip_pairs', [])
        self.rot_factor = aug.get('rotation', 40)
        self.scale_factor = aug.get('scale', [0.65, 1.35])
        self.color_jitter = aug.get('color_jitter', True)

        self.normalize = get_normalize_transform()

        if self.color_jitter and is_train:
            self.jitter = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )
        else:
            self.jitter = None

    def __call__(self, image, joints, joints_vis, center, scale):
        """
        Args:
            image: HxWx3 uint8 numpy array (BGR from OpenCV)
            joints: (num_kps, 3) float32
            joints_vis: (num_kps, 3) float32
            center: (2,) float32
            scale: (2,) float32

        Returns:
            dict with 'image', 'target', 'target_weight', 'center', 'scale'
        """
        joints = joints.copy()
        joints_vis = joints_vis.copy()

        # ── Augmentations (train only) ────────────────────────────────────────
        rot = 0
        if self.is_train:
            # Scale jitter
            sf = self.scale_factor
            scale *= np.clip(np.random.randn() * 0.1 + 1.0,
                             sf[0], sf[1])

            # Rotation jitter
            rf = self.rot_factor
            if np.random.random() < 0.6:
                rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

            # Horizontal flip
            if self.flip and np.random.random() < 0.5:
                image, joints, joints_vis = self._flip(image, joints, joints_vis)
                center[0] = image.shape[1] - center[0] - 1

        # ── Affine crop ───────────────────────────────────────────────────────
        trans = get_affine_transform(center, scale, rot, self.image_size)

        inp = cv2.warpAffine(
            image, trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )

        for i in range(self.num_keypoints):
            if joints_vis[i, 0] > 0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        # ── Color jitter ──────────────────────────────────────────────────────
        if self.jitter is not None:
            from PIL import Image as PILImage
            inp_pil = PILImage.fromarray(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))
            inp_pil = self.jitter(inp_pil)
            inp = cv2.cvtColor(np.array(inp_pil), cv2.COLOR_RGB2BGR)

        # ── Normalize ─────────────────────────────────────────────────────────
        inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp_tensor = self.normalize(inp_rgb)

        # ── Heatmaps ──────────────────────────────────────────────────────────
        target, target_weight = generate_target_heatmaps(
            joints, joints_vis,
            self.heatmap_size, self.image_size,
            self.sigma
        )

        return {
            'image': inp_tensor,
            'target': torch.from_numpy(target),
            'target_weight': torch.from_numpy(target_weight),
            'center': torch.from_numpy(np.asarray(center, dtype=np.float32)),
            'scale': torch.from_numpy(np.asarray(scale, dtype=np.float32)),
            # Crop-space GT joints for coordinate losses (not heatmaps)
            'joints': torch.from_numpy(np.asarray(joints, dtype=np.float32)),
            'joints_vis': torch.from_numpy(np.asarray(joints_vis, dtype=np.float32)),
        }

    def _flip(self, image, joints, joints_vis):
        image = image[:, ::-1, :].copy()
        W = image.shape[1]

        joints_flipped = joints.copy()
        joints_vis_flipped = joints_vis.copy()

        joints_flipped[:, 0] = W - joints[:, 0] - 1

        for left, right in self.flip_pairs:
            joints_flipped[left], joints_flipped[right] = \
                joints[right].copy(), joints[left].copy()
            joints_vis_flipped[left], joints_vis_flipped[right] = \
                joints_vis[right].copy(), joints_vis[left].copy()

        return image, joints_flipped, joints_vis_flipped
