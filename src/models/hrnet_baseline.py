# src/models/hrnet_baseline.py
"""
HRNet (High-Resolution Network) baseline for pose estimation.

Uses timm's pretrained HRNet backbone with a simple 1x1 conv head
that outputs per-keypoint heatmaps.

This is the Milestone 1 baseline. All future ablations (Mamba backbone,
occlusion module, Cauchy loss) will be compared against this.

Reference: Wang et al., "Deep High-Resolution Representation Learning
for Visual Recognition", TPAMI 2020.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

logger = logging.getLogger(__name__)

# Map config version string to timm model name
HRNET_VARIANTS = {
    'w32': 'hrnet_w32',
    'w48': 'hrnet_w48',
    'w18': 'hrnet_w18',
    'w64': 'hrnet_w64',
}


class HRNetBaseline(nn.Module):
    """
    HRNet backbone + keypoint head.

    Architecture:
        Input (B, 3, H, W)
            ↓
        HRNet backbone (pretrained on ImageNet)
            ↓
        Multi-scale feature fusion → high-res feature map (B, C, H/4, W/4)
            ↓
        Final layer: 1x1 conv → (B, num_keypoints, H/4, W/4) heatmaps

    Args:
        num_keypoints: number of output keypoints (17 for COCO, 14 for CrowdPose)
        version: 'w32' | 'w48' | 'w18' | 'w64'
        pretrained: load pretrained ImageNet weights
    """

    def __init__(self, num_keypoints=17, version='w32', pretrained=True,
                 heatmap_size=(64, 48)):
        super().__init__()
        self.num_keypoints = num_keypoints
        # heatmap_size: (W, H) to match dataset targets
        self.heatmap_size = tuple(heatmap_size)

        model_name = HRNET_VARIANTS.get(version)
        if model_name is None:
            raise ValueError(f"Unknown HRNet version '{version}'. "
                             f"Choose from {list(HRNET_VARIANTS.keys())}")

        logger.info(f"Loading HRNet backbone: {model_name} "
                    f"(pretrained={pretrained})")

        # timm HRNet with features_only=True returns multi-scale feature maps
        # We need the last feature (highest resolution = stride 4)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,  # full model, we'll hook into it
            num_classes=0,        # remove classifier head
        )

        # Infer backbone output channels (timm HRNet variants differ by version)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 192)
            feats = self.backbone.forward_features(dummy)
            backbone_out_channels = feats.shape[1]

        # Final prediction head: 1x1 conv
        self.final_layer = nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0
        )
        nn.init.normal_(self.final_layer.weight, std=0.001)
        nn.init.constant_(self.final_layer.bias, 0)

        logger.info(f"HRNetBaseline: {num_keypoints} keypoints, "
                    f"backbone_out={backbone_out_channels}ch")

    def _get_backbone_out_channels(self, version):
        channel_map = {
            'w18': 18,
            'w32': 32,
            'w48': 48,
            'w64': 64,
        }
        return channel_map[version]

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) normalized input images

        Returns:
            heatmaps: (B, num_keypoints, H/4, W/4) predicted heatmaps
        """
        # HRNet internal forward gives us multi-resolution features
        # The backbone's forward returns the final fused feature map
        features = self.backbone.forward_features(x)

        # features shape: (B, C, H/4, W/4) — highest resolution branch
        heatmaps = self.final_layer(features)

        # Upsample to target heatmap resolution (W, H)
        hm_w, hm_h = self.heatmap_size
        if heatmaps.shape[-2] != hm_h or heatmaps.shape[-1] != hm_w:
            heatmaps = F.interpolate(
                heatmaps, size=(hm_h, hm_w), mode='bilinear', align_corners=False
            )

        return heatmaps


# ─── Loss ─────────────────────────────────────────────────────────────────────

class JointsMSELoss(nn.Module):
    """
    Weighted MSE loss on heatmaps — standard baseline loss.
    Will be replaced by Cauchy loss in Milestone 3.

    Args:
        use_target_weight: weight loss by joint visibility
    """

    def __init__(self, use_target_weight=True):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, output, target, target_weight):
        """
        Args:
            output: (B, K, H, W) predicted heatmaps
            target: (B, K, H, W) GT heatmaps
            target_weight: (B, K, 1) per-joint visibility weights

        Returns:
            scalar loss
        """
        B, K, H, W = output.shape

        # Reshape for per-joint weighting
        pred = output.reshape(B, K, -1)      # (B, K, H*W)
        gt   = target.reshape(B, K, -1)      # (B, K, H*W)

        if self.use_target_weight:
            # target_weight: (B, K, 1)
            pred = pred * target_weight
            gt   = gt   * target_weight

        loss = self.mse(pred, gt)
        return loss


# ─── Post-processing ──────────────────────────────────────────────────────────

def get_max_preds(batch_heatmaps):
    """
    Get predictions from heatmaps using argmax.
    Returns keypoint coordinates in heatmap space.

    Args:
        batch_heatmaps: (B, K, H, W) numpy float32

    Returns:
        preds: (B, K, 2) coordinates [x, y] in heatmap space
        maxvals: (B, K, 1) confidence scores
    """
    import numpy as np
    B, K, H, W = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape((B, K, -1))

    idx = np.argmax(heatmaps_reshaped, axis=2)
    maxvals = np.amax(heatmaps_reshaped, axis=2)

    preds = np.zeros((B, K, 2), dtype=np.float32)
    preds[:, :, 0] = idx % W   # x
    preds[:, :, 1] = idx // W  # y

    preds = np.where(
        np.tile(maxvals[:, :, np.newaxis], (1, 1, 2)) > 0,
        preds, 0
    )
    return preds, maxvals[:, :, np.newaxis]


def decode_heatmaps(heatmaps, center, scale, heatmap_size, image_size,
                    use_dark=True):
    """
    Decode heatmap predictions back to original image coordinates.

    Supports DARK post-processing (Distribution-Aware coordinate Representation
    of Keypoints, Zhang et al., 2020) which improves AP ~1%.

    Args:
        heatmaps: (B, K, H, W) numpy float32
        center: (B, 2) person center
        scale: (B, 2) person scale
        heatmap_size: [W, H]
        image_size: [W, H]
        use_dark: use DARK post-processing (recommended)

    Returns:
        preds: (B, K, 2) keypoint coordinates in original image space
        maxvals: (B, K, 1) confidence scores
    """
    import numpy as np
    from src.data.transforms import get_affine_transform, affine_transform

    coords, maxvals = get_max_preds(heatmaps)

    heatmap_h, heatmap_w = heatmaps.shape[2], heatmaps.shape[3]

    if use_dark:
        coords = _dark_postprocess(heatmaps, coords)

    # Scale to image space
    for i in range(coords.shape[0]):
        trans_inv = get_affine_transform(
            center[i], scale[i], 0, image_size, inv=True
        )
        for j in range(coords.shape[1]):
            # Scale from heatmap to input image
            pt_hm = coords[i, j].copy()
            pt_hm[0] = pt_hm[0] / heatmap_w * image_size[0]
            pt_hm[1] = pt_hm[1] / heatmap_h * image_size[1]
            coords[i, j] = affine_transform(pt_hm, trans_inv)

    return coords, maxvals


def _dark_postprocess(batch_heatmaps, coords):
    """
    DARK post-processing: fit a 2D Gaussian to the heatmap peak region
    and use the analytical peak for sub-pixel accuracy.
    """
    import numpy as np

    B, K, H, W = batch_heatmaps.shape

    # Apply Gaussian blur to smooth noisy heatmaps
    import cv2
    batch_heatmaps = batch_heatmaps.copy()
    for b in range(B):
        for k in range(K):
            batch_heatmaps[b, k] = cv2.GaussianBlur(
                batch_heatmaps[b, k], (11, 11), 0
            )

    # Compute log of heatmaps (avoid log(0))
    batch_heatmaps = np.maximum(batch_heatmaps, 1e-10)
    batch_heatmaps = np.log(batch_heatmaps)

    coords = coords.astype(np.int32)

    for b in range(B):
        for k in range(K):
            heatmap = batch_heatmaps[b, k]
            x, y = coords[b, k, 0], coords[b, k, 1]

            if 1 < x < W - 2 and 1 < y < H - 2:
                dx  = 0.5 * (heatmap[y, x+1] - heatmap[y, x-1])
                dy  = 0.5 * (heatmap[y+1, x] - heatmap[y-1, x])
                dxx = 0.25 * (heatmap[y, x+2] - 2*heatmap[y, x] + heatmap[y, x-2])
                dyy = 0.25 * (heatmap[y+2, x] - 2*heatmap[y, x] + heatmap[y-2, x])
                dxy = 0.25 * (heatmap[y+1, x+1] - heatmap[y-1, x+1]
                              - heatmap[y+1, x-1] + heatmap[y-1, x-1])

                det = dxx * dyy - dxy ** 2
                if abs(det) > 1e-6:
                    inv_dxx = dyy / det
                    inv_dyy = dxx / det
                    inv_dxy = -dxy / det
                    offset_x = -(inv_dxx * dx + inv_dxy * dy)
                    offset_y = -(inv_dxy * dx + inv_dyy * dy)
                    offset_x = max(-0.5, min(0.5, offset_x))
                    offset_y = max(-0.5, min(0.5, offset_y))
                    coords[b, k, 0] += offset_x
                    coords[b, k, 1] += offset_y

    return coords.astype(np.float32)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_model(cfg):
    """Build model from config dict."""
    model_cfg = cfg['model']
    model = HRNetBaseline(
        num_keypoints=model_cfg['num_keypoints'],
        version=model_cfg.get('version', 'w32'),
        pretrained=model_cfg.get('pretrained', True),
        heatmap_size=model_cfg.get('heatmap_size', [64, 48]),
    )
    return model
