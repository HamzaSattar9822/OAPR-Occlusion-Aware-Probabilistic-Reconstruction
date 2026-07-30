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
        self.out_channels = backbone_out_channels

    def _get_backbone_out_channels(self, version):
        channel_map = {
            'w18': 18,
            'w32': 32,
            'w48': 48,
            'w64': 64,
        }
        return channel_map[version]

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, 3, H, W) normalized input images
            return_features: if True, also return backbone feature maps

        Returns:
            heatmaps: (B, num_keypoints, hm_h, hm_w)
            features (optional): (B, C, H/4, W/4) backbone feature maps
        """
        features = self.backbone.forward_features(x)
        heatmaps = self.final_layer(features)

        hm_w, hm_h = self.heatmap_size
        if heatmaps.shape[-2] != hm_h or heatmaps.shape[-1] != hm_w:
            heatmaps = F.interpolate(
                heatmaps, size=(hm_h, hm_w), mode='bilinear', align_corners=False
            )

        if return_features:
            return heatmaps, features
        return heatmaps


def sample_joint_features(feature_map, coords_crop, image_size):
    """
    Bilinear-sample backbone features at predicted joint locations.

    Args:
        feature_map: (B, C, Hf, Wf)
        coords_crop: (B, K, 2) keypoints in crop pixel space [W, H]
        image_size: (W, H) model input crop size

    Returns:
        joint_features: (B, K, C)
    """
    B, C, Hf, Wf = feature_map.shape
    W, H = int(image_size[0]), int(image_size[1])
    x = coords_crop[..., 0] / max(W - 1, 1) * 2.0 - 1.0
    y = coords_crop[..., 1] / max(H - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).unsqueeze(2)  # (B, K, 1, 2)
    sampled = F.grid_sample(
        feature_map, grid, mode='bilinear', padding_mode='border', align_corners=True
    )  # (B, C, K, 1)
    return sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B, K, C)


def _gaussian_kernel2d(kernel_size, sigma, device, dtype):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def dark_decode_torch(heatmaps, image_size, use_dark=True):
    """
    Decode heatmaps to crop-space keypoints with optional DARK refinement.

    Args:
        heatmaps: (B, K, H, W) torch tensor
        image_size: (W, H) crop size
        use_dark: apply Distribution-Aware coordinate Representation refinement

    Returns:
        coords: (B, K, 2) in crop pixel coordinates
        maxvals: (B, K, 1) peak confidence scores
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device
    dtype = heatmaps.dtype

    flat = heatmaps.reshape(B, K, -1)
    maxvals, idx = flat.max(dim=-1)
    coords = torch.zeros(B, K, 2, device=device, dtype=dtype)
    coords[..., 0] = (idx % W).to(dtype)
    coords[..., 1] = (idx // W).to(dtype)

    if use_dark:
        # Gaussian blur in log-space, then Taylor refinement around the peak
        kernel = _gaussian_kernel2d(11, 2.0, device, dtype).view(1, 1, 11, 11)
        hm = heatmaps.reshape(B * K, 1, H, W)
        hm_blur = F.conv2d(hm, kernel, padding=5).reshape(B, K, H, W)
        hm_log = torch.log(torch.clamp(hm_blur, min=1e-10))

        x = coords[..., 0].long().clamp(2, W - 3)
        y = coords[..., 1].long().clamp(2, H - 3)
        b_idx = torch.arange(B, device=device)[:, None].expand(B, K)
        k_idx = torch.arange(K, device=device)[None, :].expand(B, K)

        def at(yy, xx):
            return hm_log[b_idx, k_idx, yy, xx]

        dx = 0.5 * (at(y, x + 1) - at(y, x - 1))
        dy = 0.5 * (at(y + 1, x) - at(y - 1, x))
        dxx = 0.25 * (at(y, x + 2) - 2 * at(y, x) + at(y, x - 2))
        dyy = 0.25 * (at(y + 2, x) - 2 * at(y, x) + at(y - 2, x))
        dxy = 0.25 * (at(y + 1, x + 1) - at(y - 1, x + 1)
                      - at(y + 1, x - 1) + at(y - 1, x - 1))

        det = dxx * dyy - dxy ** 2
        valid = det.abs() > 1e-6
        inv_dxx = torch.where(valid, dyy / det, torch.zeros_like(det))
        inv_dyy = torch.where(valid, dxx / det, torch.zeros_like(det))
        inv_dxy = torch.where(valid, -dxy / det, torch.zeros_like(det))
        offset_x = -(inv_dxx * dx + inv_dxy * dy).clamp(-0.5, 0.5)
        offset_y = -(inv_dxy * dx + inv_dyy * dy).clamp(-0.5, 0.5)

        # Only refine interior peaks (same guard as numpy DARK)
        interior = (coords[..., 0] > 1) & (coords[..., 0] < W - 2) & \
                   (coords[..., 1] > 1) & (coords[..., 1] < H - 2)
        offset_x = torch.where(interior & valid, offset_x, torch.zeros_like(offset_x))
        offset_y = torch.where(interior & valid, offset_y, torch.zeros_like(offset_y))
        coords = coords.clone()
        coords[..., 0] = coords[..., 0] + offset_x
        coords[..., 1] = coords[..., 1] + offset_y

    # Heatmap space → crop pixel space
    img_w, img_h = float(image_size[0]), float(image_size[1])
    coords = coords.clone()
    coords[..., 0] = coords[..., 0] / max(W, 1) * img_w
    coords[..., 1] = coords[..., 1] / max(H, 1) * img_h
    return coords, maxvals.unsqueeze(-1)


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


def decode_heatmaps_to_crop(heatmaps, heatmap_size, image_size, use_dark=True):
    """
    Decode heatmaps to coordinates in the **model input crop** space (image_size).

    Use this for drawing skeletons on the normalized person crops produced by the
    dataloader. ``decode_heatmaps`` maps to full original-image coordinates and
    must not be used for crop visualization.

    Args:
        heatmaps: (B, K, H, W) numpy float32
        heatmap_size: [W, H] (config field; used for API consistency)
        image_size: [W, H] model input crop size
        use_dark: apply DARK sub-pixel refinement

    Returns:
        coords: (B, K, 2) in crop pixel space
        maxvals: (B, K, 1) heatmap peak scores
    """
    import numpy as np

    coords, maxvals = get_max_preds(heatmaps)
    heatmap_h, heatmap_w = heatmaps.shape[2], heatmaps.shape[3]

    if use_dark:
        coords = _dark_postprocess(heatmaps, coords)

    # Scale heatmap (x, y) -> crop (W, H)
    coords = coords.astype(np.float32)
    coords[:, :, 0] = coords[:, :, 0] / heatmap_w * image_size[0]
    coords[:, :, 1] = coords[:, :, 1] / heatmap_h * image_size[1]

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
