"""
OAPR: Occlusion-Aware Probabilistic Pose Reconstruction

Unified end-to-end model combining:
1. HRNet image encoder (heatmaps + features)
2. HybridMambaTransformer backbone (spatiotemporal modeling)
3. OcclusionAwarePoseReconstruction module (occlusion handling)
4. Probabilistic robust loss (distribution modeling)

This is the complete framework for Milestones 2 & 3.
"""

import logging
import torch
import torch.nn as nn
from einops import repeat

from .hrnet_baseline import (
    HRNetBaseline,
    dark_decode_torch,
    sample_joint_features,
)
from .mamba_backbone import HybridMambaTransformer
from .occlusion_module import OcclusionAwarePoseReconstruction
from .robust_loss import ProbabilisticPoseLoss

logger = logging.getLogger(__name__)


class OAPRFramework(nn.Module):
    """
    End-to-end OAPR (Occlusion-Aware Probabilistic Pose Reconstruction) framework.

    Architecture:
        Image crop (B, 3, H, W)
            ↓
        HRNet → heatmaps + feature maps
            ↓
        DARK decode → keypoints (B, K, 2) + conf; sample per-joint HRNet features
            ↓
        Hybrid Mamba-Transformer → spatiotemporal features + initial predictions
            ↓
        Occlusion-Aware Reconstruction → detect & refine occluded joints
            ↓
        Refined Keypoints (B, K, 2) + confidence
    """

    def __init__(self, num_keypoints=17, seq_len=7, hidden_size=256,
                 num_heads=8, num_spatial_layers=3, use_mamba=True,
                 occlusion_threshold=0.5, loss_type='cauchy_mixture',
                 backbone_type='mamba', hrnet_version='w32',
                 pretrained=True, heatmap_size=(64, 48),
                 image_size=(192, 256), use_dark=True):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.backbone_type = backbone_type
        self.image_size = tuple(image_size)  # (W, H)
        self.heatmap_size = tuple(heatmap_size)
        self.use_dark = use_dark

        logger.info(f"Building OAPR Framework (seq_len={seq_len}, hidden={hidden_size})")

        # Image encoder: HRNet → heatmaps + multi-scale features
        self.image_encoder = HRNetBaseline(
            num_keypoints=num_keypoints,
            version=hrnet_version,
            pretrained=pretrained,
            heatmap_size=heatmap_size,
        )
        logger.info(f"✓ HRNet-{hrnet_version} image encoder initialized")

        # Project sampled HRNet channels → temporal hidden size
        self.hrnet_proj = nn.Linear(self.image_encoder.out_channels, hidden_size)

        # Milestone 2: Spatiotemporal backbone
        self.backbone = HybridMambaTransformer(
            num_keypoints=num_keypoints,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_spatial_layers,
            use_mamba=use_mamba,
        )
        logger.info(f"✓ Backbone initialized (use_mamba={use_mamba})")

        # Milestone 3: Occlusion-aware reconstruction
        self.occlusion_module = OcclusionAwarePoseReconstruction(
            hidden_size=hidden_size,
            num_joints=num_keypoints,
            seq_len=seq_len,
            occlusion_threshold=occlusion_threshold,
        )
        logger.info("✓ Occlusion module initialized")

        # Milestone 3: Robust probabilistic loss
        self.criterion = ProbabilisticPoseLoss(
            loss_type=loss_type,
            scale_loss_weight=0.1,
            occlusion_loss_weight=0.1,
        )
        logger.info(f"✓ Robust loss initialized ({loss_type})")

    def forward(self, images, return_intermediate=False):
        """
        Forward pass for the full image→keypoint pipeline.

        Args:
            images: (B, 3, H, W) person-crop images
                    OR (B, T, K, 2) keypoint sequences (legacy/test path)
            return_intermediate: if True, return backbone + encoder outputs

        Returns:
            output dict with keypoints, confidence, occlusion_*, heatmaps
        """
        # Legacy / unit-test path: already-decoded keypoint sequences
        if images.dim() == 4 and images.shape[-1] == 2 and images.shape[2] == self.num_keypoints:
            return self._forward_from_keypoints(images, return_intermediate)

        # Stage 0: HRNet image encoder
        heatmaps, hrnet_feat_map = self.image_encoder(images, return_features=True)
        # heatmaps: (B, K, hm_h, hm_w); features: (B, C, Hf, Wf)

        # Stage 0b: DARK decode → crop-space keypoints + peak confidence
        coords, hm_conf = dark_decode_torch(
            heatmaps, self.image_size, use_dark=self.use_dark
        )  # (B, K, 2), (B, K, 1)

        # Per-joint HRNet features at predicted locations
        hrnet_joint = sample_joint_features(
            hrnet_feat_map, coords, self.image_size
        )  # (B, K, C)
        hrnet_joint = self.hrnet_proj(hrnet_joint)  # (B, K, hidden)

        # Single-image COCO: replicate decoded pose across the temporal window
        video_clip = coords.unsqueeze(1).expand(-1, self.seq_len, -1, -1).contiguous()
        # (B, T, K, 2)

        return self._forward_temporal(
            video_clip, hrnet_joint, heatmaps, hm_conf, return_intermediate
        )

    def _forward_from_keypoints(self, video_clip, return_intermediate=False):
        """Spatiotemporal path when inputs are already (B, T, K, 2)."""
        B = video_clip.shape[0]
        hrnet_joint = torch.zeros(
            B, self.num_keypoints, self.hidden_size,
            device=video_clip.device, dtype=video_clip.dtype,
        )
        return self._forward_temporal(
            video_clip, hrnet_joint, None, None, return_intermediate
        )

    def _forward_temporal(self, video_clip, hrnet_joint, heatmaps, hm_conf,
                          return_intermediate=False):
        # Stage 1: Spatiotemporal backbone — returns predictions + real hidden features
        backbone_output, joint_features = self.backbone(
            video_clip, joint_init_features=hrnet_joint
        )  # (B, K, 3), (B, K, hidden)

        predictions = backbone_output
        confidence = backbone_output[:, :, 2:3]  # (B, K, 1)
        if hm_conf is not None:
            # Blend heatmap peak score into backbone confidence
            confidence = 0.5 * confidence + 0.5 * hm_conf.clamp(0, 1)

        # Stage 2: Occlusion-aware reconstruction with real backbone features
        oapr_output = self.occlusion_module(
            joint_features, predictions, confidence, video_clip
        )

        output = {
            'keypoints': oapr_output['coordinates'],       # (B, K, 2) crop space
            'confidence': oapr_output['confidence'],       # (B, K, 1)
            'occlusion_mask': oapr_output['occlusion_mask'],
            'occlusion_score': oapr_output['occlusion_score'],
        }
        if heatmaps is not None:
            output['heatmaps'] = heatmaps

        if return_intermediate:
            output['backbone_output'] = backbone_output
            output['joint_features'] = joint_features
            output['hrnet_joint_features'] = hrnet_joint

        return output

    def compute_loss(self, predictions, targets, weights=None,
                     uncertainties=None, occlusion_scores=None):
        """
        Compute training loss using robust probabilistic objective.

        Args:
            predictions: (B, K, 2) or (B, K, 3) model outputs
            targets: (B, K, 2) ground truth (crop coords) — not heatmaps
            weights: (B, K, 1) visibility
            uncertainties: (B, K, 1) model confidence
            occlusion_scores: (B, K) occlusion detection

        Returns:
            loss: scalar
            loss_dict: breakdown
        """
        loss, loss_dict = self.criterion(
            predictions, targets, weights,
            uncertainties, occlusion_scores
        )
        return loss, loss_dict


class InstanceAwareRepresentation(nn.Module):
    """
    Structured instance-aware representation for multi-person modeling.

    Each person is represented as a separate token/embedding stream,
    preventing ambiguity in crowded scenes.
    """

    def __init__(self, hidden_size=256, num_instances=5, num_joints=17):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_instances = num_instances
        self.num_joints = num_joints

        # Instance tokens (learnable per-person embeddings)
        self.instance_tokens = nn.Parameter(
            torch.randn(num_instances, hidden_size)
        )
        nn.init.normal_(self.instance_tokens, std=0.02)

        # Cross-attention between instances and joints
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True, dropout=0.1
        )

        # Instance-specific pose heads
        self.pose_heads = nn.ModuleList([
            nn.Linear(hidden_size, num_joints * 3)
            for _ in range(num_instances)
        ])

    def forward(self, joint_features):
        """
        Args:
            joint_features: (B, num_joints, hidden_size)

        Returns:
            instance_poses: list of (B, num_joints, 3) per instance
        """
        B, K, C = joint_features.shape

        instance_poses = []
        for inst_idx in range(self.num_instances):
            inst_token = repeat(self.instance_tokens[inst_idx], 'd -> b 1 d', b=B)

            # Cross-attend to joints
            inst_feat, _ = self.cross_attn(
                inst_token, joint_features, joint_features
            )

            # Predict pose for this instance
            pose = self.pose_heads[inst_idx](inst_feat.squeeze(1))  # (B, K*3)
            pose = pose.reshape(B, K, 3)
            instance_poses.append(pose)

        return instance_poses


# ─── Factory Functions ─────────────────────────────────────────────────────

def build_oapr_framework(cfg):
    """Factory function to build complete OAPR framework from config."""
    model_cfg = cfg.get('model', {})
    loss_cfg = cfg.get('loss', {})
    if not loss_cfg:
        loss_cfg = cfg.get('training', {}).get('loss', {})
    dataset_cfg = cfg.get('dataset', {})
    eval_cfg = cfg.get('evaluation', {})

    image_size = model_cfg.get('image_size', dataset_cfg.get('image_size', [192, 256]))
    heatmap_size = model_cfg.get('heatmap_size', dataset_cfg.get('heatmap_size', [64, 48]))
    use_dark = eval_cfg.get('post_processing', 'dark') == 'dark'

    model = OAPRFramework(
        num_keypoints=model_cfg.get('num_keypoints', 17),
        seq_len=model_cfg.get('seq_len', 7),
        hidden_size=model_cfg.get('hidden_size', 256),
        num_heads=model_cfg.get('num_heads', 8),
        num_spatial_layers=model_cfg.get('num_spatial_layers', 3),
        use_mamba=model_cfg.get('use_mamba', True),
        occlusion_threshold=model_cfg.get('occlusion_threshold', 0.5),
        loss_type=loss_cfg.get('type', 'cauchy_mixture'),
        backbone_type=model_cfg.get('backbone_type', 'mamba'),
        hrnet_version=model_cfg.get('version', 'w32'),
        pretrained=model_cfg.get('pretrained', True),
        heatmap_size=heatmap_size,
        image_size=image_size,
        use_dark=use_dark,
    )

    return model


if __name__ == '__main__':
    # Test image→keypoint forward pass
    B, K = 2, 17
    config = {
        'model': {
            'num_keypoints': K,
            'seq_len': 7,
            'hidden_size': 256,
            'num_heads': 8,
            'num_spatial_layers': 2,
            'use_mamba': False,
            'version': 'w32',
            'pretrained': False,
            'heatmap_size': [64, 48],
            'image_size': [192, 256],
        },
        'loss': {'type': 'cauchy_mixture'},
        'evaluation': {'post_processing': 'dark'},
    }

    model = build_oapr_framework(config)
    images = torch.randn(B, 3, 256, 192)
    output = model(images)

    print(f"Output keys: {output.keys()}")
    print(f"Keypoints shape: {output['keypoints'].shape}")
    print(f"Confidence shape: {output['confidence'].shape}")
    print(f"Heatmaps shape: {output['heatmaps'].shape}")
    print(f"Occlusion mask shape: {output['occlusion_mask'].shape}")
