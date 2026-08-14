"""
OAPR: Occlusion-Aware Probabilistic Pose Reconstruction (single-image).

Article pipeline:
    person crop (B, 3, 256, 192)
        → HRNet-W32 features / heatmaps
        → DARK decode + per-joint feature sampling
        → Hybrid Mamba–Transformer joint encoder (sequence = K joints)
        → confidence head
        → occlusion gate (τ)
        → GCN skeleton-graph reconstruction
        → β-blend fusion
        → Cauchy-mixture loss (training)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

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
    End-to-end single-image OAPR.

        Image → HRNet → Mamba(joints) → Transformer(joints)
             → confidence → gate(τ) → GCN → β-blend → keypoints
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_spatial_layers: int = 4,
        use_mamba: bool = True,
        occlusion_threshold: float = 0.5,
        fusion_beta: float = 0.5,
        loss_type: str = "cauchy_mixture",
        hrnet_version: str = "w32",
        pretrained: bool = True,
        heatmap_size=(64, 48),
        image_size=(192, 256),
        use_dark: bool = True,
        use_gcn: bool = True,
        use_confidence_gate: bool = True,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.hidden_size = hidden_size
        self.image_size = tuple(image_size)  # (W, H)
        self.heatmap_size = tuple(heatmap_size)
        self.use_dark = use_dark
        self.fusion_beta = float(fusion_beta)
        self.occlusion_threshold = float(occlusion_threshold)
        # Alias kept for older gpu_jobs / sensitivity scripts
        self.confidence_beta = self.fusion_beta

        logger.info(
            f"Building OAPR (single-image, hidden={hidden_size}, "
            f"τ={occlusion_threshold}, β={fusion_beta})"
        )

        self.image_encoder = HRNetBaseline(
            num_keypoints=num_keypoints,
            version=hrnet_version,
            pretrained=pretrained,
            heatmap_size=heatmap_size,
        )
        self.hrnet_proj = nn.Linear(self.image_encoder.out_channels, hidden_size)

        self.backbone = HybridMambaTransformer(
            num_keypoints=num_keypoints,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_spatial_layers,
            use_mamba=use_mamba,
        )

        self.occlusion_module = OcclusionAwarePoseReconstruction(
            hidden_size=hidden_size,
            num_joints=num_keypoints,
            occlusion_threshold=occlusion_threshold,
            fusion_beta=fusion_beta,
            use_gcn=use_gcn,
            use_confidence_gate=use_confidence_gate,
        )

        self.criterion = ProbabilisticPoseLoss(
            loss_type=loss_type,
            scale_loss_weight=0.1,
            occlusion_loss_weight=0.1,
        )

    def set_ablation(
        self,
        use_gcn=None,
        use_confidence_gate=None,
        use_mamba=None,
        occlusion_threshold=None,
        confidence_beta=None,
        fusion_beta=None,
    ):
        """Runtime ablation / sensitivity controls (no weight reload)."""
        beta = fusion_beta if fusion_beta is not None else confidence_beta
        self.occlusion_module.set_ablation(
            use_gcn=use_gcn,
            use_confidence_gate=use_confidence_gate,
            occlusion_threshold=occlusion_threshold,
            fusion_beta=beta,
        )
        if occlusion_threshold is not None:
            self.occlusion_threshold = float(occlusion_threshold)
        if beta is not None:
            self.fusion_beta = float(beta)
            self.confidence_beta = self.fusion_beta
        if use_mamba is not None:
            # Informational only after construction — rebuild for a true Mamba-off run
            self.backbone.use_mamba = bool(use_mamba) and getattr(
                self.backbone, "use_mamba", False
            )

    def forward(self, images: torch.Tensor, return_intermediate: bool = False):
        """
        Args:
            images: (B, 3, H, W) person-crop images (e.g. H=256, W=192)
        """
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(
                f"OAPR expects single-image crops (B, 3, H, W); got {tuple(images.shape)}"
            )

        # 1) HRNet features
        heatmaps, hrnet_feat_map = self.image_encoder(images, return_features=True)
        coords_init, _hm_conf = dark_decode_torch(
            heatmaps, self.image_size, use_dark=self.use_dark
        )
        hrnet_joint = sample_joint_features(
            hrnet_feat_map, coords_init, self.image_size
        )
        joint_tokens = self.hrnet_proj(hrnet_joint)  # (B, K, C)

        # 2–4) Mamba(joints) → Transformer(joints) → confidence head (+ coords)
        p, confidence, joint_features = self.backbone(joint_tokens)

        # Residual toward DARK init (stable single-image localization)
        p = p + coords_init

        # 5–7) Occlusion gate (τ) → GCN → β-blend
        oapr_output = self.occlusion_module(joint_features, p, confidence)

        output = {
            "keypoints": oapr_output["coordinates"],
            "confidence": oapr_output["confidence"],
            "occlusion_mask": oapr_output["occlusion_mask"],
            "occlusion_score": oapr_output["occlusion_score"],
            "heatmaps": heatmaps,
        }
        if return_intermediate:
            output["backbone_coords"] = p
            output["joint_features"] = joint_features
            output["hrnet_joint_features"] = joint_tokens
            output["p_recon"] = oapr_output["p_recon"]
        return output

    def compute_loss(
        self,
        predictions,
        targets,
        weights=None,
        uncertainties=None,
        occlusion_scores=None,
    ):
        """Cauchy-mixture (or configured) probabilistic pose loss."""
        return self.criterion(
            predictions, targets, weights, uncertainties, occlusion_scores
        )


def build_oapr_framework(cfg) -> OAPRFramework:
    """Build OAPR from a YAML-loaded config dict."""
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {}) or cfg.get("training", {}).get("loss", {})
    dataset_cfg = cfg.get("dataset", {})
    eval_cfg = cfg.get("evaluation", {})

    image_size = model_cfg.get(
        "image_size", dataset_cfg.get("image_size", [192, 256])
    )
    heatmap_size = model_cfg.get(
        "heatmap_size", dataset_cfg.get("heatmap_size", [64, 48])
    )
    use_dark = eval_cfg.get("post_processing", "dark") == "dark"
    fusion_beta = model_cfg.get(
        "fusion_beta", model_cfg.get("confidence_beta", 0.5)
    )

    return OAPRFramework(
        num_keypoints=model_cfg.get("num_keypoints", 17),
        hidden_size=model_cfg.get("hidden_size", 256),
        num_heads=model_cfg.get("num_heads", 8),
        num_spatial_layers=model_cfg.get("num_spatial_layers", 4),
        use_mamba=model_cfg.get("use_mamba", True),
        occlusion_threshold=model_cfg.get("occlusion_threshold", 0.5),
        fusion_beta=fusion_beta,
        loss_type=loss_cfg.get("type", "cauchy_mixture"),
        hrnet_version=model_cfg.get("version", "w32"),
        pretrained=model_cfg.get("pretrained", True),
        heatmap_size=heatmap_size,
        image_size=image_size,
        use_dark=use_dark,
        use_gcn=model_cfg.get("use_gcn", True),
        use_confidence_gate=model_cfg.get("use_confidence_gate", True),
    )


if __name__ == "__main__":
    B, K = 2, 17
    config = {
        "model": {
            "num_keypoints": K,
            "hidden_size": 256,
            "num_heads": 8,
            "num_spatial_layers": 2,
            "use_mamba": False,
            "version": "w32",
            "pretrained": False,
            "heatmap_size": [64, 48],
            "image_size": [192, 256],
            "fusion_beta": 0.5,
        },
        "loss": {"type": "cauchy_mixture"},
        "evaluation": {"post_processing": "dark"},
    }
    model = build_oapr_framework(config)
    images = torch.zeros(B, 3, 256, 192)
    output = model(images)
    print({k: tuple(v.shape) for k, v in output.items() if hasattr(v, "shape")})
