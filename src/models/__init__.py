# src/models/__init__.py
from .hrnet_baseline import (
    HRNetBaseline, JointsMSELoss, build_model,
    decode_heatmaps, decode_heatmaps_to_crop,
    dark_decode_torch, sample_joint_features,
)
from .mamba_backbone import HybridMambaTransformer, build_spatiotemporal_model
from .occlusion_module import OcclusionAwarePoseReconstruction, build_oapr_module
from .robust_loss import (
    CauchyLoss, LaplaceLoss, CauchyMixtureLoss,
    ProbabilisticPoseLoss, build_robust_loss
)
from .oapr_framework import OAPRFramework, build_oapr_framework


def build_pose_model(cfg):
    """Build HRNet baseline or full OAPR framework from config ``model.name``."""
    name = str(cfg.get('model', {}).get('name', 'hrnet')).lower()
    if name in ('oapr_complete', 'oapr', 'oapr_m3', 'mamba', 'mamba_temporal'):
        return build_oapr_framework(cfg)
    return build_model(cfg)
