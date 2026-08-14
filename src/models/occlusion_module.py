"""
Occlusion-Aware Pose Reconstruction (OAPR) — GCN skeleton-graph module.

Detects occluded joints with a confidence gate (threshold τ) and reconstructs
them with a GCN over the COCO skeleton graph. No LSTM / temporal branch.

Article fusion (β-blend):
    p* = (1 − m) · p + m · (β · p_recon + (1 − β) · p)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)

# Undirected COCO-17 skeleton edges for the GCN
COCO_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class OcclusionDetector(nn.Module):
    """Confidence-gate occlusion detector (threshold τ)."""

    def __init__(self, num_joints: int = 17, occlusion_threshold: float = 0.5):
        super().__init__()
        self.num_joints = num_joints
        self.occlusion_threshold = float(occlusion_threshold)

    def forward(self, confidence: torch.Tensor):
        """
        Args:
            confidence: (B, K, 1)
        Returns:
            occluded_mask m: (B, K)  — 1 = reconstruct
            occlusion_score: (B, K)
        """
        occlusion_soft = 1.0 - confidence.squeeze(-1)
        occluded_mask = (occlusion_soft > self.occlusion_threshold).float()
        return occluded_mask, occlusion_soft


class SkeletonGCN(nn.Module):
    """One-hop message passing over the COCO skeleton graph."""

    def __init__(self, hidden_size: int = 256, num_joints: int = 17):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        self.edges = COCO_SKELETON_EDGES
        self.msg = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.ReLU(inplace=True)

    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_features: (B, K, C)
        Returns:
            (B, K, C) graph-refined features
        """
        B, K, C = joint_features.shape
        agg = torch.zeros_like(joint_features)
        counts = joint_features.new_zeros(B, K, 1)
        for i, j in self.edges:
            if i >= K or j >= K:
                continue
            pair_ij = torch.cat(
                [joint_features[:, i, :], joint_features[:, j, :]], dim=-1
            )
            pair_ji = torch.cat(
                [joint_features[:, j, :], joint_features[:, i, :]], dim=-1
            )
            msg_ij = self.act(self.msg(pair_ij))
            msg_ji = self.act(self.msg(pair_ji))
            agg[:, i, :] = agg[:, i, :] + msg_ij
            agg[:, j, :] = agg[:, j, :] + msg_ji
            counts[:, i, :] += 1.0
            counts[:, j, :] += 1.0
        agg = agg / counts.clamp(min=1.0)
        return self.norm(joint_features + agg)


class GCNPoseReconstructor(nn.Module):
    """Predict reconstructed coordinates from GCN-refined joint features."""

    def __init__(self, hidden_size: int = 256, num_joints: int = 17):
        super().__init__()
        self.gcn = SkeletonGCN(hidden_size, num_joints)
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_features: (B, K, C)
        Returns:
            p_recon: (B, K, 2)
        """
        refined = self.gcn(joint_features)
        return self.coord_head(refined)


class OcclusionAwarePoseReconstruction(nn.Module):
    """
    Occlusion gate (τ) + GCN reconstruction + β-blend fusion.

        p* = (1 − m) · p + m · (β · p_recon + (1 − β) · p)
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_joints: int = 17,
        occlusion_threshold: float = 0.5,
        fusion_beta: float = 0.5,
        use_gcn: bool = True,
        use_confidence_gate: bool = True,
    ):
        super().__init__()
        self.detector = OcclusionDetector(num_joints, occlusion_threshold)
        self.reconstructor = GCNPoseReconstructor(hidden_size, num_joints)
        self.fusion_beta = float(fusion_beta)
        self.use_gcn = bool(use_gcn)
        self.use_confidence_gate = bool(use_confidence_gate)

    def set_ablation(
        self,
        use_gcn=None,
        use_confidence_gate=None,
        occlusion_threshold=None,
        fusion_beta=None,
    ):
        if use_gcn is not None:
            self.use_gcn = bool(use_gcn)
        if use_confidence_gate is not None:
            self.use_confidence_gate = bool(use_confidence_gate)
        if occlusion_threshold is not None:
            self.detector.occlusion_threshold = float(occlusion_threshold)
        if fusion_beta is not None:
            self.fusion_beta = float(fusion_beta)

    def forward(
        self,
        joint_features: torch.Tensor,
        predictions: torch.Tensor,
        confidence: torch.Tensor,
    ):
        """
        Args:
            joint_features: (B, K, C)
            predictions p: (B, K, 2)
            confidence: (B, K, 1)
        """
        p = predictions[:, :, :2]

        if self.use_confidence_gate:
            m, occlusion_score = self.detector(confidence)
        else:
            m = torch.zeros(p.shape[:2], device=p.device, dtype=p.dtype)
            occlusion_score = 1.0 - confidence.squeeze(-1)

        if self.use_gcn:
            p_recon = self.reconstructor(joint_features)
        else:
            p_recon = p

        # p* = (1−m)·p + m·(β·p_recon + (1−β)·p)
        beta = self.fusion_beta
        m_exp = rearrange(m, "b k -> b k 1")
        blended = beta * p_recon + (1.0 - beta) * p
        p_star = (1.0 - m_exp) * p + m_exp * blended

        return {
            "coordinates": p_star,
            "confidence": confidence,
            "occlusion_mask": m,
            "occlusion_score": occlusion_score,
            "p_recon": p_recon,
        }


def build_oapr_module(cfg) -> OcclusionAwarePoseReconstruction:
    model_cfg = cfg.get("model", {})
    # fusion_beta (β); accept legacy key confidence_beta
    fusion_beta = model_cfg.get(
        "fusion_beta", model_cfg.get("confidence_beta", 0.5)
    )
    return OcclusionAwarePoseReconstruction(
        hidden_size=model_cfg.get("hidden_size", 256),
        num_joints=model_cfg.get("num_keypoints", 17),
        occlusion_threshold=model_cfg.get("occlusion_threshold", 0.5),
        fusion_beta=fusion_beta,
        use_gcn=model_cfg.get("use_gcn", True),
        use_confidence_gate=model_cfg.get("use_confidence_gate", True),
    )
