"""
Occlusion-Aware Pose Reconstruction (OAPR) Module.

Detects occluded joints using uncertainty estimates and reconstructs them using:
- Temporal context (motion continuity) — LSTM branch
- Spatial context (joint relationships) — GCN / graph branch
- Confidence gate — occlusion mask from inverse confidence vs τ

Supports runtime ablation toggles used by gpu_jobs/02_ablation.py.
"""

import logging
import torch
import torch.nn as nn
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


class OcclusionDetector(nn.Module):
    """Confidence-gate occlusion detector (threshold τ)."""

    def __init__(self, hidden_size=256, num_joints=17, occlusion_threshold=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        self.occlusion_threshold = occlusion_threshold
        self.occlusion_correlation = nn.Parameter(
            torch.ones(num_joints, num_joints) / num_joints
        )

    def forward(self, predictions, confidence):
        """
        Args:
            predictions: (B, K, 2)
            confidence: (B, K, 1)
        Returns:
            occluded_mask: (B, K)
            occlusion_score: (B, K)
        """
        occlusion_soft = 1.0 - confidence.squeeze(-1)  # (B, K)
        occluded_mask = (occlusion_soft > self.occlusion_threshold).float()
        return occluded_mask, occlusion_soft


class SpatialContextEncoder(nn.Module):
    """Graph / GCN-style spatial context over COCO skeleton edges."""

    def __init__(self, hidden_size=256, num_joints=17):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        self.coco_skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6),
            (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12),
            (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
        ]
        self.edge_embed = nn.Embedding(len(self.coco_skeleton), hidden_size)
        self.graph_conv = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, joint_features):
        B, K, C = joint_features.shape
        spatial_context = joint_features.clone()
        for edge_idx, (i, j) in enumerate(self.coco_skeleton):
            if i < K and j < K:
                neighbor_feat = torch.cat([
                    joint_features[:, i, :],
                    joint_features[:, j, :],
                ], dim=-1)
                aggregated = self.graph_conv(neighbor_feat)
                spatial_context[:, i, :] = spatial_context[:, i, :] + aggregated * 0.1
        return self.norm(spatial_context)


class TemporalContextEncoder(nn.Module):
    """LSTM temporal / motion branch."""

    def __init__(self, hidden_size=256, num_joints=17, seq_len=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.motion_head = nn.Linear(hidden_size, 2)

    def forward(self, joint_trajectories):
        B, T, K, _ = joint_trajectories.shape
        motion_vectors = []
        for k in range(K):
            traj_k = joint_trajectories[:, :, k, :]
            _, (h_n, _) = self.lstm(traj_k)
            motion = self.motion_head(h_n[-1])
            motion_vectors.append(motion)
        return torch.stack(motion_vectors, dim=1)  # (B, K, 2)


class PoseReconstructor(nn.Module):
    """Multi-context reconstruction (GCN spatial + LSTM temporal)."""

    def __init__(self, hidden_size=256, num_joints=17, seq_len=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        self.spatial_encoder = SpatialContextEncoder(hidden_size, num_joints)
        self.temporal_encoder = TemporalContextEncoder(hidden_size, num_joints, seq_len)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + 2 + 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, joint_features, predictions, motion_vectors,
                occluded_mask, joint_trajectories,
                use_gcn=True, use_lstm=True):
        B, K, C = joint_features.shape

        if use_gcn:
            spatial_ctx = self.spatial_encoder(joint_features)
        else:
            spatial_ctx = joint_features

        if use_lstm:
            temporal_ctx = self.temporal_encoder(joint_trajectories)
        else:
            temporal_ctx = torch.zeros_like(predictions)

        fused = torch.cat([spatial_ctx, predictions, temporal_ctx], dim=-1)
        reconstructed = self.fusion(fused)
        recon_conf = self.confidence_head(spatial_ctx)

        occluded_expanded = rearrange(occluded_mask, 'b k -> b k 1')
        final_coords = (
            predictions * (1 - occluded_expanded)
            + reconstructed * occluded_expanded
        )
        final_conf = 0.7 * recon_conf * occluded_expanded + (1 - occluded_expanded)
        return final_coords, final_conf


class OcclusionAwarePoseReconstruction(nn.Module):
    """Detect + reconstruct occluded joints (with ablation toggles)."""

    def __init__(self, hidden_size=256, num_joints=17, seq_len=7,
                 occlusion_threshold=0.5,
                 use_gcn=True, use_lstm=True, use_confidence_gate=True):
        super().__init__()
        self.detector = OcclusionDetector(
            hidden_size, num_joints, occlusion_threshold
        )
        self.reconstructor = PoseReconstructor(hidden_size, num_joints, seq_len)
        self.use_gcn = use_gcn
        self.use_lstm = use_lstm
        self.use_confidence_gate = use_confidence_gate

    def set_ablation(self, use_gcn=None, use_lstm=None, use_confidence_gate=None,
                     occlusion_threshold=None):
        """Runtime ablation / sensitivity knobs (no weight reload)."""
        if use_gcn is not None:
            self.use_gcn = bool(use_gcn)
        if use_lstm is not None:
            self.use_lstm = bool(use_lstm)
        if use_confidence_gate is not None:
            self.use_confidence_gate = bool(use_confidence_gate)
        if occlusion_threshold is not None:
            self.detector.occlusion_threshold = float(occlusion_threshold)

    def forward(self, joint_features, predictions, confidence, joint_trajectories):
        pred_coords = predictions[:, :, :2]

        if self.use_confidence_gate:
            occluded_mask, occlusion_score = self.detector(pred_coords, confidence)
        else:
            # Gate OFF → never trigger reconstruction (pass-through backbone)
            occluded_mask = torch.zeros(
                pred_coords.shape[:2], device=pred_coords.device, dtype=pred_coords.dtype
            )
            occlusion_score = 1.0 - confidence.squeeze(-1)

        reconstructed_coords, reconstructed_conf = self.reconstructor(
            joint_features, pred_coords,
            None,
            occluded_mask,
            joint_trajectories,
            use_gcn=self.use_gcn,
            use_lstm=self.use_lstm,
        )

        return {
            'coordinates': reconstructed_coords,
            'confidence': reconstructed_conf,
            'occlusion_mask': occluded_mask,
            'occlusion_score': occlusion_score,
        }


def build_oapr_module(cfg):
    model_cfg = cfg.get('model', {})
    return OcclusionAwarePoseReconstruction(
        hidden_size=model_cfg.get('hidden_size', 256),
        num_joints=model_cfg.get('num_keypoints', 17),
        seq_len=model_cfg.get('seq_len', 7),
        occlusion_threshold=model_cfg.get('occlusion_threshold', 0.5),
        use_gcn=model_cfg.get('use_gcn', True),
        use_lstm=model_cfg.get('use_lstm', True),
        use_confidence_gate=model_cfg.get('use_confidence_gate', True),
    )
