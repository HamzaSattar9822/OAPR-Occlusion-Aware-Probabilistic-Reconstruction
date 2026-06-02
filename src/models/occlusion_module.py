"""
Occlusion-Aware Pose Reconstruction (OAPR) Module.

Detects occluded joints using uncertainty estimates and reconstructs them using:
- Temporal context (motion continuity)
- Spatial context (joint relationships)
- Instance context (other visible joints)

This is the CORE NOVELTY of the framework.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


class OcclusionDetector(nn.Module):
    """
    Detects occluded or low-confidence joints.
    Uses uncertainty estimates from the backbone.
    
    Args:
        hidden_size: model dimension
        num_joints: number of keypoints
        occlusion_threshold: confidence threshold for occlusion detection
    """
    
    def __init__(self, hidden_size=256, num_joints=17, occlusion_threshold=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        self.occlusion_threshold = occlusion_threshold
        
        # Learn which joints are likely to be occluded together
        self.occlusion_correlation = nn.Parameter(
            torch.ones(num_joints, num_joints) / num_joints
        )
    
    def forward(self, predictions, confidence):
        """
        Args:
            predictions: (B, K, 2) joint coordinates
            confidence: (B, K, 1) per-joint confidence scores
        
        Returns:
            occluded_mask: (B, K) binary mask where 1 = occluded
            occlusion_score: (B, K) soft occlusion probability
        """
        B, K, _ = predictions.shape
        
        # Soft occlusion score (inverse of confidence)
        occlusion_soft = 1.0 - confidence.squeeze(-1)  # (B, K)
        
        # Hard occlusion mask
        occluded_mask = (occlusion_soft > self.occlusion_threshold).float()
        
        return occluded_mask, occlusion_soft


class SpatialContextEncoder(nn.Module):
    """
    Encodes spatial relationships between joints for occlusion recovery.
    Uses graph-based reasoning over skeletal structure.
    """
    
    def __init__(self, hidden_size=256, num_joints=17):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        
        # COCO skeleton adjacency
        self.coco_skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # head
            (5, 6),                           # shoulders
            (5, 7), (7, 9), (6, 8), (8, 10), # arms
            (5, 11), (6, 12),                # torso
            (11, 12),                        # hips
            (11, 13), (13, 15), (12, 14), (14, 16),  # legs
        ]
        
        # Learned edge embeddings
        self.edge_embed = nn.Embedding(len(self.coco_skeleton), hidden_size)
        
        # Graph convolution layer
        self.graph_conv = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, joint_features):
        """
        Args:
            joint_features: (B, K, hidden_size)
        
        Returns:
            spatial_context: (B, K, hidden_size)
        """
        B, K, C = joint_features.shape
        
        spatial_context = joint_features.clone()
        
        # Aggregate information from neighbors
        for edge_idx, (i, j) in enumerate(self.coco_skeleton):
            if i < K and j < K:
                neighbor_feat = torch.cat([
                    joint_features[:, i, :],
                    joint_features[:, j, :]
                ], dim=-1)  # (B, 2*C)
                
                edge_emb = self.edge_embed(torch.tensor(edge_idx, device=joint_features.device))
                edge_emb = repeat(edge_emb, 'd -> b d', b=B)
                
                aggregated = self.graph_conv(neighbor_feat)  # (B, C)
                spatial_context[:, i, :] = spatial_context[:, i, :] + aggregated * 0.1
        
        return self.norm(spatial_context)


class TemporalContextEncoder(nn.Module):
    """
    Encodes temporal motion history for smooth pose reconstruction.
    """
    
    def __init__(self, hidden_size=256, num_joints=17, seq_len=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        self.seq_len = seq_len
        
        self.lstm = nn.LSTM(
            input_size=2,  # (x, y)
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.motion_head = nn.Linear(hidden_size, 2)  # predict motion vector
    
    def forward(self, joint_trajectories):
        """
        Args:
            joint_trajectories: (B, T, K, 2)
        
        Returns:
            motion_vectors: (B, K, 2)
        """
        B, T, K, _ = joint_trajectories.shape
        
        # Process each joint's trajectory
        motion_vectors = []
        for k in range(K):
            traj_k = joint_trajectories[:, :, k, :]  # (B, T, 2)
            _, (h_n, _) = self.lstm(traj_k)
            motion = self.motion_head(h_n[-1])  # (B, 2)
            motion_vectors.append(motion)
        
        motion_vectors = torch.stack(motion_vectors, dim=1)  # (B, K, 2)
        return motion_vectors


class PoseReconstructor(nn.Module):
    """
    Reconstructs missing/occluded joints using multi-context fusion.
    
    Core innovation: Instead of predicting, we RECONSTRUCT using:
    1. Visible joints (spatial context)
    2. Past motion (temporal context)
    3. Skeleton structure (instance context)
    """
    
    def __init__(self, hidden_size=256, num_joints=17, seq_len=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        
        self.spatial_encoder = SpatialContextEncoder(hidden_size, num_joints)
        self.temporal_encoder = TemporalContextEncoder(hidden_size, num_joints, seq_len)
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + 2 + 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),  # output coordinates
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, joint_features, predictions, motion_vectors, 
                occluded_mask, joint_trajectories):
        """
        Args:
            joint_features: (B, K, hidden_size) from backbone
            predictions: (B, K, 2) initial predictions
            motion_vectors: (B, K, 2) temporal motion history
            occluded_mask: (B, K) occlusion mask
            joint_trajectories: (B, T, K, 2) full trajectory
        
        Returns:
            reconstructed_coords: (B, K, 2)
            reconstruction_conf: (B, K, 1)
        """
        B, K, C = joint_features.shape
        
        # Spatial context
        spatial_ctx = self.spatial_encoder(joint_features)
        
        # Temporal context
        temporal_ctx = self.temporal_encoder(joint_trajectories)
        
        # Fuse all information
        fused = torch.cat([
            spatial_ctx,                    # (B, K, hidden)
            predictions,                    # (B, K, 2)
            temporal_ctx,                   # (B, K, 2)
        ], dim=-1)  # (B, K, hidden + 4)
        
        # Reconstruct coordinates
        reconstructed = self.fusion(fused)  # (B, K, 2)
        recon_conf = self.confidence_head(spatial_ctx)  # (B, K, 1)
        
        # Only apply reconstruction where occluded
        occluded_expanded = rearrange(occluded_mask, 'b k -> b k 1')
        final_coords = predictions * (1 - occluded_expanded) + reconstructed * occluded_expanded
        
        # Blend confidences: use high confidence for visible, lower for reconstructed
        final_conf = 0.7 * recon_conf * occluded_expanded + (1 - occluded_expanded)
        
        return final_coords, final_conf


class OcclusionAwarePoseReconstruction(nn.Module):
    """
    Complete OAPR module: detect + reconstruct occluded joints.
    
    This is the PRIMARY CONTRIBUTION that makes the paper novel.
    """
    
    def __init__(self, hidden_size=256, num_joints=17, seq_len=7, 
                 occlusion_threshold=0.5):
        super().__init__()
        
        self.detector = OcclusionDetector(hidden_size, num_joints, occlusion_threshold)
        self.reconstructor = PoseReconstructor(hidden_size, num_joints, seq_len)
    
    def forward(self, joint_features, predictions, confidence, joint_trajectories):
        """
        Args:
            joint_features: (B, K, hidden_size) from spatiotemporal backbone
            predictions: (B, K, 3) initial predictions (x, y, conf)
            confidence: (B, K, 1) uncertainty from backbone
            joint_trajectories: (B, T, K, 2) full video sequence
        
        Returns:
            refined_coords: (B, K, 2)
            refined_conf: (B, K, 1)
            occlusion_mask: (B, K) for interpretability
        """
        pred_coords = predictions[:, :, :2]
        
        # Detect occlusions
        occluded_mask, occlusion_score = self.detector(pred_coords, confidence)
        
        # Reconstruct occluded joints
        reconstructed_coords, reconstructed_conf = self.reconstructor(
            joint_features, pred_coords, 
            None,  # motion vectors computed internally
            occluded_mask, 
            joint_trajectories
        )
        
        return {
            'coordinates': reconstructed_coords,      # (B, K, 2)
            'confidence': reconstructed_conf,          # (B, K, 1)
            'occlusion_mask': occluded_mask,           # (B, K)
            'occlusion_score': occlusion_score,        # (B, K)
        }


def build_oapr_module(cfg):
    """Factory function."""
    model_cfg = cfg.get('model', {})
    return OcclusionAwarePoseReconstruction(
        hidden_size=model_cfg.get('hidden_size', 256),
        num_joints=model_cfg.get('num_keypoints', 17),
        seq_len=model_cfg.get('seq_len', 7),
        occlusion_threshold=model_cfg.get('occlusion_threshold', 0.5),
    )
