"""
Mamba-based backbone for spatiotemporal pose estimation.
Combines state-space modeling (Mamba) for temporal dependencies
with transformer attention for spatial refinement.

Reference:
- Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", ICLR 2024
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False
    msg = (
        "mamba_ssm not installed — temporal path will use "
        "attention fallback (TemporalTransformerFallback)."
    )
    logger.warning(msg)
    print(f"[OAPR] {msg}")


class TemporalMamba(nn.Module):
    """
    Mamba layer for temporal sequence modeling.
    Processes joint trajectories over time using state-space model.
    
    Args:
        hidden_size: dimension of state space
        num_joints: number of keypoints
        seq_len: sequence length (number of frames)
    """
    
    def __init__(self, hidden_size=256, num_joints=17, seq_len=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        self.seq_len = seq_len
        
        if Mamba is None:
            raise ImportError("mamba_ssm required. Install: pip install mamba-ssm causal-conv1d")
        
        # Per-timestep coordinate embedding → temporal Mamba over T
        self.joint_embed = nn.Linear(2, hidden_size)
        
        # Mamba layer processes temporal dependencies
        self.mamba = Mamba(
            d_model=hidden_size,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        
        # Project back to coordinate space
        self.coord_head = nn.Linear(hidden_size, 2)
        
        # Uncertainty (confidence) output
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, joint_trajectories):
        """
        Args:
            joint_trajectories: (B, T, K, 2) batch of temporal sequences
                B: batch size, T: time steps, K: keypoints, 2: x,y coordinates
        
        Returns:
            refined_coords: (B, K, 2)
            uncertainty: (B, K, 1)
            joint_features: (B, K, hidden_size) last-step temporal features
        """
        B, T, K, _ = joint_trajectories.shape
        
        # (B*K, T, 2) → embed → (B*K, T, hidden)
        traj = rearrange(joint_trajectories, 'b t k c -> (b k) t c')
        feat = self.joint_embed(traj)
        hidden_seq = self.mamba(feat)  # (B*K, T, hidden_size)
        hidden = hidden_seq[:, -1, :]  # (B*K, hidden_size)
        
        # Decode coordinates and uncertainty
        coords = self.coord_head(hidden)  # (B*K, 2)
        conf = self.uncertainty_head(hidden)  # (B*K, 1)
        
        # Reshape back
        coords = rearrange(coords, '(b k) c -> b k c', b=B, k=K)
        conf = rearrange(conf, '(b k) u -> b k u', b=B, k=K)
        joint_features = rearrange(hidden, '(b k) c -> b k c', b=B, k=K)
        
        return coords, conf, joint_features


class SpatialTransformer(nn.Module):
    """
    Multi-head self-attention for spatial refinement of joints.
    Allows joint-to-joint interaction for occlusion reasoning.
    
    Args:
        hidden_size: dimension
        num_heads: number of attention heads
        num_joints: number of keypoints
    """
    
    def __init__(self, hidden_size=256, num_heads=8, num_joints=17):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_joints = num_joints
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(0.1)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(0.1),
        )
    
    def forward(self, joint_features):
        """
        Args:
            joint_features: (B, K, hidden_size)
        
        Returns:
            refined features: (B, K, hidden_size)
        """
        B, K, C = joint_features.shape
        
        # Self-attention
        x = joint_features
        x = self.norm1(x)
        
        qkv = self.qkv_proj(x).reshape(B, K, 3, self.num_heads, C // self.num_heads)
        qkv = rearrange(qkv, 'b k t h d -> t b h k d')
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, K, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Residual + FFN
        x = joint_features + x
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        
        return x


class HybridMambaTransformer(nn.Module):
    """
    Hybrid backbone combining:
    1. Temporal Mamba: captures long-range temporal dependencies
    2. Spatial Transformer: refines joint relationships
    3. Instance-aware modeling: per-person representations
    
    Designed for multi-human pose estimation with occlusion robustness.
    
    Args:
        num_keypoints: number of keypoints (17 for COCO, 14 for CrowdPose)
        seq_len: video sequence length (5-9 frames)
        hidden_size: model dimension
        num_heads: transformer attention heads
        num_layers: number of spatial transformer layers
        use_mamba: if False, uses temporal transformer fallback
    """
    
    def __init__(self, num_keypoints=17, seq_len=7, hidden_size=256,
                 num_heads=8, num_layers=3, use_mamba=True):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.use_mamba = bool(use_mamba) and MAMBA_AVAILABLE
        
        # Temporal modeling (Mamba optional — attention fallback if missing)
        if self.use_mamba:
            path_msg = f"temporal path ACTIVE: Mamba (seq_len={seq_len})"
            logger.info(path_msg)
            print(f"[OAPR] {path_msg}")
            self.temporal = TemporalMamba(hidden_size, num_keypoints, seq_len)
        else:
            reason = "use_mamba=False" if not use_mamba else "mamba_ssm unavailable"
            path_msg = (
                f"temporal path ACTIVE: attention fallback "
                f"(TemporalTransformerFallback; {reason}; seq_len={seq_len})"
            )
            logger.info(path_msg)
            print(f"[OAPR] {path_msg}")
            self.temporal = TemporalTransformerFallback(
                hidden_size, num_keypoints, seq_len, num_heads
            )
        
        # Spatial refinement layers
        self.spatial_layers = nn.ModuleList([
            SpatialTransformer(hidden_size, num_heads, num_keypoints)
            for _ in range(num_layers)
        ])
        
        # Fuse optional HRNet per-joint features with temporal hidden state
        self.feature_fuse = nn.Linear(hidden_size * 2, hidden_size)
        
        # Final coordinate regression head
        self.final_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),  # x, y, confidence
        )
    
    def forward(self, video_clip, joint_init_features=None):
        """
        Args:
            video_clip: (B, T, K, 2) spatiotemporal keypoint sequences
                B: batch, T: frames, K: joints, 2: coordinates
            joint_init_features: optional (B, K, hidden_size) from image encoder
        
        Returns:
            refined_keypoints: (B, K, 3) x, y, confidence per joint
            joint_features: (B, K, hidden_size) spatial-refined hidden features
        """
        # Temporal modeling → real per-joint hidden features (not random noise)
        _, _, temporal_hidden = self.temporal(video_clip)  # (B, K, hidden)
        
        if joint_init_features is not None:
            joint_feat = self.feature_fuse(
                torch.cat([temporal_hidden, joint_init_features], dim=-1)
            )
        else:
            joint_feat = temporal_hidden
        
        # Spatial refinement
        for spatial_layer in self.spatial_layers:
            joint_feat = spatial_layer(joint_feat)
        
        # Final head
        output = self.final_head(joint_feat)  # (B, K, 3)
        
        return output, joint_feat


class TemporalTransformerFallback(nn.Module):
    """
    Fallback when Mamba is not available.
    Uses multi-head attention over temporal dimension.
    """
    
    def __init__(self, hidden_size=256, num_joints=17, seq_len=7, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        self.seq_len = seq_len
        
        self.embed = nn.Linear(2, hidden_size)
        self.temporal_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=0.1
        )
        self.coord_head = nn.Linear(hidden_size, 2)
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, joint_trajectories):
        """
        Args:
            joint_trajectories: (B, T, K, 2)
        
        Returns:
            refined_coords: (B, K, 2)
            uncertainty: (B, K, 1)
            joint_features: (B, K, hidden_size)
        """
        B, T, K, _ = joint_trajectories.shape
        
        # Reshape: (B*K, T, 2) -> (B*K, T, hidden)
        traj = rearrange(joint_trajectories, 'b t k c -> (b k) t c')
        feat = self.embed(traj)
        
        # Self-attention over temporal dimension
        attn_out, _ = self.temporal_attn(feat, feat, feat)
        
        # Take last time step
        last_feat = attn_out[:, -1, :]  # (B*K, hidden)
        
        coords = self.coord_head(last_feat)
        conf = self.conf_head(last_feat)
        
        coords = rearrange(coords, '(b k) c -> b k c', b=B, k=K)
        conf = rearrange(conf, '(b k) u -> b k u', b=B, k=K)
        joint_features = rearrange(last_feat, '(b k) c -> b k c', b=B, k=K)
        
        return coords, conf, joint_features


def build_spatiotemporal_model(cfg):
    """Factory function to build hybrid model."""
    model_cfg = cfg.get('model', {})
    return HybridMambaTransformer(
        num_keypoints=model_cfg.get('num_keypoints', 17),
        seq_len=model_cfg.get('seq_len', 7),
        hidden_size=model_cfg.get('hidden_size', 256),
        num_heads=model_cfg.get('num_heads', 8),
        num_layers=model_cfg.get('num_spatial_layers', 3),
        use_mamba=model_cfg.get('use_mamba', True),
    )
