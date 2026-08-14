"""
Hybrid Mamba–Transformer joint encoder (single-image).

Mamba (or attention fallback) operates over the K=17 joint token sequence of
one person crop — not over time. A Transformer then refines joint–joint
relations. Matches article Section 3.4.

Reference:
- Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
  ICLR 2024
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False
    msg = (
        "mamba_ssm not installed — joint-sequence path will use "
        "attention fallback (JointAttentionFallback)."
    )
    logger.warning(msg)
    print(f"[OAPR] {msg}")


class JointMambaEncoder(nn.Module):
    """Mamba SSM over the joint token sequence (length = K)."""

    def __init__(self, hidden_size: int = 256, num_joints: int = 17):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        if Mamba is None:
            raise ImportError(
                "mamba_ssm required. Install: pip install mamba-ssm causal-conv1d"
            )
        self.mamba = Mamba(
            d_model=hidden_size,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_features: (B, K, C)
        Returns:
            (B, K, C) joint-sequence encoded features
        """
        x = self.norm(joint_features)
        return joint_features + self.mamba(x)


class JointAttentionFallback(nn.Module):
    """Self-attention over joints when mamba_ssm is unavailable."""

    def __init__(self, hidden_size: int = 256, num_joints: int = 17, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_features: (B, K, C)
        Returns:
            (B, K, C)
        """
        x = self.norm(joint_features)
        out, _ = self.attn(x, x, x)
        return joint_features + out


class JointTransformerBlock(nn.Module):
    """Transformer block over joint tokens."""

    def __init__(self, hidden_size: int = 256, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(0.1)
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(0.1),
        )

    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        B, K, C = joint_features.shape
        x = self.norm1(joint_features)
        qkv = self.qkv_proj(x).reshape(B, K, 3, self.num_heads, C // self.num_heads)
        qkv = rearrange(qkv, "b k t h d -> t b h k d")
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, K, C)
        x = joint_features + self.proj_drop(self.proj(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HybridMambaTransformer(nn.Module):
    """
    Hybrid Mamba–Transformer joint encoder.

    Pipeline:
        joint features (B, K, C)
            → Mamba (or attention fallback) over K joints
            → Transformer layers over K joints
            → coordinate head + confidence head
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        use_mamba: bool = True,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.hidden_size = hidden_size
        self.use_mamba = bool(use_mamba) and MAMBA_AVAILABLE

        if self.use_mamba:
            path_msg = (
                f"joint-sequence path ACTIVE: Mamba "
                f"(num_joints={num_keypoints})"
            )
            logger.info(path_msg)
            print(f"[OAPR] {path_msg}")
            self.joint_encoder = JointMambaEncoder(hidden_size, num_keypoints)
        else:
            reason = "use_mamba=False" if not use_mamba else "mamba_ssm unavailable"
            path_msg = (
                f"joint-sequence path ACTIVE: attention fallback "
                f"(JointAttentionFallback; {reason}; num_joints={num_keypoints})"
            )
            logger.info(path_msg)
            print(f"[OAPR] {path_msg}")
            self.joint_encoder = JointAttentionFallback(
                hidden_size, num_keypoints, num_heads
            )

        self.transformer_layers = nn.ModuleList(
            [JointTransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )

        self.coord_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, joint_features: torch.Tensor):
        """
        Args:
            joint_features: (B, K, hidden_size) from HRNet sampling + projection

        Returns:
            coords: (B, K, 2)
            confidence: (B, K, 1)
            joint_features: (B, K, hidden_size) refined tokens
        """
        x = self.joint_encoder(joint_features)
        for block in self.transformer_layers:
            x = block(x)
        coords = self.coord_head(x)
        confidence = self.confidence_head(x)
        return coords, confidence, x


def build_joint_encoder(cfg) -> HybridMambaTransformer:
    """Factory: Hybrid Mamba–Transformer joint encoder from config."""
    model_cfg = cfg.get("model", {})
    return HybridMambaTransformer(
        num_keypoints=model_cfg.get("num_keypoints", 17),
        hidden_size=model_cfg.get("hidden_size", 256),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_spatial_layers", 4),
        use_mamba=model_cfg.get("use_mamba", True),
    )


# Back-compat alias used by older imports
build_spatiotemporal_model = build_joint_encoder
