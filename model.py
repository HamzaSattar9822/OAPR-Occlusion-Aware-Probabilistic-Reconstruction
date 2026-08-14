"""
model.py — Plug-and-play OAPR architecture (single-image, article pipeline).

    Image crop (B, 3, 256, 192)
        → HRNet-W32
        → Hybrid Mamba–Transformer joint encoder (K joints)
        → confidence head
        → occlusion gate (τ)
        → GCN skeleton-graph reconstruction
        → β-blend fusion
        → keypoints (B, K, 2), confidence (B, K, 1)

Usage:
    from model import OAPR, build_oapr, load_weights, describe_components

    model = build_oapr(pretrained=False)
    load_weights(model, "path/to/weights.pth")
    print(describe_components(model))
    out = model(images)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.mamba_backbone import MAMBA_AVAILABLE  # noqa: E402
from src.models.oapr_framework import (  # noqa: E402
    OAPRFramework,
    build_oapr_framework,
)

OAPR = OAPRFramework
build_oapr_from_config = build_oapr_framework

__all__ = [
    "OAPR",
    "OAPRFramework",
    "MAMBA_AVAILABLE",
    "build_oapr",
    "build_oapr_from_config",
    "load_weights",
    "describe_components",
]


def build_oapr(
    num_keypoints: int = 17,
    hidden_size: int = 256,
    num_heads: int = 8,
    num_spatial_layers: int = 4,
    use_mamba: bool = True,
    occlusion_threshold: float = 0.5,
    fusion_beta: float = 0.5,
    use_gcn: bool = True,
    use_confidence_gate: bool = True,
    hrnet_version: str = "w32",
    pretrained: bool = False,
    image_size=(192, 256),
    heatmap_size=(64, 48),
    use_dark: bool = True,
    loss_type: str = "cauchy_mixture",
    **deprecated_kwargs,
) -> OAPRFramework:
    """Build the article OAPR stack (ignores legacy seq_len / use_lstm kwargs)."""
    deprecated_kwargs.pop("seq_len", None)
    deprecated_kwargs.pop("use_lstm", None)
    if "confidence_beta" in deprecated_kwargs:
        fusion_beta = float(deprecated_kwargs.pop("confidence_beta"))

    return OAPRFramework(
        num_keypoints=num_keypoints,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_spatial_layers=num_spatial_layers,
        use_mamba=use_mamba,
        occlusion_threshold=occlusion_threshold,
        fusion_beta=fusion_beta,
        loss_type=loss_type,
        hrnet_version=hrnet_version,
        pretrained=pretrained,
        heatmap_size=heatmap_size,
        image_size=image_size,
        use_dark=use_dark,
        use_gcn=use_gcn,
        use_confidence_gate=use_confidence_gate,
    )


def describe_components(model: OAPRFramework) -> str:
    use_mamba = bool(getattr(model.backbone, "use_mamba", False))
    if use_mamba and MAMBA_AVAILABLE:
        joint_path = "Mamba (mamba_ssm) over K joints"
    else:
        joint_path = "attention fallback (JointAttentionFallback) over K joints"
        if not MAMBA_AVAILABLE:
            joint_path += " — mamba_ssm not installed"

    occ = model.occlusion_module
    return "\n".join(
        [
            "OAPR active components (single-image):",
            "  backbone          : HRNet-W32 → DARK + per-joint features",
            f"  joint encoder     : {joint_path} → Transformer",
            "  confidence head   : ON",
            f"  occlusion gate    : {'ON' if getattr(occ, 'use_confidence_gate', True) else 'OFF'}"
            f"  (τ={getattr(model, 'occlusion_threshold', '?')})",
            f"  GCN reconstruction: {'ON' if getattr(occ, 'use_gcn', True) else 'OFF'}",
            f"  β-blend fusion    : β={getattr(model, 'fusion_beta', getattr(model, 'confidence_beta', '?'))}",
            "  loss              : Cauchy-mixture (training)",
        ]
    )


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError(
        "Unrecognized checkpoint format — expected a state_dict or a dict "
        "with model_state_dict / state_dict."
    )


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state):
        return state
    n = len(prefix)
    return {k[n:] if k.startswith(prefix) else k: v for k, v in state.items()}


def load_weights(
    model: nn.Module,
    path: Union[str, Path],
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load matching tensors; warn on missing / unexpected / shape mismatch."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Weights not found: {path}")

    map_location = device if device is not None else "cpu"
    try:
        ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location=map_location)

    state = _strip_prefix(_strip_prefix(_extract_state_dict(ckpt), "module."), "model.")
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    unexpected = sorted(set(state.keys()) - model_keys)

    filtered = {}
    shape_mismatch = []
    for k, v in state.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            shape_mismatch.append(
                f"{k}: ckpt {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}"
            )
            continue
        filtered[k] = v

    incompatible = model.load_state_dict(filtered, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected_after = sorted(set(list(incompatible.unexpected_keys) + unexpected))

    report = {
        "path": str(path),
        "loaded_tensors": len(filtered),
        "missing_keys": missing,
        "unexpected_keys": unexpected_after,
        "shape_mismatch": shape_mismatch,
    }

    if verbose:
        print(f"[OAPR] Loaded {len(filtered)} / {len(model_keys)} tensors from {path}")
        if missing:
            preview = ", ".join(missing[:8])
            more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            print(f"[OAPR] WARNING: {len(missing)} missing keys (kept init): {preview}{more}")
        if unexpected_after:
            preview = ", ".join(unexpected_after[:8])
            more = (
                f" (+{len(unexpected_after) - 8} more)"
                if len(unexpected_after) > 8
                else ""
            )
            print(
                f"[OAPR] WARNING: {len(unexpected_after)} unexpected "
                f"checkpoint keys (ignored): {preview}{more}"
            )
        if shape_mismatch:
            print(f"[OAPR] WARNING: {len(shape_mismatch)} shape-mismatched keys skipped:")
            for line in shape_mismatch[:8]:
                print(f"         {line}")

    return report
