#!/usr/bin/env python3
"""
test.py — Single-image OAPR pose overlay demo (inference only).

Example:
    python test.py --image path/to/person.jpg --weights path/to/best.pth

No evaluation metrics are computed or printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import MAMBA_AVAILABLE, build_oapr, describe_components, load_weights
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_affine_transform
from src.utils.pose_viz import draw_pose_with_neck


def _box_to_center_scale(w: float, h: float, image_size, pad: float = 1.25):
    """Full-frame person box → HRNet center/scale (pixel_std = 200)."""
    aspect = float(image_size[0]) / float(image_size[1])  # W / H
    bw, bh = float(w), float(h)
    if bw > aspect * bh:
        bh = bw / aspect
    elif bw < aspect * bh:
        bw = bh * aspect
    center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
    scale = np.array([bw / 200.0, bh / 200.0], dtype=np.float32) * pad
    return center, scale


def preprocess_image(bgr: np.ndarray, image_size=(192, 256)):
    """
    Warp full image into model crop and return tensor + center/scale for inverse map.

    image_size: (W, H) as in configs.
    """
    h, w = bgr.shape[:2]
    center, scale = _box_to_center_scale(w, h, image_size)
    trans = get_affine_transform(center, scale, 0, image_size)
    crop = cv2.warpAffine(
        bgr, trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR
    )
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean) / std
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)  # (1,3,H,W)
    return tensor, center, scale


def crop_to_image_coords(coords_crop: np.ndarray, center, scale, image_size):
    """Map keypoints from crop space back to original image pixels."""
    inv = get_affine_transform(center, scale, 0, image_size, inv=True)
    out = np.zeros_like(coords_crop)
    for i in range(coords_crop.shape[0]):
        pt = np.array([coords_crop[i, 0], coords_crop[i, 1], 1.0], dtype=np.float32)
        mapped = inv @ pt
        out[i, 0] = mapped[0]
        out[i, 1] = mapped[1]
    return out


def parse_args():
    p = argparse.ArgumentParser(description="OAPR single-image pose overlay demo")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--weights", required=True, help="Path to .pth checkpoint")
    p.add_argument(
        "--out",
        default=None,
        help="Output overlay path (default: <image_stem>_oapr_pose.png)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda | cpu | mps (default: auto)",
    )
    p.add_argument(
        "--conf-threshold",
        type=float,
        default=0.2,
        help="Min joint confidence to draw (visualization only)",
    )
    p.add_argument(
        "--no-mamba",
        action="store_true",
        help="Force joint-sequence attention fallback even if mamba_ssm is installed",
    )
    return p.parse_args()


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    image_path = Path(args.image)
    weights_path = Path(args.weights)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    if not weights_path.is_file():
        raise SystemExit(f"Weights not found: {weights_path}")

    out_path = Path(args.out) if args.out else image_path.with_name(
        f"{image_path.stem}_oapr_pose.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    image_size = (192, 256)  # (W, H)

    print("=" * 60)
    print("OAPR single-image inference demo")
    print("=" * 60)
    print(f"image   : {image_path}")
    print(f"weights : {weights_path}")
    print(f"device  : {device}")
    print(f"mamba_ssm package available: {MAMBA_AVAILABLE}")

    model = build_oapr(pretrained=False, use_mamba=not args.no_mamba)
    model = model.to(device).eval()
    load_weights(model, weights_path, device=device, verbose=True)
    print(describe_components(model))

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    tensor, center, scale = preprocess_image(bgr, image_size)
    tensor = tensor.to(device)

    with torch.no_grad():
        out = model(tensor)

    kps_crop = out["keypoints"][0].detach().cpu().numpy()
    conf = out["confidence"][0].detach().cpu().numpy().reshape(-1)
    kps_img = crop_to_image_coords(kps_crop, center, scale, image_size)

    overlay = draw_pose_with_neck(
        bgr, kps_img, conf, threshold=args.conf_threshold
    )
    if not cv2.imwrite(str(out_path), overlay):
        raise SystemExit(f"Failed to write overlay: {out_path}")

    print(f"saved   : {out_path}")
    print(f"joints  : {kps_img.shape[0]}  (conf mean={float(conf.mean()):.3f})")
    print("done (inference only — no metrics).")


if __name__ == "__main__":
    main()
