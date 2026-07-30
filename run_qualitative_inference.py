#!/usr/bin/env python3
"""Run real multi-person pose inference on the qualitative COCO subset.

This uses torchvision's COCO-pretrained Keypoint R-CNN (ResNet50-FPN) to produce
genuine multi-person 17-keypoint skeleton predictions on the 100 images selected
by ``select_qualitative_subset.py``. It is a real, published model used here as a
reference baseline for qualitative inspection (NOT the OAPR model, which still
requires training).

For every input image it:
    * detects all persons above a score threshold
    * draws the COCO skeleton + joints for each detected person
    * saves the annotated image to the output directory
    * records the number of detected persons and per-person score

Outputs:
    * annotated JPGs in ``--out-dir``
    * ``qualitative_inference_results.csv`` summary
    * ``side_by_side`` originals+overlay panels (optional, --panels)

Usage:
    python run_qualitative_inference.py \
        --subset-dir coco_qualitative_subset \
        --manifest coco_qualitative_subset/qualitative_subset.csv \
        --out-dir outputs/qualitative_results \
        --score-thresh 0.7 --kp-thresh 2.0
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.pose_viz import draw_pose_with_neck


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset-dir", default="coco_qualitative_subset")
    p.add_argument("--manifest", default="coco_qualitative_subset/qualitative_subset.csv")
    p.add_argument("--out-dir", default="outputs/qualitative_results")
    p.add_argument("--score-thresh", type=float, default=0.7,
                   help="Minimum person detection score to draw")
    p.add_argument("--kp-thresh", type=float, default=2.0,
                   help="Minimum per-keypoint visibility logit to draw a joint")
    p.add_argument("--panels", action="store_true",
                   help="Also save side-by-side original|prediction panels")
    return p.parse_args()


def draw_pose(image, keypoints, kp_scores, kp_thresh):
    """Draw one person's skeleton. keypoints: (17,3) [x,y,vis], kp_scores: (17,)."""
    return draw_pose_with_neck(image, keypoints, kp_scores, threshold=kp_thresh)


def load_manifest(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    panels_dir = os.path.join(args.out_dir, "panels")
    if args.panels:
        os.makedirs(panels_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    print(f"Loading pretrained model: Keypoint R-CNN ResNet50-FPN ({weights})")
    model = keypointrcnn_resnet50_fpn(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    manifest = load_manifest(args.manifest)
    print(f"Images to process: {len(manifest)}")

    results = []
    for idx, row in enumerate(manifest):
        file_name = row["file_name"]
        gt_n = int(row["number_of_persons"])
        src = os.path.join(args.subset_dir, file_name)
        img_bgr = cv2.imread(src)
        if img_bgr is None:
            print(f"  [skip] could not read {src}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
        batch = [preprocess(tensor).to(device)]

        with torch.no_grad():
            out = model(batch)[0]

        scores = out["scores"].cpu().numpy()
        keep = scores >= args.score_thresh
        kps = out["keypoints"].cpu().numpy()[keep]          # (N,17,3)
        kp_scores = out["keypoints_scores"].cpu().numpy()[keep]  # (N,17)
        person_scores = scores[keep]

        annotated = img_bgr.copy()
        for person in range(kps.shape[0]):
            annotated = draw_pose(annotated, kps[person], kp_scores[person], args.kp_thresh)

        out_path = os.path.join(args.out_dir, f"pred_{file_name}")
        cv2.imwrite(out_path, annotated)

        if args.panels:
            panel = np.hstack([img_bgr, annotated])
            cv2.imwrite(os.path.join(panels_dir, f"panel_{file_name}"), panel)

        n_det = int(kps.shape[0])
        mean_score = float(person_scores.mean()) if n_det else 0.0
        results.append({
            "image_id": row["image_id"],
            "file_name": file_name,
            "gt_persons": gt_n,
            "detected_persons": n_det,
            "mean_person_score": round(mean_score, 4),
        })

        if (idx + 1) % 10 == 0:
            print(f"  processed {idx + 1}/{len(manifest)}")

    csv_path = os.path.join(args.out_dir, "qualitative_inference_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "file_name", "gt_persons",
                                          "detected_persons", "mean_person_score"])
        w.writeheader()
        for r in sorted(results, key=lambda x: -x["gt_persons"]):
            w.writerow(r)

    # Aggregate stats.
    total_gt = sum(r["gt_persons"] for r in results)
    total_det = sum(r["detected_persons"] for r in results)
    print(f"\nAnnotated images: {len(results)} -> {args.out_dir}")
    print(f"Summary CSV: {csv_path}")
    print(f"Total GT persons: {total_gt} | Total detected: {total_det} "
          f"({100.0 * total_det / max(total_gt, 1):.1f}% recall vs GT count)")


if __name__ == "__main__":
    main()
