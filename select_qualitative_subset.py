#!/usr/bin/env python3
"""Select 100 representative COCO val2017 images for qualitative pose evaluation.

The images are stratified by crowd density so the qualitative figures cover the
full range from single-person scenes to heavily crowded scenes:

    * 25 images with exactly one person
    * 25 images with 2-3 persons
    * 25 images with 4-6 persons
    * 25 images with more than 6 persons

Selected images are copied into an output directory and a CSV manifest is
written with image_id, file_name and number_of_persons.

Usage:
    python select_qualitative_subset.py \
        --ann data/coco/annotations/person_keypoints_val2017.json \
        --img-dir data/coco/images/val2017 \
        --out-dir coco_qualitative_subset \
        --csv coco_qualitative_subset/qualitative_subset.csv \
        --seed 42
"""

import argparse
import csv
import os
import random
import shutil
from collections import defaultdict

from pycocotools.coco import COCO


# (label, min_persons, max_persons, count_to_sample); max_persons=None means open-ended.
CATEGORIES = [
    ("1_person", 1, 1, 25),
    ("2-3_persons", 2, 3, 25),
    ("4-6_persons", 4, 6, 25),
    ("7+_persons", 7, None, 25),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann", default="data/coco/annotations/person_keypoints_val2017.json",
                        help="Path to COCO person_keypoints_val2017.json")
    parser.add_argument("--img-dir", default="data/coco/images/val2017",
                        help="Directory containing the val2017 images")
    parser.add_argument("--out-dir", default="coco_qualitative_subset",
                        help="Directory to copy selected images into")
    parser.add_argument("--csv", default=None,
                        help="Path to the CSV manifest (default: <out-dir>/qualitative_subset.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    return parser.parse_args()


def count_persons_per_image(coco):
    """Return {image_id: number_of_person_instances} for the person category."""
    person_cat_ids = coco.getCatIds(catNms=["person"])
    person_counts = defaultdict(int)
    for ann in coco.loadAnns(coco.getAnnIds(catIds=person_cat_ids, iscrowd=False)):
        person_counts[ann["image_id"]] += 1
    return person_counts


def bucket_images(coco, person_counts, img_dir):
    """Group image ids into the configured crowd-density buckets.

    Only images that exist on disk are kept so the copy step cannot fail.
    """
    buckets = {label: [] for label, *_ in CATEGORIES}
    for image_id, n_persons in person_counts.items():
        file_name = coco.loadImgs(image_id)[0]["file_name"]
        if not os.path.isfile(os.path.join(img_dir, file_name)):
            continue
        for label, lo, hi, _ in CATEGORIES:
            if n_persons >= lo and (hi is None or n_persons <= hi):
                buckets[label].append((image_id, file_name, n_persons))
                break
    return buckets


def main():
    args = parse_args()
    random.seed(args.seed)

    csv_path = args.csv or os.path.join(args.out_dir, "qualitative_subset.csv")

    print(f"Loading annotations: {args.ann}")
    coco = COCO(args.ann)

    person_counts = count_persons_per_image(coco)
    buckets = bucket_images(coco, person_counts, args.img_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    selected = []
    for label, lo, hi, want in CATEGORIES:
        available = buckets[label]
        if len(available) < want:
            print(f"WARNING: category '{label}' has only {len(available)} images "
                  f"available (requested {want}); selecting all of them.")
            chosen = list(available)
        else:
            chosen = random.sample(available, want)
        print(f"  {label:>12}: selected {len(chosen)} / {len(available)} available")
        for image_id, file_name, n_persons in chosen:
            selected.append((image_id, file_name, n_persons, label))

    # Copy images.
    copied = 0
    for image_id, file_name, _n, _label in selected:
        src = os.path.join(args.img_dir, file_name)
        dst = os.path.join(args.out_dir, file_name)
        shutil.copy2(src, dst)
        copied += 1

    # Write CSV manifest.
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "file_name", "number_of_persons"])
        for image_id, file_name, n_persons, _label in sorted(selected, key=lambda r: -r[2]):
            writer.writerow([image_id, file_name, n_persons])

    print(f"\nCopied {copied} images into: {args.out_dir}")
    print(f"Wrote manifest: {csv_path}")
    print(f"Total selected: {len(selected)} images")


if __name__ == "__main__":
    main()
