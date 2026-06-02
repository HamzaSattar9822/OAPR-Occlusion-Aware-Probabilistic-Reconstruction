#!/bin/bash
# download_coco.sh — Download COCO 2017 Keypoints dataset
# Total size: ~20GB (images) + ~240MB (annotations)

set -e

DATA_DIR="./data/coco"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  Downloading COCO 2017 Keypoints Dataset"
echo "============================================"
echo "Target directory: $DATA_DIR"
echo ""

# --- Annotations ---
echo "[1/4] Downloading annotations..."
if [ ! -f "$DATA_DIR/annotations/person_keypoints_train2017.json" ]; then
    wget -c "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
         -O "$DATA_DIR/annotations_trainval2017.zip"
    unzip -q "$DATA_DIR/annotations_trainval2017.zip" -d "$DATA_DIR/"
    rm "$DATA_DIR/annotations_trainval2017.zip"
    echo "  ✅ Annotations extracted."
else
    echo "  ✅ Annotations already present. Skipping."
fi

# --- Train images ---
echo "[2/4] Downloading train2017 images (~18GB, this will take a while)..."
if [ ! -d "$DATA_DIR/images/train2017" ]; then
    wget -c "http://images.cocodataset.org/zips/train2017.zip" \
         -O "$DATA_DIR/train2017.zip"
    unzip -q "$DATA_DIR/train2017.zip" -d "$DATA_DIR/images/"
    rm "$DATA_DIR/train2017.zip"
    echo "  ✅ train2017 images extracted."
else
    echo "  ✅ train2017 images already present. Skipping."
fi

# --- Val images ---
echo "[3/4] Downloading val2017 images (~1GB)..."
if [ ! -d "$DATA_DIR/images/val2017" ]; then
    wget -c "http://images.cocodataset.org/zips/val2017.zip" \
         -O "$DATA_DIR/val2017.zip"
    unzip -q "$DATA_DIR/val2017.zip" -d "$DATA_DIR/images/"
    rm "$DATA_DIR/val2017.zip"
    echo "  ✅ val2017 images extracted."
else
    echo "  ✅ val2017 images already present. Skipping."
fi

# --- Verify ---
echo "[4/4] Verifying dataset..."
TRAIN_COUNT=$(ls "$DATA_DIR/images/train2017/" | wc -l)
VAL_COUNT=$(ls "$DATA_DIR/images/val2017/" | wc -l)
echo "  train2017: $TRAIN_COUNT images (expected ~118,287)"
echo "  val2017:   $VAL_COUNT images (expected ~5,000)"

echo ""
echo "✅ COCO 2017 Keypoints dataset ready at: $DATA_DIR"
echo ""
echo "Expected structure:"
echo "  data/coco/"
echo "  ├── annotations/"
echo "  │   ├── person_keypoints_train2017.json"
echo "  │   └── person_keypoints_val2017.json"
echo "  └── images/"
echo "      ├── train2017/  (~118k images)"
echo "      └── val2017/    (~5k images)"
