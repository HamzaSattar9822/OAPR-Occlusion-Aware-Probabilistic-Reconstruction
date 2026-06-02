#!/bin/bash
# download_crowdpose.sh — Download CrowdPose dataset
# Total size: ~3.4GB

set -e

DATA_DIR="./data/crowdpose"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  Downloading CrowdPose Dataset"
echo "============================================"
echo "Target directory: $DATA_DIR"
echo ""
echo "NOTE: CrowdPose images must be downloaded from:"
echo "  https://drive.google.com/file/d/1VprytECcLtU4tKP32SYi_7oDRbw7yUTL/view"
echo "  (Google Drive — manual download required)"
echo ""
echo "Annotations (JSON) will be downloaded automatically below."
echo ""

# --- Annotations via GitHub ---
echo "[1/2] Downloading CrowdPose annotations from GitHub..."
if [ ! -f "$DATA_DIR/annotations/crowdpose_train.json" ]; then
    mkdir -p "$DATA_DIR/annotations"
    BASE_URL="https://raw.githubusercontent.com/Jeff-sjtu/CrowdPose/master/crowdpose-api/data"
    wget -c "$BASE_URL/crowdpose_train.json"    -O "$DATA_DIR/annotations/crowdpose_train.json"
    wget -c "$BASE_URL/crowdpose_val.json"      -O "$DATA_DIR/annotations/crowdpose_val.json"
    wget -c "$BASE_URL/crowdpose_test.json"     -O "$DATA_DIR/annotations/crowdpose_test.json"
    wget -c "$BASE_URL/crowdpose_trainval.json" -O "$DATA_DIR/annotations/crowdpose_trainval.json"
    echo "  ✅ Annotations downloaded."
else
    echo "  ✅ Annotations already present. Skipping."
fi

# --- Images placeholder ---
echo "[2/2] Checking for images..."
if [ ! -d "$DATA_DIR/images" ] || [ -z "$(ls -A $DATA_DIR/images 2>/dev/null)" ]; then
    mkdir -p "$DATA_DIR/images"
    echo ""
    echo "  ⚠️  Images not found. Please:"
    echo "  1. Download images.zip from Google Drive:"
    echo "     https://drive.google.com/file/d/1VprytECcLtU4tKP32SYi_7oDRbw7yUTL/view"
    echo "  2. Place and extract into: $DATA_DIR/images/"
    echo ""
    echo "  After extraction, structure should be:"
    echo "    data/crowdpose/images/*.jpg  (~20,000 images)"
else
    IMG_COUNT=$(ls "$DATA_DIR/images/" | wc -l)
    echo "  ✅ Found $IMG_COUNT images in $DATA_DIR/images/"
fi

echo ""
echo "✅ CrowdPose setup at: $DATA_DIR"
echo ""
echo "Expected final structure:"
echo "  data/crowdpose/"
echo "  ├── annotations/"
echo "  │   ├── crowdpose_train.json"
echo "  │   ├── crowdpose_val.json"
echo "  │   ├── crowdpose_test.json"
echo "  │   └── crowdpose_trainval.json"
echo "  └── images/"
echo "      └── *.jpg  (~20k images)"
