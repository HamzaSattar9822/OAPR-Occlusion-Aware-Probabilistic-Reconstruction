#!/bin/bash
# run_remote_test.sh
# Quick script to run remote dataset testing

echo "=================================================="
echo "OAPR Framework - Remote Dataset Testing"
echo "=================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found"
    exit 1
fi

# Change to project directory
cd "$(dirname "$0")"

echo "Testing OAPR framework with remote CrowdPose dataset..."
echo ""

# Run with different configurations
echo "Configuration:"
echo "  - Dataset: CrowdPose (remote from GitHub)"
echo "  - Epochs: 5"
echo "  - Batch Size: 4"
echo "  - Samples: 50"
echo "  - Device: CPU (or CUDA if available)"
echo ""
echo "Starting test..."
echo "=================================================="
echo ""

# Run the test
python3 test_remote_dataset.py \
    --epochs 5 \
    --batch_size 4 \
    --dataset crowdpose \
    --num_samples 50 \
    --lr 0.001 \
    --device cpu

echo ""
echo "=================================================="
echo "Test completed!"
echo "=================================================="
echo ""
echo "For custom testing, use:"
echo "  python3 test_remote_dataset.py --help"
echo ""
echo "Examples:"
echo ""
echo "1. Faster test (20 samples, 3 epochs):"
echo "   python3 test_remote_dataset.py --epochs 3 --num_samples 20"
echo ""
echo "2. Thorough test (100 samples, 10 epochs):"
echo "   python3 test_remote_dataset.py --epochs 10 --num_samples 100"
echo ""
echo "3. With GPU (if CUDA available):"
echo "   python3 test_remote_dataset.py --device cuda --batch_size 16"
echo ""
echo "See REMOTE_TESTING.md for detailed documentation."
