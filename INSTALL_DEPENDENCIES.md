# Fix: Missing Dependencies Error

## Error You Got
```
ModuleNotFoundError: No module named 'cv2'
```

## Quick Fix - Install Missing Packages

### Option 1: Install Only Essential Packages (FASTEST)
```bash
pip3 install opencv-python pyyaml scipy matplotlib tqdm tensorboard
```

Then run:
```bash
cd /Users/saad/Downloads/oapr_pose && python3 test_remote_dataset.py --epochs 5
```

### Option 2: Install Everything from requirements.txt
```bash
cd /Users/saad/Downloads/oapr_pose && pip3 install -r requirements.txt
```

This installs all dependencies including optional ones.

### Option 3: Install Minimal Set (QUICKEST)
```bash
pip3 install opencv-python numpy scipy matplotlib
```

---

## What Was Missing

You installed:
- torch
- torchvision  
- numpy

But the code also needs:
- **opencv-python** (cv2) ← This was missing!
- scipy
- matplotlib
- pyyaml
- tqdm
- tensorboard

---

## Complete Installation Command (Copy & Paste)

```bash
pip3 install torch torchvision torchaudio opencv-python numpy scipy matplotlib pyyaml tqdm tensorboard einops pillow
```

Then run your command:
```bash
cd /Users/saad/Downloads/oapr_pose && python3 test_remote_dataset.py --epochs 5
```

---

## Why This Happened

The file `src/data/coco_dataset.py` imports `cv2` (OpenCV) for image processing:

```python
import cv2  # <- Line that failed
```

When you ran the code, Python couldn't find `cv2` because it wasn't installed.

---

## Solution Summary

**Run this ONE command:**

```bash
pip3 install opencv-python && cd /Users/saad/Downloads/oapr_pose && python3 test_remote_dataset.py --epochs 5
```

**OR for complete setup:**

```bash
cd /Users/saad/Downloads/oapr_pose && pip3 install -r requirements.txt && python3 test_remote_dataset.py --epochs 5
```

---

## Monitor Installation

If installing, you'll see:
```
Collecting opencv-python
Downloading opencv-python-4.8.0...
Installing collected packages: opencv-python
Successfully installed opencv-python
```

Once done, your test will run!

---

## If Installation Is Still Running

The `pip install -r requirements.txt` command might still be running in the background. 

**Quick workaround - install just what you need:**

```bash
pip3 install opencv-python --force-reinstall
```

Then try running the test again.
