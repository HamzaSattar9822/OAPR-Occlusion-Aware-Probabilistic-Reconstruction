# ✓ FIX FOR: ModuleNotFoundError: No module named 'cv2'

## The Problem
```
ERROR: Could not install packages due to an OSError: 
[Errno 1] Operation not permitted: '/Users/saad/Library/Python/3.11/lib/python/site-packages/cv2'
```

This is a **permission issue with system Python**. The solution is to use a **virtual environment**.

---

## SOLUTION: Use Python Virtual Environment

### Step 1: Create Virtual Environment (One Time)
```bash
cd /Users/saad/Downloads/oapr_pose
python3 -m venv venv
```

### Step 2: Activate Virtual Environment
```bash
source /Users/saad/Downloads/oapr_pose/venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3: Install Dependencies in Virtual Environment
```bash
cd /Users/saad/Downloads/oapr_pose
pip3 install -r requirements.txt
```

Or just the essentials:
```bash
pip3 install opencv-python torch torchvision numpy scipy matplotlib pyyaml tqdm tensorboard einops
```

### Step 4: Run Your Test
```bash
python3 test_remote_dataset.py --epochs 5
```

---

## COMPLETE COMMAND (Copy & Paste All at Once)

```bash
cd /Users/saad/Downloads/oapr_pose && \
python3 -m venv venv && \
source venv/bin/activate && \
pip3 install -r requirements.txt && \
python3 test_remote_dataset.py --epochs 5
```

---

## Why Virtual Environment?

Without it:
- ❌ Installing to system Python (permission denied)
- ❌ Conflicts with other projects
- ❌ Need admin rights

With virtual environment:
- ✓ Isolated Python environment
- ✓ Can install without permissions
- ✓ No conflicts with system packages
- ✓ Easy to remove (just delete `venv` folder)

---

## What to Expect

When you run the complete command above, you'll see:

```
Creating virtual environment... done
Collecting torch...
Collecting opencv-python...
[... installation progress ...]
Successfully installed opencv-python torch torchvision ...
[Training output...]
```

Then your test will run successfully!

---

## If You Keep Getting Permission Errors

Try installing with `--user` flag (alternative, not ideal):

```bash
pip3 install --user opencv-python torch torchvision numpy scipy
```

But **virtual environment is the proper solution**.

---

## Quick Summary

**The issue:** System Python permissions
**The fix:** Use virtual environment
**The command:**
```bash
cd /Users/saad/Downloads/oapr_pose && \
python3 -m venv venv && \
source venv/bin/activate && \
pip3 install -r requirements.txt
```

After this works, future runs just need:
```bash
source /Users/saad/Downloads/oapr_pose/venv/bin/activate
python3 test_remote_dataset.py --epochs 5
```

---

## Verify Virtual Environment is Working

After activation, you should see:
```bash
(venv) $ python3 -c "import cv2; print('✓ OpenCV installed')"
✓ OpenCV installed
```

If you see this, you're ready to run the training!
