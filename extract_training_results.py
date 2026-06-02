#!/usr/bin/env python3
"""Parse training log and write TRAINING_10EPOCH_RESULTS.txt summary."""

import re
import json
import glob
import os
from datetime import datetime

LOG_CANDIDATES = [
    'training_10epochs.log',
    'logs/hrnet_baseline/train_*.log',
]


def parse_log(text):
    epochs = []
    epoch_re = re.compile(
        r'Epoch \[(\d+)\]\[(\d+)/(\d+)\].*Loss: ([\d.]+).*Acc: ([\d.]+)'
    )
    ap_re = re.compile(r'\s+(AP|AP50|AP75|APm|APl)\s*:\s*([\d.]+)')

    current_epoch = None
    for line in text.splitlines():
        m = re.search(r'Epoch (\d+)/10\s+\|\s+LR:', line)
        if m:
            current_epoch = int(m.group(1))

        em = epoch_re.search(line)
        if em:
            epochs.append({
                'epoch': int(em.group(1)) + 1,
                'batch': int(em.group(2)),
                'total_batches': int(em.group(3)),
                'loss': float(em.group(4)),
                'acc': float(em.group(5)),
            })

    ap_blocks = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if 'Evaluation Results' in line or 'AP' in line:
            metrics = {}
            for j in range(i, min(i + 20, len(lines))):
                for name, val in ap_re.findall(lines[j]):
                    metrics[name] = float(val)
            if metrics:
                ap_blocks.append(metrics)

    best_ap = 0.0
    m = re.search(r'Best AP:\s*([\d.]+)', text)
    if m:
        best_ap = float(m.group(1))
    m2 = re.search(r'New best AP:\s*([\d.]+)', text)
    if m2:
        best_ap = max(best_ap, float(m2.group(1)))

    return epochs, ap_blocks, best_ap


def main():
    log_path = None
    for cand in LOG_CANDIDATES:
        if '*' in cand:
            files = sorted(glob.glob(cand))
            if files:
                log_path = files[-1]
                break
        elif os.path.isfile(cand):
            log_path = cand
            break

    if not log_path:
        print('No training log found.')
        return

    with open(log_path) as f:
        text = f.read()

    epochs, ap_blocks, best_ap = parse_log(text)

    ckpt_dir = 'checkpoints/hrnet_baseline'
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth')))

    out = {
        'generated': datetime.now().isoformat(),
        'log_file': log_path,
        'checkpoints': ckpts,
        'best_ap': best_ap,
        'ap_evaluations': ap_blocks,
        'last_batch_metrics': epochs[-5:] if epochs else [],
    }

    txt_path = 'TRAINING_10EPOCH_RESULTS.txt'
    with open(txt_path, 'w') as f:
        f.write('HRNet-W32 Baseline — 10 Epoch Training Results\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Generated: {out["generated"]}\n')
        f.write(f'Log: {log_path}\n')
        f.write('Note: Training used 5%% COCO train subset (partial download) on CPU.\n')
        f.write('      Val set: full COCO val person instances with available images.\n\n')
        f.write(f'Best AP: {best_ap:.4f}\n\n')
        if ckpts:
            f.write('Checkpoints:\n')
            for c in ckpts:
                f.write(f'  - {c}\n')
            f.write('\n')
        if ap_blocks:
            f.write('Validation AP (per eval):\n')
            for i, m in enumerate(ap_blocks, 1):
                f.write(f'  Eval #{i}: {m}\n')
            f.write('\n')
        if epochs:
            f.write('Recent training batches:\n')
            for e in epochs[-10:]:
                f.write(
                    f"  Epoch {e['epoch']} batch {e['batch']}/{e['total_batches']} "
                    f"loss={e['loss']:.4f} acc={e['acc']:.4f}\n"
                )

    json_path = 'training_10epochs_results.json'
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'Wrote {txt_path} and {json_path}')


if __name__ == '__main__':
    main()
