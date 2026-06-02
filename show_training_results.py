#!/usr/bin/env python3
"""
Real-time Training Results Analyzer for M3 OAPR 50-Epoch Run
Displays all results directly in terminal as training progresses
"""

import os
import sys
import time
import re
from pathlib import Path

def parse_training_log(log_file):
    """Parse training log for metrics"""
    results = {
        'epochs_completed': 0,
        'current_epoch': 0,
        'epochs': [],
        'best_ap': 0,
        'best_loss': float('inf'),
    }
    
    if not os.path.exists(log_file):
        return results
    
    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()
    
    # Extract epoch starts
    epoch_pattern = r'Epoch\s+(\d+)/50.*?LR:\s+([\d.]+)'
    epoch_matches = re.findall(epoch_pattern, content)
    results['epochs_completed'] = len(epoch_matches)
    if epoch_matches:
        results['current_epoch'] = int(epoch_matches[-1][0])
    
    # Extract loss values
    loss_pattern = r'Loss:\s+([\d.]+)'
    losses = re.findall(loss_pattern, content)
    if losses:
        results['final_loss'] = float(losses[-1])
        results['best_loss'] = min(float(l) for l in losses)
    
    # Extract accuracy values
    ap_pattern = r'AP:\s+([\d.]+)|Accuracy:\s+([\d.]+)'
    ap_matches = re.findall(ap_pattern, content)
    if ap_matches:
        aps = [float(m[0]) if m[0] else float(m[1]) for m in ap_matches if m[0] or m[1]]
        if aps:
            results['best_ap'] = max(aps)
            results['current_ap'] = aps[-1]
    
    return results

def print_header():
    print("\n" + "="*80)
    print("M3 OAPR 50-EPOCH COCO TRAINING - LIVE RESULTS MONITOR".center(80))
    print("="*80 + "\n")

def print_status(log_file, terminal_file):
    """Print current training status"""
    print_header()
    
    results = parse_training_log(log_file)
    
    print(f"Current Epoch: {results['current_epoch']}/50")
    print(f"Epochs Completed: {results['epochs_completed']}")
    
    if 'final_loss' in results:
        print(f"\nCurrent Training Loss: {results['final_loss']:.4f}")
        print(f"Best Training Loss: {results['best_loss']:.4f}")
    
    if 'current_ap' in results:
        print(f"\nCurrent Accuracy (AP): {results['current_ap']:.2f}%")
        print(f"Best Accuracy (AP): {results['best_ap']:.2f}%")
    
    print(f"\nTraining Output Log:")
    print(f"  Full results: {log_file}")
    print(f"  Terminal: {terminal_file}")
    
    print("\n" + "-"*80)
    print("LAST 20 LINES OF OUTPUT:")
    print("-"*80)
    
    if os.path.exists(log_file):
        with open(log_file, 'r', errors='ignore') as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(line.rstrip())
    else:
        print("Waiting for training to start...")
    
    print("\n" + "="*80)
    print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

def main():
    log_file = "/Users/saad/Downloads/oapr_pose/m3_coco_50epochs_full_results.log"
    terminal_file = "/Users/saad/.cursor/projects/Users-saad-Downloads-oapr-pose/terminals/554710.txt"
    
    print_status(log_file, terminal_file)

if __name__ == '__main__':
    main()
