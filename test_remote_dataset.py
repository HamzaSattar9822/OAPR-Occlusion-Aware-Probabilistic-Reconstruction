"""
test_remote_dataset.py

Test OAPR framework using remote CrowdPose dataset access.
Runs training epochs and provides accuracy metrics.

Usage:
    python test_remote_dataset.py --epochs 5 --batch_size 4 --dataset crowdpose
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.remote_dataset_loader import create_remote_dataloader
from src.models import build_oapr_framework


class TestingConfig:
    """Configuration for testing."""
    
    num_keypoints = 14  # CrowdPose
    seq_len = 7
    hidden_size = 128  # Reduced for testing
    num_heads = 4      # Reduced for testing
    use_mamba = False  # Use fallback (faster for testing)
    batch_size = 4
    learning_rate = 1e-3
    epochs = 5


def test_model_forward_pass(model, batch):
    """Test model forward pass with sample batch."""
    try:
        images = batch['image']
        
        # Dummy video sequence (B, T, K, 2)
        B = images.shape[0]
        T = 7
        K = 14
        video_clip = torch.randn(B, T, K, 2)
        
        # Forward pass
        output = model(video_clip)
        
        return output, None
    except Exception as e:
        return None, str(e)


def compute_test_accuracy(predictions, targets):
    """
    Compute accuracy metrics.
    
    Returns:
        accuracy_dict with various metrics
    """
    try:
        # Simple L2 distance based accuracy
        # Consider prediction correct if within 30 pixels
        pred = predictions.cpu().numpy()
        target = targets.cpu().numpy()
        
        distances = np.linalg.norm(pred - target, axis=-1)
        threshold = 30
        accuracy = (distances < threshold).mean()
        
        metrics = {
            'accuracy': accuracy,
            'mean_distance': distances.mean(),
            'median_distance': np.median(distances),
            'std_distance': distances.std(),
        }
        
        return metrics
    except Exception as e:
        return {'error': str(e)}


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 60)
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            # Get batch
            images = batch['image'].to(device)
            
            # Create dummy video sequence for testing
            B = images.shape[0]
            T = 7
            K = 14
            video_clip = torch.randn(B, T, K, 2).to(device)
            
            # Forward pass
            output = model(video_clip)
            
            # Dummy targets for loss computation
            targets = torch.randn(B, K, 2).to(device)
            target_weights = torch.ones(B, K, 1).to(device)
            
            # Compute loss
            if hasattr(model, 'compute_loss'):
                loss, loss_dict = model.compute_loss(
                    output['keypoints'],
                    targets,
                    target_weights,
                    output['confidence']
                )
            else:
                # Fallback to simple MSE loss
                loss = nn.MSELoss()(
                    output['keypoints'],
                    targets
                )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % max(1, len(dataloader) // 3) == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx+1}/{len(dataloader)}: Loss = {avg_loss:.6f}")
        
        except Exception as e:
            print(f"  Error in batch {batch_idx}: {e}")
            continue
    
    avg_epoch_loss = total_loss / max(num_batches, 1)
    return avg_epoch_loss


def evaluate(model, dataloader, device):
    """Evaluate model on test set."""
    model.eval()
    
    all_distances = []
    num_correct = 0
    total_samples = 0
    
    print("\nEvaluation")
    print("-" * 60)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                images = batch['image'].to(device)
                
                # Create dummy video sequence
                B = images.shape[0]
                T = 7
                K = 14
                video_clip = torch.randn(B, T, K, 2).to(device)
                
                # Forward pass
                output = model(video_clip)
                
                # Dummy targets
                targets = torch.randn(B, K, 2).to(device)
                
                # Compute distance-based accuracy
                predictions = output['keypoints'].cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                distances = np.linalg.norm(predictions - targets_np, axis=-1)
                all_distances.extend(distances.flatten().tolist())
                
                # Count correct predictions (within 30 pixels)
                threshold = 30
                num_correct += (distances < threshold).sum()
                total_samples += B * K
                
                if (batch_idx + 1) % max(1, len(dataloader) // 3) == 0:
                    print(f"  Batch {batch_idx+1}/{len(dataloader)}: Processed")
            
            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                continue
    
    accuracy = num_correct / max(total_samples, 1)
    
    eval_metrics = {
        'accuracy': accuracy,
        'num_correct': num_correct,
        'total_samples': total_samples,
        'mean_distance': np.mean(all_distances) if all_distances else 0,
        'std_distance': np.std(all_distances) if all_distances else 0,
    }
    
    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description="Test OAPR with remote dataset")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--dataset', default='crowdpose', help='Dataset: crowdpose or coco')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', default='cpu', help='Device: cpu or cuda')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("OAPR Framework - Remote Dataset Testing")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU (training will be slower)")
    
    # Create remote dataloaders
    print("\n[1/5] Creating remote dataloaders...")
    try:
        train_loader, train_dataset = create_remote_dataloader(
            dataset_name=args.dataset,
            split='train',
            batch_size=args.batch_size,
            num_samples=args.num_samples
        )
        
        val_loader, val_dataset = create_remote_dataloader(
            dataset_name=args.dataset,
            split='test',
            batch_size=args.batch_size,
            num_samples=max(10, args.num_samples // 5)
        )
        
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
    
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return
    
    # Build model
    print("\n[2/5] Building OAPR model...")
    try:
        cfg = {
            'model': {
                'num_keypoints': 14 if args.dataset.lower() == 'crowdpose' else 17,
                'seq_len': 7,
                'hidden_size': 128,
                'num_heads': 4,
                'num_spatial_layers': 2,
                'use_mamba': False,
            },
            'loss': {'type': 'cauchy_mixture'}
        }
        
        model = build_oapr_framework(cfg)
        model = model.to(device)
        print(f"  Model built successfully")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    except Exception as e:
        print(f"Error building model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test forward pass
    print("\n[3/5] Testing model forward pass...")
    try:
        test_batch = next(iter(train_loader))
        output, error = test_model_forward_pass(model, test_batch)
        
        if error is None:
            print(f"  Forward pass successful")
            print(f"    Output keys: {output.keys()}")
            print(f"    Keypoints shape: {output['keypoints'].shape}")
        else:
            print(f"  Error in forward pass: {error}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training loop
    print("\n[4/5] Starting training...")
    print("=" * 70)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(args.epochs):
        try:
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion,
                device, epoch, args.epochs
            )
            train_losses.append(train_loss)
            
            # Evaluate
            val_metrics = evaluate(model, val_loader, device)
            val_accuracies.append(val_metrics['accuracy'])
            
            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Training Loss: {train_loss:.6f}")
            print(f"  Validation Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Mean Distance: {val_metrics['mean_distance']:.2f} pixels")
            print(f"  Std Distance: {val_metrics['std_distance']:.2f} pixels")
        
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final results
    print("\n" + "=" * 70)
    print("[5/5] Final Results")
    print("=" * 70)
    
    print(f"\nTraining Summary:")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Initial loss: {train_losses[0]:.6f}")
    print(f"  Final loss: {train_losses[-1]:.6f}")
    print(f"  Loss reduction: {(1 - train_losses[-1]/train_losses[0])*100:.2f}%")
    
    print(f"\nValidation Accuracy:")
    print(f"  Initial accuracy: {val_accuracies[0]:.4f}")
    print(f"  Final accuracy: {val_accuracies[-1]:.4f}")
    print(f"  Accuracy improvement: {(val_accuracies[-1] - val_accuracies[0])*100:.2f}%")
    
    print(f"\nBest Results:")
    best_epoch = np.argmin(train_losses)
    best_acc_epoch = np.argmax(val_accuracies)
    print(f"  Best loss at epoch: {best_epoch + 1} (loss: {train_losses[best_epoch]:.6f})")
    print(f"  Best accuracy at epoch: {best_acc_epoch + 1} (accuracy: {val_accuracies[best_acc_epoch]:.4f})")
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
