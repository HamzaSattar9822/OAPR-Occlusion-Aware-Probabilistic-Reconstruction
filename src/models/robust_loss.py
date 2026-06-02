"""
Robust probabilistic loss functions for pose estimation.

Standard L2 loss is sensitive to outliers and occlusion noise.
We implement heavy-tailed distributions (Cauchy, Laplace) and
learnable mixture models for robust regression.

References:
- Barron et al., "A General and Adaptive Robust Loss Function", CVPR 2019
- Modeling uncertainty in pose via probabilistic outputs
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class CauchyLoss(nn.Module):
    """
    Cauchy (Lorentzian) loss for robust pose regression.
    
    L(y, ŷ, σ) = log(1 + ((y - ŷ) / σ)²)
    
    This is more robust to outliers than L2/L1 due to its heavy tails.
    
    Args:
        scale: initial scale parameter (learned per joint)
        learnable_scale: if True, scale is trainable
        reduction: 'mean' or 'none'
    """
    
    def __init__(self, scale=1.0, learnable_scale=True, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.learnable_scale = learnable_scale
        
        if learnable_scale:
            self.log_scale = nn.Parameter(torch.tensor(np.log(scale)))
        else:
            self.register_buffer('log_scale', torch.tensor(np.log(scale)))
    
    def forward(self, predictions, targets, weights=None):
        """
        Args:
            predictions: (B, K, 2) predicted coordinates
            targets: (B, K, 2) ground truth coordinates
            weights: (B, K, 1) per-joint visibility weight
        
        Returns:
            loss: scalar if reduction='mean' else (B, K)
        """
        scale = torch.exp(self.log_scale)
        
        # Compute residuals
        residuals = (predictions - targets) / (scale + 1e-8)  # (B, K, 2)
        
        # Cauchy loss: log(1 + x²)
        loss = torch.log(1.0 + residuals ** 2)  # (B, K, 2)
        
        # Mean over coordinates
        loss = loss.mean(dim=-1)  # (B, K)
        
        # Apply weights
        if weights is not None:
            weights = weights.squeeze(-1) if weights.dim() > 2 else weights
            loss = loss * weights
        
        # Reduction
        if self.reduction == 'mean':
            if weights is not None:
                loss = loss.sum() / (weights.sum() + 1e-8)
            else:
                loss = loss.mean()
        
        return loss


class LaplaceLoss(nn.Module):
    """
    Laplace (L1-like) loss for robust pose regression.
    
    L(y, ŷ, b) = |y - ŷ| / b + log(b)
    
    More robust than L2 but less than Cauchy.
    """
    
    def __init__(self, scale=1.0, learnable_scale=True, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.learnable_scale = learnable_scale
        
        if learnable_scale:
            self.log_scale = nn.Parameter(torch.tensor(np.log(scale)))
        else:
            self.register_buffer('log_scale', torch.tensor(np.log(scale)))
    
    def forward(self, predictions, targets, weights=None):
        """
        Args:
            predictions: (B, K, 2)
            targets: (B, K, 2)
            weights: (B, K, 1)
        
        Returns:
            loss: scalar
        """
        scale = torch.exp(self.log_scale)
        
        # Compute residuals
        residuals = torch.abs(predictions - targets)  # (B, K, 2)
        
        # Laplace loss: |x|/b + log(b)
        loss = residuals / (scale + 1e-8) + torch.log(scale + 1e-8)
        
        loss = loss.mean(dim=-1)  # (B, K)
        
        if weights is not None:
            weights = weights.squeeze(-1) if weights.dim() > 2 else weights
            loss = loss * weights
        
        if self.reduction == 'mean':
            if weights is not None:
                loss = loss.sum() / (weights.sum() + 1e-8)
            else:
                loss = loss.mean()
        
        return loss


class CauchyMixtureLoss(nn.Module):
    """
    Mixture of Cauchy distributions for adaptive robustness.
    
    Learns a soft mixture of scales, allowing the loss to adaptively
    decide between strict (small scale) and robust (large scale) penalties.
    
    This is the approach we recommend for the paper.
    """
    
    def __init__(self, num_mixtures=3, initial_scale=1.0, learnable=True):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.learnable = learnable
        
        # Mixture component scales
        if learnable:
            scales = torch.logspace(-1, 1, num_mixtures) * initial_scale
            self.log_scales = nn.Parameter(torch.log(scales))
            self.mixture_weights = nn.Parameter(torch.ones(num_mixtures) / num_mixtures)
        else:
            scales = torch.logspace(-1, 1, num_mixtures) * initial_scale
            self.register_buffer('log_scales', torch.log(scales))
            self.register_buffer('mixture_weights', torch.ones(num_mixtures) / num_mixtures)
    
    def forward(self, predictions, targets, weights=None):
        """
        Args:
            predictions: (B, K, 2)
            targets: (B, K, 2)
            weights: (B, K, 1)
        
        Returns:
            loss: scalar
        """
        B, K, _ = predictions.shape
        
        scales = torch.exp(self.log_scales)  # (num_mix,)
        mix_w = F.softmax(self.mixture_weights, dim=0)  # (num_mix,)
        
        # Compute residuals
        residuals = predictions - targets  # (B, K, 2)
        
        # Compute Cauchy loss for each mixture component
        loss_per_component = []
        for scale in scales:
            residuals_scaled = residuals / (scale + 1e-8)
            loss_c = torch.log(1.0 + residuals_scaled ** 2).mean(dim=-1)  # (B, K)
            loss_per_component.append(loss_c)
        
        loss_per_component = torch.stack(loss_per_component, dim=-1)  # (B, K, num_mix)
        
        # Mixture: log-sum-exp for numerical stability
        log_mixture = torch.logsumexp(
            loss_per_component + torch.log(mix_w).unsqueeze(0).unsqueeze(0),
            dim=-1
        )  # (B, K)
        
        # Apply weights
        if weights is not None:
            weights = weights.squeeze(-1) if weights.dim() > 2 else weights
            log_mixture = log_mixture * weights
            return log_mixture.sum() / (weights.sum() + 1e-8)
        else:
            return log_mixture.mean()


class ProbabilisticPoseLoss(nn.Module):
    """
    Complete probabilistic loss incorporating:
    1. Robust coordinate regression (Cauchy/Laplace)
    2. Uncertainty regularization
    3. Occlusion-aware weighting
    
    This allows the model to output both predictions AND uncertainty estimates.
    """
    
    def __init__(self, loss_type='cauchy_mixture', scale_loss_weight=0.1,
                 occlusion_loss_weight=0.1):
        super().__init__()
        self.scale_loss_weight = scale_loss_weight
        self.occlusion_loss_weight = occlusion_loss_weight
        
        # Main coordinate loss
        if loss_type == 'cauchy':
            self.coord_loss = CauchyLoss(scale=1.0, learnable_scale=True)
        elif loss_type == 'laplace':
            self.coord_loss = LaplaceLoss(scale=1.0, learnable_scale=True)
        elif loss_type == 'cauchy_mixture':
            self.coord_loss = CauchyMixtureLoss(num_mixtures=3, initial_scale=1.0)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        logger.info(f"Using {loss_type} as main coordinate loss")
    
    def forward(self, predictions, targets, weights=None, 
                uncertainties=None, occlusion_scores=None):
        """
        Args:
            predictions: (B, K, 3) [x, y, confidence]
            targets: (B, K, 2) ground truth
            weights: (B, K, 1) visibility weights
            uncertainties: (B, K, 1) model confidence (optional)
            occlusion_scores: (B, K) occlusion likelihood (optional)
        
        Returns:
            total_loss: scalar
            loss_dict: breakdown of components
        """
        pred_coords = predictions[:, :, :2]
        
        # Main coordinate regression loss
        coord_loss = self.coord_loss(pred_coords, targets, weights)
        
        total_loss = coord_loss
        loss_dict = {'coord_loss': coord_loss.item()}
        
        # Uncertainty regularization (encourage confident predictions)
        if uncertainties is not None and self.scale_loss_weight > 0:
            # Penalize predictions with very low confidence
            unc_loss = -torch.log(uncertainties + 1e-8).mean()
            total_loss = total_loss + self.scale_loss_weight * unc_loss
            loss_dict['uncertainty_loss'] = unc_loss.item()
        
        # Occlusion-aware loss adjustment
        if occlusion_scores is not None and self.occlusion_loss_weight > 0:
            # Reduce loss weight for occluded joints (they're harder)
            weighted_coords = pred_coords * (1.0 - occlusion_scores.unsqueeze(-1))
            weighted_targets = targets * (1.0 - occlusion_scores.unsqueeze(-1))
            
            occlusion_loss = F.mse_loss(weighted_coords, weighted_targets)
            total_loss = total_loss + self.occlusion_loss_weight * occlusion_loss
            loss_dict['occlusion_loss'] = occlusion_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


# ─── Factory Functions ─────────────────────────────────────────────────────

def build_robust_loss(cfg):
    """Build loss function from config."""
    loss_cfg = cfg.get('loss', {})
    
    loss_type = loss_cfg.get('type', 'cauchy_mixture')
    scale_weight = loss_cfg.get('scale_loss_weight', 0.1)
    occlusion_weight = loss_cfg.get('occlusion_loss_weight', 0.1)
    
    logger.info(f"Building robust loss: {loss_type}")
    
    return ProbabilisticPoseLoss(
        loss_type=loss_type,
        scale_loss_weight=scale_weight,
        occlusion_loss_weight=occlusion_weight,
    )


# ─── Test / Demo ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Quick test
    B, K = 4, 17
    
    pred = torch.randn(B, K, 3)
    targets = torch.randn(B, K, 2)
    weights = torch.ones(B, K, 1)
    
    loss_fn = ProbabilisticPoseLoss(loss_type='cauchy_mixture')
    loss, loss_dict = loss_fn(pred, targets, weights)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
