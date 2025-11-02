"""
Focal Loss Implementation for Multi-Label Classification

Focal Loss focuses on hard-to-classify examples by down-weighting
easy examples and focusing training on hard negatives.

Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples.
               Default: 0.25 (weights positive examples more)
        gamma: Focusing parameter for modulating loss. Higher gamma = more focus on hard examples.
               Default: 2.0
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    
    Forward Args:
        inputs: Logits from model (before sigmoid), shape [batch_size, num_classes]
        targets: Ground truth labels, shape [batch_size, num_classes]
    
    Returns:
        Focal loss value
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits), shape [N, C]
            targets: Ground truth labels, shape [N, C]
        
        Returns:
            Loss value
        """
        # Get probabilities from logits
        probs = torch.sigmoid(inputs)
        
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate pt (probability of correct class)
        # If target=1, pt=prob; if target=0, pt=1-prob
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Calculate focal weight: (1-pt)^gamma
        # Harder examples (low pt) get higher weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Calculate alpha weight
        # Balance positive/negative examples
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Combine: alpha * focal_weight * BCE
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with additional per-class weighting for severe class imbalance.
    
    Combines:
    1. Focal loss (focuses on hard examples)
    2. Class weights (handles class imbalance)
    
    Args:
        alpha: Focal loss alpha parameter (default: 0.25)
        gamma: Focal loss gamma parameter (default: 2.0)
        class_weights: Tensor of shape [num_classes] with per-class weights
                      Higher weight = more important class
        reduction: 'none' | 'mean' | 'sum'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute weighted focal loss.
        
        Args:
            inputs: Model predictions (logits), shape [N, C]
            targets: Ground truth labels, shape [N, C]
        
        Returns:
            Loss value
        """
        # Get probabilities
        probs = torch.sigmoid(inputs)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Focal weight: (1-pt)^gamma
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weight
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Focal loss per sample per class
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply class weights if provided
        if self.class_weights is not None:
            if self.class_weights.device != focal_loss.device:
                self.class_weights = self.class_weights.to(focal_loss.device)
            
            # Expand class weights to match focal_loss shape [N, C]
            class_weights_expanded = self.class_weights.unsqueeze(0).expand_as(focal_loss)
            focal_loss = focal_loss * class_weights_expanded
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(labels, method='inverse_freq', power=1.0):
    """
    Compute class weights for handling class imbalance.
    
    Args:
        labels: numpy array of shape [N, C] with binary labels
        method: 'inverse_freq' or 'effective_num'
        power: Exponent for inverse frequency (default: 1.0)
               Higher power = more aggressive weighting
    
    Returns:
        torch.Tensor of shape [C] with class weights
    """
    import numpy as np
    
    num_samples = len(labels)
    num_classes = labels.shape[1]
    
    if method == 'inverse_freq':
        # Count positive samples per class
        pos_counts = labels.sum(axis=0)
        
        # Inverse frequency: weight = total_samples / (num_classes * positive_count)
        # Add small epsilon to avoid division by zero
        weights = num_samples / (num_classes * (pos_counts + 1e-6))
        
        # Apply power to make weighting more/less aggressive
        weights = weights ** power
        
    elif method == 'effective_num':
        # Effective number of samples (Cui et al. 2019)
        # Better for extreme imbalance
        beta = 0.9999
        pos_counts = labels.sum(axis=0)
        effective_num = 1.0 - np.power(beta, pos_counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights to have mean=1
    weights = weights / weights.mean()
    
    return torch.FloatTensor(weights)


if __name__ == '__main__':
    # Test focal loss
    print("Testing Focal Loss...")
    
    # Create dummy data
    batch_size = 32
    num_classes = 8
    
    # Random logits and labels
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Test standard focal loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss_value = focal_loss(logits, labels)
    print(f"Focal Loss: {loss_value.item():.4f}")
    
    # Test with class weights
    import numpy as np
    dummy_labels = np.random.randint(0, 2, (1000, num_classes))
    class_weights = compute_class_weights(dummy_labels)
    print(f"Class weights: {class_weights}")
    
    weighted_focal = WeightedFocalLoss(alpha=0.25, gamma=2.0, class_weights=class_weights)
    weighted_loss = weighted_focal(logits, labels)
    print(f"Weighted Focal Loss: {weighted_loss.item():.4f}")
    
    print("\n✓ Focal loss implementation working correctly!")
