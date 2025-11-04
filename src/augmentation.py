"""
Advanced data augmentation for retinal fundus images.
Includes medical imaging-specific augmentations.
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image


class RetinalAugmentation:
    """
    Advanced augmentation pipeline for retinal fundus images.
    Includes both standard and medical imaging-specific augmentations.
    """
    
    def __init__(self, mode='train', image_size=224):
        """
        Args:
            mode: 'train' or 'val' - determines augmentation strength
            image_size: Target image size (default: 224)
        """
        self.mode = mode
        self.image_size = image_size
        
        if mode == 'train':
            self.augmentations = self._get_train_augmentations()
        else:
            self.augmentations = self._get_val_augmentations()
    
    def _get_train_augmentations(self):
        """Get training augmentations - aggressive but medical-safe."""
        return [
            # Geometric transformations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20, interpolation=transforms.InterpolationMode.BILINEAR),
            
            # Color augmentations (mild - preserve medical characteristics)
            transforms.ColorJitter(
                brightness=0.2,  # Lighting variations
                contrast=0.2,    # Contrast changes
                saturation=0.1,  # Mild saturation (preserve blood vessel appearance)
                hue=0.05        # Very mild hue shift
            ),
            
            # Affine transformations (slight)
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # 10% translation
                scale=(0.9, 1.1),      # 10% zoom
                shear=5                # Slight shear
            ),
            
            # Resize and normalize
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            
            # Random erasing (simulates occlusions/artifacts)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ]
    
    def _get_val_augmentations(self):
        """Get validation augmentations - no randomness."""
        return [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]
    
    def __call__(self, image):
        """Apply augmentation pipeline."""
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        
        # Apply transformations
        for transform in self.augmentations:
            image = transform(image)
        
        return image


class MixUp:
    """
    MixUp augmentation for multi-label classification.
    Mixes two samples and their labels.
    """
    
    def __init__(self, alpha=0.2):
        """
        Args:
            alpha: Beta distribution parameter (lower = less mixing)
        """
        self.alpha = alpha
    
    def __call__(self, batch_images, batch_labels):
        """
        Apply MixUp to a batch.
        
        Args:
            batch_images: Tensor of shape (B, C, H, W)
            batch_labels: Tensor of shape (B, num_classes)
        
        Returns:
            Mixed images and labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size).to(batch_images.device)
        
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index]
        mixed_labels = lam * batch_labels + (1 - lam) * batch_labels[index]
        
        return mixed_images, mixed_labels


class CutMix:
    """
    CutMix augmentation for multi-label classification.
    Cuts and pastes patches between samples.
    """
    
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, batch_images, batch_labels):
        """
        Apply CutMix to a batch.
        
        Args:
            batch_images: Tensor of shape (B, C, H, W)
            batch_labels: Tensor of shape (B, num_classes)
        
        Returns:
            Mixed images and labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size).to(batch_images.device)
        
        # Get image dimensions
        _, _, H, W = batch_images.size()
        
        # Random box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Copy patch
        mixed_images = batch_images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = batch_images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        
        mixed_labels = lam * batch_labels + (1 - lam) * batch_labels[index]
        
        return mixed_images, mixed_labels


def get_augmentation_policy(mode='train', image_size=224, use_mixup=False, use_cutmix=False):
    """
    Get augmentation policy.
    
    Args:
        mode: 'train' or 'val'
        image_size: Target image size
        use_mixup: Whether to use MixUp
        use_cutmix: Whether to use CutMix
    
    Returns:
        Augmentation function and optional batch augmentation
    """
    image_aug = RetinalAugmentation(mode=mode, image_size=image_size)
    
    batch_aug = None
    if mode == 'train':
        if use_mixup and use_cutmix:
            # Randomly choose between MixUp and CutMix
            batch_aug = lambda img, lbl: MixUp(0.2)(img, lbl) if random.random() > 0.5 else CutMix(1.0)(img, lbl)
        elif use_mixup:
            batch_aug = MixUp(alpha=0.2)
        elif use_cutmix:
            batch_aug = CutMix(alpha=1.0)
    
    return image_aug, batch_aug
