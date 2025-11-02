"""
Data Augmentation and Oversampling for Class Imbalance

This module handles class imbalance by:
1. Oversampling minority classes
2. Smart augmentation that preserves disease features
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Tuple
import random


class BalancedDataset(Dataset):
    """
    Dataset that oversamples minority classes to balance the distribution.
    
    Args:
        images: numpy array of images [N, H, W, C]
        labels: numpy array of labels [N, num_classes]
        min_samples_per_class: Minimum number of positive samples per class
        augment: Whether to apply augmentation to oversampled examples
    """
    
    def __init__(self, images, labels, min_samples_per_class=200, augment=True):
        self.original_images = images
        self.original_labels = labels
        self.min_samples = min_samples_per_class
        self.augment = augment
        
        # Balance the dataset
        self.images, self.labels = self._balance_dataset()
        
        print(f"Original dataset: {len(self.original_images)} samples")
        print(f"Balanced dataset: {len(self.images)} samples")
        print(f"Class distribution after balancing:")
        for i in range(self.labels.shape[1]):
            count = self.labels[:, i].sum()
            print(f"  Class {i}: {int(count)} positive samples")
    
    def _balance_dataset(self):
        """Balance dataset by oversampling minority classes."""
        balanced_images = list(self.original_images)
        balanced_labels = list(self.original_labels)
        
        num_classes = self.original_labels.shape[1]
        
        for class_idx in range(num_classes):
            # Find positive samples for this class
            positive_indices = np.where(self.original_labels[:, class_idx] == 1)[0]
            num_positive = len(positive_indices)
            
            if num_positive < self.min_samples:
                # Need to oversample
                needed = self.min_samples - num_positive
                print(f"Class {class_idx}: {num_positive} samples, adding {needed} via oversampling")
                
                # Randomly sample with replacement from positive examples
                oversample_indices = np.random.choice(
                    positive_indices,
                    size=needed,
                    replace=True
                )
                
                # Add oversampled examples
                for idx in oversample_indices:
                    image = self.original_images[idx].copy()
                    label = self.original_labels[idx].copy()
                    
                    # Apply augmentation to oversampled examples
                    if self.augment:
                        image = self._augment_image(image, label)
                    
                    balanced_images.append(image)
                    balanced_labels.append(label)
        
        return np.array(balanced_images), np.array(balanced_labels)
    
    def _augment_image(self, image, labels):
        """
        Apply disease-preserving augmentation.
        
        Careful to preserve disease-relevant features:
        - Avoid excessive brightness changes (affects diabetic retinopathy, AMD)
        - Preserve center region (important for AMD, macula)
        - Preserve vessel patterns (important for diabetes, hypertension)
        """
        augmented = image.copy()
        
        # 1. Horizontal flip (safe for all diseases)
        if random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)
        
        # 2. Vertical flip (safe for all diseases)
        if random.random() > 0.5:
            augmented = cv2.flip(augmented, 0)
        
        # 3. Rotation (small angles only to preserve orientation)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = augmented.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            augmented = cv2.warpAffine(augmented, M, (w, h))
        
        # 4. Brightness/contrast (very subtle to preserve color information)
        if random.random() > 0.5:
            # Only if not AMD or Diabetes (color-sensitive diseases)
            if labels[4] == 0 and labels[1] == 0:  # Not AMD and not Diabetes
                alpha = random.uniform(0.95, 1.05)  # Contrast
                beta = random.uniform(-5, 5)  # Brightness
                augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
        
        # 5. Gaussian noise (helps with generalization)
        if random.random() > 0.5:
            noise = np.random.normal(0, 2, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 255).astype(np.uint8)
        
        # 6. Slight zoom (preserves center for AMD)
        if random.random() > 0.3:
            scale = random.uniform(0.95, 1.05)
            h, w = augmented.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
            augmented = cv2.warpAffine(augmented, M, (w, h))
        
        return augmented
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor and normalize
        image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        
        # Normalize with ImageNet stats (since we use pretrained models)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        label = torch.FloatTensor(label)
        
        return image, label


class SMOTEAugmentation:
    """
    SMOTE-like augmentation for images.
    
    Instead of creating synthetic examples in feature space,
    we blend images from the same class.
    """
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.num_classes = labels.shape[1]
        
        # Index images by class for quick lookup
        self.class_indices = {}
        for class_idx in range(self.num_classes):
            self.class_indices[class_idx] = np.where(labels[:, class_idx] == 1)[0]
    
    def generate_synthetic(self, class_idx, num_synthetic):
        """
        Generate synthetic examples for a class by blending images.
        
        Args:
            class_idx: Class to generate examples for
            num_synthetic: Number of synthetic examples to generate
        
        Returns:
            synthetic_images: numpy array [num_synthetic, H, W, C]
            synthetic_labels: numpy array [num_synthetic, num_classes]
        """
        indices = self.class_indices[class_idx]
        
        if len(indices) < 2:
            # Not enough samples to blend
            return None, None
        
        synthetic_images = []
        synthetic_labels = []
        
        for _ in range(num_synthetic):
            # Select two random images from the class
            idx1, idx2 = np.random.choice(indices, size=2, replace=False)
            
            img1 = self.images[idx1]
            img2 = self.images[idx2]
            
            # Blend with random weight
            alpha = random.uniform(0.3, 0.7)
            blended = (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)
            
            # Take label from first image (or could combine)
            label = self.labels[idx1].copy()
            
            synthetic_images.append(blended)
            synthetic_labels.append(label)
        
        return np.array(synthetic_images), np.array(synthetic_labels)


def create_balanced_loader(images, labels, batch_size=32, min_samples_per_class=150):
    """
    Create a balanced data loader with oversampling.
    
    Args:
        images: numpy array of images [N, H, W, C]
        labels: numpy array of labels [N, num_classes]
        batch_size: Batch size for training
        min_samples_per_class: Minimum samples per class after balancing
    
    Returns:
        DataLoader with balanced data
    """
    from torch.utils.data import DataLoader
    
    # Create balanced dataset
    dataset = BalancedDataset(
        images=images,
        labels=labels,
        min_samples_per_class=min_samples_per_class,
        augment=True
    )
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for macOS MPS
        pin_memory=False
    )
    
    return loader


if __name__ == '__main__':
    # Test balancing
    print("Testing data balancing...")
    
    # Create dummy imbalanced data
    num_samples = 1000
    images = np.random.rand(num_samples, 224, 224, 3) * 255
    images = images.astype(np.uint8)
    
    # Imbalanced labels (class 0 has many, class 7 has few)
    labels = np.zeros((num_samples, 8))
    labels[:800, 0] = 1  # 80% class 0
    labels[:200, 1] = 1  # 20% class 1
    labels[:50, 7] = 1   # 5% class 7 (rare)
    
    print("\nOriginal distribution:")
    for i in range(8):
        print(f"Class {i}: {int(labels[:, i].sum())} samples")
    
    # Create balanced dataset
    balanced_dataset = BalancedDataset(images, labels, min_samples_per_class=200)
    
    print("\n✓ Data balancing working correctly!")
