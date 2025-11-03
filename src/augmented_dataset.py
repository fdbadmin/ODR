"""
Augmented dataset class for ODIR-5K with Albumentations.
Applies data augmentation during training for improved generalization.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AugmentedODIRDataset(Dataset):
    """
    PyTorch Dataset with augmentation support for ODIR-5K.
    
    Uses Albumentations for efficient augmentation pipeline.
    Augmentations are applied on-the-fly during training.
    """
    
    def __init__(self, images_path: str, labels_path: str, 
                 metadata_path: str = None, 
                 augment: bool = False,
                 augment_probability: float = 0.5):
        """
        Args:
            images_path: Path to .npy file containing images
            labels_path: Path to .npy file containing labels
            metadata_path: Optional path to .npy file containing metadata
            augment: Whether to apply augmentation
            augment_probability: Probability of applying each augmentation
        """
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        self.metadata = np.load(metadata_path) if metadata_path else None
        self.augment = augment
        
        print(f"Loaded dataset: {len(self.images)} samples")
        print(f"  Image shape: {self.images.shape}")
        print(f"  Label shape: {self.labels.shape}")
        if self.metadata is not None:
            print(f"  Metadata shape: {self.metadata.shape}")
        print(f"  Augmentation: {'ENABLED' if augment else 'DISABLED'}")
        
        # Define augmentation pipeline
        if augment:
            self.transform = self._get_augmentation_pipeline(augment_probability)
        else:
            self.transform = None
    
    def _get_augmentation_pipeline(self, p: float = 0.5):
        """
        Create augmentation pipeline for retinal fundus images.
        
        Carefully designed for medical imaging:
        - Geometric transforms (rotation, flip, shift)
        - Color/brightness adjustments (retinal color is diagnostic)
        - Noise and blur (simulates real-world variation)
        
        Args:
            p: Base probability for augmentations
            
        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            # Geometric transformations (fundus images can be captured from any angle)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=15,  # ±15 degrees (small rotations only)
                border_mode=0,  # cv2.BORDER_CONSTANT
                p=p
            ),
            A.Affine(
                scale=(0.9, 1.1),  # ±10% zoom
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # ±10% shift
                rotate=(-15, 15),  # ±15 degrees
                mode=0,  # cv2.BORDER_CONSTANT
                p=p
            ),
            
            # Optical distortions (simulates lens effects)
            A.OneOf([
                A.OpticalDistortion(
                    distort_limit=0.05,
                    border_mode=0,
                    p=1.0
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.1,
                    border_mode=0,
                    p=1.0
                ),
            ], p=p * 0.3),  # Less frequent (30%)
            
            # Color augmentations (CRITICAL for retinal images - color indicates disease)
            # More conservative to preserve diagnostic information
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,  # ±20%
                    contrast_limit=0.2,    # ±20%
                    p=1.0
                ),
                A.RandomGamma(
                    gamma_limit=(80, 120),  # 0.8-1.2x
                    p=1.0
                ),
                A.CLAHE(
                    clip_limit=4.0,
                    tile_grid_size=(8, 8),
                    p=1.0
                ),
            ], p=p),
            
            # Hue/Saturation (very subtle - retinal color is diagnostic!)
            A.HueSaturationValue(
                hue_shift_limit=10,     # ±10 degrees
                sat_shift_limit=15,     # ±15%
                val_shift_limit=10,     # ±10%
                p=p * 0.3  # Only 30% chance
            ),
            
            # Noise and blur (simulates image quality variation)
            A.OneOf([
                A.GaussNoise(
                    var_limit=(10.0, 50.0),
                    mean=0,
                    p=1.0
                ),
                A.GaussianBlur(
                    blur_limit=(3, 5),
                    p=1.0
                ),
                A.MotionBlur(
                    blur_limit=5,
                    p=1.0
                ),
            ], p=p * 0.3),  # Less frequent (30%)
            
            # Coarse dropout (simulates occlusions, shadows)
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                p=p * 0.2  # Rare (20%)
            ),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply augmentation if enabled
        if self.transform is not None:
            # Albumentations expects uint8 images [0-255]
            # Our preprocessed images are float32 [0-1], so convert
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Apply augmentation
            augmented = self.transform(image=image_uint8)
            image = augmented['image']
            
            # Convert back to float32 [0-1]
            image = image.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor and rearrange from HWC to CHW
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.from_numpy(label).float()
        
        # Return metadata if available
        if self.metadata is not None:
            metadata = torch.from_numpy(self.metadata[idx]).float()
            return image, label, metadata
        
        return image, label


def visualize_augmentations(dataset, num_samples=5, save_path='results/augmentation_samples.png'):
    """
    Visualize augmentation effects for quality checking.
    
    Args:
        dataset: AugmentedODIRDataset with augmentation enabled
        num_samples: Number of samples to show
        save_path: Where to save the visualization
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))
    fig.suptitle('Augmentation Examples (Original + 4 Augmented Versions)', fontsize=14)
    
    # Temporarily enable augmentation
    original_augment_state = dataset.augment
    dataset.augment = False
    
    for i in range(num_samples):
        # Get original image
        dataset.augment = False
        dataset.transform = None
        original_img, label = dataset[i]
        
        # Show original
        axes[i, 0].imshow(original_img.permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Show 4 augmented versions
        dataset.augment = True
        dataset.transform = dataset._get_augmentation_pipeline(p=0.8)  # High prob for demo
        
        for j in range(1, 5):
            aug_img, _ = dataset[i]
            axes[i, j].imshow(aug_img.permute(1, 2, 0).numpy())
            axes[i, j].set_title(f'Augmented {j}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved augmentation examples to {save_path}")
    
    # Restore original state
    dataset.augment = original_augment_state


if __name__ == '__main__':
    """Test augmentation pipeline"""
    print("="*70)
    print("AUGMENTED DATASET TEST")
    print("="*70)
    
    # Load training data with augmentation
    train_dataset = AugmentedODIRDataset(
        'preprocessed_data/train_images.npy',
        'preprocessed_data/train_labels.npy',
        augment=True,
        augment_probability=0.5
    )
    
    # Test a few samples
    print(f"\nTesting augmentation on 3 samples...")
    for i in range(3):
        image, label = train_dataset[i]
        print(f"Sample {i+1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Label: {label.numpy()}")
    
    print("\n✓ Augmentation pipeline working correctly!")
    print("\nNext steps:")
    print("  1. Visualize augmentations: visualize_augmentations(train_dataset)")
    print("  2. Integrate into train.py")
    print("  3. Train models with augmentation + Focal Loss")
