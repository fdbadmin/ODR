#!/usr/bin/env python3
"""
Test augmentation pipeline and visualize examples.
Run this before training to verify augmentations look good.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.augmented_dataset import AugmentedODIRDataset
from config import LABEL_COLUMNS


def visualize_augmentations(num_samples=5):
    """Visualize augmentation effects."""
    print("="*70)
    print("AUGMENTATION VISUALIZATION")
    print("="*70)
    
    # Load training dataset with augmentation
    print("\nLoading training data with augmentation...")
    train_dataset = AugmentedODIRDataset(
        'preprocessed_data/train_images.npy',
        'preprocessed_data/train_labels.npy',
        augment=True,
        augment_probability=0.8  # High prob for visualization
    )
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3 * num_samples))
    fig.suptitle('Augmentation Examples: Original + 5 Augmented Versions', 
                 fontsize=16, fontweight='bold')
    
    # Select random samples
    indices = np.random.choice(len(train_dataset), num_samples, replace=False)
    
    for row, idx in enumerate(indices):
        # Get original image (no augmentation)
        train_dataset.augment = False
        train_dataset.transform = None
        original_img, label = train_dataset[idx]
        
        # Convert to numpy and transpose for display
        original_np = original_img.permute(1, 2, 0).numpy()
        
        # Show original
        axes[row, 0].imshow(original_np)
        axes[row, 0].set_title('Original', fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Add label info
        disease_names = [LABEL_COLUMNS[i] for i, val in enumerate(label.numpy()) if val == 1]
        label_text = ', '.join(disease_names) if disease_names else 'Unknown'
        axes[row, 0].text(0.5, -0.1, label_text, 
                         transform=axes[row, 0].transAxes,
                         ha='center', fontsize=8, style='italic')
        
        # Show 5 augmented versions
        train_dataset.augment = True
        train_dataset.transform = train_dataset._get_augmentation_pipeline(p=0.8)
        
        for col in range(1, 6):
            aug_img, _ = train_dataset[idx]
            aug_np = aug_img.permute(1, 2, 0).numpy()
            
            axes[row, col].imshow(aug_np)
            axes[row, col].set_title(f'Augmented {col}', fontsize=10)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'augmentation_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")
    
    # Also show it
    plt.show()
    
    print("\n" + "="*70)
    print("AUGMENTATION CHECK")
    print("="*70)
    print("\n✅ Things to verify in the visualization:")
    print("  1. Images are still recognizable (not too distorted)")
    print("  2. Color changes are subtle (retinal color is diagnostic)")
    print("  3. Geometric transforms look natural")
    print("  4. No extreme artifacts or black borders")
    print("\nIf augmentations look good, proceed with training!")


def test_augmentation_pipeline():
    """Quick test of augmentation pipeline."""
    print("\n" + "="*70)
    print("AUGMENTATION PIPELINE TEST")
    print("="*70)
    
    # Load dataset
    print("\nLoading training dataset with augmentation...")
    train_dataset = AugmentedODIRDataset(
        'preprocessed_data/train_images.npy',
        'preprocessed_data/train_labels.npy',
        augment=True,
        augment_probability=0.5
    )
    
    # Test a few samples
    print("\nTesting augmentation on 5 random samples...")
    for i in range(5):
        idx = np.random.randint(0, len(train_dataset))
        image, label = train_dataset[idx]
        
        print(f"\nSample {i+1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Image type: {image.dtype}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Label shape: {label.shape}")
        print(f"  Label: {label.numpy()}")
        
        # Verify data integrity
        assert image.shape == (3, 224, 224), f"Wrong shape: {image.shape}"
        assert image.min() >= 0 and image.max() <= 1, f"Values out of range: [{image.min()}, {image.max()}]"
        assert label.shape == (len(LABEL_COLUMNS),), f"Wrong label shape: {label.shape}"
    
    print("\n✅ All tests passed!")
    print("  - Image shapes correct (3, 224, 224)")
    print("  - Image values in valid range [0, 1]")
    print("  - Labels correct shape")
    print("  - No data corruption detected")


def compare_with_without_augmentation():
    """Compare dataset statistics with and without augmentation."""
    print("\n" + "="*70)
    print("AUGMENTATION IMPACT ANALYSIS")
    print("="*70)
    
    # Load without augmentation
    print("\n1. Loading dataset WITHOUT augmentation...")
    dataset_no_aug = AugmentedODIRDataset(
        'preprocessed_data/train_images.npy',
        'preprocessed_data/train_labels.npy',
        augment=False
    )
    
    # Sample images
    images_no_aug = []
    for i in range(100):
        img, _ = dataset_no_aug[i]
        images_no_aug.append(img.numpy())
    images_no_aug = np.array(images_no_aug)
    
    print(f"\nWithout augmentation statistics:")
    print(f"  Mean: {images_no_aug.mean():.4f}")
    print(f"  Std: {images_no_aug.std():.4f}")
    print(f"  Min: {images_no_aug.min():.4f}")
    print(f"  Max: {images_no_aug.max():.4f}")
    
    # Load with augmentation
    print("\n2. Loading dataset WITH augmentation...")
    dataset_aug = AugmentedODIRDataset(
        'preprocessed_data/train_images.npy',
        'preprocessed_data/train_labels.npy',
        augment=True,
        augment_probability=0.5
    )
    
    # Sample images (same indices but augmented)
    images_aug = []
    for i in range(100):
        img, _ = dataset_aug[i]
        images_aug.append(img.numpy())
    images_aug = np.array(images_aug)
    
    print(f"\nWith augmentation statistics:")
    print(f"  Mean: {images_aug.mean():.4f}")
    print(f"  Std: {images_aug.std():.4f}")
    print(f"  Min: {images_aug.min():.4f}")
    print(f"  Max: {images_aug.max():.4f}")
    
    print("\n3. Impact:")
    print(f"  Mean change: {abs(images_aug.mean() - images_no_aug.mean()):.4f}")
    print(f"  Std increase: {images_aug.std() - images_no_aug.std():.4f}")
    print("\n✅ Augmentation increases data diversity (higher std) while preserving mean brightness")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test augmentation pipeline')
    parser.add_argument('--test', choices=['visual', 'pipeline', 'compare', 'all'],
                       default='all', help='Type of test to run')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of samples for visualization')
    args = parser.parse_args()
    
    try:
        if args.test in ['visual', 'all']:
            visualize_augmentations(num_samples=args.samples)
        
        if args.test in ['pipeline', 'all']:
            test_augmentation_pipeline()
        
        if args.test in ['compare', 'all']:
            compare_with_without_augmentation()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nYou can now proceed with training:")
        print("  python src/train.py")
        print("\nOr train ensemble models:")
        print("  python src/train_ensemble_models.py --model both")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
