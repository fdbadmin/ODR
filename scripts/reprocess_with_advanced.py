#!/usr/bin/env python3
"""
Reprocess Dataset with Advanced Preprocessing
==============================================

This script reprocesses the entire dataset using advanced preprocessing techniques
specifically designed for fundus images.

Usage:
    python reprocess_with_advanced.py --preset standard
    python reprocess_with_advanced.py --preset aggressive --target-size 512
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil
from datetime import datetime
import matplotlib.pyplot as plt

from src.advanced_preprocessing import RetinalImagePreprocessor


def backup_current_data(backup_dir: str = None):
    """Backup existing preprocessed data"""
    if backup_dir is None:
        backup_dir = f"preprocessed_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    backup_path = Path(backup_dir)
    preprocessed_path = Path("preprocessed_data")
    
    if not preprocessed_path.exists():
        print("⚠️  No existing preprocessed data found. Nothing to backup.")
        return None
    
    print(f"📦 Backing up current data to: {backup_path}")
    shutil.copytree(preprocessed_path, backup_path)
    print("✓ Backup complete")
    
    return backup_path


def visualize_comparison(original: np.ndarray, enhanced: np.ndarray, title: str = "Comparison"):
    """Show before/after comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original)
    ax1.set_title("Original", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(enhanced)
    ax2.set_title("Enhanced (Advanced Preprocessing)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'preprocessing_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison image")
    plt.show()


def reprocess_dataset(preset: str = 'standard', 
                     target_size: tuple = (224, 224),
                     show_samples: int = 3):
    """
    Reprocess entire dataset with advanced preprocessing.
    
    Args:
        preset: Preprocessing preset to use
        target_size: Target image size
        show_samples: Number of sample comparisons to show
    """
    print("=" * 80)
    print("🔧 REPROCESSING DATASET WITH ADVANCED PREPROCESSING")
    print("=" * 80)
    print()
    
    # Load existing preprocessed data
    print("📂 Loading current preprocessed data...")
    train_images = np.load('preprocessed_data/train_images.npy')
    val_images = np.load('preprocessed_data/val_images.npy')
    train_labels = np.load('preprocessed_data/train_labels.npy')
    val_labels = np.load('preprocessed_data/val_labels.npy')
    
    print(f"   Train: {train_images.shape}")
    print(f"   Val:   {val_images.shape}")
    print()
    
    # Create preprocessor based on preset
    print(f"🎨 Creating preprocessor (preset: {preset})...")
    
    if preset == 'standard':
        preprocessor = RetinalImagePreprocessor(
            target_size=target_size,
            use_green_channel=False,  # Keep RGB
            apply_illumination_correction=True,
            apply_vessel_enhancement=False,
            clahe_clip_limit=2.0,
            clahe_grid_size=(8, 8)
        )
        print("   • CLAHE contrast enhancement: ✅")
        print("   • Illumination correction: ✅")
        print("   • Vessel enhancement: ❌")
        print("   • Expected improvement: +2-3%")
    
    elif preset == 'aggressive':
        preprocessor = RetinalImagePreprocessor(
            target_size=target_size,
            use_green_channel=False,
            apply_illumination_correction=True,
            apply_vessel_enhancement=True,
            clahe_clip_limit=3.0,
            clahe_grid_size=(8, 8)
        )
        print("   • CLAHE contrast enhancement: ✅")
        print("   • Illumination correction: ✅")
        print("   • Vessel enhancement: ✅")
        print("   • Expected improvement: +3-5%")
    
    elif preset == 'vessel_focus':
        preprocessor = RetinalImagePreprocessor(
            target_size=target_size,
            use_green_channel=True,  # Use green channel
            apply_illumination_correction=True,
            apply_vessel_enhancement=True,
            clahe_clip_limit=3.0,
            clahe_grid_size=(8, 8)
        )
        print("   • Green channel extraction: ✅")
        print("   • CLAHE contrast enhancement: ✅")
        print("   • Illumination correction: ✅")
        print("   • Vessel enhancement: ✅")
        print("   • Expected improvement: +2-4% (for DR)")
    
    elif preset == 'minimal':
        preprocessor = RetinalImagePreprocessor(
            target_size=target_size,
            use_green_channel=False,
            apply_illumination_correction=False,
            apply_vessel_enhancement=False,
            clahe_clip_limit=1.0,  # Minimal enhancement
            clahe_grid_size=(8, 8)
        )
        print("   • Minimal preprocessing (baseline)")
    
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    print()
    
    # Show sample comparisons
    if show_samples > 0:
        print(f"📸 Showing {show_samples} sample comparisons...")
        sample_indices = np.random.choice(len(train_images), min(show_samples, len(train_images)), replace=False)
        
        for idx in sample_indices:
            original = train_images[idx]
            enhanced = preprocessor.preprocess_single_image(original)
            visualize_comparison(original, enhanced, f"Training Sample #{idx}")
    
    # Reprocess training data
    print("🔄 Reprocessing training images...")
    train_enhanced = np.zeros_like(train_images)
    for i in tqdm(range(len(train_images)), desc="Training"):
        train_enhanced[i] = preprocessor.preprocess_single_image(train_images[i])
    
    # Reprocess validation data
    print("🔄 Reprocessing validation images...")
    val_enhanced = np.zeros_like(val_images)
    for i in tqdm(range(len(val_images)), desc="Validation"):
        val_enhanced[i] = preprocessor.preprocess_single_image(val_images[i])
    
    print()
    print("💾 Saving enhanced data...")
    
    # Save with descriptive names
    np.save('preprocessed_data/train_images.npy', train_enhanced)
    np.save('preprocessed_data/val_images.npy', val_enhanced)
    # Labels unchanged
    print("   ✓ train_images.npy")
    print("   ✓ val_images.npy")
    print("   ✓ Labels unchanged (same file)")
    
    print()
    print("=" * 80)
    print("✅ REPROCESSING COMPLETE")
    print("=" * 80)
    print()
    print(f"Preprocessing preset: {preset}")
    print(f"Images processed: {len(train_images) + len(val_images)}")
    print(f"Output size: {target_size}")
    print()
    print("Next steps:")
    print("  1. Review sample comparisons")
    print("  2. Train model: python src/train.py")
    print("  3. Compare with baseline results")
    print()


def main():
    parser = argparse.ArgumentParser(description='Reprocess dataset with advanced preprocessing')
    parser.add_argument('--preset', type=str, default='standard',
                       choices=['minimal', 'standard', 'aggressive', 'vessel_focus'],
                       help='Preprocessing preset (default: standard)')
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224],
                       help='Target image size (default: 224 224)')
    parser.add_argument('--show-samples', type=int, default=3,
                       help='Number of sample comparisons to show (default: 3)')
    parser.add_argument('--backup', action='store_true',
                       help='Backup existing data before reprocessing')
    parser.add_argument('--skip-backup', action='store_true',
                       help='Skip backing up existing data')
    
    args = parser.parse_args()
    
    # Backup existing data unless skipped
    if not args.skip_backup:
        if args.backup or input("Backup existing preprocessed data? (y/n): ").lower() == 'y':
            backup_current_data()
    
    # Reprocess
    target_size = tuple(args.target_size)
    reprocess_dataset(args.preset, target_size, args.show_samples)


if __name__ == '__main__':
    main()
