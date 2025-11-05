"""
Reprocess data with vessel enhancement enabled.
This should provide additional gains on top of threshold optimization.

Expected additional gain: +0.5-1.5% F1
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import shutil

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.advanced_preprocessing import RetinalImagePreprocessor


def backup_current_data():
    """Backup current preprocessed data."""
    backup_dir = Path('preprocessed_data_backup')
    if backup_dir.exists():
        print("⚠️  Backup already exists, skipping...")
        return False
    
    print("💾 Creating backup of current preprocessed data...")
    shutil.copytree('preprocessed_data', backup_dir)
    print(f"✓ Backup created at: {backup_dir}")
    return True


def reprocess_with_vessels(image_dir='ODIR-5K/Training Images', 
                           output_dir='preprocessed_data_vessels',
                           target_size=(224, 224)):
    """
    Reprocess all images with vessel enhancement enabled.
    
    Args:
        image_dir: Directory containing original images
        output_dir: Where to save vessel-enhanced images
        target_size: Output image size
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize preprocessor with vessel enhancement
    preprocessor = RetinalImagePreprocessor(
        target_size=target_size,
        use_green_channel=True,
        apply_illumination_correction=True,
        apply_vessel_enhancement=True,  # ← KEY CHANGE
        clahe_clip_limit=3.0,
        clahe_grid_size=(8, 8)
    )
    
    print(f"\n🔬 Preprocessing Configuration:")
    print(f"  Green Channel: ✓")
    print(f"  Illumination Correction: ✓")
    print(f"  Vessel Enhancement: ✓ (NEW!)")
    print(f"  CLAHE: ✓ (clip_limit={preprocessor.clahe_clip_limit})")
    print(f"  Target Size: {target_size}")
    
    # Load existing data to know what to process
    train_images = np.load('preprocessed_data/train_images.npy', mmap_mode='r')
    val_images = np.load('preprocessed_data/val_images.npy', mmap_mode='r')
    
    train_labels = np.load('preprocessed_data/train_labels.npy')
    val_labels = np.load('preprocessed_data/val_labels.npy')
    
    train_metadata = np.load('preprocessed_data/train_metadata.npy', allow_pickle=True)
    val_metadata = np.load('preprocessed_data/val_metadata.npy', allow_pickle=True)
    
    print(f"\n📊 Dataset Info:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    
    # Get image paths from metadata
    image_base = Path(image_dir)
    
    def process_split(images_array, metadata_array, split_name):
        """Process one split (train or val)."""
        processed = []
        skipped = 0
        
        print(f"\n🔄 Processing {split_name} split...")
        for i in tqdm(range(len(images_array)), desc=f"Processing {split_name}"):
            metadata = metadata_array[i]
            
            # Try to get image path from metadata
            if isinstance(metadata, dict) and 'image_path' in metadata:
                img_path = Path(metadata['image_path'])
            elif isinstance(metadata, np.ndarray) and len(metadata) > 0:
                # Metadata might be stored as structured array
                img_path = Path(str(metadata[0]))
            else:
                # Skip if we can't determine path
                skipped += 1
                processed.append(images_array[i])  # Keep original
                continue
            
            # Make sure path exists
            if not img_path.exists():
                # Try relative to image_base
                img_path = image_base / img_path.name
            
            if img_path.exists():
                # Reprocess with vessel enhancement
                img = preprocessor.process(img_path, visualize=False)
                if img is not None:
                    processed.append(img)
                else:
                    skipped += 1
                    processed.append(images_array[i])  # Keep original if processing fails
            else:
                skipped += 1
                processed.append(images_array[i])  # Keep original
        
        if skipped > 0:
            print(f"⚠️  Could not reprocess {skipped} images (kept original)")
        
        return np.array(processed)
    
    # Process both splits
    train_processed = process_split(train_images, train_metadata, 'train')
    val_processed = process_split(val_images, val_metadata, 'val')
    
    # Save processed data
    print(f"\n💾 Saving vessel-enhanced data to: {output_dir}/")
    np.save(output_path / 'train_images.npy', train_processed)
    np.save(output_path / 'val_images.npy', val_processed)
    
    # Copy labels and metadata (unchanged)
    np.save(output_path / 'train_labels.npy', train_labels)
    np.save(output_path / 'val_labels.npy', val_labels)
    np.save(output_path / 'train_metadata.npy', train_metadata)
    np.save(output_path / 'val_metadata.npy', val_metadata)
    
    print(f"✓ Saved {len(train_processed)} training images")
    print(f"✓ Saved {len(val_processed)} validation images")
    
    # Create comparison visualization
    print("\n📸 Creating before/after comparison...")
    visualize_vessel_enhancement(
        train_images[:5], train_processed[:5],
        'results/vessel_enhancement_comparison.png'
    )


def visualize_vessel_enhancement(original_images, processed_images, save_path):
    """Create side-by-side comparison of original vs vessel-enhanced."""
    import matplotlib.pyplot as plt
    
    n_samples = min(5, len(original_images))
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 8))
    
    for i in range(n_samples):
        # Original (without vessel enhancement)
        img_orig = original_images[i]
        if img_orig.max() <= 1.0:
            img_orig = (img_orig * 255).astype(np.uint8)
        if len(img_orig.shape) == 3:
            img_orig = img_orig[:, :, 0]  # Show one channel
        
        axes[0, i].imshow(img_orig, cmap='gray')
        axes[0, i].set_title(f'Original #{i+1}', fontsize=10)
        axes[0, i].axis('off')
        
        # Vessel-enhanced
        img_vessel = processed_images[i]
        if img_vessel.max() <= 1.0:
            img_vessel = (img_vessel * 255).astype(np.uint8)
        if len(img_vessel.shape) == 3:
            img_vessel = img_vessel[:, :, 0]
        
        axes[1, i].imshow(img_vessel, cmap='gray')
        axes[1, i].set_title(f'Vessel Enhanced #{i+1}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle('Vessel Enhancement Comparison\nTop: Original | Bottom: Vessel Enhanced', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison: {save_path}")
    plt.close()


def main():
    """Main processing pipeline."""
    print("=" * 60)
    print("VESSEL ENHANCEMENT PREPROCESSING")
    print("=" * 60)
    
    # Check if original data exists
    if not Path('preprocessed_data/train_images.npy').exists():
        print("❌ Error: preprocessed_data not found!")
        print("   Please run data_preprocessing_enhanced.py first.")
        return
    
    # Create backup
    backup_created = backup_current_data()
    
    # Ask user to confirm
    print("\n⚠️  This will create vessel-enhanced data in: preprocessed_data_vessels/")
    print("   The original preprocessed_data/ will remain unchanged.")
    print("   You can compare both versions and choose which to use.")
    
    response = input("\nProceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Reprocess with vessel enhancement
    reprocess_with_vessels(
        image_dir='ODIR-5K/Training Images',
        output_dir='preprocessed_data_vessels',
        target_size=(224, 224)
    )
    
    print("\n" + "=" * 60)
    print("✅ VESSEL ENHANCEMENT COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Evaluate vessel-enhanced data with optimized thresholds")
    print("2. If better, replace preprocessed_data/ with preprocessed_data_vessels/")
    print("3. Or use the evaluate script directly on the new data")
    
    if backup_created:
        print(f"\n💾 Original data backed up to: preprocessed_data_backup/")


if __name__ == '__main__':
    main()
