"""
Phase 4C Data Preprocessing Script
Prepares ODIR-5K dataset with RGB color preservation and optimized vessel enhancement.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.phase4c_preprocessing import Phase4CPreprocessor


def load_labels(csv_path: str):
    """Load and process labels from CSV."""
    df = pd.read_csv(csv_path)
    
    # Disease columns (7 classes - excluding Hypertension)
    disease_cols = ['N', 'D', 'G', 'C', 'A', 'M', 'O']
    
    labels = {}
    for _, row in df.iterrows():
        img_id = str(row['ID'])
        left_id = f"{img_id}_left"
        right_id = f"{img_id}_right"
        
        # Get disease labels (0 or 1)
        label_vector = row[disease_cols].values.astype(np.float32)
        
        labels[left_id] = label_vector
        labels[right_id] = label_vector
    
    return labels


def preprocess_phase4c():
    """Main preprocessing function for Phase 4C."""
    
    print("\n" + "="*80)
    print("PHASE 4C: RGB COLOR PRESERVATION + VESSEL ENHANCEMENT")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nKey Features:")
    print("  ✓ RGB color preservation (RED, GREEN, BLUE channels)")
    print("  ✓ Vessel enhancement on GREEN channel only")
    print("  ✓ 7×7 kernel (optimized for fine details)")
    print("  ✓ 384×384 resolution (up from 224×224)")
    print("  ✓ CLAHE + bilateral filtering per channel")
    print("="*80 + "\n")
    
    # Paths
    data_dir = Path('ODIR-5K')
    csv_path = data_dir / 'full_df.csv'
    images_dir = data_dir / 'preprocessed_images'
    output_dir = Path('preprocessed_data_phase4c')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = Phase4CPreprocessor(
        target_size=(384, 384),
        vessel_kernel_size=7
    )
    
    # Load labels
    print("Loading labels...")
    labels_dict = load_labels(csv_path)
    print(f"Loaded labels for {len(labels_dict)} images\n")
    
    # Get all image files
    image_files = []
    for pattern in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(images_dir.glob(f'**/{pattern}')))
    
    print(f"Found {len(image_files)} image files")
    
    # Process images
    processed_images = []
    processed_labels = []
    image_ids = []
    
    print("\nPreprocessing images...")
    for img_path in tqdm(image_files, desc="Processing"):
        # Extract image ID (e.g., "123_left" or "123_right")
        img_id = img_path.stem
        
        # Skip if no label
        if img_id not in labels_dict:
            continue
        
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Preprocess
            img_processed = preprocessor.preprocess(img)
            
            # Store
            processed_images.append(img_processed)
            processed_labels.append(labels_dict[img_id])
            image_ids.append(img_id)
            
        except Exception as e:
            print(f"\nError processing {img_id}: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(processed_images)} images")
    
    # Convert to arrays
    images_array = np.array(processed_images, dtype=np.float32)
    labels_array = np.array(processed_labels, dtype=np.float32)
    
    print(f"\nImages shape: {images_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    
    # Split train/val (80/20)
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(images_array))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=42,
        stratify=labels_array[:, 1]  # Stratify by diabetes (most common)
    )
    
    train_images = images_array[train_idx]
    train_labels = labels_array[train_idx]
    val_images = images_array[val_idx]
    val_labels = labels_array[val_idx]
    
    print(f"\nTrain: {len(train_images)} images")
    print(f"Val:   {len(val_images)} images")
    
    # Save
    print("\nSaving preprocessed data...")
    np.save(output_dir / 'train_images.npy', train_images)
    np.save(output_dir / 'train_labels.npy', train_labels)
    np.save(output_dir / 'val_images.npy', val_images)
    np.save(output_dir / 'val_labels.npy', val_labels)
    
    # Save image IDs for reference
    np.save(output_dir / 'train_ids.npy', np.array([image_ids[i] for i in train_idx]))
    np.save(output_dir / 'val_ids.npy', np.array([image_ids[i] for i in val_idx]))
    
    # Print statistics
    print("\n" + "="*80)
    print("PHASE 4C PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  • train_images.npy: {train_images.shape} ({train_images.nbytes / 1e9:.2f} GB)")
    print(f"  • train_labels.npy: {train_labels.shape}")
    print(f"  • val_images.npy: {val_images.shape} ({val_images.nbytes / 1e9:.2f} GB)")
    print(f"  • val_labels.npy: {val_labels.shape}")
    print(f"\nClass distribution (training):")
    disease_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Myopia', 'Other']
    for i, name in enumerate(disease_names):
        count = train_labels[:, i].sum()
        pct = 100 * count / len(train_labels)
        print(f"  {name:15s}: {int(count):5d} ({pct:5.1f}%)")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\nNext steps:")
    print("  1. Update training script to use 'preprocessed_data_phase4c' directory")
    print("  2. Run training: M5_NUM_WORKERS=4 python scripts/train_advanced.py --model convnext_tiny --epochs 50 --batch-size 32")
    print("     (Note: Smaller batch size due to larger images)")
    print("  3. Expected improvement: 5-10% F1 increase from RGB color + higher resolution")


if __name__ == '__main__':
    preprocess_phase4c()
