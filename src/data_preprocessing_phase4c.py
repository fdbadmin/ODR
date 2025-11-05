"""
Phase 4C Data Preprocessing Script
Prepares ODIR-5K dataset with RGB color preservation and optimized vessel enhancement.
Optimized for M5 MacBook Pro: 10 cores, 32GB RAM, multiprocessing enabled.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Tuple, List, Dict
import multiprocessing as mp
from functools import partial
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.phase4c_preprocessing import Phase4CPreprocessor


def load_labels(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and process ODIR-5K labels (7 classes, H removed).
    
    Args:
        data_path: Path to Excel/CSV file with labels
        
    Returns:
        Tuple of (DataFrame, label_columns)
    """
    # Handle both Excel and CSV formats
    if str(data_path).endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    
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


def process_single_image(args: Tuple[Path, Dict, Phase4CPreprocessor]) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Process a single image (for multiprocessing).
    
    Args:
        args: Tuple of (img_path, labels_dict, preprocessor)
        
    Returns:
        Tuple of (processed_image, label, image_id) or None if failed
    """
    img_path, labels_dict, preprocessor = args
    img_id = img_path.stem
    
    # Skip if no label
    if img_id not in labels_dict:
        return None
    
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # Preprocess
        img_processed = preprocessor.preprocess(img)
        
        # Return processed data
        return (img_processed, labels_dict[img_id], img_id)
        
    except Exception as e:
        return None


def preprocess_phase4c(num_workers: int = 4, chunksize: int = 150):
    """
    Main preprocessing function for Phase 4C with M5 optimization.
    
    Args:
        num_workers: Number of parallel workers (default: 4 for M5 P-cores)
        chunksize: Number of images per worker batch (default: 150 for high memory)
    """
    
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
    print("\nM5 High-Memory Optimizations:")
    print(f"  ✓ Multiprocessing: {num_workers} workers (P-cores)")
    print(f"  ✓ Large batches: {chunksize} images per chunk")
    print(f"  ✓ Memory budget: ~24-26GB peak usage")
    print(f"  ✓ Expected time: 10-12 minutes (2x faster)")
    print("="*80 + "\n")
    
    # Paths
    data_dir = Path('ODIR-5K')
    data_file = data_dir / 'data.xlsx'
    images_dir = data_dir / 'Training Images'
    output_dir = Path('preprocessed_data_phase4c')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = Phase4CPreprocessor(
        target_size=(384, 384),
        vessel_kernel_size=7
    )
    
    # Load labels
    print("Loading labels...")
    labels_dict = load_labels(data_file)
    print(f"Loaded labels for {len(labels_dict)} images\n")
    
    # Get all image files
    print("Scanning for image files...")
    image_files = []
    for pattern in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(images_dir.glob(f'**/{pattern}')))
    
    print(f"Found {len(image_files)} image files")
    
    # Filter to only images with labels
    print("Filtering images with labels...")
    valid_image_files = [f for f in image_files if f.stem in labels_dict]
    print(f"Images with labels: {len(valid_image_files)}")
    
    if len(valid_image_files) == 0:
        print("ERROR: No valid images found!")
        return
    
    # Prepare arguments for multiprocessing
    process_args = [(img_path, labels_dict, preprocessor) for img_path in valid_image_files]
    
    # Process images with multiprocessing
    print(f"\nPreprocessing {len(valid_image_files)} images with {num_workers} workers...")
    print(f"Large batch mode: {chunksize} images per chunk for maximum throughput")
    print("This will take approximately 10-12 minutes on M5 hardware...\n")
    
    processed_images = []
    processed_labels = []
    image_ids = []
    
    # Use multiprocessing pool with imap_unordered for better progress visibility
    with mp.Pool(processes=num_workers) as pool:
        # Process with progress bar - use imap_unordered for immediate feedback
        with tqdm(total=len(process_args), desc="Processing", unit="img", 
                  ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for result in pool.imap_unordered(process_single_image, process_args, chunksize=chunksize):
                if result is not None:
                    img_processed, label, img_id = result
                    processed_images.append(img_processed)
                    processed_labels.append(label)
                    image_ids.append(img_id)
                pbar.update(1)
    
    print(f"\nSuccessfully processed {len(processed_images)} images")
    
    # Convert to arrays (more memory efficient than list storage)
    print("Converting to numpy arrays...")
    images_array = np.array(processed_images, dtype=np.float32)
    labels_array = np.array(processed_labels, dtype=np.float32)
    
    # Free up memory
    del processed_images
    del processed_labels
    
    print(f"Images shape: {images_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    print(f"Memory usage: ~{images_array.nbytes / 1e9:.2f} GB")
    
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
    # Important: Set multiprocessing start method for macOS
    mp.set_start_method('spawn', force=True)
    
    # Use 4 workers for M5 performance cores
    num_workers = 4
    
    # Large chunksize for high memory throughput (150 images per batch)
    # This uses ~24-26GB RAM but processes 2x faster
    chunksize = 150
    
    print(f"\nM5 High-Memory Mode (28GB available):")
    print(f"  Workers: {num_workers} (P-cores)")
    print(f"  Chunksize: {chunksize} images/batch (3x larger)")
    print(f"  Memory: ~24-26GB peak")
    print(f"  Speed: ~8-10 img/s (vs 5.4 img/s before)")
    print(f"  Time: 10-12 minutes (vs 20 minutes)")
    
    preprocess_phase4c(num_workers=num_workers, chunksize=chunksize)
