"""
Data preprocessing utilities for ODIR-5K dataset.
Handles train/test splitting, image preprocessing, and data augmentation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    CSV_PATH, TRAIN_IMAGE_DIR, TEST_IMAGE_DIR, 
    LABEL_COLUMNS, DEFAULT_IMAGE_SIZE, RANDOM_SEED
)


def load_and_split_data(csv_path: Path = CSV_PATH, 
                        test_size: float = 0.2,
                        stratify_column: str = 'N',
                        random_state: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset and split into train and validation sets.
    
    Args:
        csv_path: Path to the CSV file.
        test_size: Proportion of the dataset to include in the validation split.
        stratify_column: Column to use for stratified splitting (default: 'N' for normal).
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_df, val_df)
    """
    df = pd.read_csv(csv_path)
    
    # If stratify column exists, use it for stratification
    stratify = df[stratify_column] if stratify_column in df.columns else None
    
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def get_image_path(filename: str, image_dir: Path = TRAIN_IMAGE_DIR) -> Optional[Path]:
    """
    Get the full path to an image file.
    
    Args:
        filename: Name of the image file.
        image_dir: Directory containing the images.
        
    Returns:
        Path to the image file, or None if not found.
    """
    possible_paths = [
        image_dir / filename,
        TRAIN_IMAGE_DIR / filename,
        TEST_IMAGE_DIR / filename,
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def preprocess_image(image_path: Path, 
                     target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                     apply_clahe: bool = True) -> Optional[np.ndarray]:
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file.
        target_size: Target size (width, height) for resizing.
        apply_clahe: Whether to apply CLAHE for contrast enhancement.
        
    Returns:
        Preprocessed image as numpy array (normalized to [0, 1]), or None if error.
    """
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if apply_clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None


def process_dataset(df: pd.DataFrame, 
                   eye: str = 'right',
                   image_dir: Path = TRAIN_IMAGE_DIR,
                   target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                   apply_clahe: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Process all images in a dataset.
    
    Args:
        df: DataFrame with image information and labels.
        eye: Which eye to use ('left' or 'right').
        image_dir: Directory containing images.
        target_size: Target size for images.
        apply_clahe: Whether to apply CLAHE preprocessing.
        
    Returns:
        Tuple of (images_list, labels_list, valid_indices)
    """
    images = []
    labels = []
    valid_indices = []
    failed_count = 0
    
    column = f'{eye.capitalize()}-Fundus'
    
    print(f"Processing {len(df)} images from {eye} eye...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        if column not in row or pd.isna(row[column]):
            failed_count += 1
            continue
        
        # Get image path
        img_path = get_image_path(row[column], image_dir=image_dir)
        
        if img_path is None:
            failed_count += 1
            continue
        
        # Preprocess image
        img = preprocess_image(img_path, target_size=target_size, apply_clahe=apply_clahe)
        
        if img is None:
            failed_count += 1
            continue
        
        # Get labels
        label_vector = row[LABEL_COLUMNS].values.astype(np.float32)
        
        images.append(img)
        labels.append(label_vector)
        valid_indices.append(idx)
    
    print(f"✓ Successfully processed: {len(images)} images")
    print(f"✗ Failed to process: {failed_count} images")
    
    return images, labels, valid_indices


def save_processed_data(images: List[np.ndarray], 
                       labels: List[np.ndarray],
                       output_file: str = 'train') -> None:
    """
    Save processed images and labels to .npy files.
    
    Args:
        images: List of preprocessed images.
        labels: List of label vectors.
        output_file: Base name for output files (without extension).
    """
    output_dir = Path('preprocessed_data')
    output_dir.mkdir(exist_ok=True)
    
    images_array = np.array(images)
    labels_array = np.array(labels)
    
    images_path = output_dir / f'{output_file}_images.npy'
    labels_path = output_dir / f'{output_file}_labels.npy'
    
    np.save(images_path, images_array)
    np.save(labels_path, labels_array)
    
    print(f"\n💾 Saved preprocessed data:")
    print(f"  - Images: {images_path} (shape: {images_array.shape})")
    print(f"  - Labels: {labels_path} (shape: {labels_array.shape})")
    print(f"  - Total size: {(images_array.nbytes + labels_array.nbytes) / (1024**3):.2f} GB")


if __name__ == "__main__":
    print("=" * 70)
    print("ODIR-5K DATA PREPROCESSING")
    print("=" * 70)
    
    # Step 1: Load and split data
    print("\n📂 Step 1: Loading and splitting data...")
    train_df, val_df = load_and_split_data(test_size=0.2)
    
    # Show class distribution
    print("\n📊 Training set class distribution:")
    train_dist = train_df[LABEL_COLUMNS].sum()
    for label in LABEL_COLUMNS:
        count = train_dist[label]
        pct = (count / len(train_df)) * 100
        print(f"  {label}: {int(count):4d} ({pct:5.2f}%)")
    
    # Step 2: Process training data
    print("\n🖼️  Step 2: Processing training images...")
    train_images, train_labels, train_indices = process_dataset(
        train_df, 
        eye='right',
        image_dir=TRAIN_IMAGE_DIR,
        apply_clahe=True
    )
    
    # Step 3: Process validation data
    print("\n🖼️  Step 3: Processing validation images...")
    val_images, val_labels, val_indices = process_dataset(
        val_df,
        eye='right', 
        image_dir=TRAIN_IMAGE_DIR,
        apply_clahe=True
    )
    
    # Step 4: Save preprocessed data
    print("\n💾 Step 4: Saving preprocessed data...")
    save_processed_data(train_images, train_labels, output_file='train')
    save_processed_data(val_images, val_labels, output_file='val')
    
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the preprocessed data in 'preprocessed_data/' folder")
    print("  2. Train a PyTorch model using the preprocessed data")
    print("  3. Evaluate model performance on validation set")
    print("=" * 70)
