"""
Enhanced data preprocessing with advanced augmentation techniques.
Replaces basic data_preprocessing.py with state-of-the-art methods.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
warnings.filterwarnings('ignore')

from config import (
    CSV_PATH, TRAIN_IMAGE_DIR, TEST_IMAGE_DIR, 
    LABEL_COLUMNS, DEFAULT_IMAGE_SIZE, RANDOM_SEED
)
from advanced_preprocessing import RetinalImagePreprocessor
from keyword_label_parser import DiagnosticKeywordParser
from severity_extractor import SeverityExtractor
from metadata_extractor import MetadataExtractor


def get_advanced_augmentation_pipeline(probability: float = 0.5) -> A.Compose:
    """
    Create advanced augmentation pipeline for training.
    
    Uses Albumentations library for efficient, GPU-accelerated augmentations.
    All augmentations are carefully chosen for retinal images.
    
    Args:
        probability: Probability of applying each augmentation
        
    Returns:
        Albumentations composition pipeline
    """
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=probability, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=probability
        ),
        
        # Optical distortions (simulates camera/eye movement)
        A.OpticalDistortion(
            distort_limit=0.1,
            shift_limit=0.1,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=probability * 0.5
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.1,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=probability * 0.5
        ),
        
        # Color augmentations (careful - retinal images sensitive to color)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=probability),
        
        # Hue/Saturation (subtle changes only)
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=probability * 0.3
        ),
        
        # Noise and blur (simulates image quality variation)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=probability * 0.3),
        
        # Coarse dropout (simulates occlusions)
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=probability * 0.2
        ),
    ])


def get_validation_pipeline() -> A.Compose:
    """
    Validation pipeline (no augmentation, only normalization).
    
    Returns:
        Albumentations composition for validation
    """
    return A.Compose([
        # No augmentations for validation
    ])


def advanced_preprocess_image(image_path: Path,
                              target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                              method: str = 'full') -> Optional[np.ndarray]:
    """
    Load and preprocess image using advanced techniques.
    
    Args:
        image_path: Path to image
        target_size: Target size for output
        method: Preprocessing method
            - 'basic': Original CLAHE method
            - 'green': Green channel extraction + CLAHE
            - 'full': Green + illumination correction (RECOMMENDED)
            - 'vessel': Full + vessel enhancement
            
    Returns:
        Preprocessed image or None
    """
    if method == 'basic':
        processor = RetinalImagePreprocessor(
            target_size=target_size,
            use_green_channel=False,
            apply_illumination_correction=False,
            apply_vessel_enhancement=False,
            clahe_clip_limit=2.0
        )
    elif method == 'green':
        processor = RetinalImagePreprocessor(
            target_size=target_size,
            use_green_channel=True,
            apply_illumination_correction=False,
            apply_vessel_enhancement=False,
            clahe_clip_limit=3.0
        )
    elif method == 'full':
        processor = RetinalImagePreprocessor(
            target_size=target_size,
            use_green_channel=True,
            apply_illumination_correction=True,
            apply_vessel_enhancement=False,
            clahe_clip_limit=3.0
        )
    elif method == 'vessel':
        processor = RetinalImagePreprocessor(
            target_size=target_size,
            use_green_channel=True,
            apply_illumination_correction=True,
            apply_vessel_enhancement=True,
            clahe_clip_limit=3.0
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return processor.process(image_path)


def process_dataset_enhanced(
    df,
    eye='both',
    image_dir: Path = TRAIN_IMAGE_DIR,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    preprocessing_method='full',
    apply_augmentation=False,
    use_eye_specific_labels=True,
    exclude_low_quality=True,
    extract_metadata=True,  # NEW: Extract age/gender
    extract_severity=True   # NEW: Extract severity levels
):
    """
    Process dataset with advanced preprocessing and optional augmentation.
    
    Args:
        df: DataFrame with image info
        eye: Which eye to use ('left', 'right', or 'both')
        image_dir: Image directory
        target_size: Target size
        preprocessing_method: 'basic', 'green', 'full', or 'vessel'
        apply_augmentation: Whether to apply augmentation
        use_eye_specific_labels: If True, parse diagnostic keywords per eye (RECOMMENDED)
        exclude_low_quality: If True, exclude images marked as low quality (RECOMMENDED)
        extract_metadata: If True, extract age/gender metadata (RECOMMENDED for training)
        extract_severity: If True, extract disease severity levels (OPTIONAL)
        
    Returns:
        Tuple of (images, labels, valid_indices, metadata, severity_info)
    """
    images = []
    labels = []
    valid_indices = []
    metadata_list = []  # NEW: Store age/gender for each image
    severity_list = []  # NEW: Store severity info for each image
    failed_count = 0
    excluded_low_quality_count = 0
    
    # Initialize extractors
    if use_eye_specific_labels:
        keyword_parser = DiagnosticKeywordParser()
        print("✓ Using eye-specific diagnostic keyword parsing")
        if exclude_low_quality:
            print("✓ Low quality images will be excluded")
    else:
        keyword_parser = None
        print("⚠ Using patient-level labels (less accurate)")
    
    # Initialize metadata extractor
    if extract_metadata:
        metadata_extractor = MetadataExtractor()
        print("✓ Extracting age/gender metadata")
    else:
        metadata_extractor = None
    
    # Initialize severity extractor
    if extract_severity:
        severity_extractor = SeverityExtractor()
        print("✓ Extracting severity levels")
    else:
        severity_extractor = None
    
    # Determine which eyes to process
    if eye == 'both':
        eye_columns = [('Left-Fundus', 'Left-Diagnostic Keywords'), 
                      ('Right-Fundus', 'Right-Diagnostic Keywords')]
    elif eye == 'left':
        eye_columns = [('Left-Fundus', 'Left-Diagnostic Keywords')]
    else:  # right
        eye_columns = [('Right-Fundus', 'Right-Diagnostic Keywords')]
    
    # Get augmentation pipeline
    if apply_augmentation:
        augmentor = get_advanced_augmentation_pipeline(probability=0.5)
        print(f"✓ Augmentation enabled (50% probability per operation)")
    else:
        augmentor = get_validation_pipeline()
    
    print(f"Processing {len(df)} patients with '{preprocessing_method}' method...")
    if eye == 'both':
        print(f"Processing both eyes (left + right) = {len(df) * 2} images expected")
    
    # Statistics for label source tracking
    label_stats = {'parsed': 0, 'fallback': 0, 'combined': 0, 'allocated': 0}
    allocation_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Get patient-level labels for smart allocation
        patient_labels = row[LABEL_COLUMNS].values.astype(np.float32)
        
        # Extract patient metadata (age/gender) - same for both eyes
        if extract_metadata and metadata_extractor:
            patient_age = row.get('Patient Age', None)
            patient_gender = row.get('Patient Sex', None)
            patient_metadata = metadata_extractor.extract_metadata(patient_age, patient_gender)
        else:
            patient_metadata = None
        
        # SMART: Process both eyes together to enable intelligent disease allocation
        if use_eye_specific_labels and keyword_parser and eye == 'both':
            left_keywords = row['Left-Diagnostic Keywords']
            right_keywords = row['Right-Diagnostic Keywords']
            
            # Extract severity for both eyes
            if extract_severity and severity_extractor:
                left_severity, right_severity = severity_extractor.extract_patient_both_eyes(
                    left_keywords, right_keywords
                )
            else:
                left_severity = None
                right_severity = None
            
            # Check for low quality and skip if requested
            if exclude_low_quality and keyword_parser:
                left_is_low_quality = keyword_parser.is_low_quality(left_keywords)
                right_is_low_quality = keyword_parser.is_low_quality(right_keywords)
                
                if left_is_low_quality:
                    excluded_low_quality_count += 1
                if right_is_low_quality:
                    excluded_low_quality_count += 1
                
                # Skip this patient entirely if both eyes are low quality
                if left_is_low_quality and right_is_low_quality:
                    failed_count += 2  # Count as 2 failed (both eyes)
                    continue
            else:
                left_is_low_quality = False
                right_is_low_quality = False
            
            # Use smart allocation method
            left_labels, right_labels, stats = keyword_parser.parse_patient_both_eyes(
                left_keywords, right_keywords, patient_labels
            )
            
            # Track statistics
            label_stats[stats['left_source']] += 1
            label_stats[stats['right_source']] += 1
            if stats['allocation_used']:
                allocation_count += 1
            
            # Process left eye image (if not low quality)
            left_img = None
            if not left_is_low_quality and 'Left-Fundus' in row and not pd.isna(row['Left-Fundus']):
                img_filename = row['Left-Fundus']
                possible_paths = [
                    image_dir / img_filename,
                    TRAIN_IMAGE_DIR / img_filename,
                    TEST_IMAGE_DIR / img_filename,
                ]
                
                img_path = None
                for path in possible_paths:
                    if path.exists():
                        img_path = path
                        break
                
                if img_path:
                    left_img = advanced_preprocess_image(
                        img_path, target_size=target_size, method=preprocessing_method
                    )
                    
                    if left_img is not None and apply_augmentation:
                        img_uint8 = (left_img * 255).astype(np.uint8)
                        augmented = augmentor(image=img_uint8)
                        left_img = augmented['image'].astype(np.float32) / 255.0
            
            # Process right eye image (if not low quality)
            right_img = None
            if not right_is_low_quality and 'Right-Fundus' in row and not pd.isna(row['Right-Fundus']):
                img_filename = row['Right-Fundus']
                possible_paths = [
                    image_dir / img_filename,
                    TRAIN_IMAGE_DIR / img_filename,
                    TEST_IMAGE_DIR / img_filename,
                ]
                
                img_path = None
                for path in possible_paths:
                    if path.exists():
                        img_path = path
                        break
                
                if img_path:
                    right_img = advanced_preprocess_image(
                        img_path, target_size=target_size, method=preprocessing_method
                    )
                    
                    if right_img is not None and apply_augmentation:
                        img_uint8 = (right_img * 255).astype(np.uint8)
                        augmented = augmentor(image=img_uint8)
                        right_img = augmented['image'].astype(np.float32) / 255.0
            
            # Add successfully processed images
            if left_img is not None:
                images.append(left_img)
                labels.append(left_labels)
                valid_indices.append(idx)
                if patient_metadata is not None:
                    metadata_list.append(patient_metadata)
                if left_severity is not None:
                    severity_list.append(left_severity)
            else:
                failed_count += 1
            
            if right_img is not None:
                images.append(right_img)
                labels.append(right_labels)
                valid_indices.append(idx)
                if patient_metadata is not None:
                    metadata_list.append(patient_metadata)
                if right_severity is not None:
                    severity_list.append(right_severity)
            else:
                failed_count += 1
        
        else:
            # Original approach: Process each eye independently (less smart)
            for fundus_col, keyword_col in eye_columns:
                if fundus_col not in row or pd.isna(row[fundus_col]):
                    failed_count += 1
                    continue
                
                # Get image path
                img_filename = row[fundus_col]
                possible_paths = [
                    image_dir / img_filename,
                    TRAIN_IMAGE_DIR / img_filename,
                    TEST_IMAGE_DIR / img_filename,
                ]
                
                img_path = None
                for path in possible_paths:
                    if path.exists():
                        img_path = path
                        break
                
                if img_path is None:
                    failed_count += 1
                    continue
                
                # Advanced preprocessing
                img = advanced_preprocess_image(
                    img_path,
                    target_size=target_size,
                    method=preprocessing_method
                )
                
                if img is None:
                    failed_count += 1
                    continue
                
                # Apply augmentation (if training)
                if apply_augmentation:
                    # Convert to uint8 for albumentations
                    img_uint8 = (img * 255).astype(np.uint8)
                    augmented = augmentor(image=img_uint8)
                    img = augmented['image'].astype(np.float32) / 255.0
                
                # Get labels - EYE-SPECIFIC or patient-level
                if use_eye_specific_labels and keyword_parser is not None:
                    # Parse diagnostic keywords for THIS specific eye
                    diagnostic_keywords = row[keyword_col]
                    
                    # Use parser with fallback
                    label_vector, source = keyword_parser.parse_with_fallback(
                        diagnostic_keywords, 
                        patient_labels,
                        confidence_threshold=0.5
                    )
                    label_stats[source] += 1
                else:
                    # Use patient-level labels (old approach)
                    label_vector = patient_labels
                
                images.append(img)
                labels.append(label_vector)
                valid_indices.append(idx)
    
    print(f"✓ Processed: {len(images)} images")
    print(f"✗ Failed: {failed_count} images")
    if excluded_low_quality_count > 0:
        print(f"🗑️  Excluded low quality: {excluded_low_quality_count} images")
    
    # Print label source statistics if using eye-specific labels
    if use_eye_specific_labels and keyword_parser is not None:
        total_labeled = sum(label_stats.values())
        if total_labeled > 0:
            print(f"\n📊 Label Source Statistics:")
            print(f"  Parsed from keywords: {label_stats['parsed']} ({label_stats['parsed']/total_labeled*100:.1f}%)")
            if label_stats.get('allocated', 0) > 0:
                print(f"  Smart allocated: {label_stats['allocated']} ({label_stats['allocated']/total_labeled*100:.1f}%)")
                print(f"    ↳ {allocation_count} patients had smart disease allocation")
            print(f"  Fallback to patient labels: {label_stats['fallback']} ({label_stats['fallback']/total_labeled*100:.1f}%)")
            if label_stats.get('combined', 0) > 0:
                print(f"  Combined (keywords + patient): {label_stats['combined']} ({label_stats['combined']/total_labeled*100:.1f}%)")
    
    # Print metadata statistics
    if extract_metadata and len(metadata_list) > 0:
        print(f"\n📊 Metadata Statistics:")
        print(f"  Extracted for: {len(metadata_list)} images")
        metadata_array = np.array(metadata_list)
        print(f"  Age range (normalized): {metadata_array[:, 0].min():.3f} - {metadata_array[:, 0].max():.3f}")
        print(f"  Gender: {int((metadata_array[:, 1] == 0).sum())} male, {int((metadata_array[:, 1] == 1).sum())} female")
    
    # Print severity statistics
    if extract_severity and len(severity_list) > 0:
        print(f"\n📊 Severity Statistics:")
        print(f"  Extracted for: {len(severity_list)} images")
        severity_scores = [s['severity_score'] for s in severity_list]
        has_severity = sum(1 for s in severity_scores if s > 0)
        print(f"  Images with severity info: {has_severity} ({has_severity/len(severity_list)*100:.1f}%)")
        from collections import Counter
        level_counts = Counter([s['severity_level'] for s in severity_list])
        for level in ['mild', 'moderate', 'severe', 'proliferative']:
            if level in level_counts:
                print(f"    {level}: {level_counts[level]}")
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Convert metadata and severity if available
    if len(metadata_list) > 0:
        metadata = np.array(metadata_list)
    else:
        metadata = None
    
    if len(severity_list) > 0:
        # Convert severity dicts to structured array
        severity_scores = np.array([s['severity_score'] for s in severity_list])
        is_proliferative = np.array([s['is_proliferative'] for s in severity_list])
        severity = np.stack([severity_scores, is_proliferative.astype(float)], axis=1)
    else:
        severity = None
    
    return images, labels, valid_indices, metadata, severity


def save_processed_data(images: np.ndarray,
                       labels: np.ndarray,
                       output_file: str = 'train',
                       output_dir: str = 'preprocessed_data_enhanced',
                       metadata: Optional[np.ndarray] = None,
                       severity: Optional[np.ndarray] = None) -> None:
    """
    Save processed data to .npy files.
    
    Args:
        images: Numpy array of processed images
        labels: Numpy array of labels
        output_file: Base name for output files ('train' or 'val')
        output_dir: Output directory
        metadata: Optional metadata (age, gender)
        severity: Optional severity scores
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save images and labels
    images_file = output_path / f'{output_file}_images.npy'
    labels_file = output_path / f'{output_file}_labels.npy'
    
    np.save(images_file, images)
    np.save(labels_file, labels)
    
    total_size = images.nbytes + labels.nbytes
    
    print(f"\n💾 Saved enhanced preprocessed data:")
    print(f"  Images: {images_file} (shape: {images.shape})")
    print(f"  Labels: {labels_file} (shape: {labels.shape})")
    
    # Save metadata if available
    if metadata is not None:
        metadata_file = output_path / f'{output_file}_metadata.npy'
        np.save(metadata_file, metadata)
        total_size += metadata.nbytes
        print(f"  Metadata: {metadata_file} (shape: {metadata.shape})")
    
    # Save severity if available
    if severity is not None:
        severity_file = output_path / f'{output_file}_severity.npy'
        np.save(severity_file, severity)
        total_size += severity.nbytes
        print(f"  Severity: {severity_file} (shape: {severity.shape})")
    
    print(f"  Total size: {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED DATA PREPROCESSING WITH ADVANCED TECHNIQUES")
    print("=" * 70)
    
    # Configuration
    PREPROCESSING_METHOD = 'full'  # 'basic', 'green', 'full', or 'vessel'
    OUTPUT_DIR = 'preprocessed_data_enhanced'
    
    print(f"\n🎯 Preprocessing method: {PREPROCESSING_METHOD.upper()}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Step 1: Load and split
    print("\n📂 Step 1: Loading and splitting data...")
    # Handle both CSV and Excel files
    if str(CSV_PATH).endswith('.xlsx'):
        df = pd.read_excel(CSV_PATH)
    else:
        df = pd.read_csv(CSV_PATH)
    
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['N']
    )
    
    print(f"Training: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    
    # Show class distribution
    print("\n📊 Class distribution (training):")
    for label in LABEL_COLUMNS:
        count = train_df[label].sum()
        pct = (count / len(train_df)) * 100
        print(f"  {label}: {int(count):4d} ({pct:5.2f}%)")
    
    # Step 2: Process training (WITH augmentation)
    print(f"\n🖼️  Step 2: Processing training images (with augmentation)...")
    train_images, train_labels, train_indices, train_metadata, train_severity = process_dataset_enhanced(
        train_df,
        eye='both',  # Process both left and right eyes
        preprocessing_method=PREPROCESSING_METHOD,
        apply_augmentation=False,  # Augmentation applied during training, not preprocessing
        extract_metadata=True,  # Extract age/gender
        extract_severity=True   # Extract severity levels
    )
    
    # Step 3: Process validation (NO augmentation)
    print(f"\n🖼️  Step 3: Processing validation images (no augmentation)...")
    val_images, val_labels, val_indices, val_metadata, val_severity = process_dataset_enhanced(
        val_df,
        eye='both',  # Process both left and right eyes
        preprocessing_method=PREPROCESSING_METHOD,
        apply_augmentation=False,
        extract_metadata=True,
        extract_severity=True
    )
    
    # Step 4: Save
    print(f"\n💾 Step 4: Saving preprocessed data...")
    save_processed_data(train_images, train_labels, 'train', OUTPUT_DIR, train_metadata, train_severity)
    save_processed_data(val_images, val_labels, 'val', OUTPUT_DIR, val_metadata, val_severity)
    
    print("\n" + "=" * 70)
    print("✅ ENHANCED PREPROCESSING COMPLETE!")
    print("=" * 70)
    print("\nImprovements over basic preprocessing:")
    print("  ✓ Green channel extraction (better vessel contrast)")
    print("  ✓ Illumination correction (removes lighting artifacts)")
    print("  ✓ Advanced CLAHE (better contrast)")
    print("  ✓ Bilateral filtering (noise reduction, edge preservation)")
    print("  ✓ ROI extraction (removes black borders)")
    print("\nNext steps:")
    print(f"  1. Train model: python train_enhanced.py")
    print(f"  2. Compare with baseline: models/best_model.pth")
    print("=" * 70)
