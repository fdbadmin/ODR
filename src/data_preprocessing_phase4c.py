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


def is_low_quality(diagnostic_keywords: str) -> bool:
    """
    Check if image is marked as low quality and should be excluded.
    
    Args:
        diagnostic_keywords: String with comma-separated diagnostic terms
        
    Returns:
        True if image should be excluded due to quality issues
    """
    if pd.isna(diagnostic_keywords):
        return False
    
    keywords_lower = str(diagnostic_keywords).lower()
    
    # Quality exclusion keywords
    quality_exclusions = [
        'low image quality', 'low quality', 'poor quality', 
        'image unclear', 'blur', 'artifact', 'unqualified'
    ]
    
    return any(kw in keywords_lower for kw in quality_exclusions)


def parse_intelligent_per_eye_labels(patient_labels: dict, diagnostic_keywords: str, eye_side: str) -> np.ndarray:
    """
    INTELLIGENT PER-EYE LABELING: Combines patient-level labels with diagnostic keywords
    to determine which diseases apply to which specific eye.
    
    Strategy:
    1. Start with patient-level labels (which diseases the patient has)
    2. Parse diagnostic keywords to determine which eye has which disease
    3. If keywords explicitly mention "normal" for this eye, remove all disease labels
    4. If keywords mention specific disease for this eye, ensure it's labeled
    
    Args:
        patient_labels: Dict with keys 'A', 'D', 'G', 'C', 'M', 'N', 'O' (patient-level)
        diagnostic_keywords: String with comma-separated diagnostic terms for this eye
        eye_side: 'left' or 'right'
        
    Returns:
        7-element binary vector [AMD, Diabetes, Glaucoma, Cataract, Myopia, Normal, Other]
    """
    if pd.isna(diagnostic_keywords):
        # No keywords - use patient labels (assume symmetric)
        labels = np.array([
            patient_labels['A'],  # AMD
            patient_labels['D'],  # Diabetes  
            patient_labels['G'],  # Glaucoma
            patient_labels['C'],  # Cataract
            patient_labels['M'],  # Myopia
            patient_labels['N'],  # Normal
            patient_labels['O']   # Other
        ], dtype=np.float32)
        return labels
    
    keywords_lower = str(diagnostic_keywords).lower()
    
    # Check if this eye is explicitly normal
    is_normal_eye = ('normal fundus' in keywords_lower or 
                     (keywords_lower.strip() == 'normal') or
                     ('normal' in keywords_lower and not any(disease in keywords_lower for disease in [
                         'diabetic', 'retinopathy', 'glaucoma', 'cataract', 'amd', 
                         'macular degeneration', 'myopia', 'drusen', 'proliferative'
                     ])))
    
    if is_normal_eye:
        # This eye is explicitly normal - override patient labels
        return np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
    
    # Start with patient-level labels (diseases patient has somewhere)
    labels = np.array([
        patient_labels['A'],  # AMD
        patient_labels['D'],  # Diabetes  
        patient_labels['G'],  # Glaucoma
        patient_labels['C'],  # Cataract
        patient_labels['M'],  # Myopia
        patient_labels['N'],  # Normal
        patient_labels['O']   # Other
    ], dtype=np.float32)
    
    # Refine with keyword detection for this specific eye
    # If keyword mentions disease explicitly, ensure it's marked
    
    # AMD keywords
    if any(kw in keywords_lower for kw in [
        'amd', 'macular degeneration', 'drusen', 'geographic atrophy', 'cnv'
    ]):
        labels[0] = 1  # AMD
        labels[5] = 0  # Not normal if disease present
    
    # Diabetes keywords
    if any(kw in keywords_lower for kw in [
        'diabetic', 'retinopathy', 'proliferative', 'nonproliferative', 
        'maculopathy', 'dme', 'diabetic macular edema'
    ]):
        labels[1] = 1  # Diabetes
        labels[5] = 0  # Not normal
    
    # Glaucoma keywords
    if any(kw in keywords_lower for kw in [
        'glaucoma', 'suspicious glaucoma', 'disc suspicious'
    ]):
        labels[2] = 1  # Glaucoma
        labels[5] = 0  # Not normal
    
    # Cataract keywords
    if any(kw in keywords_lower for kw in [
        'cataract', 'lens opacity', 'nuclear cataract', 'cortical cataract'
    ]):
        labels[3] = 1  # Cataract
        labels[5] = 0  # Not normal
    
    # Myopia keywords
    if any(kw in keywords_lower for kw in [
        'myopia', 'pathological myopia', 'high myopia', 'myopic'
    ]):
        labels[4] = 1  # Myopia
        labels[5] = 0  # Not normal
    
    # Other diseases
    if any(kw in keywords_lower for kw in [
        'epiretinal', 'myelinated', 'laser', 'vitreous', 'membrane',
        'retinal vein occlusion', 'brvo', 'crvo', 'optic atrophy',
        'retinitis', 'retinoschisis', 'pigmentosa', 'punctate', 'spots',
        'hypertensive'
    ]):
        labels[6] = 1  # Other
        labels[5] = 0  # Not normal
    
    # If patient has disease but this eye's keywords don't mention it,
    # and keywords ARE specific (not empty), then remove that disease for this eye
    has_specific_keywords = any(disease in keywords_lower for disease in [
        'diabetic', 'retinopathy', 'glaucoma', 'cataract', 'amd', 
        'macular degeneration', 'myopia', 'drusen', 'epiretinal',
        'laser', 'vitreous', 'membrane'
    ])
    
    if has_specific_keywords:
        # Keywords are specific - if disease not mentioned, remove it for this eye
        if patient_labels['A'] == 1 and labels[0] == 1:
            if not any(kw in keywords_lower for kw in ['amd', 'macular degeneration', 'drusen']):
                labels[0] = 0  # Patient has AMD, but not in this eye
        
        if patient_labels['D'] == 1 and labels[1] == 1:
            if not any(kw in keywords_lower for kw in ['diabetic', 'retinopathy', 'maculopathy']):
                labels[1] = 0  # Patient has diabetes, but not this eye
        
        if patient_labels['G'] == 1 and labels[2] == 1:
            if not any(kw in keywords_lower for kw in ['glaucoma']):
                labels[2] = 0  # Patient has glaucoma, but not this eye
        
        if patient_labels['C'] == 1 and labels[3] == 1:
            if not any(kw in keywords_lower for kw in ['cataract', 'lens opacity']):
                labels[3] = 0  # Patient has cataract, but not this eye
        
        if patient_labels['M'] == 1 and labels[4] == 1:
            if not any(kw in keywords_lower for kw in ['myopia', 'myopic']):
                labels[4] = 0  # Patient has myopia, but not this eye
    
    # Final check: if no diseases marked, set as normal
    if labels[:5].sum() == 0 and labels[6] == 0:
        labels[5] = 1
    
    return labels


def load_labels(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and process ODIR-5K labels with INTELLIGENT PER-EYE labeling.
    
    Uses BOTH patient-level labels AND diagnostic keywords to intelligently determine
    which diseases apply to which specific eye. This handles cases like:
    - Patient has glaucoma, but only in one eye
    - One eye is normal while the other has disease
    - Different diseases in each eye
    
    Args:
        data_path: Path to Excel/CSV file with labels
        
    Returns:
        Tuple of (labels_dict, disease_columns)
    """
    # Handle both Excel and CSV formats
    if str(data_path).endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    
    labels = {}
    excluded_count = 0
    
    for _, row in df.iterrows():
        img_id = str(row['ID'])
        left_id = f"{img_id}_left"
        right_id = f"{img_id}_right"
        
        # Get patient-level labels (which diseases the patient has overall)
        patient_labels = {
            'A': row['A'],  # AMD
            'D': row['D'],  # Diabetes
            'G': row['G'],  # Glaucoma
            'C': row['C'],  # Cataract
            'M': row['M'],  # Myopia
            'N': row['N'],  # Normal
            'O': row['O']   # Other
        }
        
        # Get diagnostic keywords for each eye
        left_keywords = row['Left-Diagnostic Keywords']
        right_keywords = row['Right-Diagnostic Keywords']
        
        # Intelligent per-eye labeling
        if not is_low_quality(left_keywords):
            left_label = parse_intelligent_per_eye_labels(patient_labels, left_keywords, 'left')
            labels[left_id] = left_label
        else:
            excluded_count += 1
            
        if not is_low_quality(right_keywords):
            right_label = parse_intelligent_per_eye_labels(patient_labels, right_keywords, 'right')
            labels[right_id] = right_label
        else:
            excluded_count += 1
    
    print(f"  Excluded {excluded_count} low-quality images")
    print(f"  Using INTELLIGENT PER-EYE labeling (patient labels + diagnostic keywords)")
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
    print("PHASE 4C: RGB COLOR PRESERVATION + OPTIMIZED VESSEL ENHANCEMENT")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nKey Features:")
    print("  ✓ RGB color preservation (RED, GREEN, BLUE channels)")
    print("  ✓ Multi-scale vessel enhancement (5×5, 7×7, 9×9 kernels)")
    print("  ✓ AMD drusen enhancement (11×11 top-hat on GREEN)")
    print("  ✓ Adaptive CLAHE (adjusts to image brightness)")
    print("  ✓ 384×384 resolution (up from 224×224)")
    print("  ✓ CLAHE + bilateral filtering per channel")
    print("\nData Split Optimization:")
    print("  ✓ Multi-label stratified split (ALL 7 classes)")
    print("  ✓ 75/25 split (larger validation set)")
    print("  ✓ Preserves disease co-occurrence patterns")
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
    
    # Split train/val with multi-label stratification (75/25 for larger val set)
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    
    print("\nPerforming multi-label stratified split...")
    print("  ✓ Stratifying by ALL 7 disease classes")
    print("  ✓ Using 75/25 split (larger validation set for rare classes)")
    print("  ✓ Preserving disease co-occurrence patterns")
    
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, 
        test_size=0.25,  # 25% validation for larger sample size on rare classes
        random_state=42
    )
    
    train_idx, val_idx = next(msss.split(images_array, labels_array))
    
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
    disease_names = ['AMD', 'Diabetes', 'Glaucoma', 'Cataract', 'Myopia', 'Normal', 'Other']
    for i, name in enumerate(disease_names):
        count = train_labels[:, i].sum()
        pct = 100 * count / len(train_labels)
        print(f"  {name:15s}: {int(count):5d} ({pct:5.1f}%)")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\nNext steps:")
    print("  1. Training optimizations applied:")
    print("     • OneCycleLR scheduler (faster convergence)")
    print("     • Label smoothing (0.1 - better calibration)")
    print("     • Multi-scale vessel + drusen enhancement")
    print("     • Adaptive CLAHE (brightness-aware)")
    print("  2. Run training: M5_NUM_WORKERS=4 python scripts/train_advanced.py --model convnext_tiny --epochs 50 --batch-size 32 --use-mixup --use-amp --grad-accum-steps 2 --grad-clip 1.0")
    print("     (Note: Smaller batch size due to larger images)")
    print("  3. Expected improvement: +5-8% F1 from all optimizations combined")


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
