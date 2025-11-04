#!/usr/bin/env python3
"""
Run Enhanced Preprocessing with All Recommended Features

This script implements Phase 2 improvements:
1. Eye-specific label parsing (+2-4% F1)
2. Advanced preprocessing (+1-2% F1)
   - Green channel extraction
   - Illumination correction
   - ROI extraction
   - Enhanced CLAHE
3. Metadata extraction (age/gender)
4. Low quality image exclusion
5. Severity level extraction

Expected improvement: +3-6% F1 (from 85.28% → 90-93%)
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import shutil
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import CSV_PATH, TRAIN_IMAGE_DIR, LABEL_COLUMNS, DEFAULT_IMAGE_SIZE, RANDOM_SEED
from src.data_preprocessing_enhanced import process_dataset_enhanced
from sklearn.model_selection import train_test_split


def backup_current_data():
    """Backup existing preprocessed data before reprocessing."""
    print("\n" + "="*80)
    print("🔒 BACKING UP CURRENT PREPROCESSED DATA")
    print("="*80)
    
    preprocessed_path = Path("preprocessed_data")
    
    if not preprocessed_path.exists():
        print("⚠️  No existing preprocessed data found. Nothing to backup.")
        return None
    
    # Create timestamped backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = Path(f"preprocessed_data_backup_{timestamp}")
    
    print(f"📦 Creating backup: {backup_path}")
    shutil.copytree(preprocessed_path, backup_path)
    
    # Check sizes
    backup_size = sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file())
    print(f"✅ Backup created: {backup_size / (1024**3):.2f} GB")
    print(f"   Location: {backup_path.absolute()}")
    
    return backup_path


def compare_preprocessing(old_images, new_images, old_labels, new_labels):
    """Compare old vs new preprocessing results."""
    print("\n" + "="*80)
    print("📊 COMPARING OLD vs NEW PREPROCESSING")
    print("="*80)
    
    print("\n1. IMAGE STATISTICS:")
    print(f"   Old images shape: {old_images.shape}")
    print(f"   New images shape: {new_images.shape}")
    print(f"   Old value range: [{old_images.min():.3f}, {old_images.max():.3f}]")
    print(f"   New value range: [{new_images.min():.3f}, {new_images.max():.3f}]")
    print(f"   Old mean: {old_images.mean():.3f} ± {old_images.std():.3f}")
    print(f"   New mean: {new_images.mean():.3f} ± {new_images.std():.3f}")
    
    print("\n2. LABEL STATISTICS:")
    print(f"   Old labels shape: {old_labels.shape}")
    print(f"   New labels shape: {new_labels.shape}")
    print(f"   Old label distribution: {old_labels.sum(axis=0)}")
    print(f"   New label distribution: {new_labels.sum(axis=0)}")
    
    # Calculate label changes
    old_total = old_labels.sum()
    new_total = new_labels.sum()
    change_pct = ((new_total - old_total) / old_total) * 100
    
    print(f"\n3. LABEL CHANGES:")
    print(f"   Old total labels: {int(old_total)}")
    print(f"   New total labels: {int(new_total)}")
    print(f"   Change: {int(new_total - old_total)} ({change_pct:+.1f}%)")
    
    # Per-disease changes
    disease_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
    print("\n4. PER-DISEASE CHANGES:")
    for i, disease in enumerate(disease_names[:7]):  # Only 7 diseases in LABEL_COLUMNS
        old_count = old_labels[:, i].sum()
        new_count = new_labels[:, i].sum()
        change = new_count - old_count
        change_pct = ((new_count - old_count) / max(old_count, 1)) * 100
        print(f"   {disease:12s}: {int(old_count):5d} → {int(new_count):5d} ({change:+5.0f}, {change_pct:+6.1f}%)")


def save_metadata(metadata_list, severity_list, output_dir: Path):
    """Save extracted metadata and severity information."""
    print("\n" + "="*80)
    print("💾 SAVING METADATA AND SEVERITY INFO")
    print("="*80)
    
    if len(metadata_list) > 0:
        metadata_array = np.array(metadata_list)
        metadata_path = output_dir / "metadata.npy"
        np.save(metadata_path, metadata_array)
        print(f"✅ Metadata saved: {metadata_path} (shape: {metadata_array.shape})")
        print(f"   Contains: age, gender for each image")
    
    if len(severity_list) > 0:
        severity_path = output_dir / "severity_info.json"
        with open(severity_path, 'w') as f:
            json.dump(severity_list, f, indent=2)
        print(f"✅ Severity info saved: {severity_path} ({len(severity_list)} records)")


def main():
    """Main preprocessing pipeline."""
    print("\n" + "="*80)
    print("🚀 ENHANCED PREPROCESSING WITH ALL RECOMMENDED FEATURES")
    print("="*80)
    print("\nPhase 2: Data Improvements")
    print("Expected improvement: +3-6% F1 (from 85.28% → 90-93%)")
    print("\nFeatures enabled:")
    print("  ✓ Eye-specific label parsing (biggest impact: +2-4% F1)")
    print("  ✓ Green channel extraction")
    print("  ✓ Illumination correction")
    print("  ✓ Enhanced CLAHE")
    print("  ✓ ROI extraction")
    print("  ✓ Low quality image exclusion")
    print("  ✓ Age/gender metadata extraction")
    print("  ✓ Severity level extraction")
    print("\n" + "="*80)
    
    # Step 1: Backup current data
    backup_path = backup_current_data()
    
    # Step 2: Load original data file (Excel format)
    print("\n" + "="*80)
    print("📂 LOADING ORIGINAL DATA")
    print("="*80)
    print(f"Loading data file: {CSV_PATH}")
    # Handle both .xlsx and .csv files
    if str(CSV_PATH).endswith('.xlsx'):
        df = pd.read_excel(CSV_PATH)
    else:
        df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded {len(df)} patients")
    
    # Step 3: Load old preprocessed data for comparison
    print("\n" + "="*80)
    print("📥 LOADING OLD PREPROCESSED DATA (for comparison)")
    print("="*80)
    old_train_images = np.load('preprocessed_data/train_images.npy')
    old_train_labels = np.load('preprocessed_data/train_labels.npy')
    old_val_images = np.load('preprocessed_data/val_images.npy')
    old_val_labels = np.load('preprocessed_data/val_labels.npy')
    print(f"✅ Old train: {old_train_images.shape} images, {old_train_labels.shape} labels")
    print(f"✅ Old val:   {old_val_images.shape} images, {old_val_labels.shape} labels")
    
    # Step 4: Split data (same as before for fair comparison)
    print("\n" + "="*80)
    print("✂️  SPLITTING DATA (80/20 train/val)")
    print("="*80)
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=RANDOM_SEED,
        stratify=df['N']  # Stratify by Normal class
    )
    print(f"✅ Train: {len(train_df)} patients")
    print(f"✅ Val:   {len(val_df)} patients")
    
    # Step 5: Process training data with ALL advanced features
    print("\n" + "="*80)
    print("🔬 PROCESSING TRAINING DATA (with ALL enhancements)")
    print("="*80)
    print("This will take ~10-15 minutes...")
    
    train_images, train_labels, train_indices, train_metadata, train_severity = process_dataset_enhanced(
        train_df,
        eye='both',  # Process both eyes
        image_dir=TRAIN_IMAGE_DIR,
        target_size=DEFAULT_IMAGE_SIZE,
        preprocessing_method='full',  # ✓ Green channel + illumination + CLAHE
        apply_augmentation=False,  # No augmentation for stored data (done during training)
        use_eye_specific_labels=True,  # ✓ Eye-specific parsing (+2-4% F1)
        exclude_low_quality=True,  # ✓ Exclude poor quality images
        extract_metadata=True,  # ✓ Extract age/gender
        extract_severity=True  # ✓ Extract severity levels
    )
    
    print(f"\n✅ Processed {len(train_images)} training images")
    print(f"   Shape: {np.array(train_images).shape}")
    
    # Step 6: Process validation data with same enhancements
    print("\n" + "="*80)
    print("🔬 PROCESSING VALIDATION DATA (with ALL enhancements)")
    print("="*80)
    print("This will take ~3-5 minutes...")
    
    val_images, val_labels, val_indices, val_metadata, val_severity = process_dataset_enhanced(
        val_df,
        eye='both',
        image_dir=TRAIN_IMAGE_DIR,
        target_size=DEFAULT_IMAGE_SIZE,
        preprocessing_method='full',  # Same as training
        apply_augmentation=False,
        use_eye_specific_labels=True,
        exclude_low_quality=True,
        extract_metadata=True,
        extract_severity=True
    )
    
    print(f"\n✅ Processed {len(val_images)} validation images")
    print(f"   Shape: {np.array(val_images).shape}")
    
    # Step 7: Convert to numpy arrays
    print("\n" + "="*80)
    print("🔄 CONVERTING TO NUMPY ARRAYS")
    print("="*80)
    
    train_images = np.array(train_images, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)
    val_images = np.array(val_images, dtype=np.float32)
    val_labels = np.array(val_labels, dtype=np.float32)
    
    print(f"✅ Train images: {train_images.shape}, dtype: {train_images.dtype}")
    print(f"✅ Train labels: {train_labels.shape}, dtype: {train_labels.dtype}")
    print(f"✅ Val images:   {val_images.shape}, dtype: {val_images.dtype}")
    print(f"✅ Val labels:   {val_labels.shape}, dtype: {val_labels.dtype}")
    
    # Step 8: Compare old vs new
    compare_preprocessing(old_train_images, train_images, old_train_labels, train_labels)
    
    # Step 9: Save new preprocessed data
    print("\n" + "="*80)
    print("💾 SAVING ENHANCED PREPROCESSED DATA")
    print("="*80)
    
    output_dir = Path("preprocessed_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save images and labels
    np.save(output_dir / 'train_images.npy', train_images)
    np.save(output_dir / 'train_labels.npy', train_labels)
    np.save(output_dir / 'val_images.npy', val_images)
    np.save(output_dir / 'val_labels.npy', val_labels)
    
    print(f"✅ Saved: train_images.npy ({train_images.nbytes / (1024**3):.2f} GB)")
    print(f"✅ Saved: train_labels.npy ({train_labels.nbytes / (1024**2):.2f} MB)")
    print(f"✅ Saved: val_images.npy ({val_images.nbytes / (1024**3):.2f} GB)")
    print(f"✅ Saved: val_labels.npy ({val_labels.nbytes / (1024**2):.2f} MB)")
    
    # Save metadata and severity info
    save_metadata(train_metadata, train_severity, output_dir)
    
    # Step 10: Save preprocessing info
    preprocessing_info = {
        'timestamp': datetime.now().isoformat(),
        'method': 'full',
        'features': {
            'eye_specific_labels': True,
            'green_channel': True,
            'illumination_correction': True,
            'enhanced_clahe': True,
            'roi_extraction': True,
            'exclude_low_quality': True,
            'metadata_extraction': True,
            'severity_extraction': True
        },
        'statistics': {
            'train_images': int(len(train_images)),
            'val_images': int(len(val_images)),
            'train_label_distribution': train_labels.sum(axis=0).tolist(),
            'val_label_distribution': val_labels.sum(axis=0).tolist(),
            'old_train_images': int(len(old_train_images)),
            'old_val_images': int(len(old_val_images)),
        },
        'backup_location': str(backup_path) if backup_path else None
    }
    
    info_path = output_dir / 'preprocessing_info.json'
    with open(info_path, 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    print(f"✅ Saved: preprocessing_info.json")
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 ENHANCED PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\n✅ All data saved to: {output_dir.absolute()}")
    if backup_path:
        print(f"✅ Backup available at: {backup_path.absolute()}")
    print(f"\n📊 Summary:")
    print(f"   Training:   {len(train_images):,} images")
    print(f"   Validation: {len(val_images):,} images")
    print(f"   Total:      {len(train_images) + len(val_images):,} images")
    print(f"\n🎯 Next Steps:")
    print(f"   1. Retrain all 3 models with enhanced data")
    print(f"   2. Optimize thresholds")
    print(f"   3. Create new ensemble")
    print(f"   4. Compare results: 85.28% F1 (baseline) → expected 90-93% F1")
    print(f"\n💡 Expected improvements:")
    print(f"   - Eye-specific labels: +2-4% F1")
    print(f"   - Advanced preprocessing: +1-2% F1")
    print(f"   - Total expected: +3-6% F1")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
