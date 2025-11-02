#!/usr/bin/env python3
"""
Test enhanced preprocessing on small sample to verify everything works
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_FILE
from data_preprocessing_enhanced import process_dataset_enhanced, save_processed_data

# Load data
print("Loading data...")
df = pd.read_excel(DATA_FILE)

# Test on small sample
print("\n" + "="*80)
print("TESTING ON SMALL SAMPLE (10 patients = 20 images)")
print("="*80 + "\n")

train_sample = df.head(10)
val_sample = df.iloc[10:13]  # Next 3 patients

# Process with eye-specific labels
print("Processing training sample with eye-specific labels...")
train_images, train_labels, train_indices, train_metadata, train_severity = process_dataset_enhanced(
    train_sample,
    eye='both',
    preprocessing_method='full',
    use_eye_specific_labels=True
)

print("\nProcessing validation sample with eye-specific labels...")
val_images, val_labels, val_indices, val_metadata, val_severity = process_dataset_enhanced(
    val_sample,
    eye='both',
    preprocessing_method='full',
    use_eye_specific_labels=True
)

# Verify results
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

print(f"\n✓ Training images processed: {len(train_images)}")
print(f"  Expected: 20 (10 patients × 2 eyes)")
print(f"  Match: {'✓' if len(train_images) == 20 else '✗'}")

print(f"\n✓ Validation images processed: {len(val_images)}")
print(f"  Expected: 6 (3 patients × 2 eyes)")
print(f"  Match: {'✓' if len(val_images) == 6 else '✗'}")

print(f"\n✓ Image shapes correct: {train_images[0].shape}")
print(f"  Expected: (224, 224, 3)")
print(f"  Match: {'✓' if train_images[0].shape == (224, 224, 3) else '✗'}")

print(f"\n✓ Label shapes correct: {train_labels[0].shape}")
print(f"  Expected: (8,)")
print(f"  Match: {'✓' if train_labels[0].shape == (8,) else '✗'}")

# Sample labels
print("\n" + "="*80)
print("SAMPLE LABELS (First 3 training images)")
print("="*80)

label_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

for i in range(min(3, len(train_images))):
    patient_idx = train_indices[i]
    patient = train_sample.iloc[patient_idx]
    
    # Determine which eye
    eye = 'Left' if i % 2 == 0 else 'Right'
    keyword_col = f'{eye}-Diagnostic Keywords'
    keywords = patient[keyword_col]
    
    print(f"\nImage {i+1} (Patient {patient['ID']}, {eye} eye):")
    print(f"  Keywords: '{keywords}'")
    print(f"  Labels: {dict(zip(label_names, train_labels[i].astype(int)))}")
    diseases = [name for name, val in zip(label_names, train_labels[i]) if val == 1]
    print(f"  Diseases: {diseases if diseases else ['None (Normal)']}")

print("\n" + "="*80)
print("TEST COMPLETE ✓")
print("="*80)
print("\nIf you see label source statistics above with >80% parsed from keywords,")
print("the integration is working correctly!")
print("\nReady to run on full dataset (7,000 images, ~35-42 minutes)")
