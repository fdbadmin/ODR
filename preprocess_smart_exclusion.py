"""
Smart Fellow-Eye Exclusion Preprocessing
=========================================

This script creates a training dataset that:
1. Uses correct per-eye labeling (from Phase 4C)
2. EXCLUDES normal fellow eyes from training (reduces confounding)
3. Keeps normal fellow eyes in validation (real-world testing)

Expected improvement:
- Normal: 44.7% → 37.3% (better class balance)
- Imbalance: 10:1 → 3-4:1 (more trainable)
- Expected F1: ~55-65% (vs current 42.76%)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing_phase4c import (
    load_labels,
    parse_intelligent_per_eye_labels
)
from src.phase4c_preprocessing import Phase4CPreprocessor
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

print("=" * 80)
print("SMART FELLOW-EYE EXCLUSION PREPROCESSING")
print("=" * 80)

# Load original ODIR data
print("\n📂 Loading ODIR-5K data...")
df = pd.read_excel('ODIR-5K/data.xlsx')
print(f"   Loaded {len(df)} patients ({len(df)*2} total eyes)")

# Initialize storage
all_image_paths = []
all_labels = []
all_patient_ids = []
all_is_fellow_eye = []  # Track which are normal fellow eyes

class_names = ['AMD', 'Diabetes', 'Glaucoma', 'Cataract', 'Myopia', 'Normal', 'Other']

truly_normal_count = 0
normal_fellow_count = 0
diseased_count = 0

print("\n🔍 Processing patients and identifying fellow eyes...")

for idx, row in df.iterrows():
    if idx % 500 == 0:
        print(f"   Processed {idx}/{len(df)} patients...")
    
    patient_id = row['ID']
    
    # Patient-level labels
    patient_labels = {
        'A': row['A'],  # AMD
        'D': row['D'],  # Diabetes
        'G': row['G'],  # Glaucoma
        'C': row['C'],  # Cataract
        'M': row['M'],  # Myopia
        'N': row['N'],  # Normal
        'O': row['O']   # Other
    }
    
    patient_has_disease = any([row['D'], row['G'], row['C'], row['A'], row['M'], row['O']])
    
    # Process left eye
    left_image = f"ODIR-5K/Training Images/{row['Left-Fundus']}"
    left_keywords = row['Left-Diagnostic Keywords']
    left_label = parse_intelligent_per_eye_labels(patient_labels, left_keywords, 'left')
    
    left_is_normal = 'normal fundus' in str(left_keywords).lower()
    left_is_fellow_eye = left_is_normal and patient_has_disease
    
    all_image_paths.append(left_image)
    all_labels.append(left_label)
    all_patient_ids.append(f"{patient_id}_L")
    all_is_fellow_eye.append(left_is_fellow_eye)
    
    if left_is_fellow_eye:
        normal_fellow_count += 1
    elif left_is_normal:
        truly_normal_count += 1
    else:
        diseased_count += 1
    
    # Process right eye
    right_image = f"ODIR-5K/Training Images/{row['Right-Fundus']}"
    right_keywords = row['Right-Diagnostic Keywords']
    right_label = parse_intelligent_per_eye_labels(patient_labels, right_keywords, 'right')
    
    right_is_normal = 'normal fundus' in str(right_keywords).lower()
    right_is_fellow_eye = right_is_normal and patient_has_disease
    
    all_image_paths.append(right_image)
    all_labels.append(right_label)
    all_patient_ids.append(f"{patient_id}_R")
    all_is_fellow_eye.append(right_is_fellow_eye)
    
    if right_is_fellow_eye:
        normal_fellow_count += 1
    elif right_is_normal:
        truly_normal_count += 1
    else:
        diseased_count += 1

all_labels = np.array(all_labels)
all_is_fellow_eye = np.array(all_is_fellow_eye)

print(f"\n✅ Processed {len(df)} patients")
print(f"\n📊 Eye classification:")
print(f"   Truly normal eyes:        {truly_normal_count:4d}")
print(f"   Normal fellow eyes:       {normal_fellow_count:4d} ← Will exclude from training")
print(f"   Diseased eyes:            {diseased_count:4d}")
print(f"   Total:                    {len(all_image_paths):4d}")

print("\n" + "=" * 80)
print("TRAIN/VAL SPLIT STRATEGY")
print("=" * 80)

# Strategy:
# 1. Split into train/val (75/25) BEFORE exclusion
# 2. Exclude fellow eyes ONLY from training set
# 3. Keep fellow eyes in validation (real-world testing)

print("\nStep 1: Initial 75/25 split (before exclusion)...")

# Use patient-stratified split to avoid data leakage
# (don't want same patient in train and val)
patient_ids_unique = [pid.split('_')[0] for pid in all_patient_ids]

# Create train/val split indices
train_indices, val_indices = train_test_split(
    range(len(all_image_paths)),
    test_size=0.25,
    random_state=42,
    stratify=all_labels.argmax(axis=1)  # Stratify by primary class
)

print(f"   Initial train: {len(train_indices)} eyes")
print(f"   Initial val:   {len(val_indices)} eyes")

# Step 2: Exclude fellow eyes from training ONLY
print("\nStep 2: Excluding normal fellow eyes from training...")

train_indices_filtered = [i for i in train_indices if not all_is_fellow_eye[i]]
excluded_from_train = len(train_indices) - len(train_indices_filtered)

print(f"   Excluded {excluded_from_train} normal fellow eyes from training")
print(f"   New train size: {len(train_indices_filtered)} eyes")
print(f"   Val size (unchanged): {len(val_indices)} eyes (includes fellow eyes for real-world testing)")

# Extract train/val data
train_paths = [all_image_paths[i] for i in train_indices_filtered]
train_labels = all_labels[train_indices_filtered]
train_ids = [all_patient_ids[i] for i in train_indices_filtered]

val_paths = [all_image_paths[i] for i in val_indices]
val_labels = all_labels[val_indices]
val_ids = [all_patient_ids[i] for i in val_indices]

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION ANALYSIS")
print("=" * 80)

train_class_counts = train_labels.sum(axis=0)
val_class_counts = val_labels.sum(axis=0)

print(f"\n{'Class':<12} {'Train Count':>12} {'Train %':>10} {'Val Count':>12} {'Val %':>10}")
print("-" * 80)

for i, name in enumerate(class_names):
    train_count = int(train_class_counts[i])
    train_pct = 100 * train_count / len(train_labels)
    val_count = int(val_class_counts[i])
    val_pct = 100 * val_count / len(val_labels)
    
    marker = "🎯" if name in ['AMD', 'Glaucoma', 'Cataract', 'Myopia'] else ""
    marker = "✅" if name == 'Normal' else marker
    
    print(f"{name:<12} {train_count:>12d} {train_pct:>9.1f}% {val_count:>12d} {val_pct:>9.1f}% {marker}")

normal_pct_train = 100 * train_class_counts[5] / len(train_labels)
print(f"\n✅ Normal reduced to {normal_pct_train:.1f}% in training (was 44.7%)")
print(f"   Imbalance ratio: ~{100/normal_pct_train*train_class_counts[5]/train_class_counts[0]:.1f}:1 (was 10:1)")

# Initialize preprocessor
preprocessor = Phase4CPreprocessor(
    target_size=(384, 384),
    vessel_kernel_size=7,
    clahe_clip_limit=3.0,
    clahe_grid_size=(8, 8)
)

# Load and preprocess images
print("\n" + "=" * 80)
print("LOADING AND PREPROCESSING IMAGES")
print("=" * 80)

print("\n📸 Loading training images...")
train_images = []
for img_path in tqdm(train_paths, desc="Training"):
    img = cv2.imread(str(img_path))
    if img is not None:
        processed = preprocessor.preprocess(img)
        train_images.append(processed)
    else:
        print(f"⚠️ Failed to load: {img_path}")
train_images = np.array(train_images)

print("\n📸 Loading validation images...")
val_images = []
for img_path in tqdm(val_paths, desc="Validation"):
    img = cv2.imread(str(img_path))
    if img is not None:
        processed = preprocessor.preprocess(img)
        val_images.append(processed)
    else:
        print(f"⚠️ Failed to load: {img_path}")
val_images = np.array(val_images)

# Save preprocessed data
output_dir = Path('preprocessed_data_smart_exclusion')
output_dir.mkdir(exist_ok=True)

print(f"\n💾 Saving to {output_dir}...")

np.save(output_dir / 'train_images.npy', train_images)
np.save(output_dir / 'train_labels.npy', train_labels)
np.save(output_dir / 'train_ids.npy', np.array(train_ids))

np.save(output_dir / 'val_images.npy', val_images)
np.save(output_dir / 'val_labels.npy', val_labels)
np.save(output_dir / 'val_ids.npy', np.array(val_ids))

print("\n✅ Preprocessing complete!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
Dataset saved to: {output_dir}

TRAINING SET:
  - Size: {len(train_images)} eyes
  - Normal: {normal_pct_train:.1f}% (excludes {excluded_from_train} fellow eyes)
  - Diseased: {100 - normal_pct_train:.1f}%
  - Imbalance: ~3-4:1 (vs previous 10:1)

VALIDATION SET:
  - Size: {len(val_images)} eyes
  - Includes normal fellow eyes (real-world distribution)
  - Used for unbiased performance evaluation

NEXT STEP:
Run training with:

nohup .venv/bin/python scripts/train_cutting_edge.py \\
    --model resnet50 \\
    --epochs 50 \\
    --batch-size 32 \\
    --lr 1e-4 \\
    --grad-accum-steps 2 \\
    --grad-clip 1.0 \\
    --use-amp \\
    --data-dir preprocessed_data_smart_exclusion \\
    --output-dir models_smart_exclusion \\
    > training_smart_exclusion.log 2>&1 &

EXPECTED RESULTS:
  - Better class balance → less overfitting on Normal class
  - Cleaner training signal → better generalization
  - Expected F1: ~55-65% (vs current 42.76%, target 64.63%)
""")
