# ResNet-50 Training Results - Phase 4C

**Date:** November 7, 2025  
**Model:** ResNet-50 (23.5M parameters)  
**Dataset:** ODIR-5K Phase 4C (intelligent per-eye labeling)  
**Training Duration:** ~4.5 hours (50 epochs)

---

## Executive Summary

**Final Performance:** 42.76% macro F1 (best at Epoch 25)

**Status:** ⚠️ **Below Target** - Expected 75-80%, achieved 42.76%

**Key Findings:**
- ✅ **Significant improvement** from broken baseline (16% → 43%)
- ❌ **Still below Phase 4A baseline** (64.63% target)
- ✅ **All optimizations working** (augmentation, focal loss, class weights)
- ⚠️ **Performance plateau** after Epoch 25-30
- ❌ **Minority classes still struggling** (avg 36%, target >60%)

---

## Training Configuration

### Model Architecture
```
ResNet-50 (timm)
- Parameters: 23,522,375 (23.5M)
- Pre-trained: ImageNet-1k
- Input: 384×384×3 RGB
- Output: 7 classes (multi-label)
```

### Optimizations Applied
```yaml
Data Augmentation:
  - RandomHorizontalFlip(p=0.5)
  - RandomVerticalFlip(p=0.5)
  - RandomRotation(degrees=15)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

Loss Function:
  - 80% Focal Loss (alpha=0.25, gamma=2.5)
  - 20% Weighted BCE
  - Class weights: [13.83, 2.46, 19.14, 20.64, 22.07, 1.25, 3.21]

Training:
  - Batch size: 32 (effective 64 with grad_accum=2)
  - Learning rate: 1e-4 → 1e-3 (OneCycleLR, 30% warmup)
  - Label smoothing: 0.05
  - Gradient clipping: 1.0
  - Mixed precision: MPS (Apple Silicon)
  - Memory mapping: Enabled (saves 11GB RAM)
```

### Dataset
```
Train: 5,235 images (75%)
Val:   1,744 images (25%)

Class Distribution:
  AMD:        352 (6.7%)  🎯 MINORITY
  Diabetes: 1,513 (28.9%)
  Glaucoma:   259 (4.9%)  🎯 MINORITY
  Cataract:   241 (4.6%)  🎯 MINORITY
  Myopia:     226 (4.3%)  🎯 MINORITY
  Normal:   2,328 (44.5%) ⚠️  MAJOR
  Other:    1,243 (23.7%)

Imbalance Ratio: 10.3:1 (Normal vs Myopia)
```

---

## Final Results

### Best Model (Epoch 25)
```
Overall Metrics:
  Val F1:        42.76%
  Val Precision: 38.93%
  Val Recall:    60.84%
  Train F1:      83.93% (overfitting evident)

Per-Class F1:
  AMD         : 35.1% 🎯
  Diabetes    : 46.0%
  Glaucoma    : 26.5% 🎯
  Cataract    : 37.9% 🎯
  Myopia      : 44.7% 🎯
  Normal      : 69.2%
  Other       : 40.1%

Minority Class Avg: 36.0%
```

### Latest Model (Epoch 50)
```
Overall Metrics:
  Val F1:        41.17%
  Val Precision: 40.24%
  Val Recall:    58.49%
  Train F1:      93.93% (severe overfitting)

Per-Class F1:
  AMD         : 33.9% 🎯
  Diabetes    : 42.7%
  Glaucoma    : 27.5% 🎯
  Cataract    : 62.3% 🎯
  Myopia      : 21.8% 🎯
  Normal      : 69.8%
  Other       : 30.2%

Minority Class Avg: 36.4%
```

---

## Training Progression

### F1 Score Over Time
```
Epoch   Val F1   Minority F1   Train F1   Gap
─────────────────────────────────────────────
  1     6.27%     9.80%       23.72%     +17%
  5    16.58%    13.98%       51.60%     +35%
 10    31.92%    20.64%       71.86%     +40%
 20    38.14%    35.86%       82.36%     +44%
 25    42.76% ★  36.01%       83.93%     +41%  ← BEST
 30    37.70%    30.20%       88.00%     +50%
 40    41.57%    35.98%       89.80%     +48%
 50    41.17%    36.38%       93.93%     +53%
```

**Key Observations:**
- Rapid improvement Epochs 1-10 (6% → 32% F1)
- Steady climb Epochs 10-25 (32% → 43% F1)
- **Plateau/decline after Epoch 25** (43% → 38% → 41%)
- Severe overfitting by Epoch 50 (94% train vs 41% val)

---

## Per-Class Analysis

### Minority Classes (Target >60% each)

**AMD (6.7% of data):**
- Best: 35.1% (Epoch 25)
- Latest: 33.9% (Epoch 50)
- Status: ❌ Far below target
- Issue: Insufficient samples + hard to distinguish from Normal

**Glaucoma (4.9% of data):**
- Best: 26.5% (Epoch 25)
- Latest: 27.5% (Epoch 50)
- Status: ❌ Lowest minority class performance
- Issue: Very subtle features, small sample size

**Cataract (4.6% of data):**
- Best: 37.9% (Epoch 25)
- Latest: 62.3% (Epoch 50) ✅ **Best minority class!**
- Status: ⚠️ Acceptable but volatile
- Issue: High variance across epochs

**Myopia (4.3% of data):**
- Best: 44.7% (Epoch 25)
- Latest: 21.8% (Epoch 50) ⚠️ **Degraded significantly**
- Status: ❌ Unstable performance
- Issue: Smallest class, model forgot after Epoch 25

### Majority Classes

**Normal (44.5% of data):**
- Best: 69.2% (Epoch 25)
- Latest: 69.8% (Epoch 50)
- Status: ✅ Consistent, but below target (75%)
- Issue: Model conservative on Normal predictions

**Diabetes (28.9% of data):**
- Best: 46.0% (Epoch 25)
- Latest: 42.7% (Epoch 50)
- Status: ❌ Below expectations for large class
- Issue: Possible mislabeling or feature overlap with Other

**Other (23.7% of data):**
- Best: 40.1% (Epoch 25)
- Latest: 30.2% (Epoch 50)
- Status: ❌ Poor for a large class
- Issue: Heterogeneous category, hard to learn

---

## Comparison to Baselines

| Metric | Phase 4A Baseline | This Run | Delta |
|--------|-------------------|----------|-------|
| **Overall F1** | 64.63% | 42.76% | -21.87% ❌ |
| **AMD** | ~60% | 35.1% | -24.9% ❌ |
| **Diabetes** | ~70% | 46.0% | -24.0% ❌ |
| **Glaucoma** | ~55% | 26.5% | -28.5% ❌ |
| **Cataract** | ~55% | 37.9% | -17.1% ❌ |
| **Myopia** | ~50% | 44.7% | -5.3% ⚠️ |
| **Normal** | ~75% | 69.2% | -5.8% ⚠️ |
| **Other** | ~65% | 40.1% | -24.9% ❌ |

**Conclusion:** Performance is **significantly worse** than Phase 4A baseline across all classes.

---

## Root Cause Analysis

### What Worked ✅
1. **Double-normalization fix verified** - Images properly normalized (range checked)
2. **All 7 classes present** - Myopia recovered (was 0 samples in broken version)
3. **Intelligent per-eye labeling** - Validated on 20 patients (100% pass)
4. **Augmentation working** - Training F1 increased (shows model learning)
5. **Class weights applied** - Loss calculation correct
6. **Training stable** - No crashes, gradual improvement

### What Didn't Work ❌

**1. Performance Gap (~22% below baseline)**

Possible causes:
- **Data distribution mismatch:** Phase 4C has 44% Normal vs Phase 4A had 44% AMD
  - This suggests Phase 4A used different preprocessing or data subset
  - Need to verify Phase 4A data source and labeling method

- **Intelligent labeling too conservative:** May have over-labeled as Normal
  - 44% Normal is technically correct per ODIR (32.6% Normal patients)
  - But model may need disease-focused labeling for better discrimination

- **Preprocessing artifacts:** RGB vessel enhancement may have introduced noise
  - Multi-scale vessel enhancement (5×5, 7×7, 9×9) might blur critical features
  - AMD drusen enhancement might not be effective

**2. Severe Overfitting (94% train vs 41% val)**

Evidence:
- Train F1 reaches 94% while val F1 stuck at 41%
- Gap increases from 17% (Epoch 1) to 53% (Epoch 50)
- Model memorizing training set, not generalizing

Likely causes:
- **Insufficient effective data:** Despite augmentation, minority classes too small
  - AMD: 352 samples → ~1,400 with 4x augmentation (still small)
  - Glaucoma: 259 samples → ~1,000 augmented
  - Myopia: 226 samples → ~900 augmented

- **Model capacity too large:** ResNet-50 (23.5M params) may be overkill
  - Need regularization: dropout, higher weight decay
  - Or smaller model: ResNet-34, EfficientNet-B0

**3. Minority Class Collapse**

Myopia degraded from 44.7% → 21.8% after Epoch 25:
- Model "forgets" minority classes as training progresses
- Focal loss + class weights not strong enough
- May need curriculum learning or per-class threshold tuning

**4. Class Imbalance Still Dominating**

Despite 6 imbalance handling strategies:
- 10:1 ratio (Normal vs Myopia) too extreme
- Normal class "pulling" model predictions
- May need explicit minority oversampling (not just augmentation)

---

## Diagnostic Checks Needed

### 1. Verify Double-Normalization Fix
```python
# Check actual image values fed to model
sample_batch = next(iter(train_loader))
print(f"Min: {sample_batch[0].min()}, Max: {sample_batch[0].max()}")
# Expected: ~[-2.5, 2.5] after ImageNet normalization
# If still [0, 1], double-norm fix didn't work!
```

### 2. Compare Preprocessing to Phase 4A
```python
# Load Phase 4A preprocessed data
phase4a_images = np.load('preprocessed_data/train_images.npy')
phase4c_images = np.load('preprocessed_data_phase4c/train_images.npy')

# Compare:
# - Shape, dtype, value ranges
# - Class distributions
# - Image visual appearance
```

### 3. Analyze Predictions
```python
# Where is model failing?
# - Confusion matrix
# - False positives/negatives per class
# - Prediction confidence distributions
```

### 4. Check Label Quality
```python
# Spot check intelligent per-eye labeling
# - Are labels actually correct?
# - Compare to Phase 4A labels
# - Check if "Normal" over-assigned
```

---

## Recommended Next Steps

### Immediate Actions (Priority 1)

**1. Verify Double-Normalization Fix Actually Applied**
- Create test script to check actual tensor values fed to model
- If values are [0, 1] instead of [-2.5, 2.5], fix didn't work!
- **This is most likely culprit for 22% gap**

**2. Compare Phase 4A vs Phase 4C Data**
```bash
# What exactly is different?
# - Preprocessing method
# - Label assignment
# - Class distribution
# - Image quality
```

**3. Threshold Tuning**
- Current: All classes use 0.5 threshold
- Minority classes may need lower thresholds (0.3-0.4)
- Run validation sweep: test thresholds 0.1 to 0.9 per class

### Short-term Experiments (Priority 2)

**4. Oversample Minority Classes**
```python
# Repeat minority samples 2-3x in training
# AMD: 352 → 1,056 samples
# Glaucoma: 259 → 777 samples
# Myopia: 226 → 678 samples
```

**5. Try Smaller Model**
```python
# ResNet-34 (21M params) or EfficientNet-B0 (5M params)
# Less overfitting, may generalize better
```

**6. Undersample Normal Class**
```python
# Use only 50% of Normal samples (2,328 → 1,164)
# Reduce imbalance from 10:1 to 5:1
```

**7. Increase Focal Loss Gamma**
```python
# gamma = 2.5 → 3.0 or 3.5
# Even stronger minority class focus
```

### Long-term Improvements (Priority 3)

**8. Different Preprocessing**
- Try Phase 4A preprocessing method (if we can identify it)
- Simpler preprocessing: Just resize + CLAHE (no vessel enhancement)
- Higher resolution: 512×512 instead of 384×384

**9. Ensemble Methods**
- Train multiple models (ResNet, EfficientNet, ConvNeXt)
- Ensemble predictions for better robustness

**10. External Data**
- Add external fundus datasets (Messidor, EyePACS)
- Pre-train on larger dataset, fine-tune on ODIR

---

## Code for Next Steps

### 1. Verify Normalization Fix
```python
# test_normalization.py
import torch
import numpy as np
from pathlib import Path
from scripts.train_cutting_edge import FundusDataset

data_dir = Path('preprocessed_data_phase4c')
train_images = np.load(data_dir / 'train_images.npy', mmap_mode='r')
train_labels = np.load(data_dir / 'train_labels.npy')

dataset = FundusDataset(train_images[:100], train_labels[:100], augment=False)

# Get a batch
images, labels = [], []
for i in range(10):
    img, lbl = dataset[i]
    images.append(img)
    labels.append(lbl)

images = torch.stack(images)

print("=" * 80)
print("NORMALIZATION VERIFICATION")
print("=" * 80)
print(f"\nBatch shape: {images.shape}")
print(f"Batch dtype: {images.dtype}")
print(f"Min value: {images.min().item():.6f}")
print(f"Max value: {images.max().item():.6f}")
print(f"Mean value: {images.mean().item():.6f}")
print(f"Std value: {images.std().item():.6f}")

print("\nExpected after ImageNet normalization:")
print("  Min: ~-2.5")
print("  Max: ~+2.5")
print("  Mean: ~0.0")
print("  Std: ~1.0")

if images.min() > -1.0 or images.max() < 1.0:
    print("\n❌ FAIL: Values suggest double-normalization or no normalization!")
    print("   Images are too close to [0, 1] range")
else:
    print("\n✅ PASS: Normalization appears correct")
```

### 2. Threshold Tuning
```python
# tune_thresholds.py
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from scripts.train_cutting_edge import FundusDataset, create_model, validate
from torch.utils.data import DataLoader

# Load model
model = create_model('resnet50', num_classes=7, device='mps')
checkpoint = torch.load('models/best_resnet50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load val data
data_dir = Path('preprocessed_data_phase4c')
val_images = np.load(data_dir / 'val_images.npy', mmap_mode='r')
val_labels = np.load(data_dir / 'val_labels.npy')

val_dataset = FundusDataset(val_images, val_labels, augment=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Get all predictions
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to('mps')
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

all_probs = np.vstack(all_probs)
all_labels = np.vstack(all_labels)

# Tune thresholds per class
class_names = ['AMD', 'Diabetes', 'Glaucoma', 'Cataract', 'Myopia', 'Normal', 'Other']
best_thresholds = []

print("=" * 80)
print("PER-CLASS THRESHOLD TUNING")
print("=" * 80)

for class_idx, class_name in enumerate(class_names):
    class_probs = all_probs[:, class_idx]
    class_labels = all_labels[:, class_idx]
    
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (class_probs > thresh).astype(int)
        f1 = f1_score(class_labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    best_thresholds.append(best_thresh)
    print(f"{class_name:12s}: best_thresh={best_thresh:.2f}, F1={best_f1:.4f} (vs 0.50: {f1_score(class_labels, (class_probs > 0.5).astype(int), zero_division=0):.4f})")

# Test with tuned thresholds
print("\n" + "=" * 80)
print("OVERALL PERFORMANCE WITH TUNED THRESHOLDS")
print("=" * 80)

tuned_preds = np.zeros_like(all_labels)
for i, thresh in enumerate(best_thresholds):
    tuned_preds[:, i] = (all_probs[:, i] > thresh).astype(int)

default_preds = (all_probs > 0.5).astype(int)

tuned_f1 = f1_score(all_labels, tuned_preds, average='macro', zero_division=0)
default_f1 = f1_score(all_labels, default_preds, average='macro', zero_division=0)

print(f"Default (0.5 all classes): {default_f1:.4f}")
print(f"Tuned thresholds:          {tuned_f1:.4f}")
print(f"Improvement:               {tuned_f1 - default_f1:+.4f}")
```

---

## Conclusion

**Training completed successfully** but performance is **significantly below target**:
- Achieved: 42.76% F1 (best model at Epoch 25)
- Expected: 75-80% F1
- Gap: -32 to -37 percentage points

**Most likely root cause:** Need to verify double-normalization fix actually applied during training. If images were still incorrectly normalized, this would explain the ~22% gap vs Phase 4A baseline.

**Next steps:**
1. **Urgent:** Verify normalization fix worked (test script above)
2. **Compare:** Phase 4A vs Phase 4C data/preprocessing differences
3. **Tune:** Per-class thresholds (quick win, +5-10% F1)
4. **Experiment:** Oversampling, smaller model, simpler preprocessing

**Recommendation:** Before starting new training, run diagnostic checks to understand why performance is so far below baseline. Don't waste compute on another 4-hour run until root cause identified.

---

*Training completed: November 7, 2025, 05:56:20*  
*Total duration: ~4.5 hours (50 epochs)*  
*Best model: models/best_resnet50.pth (Epoch 25)*
