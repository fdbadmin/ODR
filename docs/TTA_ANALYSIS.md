# TTA Analysis - Why It Didn't Work

## Results Summary

**Baseline (No TTA):** 72.26% F1  
**With TTA:** 70.77% F1  
**Change:** **-1.49% F1** ❌

## Why TTA Failed

### Hypothesis 1: Over-Averaging Dilutes Confident Predictions

The models were trained on **specific orientations** and when we average across 8 augmentations, we're including views the models were never trained to handle well.

**Evidence:**
- All disease classes decreased (AMD, Diabetes, Glaucoma, Cataract, Myopia)
- Only Normal and Other improved slightly (+0.46%, +0.16%)
- This suggests the augmentations are confusing the models for disease detection

### Hypothesis 2: Fundus Images Are NOT Rotation-Invariant

Unlike natural images (cats, dogs), **fundus images have anatomical structure**:
- Optic disc is typically on one side
- Macula is in a specific location
- Blood vessel patterns have directionality

**Rotating 90°/180°/270° breaks this anatomical prior!**

### Hypothesis 3: Training Augmentation Mismatch

Our training augmentation:
```python
T.RandomHorizontalFlip(p=0.5)  # ✅ Mirrors
T.RandomVerticalFlip(p=0.5)    # ✅ Mirrors  
T.RandomRotation(degrees=15)   # ✅ Small rotations only
```

TTA augmentation:
```python
- 90° rotation   # ❌ Never seen during training!
- 180° rotation  # ❌ Never seen during training!
- 270° rotation  # ❌ Never seen during training!
```

**Models trained with ±15° rotation can't handle 90° rotations!**

## What Works vs What Doesn't

### ✅ Likely to Help:
- **Horizontal flip** (left/right eye symmetry)
- **Vertical flip** (some anatomical symmetry)
- **Small rotations** (±15°, matching training)
- **Minor brightness/contrast shifts**

### ❌ Likely to Hurt:
- **90°/180°/270° rotations** (breaks anatomy)
- **Large color shifts** (changes disease appearance)
- **Extreme crops** (loses anatomical context)

## Corrected TTA Strategy

### Strategy 1: Conservative TTA (Only Safe Augmentations)
```python
augmentations = [
    original,
    horizontal_flip,
    vertical_flip,
    both_flips
]
# 4 views instead of 8 → faster + potentially more accurate
```

### Strategy 2: Weighted TTA
```python
# Give more weight to original orientation
weights = {
    'original': 0.5,      # 50% weight
    'h_flip': 0.2,        # 20% weight
    'v_flip': 0.2,        # 20% weight
    'both_flips': 0.1     # 10% weight
}
```

### Strategy 3: Selective TTA (Per-Class)
```python
# Use TTA only for classes that benefit
tta_beneficial_classes = ['Normal', 'Other']  # These improved with TTA
standard_classes = ['AMD', 'Diabetes', 'Glaucoma', 'Cataract', 'Myopia']
```

## Lessons Learned

1. **TTA is NOT universally beneficial** - especially for medical images with anatomical structure
2. **Test augmentation should match training augmentation** - don't introduce transformations the model never saw
3. **Fundus images have directional structure** - unlike natural images that are more rotation-invariant
4. **Always measure before deploying** - even "obvious" improvements need validation

## Next Steps

### Immediate Actions:
1. ~~Try TTA~~ ❌ **Skip TTA** (confirmed it hurts performance)
2. Move to **Priority 2: Class-Balanced Sampling** (+3-5% expected)
3. Implement **Asymmetric Loss** (+2-3% expected)

### Alternative to TTA:
Instead of TTA at inference, improve **training-time augmentation**:
- More aggressive flips (p=0.7 instead of 0.5)
- Stronger color jitter
- Add elastic deformations
- Add MixUp/CutMix

**Expected gain from better training augmentation: +2-3% F1**

## Conclusion

TTA decreased performance because:
1. Large rotations (90°+) break anatomical structure
2. Models weren't trained to handle those orientations
3. Over-averaging dilutes confident disease predictions

**Recommendation:** Skip TTA, focus on training improvements instead.

