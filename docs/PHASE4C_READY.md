# Phase 4C: RGB Color Preservation + Optimized Vessel Enhancement

**Branch:** `phase4C-rgb-highres`  
**Status:** Ready to start after Phase 4B completes  
**Expected F1:** 70-76% (target: +6-12% over Phase 4A's 64.63%)

---

## Key Improvements Over Phase 4A & 4B

### 1. **RGB Color Preservation** ✨ NEW
- **Phase 4A/4B:** Converted to grayscale (green channel only)
- **Phase 4C:** Keeps all RGB color information
- **Why:** Critical for detecting:
  - **AMD drusen** (yellowish deposits)
  - **DR hard exudates** (yellow/white spots)
  - **Hemorrhages** (red in red channel)

### 2. **Selective Vessel Enhancement** ✨ IMPROVED
- **Phase 4B:** Applied to grayscale image with 15×15 kernel
- **Phase 4C:** Applied to GREEN channel only with 7×7 kernel
- **Why:** 
  - Preserves color info in R/B channels
  - Smaller kernel better for fine details
  - Less likely to obscure subtle lesions

### 3. **Higher Resolution** ✨ NEW
- **Phase 4A/4B:** 224×224 pixels
- **Phase 4C:** 384×384 pixels (73% increase in resolution)
- **Why:** Better detection of:
  - Small microaneurysms (1-5 pixels)
  - Fine drusen in AMD
  - Subtle vessel changes

### 4. **Per-Channel Processing** ✨ NEW
Each channel processed independently:
- **RED:** Illumination + CLAHE + Bilateral → Preserves hemorrhages
- **GREEN:** Illumination + **Vessel Enhancement** + CLAHE + Bilateral → Enhances vessels
- **BLUE:** Illumination + CLAHE + Bilateral → Preserves drusen/exudates

---

## Disease-Specific Benefits

| Disease | Phase 4A/4B | Phase 4C Improvement |
|---------|-------------|----------------------|
| **Diabetes (D)** | Green channel good for vessels | + RED channel for hemorrhages<br>+ Higher res for microaneurysms<br>+ Vessel enhancement |
| **Glaucoma (G)** | Vessel patterns captured | + Better vessel detail (7×7 kernel)<br>+ Higher resolution |
| **Cataract (C)** | Basic detection | + Color info for lens opacity<br>+ Texture analysis |
| **AMD (A)** | **Poor** (lost yellow color) | + **Yellow drusen visible**<br>+ Higher res for small drusen<br>+ No vessel noise |
| **Myopia (M)** | Moderate | + Better geometry preservation<br>+ Clearer peripheral features |
| **Other (O)** | Moderate | + More information overall |

---

## Technical Specifications

### Preprocessing Pipeline

```python
Input: BGR image from OpenCV (variable size)
↓
1. Convert to RGB
2. Extract ROI (circular fundus region)
3. Split into R, G, B channels
4. Process each channel:
   - Illumination correction (Gaussian σ=50)
   - [GREEN only] Vessel enhancement (7×7 elliptical kernel)
   - CLAHE (clip=3.0, grid=8×8)
   - Bilateral filter (d=5, σ_color=50, σ_space=50)
5. Recombine channels
6. Resize to 384×384 (LANCZOS4 interpolation)
7. Normalize to [0, 1]
↓
Output: RGB image (384×384×3) as float32
```

### Training Adjustments

**Batch Size:** Reduce from 64 to 32
- 384×384 images use ~3× more memory than 224×224
- Still fits in 32GB M5 unified memory

**Data Size:**
- Training: ~5,584 images → ~11-12 GB (vs 3.1 GB in Phase 4A)
- Validation: ~1,395 images → ~2.8 GB (vs 801 MB in Phase 4A)

**Training Time:**
- Per epoch: ~20-30% longer due to larger images
- Total: ~4-5 hours per model (vs ~3-4 hours)

---

## Files Created

### Core Files
- `src/phase4c_preprocessing.py` - Phase4CPreprocessor class
- `src/data_preprocessing_phase4c.py` - Preprocessing script

### Output
- `preprocessed_data_phase4c/train_images.npy`
- `preprocessed_data_phase4c/train_labels.npy`
- `preprocessed_data_phase4c/val_images.npy`
- `preprocessed_data_phase4c/val_labels.npy`

---

## How to Run Phase 4C

### Step 1: Wait for Phase 4B to Complete
```bash
# Check if EfficientNetV2 training is done
# Look for "Training Complete!" in Python terminal
```

### Step 2: Create Phase 4C Branch
```bash
git checkout -b phase4C-rgb-highres
git add src/phase4c_preprocessing.py src/data_preprocessing_phase4c.py
git commit -m "Phase 4C: RGB color preservation + 384×384 resolution"
```

### Step 3: Run Preprocessing
```bash
# This will take ~15-20 minutes
python src/data_preprocessing_phase4c.py
```

### Step 4: Update Training Script Data Path
Modify `scripts/train_advanced.py` line ~250:
```python
# Change from:
data_dir = Path('preprocessed_data')

# To:
data_dir = Path('preprocessed_data_phase4c')
```

### Step 5: Train Models (Smaller Batch Size)
```bash
# ConvNeXt Tiny
M5_NUM_WORKERS=4 python scripts/train_advanced.py \
  --model convnext_tiny --epochs 50 --batch-size 32

# ViT Small  
M5_NUM_WORKERS=4 python scripts/train_advanced.py \
  --model vit_small --epochs 50 --batch-size 32

# EfficientNetV2 Small
M5_NUM_WORKERS=4 python scripts/train_advanced.py \
  --model efficientnetv2_s --epochs 50 --batch-size 32
```

### Step 6: Threshold Optimization
```bash
python scripts/optimize_phase4_thresholds.py --phase 4c
```

---

## Expected Results

### Individual Models (Before Threshold Optimization)
| Model | Phase 4A | Phase 4B | Phase 4C Target |
|-------|----------|----------|-----------------|
| ConvNeXt Tiny | 60.30% | 59.22% | **65-68%** |
| ViT Small | 54.99% | 55.33% | **60-63%** |
| EfficientNetV2 | 54.97% | TBD | **60-63%** |

### After Threshold Optimization
- **Phase 4A:** 64.63% F1
- **Phase 4C Target:** **73-76% F1** (+8-11% improvement)

### Why the Big Jump?
1. **RGB color:** +3-5% (especially helps AMD, DR exudates)
2. **Higher resolution:** +3-5% (small lesion detection)
3. **Optimized vessel enhancement:** +1-2% (less noise, better details)
4. **Combined effect:** 7-12% total improvement

---

## Key Differences from Phase 4B

| Aspect | Phase 4B | Phase 4C |
|--------|----------|----------|
| Color | Grayscale (green only) | **RGB (all channels)** |
| Resolution | 224×224 | **384×384** |
| Vessel Enhancement | 15×15 kernel on gray | **7×7 kernel on GREEN only** |
| Kernel Size | Too large | **Optimized for details** |
| AMD Performance | Poor (no yellow) | **Should improve significantly** |
| DR Performance | Moderate | **Should improve** (red channel) |
| File Size | 3.1 GB training | **~11 GB training** |
| Batch Size | 64 | **32** (memory constraint) |
| Training Time | ~3-4h/model | **~4-5h/model** |

---

## Success Criteria

✅ **Minimum Success:** 70% F1 (Phase 4C > Phase 4A)  
✅ **Target Success:** 73-76% F1  
🎯 **Stretch Goal:** 78%+ F1 (ahead of schedule for Phase 4D)

**Key Metrics to Watch:**
- AMD (class A) F1: Should increase significantly (currently worst)
- Diabetes (class D) F1: Should increase (red channel + vessel enhancement)
- Overall macro F1: Target 73-76%

---

## Troubleshooting

### Out of Memory
**Issue:** CUDA/MPS out of memory during training  
**Solution:** Reduce batch size from 32 to 24 or 16

### Preprocessing Too Slow
**Issue:** Takes too long to preprocess 5,584 images  
**Solution:** Already optimized, but could add multiprocessing if needed

### Models Not Loading
**Issue:** Trying to load Phase 4A/4B models with new data  
**Solution:** Train new models from scratch for Phase 4C

---

## Next Steps After Phase 4C

If Phase 4C reaches 73-76% F1:
- **Phase 4D:** Try different base models (Swin Transformer, larger ConvNeXt)
- **Phase 4E:** Final ensemble tuning to reach 85-90% clinical-grade

If Phase 4C doesn't reach 70% F1:
- Analyze per-class performance
- Consider attention mechanisms or region-specific processing
- May need more advanced augmentation

---

**Created:** 2025-11-05  
**Status:** Ready to execute after Phase 4B completion  
**Estimated Time to Complete:** 1 day (preprocessing + training + evaluation)
