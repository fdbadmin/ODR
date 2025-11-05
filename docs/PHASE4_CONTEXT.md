# Phase 4 Context Document
**Last Updated:** 2025-11-05  
**Current Branch:** `phase4B-vessel-enhancement`  
**Purpose:** Quick context restoration for new chat sessions

---

## Critical Context: Phase 2 vs Phase 4

**Phase 2 (89.52% F1):**
- Used **patient-level labels** (same label for both left and right eye)
- Easier task but less medically accurate
- ~5,584 images with duplicated labels

**Phase 4 (Current: 64.63% F1):**
- Uses **eye-specific labels** (different labels per eye based on diagnostic keywords)
- Harder task but clinically correct - this is what doctors actually need!
- ~5,584 images with unique, accurate labels per eye
- **Target:** 85-90% F1 for clinical-grade eye-specific diagnosis

⚠️ **Do NOT compare Phase 2 and Phase 4 directly** - they solve different problems!

---

## Phase 4 Roadmap

| Phase | Status | Target F1 | Key Improvement | Time |
|-------|--------|-----------|-----------------|------|
| **4A** | ✅ Complete | 64.63% | Baseline + threshold optimization | 11h |
| **4B** | 🔄 In Progress | 67-70% | + Vessel enhancement | 14h |
| **4C** | ⏳ Planned | 73-76% | + Resolution 384×384 | 24h |
| **4D** | ⏳ Planned | 78-84% | + Base model variants | 72h |
| **4E** | ⏳ Planned | 85-90% | + Final tuning (CLINICAL-GRADE) | 48h |

**Total Timeline:** 6-7 days to clinical-grade

---

## Current Status: Phase 4B

### Completed ✅
1. **Branch created:** `phase4B-vessel-enhancement`
2. **Vessel enhancement enabled** in `src/advanced_preprocessing.py` (line 30: `True`)
3. **Preprocessing completed:** November 5, 01:47
   - Training: 5,584 images, 3.1 GB (`train_images.npy`)
   - Validation: 1,395 images, 801 MB (`val_images.npy`)
   - Method: Full preprocessing with vessel enhancement (morphological top-hat/black-hat)
4. **Training optimized for M5 hardware:**
   - Modified `scripts/train_advanced.py` DataLoader settings
   - Created `scripts/train_phase4B_sequential_m5.py` (sequential training script)
   - Created `scripts/run_phase4B_overnight.sh` (launcher)

### Ready to Run ⏳
- **Overnight training:** 3 models sequentially (~10-12 hours)
  1. ConvNeXt Tiny (MixUp) - Expected 63-66% F1
  2. ViT Small (CutMix) - Expected 58-61% F1
  3. EfficientNetV2 Small (MixUp+CutMix) - Expected 58-61% F1

### After Training ⏳
1. Threshold optimization: `python scripts/optimize_phase4_thresholds.py`
2. Ensemble evaluation
3. Compare with Phase 4A (64.63%)
4. Target: 67-70% F1 (+3-5% improvement)

---

## Hardware: MacBook Pro M5

**Specifications:**
- M5 chip with 32GB unified memory
- 10-core CPU
- MPS (Metal Performance Shaders) acceleration

**M5 Optimizations Applied:**
```python
batch_size = 96              # vs typical 32-64
num_workers = 10             # match 10-core CPU
persistent_workers = True    # reduce overhead
pin_memory = True            # unified memory benefit
prefetch_factor = 2          # prefetch 2 batches per worker
```

---

## Model Architecture (Phase 4B)

| Model | Params | Phase 4A F1 | Expected 4B | Augmentation |
|-------|--------|-------------|-------------|--------------|
| ConvNeXt Tiny | 27.8M | 60.30% | 63-66% | MixUp |
| ViT Small | 21.7M | 54.99% | 58-61% | CutMix |
| EfficientNetV2 Small | 20.2M | 54.97% | 58-61% | MixUp+CutMix |

**Phase 4A Results:**
- Best individual: ConvNeXt Tiny (60.30%)
- Ensemble: 61.80% F1
- After threshold optimization: **64.63% F1**

---

## Preprocessing Pipeline (Phase 4B)

**Input:** ODIR-5K retinal images (both eyes per patient)  
**Output:** 224×224×3 preprocessed images

**Steps:**
1. ROI extraction (circular crop)
2. Green channel extraction
3. Illumination correction (Gaussian σ=50)
4. **Vessel enhancement** (morphological top-hat/black-hat) ✅ NEW in 4B
5. CLAHE (clip_limit=3.0, grid_size=8×8)
6. Bilateral filtering (d=5, σ_color=50, σ_space=50)
7. Resize to 224×224
8. Normalization [0,1]
9. Convert grayscale→3-channel

**File:** `src/advanced_preprocessing.py`  
**Config:** Line 30 - `apply_vessel_enhancement: bool = True`

---

## Training Configuration

**Loss:** FocalLoss (α=0.25, γ=2.0) + WeightedBCE (70%/30% mix)  
**Optimizer:** AdamW (lr=1e-4, weight_decay=1e-5)  
**Scheduler:** CosineAnnealingLR  
**Epochs:** 50  
**Classes:** 7 (N, D, G, C, A, M, O)

**Augmentation:**
- Per-image: Albumentations pipeline
- Batch-level: MixUp/CutMix (model-specific)

---

## Key Files

### Training Scripts
- `scripts/train_advanced.py` - Base training script (M5 optimized)
- `scripts/train_phase4B_sequential_m5.py` - Sequential overnight training
- `scripts/run_phase4B_overnight.sh` - Launch script

### Models
- `models/convnext_tiny_advanced_best.pth`
- `models/vit_small_advanced_best.pth`
- `models/efficientnetv2_s_advanced_best.pth`

### Preprocessing
- `src/advanced_preprocessing.py` - Preprocessing pipeline
- `src/data_preprocessing_enhanced.py` - Orchestration script
- `preprocessed_data/*.npy` - Preprocessed images and labels

### Documentation
- `docs/PHASE4_ROADMAP.md` - Complete 4A→4E progression plan
- `docs/PHASE4B_STATUS.md` - Phase 4B progress tracking
- `results/PHASE4_PATHA_SUCCESS_REPORT.md` - Phase 4A results (corrected)
- `docs/PHASE4_CONTEXT.md` - **This file** (for chat restoration)

---

## Quick Start Commands

### Launch Overnight Training
```bash
# Option 1: Using launcher (recommended)
./scripts/run_phase4B_overnight.sh

# Option 2: Direct Python
python scripts/train_phase4B_sequential_m5.py
```

### After Training
```bash
# Optimize thresholds
python scripts/optimize_phase4_thresholds.py

# Check results
ls -lh models/*_advanced_best.pth
cat results/PHASE4B_TRAINING_REPORT_*.md
```

### Monitor Progress (if needed)
```bash
# Check latest log
tail -f logs/phase4B_training_*.log

# Check progress file
cat results/phase4B_training_progress.json
```

---

## Common Issues & Solutions

### Issue: "Preprocessed data not found"
```bash
python src/data_preprocessing_enhanced.py
```

### Issue: MPS not available
- Verify PyTorch with MPS support is installed
- Check: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Issue: Out of memory
- Reduce batch_size in `scripts/train_phase4B_sequential_m5.py`
- Change `batch_size = 96` to `64` or `48`

### Issue: Training too slow
- Check num_workers isn't too high for your system
- Verify MPS is being used (check logs for "Using MPS device")

---

## Git Status

**Current Branch:** `phase4B-vessel-enhancement`  
**Recent Commits:**
- `ba474cb` - Corrected Phase 2 vs Phase 4 documentation
- `b024c92` - Phase 4B setup (vessel enhancement enabled)

**To merge after Phase 4B success:**
```bash
git add .
git commit -m "Phase 4B complete: XX.XX% F1 with vessel enhancement"
git push origin phase4B-vessel-enhancement
# Then merge to main via PR
```

---

## Expected Timeline

**Tonight (10-12 hours):**
- Sequential training of 3 models
- Automatic checkpointing and logging
- Final report generation

**Tomorrow morning:**
1. Review training report (~5 min)
2. Run threshold optimization (~30 min)
3. Evaluate results (~30 min)
4. Compare with Phase 4A baseline (~15 min)

**Success Criteria:**
- All 3 models train successfully
- Phase 4B F1 reaches 67-70% (+3-5% over Phase 4A's 64.63%)
- Ready to proceed to Phase 4C (384×384 resolution)

---

## Questions for New Chat Session?

**Restore context by asking:**
> "Read docs/PHASE4_CONTEXT.md for project context"

**Then proceed with:**
- "Check Phase 4B training results"
- "Run threshold optimization"
- "Evaluate Phase 4B vs Phase 4A"
- "Start Phase 4C planning"

---

**Last Session Note:** About to launch overnight training with M5 optimizations. All preprocessing complete, scripts ready.
