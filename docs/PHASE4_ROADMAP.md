# Phase 4: Eye-Specific Diagnosis Roadmap

**Goal**: Achieve 85-90% F1 on eye-specific diagnosis (clinical-grade performance)

**Current Status**: Phase 4A Complete - 64.63% F1

**Task**: Per-eye diagnosis using diagnostic keywords (medically accurate, harder than patient-level)

---

## 📊 Progress Overview

| Phase | Status | F1 Score | Improvement | Methods | Time |
|-------|--------|----------|-------------|---------|------|
| **4A** | ✅ Complete | **64.63%** | Baseline | Ensemble + Threshold Opt | 11h |
| **4B** | 🔄 In Progress | 67-70% (target) | +3-5% | + Vessel Enhancement | 14h |
| **4C** | ⏳ Planned | 73-76% (target) | +6-9% | + Resolution 384×384 | 1 day |
| **4D** | ⏳ Planned | 78-84% (target) | +11-17% | + Base Models | 3 days |
| **4E** | ⏳ Planned | 85-90% (target) | +18-23% | + Advanced Aug + Tuning | 2 days |

**Total Estimated Time**: 6-7 days from Phase 4A to clinical-grade (85-90%)

---

## Phase 4A: Baseline Established ✅

**Completed**: November 5, 2025

### What We Did:
1. ✅ Trained 3 advanced models (ConvNeXt Tiny, ViT Small, EfficientNetV2 Small)
2. ✅ Created weighted ensemble
3. ✅ Tested TTA (found ineffective with MixUp/CutMix)
4. ✅ Optimized per-class thresholds (+2.83% gain)

### Results:
- **F1 Macro**: 64.63%
- **F1 Weighted**: 63.93%
- **Best Individual**: ConvNeXt Tiny @ 60.30%
- **Best Ensemble**: Weighted by Val F1

### Per-Class Performance:
| Class | F1 Score | Status |
|-------|----------|--------|
| Normal | 72.41% | ✅ Good |
| Diabetes | 51.72% | ⚠️ Needs improvement |
| Glaucoma | 67.88% | ✅ Good |
| Cataract | 60.45% | ⚠️ Moderate |
| AMD | 50.78% | ⚠️ Needs improvement |
| Myopia | 88.42% | 🥇 Excellent |
| Other | 60.76% | ⚠️ Moderate |

### Key Insights:
- Threshold optimization was critical (+2.83%)
- TTA doesn't help with heavy augmentation training
- Small models (Tiny/Small) sufficient for baseline
- Eye-specific labeling makes this harder than patient-level

### Files Created:
- `scripts/evaluate_phase4_ensemble.py`
- `scripts/optimize_phase4_thresholds.py`
- `scripts/evaluate_phase4_with_tta.py`
- `results/phase4_threshold_optimization.json`
- `results/PHASE4_PATHA_SUCCESS_REPORT.md`

---

## Phase 4B: Vessel Enhancement 🔄

**Status**: In Progress  
**Expected**: 67-70% F1 (+3-5%)  
**Time**: 14 hours (4h preprocessing + 10h training)

### Why Vessel Enhancement?

Blood vessels are **medically critical** features:
- **Glaucoma**: Vessel patterns indicate optic nerve damage
- **Diabetes**: Microaneurysms, hemorrhages in retinal vessels
- **AMD**: Vessel changes in macula region
- **Hypertension**: Vessel narrowing, AV nicking

Current Phase 4A models don't emphasize vessels enough.

### What Will Change:

**Preprocessing**:
```python
# Phase 4A (current)
apply_vessel_enhancement = False

# Phase 4B (new)
apply_vessel_enhancement = True
```

**Vessel Enhancement Method** (morphological operations):
1. Top-hat transform: Enhance bright structures
2. Black-hat transform: Enhance dark structures (vessels)
3. Combine: `enhanced = img + tophat - blackhat`
4. Kernel: Ellipse 15×15

### Expected Improvements:

**Per-Class Predictions**:
| Class | Phase 4A | Phase 4B Target | Expected Gain |
|-------|----------|-----------------|---------------|
| Diabetes | 51.72% | 55-58% | +3-6% (vessels critical) |
| Glaucoma | 67.88% | 71-74% | +3-6% (optic nerve) |
| AMD | 50.78% | 54-57% | +3-6% (macula vessels) |
| Others | ~65% | ~67% | +2-3% |

**Overall**: 64.63% → 67-70% F1

### Implementation Steps:

1. ✅ **Enable vessel enhancement** (src/advanced_preprocessing.py)
   ```bash
   # Already done: apply_vessel_enhancement=True
   ```

2. ⏳ **Backup Phase 4A data**
   ```bash
   python scripts/phase4B_vessel_enhancement.py
   ```

3. ⏳ **Reprocess dataset** (~4 hours)
   ```bash
   python src/data_preprocessing_enhanced.py
   ```

4. ⏳ **Retrain all 3 models** (~10 hours)
   ```bash
   python scripts/train_advanced.py
   ```

5. ⏳ **Re-evaluate with optimized thresholds**
   ```bash
   python scripts/optimize_phase4_thresholds.py
   ```

### Success Criteria:
- [ ] F1 Macro ≥ 67%
- [ ] Diabetes F1 ≥ 55%
- [ ] Glaucoma F1 ≥ 71%
- [ ] AMD F1 ≥ 54%

---

## Phase 4C: Resolution Scaling ⏳

**Status**: Planned  
**Expected**: 73-76% F1 (+6-9% from 4B)  
**Time**: 1 day (preprocessing + retraining)

### Why Higher Resolution?

Medical images benefit from **fine detail capture**:
- Current: 224×224 (standard ImageNet size)
- Target: 384×384 (medical imaging standard)
- Potential: 512×512 (highest quality, slower)

**What You Gain**:
- Microaneurysms (diabetes) - tiny dots
- Early vessel changes - subtle patterns
- Optic disc detail - glaucoma detection
- Macula texture - AMD features

### Literature Support:
- Medical imaging papers commonly use 384-512
- Diabetic retinopathy: 512×512 standard
- Glaucoma detection: 384×384 minimum

### Implementation:

**Option A: 384×384** (Recommended)
- Moderate compute increase (3-4x)
- Significant quality gain
- Proven in medical imaging

**Option B: 512×512** (If compute allows)
- High compute increase (5-6x)
- Maximum quality
- May need smaller batch size

**Changes Required**:
1. Modify `target_size=(384, 384)` in preprocessing
2. Reprocess all data (~5 hours)
3. Retrain models with larger input (~15 hours)
   - Batch size: 64 → 32 (or 48 → 24)
   - Same models, just larger input

### Expected Results:
| Class | Phase 4B | Phase 4C Target | Gain |
|-------|----------|-----------------|------|
| All classes | ~67-70% | ~73-76% | +6-9% |

**Why the big jump?**
- Fine details become visible
- Medical diagnosis relies on subtle patterns
- Resolution is often THE limiting factor

---

## Phase 4D: Base Model Variants ⏳

**Status**: Planned  
**Expected**: 78-84% F1 (+5-8% from 4C)  
**Time**: 3 days (parallel training recommended)

### Why Larger Models?

Eye-specific diagnosis is **complex**:
- 7 classes with overlapping symptoms
- Per-eye patterns vs patient-level
- Subtle differences require capacity

**Current Models** (Phase 4A-C):
- ConvNeXt **Tiny**: 27.8M params
- ViT **Small**: 21.7M params  
- EfficientNetV2 **Small**: 20.2M params

**Phase 4D Models**:
- ConvNeXt **Base**: 88M params (+3.2× capacity)
- ViT **Base**: 86M params (+4.0× capacity)
- EfficientNetV2 **Medium**: 54M params (+2.7× capacity)

### Training Strategy:

**Parallel Training** (if resources available):
```bash
# Terminal 1
python scripts/train_convnext_base.py

# Terminal 2  
python scripts/train_vit_base.py

# Terminal 3
python scripts/train_efficientnetv2_medium.py
```

**Sequential Training** (limited resources):
- Day 1: ConvNeXt Base (~10-12 hours)
- Day 2: ViT Base (~10-12 hours)
- Day 3: EfficientNetV2 Medium (~8-10 hours)

### Expected Improvements:

Base models learn better:
- Fine-grained features (vessel patterns)
- Class boundaries (overlapping diseases)
- Eye-specific patterns (left vs right differences)

**Per-Class Predictions**:
| Class | Phase 4C | Phase 4D Target | Gain |
|-------|----------|-----------------|------|
| Diabetes | ~58% | 65-70% | +7-12% |
| AMD | ~57% | 63-68% | +6-11% |
| Glaucoma | ~74% | 78-82% | +4-8% |
| Others | ~73% | 77-83% | +4-10% |

**Overall**: 73-76% → 78-84% F1

### Success Criteria:
- [ ] F1 Macro ≥ 78%
- [ ] All classes ≥ 60% F1
- [ ] Diabetes + AMD ≥ 65%

---

## Phase 4E: Final Tuning ⏳

**Status**: Planned  
**Expected**: 85-90% F1 (+7-12% from 4D)  
**Time**: 2 days

### Advanced Augmentation Tuning

**Current** (Phase 4A-D):
- MixUp (α=0.2)
- CutMix (α=1.0)
- Standard geometric augmentations

**Phase 4E Additions**:

1. **Multi-Scale Training**
   - Random resize: [320, 352, 384, 416]
   - Forces model to learn scale-invariant features
   - Expected: +1-2% F1

2. **Fundus-Specific Augmentations**
   ```python
   - GridDistortion (vessel warping)
   - OpticalDistortion (lens effects)
   - Elastic transforms (realistic deformation)
   - Color jitter (imaging variations)
   ```
   - Expected: +1-2% F1

3. **Advanced Mixing**
   - FMix (Fourier domain mixing)
   - SaliencyMix (attention-based)
   - Expected: +1-2% F1

### Label Refinement

**Quality Check**:
1. Review lowest-confidence predictions
2. Validate diagnostic keyword parsing
3. Correct ambiguous labels
4. Re-allocate uncertain cases

**Expected**: +2-3% F1 from cleaner labels

### Ensemble Optimization

**Phase 4A-D**: Equal or F1-weighted ensemble

**Phase 4E**: Optimize ensemble
- Stacking with meta-learner
- Class-specific ensemble weights
- Confidence-based weighting

**Expected**: +1-2% F1

### Final Target:

**85-90% F1** = Clinical-Grade Eye-Specific Diagnosis

| Class | Phase 4E Target | Clinical Value |
|-------|-----------------|----------------|
| Diabetes | 80-85% | High (early detection critical) |
| Glaucoma | 82-87% | High (prevent blindness) |
| AMD | 78-83% | High (early treatment effective) |
| Cataract | 75-80% | Moderate (obvious, treatable) |
| Myopia | 90-95% | High (already excellent) |
| Normal | 85-90% | High (rule out disease) |
| Other | 75-80% | Moderate (catch-all category) |

---

## Comparison: Phase 2 vs Phase 4

### Why Phase 4 is Better (Despite Lower F1):

**Phase 2: Patient-Level Labels**
- F1: 89.52%
- Task: Same label for both eyes
- Example: Patient has "Diabetes" → Both eyes labeled "Diabetes"
- Problem: **Medically inaccurate**
  - Left eye might be healthy
  - Right eye might show early changes
  - Can't detect asymmetric disease
- Clinical Value: **Limited**

**Phase 4: Eye-Specific Labels**
- F1: 64.63% → 85-90% (target)
- Task: Different labels per eye
- Example: Left "Normal", Right "Diabetes (early stage)"
- Advantage: **Medically accurate**
  - Detects asymmetric disease
  - Supports surgical planning (which eye?)
  - Tracks progression per eye
- Clinical Value: **High**

### Real-World Scenario:

**Patient with asymmetric diabetes**:
- Left eye: Normal (no retinopathy)
- Right eye: Mild diabetic retinopathy

**Phase 2 Prediction**: Both eyes "Diabetes" (89% confidence)
- Problem: Can't tell which eye needs treatment
- Can't track individual eye progression

**Phase 4 Prediction** (target):
- Left eye: "Normal" (85% confidence) ✅
- Right eye: "Diabetes" (87% confidence) ✅
- Actionable: Treat right eye, monitor left eye

**Medical Value**: Phase 4 @ 85-90% >>> Phase 2 @ 89.52%

---

## Timeline & Resource Requirements

### Conservative Timeline (Sequential):
- **Week 1**: Phase 4B (Vessel) - 2 days
- **Week 1-2**: Phase 4C (Resolution) - 1-2 days  
- **Week 2**: Phase 4D (Base Models) - 3 days
- **Week 2-3**: Phase 4E (Final Tuning) - 2 days

**Total**: 8-10 days to clinical-grade

### Aggressive Timeline (Parallel):
- **Days 1-2**: Phase 4B (Vessel) + Start 4C preprocessing
- **Days 3-5**: Phase 4C (Resolution) + Start 4D parallel training
- **Days 6-7**: Phase 4D complete + Start 4E tuning
- **Days 8-9**: Phase 4E complete

**Total**: 6-7 days to clinical-grade

### Compute Requirements:

**Current Setup** (Apple M1 Max 32GB):
- Sufficient for Phases 4A-C
- May need sequential training for 4D
- 4E should work fine

**Optimal Setup**:
- GPU: RTX 4090 or A100
- RAM: 64GB+
- Parallel training possible

---

## Success Metrics

### Phase 4B Success:
- [ ] F1 ≥ 67%
- [ ] Diabetes F1 ≥ 55%
- [ ] Training completes without errors
- [ ] Vessel enhancement visible in samples

### Phase 4C Success:
- [ ] F1 ≥ 73%
- [ ] Visual inspection shows better detail
- [ ] All classes show improvement
- [ ] Resolution scaling stable

### Phase 4D Success:
- [ ] F1 ≥ 78%
- [ ] Diabetes, AMD ≥ 65%
- [ ] Base models converge properly
- [ ] Ensemble beats individual models

### Phase 4E Success (Clinical-Grade):
- [x] **F1 ≥ 85%** (PRIMARY GOAL)
- [ ] All classes ≥ 75% F1
- [ ] Diabetes, Glaucoma, AMD ≥ 80%
- [ ] Stable across multiple runs
- [ ] Ready for clinical validation

---

## Monitoring & Validation

### Per-Phase Checklist:

After each phase:
1. ✅ Train all models to completion
2. ✅ Optimize per-class thresholds
3. ✅ Create weighted ensemble
4. ✅ Evaluate on validation set
5. ✅ Compare with previous phase
6. ✅ Visualize improvements per class
7. ✅ Git commit with detailed results
8. ✅ Update this roadmap

### Key Files to Track:

```bash
# Results
results/phase4A_*.json  # Baseline
results/phase4B_*.json  # + Vessel
results/phase4C_*.json  # + Resolution  
results/phase4D_*.json  # + Base Models
results/phase4E_*.json  # Final

# Models
models/*_phase4A_best.pth
models/*_phase4B_best.pth
models/*_phase4C_best.pth
models/*_phase4D_best.pth
models/*_phase4E_best.pth

# Visualizations
results/phase4*_comparison.png
results/phase4*_progress.png
```

---

## Decision Points

### After Phase 4B:
- **If F1 ≥ 70%**: Proceed to 4C immediately ✅
- **If F1 = 67-70%**: Proceed to 4C ✅
- **If F1 < 67%**: Debug vessel enhancement, check preprocessing

### After Phase 4C:
- **If F1 ≥ 76%**: Proceed to 4D ✅
- **If F1 = 73-76%**: Proceed to 4D ✅
- **If F1 < 73%**: Review resolution impact, check if overfitting

### After Phase 4D:
- **If F1 ≥ 80%**: Proceed to 4E for final polish ✅
- **If F1 = 78-80%**: Evaluate if 4E worth it (may already be good enough)
- **If F1 < 78%**: Review Base model training, check convergence

### After Phase 4E:
- **If F1 ≥ 85%**: 🎉 Clinical-grade achieved! Begin validation studies
- **If F1 = 82-85%**: Good enough for pilot deployment
- **If F1 < 82%**: Review label quality, consider external validation set

---

## Risks & Mitigation

### Risk 1: Overfitting with Larger Models
**Mitigation**:
- Strong augmentation (MixUp, CutMix)
- Dropout, weight decay
- Early stopping
- Monitor train/val gap

### Risk 2: Compute Limitations
**Mitigation**:
- Sequential training if needed
- Gradient accumulation for larger batches
- Mixed precision training
- Consider cloud GPU (RunPod, Vast.ai)

### Risk 3: Diminishing Returns
**Mitigation**:
- Clear success criteria per phase
- Decision points after each phase
- ROI analysis (time vs gain)
- Skip phases if needed

### Risk 4: Label Quality Issues
**Mitigation**:
- Manual review of low-confidence samples
- Validate diagnostic keyword parsing
- Cross-check with patient-level labels
- Consider external validation

---

## Current Status Summary

**Phase 4A**: ✅ Complete (64.63% F1)  
**Phase 4B**: 🔄 In Progress (vessel enhancement enabled)  
**Phase 4C**: ⏳ Planned (resolution scaling)  
**Phase 4D**: ⏳ Planned (base models)  
**Phase 4E**: ⏳ Planned (final tuning)

**Next Immediate Actions**:
1. Run Phase 4B preprocessing
2. Retrain 3 models with vessel enhancement
3. Evaluate and compare with 4A
4. If successful, proceed to 4C

**Estimated Completion**: November 12-13, 2025 (6-7 days from now)

---

**Document Version**: 1.0  
**Last Updated**: November 5, 2025  
**Status**: Active Development
