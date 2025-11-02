# Stage 2 Training Insights from Stage 1 Evaluation

## Key Findings from Stage 1 Baseline (Image-Only Model)

### Overall Performance
- **Label Accuracy**: 87.08% ✅ (exceeded 85.21% target by +1.87%)
- **Sample Accuracy**: 43.23% (exact match all 8 labels)
- **Mean ROC-AUC**: 0.80 (good discrimination)
- **But**: Mean F1-Score only 0.51 (significant imbalance issues)

---

## Critical Insights for Stage 2

### 1. **Severe Class Imbalance Problem**

**Performance by Prevalence:**
| Disease | Prevalence | F1-Score | ROC-AUC | Support | Performance |
|---------|-----------|----------|---------|---------|-------------|
| Normal | 44.2% | 0.633 | 0.713 | 617 | Moderate |
| Diabetes | 25.5% | 0.469 | 0.707 | 356 | Poor |
| Cataract | 11.2% | 0.564 | 0.826 | 156 | Moderate |
| Other | 11.0% | 0.434 | 0.758 | 153 | Poor |
| AMD | 7.5% | 0.435 | 0.830 | 105 | Poor |
| Glaucoma | 5.6% | 0.564 | 0.895 | 78 | Moderate |
| Myopia | 3.6% | 0.781 | 0.975 | 50 | **Excellent** |
| **Hypertension** | **3.2%** | **0.179** | **0.694** | **45** | **Critical** |

**Key Pattern**: 
- ❌ Hypertension (rarest, 3.2%) performs worst (F1=0.18)
- ✅ Myopia (also rare, 3.6%) performs best (F1=0.78)
- 🤔 **Prevalence alone doesn't explain performance**

### 2. **Precision vs Recall Imbalance**

**Diseases with Low Recall (missing positive cases):**
- Hypertension: Recall=0.13 (missing 87% of cases!)
- AMD: Recall=0.43 (missing 57%)
- Diabetes: Recall=0.48 (missing 52%)
- Other: Recall=0.47 (missing 53%)

**Diseases with Low Precision (too many false alarms):**
- Other: Precision=0.40 (60% false positives)
- AMD: Precision=0.44 (56% false positives)
- Diabetes: Precision=0.46 (54% false positives)

### 3. **Prediction Confidence Analysis**

**Confidence Distribution Issues:**

| Disease | Mean Prob | Issue |
|---------|-----------|-------|
| Normal | 0.48 | Close to threshold (0.5) - many borderline cases |
| Diabetes | 0.28 | Low confidence - model uncertain |
| Hypertension | **0.02** | **Extremely low** - model rarely predicts it |
| AMD | 0.08 | Very low - struggling to detect |
| Other | 0.14 | Low - unclear visual patterns |

**Observation**: Model is **overconfident on negatives** for rare classes (especially Hypertension)

### 4. **ROC-AUC vs F1 Discrepancy**

**High ROC-AUC but Low F1:**
- AMD: ROC-AUC=0.83, F1=0.43 (40% gap)
- Glaucoma: ROC-AUC=0.90, F1=0.56 (34% gap)
- Cataract: ROC-AUC=0.83, F1=0.56 (27% gap)

**Meaning**: Model can **separate** classes well (good discrimination) but **threshold=0.5 is suboptimal** for rare classes

---

## Recommendations for Stage 2

### ✅ **1. Keep Current Class Weights** (CRITICAL)
```python
# Current weights in train.py are correct:
pos_weight = torch.tensor([
    1.26,   # Normal (most common)
    1.79,   # Diabetes
    16.88,  # Glaucoma
    7.94,   # Cataract
    12.29,  # AMD
    30.00,  # Hypertension (HIGHEST - keep this!)
    27.80,  # Myopia
    8.12    # Other
])
```
**Why**: These weights help rare classes during training. Hypertension needs maximum emphasis.

### ✅ **2. Use Two-Stage Architecture** (ALREADY PLANNED)
```python
# MetadataRefinementModel approach:
1. Frozen baseline predictions (87.08% floor guaranteed)
2. Small refinement network (~5K params) 
3. Combine: baseline_logits + age + gender
```

**Why This Will Help:**
- **Age patterns**: 
  - Diabetes: older patients
  - Myopia: younger patients
  - AMD: age-related (in the name!)
  - Hypertension: strongly age-correlated
  
- **Gender patterns**:
  - Some diseases have gender prevalence differences
  - Can help disambiguate similar retinal features

### ❌ **3. DO NOT Use Adaptive Thresholds** (PREVIOUSLY FAILED)

**Why not:**
- Previously tested and **removed** due to poorer performance
- Creates train/val mismatch (model trained on 0.5, evaluated on different thresholds)
- Optimizing per-class thresholds on validation set = overfitting to validation
- Better approach: Let model learn to produce better probabilities through:
  - Class weights (already doing this ✅)
  - Better training data representation
  - Metadata features to disambiguate

**The Real Problem**: Model produces low probabilities for rare classes because it's **correctly uncertain** without additional context. Fix: Add metadata, not threshold manipulation.

### ✅ **3. Metadata Feature Engineering** (PRIMARY SOLUTION)

**Add these features to Stage 2:**
```python
# Current: age, gender (2 features)
# Enhanced: 
metadata_features = {
    'age': age,
    'age_squared': age ** 2,  # Non-linear age effects
    'age_bins': [
        age < 30,   # Young
        30 <= age < 50,  # Middle
        50 <= age < 70,  # Senior
        age >= 70   # Elderly
    ],
    'gender': gender,
    'age_gender_interaction': age * gender  # Combined effect
}
# Total: ~7 features instead of 2
```

**Why**: AMD, Diabetes, Hypertension have **non-linear age relationships**

### ✅ **4. Focus Training on Problem Classes**

**Use sample weighting during Stage 2:**
```python
# Weight samples based on baseline model errors
sample_weights = np.ones(len(train_dataset))

for i, (y_true, y_pred) in enumerate(zip(train_labels, baseline_preds)):
    # Upweight samples where baseline failed on rare classes
    errors = (y_true != y_pred)
    rare_class_errors = errors[[3, 4, 5, 7]]  # C, A, H, O indices
    if rare_class_errors.sum() > 0:
        sample_weights[i] *= 2.0  # Double importance
```

### ❌ **5. DO NOT Use Focal Loss** 
**Reason**: Standard BCE is working. Focal loss adds complexity without clear benefit given our class weights are already handling imbalance.

### ❌ **6. DO NOT Oversample Rare Classes**
**Reason**: 
- Creates distribution shift
- Validation won't match real-world prevalence
- Our evaluation shows we can get 87% without it

---

## Expected Stage 2 Improvements

### Conservative Estimates:
| Metric | Stage 1 | Stage 2 Target | How |
|--------|---------|----------------|-----|
| **Label Accuracy** | 87.08% | 88-90% | Metadata disambiguation |
| **Hypertension F1** | 0.18 | 0.35-0.45 | Lower threshold + age features |
| **Diabetes F1** | 0.47 | 0.55-0.60 | Age/gender patterns |
| **AMD F1** | 0.43 | 0.50-0.55 | Age features (AMD = Age-related!) |
| **Mean F1** | 0.51 | 0.55-0.60 | Better rare class handling |
| **Mean ROC-AUC** | 0.80 | 0.82-0.85 | Improved discrimination |

### Key Success Metrics for Stage 2:
1. ✅ **Maintain** 87%+ label accuracy (frozen baseline ensures this)
2. ✅ **Improve Hypertension** F1 from 0.18 to 0.35+ (double it!)
3. ✅ **Improve rare class recall** (AMD, Hypertension, Other)
4. ✅ **Keep Myopia performance** (0.78 F1 - don't break what works!)

---

## Implementation Checklist for Stage 2

- [x] Use two-stage frozen baseline architecture (already in code)
- [x] Keep current class weights (already correct)
- [x] ~~Adaptive thresholds~~ (TESTED - hurt performance, correctly removed)
- [ ] **Enhanced metadata features** (age^2, bins, interactions) - PRIORITY
- [ ] **Sample weighting** for baseline-error cases (optional)
- [ ] Validate on same data to compare directly
- [ ] Track per-class metrics throughout training
- [ ] Monitor validation accuracy (label-level, not sample-level)

---

## Stage 2 Training Configuration

```python
# Recommended settings:
STAGE = 2
USE_METADATA = True
USE_TWO_STAGE = True
BASELINE_MODEL_PATH = 'models/baseline_model.pth'
NUM_EPOCHS = 15  # May converge faster with frozen baseline
LEARNING_RATE = 5e-5  # Small - only training refinement network
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5
MONITOR_METRIC = 'val_acc'  # Label accuracy (what worked for Stage 1)
THRESHOLD = 0.5  # Standard threshold (adaptive thresholds failed)
```

---

## Bottom Line

**Stage 1 achieved the accuracy target (87.08%)** but revealed:
1. ⚠️ **Hypertension is the critical failure case** (F1=0.18, Recall=0.13)
2. ❌ **Adaptive thresholds don't work** (tested, hurt performance, correctly removed)
3. 🎯 **The REAL solution: Metadata** to help age-related diseases (AMD, Diabetes, Hypertension)
4. 🔒 **Two-stage approach protects our gains** (87% floor)

**Why Hypertension fails without metadata:**
- Visual features alone are ambiguous (hypertensive retinopathy overlaps with diabetes)
- Age is a **strong predictor** (hypertension prevalence increases with age)
- Only 45 validation samples (3.2%) - model cautious without context
- Class weights help during training, but **inference needs metadata**

**Stage 2 should focus on**:
- ✅ Enhanced metadata features (age^2, age bins, age×gender)
- ✅ Two-stage frozen baseline (protect 87% floor)
- ✅ Keep class weights (already optimal)
- ✅ Standard 0.5 threshold (adaptive failed)
- ❌ No oversampling (creates distribution shift)
- ❌ No focal loss (unnecessary complexity)

**Expected outcome**: 88-90% accuracy with better rare class performance through **metadata disambiguation**, not threshold tricks.
