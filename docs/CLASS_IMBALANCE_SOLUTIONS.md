# Class Imbalance Solutions for 7-Class Multi-Label System

## Current Imbalance Problem

**Class Distribution:**
- Normal: 1140 (32.57%)
- Diabetes: 1128 (32.23%)
- **Glaucoma: 215 (6.14%)** ⚠️
- **Cataract: 212 (6.06%)** ⚠️
- **AMD: 164 (4.69%)** ⚠️ MOST IMBALANCED
- **Myopia: 174 (4.97%)** ⚠️
- Other: 979 (27.97%)

**Imbalance Ratio:** 6.95x (Normal vs AMD)

## ✅ Solutions Already Implemented

### 1. **Class Weights in Loss Function** ✓ ALREADY DONE
**Status:** Currently implemented in `src/train.py`

```python
# Calculate class weights (inverse frequency)
class_weights = len(train_labels) / (num_classes * train_labels.sum(axis=0))
# Results in higher weight for rare classes (AMD, Myopia)
```

**Effectiveness:** Moderate - helps but not sufficient for 7x imbalance

---

## 🚀 Additional Strategies (Recommended)

### 2. **Focal Loss** ⭐ HIGHLY RECOMMENDED
**What it does:** Focuses on hard-to-classify examples, downweights easy ones

**Advantages:**
- ✓ Specifically designed for class imbalance
- ✓ Works better than class weights for extreme imbalance
- ✓ No data modification needed
- ✓ Easy to implement

**Implementation:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
```

**Expected improvement:** +2-4% F1 score (93-95% F1)

---

### 3. **Data Augmentation for Minority Classes** ⭐ RECOMMENDED
**What it does:** Create more synthetic samples for rare diseases

**Strategies:**
- **A. Aggressive augmentation for rare classes:**
  - AMD, Myopia, Glaucoma, Cataract: 3x augmentation
  - Normal, Diabetes, Other: 1x (original only)

**Augmentation techniques:**
```python
# Minority class augmentation
minority_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.GaussianBlur(kernel_size=3),
])
```

**Expected improvement:** +1-3% F1 score

---

### 4. **Oversampling Minority Classes** 
**What it does:** Duplicate samples from rare classes during training

**Strategies:**
- **A. Random oversampling:** Duplicate minority samples
- **B. SMOTE-like:** Create synthetic samples by interpolating features
- **C. Weighted sampling:** Sample rare classes more frequently

**Implementation (Weighted Sampling):**
```python
# Calculate sample weights based on labels
sample_weights = []
for label in train_labels:
    # Weight = sum of class weights for all positive labels
    weight = sum([class_weights[i] for i in range(7) if label[i] == 1])
    sample_weights.append(weight)

# Use WeightedRandomSampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=48,
    sampler=sampler  # Instead of shuffle=True
)
```

**Expected improvement:** +2-3% F1 score

---

### 5. **Class-Balanced Loss** ⭐ VERY EFFECTIVE
**What it does:** Re-weight loss based on effective number of samples

**Formula:**
```python
def get_effective_num(n, beta=0.9999):
    """Calculate effective number of samples"""
    return (1 - beta**n) / (1 - beta)

# Calculate class-balanced weights
effective_nums = [get_effective_num(count) for count in class_counts]
cb_weights = [(1 - beta) / en for en in effective_nums]
```

**Advantages:**
- ✓ More sophisticated than simple inverse frequency
- ✓ Accounts for information overlap in augmented data
- ✓ Used in state-of-the-art models

**Expected improvement:** +2-3% F1 score

---

### 6. **Two-Stage Training**
**What it does:** Train on balanced subset first, then fine-tune on full data

**Procedure:**
1. **Stage 1:** Train on balanced subset (undersample majority classes)
   - Take all minority samples (AMD, Myopia, Glaucoma, Cataract)
   - Randomly sample majority classes to match
   - Train for 15 epochs

2. **Stage 2:** Fine-tune on full dataset
   - Load Stage 1 model
   - Train on full data with class weights
   - Train for 10 more epochs

**Expected improvement:** +1-2% F1 score

---

### 7. **Ensemble with Specialized Models**
**What it does:** Train separate models for minority classes

**Strategy:**
- **Model 1:** General model (all classes)
- **Model 2:** Specialized for AMD + Glaucoma + Myopia + Cataract
  - Only trained on samples with these diseases
  - Uses balanced subset

**Expected improvement:** +2-3% F1 score

---

## 📊 Recommended Combination Strategy

### **Best Approach: Focal Loss + Augmentation + Weighted Sampling**

**Why this combination:**
1. **Focal Loss:** Handles loss-level imbalance (most effective)
2. **Augmentation:** Increases minority class diversity
3. **Weighted Sampling:** Ensures rare classes seen more often

**Expected total improvement:** +4-6% F1 score (95-97% F1)

**Implementation Priority:**
1. ⭐ **Start with Focal Loss** (easiest, biggest impact)
2. ⭐ Add minority class augmentation
3. ⭐ Add weighted sampling
4. Test and iterate

---

## 🔧 Implementation Plan

### Quick Win: Focal Loss (30 minutes)
```bash
# 1. Add focal loss to src/train.py
# 2. Replace BCEWithLogitsLoss
# 3. Retrain baseline model
# Expected: 93-94% F1 (vs current 88.72%)
```

### Medium Effort: Full Pipeline (2-3 hours)
```bash
# 1. Implement focal loss
# 2. Add minority class augmentation in data loader
# 3. Implement weighted sampling
# 4. Retrain all 3 models in ensemble
# Expected: 95-97% F1 (vs current 91.51%)
```

---

## 📈 Comparison: What You're Already Doing vs What We Can Do

### Current Approach (YOUR SYSTEM)
```
✓ Class weights in loss function
✓ Ensemble of 3 models
✓ Per-class threshold optimization
→ Result: 91.51% F1
```

### Enhanced Approach (RECOMMENDED)
```
✓ Focal Loss (replaces class weights)
✓ Minority class augmentation (3x for rare diseases)
✓ Weighted sampling in data loader
✓ Ensemble of 3 models (keep this)
✓ Per-class threshold optimization (keep this)
→ Expected: 95-97% F1 (+4-6% improvement)
```

---

## 🎯 Quick Decision Guide

**Want quick improvement with minimal changes?**
→ Use **Focal Loss** only (30 min work, +2-3% F1)

**Want maximum performance?**
→ Use **Focal Loss + Augmentation + Weighted Sampling** (3 hours, +4-6% F1)

**Want to experiment?**
→ Try **two-stage training** or **specialized ensemble models**

---

## 📝 Next Steps

Would you like me to:

1. **Implement Focal Loss** (quick win, 30 minutes)
2. **Implement full pipeline** (focal loss + augmentation + sampling)
3. **Show comparison** between current and enhanced approach
4. **Something else?**

Let me know and I'll set it up! 🚀
