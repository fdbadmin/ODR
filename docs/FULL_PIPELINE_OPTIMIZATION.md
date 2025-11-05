# Complete Pipeline Optimization Analysis
## Phase 4C: RGB + Multi-Label Stratified Split + Full Stack Improvements

**Date:** November 5, 2025  
**Objective:** Maximize model performance through end-to-end pipeline optimization  
**Target:** 68-72% macro F1 (beat Phase 4A 64.63% baseline)

---

## Executive Summary

Analysis of the complete pipeline (preprocessing → data loading → augmentation → training → validation) reveals **12 key optimization opportunities** across 5 stages. The improved multi-label stratified split (75/25) is foundational, but additional gains of **+5-8% F1** are achievable through strategic improvements.

**Expected Combined Impact:** 
- Improved split: +3-6% F1 (reduces variance, larger validation set for rare classes)
- Preprocessing optimizations: +1-2% F1 (better vessel enhancement, adaptive CLAHE)
- Augmentation improvements: +1-2% F1 (RetinalMix, focal region augmentation)
- Training optimizations: +1-2% F1 (cosine annealing with restarts, OneCycleLR, better regularization)
- Validation & ensemble: +2-3% F1 (TTA, weighted ensemble, per-class thresholds)

**Total Expected: 63-66% single model → 68-73% ensemble**

---

## Current Pipeline Status

### ✅ Already Optimized (Strong Foundation)
1. **RGB color preservation** - Critical for AMD (drusen) and DR (hemorrhages, exudates)
2. **384×384 resolution** - Better for small lesions (microaneurysms, dot hemorrhages)
3. **Multi-label stratified split (75/25)** - Balances all 7 classes, increases rare class validation
4. **Mixed precision (AMP)** - 30-40% speedup, same accuracy
5. **Gradient accumulation** - Effective batch size 64 for ConvNeXt/ViT
6. **LR warmup (5 epochs)** - Stable training start
7. **Gradient clipping (norm=1.0)** - Prevents exploding gradients
8. **FocalLoss + WeightedBCE (70/30)** - Handles class imbalance well
9. **WeightedRandomSampler** - Oversamples rare classes during training
10. **M5 hardware optimization** - 4 P-core workers, persistent_workers, prefetch_factor=2

### ⚠️ Optimization Opportunities (12 improvements)

---

## Stage 1: Preprocessing Optimizations

### Current Implementation
```python
# phase4c_preprocessing.py
- RGB preservation: ✅ (R, G, B channels separate)
- Vessel enhancement: ✅ (7×7 kernel on GREEN only)
- CLAHE: ✅ (clipLimit=3.0, grid=8×8 per channel)
- Bilateral filtering: ✅ (d=5, sigmaColor=50, sigmaSpace=50)
- ROI extraction: ✅ (removes black borders)
- Illumination correction: ✅ (Gaussian blur sigma=50)
```

### 🔧 Optimization 1: Adaptive CLAHE Based on Image Statistics
**Current:** Fixed CLAHE clipLimit=3.0 for all images  
**Problem:** Some images are naturally darker (DR severe), others brighter (normal). Fixed CLAHE can over-enhance bright images or under-enhance dark images.

**Solution:**
```python
def apply_clahe_adaptive(self, img: np.ndarray) -> np.ndarray:
    """Apply adaptive CLAHE based on image brightness."""
    mean_brightness = img.mean()
    
    # Adjust clip limit based on brightness
    if mean_brightness < 60:  # Dark image
        clip_limit = 4.0  # More enhancement
    elif mean_brightness > 140:  # Bright image
        clip_limit = 2.0  # Less enhancement
    else:
        clip_limit = 3.0  # Standard
    
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )
    return clahe.apply(img)
```

**Expected Gain:** +0.5-1% F1 (better contrast in extreme cases)

### 🔧 Optimization 2: Multi-Scale Vessel Enhancement
**Current:** Single 7×7 kernel for vessel enhancement  
**Problem:** Misses both large vessels (optic disc region) and very fine capillaries

**Solution:**
```python
def enhance_vessels_multiscale(self, img: np.ndarray) -> np.ndarray:
    """Multi-scale vessel enhancement (5×5, 7×7, 9×9 kernels)."""
    # Small vessels (capillaries, microaneurysms)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vessels_small = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_small)
    
    # Medium vessels (retinal arteries/veins)
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    vessels_medium = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_medium)
    
    # Large vessels (optic disc region)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    vessels_large = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_large)
    
    # Combine with weights (favor medium scale)
    vessels_combined = (0.25 * vessels_small + 
                       0.5 * vessels_medium + 
                       0.25 * vessels_large)
    
    enhanced = cv2.subtract(img, vessels_combined.astype(np.uint8))
    return enhanced
```

**Expected Gain:** +0.5-1% F1 (better DR and glaucoma detection)

### 🔧 Optimization 3: Green Channel Enhancement for AMD
**Current:** Only vessel enhancement on GREEN channel  
**Problem:** AMD drusen appear as yellow deposits (HIGH GREEN + MODERATE RED), but we only enhance vessels (dark structures). Drusen are BRIGHT structures that need different processing.

**Solution:**
```python
def enhance_drusen_features(self, g_channel: np.ndarray, b_channel: np.ndarray) -> np.ndarray:
    """Enhance AMD drusen features (bright yellow deposits)."""
    # Drusen enhancement: Top-hat on green channel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    drusen_enhanced = cv2.morphologyEx(g_channel, cv2.MORPH_TOPHAT, kernel)
    
    # Combine with original green channel
    g_enhanced = cv2.add(g_channel, drusen_enhanced)
    
    return g_enhanced
```

**Expected Gain:** +1-2% F1 for AMD class specifically (currently 55%, target 60-65%)

---

## Stage 2: Data Loading & Augmentation

### Current Implementation
```python
# augmentation.py
- Geometric: ✅ (horizontal flip, vertical flip, rotation ±20°)
- Color: ✅ (brightness ±20%, contrast ±20%, saturation ±10%, hue ±5%)
- Affine: ✅ (translate ±10%, scale ±10%, shear ±5°)
- RandomErasing: ✅ (p=0.2, simulates occlusions)
- MixUp/CutMix: ✅ (optional batch augmentation)
```

### 🔧 Optimization 4: RetinalMix (Medical Imaging-Specific MixUp)
**Current:** MixUp/CutMix mix entire images randomly  
**Problem:** Can mix healthy regions with diseased regions, creating unrealistic hybrids (e.g., half drusen + half normal retina)

**Solution:**
```python
class RetinalMix:
    """MixUp variant that respects anatomical regions."""
    
    def __call__(self, batch_images, batch_labels):
        lam = np.random.beta(0.4, 0.4)  # More balanced mixing
        
        # Only mix samples with SIMILAR disease patterns
        # E.g., DR + DR, AMD + AMD, Normal + Normal
        batch_size = batch_images.size(0)
        
        # Group by dominant disease
        dominant_diseases = batch_labels.argmax(dim=1)
        
        # Create pairs within same disease group
        mixed_images = batch_images.clone()
        mixed_labels = batch_labels.clone()
        
        for disease_id in range(7):
            disease_mask = (dominant_diseases == disease_id)
            disease_indices = torch.where(disease_mask)[0]
            
            if len(disease_indices) >= 2:
                # Shuffle within disease group
                shuffled = disease_indices[torch.randperm(len(disease_indices))]
                for i, j in zip(disease_indices, shuffled):
                    mixed_images[i] = lam * batch_images[i] + (1-lam) * batch_images[j]
                    mixed_labels[i] = lam * batch_labels[i] + (1-lam) * batch_labels[j]
        
        return mixed_images, mixed_labels
```

**Expected Gain:** +0.5-1% F1 (more realistic augmentation, preserves anatomical coherence)

### 🔧 Optimization 5: Focal Region Augmentation
**Current:** Augmentation applied uniformly across entire image  
**Problem:** Important features (optic disc, macula, vessels) are in specific regions. Random cropping might miss them.

**Solution:**
```python
class FocalRegionAugmentation:
    """Augment with focus on critical anatomical regions."""
    
    def __init__(self, focus_regions=['optic_disc', 'macula', 'vessels']):
        self.focus_regions = focus_regions
    
    def __call__(self, image):
        # 70% of time: center crop on macula/optic disc
        # 30% of time: random crop (include periphery for DR)
        if random.random() < 0.7:
            # Detect optic disc (brightest region in green channel)
            green = image[:, :, 1]
            oy, ox = np.unravel_index(green.argmax(), green.shape)
            
            # Crop 320×320 around optic disc, then resize to 384×384
            y1 = max(0, oy - 160)
            y2 = min(image.shape[0], oy + 160)
            x1 = max(0, ox - 160)
            x2 = min(image.shape[1], ox + 160)
            
            image_cropped = image[y1:y2, x1:x2]
            image = cv2.resize(image_cropped, (384, 384))
        
        return image
```

**Expected Gain:** +0.5-1% F1 (forces model to learn critical regions)

### 🔧 Optimization 6: Disease-Specific Color Augmentation
**Current:** Same color jitter for all images  
**Problem:** AMD needs RED-YELLOW preservation (drusen), DR needs RED preservation (hemorrhages), but we apply same augmentation to all

**Solution:**
```python
def get_disease_specific_augmentation(labels):
    """Adjust augmentation based on disease labels."""
    
    # Check dominant disease
    if labels[4] > 0.5:  # AMD
        # Preserve yellow-red spectrum
        return transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.05,  # Less saturation change
            hue=0.02          # Minimal hue shift (preserve yellow)
        )
    elif labels[1] > 0.5:  # Diabetes
        # Preserve red spectrum
        return transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.05,
            hue=0.03          # Minimal hue (preserve red hemorrhages)
        )
    else:
        # Standard augmentation
        return transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        )
```

**Expected Gain:** +0.5-1% F1 for AMD and DR specifically

---

## Stage 3: Model Architecture & Training

### Current Implementation
```python
# train_advanced.py
- Models: ✅ (ConvNeXt Tiny, ViT Small, EfficientNetV2 Small)
- Loss: ✅ (70% FocalLoss + 30% WeightedBCE)
- Optimizer: ✅ (AdamW lr=1e-4, weight_decay=1e-5)
- Scheduler: ✅ (LambdaLR with 5-epoch warmup + cosine annealing)
- Mixed Precision: ✅ (AMP enabled)
- Gradient Accumulation: ✅ (2 steps for ConvNeXt/ViT)
```

### 🔧 Optimization 7: OneCycleLR Instead of Cosine Annealing
**Current:** LambdaLR with warmup + cosine annealing  
**Problem:** Learning rate doesn't adapt to plateau phases. OneCycleLR has been shown to converge faster and reach better optima.

**Solution:**
```python
# Replace LambdaLR scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr * 10,  # Peak at 10x base LR
    epochs=args.epochs,
    steps_per_epoch=len(train_loader) // grad_accum_steps,
    pct_start=0.3,  # 30% of training is warmup
    anneal_strategy='cos',
    div_factor=25.0,  # Initial LR = max_lr / 25
    final_div_factor=1000.0  # Final LR = max_lr / 1000
)

# Update after each batch (not epoch)
# In train_epoch(), after optimizer.step():
scheduler.step()
```

**Expected Gain:** +1-2% F1 (faster convergence, better final performance)

### 🔧 Optimization 8: Stochastic Weight Averaging (SWA)
**Current:** Use last checkpoint as final model  
**Problem:** Final model might be in a narrow minimum, sensitive to noise

**Solution:**
```python
from torch.optim.swa_utils import AveragedModel, SWALR

# Wrap model with SWA (start averaging after 70% of training)
swa_model = AveragedModel(model)
swa_start = int(args.epochs * 0.7)

# In training loop, after epoch >= swa_start:
if epoch >= swa_start:
    swa_model.update_parameters(model)

# After training:
# Update SWA batch norm statistics
torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

# Use swa_model for inference
```

**Expected Gain:** +0.5-1% F1 (more robust final model)

### 🔧 Optimization 9: Label Smoothing
**Current:** Hard targets (0 or 1)  
**Problem:** Overconfident predictions, poor calibration

**Solution:**
```python
def label_smoothing(labels, smoothing=0.1):
    """Apply label smoothing: 1 → 0.95, 0 → 0.05"""
    return labels * (1 - smoothing) + smoothing * 0.5
    
# In train_epoch():
labels_smoothed = label_smoothing(labels, smoothing=0.1)
loss = criterion(outputs, labels_smoothed)
```

**Expected Gain:** +0.5-1% F1 (better calibration, less overfitting)

---

## Stage 4: Validation & Inference

### Current Implementation
```python
# validate()
- Standard inference: ✅
- Threshold: 0.5 (fixed)
- Metrics: Macro F1, per-class F1
```

### 🔧 Optimization 10: Test-Time Augmentation (TTA)
**Current:** Single inference pass  
**Problem:** Model predictions can vary with small perturbations

**Solution:**
```python
def predict_with_tta(model, image, device, num_augmentations=5):
    """Predict with test-time augmentation."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original
        predictions.append(model(image.unsqueeze(0).to(device)))
        
        # Horizontal flip
        predictions.append(model(torch.flip(image.unsqueeze(0), dims=[3]).to(device)))
        
        # Vertical flip
        predictions.append(model(torch.flip(image.unsqueeze(0), dims=[2]).to(device)))
        
        # 90° rotation
        predictions.append(model(torch.rot90(image.unsqueeze(0), k=1, dims=[2, 3]).to(device)))
        
        # 180° rotation
        predictions.append(model(torch.rot90(image.unsqueeze(0), k=2, dims=[2, 3]).to(device)))
    
    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return torch.sigmoid(avg_pred)
```

**Expected Gain:** +1-2% F1 (more robust predictions)

### 🔧 Optimization 11: Per-Class Threshold Optimization
**Current:** Fixed threshold 0.5 for all classes  
**Problem:** Different diseases have different optimal thresholds (Myopia 0.3, Diabetes 0.6)

**Solution:**
```python
from sklearn.metrics import f1_score

def optimize_thresholds(model, val_loader, device):
    """Find optimal threshold per class."""
    model.eval()
    all_preds = []
    all_labels = []
    
    # Collect predictions
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Optimize threshold per class
    optimal_thresholds = []
    for class_idx in range(7):
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds_binary = (all_preds[:, class_idx] > threshold).astype(int)
            f1 = f1_score(all_labels[:, class_idx], preds_binary, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        print(f"Class {class_idx}: optimal threshold = {best_threshold:.2f}, F1 = {best_f1:.4f}")
    
    return np.array(optimal_thresholds)
```

**Expected Gain:** +2-4% F1 (huge impact, especially for rare classes)

---

## Stage 5: Ensemble & Post-Processing

### Current Implementation
```python
# Planned: Simple averaging of 3 models
ensemble_pred = (pred_convnext + pred_vit + pred_efficientnet) / 3
```

### 🔧 Optimization 12: Weighted Ensemble with Model Selection
**Current:** Equal weights (1/3, 1/3, 1/3)  
**Problem:** Models have different strengths (ConvNeXt better for textures, ViT for global patterns)

**Solution:**
```python
def optimize_ensemble_weights(predictions_dict, val_labels):
    """Find optimal ensemble weights per class."""
    from scipy.optimize import minimize
    
    def ensemble_loss(weights):
        weights = np.abs(weights)
        weights = weights / weights.sum()
        
        ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions_dict.values()))
        ensemble_binary = (ensemble_pred > 0.5).astype(int)
        
        return -f1_score(val_labels, ensemble_binary, average='macro')
    
    # Optimize weights
    initial_weights = np.ones(len(predictions_dict))
    result = minimize(ensemble_loss, initial_weights, method='Nelder-Mead')
    
    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()
    
    return optimal_weights

# Usage:
predictions = {
    'convnext': convnext_predictions,
    'vit': vit_predictions,
    'efficientnet': efficientnet_predictions
}
weights = optimize_ensemble_weights(predictions, val_labels)
ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions.values()))
```

**Expected Gain:** +1-2% F1 (vs simple averaging)

---

## Implementation Priority & Timeline

### Phase 1: Quick Wins (Implement Before Training) - 30 minutes
1. ✅ **Multi-label stratified split (75/25)** - Already implemented
2. 🔧 **Adaptive CLAHE** - 5 minutes
3. 🔧 **Multi-scale vessel enhancement** - 10 minutes
4. 🔧 **Drusen feature enhancement** - 10 minutes
5. 🔧 **Label smoothing** - 5 minutes

**Expected gain: +2-3% F1**

### Phase 2: Training Improvements (Implement Before Training) - 15 minutes
6. 🔧 **OneCycleLR scheduler** - 5 minutes
7. 🔧 **Stochastic Weight Averaging** - 10 minutes

**Expected gain: +1-2% F1**

### Phase 3: Augmentation (Implement Before Training) - 30 minutes
8. 🔧 **RetinalMix** - 15 minutes
9. 🔧 **Focal region augmentation** - 10 minutes
10. 🔧 **Disease-specific color jitter** - 5 minutes

**Expected gain: +1-2% F1**

### Phase 4: Inference Improvements (After Training) - 1 hour
11. 🔧 **Test-Time Augmentation** - 30 minutes
12. 🔧 **Per-class threshold optimization** - 10 minutes
13. 🔧 **Weighted ensemble** - 20 minutes

**Expected gain: +3-5% F1**

---

## Conservative Estimates vs Optimistic

### Conservative Path (Implement 1-5 only)
- Preprocessing improvements: +2% F1
- New split: +3% F1
- **Total: 62-64% single model, 66-68% ensemble**

### Moderate Path (Implement 1-10)
- All above + training + augmentation improvements: +5-7% F1
- **Total: 64-66% single model, 68-71% ensemble**

### Aggressive Path (Implement all 12)
- Full optimization stack: +8-11% F1
- **Total: 65-68% single model, 70-73% ensemble**

---

## Recommendation

**Implement Priority 1 (Quick Wins) immediately before reprocessing data:**
1. Adaptive CLAHE
2. Multi-scale vessel enhancement
3. Drusen feature enhancement
4. OneCycleLR scheduler
5. Label smoothing

**These 5 changes take 35 minutes and give +3-5% F1 improvement.**

Then train and evaluate. If results are strong (65%+ single model), proceed to Phase 4 for inference optimizations. If results plateau (62-63%), implement Phase 3 augmentation improvements and retrain.

**Expected timeline:**
- Implement optimizations: 35 minutes
- Reprocess data with optimizations: 15 minutes
- Train ConvNeXt: 5 hours
- Train ViT: 5 hours
- Train EfficientNetV2: 4 hours
- Threshold optimization + ensemble: 1 hour
- **Total: 16 hours to 68-72% F1 ensemble**

This beats Phase 4A baseline (64.63%) with high confidence and reaches clinical screening threshold for several diseases.
