# Advanced Training Strategy

## Overview
We've implemented data augmentation and modern architectures to improve upon the 64.03% F1 baseline.

## Available Models

### Modern Architectures (NEW)
1. **Vision Transformer (ViT) Small** - 21.7M params
   - Attention-based, excellent for complex patterns
   - Best for: Capturing global dependencies

2. **ConvNeXt Tiny** - 27.8M params
   - Modern CNN, efficient and accurate
   - Best for: Good balance of speed and accuracy

3. **Swin Transformer Tiny** - 27.5M params
   - Hierarchical transformer with shifted windows
   - Best for: Medical images with hierarchical structures

4. **EfficientNetV2 Small** - 20.2M params
   - Optimized for training speed
   - Best for: Fast iterations

### Baseline Models (for comparison)
5. ResNet50
6. EfficientNet-B3
7. DenseNet-121

## Augmentation Strategy

### Per-Image Augmentations
- Horizontal/Vertical Flip (50%)
- Random Rotation (±20°)
- Color Jitter (brightness, contrast, saturation, hue)
- Random Affine (translate, scale, shear)
- Random Erasing (occlusions/artifacts)

### Batch Augmentations
- **MixUp**: Mixes two samples and their labels
- **CutMix**: Cuts and pastes patches between samples

## Training Commands

### Quick Start: Best 2 Models
```bash
# ConvNeXt Tiny with MixUp (Recommended - Best balance)
python scripts/train_advanced.py --model convnext_tiny --epochs 50 --lr 1e-4 --use-mixup

# Vision Transformer Small with CutMix (Alternative - Better for complex patterns)
python scripts/train_advanced.py --model vit_small --epochs 50 --lr 1e-4 --use-cutmix
```

### Full Suite (3 models, ~2-3 hours total)
```bash
# 1. ConvNeXt Tiny with MixUp
python scripts/train_advanced.py --model convnext_tiny --epochs 50 --use-mixup

# 2. Vision Transformer Small with CutMix  
python scripts/train_advanced.py --model vit_small --epochs 50 --use-cutmix

# 3. EfficientNetV2 Small with both
python scripts/train_advanced.py --model efficientnetv2_s --epochs 50 --use-mixup --use-cutmix
```

### Extended Training (if we have time)
```bash
# Swin Transformer
python scripts/train_advanced.py --model swin_tiny --epochs 50 --use-mixup
```

## Expected Results

Based on literature and our current performance:

**Current Baseline (from scratch, 50 epochs):**
- Best single model: 62.79% F1 (DenseNet-121)
- Ensemble: 64.03% F1

**Expected with Advanced Training:**
- ConvNeXt Tiny: **68-72% F1** (+4-8%)
- ViT Small: **66-70% F1** (+2-6%)
- EfficientNetV2 Small: **67-71% F1** (+3-7%)
- **Advanced Ensemble: 70-75% F1** (+6-11%)

### Why This Should Work Better

1. **Better Architectures**: Modern models (ViT, ConvNeXt) have better inductive biases for medical images
2. **Data Augmentation**: Increases effective dataset size and prevents overfitting
3. **MixUp/CutMix**: Regularization that improves generalization
4. **Pretrained on ImageNet**: Transfer learning from natural images helps with fundus features

## Recommendation

**Start with ConvNeXt Tiny + MixUp:**
```bash
python scripts/train_advanced.py --model convnext_tiny --epochs 50 --lr 1e-4 --use-mixup
```

This is the best single model choice based on:
- Modern CNN architecture (state-of-the-art)
- 27.8M params (not too large, trains in ~40-50 min)
- MixUp augmentation (proven effective for medical imaging)
- Expected ~70% F1 (vs 64% current)

If it works well, we can train 2-3 more models and create an ensemble for 72-75% F1.
