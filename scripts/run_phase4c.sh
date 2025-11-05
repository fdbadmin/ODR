#!/bin/bash
# Phase 4C Quick Start Script
# Run this after Phase 4B completes

echo "=========================================="
echo "PHASE 4C SETUP & EXECUTION"
echo "=========================================="
echo ""

# Step 1: Create branch
echo "Step 1: Creating Phase 4C branch..."
git checkout -b phase4C-rgb-highres
git add src/phase4c_preprocessing.py src/data_preprocessing_phase4c.py docs/PHASE4C_READY.md
git commit -m "Phase 4C: RGB color + 384x384 + optimized vessel enhancement"
echo "✓ Branch created"
echo ""

# Step 2: Run preprocessing
echo "Step 2: Running preprocessing (~15-20 minutes)..."
python src/data_preprocessing_phase4c.py
echo "✓ Preprocessing complete"
echo ""

# Step 3: Update training script data path
echo "Step 3: Update training script to use phase4c data..."
echo "Manual step: Edit scripts/train_advanced.py line ~250"
echo "Change: data_dir = Path('preprocessed_data')"
echo "To:     data_dir = Path('preprocessed_data_phase4c')"
echo ""
read -p "Press Enter after updating train_advanced.py..."

# Step 4: Train models sequentially
echo ""
echo "Step 4: Starting sequential training (~12-15 hours total)..."
echo ""

# ConvNeXt Tiny
echo "Training ConvNeXt Tiny..."
M5_NUM_WORKERS=4 python scripts/train_advanced.py \
  --model convnext_tiny --epochs 50 --batch-size 32

# ViT Small
echo ""
echo "Training ViT Small..."
M5_NUM_WORKERS=4 python scripts/train_advanced.py \
  --model vit_small --epochs 50 --batch-size 32

# EfficientNetV2 Small
echo ""
echo "Training EfficientNetV2 Small..."
M5_NUM_WORKERS=4 python scripts/train_advanced.py \
  --model efficientnetv2_s --epochs 50 --batch-size 32

echo ""
echo "=========================================="
echo "PHASE 4C TRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "Next: Run threshold optimization"
echo "  python scripts/optimize_phase4_thresholds.py --phase 4c"
echo ""
