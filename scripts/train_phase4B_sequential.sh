#!/bin/bash
# Phase 4B: Train all 3 models sequentially overnight
# Expected total time: ~10 hours (3-4 hours per model)

set -e  # Exit on error

echo "=========================================="
echo "PHASE 4B: SEQUENTIAL TRAINING"
echo "=========================================="
echo "Started: $(date)"
echo ""
echo "Models to train:"
echo "  1. ConvNeXt Tiny (~3.5 hours)"
echo "  2. ViT Small (~3.5 hours)"
echo "  3. EfficientNetV2 Small (~3 hours)"
echo ""
echo "Total estimated time: ~10 hours"
echo "=========================================="
echo ""

# Activate virtual environment
source .venv/bin/activate

# Model 1: ConvNeXt Tiny
echo "=========================================="
echo "MODEL 1/3: ConvNeXt Tiny"
echo "Started: $(date)"
echo "=========================================="
python scripts/train_advanced.py --model convnext_tiny --epochs 50 --batch-size 64
echo "✓ ConvNeXt Tiny complete: $(date)"
echo ""

# Model 2: ViT Small
echo "=========================================="
echo "MODEL 2/3: ViT Small"
echo "Started: $(date)"
echo "=========================================="
python scripts/train_advanced.py --model vit_small --epochs 50 --batch-size 64
echo "✓ ViT Small complete: $(date)"
echo ""

# Model 3: EfficientNetV2 Small
echo "=========================================="
echo "MODEL 3/3: EfficientNetV2 Small"
echo "Started: $(date)"
echo "=========================================="
python scripts/train_advanced.py --model efficientnetv2_s --epochs 50 --batch-size 64
echo "✓ EfficientNetV2 Small complete: $(date)"
echo ""

# Summary
echo "=========================================="
echo "PHASE 4B TRAINING COMPLETE!"
echo "=========================================="
echo "Finished: $(date)"
echo ""
echo "Next steps:"
echo "  1. Optimize thresholds: python scripts/optimize_phase4_thresholds.py"
echo "  2. Compare with Phase 4A: python scripts/compare_phase4A_vs_4B.py"
echo "  3. Expected result: 67-70% F1 (+3-5% from Phase 4A)"
echo ""
