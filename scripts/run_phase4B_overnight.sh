#!/bin/bash
# Phase 4B Overnight Training Launcher
# Optimized for MacBook Pro M5 with 32GB unified memory

echo "=================================================="
echo "Phase 4B Sequential Training - M5 Optimized"
echo "=================================================="
echo ""
echo "This will train 3 models sequentially:"
echo "  1. ConvNeXt Tiny (~3-4 hours)"
echo "  2. ViT Small (~3-4 hours)"
echo "  3. EfficientNetV2 Small (~3-4 hours)"
echo ""
echo "Total estimated time: 10-12 hours"
echo ""
echo "M5 Optimizations:"
echo "  - Batch size: 96 (vs typical 32-64)"
echo "  - Workers: 10 (match 10-core CPU)"
echo "  - Persistent workers: Enabled"
echo "  - Pin memory: Enabled"
echo ""

# Check if preprocessed data exists
if [ ! -f "preprocessed_data/train_images.npy" ]; then
    echo "ERROR: Preprocessed data not found!"
    echo "Run preprocessing first: python src/data_preprocessing_enhanced.py"
    exit 1
fi

echo "✓ Preprocessed data found"
echo ""

# Ask for confirmation
read -p "Start training now? This will run overnight. (y/N): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "Starting training..."
echo ""

# Run the sequential training script
python scripts/train_phase4B_sequential_m5.py

echo ""
echo "Training completed!"
echo ""
echo "Check the results:"
echo "  - Training log: logs/phase4B_training_*.log"
echo "  - Report: results/PHASE4B_TRAINING_REPORT_*.md"
echo "  - Models: models/*_advanced_best.pth"
echo ""
