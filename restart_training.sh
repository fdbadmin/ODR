#!/bin/bash
# Clean and restart training with 7 classes (no Hypertension)

echo "=============================================================================="
echo "🧹 CLEANING OLD MODELS"
echo "=============================================================================="
echo ""

# Backup old models
if [ -d "models" ] && [ "$(ls -A models/*.pth 2>/dev/null)" ]; then
    echo "📦 Backing up old 8-class models..."
    mkdir -p models_8class_backup
    mv models/*.pth models_8class_backup/ 2>/dev/null
    mv models/*.json models_8class_backup/ 2>/dev/null
    echo "✓ Old models backed up to: models_8class_backup/"
else
    echo "ℹ️  No old models found to backup"
fi

# Clean any old optimal thresholds
if [ -f "optimal_thresholds.json" ]; then
    echo "📦 Backing up old optimal thresholds..."
    mv optimal_thresholds.json models_8class_backup/optimal_thresholds_8class.json 2>/dev/null
    echo "✓ Old thresholds backed up"
fi

echo ""
echo "=============================================================================="
echo "📊 CURRENT CONFIGURATION"
echo "=============================================================================="
echo ""
echo "Label columns: ['N', 'D', 'G', 'C', 'A', 'M', 'O']"
echo "Number of classes: 7"
echo "Default epochs: 25"
echo "Removed: Hypertension (H)"
echo ""

# Check preprocessed data
if [ -f "preprocessed_data/train_labels.npy" ]; then
    echo "✅ Preprocessed data ready"
    echo "   Train set: $(python3 -c "import numpy as np; print(np.load('preprocessed_data/train_labels.npy').shape)")"
    echo "   Val set:   $(python3 -c "import numpy as np; print(np.load('preprocessed_data/val_labels.npy').shape)")"
else
    echo "⚠️  Warning: Preprocessed data not found!"
    echo "   Run: python src/utils.py to create it"
fi

echo ""
echo "=============================================================================="
echo "🚀 READY TO TRAIN"
echo "=============================================================================="
echo ""
echo "Start training with:"
echo ""
echo "  1. Baseline ResNet50 (25 epochs):"
echo "     python src/train.py"
echo ""
echo "  2. Ensemble models (25 epochs each):"
echo "     python src/train_ensemble_models.py --model efficientnet_b3 --epochs 25"
echo "     python src/train_ensemble_models.py --model densenet121 --epochs 25"
echo ""
echo "  3. Or train both ensemble models:"
echo "     python src/train_ensemble_models.py --model both --epochs 25"
echo ""
echo "=============================================================================="
