#!/bin/bash

echo "════════════════════════════════════════════════════════════════════════════"
echo "              RETFound Setup - Download Pre-trained Weights"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "RETFound is a foundation model trained on 1.6M retinal fundus images"
echo "for generalizable disease detection."
echo ""
echo "Reference: https://github.com/rmaphoh/RETFound_MAE"
echo "Paper: Nature (2023)"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Create models directory
mkdir -p models

# Check if weights already exist
if [ -f "models/RETFound_cfp_weights.pth" ]; then
    echo "✅ RETFound weights already downloaded: models/RETFound_cfp_weights.pth"
    echo ""
    exit 0
fi

echo "Downloading RETFound pre-trained weights..."
echo ""
echo "Available versions:"
echo "  1. RETFound_cfp_weights.pth - Color fundus photography (RECOMMENDED for ODIR-5K)"
echo "  2. RETFound_oct_weights.pth - Optical coherence tomography"
echo ""

# Download CFP weights (best for fundus photography)
echo "Downloading CFP weights (best for your fundus dataset)..."
cd models

# Try wget first
if command -v wget &> /dev/null; then
    wget https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_cfp_weights.pth
elif command -v curl &> /dev/null; then
    curl -L -O https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_cfp_weights.pth
else
    echo "⚠️  Neither wget nor curl found. Please download manually:"
    echo "   https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_cfp_weights.pth"
    echo ""
    echo "   Save to: models/RETFound_cfp_weights.pth"
    exit 1
fi

cd ..

# Check if download was successful
if [ -f "models/RETFound_cfp_weights.pth" ]; then
    file_size=$(du -h models/RETFound_cfp_weights.pth | cut -f1)
    echo ""
    echo "✅ RETFound weights downloaded successfully!"
    echo "   Location: models/RETFound_cfp_weights.pth"
    echo "   Size: $file_size"
    echo ""
    echo "You can now train with:"
    echo "   python scripts/train_cutting_edge.py --model retfound --retfound-path models/RETFound_cfp_weights.pth"
    echo ""
else
    echo "❌ Download failed. Please download manually from:"
    echo "   https://github.com/rmaphoh/RETFound_MAE/releases"
    echo ""
    exit 1
fi
