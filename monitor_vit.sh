#!/bin/bash

echo "================================================================================"
echo "                  ViT-BASE TRAINING MONITOR"
echo "================================================================================"
echo ""

# Check if training is running
if ps aux | grep -E "[P]ython.*vit_base" > /dev/null; then
    PID=$(ps aux | grep -E "[P]ython.*vit_base" | awk '{print $2}')
    echo "✅ Training is RUNNING (PID: $PID)"
else
    echo "❌ Training is NOT running"
    exit 1
fi

echo ""
echo "Latest output:"
echo "--------------------------------------------------------------------------------"
tail -30 training_vit_base.log
echo "--------------------------------------------------------------------------------"
echo ""
echo "To monitor live: tail -f training_vit_base.log"
echo "To stop training: kill $PID"
echo "================================================================================"
