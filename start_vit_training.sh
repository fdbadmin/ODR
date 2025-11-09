#!/bin/bash

cd /Users/fdb/VSCode/ODR

# Kill any existing training
ps aux | grep -E "[P]ython.*vit_base" | awk '{print $2}' | xargs kill -9 2>/dev/null

# Start training
.venv/bin/python scripts/train_cutting_edge.py \
  --model vit_base \
  --epochs 40 \
  --batch-size 16 \
  --lr 3e-5 \
  --grad-accum-steps 4 \
  --grad-clip 1.0 \
  --use-amp \
  --data-dir preprocessed_data_smart_exclusion \
  --output-dir models_vit_base \
  > training_vit_base.log 2>&1 &

# Get PID
PID=$!
echo "ViT-Base training started with PID: $PID"
echo $PID > vit_training.pid

# Wait a bit and check if it's running
sleep 5
if ps -p $PID > /dev/null; then
   echo "✅ Training is running!"
   echo "Monitor with: tail -f training_vit_base.log"
else
   echo "❌ Training failed to start. Check training_vit_base.log"
   cat training_vit_base.log
fi
