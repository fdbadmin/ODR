#!/bin/bash

# Monitor EfficientNet-B5 Training Progress
# Shows epoch progress, loss, and F1 scores in real-time

LOG_FILE="training_efficientnet_b5.log"

echo "======================================================================"
echo "  EfficientNet-B5 Training Monitor"
echo "======================================================================"
echo ""

# Check if training is running
if ps aux | grep -E "[P]ython.*train_cutting_edge.*efficientnet_b5" > /dev/null; then
    echo "✅ Training process is RUNNING"
    PID=$(ps aux | grep -E "[P]ython.*train_cutting_edge.*efficientnet_b5" | awk '{print $2}')
    echo "   Process ID: $PID"
else
    echo "⚠️  Training process NOT FOUND - checking log file..."
fi

echo ""
echo "----------------------------------------------------------------------"
echo "  Recent Progress (last 50 lines)"
echo "----------------------------------------------------------------------"
echo ""

if [ -f "$LOG_FILE" ]; then
    tail -50 "$LOG_FILE"
    echo ""
    echo "----------------------------------------------------------------------"
    echo "  Live Monitoring (Ctrl+C to exit)"
    echo "----------------------------------------------------------------------"
    echo ""
    
    # Follow the log file in real-time
    tail -f "$LOG_FILE"
else
    echo "❌ Log file not found: $LOG_FILE"
    echo ""
    echo "Expected location: $(pwd)/$LOG_FILE"
    echo ""
    echo "Available log files:"
    ls -lh training_*.log 2>/dev/null || echo "No training logs found"
fi
