#!/bin/bash
# Training Monitor Script - Run this in a separate terminal while training

MODEL_NAME=${1:-convnextv2_tiny}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Training Monitor: $MODEL_NAME"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Function to display progress
show_progress() {
    if [ -f "models/progress_${MODEL_NAME}.txt" ]; then
        clear
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║         Training Monitor: $MODEL_NAME"
        echo "╚══════════════════════════════════════════════════════════════╝"
        echo ""
        cat "models/progress_${MODEL_NAME}.txt"
        echo ""
        echo "────────────────────────────────────────────────────────────────"
        echo "Last updated: $(date)"
        echo ""
        
        # Show process status
        if ps aux | grep -q "[t]rain_cutting_edge.py.*$MODEL_NAME"; then
            echo "✅ Training process is RUNNING"
            MEM_USED=$(ps aux | grep "[t]rain_cutting_edge.py" | awk '{print $4}')
            echo "   Memory usage: ${MEM_USED}%"
        else
            echo "⚠️  Training process NOT FOUND"
        fi
    else
        echo "⏳ Waiting for training to start..."
        echo "   Looking for: models/progress_${MODEL_NAME}.txt"
    fi
}

# Monitor loop
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    show_progress
    sleep 30  # Update every 30 seconds
done
