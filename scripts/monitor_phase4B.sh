#!/bin/bash
# Monitor Phase 4B training progress

echo "=========================================="
echo "PHASE 4B TRAINING MONITOR"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "PHASE 4B TRAINING MONITOR"
    echo "Updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Check if training is running
    if ps aux | grep -q "[t]rain_advanced.py"; then
        echo "✅ STATUS: Training is RUNNING"
        echo ""
        
        # Show last 30 lines of log (captures progress info)
        echo "📊 LATEST PROGRESS:"
        echo "----------------------------------------"
        tail -30 logs/phase4B_convnext_*.log 2>/dev/null | grep -E "(Epoch|Training:|Validation:|Best|loss)" | tail -10
        echo ""
        
        # Show model checkpoints
        echo "💾 MODEL CHECKPOINTS:"
        ls -lht models/*advanced*.pth 2>/dev/null | head -3
        echo ""
        
    else
        echo "⚠️  STATUS: Training not running"
        echo ""
        echo "Recent completion or error:"
        tail -20 logs/phase4B_*.log 2>/dev/null | tail -10
        echo ""
    fi
    
    echo "=========================================="
    echo "Press Ctrl+C to exit monitor"
    echo "Auto-refresh every 30 seconds..."
    echo "=========================================="
    
    sleep 30
done
