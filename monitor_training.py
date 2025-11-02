#!/usr/bin/env python3
"""
Training Monitor Script
Checks training progress and provides estimates
"""

import json
import time
import os
from pathlib import Path

def load_history():
    """Load training history if it exists"""
    history_file = Path("models/improved_focal_balanced_history.json")
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    return None

def print_status():
    """Print current training status"""
    history = load_history()
    
    print("=" * 80)
    print("🚀 TRAINING PROGRESS MONITOR")
    print("=" * 80)
    
    if history is None:
        print("❌ Training history not found yet. Training may have just started.")
        print("   History is saved after each epoch completes.")
        return
    
    # Get latest epoch data
    num_epochs = len(history['train_loss'])
    if num_epochs == 0:
        print("📊 No completed epochs yet...")
        return
    
    print(f"\n📈 Progress: {num_epochs}/15 epochs completed ({num_epochs/15*100:.1f}%)")
    print(f"⏱️  Estimated time remaining: {(15 - num_epochs) * 1.7:.0f} minutes")
    
    # Latest metrics
    print(f"\n📊 Latest Metrics (Epoch {num_epochs}):")
    print(f"   Train Loss:      {history['train_loss'][-1]:.4f}")
    print(f"   Val Label Acc:   {history['val_label_acc'][-1]:.2%}")
    print(f"   Val Sample Acc:  {history['val_sample_acc'][-1]:.2%}")
    print(f"   Val Mean F1:     {history['val_mean_f1'][-1]:.4f}")
    
    # Best so far
    best_idx = history['val_mean_f1'].index(max(history['val_mean_f1']))
    print(f"\n🏆 Best Performance So Far (Epoch {best_idx + 1}):")
    print(f"   Val Label Acc:   {history['val_label_acc'][best_idx]:.2%}")
    print(f"   Val Sample Acc:  {history['val_sample_acc'][best_idx]:.2%}")
    print(f"   Val Mean F1:     {history['val_mean_f1'][best_idx]:.4f}")
    
    # Trend analysis
    if num_epochs >= 3:
        recent_f1 = history['val_mean_f1'][-3:]
        if recent_f1[-1] > recent_f1[0]:
            print(f"\n📈 Trend: IMPROVING (F1 increased over last 3 epochs)")
        elif recent_f1[-1] < recent_f1[0] * 0.95:
            print(f"\n⚠️  Trend: DECLINING (F1 decreased over last 3 epochs)")
        else:
            print(f"\n➡️  Trend: STABLE (F1 roughly stable over last 3 epochs)")
    
    # Comparison with baseline
    baseline_f1 = 0.550  # Our baseline ensemble
    current_best = max(history['val_mean_f1'])
    if current_best > baseline_f1:
        improvement = (current_best - baseline_f1) / baseline_f1 * 100
        print(f"\n✅ BEATING BASELINE by {improvement:.1f}%!")
        print(f"   Baseline F1: {baseline_f1:.4f}")
        print(f"   Current F1:  {current_best:.4f}")
    else:
        deficit = (baseline_f1 - current_best) / baseline_f1 * 100
        print(f"\n⏳ Still training... ({deficit:.1f}% below baseline)")
        print(f"   Baseline F1: {baseline_f1:.4f}")
        print(f"   Current F1:  {current_best:.4f}")
        print(f"   This is normal in early epochs!")
    
    # Per-disease breakdown (latest epoch)
    if 'val_f1_scores' in history and len(history['val_f1_scores']) > 0:
        print(f"\n📋 Per-Disease F1 Scores (Latest Epoch):")
        diseases = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 
                   'AMD', 'Hypertension', 'Myopia', 'Other']
        latest_f1 = history['val_f1_scores'][-1]
        for disease, f1 in zip(diseases, latest_f1):
            status = "✅" if f1 > 0.5 else "⚠️" if f1 > 0.3 else "❌"
            print(f"      {status} {disease:15s} {f1:.4f}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--watch':
        print("👀 Watching training progress (updates every 30 seconds)...")
        print("   Press Ctrl+C to stop\n")
        try:
            while True:
                print_status()
                print("\n⏰ Refreshing in 30 seconds...")
                time.sleep(30)
                print("\n" * 2)
        except KeyboardInterrupt:
            print("\n\n👋 Stopped monitoring.")
    else:
        print_status()
        print("\nTip: Run with --watch flag to monitor continuously:")
        print("     python monitor_training.py --watch")
