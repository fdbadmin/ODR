"""
Threshold Tuning for Smart Exclusion Model
==========================================

Optimizes per-class decision thresholds to maximize F1 score.
Default threshold of 0.5 is rarely optimal for imbalanced datasets.

Expected improvement: +3-7% F1 score
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import json

# Import model architecture
from scripts.train_cutting_edge import create_model, FundusDataset


def load_data(data_dir):
    """Load validation data"""
    print(f"📂 Loading validation data from {data_dir}...")
    
    val_images = np.load(f'{data_dir}/val_images.npy', mmap_mode='r')
    val_labels = np.load(f'{data_dir}/val_labels.npy')
    
    print(f"   Validation: {len(val_images)} images")
    return val_images, val_labels


def get_predictions(model, val_dataset, device='mps', batch_size=32):
    """Get model predictions (probabilities) for validation set"""
    print("\n📊 Getting model predictions...")
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    return all_probs, all_labels


def optimize_threshold_per_class(probs, labels, class_idx, class_name):
    """Find optimal threshold for a single class"""
    best_threshold = 0.5
    best_f1 = 0.0
    
    # Try thresholds from 0.1 to 0.9 in steps of 0.01
    thresholds = np.arange(0.1, 0.91, 0.01)
    
    for threshold in thresholds:
        preds = (probs[:, class_idx] >= threshold).astype(int)
        f1 = f1_score(labels[:, class_idx], preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def evaluate_with_thresholds(probs, labels, thresholds, class_names):
    """Evaluate using optimized thresholds"""
    preds = np.zeros_like(probs)
    
    for i in range(probs.shape[1]):
        preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)
    
    # Calculate metrics
    f1_per_class = []
    precision_per_class = []
    recall_per_class = []
    
    for i, name in enumerate(class_names):
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        precision = precision_score(labels[:, i], preds[:, i], zero_division=0)
        recall = recall_score(labels[:, i], preds[:, i], zero_division=0)
        
        f1_per_class.append(f1)
        precision_per_class.append(precision)
        recall_per_class.append(recall)
    
    mean_f1 = np.mean(f1_per_class)
    
    return {
        'f1_per_class': f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'mean_f1': mean_f1
    }


def main():
    # Configuration
    MODEL_PATH = 'models_smart_exclusion/best_resnet50.pth'
    DATA_DIR = 'preprocessed_data_smart_exclusion'
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    CLASS_NAMES = ['AMD', 'Diabetes', 'Glaucoma', 'Cataract', 'Myopia', 'Normal', 'Other']
    
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION FOR SMART EXCLUSION MODEL")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    
    # Load validation data
    val_images, val_labels = load_data(DATA_DIR)
    val_dataset = FundusDataset(val_images, val_labels, augment=False)
    
    # Load model
    print(f"\n🔨 Loading model from {MODEL_PATH}...")
    model = create_model('resnet50', num_classes=7, device=DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    print("✅ Model loaded")
    
    # Get predictions
    probs, labels = get_predictions(model, val_dataset, DEVICE)
    
    # Baseline performance (threshold=0.5 for all)
    print("\n" + "=" * 80)
    print("BASELINE PERFORMANCE (threshold=0.5 for all classes)")
    print("=" * 80)
    
    baseline_thresholds = [0.5] * len(CLASS_NAMES)
    baseline_results = evaluate_with_thresholds(probs, labels, baseline_thresholds, CLASS_NAMES)
    
    print(f"\nOverall F1: {baseline_results['mean_f1']:.4f} ({baseline_results['mean_f1']*100:.2f}%)")
    print("\nPer-class performance:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}: F1={baseline_results['f1_per_class'][i]:.4f}, "
              f"Precision={baseline_results['precision_per_class'][i]:.4f}, "
              f"Recall={baseline_results['recall_per_class'][i]:.4f}")
    
    # Optimize thresholds
    print("\n" + "=" * 80)
    print("OPTIMIZING THRESHOLDS")
    print("=" * 80)
    
    optimal_thresholds = []
    optimal_f1s = []
    
    for i, name in enumerate(CLASS_NAMES):
        threshold, f1 = optimize_threshold_per_class(probs, labels, i, name)
        optimal_thresholds.append(threshold)
        optimal_f1s.append(f1)
        print(f"  {name:12s}: threshold={threshold:.3f}, F1={f1:.4f}")
    
    # Evaluate with optimized thresholds
    print("\n" + "=" * 80)
    print("OPTIMIZED PERFORMANCE")
    print("=" * 80)
    
    optimized_results = evaluate_with_thresholds(probs, labels, optimal_thresholds, CLASS_NAMES)
    
    print(f"\nOverall F1: {optimized_results['mean_f1']:.4f} ({optimized_results['mean_f1']*100:.2f}%)")
    print("\nPer-class performance:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}: F1={optimized_results['f1_per_class'][i]:.4f}, "
              f"Precision={optimized_results['precision_per_class'][i]:.4f}, "
              f"Recall={optimized_results['recall_per_class'][i]:.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("IMPROVEMENT SUMMARY")
    print("=" * 80)
    
    improvement = optimized_results['mean_f1'] - baseline_results['mean_f1']
    improvement_pct = (improvement / baseline_results['mean_f1']) * 100
    
    print(f"\nBaseline F1:  {baseline_results['mean_f1']:.4f} ({baseline_results['mean_f1']*100:.2f}%)")
    print(f"Optimized F1: {optimized_results['mean_f1']:.4f} ({optimized_results['mean_f1']*100:.2f}%)")
    print(f"Improvement:  +{improvement:.4f} (+{improvement_pct:.2f}%)")
    
    print("\nPer-class improvements:")
    for i, name in enumerate(CLASS_NAMES):
        base_f1 = baseline_results['f1_per_class'][i]
        opt_f1 = optimized_results['f1_per_class'][i]
        diff = opt_f1 - base_f1
        marker = "✅" if diff > 0.01 else "⚠️" if diff < -0.01 else "➖"
        print(f"  {marker} {name:12s}: {base_f1:.4f} → {opt_f1:.4f} ({diff:+.4f})")
    
    # Save optimized thresholds
    output_file = 'models_smart_exclusion/optimized_thresholds.json'
    thresholds_data = {
        'thresholds': {name: float(thresh) for name, thresh in zip(CLASS_NAMES, optimal_thresholds)},
        'baseline_f1': float(baseline_results['mean_f1']),
        'optimized_f1': float(optimized_results['mean_f1']),
        'improvement': float(improvement),
        'per_class_f1': {name: float(f1) for name, f1 in zip(CLASS_NAMES, optimized_results['f1_per_class'])}
    }
    
    with open(output_file, 'w') as f:
        json.dump(thresholds_data, f, indent=2)
    
    print(f"\n💾 Optimized thresholds saved to: {output_file}")
    print("\n✅ Threshold optimization complete!")


if __name__ == '__main__':
    main()
