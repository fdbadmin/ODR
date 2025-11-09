"""
Threshold Tuning for EfficientNet-B5 Model
==========================================

Optimizes per-class decision thresholds to maximize F1 score.
Default threshold of 0.5 is rarely optimal for imbalanced datasets.

Expected improvement: +4-6% F1 score (based on ResNet-50 results)
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
        for images, labels in tqdm(val_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    print(f"   Got predictions for {len(all_probs)} samples")
    return all_probs, all_labels


def optimize_thresholds(probs, labels, class_names):
    """Find optimal threshold for each class"""
    print("\n🔍 Optimizing thresholds per class...")
    print("   Testing thresholds from 0.1 to 0.9...")
    
    n_classes = len(class_names)
    optimal_thresholds = []
    baseline_f1_per_class = []
    optimized_f1_per_class = []
    
    # Test thresholds
    test_thresholds = np.arange(0.1, 0.91, 0.01)
    
    for class_idx in range(n_classes):
        class_probs = probs[:, class_idx]
        class_labels = labels[:, class_idx]
        
        # Baseline with 0.5
        baseline_preds = (class_probs >= 0.5).astype(int)
        baseline_f1 = f1_score(class_labels, baseline_preds, zero_division=0)
        baseline_f1_per_class.append(baseline_f1)
        
        # Find best threshold
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in test_thresholds:
            preds = (class_probs >= threshold).astype(int)
            f1 = f1_score(class_labels, preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        optimized_f1_per_class.append(best_f1)
        
        improvement = best_f1 - baseline_f1
        print(f"   {class_names[class_idx]:12s}: threshold={best_threshold:.3f}, "
              f"F1={best_f1:.4f} (baseline: {baseline_f1:.4f}, Δ={improvement:+.4f})")
    
    return optimal_thresholds, baseline_f1_per_class, optimized_f1_per_class


def evaluate_with_thresholds(probs, labels, thresholds):
    """Evaluate model using optimized thresholds"""
    n_classes = probs.shape[1]
    preds = np.zeros_like(probs)
    
    for i in range(n_classes):
        preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)
    
    # Calculate metrics
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    
    return f1_macro, f1_per_class, precision, recall


def main():
    print("="*80)
    print("  EfficientNet-B5 Threshold Optimization")
    print("="*80)
    
    # Configuration
    data_dir = 'preprocessed_data_smart_exclusion'
    model_path = 'models_efficientnet_b5/best_efficientnet_b5.pth'
    output_path = 'models_efficientnet_b5/optimized_thresholds.json'
    
    class_names = ['AMD', 'Diabetes', 'Glaucoma', 'Cataract', 'Myopia', 'Normal', 'Other']
    num_classes = len(class_names)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("\n✅ Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print("\n⚠️  MPS not available, using CPU")
    
    # Load data
    val_images, val_labels = load_data(data_dir)
    
    # Create dataset (no augmentation for validation)
    val_dataset = FundusDataset(
        val_images, 
        val_labels, 
        augment=False
    )
    
    # Load model
    print(f"\n🔨 Loading EfficientNet-B5 model from {model_path}...")
    model = create_model('efficientnet_b5', num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("   ✅ Model loaded successfully")
    
    # Get predictions
    probs, labels = get_predictions(model, val_dataset, device=device, batch_size=32)
    
    # Baseline performance (threshold=0.5)
    print("\n" + "="*80)
    print("  BASELINE PERFORMANCE (threshold=0.5 for all classes)")
    print("="*80)
    
    baseline_preds = (probs >= 0.5).astype(int)
    baseline_f1_macro = f1_score(labels, baseline_preds, average='macro', zero_division=0)
    baseline_f1_per_class = f1_score(labels, baseline_preds, average=None, zero_division=0)
    baseline_precision = precision_score(labels, baseline_preds, average='macro', zero_division=0)
    baseline_recall = recall_score(labels, baseline_preds, average='macro', zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  F1 Score:  {baseline_f1_macro:.4f} ({baseline_f1_macro*100:.2f}%)")
    print(f"  Precision: {baseline_precision:.4f} ({baseline_precision*100:.2f}%)")
    print(f"  Recall:    {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
    
    print(f"\nPer-Class F1 Scores:")
    for i, name in enumerate(class_names):
        print(f"  {name:12s}: {baseline_f1_per_class[i]:.4f}")
    
    # Optimize thresholds
    print("\n" + "="*80)
    print("  OPTIMIZING THRESHOLDS")
    print("="*80)
    
    optimal_thresholds, _, _ = optimize_thresholds(probs, labels, class_names)
    
    # Evaluate with optimized thresholds
    print("\n" + "="*80)
    print("  OPTIMIZED PERFORMANCE (per-class thresholds)")
    print("="*80)
    
    opt_f1_macro, opt_f1_per_class, opt_precision, opt_recall = evaluate_with_thresholds(
        probs, labels, optimal_thresholds
    )
    
    print(f"\nOverall Metrics:")
    print(f"  F1 Score:  {opt_f1_macro:.4f} ({opt_f1_macro*100:.2f}%)")
    print(f"  Precision: {opt_precision:.4f} ({opt_precision*100:.2f}%)")
    print(f"  Recall:    {opt_recall:.4f} ({opt_recall*100:.2f}%)")
    
    print(f"\nPer-Class F1 Scores:")
    for i, name in enumerate(class_names):
        improvement = opt_f1_per_class[i] - baseline_f1_per_class[i]
        print(f"  {name:12s}: {opt_f1_per_class[i]:.4f} "
              f"(baseline: {baseline_f1_per_class[i]:.4f}, Δ={improvement:+.4f})")
    
    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    
    f1_improvement = opt_f1_macro - baseline_f1_macro
    f1_improvement_pct = (f1_improvement / baseline_f1_macro) * 100
    
    print(f"\nBaseline F1:  {baseline_f1_macro:.4f} ({baseline_f1_macro*100:.2f}%)")
    print(f"Optimized F1: {opt_f1_macro:.4f} ({opt_f1_macro*100:.2f}%)")
    print(f"Improvement:  {f1_improvement:+.4f} ({f1_improvement_pct:+.2f}%)")
    
    print(f"\nOptimized Thresholds:")
    for i, name in enumerate(class_names):
        print(f"  {name:12s}: {optimal_thresholds[i]:.3f}")
    
    # Save thresholds
    threshold_config = {
        'thresholds': {name: float(thresh) for name, thresh in zip(class_names, optimal_thresholds)},
        'baseline_f1': float(baseline_f1_macro),
        'optimized_f1': float(opt_f1_macro),
        'improvement': float(f1_improvement),
        'baseline_f1_per_class': {name: float(f1) for name, f1 in zip(class_names, baseline_f1_per_class)},
        'optimized_f1_per_class': {name: float(f1) for name, f1 in zip(class_names, opt_f1_per_class)},
        'model': 'efficientnet_b5',
        'model_path': model_path
    }
    
    with open(output_path, 'w') as f:
        json.dump(threshold_config, f, indent=2)
    
    print(f"\n✅ Optimized thresholds saved to: {output_path}")
    print("\n" + "="*80)
    print("  🎉 Threshold optimization complete!")
    print("="*80)


if __name__ == '__main__':
    main()
