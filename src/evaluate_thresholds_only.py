"""
Evaluate ensemble with only optimal thresholds (no TTA).
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

from src.evaluate_ensemble import EnsemblePredictor, LABEL_COLUMNS

LABEL_NAMES = {
    'N': 'Normal',
    'D': 'Diabetes',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'AMD',
    'M': 'Myopia',
    'O': 'Other'
}


def main():
    """Evaluate ensemble with optimal thresholds only (no TTA)."""
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION: OPTIMAL THRESHOLDS ONLY (no TTA)")
    print("="*70 + "\n")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Model configurations
    model_configs = [
        {'type': 'resnet50', 'path': 'models/baseline_model.pth'},
        {'type': 'efficientnet_b3', 'path': 'models/efficientnet_b3_model.pth'},
        {'type': 'densenet121', 'path': 'models/densenet121_model.pth'}
    ]
    
    # Load optimal thresholds
    print("Loading optimal thresholds...")
    with open('results/optimal_thresholds.json', 'r') as f:
        threshold_results = json.load(f)
    optimal_thresholds = threshold_results['optimal_thresholds']
    print("  Optimal thresholds loaded")
    for disease in LABEL_COLUMNS:
        print(f"    {LABEL_NAMES[disease]:12}: {optimal_thresholds[disease]:.3f}")
    
    # Load validation data
    print("\nLoading validation data...")
    val_images = np.load('preprocessed_data/val_images.npy')
    val_labels = np.load('preprocessed_data/val_labels.npy')
    print(f"  Images: {val_images.shape}")
    print(f"  Labels: {val_labels.shape}")
    
    # Create DataLoader
    val_images_tensor = torch.FloatTensor(val_images).permute(0, 3, 1, 2)
    val_dataset = TensorDataset(val_images_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create ensemble (no TTA)
    print("\nLoading ensemble models...")
    ensemble = EnsemblePredictor(model_configs, device=device)
    
    # Generate predictions
    print("\nGenerating predictions (no TTA)...")
    predictions = ensemble.predict_batch(val_loader, desc="  Predicting")
    
    print(f"\nPredictions shape: {predictions.shape}")
    
    # Apply optimal thresholds
    print("\nApplying optimal thresholds...")
    pred_binary = np.zeros_like(predictions)
    for i, disease in enumerate(LABEL_COLUMNS):
        threshold = optimal_thresholds[disease]
        pred_binary[:, i] = (predictions[:, i] >= threshold).astype(int)
    
    # Calculate metrics
    print("Calculating metrics...")
    
    per_class_metrics = {}
    f1_scores = []
    
    for i, label in enumerate(LABEL_COLUMNS):
        f1 = f1_score(val_labels[:, i], pred_binary[:, i], zero_division=0)
        precision = precision_score(val_labels[:, i], pred_binary[:, i], zero_division=0)
        recall = recall_score(val_labels[:, i], pred_binary[:, i], zero_division=0)
        
        per_class_metrics[label] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'threshold': optimal_thresholds[label],
            'support': int(val_labels[:, i].sum())
        }
        f1_scores.append(f1)
    
    mean_f1 = np.mean(f1_scores)
    label_accuracy = (pred_binary == val_labels).mean()
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS: OPTIMAL THRESHOLDS ONLY")
    print("="*70)
    print(f"\n  Mean F1: {mean_f1:.4f}")
    print(f"  Label Accuracy: {label_accuracy:.4f}")
    
    print(f"\n  Per-class F1 (with optimal thresholds):")
    for i, label in enumerate(LABEL_COLUMNS):
        metrics = per_class_metrics[label]
        print(f"    {LABEL_NAMES[label]:12}: F1={metrics['f1']:.4f}, "
              f"Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, "
              f"Threshold={metrics['threshold']:.3f}")
    
    # Load baseline results for comparison
    with open('results/ensemble_evaluation_results.json', 'r') as f:
        baseline_results = json.load(f)
    
    baseline_f1 = baseline_results['ensemble']['mean_f1']
    
    # Also load threshold optimization results
    default_f1 = threshold_results['default_mean_f1']
    expected_f1 = threshold_results['optimized_mean_f1']
    
    improvement_vs_baseline = ((mean_f1 - baseline_f1) / baseline_f1) * 100
    improvement_vs_default = ((mean_f1 - default_f1) / default_f1) * 100
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\n  Baseline Ensemble (threshold=0.5):           {baseline_f1:.4f}")
    print(f"  Optimized Ensemble (optimal thresholds):      {mean_f1:.4f}")
    print(f"  Improvement:                                  {improvement_vs_baseline:+.2f}%")
    print(f"\n  Expected from threshold optimization:        {expected_f1:.4f}")
    print(f"  Actual result:                                {mean_f1:.4f}")
    print(f"  Match: {'✓ YES' if abs(mean_f1 - expected_f1) < 0.001 else '✗ NO'}")
    
    # Save results
    results = {
        'mean_f1': float(mean_f1),
        'label_accuracy': float(label_accuracy),
        'per_class_f1': {label: float(per_class_metrics[label]['f1']) for label in LABEL_COLUMNS},
        'per_class_metrics': {label: {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                     for k, v in metrics.items()} 
                             for label, metrics in per_class_metrics.items()},
        'baseline_f1': float(baseline_f1),
        'improvement_percent': float(improvement_vs_baseline),
        'configuration': {
            'tta': False,
            'optimal_thresholds': optimal_thresholds
        }
    }
    
    output_path = Path('results/thresholds_only_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("\n" + "="*70)
    print("✓ THRESHOLD OPTIMIZATION EVALUATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
