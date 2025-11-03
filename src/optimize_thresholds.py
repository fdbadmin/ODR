"""
Optimize classification thresholds for each disease class.
Find the threshold that maximizes F1 score for each class independently.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from tqdm import tqdm
import json
from pathlib import Path

# Import ensemble predictor
from src.evaluate_ensemble import EnsemblePredictor, LABEL_COLUMNS

# Map label codes to full names
LABEL_NAMES = {
    'N': 'Normal',
    'D': 'Diabetes',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'AMD',
    'M': 'Myopia',
    'O': 'Other'
}


def find_optimal_threshold_for_class(predictions, targets, class_idx, 
                                     threshold_range=np.arange(0.1, 0.9, 0.01)):
    """
    Find optimal threshold for a single class.
    
    Args:
        predictions: Raw probabilities (num_samples,)
        targets: Ground truth labels (num_samples,)
        class_idx: Index of the class
        threshold_range: Range of thresholds to try
        
    Returns:
        best_threshold, best_f1
    """
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in threshold_range:
        pred_binary = (predictions >= threshold).astype(int)
        f1 = f1_score(targets, pred_binary, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def optimize_thresholds(model_configs, val_loader, val_labels, device='mps'):
    """
    Find optimal thresholds for each disease class.
    
    Args:
        model_configs: List of model configurations
        val_loader: DataLoader for validation set
        val_labels: Ground truth validation labels
        device: Device for inference
        
    Returns:
        Dictionary of optimal thresholds per class
    """
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZATION")
    print("="*70 + "\n")
    
    # Create ensemble predictor
    print("Loading ensemble models...")
    ensemble = EnsemblePredictor(model_configs, device=device)
    
    # Generate predictions (probabilities)
    print("\nGenerating predictions on validation set...")
    predictions = ensemble.predict_batch(val_loader, desc="  Generating predictions")
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Labels shape: {val_labels.shape}")
    
    # Find optimal threshold for each class
    optimal_thresholds = {}
    default_f1s = {}
    optimized_f1s = {}
    
    print("\n" + "="*70)
    print("OPTIMIZING THRESHOLDS PER CLASS")
    print("="*70 + "\n")
    
    for i, disease in enumerate(LABEL_COLUMNS):
        disease_name = LABEL_NAMES[disease]
        
        # Get predictions and targets for this class
        class_predictions = predictions[:, i]
        class_targets = val_labels[:, i]
        
        # Calculate F1 with default threshold (0.5)
        default_pred = (class_predictions >= 0.5).astype(int)
        default_f1 = f1_score(class_targets, default_pred, zero_division=0)
        
        # Find optimal threshold
        optimal_threshold, optimal_f1 = find_optimal_threshold_for_class(
            class_predictions, class_targets, i
        )
        
        optimal_thresholds[disease] = float(optimal_threshold)
        default_f1s[disease] = float(default_f1)
        optimized_f1s[disease] = float(optimal_f1)
        
        improvement = ((optimal_f1 - default_f1) / default_f1) * 100 if default_f1 > 0 else 0
        
        print(f"{disease_name:12} | Default (0.5): {default_f1:.4f} | "
              f"Optimal ({optimal_threshold:.2f}): {optimal_f1:.4f} | "
              f"Improvement: {improvement:+.2f}%")
    
    # Calculate overall improvement
    default_mean_f1 = np.mean(list(default_f1s.values()))
    optimized_mean_f1 = np.mean(list(optimized_f1s.values()))
    overall_improvement = ((optimized_mean_f1 - default_mean_f1) / default_mean_f1) * 100
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nDefault Mean F1 (threshold=0.5):     {default_mean_f1:.4f}")
    print(f"Optimized Mean F1 (optimal thresholds): {optimized_mean_f1:.4f}")
    print(f"Overall Improvement:                    {overall_improvement:+.2f}%")
    
    return {
        'optimal_thresholds': optimal_thresholds,
        'default_f1': default_f1s,
        'optimized_f1': optimized_f1s,
        'default_mean_f1': default_mean_f1,
        'optimized_mean_f1': optimized_mean_f1,
        'improvement_percent': overall_improvement
    }


def evaluate_with_thresholds(predictions, targets, thresholds):
    """
    Evaluate predictions using custom thresholds.
    
    Args:
        predictions: Raw probabilities (num_samples, num_classes)
        targets: Ground truth labels (num_samples, num_classes)
        thresholds: Dictionary of thresholds per class
        
    Returns:
        Dictionary of metrics
    """
    pred_binary = np.zeros_like(predictions)
    
    for i, disease in enumerate(LABEL_COLUMNS):
        threshold = thresholds.get(disease, 0.5)
        pred_binary[:, i] = (predictions[:, i] >= threshold).astype(int)
    
    # Calculate metrics
    f1_scores = []
    for i, disease in enumerate(LABEL_COLUMNS):
        f1 = f1_score(targets[:, i], pred_binary[:, i], zero_division=0)
        f1_scores.append(f1)
    
    mean_f1 = np.mean(f1_scores)
    label_accuracy = (pred_binary == targets).mean()
    
    return {
        'mean_f1': mean_f1,
        'label_accuracy': label_accuracy,
        'f1_per_class': {LABEL_COLUMNS[i]: f1_scores[i] for i in range(len(LABEL_COLUMNS))}
    }


def main():
    """Main threshold optimization pipeline."""
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZATION FOR ENSEMBLE MODEL")
    print("="*70 + "\n")
    
    # Configuration
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Model configurations
    model_configs = [
        {
            'type': 'resnet50',
            'path': 'models/baseline_model.pth'
        },
        {
            'type': 'efficientnet_b3',
            'path': 'models/efficientnet_b3_model.pth'
        },
        {
            'type': 'densenet121',
            'path': 'models/densenet121_model.pth'
        }
    ]
    
    # Load validation data
    print("Loading validation data...")
    val_images = np.load('preprocessed_data/val_images.npy')
    val_labels = np.load('preprocessed_data/val_labels.npy')
    print(f"  Images: {val_images.shape}")
    print(f"  Labels: {val_labels.shape}")
    
    # Create DataLoader
    val_images_tensor = torch.FloatTensor(val_images).permute(0, 3, 1, 2)
    val_dataset = TensorDataset(val_images_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Optimize thresholds
    results = optimize_thresholds(model_configs, val_loader, val_labels, device)
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'optimal_thresholds.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Create visualization-ready summary
    print("\n" + "="*70)
    print("OPTIMAL THRESHOLDS")
    print("="*70 + "\n")
    
    print("optimal_thresholds = {")
    for disease in LABEL_COLUMNS:
        threshold = results['optimal_thresholds'][disease]
        print(f"    '{disease}': {threshold:.3f},  # {LABEL_NAMES[disease]}")
    print("}")
    
    print("\n" + "="*70)
    print("✓ THRESHOLD OPTIMIZATION COMPLETE!")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    results = main()
