"""
Threshold Optimization for DenseNet-121 Model
Finds optimal classification thresholds for each disease class to maximize F1-score.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from augmented_dataset import AugmentedODIRDataset

# Disease labels
DISEASE_LABELS = ['N', 'D', 'G', 'C', 'A', 'M', 'O']
DISEASE_NAMES = {
    'N': 'Normal',
    'D': 'Diabetes',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'AMD',
    'M': 'Myopia',
    'O': 'Other'
}

class DenseNet121Classifier(nn.Module):
    """DenseNet-121 model for multi-label classification"""
    def __init__(self, num_classes=7):
        super(DenseNet121Classifier, self).__init__()
        from torchvision.models import densenet121, DenseNet121_Weights
        
        # Load pretrained DenseNet-121 - matches training script architecture
        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        
        # Replace classifier - matches training script
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


def find_optimal_threshold_for_class(predictions, targets, class_idx, 
                                     threshold_range=np.arange(0.1, 0.95, 0.01)):
    """
    Find optimal threshold for a single class that maximizes F1-score.
    
    Args:
        predictions: Predicted probabilities for the class (after sigmoid)
        targets: True binary labels for the class
        class_idx: Index of the class
        threshold_range: Range of thresholds to test
    
    Returns:
        best_threshold: Optimal threshold value
        best_f1: Best F1-score achieved
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


def optimize_thresholds(model, val_loader, val_labels, device='mps'):
    """
    Optimize classification thresholds for each disease class.
    
    Args:
        model: Trained DenseNet-121 model
        val_loader: Validation data loader
        val_labels: True validation labels
        device: Device to run on ('mps', 'cuda', or 'cpu')
    
    Returns:
        optimal_thresholds: Dictionary with optimal thresholds for each class
        results: Dictionary with detailed results
    """
    model.eval()
    all_predictions = []
    
    print("\nGenerating predictions on validation set...")
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Generating predictions"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_predictions.append(probs.cpu().numpy())
    
    # Concatenate all predictions
    predictions = np.vstack(all_predictions)
    print(f"Predictions shape: {predictions.shape}")
    
    # Find optimal thresholds for each class
    optimal_thresholds = {}
    baseline_f1_scores = {}
    optimized_f1_scores = {}
    
    print("\nOptimizing thresholds for each class:")
    print("-" * 80)
    
    for i, (label, name) in enumerate(zip(DISEASE_LABELS, DISEASE_NAMES.values())):
        class_predictions = predictions[:, i]
        class_targets = val_labels[:, i]
        
        # Calculate baseline F1 with 0.5 threshold
        baseline_pred = (class_predictions >= 0.5).astype(int)
        baseline_f1 = f1_score(class_targets, baseline_pred, zero_division=0)
        baseline_f1_scores[label] = baseline_f1
        
        # Find optimal threshold
        optimal_threshold, optimal_f1 = find_optimal_threshold_for_class(
            class_predictions, class_targets, i, threshold_range=np.arange(0.1, 0.95, 0.01)
        )
        
        optimal_thresholds[label] = float(optimal_threshold)
        optimized_f1_scores[label] = optimal_f1
        
        improvement = optimal_f1 - baseline_f1
        support = int(class_targets.sum())
        
        print(f"{name:12s} | Threshold: {optimal_threshold:.3f} | "
              f"F1: {baseline_f1:.4f} → {optimal_f1:.4f} ({improvement:+.4f}) | "
              f"Support: {support}")
    
    print("-" * 80)
    
    # Calculate mean F1 scores
    baseline_mean_f1 = np.mean(list(baseline_f1_scores.values()))
    optimized_mean_f1 = np.mean(list(optimized_f1_scores.values()))
    
    print(f"\nMean F1: {baseline_mean_f1:.4f} → {optimized_mean_f1:.4f} "
          f"({optimized_mean_f1 - baseline_mean_f1:+.4f})")
    
    # Evaluate with optimal thresholds
    print("\n" + "="*80)
    print("EVALUATION WITH OPTIMAL THRESHOLDS")
    print("="*80)
    
    predictions_with_optimal = np.zeros_like(predictions)
    for i, label in enumerate(DISEASE_LABELS):
        threshold = optimal_thresholds[label]
        predictions_with_optimal[:, i] = (predictions[:, i] >= threshold).astype(int)
    
    # Calculate detailed metrics for each class
    print(f"\n{'Disease':<12s}  {'Threshold':<10s}  {'Precision':<12s}  {'Recall':<10s}  "
          f"{'F1-Score':<10s}  {'Support':<8s}")
    print("-" * 80)
    
    for i, (label, name) in enumerate(zip(DISEASE_LABELS, DISEASE_NAMES.values())):
        threshold = optimal_thresholds[label]
        pred = predictions_with_optimal[:, i]
        true = val_labels[:, i]
        
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        support = int(true.sum())
        
        print(f"{name:<12s}  {threshold:<10.3f}  {precision:<12.4f}  {recall:<10.4f}  "
              f"{f1:<10.4f}  {support:<8d}")
    
    print("-" * 80)
    print(f"\n{'MEAN':<12s}  {'':<10s}  {'':<12s}  {'':<10s}  {optimized_mean_f1:<10.4f}")
    
    # Calculate label accuracy
    label_accuracy = np.mean(np.all(predictions_with_optimal == val_labels, axis=1))
    print(f"Label Accuracy: {label_accuracy:.4f}")
    
    # Prepare results dictionary
    results = {
        'optimal_thresholds': optimal_thresholds,
        'baseline_f1_scores': {k: float(v) for k, v in baseline_f1_scores.items()},
        'optimized_f1_scores': {k: float(v) for k, v in optimized_f1_scores.items()},
        'baseline_mean_f1': float(baseline_mean_f1),
        'optimized_mean_f1': float(optimized_mean_f1),
        'improvement': float(optimized_mean_f1 - baseline_mean_f1),
        'label_accuracy': float(label_accuracy)
    }
    
    return optimal_thresholds, results


def main():
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: mps")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using device: cuda")
    else:
        device = torch.device('cpu')
        print("Using device: cpu")
    
    # Load trained model
    print("\nLoading trained DenseNet-121 model...")
    model = DenseNet121Classifier(num_classes=7)
    
    checkpoint = torch.load('models/densenet121_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ Model loaded from models/densenet121_model.pth")
    print(f"✓ Best epoch: {checkpoint['epoch']}")
    
    # Load validation data
    print("\nLoading validation data...")
    val_dataset = AugmentedODIRDataset(
        'preprocessed_data/val_images.npy',
        'preprocessed_data/val_labels.npy',
        augment=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )
    
    val_labels = np.load('preprocessed_data/val_labels.npy')
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Optimize thresholds
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION - DENSENET-121")
    print("="*80)
    
    optimal_thresholds, results = optimize_thresholds(model, val_loader, val_labels, device)
    
    # Save results
    output_path = 'results/densenet121_optimal_thresholds.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"Baseline (0.5 threshold):     Mean F1 = {results['baseline_mean_f1']:.4f}")
    print(f"Optimized (per-class):        Mean F1 = {results['optimized_mean_f1']:.4f}")
    print(f"Improvement:                  {results['improvement']:+.4f} ({results['improvement']/results['baseline_mean_f1']*100:.1f}%)")
    print("="*80)


if __name__ == '__main__':
    main()
