"""
Create and evaluate 3-model ensemble (ResNet50, EfficientNet-B3, DenseNet-121)
Combines predictions using weighted averaging with optimized per-model thresholds.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from augmented_dataset import AugmentedODIRDataset
from train import MultiLabelClassifier  # ResNet50
from ensemble_models import EfficientNetB3Classifier, DenseNet121Classifier

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


def load_model_with_thresholds(model_class, model_path, threshold_path, device='mps', num_classes=7):
    """
    Load a trained model and its optimal thresholds.
    
    Args:
        model_class: Model class to instantiate
        model_path: Path to model checkpoint
        threshold_path: Path to optimal thresholds JSON
        device: Device to load model on
        num_classes: Number of output classes
    
    Returns:
        model: Loaded model in eval mode
        thresholds: Dictionary of optimal thresholds per class
    """
    # Load model
    model = model_class(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load thresholds
    with open(threshold_path, 'r') as f:
        threshold_data = json.load(f)
        thresholds = threshold_data['optimal_thresholds']
    
    return model, thresholds


def generate_ensemble_predictions(models, val_loader, device='mps'):
    """
    Generate predictions from all models in the ensemble.
    
    Args:
        models: List of (model, name) tuples
        val_loader: Validation data loader
        device: Device to run on
    
    Returns:
        predictions_dict: Dictionary mapping model name to predictions array
    """
    predictions_dict = {}
    
    for model, name in models:
        print(f"\nGenerating predictions for {name}...")
        model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc=f"{name}"):
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                all_predictions.append(probs.cpu().numpy())
        
        predictions_dict[name] = np.vstack(all_predictions)
        print(f"✓ {name} predictions: {predictions_dict[name].shape}")
    
    return predictions_dict


def evaluate_ensemble(predictions_dict, thresholds_dict, val_labels, weights=None):
    """
    Evaluate ensemble with weighted averaging and per-model thresholds.
    
    Args:
        predictions_dict: Dictionary of model predictions
        thresholds_dict: Dictionary of per-model thresholds
        val_labels: Ground truth labels
        weights: Optional weights for each model (default: equal)
    
    Returns:
        results: Dictionary with detailed metrics
    """
    model_names = list(predictions_dict.keys())
    
    # Default to equal weights
    if weights is None:
        weights = {name: 1.0 / len(model_names) for name in model_names}
    
    print("\n" + "="*80)
    print("ENSEMBLE EVALUATION")
    print("="*80)
    print(f"\nModel weights: {weights}")
    
    # Apply thresholds to each model's predictions first
    thresholded_predictions = {}
    for name in model_names:
        preds = predictions_dict[name]
        thresholds = thresholds_dict[name]
        
        # Apply per-class thresholds
        binary_preds = np.zeros_like(preds)
        for i, label in enumerate(DISEASE_LABELS):
            threshold = thresholds[label]
            binary_preds[:, i] = (preds[:, i] >= threshold).astype(float)
        
        thresholded_predictions[name] = binary_preds
    
    # Weighted voting: average the binary predictions
    ensemble_predictions = np.zeros_like(val_labels)
    for name, binary_preds in thresholded_predictions.items():
        ensemble_predictions += binary_preds * weights[name]
    
    # Final decision: majority vote (>0.5 after weighted averaging)
    final_predictions = (ensemble_predictions >= 0.5).astype(float)
    
    # Calculate metrics
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    # Per-class metrics
    print("\n" + "-"*80)
    print(f"{'Disease':<12s}  {'Precision':<12s}  {'Recall':<10s}  {'F1-Score':<10s}  {'Support':<8s}")
    print("-"*80)
    
    f1_scores = []
    for i, (label, name) in enumerate(zip(DISEASE_LABELS, DISEASE_NAMES.values())):
        pred = final_predictions[:, i]
        true = val_labels[:, i]
        
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        support = int(true.sum())
        
        f1_scores.append(f1)
        
        print(f"{name:<12s}  {precision:<12.4f}  {recall:<10.4f}  {f1:<10.4f}  {support:<8d}")
    
    mean_f1 = np.mean(f1_scores)
    
    print("-"*80)
    print(f"{'MEAN':<12s}  {'':<12s}  {'':<10s}  {mean_f1:<10.4f}")
    
    # Label accuracy
    label_accuracy = np.mean(np.all(final_predictions == val_labels, axis=1))
    print(f"\nLabel Accuracy (exact match): {label_accuracy:.4f}")
    
    # Sample accuracy
    sample_accuracy = np.mean(final_predictions == val_labels)
    print(f"Sample Accuracy (label-wise): {sample_accuracy:.4f}")
    
    # Results dictionary
    results = {
        'ensemble_weights': weights,
        'mean_f1': float(mean_f1),
        'label_accuracy': float(label_accuracy),
        'sample_accuracy': float(sample_accuracy),
        'per_class_f1': {label: float(f1) for label, f1 in zip(DISEASE_LABELS, f1_scores)},
        'final_predictions': final_predictions.tolist()
    }
    
    return results, final_predictions


def compare_with_individual_models(ensemble_f1, individual_results):
    """
    Compare ensemble performance with individual models.
    """
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"\n{'Model':<20s}  {'F1-Score':<10s}  {'Improvement vs Ensemble':<25s}")
    print("-"*80)
    
    for model_name, f1 in individual_results.items():
        diff = ensemble_f1 - f1
        print(f"{model_name:<20s}  {f1:<10.4f}  {diff:+.4f} ({diff/f1*100:+.1f}%)")
    
    print(f"{'ENSEMBLE':<20s}  {ensemble_f1:<10.4f}  {'---':<25s}")
    print("-"*80)
    
    best_individual = max(individual_results.values())
    improvement = ensemble_f1 - best_individual
    print(f"\nEnsemble improvement over best individual: {improvement:+.4f} ({improvement/best_individual*100:+.1f}%)")


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
    
    print("\n" + "="*80)
    print("3-MODEL ENSEMBLE CREATION")
    print("="*80)
    
    # Model configurations
    model_configs = [
        {
            'name': 'ResNet50',
            'class': MultiLabelClassifier,
            'model_path': 'models/baseline_model.pth',
            'threshold_path': 'results/single_model_optimal_thresholds.json'
        },
        {
            'name': 'EfficientNet-B3',
            'class': EfficientNetB3Classifier,
            'model_path': 'models/efficientnet_b3_model.pth',
            'threshold_path': 'results/efficientnet_b3_optimal_thresholds.json'
        },
        {
            'name': 'DenseNet-121',
            'class': DenseNet121Classifier,
            'model_path': 'models/densenet121_model.pth',
            'threshold_path': 'results/densenet121_optimal_thresholds.json'
        }
    ]
    
    # Load all models
    print("\n📦 Loading models...")
    models = []
    thresholds_dict = {}
    individual_f1_scores = {}
    
    for config in model_configs:
        print(f"\nLoading {config['name']}...")
        model, thresholds = load_model_with_thresholds(
            config['class'],
            config['model_path'],
            config['threshold_path'],
            device=device
        )
        models.append((model, config['name']))
        thresholds_dict[config['name']] = thresholds
        
        # Load individual F1 scores
        with open(config['threshold_path'], 'r') as f:
            data = json.load(f)
            individual_f1_scores[config['name']] = data['optimized_mean_f1']
        
        print(f"✓ {config['name']} loaded (F1: {individual_f1_scores[config['name']]:.4f})")
    
    # Load validation data
    print("\n📂 Loading validation data...")
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
    
    # Generate predictions from all models
    predictions_dict = generate_ensemble_predictions(models, val_loader, device)
    
    # Evaluate ensemble with equal weights
    print("\n" + "="*80)
    print("EVALUATING ENSEMBLE (EQUAL WEIGHTS)")
    print("="*80)
    
    results, final_predictions = evaluate_ensemble(
        predictions_dict,
        thresholds_dict,
        val_labels,
        weights=None  # Equal weights
    )
    
    # Compare with individual models
    compare_with_individual_models(results['mean_f1'], individual_f1_scores)
    
    # Save results
    output_path = 'results/ensemble_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Ensemble results saved to {output_path}")
    
    # Save ensemble predictions
    predictions_output = {
        'individual_predictions': {name: preds.tolist() for name, preds in predictions_dict.items()},
        'ensemble_predictions': final_predictions.tolist(),
        'ground_truth': val_labels.tolist()
    }
    
    predictions_path = 'results/ensemble_predictions.json'
    with open(predictions_path, 'w') as f:
        json.dump(predictions_output, f, indent=2)
    
    print(f"✅ Ensemble predictions saved to {predictions_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("ENSEMBLE CREATION COMPLETE!")
    print("="*80)
    print(f"\nFinal Ensemble Performance:")
    print(f"  Mean F1-Score: {results['mean_f1']:.4f} ({results['mean_f1']*100:.2f}%)")
    print(f"  Label Accuracy: {results['label_accuracy']:.4f} ({results['label_accuracy']*100:.2f}%)")
    print(f"  Sample Accuracy: {results['sample_accuracy']:.4f} ({results['sample_accuracy']*100:.2f}%)")
    
    print("\nPer-Class F1 Scores:")
    for label, name in DISEASE_NAMES.items():
        f1 = results['per_class_f1'][label]
        print(f"  {name:<12s}: {f1:.4f} ({f1*100:.2f}%)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
