"""
Evaluate ensemble model combining ResNet50, EfficientNet-B3, and DenseNet-121.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json
from pathlib import Path
from tqdm import tqdm

# Import model architectures
from src.train import MultiLabelClassifier
from src.ensemble_models import EfficientNetB3Classifier, DenseNet121Classifier

# Disease labels (7 classes)
LABEL_COLUMNS = ['N', 'D', 'G', 'C', 'A', 'M', 'O']


class EnsemblePredictor:
    """Ensemble predictor combining multiple models."""
    
    def __init__(self, model_configs, device='mps', weights=None):
        """
        Args:
            model_configs: List of dicts with 'path', 'type', and 'checkpoint_key'
            device: Device to run inference on
            weights: Optional weights for each model (default: equal weights)
        """
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        self.models = []
        self.weights = weights if weights is not None else [1.0 / len(model_configs)] * len(model_configs)
        
        # Load each model
        for config in model_configs:
            model = self._load_model(config)
            self.models.append(model)
    
    def _load_model(self, config):
        """Load a single model from checkpoint."""
        model_type = config['type']
        checkpoint_path = config['path']
        
        print(f"Loading {model_type}...")
        
        # Create model architecture
        if model_type == 'resnet50':
            model = MultiLabelClassifier(num_classes=7)
        elif model_type == 'efficientnet_b3':
            model = EfficientNetB3Classifier(num_classes=7)
        elif model_type == 'densenet121':
            model = DenseNet121Classifier(num_classes=7)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint (weights_only=False for PyTorch 2.6+)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Print model info
        val_f1 = checkpoint.get('val_mean_f1', checkpoint.get('val_f1', 'N/A'))
        val_acc = checkpoint.get('val_acc', 'N/A')
        epoch = checkpoint.get('epoch', 'N/A')
        
        print(f"  ✓ {model_type}: Epoch {epoch}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
        
        return model
    
    def predict(self, images_tensor):
        """
        Generate ensemble predictions.
        
        Args:
            images_tensor: Torch tensor of images (batch_size, 3, 224, 224)
            
        Returns:
            Weighted average of probabilities from all models
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(images_tensor)
                probs = torch.sigmoid(logits)
                predictions.append(probs.cpu().numpy())
        
        # Weighted average
        predictions = np.array(predictions)  # (num_models, batch_size, num_classes)
        weights = np.array(self.weights).reshape(-1, 1, 1)
        ensemble_probs = np.sum(predictions * weights, axis=0)
        
        return ensemble_probs
    
    def predict_batch(self, dataloader, desc="Predicting"):
        """
        Generate predictions for entire dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            desc: Description for progress bar
            
        Returns:
            numpy array of predictions (num_samples, num_classes)
        """
        all_predictions = []
        
        for images in tqdm(dataloader, desc=desc):
            images = images[0] if isinstance(images, (list, tuple)) else images
            images = images.to(self.device)
            
            preds = self.predict(images)
            all_predictions.append(preds)
        
        return np.vstack(all_predictions)


def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculate comprehensive metrics for multi-label classification.
    
    Args:
        predictions: Model predictions (num_samples, num_classes)
        targets: Ground truth labels (num_samples, num_classes)
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Binarize predictions
    pred_binary = (predictions >= threshold).astype(int)
    
    # Overall metrics
    label_acc = accuracy_score(targets.flatten(), pred_binary.flatten())
    
    # Per-class metrics
    per_class_metrics = {}
    f1_scores = []
    
    for i, label in enumerate(LABEL_COLUMNS):
        f1 = f1_score(targets[:, i], pred_binary[:, i], zero_division=0)
        precision = precision_score(targets[:, i], pred_binary[:, i], zero_division=0)
        recall = recall_score(targets[:, i], pred_binary[:, i], zero_division=0)
        
        per_class_metrics[label] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'support': int(targets[:, i].sum())
        }
        f1_scores.append(f1)
    
    mean_f1 = np.mean(f1_scores)
    
    return {
        'label_accuracy': label_acc,
        'mean_f1': mean_f1,
        'f1_per_class': f1_scores,
        'per_class': per_class_metrics
    }


def evaluate_individual_models(model_configs, val_loader, val_labels, device='mps'):
    """Evaluate each model individually."""
    print("\n" + "="*70)
    print("INDIVIDUAL MODEL EVALUATION")
    print("="*70 + "\n")
    
    results = {}
    
    for config in model_configs:
        print(f"\nEvaluating {config['type']}...")
        print("-" * 50)
        
        # Create single-model ensemble
        predictor = EnsemblePredictor([config], device=device)
        
        # Generate predictions
        predictions = predictor.predict_batch(val_loader, desc=f"  {config['type']}")
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, val_labels)
        
        # Store results
        results[config['type']] = {
            'predictions': predictions,
            'metrics': metrics
        }
        
        # Print metrics
        print(f"\n  Mean F1: {metrics['mean_f1']:.4f}")
        print(f"  Label Accuracy: {metrics['label_accuracy']:.4f}")
        print(f"  Per-class F1:")
        for i, label in enumerate(LABEL_COLUMNS):
            f1 = metrics['f1_per_class'][i]
            print(f"    {label}: {f1:.4f}")
    
    return results


def evaluate_ensemble(model_configs, val_loader, val_labels, device='mps', weights=None):
    """Evaluate ensemble model."""
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION")
    print("="*70 + "\n")
    
    if weights is not None:
        print(f"Ensemble weights: {weights}")
    else:
        print("Using equal weights for all models")
    
    # Create ensemble
    ensemble = EnsemblePredictor(model_configs, device=device, weights=weights)
    
    # Generate predictions
    print("\nGenerating ensemble predictions...")
    predictions = ensemble.predict_batch(val_loader, desc="  Ensemble")
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, val_labels)
    
    # Print results
    print("\n" + "="*70)
    print("ENSEMBLE RESULTS")
    print("="*70)
    print(f"\n  Mean F1: {metrics['mean_f1']:.4f}")
    print(f"  Label Accuracy: {metrics['label_accuracy']:.4f}")
    
    print(f"\n  Per-class F1:")
    for i, label in enumerate(LABEL_COLUMNS):
        f1 = metrics['f1_per_class'][i]
        pm = metrics['per_class'][label]
        print(f"    {label}: F1={f1:.4f}, Precision={pm['precision']:.4f}, "
              f"Recall={pm['recall']:.4f}, Support={pm['support']}")
    
    return {
        'predictions': predictions,
        'metrics': metrics
    }


def optimize_ensemble_weights(individual_results, val_labels):
    """
    Find optimal ensemble weights using validation performance.
    Simple grid search over weight combinations.
    """
    print("\n" + "="*70)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("="*70 + "\n")
    
    model_names = list(individual_results.keys())
    best_f1 = 0
    best_weights = None
    
    # Grid search over weight combinations
    print("Testing weight combinations...")
    
    # Test equal weights
    weights = [1.0/3, 1.0/3, 1.0/3]
    predictions = sum(w * individual_results[name]['predictions'] 
                     for w, name in zip(weights, model_names))
    metrics = calculate_metrics(predictions, val_labels)
    print(f"  Equal weights [0.33, 0.33, 0.33]: F1={metrics['mean_f1']:.4f}")
    best_f1 = metrics['mean_f1']
    best_weights = weights
    
    # Test performance-weighted (by F1)
    f1_scores = [individual_results[name]['metrics']['mean_f1'] for name in model_names]
    total_f1 = sum(f1_scores)
    weights = [f1 / total_f1 for f1 in f1_scores]
    predictions = sum(w * individual_results[name]['predictions'] 
                     for w, name in zip(weights, model_names))
    metrics = calculate_metrics(predictions, val_labels)
    print(f"  F1-weighted {[f'{w:.3f}' for w in weights]}: F1={metrics['mean_f1']:.4f}")
    if metrics['mean_f1'] > best_f1:
        best_f1 = metrics['mean_f1']
        best_weights = weights
    
    # Test boosting best model
    best_model_idx = np.argmax(f1_scores)
    for boost in [0.4, 0.5, 0.6]:
        weights = [boost if i == best_model_idx else (1-boost)/2 
                  for i in range(len(model_names))]
        predictions = sum(w * individual_results[name]['predictions'] 
                         for w, name in zip(weights, model_names))
        metrics = calculate_metrics(predictions, val_labels)
        print(f"  Boost best {[f'{w:.3f}' for w in weights]}: F1={metrics['mean_f1']:.4f}")
        if metrics['mean_f1'] > best_f1:
            best_f1 = metrics['mean_f1']
            best_weights = weights
    
    print(f"\n✓ Best weights: {[f'{w:.3f}' for w in best_weights]}")
    print(f"  Best F1: {best_f1:.4f}")
    
    return best_weights


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*70)
    print("ENSEMBLE MODEL EVALUATION - 7 CLASSES")
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
    
    # Evaluate individual models
    individual_results = evaluate_individual_models(model_configs, val_loader, val_labels, device)
    
    # Optimize ensemble weights
    optimal_weights = optimize_ensemble_weights(individual_results, val_labels)
    
    # Evaluate ensemble with optimal weights
    ensemble_results = evaluate_ensemble(model_configs, val_loader, val_labels, device, optimal_weights)
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70 + "\n")
    
    results_summary = {
        'individual_models': {
            name: {
                'mean_f1': res['metrics']['mean_f1'],
                'label_accuracy': res['metrics']['label_accuracy'],
                'per_class_f1': {LABEL_COLUMNS[i]: float(res['metrics']['f1_per_class'][i]) 
                                for i in range(len(LABEL_COLUMNS))}
            }
            for name, res in individual_results.items()
        },
        'ensemble': {
            'weights': {model_configs[i]['type']: float(optimal_weights[i]) 
                       for i in range(len(model_configs))},
            'mean_f1': float(ensemble_results['metrics']['mean_f1']),
            'label_accuracy': float(ensemble_results['metrics']['label_accuracy']),
            'per_class_f1': {LABEL_COLUMNS[i]: float(ensemble_results['metrics']['f1_per_class'][i]) 
                           for i in range(len(LABEL_COLUMNS))}
        }
    }
    
    # Save to file
    output_file = 'results/ensemble_evaluation_results.json'
    Path('results').mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")
    
    # Print final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70 + "\n")
    
    print(f"{'Model':<20} {'Mean F1':<12} {'Accuracy':<12} {'Improvement'}")
    print("-" * 70)
    
    for name, res in individual_results.items():
        f1 = res['metrics']['mean_f1']
        acc = res['metrics']['label_accuracy']
        print(f"{name:<20} {f1:.4f}       {acc:.4f}")
    
    ensemble_f1 = ensemble_results['metrics']['mean_f1']
    ensemble_acc = ensemble_results['metrics']['label_accuracy']
    best_individual_f1 = max(res['metrics']['mean_f1'] for res in individual_results.values())
    improvement = ((ensemble_f1 - best_individual_f1) / best_individual_f1) * 100
    
    print("-" * 70)
    print(f"{'Ensemble':<20} {ensemble_f1:.4f}       {ensemble_acc:.4f}       "
          f"+{improvement:.2f}%")
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
