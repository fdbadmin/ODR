"""
Ensemble model with Test Time Augmentation (TTA) and optimal thresholds.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
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


class EnsembleTTAPredictor:
    """Ensemble predictor with Test-Time Augmentation."""
    
    def __init__(self, model_configs, device='mps', num_augmentations=6):
        """
        Args:
            model_configs: List of model configurations
            device: Device for inference
            num_augmentations: Number of augmentations to apply
        """
        self.ensemble = EnsemblePredictor(model_configs, device=device)
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        self.num_augmentations = num_augmentations
        self.augmentations = self._get_augmentations()
    
    def _get_augmentations(self):
        """Define TTA augmentations for fundus images."""
        augmentations = [
            ('original', lambda x: x),
            ('flip_h', lambda x: torch.flip(x, dims=[3])),
            ('flip_v', lambda x: torch.flip(x, dims=[2])),
            ('flip_hv', lambda x: torch.flip(torch.flip(x, dims=[2]), dims=[3])),
            ('rotate_90', lambda x: torch.rot90(x, k=1, dims=[2, 3])),
            ('rotate_270', lambda x: torch.rot90(x, k=3, dims=[2, 3])),
        ]
        return augmentations[:self.num_augmentations]
    
    def predict_with_tta(self, images_tensor):
        """
        Generate predictions with TTA.
        
        Args:
            images_tensor: Torch tensor of images (batch_size, 3, 224, 224)
            
        Returns:
            Averaged predictions across all augmentations
        """
        all_predictions = []
        
        for aug_name, aug_fn in self.augmentations:
            # Apply augmentation
            aug_images = aug_fn(images_tensor)
            
            # Get ensemble predictions
            predictions = self.ensemble.predict(aug_images)
            all_predictions.append(predictions)
        
        # Average across augmentations
        avg_predictions = np.mean(all_predictions, axis=0)
        return avg_predictions
    
    def predict_batch(self, dataloader, desc="TTA Prediction"):
        """
        Generate TTA predictions for entire dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            desc: Description for progress bar
            
        Returns:
            numpy array of predictions (num_samples, num_classes)
        """
        all_predictions = []
        
        for batch in tqdm(dataloader, desc=desc):
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(self.device)
            
            # Predict with TTA
            batch_preds = self.predict_with_tta(images)
            all_predictions.append(batch_preds)
        
        return np.vstack(all_predictions)


def evaluate_ensemble_with_tta_and_thresholds(model_configs, val_loader, val_labels, 
                                               optimal_thresholds, device='mps'):
    """
    Evaluate ensemble with TTA and optimal thresholds.
    
    Args:
        model_configs: List of model configurations
        val_loader: DataLoader for validation set
        val_labels: Ground truth labels
        optimal_thresholds: Dictionary of optimal thresholds per class
        device: Device for inference
        
    Returns:
        Dictionary with results
    """
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION WITH TTA + OPTIMAL THRESHOLDS")
    print("="*70 + "\n")
    
    # Create TTA predictor
    print("Loading ensemble models with TTA...")
    tta_ensemble = EnsembleTTAPredictor(model_configs, device=device, num_augmentations=6)
    
    # Generate predictions with TTA
    print("\nGenerating predictions with TTA (6 augmentations)...")
    print("  This may take a few minutes...")
    predictions = tta_ensemble.predict_batch(val_loader, desc="  TTA Prediction")
    
    print(f"\nPredictions shape: {predictions.shape}")
    
    # Apply optimal thresholds
    pred_binary = np.zeros_like(predictions)
    for i, disease in enumerate(LABEL_COLUMNS):
        threshold = optimal_thresholds.get(disease, 0.5)
        pred_binary[:, i] = (predictions[:, i] >= threshold).astype(int)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    from sklearn.metrics import f1_score, precision_score, recall_score
    
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
            'threshold': optimal_thresholds.get(label, 0.5),
            'support': int(val_labels[:, i].sum())
        }
        f1_scores.append(f1)
    
    mean_f1 = np.mean(f1_scores)
    label_accuracy = (pred_binary == val_labels).mean()
    
    return {
        'predictions': pred_binary,
        'probabilities': predictions,
        'mean_f1': mean_f1,
        'label_accuracy': label_accuracy,
        'f1_per_class': f1_scores,
        'per_class_metrics': per_class_metrics
    }


def main():
    """Main evaluation pipeline with TTA and optimal thresholds."""
    print("\n" + "="*70)
    print("FINAL ENSEMBLE EVALUATION: TTA + OPTIMAL THRESHOLDS")
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
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Evaluate with TTA and optimal thresholds
    results = evaluate_ensemble_with_tta_and_thresholds(
        model_configs, val_loader, val_labels, optimal_thresholds, device
    )
    
    # Print results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\n  Mean F1: {results['mean_f1']:.4f}")
    print(f"  Label Accuracy: {results['label_accuracy']:.4f}")
    
    print(f"\n  Per-class F1 (with optimal thresholds):")
    for i, label in enumerate(LABEL_COLUMNS):
        metrics = results['per_class_metrics'][label]
        print(f"    {LABEL_NAMES[label]:12}: F1={metrics['f1']:.4f}, "
              f"Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, "
              f"Threshold={metrics['threshold']:.3f}")
    
    # Load baseline results for comparison
    with open('results/ensemble_evaluation_results.json', 'r') as f:
        baseline_results = json.load(f)
    
    baseline_f1 = baseline_results['ensemble']['mean_f1']
    improvement = ((results['mean_f1'] - baseline_f1) / baseline_f1) * 100
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\n  Baseline Ensemble (equal weights, threshold=0.5): {baseline_f1:.4f}")
    print(f"  Optimized Ensemble (TTA + optimal thresholds):     {results['mean_f1']:.4f}")
    print(f"  Improvement:                                        {improvement:+.2f}%")
    
    # Save results
    output_file = Path('results/final_ensemble_results.json')
    
    # Prepare results for JSON serialization
    save_results = {
        'mean_f1': float(results['mean_f1']),
        'label_accuracy': float(results['label_accuracy']),
        'per_class_f1': {LABEL_COLUMNS[i]: float(results['f1_per_class'][i]) 
                        for i in range(len(LABEL_COLUMNS))},
        'per_class_metrics': {
            label: {
                'f1': float(metrics['f1']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'threshold': float(metrics['threshold']),
                'support': metrics['support']
            }
            for label, metrics in results['per_class_metrics'].items()
        },
        'baseline_f1': baseline_f1,
        'improvement_percent': float(improvement),
        'configuration': {
            'tta_augmentations': 6,
            'optimal_thresholds': optimal_thresholds
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    print("\n" + "="*70)
    print("✓ FINAL ENSEMBLE EVALUATION COMPLETE!")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    results = main()
