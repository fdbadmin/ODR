"""
Evaluate ensemble with only TTA (no threshold optimization).
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


class EnsembleTTAPredictor:
    """Ensemble predictor with Test-Time Augmentation."""
    
    def __init__(self, model_configs, device='mps', num_augmentations=6):
        self.ensemble = EnsemblePredictor(model_configs, device=device)
        self.device = device
        self.num_augmentations = num_augmentations
        
        # Define augmentations
        self.augmentations = [
            ('original', lambda x: x),
            ('flip_h', lambda x: torch.flip(x, dims=[3])),
            ('flip_v', lambda x: torch.flip(x, dims=[2])),
            ('flip_hv', lambda x: torch.flip(torch.flip(x, dims=[2]), dims=[3])),
            ('rotate_90', lambda x: torch.rot90(x, k=1, dims=[2, 3])),
            ('rotate_270', lambda x: torch.rot90(x, k=3, dims=[2, 3]))
        ]
    
    def predict_with_tta(self, images_tensor):
        """Generate predictions with TTA."""
        all_predictions = []
        
        for aug_name, aug_fn in self.augmentations:
            aug_images = aug_fn(images_tensor)
            predictions = self.ensemble.predict(aug_images)
            all_predictions.append(predictions)
        
        # Average across all augmentations
        avg_predictions = np.mean(all_predictions, axis=0)
        return avg_predictions
    
    def predict_batch(self, dataloader, desc="Predicting"):
        """Generate predictions for entire dataset with TTA."""
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                images = batch[0].to(self.device)
                predictions = self.predict_with_tta(images)
                all_predictions.append(predictions)
        
        return np.vstack(all_predictions)


def main():
    """Evaluate ensemble with TTA only (threshold=0.5)."""
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION: TTA ONLY (threshold=0.5)")
    print("="*70 + "\n")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Model configurations
    model_configs = [
        {'type': 'resnet50', 'path': 'models/baseline_model.pth'},
        {'type': 'efficientnet_b3', 'path': 'models/efficientnet_b3_model.pth'},
        {'type': 'densenet121', 'path': 'models/densenet121_model.pth'}
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
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Create TTA ensemble
    print("\nLoading ensemble models with TTA...")
    tta_ensemble = EnsembleTTAPredictor(model_configs, device=device)
    
    # Generate predictions
    print("\nGenerating predictions with TTA (6 augmentations)...")
    print("  This may take a few minutes...")
    predictions = tta_ensemble.predict_batch(val_loader, desc="  TTA Prediction")
    
    print(f"\nPredictions shape: {predictions.shape}")
    
    # Apply default threshold of 0.5
    pred_binary = (predictions >= 0.5).astype(int)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
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
            'support': int(val_labels[:, i].sum())
        }
        f1_scores.append(f1)
    
    mean_f1 = np.mean(f1_scores)
    label_accuracy = (pred_binary == val_labels).mean()
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS: TTA ONLY (threshold=0.5)")
    print("="*70)
    print(f"\n  Mean F1: {mean_f1:.4f}")
    print(f"  Label Accuracy: {label_accuracy:.4f}")
    
    print(f"\n  Per-class F1:")
    for i, label in enumerate(LABEL_COLUMNS):
        metrics = per_class_metrics[label]
        print(f"    {LABEL_NAMES[label]:12}: F1={metrics['f1']:.4f}, "
              f"Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}")
    
    # Load baseline results for comparison
    with open('results/ensemble_evaluation_results.json', 'r') as f:
        baseline_results = json.load(f)
    
    baseline_f1 = baseline_results['ensemble']['mean_f1']
    improvement = ((mean_f1 - baseline_f1) / baseline_f1) * 100
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\n  Baseline Ensemble (no TTA, threshold=0.5):  {baseline_f1:.4f}")
    print(f"  TTA Ensemble (6 augmentations, threshold=0.5): {mean_f1:.4f}")
    print(f"  Improvement from TTA alone:                     {improvement:+.2f}%")
    
    # Save results
    results = {
        'mean_f1': float(mean_f1),
        'label_accuracy': float(label_accuracy),
        'per_class_f1': {label: float(per_class_metrics[label]['f1']) for label in LABEL_COLUMNS},
        'per_class_metrics': {label: {k: float(v) for k, v in metrics.items()} 
                             for label, metrics in per_class_metrics.items()},
        'baseline_f1': float(baseline_f1),
        'improvement_percent': float(improvement),
        'configuration': {
            'tta_augmentations': 6,
            'threshold': 0.5
        }
    }
    
    output_path = Path('results/tta_only_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("\n" + "="*70)
    print("✓ TTA EVALUATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
