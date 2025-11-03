"""
Optimize classification thresholds for EfficientNet-B3 model.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import LABEL_COLUMNS
from ensemble_models import EfficientNetB3Classifier
from augmented_dataset import AugmentedODIRDataset

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
                                     threshold_range=np.arange(0.1, 0.95, 0.01)):
    """Find optimal threshold for a single class."""
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in threshold_range:
        pred_binary = (predictions >= threshold).astype(int)
        f1 = f1_score(targets, pred_binary, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def generate_predictions(model, data_loader, device):
    """Generate predictions on validation set."""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating predictions"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_predictions.append(probs.cpu().numpy())
    
    return np.vstack(all_predictions)


def optimize_thresholds(model, val_loader, val_labels, device='mps'):
    """Find optimal thresholds for each disease class."""
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZATION - EFFICIENTNET-B3")
    print("="*70 + "\n")
    
    # Generate predictions (probabilities)
    print("Generating predictions on validation set...")
    predictions = generate_predictions(model, val_loader, device)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Labels shape: {val_labels.shape}\n")
    
    # Find optimal threshold for each class
    print("Optimizing thresholds for each class...")
    print("-" * 70)
    
    optimal_thresholds = {}
    baseline_f1_scores = []
    optimized_f1_scores = []
    
    for i, label in enumerate(LABEL_COLUMNS):
        class_predictions = predictions[:, i]
        class_labels = val_labels[:, i]
        
        # Find optimal threshold
        optimal_threshold, optimal_f1 = find_optimal_threshold_for_class(
            class_predictions, class_labels, i
        )
        
        # Calculate baseline F1 (with 0.5 threshold)
        baseline_pred = (class_predictions >= 0.5).astype(int)
        baseline_f1 = f1_score(class_labels, baseline_pred, zero_division=0)
        
        optimal_thresholds[label] = float(optimal_threshold)
        baseline_f1_scores.append(baseline_f1)
        optimized_f1_scores.append(optimal_f1)
        
        support = int(class_labels.sum())
        improvement = optimal_f1 - baseline_f1
        
        print(f"{LABEL_NAMES[label]:12} | "
              f"Threshold: {optimal_threshold:.3f} | "
              f"F1: {baseline_f1:.4f} → {optimal_f1:.4f} "
              f"({improvement:+.4f}) | "
              f"Support: {support}")
    
    baseline_mean_f1 = np.mean(baseline_f1_scores)
    optimized_mean_f1 = np.mean(optimized_f1_scores)
    improvement = optimized_mean_f1 - baseline_mean_f1
    
    print("-" * 70)
    print(f"\n{'Mean F1':12} | "
          f"{'':18} | "
          f"F1: {baseline_mean_f1:.4f} → {optimized_mean_f1:.4f} "
          f"({improvement:+.4f})")
    
    # Evaluate with optimal thresholds
    print("\n" + "="*70)
    print("EVALUATION WITH OPTIMAL THRESHOLDS")
    print("="*70 + "\n")
    
    pred_binary = np.zeros_like(predictions)
    for i, label in enumerate(LABEL_COLUMNS):
        threshold = optimal_thresholds[label]
        pred_binary[:, i] = (predictions[:, i] >= threshold).astype(int)
    
    # Calculate detailed metrics
    per_class_metrics = {}
    
    print(f"{'Disease':<12} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 70)
    
    for i, label in enumerate(LABEL_COLUMNS):
        precision = precision_score(val_labels[:, i], pred_binary[:, i], zero_division=0)
        recall = recall_score(val_labels[:, i], pred_binary[:, i], zero_division=0)
        f1 = f1_score(val_labels[:, i], pred_binary[:, i], zero_division=0)
        support = int(val_labels[:, i].sum())
        
        per_class_metrics[label] = {
            'threshold': optimal_thresholds[label],
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': support
        }
        
        print(f"{LABEL_NAMES[label]:<12} {optimal_thresholds[label]:>10.3f} "
              f"{precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")
    
    mean_f1 = np.mean([per_class_metrics[label]['f1'] for label in LABEL_COLUMNS])
    label_accuracy = (pred_binary == val_labels).mean()
    
    print("-" * 70)
    print(f"{'MEAN':<12} {'':>10} {'':>10} {'':>10} {mean_f1:>10.4f}")
    print(f"\nLabel Accuracy: {label_accuracy:.4f}")
    
    # Save results
    results = {
        'model': 'efficientnet_b3',
        'optimal_thresholds': optimal_thresholds,
        'baseline_mean_f1': float(baseline_mean_f1),
        'optimized_mean_f1': float(optimized_mean_f1),
        'improvement': float(improvement),
        'per_class_metrics': per_class_metrics,
        'label_accuracy': float(label_accuracy)
    }
    
    output_file = 'results/efficientnet_b3_optimal_thresholds.json'
    Path('results').mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    return results


def main():
    """Main threshold optimization pipeline."""
    
    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\n📂 Loading trained EfficientNet-B3 model...")
    model = EfficientNetB3Classifier(num_classes=len(LABEL_COLUMNS))
    
    checkpoint = torch.load('models/efficientnet_b3_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from models/efficientnet_b3_model.pth")
    print(f"✓ Best epoch: {checkpoint.get('epoch', 'unknown')}")
    best_f1 = checkpoint.get('mean_f1', None)
    if best_f1 is not None:
        print(f"✓ Best val F1: {best_f1:.4f}")
    else:
        print(f"✓ Best val F1: unknown")
    
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
    print(f"✓ Validation samples: {len(val_labels)}")
    
    # Optimize thresholds
    results = optimize_thresholds(model, val_loader, val_labels, device)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"\nBaseline (0.5 threshold):     Mean F1 = {results['baseline_mean_f1']:.4f}")
    print(f"Optimized (per-class):        Mean F1 = {results['optimized_mean_f1']:.4f}")
    print(f"Improvement:                  +{results['improvement']:.4f} ({results['improvement']/results['baseline_mean_f1']*100:.1f}%)")
    print(f"\nLabel Accuracy: {results['label_accuracy']:.4f}")
    print("\nOptimal Thresholds:")
    for label in LABEL_COLUMNS:
        print(f"  {LABEL_NAMES[label]:12}: {results['optimal_thresholds'][label]:.3f}")


if __name__ == '__main__':
    main()
