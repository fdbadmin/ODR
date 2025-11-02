"""
Comprehensive model evaluation script
Generates detailed metrics, per-class performance, and confusion matrices
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    average_precision_score,
    hamming_loss,
    accuracy_score
)
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train import ODIRDataset, MultiLabelClassifier

# Disease classes
DISEASE_CLASSES = [
    'Normal', 'Diabetes', 'Glaucoma', 'Cataract',
    'AMD', 'Hypertension', 'Myopia', 'Other'
]


def load_model(model_path: str, device: torch.device):
    """Load trained model"""
    print(f"📂 Loading model from {model_path}")
    
    model = MultiLabelClassifier(num_classes=8)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        train_loss = checkpoint.get('train_loss', None)
        val_loss = checkpoint.get('val_loss', None)
        if train_loss is not None:
            print(f"  Training loss: {train_loss:.4f}")
        if val_loss is not None:
            print(f"  Validation loss: {val_loss:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader, device):
    """Run inference and collect predictions"""
    print("\n🔍 Running inference...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Concatenate all batches
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    y_probs = np.vstack(all_probs)
    
    return y_true, y_pred, y_probs


def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive metrics"""
    print("\n📊 Calculating metrics...")
    
    metrics = {}
    
    # Overall metrics
    metrics['exact_match_accuracy'] = accuracy_score(y_true, y_pred)
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-sample accuracy (what % of labels correct per sample)
    per_sample_acc = (y_true == y_pred).mean(axis=1)
    metrics['mean_sample_accuracy'] = per_sample_acc.mean()
    metrics['median_sample_accuracy'] = np.median(per_sample_acc)
    
    # AUC and AP scores
    try:
        metrics['macro_auc'] = roc_auc_score(y_true, y_probs, average='macro')
        metrics['micro_auc'] = roc_auc_score(y_true, y_probs, average='micro')
        metrics['weighted_auc'] = roc_auc_score(y_true, y_probs, average='weighted')
    except ValueError as e:
        print(f"  Warning: Could not calculate AUC: {e}")
        metrics['macro_auc'] = None
        metrics['micro_auc'] = None
        metrics['weighted_auc'] = None
    
    try:
        metrics['macro_ap'] = average_precision_score(y_true, y_probs, average='macro')
        metrics['micro_ap'] = average_precision_score(y_true, y_probs, average='micro')
    except ValueError as e:
        print(f"  Warning: Could not calculate AP: {e}")
        metrics['macro_ap'] = None
        metrics['micro_ap'] = None
    
    return metrics


def per_class_analysis(y_true, y_pred, y_probs):
    """Detailed per-class performance"""
    print("\n📋 Per-class analysis...")
    
    per_class_metrics = {}
    
    for i, disease in enumerate(DISEASE_CLASSES):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        y_probs_class = y_probs[:, i]
        
        # Basic counts
        true_positives = ((y_true_class == 1) & (y_pred_class == 1)).sum()
        true_negatives = ((y_true_class == 0) & (y_pred_class == 0)).sum()
        false_positives = ((y_true_class == 0) & (y_pred_class == 1)).sum()
        false_negatives = ((y_true_class == 1) & (y_pred_class == 0)).sum()
        
        # Metrics
        total = len(y_true_class)
        support = y_true_class.sum()
        accuracy = (true_positives + true_negatives) / total
        
        # Precision, Recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        
        # AUC
        try:
            if len(np.unique(y_true_class)) > 1:
                auc = roc_auc_score(y_true_class, y_probs_class)
            else:
                auc = None
        except:
            auc = None
        
        per_class_metrics[disease] = {
            'support': int(support),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'auc': float(auc) if auc is not None else None,
            'true_positives': int(true_positives),
            'true_negatives': int(true_negatives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        }
    
    return per_class_metrics


def print_results(metrics, per_class_metrics):
    """Pretty print all results"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\n📈 Overall Metrics:")
    print(f"  Exact Match Accuracy:    {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"  Mean Sample Accuracy:    {metrics['mean_sample_accuracy']:.4f} ({metrics['mean_sample_accuracy']*100:.2f}%)")
    print(f"  Hamming Loss:            {metrics['hamming_loss']:.4f}")
    
    if metrics['macro_auc'] is not None:
        print(f"\n🎯 AUC Scores:")
        print(f"  Macro AUC:               {metrics['macro_auc']:.4f}")
        print(f"  Micro AUC:               {metrics['micro_auc']:.4f}")
        print(f"  Weighted AUC:            {metrics['weighted_auc']:.4f}")
    
    if metrics['macro_ap'] is not None:
        print(f"\n📊 Average Precision:")
        print(f"  Macro AP:                {metrics['macro_ap']:.4f}")
        print(f"  Micro AP:                {metrics['micro_ap']:.4f}")
    
    print("\n" + "="*70)
    print("PER-CLASS PERFORMANCE")
    print("="*70)
    
    # Header
    print(f"\n{'Disease':<15} {'Support':>8} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Spec':>7} {'AUC':>7}")
    print("-" * 70)
    
    # Sort by support (most common diseases first)
    sorted_classes = sorted(per_class_metrics.items(), key=lambda x: x[1]['support'], reverse=True)
    
    for disease, m in sorted_classes:
        auc_str = f"{m['auc']:.3f}" if m['auc'] is not None else "N/A"
        print(f"{disease:<15} {m['support']:>8} {m['accuracy']:>7.3f} {m['precision']:>7.3f} "
              f"{m['recall']:>7.3f} {m['f1_score']:>7.3f} {m['specificity']:>7.3f} {auc_str:>7}")
    
    print("\n" + "="*70)
    print("DETAILED CONFUSION MATRICES (per class)")
    print("="*70)
    
    for disease, m in sorted_classes:
        print(f"\n{disease} (Support: {m['support']}):")
        print(f"  TP: {m['true_positives']:>5}  FP: {m['false_positives']:>5}")
        print(f"  FN: {m['false_negatives']:>5}  TN: {m['true_negatives']:>5}")
        print(f"  Sensitivity: {m['recall']:.3f}  Specificity: {m['specificity']:.3f}")


def main():
    """Main evaluation pipeline"""
    print("="*70)
    print("ODIR-5K MODEL EVALUATION")
    print("="*70)
    
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Paths
    model_path = 'models/best_model.pth'
    val_images_path = 'preprocessed_data_enhanced/val_images.npy'
    val_labels_path = 'preprocessed_data_enhanced/val_labels.npy'
    
    # Load data
    print(f"\n📂 Loading validation data...")
    val_dataset = ODIRDataset(val_images_path, val_labels_path)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"✓ Loaded {len(val_dataset)} validation samples")
    
    # Load model
    model = load_model(model_path, device)
    
    # Evaluate
    y_true, y_pred, y_probs = evaluate_model(model, val_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_probs)
    per_class_metrics = per_class_analysis(y_true, y_pred, y_probs)
    
    # Print results
    print_results(metrics, per_class_metrics)
    
    # Save results
    results = {
        'overall_metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'disease_classes': DISEASE_CLASSES,
        'num_samples': len(y_true)
    }
    
    output_path = 'models/evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {output_path}")
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
