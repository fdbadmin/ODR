"""
Optimize per-class thresholds for Phase 4 ensemble.
This is a quick win - no retraining or reprocessing needed!

Expected gain: +0.5-1.0% F1
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.advanced_models import get_model


class RetinalDataset(Dataset):
    """Simple dataset for evaluation."""
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        if image.max() > 1.0:
            image = image / 255.0
        
        return image, torch.from_numpy(label).float()


def load_model(model_name, checkpoint_path, device):
    """Load a trained model from checkpoint."""
    print(f"Loading {model_name}...")
    
    # Get model architecture
    model = get_model(model_name, num_classes=7, pretrained=False)
    model = model.to(device)
    
    # Load checkpoint (PyTorch 2.6 compatibility)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def get_predictions(models, dataloader, device):
    """Get predictions from all models."""
    all_predictions = [[] for _ in models]
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Getting predictions"):
            images = images.to(device)
            
            for i, model in enumerate(models):
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                all_predictions[i].append(probs.cpu().numpy())
            
            all_labels.append(labels.numpy())
    
    # Concatenate batches
    predictions_list = [np.vstack(preds) for preds in all_predictions]
    labels = np.vstack(all_labels)
    
    return predictions_list, labels


def ensemble_weighted(predictions_list, weights):
    """Weighted ensemble by validation F1."""
    weighted = sum(pred * weight for pred, weight in zip(predictions_list, weights))
    return weighted / sum(weights)


def optimize_threshold_per_class(probabilities, labels, class_idx, thresholds=None):
    """
    Find optimal threshold for a single class.
    
    Args:
        probabilities: Predicted probabilities (N, num_classes)
        labels: True labels (N, num_classes)
        class_idx: Which class to optimize
        thresholds: List of thresholds to try
    
    Returns:
        best_threshold, best_f1
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = (probabilities[:, class_idx] > threshold).astype(int)
        true = labels[:, class_idx].astype(int)
        
        # Calculate F1 for this class
        f1 = f1_score(true, preds, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def optimize_all_thresholds(probabilities, labels, class_names):
    """
    Optimize thresholds for all classes.
    
    Returns:
        optimal_thresholds: dict mapping class names to optimal thresholds
        per_class_f1: dict mapping class names to F1 scores
    """
    num_classes = len(class_names)
    optimal_thresholds = {}
    per_class_f1 = {}
    
    print("\n🔍 Optimizing thresholds per class...")
    for i, class_name in enumerate(class_names):
        threshold, f1 = optimize_threshold_per_class(probabilities, labels, i)
        optimal_thresholds[class_name] = float(threshold)
        per_class_f1[class_name] = float(f1)
        print(f"  {class_name}: threshold={threshold:.3f}, F1={f1:.4f}")
    
    return optimal_thresholds, per_class_f1


def evaluate_with_thresholds(probabilities, labels, thresholds, class_names):
    """
    Evaluate predictions using per-class thresholds.
    
    Args:
        probabilities: Predicted probabilities (N, num_classes)
        labels: True labels (N, num_classes)
        thresholds: dict mapping class names to thresholds
        class_names: list of class names
    
    Returns:
        f1_macro, f1_weighted, f1_per_class
    """
    predictions = np.zeros_like(probabilities)
    
    for i, class_name in enumerate(class_names):
        threshold = thresholds[class_name]
        predictions[:, i] = (probabilities[:, i] > threshold).astype(int)
    
    # Calculate metrics
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    
    return f1_macro, f1_weighted, f1_per_class


def visualize_threshold_comparison(baseline_f1, optimized_f1, class_names, save_path):
    """Create visualization comparing baseline vs optimized thresholds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Per-class comparison
    x = np.arange(len(class_names))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_f1['f1_per_class'], width, 
            label='Baseline (0.5)', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, optimized_f1['f1_per_class'], width,
            label='Optimized Thresholds', alpha=0.8, color='coral')
    
    ax1.set_xlabel('Disease Class', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Overall metrics comparison
    metrics = ['F1 Macro', 'F1 Weighted']
    baseline_vals = [baseline_f1['f1_macro'], baseline_f1['f1_weighted']]
    optimized_vals = [optimized_f1['f1_macro'], optimized_f1['f1_weighted']]
    
    x2 = np.arange(len(metrics))
    ax2.bar(x2 - width/2, baseline_vals, width,
            label='Baseline (0.5)', alpha=0.8, color='steelblue')
    ax2.bar(x2 + width/2, optimized_vals, width,
            label='Optimized Thresholds', alpha=0.8, color='coral')
    
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0.5, 0.7)
    
    # Add value labels
    for i, (b, o) in enumerate(zip(baseline_vals, optimized_vals)):
        ax2.text(i - width/2, b + 0.005, f'{b:.3f}', ha='center', va='bottom', fontsize=10)
        ax2.text(i + width/2, o + 0.005, f'{o:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {save_path}")
    plt.close()


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Class names
    class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Myopia', 'Other']
    
    # Model configurations
    model_configs = [
        {
            'name': 'convnext_tiny',
            'checkpoint': 'models/convnext_tiny_advanced_best.pth',
            'val_f1': 0.6030
        },
        {
            'name': 'vit_small',
            'checkpoint': 'models/vit_small_advanced_best.pth',
            'val_f1': 0.5499
        },
        {
            'name': 'efficientnetv2_s',
            'checkpoint': 'models/efficientnetv2_s_advanced_best.pth',
            'val_f1': 0.5497
        }
    ]
    
    # Load validation data
    print("\n📂 Loading validation data...")
    val_data = np.load('preprocessed_data/val_images.npy')
    val_labels = np.load('preprocessed_data/val_labels.npy')
    
    print(f"Validation set: {len(val_data)} images")
    print(f"Label distribution: {val_labels.sum(axis=0)}")
    
    # Create dataset and dataloader
    val_dataset = RetinalDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Load models
    print("\n🔄 Loading Phase 4 models...")
    models = []
    weights = []
    for config in model_configs:
        model = load_model(config['name'], config['checkpoint'], device)
        models.append(model)
        weights.append(config['val_f1'])
    
    # Get predictions from all models
    print("\n🔮 Getting predictions from ensemble...")
    predictions_list, labels = get_predictions(models, val_loader, device)
    
    # Create weighted ensemble
    ensemble_probs = ensemble_weighted(predictions_list, weights)
    
    # Baseline evaluation (threshold = 0.5)
    print("\n📊 Baseline Performance (threshold=0.5):")
    baseline_preds = (ensemble_probs > 0.5).astype(int)
    baseline_f1_macro = f1_score(labels, baseline_preds, average='macro', zero_division=0)
    baseline_f1_weighted = f1_score(labels, baseline_preds, average='weighted', zero_division=0)
    baseline_f1_per_class = f1_score(labels, baseline_preds, average=None, zero_division=0)
    
    print(f"  F1 Macro: {baseline_f1_macro:.4f}")
    print(f"  F1 Weighted: {baseline_f1_weighted:.4f}")
    print(f"  Per-class F1:")
    for name, f1 in zip(class_names, baseline_f1_per_class):
        print(f"    {name}: {f1:.4f}")
    
    # Optimize thresholds
    optimal_thresholds, optimized_per_class_f1 = optimize_all_thresholds(
        ensemble_probs, labels, class_names
    )
    
    # Evaluate with optimized thresholds
    print("\n📊 Optimized Performance (per-class thresholds):")
    optimized_f1_macro, optimized_f1_weighted, optimized_f1_per_class = evaluate_with_thresholds(
        ensemble_probs, labels, optimal_thresholds, class_names
    )
    
    print(f"  F1 Macro: {optimized_f1_macro:.4f}")
    print(f"  F1 Weighted: {optimized_f1_weighted:.4f}")
    print(f"  Per-class F1:")
    for name, f1 in zip(class_names, optimized_f1_per_class):
        print(f"    {name}: {f1:.4f}")
    
    # Calculate improvement
    improvement = (optimized_f1_macro - baseline_f1_macro) * 100
    print(f"\n✨ Improvement: +{improvement:.2f}% ({baseline_f1_macro:.4f} → {optimized_f1_macro:.4f})")
    
    # Save results
    results = {
        'baseline': {
            'f1_macro': float(baseline_f1_macro),
            'f1_weighted': float(baseline_f1_weighted),
            'f1_per_class': [float(f1) for f1 in baseline_f1_per_class],
            'threshold': 0.5
        },
        'optimized': {
            'f1_macro': float(optimized_f1_macro),
            'f1_weighted': float(optimized_f1_weighted),
            'f1_per_class': [float(f1) for f1 in optimized_f1_per_class],
            'thresholds': optimal_thresholds
        },
        'improvement': {
            'f1_macro': float(optimized_f1_macro - baseline_f1_macro),
            'f1_weighted': float(optimized_f1_weighted - baseline_f1_weighted),
            'percentage': float(improvement)
        },
        'class_names': class_names
    }
    
    results_path = 'results/phase4_threshold_optimization.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results: {results_path}")
    
    # Visualize comparison
    baseline_dict = {
        'f1_macro': baseline_f1_macro,
        'f1_weighted': baseline_f1_weighted,
        'f1_per_class': baseline_f1_per_class
    }
    optimized_dict = {
        'f1_macro': optimized_f1_macro,
        'f1_weighted': optimized_f1_weighted,
        'f1_per_class': optimized_f1_per_class
    }
    
    visualize_threshold_comparison(
        baseline_dict, optimized_dict, class_names,
        'results/phase4_threshold_comparison.png'
    )
    
    # Compare with Phase 2 baseline
    phase2_f1 = 0.6403  # From previous results
    gap_remaining = (phase2_f1 - optimized_f1_macro) * 100
    
    print(f"\n📈 Gap Analysis:")
    print(f"  Phase 2 Baseline: {phase2_f1:.4f}")
    print(f"  Phase 4 Before: {baseline_f1_macro:.4f} (gap: -{(phase2_f1 - baseline_f1_macro)*100:.2f}%)")
    print(f"  Phase 4 After: {optimized_f1_macro:.4f} (gap: -{gap_remaining:.2f}%)")
    print(f"  Gap Closed: {((phase2_f1 - baseline_f1_macro) - (phase2_f1 - optimized_f1_macro))*100:.2f}%")
    
    if gap_remaining < 0:
        print(f"\n🎉 SUCCESS! We exceeded Phase 2 by {-gap_remaining:.2f}%!")
    elif gap_remaining < 1.0:
        print(f"\n✅ Almost there! Only {gap_remaining:.2f}% gap remaining.")
    else:
        print(f"\n📝 Next step: Enable vessel enhancement to close remaining {gap_remaining:.2f}% gap")


if __name__ == '__main__':
    main()
