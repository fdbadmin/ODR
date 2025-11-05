"""
Evaluate Phase 4 ensemble (ConvNeXt Tiny + ViT Small + EfficientNetV2 Small)
Tests different ensemble strategies and compares with Phase 2 baseline.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model
    model = get_model(model_name, num_classes=7, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def get_predictions(model, dataloader, device):
    """Get predictions from a single model."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Predicting', leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    
    return np.vstack(all_preds), np.vstack(all_labels)


def ensemble_average(predictions_list):
    """Simple averaging ensemble."""
    return np.mean(predictions_list, axis=0)


def ensemble_weighted(predictions_list, weights):
    """Weighted averaging ensemble."""
    weighted = np.zeros_like(predictions_list[0])
    for pred, weight in zip(predictions_list, weights):
        weighted += pred * weight
    return weighted / sum(weights)


def ensemble_voting(predictions_list, threshold=0.5):
    """Majority voting ensemble."""
    binary_preds = [(pred > threshold).astype(int) for pred in predictions_list]
    votes = np.stack(binary_preds, axis=0)
    return (votes.sum(axis=0) >= len(predictions_list) / 2).astype(int)


def evaluate_predictions(y_true, y_pred_probs, threshold=0.5, ensemble_name="Ensemble"):
    """Evaluate predictions and return metrics."""
    y_pred = (y_pred_probs > threshold).astype(int)
    
    # Overall F1
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    disease_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Myopia', 'Other']
    
    print(f"\n{'='*70}")
    print(f"{ensemble_name} Results")
    print(f"{'='*70}")
    print(f"Macro F1:    {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"Weighted F1: {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    print(f"\nPer-Class F1:")
    for name, f1 in zip(disease_names, f1_per_class):
        print(f"  {name:12s}: {f1:.4f} ({f1*100:.2f}%)")
    
    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'predictions': y_pred,
        'probabilities': y_pred_probs
    }


def plot_comparison(results_dict, save_path='results/phase4_ensemble_comparison.png'):
    """Plot comparison of different ensemble strategies."""
    Path(save_path).parent.mkdir(exist_ok=True)
    
    disease_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Myopia', 'Other']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Overall F1 comparison
    ax = axes[0, 0]
    models = list(results_dict.keys())
    f1_scores = [results_dict[m]['f1_macro'] for m in models]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.bar(range(len(models)), f1_scores, color=colors[:len(models)])
    ax.set_xlabel('Model/Ensemble', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.4f}\n({f1*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add Phase 2 baseline line
    phase2_baseline = 0.6403
    ax.axhline(y=phase2_baseline, color='red', linestyle='--', linewidth=2, 
               label=f'Phase 2 Baseline ({phase2_baseline:.4f})')
    ax.legend(loc='upper right', fontsize=10)
    
    # Plot 2: Per-class F1 comparison
    ax = axes[0, 1]
    x = np.arange(len(disease_names))
    width = 0.15
    
    for i, (model_name, color) in enumerate(zip(models[:4], colors[:4])):  # Show top 4
        f1_per_class = results_dict[model_name]['f1_per_class']
        offset = (i - 1.5) * width
        ax.bar(x + offset, f1_per_class, width, label=model_name, color=color)
    
    ax.set_xlabel('Disease', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(disease_names, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Plot 3: Best ensemble per-class heatmap
    ax = axes[1, 0]
    best_ensemble_name = max(results_dict.keys(), key=lambda k: results_dict[k]['f1_macro'])
    f1_matrix = results_dict[best_ensemble_name]['f1_per_class'].reshape(1, -1)
    
    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=disease_names, yticklabels=[best_ensemble_name],
                vmin=0, vmax=1, cbar_kws={'label': 'F1 Score'}, ax=ax)
    ax.set_title(f'Best Ensemble: {best_ensemble_name}', fontsize=14, fontweight='bold')
    
    # Plot 4: Improvement over Phase 2
    ax = axes[1, 1]
    improvements = [(results_dict[m]['f1_macro'] - phase2_baseline) * 100 for m in models]
    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.barh(range(len(models)), improvements, color=colors_imp, alpha=0.7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Improvement over Phase 2 Baseline (percentage points)', 
                   fontsize=12, fontweight='bold')
    ax.set_title('Phase 4 vs Phase 2 Baseline', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        ax.text(width + (0.5 if width > 0 else -0.5), bar.get_y() + bar.get_height()/2.,
                f'{imp:+.2f}%',
                ha='left' if width > 0 else 'right', va='center', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {save_path}")
    plt.close()


def main():
    print(f"\n{'='*70}")
    print("PHASE 4 ENSEMBLE EVALUATION")
    print(f"{'='*70}\n")
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device\n")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device\n")
    else:
        device = torch.device("cpu")
        print("Using CPU device\n")
    
    # Load validation data
    print("Loading validation data...")
    data_dir = Path('preprocessed_data')
    val_images = np.load(data_dir / 'val_images.npy')
    val_labels = np.load(data_dir / 'val_labels.npy')
    print(f"Validation set: {val_images.shape[0]} images\n")
    
    # Create dataloader
    val_dataset = RetinalDataset(val_images, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Load models
    models_config = {
        'convnext_tiny': 'models/convnext_tiny_advanced_best.pth',
        'vit_small': 'models/vit_small_advanced_best.pth',
        'efficientnetv2_s': 'models/efficientnetv2_s_advanced_best.pth'
    }
    
    models = {}
    model_f1s = {}
    for model_name, checkpoint_path in models_config.items():
        model, checkpoint = load_model(model_name, checkpoint_path, device)
        models[model_name] = model
        model_f1s[model_name] = checkpoint['best_val_f1']
    
    print(f"\n{'='*70}")
    print("Getting predictions from individual models...")
    print(f"{'='*70}\n")
    
    # Get predictions from each model
    all_predictions = {}
    all_labels = None
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        preds, labels = get_predictions(model, val_loader, device)
        all_predictions[model_name] = preds
        if all_labels is None:
            all_labels = labels
    
    # Evaluate individual models
    results = {}
    
    print(f"\n{'='*70}")
    print("INDIVIDUAL MODEL RESULTS")
    print(f"{'='*70}")
    
    results['ConvNeXt Tiny'] = evaluate_predictions(
        all_labels, all_predictions['convnext_tiny'], 
        ensemble_name="ConvNeXt Tiny (Single Model)"
    )
    
    results['ViT Small'] = evaluate_predictions(
        all_labels, all_predictions['vit_small'],
        ensemble_name="ViT Small (Single Model)"
    )
    
    results['EfficientNetV2 S'] = evaluate_predictions(
        all_labels, all_predictions['efficientnetv2_s'],
        ensemble_name="EfficientNetV2 Small (Single Model)"
    )
    
    # Ensemble strategies
    print(f"\n{'='*70}")
    print("ENSEMBLE STRATEGIES")
    print(f"{'='*70}")
    
    predictions_list = [
        all_predictions['convnext_tiny'],
        all_predictions['vit_small'],
        all_predictions['efficientnetv2_s']
    ]
    
    # 1. Simple Average
    avg_preds = ensemble_average(predictions_list)
    results['Average Ensemble'] = evaluate_predictions(
        all_labels, avg_preds,
        ensemble_name="Average Ensemble (Equal Weights)"
    )
    
    # 2. Weighted by validation F1
    weights = [model_f1s['convnext_tiny'], model_f1s['vit_small'], model_f1s['efficientnetv2_s']]
    weighted_preds = ensemble_weighted(predictions_list, weights)
    results['Weighted Ensemble'] = evaluate_predictions(
        all_labels, weighted_preds,
        ensemble_name="Weighted Ensemble (by Val F1)"
    )
    
    # 3. Majority Voting
    voting_preds = ensemble_voting(predictions_list, threshold=0.5)
    # Convert back to probabilities for consistent evaluation
    voting_preds_probs = voting_preds.astype(float)
    results['Voting Ensemble'] = evaluate_predictions(
        all_labels, voting_preds_probs, threshold=0.5,
        ensemble_name="Voting Ensemble (Majority)"
    )
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    
    phase2_baseline = 0.6403
    
    print(f"\n{'Model/Ensemble':<30} {'F1 Score':<12} {'vs Phase 2':<15}")
    print("-" * 70)
    
    for name, result in results.items():
        f1 = result['f1_macro']
        diff = (f1 - phase2_baseline) * 100
        diff_str = f"{diff:+.2f}%"
        print(f"{name:<30} {f1:.4f} ({f1*100:.2f}%)  {diff_str:<15}")
    
    print("-" * 70)
    print(f"{'Phase 2 Baseline':<30} {phase2_baseline:.4f} ({phase2_baseline*100:.2f}%)  {'(reference)':<15}")
    
    # Find best ensemble
    best_name = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    best_f1 = results[best_name]['f1_macro']
    
    print(f"\n{'='*70}")
    print(f"BEST PERFORMER: {best_name}")
    print(f"F1 Score: {best_f1:.4f} ({best_f1*100:.2f}%)")
    
    if best_f1 > phase2_baseline:
        improvement = (best_f1 - phase2_baseline) * 100
        print(f"✅ BEATS Phase 2 baseline by {improvement:.2f} percentage points!")
    else:
        gap = (phase2_baseline - best_f1) * 100
        print(f"⚠️  Below Phase 2 baseline by {gap:.2f} percentage points")
    
    print(f"{'='*70}\n")
    
    # Generate comparison plot
    plot_comparison(results)
    
    # Save detailed results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    import json
    results_json = {}
    for name, result in results.items():
        results_json[name] = {
            'f1_macro': float(result['f1_macro']),
            'f1_weighted': float(result['f1_weighted']),
            'f1_per_class': result['f1_per_class'].tolist()
        }
    
    with open(results_dir / 'phase4_ensemble_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Saved detailed results to: results/phase4_ensemble_results.json")
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
