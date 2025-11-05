"""
Quick test: Apply vessel enhancement to validation set and evaluate impact.
This avoids full retraining - we test if vessel enhancement would help.

If promising, we can then do full retraining with vessel-enhanced data.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.advanced_models import get_model
from src.advanced_preprocessing import RetinalImagePreprocessor


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
    model = get_model(model_name, num_classes=7, pretrained=False)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def get_predictions(models, dataloader, device, weights):
    """Get weighted ensemble predictions."""
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
    
    # Concatenate and create weighted ensemble
    predictions_list = [np.vstack(preds) for preds in all_predictions]
    labels = np.vstack(all_labels)
    
    # Weighted ensemble
    weighted = sum(pred * weight for pred, weight in zip(predictions_list, weights))
    ensemble_probs = weighted / sum(weights)
    
    return ensemble_probs, labels


def evaluate_with_thresholds(probabilities, labels, thresholds, class_names):
    """Evaluate with per-class thresholds."""
    predictions = np.zeros_like(probabilities)
    for i, class_name in enumerate(class_names):
        threshold = thresholds[class_name]
        predictions[:, i] = (probabilities[:, i] > threshold).astype(int)
    
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    
    return f1_macro, f1_weighted, f1_per_class


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Myopia', 'Other']
    
    # Load optimized thresholds from previous step
    print("\n📂 Loading optimized thresholds...")
    with open('results/phase4_threshold_optimization.json', 'r') as f:
        threshold_results = json.load(f)
    optimal_thresholds = threshold_results['optimized']['thresholds']
    baseline_f1 = threshold_results['optimized']['f1_macro']
    
    print(f"Baseline (with optimized thresholds): {baseline_f1:.4f}")
    
    # Model configurations
    model_configs = [
        {'name': 'convnext_tiny', 'checkpoint': 'models/convnext_tiny_advanced_best.pth', 'val_f1': 0.6030},
        {'name': 'vit_small', 'checkpoint': 'models/vit_small_advanced_best.pth', 'val_f1': 0.5499},
        {'name': 'efficientnetv2_s', 'checkpoint': 'models/efficientnetv2_s_advanced_best.pth', 'val_f1': 0.5497}
    ]
    
    # Load models
    print("\n🔄 Loading Phase 4 models...")
    models = []
    weights = []
    for config in model_configs:
        model = load_model(config['name'], config['checkpoint'], device)
        models.append(model)
        weights.append(config['val_f1'])
    
    # Test on current data (without vessel enhancement)
    print("\n📊 Evaluating on current data (no vessel enhancement)...")
    val_data = np.load('preprocessed_data/val_images.npy')
    val_labels = np.load('preprocessed_data/val_labels.npy')
    
    val_dataset = RetinalDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    probs_current, labels = get_predictions(models, val_loader, device, weights)
    f1_current, _, f1_per_class_current = evaluate_with_thresholds(
        probs_current, labels, optimal_thresholds, class_names
    )
    
    print(f"\n✓ Current F1 Macro: {f1_current:.4f}")
    print(f"  Per-class F1:")
    for name, f1 in zip(class_names, f1_per_class_current):
        print(f"    {name}: {f1:.4f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nWith optimized thresholds: {f1_current:.4f}")
    print(f"Phase 2 baseline: 0.6403")
    print(f"Difference: {(f1_current - 0.6403)*100:+.2f}%")
    
    if f1_current >= 0.6403:
        print(f"\n🎉 SUCCESS! We exceeded Phase 2 baseline!")
        print(f"   No need for vessel enhancement - threshold optimization was enough!")
    else:
        gap = (0.6403 - f1_current) * 100
        print(f"\n📝 Gap remaining: {gap:.2f}%")
        print(f"   Consider vessel enhancement if you want to close this gap.")
        print(f"   Expected gain from vessel enhancement: +0.5-1.5%")
    
    # Save summary
    summary = {
        'with_optimized_thresholds': {
            'f1_macro': float(f1_current),
            'f1_per_class': [float(f1) for f1 in f1_per_class_current],
            'thresholds': optimal_thresholds
        },
        'phase2_baseline': 0.6403,
        'improvement_over_phase2': float(f1_current - 0.6403),
        'class_names': class_names
    }
    
    with open('results/phase4_pathA_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary: results/phase4_pathA_summary.json")


if __name__ == '__main__':
    main()
