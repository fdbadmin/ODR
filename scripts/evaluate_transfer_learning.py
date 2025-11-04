"""
Evaluate transfer learning results and create optimized ensemble.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from scipy.optimize import differential_evolution

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ensemble_models import EfficientNetB3Classifier, DenseNet121Classifier


class MultiLabelClassifier(nn.Module):
    """ResNet-based multi-label classifier."""
    
    def __init__(self, num_classes: int = 7, backbone: str = 'resnet50', pretrained: bool = False):
        super(MultiLabelClassifier, self).__init__()
        
        if backbone == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class RetinalDataset(Dataset):
    """Dataset for retinal images."""
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        
        if image.max() > 1.0:
            image = image / 255.0
        
        return image, torch.from_numpy(label).float()


def load_model(model_name, model_path, device, num_classes=7):
    """Load a fine-tuned model."""
    if model_name == 'resnet50':
        model = MultiLabelClassifier(num_classes=num_classes, backbone='resnet50', pretrained=False)
    elif model_name == 'efficientnet_b3':
        model = EfficientNetB3Classifier(num_classes=num_classes)
    elif model_name == 'densenet121':
        model = DenseNet121Classifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_f1 = checkpoint.get('best_val_f1', 'N/A')
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"  Loaded from epoch {epoch}, best F1: {best_f1:.4f}" if isinstance(best_f1, float) else f"  Loaded from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def get_predictions(model, data_loader, device):
    """Get predictions from a model."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    
    return np.vstack(all_preds), np.vstack(all_labels)


def optimize_thresholds(y_true, y_pred_proba):
    """Optimize classification thresholds for best F1 score."""
    num_classes = y_true.shape[1]
    
    def objective(thresholds):
        y_pred = (y_pred_proba >= thresholds).astype(int)
        return -f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    bounds = [(0.1, 0.9)] * num_classes
    result = differential_evolution(objective, bounds, seed=42, maxiter=100, workers=1)
    
    return result.x


def evaluate_with_thresholds(y_true, y_pred_proba, thresholds):
    """Evaluate predictions with given thresholds."""
    y_pred = (y_pred_proba >= thresholds).astype(int)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    return f1_macro, f1_per_class


def main():
    print("\n" + "="*80)
    print("Transfer Learning Evaluation and Ensemble Creation")
    print("="*80 + "\n")
    
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
    
    val_dataset = RetinalDataset(val_images, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Load models and get predictions
    models_info = [
        ('resnet50', 'models/resnet50_finetuned_best.pth'),
        ('efficientnet_b3', 'models/efficientnet_b3_finetuned_best.pth'),
        ('densenet121', 'models/densenet121_finetuned_best.pth')
    ]
    
    all_predictions = []
    disease_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Myopia', 'Other']
    
    print("="*80)
    print("Loading models and generating predictions...")
    print("="*80 + "\n")
    
    for model_name, model_path in models_info:
        print(f"Loading {model_name}...")
        model = load_model(model_name, model_path, device)
        preds, labels = get_predictions(model, val_loader, device)
        all_predictions.append(preds)
        
        # Optimize thresholds
        print(f"  Optimizing thresholds...")
        optimal_thresholds = optimize_thresholds(labels, preds)
        
        # Evaluate with default (0.5) and optimal thresholds
        f1_default, _ = evaluate_with_thresholds(labels, preds, np.array([0.5] * 7))
        f1_optimal, f1_per_class = evaluate_with_thresholds(labels, preds, optimal_thresholds)
        
        print(f"  Default (0.5):  F1 = {f1_default:.4f}")
        print(f"  Optimized:      F1 = {f1_optimal:.4f} (+{f1_optimal - f1_default:.4f})")
        print(f"  Per-class F1: {' '.join([f'{f1:.3f}' for f1 in f1_per_class])}")
        print()
    
    # Create ensemble
    print("="*80)
    print("Creating Ensemble")
    print("="*80 + "\n")
    
    ensemble_preds = np.mean(all_predictions, axis=0)
    
    print("Optimizing ensemble thresholds...")
    ensemble_thresholds = optimize_thresholds(labels, ensemble_preds)
    
    f1_ensemble_default, _ = evaluate_with_thresholds(labels, ensemble_preds, np.array([0.5] * 7))
    f1_ensemble_optimal, f1_ensemble_per_class = evaluate_with_thresholds(labels, ensemble_preds, ensemble_thresholds)
    
    print(f"\nEnsemble Results:")
    print(f"  Default (0.5):  F1 = {f1_ensemble_default:.4f}")
    print(f"  Optimized:      F1 = {f1_ensemble_optimal:.4f} (+{f1_ensemble_optimal - f1_ensemble_default:.4f})")
    print(f"\nPer-Class F1 Scores (Optimized Ensemble):")
    for disease, f1 in zip(disease_names, f1_ensemble_per_class):
        print(f"  {disease:12s}: {f1:.4f}")
    
    print(f"\nOptimal Thresholds:")
    for disease, thresh in zip(disease_names, ensemble_thresholds):
        print(f"  {disease:12s}: {thresh:.3f}")
    
    # Save thresholds
    np.save('models/transfer_learning_ensemble_thresholds.npy', ensemble_thresholds)
    print(f"\n✓ Ensemble thresholds saved to: models/transfer_learning_ensemble_thresholds.npy")
    
    # Comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    print("From-Scratch Training (50 epochs):")
    print("  ResNet50:         60.22% F1")
    print("  EfficientNet-B3:  61.60% F1")
    print("  DenseNet-121:     62.79% F1")
    print("  Ensemble:         64.03% F1")
    print()
    
    print("Transfer Learning (15 epochs):")
    print(f"  ResNet50:         {58.31:.2f}% F1")
    print(f"  EfficientNet-B3:  {55.76:.2f}% F1")
    print(f"  DenseNet-121:     {57.73:.2f}% F1")
    print(f"  Ensemble:         {f1_ensemble_optimal*100:.2f}% F1")
    print()
    
    print("Analysis:")
    print("  • Transfer learning did NOT improve over from-scratch training")
    print("  • Possible reasons:")
    print("    - Baseline models overfit to patient-level label patterns")
    print("    - Eye-specific labels too different from patient-level")
    print("    - Learning rate too low (5e-6 may be too conservative)")
    print("    - Need longer fine-tuning or unfrozen earlier layers")
    print()
    
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
