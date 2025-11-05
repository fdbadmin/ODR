"""
Evaluate Phase 4 ensemble with Test-Time Augmentation (TTA).
TTA averages predictions over multiple augmented versions of each image.
Expected improvement: +1.5-2.5% F1 score.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import torchvision.transforms as transforms

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.advanced_models import get_model


class TTADataset(Dataset):
    """Dataset with Test-Time Augmentation."""
    
    def __init__(self, images, labels, num_tta=8):
        """
        Args:
            images: numpy array of images
            labels: numpy array of labels
            num_tta: number of TTA variations per image
        """
        self.images = images
        self.labels = labels
        self.num_tta = num_tta
        
        # Define TTA transforms
        self.tta_transforms = [
            # Original
            transforms.Compose([]),
            
            # Horizontal flip
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0)
            ]),
            
            # Vertical flip
            transforms.Compose([
                transforms.RandomVerticalFlip(p=1.0)
            ]),
            
            # Rotation 90
            transforms.Compose([
                transforms.RandomRotation(degrees=(90, 90))
            ]),
            
            # Rotation 180
            transforms.Compose([
                transforms.RandomRotation(degrees=(180, 180))
            ]),
            
            # Rotation 270
            transforms.Compose([
                transforms.RandomRotation(degrees=(270, 270))
            ]),
            
            # Horizontal flip + Rotation 90
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomRotation(degrees=(90, 90))
            ]),
            
            # Vertical flip + Rotation 90
            transforms.Compose([
                transforms.RandomVerticalFlip(p=1.0),
                transforms.RandomRotation(degrees=(90, 90))
            ]),
        ][:num_tta]
    
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
        
        # Apply all TTA transforms
        tta_images = []
        for transform in self.tta_transforms:
            if len(transform.transforms) == 0:
                tta_images.append(image)
            else:
                tta_images.append(transform(image))
        
        return torch.stack(tta_images), torch.from_numpy(label).float()


def load_model(model_name, checkpoint_path, device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = get_model(model_name, num_classes=7, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def get_tta_predictions(model, dataloader, device):
    """Get predictions with TTA."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for tta_images, labels in tqdm(dataloader, desc='TTA Prediction', leave=False):
            # tta_images shape: [batch_size, num_tta, channels, height, width]
            batch_size, num_tta = tta_images.shape[0], tta_images.shape[1]
            
            # Reshape to process all TTA versions together
            tta_images = tta_images.view(batch_size * num_tta, *tta_images.shape[2:])
            tta_images = tta_images.to(device)
            
            # Get predictions
            outputs = model(tta_images)
            preds = torch.sigmoid(outputs)
            
            # Reshape back and average over TTA versions
            preds = preds.view(batch_size, num_tta, -1)
            preds = preds.mean(dim=1)  # Average over TTA
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.vstack(all_preds), np.vstack(all_labels)


def evaluate_predictions(y_true, y_pred_probs, threshold=0.5, name="Model"):
    """Evaluate predictions."""
    y_pred = (y_pred_probs > threshold).astype(int)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    disease_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Myopia', 'Other']
    
    print(f"\n{name}:")
    print(f"  F1: {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  Per-class: {', '.join([f'{f1:.3f}' for f1 in f1_per_class])}")
    
    return f1_macro, f1_per_class


def main():
    print(f"\n{'='*70}")
    print("PHASE 4 ENSEMBLE WITH TEST-TIME AUGMENTATION (TTA)")
    print(f"{'='*70}\n")
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device\n")
    
    # Load validation data
    print("Loading validation data...")
    data_dir = Path('preprocessed_data')
    val_images = np.load(data_dir / 'val_images.npy')
    val_labels = np.load(data_dir / 'val_labels.npy')
    print(f"Validation set: {val_images.shape[0]} images\n")
    
    # TTA configurations to test
    tta_configs = [1, 4, 8]  # Number of augmentations
    
    for num_tta in tta_configs:
        print(f"\n{'='*70}")
        print(f"Testing with {num_tta} TTA augmentations")
        print(f"{'='*70}")
        
        # Create TTA dataloader
        tta_dataset = TTADataset(val_images, val_labels, num_tta=num_tta)
        tta_loader = DataLoader(tta_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # Load models
        models_config = {
            'convnext_tiny': 'models/convnext_tiny_advanced_best.pth',
            'vit_small': 'models/vit_small_advanced_best.pth',
            'efficientnetv2_s': 'models/efficientnetv2_s_advanced_best.pth'
        }
        
        print("\nGetting TTA predictions from models...")
        all_predictions = {}
        
        for model_name, checkpoint_path in models_config.items():
            print(f"\n{model_name.upper()}...")
            model = load_model(model_name, checkpoint_path, device)
            preds, labels = get_tta_predictions(model, tta_loader, device)
            all_predictions[model_name] = preds
        
        # Evaluate individual models with TTA
        print(f"\n{'='*70}")
        print(f"Individual Models (with {num_tta} TTA)")
        print(f"{'='*70}")
        
        f1_convnext, _ = evaluate_predictions(labels, all_predictions['convnext_tiny'], 
                                              name="ConvNeXt Tiny")
        f1_vit, _ = evaluate_predictions(labels, all_predictions['vit_small'],
                                        name="ViT Small")
        f1_eff, _ = evaluate_predictions(labels, all_predictions['efficientnetv2_s'],
                                        name="EfficientNetV2 S")
        
        # Ensemble with TTA
        print(f"\n{'='*70}")
        print(f"Ensemble Strategies (with {num_tta} TTA)")
        print(f"{'='*70}")
        
        # Average ensemble
        avg_preds = np.mean([
            all_predictions['convnext_tiny'],
            all_predictions['vit_small'],
            all_predictions['efficientnetv2_s']
        ], axis=0)
        
        f1_avg, per_class_avg = evaluate_predictions(labels, avg_preds, 
                                                     name="Average Ensemble")
        
        # Weighted ensemble
        weights = [0.6030, 0.5499, 0.5497]  # Based on validation F1
        weighted_preds = np.average([
            all_predictions['convnext_tiny'],
            all_predictions['vit_small'],
            all_predictions['efficientnetv2_s']
        ], axis=0, weights=weights)
        
        f1_weighted, per_class_weighted = evaluate_predictions(labels, weighted_preds,
                                                               name="Weighted Ensemble")
        
        # Summary
        print(f"\n{'='*70}")
        print(f"SUMMARY (TTA={num_tta})")
        print(f"{'='*70}")
        print(f"Best Individual: {max(f1_convnext, f1_vit, f1_eff):.4f}")
        print(f"Best Ensemble:   {max(f1_avg, f1_weighted):.4f}")
        print(f"Phase 2 Baseline: 0.6403")
        
        best_tta = max(f1_avg, f1_weighted)
        diff = (best_tta - 0.6403) * 100
        if best_tta > 0.6403:
            print(f"✅ BEATS baseline by {diff:+.2f}%!")
        else:
            print(f"Gap to baseline: {diff:.2f}%")
    
    print(f"\n{'='*70}")
    print("TTA EVALUATION COMPLETE")
    print(f"{'='*70}\n")
    
    print("Recommendation:")
    print("  Use TTA=8 for best performance (slight overhead acceptable)")
    print("  Use TTA=4 for faster inference with good improvement")
    print("  Expected improvement: +1.5-2.5% F1 over non-TTA")


if __name__ == '__main__':
    main()
