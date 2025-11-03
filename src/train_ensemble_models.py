"""
Train additional architectures (EfficientNet-B3, DenseNet-121) for ensemble.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import time
from tqdm import tqdm
import json
import sys
sys.path.append('/Users/fdb/VSCode/ODR')

from config import LABEL_COLUMNS, RANDOM_SEED
from src.ensemble_models import EfficientNetB3Classifier, DenseNet121Classifier
from src.augmented_dataset import AugmentedODIRDataset
from src.imbalance_solutions import FocalLoss, get_weighted_sampler

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class ODIRDataset(Dataset):
    """PyTorch Dataset for ODIR-5K preprocessed data."""
    
    def __init__(self, images_path: str, labels_path: str):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        print(f"Loaded dataset: {len(self.images)} samples")
        print(f"  Image shape: {self.images.shape}")
        print(f"  Label shape: {self.labels.shape}")
        print(f"  NOTE: This is the old dataset class. Consider using AugmentedODIRDataset for augmentation support.")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PyTorch tensor and transpose to (C, H, W)
        image = torch.FloatTensor(image).permute(2, 0, 1)
        label = torch.FloatTensor(label)
        
        return image, label


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 Using MPS (Metal Performance Shaders) - Apple Silicon acceleration!")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 Using CUDA - GPU acceleration!")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU - training will be slower")
    return device


def calculate_class_weights(train_labels: np.ndarray) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset."""
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    weights = neg_counts / pos_counts
    return torch.FloatTensor(weights)


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> Dict:
    """Calculate multi-label classification metrics."""
    pred_labels = (predictions >= threshold).astype(float)
    
    # Label-wise accuracy
    label_acc = np.mean(pred_labels == targets)
    
    # Sample-wise accuracy (all labels must match)
    sample_acc = np.mean(np.all(pred_labels == targets, axis=1))
    
    # Per-class metrics
    per_class_metrics = {}
    f1_scores = []
    for i, label in enumerate(LABEL_COLUMNS):
        tp = np.sum((pred_labels[:, i] == 1) & (targets[:, i] == 1))
        fp = np.sum((pred_labels[:, i] == 1) & (targets[:, i] == 0))
        fn = np.sum((pred_labels[:, i] == 0) & (targets[:, i] == 1))
        tn = np.sum((pred_labels[:, i] == 0) & (targets[:, i] == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        f1_scores.append(f1)
    
    # Calculate mean F1 score
    mean_f1 = np.mean(f1_scores)
    
    return {
        'label_accuracy': label_acc,
        'sample_accuracy': sample_acc,
        'mean_f1': mean_f1,
        'per_class': per_class_metrics
    }


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions for metrics
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    
    return total_loss / len(train_loader), metrics


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Store predictions
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    
    return total_loss / len(val_loader), metrics


def train_model(model_name: str, model: nn.Module, train_loader, val_loader, 
                device, num_epochs: int = 25, save_path: str = None,
                use_focal_loss: bool = True, focal_gamma: float = 2.0):
    """Train a model with advanced loss functions."""
    print(f"\n{'='*70}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*70}\n")
    
    # Calculate class weights
    train_labels = train_loader.dataset.labels
    class_weights = calculate_class_weights(train_labels).to(device)
    
    # Setup training with Focal Loss or BCE
    if use_focal_loss:
        print(f"✓ Using Focal Loss (alpha=0.25, gamma={focal_gamma})")
        criterion = FocalLoss(alpha=0.25, gamma=focal_gamma)
    else:
        print("✓ Using BCEWithLogitsLoss with class weights")
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Training loop
    best_mean_f1 = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_mean_f1': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler based on Mean F1
        scheduler.step(val_metrics['mean_f1'])
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['label_accuracy'])
        history['val_mean_f1'].append(val_metrics['mean_f1'])
        
        # Print metrics
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Acc: {val_metrics['label_accuracy']:.4f} ({val_metrics['label_accuracy']*100:.2f}%)")
        print(f"  Val Mean F1: {val_metrics['mean_f1']:.4f}")
        
        # Print per-class F1 scores
        print(f"  Per-class F1:", end=" ")
        for label in LABEL_COLUMNS:
            f1 = val_metrics['per_class'][label]['f1']
            print(f"{label}:{f1:.3f}", end=" ")
        print()
        
        # Save best model based on Mean F1 (same as baseline)
        if val_metrics['mean_f1'] > best_mean_f1:
            best_mean_f1 = val_metrics['mean_f1']
            if save_path:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_metrics['label_accuracy'],
                    'val_mean_f1': best_mean_f1,
                    'val_metrics': val_metrics,
                    'label_columns': LABEL_COLUMNS,
                    'history': history
                }
                torch.save(checkpoint, save_path, _use_new_zipfile_serialization=False)
                print(f"  ✓ Saved best model (mean_f1: {best_mean_f1:.4f})")
    
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} TRAINING COMPLETE")
    print(f"Best Mean F1: {best_mean_f1:.4f}, Val Acc: {val_metrics['label_accuracy']:.4f} ({val_metrics['label_accuracy']*100:.2f}%)")
    print(f"{'='*70}\n")
    
    return history


def main():
    """Main training function."""
    import argparse
    parser = argparse.ArgumentParser(description='Train ensemble models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['efficientnet_b3', 'densenet121', 'both'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of epochs (default: 25)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-focal', action='store_true',
                       help='Use BCE instead of Focal Loss')
    parser.add_argument('--no-weighted-sampling', action='store_true',
                       help='Disable weighted sampling')
    args = parser.parse_args()
    
    # Configuration (M5 MacBook Pro Optimized)
    BATCH_SIZE = 64  # Optimized for M5 with 32GB
    NUM_EPOCHS = args.epochs
    NUM_WORKERS = 0  # Single-threaded for macOS stability
    USE_AUGMENTATION = not args.no_augment
    USE_FOCAL_LOSS = not args.no_focal
    USE_WEIGHTED_SAMPLING = not args.no_weighted_sampling
    FOCAL_GAMMA = 2.0
    
    print("="*70)
    print("ENSEMBLE MODEL TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Num workers: {NUM_WORKERS}")
    print(f"\nAdvanced Features:")
    print(f"  Data augmentation: {USE_AUGMENTATION}")
    print(f"  Focal Loss: {USE_FOCAL_LOSS} (gamma={FOCAL_GAMMA})")
    print(f"  Weighted sampling: {USE_WEIGHTED_SAMPLING}")
    
    # Get device
    device = get_device()
    
    # Load datasets (7-class data) with augmentation support
    print("\n📂 Loading preprocessed data...")
    train_dataset = AugmentedODIRDataset(
        'preprocessed_data/train_images.npy',
        'preprocessed_data/train_labels.npy',
        augment=USE_AUGMENTATION,
        augment_probability=0.5
    )
    val_dataset = AugmentedODIRDataset(
        'preprocessed_data/val_images.npy',
        'preprocessed_data/val_labels.npy',
        augment=False  # Never augment validation
    )
    
    # Create data loaders with optional weighted sampling
    if USE_WEIGHTED_SAMPLING:
        print("\n📊 Using weighted sampling to balance classes...")
        train_labels = train_dataset.labels
        pos_counts = train_labels.sum(axis=0)
        class_weights = len(train_labels) / (len(LABEL_COLUMNS) * pos_counts)
        sampler = get_weighted_sampler(train_labels, class_weights)
        
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=False
        )
    else:
        print("\n📊 Using standard random sampling...")
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=False
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    
    # Train models
    Path('models').mkdir(exist_ok=True)
    
    if args.model in ['efficientnet_b3', 'both']:
        print("\n" + "="*70)
        print("1. TRAINING EFFICIENTNET-B3")
        print("="*70)
        model = EfficientNetB3Classifier(num_classes=len(LABEL_COLUMNS))
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        train_model(
            'EfficientNet-B3',
            model,
            train_loader,
            val_loader,
            device,
            num_epochs=NUM_EPOCHS,
            save_path='models/efficientnet_b3_model.pth',
            use_focal_loss=USE_FOCAL_LOSS,
            focal_gamma=FOCAL_GAMMA
        )
    
    if args.model in ['densenet121', 'both']:
        print("\n" + "="*70)
        print("2. TRAINING DENSENET-121")
        print("="*70)
        model = DenseNet121Classifier(num_classes=len(LABEL_COLUMNS))
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        train_model(
            'DenseNet-121',
            model,
            train_loader,
            val_loader,
            device,
            num_epochs=NUM_EPOCHS,
            save_path='models/densenet121_model.pth',
            use_focal_loss=USE_FOCAL_LOSS,
            focal_gamma=FOCAL_GAMMA
        )
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
