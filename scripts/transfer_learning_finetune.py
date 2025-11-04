"""
Transfer Learning Fine-Tuning Script
Fine-tune baseline models (trained on patient-level labels) on eye-specific labels.

This script:
1. Loads pre-trained baseline models (85.28% F1 ensemble)
2. Fine-tunes them on eye-specific labeled data
3. Uses much lower learning rate to preserve learned fundus features
4. Expected result: 75-80% F1 (vs 64% from scratch)
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import only ensemble models, define ResNet locally to avoid circular imports
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
        
        # Replace final layer for multi-label classification
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
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # Ensure correct shape: [C, H, W]
        if image.shape[-1] == 3:  # [H, W, C]
            image = image.permute(2, 0, 1)
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, torch.from_numpy(label).float()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def calculate_class_weights(labels):
    """Calculate class weights for imbalanced dataset."""
    # Count positive samples for each class
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    
    # Calculate weights (inverse frequency)
    total = len(labels)
    weights = total / (2 * pos_counts)
    
    # Normalize weights
    weights = weights / weights.min()
    
    return weights


def get_sample_weights(labels, class_weights):
    """Calculate sample weights for weighted sampling."""
    # For each sample, use the maximum class weight it belongs to
    sample_weights = np.zeros(len(labels))
    for i, label in enumerate(labels):
        positive_classes = np.where(label > 0)[0]
        if len(positive_classes) > 0:
            sample_weights[i] = class_weights[positive_classes].max()
        else:
            sample_weights[i] = 1.0  # Default weight for all-negative samples
    
    return sample_weights


def train_epoch(model, train_loader, criterion, optimizer, device, class_weights):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss with class weights
        loss = criterion(outputs, labels)
        
        # Add weighted BCE component for stability
        weighted_bce = nn.functional.binary_cross_entropy_with_logits(
            outputs, labels, pos_weight=class_weights.to(device), reduction='mean'
        )
        total_loss = 0.7 * loss + 0.3 * weighted_bce
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += total_loss.item()
        preds = torch.sigmoid(outputs).cpu().detach().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_preds_binary = (all_preds > 0.5).astype(int)
    
    f1 = f1_score(all_labels, all_preds_binary, average='macro', zero_division=0)
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss, f1


def validate(model, val_loader, criterion, device, class_weights):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            weighted_bce = nn.functional.binary_cross_entropy_with_logits(
                outputs, labels, pos_weight=class_weights.to(device), reduction='mean'
            )
            total_loss = 0.7 * loss + 0.3 * weighted_bce
            
            running_loss += total_loss.item()
            
            # Store predictions
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_preds_binary = (all_preds > 0.5).astype(int)
    
    f1 = f1_score(all_labels, all_preds_binary, average='macro', zero_division=0)
    avg_loss = running_loss / len(val_loader)
    
    # Per-class F1 scores
    per_class_f1 = f1_score(all_labels, all_preds_binary, average=None, zero_division=0)
    
    return avg_loss, f1, per_class_f1, all_preds


def fine_tune_model(model_name, num_epochs=15, learning_rate=5e-6, batch_size=64):
    """
    Fine-tune a baseline model on eye-specific labels.
    
    Args:
        model_name: 'resnet50', 'efficientnet_b3', or 'densenet121'
        num_epochs: Number of fine-tuning epochs (default: 15)
        learning_rate: Learning rate (default: 5e-6, much lower than baseline)
        batch_size: Batch size (default: 64)
    """
    print(f"\n{'='*80}")
    print(f"Transfer Learning Fine-Tuning: {model_name}")
    print(f"{'='*80}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate} (vs 1e-4 baseline)")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*80}\n")
    
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Load enhanced data (eye-specific labels)
    print("\nLoading enhanced data with eye-specific labels...")
    data_dir = Path('preprocessed_data')
    
    train_images = np.load(data_dir / 'train_images.npy')
    train_labels = np.load(data_dir / 'train_labels.npy')
    val_images = np.load(data_dir / 'val_images.npy')
    val_labels = np.load(data_dir / 'val_labels.npy')
    
    print(f"Train: {train_images.shape[0]} images, {train_labels.shape[1]} classes")
    print(f"Val: {val_images.shape[0]} images, {val_labels.shape[1]} classes")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_labels)
    class_weights_tensor = torch.from_numpy(class_weights).float()
    
    print(f"\nClass weights (for imbalance):")
    disease_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Myopia', 'Other']
    for i, (name, weight) in enumerate(zip(disease_names, class_weights)):
        pos_count = train_labels[:, i].sum()
        print(f"  {name:12s}: {weight:6.2f}x (positive samples: {pos_count:.0f})")
    
    # Create datasets
    train_dataset = RetinalDataset(train_images, train_labels)
    val_dataset = RetinalDataset(val_images, val_labels)
    
    # Create weighted sampler for training
    sample_weights = get_sample_weights(train_labels, class_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,  # macOS stability
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Load pre-trained baseline model
    print(f"\nLoading baseline model from: models_baseline_transfer/{model_name}_model.pth")
    if model_name == 'resnet50':
        baseline_path = Path('models_baseline_transfer/baseline_model.pth')
    else:
        baseline_path = Path(f'models_baseline_transfer/{model_name}_model.pth')
    
    # Initialize model architecture based on model type
    num_classes = 7
    if model_name == 'resnet50':
        model = MultiLabelClassifier(num_classes=num_classes, backbone='resnet50', pretrained=False)
    elif model_name == 'efficientnet_b3':
        model = EfficientNetB3Classifier(num_classes=num_classes)
    elif model_name == 'densenet121':
        model = DenseNet121Classifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Load baseline weights
    checkpoint = torch.load(baseline_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        baseline_f1 = checkpoint.get('best_val_f1', checkpoint.get('val_f1', 'N/A'))
        print(f"Loaded baseline model from epoch {checkpoint.get('epoch', 'N/A')}")
        if isinstance(baseline_f1, float):
            print(f"Baseline validation F1: {baseline_f1:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded baseline model weights")
    
    print(f"\n✓ Starting from pre-trained baseline (patient-level labels)")
    print(f"✓ Fine-tuning on eye-specific labels with lower learning rate")
    
    # Setup training components
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler (optional, for longer fine-tuning)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_f1': [],
        'val_loss': [],
        'val_f1': [],
        'per_class_f1': []
    }
    
    best_val_f1 = 0.0
    best_epoch = 0
    
    # Training loop
    print(f"\nStarting fine-tuning for {num_epochs} epochs...")
    print(f"{'='*80}\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, class_weights_tensor
        )
        
        # Validate
        val_loss, val_f1, per_class_f1, _ = validate(
            model, val_loader, criterion, device, class_weights_tensor
        )
        
        # Update scheduler
        scheduler.step(val_f1)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['per_class_f1'].append(per_class_f1)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")
        print(f"Per-class F1: {' '.join([f'{f1:.3f}' for f1 in per_class_f1])}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            
            # Save checkpoint
            save_path = Path(f'models/{model_name}_finetuned_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'per_class_f1': per_class_f1,
                'history': history
            }, save_path)
            
            print(f"✓ New best model saved (F1: {val_f1:.4f})")
        
        print()
    
    # Training complete
    print(f"\n{'='*80}")
    print(f"Fine-tuning Complete!")
    print(f"{'='*80}")
    print(f"Best Validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Model saved to: models/{model_name}_finetuned_best.pth")
    
    # Plot training history
    plot_history(history, model_name)
    
    return best_val_f1, history


def plot_history(history, model_name):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[1].plot(history['train_f1'], label='Train F1', linewidth=2)
    axes[1].plot(history['val_f1'], label='Val F1', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title(f'{model_name} - F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'results/{model_name}_finetuning_history.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {save_path}")
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Fine-tune baseline models on eye-specific labels'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['resnet50', 'efficientnet_b3', 'densenet121'],
        help='Model to fine-tune'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Number of fine-tuning epochs (default: 15)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-6,
        help='Learning rate (default: 5e-6)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    
    args = parser.parse_args()
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Fine-tune model
    best_f1, history = fine_tune_model(
        model_name=args.model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    
    print(f"\n{'='*80}")
    print(f"Transfer Learning Results for {args.model}:")
    print(f"  Best Validation F1: {best_f1:.4f}")
    print(f"  Expected improvement over from-scratch: +10-15% F1")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
