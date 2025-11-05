"""
Advanced training script with data augmentation and modern architectures.
Trains models with MixUp/CutMix and advanced architectures (ViT, ConvNeXt, etc.)
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.augmentation import get_augmentation_policy
from src.advanced_models import get_model, count_parameters
from src.ensemble_models import EfficientNetB3Classifier, DenseNet121Classifier


class RetinalDataset(Dataset):
    """Dataset with augmentation support."""
    
    def __init__(self, images, labels, transform=None, batch_augmentation=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.batch_augmentation = batch_augmentation
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply per-image augmentation
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float()
            if image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            if image.max() > 1.0:
                image = image / 255.0
        
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
    pos_counts = labels.sum(axis=0)
    total = len(labels)
    weights = total / (2 * pos_counts)
    weights = weights / weights.min()
    return weights


def get_sample_weights(labels, class_weights):
    """Calculate sample weights for weighted sampling."""
    sample_weights = np.zeros(len(labels))
    for i, label in enumerate(labels):
        positive_classes = np.where(label > 0)[0]
        if len(positive_classes) > 0:
            sample_weights[i] = class_weights[positive_classes].max()
        else:
            sample_weights[i] = 1.0
    return sample_weights


def train_epoch(model, train_loader, criterion, optimizer, device, class_weights, batch_aug=None, 
                scaler=None, grad_accum_steps=1, grad_clip=1.0):
    """Train for one epoch with AMP and gradient accumulation support."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Add progress bar
    pbar = tqdm(train_loader, desc='Training', leave=False)
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Apply batch augmentation if provided (MixUp/CutMix)
        if batch_aug is not None:
            images, labels = batch_aug(images, labels)
        
        # Mixed precision forward pass
        if scaler is not None:
            with torch.amp.autocast(device_type='mps' if device.type == 'mps' else 'cuda', dtype=torch.float16):
                outputs = model(images)
                
                # Combined loss: 70% FocalLoss + 30% Weighted BCE
                focal_loss = criterion(outputs, labels)
                bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs, labels, pos_weight=class_weights.to(device), reduction='mean'
                )
                loss = 0.7 * focal_loss + 0.3 * bce_loss
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
        else:
            # Regular forward pass (FP32)
            outputs = model(images)
            
            # Combined loss
            focal_loss = criterion(outputs, labels)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs, labels, pos_weight=class_weights.to(device), reduction='mean'
            )
            loss = 0.7 * focal_loss + 0.3 * bce_loss
            loss = loss / grad_accum_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                # Gradient clipping (unscale first for AMP)
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Gradient clipping (FP32)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
            
            optimizer.zero_grad()
        
        running_loss += loss.item() * grad_accum_steps
        
        # Collect predictions
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.4f}'})
    
    # Handle any remaining gradients
    if len(train_loader) % grad_accum_steps != 0:
        if scaler is not None:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad()
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    avg_loss = running_loss / len(train_loader)
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='macro', zero_division=0)
    
    return avg_loss, f1


def validate(model, val_loader, criterion, device, class_weights, scaler=None):
    """Validate the model with AMP support."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # Add progress bar
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Mixed precision inference
            if scaler is not None:
                with torch.amp.autocast(device_type='mps' if device.type == 'mps' else 'cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    weighted_bce = nn.functional.binary_cross_entropy_with_logits(
                        outputs, labels, pos_weight=class_weights.to(device), reduction='mean'
                    )
                    total_loss = 0.7 * loss + 0.3 * weighted_bce
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                weighted_bce = nn.functional.binary_cross_entropy_with_logits(
                    outputs, labels, pos_weight=class_weights.to(device), reduction='mean'
                )
                total_loss = 0.7 * loss + 0.3 * weighted_bce
            
            running_loss += total_loss.item()
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_preds_binary = (all_preds > 0.5).astype(int)
    
    f1 = f1_score(all_labels, all_preds_binary, average='macro', zero_division=0)
    avg_loss = running_loss / len(val_loader)
    per_class_f1 = f1_score(all_labels, all_preds_binary, average=None, zero_division=0)
    
    return avg_loss, f1, per_class_f1


def main():
    parser = argparse.ArgumentParser(description='Train with advanced augmentation and architectures')
    parser.add_argument('--model', type=str, required=True,
                       choices=['vit_small', 'convnext_tiny', 'swin_tiny', 'efficientnetv2_s',
                               'resnet50', 'efficientnet_b3', 'densenet121'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--use-mixup', action='store_true', help='Use MixUp augmentation')
    parser.add_argument('--use-cutmix', action='store_true', help='Use CutMix augmentation')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable augmentation')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--use-amp', action='store_true', help='Use Automatic Mixed Precision (AMP)')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Learning rate warmup epochs (default: 5)')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping max norm (default: 1.0, 0 to disable)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Advanced Training: {args.model}")
    print(f"{'='*80}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Augmentation: {'Disabled' if args.no_augmentation else 'Enabled'}")
    print(f"MixUp: {args.use_mixup}")
    print(f"CutMix: {args.use_cutmix}")
    print(f"{'='*80}\n")
    
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
    
    # Load data
    print("Loading data...")
    # Use Phase 4C data (RGB, 384x384)
    data_dir = Path('preprocessed_data_phase4c')
    train_images = np.load(data_dir / 'train_images.npy')
    train_labels = np.load(data_dir / 'train_labels.npy')
    val_images = np.load(data_dir / 'val_images.npy')
    val_labels = np.load(data_dir / 'val_labels.npy')
    
    # Detect image size from data
    image_size = train_images.shape[1]  # Should be 384 for Phase 4C
    print(f"Train: {train_images.shape[0]} images at {image_size}x{image_size}")
    print(f"Val: {val_images.shape[0]} images at {image_size}x{image_size}\n")
    
    # Class weights
    class_weights = calculate_class_weights(train_labels)
    class_weights_tensor = torch.from_numpy(class_weights).float()
    disease_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Myopia', 'Other']
    print("Class weights:")
    for name, weight in zip(disease_names, class_weights):
        print(f"  {name:12s}: {weight:6.2f}x")
    print()
    
    # Get augmentation
    if args.no_augmentation:
        train_transform, batch_aug = None, None
        val_transform, _ = None, None
    else:
        train_transform, batch_aug = get_augmentation_policy(
            mode='train', image_size=image_size,
            use_mixup=args.use_mixup,
            use_cutmix=args.use_cutmix
        )
        val_transform, _ = get_augmentation_policy(mode='val', image_size=image_size)
    
    # Create datasets
    train_dataset = RetinalDataset(train_images, train_labels, transform=train_transform)
    val_dataset = RetinalDataset(val_images, val_labels, transform=val_transform)
    
    # Weighted sampling
    sample_weights = get_sample_weights(train_labels, class_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # M5 Optimized DataLoader settings for Phase 4C (384x384 images)
    import os
    num_workers = int(os.environ.get('M5_NUM_WORKERS', '4'))  # Default to 4 P-cores
    persistent_workers = os.environ.get('M5_PERSISTENT_WORKERS', '1') == '1'
    pin_memory = os.environ.get('M5_PIN_MEMORY', '1') == '1'
    prefetch_factor = int(os.environ.get('M5_PREFETCH_FACTOR', '2'))
    
    # Adjust batch size for larger images if not specified
    if image_size > 224 and args.batch_size == 64:
        # Auto-reduce batch size for 384x384 images
        original_batch = args.batch_size
        args.batch_size = 32
        print(f"NOTE: Auto-reduced batch size from {original_batch} to {args.batch_size} for {image_size}x{image_size} images")
        print(f"      (Larger images require more GPU memory)\n")
    
    # Fix for num_workers=0: can't use persistent_workers or prefetch_factor
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None
    
    print(f"M5 DataLoader Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Persistent workers: {persistent_workers}")
    print(f"  Pin memory: {pin_memory}")
    print(f"  Prefetch factor: {prefetch_factor}\n")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    # Load model
    print(f"Loading {args.model} model...")
    if args.model in ['vit_small', 'convnext_tiny', 'swin_tiny', 'efficientnetv2_s']:
        from src.advanced_models import get_model as get_advanced_model
        model = get_advanced_model(args.model, num_classes=7, pretrained=True)
    elif args.model == 'resnet50':
        from src.train import MultiLabelClassifier
        model = MultiLabelClassifier(num_classes=7, backbone='resnet50', pretrained=True)
    elif args.model == 'efficientnet_b3':
        model = EfficientNetB3Classifier(num_classes=7)
    elif args.model == 'densenet121':
        model = DenseNet121Classifier(num_classes=7)
    
    model = model.to(device)
    params = count_parameters(model)
    print(f"Parameters: {params/1e6:.1f}M\n")
    
    # Training setup
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            # Linear warmup
            return (epoch + 1) / args.warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize AMP scaler if requested
    scaler = None
    if args.use_amp:
        scaler = torch.amp.GradScaler('mps' if device.type == 'mps' else 'cuda')
        print(f"✓ Mixed Precision (AMP) enabled for {device.type.upper()}")
    
    # Determine gradient accumulation steps
    grad_accum_steps = args.grad_accum_steps
    # Skip grad accumulation for EfficientNetV2 (uses BatchNorm)
    if 'efficientnet' in args.model.lower() and grad_accum_steps > 1:
        print(f"⚠️  Gradient accumulation disabled for {args.model} (uses BatchNorm)")
        grad_accum_steps = 1
    
    # Print optimization settings
    print(f"\n{'='*60}")
    print("Training Optimizations:")
    print(f"{'='*60}")
    print(f"  AMP (Mixed Precision): {'✓ Enabled' if args.use_amp else '✗ Disabled'}")
    print(f"  Gradient Accumulation: {'✓ ' + str(grad_accum_steps) + ' steps' if grad_accum_steps > 1 else '✗ Disabled'}")
    if grad_accum_steps > 1:
        effective_batch = args.batch_size * grad_accum_steps
        print(f"    → Effective batch size: {effective_batch}")
    print(f"  LR Warmup: {'✓ ' + str(args.warmup_epochs) + ' epochs' if args.warmup_epochs > 0 else '✗ Disabled'}")
    print(f"  Gradient Clipping: {'✓ max_norm=' + str(args.grad_clip) if args.grad_clip > 0 else '✗ Disabled'}")
    print(f"{'='*60}\n")
    
    # Training loop
    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': [], 'per_class_f1': []}
    best_val_f1 = 0.0
    best_epoch = 0
    
    print("Starting training...\n")
    # Add epoch progress bar
    epoch_pbar = tqdm(range(args.epochs), desc='Epochs', position=0)
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, class_weights_tensor, batch_aug,
            scaler=scaler, grad_accum_steps=grad_accum_steps, grad_clip=args.grad_clip
        )
        val_loss, val_f1, per_class_f1 = validate(
            model, val_loader, criterion, device, class_weights_tensor, scaler=scaler
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['per_class_f1'].append(per_class_f1)
        
        # Update progress bar with metrics
        epoch_pbar.set_postfix({
            'train_f1': f'{train_f1:.4f}',
            'val_f1': f'{val_f1:.4f}',
            'best': f'{best_val_f1:.4f}'
        })
        
        # Print detailed results
        tqdm.write(f"\nEpoch {epoch+1}/{args.epochs} Results:")
        tqdm.write(f"  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        tqdm.write(f"  Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")
        tqdm.write(f"  Per-class F1: {' '.join([f'{f1:.3f}' for f1 in per_class_f1])}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            
            save_path = Path(f'models/{args.model}_advanced_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'per_class_f1': per_class_f1,
                'history': history,
                'args': vars(args)
            }, save_path)
            
            tqdm.write(f"  ✓ New best model saved (F1: {val_f1:.4f})\n")
        else:
            tqdm.write("")
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Best Validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Model saved to: models/{args.model}_advanced_best.pth")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
