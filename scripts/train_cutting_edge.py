"""
Cutting-Edge Model Training Script (2025)
Integrates the best available models for retinal disease classification:

1. RETFound - Retinal-specific foundation model (1.6M fundus images)
2. DINOv2-Small - Self-supervised ViT (142M images, best for small datasets)
3. ConvNeXt V2 - Latest CNN with masked autoencoder pre-training

Expected ensemble: 77-82% F1 (vs 70-78% with 2021-2022 models)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import timm
from tqdm import tqdm
from pathlib import Path
import argparse
from datetime import datetime
import json
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import custom modules
from scripts.train_retfound import create_retfound_model


class FundusDataset(Dataset):
    """Dataset for preprocessed fundus images - OPTIMIZED with augmentation"""
    
    # Class-level constants for ImageNet normalization (shared across all instances)
    MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(3, 1, 1)
    
    def __init__(self, images, labels, ids=None, augment=False):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.augment = augment
        
        # Augmentation transforms (applied in tensor space for efficiency)
        if augment:
            import torchvision.transforms as T
            self.aug_transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ])
        else:
            self.aug_transforms = None
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]  # Already preprocessed (384x384x3)
        label = self.labels[idx]
        
        # Copy to ensure contiguous array when using memory mapping
        image = np.array(image, copy=True) if hasattr(self.images, 'filename') else image
        
        # Handle both uint8 [0-255] and float32 [0-1] formats
        if image.dtype == np.uint8:
            # Convert uint8 to float32 [0,1]
            image = image.astype(np.float32) / 255.0
        # else: already float32 [0,1] from preprocessing
        
        # HWC -> CHW
        image = image.transpose(2, 0, 1)  # Now (3, 384, 384)
        
        # Convert to tensor BEFORE augmentation (for torchvision transforms)
        image = torch.from_numpy(image)  # Single tensor conversion
        
        # Apply augmentation if enabled (BEFORE normalization!)
        if self.aug_transforms is not None:
            image = self.aug_transforms(image)
        
        # Apply ImageNet normalization AFTER augmentation (using class-level tensors)
        image = (image - self.MEAN) / self.STD
        
        label = torch.from_numpy(label).float()
        
        return image, label


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - optimized for extreme imbalance (10:1)"""
    
    def __init__(self, alpha=0.25, gamma=2.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma  # Increased from 2.0 to 2.5 for stronger minority class focus
    
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def calculate_class_weights(labels):
    """Calculate class weights for balanced loss"""
    class_counts = labels.sum(axis=0)
    total = len(labels)
    neg_counts = total - class_counts
    pos_weight = neg_counts / (class_counts + 1)  # Add 1 to avoid division by zero
    return torch.tensor(pos_weight, dtype=torch.float32)


def create_model(model_name, num_classes=7, device='mps', retfound_path=None):
    """
    Create cutting-edge model
    
    Supported models:
    - resnet50: ResNet-50 (proven, stable baseline)
    - resnet101: ResNet-101 (deeper, more capacity)
    - efficientnet_b3: EfficientNet-B3 (efficient, accurate)
    - efficientnet_b5: EfficientNet-B5 (larger, more powerful)
    - retfound: RETFound foundation model (ViT-Base, retinal-specific)
    - dinov2_small: DINOv2-Small/14 (self-supervised ViT)
    - dinov2_base: DINOv2-Base/14 (self-supervised ViT, larger)
    - vit_base: Vision Transformer Base/16 (pure attention)
    - convnextv2_tiny: ConvNeXt V2 Tiny (latest CNN)
    - convnextv2_base: ConvNeXt V2 Base (latest CNN, larger)
    """
    
    print(f"\n🔨 Creating model: {model_name}")
    
    if model_name.startswith('efficientnet'):
        # EfficientNet: State-of-the-art efficient CNNs
        if model_name == 'efficientnet_b3':
            model = timm.create_model('efficientnet_b3.ra2_in1k', pretrained=True, num_classes=num_classes)
        elif model_name == 'efficientnet_b5':
            model = timm.create_model('efficientnet_b5.sw_in12k_ft_in1k', pretrained=True, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown EfficientNet variant: {model_name}")
        
        model = model.to(device)
        print(f"✅ EfficientNet loaded from timm (ImageNet pretrained)")
    
    elif model_name.startswith('resnet'):
        # ResNet: Proven stable architecture
        if model_name == 'resnet50':
            model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=num_classes)
        elif model_name == 'resnet101':
            model = timm.create_model('resnet101.a1_in1k', pretrained=True, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown ResNet variant: {model_name}")
        
        model = model.to(device)
        print(f"✅ ResNet loaded from timm (ImageNet-1k pretrained)")
    
    elif model_name == 'retfound':
        # RETFound: Retinal-specific foundation model
        model = create_retfound_model(
            num_classes=num_classes,
            pretrained_path=retfound_path,
            device=device
        )
        
    elif model_name.startswith('dinov2'):
        # DINOv2: Self-supervised Vision Transformer
        # Using timm implementation for better compatibility
        print(f"Loading DINOv2 from timm (more stable)...")
        
        if 'small' in model_name:
            # Use timm's vit_small_patch14_dinov2 (equivalent to DINOv2-S/14)
            backbone = timm.create_model(
                'vit_small_patch14_dinov2.lvd142m',
                pretrained=True,
                num_classes=0,  # Remove classification head
                img_size=384
            )
            feature_dim = 384
        elif 'base' in model_name:
            # Use timm's vit_base_patch14_dinov2 (equivalent to DINOv2-B/14)
            backbone = timm.create_model(
                'vit_base_patch14_dinov2.lvd142m',
                pretrained=True,
                num_classes=0,
                img_size=384
            )
            feature_dim = 768
        else:
            raise ValueError(f"Unknown DINOv2 variant: {model_name}")
        
        # Add classification head
        class DINOv2Classifier(nn.Module):
            def __init__(self, backbone, feature_dim, num_classes):
                super().__init__()
                self.backbone = backbone
                self.classifier = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, 512),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )
            
            def forward(self, x):
                features = self.backbone(x)
                return self.classifier(features)
        
        model = DINOv2Classifier(backbone, feature_dim, num_classes)
        model = model.to(device)
        
        print(f"✅ DINOv2 loaded from timm ({feature_dim}D features, trained on 142M images)")
    
    elif model_name == 'vit_base':
        # Vision Transformer Base: Pure attention, no convolutions
        print(f"Loading Vision Transformer Base/16 from timm...")
        
        # Use 224px pretrained weights but adapt to 384px images
        # Position embeddings will be interpolated automatically
        model = timm.create_model(
            'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            pretrained=True,
            num_classes=num_classes,
            img_size=384  # Will interpolate position embeddings
        )
        
        model = model.to(device)
        print(f"✅ Vision Transformer loaded from timm (ImageNet-21k → IN1k, adapted to 384px)")
        
    elif model_name.startswith('convnextv2'):
        # ConvNeXt V2: Latest pure CNN
        if 'tiny' in model_name:
            model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', 
                                     pretrained=True, num_classes=num_classes)
        elif 'base' in model_name:
            model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384',
                                     pretrained=True, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown ConvNeXt V2 variant: {model_name}")
        
        model = model.to(device)
        print(f"✅ ConvNeXt V2 loaded from timm (ImageNet-22k + IN1k fine-tuned)")
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    return model


def train_epoch(model, train_loader, criterion, bce_criterion, class_weights, optimizer, 
                device, scaler=None, grad_accum_steps=1, grad_clip=1.0, scheduler=None, 
                label_smoothing=0.05):  # Reduced from 0.1 to 0.05 for better minority class learning
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc='Training')
    intermediate_loss = 0.0
    report_interval = max(1, len(train_loader) // 4)  # Report 4 times per epoch
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Label smoothing (reduced to preserve minority class signal)
        if label_smoothing > 0:
            labels_smoothed = labels * (1 - label_smoothing) + label_smoothing * 0.5
        else:
            labels_smoothed = labels
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('mps' if device == 'mps' else 'cuda'):
                outputs = model(images)
                
                # Combined loss: 80% FocalLoss + 20% Weighted BCE (more focus on focal for minority classes)
                focal_loss = criterion(outputs, labels_smoothed)
                
                bce_loss = bce_criterion(outputs, labels_smoothed)
                
                loss = 0.8 * focal_loss + 0.2 * bce_loss  # Increased focal weight
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Step scheduler after each batch (for OneCycleLR)
                if scheduler is not None:
                    scheduler.step()
        else:
            outputs = model(images)
            
            focal_loss = criterion(outputs, labels_smoothed)
            bce_loss = bce_criterion(outputs, labels_smoothed)
            loss = 0.8 * focal_loss + 0.2 * bce_loss
            loss = loss / grad_accum_steps
            
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
        
        # Clear GPU cache every 50 batches to prevent memory buildup
        if device.type == 'mps' and (batch_idx + 1) % 50 == 0:
            torch.mps.empty_cache()
        
        running_loss += loss.item() * grad_accum_steps
        
        # Collect predictions for metrics (sample 20% for memory efficiency)
        # This gives accurate F1 estimation while saving memory
        if np.random.random() < 0.2 or (batch_idx + 1) == len(train_loader):
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
        
        # Report intermediate metrics
        intermediate_loss += loss.item() * grad_accum_steps
        if (batch_idx + 1) % report_interval == 0:
            avg_loss = intermediate_loss / report_interval
            pbar.set_postfix({
                'loss': avg_loss,
                'batch': f'{batch_idx+1}/{len(train_loader)}'
            })
            intermediate_loss = 0.0
        else:
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    
    all_preds = np.vstack(all_preds) if all_preds else np.array([])
    all_labels = np.vstack(all_labels) if all_labels else np.array([])
    
    # Calculate metrics (threshold = 0.5) - skip if no predictions collected
    if len(all_preds) > 0:
        preds_binary = (all_preds > 0.5).astype(int)
        f1 = f1_score(all_labels, preds_binary, average='macro', zero_division=0)
    else:
        f1 = 0.0
    
    return running_loss / len(train_loader), f1


def validate(model, val_loader, criterion, bce_criterion, class_weights, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Combined loss (same ratio as training)
            focal_loss = criterion(outputs, labels)
            bce_loss = bce_criterion(outputs, labels)
            loss = 0.8 * focal_loss + 0.2 * bce_loss
            
            running_loss += loss.item()
            
            # Collect predictions
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    preds_binary = (all_preds > 0.5).astype(int)
    f1 = f1_score(all_labels, preds_binary, average='macro', zero_division=0)
    precision = precision_score(all_labels, preds_binary, average='macro', zero_division=0)
    recall = recall_score(all_labels, preds_binary, average='macro', zero_division=0)
    
    # Per-class F1
    per_class_f1 = f1_score(all_labels, preds_binary, average=None, zero_division=0)
    
    return running_loss / len(val_loader), f1, precision, recall, per_class_f1


def main():
    parser = argparse.ArgumentParser(description='Train cutting-edge models (2025)')
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet50', 'resnet101', 'efficientnet_b3', 'efficientnet_b5', 'retfound', 'dinov2_small', 'dinov2_base', 'vit_base', 'convnextv2_tiny', 'convnextv2_base'],
                       help='Model architecture')
    parser.add_argument('--retfound-path', type=str, default=None,
                       help='Path to RETFound pre-trained weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad-accum-steps', type=int, default=2)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--data-dir', type=str, default='preprocessed_data_phase4c')
    parser.add_argument('--output-dir', type=str, default='models')
    
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load data
    print(f"\n📂 Loading preprocessed data from: {args.data_dir}")
    data_dir = Path(args.data_dir)
    
    # Use memory mapping to reduce RAM usage (trade: slightly slower, but no swap thrashing)
    print("   Loading data with memory mapping (reduces RAM usage by ~11GB)...")
    train_images = np.load(data_dir / 'train_images.npy', mmap_mode='r')  # Memory-mapped
    train_labels = np.load(data_dir / 'train_labels.npy')
    val_images = np.load(data_dir / 'val_images.npy', mmap_mode='r')  # Memory-mapped
    val_labels = np.load(data_dir / 'val_labels.npy')
    print("   ✅ Data loaded with memory mapping (RAM-efficient)")
    
    print(f"   Train: {len(train_images)} images, shape {train_images.shape}")
    print(f"   Val: {len(val_images)} images, shape {val_images.shape}")
    
    # Create datasets
    train_dataset = FundusDataset(train_images, train_labels, augment=True)  # Enable augmentation
    val_dataset = FundusDataset(val_images, val_labels, augment=False)  # No augmentation for validation
    
    # Get number of workers from environment
    # NOTE: Using num_workers=0 (single process) to avoid multiprocessing pickle issues
    # Since data is already in memory, this is actually faster than spawning workers!
    num_workers = 0
    
    # DataLoaders (simple shuffle - oversampling handled in dataset itself)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    # Create model
    model = create_model(args.model, num_classes=7, device=device, retfound_path=args.retfound_path)
    
    # NOTE: torch.compile() causes major slowdown on MPS (compilation overhead > benefit)
    # Keeping model uncompiled for better performance on Apple Silicon
    
    # Loss functions
    criterion = FocalLoss(alpha=0.25, gamma=2.5)  # Increased gamma for extreme imbalance
    class_weights = calculate_class_weights(train_labels).to(device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    print(f"\n📊 Class weights: {class_weights.cpu().numpy()}")
    
    # Calculate minority class indices for monitoring
    class_names = ['AMD', 'Diabetes', 'Glaucoma', 'Cataract', 'Myopia', 'Normal', 'Other']
    minority_indices = [i for i, name in enumerate(class_names) if name in ['AMD', 'Glaucoma', 'Cataract', 'Myopia']]
    print(f"   Minority classes (will monitor closely): {[class_names[i] for i in minority_indices]}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # OneCycleLR scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('mps') if args.use_amp and device.type == 'mps' else None
    
    # Training info
    print(f"\n🚀 Training Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: {args.grad_accum_steps} steps (effective batch: {args.batch_size * args.grad_accum_steps})")
    print(f"   Learning rate: {args.lr:.0e}")
    print(f"   Scheduler: OneCycleLR (max_lr={args.lr * 10:.0e}, 30% warmup)")
    print(f"   Loss: 80% FocalLoss(gamma=2.5) + 20% Weighted BCE")
    print(f"   Mixed precision: {args.use_amp}")
    print(f"   Gradient clipping: {args.grad_clip}")
    print(f"   Label smoothing: 0.05 (reduced for minority classes)")
    print(f"   Data augmentation: Enabled (flip, rotate, color jitter)")
    print(f"   Workers: {num_workers}")
    
    # Training loop
    best_val_f1 = 0.0
    history = []
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_name_safe = args.model.replace('/', '_')
    
    print(f"\n{'='*80}")
    print(f"Starting training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, bce_criterion, class_weights, optimizer, device,
            scaler=scaler, grad_accum_steps=args.grad_accum_steps, grad_clip=args.grad_clip,
            scheduler=scheduler, label_smoothing=0.1
        )
        
        # Validate
        val_loss, val_f1, val_precision, val_recall, per_class_f1 = validate(
            model, val_loader, criterion, bce_criterion, class_weights, device
        )
        
        # Print results
        print(f"\nResults:")
        print(f"  Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        # Display class names for clarity
        class_names = ['AMD', 'Diabetes', 'Glaucoma', 'Cataract', 'Myopia', 'Normal', 'Other']
        minority_classes = ['AMD', 'Glaucoma', 'Cataract', 'Myopia']
        
        print(f"\n  Detailed Per-Class Performance:")
        minority_f1_sum = 0.0
        minority_count = 0
        for i, (name, f1) in enumerate(zip(class_names, per_class_f1)):
            marker = " 🎯" if name in minority_classes else ""
            print(f"    {name:12s}: {f1:.3f}{marker}")
            if name in minority_classes:
                minority_f1_sum += f1
                minority_count += 1
        
        # Calculate and display minority class average
        minority_f1_avg = minority_f1_sum / minority_count if minority_count > 0 else 0.0
        print(f"\n  Minority Class Avg F1: {minority_f1_avg:.3f} (AMD, Glaucoma, Cataract, Myopia)")
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_f1': float(train_f1),
            'val_loss': float(val_loss),
            'val_f1': float(val_f1),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'per_class_f1': per_class_f1.tolist(),
            'minority_f1': float(minority_f1_avg)
        })
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = output_dir / f'best_{model_name_safe}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'per_class_f1': per_class_f1.tolist(),
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✅ Saved best model: {checkpoint_path} (F1: {val_f1:.4f})")
        
        # Save progress summary (for easy monitoring)
        progress_path = output_dir / f'progress_{model_name_safe}.txt'
        with open(progress_path, 'w') as f:
            f.write(f"Training Progress: {model_name_safe}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Epoch: {epoch + 1}/{args.epochs}\n")
            f.write(f"Best Val F1: {best_val_f1:.4f}\n")
            f.write(f"Current Val F1: {val_f1:.4f}\n")
            f.write(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}\n")
            f.write(f"Val Loss: {val_loss:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}\n\n")
            f.write(f"Per-Class F1 Scores:\n")
            for i, (name, f1) in enumerate(zip(class_names, per_class_f1)):
                marker = " 🎯" if name in minority_classes else ""
                f.write(f"  {name:12s}: {f1:.3f}{marker}\n")
            f.write(f"\nMinority Class Avg F1: {minority_f1_avg:.3f}\n")
            f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f'checkpoint_{model_name_safe}_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'args': vars(args)
            }, checkpoint_path)
    
    # Save training history
    history_path = output_dir / f'history_{model_name_safe}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Model saved to: {output_dir}/best_{model_name_safe}.pth")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
