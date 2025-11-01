"""
PyTorch training script for ODIR-5K multi-label classification.
Optimized for Apple Silicon (M5) with MPS acceleration.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import time
from tqdm import tqdm
import json

from config import LABEL_COLUMNS, RANDOM_SEED

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class ODIRDataset(Dataset):
    """PyTorch Dataset for ODIR-5K preprocessed data."""
    
    def __init__(self, images_path: str, labels_path: str, transform=None):
        """
        Args:
            images_path: Path to .npy file containing images.
            labels_path: Path to .npy file containing labels.
            transform: Optional transform to apply to images.
        """
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        self.transform = transform
        
        print(f"Loaded dataset: {len(self.images)} samples")
        print(f"  Image shape: {self.images.shape}")
        print(f"  Label shape: {self.labels.shape}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor and rearrange from HWC to CHW
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.from_numpy(label).float()
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MultiLabelClassifier(nn.Module):
    """ResNet-based multi-label classifier."""
    
    def __init__(self, num_classes: int = 8, backbone: str = 'resnet50', pretrained: bool = True):
        """
        Args:
            num_classes: Number of classes (8 for ODIR-5K).
            backbone: Backbone architecture ('resnet50', 'resnet34', 'resnet18').
            pretrained: Whether to use pretrained weights.
        """
        super(MultiLabelClassifier, self).__init__()
        
        # Load backbone
        if backbone == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
        elif backbone == 'resnet34':
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet34(weights=weights)
            num_features = self.backbone.fc.in_features
        elif backbone == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet18(weights=weights)
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


def get_device():
    """Get the best available device (MPS for M5, CUDA for GPU, or CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using MPS (Metal Performance Shaders) - Apple Silicon acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU acceleration!")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (consider using MPS or CUDA for faster training)")
    return device


def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.numel()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / total_predictions
    
    return epoch_loss, epoch_acc


def validate_epoch(model: nn.Module, 
                   dataloader: DataLoader, 
                   criterion: nn.Module,
                   device: torch.device,
                   epoch: int) -> Tuple[float, float]:
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.numel()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / total_predictions
    
    return epoch_loss, epoch_acc


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler._LRScheduler,
                device: torch.device,
                num_epochs: int = 50,
                save_dir: str = 'models') -> Dict:
    """
    Train the model and save checkpoints.
    
    Returns:
        Dictionary containing training history.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    start_time = time.time()
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            }, save_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})")
    print("="*70)
    
    # Save final model and history
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, save_dir / 'final_model.pth')
    
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


def main():
    """Main training function."""
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 4  # M5 can handle parallel data loading
    
    print("="*70)
    print("ODIR-5K MULTI-LABEL CLASSIFICATION TRAINING")
    print("="*70)
    print(f"\nHyperparameters:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Num workers: {NUM_WORKERS}")
    
    # Get device
    device = get_device()
    
    # Load datasets
    print("\n📂 Loading preprocessed data...")
    train_dataset = ODIRDataset(
        'preprocessed_data/train_images.npy',
        'preprocessed_data/train_labels.npy'
    )
    val_dataset = ODIRDataset(
        'preprocessed_data/val_images.npy',
        'preprocessed_data/val_labels.npy'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create model
    print("\n🏗️  Building model...")
    model = MultiLabelClassifier(
        num_classes=len(LABEL_COLUMNS),
        backbone='resnet50',
        pretrained=True
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Loss function (Binary Cross Entropy for multi-label)
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6
    )
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS
    )
    
    print("\n✅ Training complete! Model saved in 'models/' directory.")
    print("\nNext steps:")
    print("  1. Evaluate model on test set")
    print("  2. Generate predictions and visualizations")
    print("  3. Analyze per-class performance metrics")


if __name__ == "__main__":
    main()
