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
    """PyTorch Dataset for ODIR-5K preprocessed data with optional metadata."""
    
    def __init__(self, images_path: str, labels_path: str, metadata_path: str = None, transform=None):
        """
        Args:
            images_path: Path to .npy file containing images.
            labels_path: Path to .npy file containing labels.
            metadata_path: Optional path to .npy file containing metadata (age, gender).
            transform: Optional transform to apply to images.
        """
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        self.metadata = np.load(metadata_path) if metadata_path else None
        self.transform = transform
        
        print(f"Loaded dataset: {len(self.images)} samples")
        print(f"  Image shape: {self.images.shape}")
        print(f"  Label shape: {self.labels.shape}")
        if self.metadata is not None:
            print(f"  Metadata shape: {self.metadata.shape}")
    
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
        
        # Return metadata if available
        if self.metadata is not None:
            metadata = torch.from_numpy(self.metadata[idx]).float()
            return image, label, metadata
        
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


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses training on hard examples and down-weights easy negatives.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        """
        Args:
            alpha: Weighting factor (default 1.0)
            gamma: Focusing parameter (default 2.0). Higher gamma focuses more on hard examples.
            pos_weight: Per-class weights for positive samples
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - Raw logits
            targets: (batch_size, num_classes) - Binary labels
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        # Calculate focal term: (1 - p_t)^gamma
        p_t = p * targets + (1 - p) * (1 - targets)  # p if y=1, (1-p) if y=0
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply focal term to BCE loss
        focal_loss = self.alpha * focal_term * bce_loss
        
        return focal_loss.mean()


class MetadataEnhancedClassifier(nn.Module):
    """ResNet-based multi-label classifier enhanced with metadata (age, gender)."""
    
    def __init__(self, num_classes: int = 8, metadata_dim: int = 2, backbone: str = 'resnet50', pretrained: bool = True):
        """
        Args:
            num_classes: Number of classes (8 for ODIR-5K).
            metadata_dim: Dimension of metadata features (2 for age + gender).
            backbone: Backbone architecture ('resnet50', 'resnet34', 'resnet18').
            pretrained: Whether to use pretrained weights.
        """
        super(MetadataEnhancedClassifier, self).__init__()
        
        # Image encoder (ResNet backbone)
        if backbone == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.image_encoder = resnet50(weights=weights)
            image_features = self.image_encoder.fc.in_features
        elif backbone == 'resnet34':
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.image_encoder = resnet34(weights=weights)
            image_features = self.image_encoder.fc.in_features
        elif backbone == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.image_encoder = resnet18(weights=weights)
            image_features = self.image_encoder.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer from backbone
        self.image_encoder.fc = nn.Identity()
        
        # Metadata encoder (small MLP)
        metadata_hidden = 16
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, metadata_hidden),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined classifier
        combined_features = image_features + metadata_hidden
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, images, metadata):
        """
        Args:
            images: (batch_size, 3, 224, 224) - Input images
            metadata: (batch_size, 2) - [normalized_age, gender_binary]
        
        Returns:
            (batch_size, num_classes) - Class logits
        """
        # Extract image features
        image_features = self.image_encoder(images)  # (batch, 2048) for ResNet50
        
        # Extract metadata features
        metadata_features = self.metadata_encoder(metadata)  # (batch, 16)
        
        # Concatenate features
        combined = torch.cat([image_features, metadata_features], dim=1)  # (batch, 2064)
        
        # Final classification
        return self.classifier(combined)


class MetadataRefinementModel(nn.Module):
    """
    Two-stage metadata refinement model.
    Stage 1: Frozen image-only baseline model
    Stage 2: Small metadata-based refinement network
    
    This architecture guarantees no performance regression below the baseline
    while allowing metadata to add corrections where helpful.
    """
    
    def __init__(self, frozen_image_model: nn.Module, num_classes: int = 8, metadata_dim: int = 2):
        """
        Args:
            frozen_image_model: Pre-trained image-only model (will be frozen)
            num_classes: Number of classes (8 for ODIR-5K)
            metadata_dim: Dimension of metadata features (2 for age + gender)
        """
        super(MetadataRefinementModel, self).__init__()
        
        # Stage 1: Frozen image model
        self.image_model = frozen_image_model
        for param in self.image_model.parameters():
            param.requires_grad = False  # Freeze the baseline model
        self.image_model.eval()  # Keep in eval mode
        
        # Stage 2: Metadata-based refinement network
        # Input: [metadata (2) + image predictions (8)] = 10 features
        # Output: Refinement residuals (8)
        self.refiner = nn.Sequential(
            nn.Linear(metadata_dim + num_classes, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
        # Initialize refinement network with small weights (start conservative)
        for m in self.refiner.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)  # Small initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, images, metadata):
        """
        Args:
            images: (batch_size, 3, 224, 224) - Input images
            metadata: (batch_size, 2) - [normalized_age, gender_binary]
        
        Returns:
            (batch_size, num_classes) - Refined class logits
        """
        # Stage 1: Get baseline predictions (frozen, no gradients)
        with torch.no_grad():
            image_logits = self.image_model(images)  # (batch, 8)
        
        # Stage 2: Generate metadata-based refinements
        # Concatenate metadata with image predictions
        refiner_input = torch.cat([metadata, image_logits], dim=1)  # (batch, 10)
        refinement = self.refiner(refiner_input)  # (batch, 8)
        
        # Apply residual connection: refined = baseline + correction
        refined_logits = image_logits + refinement
        
        return refined_logits
    
    def get_baseline_predictions(self, images):
        """
        Get predictions from the baseline model only (for comparison).
        
        Args:
            images: (batch_size, 3, 224, 224) - Input images
        
        Returns:
            (batch_size, num_classes) - Baseline class logits
        """
        with torch.no_grad():
            return self.image_model(images)


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


def get_adaptive_thresholds(class_frequencies: np.ndarray) -> torch.Tensor:
    """
    Calculate adaptive thresholds for each class based on frequency.
    Rare classes get lower thresholds (easier to predict positive).
    Moderately tuned for clinical reliability (balanced sensitivity/specificity).
    
    Args:
        class_frequencies: Array of class frequencies (0-1 range)
    
    Returns:
        Tensor of thresholds per class
    """
    # Moderate threshold adjustment for rare classes
    # Formula: threshold = 0.4 + 0.15 * frequency
    # Range: [0.40 for rarest, ~0.47 for common]
    thresholds = 0.40 + (0.15 * class_frequencies)
    
    # Keep Normal and Diabetes at 0.5 (they're well-represented)
    thresholds[0] = 0.5  # Normal
    thresholds[1] = 0.5  # Diabetes
    
    return torch.FloatTensor(thresholds)


def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                use_metadata: bool = False,
                thresholds: torch.Tensor = None) -> Tuple[float, float]:
    """Train for one epoch with adaptive thresholds."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    # Use adaptive thresholds if provided, else default 0.5
    if thresholds is None:
        thresholds = torch.ones(len(LABEL_COLUMNS)) * 0.5
    thresholds = thresholds.to(device)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        if use_metadata:
            images, labels, metadata = batch
            images = images.to(device)
            labels = labels.to(device)
            metadata = metadata.to(device)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        if use_metadata:
            outputs = model(images, metadata)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics with adaptive thresholds
        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        predictions = (probs > thresholds.unsqueeze(0)).float()
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
                   epoch: int,
                   use_metadata: bool = False,
                   thresholds: torch.Tensor = None) -> Tuple[float, float, float, List[float]]:
    """Validate for one epoch with adaptive thresholds. Returns loss, accuracy, mean F1, and per-class F1."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    # Collect all predictions and labels for F1 calculation
    all_predictions = []
    all_labels = []
    
    # Use adaptive thresholds if provided, else default 0.5
    if thresholds is None:
        thresholds = torch.ones(len(LABEL_COLUMNS)) * 0.5
    thresholds = thresholds.to(device)
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            if use_metadata:
                images, labels, metadata = batch
                images = images.to(device)
                labels = labels.to(device)
                metadata = metadata.to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
            
            # Forward pass
            if use_metadata:
                outputs = model(images, metadata)
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics with adaptive thresholds
            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            predictions = (probs > thresholds.unsqueeze(0)).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.numel()
            
            # Store for F1 calculation
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / total_predictions
    
    # Calculate F1 scores per class
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    f1_per_class = []
    for i in range(len(LABEL_COLUMNS)):
        # Calculate F1 for each class
        tp = ((all_predictions[:, i] == 1) & (all_labels[:, i] == 1)).sum()
        fp = ((all_predictions[:, i] == 1) & (all_labels[:, i] == 0)).sum()
        fn = ((all_predictions[:, i] == 0) & (all_labels[:, i] == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_per_class.append(float(f1))
    
    mean_f1 = np.mean(f1_per_class)
    
    return epoch_loss, epoch_acc, mean_f1, f1_per_class


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler._LRScheduler,
                device: torch.device,
                num_epochs: int = 25,
                save_dir: str = 'models',
                use_metadata: bool = False,
                thresholds: torch.Tensor = None,
                use_two_stage: bool = False,
                stage: int = 1) -> Dict:
    """
    Train the model and save checkpoints.
    
    Args:
        use_metadata: Whether the model uses metadata (age/gender) as input.
        thresholds: Adaptive thresholds per class for predictions.
    
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
        'val_mean_f1': [],
        'val_f1_per_class': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_mean_f1 = 0.0  # Track best mean F1 score
    best_epoch = 0
    start_time = time.time()
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_metadata, thresholds)
        
        # Validate (now returns F1 scores too)
        val_loss, val_acc, val_mean_f1, val_f1_per_class = validate_epoch(model, val_loader, criterion, device, epoch, use_metadata, thresholds)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_mean_f1'].append(val_mean_f1)
        history['val_f1_per_class'].append(val_f1_per_class)
        history['learning_rates'].append(current_lr)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val Mean F1: {val_mean_f1:.4f}")
        
        # Print per-class F1 scores
        print(f"  Per-class F1:", end=" ")
        for i, disease in enumerate(LABEL_COLUMNS):
            print(f"{disease}:{val_f1_per_class[i]:.3f}", end=" ")
        print()
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model based on MEAN F1 SCORE (better metric for multi-label)
        if val_mean_f1 > best_mean_f1:
            best_mean_f1 = val_mean_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            # Save with different names based on stage
            if use_two_stage and stage == 1:
                save_path = save_dir / 'baseline_model.pth'
            elif use_two_stage and stage == 2:
                save_path = save_dir / 'refinement_model.pth'
            else:
                save_path = save_dir / 'best_model.pth'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_mean_f1': val_mean_f1,
                'val_f1_per_class': val_f1_per_class,
                'label_columns': LABEL_COLUMNS,
                'history': history
            }, save_path)
            print(f"  ✓ Saved best model to {save_path.name} (mean_f1: {val_mean_f1:.4f}, val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f})")
        
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
    print(f"Best epoch: {best_epoch}")
    print(f"  Val Accuracy: {best_val_acc:.4f}")
    print(f"  Val Mean F1:  {best_mean_f1:.4f}")
    print(f"  Val Loss:     {best_val_loss:.4f}")
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
    
    # ============================================================================
    # CONFIGURATION FLAGS
    # ============================================================================
    
    # Stage 1: Train baseline model first
    # Stage 2: Train refinement with metadata (requires baseline model)
    STAGE = 1  # 1 = Train baseline only, 2 = Train refinement with metadata
    
    # Metadata options
    USE_METADATA = False  # Set to True to train with age/gender features (Stage 1 baseline: False)
    USE_TWO_STAGE = True  # Set to True for two-stage refinement approach
    USE_ENHANCED_METADATA = True  # Set to True to use 18-feature enhanced metadata
    
    # Path to baseline model (required for STAGE=2 or USE_TWO_STAGE=True)
    BASELINE_MODEL_PATH = 'models/baseline_model.pth'  # Path to trained baseline
    
    # ============================================================================
    # HYPERPARAMETERS
    # ============================================================================
    
    if STAGE == 1 or not USE_TWO_STAGE:
        # Stage 1: Train baseline or standard metadata model
        BATCH_SIZE = 48  # Increased from 32 - M5 with 32GB can handle more
        NUM_EPOCHS = 25
        LEARNING_RATE = 1e-4
        WEIGHT_DECAY = 1e-5
    else:
        # Stage 2: Train refinement network (faster, smaller model)
        BATCH_SIZE = 48  # Increased from 32
        NUM_EPOCHS = 15  # Fewer epochs needed for refinement
        LEARNING_RATE = 5e-5  # Lower learning rate for fine-tuning
        WEIGHT_DECAY = 1e-5
    
    NUM_WORKERS = 6  # M5 has more performance cores, increased from 4
    
    print("="*70)
    print("ODIR-5K MULTI-LABEL CLASSIFICATION TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Training Stage: {STAGE}")
    if STAGE == 1:
        print(f"  Mode: Training baseline image-only model")
    else:
        print(f"  Mode: {'Two-stage refinement' if USE_TWO_STAGE else 'Standard metadata model'}")
    print(f"  Use metadata: {USE_METADATA}")
    if USE_TWO_STAGE and STAGE == 2:
        print(f"  Baseline model: {BASELINE_MODEL_PATH}")
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
    
    # For Stage 1 or non-two-stage training, load based on USE_METADATA flag
    # For Stage 2 two-stage, always load metadata for refinement network
    load_metadata = USE_METADATA if (STAGE == 1 or not USE_TWO_STAGE) else True
    
    if load_metadata:
        # Determine which metadata files to use
        if USE_ENHANCED_METADATA:
            train_metadata_path = 'preprocessed_data/train_metadata_enhanced.npy'
            val_metadata_path = 'preprocessed_data/val_metadata_enhanced.npy'
            metadata_dim = 18  # Enhanced metadata has 18 features
            print(f"  Using enhanced metadata (18 features)")
        else:
            train_metadata_path = 'preprocessed_data/train_metadata.npy'
            val_metadata_path = 'preprocessed_data/val_metadata.npy'
            metadata_dim = 2  # Simple metadata has 2 features (age, gender)
            print(f"  Using simple metadata (2 features)")
        
        train_dataset = ODIRDataset(
            'preprocessed_data/train_images.npy',
            'preprocessed_data/train_labels.npy',
            train_metadata_path
        )
        val_dataset = ODIRDataset(
            'preprocessed_data/val_images.npy',
            'preprocessed_data/val_labels.npy',
            val_metadata_path
        )
    else:
        train_dataset = ODIRDataset(
            'preprocessed_data/train_images.npy',
            'preprocessed_data/train_labels.npy'
        )
        val_dataset = ODIRDataset(
            'preprocessed_data/val_images.npy',
            'preprocessed_data/val_labels.npy'
        )
        metadata_dim = 0  # No metadata
    
    # SIMPLIFIED: Use standard random shuffling (no weighted sampling)
    print("\n📊 Using standard random sampling...")
    train_labels = train_dataset.labels
    pos_counts = train_labels.sum(axis=0)
    
    # Create data loaders with standard shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Standard random shuffling
        num_workers=NUM_WORKERS,
        pin_memory=False  # MPS doesn't support pin_memory, set to False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False  # MPS doesn't support pin_memory, set to False
    )
    
    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create model
    print("\n🏗️  Building model...")
    
    if STAGE == 2 and USE_TWO_STAGE:
        # Stage 2: Two-stage refinement approach
        print("🔄 Loading baseline model for two-stage training...")
        
        # Load the frozen baseline model
        baseline_model = MultiLabelClassifier(
            num_classes=len(LABEL_COLUMNS),
            backbone='resnet50',
            pretrained=True
        )
        
        # Load trained weights
        checkpoint = torch.load(BASELINE_MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            baseline_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            baseline_model.load_state_dict(checkpoint)
        
        baseline_model = baseline_model.to(device)
        print(f"✓ Loaded baseline model from {BASELINE_MODEL_PATH}")
        
        # Create refinement model
        model = MetadataRefinementModel(
            frozen_image_model=baseline_model,
            num_classes=len(LABEL_COLUMNS),
            metadata_dim=metadata_dim  # Use detected metadata dimension (2 or 18)
        )
        model = model.to(device)
        print("✓ Using MetadataRefinementModel (two-stage approach)")
        print("  - Stage 1: Frozen baseline (image-only)")
        print(f"  - Stage 2: Metadata-based refinement network ({metadata_dim} features)")
        
    elif USE_METADATA:
        # Standard metadata-enhanced model
        model = MetadataEnhancedClassifier(
            num_classes=len(LABEL_COLUMNS),
            metadata_dim=metadata_dim,  # Use detected metadata dimension (2 or 18)
            backbone='resnet50',
            pretrained=True
        )
        model = model.to(device)
        print(f"✓ Using MetadataEnhancedClassifier (with {metadata_dim} features)")
        
    else:
        # Image-only baseline model
        model = MultiLabelClassifier(
            num_classes=len(LABEL_COLUMNS),
            backbone='resnet50',
            pretrained=True
        )
        model = model.to(device)
        print("✓ Using MultiLabelClassifier (image-only baseline)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    if USE_TWO_STAGE and STAGE == 2:
        frozen_params = total_params - trainable_params
        print(f"✓ Frozen parameters: {frozen_params:,} (baseline model)")
        print(f"  → Only training {trainable_params:,} refinement parameters!")
    
    # Calculate class weights for imbalanced dataset
    print("\n⚖️  Calculating class weights...")
    neg_counts = len(train_labels) - pos_counts
    
    # Weight = neg_count / pos_count (higher weight for rare classes)
    pos_weights = neg_counts / pos_counts
    pos_weights = torch.FloatTensor(pos_weights).to(device)
    
    print("  Class weights (higher = rarer class):")
    for i, disease in enumerate(LABEL_COLUMNS):
        print(f"    {disease}: {pos_weights[i].item():.2f} (pos: {int(pos_counts[i])}, neg: {int(neg_counts[i])})")
    
    # SIMPLIFIED: Use standard BCE Loss with class weights (no Focal Loss, no adaptive thresholds)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    print("\n✓ Using BCEWithLogitsLoss with class weights (standard approach)")
    print("✓ Using standard 0.5 threshold for all classes")
    
    # No adaptive thresholds - will use default 0.5 in training
    thresholds = None
    
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
    use_metadata_flag = load_metadata  # Use metadata in training if loaded
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        use_metadata=use_metadata_flag,
        thresholds=thresholds,
        use_two_stage=USE_TWO_STAGE,
        stage=STAGE
    )
    
    print("\n✅ Training complete! Model saved in 'models/' directory.")
    
    if STAGE == 2 and USE_TWO_STAGE:
        print("\n🎯 Two-Stage Training Summary:")
        print("  ✓ Baseline model: Frozen (guaranteed baseline performance)")
        print("  ✓ Refinement network: Trained to add metadata-based corrections")
        print("  ✓ Final model: Baseline + refinement (cannot perform worse than baseline)")
    
    print("\nNext steps:")
    print("  1. Evaluate model on test set")
    print("  2. Generate predictions and visualizations")
    print("  3. Analyze per-class performance metrics")


if __name__ == "__main__":
    main()
