"""
Improved Training Script with:
1. Optimal thresholds (applied during evaluation)
2. Class-weighted loss
3. Focal loss
4. Data balancing (oversampling minority classes)

Usage:
    python src/train_improved.py --epochs 15 --use-focal-loss --balance-data
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import MultiLabelClassifier
from src.focal_loss import FocalLoss, WeightedFocalLoss, compute_class_weights
from src.data_balancing import create_balanced_loader
from sklearn.metrics import f1_score, accuracy_score


LABEL_COLUMNS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
LABEL_NAMES = {
    'N': 'Normal', 'D': 'Diabetes', 'G': 'Glaucoma',
    'C': 'Cataract', 'A': 'AMD', 'H': 'Hypertension',
    'M': 'Myopia', 'O': 'Other'
}

# Optimal thresholds found from optimization
OPTIMAL_THRESHOLDS = [0.125, 0.175, 0.875, 0.775, 0.150, 0.750, 0.750, 0.275]


def load_data():
    """Load preprocessed data."""
    print("Loading data...")
    
    train_images = np.load('preprocessed_data_enhanced/train_images.npy')
    train_labels = np.load('preprocessed_data_enhanced/train_labels.npy')
    val_images = np.load('preprocessed_data_enhanced/val_images.npy')
    val_labels = np.load('preprocessed_data_enhanced/val_labels.npy')
    
    print(f"Train: {train_images.shape}, Val: {val_images.shape}")
    print(f"\nClass distribution (training):")
    for i, label in enumerate(LABEL_COLUMNS):
        count = int(train_labels[:, i].sum())
        pct = count / len(train_labels) * 100
        print(f"  {LABEL_NAMES[label]:<15} {count:>4} ({pct:>5.1f}%)")
    
    return train_images, train_labels, val_images, val_labels


def create_dataloaders(train_images, train_labels, val_images, val_labels, 
                       batch_size=32, balance_data=False, min_samples=150):
    """Create train and validation data loaders."""
    
    if balance_data:
        print(f"\n{'='*70}")
        print("BALANCING TRAINING DATA")
        print(f"{'='*70}")
        train_loader = create_balanced_loader(
            images=train_images,
            labels=train_labels,
            batch_size=batch_size,
            min_samples_per_class=min_samples
        )
    else:
        # Standard data loader without balancing
        train_images_tensor = torch.FloatTensor(train_images).permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        train_images_tensor = (train_images_tensor - mean) / std
        
        train_labels_tensor = torch.FloatTensor(train_labels)
        train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation loader (never balanced)
    val_images_tensor = torch.FloatTensor(val_images).permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    val_images_tensor = (val_images_tensor - mean) / std
    
    val_labels_tensor = torch.FloatTensor(val_labels)
    val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def evaluate_model(model, val_loader, device, use_optimal_thresholds=False):
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    # Apply thresholds
    if use_optimal_thresholds:
        all_preds = np.zeros_like(all_probs)
        for i in range(len(LABEL_COLUMNS)):
            all_preds[:, i] = (all_probs[:, i] > OPTIMAL_THRESHOLDS[i]).astype(float)
    else:
        all_preds = (all_probs > 0.5).astype(float)
    
    # Calculate metrics
    label_acc = np.mean(all_preds == all_labels)
    sample_acc = np.mean(np.all(all_preds == all_labels, axis=1))
    
    # Per-class F1 scores
    f1_scores = []
    for i in range(len(LABEL_COLUMNS)):
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        f1_scores.append(f1)
    
    mean_f1 = np.mean(f1_scores)
    
    return label_acc, sample_acc, mean_f1, f1_scores


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train with improved techniques')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--use-focal-loss', action='store_true', help='Use focal loss')
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal loss alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--balance-data', action='store_true', help='Balance training data')
    parser.add_argument('--min-samples', type=int, default=150, help='Min samples per class when balancing')
    parser.add_argument('--use-class-weights', action='store_true', help='Use class weights')
    parser.add_argument('--use-optimal-thresholds', action='store_true', help='Use optimal thresholds for evaluation')
    parser.add_argument('--model-save-path', type=str, default='models/improved_model.pth', help='Path to save model')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_images, train_labels, val_images, val_labels = load_data()
    
    # Compute class weights
    class_weights = None
    if args.use_class_weights:
        print(f"\n{'='*70}")
        print("COMPUTING CLASS WEIGHTS")
        print(f"{'='*70}")
        class_weights = compute_class_weights(train_labels, method='inverse_freq', power=1.0)
        print("Class weights:")
        for i, label in enumerate(LABEL_COLUMNS):
            print(f"  {LABEL_NAMES[label]:<15} {class_weights[i]:.3f}")
        class_weights = class_weights.to(device)
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_images, train_labels, val_images, val_labels,
        batch_size=args.batch_size,
        balance_data=args.balance_data,
        min_samples=args.min_samples
    )
    
    # Create model
    print(f"\n{'='*70}")
    print("INITIALIZING MODEL")
    print(f"{'='*70}")
    model = MultiLabelClassifier(num_classes=8)
    model = model.to(device)
    
    # Create loss function
    if args.use_focal_loss:
        if args.use_class_weights:
            criterion = WeightedFocalLoss(
                alpha=args.focal_alpha,
                gamma=args.focal_gamma,
                class_weights=class_weights
            )
            print(f"Using Weighted Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
        else:
            criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
            print(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    else:
        if args.use_class_weights:
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            print("Using BCE Loss with class weights")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("Using standard BCE Loss")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")
    
    best_f1 = 0
    best_epoch = 0
    history = {
        'train_loss': [],
        'val_label_acc': [],
        'val_sample_acc': [],
        'val_mean_f1': [],
        'val_f1_per_class': []
    }
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        
        # Evaluate
        val_label_acc, val_sample_acc, val_mean_f1, val_f1_scores = evaluate_model(
            model, val_loader, device, use_optimal_thresholds=args.use_optimal_thresholds
        )
        
        # Update scheduler
        scheduler.step(val_mean_f1)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_label_acc'].append(val_label_acc)
        history['val_sample_acc'].append(val_sample_acc)
        history['val_mean_f1'].append(val_mean_f1)
        history['val_f1_per_class'].append(val_f1_scores)
        
        # Print results
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss:      {train_loss:.4f}")
        print(f"  Val Label Acc:   {val_label_acc*100:.2f}%")
        print(f"  Val Sample Acc:  {val_sample_acc*100:.2f}%")
        print(f"  Val Mean F1:     {val_mean_f1:.4f}")
        print(f"  Per-class F1:")
        for i, label in enumerate(LABEL_COLUMNS):
            print(f"    {LABEL_NAMES[label]:<15} {val_f1_scores[i]:.4f}")
        
        # Save best model
        if val_mean_f1 > best_f1:
            best_f1 = val_mean_f1
            best_epoch = epoch
            
            os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_label_acc': val_label_acc,
                'val_sample_acc': val_sample_acc,
                'val_mean_f1': val_mean_f1,
                'val_f1_scores': val_f1_scores,
                'args': vars(args),
                'history': history
            }, args.model_save_path)
            
            print(f"  ✓ New best model saved! (F1={val_mean_f1:.4f})")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best mean F1: {best_f1:.4f}")
    print(f"Model saved to: {args.model_save_path}")
    
    # Save training history
    history_path = args.model_save_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {
            'train_loss': history['train_loss'],
            'val_label_acc': history['val_label_acc'],
            'val_sample_acc': history['val_sample_acc'],
            'val_mean_f1': history['val_mean_f1'],
            'val_f1_per_class': [scores.tolist() for scores in history['val_f1_per_class']]
        }
        json.dump(history_json, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
