"""
Fine-tune the ODIR model with additional glaucoma images
This script adds external glaucoma data to improve model performance on glaucoma detection
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

from train import MultiLabelClassifier, get_device
from config import DISEASE_LABELS, DEFAULT_IMAGE_SIZE

# Paths
EXTERNAL_GLAUCOMA_DIR = Path("Eye Disease Retinal Images/glaucoma")
ODIR_TRAIN_IMAGES = Path("preprocessed_data/train_images.npy")
ODIR_TRAIN_LABELS = Path("preprocessed_data/train_labels.npy")
ODIR_VAL_IMAGES = Path("preprocessed_data/val_images.npy")
ODIR_VAL_LABELS = Path("preprocessed_data/val_labels.npy")

# Output paths
OUTPUT_DIR = Path("models")
AUGMENTED_MODEL_PATH = OUTPUT_DIR / "glaucoma_enhanced_model.pth"

def preprocess_image(image_path):
    """
    Preprocess external image to match ODIR preprocessing
    Same pipeline as data_preprocessing.py
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Resize
    img = cv2.resize(img, DEFAULT_IMAGE_SIZE)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def load_external_glaucoma_data():
    """
    Load and preprocess external glaucoma images
    """
    print("Loading external glaucoma images...")
    
    images = []
    valid_paths = []
    
    image_files = list(EXTERNAL_GLAUCOMA_DIR.glob("*.jpg"))
    
    for img_path in tqdm(image_files, desc="Preprocessing glaucoma images"):
        img = preprocess_image(img_path)
        if img is not None:
            images.append(img)
            valid_paths.append(img_path)
    
    images = np.array(images)
    
    # Create labels: [N, D, G, C, A, H, M, O]
    # Glaucoma = index 2 (G)
    # All images are glaucoma-positive
    labels = np.zeros((len(images), 8), dtype=np.float32)
    labels[:, 2] = 1.0  # Set Glaucoma to 1
    
    print(f"Loaded {len(images)} glaucoma images")
    print(f"Image shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
    
    return images, labels

def combine_datasets():
    """
    Combine ODIR dataset with external glaucoma data
    """
    print("\nLoading original ODIR datasets...")
    
    # Load original ODIR data
    odir_train_images = np.load(ODIR_TRAIN_IMAGES)
    odir_train_labels = np.load(ODIR_TRAIN_LABELS)
    odir_val_images = np.load(ODIR_VAL_IMAGES)
    odir_val_labels = np.load(ODIR_VAL_LABELS)
    
    print(f"ODIR train: {odir_train_images.shape[0]} images")
    print(f"ODIR val: {odir_val_images.shape[0]} images")
    
    # Load external glaucoma data
    glaucoma_images, glaucoma_labels = load_external_glaucoma_data()
    
    # Split glaucoma data into train/val (80/20)
    glauc_train_img, glauc_val_img, glauc_train_lbl, glauc_val_lbl = train_test_split(
        glaucoma_images, glaucoma_labels, test_size=0.2, random_state=42
    )
    
    print(f"\nGlaucoma split:")
    print(f"  Train: {len(glauc_train_img)} images")
    print(f"  Val: {len(glauc_val_img)} images")
    
    # Combine datasets
    combined_train_images = np.concatenate([odir_train_images, glauc_train_img], axis=0)
    combined_train_labels = np.concatenate([odir_train_labels, glauc_train_lbl], axis=0)
    combined_val_images = np.concatenate([odir_val_images, glauc_val_img], axis=0)
    combined_val_labels = np.concatenate([odir_val_labels, glauc_val_lbl], axis=0)
    
    print(f"\nCombined datasets:")
    print(f"  Train: {combined_train_images.shape[0]} images (+{len(glauc_train_img)})")
    print(f"  Val: {combined_val_images.shape[0]} images (+{len(glauc_val_img)})")
    
    # Analyze glaucoma distribution
    odir_glaucoma_train = odir_train_labels[:, 2].sum()
    odir_glaucoma_val = odir_val_labels[:, 2].sum()
    total_glaucoma_train = combined_train_labels[:, 2].sum()
    total_glaucoma_val = combined_val_labels[:, 2].sum()
    
    print(f"\nGlaucoma distribution:")
    print(f"  ODIR train: {odir_glaucoma_train:.0f} ({odir_glaucoma_train/len(odir_train_labels)*100:.1f}%)")
    print(f"  ODIR val: {odir_glaucoma_val:.0f} ({odir_glaucoma_val/len(odir_val_labels)*100:.1f}%)")
    print(f"  Enhanced train: {total_glaucoma_train:.0f} ({total_glaucoma_train/len(combined_train_labels)*100:.1f}%)")
    print(f"  Enhanced val: {total_glaucoma_val:.0f} ({total_glaucoma_val/len(combined_val_labels)*100:.1f}%)")
    
    return combined_train_images, combined_train_labels, combined_val_images, combined_val_labels

class AugmentedDataset(Dataset):
    """Dataset for combined ODIR + external glaucoma data"""
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Convert HWC to CHW for PyTorch
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1)
        label = torch.from_numpy(self.labels[idx])
        return image, label

def fine_tune_model(train_images, train_labels, val_images, val_labels, 
                     base_model_path="models/best_model.pth", epochs=20):
    """
    Fine-tune existing model with augmented dataset
    """
    print(f"\n{'='*70}")
    print("FINE-TUNING MODEL WITH GLAUCOMA-ENHANCED DATASET")
    print(f"{'='*70}\n")
    
    device = get_device()
    
    # Load pre-trained model
    print(f"Loading base model from {base_model_path}...")
    model = MultiLabelClassifier(num_classes=8)
    checkpoint = torch.load(base_model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Create datasets
    train_dataset = AugmentedDataset(train_images, train_labels)
    val_dataset = AugmentedDataset(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Training setup
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)  # Lower LR for fine-tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }
    
    print(f"\nStarting fine-tuning for {epochs} epochs...")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.numel()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.numel()
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            }, AUGMENTED_MODEL_PATH)
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = OUTPUT_DIR / f"glaucoma_enhanced_checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            }, checkpoint_path)
    
    print(f"\n{'='*70}")
    print(f"Fine-tuning complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {AUGMENTED_MODEL_PATH}")
    print(f"{'='*70}\n")
    
    # Save history
    history_path = OUTPUT_DIR / "glaucoma_enhanced_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    return model, history

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("GLAUCOMA-ENHANCED MODEL TRAINING")
    print("="*70 + "\n")
    
    # Step 1: Combine datasets
    train_images, train_labels, val_images, val_labels = combine_datasets()
    
    # Step 2: Fine-tune model
    model, history = fine_tune_model(
        train_images, train_labels, 
        val_images, val_labels,
        base_model_path="models/best_model.pth",
        epochs=20  # Fine-tuning typically needs fewer epochs
    )
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. Test the enhanced model:")
    print("   python predict.py path/to/image.jpg --model models/glaucoma_enhanced_model.pth")
    print("\n2. Compare performance:")
    print("   - Original model: models/best_model.pth")
    print("   - Enhanced model: models/glaucoma_enhanced_model.pth")
    print("\n3. Update API to use enhanced model:")
    print("   Edit api.py line 84: model_path = Path('models/glaucoma_enhanced_model.pth')")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
