"""
Test-Time Augmentation (TTA) for improved inference accuracy.
Apply multiple augmentations at test time and average predictions.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Optional
import torchvision.transforms as T


class TTAPredictor:
    """
    Test-Time Augmentation predictor.
    Applies multiple augmentations and averages predictions.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, num_augmentations: int = 8):
        """
        Args:
            model: Trained PyTorch model
            device: torch.device for computation
            num_augmentations: Number of augmentations to apply (default: 8)
        """
        self.model = model
        self.device = device
        self.num_augmentations = num_augmentations
        self.model.eval()
        self.augmentations = self.get_augmentations()
        
    def get_augmentations(self) -> List[Callable]:
        """
        Define TTA augmentations for retinal images.
        Returns list of augmentation functions.
        """
        augmentations = [
            lambda x: x,  # Original (no augmentation)
            lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
            lambda x: torch.flip(x, dims=[2]),  # Vertical flip
            lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # Rotate 90°
            lambda x: torch.rot90(x, k=2, dims=[2, 3]),  # Rotate 180°
            lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # Rotate 270°
            lambda x: x * 0.9,  # Brightness down
            lambda x: torch.clamp(x * 1.1, 0, 1),  # Brightness up
        ]
        
        return augmentations[:self.num_augmentations]
    
    def predict_single(self, image: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make prediction on a single image using all augmentations.
        
        Args:
            image: Input image tensor [C, H, W]
            metadata: Optional metadata tensor [metadata_dim]
            
        Returns:
            Average probabilities across all augmentations [num_classes]
        """
        all_probs = []
        
        # Add batch dimension and move to device
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        
        if metadata is not None:
            if metadata.dim() == 1:
                metadata = metadata.unsqueeze(0)
            metadata = metadata.to(self.device)
        
        with torch.no_grad():
            for aug_fn in self.augmentations:
                # Apply augmentation
                aug_image = aug_fn(image)
                
                # Get model prediction
                if metadata is not None:
                    logits = self.model(aug_image, metadata)
                else:
                    logits = self.model(aug_image)
                
                # Get probabilities
                probs = torch.sigmoid(logits)
                all_probs.append(probs)
        
        # Average probabilities across all augmentations
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs.squeeze(0)  # Remove batch dimension
    
    def predict_batch(self, images: torch.Tensor, metadata: torch.Tensor = None,
                     batch_size: int = 32) -> np.ndarray:
        """
        Predict with TTA for a batch of images.
        
        Args:
            images: (N, 3, 224, 224) tensor
            metadata: Optional (N, 2) tensor for metadata models
            batch_size: Batch size for processing
            
        Returns:
            (N, num_classes) numpy array of probabilities
        """
        all_predictions = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size] if metadata is not None else None
            
            batch_preds = []
            for j in range(len(batch_images)):
                img = batch_images[j]
                meta = batch_metadata[j] if batch_metadata is not None else None
                pred = self.predict_single(img, meta)
                batch_preds.append(pred.cpu().numpy())
            
            all_predictions.extend(batch_preds)
        
        return np.array(all_predictions)


def evaluate_with_tta(model, dataloader, device, use_metadata=False, threshold=0.5):
    """
    Evaluate model with Test-Time Augmentation.
    
    Args:
        model: Trained model
        dataloader: DataLoader for validation/test data
        device: torch.device
        use_metadata: Whether model uses metadata
        threshold: Classification threshold
        
    Returns:
        Dictionary with predictions, probabilities, and labels
    """
    tta_predictor = TTAPredictor(model, device, num_augmentations=8)
    
    all_probs = []
    all_labels = []
    
    print("Evaluating with Test-Time Augmentation...")
    
    for batch_data in dataloader:
        if use_metadata:
            images, labels, metadata = batch_data
            metadata = metadata.to(device)
        else:
            images, labels = batch_data
            metadata = None
        
        images = images.to(device)
        labels = labels.numpy()
        
        # TTA prediction for batch
        batch_probs = tta_predictor.predict_batch(images, metadata)
        
        all_probs.append(batch_probs)
        all_labels.append(labels)
    
    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_labels)
    y_pred = (y_probs > threshold).astype(float)
    
    return {
        'predictions': y_pred,
        'probabilities': y_probs,
        'labels': y_true
    }


if __name__ == "__main__":
    # Example usage
    print("Test-Time Augmentation module loaded successfully!")
    print("\nUsage example:")
    print("  from src.tta_inference import TTAPredictor, evaluate_with_tta")
    print("  tta_predictor = TTAPredictor(model, device)")
    print("  results = evaluate_with_tta(model, val_loader, device)")
