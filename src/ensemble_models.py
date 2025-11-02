"""
Ensemble model training for ODIR-5K classification.
Trains multiple architectures and combines predictions.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict
import numpy as np


class EfficientNetB3Classifier(nn.Module):
    """EfficientNet-B3 based classifier."""
    
    def __init__(self, num_classes=8):
        super(EfficientNetB3Classifier, self).__init__()
        # Load pre-trained EfficientNet-B3
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class DenseNet121Classifier(nn.Module):
    """DenseNet-121 based classifier."""
    
    def __init__(self, num_classes=8):
        super(DenseNet121Classifier, self).__init__()
        # Load pre-trained DenseNet-121
        from torchvision.models import densenet121, DenseNet121_Weights
        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        
        # Replace classifier
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models with weighted averaging.
    """
    
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        """
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
        
        # Set all models to eval mode
        for model in self.models:
            model.eval()
    
    def forward(self, x):
        """
        Forward pass through all models and combine predictions.
        
        Args:
            x: Input tensor (batch_size, 3, 224, 224)
            
        Returns:
            Weighted average of logits from all models
        """
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Stack predictions
        stacked = torch.stack(predictions)  # (num_models, batch_size, num_classes)
        
        # Weighted average
        weights = self.weights.to(stacked.device).view(-1, 1, 1)
        weighted_pred = torch.sum(stacked * weights, dim=0)
        
        return weighted_pred


class EnsembleMetadataModel(nn.Module):
    """
    Ensemble with metadata refinement.
    Combines multiple image models + metadata refinement.
    """
    
    def __init__(self, image_models: List[nn.Module], metadata_dim: int = 18,
                 num_classes: int = 8, model_weights: List[float] = None):
        """
        Args:
            image_models: List of trained image-only models
            metadata_dim: Dimension of metadata features
            num_classes: Number of output classes
            model_weights: Optional weights for each image model
        """
        super(EnsembleMetadataModel, self).__init__()
        
        # Image ensemble
        self.image_ensemble = EnsembleModel(image_models, model_weights)
        
        # Freeze image models
        for param in self.image_ensemble.parameters():
            param.requires_grad = False
        
        # Metadata refinement network
        self.refiner = nn.Sequential(
            nn.Linear(num_classes + metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Initialize with small weights
        for m in self.refiner.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, images, metadata):
        """
        Forward pass combining ensemble predictions with metadata.
        
        Args:
            images: (batch_size, 3, 224, 224)
            metadata: (batch_size, metadata_dim)
            
        Returns:
            Refined predictions
        """
        # Get ensemble predictions
        ensemble_logits = self.image_ensemble(images)
        
        # Combine with metadata
        combined = torch.cat([ensemble_logits, metadata], dim=1)
        
        # Refine predictions
        refined_logits = self.refiner(combined)
        
        # Residual connection
        final_logits = ensemble_logits + 0.1 * refined_logits
        
        return final_logits


def create_ensemble_from_checkpoints(checkpoint_paths: List[str], 
                                    model_types: List[str],
                                    device: torch.device) -> EnsembleModel:
    """
    Create ensemble from saved model checkpoints.
    
    Args:
        checkpoint_paths: List of paths to model checkpoint files
        model_types: List of model type strings ('resnet50', 'efficientnet_b3', 'densenet121')
        device: torch.device for loading models
        
    Returns:
        EnsembleModel instance
    """
    from src.train import MultiLabelClassifier
    
    models = []
    
    for checkpoint_path, model_type in zip(checkpoint_paths, model_types):
        print(f"Loading {model_type} from {checkpoint_path}...")
        
        # Create model
        if model_type == 'resnet50':
            model = MultiLabelClassifier(num_classes=8)
        elif model_type == 'efficientnet_b3':
            model = EfficientNetB3Classifier(num_classes=8)
        elif model_type == 'densenet121':
            model = DenseNet121Classifier(num_classes=8)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        models.append(model)
        print(f"  ✓ Loaded (val_acc: {checkpoint['val_acc']:.4f})")
    
    # Create ensemble
    ensemble = EnsembleModel(models)
    ensemble.to(device)
    
    print(f"\n✓ Ensemble created with {len(models)} models")
    return ensemble


if __name__ == "__main__":
    print("Ensemble models module loaded successfully!")
    print("\nAvailable models:")
    print("  - EfficientNetB3Classifier")
    print("  - DenseNet121Classifier")
    print("  - EnsembleModel")
    print("  - EnsembleMetadataModel")
