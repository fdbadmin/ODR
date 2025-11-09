"""
RETFound Model Integration for Retinal Disease Classification
Uses retinal-specific foundation model trained on 1.6M fundus images
Expected improvement: +5-10% F1 over ImageNet pre-training
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class RETFoundModel(nn.Module):
    """
    RETFound foundation model adapted for multi-label classification.
    
    Architecture:
    - Backbone: ViT-Base/16 (86M parameters)
    - Pre-training: Self-supervised MAE on 1.6M retinal images
    - Fine-tuning: Classification head for 7 diseases
    
    Reference: https://github.com/rmaphoh/RETFound_MAE
    Paper: "A Foundation Model for Generalizable Disease Detection from Retinal Images"
    """
    
    def __init__(self, num_classes=7, pretrained_path=None, img_size=384):
        super(RETFoundModel, self).__init__()
        
        # Load ViT-Base architecture (matches RETFound)
        # Note: RETFound uses patch_size=16, so we use vit_base_patch16_384
        self.backbone = timm.create_model(
            'vit_base_patch16_384',
            pretrained=False,  # We'll load RETFound weights separately
            num_classes=0,     # Remove classification head
            img_size=img_size,
            drop_path_rate=0.1,  # Stochastic depth
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features  # 768 for ViT-Base
        
        # Custom classification head for multi-label
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Load RETFound pre-trained weights if available
        if pretrained_path and Path(pretrained_path).exists():
            self._load_retfound_weights(pretrained_path)
            print(f"✅ Loaded RETFound weights from: {pretrained_path}")
        else:
            print("⚠️  RETFound weights not found. Using random initialization.")
            print("   Download from: https://github.com/rmaphoh/RETFound_MAE/releases")
            print("   Or using ImageNet pre-training as fallback...")
            # Fallback to ImageNet pre-training
            self.backbone = timm.create_model(
                'vit_base_patch16_384',
                pretrained=True,
                num_classes=0,
                img_size=img_size,
            )
    
    def _load_retfound_weights(self, checkpoint_path):
        """Load RETFound pre-trained weights"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # RETFound checkpoint structure
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'backbone.' prefix if present
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            
            # Load weights (strict=False allows classifier to be randomly initialized)
            msg = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"   Loaded backbone weights. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
            
        except Exception as e:
            print(f"⚠️  Error loading RETFound weights: {e}")
            print("   Falling back to ImageNet pre-training...")
            self.backbone = timm.create_model(
                'vit_base_patch16_384',
                pretrained=True,
                num_classes=0,
            )
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def get_attention_maps(self, x):
        """Get attention maps for visualization (optional)"""
        # Useful for explainability
        self.backbone.eval()
        with torch.no_grad():
            # Forward through backbone
            features = self.backbone.forward_features(x)
            
            # Get attention from last layer
            if hasattr(self.backbone, 'blocks'):
                last_block = self.backbone.blocks[-1]
                if hasattr(last_block, 'attn'):
                    attn = last_block.attn.get_attention_map()
                    return attn
        
        return None


def create_retfound_model(num_classes=7, pretrained_path=None, device='mps'):
    """
    Factory function to create RETFound model
    
    Args:
        num_classes: Number of disease classes (default: 7 for ODIR-5K)
        pretrained_path: Path to RETFound checkpoint (.pth file)
        device: Device to load model on
    
    Returns:
        model: RETFoundModel instance
    
    Example usage:
        # Download RETFound weights first:
        # wget https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_oct_weights.pth
        
        model = create_retfound_model(
            pretrained_path='models/RETFound_oct_weights.pth',
            device='mps'
        )
    """
    model = RETFoundModel(
        num_classes=num_classes,
        pretrained_path=pretrained_path,
        img_size=384
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 RETFound Model Summary:")
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"   Backbone: ViT-Base/16 (86M params)")
    print(f"   Classifier: {num_classes}-class multi-label head")
    print(f"   Device: {device}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing RETFound model creation...")
    
    # Check if weights exist
    possible_paths = [
        'models/RETFound_oct_weights.pth',
        'models/RETFound_cfp_weights.pth',
        '../models/RETFound_oct_weights.pth',
    ]
    
    pretrained_path = None
    for path in possible_paths:
        if Path(path).exists():
            pretrained_path = path
            break
    
    if pretrained_path:
        print(f"Found RETFound weights: {pretrained_path}")
    else:
        print("RETFound weights not found. Download from:")
        print("https://github.com/rmaphoh/RETFound_MAE/releases")
    
    # Create model
    model = create_retfound_model(
        num_classes=7,
        pretrained_path=pretrained_path,
        device='cpu'  # Use CPU for testing
    )
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 384, 384)  # Batch of 2 images
    with torch.no_grad():
        out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # Should be [2, 7]
    print("✅ Model test passed!")
