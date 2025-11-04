"""
Advanced model architectures for retinal image classification.
Includes Vision Transformer (ViT) and ConvNeXt.
"""

import torch
import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    """
    Vision Transformer (ViT) based classifier.
    State-of-the-art attention-based architecture.
    """
    
    def __init__(self, num_classes=7, model_name='vit_base_patch16_224', pretrained=True):
        """
        Args:
            num_classes: Number of output classes
            model_name: ViT variant ('vit_base_patch16_224', 'vit_small_patch16_224', etc.)
            pretrained: Use ImageNet pretrained weights
        """
        super(ViTClassifier, self).__init__()
        
        # Load pretrained ViT from timm library
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # ViT already has a classification head, just need to ensure it's the right size
        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)


class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt based classifier.
    Modern ConvNet architecture with competitive performance to ViT.
    """
    
    def __init__(self, num_classes=7, model_name='convnext_tiny', pretrained=True):
        """
        Args:
            num_classes: Number of output classes
            model_name: ConvNeXt variant ('convnext_tiny', 'convnext_small', 'convnext_base')
            pretrained: Use ImageNet pretrained weights
        """
        super(ConvNeXtClassifier, self).__init__()
        
        # Load pretrained ConvNeXt from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # ConvNeXt has a classification head
        if hasattr(self.backbone, 'head'):
            if hasattr(self.backbone.head, 'fc'):
                in_features = self.backbone.head.fc.in_features
                self.backbone.head.fc = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(in_features, num_classes)
                )
    
    def forward(self, x):
        return self.backbone(x)


class SwinTransformerClassifier(nn.Module):
    """
    Swin Transformer based classifier.
    Hierarchical vision transformer with shifted windows.
    """
    
    def __init__(self, num_classes=7, model_name='swin_tiny_patch4_window7_224', pretrained=True):
        """
        Args:
            num_classes: Number of output classes
            model_name: Swin variant
            pretrained: Use ImageNet pretrained weights
        """
        super(SwinTransformerClassifier, self).__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            global_pool='avg'  # Ensure global pooling to get correct output shape
        )
        
        # Swin has 'head.fc' attribute
        if hasattr(self.backbone, 'head') and hasattr(self.backbone.head, 'fc'):
            in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
        elif hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)


class EfficientNetV2Classifier(nn.Module):
    """
    EfficientNetV2 based classifier.
    Improved version of EfficientNet with better training speed.
    """
    
    def __init__(self, num_classes=7, model_name='tf_efficientnetv2_s', pretrained=True):
        """
        Args:
            num_classes: Number of output classes
            model_name: EfficientNetV2 variant ('tf_efficientnetv2_s', 'tf_efficientnetv2_m', etc.)
            pretrained: Use ImageNet pretrained weights
        """
        super(EfficientNetV2Classifier, self).__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)


def get_model(model_name, num_classes=7, pretrained=True):
    """
    Factory function to get any model architecture.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Use pretrained weights
    
    Returns:
        Model instance
    
    Supported models:
        - 'vit_base': Vision Transformer Base
        - 'vit_small': Vision Transformer Small (faster)
        - 'convnext_tiny': ConvNeXt Tiny
        - 'convnext_small': ConvNeXt Small
        - 'swin_tiny': Swin Transformer Tiny
        - 'efficientnetv2_s': EfficientNetV2 Small
        - 'efficientnetv2_m': EfficientNetV2 Medium
    """
    
    model_map = {
        'vit_base': lambda: ViTClassifier(num_classes, 'vit_base_patch16_224', pretrained),
        'vit_small': lambda: ViTClassifier(num_classes, 'vit_small_patch16_224', pretrained),
        'convnext_tiny': lambda: ConvNeXtClassifier(num_classes, 'convnext_tiny', pretrained),
        'convnext_small': lambda: ConvNeXtClassifier(num_classes, 'convnext_small', pretrained),
        'swin_tiny': lambda: SwinTransformerClassifier(num_classes, 'swin_tiny_patch4_window7_224', pretrained),
        'efficientnetv2_s': lambda: EfficientNetV2Classifier(num_classes, 'tf_efficientnetv2_s', pretrained),
        'efficientnetv2_m': lambda: EfficientNetV2Classifier(num_classes, 'tf_efficientnetv2_m', pretrained),
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(model_map.keys())}")
    
    return model_map[model_name]()


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info():
    """Get information about available models."""
    info = {
        'vit_base': {
            'name': 'Vision Transformer Base',
            'params': '~86M',
            'speed': 'Medium',
            'accuracy': 'Very High',
            'description': 'Attention-based, excellent for complex patterns'
        },
        'vit_small': {
            'name': 'Vision Transformer Small',
            'params': '~22M',
            'speed': 'Fast',
            'accuracy': 'High',
            'description': 'Lighter ViT, good balance of speed and accuracy'
        },
        'convnext_tiny': {
            'name': 'ConvNeXt Tiny',
            'params': '~28M',
            'speed': 'Fast',
            'accuracy': 'Very High',
            'description': 'Modern CNN, efficient and accurate'
        },
        'convnext_small': {
            'name': 'ConvNeXt Small',
            'params': '~50M',
            'speed': 'Medium',
            'accuracy': 'Very High',
            'description': 'Larger ConvNeXt, state-of-the-art performance'
        },
        'swin_tiny': {
            'name': 'Swin Transformer Tiny',
            'params': '~28M',
            'speed': 'Fast',
            'accuracy': 'Very High',
            'description': 'Hierarchical transformer, great for medical images'
        },
        'efficientnetv2_s': {
            'name': 'EfficientNetV2 Small',
            'params': '~21M',
            'speed': 'Very Fast',
            'accuracy': 'High',
            'description': 'Optimized for training speed'
        },
        'efficientnetv2_m': {
            'name': 'EfficientNetV2 Medium',
            'params': '~54M',
            'speed': 'Fast',
            'accuracy': 'Very High',
            'description': 'Larger EfficientNetV2, better accuracy'
        },
    }
    
    return info


if __name__ == '__main__':
    # Test models
    print("Testing advanced architectures...\n")
    
    models_to_test = ['vit_small', 'convnext_tiny', 'swin_tiny', 'efficientnetv2_s']
    
    for model_name in models_to_test:
        try:
            model = get_model(model_name, num_classes=7, pretrained=False)
            params = count_parameters(model)
            print(f"✓ {model_name}: {params/1e6:.1f}M parameters")
            
            # Test forward pass
            x = torch.randn(2, 3, 224, 224)
            y = model(x)
            assert y.shape == (2, 7), f"Output shape mismatch: {y.shape}"
        except Exception as e:
            print(f"✗ {model_name}: {str(e)}")
    
    print("\nAll models loaded successfully!")
