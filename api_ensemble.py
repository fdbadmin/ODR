"""
Flask API for 3-Model Ensemble Predictions
==========================================

Serves predictions from the Phase 4C ensemble:
- ResNet-50 (CNN, 23.5M params)
- EfficientNet-B5 (CNN, 28.4M params)
- ViT-Base (Transformer, 86.1M params)

Performance: 72.26% Macro F1 with optimized thresholds
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import json
import timm
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Configuration
CLASS_NAMES = ['AMD', 'Diabetes', 'Glaucoma', 'Cataract', 'Myopia', 'Normal', 'Other']
NUM_CLASSES = 7

# Device configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("✅ Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("✅ Using CUDA GPU")
else:
    DEVICE = torch.device('cpu')
    print("⚠️  Using CPU")


class EnsembleModel:
    """3-Model Ensemble for fundus image classification"""
    
    def __init__(self, device='mps'):
        self.device = torch.device(device)
        self.models = {}
        self.thresholds = {}
        
        # Load ensemble thresholds
        with open('ensemble_optimized_thresholds_simple_average.json', 'r') as f:
            threshold_config = json.load(f)
            self.thresholds = {name: threshold_config['thresholds'][name] 
                             for name in CLASS_NAMES}
        
        print("\n📊 Loading 3-Model Ensemble...")
        print(f"   Optimized thresholds: {self.thresholds}")
        
        # Load models
        self._load_models()
        
    def _create_resnet50(self):
        """Create ResNet-50 model"""
        model = timm.create_model('resnet50', pretrained=False, num_classes=NUM_CLASSES)
        return model
    
    def _create_efficientnet_b5(self):
        """Create EfficientNet-B5 model"""
        model = timm.create_model('tf_efficientnet_b5', pretrained=False, num_classes=NUM_CLASSES)
        return model
    
    def _create_vit_base(self):
        """Create ViT-Base model"""
        model = timm.create_model(
            'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            pretrained=False,
            num_classes=NUM_CLASSES,
            img_size=384  # Our training resolution
        )
        return model
    
    def _load_models(self):
        """Load all three models"""
        model_configs = {
            'resnet50': {
                'path': 'models_smart_exclusion/best_resnet50.pth',
                'create_fn': self._create_resnet50
            },
            'efficientnet_b5': {
                'path': 'models_efficientnet_b5/best_efficientnet_b5.pth',
                'create_fn': self._create_efficientnet_b5
            },
            'vit_base': {
                'path': 'models_vit_base/best_vit_base.pth',
                'create_fn': self._create_vit_base
            }
        }
        
        for model_name, config in model_configs.items():
            print(f"\n   Loading {model_name}...")
            
            # Create model
            model = config['create_fn']()
            
            # Load weights
            checkpoint = torch.load(config['path'], map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            self.models[model_name] = model
            print(f"   ✅ {model_name} loaded")
        
        print(f"\n✅ All 3 models loaded successfully!")
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Resize to 384x384 (our training resolution)
        image = image.resize((384, 384), Image.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Convert to CHW format and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()
        
        return img_tensor.to(self.device)
    
    def predict(self, image):
        """
        Get ensemble predictions
        
        Returns:
            dict with:
                - detected_diseases: list of detected diseases
                - all_probabilities: dict of all probabilities
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Get predictions from all models
        predictions = []
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                output = model(img_tensor)
                probs = torch.sigmoid(output)
                predictions.append(probs.cpu().numpy()[0])
        
        # Simple average ensemble
        avg_probs = np.mean(predictions, axis=0)
        
        # Apply thresholds
        detected_diseases = []
        all_probabilities = {}
        
        for idx, class_name in enumerate(CLASS_NAMES):
            prob = float(avg_probs[idx])
            threshold = self.thresholds[class_name]
            all_probabilities[class_name] = prob
            
            if prob >= threshold:
                detected_diseases.append({
                    'name': class_name,
                    'confidence': prob,
                    'threshold': threshold
                })
        
        return {
            'detected_diseases': detected_diseases,
            'all_probabilities': all_probabilities
        }


# Initialize ensemble model (once at startup)
print("\n" + "="*80)
print("  INITIALIZING 3-MODEL ENSEMBLE API")
print("="*80)

try:
    ensemble = EnsembleModel(device=DEVICE)
    print("\n✅ Ensemble API ready!")
    print("="*80 + "\n")
except Exception as e:
    print(f"\n❌ Error initializing ensemble: {e}")
    print("="*80 + "\n")
    ensemble = None


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if ensemble is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    # Check if image is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get predictions
        result = ensemble.predict(image)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': '3-model ensemble',
        'models': ['ResNet-50', 'EfficientNet-B5', 'ViT-Base'],
        'performance': '72.26% Macro F1',
        'device': str(DEVICE)
    })


@app.route('/', methods=['GET'])
def index():
    """API info endpoint"""
    return jsonify({
        'name': 'ODIR-5K Ensemble API',
        'version': 'Phase 4C',
        'models': ['ResNet-50', 'EfficientNet-B5', 'ViT-Base'],
        'performance': {
            'macro_f1': 0.7226,
            'precision': 0.7182,
            'recall': 0.7325
        },
        'classes': CLASS_NAMES,
        'endpoints': {
            '/predict': 'POST - Upload image for prediction',
            '/health': 'GET - Health check',
            '/': 'GET - API information'
        }
    })


if __name__ == '__main__':
    print("\n🚀 Starting Flask API server...")
    print("   Access web interface at: http://localhost:5001")
    print("   API endpoint: http://localhost:5001/predict")
    print("   Health check: http://localhost:5001/health")
    print("\n   Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
