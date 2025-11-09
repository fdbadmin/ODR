# Ensemble Deployment Guide

Complete guide for deploying the 3-model ensemble in production.

## Overview

The production ensemble consists of three complementary models:

1. **ResNet-50** (CNN): 67.28% F1, 23.5M parameters
2. **EfficientNet-B5** (CNN): 68.77% F1, 28.4M parameters  
3. **ViT-Base** (Transformer): 67.68% F1, 86.1M parameters

**Ensemble Performance**: 72.26% Macro F1 (with optimized thresholds)

## Quick Start

### 1. Model Files

Ensure all required files are present:

```
models_smart_exclusion/
├── best_resnet50.pth              # 270MB
└── optimized_thresholds.json

models_efficientnet_b5/
├── best_efficientnet_b5.pth       # 326MB
└── optimized_thresholds.json

models_vit_base/
├── best_vit_base.pth              # 985MB
└── optimized_thresholds.json

ensemble_optimized_thresholds_simple_average.json
```

**Total Storage**: ~1.6GB for all models

### 2. Python Dependencies

```bash
pip install torch torchvision timm numpy pillow
```

### 3. Basic Usage

```python
from src.predict_ensemble import EnsemblePredictor

# Initialize (one time setup)
predictor = EnsemblePredictor(device='cuda')  # or 'mps', 'cpu'

# Predict on image
results = predictor.predict('path/to/fundus.jpg')

# Check predictions
for disease, info in results.items():
    if info['predicted']:
        print(f"{disease}: {info['confidence']*100:.1f}%")
```

## Deployment Options

### Option 1: Command Line Interface

```bash
# Single image
python src/predict_ensemble.py --image fundus.jpg

# Multiple images
python src/predict_ensemble.py --images img1.jpg img2.jpg img3.jpg

# Save to JSON
python src/predict_ensemble.py --image fundus.jpg --output results.json
```

### Option 2: Python API

```python
from src.predict_ensemble import EnsemblePredictor
import json

# Initialize ensemble
predictor = EnsemblePredictor(device='cuda')

# Batch prediction
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = {}

for img_path in images:
    results[img_path] = predictor.predict(img_path)

# Save results
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Option 3: REST API (Flask)

```python
from flask import Flask, request, jsonify
from src.predict_ensemble import EnsemblePredictor
import os

app = Flask(__name__)
predictor = EnsemblePredictor(device='cuda')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    temp_path = f'/tmp/{file.filename}'
    file.save(temp_path)
    
    try:
        results = predictor.predict(temp_path)
        return jsonify(results)
    finally:
        os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Test the API:**

```bash
curl -X POST -F "image=@fundus.jpg" http://localhost:5000/predict
```

## Performance Optimization

### GPU Memory Management

**Single Model Memory Usage:**

- ResNet-50: ~1GB VRAM
- EfficientNet-B5: ~2GB VRAM
- ViT-Base: ~4GB VRAM

**Total Ensemble**: ~7GB VRAM required

**For Limited VRAM (<8GB):**

```python
# Load models sequentially instead of keeping all in memory
class SequentialEnsemble:
    def __init__(self, device='cuda'):
        self.device = device
        
    def predict(self, image):
        import torch
        
        # Process each model and clear memory
        predictions = []
        
        # Model 1: ResNet-50
        model1 = load_resnet50()
        pred1 = model1(image)
        predictions.append(pred1)
        del model1
        torch.cuda.empty_cache()
        
        # Model 2: EfficientNet-B5
        model2 = load_efficientnet_b5()
        pred2 = model2(image)
        predictions.append(pred2)
        del model2
        torch.cuda.empty_cache()
        
        # Model 3: ViT-Base
        model3 = load_vit_base()
        pred3 = model3(image)
        predictions.append(pred3)
        del model3
        torch.cuda.empty_cache()
        
        # Average predictions
        return np.mean(predictions, axis=0)
```

### Batch Processing

For high throughput:

```python
predictor = EnsemblePredictor(device='cuda')

# Process in batches
batch_size = 8
images = load_image_batch(batch_size)

# All models support batch inference
results = predictor.predict_batch(images)
```

### CPU Deployment

For environments without GPU:

```python
# Use CPU (slower but works everywhere)
predictor = EnsemblePredictor(device='cpu')

# Consider using fewer models for speed
predictor = EnsemblePredictor(
    device='cpu',
    models=['efficientnet_b5']  # Use only best single model
)
```

## Output Format

### Standard Output

```json
{
  "AMD": {
    "predicted": false,
    "confidence": 0.423,
    "threshold": 0.580
  },
  "Diabetes": {
    "predicted": true,
    "confidence": 0.687,
    "threshold": 0.560
  },
  "Glaucoma": {
    "predicted": false,
    "confidence": 0.234,
    "threshold": 0.590
  },
  "Cataract": {
    "predicted": false,
    "confidence": 0.512,
    "threshold": 0.750
  },
  "Myopia": {
    "predicted": true,
    "confidence": 0.891,
    "threshold": 0.810
  },
  "Normal": {
    "predicted": false,
    "confidence": 0.198,
    "threshold": 0.370
  },
  "Other": {
    "predicted": false,
    "confidence": 0.345,
    "threshold": 0.490
  }
}
```

### Interpretation

- **predicted**: Boolean - whether disease is detected (confidence > threshold)
- **confidence**: Float (0-1) - model's confidence score
- **threshold**: Float (0-1) - optimized decision threshold for this disease

**Clinical Interpretation:**

- `predicted=true`: Disease detected, recommend further examination
- `confidence > 0.8`: High confidence, priority case
- `confidence 0.6-0.8`: Moderate confidence, standard review
- `confidence < 0.6`: Low confidence, may need additional imaging

## Threshold Customization

### Using Default Optimized Thresholds

```python
# Loads optimized thresholds automatically
predictor = EnsemblePredictor()
```

**Default Thresholds** (optimized for macro F1):

- AMD: 0.580
- Diabetes: 0.560
- Glaucoma: 0.590
- Cataract: 0.750
- Myopia: 0.810
- Normal: 0.370
- Other: 0.490

### Custom Thresholds

For specific clinical requirements:

```python
# High sensitivity (catch more cases, more false positives)
high_sensitivity_thresholds = {
    'AMD': 0.3,
    'Diabetes': 0.3,
    'Glaucoma': 0.3,
    'Cataract': 0.4,
    'Myopia': 0.5,
    'Normal': 0.2,
    'Other': 0.3
}

predictor = EnsemblePredictor(thresholds=high_sensitivity_thresholds)

# High specificity (fewer false positives, may miss some cases)
high_specificity_thresholds = {
    'AMD': 0.8,
    'Diabetes': 0.8,
    'Glaucoma': 0.8,
    'Cataract': 0.9,
    'Myopia': 0.9,
    'Normal': 0.6,
    'Other': 0.7
}

predictor = EnsemblePredictor(thresholds=high_specificity_thresholds)
```

## Error Handling

### Robust Prediction

```python
def safe_predict(image_path):
    try:
        results = predictor.predict(image_path)
        return {'status': 'success', 'predictions': results}
    except FileNotFoundError:
        return {'status': 'error', 'message': 'Image file not found'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# Use it
result = safe_predict('fundus.jpg')
if result['status'] == 'success':
    print(result['predictions'])
else:
    print(f"Error: {result['message']}")
```

### Image Quality Validation

```python
from PIL import Image

def validate_image(image_path):
    """Check if image meets minimum requirements"""
    try:
        img = Image.open(image_path)
        
        # Check file format
        if img.format not in ['JPEG', 'PNG']:
            return False, "Only JPEG/PNG formats supported"
        
        # Check dimensions
        if min(img.size) < 224:
            return False, "Image too small (minimum 224x224)"
        
        # Check if grayscale or RGB
        if img.mode not in ['RGB', 'L']:
            return False, "Invalid color mode"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)

# Validate before prediction
valid, message = validate_image('fundus.jpg')
if valid:
    results = predictor.predict('fundus.jpg')
else:
    print(f"Invalid image: {message}")
```

## Monitoring & Logging

### Basic Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='predictions.log'
)

class LoggingEnsemble:
    def __init__(self, predictor):
        self.predictor = predictor
        
    def predict(self, image_path):
        logging.info(f"Processing: {image_path}")
        
        try:
            results = self.predictor.predict(image_path)
            
            # Log positive predictions
            positives = [d for d, info in results.items() if info['predicted']]
            if positives:
                logging.info(f"Detected: {', '.join(positives)}")
            else:
                logging.info("No diseases detected")
            
            return results
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

predictor = LoggingEnsemble(EnsemblePredictor())
```

### Performance Metrics

```python
import time

class TimedPredictor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.times = []
        
    def predict(self, image_path):
        start = time.time()
        results = self.predictor.predict(image_path)
        duration = time.time() - start
        
        self.times.append(duration)
        print(f"Prediction time: {duration:.2f}s")
        print(f"Average time: {np.mean(self.times):.2f}s")
        
        return results

predictor = TimedPredictor(EnsemblePredictor())
```

## Production Checklist

### Before Deployment

- [ ] All model files downloaded and verified
- [ ] Dependencies installed (`requirements.txt`)
- [ ] GPU/CPU resources allocated (7GB VRAM for GPU)
- [ ] Input validation implemented
- [ ] Error handling configured
- [ ] Logging system set up
- [ ] Load testing completed

### Security Considerations

- [ ] File upload size limits configured
- [ ] Input sanitization implemented
- [ ] Rate limiting enabled
- [ ] HTTPS enabled for API endpoints
- [ ] Model files secured (read-only access)
- [ ] Patient data anonymization

### Performance Targets

**Single Image (GPU):**

- ResNet-50: ~50ms
- EfficientNet-B5: ~100ms
- ViT-Base: ~150ms
- **Total Ensemble**: ~300ms

**Single Image (CPU):**

- **Total Ensemble**: ~2-3 seconds

**Throughput (GPU, batch size 8):**

- ~100 images/minute

## Troubleshooting

### Common Issues

**Issue**: Out of memory error

```python
# Solution: Use sequential loading or reduce batch size
predictor = EnsemblePredictor(device='cpu')  # or use sequential loading
```

**Issue**: Slow predictions

```python
# Solution: Enable GPU, use mixed precision
predictor = EnsemblePredictor(device='cuda', fp16=True)
```

**Issue**: Inconsistent results

```python
# Solution: Ensure preprocessing is consistent
from src.predict_ensemble import preprocess_image
image = preprocess_image('fundus.jpg')  # Always use same preprocessing
```

**Issue**: Model files not found

```python
# Solution: Check file paths and structure
import os
assert os.path.exists('models_smart_exclusion/best_resnet50.pth')
assert os.path.exists('models_efficientnet_b5/best_efficientnet_b5.pth')
assert os.path.exists('models_vit_base/best_vit_base.pth')
```

## Support

For issues or questions:

1. Check logs for error messages
2. Verify model files are not corrupted
3. Test with sample images from `tests/fixtures/`
4. Open an issue on GitHub with reproduction steps

## Updates & Maintenance

### Model Updates

To update ensemble with retrained models:

1. Replace model files in respective directories
2. Update threshold files if thresholds changed
3. Update version number in code
4. Re-run validation tests
5. Document changes in changelog

### Monitoring Production Performance

Track these metrics:

- Prediction latency (p50, p95, p99)
- Error rate
- Throughput (predictions/second)
- Resource utilization (GPU/CPU/memory)
- Disease detection rates per class

## License & Disclaimer

**For Research and Educational Use Only**

This system is NOT FDA-approved and should NOT be used as the sole basis for clinical decisions. Always confirm findings with qualified ophthalmologists and additional diagnostic tests.

---

**Version**: 1.0 (Phase 4C)  
**Last Updated**: November 2025  
**Performance**: 72.26% Macro F1
