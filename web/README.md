# Web Deployment Guide - Phase 4C

## Overview

This directory contains the web interface for the ODIR-5K Ocular Disease Classifier **Phase 4C** (72.26% Macro F1).

The system consists of:
- **Frontend**: Beautiful HTML/CSS/JavaScript interface (`index.html`)
- **Backend**: Flask API server (`api_ensemble.py` in parent directory)
- **Models**: 3-model ensemble (ResNet-50 + EfficientNet-B5 + ViT-Base)

## Deployment Options

### Option 1: GitHub Pages (Static Demo - No Real Predictions)

The `index.html` includes a demo mode that works without a backend. This is perfect for showcasing the interface.

**Steps:**
1. Push the `web/` folder to your GitHub repository
2. Go to repository Settings → Pages
3. Select the branch and `/web` folder
4. Your site will be available at: `https://fdbadmin.github.io/ODR/`

**Note:** In demo mode, predictions are simulated. For real predictions, deploy the backend API.

### Option 2: Full Deployment with Backend API

For production use with real predictions, you need to deploy both frontend and backend.

#### Backend Deployment Options:

**A. Local Server (Development)**
```bash
# Install dependencies
pip install flask flask-cors torch torchvision timm pillow

# Run API server (loads 3 models ~1.6GB)
python api_ensemble.py

# API will be available at http://localhost:5001
# Note: Port 5000 is used by macOS AirPlay, so we use 5001
```

**B. Cloud Deployment (Recommended)**

**Heroku:**
1. Create `Procfile`:
   ```
   web: gunicorn api_ensemble:app
   ```
2. Install gunicorn: `pip install gunicorn`
3. Deploy to Heroku (Note: Model files ~1.6GB may require larger dyno)
4. Update `API_URL` in `index.html` to your Heroku URL

**AWS Lambda + API Gateway:**
- Use Zappa or AWS SAM for serverless deployment
- Benefits: Auto-scaling, pay-per-use
- Update `API_URL` in `index.html`

**Google Cloud Run:**
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/odir-api

# Deploy
gcloud run deploy odir-api --image gcr.io/PROJECT_ID/odir-api --platform managed
```

**Railway/Render:**
- Connect GitHub repo
- Set start command: `python api.py`
- Environment auto-detected
- Update `API_URL` in `index.html`

### Option 3: Convert Model to TensorFlow.js (Browser-Based)

For a completely static deployment on GitHub Pages with real predictions:

**Steps:**
1. Convert PyTorch model to ONNX format
2. Convert ONNX to TensorFlow.js
3. Load model directly in browser
4. No backend server needed!

Would you like me to create the conversion script for this option?

## Configuration

### Update API URL

In `web/index.html`, find this line:
```javascript
const API_URL = 'http://localhost:5000/predict';
```

Change it to your deployed API URL:
```javascript
const API_URL = 'https://your-api-url.com/predict';
```

### Enable CORS

If your frontend and backend are on different domains, CORS is already enabled in `api_ensemble.py` via:
```python
from flask_cors import CORS
CORS(app)
```

## Testing Locally

### Terminal 1 - Start Backend:
```bash
python api_ensemble.py  # Loads 3 models (~1.6GB)
```

### Terminal 2 - Start Frontend:
```bash
cd web
python -m http.server 8000
```

Open browser: `http://localhost:8000`

## Features

### Frontend Features:
- ✅ Drag & drop image upload
- ✅ Click to upload
- ✅ Image preview
- ✅ Beautiful results visualization
- ✅ Confidence bars for each disease
- ✅ Professional medical UI design
- ✅ Responsive (mobile-friendly)
- ✅ Demo mode for GitHub Pages

### Backend API Features:
- ✅ `/predict` - Single image classification
- ✅ `/batch_predict` - Multiple images
- ✅ `/health` - Health check
- ✅ Adjustable confidence threshold
- ✅ CORS enabled
- ✅ Error handling

## API Documentation

### POST /predict

Upload a single fundus image for classification.

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@fundus_image.jpg" \
  -F "threshold=0.5"
```

**Response:**
```json
{
  "success": true,
  "image_name": "fundus_image.jpg",
  "threshold": 0.5,
  "detected_diseases": [
    {
      "code": "D",
      "name": "Diabetic Retinopathy",
      "probability": 0.946
    }
  ],
  "all_probabilities": {
    "Normal": 0.049,
    "Diabetic Retinopathy": 0.946,
    "Glaucoma": 0.007,
    ...
  }
}
```

### POST /batch_predict

Upload multiple images for batch classification.

**Request:**
```bash
curl -X POST http://localhost:5000/batch_predict \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg"
```

## Security Considerations

⚠️ **Important for Production:**

1. **Add Authentication**: Protect your API with API keys or OAuth
2. **Rate Limiting**: Prevent abuse with rate limits
3. **File Size Limits**: Limit upload sizes
4. **Input Validation**: Validate file types and sizes
5. **HTTPS Only**: Use SSL certificates
6. **Model Security**: Don't expose model weights publicly

Example rate limiting with Flask-Limiter:
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per hour"]
)
```

## Cost Optimization

**Model Size:** ~1.6GB total (ResNet-50: 270MB, EfficientNet-B5: 326MB, ViT-Base: 985MB)
- Large model size requires sufficient disk space and memory
- Consider deploying single best model (EfficientNet-B5: 68.77% F1) for resource constraints
- Model compression techniques (quantization, pruning) can reduce size

**API Hosting:**
- Free tier: Limited by model size (1.6GB may exceed free tier limits)
- Recommended: Railway, Render, or AWS EC2 with sufficient storage/RAM
- Budget: $10-30/month for medium traffic with 1.6GB models

**Alternatives:**
- Single model deployment: Use only EfficientNet-B5 (326MB) to reduce costs
- Edge deployment: Not practical with 1.6GB ensemble
- Quantization: PyTorch quantization can reduce model size by 4x

## Troubleshooting

### CORS Errors
- Ensure `flask-cors` is installed
- Check API URL in `index.html`
- Verify backend is running

### Model Loading Errors
- Ensure model files exist:
  - `models_smart_exclusion/best_resnet50.pth` (270MB)
  - `models_efficientnet_b5/best_efficientnet_b5.pth` (326MB)
  - `models_vit_base/best_vit_base.pth` (985MB)
- Check PyTorch version compatibility (2.0+)
- Verify device availability (MPS/CUDA/CPU)
- Ensure sufficient RAM (~4GB for model inference)

### Image Upload Errors
- Check file size limits
- Verify image format (JPG/PNG)
- Check API endpoint URL

## Next Steps

1. **Test Locally**: Run both frontend and backend locally
2. **Choose Deployment**: Select deployment option based on needs
3. **Update URLs**: Configure API endpoint in frontend
4. **Deploy Backend**: Deploy API to cloud service
5. **Deploy Frontend**: Host on GitHub Pages or with backend
6. **Test Live**: Verify predictions work end-to-end
7. **Add Features**: Authentication, logging, analytics

## Performance Tips

- **Caching**: Cache model in memory (already done)
- **Batch Processing**: Use `/batch_predict` for multiple images
- **CDN**: Serve static files via CDN
- **Compression**: Enable gzip compression
- **Async**: Use async/await for better performance

## Need Help?

See the main README.md for more information or open an issue on GitHub.

---

**Disclaimer:** This tool is for educational and research purposes only. Not intended for clinical diagnosis. Always consult healthcare professionals for medical advice.
