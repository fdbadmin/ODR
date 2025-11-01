"""
Flask API for ODIR-5K Ocular Disease Classification
Serves predictions for the web interface
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
from pathlib import Path
import io
from PIL import Image
import sys

# Import our model
from train import MultiLabelClassifier
from config import DISEASE_LABELS, DEFAULT_IMAGE_SIZE

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model variable
model = None
device = None

def preprocess_image(image_bytes):
    """
    Preprocess image for model input
    Same preprocessing as training
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE (same as preprocessing)
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
    
    # Convert to tensor (HWC -> CHW)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def load_model():
    """Load the trained model"""
    global model, device
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    model_path = Path("models/best_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = MultiLabelClassifier(num_classes=8)
    
    # Load checkpoint (may contain additional keys like optimizer state)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")

@app.route('/')
def home():
    """Home endpoint with API info"""
    return jsonify({
        "name": "ODIR-5K Ocular Disease Classifier API",
        "version": "1.0",
        "model": "ResNet50",
        "accuracy": "96.5%",
        "diseases": list(DISEASE_LABELS.values()),
        "endpoints": {
            "/predict": "POST - Upload image for prediction",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict diseases from uploaded fundus image
    
    Expected: multipart/form-data with 'image' file
    Returns: JSON with predictions
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Preprocess image
        img_tensor = preprocess_image(image_bytes)
        img_tensor = img_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Get threshold from query params (default 0.5)
        threshold = float(request.args.get('threshold', 0.5))
        
        # Format results
        results = {
            "success": True,
            "image_name": file.filename,
            "threshold": threshold,
            "detected_diseases": [],
            "all_probabilities": {}
        }
        
        for idx, (disease_code, disease_name) in enumerate(DISEASE_LABELS.items()):
            prob = float(probabilities[idx])
            results["all_probabilities"][disease_name] = prob
            
            if prob > threshold:
                results["detected_diseases"].append({
                    "code": disease_code,
                    "name": disease_name,
                    "probability": prob
                })
        
        # Sort detected diseases by probability
        results["detected_diseases"].sort(key=lambda x: x["probability"], reverse=True)
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict diseases from multiple images
    
    Expected: multipart/form-data with multiple 'images' files
    Returns: JSON with predictions for each image
    """
    try:
        if 'images' not in request.files:
            return jsonify({"error": "No image files provided"}), 400
        
        files = request.files.getlist('images')
        
        if len(files) == 0:
            return jsonify({"error": "No files selected"}), 400
        
        threshold = float(request.args.get('threshold', 0.5))
        
        results = {
            "success": True,
            "total_images": len(files),
            "threshold": threshold,
            "predictions": []
        }
        
        for file in files:
            try:
                # Read and preprocess image
                image_bytes = file.read()
                img_tensor = preprocess_image(image_bytes)
                img_tensor = img_tensor.to(device)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
                
                # Format result for this image
                image_result = {
                    "image_name": file.filename,
                    "detected_diseases": [],
                    "all_probabilities": {}
                }
                
                for idx, (disease_code, disease_name) in enumerate(DISEASE_LABELS.items()):
                    prob = float(probabilities[idx])
                    image_result["all_probabilities"][disease_name] = prob
                    
                    if prob > threshold:
                        image_result["detected_diseases"].append({
                            "code": disease_code,
                            "name": disease_name,
                            "probability": prob
                        })
                
                image_result["detected_diseases"].sort(key=lambda x: x["probability"], reverse=True)
                results["predictions"].append(image_result)
                
            except Exception as e:
                results["predictions"].append({
                    "image_name": file.filename,
                    "error": str(e)
                })
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error during batch prediction: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("\nStarting Flask API server...")
    print("API will be available at: http://localhost:5001")
    print("\nEndpoints:")
    print("  GET  /          - API information")
    print("  GET  /health    - Health check")
    print("  POST /predict   - Single image prediction")
    print("  POST /batch_predict - Multiple image predictions")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
