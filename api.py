"""
Flask API for ODIR-5K Ocular Disease Classification
Serves predictions for the web interface using the final ensemble model
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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import our production ensemble
from production_ensemble import ProductionEnsemble
from config import DISEASE_LABELS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global ensemble model
ensemble = None

def preprocess_fundus_image(image_bytes):
    """
    Complete preprocessing matching training pipeline
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract green channel (most informative for fundus)
    green_channel = image[:, :, 1]
    
    # Apply CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)
    
    # Illumination correction
    median_filtered = cv2.medianBlur(enhanced, 21)
    corrected = cv2.subtract(enhanced, median_filtered)
    corrected = cv2.add(corrected, 128)
    
    # Create 3-channel image
    preprocessed = np.stack([corrected, corrected, corrected], axis=-1)
    
    # Resize to 224x224
    preprocessed = cv2.resize(preprocessed, (224, 224))
    
    # Normalize to [0, 1]
    preprocessed = preprocessed.astype(np.float32) / 255.0
    
    return preprocessed

def load_model():
    """Load the trained ensemble model"""
    global ensemble
    
    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        print("Note: Running on CPU (normal for cloud deployments)")
    
    print(f"Using device: {device}")
    
    # Load production ensemble
    try:
        ensemble = ProductionEnsemble(
            device=device,
            models_dir='models',
            thresholds_path='results/optimal_thresholds.json'
        )
        
        # Get model info
        info = ensemble.get_model_info()
        print("✓ Ensemble model loaded successfully!")
        print(f"  Models: {', '.join(info['models'])}")
        print(f"  Performance: F1={info['performance']['mean_f1']:.4f}, Accuracy={info['performance']['label_accuracy']:.4f}")
    except Exception as e:
        print(f"Error loading ensemble: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.route('/')
def home():
    """Home endpoint with API info"""
    model_info = ensemble.get_model_info() if ensemble else {}
    
    return jsonify({
        "name": "ODIR-5K Ocular Disease Classifier API",
        "version": "2.0",
        "model": "3-Model Ensemble (ResNet50 + EfficientNet-B3 + DenseNet-121)",
        "performance": model_info.get('performance', {}),
        "diseases": model_info.get('diseases', list(DISEASE_LABELS.values())),
        "endpoints": {
            "/predict": "POST - Upload image for prediction",
            "/batch_predict": "POST - Upload multiple images",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    model_info = ensemble.get_model_info() if ensemble else {}
    
    return jsonify({
        "status": "healthy",
        "model_loaded": ensemble is not None,
        "device": ensemble.device if ensemble else None,
        "models": model_info.get('models', [])
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict diseases from uploaded fundus image using ensemble model
    
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
        
        # Read and preprocess image
        image_bytes = file.read()
        preprocessed_image = preprocess_fundus_image(image_bytes)
        
        # Make prediction using ensemble
        predictions = ensemble.predict(preprocessed_image)
        
        # Format results
        results = {
            "success": True,
            "image_name": file.filename,
            "model": "3-Model Ensemble",
            "detected_diseases": [],
            "all_probabilities": {}
        }
        
        for disease_name, info in predictions.items():
            results["all_probabilities"][disease_name] = info['confidence']
            
            if info['predicted']:
                results["detected_diseases"].append({
                    "name": disease_name,
                    "confidence": info['confidence'],
                    "threshold": info['threshold']
                })
        
        # Sort detected diseases by confidence
        results["detected_diseases"].sort(key=lambda x: x["confidence"], reverse=True)
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict diseases from multiple images using ensemble model
    
    Expected: multipart/form-data with multiple 'images' files
    Returns: JSON with predictions for each image
    """
    try:
        if 'images' not in request.files:
            return jsonify({"error": "No image files provided"}), 400
        
        files = request.files.getlist('images')
        
        if len(files) == 0:
            return jsonify({"error": "No files selected"}), 400
        
        results = {
            "success": True,
            "total_images": len(files),
            "model": "3-Model Ensemble",
            "predictions": []
        }
        
        for file in files:
            try:
                # Read and preprocess image
                image_bytes = file.read()
                preprocessed_image = preprocess_fundus_image(image_bytes)
                
                # Make prediction using ensemble
                predictions = ensemble.predict(preprocessed_image)
                
                # Format result for this image
                image_result = {
                    "image_name": file.filename,
                    "detected_diseases": [],
                    "all_probabilities": {}
                }
                
                for disease_name, info in predictions.items():
                    image_result["all_probabilities"][disease_name] = info['confidence']
                    
                    if info['predicted']:
                        image_result["detected_diseases"].append({
                            "name": disease_name,
                            "confidence": info['confidence'],
                            "threshold": info['threshold']
                        })
                
                image_result["detected_diseases"].sort(key=lambda x: x["confidence"], reverse=True)
                results["predictions"].append(image_result)
                
            except Exception as e:
                results["predictions"].append({
                    "image_name": file.filename,
                    "error": str(e)
                })
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error during batch prediction: {str(e)}")
        import traceback
        traceback.print_exc()
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
