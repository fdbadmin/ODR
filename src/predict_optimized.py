#!/usr/bin/env python3
"""
Optimized Prediction Script
Uses ensemble model with optimal per-disease thresholds

Usage:
    python src/predict_optimized.py --image path/to/image.jpg
    python src/predict_optimized.py --image-dir path/to/images/ --output results.csv
"""

import torch
import numpy as np
import json
from pathlib import Path
import argparse
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

from ensemble_models import ResNet50Classifier, EfficientNetB3Classifier, DenseNet121Classifier, EnsembleModel
from train import MultiLabelClassifier

# Disease labels
LABEL_COLUMNS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
LABEL_NAMES = {
    'N': 'Normal',
    'D': 'Diabetes',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'AMD',
    'H': 'Hypertension',
    'M': 'Myopia',
    'O': 'Other'
}

# Load optimal thresholds
def load_optimal_thresholds():
    """Load optimal per-disease thresholds"""
    with open('optimal_thresholds.json', 'r') as f:
        thresholds = json.load(f)
    return [thresholds[label] for label in LABEL_COLUMNS]

# Image preprocessing
def preprocess_image(image_path):
    """Load and preprocess a single image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Load models
def load_models(device):
    """Load the ensemble model"""
    print("Loading models...")
    
    # Load individual models
    resnet50 = MultiLabelClassifier(num_classes=8, model_name='resnet50')
    resnet50.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    
    efficientnet = EfficientNetB3Classifier(num_classes=8)
    efficientnet.load_state_dict(torch.load('models/efficientnet_b3_best.pth', map_location=device))
    
    densenet = DenseNet121Classifier(num_classes=8)
    densenet.load_state_dict(torch.load('models/densenet121_best.pth', map_location=device))
    
    # Create ensemble
    ensemble = EnsembleModel([resnet50, efficientnet, densenet])
    ensemble = ensemble.to(device)
    ensemble.eval()
    
    print("✓ Models loaded successfully")
    return ensemble

# Predict with optimal thresholds
def predict_with_optimal_thresholds(model, image_tensor, thresholds, device):
    """
    Predict using optimal per-disease thresholds
    
    Args:
        model: Ensemble model
        image_tensor: Preprocessed image tensor
        thresholds: List of optimal thresholds per disease
        device: Device to run on
        
    Returns:
        probabilities: Raw probabilities for each disease
        predictions: Binary predictions using optimal thresholds
        confidence: Confidence scores for predicted diseases
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Apply optimal thresholds
    predictions = (probabilities >= np.array(thresholds)).astype(int)
    
    return probabilities, predictions

# Format results
def format_results(probabilities, predictions, include_normal=False):
    """
    Format prediction results in human-readable form
    
    Args:
        probabilities: Raw probabilities
        predictions: Binary predictions
        include_normal: Whether to include "Normal" in results
        
    Returns:
        results: List of (disease, probability, predicted) tuples
    """
    results = []
    
    for i, (label, prob, pred) in enumerate(zip(LABEL_COLUMNS, probabilities, predictions)):
        # Skip Normal unless explicitly requested
        if label == 'N' and not include_normal:
            continue
            
        if pred == 1:
            results.append({
                'disease': LABEL_NAMES[label],
                'code': label,
                'probability': float(prob),
                'predicted': bool(pred),
                'confidence': 'High' if prob > 0.8 else 'Medium' if prob > 0.6 else 'Low'
            })
    
    # Sort by probability (highest first)
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return results

# Main prediction function
def predict_single_image(image_path, model, thresholds, device):
    """Predict diseases for a single image"""
    # Preprocess
    image_tensor = preprocess_image(image_path)
    
    # Predict
    probabilities, predictions = predict_with_optimal_thresholds(
        model, image_tensor, thresholds, device
    )
    
    # Format
    results = format_results(probabilities, predictions)
    
    # Check if normal (no diseases predicted)
    if len(results) == 0 or (len(results) == 1 and results[0]['code'] == 'N'):
        return {
            'image': str(image_path),
            'status': 'Normal',
            'diseases': [],
            'all_probabilities': {LABEL_NAMES[label]: float(prob) 
                                 for label, prob in zip(LABEL_COLUMNS, probabilities)}
        }
    
    return {
        'image': str(image_path),
        'status': 'Abnormal',
        'diseases': results,
        'all_probabilities': {LABEL_NAMES[label]: float(prob) 
                             for label, prob in zip(LABEL_COLUMNS, probabilities)}
    }

# Batch prediction
def predict_directory(image_dir, model, thresholds, device, output_csv=None):
    """Predict diseases for all images in a directory"""
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    print(f"Found {len(image_files)} images")
    
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {image_path.name}...", end='\r')
        result = predict_single_image(image_path, model, thresholds, device)
        results.append(result)
    
    print()
    
    # Save to CSV if requested
    if output_csv:
        # Flatten results for CSV
        rows = []
        for result in results:
            row = {'image': result['image'], 'status': result['status']}
            
            # Add disease predictions
            for label in LABEL_COLUMNS:
                disease_name = LABEL_NAMES[label]
                row[f'{disease_name}_prob'] = result['all_probabilities'][disease_name]
                
                # Check if disease was predicted
                predicted = any(d['disease'] == disease_name for d in result['diseases'])
                row[f'{disease_name}_pred'] = 1 if predicted else 0
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"✓ Results saved to {output_csv}")
    
    return results

# Display results
def display_results(result):
    """Display prediction results in a nice format"""
    print("=" * 80)
    print(f"📸 Image: {result['image']}")
    print("=" * 80)
    
    if result['status'] == 'Normal':
        print("✅ Status: NORMAL (No diseases detected)")
    else:
        print("⚠️  Status: ABNORMAL")
        print()
        print("🔬 Detected Diseases:")
        for disease in result['diseases']:
            confidence_emoji = "🔴" if disease['confidence'] == 'High' else "🟡" if disease['confidence'] == 'Medium' else "⚪"
            print(f"   {confidence_emoji} {disease['disease']:15s} - {disease['probability']:5.1%} ({disease['confidence']} confidence)")
    
    print()
    print("📊 All Disease Probabilities:")
    for disease, prob in result['all_probabilities'].items():
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {disease:15s} {bar} {prob:5.1%}")
    
    print("=" * 80)

# Main function
def main():
    parser = argparse.ArgumentParser(description='Predict eye diseases from fundus images')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image-dir', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, help='Output CSV file for batch predictions')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto, cuda, mps, cpu)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load models and thresholds
    model = load_models(device)
    thresholds = load_optimal_thresholds()
    
    print()
    print("🎯 Using Optimized Thresholds:")
    for label, thresh in zip(LABEL_COLUMNS, thresholds):
        print(f"   {LABEL_NAMES[label]:15s}: {thresh:.3f}")
    print()
    
    # Single image or batch?
    if args.image:
        result = predict_single_image(args.image, model, thresholds, device)
        display_results(result)
    
    elif args.image_dir:
        results = predict_directory(args.image_dir, model, thresholds, device, args.output)
        
        # Display summary
        print()
        print("📈 Batch Prediction Summary:")
        print(f"   Total images: {len(results)}")
        normal_count = sum(1 for r in results if r['status'] == 'Normal')
        print(f"   Normal: {normal_count} ({normal_count/len(results)*100:.1f}%)")
        print(f"   Abnormal: {len(results) - normal_count} ({(len(results) - normal_count)/len(results)*100:.1f}%)")
        
        # Disease distribution
        disease_counts = {disease: 0 for disease in LABEL_NAMES.values() if disease != 'Normal'}
        for result in results:
            for disease_info in result['diseases']:
                if disease_info['disease'] != 'Normal':
                    disease_counts[disease_info['disease']] += 1
        
        print()
        print("   Disease Distribution:")
        for disease, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"      {disease:15s}: {count} images ({count/len(results)*100:.1f}%)")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
