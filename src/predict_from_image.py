"""
Predict eye diseases from raw JPEG fundus images.
Handles preprocessing and inference using the optimized ensemble.
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Union, List, Dict

from src.production_ensemble import ProductionEnsemble


def preprocess_fundus_image(image_path: str, target_size: int = 224) -> np.ndarray:
    """
    Preprocess a raw fundus image to match training preprocessing.
    
    Preprocessing steps (must match training):
    1. Load image
    2. Extract green channel (most informative for fundus images)
    3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Illumination correction
    5. Resize to 224x224
    6. Normalize to [0, 1]
    
    Args:
        image_path: Path to JPEG image
        target_size: Target size (default: 224)
    
    Returns:
        Preprocessed image as (224, 224, 3) numpy array
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract green channel (most informative for retinal images)
    green_channel = image[:, :, 1]
    
    # Apply CLAHE with clip_limit=3.0 (matches training)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)
    
    # Illumination correction (subtract median-filtered version)
    # This removes uneven illumination
    median_filtered = cv2.medianBlur(enhanced, 21)
    corrected = cv2.subtract(enhanced, median_filtered)
    corrected = cv2.add(corrected, 128)  # Re-center
    
    # Create 3-channel image (duplicate green channel to all channels)
    # This matches training where we used green channel preprocessing
    preprocessed = np.stack([corrected, corrected, corrected], axis=-1)
    
    # Resize to target size
    preprocessed = cv2.resize(preprocessed, (target_size, target_size), 
                             interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    preprocessed = preprocessed.astype(np.float32) / 255.0
    
    return preprocessed


def predict_single_image(image_path: str, ensemble: ProductionEnsemble, 
                        show_details: bool = True) -> Dict:
    """
    Predict diseases for a single image.
    
    Args:
        image_path: Path to JPEG image
        ensemble: Loaded ProductionEnsemble model
        show_details: If True, print detailed results
    
    Returns:
        Dictionary with predictions
    """
    print(f"\nProcessing: {Path(image_path).name}")
    
    # Preprocess image
    preprocessed = preprocess_fundus_image(image_path)
    
    # Get predictions
    predictions = ensemble.predict(preprocessed)
    
    if show_details:
        print("\n" + "="*70)
        print("PREDICTIONS")
        print("="*70)
        
        # Check if any diseases detected
        detected_diseases = [disease for disease, info in predictions.items() 
                           if info['predicted']]
        
        if detected_diseases:
            print(f"\n⚠️  DETECTED DISEASES: {len(detected_diseases)}")
            print("\n" + "-"*70)
            for disease, info in predictions.items():
                if info['predicted']:
                    confidence_pct = info['confidence'] * 100
                    print(f"  🔴 {disease:<15} YES  (confidence: {confidence_pct:5.1f}%, "
                          f"threshold: {info['threshold']:.2f})")
        else:
            print("\n✅ NO DISEASES DETECTED - Eye appears healthy")
        
        print("\n" + "-"*70)
        print("All predictions:")
        print(f"  {'Disease':<15} {'Predicted':<12} {'Confidence':<12} {'Threshold':<12}")
        print("  " + "-"*60)
        
        for disease, info in predictions.items():
            status = "YES" if info['predicted'] else "No"
            confidence_pct = info['confidence'] * 100
            icon = "🔴" if info['predicted'] else "⚪"
            print(f"  {icon} {disease:<15} {status:<12} {confidence_pct:5.1f}%      "
                  f"{info['threshold']:.2f}")
        
        print("="*70)
    
    return predictions


def predict_batch_images(image_paths: List[str], ensemble: ProductionEnsemble) -> List[Dict]:
    """
    Predict diseases for multiple images.
    
    Args:
        image_paths: List of paths to JPEG images
        ensemble: Loaded ProductionEnsemble model
    
    Returns:
        List of prediction dictionaries
    """
    print(f"\nProcessing {len(image_paths)} images...")
    
    # Preprocess all images
    preprocessed_images = []
    valid_paths = []
    
    for image_path in image_paths:
        try:
            preprocessed = preprocess_fundus_image(image_path)
            preprocessed_images.append(preprocessed)
            valid_paths.append(image_path)
        except Exception as e:
            print(f"⚠️  Skipping {Path(image_path).name}: {e}")
    
    if not preprocessed_images:
        print("❌ No valid images to process")
        return []
    
    # Stack into batch
    batch = np.stack(preprocessed_images, axis=0)
    
    # Get predictions
    predictions_list = ensemble.predict_batch(batch)
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PREDICTION SUMMARY")
    print("="*70)
    
    for image_path, predictions in zip(valid_paths, predictions_list):
        detected = sum(1 for info in predictions.values() if info['predicted'])
        status = "✅ Healthy" if detected == 0 else f"⚠️  {detected} disease(s)"
        print(f"  {Path(image_path).name:<40} {status}")
    
    print("="*70)
    
    return predictions_list


def save_predictions(predictions: Union[Dict, List[Dict]], 
                    output_path: str,
                    image_paths: Union[str, List[str]]):
    """Save predictions to JSON file."""
    if isinstance(predictions, dict):
        # Single prediction
        output = {
            'image': str(Path(image_paths).name),
            'predictions': {
                disease: {
                    'predicted': bool(info['predicted']),
                    'confidence': float(info['confidence']),
                    'threshold': float(info['threshold'])
                }
                for disease, info in predictions.items()
            },
            'detected_diseases': [disease for disease, info in predictions.items() 
                                 if info['predicted']]
        }
    else:
        # Batch predictions
        output = []
        for image_path, preds in zip(image_paths, predictions):
            output.append({
                'image': str(Path(image_path).name),
                'predictions': {
                    disease: {
                        'predicted': bool(info['predicted']),
                        'confidence': float(info['confidence']),
                        'threshold': float(info['threshold'])
                    }
                    for disease, info in preds.items()
                },
                'detected_diseases': [disease for disease, info in preds.items() 
                                     if info['predicted']]
            })
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Predictions saved to: {output_path}")


def main():
    """Command-line interface for predictions."""
    parser = argparse.ArgumentParser(
        description='Predict eye diseases from fundus images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python src/predict_from_image.py --image path/to/fundus.jpg
  
  # Multiple images
  python src/predict_from_image.py --images image1.jpg image2.jpg image3.jpg
  
  # Save results to JSON
  python src/predict_from_image.py --image fundus.jpg --output results.json
  
  # Use different device
  python src/predict_from_image.py --image fundus.jpg --device cpu
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--images', nargs='+', help='Paths to multiple images')
    parser.add_argument('--output', type=str, help='Save predictions to JSON file')
    parser.add_argument('--device', type=str, default='mps', 
                       choices=['mps', 'cuda', 'cpu'],
                       help='Device to run inference on (default: mps)')
    
    args = parser.parse_args()
    
    if not args.image and not args.images:
        parser.error("Must provide either --image or --images")
    
    # Initialize ensemble
    print("Loading ensemble model...")
    ensemble = ProductionEnsemble(device=args.device)
    print()
    
    # Make predictions
    if args.image:
        # Single image
        predictions = predict_single_image(args.image, ensemble, show_details=True)
        
        if args.output:
            save_predictions(predictions, args.output, args.image)
    
    elif args.images:
        # Multiple images
        predictions = predict_batch_images(args.images, ensemble)
        
        if args.output:
            save_predictions(predictions, args.output, args.images)


if __name__ == '__main__':
    main()
