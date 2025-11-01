"""
Inference script for ODIR-5K multi-label classification.
Make predictions on new fundus images using the trained model.
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Union
import argparse
import json

from train import MultiLabelClassifier, get_device
from config import LABEL_COLUMNS, DISEASE_LABELS


class ODIRPredictor:
    """Class for making predictions on fundus images."""
    
    def __init__(self, model_path: str, device: torch.device = None, threshold: float = 0.5):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on (None for auto-detect)
            threshold: Probability threshold for positive prediction
        """
        self.device = device if device else get_device()
        self.threshold = threshold
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = MultiLabelClassifier(num_classes=len(LABEL_COLUMNS))
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"  Val Acc: {checkpoint.get('val_acc', 'N/A')}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image (same as training preprocessing).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE (same as preprocessing)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize to 224x224
        image = cv2.resize(image, (224, 224))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and rearrange from HWC to CHW
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image_path: str) -> Dict:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with predictions, probabilities, and disease names
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            predictions = (probabilities > self.threshold).astype(int)
        
        # Format results
        results = {
            'image_path': str(image_path),
            'predictions': {},
            'detected_diseases': [],
            'all_probabilities': {}
        }
        
        for i, label in enumerate(LABEL_COLUMNS):
            disease_name = DISEASE_LABELS.get(label, label)
            results['predictions'][label] = int(predictions[i])
            results['all_probabilities'][disease_name] = float(probabilities[i])
            
            if predictions[i] == 1:
                results['detected_diseases'].append({
                    'code': label,
                    'name': disease_name,
                    'probability': float(probabilities[i])
                })
        
        # Add summary
        if len(results['detected_diseases']) == 0:
            results['summary'] = "No diseases detected (Normal)"
        else:
            disease_names = [d['name'] for d in results['detected_diseases']]
            results['summary'] = f"Detected: {', '.join(disease_names)}"
        
        return results
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Make predictions on multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        return results
    
    def print_prediction(self, result: Dict, detailed: bool = False):
        """Print prediction results in a formatted way."""
        print("\n" + "="*70)
        print(f"Image: {Path(result['image_path']).name}")
        print("="*70)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"\n{result['summary']}")
        
        if result['detected_diseases']:
            print(f"\n🔍 Detected Diseases ({len(result['detected_diseases'])}):")
            for disease in result['detected_diseases']:
                prob_pct = disease['probability'] * 100
                print(f"  • {disease['name']} ({disease['code']}): {prob_pct:.1f}% confidence")
        
        if detailed:
            print(f"\n📊 All Disease Probabilities:")
            for disease_name, prob in sorted(result['all_probabilities'].items(), 
                                            key=lambda x: x[1], reverse=True):
                prob_pct = prob * 100
                bar = "█" * int(prob_pct / 5)
                print(f"  {disease_name:<40} {prob_pct:6.2f}% {bar}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Make predictions on fundus images using trained ODIR-5K model'
    )
    parser.add_argument('input', type=str, 
                       help='Path to image file or directory of images')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to model checkpoint (default: models/best_model.pth)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold (default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (optional)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed probabilities for all diseases')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ODIRPredictor(args.model, threshold=args.threshold)
    
    # Get image paths
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(input_path.glob(ext))
        print(f"\nFound {len(image_paths)} images in {input_path}")
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    
    # Make predictions
    print(f"\n🔮 Making predictions...")
    results = predictor.predict_batch([str(p) for p in image_paths])
    
    # Print results
    for result in results:
        predictor.print_prediction(result, detailed=args.detailed)
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    # Summary
    successful = sum(1 for r in results if 'error' not in r)
    print("\n" + "="*70)
    print(f"✅ Processed {successful}/{len(results)} images successfully")
    print("="*70)


if __name__ == "__main__":
    main()
