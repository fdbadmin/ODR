"""
Test-Time Augmentation (TTA) for Ensemble Evaluation
=====================================================

Applies TTA to the 3-model ensemble to boost performance without retraining.

TTA Strategy:
- Original image
- Horizontal flip
- Vertical flip
- Both flips
- 90° rotation
- 180° rotation
- 270° rotation
- Average predictions across all 8 views

Expected gain: +2-4% F1 (from 72.26% to 74-76%)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import timm

# Device setup
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("✅ Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ Using CUDA GPU")
else:
    device = torch.device('cpu')
    print("⚠️ Using CPU")

# ImageNet normalization constants
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1).to(device)

class_names = ['AMD', 'Diabetes', 'Glaucoma', 'Cataract', 'Myopia', 'Normal', 'Other']


def load_model(model_name, model_path, num_classes=7):
    """Load a trained model."""
    print(f"   Loading {model_name}...")
    
    if model_name == 'resnet50':
        model = timm.create_model('resnet50.a1_in1k', pretrained=False, num_classes=num_classes)
    elif model_name == 'efficientnet_b5':
        model = timm.create_model('efficientnet_b5.sw_in12k_ft_in1k', pretrained=False, num_classes=num_classes)
    elif model_name == 'vit_base':
        model = timm.create_model('vit_base_patch16_384.augreg_in21k_ft_in1k', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def apply_tta_augmentations(image):
    """
    Apply TTA augmentations to a single image.
    
    Args:
        image: (H, W, 3) numpy array, float32 [0, 1]
    
    Returns:
        List of 8 augmented versions as tensors (1, 3, H, W)
    """
    augmented = []
    
    # Convert to tensor (H, W, 3) -> (3, H, W)
    img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    # Apply ImageNet normalization
    img_normalized = (img_tensor - MEAN) / STD
    
    # 1. Original
    augmented.append(img_normalized)
    
    # 2. Horizontal flip
    augmented.append(torch.flip(img_normalized, dims=[3]))
    
    # 3. Vertical flip
    augmented.append(torch.flip(img_normalized, dims=[2]))
    
    # 4. Both flips
    augmented.append(torch.flip(img_normalized, dims=[2, 3]))
    
    # 5. Rotate 90°
    augmented.append(torch.rot90(img_normalized, k=1, dims=[2, 3]))
    
    # 6. Rotate 180°
    augmented.append(torch.rot90(img_normalized, k=2, dims=[2, 3]))
    
    # 7. Rotate 270°
    augmented.append(torch.rot90(img_normalized, k=3, dims=[2, 3]))
    
    # 8. Rotate 90° + horizontal flip
    augmented.append(torch.flip(torch.rot90(img_normalized, k=1, dims=[2, 3]), dims=[3]))
    
    return augmented


def predict_with_tta(models, image, return_all=False):
    """
    Predict using TTA across all models.
    
    Args:
        models: List of PyTorch models
        image: (H, W, 3) numpy array, float32 [0, 1]
        return_all: If True, return all individual predictions
    
    Returns:
        averaged_probs: (7,) averaged probabilities across all models and augmentations
        all_preds: (optional) list of all predictions for analysis
    """
    # Get augmented versions
    augmented_images = apply_tta_augmentations(image)
    
    all_preds = []
    
    with torch.no_grad():
        for model in models:
            model_preds = []
            
            # Predict on each augmented version
            for aug_img in augmented_images:
                logits = model(aug_img)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                model_preds.append(probs)
            
            # Average across augmentations for this model
            model_avg = np.mean(model_preds, axis=0)
            all_preds.append(model_avg)
    
    # Average across models
    averaged_probs = np.mean(all_preds, axis=0)
    
    if return_all:
        return averaged_probs, all_preds
    return averaged_probs


def evaluate_with_tta(models, val_images, val_labels, thresholds):
    """
    Evaluate ensemble with TTA on validation set.
    
    Args:
        models: List of trained models
        val_images: Validation images (N, H, W, 3)
        val_labels: Validation labels (N, 7)
        thresholds: Dict of per-class thresholds
    
    Returns:
        metrics: Dict with F1, precision, recall, per-class F1
    """
    print(f"\n🔮 Running TTA evaluation on {len(val_images)} validation images...")
    print(f"   Strategy: 8 augmentations × {len(models)} models = {8 * len(models)} predictions per image")
    
    all_probs = []
    
    for i in tqdm(range(len(val_images)), desc="TTA Inference"):
        image = val_images[i]
        
        # Predict with TTA
        avg_probs = predict_with_tta(models, image)
        all_probs.append(avg_probs)
    
    all_probs = np.array(all_probs)
    
    # Apply optimized thresholds
    threshold_array = np.array([
        thresholds['AMD'],
        thresholds['Diabetes'],
        thresholds['Glaucoma'],
        thresholds['Cataract'],
        thresholds['Myopia'],
        thresholds['Normal'],
        thresholds['Other']
    ])
    
    predictions = (all_probs >= threshold_array).astype(int)
    
    # Calculate metrics
    f1_macro = f1_score(val_labels, predictions, average='macro', zero_division=0)
    precision = precision_score(val_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(val_labels, predictions, average='macro', zero_division=0)
    per_class_f1 = f1_score(val_labels, predictions, average=None, zero_division=0)
    
    return {
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'per_class_f1': per_class_f1,
        'predictions': predictions,
        'probabilities': all_probs
    }


def main():
    print("=" * 80)
    print("TEST-TIME AUGMENTATION (TTA) EVALUATION")
    print("=" * 80)
    
    # Load validation data
    print("\n📂 Loading validation data...")
    data_dir = Path('preprocessed_data_smart_exclusion')
    
    val_images = np.load(data_dir / 'val_images.npy', mmap_mode='r')
    val_labels = np.load(data_dir / 'val_labels.npy')
    
    print(f"   Loaded {len(val_images)} validation images")
    
    # Load optimized thresholds
    print("\n📊 Loading optimized thresholds...")
    with open('ensemble_optimized_thresholds_simple_average.json', 'r') as f:
        threshold_data = json.load(f)
    
    # Extract thresholds from nested structure
    thresholds = threshold_data['thresholds']
    print(f"   Thresholds: {thresholds}")
    
    # Load all 3 models
    print("\n🔨 Loading ensemble models...")
    models = [
        load_model('resnet50', 'models_smart_exclusion/best_resnet50.pth'),
        load_model('efficientnet_b5', 'models_efficientnet_b5/best_efficientnet_b5.pth'),
        load_model('vit_base', 'models_vit_base/best_vit_base.pth')
    ]
    print(f"   ✅ Loaded {len(models)} models")
    
    # Evaluate WITHOUT TTA (baseline)
    print("\n" + "=" * 80)
    print("BASELINE: Ensemble WITHOUT TTA")
    print("=" * 80)
    
    print("\n📊 Running baseline evaluation (no TTA)...")
    baseline_probs = []
    
    for i in tqdm(range(len(val_images)), desc="Baseline"):
        image = val_images[i]
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        img_normalized = (img_tensor - MEAN) / STD
        
        # Predict with each model
        model_preds = []
        with torch.no_grad():
            for model in models:
                logits = model(img_normalized)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                model_preds.append(probs)
        
        # Average across models
        avg_probs = np.mean(model_preds, axis=0)
        baseline_probs.append(avg_probs)
    
    baseline_probs = np.array(baseline_probs)
    
    # Apply thresholds
    threshold_array = np.array([
        thresholds['AMD'],
        thresholds['Diabetes'],
        thresholds['Glaucoma'],
        thresholds['Cataract'],
        thresholds['Myopia'],
        thresholds['Normal'],
        thresholds['Other']
    ])
    
    baseline_preds = (baseline_probs >= threshold_array).astype(int)
    
    baseline_f1 = f1_score(val_labels, baseline_preds, average='macro', zero_division=0)
    baseline_precision = precision_score(val_labels, baseline_preds, average='macro', zero_division=0)
    baseline_recall = recall_score(val_labels, baseline_preds, average='macro', zero_division=0)
    baseline_per_class_f1 = f1_score(val_labels, baseline_preds, average=None, zero_division=0)
    
    print(f"\n📈 Baseline Results (No TTA):")
    print(f"   Macro F1:    {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
    print(f"   Precision:   {baseline_precision:.4f} ({baseline_precision*100:.2f}%)")
    print(f"   Recall:      {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
    
    print(f"\n   Per-Class F1:")
    for i, name in enumerate(class_names):
        print(f"   {name:<12} {baseline_per_class_f1[i]:.4f} ({baseline_per_class_f1[i]*100:.2f}%)")
    
    # Evaluate WITH TTA
    print("\n" + "=" * 80)
    print("TTA: Ensemble WITH Test-Time Augmentation")
    print("=" * 80)
    
    tta_results = evaluate_with_tta(models, val_images, val_labels, thresholds)
    
    print(f"\n📈 TTA Results:")
    print(f"   Macro F1:    {tta_results['f1_macro']:.4f} ({tta_results['f1_macro']*100:.2f}%)")
    print(f"   Precision:   {tta_results['precision']:.4f} ({tta_results['precision']*100:.2f}%)")
    print(f"   Recall:      {tta_results['recall']:.4f} ({tta_results['recall']*100:.2f}%)")
    
    print(f"\n   Per-Class F1:")
    for i, name in enumerate(class_names):
        print(f"   {name:<12} {tta_results['per_class_f1'][i]:.4f} ({tta_results['per_class_f1'][i]*100:.2f}%)")
    
    # Calculate improvement
    print("\n" + "=" * 80)
    print("TTA IMPROVEMENT ANALYSIS")
    print("=" * 80)
    
    f1_improvement = tta_results['f1_macro'] - baseline_f1
    f1_improvement_pct = (f1_improvement / baseline_f1) * 100
    
    print(f"\n🎯 Overall Improvement:")
    print(f"   Baseline F1:     {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
    print(f"   TTA F1:          {tta_results['f1_macro']:.4f} ({tta_results['f1_macro']*100:.2f}%)")
    print(f"   Absolute Gain:   {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)")
    print(f"   Relative Gain:   {f1_improvement_pct:+.2f}%")
    
    print(f"\n📊 Per-Class Improvements:")
    for i, name in enumerate(class_names):
        improvement = tta_results['per_class_f1'][i] - baseline_per_class_f1[i]
        marker = "🔥" if improvement > 0.02 else "✅" if improvement > 0 else "➖"
        print(f"   {name:<12} {baseline_per_class_f1[i]:.4f} → {tta_results['per_class_f1'][i]:.4f} ({improvement:+.4f}) {marker}")
    
    # Save TTA results
    print("\n💾 Saving TTA results...")
    
    results_dict = {
        'baseline': {
            'f1_macro': float(baseline_f1),
            'precision': float(baseline_precision),
            'recall': float(baseline_recall),
            'per_class_f1': {name: float(f1) for name, f1 in zip(class_names, baseline_per_class_f1)}
        },
        'tta': {
            'f1_macro': float(tta_results['f1_macro']),
            'precision': float(tta_results['precision']),
            'recall': float(tta_results['recall']),
            'per_class_f1': {name: float(f1) for name, f1 in zip(class_names, tta_results['per_class_f1'])}
        },
        'improvement': {
            'f1_absolute': float(f1_improvement),
            'f1_relative_pct': float(f1_improvement_pct)
        },
        'config': {
            'augmentations': 8,
            'models': 3,
            'total_predictions_per_image': 24
        }
    }
    
    with open('tta_evaluation_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"   ✅ Saved to tta_evaluation_results.json")
    
    # Save TTA predictions
    np.save('tta_predictions.npy', tta_results['predictions'])
    np.save('tta_probabilities.npy', tta_results['probabilities'])
    print(f"   ✅ Saved predictions and probabilities")
    
    print("\n" + "=" * 80)
    print("TTA EVALUATION COMPLETE!")
    print("=" * 80)
    
    if f1_improvement > 0:
        print(f"\n🎉 TTA improved performance by {f1_improvement*100:.2f}% F1!")
        print(f"   New best: {tta_results['f1_macro']*100:.2f}% Macro F1")
    else:
        print(f"\n⚠️ TTA did not improve performance (baseline was better)")
    
    print(f"\nNext steps:")
    print(f"1. Update api_ensemble.py to use TTA for inference")
    print(f"2. Implement other optimizations (class-balanced sampling, asymmetric loss)")
    print(f"3. Continue toward 80% F1 target!")


if __name__ == '__main__':
    main()
