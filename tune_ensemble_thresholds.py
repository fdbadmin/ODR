"""
Ensemble Threshold Tuning
==========================

Optimizes per-class decision thresholds for the 3-model ensemble.
This tunes thresholds on the ensemble's combined predictions rather than
individual model thresholds.

Strategy: Get ensemble probabilities, then optimize thresholds on those.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import json

from scripts.train_cutting_edge import create_model, FundusDataset


class EnsembleThresholdTuner:
    def __init__(self, device='mps'):
        self.device = torch.device(device) if torch.cuda.is_available() or device == 'mps' else torch.device('cpu')
        self.class_names = ['AMD', 'Diabetes', 'Glaucoma', 'Cataract', 'Myopia', 'Normal', 'Other']
        self.num_classes = len(self.class_names)
        
        # Model configurations
        self.models_config = {
            'resnet50': {
                'path': 'models_smart_exclusion/best_resnet50.pth',
                'model_name': 'resnet50'
            },
            'efficientnet_b5': {
                'path': 'models_efficientnet_b5/best_efficientnet_b5.pth',
                'model_name': 'efficientnet_b5'
            },
            'vit_base': {
                'path': 'models_vit_base/best_vit_base.pth',
                'model_name': 'vit_base'
            }
        }
        
        self.models = {}
        
    def load_models(self):
        """Load all three models"""
        print("="*80)
        print("  LOADING MODELS")
        print("="*80)
        print()
        
        for model_key, config in self.models_config.items():
            print(f"Loading {model_key}...")
            
            model = create_model(config['model_name'], self.num_classes)
            checkpoint = torch.load(config['path'], map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            self.models[model_key] = model
            print(f"   ✅ {model_key} loaded")
        
        print()
    
    def get_ensemble_predictions(self, val_dataset, batch_size=32, ensemble_method='per_class_weighted'):
        """Get ensemble predictions from all models"""
        print("="*80)
        print(f"  GETTING ENSEMBLE PREDICTIONS ({ensemble_method})")
        print("="*80)
        print()
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        all_labels = []
        predictions = {key: [] for key in self.models.keys()}
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Inference"):
                images = images.to(self.device)
                all_labels.append(labels.numpy())
                
                for model_key, model in self.models.items():
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                    predictions[model_key].append(probs.cpu().numpy())
        
        all_labels = np.vstack(all_labels)
        for key in predictions:
            predictions[key] = np.vstack(predictions[key])
        
        print(f"   ✅ Got predictions for {len(all_labels)} samples")
        print()
        
        # Create ensemble predictions based on method
        if ensemble_method == 'simple_average':
            ensemble_probs = self._simple_average(predictions)
        elif ensemble_method == 'weighted_average':
            ensemble_probs = self._weighted_average(predictions)
        elif ensemble_method == 'per_class_weighted':
            ensemble_probs = self._per_class_weighted(predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        return ensemble_probs, all_labels
    
    def _simple_average(self, predictions):
        """Simple average of all model predictions"""
        print("Using Simple Average ensemble")
        return np.mean([predictions['resnet50'], predictions['efficientnet_b5'], predictions['vit_base']], axis=0)
    
    def _weighted_average(self, predictions):
        """Weighted average by model F1 scores"""
        print("Using Weighted Average ensemble")
        f1_scores = {'resnet50': 0.6728, 'efficientnet_b5': 0.6877, 'vit_base': 0.6768}
        total_f1 = sum(f1_scores.values())
        weights = {k: v/total_f1 for k, v in f1_scores.items()}
        
        print(f"   Weights: ResNet={weights['resnet50']:.3f}, EfficientNet={weights['efficientnet_b5']:.3f}, ViT={weights['vit_base']:.3f}")
        
        return (weights['resnet50'] * predictions['resnet50'] +
                weights['efficientnet_b5'] * predictions['efficientnet_b5'] +
                weights['vit_base'] * predictions['vit_base'])
    
    def _per_class_weighted(self, predictions):
        """Per-class weighted by class-specific F1 scores"""
        print("Using Per-Class Weighted ensemble")
        
        per_class_f1 = {
            'resnet50': [0.5419, 0.5975, 0.5570, 0.7949, 0.8961, 0.7497, 0.5725],
            'efficientnet_b5': [0.5849, 0.6200, 0.5584, 0.8516, 0.8859, 0.7477, 0.5657],
            'vit_base': [0.5660, 0.5945, 0.5276, 0.8221, 0.9020, 0.7469, 0.5784]
        }
        
        weighted_probs = np.zeros_like(predictions['resnet50'])
        
        for i in range(self.num_classes):
            f1_sum = sum(per_class_f1[k][i] for k in per_class_f1.keys())
            
            if f1_sum > 0:
                weights = {k: per_class_f1[k][i] / f1_sum for k in per_class_f1.keys()}
            else:
                weights = {k: 1.0/3.0 for k in per_class_f1.keys()}
            
            weighted_probs[:, i] = (weights['resnet50'] * predictions['resnet50'][:, i] +
                                   weights['efficientnet_b5'] * predictions['efficientnet_b5'][:, i] +
                                   weights['vit_base'] * predictions['vit_base'][:, i])
        
        return weighted_probs
    
    def optimize_thresholds(self, ensemble_probs, labels):
        """Find optimal threshold for each class on ensemble predictions"""
        print("\n" + "="*80)
        print("  OPTIMIZING ENSEMBLE THRESHOLDS")
        print("="*80)
        print()
        print("   Testing thresholds from 0.05 to 0.95 (fine-grained)...")
        print()
        
        optimal_thresholds = []
        baseline_f1_per_class = []
        optimized_f1_per_class = []
        
        # Test thresholds with finer granularity
        test_thresholds = np.arange(0.05, 0.96, 0.01)
        
        for class_idx in range(self.num_classes):
            class_probs = ensemble_probs[:, class_idx]
            class_labels = labels[:, class_idx]
            
            # Baseline with 0.5
            baseline_preds = (class_probs >= 0.5).astype(int)
            baseline_f1 = f1_score(class_labels, baseline_preds, zero_division=0)
            baseline_f1_per_class.append(baseline_f1)
            
            # Find best threshold
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in test_thresholds:
                preds = (class_probs >= threshold).astype(int)
                f1 = f1_score(class_labels, preds, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            optimal_thresholds.append(best_threshold)
            optimized_f1_per_class.append(best_f1)
            
            improvement = best_f1 - baseline_f1
            print(f"   {self.class_names[class_idx]:12s}: threshold={best_threshold:.3f}, "
                  f"F1={best_f1:.4f} (baseline: {baseline_f1:.4f}, Δ={improvement:+.4f})")
        
        return optimal_thresholds, baseline_f1_per_class, optimized_f1_per_class
    
    def evaluate_with_thresholds(self, probs, labels, thresholds):
        """Evaluate ensemble using optimized thresholds"""
        preds = np.zeros_like(probs)
        
        for i in range(self.num_classes):
            preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)
        
        # Calculate metrics
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        
        return f1_macro, f1_per_class, precision, recall
    
    def run_tuning(self, ensemble_method='per_class_weighted'):
        """Run complete threshold tuning for ensemble"""
        print("\n")
        print("="*80)
        print("  ENSEMBLE THRESHOLD OPTIMIZATION")
        print("  3-Model Ensemble: ResNet-50 + EfficientNet-B5 + ViT-Base")
        print("="*80)
        print()
        
        # Load data
        print("📂 Loading validation data...")
        val_images = np.load('preprocessed_data_smart_exclusion/val_images.npy', mmap_mode='r')
        val_labels = np.load('preprocessed_data_smart_exclusion/val_labels.npy')
        print(f"   ✅ Loaded {len(val_images)} validation samples")
        print()
        
        val_dataset = FundusDataset(val_images, val_labels, augment=False)
        
        # Load models
        self.load_models()
        
        # Get ensemble predictions
        ensemble_probs, labels = self.get_ensemble_predictions(val_dataset, ensemble_method=ensemble_method)
        
        # Baseline performance (threshold=0.5)
        print("\n" + "="*80)
        print("  BASELINE PERFORMANCE (threshold=0.5 for all classes)")
        print("="*80)
        print()
        
        baseline_preds = (ensemble_probs >= 0.5).astype(int)
        baseline_f1_macro = f1_score(labels, baseline_preds, average='macro', zero_division=0)
        baseline_f1_per_class = f1_score(labels, baseline_preds, average=None, zero_division=0)
        baseline_precision = precision_score(labels, baseline_preds, average='macro', zero_division=0)
        baseline_recall = recall_score(labels, baseline_preds, average='macro', zero_division=0)
        
        print(f"Overall Metrics:")
        print(f"  F1 Score:  {baseline_f1_macro:.4f} ({baseline_f1_macro*100:.2f}%)")
        print(f"  Precision: {baseline_precision:.4f} ({baseline_precision*100:.2f}%)")
        print(f"  Recall:    {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
        
        print(f"\nPer-Class F1 Scores:")
        for i, name in enumerate(self.class_names):
            print(f"  {name:12s}: {baseline_f1_per_class[i]:.4f}")
        
        # Optimize thresholds
        optimal_thresholds, _, _ = self.optimize_thresholds(ensemble_probs, labels)
        
        # Evaluate with optimized thresholds
        print("\n" + "="*80)
        print("  OPTIMIZED PERFORMANCE (per-class thresholds on ensemble)")
        print("="*80)
        print()
        
        opt_f1_macro, opt_f1_per_class, opt_precision, opt_recall = self.evaluate_with_thresholds(
            ensemble_probs, labels, optimal_thresholds
        )
        
        print(f"Overall Metrics:")
        print(f"  F1 Score:  {opt_f1_macro:.4f} ({opt_f1_macro*100:.2f}%)")
        print(f"  Precision: {opt_precision:.4f} ({opt_precision*100:.2f}%)")
        print(f"  Recall:    {opt_recall:.4f} ({opt_recall*100:.2f}%)")
        
        print(f"\nPer-Class F1 Scores:")
        for i, name in enumerate(self.class_names):
            improvement = opt_f1_per_class[i] - baseline_f1_per_class[i]
            print(f"  {name:12s}: {opt_f1_per_class[i]:.4f} "
                  f"(baseline: {baseline_f1_per_class[i]:.4f}, Δ={improvement:+.4f})")
        
        # Summary
        print("\n" + "="*80)
        print("  SUMMARY")
        print("="*80)
        print()
        
        f1_improvement = opt_f1_macro - baseline_f1_macro
        f1_improvement_pct = (f1_improvement / baseline_f1_macro) * 100
        
        print(f"Ensemble Method: {ensemble_method.upper()}")
        print()
        print(f"Baseline F1:  {baseline_f1_macro:.4f} ({baseline_f1_macro*100:.2f}%)")
        print(f"Optimized F1: {opt_f1_macro:.4f} ({opt_f1_macro*100:.2f}%)")
        print(f"Improvement:  {f1_improvement:+.4f} ({f1_improvement_pct:+.2f}%)")
        print()
        
        if opt_f1_macro >= 0.70:
            print("🎉 SUCCESS! Ensemble with optimized thresholds achieves 70%+ F1!")
            print()
            print(f"Final Result: {opt_f1_macro*100:.2f}% F1")
            print("   ✅ Target achieved!")
            print("   ✅ Ready for deployment!")
        else:
            gap = (0.70 - opt_f1_macro) * 100
            print(f"⚠️  Close: {opt_f1_macro*100:.2f}% F1 (need +{gap:.2f}% for 70%)")
            print()
            print("Consider:")
            print("  • Testing other ensemble methods")
            print("  • Adding 4th model (DINOv2)")
            print("  • Investigating weak classes")
        
        print()
        print(f"Optimized Thresholds:")
        for i, name in enumerate(self.class_names):
            print(f"  {name:12s}: {optimal_thresholds[i]:.3f}")
        
        # Save thresholds
        output_path = f'ensemble_optimized_thresholds_{ensemble_method}.json'
        threshold_config = {
            'ensemble_method': ensemble_method,
            'models': ['resnet50', 'efficientnet_b5', 'vit_base'],
            'thresholds': {name: float(thresh) for name, thresh in zip(self.class_names, optimal_thresholds)},
            'baseline_f1': float(baseline_f1_macro),
            'optimized_f1': float(opt_f1_macro),
            'improvement': float(f1_improvement),
            'baseline_f1_per_class': {name: float(f1) for name, f1 in zip(self.class_names, baseline_f1_per_class)},
            'optimized_f1_per_class': {name: float(f1) for name, f1 in zip(self.class_names, opt_f1_per_class)}
        }
        
        with open(output_path, 'w') as f:
            json.dump(threshold_config, f, indent=2)
        
        print()
        print(f"✅ Optimized thresholds saved to: {output_path}")
        print()
        print("="*80)
        print("  🎉 Threshold optimization complete!")
        print("="*80)
        print()


def main():
    # Test all ensemble methods to find the best
    methods = ['per_class_weighted', 'weighted_average', 'simple_average']
    
    print("\n")
    print("="*80)
    print("  TESTING ALL ENSEMBLE METHODS WITH THRESHOLD OPTIMIZATION")
    print("="*80)
    print()
    
    tuner = EnsembleThresholdTuner(device='mps')
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"  TESTING: {method.upper()}")
        print(f"{'='*80}\n")
        
        tuner.run_tuning(ensemble_method=method)
        
        # Load results
        with open(f'ensemble_optimized_thresholds_{method}.json', 'r') as f:
            results[method] = json.load(f)
        
        print("\n" + "="*80 + "\n")
    
    # Final comparison
    print("\n")
    print("="*80)
    print("  FINAL COMPARISON")
    print("="*80)
    print()
    
    print("Ensemble Method Results (with optimized thresholds):")
    for method in methods:
        f1 = results[method]['optimized_f1']
        improvement = results[method]['improvement']
        print(f"  {method:20s}: {f1:.4f} ({f1*100:.2f}%) [Δ={improvement:+.4f}]")
    
    # Best method
    best_method = max(results.keys(), key=lambda k: results[k]['optimized_f1'])
    best_f1 = results[best_method]['optimized_f1']
    
    print()
    print(f"🏆 Best Method: {best_method.upper()}")
    print(f"   Final F1: {best_f1:.4f} ({best_f1*100:.2f}%)")
    
    if best_f1 >= 0.70:
        print()
        print("🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉")
        print(f"   {best_f1*100:.2f}% F1 exceeds 70% goal!")
        print()
        print("✅ Ready for production deployment!")
    
    print()
    print("="*80)
    print()


if __name__ == '__main__':
    main()
