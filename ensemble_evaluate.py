"""
3-Model Ensemble Evaluation
============================

Tests multiple ensemble strategies with 3 models for maximum diversity:
- ResNet-50 (CNN)
- EfficientNet-B5 (CNN) 
- ViT-Base (Transformer)

Ensemble Methods:
1. Simple Average
2. Weighted Average (by F1 score)
3. Per-Class Weighted (use best model per class)

Evaluates on validation set and provides comprehensive analysis.
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


class EnsembleEvaluator:
    def __init__(self, device='mps'):
        self.device = torch.device(device) if torch.cuda.is_available() or device == 'mps' else torch.device('cpu')
        self.class_names = ['AMD', 'Diabetes', 'Glaucoma', 'Cataract', 'Myopia', 'Normal', 'Other']
        self.num_classes = len(self.class_names)
        
        # Model configurations
        self.models_config = {
            'resnet50': {
                'path': 'models_smart_exclusion/best_resnet50.pth',
                'thresholds_path': 'models_smart_exclusion/optimized_thresholds.json',
                'f1': 0.6728,
                'model_name': 'resnet50'
            },
            'efficientnet_b5': {
                'path': 'models_efficientnet_b5/best_efficientnet_b5.pth',
                'thresholds_path': 'models_efficientnet_b5/optimized_thresholds.json',
                'f1': 0.6877,
                'model_name': 'efficientnet_b5'
            },
            'vit_base': {
                'path': 'models_vit_base/best_vit_base.pth',
                'thresholds_path': 'models_vit_base/optimized_thresholds.json',
                'f1': 0.6768,
                'model_name': 'vit_base'
            }
        }
        
        # Per-class performance (for per-class weighting)
        self.per_class_f1 = {
            'resnet50': [0.5419, 0.5975, 0.5570, 0.7949, 0.8961, 0.7497, 0.5725],
            'efficientnet_b5': [0.5849, 0.6200, 0.5584, 0.8516, 0.8859, 0.7477, 0.5657],
            'vit_base': [0.5660, 0.5945, 0.5276, 0.8221, 0.9020, 0.7469, 0.5784]
        }
        
        self.models = {}
        self.thresholds = {}
        
    def load_models(self):
        """Load both models"""
        print("="*80)
        print("  LOADING MODELS")
        print("="*80)
        print()
        
        for model_key, config in self.models_config.items():
            print(f"Loading {model_key}...")
            
            # Load model
            model = create_model(config['model_name'], self.num_classes)
            checkpoint = torch.load(config['path'], map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            self.models[model_key] = model
            print(f"   ✅ Model loaded from {config['path']}")
            
            # Load thresholds
            with open(config['thresholds_path'], 'r') as f:
                thresh_data = json.load(f)
                thresholds = [thresh_data['thresholds'][name] for name in self.class_names]
                self.thresholds[model_key] = thresholds
            print(f"   ✅ Thresholds loaded from {config['thresholds_path']}")
            print()
    
    def get_predictions(self, val_dataset, batch_size=32):
        """Get predictions from both models"""
        print("="*80)
        print("  GETTING MODEL PREDICTIONS")
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
        return predictions, all_labels
    
    def evaluate_single_model(self, predictions, labels, model_key):
        """Evaluate a single model with its optimized thresholds"""
        probs = predictions[model_key]
        thresholds = self.thresholds[model_key]
        
        preds = np.zeros_like(probs)
        for i in range(self.num_classes):
            preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)
        
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        
        return f1_macro, f1_per_class, precision, recall, preds
    
    def ensemble_simple_average(self, predictions, labels):
        """Method 1: Simple average of probabilities"""
        print("-"*80)
        print("  METHOD 1: SIMPLE AVERAGE (3 Models)")
        print("-"*80)
        print()
        
        # Average probabilities
        avg_probs = np.mean([predictions['resnet50'], predictions['efficientnet_b5'], predictions['vit_base']], axis=0)
        
        # Use average of thresholds
        avg_thresholds = np.mean([
            self.thresholds['resnet50'],
            self.thresholds['efficientnet_b5'],
            self.thresholds['vit_base']
        ], axis=0)
        
        preds = np.zeros_like(avg_probs)
        for i in range(self.num_classes):
            preds[:, i] = (avg_probs[:, i] >= avg_thresholds[i]).astype(int)
        
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        
        return f1_macro, f1_per_class, precision, recall, preds
    
    def ensemble_weighted_average(self, predictions, labels):
        """Method 2: Weighted average by F1 scores"""
        print("-"*80)
        print("  METHOD 2: WEIGHTED AVERAGE (by F1)")
        print("-"*80)
        print()
        
        # Calculate weights based on F1 scores
        total_f1 = (self.models_config['resnet50']['f1'] + 
                    self.models_config['efficientnet_b5']['f1'] + 
                    self.models_config['vit_base']['f1'])
        w_resnet = self.models_config['resnet50']['f1'] / total_f1
        w_effnet = self.models_config['efficientnet_b5']['f1'] / total_f1
        w_vit = self.models_config['vit_base']['f1'] / total_f1
        
        print(f"   Weights: ResNet-50={w_resnet:.3f}, EfficientNet-B5={w_effnet:.3f}, ViT-Base={w_vit:.3f}")
        print()
        
        # Weighted average of probabilities
        weighted_probs = (w_resnet * predictions['resnet50'] + 
                         w_effnet * predictions['efficientnet_b5'] +
                         w_vit * predictions['vit_base'])
        
        # Weighted average of thresholds
        weighted_thresholds = (w_resnet * np.array(self.thresholds['resnet50']) +
                              w_effnet * np.array(self.thresholds['efficientnet_b5']) +
                              w_vit * np.array(self.thresholds['vit_base']))
        
        preds = np.zeros_like(weighted_probs)
        for i in range(self.num_classes):
            preds[:, i] = (weighted_probs[:, i] >= weighted_thresholds[i]).astype(int)
        
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        
        return f1_macro, f1_per_class, precision, recall, preds
    
    def ensemble_per_class_weighted(self, predictions, labels):
        """Method 3: Per-class weighted (use best model for each class)"""
        print("-"*80)
        print("  METHOD 3: PER-CLASS WEIGHTED (3 Models)")
        print("-"*80)
        print()
        
        # For each class, weight by per-class F1 performance
        weighted_probs = np.zeros_like(predictions['resnet50'])
        
        for i in range(self.num_classes):
            f1_resnet = self.per_class_f1['resnet50'][i]
            f1_effnet = self.per_class_f1['efficientnet_b5'][i]
            f1_vit = self.per_class_f1['vit_base'][i]
            total = f1_resnet + f1_effnet + f1_vit
            
            if total > 0:
                w_resnet = f1_resnet / total
                w_effnet = f1_effnet / total
                w_vit = f1_vit / total
            else:
                w_resnet = w_effnet = w_vit = 1.0/3.0
            
            weighted_probs[:, i] = (w_resnet * predictions['resnet50'][:, i] +
                                   w_effnet * predictions['efficientnet_b5'][:, i] +
                                   w_vit * predictions['vit_base'][:, i])
            
            print(f"   {self.class_names[i]:12s}: ResNet={w_resnet:.3f}, EfficientNet={w_effnet:.3f}, ViT={w_vit:.3f}")
        print()
        
        # Use weighted thresholds
        weighted_thresholds = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            f1_resnet = self.per_class_f1['resnet50'][i]
            f1_effnet = self.per_class_f1['efficientnet_b5'][i]
            f1_vit = self.per_class_f1['vit_base'][i]
            total = f1_resnet + f1_effnet + f1_vit
            
            if total > 0:
                w_resnet = f1_resnet / total
                w_effnet = f1_effnet / total
                w_vit = f1_vit / total
            else:
                w_resnet = w_effnet = w_vit = 1.0/3.0
            
            weighted_thresholds[i] = (w_resnet * self.thresholds['resnet50'][i] +
                                     w_effnet * self.thresholds['efficientnet_b5'][i] +
                                     w_vit * self.thresholds['vit_base'][i])
        
        preds = np.zeros_like(weighted_probs)
        for i in range(self.num_classes):
            preds[:, i] = (weighted_probs[:, i] >= weighted_thresholds[i]).astype(int)
        
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        
        return f1_macro, f1_per_class, precision, recall, preds
    
    def print_results(self, name, f1_macro, f1_per_class, precision, recall):
        """Print results in a nice format"""
        print(f"\n{name}")
        print(f"  Overall F1:  {f1_macro:.4f} ({f1_macro*100:.2f}%)")
        print(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)")
        print(f"\n  Per-Class F1:")
        for i, name in enumerate(self.class_names):
            print(f"    {name:12s}: {f1_per_class[i]:.4f} ({f1_per_class[i]*100:.2f}%)")
    
    def run_evaluation(self):
        """Run complete evaluation"""
        print("\n")
        print("="*80)
        print("  3-MODEL ENSEMBLE EVALUATION")
        print("  ResNet-50 (CNN) + EfficientNet-B5 (CNN) + ViT-Base (Transformer)")
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
        
        # Get predictions
        predictions, labels = self.get_predictions(val_dataset)
        
        # Evaluate individual models (baseline)
        print("="*80)
        print("  BASELINE: INDIVIDUAL MODEL PERFORMANCE")
        print("="*80)
        print()
        
        baseline_results = {}
        for model_key in self.models.keys():
            f1, f1_per_class, prec, rec, _ = self.evaluate_single_model(predictions, labels, model_key)
            baseline_results[model_key] = (f1, f1_per_class, prec, rec)
            self.print_results(model_key.upper(), f1, f1_per_class, prec, rec)
            print()
        
        # Evaluate ensemble methods
        print("\n")
        print("="*80)
        print("  ENSEMBLE METHODS")
        print("="*80)
        print()
        
        ensemble_results = {}
        
        # Method 1: Simple Average
        f1_simple, f1_per_simple, prec_simple, rec_simple, _ = self.ensemble_simple_average(predictions, labels)
        ensemble_results['simple'] = (f1_simple, f1_per_simple, prec_simple, rec_simple)
        self.print_results("SIMPLE AVERAGE", f1_simple, f1_per_simple, prec_simple, rec_simple)
        print()
        
        # Method 2: Weighted Average
        f1_weighted, f1_per_weighted, prec_weighted, rec_weighted, _ = self.ensemble_weighted_average(predictions, labels)
        ensemble_results['weighted'] = (f1_weighted, f1_per_weighted, prec_weighted, rec_weighted)
        self.print_results("WEIGHTED AVERAGE", f1_weighted, f1_per_weighted, prec_weighted, rec_weighted)
        print()
        
        # Method 3: Per-Class Weighted
        f1_perclass, f1_per_perclass, prec_perclass, rec_perclass, _ = self.ensemble_per_class_weighted(predictions, labels)
        ensemble_results['per_class'] = (f1_perclass, f1_per_perclass, prec_perclass, rec_perclass)
        self.print_results("PER-CLASS WEIGHTED", f1_perclass, f1_per_perclass, prec_perclass, rec_perclass)
        print()
        
        # Summary
        print("\n")
        print("="*80)
        print("  SUMMARY & RECOMMENDATIONS")
        print("="*80)
        print()
        
        best_f1 = max(baseline_results['resnet50'][0], baseline_results['efficientnet_b5'][0])
        print(f"Best Single Model F1:     {best_f1:.4f} ({best_f1*100:.2f}%)")
        print()
        
        print("Ensemble Results:")
        print(f"  Simple Average:         {f1_simple:.4f} ({f1_simple*100:.2f}%) [+{(f1_simple-best_f1)*100:+.2f}%]")
        print(f"  Weighted Average:       {f1_weighted:.4f} ({f1_weighted*100:.2f}%) [+{(f1_weighted-best_f1)*100:+.2f}%]")
        print(f"  Per-Class Weighted:     {f1_perclass:.4f} ({f1_perclass*100:.2f}%) [+{(f1_perclass-best_f1)*100:+.2f}%]")
        print()
        
        # Find best ensemble method
        best_ensemble_name = max(ensemble_results.keys(), key=lambda k: ensemble_results[k][0])
        best_ensemble_f1 = ensemble_results[best_ensemble_name][0]
        
        print(f"🏆 Best Ensemble Method: {best_ensemble_name.upper()}")
        print(f"   F1 Score: {best_ensemble_f1:.4f} ({best_ensemble_f1*100:.2f}%)")
        print(f"   Improvement over best single model: +{(best_ensemble_f1-best_f1)*100:.2f}%")
        print()
        
        # Recommendations
        print("="*80)
        print("  RECOMMENDATIONS")
        print("="*80)
        print()
        
        if best_ensemble_f1 >= 0.70:
            print(f"🎉 EXCELLENT RESULT! 3-model ensemble achieves {best_ensemble_f1*100:.2f}% F1")
            print()
            print(f"✅ TARGET ACHIEVED: {best_ensemble_f1*100:.2f}% exceeds 70% goal!")
            print()
            print("Next Steps:")
            print(f"  1. DEPLOY - This 3-model ensemble is production-ready")
            print(f"  2. DOCUMENT - Save model architecture and ensemble config")
            print(f"  3. MONITOR - Track performance on real-world data")
            print()
            print("Architectural Diversity:")
            print("  • ResNet-50: Deep CNN with residual connections")
            print("  • EfficientNet-B5: Efficient CNN with compound scaling")
            print("  • ViT-Base: Vision Transformer with self-attention")
            print()
            print("This combination provides excellent coverage of different")
            print("feature extraction strategies!")
            
        elif best_ensemble_f1 >= 0.68:
            gap = (0.70 - best_ensemble_f1) * 100
            print(f"⚠️  CLOSE TO TARGET - {best_ensemble_f1*100:.2f}% F1 (need +{gap:.2f}% to reach 70%)")
            print()
            print("Analysis:")
            print(f"  • 3-model ensemble with max diversity achieved {best_ensemble_f1*100:.2f}%")
            print(f"  • Just {gap:.2f}% short of 70% goal")
            print()
            print("Options:")
            print("  1. DEPLOY AS-IS - {:.2f}% is still a strong result".format(best_ensemble_f1*100))
            print("  2. TRY DATA AUGMENTATION - May gain 1-2%")
            print("  3. ADD 4TH MODEL (DINOv2) - May push over 70%")
            print()
            
            # Identify weak classes
            print("Weakest Classes (limiting performance):")
            avg_f1_per_class = (np.array(f1_per_simple) + np.array(f1_per_weighted) + np.array(f1_per_perclass)) / 3
            weak_indices = np.argsort(avg_f1_per_class)[:3]
            for idx in weak_indices:
                print(f"  • {self.class_names[idx]:12s}: {avg_f1_per_class[idx]:.4f} ({avg_f1_per_class[idx]*100:.2f}%)")
        
        else:
            print(f"⚠️  3-MODEL ENSEMBLE: {best_ensemble_f1*100:.2f}% F1")
            print()
            print("Analysis:")
            print("  • Added ViT-Base (Transformer) for architectural diversity")
            print("  • Ensemble includes: 2 CNNs + 1 Transformer")
            print()
            if best_ensemble_f1 < best_f1:
                print("⚠️  ENSEMBLE WORSE THAN SINGLE MODEL!")
                print("    This suggests models are making correlated errors")
                print()
            print("Recommendations:")
            print("  1. REVIEW DATA - May have data quality issues")
            print("  2. ANALYZE ERRORS - Where are models agreeing on wrong answers?")
            print("  3. TRY DIFFERENT FEATURES - Consider adding metadata")
        
        print()
        print("="*80)
        print("  EVALUATION COMPLETE")
        print("="*80)
        print()


def main():
    evaluator = EnsembleEvaluator(device='mps')
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()
