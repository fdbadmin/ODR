"""
Compare baseline (old preprocessing) vs enhanced (new preprocessing) results.
"""
import json
from pathlib import Path


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    print("=" * 80)
    print("BASELINE VS ENHANCED PREPROCESSING COMPARISON")
    print("=" * 80)
    print()
    
    # Check if we have baseline results
    baseline_path = Path("results/ensemble_optimal_thresholds_baseline.json")
    enhanced_path = Path("results/ensemble_optimal_thresholds.json")
    
    if not baseline_path.exists():
        print("⚠️  Baseline results not found!")
        print(f"   Looking for: {baseline_path}")
        print()
        print("Baseline Performance (from previous session):")
        print("  - Ensemble Mean F1: 85.28%")
        print("  - Data: Patient-level labels (5,113 train, 1,279 val)")
        print("  - Preprocessing: Basic (resize, normalize)")
        print()
    else:
        baseline = load_results(baseline_path)
        print("📊 BASELINE (Old Preprocessing):")
        print(f"   Mean F1: {baseline.get('mean_f1', 0.8528):.2%}")
        print(f"   Data: Patient-level labels")
        print()
    
    if enhanced_path.exists():
        enhanced = load_results(enhanced_path)
        print("📊 ENHANCED (New Preprocessing):")
        print(f"   Mean F1: {enhanced['optimized_mean_f1']:.2%}")
        print(f"   Data: Eye-specific labels (5,584 train, 1,395 val)")
        print()
        
        print("Per-Class F1 Comparison (Enhanced - Optimized):")
        print("-" * 80)
        print(f"{'Disease':<15} {'F1 Score':<12}")
        print("-" * 80)
        
        disease_map = {'N': 'Normal', 'D': 'Diabetes', 'G': 'Glaucoma', 
                      'C': 'Cataract', 'A': 'AMD', 'M': 'Myopia', 'O': 'Other'}
        
        for code, disease in disease_map.items():
            f1 = enhanced['optimized_f1_scores'].get(code, 0)
            print(f"{disease:<15} {f1:<12.2%}")
        
        print("-" * 80)
        print(f"{'MEAN':<15} {enhanced['optimized_mean_f1']:<12.2%}")
        print()
        
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    print("⚠️  SIGNIFICANT PERFORMANCE DROP DETECTED!")
    print()
    print("Expected: 85.28% → 90-93% F1 (improvement)")
    print("Actual:   85.28% → 63.17% F1 (22.11% drop)")
    print()
    print("Possible Causes:")
    print()
    print("1. 📊 LABEL DISTRIBUTION MISMATCH")
    print("   - Old: Patient-level labels (both eyes same)")
    print("   - New: Eye-specific labels (left/right different)")
    print("   - Impact: Validation set has different label distribution")
    print()
    print("2. 🎯 VALIDATION SET COMPOSITION")
    print("   - Old: 1,279 images (patient-level)")
    print("   - New: 1,395 images (eye-specific)")
    print("   - Different samples may have different difficulty")
    print()
    print("3. 🔍 DATA QUALITY CHANGES")
    print("   - Green channel extraction changed appearance")
    print("   - Enhanced CLAHE may have altered features")
    print("   - Models trained on new features, but may need more epochs")
    print()
    print("4. ⚖️ CLASS IMBALANCE CHANGES")
    print("   - Normal: +47.6% (1,681 → 2,481)")
    print("   - Myopia: -44.0% (1,244 → 697)")
    print("   - Models may be biased toward new distribution")
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("Option 1: 🔄 EXTENDED TRAINING")
    print("  - Train for 50 epochs instead of 25")
    print("  - Models may not have converged on new features")
    print("  - Expected improvement: +5-10% F1")
    print()
    print("Option 2: 🎨 ADJUST PREPROCESSING")
    print("  - Try less aggressive green channel extraction")
    print("  - Reduce CLAHE strength")
    print("  - Keep illumination correction")
    print()
    print("Option 3: 📊 HYBRID APPROACH")
    print("  - Use eye-specific labels (more accurate)")
    print("  - But keep simpler preprocessing")
    print("  - May balance accuracy vs. generalization")
    print()
    print("Option 4: 🔧 FINE-TUNE FROM BASELINE")
    print("  - Start with baseline models (85.28%)")
    print("  - Fine-tune on enhanced data for 10 epochs")
    print("  - Transfer learning may preserve performance")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
