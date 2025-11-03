"""
Detailed augmentation analysis and quality assessment.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from src.augmented_dataset import AugmentedODIRDataset
from config import LABEL_COLUMNS


def analyze_augmentation_quality():
    """Detailed analysis of augmentation quality."""
    print("="*70)
    print("AUGMENTATION QUALITY ANALYSIS")
    print("="*70)
    
    # Load datasets
    print("\nLoading datasets...")
    dataset_no_aug = AugmentedODIRDataset(
        'preprocessed_data/train_images.npy',
        'preprocessed_data/train_labels.npy',
        augment=False
    )
    
    dataset_aug = AugmentedODIRDataset(
        'preprocessed_data/train_images.npy',
        'preprocessed_data/train_labels.npy',
        augment=True,
        augment_probability=0.5
    )
    
    # Sample 1000 images
    n_samples = 1000
    print(f"\nAnalyzing {n_samples} samples...")
    
    # Collect statistics
    original_stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'brightness': []
    }
    
    augmented_stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'brightness': []
    }
    
    for i in range(n_samples):
        # Original
        img_orig, _ = dataset_no_aug[i]
        img_orig_np = img_orig.numpy()
        original_stats['mean'].append(img_orig_np.mean())
        original_stats['std'].append(img_orig_np.std())
        original_stats['min'].append(img_orig_np.min())
        original_stats['max'].append(img_orig_np.max())
        original_stats['brightness'].append(img_orig_np.mean())
        
        # Augmented
        img_aug, _ = dataset_aug[i]
        img_aug_np = img_aug.numpy()
        augmented_stats['mean'].append(img_aug_np.mean())
        augmented_stats['std'].append(img_aug_np.std())
        augmented_stats['min'].append(img_aug_np.min())
        augmented_stats['max'].append(img_aug_np.max())
        augmented_stats['brightness'].append(img_aug_np.mean())
    
    # Convert to numpy
    for key in original_stats:
        original_stats[key] = np.array(original_stats[key])
        augmented_stats[key] = np.array(augmented_stats[key])
    
    # Print statistics
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<15} {'Original':<20} {'Augmented':<20} {'Change':<15}")
    print("-"*70)
    
    metrics = ['mean', 'std', 'min', 'max', 'brightness']
    for metric in metrics:
        orig = original_stats[metric].mean()
        aug = augmented_stats[metric].mean()
        change = aug - orig
        change_pct = (change / orig * 100) if orig > 0 else 0
        
        print(f"{metric.capitalize():<15} {orig:<20.4f} {aug:<20.4f} {change:+.4f} ({change_pct:+.1f}%)")
    
    # Check for outliers
    print("\n" + "="*70)
    print("OUTLIER DETECTION")
    print("="*70)
    
    # Check for extreme values
    extreme_bright = augmented_stats['brightness'] > 0.8
    extreme_dark = augmented_stats['brightness'] < 0.05
    extreme_std = augmented_stats['std'] > 0.4
    
    print(f"\nExtremely bright images (mean > 0.8): {extreme_bright.sum()} / {n_samples} ({extreme_bright.sum()/n_samples*100:.1f}%)")
    print(f"Extremely dark images (mean < 0.05): {extreme_dark.sum()} / {n_samples} ({extreme_dark.sum()/n_samples*100:.1f}%)")
    print(f"High variance images (std > 0.4): {extreme_std.sum()} / {n_samples} ({extreme_std.sum()/n_samples*100:.1f}%)")
    
    # Distribution comparison
    print("\n" + "="*70)
    print("DISTRIBUTION ANALYSIS")
    print("="*70)
    
    print("\nBrightness distribution:")
    print(f"  Original:  Mean={original_stats['brightness'].mean():.3f}, Std={original_stats['brightness'].std():.3f}")
    print(f"  Augmented: Mean={augmented_stats['brightness'].mean():.3f}, Std={augmented_stats['brightness'].std():.3f}")
    print(f"  → Diversity increase: {(augmented_stats['brightness'].std() - original_stats['brightness'].std()) / original_stats['brightness'].std() * 100:.1f}%")
    
    # Create comparison plots
    create_distribution_plots(original_stats, augmented_stats)
    
    # Quality assessment
    print("\n" + "="*70)
    print("QUALITY ASSESSMENT")
    print("="*70)
    
    assessments = []
    
    # Check 1: Mean brightness preserved
    mean_change = abs(augmented_stats['brightness'].mean() - original_stats['brightness'].mean())
    if mean_change < 0.05:
        assessments.append(("✅ PASS", "Mean brightness preserved", f"Change: {mean_change:.4f} < 0.05"))
    else:
        assessments.append(("⚠️  WARN", "Mean brightness changed", f"Change: {mean_change:.4f} > 0.05"))
    
    # Check 2: Diversity increased
    std_increase = augmented_stats['brightness'].std() - original_stats['brightness'].std()
    if std_increase > 0.005:
        assessments.append(("✅ PASS", "Diversity increased", f"Std increase: {std_increase:.4f}"))
    else:
        assessments.append(("⚠️  WARN", "Low diversity increase", f"Std increase: {std_increase:.4f}"))
    
    # Check 3: No extreme outliers
    outlier_pct = (extreme_bright.sum() + extreme_dark.sum()) / n_samples * 100
    if outlier_pct < 5:
        assessments.append(("✅ PASS", "Few outliers", f"{outlier_pct:.1f}% < 5%"))
    else:
        assessments.append(("⚠️  WARN", "Many outliers", f"{outlier_pct:.1f}% > 5%"))
    
    # Check 4: Values in valid range
    if augmented_stats['min'].min() >= 0 and augmented_stats['max'].max() <= 1:
        assessments.append(("✅ PASS", "Valid value range", "[0, 1]"))
    else:
        assessments.append(("❌ FAIL", "Invalid value range", f"[{augmented_stats['min'].min()}, {augmented_stats['max'].max()}]"))
    
    # Print assessments
    for status, check, detail in assessments:
        print(f"\n{status} {check}")
        print(f"     {detail}")
    
    # Overall verdict
    passed = sum(1 for s, _, _ in assessments if "PASS" in s)
    total = len(assessments)
    
    print("\n" + "="*70)
    print("OVERALL VERDICT")
    print("="*70)
    print(f"\nPassed: {passed}/{total} checks")
    
    if passed == total:
        print("\n✅ EXCELLENT: Augmentation quality is high!")
        print("   → Safe to proceed with training")
        print("   → Expected to improve generalization")
    elif passed >= total - 1:
        print("\n✅ GOOD: Augmentation quality is acceptable")
        print("   → Can proceed with training")
        print("   → Monitor training curves closely")
    else:
        print("\n⚠️  CAUTION: Some quality concerns")
        print("   → Review augmentation parameters")
        print("   → Consider reducing augmentation_probability")
    
    return augmented_stats, original_stats


def create_distribution_plots(original_stats, augmented_stats):
    """Create distribution comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Augmentation Impact on Image Statistics', fontsize=14, fontweight='bold')
    
    metrics = [
        ('brightness', 'Brightness Distribution'),
        ('std', 'Standard Deviation Distribution'),
        ('min', 'Minimum Value Distribution'),
        ('max', 'Maximum Value Distribution')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Plot histograms
        ax.hist(original_stats[metric], bins=50, alpha=0.5, label='Original', color='blue', density=True)
        ax.hist(augmented_stats[metric], bins=50, alpha=0.5, label='Augmented', color='red', density=True)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(metric.capitalize())
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        orig_mean = original_stats[metric].mean()
        aug_mean = augmented_stats[metric].mean()
        change = aug_mean - orig_mean
        
        stats_text = f"Original: μ={orig_mean:.3f}\nAugmented: μ={aug_mean:.3f}\nΔ={change:+.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = Path('results/augmentation_statistics.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved distribution plots to {output_path}")
    plt.close()


def show_side_by_side_comparison(num_samples=4):
    """Show side-by-side comparison of original vs augmented."""
    print("\n" + "="*70)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*70)
    
    # Load datasets
    dataset_orig = AugmentedODIRDataset(
        'preprocessed_data/train_images.npy',
        'preprocessed_data/train_labels.npy',
        augment=False
    )
    
    dataset_aug = AugmentedODIRDataset(
        'preprocessed_data/train_images.npy',
        'preprocessed_data/train_labels.npy',
        augment=True,
        augment_probability=0.8
    )
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    fig.suptitle('Original vs Single Augmented Version', fontsize=14, fontweight='bold')
    
    # Select random samples
    indices = np.random.choice(len(dataset_orig), num_samples, replace=False)
    
    for row, idx in enumerate(indices):
        # Original
        img_orig, label = dataset_orig[idx]
        img_orig_np = img_orig.permute(1, 2, 0).numpy()
        
        axes[row, 0].imshow(img_orig_np)
        axes[row, 0].set_title('Original', fontweight='bold')
        axes[row, 0].axis('off')
        
        # Stats
        stats_text = f"Mean: {img_orig_np.mean():.3f}\nStd: {img_orig_np.std():.3f}"
        axes[row, 0].text(0.02, 0.98, stats_text, transform=axes[row, 0].transAxes,
                         verticalalignment='top', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Augmented
        img_aug, _ = dataset_aug[idx]
        img_aug_np = img_aug.permute(1, 2, 0).numpy()
        
        axes[row, 1].imshow(img_aug_np)
        axes[row, 1].set_title('Augmented', fontweight='bold')
        axes[row, 1].axis('off')
        
        # Stats
        stats_text = f"Mean: {img_aug_np.mean():.3f}\nStd: {img_aug_np.std():.3f}"
        axes[row, 1].text(0.02, 0.98, stats_text, transform=axes[row, 1].transAxes,
                         verticalalignment='top', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Label info
        disease_names = [LABEL_COLUMNS[i] for i, val in enumerate(label.numpy()) if val == 1]
        label_text = ', '.join(disease_names) if disease_names else 'Unknown'
        fig.text(0.5, 0.96 - (row * 0.24), f"Labels: {label_text}",
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    output_path = Path('results/augmentation_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to {output_path}")
    plt.close()


if __name__ == '__main__':
    try:
        print("\n🔬 Running comprehensive augmentation analysis...\n")
        
        # 1. Quality analysis
        aug_stats, orig_stats = analyze_augmentation_quality()
        
        # 2. Side-by-side comparison
        show_side_by_side_comparison(num_samples=4)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        print("\nGenerated files:")
        print("  - results/augmentation_visualization.png (6 versions per image)")
        print("  - results/augmentation_statistics.png (distribution plots)")
        print("  - results/augmentation_comparison.png (side-by-side)")
        
        print("\n📊 Review these visualizations to verify:")
        print("  1. Augmentations preserve diagnostic features")
        print("  2. Color changes are subtle (retinal color is important)")
        print("  3. No extreme distortions or artifacts")
        print("  4. Images remain recognizable")
        
        print("\n🚀 If everything looks good, proceed with training:")
        print("   python src/train.py")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
