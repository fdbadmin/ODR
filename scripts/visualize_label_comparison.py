"""
Visualize comparison between different labeling schemes.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


def create_comparison_plot():
    """Create visual comparison of different label systems."""
    # Load metadata
    metadata_path = Path('results/simplified_labels/metadata.json')
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Load current 7-class distribution
    import pandas as pd
    df = pd.read_excel('ODIR-5K/data.xlsx')
    current_dist = {
        'Normal': int((df['N'] == 1).sum()),
        'Diabetes': int((df['D'] == 1).sum()),
        'Glaucoma': int((df['G'] == 1).sum()),
        'Cataract': int((df['C'] == 1).sum()),
        'AMD': int((df['A'] == 1).sum()),
        'Myopia': int((df['M'] == 1).sum()),
        'Other': int((df['O'] == 1).sum())
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comparison of Different Label Systems for ODIR-5K Dataset', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Current 7-class (multi-label)
    ax = axes[0, 0]
    labels = list(current_dist.keys())
    values = list(current_dist.values())
    colors = plt.cm.Set3(range(len(labels)))
    
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xlabel('Disease Class', fontweight='bold')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Current: 7-Class Multi-Label\n(Labels not mutually exclusive)', fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Binary
    ax = axes[0, 1]
    data = metadata['binary']
    labels = data['class_names']
    values = [data['distribution']['Normal'], data['distribution']['Disease']]
    colors = ['#2ecc71', '#e74c3c']
    
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax.set_title('Option 1: Binary Classification\n(Normal vs Disease)', fontweight='bold')
    
    # Plot 3: 3-class
    ax = axes[0, 2]
    data = metadata['multiclass_3']
    labels = data['class_names']
    values = [data['distribution']['Normal'], 
              data['distribution']['DR'],
              data['distribution']['Other']]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax.set_title('Option 2: 3-Class\n(Normal, DR, Other)', fontweight='bold')
    
    # Plot 4: 5-class
    ax = axes[1, 0]
    data = metadata['major_diseases_5']
    labels = data['class_names']
    values = [data['distribution']['Normal'],
              data['distribution']['DR'],
              data['distribution']['Glaucoma'],
              data['distribution']['AMD'],
              data['distribution']['Other']]
    colors = plt.cm.Set2(range(len(labels)))
    
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xlabel('Disease Class', fontweight='bold')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Option 3: 5-Class Major Diseases\n(Normal, DR, Glaucoma, AMD, Other)', 
                 fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/3500*100:.1f}%)',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 5: Severity-based
    ax = axes[1, 1]
    data = metadata['severity_4']
    labels = data['class_names']
    values = [data['distribution']['Normal'],
              data['distribution']['Mild'],
              data['distribution']['Moderate'],
              data['distribution']['Severe']]
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#c0392b']
    
    # Remove 0 values and corresponding labels/colors
    non_zero_idx = [i for i, v in enumerate(values) if v > 0]
    values = [values[i] for i in non_zero_idx]
    labels = [labels[i] for i in non_zero_idx]
    colors = [colors[i] for i in non_zero_idx]
    
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax.set_title('Option 4: Severity-Based\n(Normal, Mild, Moderate, Severe)', 
                 fontweight='bold')
    
    # Plot 6: Summary comparison table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_data = [
        ['Label System', 'Classes', 'Balance', 'Complexity'],
        ['Current (7-class)', '7', 'Poor', 'High'],
        ['Binary', '2', 'Moderate', 'Very Low'],
        ['3-Class', '3', 'Good', 'Low'],
        ['5-Class', '5', 'Fair', 'Medium'],
        ['Severity', '3-4', 'Good', 'Low']
    ]
    
    table = ax.table(cellText=summary_data, loc='center', cellLoc='center',
                     colWidths=[0.3, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color data rows alternately
    for i in range(1, 6):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax.set_title('Summary Comparison', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('results/simplified_labels/label_systems_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {output_path}")
    
    plt.show()


def print_recommendations():
    """Print detailed recommendations."""
    print("\n" + "="*80)
    print("DETAILED RECOMMENDATIONS")
    print("="*80)
    
    print("""
Based on the analysis of columns F and G (diagnostic keywords):

╔══════════════════════════════════════════════════════════════════════╗
║                        COMPARISON SUMMARY                            ║
╚══════════════════════════════════════════════════════════════════════╝

CURRENT SYSTEM (7-class multi-label):
  ✓ Comprehensive disease coverage
  ✗ Complex multi-label classification
  ✗ Class imbalance (Myopia: 1014 vs AMD: 424)
  ✗ Requires threshold optimization per class
  → Current F1: 91.51% (with optimal thresholds)

OPTION 1: BINARY (Normal vs Disease)
  Distribution: 29% Normal, 71% Disease
  ✓ Simplest possible model
  ✓ Good for screening: "Does this need attention?"
  ✓ High accuracy expected (>95%)
  ✗ No specific disease information
  ✗ Can't guide treatment decisions
  
  USE CASE: First-stage screening in low-resource settings

OPTION 2: 3-CLASS (Normal, DR, Other) ⭐ RECOMMENDED
  Distribution: 29% Normal, 34% DR, 37% Other
  ✓ Well balanced (best among all options)
  ✓ Separates most common disease (Diabetic Retinopathy)
  ✓ Simpler than current system
  ✓ Still clinically useful
  ✓ Standard single-label classification
  ✗ Lumps all non-DR diseases together
  
  USE CASE: Diabetes screening programs, general screening

OPTION 3: 5-CLASS (Normal, DR, Glaucoma, AMD, Other)
  Distribution: 29% Normal, 34% DR, 5% Glaucoma, 4% AMD, 29% Other
  ✓ Covers major sight-threatening diseases
  ✓ More informative than 3-class
  ✗ Glaucoma and AMD are quite rare (4-5%)
  ✗ "Other" is still a large catch-all (29%)
  
  USE CASE: Comprehensive eye screening with major disease focus

OPTION 4: SEVERITY-BASED (Normal, Mild, Moderate, Severe)
  Distribution: 29% Normal, 40% Moderate, 31% Severe
  ✓ Clinically meaningful severity progression
  ✓ Could prioritize treatment urgency
  ✗ Loses specific disease information
  ✗ "Mild" class has very few samples
  
  USE CASE: Triage system for treatment priority

╔══════════════════════════════════════════════════════════════════════╗
║                     MY RECOMMENDATION                                ║
╚══════════════════════════════════════════════════════════════════════╝

→ START WITH OPTION 2 (3-CLASS)

Reasons:
1. BEST BALANCED: 29%, 34%, 37% - nearly perfect balance
2. SIMPLER: Single-label classification (no threshold optimization needed)
3. CLINICALLY USEFUL: Separates DR (most common, diabetes-related)
4. EASIER TO TRAIN: Expected accuracy >93-95% with simpler model
5. FASTER INFERENCE: Single model, no ensemble needed
6. EASIER TO INTERPRET: Clear decision boundaries

Training Strategy for 3-Class:
  • Use standard cross-entropy loss (no need for class weights)
  • Single ResNet50 or EfficientNet should suffice
  • No need for ensemble (balanced classes)
  • No threshold optimization needed
  • Expected training time: 2-3 hours (vs 15+ for current)
  • Expected F1 score: 93-96%

Comparison with Current System:
  Current:  7 classes, multi-label, needs ensemble + thresholds → 91.51% F1
  3-Class:  3 classes, single-label, balanced → Expected 93-96% F1

╔══════════════════════════════════════════════════════════════════════╗
║                    NEXT STEPS (if interested)                        ║
╚══════════════════════════════════════════════════════════════════════╝

If you want to try the 3-class system:

1. I can create a new preprocessing script for 3-class labels
2. Train a single model (ResNet50 or EfficientNet)
3. Compare performance with current 7-class system
4. Deploy whichever works better

Would you like me to set this up? It would be a good comparison to see
if simplification improves performance!
""")


if __name__ == '__main__':
    create_comparison_plot()
    print_recommendations()
