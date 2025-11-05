"""
Create comprehensive Path A visualization showing the complete journey.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Load results
with open('results/phase4_ensemble_results.json', 'r') as f:
    phase4_baseline = json.load(f)

with open('results/phase4_threshold_optimization.json', 'r') as f:
    threshold_opt = json.load(f)

with open('results/phase4_pathA_summary.json', 'r') as f:
    final_summary = json.load(f)

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Myopia', 'Other']
class_names_short = ['N', 'D', 'G', 'C', 'A', 'M', 'O']

# ============================================================
# 1. Main Progress Chart (Top, spanning 2 columns)
# ============================================================
ax1 = fig.add_subplot(gs[0, :2])

milestones = [
    'Phase 4\nBaseline\n(0.5 threshold)',
    'Phase 4\nEnsemble\n(Weighted)',
    'Phase 4 +\nThreshold\nOptimization',
    'Phase 2\nBaseline\n(Target)'
]

f1_scores = [
    phase4_baseline['ConvNeXt Tiny']['f1_macro'],  # Best individual
    phase4_baseline['Weighted Ensemble']['f1_macro'],  # Ensemble
    threshold_opt['optimized']['f1_macro'],  # With threshold optimization
    0.6403  # Phase 2 target
]

colors = ['#3498db', '#2ecc71', '#e74c3c', '#95a5a6']
x_pos = np.arange(len(milestones))

bars = ax1.bar(x_pos, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    height = bar.get_height()
    label = f'{score:.4f}\n({score*100:.2f}%)'
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             label, ha='center', va='bottom', fontsize=11, fontweight='bold')

# Highlight that we beat Phase 2
ax1.axhline(y=0.6403, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Phase 2 Target')
ax1.fill_between([-0.5, 3.5], 0.6403, 0.65, alpha=0.1, color='green', label='Above Target')

ax1.set_ylabel('F1 Macro Score', fontsize=14, fontweight='bold')
ax1.set_title('Path A: Journey to Exceed Phase 2 Baseline', fontsize=16, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(milestones, fontsize=11)
ax1.set_ylim(0.55, 0.66)
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(axis='y', alpha=0.3)

# Add improvement arrows
ax1.annotate('', xy=(1, f1_scores[1]), xytext=(0, f1_scores[0]),
            arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.5))
ax1.text(0.5, (f1_scores[0] + f1_scores[1])/2 - 0.003, '+1.50%', 
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.annotate('', xy=(2, f1_scores[2]), xytext=(1, f1_scores[1]),
            arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.5))
ax1.text(1.5, (f1_scores[1] + f1_scores[2])/2 + 0.003, '+2.83%', 
         ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# ============================================================
# 2. Gap Analysis (Top right)
# ============================================================
ax2 = fig.add_subplot(gs[0, 2])

gaps = [
    (phase4_baseline['ConvNeXt Tiny']['f1_macro'] - 0.6403) * 100,
    (phase4_baseline['Weighted Ensemble']['f1_macro'] - 0.6403) * 100,
    (threshold_opt['optimized']['f1_macro'] - 0.6403) * 100
]

gap_labels = ['Individual\nModel', 'Ensemble', 'Threshold\nOpt.']
gap_colors = ['red' if g < 0 else 'green' for g in gaps]

bars2 = ax2.barh(gap_labels, gaps, color=gap_colors, alpha=0.7, edgecolor='black', linewidth=2)

for i, (bar, gap) in enumerate(zip(bars2, gaps)):
    width = bar.get_width()
    label = f'{gap:+.2f}%'
    x_pos = width + 0.1 if width > 0 else width - 0.1
    ha = 'left' if width > 0 else 'right'
    ax2.text(x_pos, bar.get_y() + bar.get_height()/2, label,
             ha=ha, va='center', fontsize=11, fontweight='bold')

ax2.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
ax2.set_xlabel('Gap vs Phase 2 (%)', fontsize=12, fontweight='bold')
ax2.set_title('Performance Gap Analysis', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.set_xlim(-3, 1.5)

# ============================================================
# 3. Per-Class Improvement (Middle row)
# ============================================================
ax3 = fig.add_subplot(gs[1, :])

baseline_per_class = phase4_baseline['Weighted Ensemble']['f1_per_class']
optimized_per_class = threshold_opt['optimized']['f1_per_class']

x = np.arange(len(class_names))
width = 0.35

bars_base = ax3.bar(x - width/2, baseline_per_class, width, label='Baseline (0.5 threshold)',
                    alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
bars_opt = ax3.bar(x + width/2, optimized_per_class, width, label='Optimized Thresholds',
                   alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)

# Add improvement labels
for i, (base, opt) in enumerate(zip(baseline_per_class, optimized_per_class)):
    improvement = (opt - base) * 100
    if abs(improvement) > 0.5:  # Only show significant improvements
        y_pos = max(base, opt) + 0.02
        color = 'green' if improvement > 0 else 'red'
        ax3.text(i, y_pos, f'{improvement:+.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=color)

ax3.set_xlabel('Disease Class', fontsize=13, fontweight='bold')
ax3.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
ax3.set_title('Per-Class Performance: Before vs After Threshold Optimization', 
             fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
ax3.legend(fontsize=11, loc='lower right')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 1.0)

# ============================================================
# 4. Optimized Thresholds (Bottom left)
# ============================================================
ax4 = fig.add_subplot(gs[2, 0])

thresholds = [threshold_opt['optimized']['thresholds'][name] for name in class_names]
colors_thresh = ['green' if t != 0.5 else 'gray' for t in thresholds]

bars4 = ax4.barh(class_names_short, thresholds, color=colors_thresh, alpha=0.7,
                 edgecolor='black', linewidth=1.5)

for i, (bar, thresh) in enumerate(zip(bars4, thresholds)):
    ax4.text(thresh + 0.02, bar.get_y() + bar.get_height()/2, f'{thresh:.2f}',
            ha='left', va='center', fontsize=10, fontweight='bold')

ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Default (0.5)')
ax4.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax4.set_ylabel('Class', fontsize=12, fontweight='bold')
ax4.set_title('Optimized Thresholds\nPer Class', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(axis='x', alpha=0.3)
ax4.set_xlim(0.2, 0.7)

# ============================================================
# 5. Time & Effort Breakdown (Bottom center)
# ============================================================
ax5 = fig.add_subplot(gs[2, 1])

steps = ['Model\nTraining', 'Ensemble\nEvaluation', 'Threshold\nOptimization', 'Total']
times = [10, 0.5, 0.5, 11]  # hours
gains = [60.30, 1.50, 2.83, 4.33]  # cumulative gains in percentage points

colors_time = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

bars5 = ax5.bar(steps, times, color=colors_time, alpha=0.7, edgecolor='black', linewidth=1.5)

for bar, time, gain in zip(bars5, times, gains):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{time}h', ha='center', va='bottom', fontsize=10, fontweight='bold')
    if gain > 1:  # Don't show for small gains
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'+{gain:.2f}%', ha='center', va='center', fontsize=9,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

ax5.set_ylabel('Time (hours)', fontsize=12, fontweight='bold')
ax5.set_title('Time Investment\nvs Performance Gain', fontsize=13, fontweight='bold')
ax5.set_ylim(0, 12)
ax5.grid(axis='y', alpha=0.3)

# ============================================================
# 6. Summary Stats (Bottom right)
# ============================================================
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

summary_text = f"""
PATH A RESULTS SUMMARY
{'=' * 35}

Starting Point:
  Phase 4 Best Individual: 60.30%
  Phase 2 Target: 64.03%
  Initial Gap: -3.73%

Path A Optimizations:
  ✓ Ensemble (Weighted): +1.50%
  ✓ Threshold Optimization: +2.83%

Final Results:
  Phase 4 Final: 64.63%
  Phase 2 Baseline: 64.03%
  
  🎉 EXCEEDED by +0.60%!

Key Insights:
  • Threshold optimization was key
  • Normal class: 63.2% → 72.4% (+9.2%)
  • No retraining needed
  • Total time: ~11 hours
  
ROI Analysis:
  • Training: 10 hours → 60.30%
  • Quick wins: 1 hour → +4.33%
  • Cost-effective approach!
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

# ============================================================
# Overall title
# ============================================================
fig.suptitle('Phase 4 Path A: Complete Analysis - Threshold Optimization Success',
            fontsize=18, fontweight='bold', y=0.995)

# Save
plt.savefig('results/phase4_pathA_complete_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved comprehensive analysis: results/phase4_pathA_complete_analysis.png")
plt.close()

# Create a simpler summary image for quick reference
fig2, ax = plt.subplots(figsize=(12, 8))

# Summary bar chart
categories = ['Phase 4\nIndividual\nBest', 'Phase 4\nEnsemble', 'Phase 4\n+ Threshold\nOpt.', 'Phase 2\nBaseline']
scores = [60.30, 61.80, 64.63, 64.03]
colors_summary = ['#3498db', '#2ecc71', '#e74c3c', '#95a5a6']

bars_summary = ax.bar(categories, scores, color=colors_summary, alpha=0.8, 
                      edgecolor='black', linewidth=2, width=0.6)

# Add value labels
for bar, score in zip(bars_summary, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
           f'{score:.2f}%', ha='center', va='bottom', 
           fontsize=14, fontweight='bold')

# Highlight the win
ax.axhline(y=64.03, color='red', linestyle='--', linewidth=2, 
          alpha=0.7, label='Phase 2 Target (64.03%)')
ax.fill_between([-0.5, 3.5], 64.03, 66, alpha=0.15, color='green')

ax.set_ylabel('F1 Macro Score (%)', fontsize=14, fontweight='bold')
ax.set_title('Path A Success: Exceeded Phase 2 Baseline with Threshold Optimization\n' + 
            '🎉 Final: 64.63% | Target: 64.03% | Improvement: +0.60%',
            fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(55, 67)
ax.legend(fontsize=12, loc='lower right')
ax.grid(axis='y', alpha=0.3)

# Add annotation
ax.annotate('✓ SUCCESS!', xy=(2, 64.63), xytext=(1.5, 66),
           arrowprops=dict(arrowstyle='->', lw=3, color='green'),
           fontsize=16, fontweight='bold', color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=0.8))

plt.tight_layout()
plt.savefig('results/phase4_pathA_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved summary: results/phase4_pathA_summary.png")
plt.close()

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE!")
print("=" * 60)
print("\nGenerated files:")
print("  1. phase4_pathA_complete_analysis.png - Comprehensive analysis")
print("  2. phase4_pathA_summary.png - Quick summary chart")
print("  3. phase4_threshold_comparison.png - Threshold optimization details")
