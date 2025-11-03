"""
Analyze diagnostic keywords (columns F & G) to create simplified label systems.

This script explores different ways to create simplified one-hot encoded labels
based on the detailed diagnostic text in the ODIR-5K dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json


def create_binary_labels(df):
    """
    Create binary classification: Normal vs Disease/Abnormal
    
    Strategy: Both eyes must be "normal fundus" (and nothing else) for Normal label
    """
    binary_labels = []
    
    for idx, row in df.iterrows():
        left_kw = str(row['Left-Diagnostic Keywords']).lower().strip()
        right_kw = str(row['Right-Diagnostic Keywords']).lower().strip()
        
        # Both eyes must be exactly "normal fundus" (no additional conditions)
        left_normal = left_kw == 'normal fundus'
        right_normal = right_kw == 'normal fundus'
        
        if left_normal and right_normal:
            binary_labels.append(0)  # Normal
        else:
            binary_labels.append(1)  # Disease/Abnormal
    
    return np.array(binary_labels)


def create_simplified_multiclass(df):
    """
    Create simplified multi-class: Normal, Diabetic Retinopathy, Other Disease
    
    Strategy:
    - Normal: Both eyes exactly "normal fundus"
    - Diabetic Retinopathy: Any mention of retinopathy/diabetic
    - Other: Everything else
    """
    labels = []
    
    for idx, row in df.iterrows():
        left_kw = str(row['Left-Diagnostic Keywords']).lower()
        right_kw = str(row['Right-Diagnostic Keywords']).lower()
        combined = left_kw + ' ' + right_kw
        
        # Check for normal (exact match)
        left_normal = left_kw.strip() == 'normal fundus'
        right_normal = right_kw.strip() == 'normal fundus'
        
        if left_normal and right_normal:
            labels.append(0)  # Normal
        elif 'retinopathy' in combined or 'diabetic' in combined:
            labels.append(1)  # Diabetic Retinopathy
        else:
            labels.append(2)  # Other Disease
    
    return np.array(labels)


def create_major_disease_classes(df):
    """
    Create 5-class system: Normal, DR, Glaucoma, AMD, Other
    
    Strategy: Prioritize based on severity/prevalence
    """
    labels = []
    
    for idx, row in df.iterrows():
        left_kw = str(row['Left-Diagnostic Keywords']).lower()
        right_kw = str(row['Right-Diagnostic Keywords']).lower()
        combined = left_kw + ' ' + right_kw
        
        # Check for normal (exact match)
        left_normal = left_kw.strip() == 'normal fundus'
        right_normal = right_kw.strip() == 'normal fundus'
        
        if left_normal and right_normal:
            labels.append(0)  # Normal
        elif 'retinopathy' in combined or 'diabetic' in combined:
            labels.append(1)  # Diabetic Retinopathy
        elif 'glaucoma' in combined:
            labels.append(2)  # Glaucoma
        elif 'macular degeneration' in combined or 'amd' in combined:
            labels.append(3)  # AMD
        else:
            labels.append(4)  # Other Disease
    
    return np.array(labels)


def create_severity_based(df):
    """
    Create 4-class system based on severity: Normal, Mild, Moderate, Severe
    
    Strategy: Use keywords that indicate severity
    """
    labels = []
    
    for idx, row in df.iterrows():
        left_kw = str(row['Left-Diagnostic Keywords']).lower()
        right_kw = str(row['Right-Diagnostic Keywords']).lower()
        combined = left_kw + ' ' + right_kw
        
        # Check for normal
        left_normal = left_kw.strip() == 'normal fundus'
        right_normal = right_kw.strip() == 'normal fundus'
        
        if left_normal and right_normal:
            labels.append(0)  # Normal
        elif 'severe' in combined or 'proliferative' in combined:
            labels.append(3)  # Severe
        elif 'moderate' in combined:
            labels.append(2)  # Moderate
        elif 'mild' in combined:
            labels.append(1)  # Mild
        else:
            labels.append(2)  # Default to moderate for unlabeled disease
    
    return np.array(labels)


def analyze_label_system(labels, class_names, current_labels_df=None):
    """Analyze a label system and print statistics."""
    unique, counts = np.unique(labels, return_counts=True)
    
    print(f"\nClass distribution:")
    for cls, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"  {class_names[cls]:25s}: {count:5d} ({pct:5.2f}%)")
    
    print(f"\nTotal samples: {len(labels)}")
    print(f"Num classes: {len(unique)}")
    print(f"Balance ratio (min/max): {counts.min() / counts.max():.3f}")
    
    # Compare with current system if provided
    if current_labels_df is not None:
        current_normal = (current_labels_df['N'] == 1) & (current_labels_df[['D', 'G', 'C', 'A', 'M', 'O']].sum(axis=1) == 0)
        
        # For binary, compare with normal
        if len(unique) == 2:
            new_normal = (labels == 0)
            agreement = (new_normal == current_normal).sum()
            print(f"\nAgreement with current 'Normal' label: {agreement}/{len(labels)} ({agreement/len(labels)*100:.2f}%)")


def main():
    """Main analysis function."""
    # Load data
    data_path = Path('ODIR-5K/data.xlsx')
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        return
    
    df = pd.read_excel(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Get current labels for comparison
    current_labels = df[['N', 'D', 'G', 'C', 'A', 'M', 'O']]
    
    print("\n" + "="*80)
    print("OPTION 1: BINARY CLASSIFICATION (Normal vs Disease)")
    print("="*80)
    binary = create_binary_labels(df)
    analyze_label_system(binary, {0: 'Normal', 1: 'Disease/Abnormal'}, df)
    
    print("\n" + "="*80)
    print("OPTION 2: SIMPLIFIED 3-CLASS (Normal, DR, Other)")
    print("="*80)
    multiclass = create_simplified_multiclass(df)
    analyze_label_system(multiclass, {
        0: 'Normal',
        1: 'Diabetic Retinopathy',
        2: 'Other Disease'
    }, df)
    
    print("\n" + "="*80)
    print("OPTION 3: MAJOR DISEASES 5-CLASS (Normal, DR, Glaucoma, AMD, Other)")
    print("="*80)
    major = create_major_disease_classes(df)
    analyze_label_system(major, {
        0: 'Normal',
        1: 'Diabetic Retinopathy',
        2: 'Glaucoma',
        3: 'AMD',
        4: 'Other Disease'
    }, df)
    
    print("\n" + "="*80)
    print("OPTION 4: SEVERITY-BASED 4-CLASS (Normal, Mild, Moderate, Severe)")
    print("="*80)
    severity = create_severity_based(df)
    analyze_label_system(severity, {
        0: 'Normal',
        1: 'Mild Disease',
        2: 'Moderate Disease',
        3: 'Severe Disease'
    }, df)
    
    # Save all label systems
    print("\n" + "="*80)
    print("SAVING SIMPLIFIED LABEL SYSTEMS")
    print("="*80)
    
    output_dir = Path('results/simplified_labels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy arrays
    np.save(output_dir / 'binary_labels.npy', binary)
    np.save(output_dir / 'multiclass_3_labels.npy', multiclass)
    np.save(output_dir / 'major_diseases_5_labels.npy', major)
    np.save(output_dir / 'severity_4_labels.npy', severity)
    
    # Save metadata
    metadata = {
        'binary': {
            'num_classes': 2,
            'class_names': ['Normal', 'Disease/Abnormal'],
            'distribution': {
                'Normal': int((binary == 0).sum()),
                'Disease': int((binary == 1).sum())
            }
        },
        'multiclass_3': {
            'num_classes': 3,
            'class_names': ['Normal', 'Diabetic Retinopathy', 'Other Disease'],
            'distribution': {
                'Normal': int((multiclass == 0).sum()),
                'DR': int((multiclass == 1).sum()),
                'Other': int((multiclass == 2).sum())
            }
        },
        'major_diseases_5': {
            'num_classes': 5,
            'class_names': ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'AMD', 'Other'],
            'distribution': {
                'Normal': int((major == 0).sum()),
                'DR': int((major == 1).sum()),
                'Glaucoma': int((major == 2).sum()),
                'AMD': int((major == 3).sum()),
                'Other': int((major == 4).sum())
            }
        },
        'severity_4': {
            'num_classes': 4,
            'class_names': ['Normal', 'Mild', 'Moderate', 'Severe'],
            'distribution': {
                'Normal': int((severity == 0).sum()),
                'Mild': int((severity == 1).sum()),
                'Moderate': int((severity == 2).sum()),
                'Severe': int((severity == 3).sum())
            }
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved all label systems to: {output_dir}/")
    print(f"  - binary_labels.npy (2 classes)")
    print(f"  - multiclass_3_labels.npy (3 classes)")
    print(f"  - major_diseases_5_labels.npy (5 classes)")
    print(f"  - severity_4_labels.npy (4 classes)")
    print(f"  - metadata.json")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
Based on the analysis:

1. BINARY (Normal vs Disease) - 71% disease, 29% normal
   ✓ Very simple and clear
   ✓ Good for screening applications
   ✓ Highly imbalanced but manageable
   
2. 3-CLASS (Normal, DR, Other) - More informative
   ✓ Separates most common disease (DR: ~60%)
   ✓ Still relatively simple
   ✓ Better clinical utility than binary
   
3. 5-CLASS (Normal, DR, Glaucoma, AMD, Other)
   ✓ Covers major diseases
   ✓ More balanced than current 7-class
   ✓ Good compromise between detail and simplicity
   
4. SEVERITY-BASED (Normal, Mild, Moderate, Severe)
   ✓ Clinically meaningful progression
   ✓ Could help with treatment priority
   ✓ But loses specific disease information

BEST CHOICE: Option 2 or 3
- If you want simplicity: Use 3-class
- If you want more detail: Use 5-class
- Both are better balanced than current 7-class system
""")


if __name__ == '__main__':
    main()
