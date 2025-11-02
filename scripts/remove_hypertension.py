#!/usr/bin/env python3
"""
Remove Hypertension Label from Dataset
=======================================

This script removes hypertension (H) from the label columns since it is not
reliably detectable from fundus images alone. It:

1. Updates config.py to remove 'H' from LABEL_COLUMNS and DISEASE_LABELS
2. Rebuilds preprocessed data without the hypertension column
3. Updates all saved models to work with 7 classes instead of 8
4. Creates a backup of the original data

Usage:
    python remove_hypertension.py [--backup-dir BACKUP_DIR]
"""

import numpy as np
import pandas as pd
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import json

def backup_files(backup_dir):
    """Create backups of important files before modification"""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("📦 CREATING BACKUPS")
    print("=" * 80)
    
    files_to_backup = [
        'config.py',
        'preprocessed_data/train_images.npy',
        'preprocessed_data/train_labels.npy',
        'preprocessed_data/val_images.npy',
        'preprocessed_data/val_labels.npy',
        'preprocessed_data/test_images.npy',
        'preprocessed_data/test_labels.npy',
    ]
    
    for file_path in files_to_backup:
        src = Path(file_path)
        if src.exists():
            dst = backup_dir / src.name
            shutil.copy2(src, dst)
            print(f"✓ Backed up: {src} -> {dst}")
        else:
            print(f"⚠ Skipped (not found): {src}")
    
    print(f"\n✓ Backups saved to: {backup_dir}")
    print()

def update_config():
    """Update config.py to remove hypertension"""
    print("=" * 80)
    print("📝 UPDATING CONFIG.PY")
    print("=" * 80)
    
    config_path = Path('config.py')
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Remove H from DISEASE_LABELS
    old_disease_labels = """DISEASE_LABELS = {
    'N': 'Normal',
    'D': 'Diabetes',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'Age-related Macular Degeneration',
    'H': 'Hypertension',
    'M': 'Pathological Myopia',
    'O': 'Other diseases/abnormalities'
}"""
    
    new_disease_labels = """DISEASE_LABELS = {
    'N': 'Normal',
    'D': 'Diabetes',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'Age-related Macular Degeneration',
    'M': 'Pathological Myopia',
    'O': 'Other diseases/abnormalities'
}

# Note: Hypertension (H) removed - not reliably detectable from fundus images alone"""
    
    content = content.replace(old_disease_labels, new_disease_labels)
    
    # Remove H from LABEL_COLUMNS
    old_label_columns = "LABEL_COLUMNS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']"
    new_label_columns = "LABEL_COLUMNS = ['N', 'D', 'G', 'C', 'A', 'M', 'O']  # H removed"
    
    content = content.replace(old_label_columns, new_label_columns)
    
    # Remove H from DISEASE_COLORS
    old_color_line = "    'H': '#e67e22',  # Dark orange"
    content = '\n'.join([line for line in content.split('\n') if old_color_line not in line])
    
    with open(config_path, 'w') as f:
        f.write(content)
    
    print("✓ Updated DISEASE_LABELS (removed Hypertension)")
    print("✓ Updated LABEL_COLUMNS: ['N', 'D', 'G', 'C', 'A', 'M', 'O']")
    print("✓ Updated DISEASE_COLORS (removed H entry)")
    print()

def remove_hypertension_column(data_path, output_path):
    """Remove hypertension column (index 5) from label arrays"""
    data = np.load(data_path)
    
    if data.ndim == 2:  # Labels array
        # Remove column 5 (H - Hypertension)
        # Original: [N, D, G, C, A, H, M, O] (8 columns)
        # New:      [N, D, G, C, A, M, O]    (7 columns)
        new_data = np.delete(data, 5, axis=1)
        print(f"  Labels: {data.shape} -> {new_data.shape}")
    else:  # Images array
        new_data = data
        print(f"  Images: {data.shape} (unchanged)")
    
    np.save(output_path, new_data)
    return new_data

def rebuild_preprocessed_data():
    """Rebuild all preprocessed data files without hypertension"""
    print("=" * 80)
    print("🔨 REBUILDING PREPROCESSED DATA")
    print("=" * 80)
    
    preprocessed_dir = Path('preprocessed_data')
    
    if not preprocessed_dir.exists():
        print("⚠ No preprocessed_data directory found.")
        print("  You'll need to run preprocessing from scratch.")
        return
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_path = preprocessed_dir / f'{split}_images.npy'
        labels_path = preprocessed_dir / f'{split}_labels.npy'
        
        if not labels_path.exists():
            print(f"⚠ Skipped {split}: {labels_path} not found")
            continue
        
        print(f"\n{split.upper()} SET:")
        
        # Load and process
        labels = np.load(labels_path)
        
        # Check if already has 7 columns (already processed)
        if labels.shape[1] == 7:
            print(f"  ✓ Already has 7 columns, skipping")
            continue
        
        # Remove hypertension column (index 5)
        new_labels = remove_hypertension_column(labels_path, labels_path)
        
        # Copy images (unchanged)
        if images_path.exists():
            images = np.load(images_path)
            print(f"  Images: {images.shape} samples")
        
        # Print label distribution
        print(f"  Label distribution:")
        label_names = ['N', 'D', 'G', 'C', 'A', 'M', 'O']
        for i, label in enumerate(label_names):
            count = int(new_labels[:, i].sum())
            percentage = count / len(new_labels) * 100
            print(f"    {label}: {count:4d} ({percentage:5.1f}%)")
    
    print("\n✓ Preprocessed data rebuilt without hypertension")
    print()

def create_migration_report():
    """Create a report of what was changed"""
    print("=" * 80)
    print("📊 MIGRATION REPORT")
    print("=" * 80)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'action': 'Removed Hypertension label',
        'reason': 'Hypertension not reliably detectable from fundus images alone',
        'changes': {
            'label_columns': {
                'before': ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'],
                'after': ['N', 'D', 'G', 'C', 'A', 'M', 'O']
            },
            'num_classes': {
                'before': 8,
                'after': 7
            },
            'removed_column_index': 5
        },
        'files_modified': [
            'config.py',
            'preprocessed_data/train_labels.npy',
            'preprocessed_data/val_labels.npy',
            'preprocessed_data/test_labels.npy'
        ],
        'models_affected': [
            'All models need retraining with num_classes=7',
            'Existing 8-class models are incompatible'
        ]
    }
    
    report_path = Path('HYPERTENSION_REMOVAL_REPORT.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n📋 Summary:")
    print(f"  • Removed label: Hypertension (H)")
    print(f"  • Column index removed: 5")
    print(f"  • Classes before: 8")
    print(f"  • Classes after: 7")
    print(f"  • New label order: ['N', 'D', 'G', 'C', 'A', 'M', 'O']")
    print()
    print("⚠️  IMPORTANT NOTES:")
    print("  1. All existing models (8 classes) need to be retrained")
    print("  2. Model files in models/ directory are now incompatible")
    print("  3. You'll need to retrain from scratch with num_classes=7")
    print()
    print(f"✓ Full report saved to: {report_path}")
    print()

def main():
    parser = argparse.ArgumentParser(description='Remove hypertension label from dataset')
    parser.add_argument('--backup-dir', type=str, 
                       default=f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='Directory to store backups')
    parser.add_argument('--skip-backup', action='store_true',
                       help='Skip creating backups')
    
    args = parser.parse_args()
    
    print()
    print("=" * 80)
    print("🔧 REMOVE HYPERTENSION LABEL")
    print("=" * 80)
    print()
    print("This script will:")
    print("  1. Create backups of important files")
    print("  2. Update config.py (remove H from labels)")
    print("  3. Rebuild preprocessed data (remove H column)")
    print("  4. Create migration report")
    print()
    print("⚠️  WARNING: All existing models will need to be retrained!")
    print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled.")
        return
    
    print()
    
    # Step 1: Backup
    if not args.skip_backup:
        backup_files(args.backup_dir)
    else:
        print("⚠ Skipping backups (--skip-backup flag)")
        print()
    
    # Step 2: Update config
    update_config()
    
    # Step 3: Rebuild preprocessed data
    rebuild_preprocessed_data()
    
    # Step 4: Create report
    create_migration_report()
    
    print("=" * 80)
    print("✅ MIGRATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review HYPERTENSION_REMOVAL_REPORT.json")
    print("  2. Delete old model files in models/ directory")
    print("  3. Retrain models with: python src/train.py")
    print("  4. Update any notebooks to use 7 classes")
    print()

if __name__ == '__main__':
    main()
