#!/usr/bin/env python3
"""
Apply M5 optimizations to train_advanced.py DataLoader configurations.
This script modifies the num_workers setting in train_advanced.py.
"""

import sys
from pathlib import Path

def apply_m5_optimizations():
    """Apply M5-specific optimizations to train_advanced.py"""
    
    script_path = Path('scripts/train_advanced.py')
    
    if not script_path.exists():
        print(f"Error: {script_path} not found!")
        return False
    
    # Read the current file
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Get M5 settings from environment or use defaults
    import os
    num_workers = int(os.environ.get('M5_NUM_WORKERS', '10'))
    persistent_workers = os.environ.get('M5_PERSISTENT_WORKERS', '1') == '1'
    pin_memory = os.environ.get('M5_PIN_MEMORY', '1') == '1'
    prefetch_factor = int(os.environ.get('M5_PREFETCH_FACTOR', '2'))
    
    # Replace the DataLoader configurations (lines 272-273)
    # Old: num_workers=0
    # New: num_workers=10, persistent_workers=True, pin_memory=True, prefetch_factor=2
    
    modifications = [
        # Training DataLoader
        (
            'train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)',
            f'train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, '
            f'num_workers={num_workers}, persistent_workers={persistent_workers}, '
            f'pin_memory={pin_memory}, prefetch_factor={prefetch_factor})'
        ),
        # Validation DataLoader
        (
            'val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)',
            f'val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, '
            f'num_workers={num_workers}, persistent_workers={persistent_workers}, '
            f'pin_memory={pin_memory}, prefetch_factor={prefetch_factor})'
        )
    ]
    
    modified_content = content
    changes_made = 0
    
    for old, new in modifications:
        if old in modified_content:
            modified_content = modified_content.replace(old, new)
            changes_made += 1
            print(f"✓ Applied modification {changes_made}")
        else:
            print(f"Warning: Pattern not found: {old[:50]}...")
    
    if changes_made > 0:
        # Backup original
        backup_path = script_path.with_suffix('.py.backup')
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"✓ Backup saved: {backup_path}")
        
        # Write modified version
        with open(script_path, 'w') as f:
            f.write(modified_content)
        print(f"✓ M5 optimizations applied to {script_path}")
        print(f"  num_workers: {num_workers}")
        print(f"  persistent_workers: {persistent_workers}")
        print(f"  pin_memory: {pin_memory}")
        print(f"  prefetch_factor: {prefetch_factor}")
        return True
    else:
        print(f"✗ No modifications applied - patterns not found")
        return False

def restore_original():
    """Restore original train_advanced.py from backup"""
    script_path = Path('scripts/train_advanced.py')
    backup_path = script_path.with_suffix('.py.backup')
    
    if backup_path.exists():
        with open(backup_path, 'r') as f:
            content = f.read()
        with open(script_path, 'w') as f:
            f.write(content)
        print(f"✓ Restored original from {backup_path}")
        return True
    else:
        print(f"✗ Backup not found: {backup_path}")
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--restore':
        restore_original()
    else:
        apply_m5_optimizations()
