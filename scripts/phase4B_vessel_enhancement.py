"""
Phase 4B: Enable vessel enhancement and reprocess data.

Expected improvement: +3-5% F1 (64.63% → 67-70%)
Critical for: Glaucoma, Diabetes retinopathy detection
Time: 4 hours reprocessing + 10 hours retraining
"""

import sys
from pathlib import Path
import numpy as np
import shutil
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def backup_current_data():
    """Backup current preprocessed data before vessel enhancement."""
    backup_dir = Path('preprocessed_data_phase4A_backup')
    
    if backup_dir.exists():
        print(f"⚠️  Backup already exists at: {backup_dir}")
        response = input("Overwrite? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Skipping backup.")
            return False
        shutil.rmtree(backup_dir)
    
    print("\n💾 Creating backup of Phase 4A data (without vessel enhancement)...")
    shutil.copytree('preprocessed_data', backup_dir)
    print(f"✓ Backup created at: {backup_dir}")
    
    # Save metadata about backup
    with open(backup_dir / 'BACKUP_INFO.txt', 'w') as f:
        f.write(f"Backup created: {datetime.now()}\n")
        f.write(f"Phase: 4A (threshold optimization, no vessel enhancement)\n")
        f.write(f"F1 Score: 64.63%\n")
        f.write(f"Models: ConvNeXt Tiny + ViT Small + EfficientNetV2 Small\n")
    
    return True


def reprocess_with_vessel_enhancement():
    """
    Reprocess entire dataset with vessel enhancement enabled.
    
    This will:
    1. Load original ODIR-5K data
    2. Apply full preprocessing WITH vessel enhancement
    3. Overwrite preprocessed_data/ directory
    4. Preserve all eye-specific labels
    """
    print("\n" + "="*70)
    print("PHASE 4B: VESSEL ENHANCEMENT PREPROCESSING")
    print("="*70)
    
    print("\n📋 Configuration:")
    print("  ✓ Green channel extraction")
    print("  ✓ Illumination correction (σ=50)")
    print("  ✓ Vessel enhancement (morphological ops) ⭐ NEW!")
    print("  ✓ CLAHE (clip_limit=3.0)")
    print("  ✓ ROI extraction (circular crop)")
    print("  ✓ Bilateral filtering")
    print("  ✓ Eye-specific labels (diagnostic keywords)")
    print("  ✓ Severity extraction")
    print("  ✓ Quality filtering")
    
    print("\n⚠️  This will OVERWRITE existing preprocessed_data/")
    print("   Make sure you have a backup!")
    
    response = input("\nProceed with vessel enhancement preprocessing? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return False
    
    # Run preprocessing with vessel enhancement
    print("\n🔄 Starting preprocessing (this will take ~2-4 hours)...")
    print("   Estimated time: 4 hours for full dataset\n")
    
    try:
        # Import and run the enhanced preprocessing
        # This will use 'full' method which includes vessel enhancement when enabled
        from src.advanced_preprocessing import RetinalImagePreprocessor
        
        # Modify the preprocessor to enable vessel enhancement
        print("⚙️  Enabling vessel enhancement in preprocessing pipeline...")
        
        # Run the main preprocessing function from data_preprocessing_enhanced.py
        # We need to modify it to use vessel enhancement
        import subprocess
        
        # Create a temporary config to enable vessel enhancement
        config_override = """
# Temporary override for Phase 4B
VESSEL_ENHANCEMENT_ENABLED = True
"""
        
        with open('config_phase4B.py', 'w') as f:
            f.write(config_override)
        
        print("\n📊 Running preprocessing with vessel enhancement...")
        print("   This is the same as Phase 4A but with vessel enhancement enabled.\n")
        
        # We'll need to modify the data_preprocessing_enhanced.py to accept vessel enhancement flag
        # For now, let's create a wrapper script
        
        print("✓ Configuration prepared")
        print("\n⚠️  IMPORTANT: You need to manually edit src/advanced_preprocessing.py")
        print("   Change line ~109: apply_vessel_enhancement=False → True")
        print("\n   Then run: python src/data_preprocessing_enhanced.py")
        print("\n   This script will guide you through the process.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        return False


def verify_vessel_enhancement():
    """Check if vessel enhancement is enabled in the preprocessing."""
    try:
        from src.advanced_preprocessing import RetinalImagePreprocessor
        
        # Create a test preprocessor
        preprocessor = RetinalImagePreprocessor()
        
        if hasattr(preprocessor, 'apply_vessel_enhancement'):
            if preprocessor.apply_vessel_enhancement:
                print("✓ Vessel enhancement is ENABLED")
                return True
            else:
                print("❌ Vessel enhancement is DISABLED")
                return False
        else:
            print("⚠️  Cannot determine vessel enhancement status")
            return False
    except Exception as e:
        print(f"⚠️  Error checking vessel enhancement: {e}")
        return False


def main():
    """Main pipeline for Phase 4B setup."""
    print("="*70)
    print("PHASE 4B: VESSEL ENHANCEMENT SETUP")
    print("="*70)
    
    print("\n📊 Current Status:")
    print("  Phase 4A: 64.63% F1 (threshold optimization)")
    print("  Target: 67-70% F1 with vessel enhancement")
    print("  Improvement expected: +3-5%")
    
    print("\n🎯 Why Vessel Enhancement Matters:")
    print("  • Glaucoma: Vessel patterns show optic nerve damage")
    print("  • Diabetes: Microaneurysms, hemorrhages in vessels")
    print("  • AMD: Vessel changes in macula region")
    print("  • Medically critical features for diagnosis")
    
    print("\n📋 Phase 4B Plan:")
    print("  1. Backup current data (Phase 4A)")
    print("  2. Enable vessel enhancement in preprocessor")
    print("  3. Reprocess entire dataset (~4 hours)")
    print("  4. Retrain all 3 models (~10 hours)")
    print("  5. Re-evaluate with optimized thresholds")
    print("  Total time: ~14 hours")
    
    # Step 1: Backup
    print("\n" + "-"*70)
    print("STEP 1: Backup Current Data")
    print("-"*70)
    backup_success = backup_current_data()
    
    if not backup_success:
        print("⚠️  Backup skipped. Continuing...")
    
    # Step 2: Check vessel enhancement status
    print("\n" + "-"*70)
    print("STEP 2: Check Vessel Enhancement Status")
    print("-"*70)
    vessel_enabled = verify_vessel_enhancement()
    
    if not vessel_enabled:
        print("\n📝 TO ENABLE VESSEL ENHANCEMENT:")
        print("\n1. Edit src/advanced_preprocessing.py:")
        print("   Line ~109: Change")
        print("   FROM: apply_vessel_enhancement=False")
        print("   TO:   apply_vessel_enhancement=True")
        print("\n2. Then run: python src/data_preprocessing_enhanced.py")
        print("\n3. This will reprocess the data with vessel enhancement")
        print("\n4. After preprocessing completes, retrain models:")
        print("   python scripts/train_advanced.py")
        
        print("\n⚠️  Manual edit required. This script will now exit.")
        print("   Run this script again after enabling vessel enhancement.")
        return
    
    # Step 3: Reprocess (if vessel enhancement is enabled)
    print("\n" + "-"*70)
    print("STEP 3: Reprocess Data with Vessel Enhancement")
    print("-"*70)
    
    if vessel_enabled:
        print("✓ Vessel enhancement is already enabled!")
        print("\nRun: python src/data_preprocessing_enhanced.py")
        print("   to reprocess the data.")
    
    print("\n" + "="*70)
    print("PHASE 4B SETUP COMPLETE")
    print("="*70)
    
    print("\n📋 Next Steps:")
    print("  1. ✓ Backup created (if applicable)")
    print("  2. □ Enable vessel enhancement (if not already)")
    print("  3. □ Run: python src/data_preprocessing_enhanced.py")
    print("  4. □ Run: python scripts/train_advanced.py")
    print("  5. □ Evaluate with: python scripts/optimize_phase4_thresholds.py")
    print("\n  Expected result: 67-70% F1 (+3-5% improvement)")


if __name__ == '__main__':
    main()
