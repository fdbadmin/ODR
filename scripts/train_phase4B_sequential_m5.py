#!/usr/bin/env python3
"""
Phase 4B Sequential Training - M5 Optimized
Train ConvNeXt Tiny, ViT Small, and EfficientNetV2 Small sequentially
Optimized for MacBook Pro M5 with 32GB unified memory
"""

import sys
import os
import time
import json
import torch
import logging
from pathlib import Path
from datetime import datetime

# Setup logging to both file and console
log_file = Path('logs') / f'phase4B_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# M5 Optimized Configuration
M5_CONFIG = {
    'batch_size': 96,           # Increased from 64 - M5 32GB can handle it
    'num_workers': 10,          # M5 has 10-core CPU
    'persistent_workers': True, # Reduce worker restart overhead
    'pin_memory': True,         # Beneficial with unified memory
    'prefetch_factor': 2,       # Prefetch 2 batches per worker
    'learning_rate': 1e-4,
    'epochs': 50,
    'weight_decay': 1e-5,
}

# Model configurations with optimal augmentation strategies
MODELS = [
    {
        'name': 'convnext_tiny',
        'description': 'ConvNeXt Tiny - Best performer in Phase 4A',
        'expected_f1': '63-66%',
        'augmentation': {'use_mixup': True, 'use_cutmix': False},
        'priority': 1  # Train first (most promising)
    },
    {
        'name': 'vit_small',
        'description': 'Vision Transformer Small',
        'expected_f1': '58-61%',
        'augmentation': {'use_mixup': False, 'use_cutmix': True},
        'priority': 2
    },
    {
        'name': 'efficientnetv2_s',
        'description': 'EfficientNetV2 Small',
        'expected_f1': '58-61%',
        'augmentation': {'use_mixup': True, 'use_cutmix': True},
        'priority': 3
    }
]

def clear_mps_cache():
    """Clear MPS cache to free memory between models."""
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.info("✓ MPS cache cleared")
        except Exception as e:
            logger.warning(f"Could not clear MPS cache: {e}")

def train_single_model(model_config, m5_config):
    """
    Train a single model with M5 optimizations.
    Returns (success, results_dict)
    """
    model_name = model_config['name']
    logger.info(f"\n{'='*80}")
    logger.info(f"Training: {model_config['description']}")
    logger.info(f"Expected F1: {model_config['expected_f1']}")
    logger.info(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Build command with M5 optimizations
    cmd_parts = [
        'python', 'scripts/train_advanced.py',
        '--model', model_name,
        '--epochs', str(m5_config['epochs']),
        '--lr', str(m5_config['learning_rate']),
        '--batch-size', str(m5_config['batch_size']),
    ]
    
    # Add augmentation flags
    if model_config['augmentation'].get('use_mixup'):
        cmd_parts.append('--use-mixup')
    if model_config['augmentation'].get('use_cutmix'):
        cmd_parts.append('--use-cutmix')
    
    cmd = ' '.join(cmd_parts)
    logger.info(f"Command: {cmd}\n")
    
    # Execute training
    try:
        # Modify train_advanced.py DataLoader settings on-the-fly via environment
        env = os.environ.copy()
        env['M5_NUM_WORKERS'] = str(m5_config['num_workers'])
        env['M5_PERSISTENT_WORKERS'] = '1' if m5_config['persistent_workers'] else '0'
        env['M5_PIN_MEMORY'] = '1' if m5_config['pin_memory'] else '0'
        env['M5_PREFETCH_FACTOR'] = str(m5_config['prefetch_factor'])
        
        import subprocess
        result = subprocess.run(
            cmd_parts,
            env=env,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        
        logger.info(f"\n✓ {model_name} training completed successfully")
        logger.info(f"  Duration: {hours}h {minutes}m")
        
        # Load best model results
        model_path = Path(f'models/{model_name}_advanced_best.pth')
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            best_f1 = checkpoint.get('best_val_f1', 0.0)
            best_epoch = checkpoint.get('epoch', -1)
            per_class_f1 = checkpoint.get('per_class_f1', [])
            
            logger.info(f"  Best F1: {best_f1:.4f} at epoch {best_epoch}")
            logger.info(f"  Per-class F1: {' '.join([f'{f1:.3f}' for f1 in per_class_f1])}\n")
            
            return True, {
                'model': model_name,
                'success': True,
                'best_f1': float(best_f1),
                'best_epoch': int(best_epoch),
                'per_class_f1': [float(f) for f in per_class_f1],
                'duration_seconds': duration,
                'duration_readable': f"{hours}h {minutes}m"
            }
        else:
            logger.warning(f"Model checkpoint not found: {model_path}")
            return True, {
                'model': model_name,
                'success': True,
                'note': 'Checkpoint not found for detailed results'
            }
            
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logger.error(f"\n✗ {model_name} training failed after {duration/60:.1f} minutes")
        logger.error(f"  Error: {e}")
        return False, {
            'model': model_name,
            'success': False,
            'error': str(e),
            'duration_seconds': duration
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"\n✗ Unexpected error training {model_name}")
        logger.error(f"  Error: {e}")
        return False, {
            'model': model_name,
            'success': False,
            'error': str(e),
            'duration_seconds': duration
        }

def generate_phase4B_report(results, total_time):
    """Generate comprehensive Phase 4B training report."""
    report_path = Path('results') / f'PHASE4B_TRAINING_REPORT_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    report_path.parent.mkdir(exist_ok=True)
    
    successful_models = [r for r in results if r.get('success', False) and 'best_f1' in r]
    failed_models = [r for r in results if not r.get('success', False)]
    
    with open(report_path, 'w') as f:
        f.write("# Phase 4B Training Report - Vessel Enhancement\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Hardware:** MacBook Pro M5 with 32GB unified memory\n\n")
        
        # Configuration
        f.write("## M5 Optimization Configuration\n\n")
        f.write("```python\n")
        f.write(f"batch_size: {M5_CONFIG['batch_size']}  # Optimized for 32GB\n")
        f.write(f"num_workers: {M5_CONFIG['num_workers']}  # Match M5 10-core CPU\n")
        f.write(f"persistent_workers: {M5_CONFIG['persistent_workers']}\n")
        f.write(f"pin_memory: {M5_CONFIG['pin_memory']}\n")
        f.write(f"prefetch_factor: {M5_CONFIG['prefetch_factor']}\n")
        f.write("```\n\n")
        
        # Summary
        f.write("## Training Summary\n\n")
        total_hours = int(total_time // 3600)
        total_minutes = int((total_time % 3600) // 60)
        f.write(f"- **Total Time:** {total_hours}h {total_minutes}m\n")
        f.write(f"- **Successful Models:** {len(successful_models)}/3\n")
        f.write(f"- **Failed Models:** {len(failed_models)}/3\n\n")
        
        # Successful models
        if successful_models:
            f.write("## Model Results\n\n")
            f.write("| Model | Best F1 | Epoch | Duration | Per-Class F1 |\n")
            f.write("|-------|---------|-------|----------|-------------|\n")
            
            for result in successful_models:
                model = result['model']
                f1 = result['best_f1']
                epoch = result['best_epoch']
                duration = result['duration_readable']
                per_class = ', '.join([f"{f:.3f}" for f in result['per_class_f1']])
                f.write(f"| {model} | {f1:.4f} | {epoch} | {duration} | {per_class} |\n")
            
            f.write("\n")
            
            # Best model
            best_result = max(successful_models, key=lambda x: x['best_f1'])
            f.write(f"**Best Model:** {best_result['model']} with F1 = {best_result['best_f1']:.4f}\n\n")
        
        # Failed models
        if failed_models:
            f.write("## Failed Models\n\n")
            for result in failed_models:
                f.write(f"- **{result['model']}:** {result.get('error', 'Unknown error')}\n")
            f.write("\n")
        
        # Next steps
        f.write("## Next Steps\n\n")
        if successful_models:
            f.write("1. Run threshold optimization: `python scripts/optimize_phase4_thresholds.py`\n")
            f.write("2. Evaluate ensemble performance\n")
            f.write("3. Compare with Phase 4A baseline (64.63%)\n")
            f.write("4. Target: 67-70% F1 with vessel enhancement\n\n")
        else:
            f.write("1. Review errors and adjust configuration\n")
            f.write("2. Retry training with fixes\n\n")
        
        # Phase comparison
        f.write("## Phase 4 Progression\n\n")
        f.write("- **Phase 4A:** 64.63% F1 (baseline + threshold optimization)\n")
        f.write("- **Phase 4B:** TBD (+ vessel enhancement)\n")
        f.write("- **Target:** 67-70% F1 (+3-5% improvement)\n\n")
        
        f.write("---\n")
        f.write(f"*Training log: {log_file}*\n")
    
    logger.info(f"\n✓ Report generated: {report_path}")
    return report_path

def main():
    """Main sequential training loop."""
    logger.info("="*80)
    logger.info("Phase 4B Sequential Training - M5 Optimized")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    
    # Verify MPS availability
    if not torch.backends.mps.is_available():
        logger.error("✗ MPS not available! Training will be slow on CPU.")
        logger.error("  This script is optimized for M5 with MPS acceleration.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            logger.info("Training cancelled.")
            return
    else:
        logger.info("✓ MPS available - using Apple Silicon acceleration")
    
    # Display M5 configuration
    logger.info("\nM5 Optimization Settings:")
    logger.info(f"  Batch size: {M5_CONFIG['batch_size']} (vs typical 32-64)")
    logger.info(f"  Workers: {M5_CONFIG['num_workers']} (vs typical 4-6)")
    logger.info(f"  Persistent workers: {M5_CONFIG['persistent_workers']}")
    logger.info(f"  Pin memory: {M5_CONFIG['pin_memory']}")
    logger.info(f"  Prefetch factor: {M5_CONFIG['prefetch_factor']}")
    logger.info("")
    
    # Verify preprocessed data exists
    data_dir = Path('preprocessed_data')
    if not (data_dir / 'train_images.npy').exists():
        logger.error(f"✗ Preprocessed data not found in {data_dir}")
        logger.error("  Run preprocessing first: python src/data_preprocessing_enhanced.py")
        return
    logger.info(f"✓ Preprocessed data found in {data_dir}\n")
    
    # Start training
    total_start = time.time()
    results = []
    
    for i, model_config in enumerate(MODELS, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"Model {i}/{len(MODELS)}: {model_config['name']}")
        logger.info(f"{'#'*80}")
        
        # Clear MPS cache before each model
        if i > 1:
            clear_mps_cache()
            time.sleep(2)  # Brief pause to ensure cleanup
        
        # Train model
        success, result = train_single_model(model_config, M5_CONFIG)
        results.append(result)
        
        # Save intermediate results
        intermediate_path = Path('results') / 'phase4B_training_progress.json'
        intermediate_path.parent.mkdir(exist_ok=True)
        with open(intermediate_path, 'w') as f:
            json.dump({
                'completed_models': len(results),
                'total_models': len(MODELS),
                'results': results,
                'elapsed_time_seconds': time.time() - total_start
            }, f, indent=2)
        
        if not success:
            logger.warning(f"✗ {model_config['name']} failed, continuing to next model...")
        
        # Brief pause between models
        if i < len(MODELS):
            logger.info("\nPreparing for next model...")
            time.sleep(5)
    
    total_time = time.time() - total_start
    total_hours = int(total_time // 3600)
    total_minutes = int((total_time % 3600) // 60)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("PHASE 4B TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {total_hours}h {total_minutes}m")
    
    successful = sum(1 for r in results if r.get('success', False))
    logger.info(f"Successful models: {successful}/{len(MODELS)}")
    
    if successful > 0:
        f1_scores = [r['best_f1'] for r in results if 'best_f1' in r]
        if f1_scores:
            best_f1 = max(f1_scores)
            avg_f1 = sum(f1_scores) / len(f1_scores)
            logger.info(f"Best F1: {best_f1:.4f}")
            logger.info(f"Average F1: {avg_f1:.4f}")
    
    logger.info("")
    
    # Generate report
    report_path = generate_phase4B_report(results, total_time)
    
    logger.info("\n" + "="*80)
    logger.info("Next Steps:")
    logger.info("="*80)
    if successful > 0:
        logger.info("1. Review training report: " + str(report_path))
        logger.info("2. Optimize thresholds: python scripts/optimize_phase4_thresholds.py")
        logger.info("3. Evaluate ensemble performance")
        logger.info("4. Compare with Phase 4A baseline (64.63%)")
    else:
        logger.info("1. Review training log: " + str(log_file))
        logger.info("2. Check for errors and adjust configuration")
        logger.info("3. Retry training")
    logger.info("="*80 + "\n")

if __name__ == '__main__':
    main()
