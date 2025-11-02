# Utility Scripts

This directory contains utility scripts for data processing and training management.

## Scripts

### Data Processing
- **`preprocess.py`** - Initial data preprocessing script
- **`reprocess_with_advanced.py`** - Advanced preprocessing with CLAHE and enhancement
- **`remove_hypertension.py`** - Migration script to convert from 8 to 7 classes

### Training Management
- **`restart_training.sh`** - Shell script to restart training from scratch
- **`monitor_training.py`** - Monitor training progress and metrics

## Usage

### Preprocessing Data
```bash
python scripts/preprocess.py
```

### Remove Hypertension Label (Already Done)
```bash
python scripts/remove_hypertension.py
```

### Monitor Training
```bash
python scripts/monitor_training.py
```

### Restart Training
```bash
bash scripts/restart_training.sh
```

## Notes

- Most preprocessing is already complete in `preprocessed_data/`
- The project now uses 7 classes (hypertension removed)
- All scripts should be run from the project root directory
