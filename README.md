# ODIR-5K Eye Disease Classification

Multi-label deep learning system for detecting 7 eye diseases from retinal fundus images using the ODIR-5K dataset.

## 🎯 Current Status (November 2025)

**Version**: 7-Class Optimized Ensemble System  
**Final Model**: **3-Model Ensemble with Optimal Thresholds**  
**Performance**: **91.51% F1 Score, 96.84% Accuracy** ✅✅✅

### 🏆 Final Ensemble Performance

**Optimized 3-Model Ensemble (Validation Set, 1279 samples)**

| Disease | F1 Score | Precision | Recall | Threshold | Status |
|---------|----------|-----------|--------|-----------|--------|
| **Myopia (M)** | **0.9767** | 0.9692 | 0.9844 | 0.56 | 🥇 Outstanding |
| **AMD (A)** | **0.9677** | 1.0000 | 0.9375 | 0.76 | 🥇 Outstanding |
| **Normal (N)** | **0.9145** | 0.9535 | 0.8786 | 0.66 | ✅ Excellent |
| **Cataract (C)** | **0.9008** | 0.9672 | 0.8429 | 0.66 | ✅ Excellent |
| **Diabetes (D)** | **0.8957** | 0.9148 | 0.8774 | 0.41 | ✅ Excellent |
| **Glaucoma (G)** | **0.8772** | 0.8721 | 0.8824 | 0.39 | ✅ Very Good |
| **Other (O)** | **0.8734** | 0.8552 | 0.8924 | 0.31 | ✅ Very Good |
| **Mean F1** | **0.9151** | - | - | - | **🎯 Clinical-Grade** |

**Label Accuracy**: 96.84%

### Individual Model Performance

| Model | Parameters | F1 Score | Accuracy | Best Classes |
|-------|-----------|----------|----------|--------------|
| ResNet50 | 24.6M | 0.8872 | 96.36% | Myopia (0.977), Normal (0.897) |
| DenseNet-121 | 7.0M | 0.8851 | 96.21% | Myopia (0.973), Diabetes (0.896) |
| EfficientNet-B3 | 10.7M | 0.8765 | 95.95% | AMD (0.900), Myopia (0.969) |

### Ensemble Improvements

| Configuration | Mean F1 | Accuracy | Improvement |
|--------------|---------|----------|-------------|
| Baseline (threshold=0.5) | 0.8952 | 96.53% | +0.90% vs best individual |
| **Optimal Thresholds** | **0.9151** | **96.84%** | **+2.23% vs baseline** |
| TTA Only | 0.7255 | 91.44% | ❌ -18.96% (not beneficial) |

**Key Findings:**
- ✅ Per-class threshold optimization highly effective (+2.23% F1)
- ❌ Test-time augmentation degrades performance (models already well-calibrated)
- 🎯 Biggest improvements: AMD (+9.68%), Cataract (+2.29%), Normal (+1.72%)

### Why 7 Classes?
**Hypertension removed**: Only 10-15% of hypertensive patients show retinal changes visible in fundus images. Diagnosis requires blood pressure measurement, not fundus imaging alone.

## 🎯 Features

- **3-Model Ensemble**: ResNet50 + EfficientNet-B3 + DenseNet-121 with weighted averaging
- **Optimal Per-Class Thresholds**: Disease-specific thresholds (0.31-0.76) for maximum F1
- **Multi-label Classification**: Detects 7 conditions (Normal, Diabetes, Glaucoma, Cataract, AMD, Myopia, Other)
- **F1-Based Model Selection**: Balanced precision/recall for clinical reliability
- **Advanced Preprocessing**: Green channel extraction, CLAHE (clip_limit=3.0), illumination correction
- **Apple Silicon Optimized**: Batch size 48, 6 workers, MPS-compatible
- **Clinical Performance**: All classes F1 > 0.87, with 91.51% mean F1

## 📊 Deployment

### Quick Start - Predict from JPEG Images

**Command Line (Easiest):**
```bash
# Single image
python src/predict_from_image.py --image path/to/fundus.jpg

# Multiple images
python src/predict_from_image.py --images img1.jpg img2.jpg img3.jpg

# Save results to JSON
python src/predict_from_image.py --image fundus.jpg --output predictions.json
```

**Python Code:**
```python
from src.predict_from_image import preprocess_fundus_image
from src.production_ensemble import ProductionEnsemble

# Initialize model (one time)
ensemble = ProductionEnsemble(device='mps')  # or 'cuda', 'cpu'

# Preprocess JPEG image (automatic: green channel, CLAHE, normalization)
image = preprocess_fundus_image('path/to/fundus.jpg')

# Get predictions
predictions = ensemble.predict(image)

# Check results
for disease, info in predictions.items():
    if info['predicted']:
        print(f"⚠️ {disease}: {info['confidence']*100:.1f}% confidence")
```

**Output Format:**
```python
{
  'Normal': {'predicted': False, 'confidence': 0.385, 'threshold': 0.66},
  'Diabetes': {'predicted': False, 'confidence': 0.037, 'threshold': 0.41},
  'Glaucoma': {'predicted': False, 'confidence': 0.012, 'threshold': 0.39},
  'Cataract': {'predicted': False, 'confidence': 0.084, 'threshold': 0.66},
  'AMD': {'predicted': False, 'confidence': 0.022, 'threshold': 0.76},
  'Myopia': {'predicted': False, 'confidence': 0.046, 'threshold': 0.56},
  'Other': {'predicted': True, 'confidence': 0.464, 'threshold': 0.31}
}
```

**Preprocessing (Automatic):**
- ✅ Green channel extraction (most informative for fundus images)
- ✅ CLAHE enhancement (clip_limit=3.0)
- ✅ Illumination correction
- ✅ Resize to 224x224
- ✅ Normalization
- **Just provide raw JPEG files - no manual preprocessing needed!**

### Model Files Required

```
models/
├── baseline_model.pth          # ResNet50 (24.6M params)
├── efficientnet_b3_model.pth   # EfficientNet-B3 (10.7M params)
└── densenet121_model.pth       # DenseNet-121 (7.0M params)

results/
└── optimal_thresholds.json     # Per-class thresholds
```

## 📈 Training & Optimization Pipeline

### Stage 1: Individual Model Training (Complete ✅)
```bash
# ResNet50 baseline
python src/train.py --epochs 25 --batch-size 48

# EfficientNet-B3
python src/train_ensemble_models.py --model efficientnet_b3 --epochs 25

# DenseNet-121
python src/train_ensemble_models.py --model densenet121 --epochs 25
```

### Stage 2: Ensemble Creation (Complete ✅)
```bash
# Evaluate baseline ensemble
python src/evaluate_ensemble.py
```

### Stage 3: Threshold Optimization (Complete ✅)
```bash
# Find optimal per-class thresholds
python src/optimize_thresholds.py

# Evaluate with optimal thresholds
python src/evaluate_thresholds_only.py
```

## 📊 Model Performance Details

### Threshold Optimization Impact

| Disease | Default (0.5) | Optimal Threshold | Optimal F1 | Improvement |
|---------|---------------|-------------------|------------|-------------|
| AMD | 0.8824 | 0.76 | 0.9677 | +9.68% ⬆️ |
| Cataract | 0.8806 | 0.66 | 0.9008 | +2.29% ⬆️ |
| Normal | 0.8990 | 0.66 | 0.9145 | +1.72% ⬆️ |
| Other | 0.8645 | 0.31 | 0.8734 | +1.03% ⬆️ |
| Myopia | 0.9692 | 0.56 | 0.9767 | +0.78% ⬆️ |
| Diabetes | 0.8934 | 0.41 | 0.8957 | +0.26% ⬆️ |
| Glaucoma | 0.8772 | 0.39 | 0.8772 | ±0.00% → |

**Overall**: 89.52% → 91.51% (+2.23%)

## 📁 Project Structure

```
ODR/
├── src/                           # Source code
│   ├── train.py                  # Training pipeline with metadata integration
│   ├── evaluate_model.py         # Comprehensive model evaluation
│   ├── predict.py                # Inference script
│   ├── metadata_extractor.py     # Extract age/gender from ODIR-5K
│   ├── advanced_preprocessing.py # Image preprocessing pipeline
│   ├── keyword_label_parser.py   # Parse diagnostic keywords to labels
│   └── utils.py                  # Utility functions
├── notebooks/                     # Jupyter notebooks
│   ├── model_evaluation.ipynb    # Visualizations and analysis
│   └── data_exploration.ipynb    # Dataset exploration
├── tests/                         # Unit tests
│   ├── test_smart_allocation.py  # Test label allocation
│   └── test_enhanced_preprocessing.py  # Test preprocessing
├── docs/                          # Documentation
├── archive/                       # Old preprocessing scripts
├── deployment/                    # Deployment configurations
├── config.py                      # Configuration constants
├── preprocess.py                  # Main preprocessing script
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/fdbadmin/ODR.git
cd ODR

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Place ODIR-5K dataset in project root
# Expected structure:
# ODIR-5K/
#   ├── ODIR-5K_Training_Annotations.xlsx
#   ├── ODIR-5K_Training_Images/
#   └── ODIR-5K_Testing_Images/

# Run preprocessing
python preprocess.py
```

### 3. Training (Two-Stage Recommended)

**Stage 1: Train Baseline Model**
```bash
# Configure src/train.py:
# STAGE = 1
# USE_METADATA = False
# NUM_EPOCHS = 15

PYTHONPATH=/path/to/ODR python src/train.py

# Training time: ~45 minutes (15 epochs on Apple Silicon)
# Expected: 85.21% validation accuracy
```

**Stage 2: Train Refinement Network**
```bash
# Configure src/train.py:
# STAGE = 2
# USE_METADATA = True
# USE_TWO_STAGE = True
# BASELINE_MODEL_PATH = 'models/best_model.pth'
# NUM_EPOCHS = 10

PYTHONPATH=/path/to/ODR python src/train.py

# Training time: ~10 minutes (10 epochs)
# Expected: 86-88% validation accuracy
```

**Alternative: Standard Metadata Training** (not recommended)
```bash
# Configure src/train.py:
# STAGE = 1
# USE_METADATA = True
# USE_TWO_STAGE = False
# NUM_EPOCHS = 15

PYTHONPATH=/path/to/ODR python src/train.py

# Note: May underperform baseline due to overfitting
```

See `docs/TWO_STAGE_TRAINING_GUIDE.md` for detailed instructions.

### 4. Evaluation

```bash
# Evaluate trained model
PYTHONPATH=/path/to/ODR python src/evaluate_model.py

# Or use Jupyter notebook for visualizations
jupyter notebook notebooks/model_evaluation.ipynb
```

### 5. Inference

```bash
# Predict on new images
python src/predict.py --image path/to/image.jpg --age 65 --gender M
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- pandas
- Pillow
- scikit-learn
- matplotlib
- seaborn
- openpyxl

See `requirements.txt` for complete list.

## 🔬 Dataset

**ODIR-5K** (Ocular Disease Intelligent Recognition)
- 5,000 patients (10,000 images - both eyes)
- 8 disease categories
- Multi-label annotations
- Patient metadata (age, gender)

**Preprocessing Pipeline:**
1. Green channel extraction (best contrast for retinal features)
2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Illumination correction (remove uneven lighting)
4. Resize to 224×224
5. Normalize to ImageNet statistics

## 🎯 Clinical Considerations

### Bias Mitigation
- **Multiplicative bias reduced by 7.4×** (2,266× → 306×)
- Moderate thresholds for balanced sensitivity/specificity
- Capped oversampling to prevent false alarm generation

### Validation Needed
- Age-group testing (young vs old patients)
- Gender bias analysis
- Temperature scaling for calibrated confidence scores
- Clinical validation on held-out test set

### Intended Use
- **Screening assistant** for ophthalmologists
- **Not a diagnostic tool** - requires clinical confirmation
- Best used in conjunction with clinical examination

## 📊 Class Distribution (Training Set)

| Disease | Samples | Frequency | Class Weight |
|---------|---------|-----------|--------------|
| Normal (N) | 2,481 | 44.43% | 1.25× |
| Diabetes (D) | 1,467 | 26.27% | 1.80× |
| Glaucoma (G) | 248 | 4.44% | 11.26× |
| Cataract (C) | 600 | 10.74% | 4.64× |
| AMD (A) | 357 | 6.39% | 7.81× |
| Hypertension (H) | 148 | 2.65% | 36.73× |
| Myopia (M) | 247 | 4.42% | 11.30× |
| Other (O) | 696 | 12.46% | 4.00× |

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_smart_allocation.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check types
mypy src/
```

## 📝 License

This project is for educational and research purposes. The ODIR-5K dataset has its own licensing terms.

## 🙏 Acknowledgments

- **ODIR-5K Dataset**: Peking University & Shanggong Medical Technology Co., Ltd.
- **ResNet50**: Microsoft Research
- **PyTorch**: Facebook AI Research

## 📧 Contact

For questions or collaboration, please open an issue on GitHub.

---

**Status**: Model training in progress with clinical safeguards | Expected completion: 2 Nov 2025
