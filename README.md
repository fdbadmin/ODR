# ODIR-5K Eye Disease Classification

Multi-label deep learning system for detecting 8 eye diseases from retinal fundus images using the ODIR-5K dataset.

## 🎯 Features

- **Multi-label Classification**: Detects 8 conditions (Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other)
- **Metadata Integration**: Uses patient age and gender alongside images for improved accuracy
- **Advanced Preprocessing**: Green channel extraction, CLAHE, illumination correction
- **Clinical Safeguards**: Balanced sensitivity/specificity for clinical reliability
- **Smart Label Parsing**: Automatically extracts labels from diagnostic keywords

## 📊 Model Performance

Current baseline (image-only):
- **Mean Sample Accuracy**: 85.21%
- **Best Performers**: Myopia (89.1% AUC), Glaucoma (73.6%), Cataract (69.9%)
- **Challenges**: Low recall on rare diseases (Hypertension: 0%, AMD: 1.9%)

Two-Stage Metadata Refinement (recommended approach):
- **Expected Accuracy**: 86-88% (+1-3% over baseline)
- **Guaranteed Floor**: ≥85.21% (cannot regress below baseline)
- **Improved Rare Disease Detection**: 10-15% recall on Hypertension, AMD, Diabetes
- **Safe Integration**: Frozen baseline + trainable refinement network

## 🏗️ Architecture

### Two-Stage Refinement Model (Recommended)
```
Stage 1: Frozen Baseline (Image-Only)
  - ResNet50 pretrained backbone
  - Trained to 85.21% accuracy
  - Frozen during Stage 2 (guarantees floor performance)

Stage 2: Metadata Refinement
  - Input: Baseline predictions + age/gender
  - Small MLP (~5K parameters)
  - Learns metadata-based corrections
  - Output: Baseline + refinements (residual connection)
```

**Key Advantages:**
- ✅ **No Regression Risk**: Cannot perform worse than baseline
- ✅ **Fast Training**: Only ~10 minutes for refinement
- ✅ **Interpretable**: Can measure metadata contribution per disease
- ✅ **Efficient**: Only 5K trainable parameters vs 24.6M

### Alternative: Standard Metadata Model
- **Image Encoder**: ResNet50 (2048 features)
- **Metadata Encoder**: Small MLP (age + gender → 16 features)
- **Late Fusion**: Concatenate features
- **Total Parameters**: 24.6M
- **Note**: Prone to overfitting, may regress below baseline

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
