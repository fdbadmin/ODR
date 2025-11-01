# Ocular Disease Intelligent Recognition (ODIR)

## Dataset Overview

The ODIR-5K dataset is a structured ophthalmic database containing information from 5,000 patients, designed to represent real-world clinical data collected by Shanggong Medical Technology Co., Ltd. from various hospitals and medical centers across China.

## Dataset Characteristics

- **Total Patients**: 5,000
- **Images per Patient**: 2 (left and right eye fundus photographs)
- **Total Images**: ~10,000 color fundus photographs
- **Image Sources**: Multiple camera manufacturers (Canon, Zeiss, Kowa)
- **Image Resolutions**: Varied (due to different camera models)
- **Annotations**: Labeled by trained human readers with quality control

## Disease Categories (8 Classes)

The dataset classifies patients into eight diagnostic labels:

| Code | Disease | Description |
|------|---------|-------------|
| **N** | Normal | Normal fundus, no abnormalities detected |
| **D** | Diabetes | Diabetic retinopathy and related complications |
| **G** | Glaucoma | Optic nerve damage, increased intraocular pressure |
| **C** | Cataract | Clouding of the eye's lens |
| **A** | Age-related Macular Degeneration | Deterioration of the macula |
| **H** | Hypertension | Hypertensive retinopathy |
| **M** | Pathological Myopia | Severe nearsightedness with complications |
| **O** | Other | Other diseases/abnormalities not listed above |

## Data Structure

```
ODR/
├── Occular Disease Data/
│   ├── full_df.csv                 # Main dataset CSV
│   ├── preprocessed_images/        # Preprocessed images
│   └── ODIR-5K/                    # Original ODIR-5K dataset
├── README.md                        # This file
├── config.py                        # Configuration settings
├── utils.py                         # Helper functions
├── data_exploration.ipynb          # Data exploration notebook
└── requirements.txt                # Python dependencies
```

## CSV File Columns

- `ID`: Patient identifier
- `Patient Age`: Age of the patient
- `Patient Sex`: Gender (Male/Female)
- `Left-Fundus`: Filename of left eye fundus image
- `Right-Fundus`: Filename of right eye fundus image
- `Left-Diagnostic Keywords`: Doctor's diagnostic notes for left eye
- `Right-Diagnostic Keywords`: Doctor's diagnostic notes for right eye
- `N`, `D`, `G`, `C`, `A`, `H`, `M`, `O`: Binary labels for each disease category

## Multi-Label Classification

Note that this is a **multi-label classification** problem - a patient can have multiple diseases simultaneously (e.g., both Diabetes and Hypertension).

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Explore the data:
   ```bash
   jupyter notebook data_exploration.ipynb
   ```

3. Use the utility functions:
   ```python
   from utils import load_dataset, visualize_samples
   
   df = load_dataset()
   visualize_samples(df, n_samples=5)
   ```

## Citation

If you use this dataset, please cite the original ODIR-5K challenge and Shanggong Medical Technology Co., Ltd.

## License

Please refer to the original ODIR-5K dataset license terms.
