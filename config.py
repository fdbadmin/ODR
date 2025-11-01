"""
Configuration file for ODIR-5K dataset paths and constants.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "ODIR-5K"
ODIR_DIR = DATA_DIR
PREPROCESSED_DIR = BASE_DIR / "preprocessed_data"

# Image directories
TRAIN_IMAGE_DIR = DATA_DIR / "Training Images"
TEST_IMAGE_DIR = DATA_DIR / "Testing Images"

# Data file path (Excel file)
DATA_FILE = DATA_DIR / "data.xlsx"
CSV_PATH = DATA_FILE  # For backward compatibility

# Disease labels
DISEASE_LABELS = {
    'N': 'Normal',
    'D': 'Diabetes',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'Age-related Macular Degeneration',
    'H': 'Hypertension',
    'M': 'Pathological Myopia',
    'O': 'Other diseases/abnormalities'
}

# Label columns in order
LABEL_COLUMNS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# Image properties
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
DEFAULT_IMAGE_SIZE = (224, 224)  # For preprocessing

# Model hyperparameters (example)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_TEST_SPLIT = 0.8

# Random seed for reproducibility
RANDOM_SEED = 42

# Color mapping for visualization
DISEASE_COLORS = {
    'N': '#2ecc71',  # Green
    'D': '#e74c3c',  # Red
    'G': '#3498db',  # Blue
    'C': '#f39c12',  # Orange
    'A': '#9b59b6',  # Purple
    'H': '#e67e22',  # Dark orange
    'M': '#1abc9c',  # Turquoise
    'O': '#95a5a6'   # Gray
}

def ensure_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    PREPROCESSED_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    print("Configuration paths:")
    print(f"Base directory: {BASE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"CSV file: {CSV_PATH}")
    print(f"CSV exists: {CSV_PATH.exists()}")
    print(f"Training images: {TRAIN_IMAGE_DIR}")
    print(f"Training images exist: {TRAIN_IMAGE_DIR.exists()}")
    print(f"Test images: {TEST_IMAGE_DIR}")
    print(f"Test images exist: {TEST_IMAGE_DIR.exists()}")
