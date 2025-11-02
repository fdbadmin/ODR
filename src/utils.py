"""
Utility functions for ODIR-5K dataset manipulation and visualization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image
import warnings

from config import (
    CSV_PATH, DATA_DIR, DISEASE_LABELS, LABEL_COLUMNS, 
    DISEASE_COLORS, DEFAULT_IMAGE_SIZE, TRAIN_IMAGE_DIR, TEST_IMAGE_DIR
)

warnings.filterwarnings('ignore')


def load_dataset(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the ODIR dataset from CSV or Excel file.
    
    Args:
        data_path: Path to data file. If None, uses default from config.
        
    Returns:
        DataFrame with the dataset.
    """
    if data_path is None:
        data_path = CSV_PATH
    
    # Check file extension and load accordingly
    if str(data_path).endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"Loaded dataset with {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    return df


def get_disease_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for each disease category.
    
    Args:
        df: DataFrame with disease labels.
        
    Returns:
        DataFrame with disease statistics.
    """
    stats = []
    total_patients = len(df)
    
    for label in LABEL_COLUMNS:
        if label in df.columns:
            count = df[label].sum()
            percentage = (count / total_patients) * 100
            stats.append({
                'Disease Code': label,
                'Disease Name': DISEASE_LABELS[label],
                'Count': int(count),
                'Percentage': f"{percentage:.2f}%"
            })
    
    return pd.DataFrame(stats)


def plot_disease_distribution(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)):
    """
    Plot the distribution of diseases in the dataset.
    
    Args:
        df: DataFrame with disease labels.
        figsize: Figure size for the plot.
    """
    stats = get_disease_statistics(df)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    colors = [DISEASE_COLORS[code] for code in stats['Disease Code']]
    ax1.bar(stats['Disease Code'], stats['Count'], color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Disease Code', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax1.set_title('Disease Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (code, count) in enumerate(zip(stats['Disease Code'], stats['Count'])):
        ax1.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(stats['Count'], labels=stats['Disease Code'], colors=colors, 
            autopct='%1.1f%%', startangle=90, textprops={'fontweight': 'bold'})
    ax2.set_title('Disease Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return stats


def analyze_multilabel(df: pd.DataFrame) -> Dict:
    """
    Analyze multi-label distribution in the dataset.
    
    Args:
        df: DataFrame with disease labels.
        
    Returns:
        Dictionary with multi-label statistics.
    """
    # Count number of diseases per patient
    disease_counts = df[LABEL_COLUMNS].sum(axis=1)
    
    analysis = {
        'single_disease': (disease_counts == 1).sum(),
        'multiple_diseases': (disease_counts > 1).sum(),
        'no_disease': (disease_counts == 0).sum(),
        'max_diseases_per_patient': disease_counts.max(),
        'avg_diseases_per_patient': disease_counts.mean()
    }
    
    print("Multi-label Analysis:")
    print(f"  Single disease: {analysis['single_disease']} patients")
    print(f"  Multiple diseases: {analysis['multiple_diseases']} patients")
    print(f"  No disease labeled: {analysis['no_disease']} patients")
    print(f"  Max diseases per patient: {analysis['max_diseases_per_patient']}")
    print(f"  Avg diseases per patient: {analysis['avg_diseases_per_patient']:.2f}")
    
    return analysis


def plot_age_gender_distribution(df: pd.DataFrame, figsize: Tuple[int, int] = (14, 5)):
    """
    Plot age and gender distribution in the dataset.
    
    Args:
        df: DataFrame with patient information.
        figsize: Figure size for the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Age distribution
    if 'Patient Age' in df.columns:
        axes[0].hist(df['Patient Age'].dropna(), bins=30, color='skyblue', 
                     edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Age', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Age Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
    
    # Gender distribution
    if 'Patient Sex' in df.columns:
        gender_counts = df['Patient Sex'].value_counts()
        axes[1].bar(gender_counts.index, gender_counts.values, 
                   color=['lightcoral', 'lightblue'], edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Gender', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[1].set_title('Gender Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (gender, count) in enumerate(zip(gender_counts.index, gender_counts.values)):
            axes[1].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Age by disease
    if 'Patient Age' in df.columns:
        age_data = []
        labels = []
        for label in LABEL_COLUMNS:
            if label in df.columns:
                ages = df[df[label] == 1]['Patient Age'].dropna()
                if len(ages) > 0:
                    age_data.append(ages)
                    labels.append(label)
        
        axes[2].boxplot(age_data, labels=labels, patch_artist=True)
        axes[2].set_xlabel('Disease Code', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Age', fontsize=12, fontweight='bold')
        axes[2].set_title('Age Distribution by Disease', fontsize=14, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def load_image(image_path: Path, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load and optionally resize an image.
    
    Args:
        image_path: Path to the image file.
        target_size: Target size (width, height) for resizing.
        
    Returns:
        Numpy array of the image.
    """
    try:
        img = Image.open(image_path)
        if target_size:
            img = img.resize(target_size, Image.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def visualize_samples(df: pd.DataFrame, n_samples: int = 5, 
                     image_column: str = 'Right-Fundus',
                     show_labels: bool = True):
    """
    Visualize sample images from the dataset.
    
    Args:
        df: DataFrame with image information.
        n_samples: Number of samples to display.
        image_column: Column name containing image filenames.
        show_labels: Whether to show disease labels.
    """
    # Sample random rows
    samples = df.sample(n=min(n_samples, len(df)))
    
    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
    if n_samples == 1:
        axes = [axes]
    
    for idx, (ax, (_, row)) in enumerate(zip(axes, samples.iterrows())):
        # Try to find and load image
        if image_column in row and pd.notna(row[image_column]):
            # Try different possible paths
            possible_paths = [
                TRAIN_IMAGE_DIR / row[image_column],
                TEST_IMAGE_DIR / row[image_column],
                DATA_DIR / row[image_column],
                DATA_DIR / 'preprocessed_images' / row[image_column],
            ]
            
            img = None
            for path in possible_paths:
                if path.exists():
                    img = load_image(path)
                    if img is not None:
                        break
            
            if img is not None:
                ax.imshow(img)
                
                if show_labels:
                    # Get disease labels for this patient
                    diseases = [DISEASE_LABELS[label] for label in LABEL_COLUMNS 
                               if label in row and row[label] == 1]
                    title = f"Age: {row.get('Patient Age', 'N/A')}, Sex: {row.get('Patient Sex', 'N/A')}\n"
                    title += f"Diseases: {', '.join(diseases) if diseases else 'None'}"
                    ax.set_title(title, fontsize=10)
            else:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
                ax.set_title(f"Image: {row[image_column]}", fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No image path', ha='center', va='center')
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def create_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)):
    """
    Create a correlation matrix for disease co-occurrence.
    
    Args:
        df: DataFrame with disease labels.
        figsize: Figure size for the plot.
    """
    import seaborn as sns
    
    # Calculate correlation matrix
    corr_matrix = df[LABEL_COLUMNS].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Disease Co-occurrence Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


if __name__ == "__main__":
    # Example usage
    print("Loading dataset...")
    df = load_dataset()
    
    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDisease statistics:")
    stats = get_disease_statistics(df)
    print(stats)
