"""
Advanced preprocessing techniques for retinal fundus images.
Implements state-of-the-art methods to maximize model performance.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt


class RetinalImagePreprocessor:
    """
    Advanced preprocessing pipeline for retinal fundus images.
    
    Key techniques:
    1. Green channel extraction (best contrast for blood vessels)
    2. Illumination correction (remove uneven lighting)
    3. Contrast enhancement (CLAHE, histogram equalization)
    4. Vessel enhancement (morphological operations)
    5. Noise reduction (Gaussian blur, bilateral filter)
    6. Image normalization (standardization)
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 use_green_channel: bool = True,
                 apply_illumination_correction: bool = True,
                 apply_vessel_enhancement: bool = False,
                 clahe_clip_limit: float = 3.0,
                 clahe_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize preprocessor with configuration.
        
        Args:
            target_size: Output image size (width, height)
            use_green_channel: Extract green channel (best for retinal vessels)
            apply_illumination_correction: Remove uneven lighting artifacts
            apply_vessel_enhancement: Enhance blood vessel visibility
            clahe_clip_limit: CLAHE contrast limit (higher = more contrast)
            clahe_grid_size: CLAHE tile grid size
        """
        self.target_size = target_size
        self.use_green_channel = use_green_channel
        self.apply_illumination_correction = apply_illumination_correction
        self.apply_vessel_enhancement = apply_vessel_enhancement
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
    
    def extract_roi(self, img: np.ndarray, margin: int = 10) -> np.ndarray:
        """
        Extract region of interest (remove black borders).
        
        Args:
            img: Input image (RGB or grayscale)
            margin: Additional margin to remove around edges
            
        Returns:
            Cropped image containing only fundus region
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Threshold to find fundus region
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add margin (but stay within bounds)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            
            return img[y:y+h, x:x+w]
        
        return img
    
    def correct_illumination(self, img: np.ndarray, sigma: int = 50) -> np.ndarray:
        """
        Remove uneven illumination using background estimation.
        
        Args:
            img: Input grayscale image
            sigma: Gaussian blur kernel size (larger = smoother background)
            
        Returns:
            Illumination-corrected image
        """
        # Estimate background with heavy Gaussian blur
        background = cv2.GaussianBlur(img, (0, 0), sigma)
        
        # Subtract background and rescale
        corrected = cv2.subtract(img, background)
        corrected = cv2.add(corrected, 128)  # Add offset to avoid clipping
        
        # Normalize to full range
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
        
        return corrected.astype(np.uint8)
    
    def enhance_vessels(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance blood vessel visibility using morphological operations.
        
        Args:
            img: Input grayscale image
            
        Returns:
            Image with enhanced vessel contrast
        """
        # Apply top-hat transform to enhance bright structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        
        # Apply bottom-hat transform to enhance dark structures (vessels)
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine
        enhanced = cv2.add(img, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)
        
        return enhanced
    
    def apply_clahe(self, img: np.ndarray, color: bool = False) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            img: Input image (RGB or grayscale)
            color: Whether to apply to color image (uses LAB color space)
            
        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )
        
        if color and len(img.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            return clahe.apply(img)
    
    def standardize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Standardize image to zero mean and unit variance.
        
        Args:
            img: Input image (normalized to [0, 1])
            
        Returns:
            Standardized image
        """
        mean = np.mean(img)
        std = np.std(img)
        
        if std > 0:
            standardized = (img - mean) / std
        else:
            standardized = img - mean
        
        return standardized
    
    def process(self, image_path: Path, visualize: bool = False) -> Optional[np.ndarray]:
        """
        Apply full preprocessing pipeline.
        
        Args:
            image_path: Path to input image
            visualize: Whether to show intermediate steps
            
        Returns:
            Preprocessed image ready for model input, or None if error
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if visualize:
                steps = [("Original", img.copy())]
            
            # Step 1: Extract ROI (remove black borders)
            img = self.extract_roi(img, margin=10)
            if visualize:
                steps.append(("ROI Extracted", img.copy()))
            
            # Step 2: Use green channel (best for retinal structures)
            if self.use_green_channel:
                img = img[:, :, 1]  # Green channel
                if visualize:
                    steps.append(("Green Channel", img.copy()))
            
            # Step 3: Illumination correction
            if self.apply_illumination_correction:
                img = self.correct_illumination(img, sigma=50)
                if visualize:
                    steps.append(("Illumination Corrected", img.copy()))
            
            # Step 4: Vessel enhancement (optional)
            if self.apply_vessel_enhancement:
                img = self.enhance_vessels(img)
                if visualize:
                    steps.append(("Vessels Enhanced", img.copy()))
            
            # Step 5: CLAHE contrast enhancement
            img = self.apply_clahe(img, color=False)
            if visualize:
                steps.append(("CLAHE Applied", img.copy()))
            
            # Step 6: Bilateral filter for noise reduction (preserves edges)
            img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
            if visualize:
                steps.append(("Noise Reduced", img.copy()))
            
            # Step 7: Resize to target size
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            if visualize:
                steps.append(("Resized", img.copy()))
            
            # Step 8: Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Step 9: Convert grayscale to 3-channel (for ResNet compatibility)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            
            if visualize:
                steps.append(("Normalized", (img * 255).astype(np.uint8)))
                self._visualize_pipeline(steps)
            
            return img
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def _visualize_pipeline(self, steps: list):
        """Visualize preprocessing steps side-by-side."""
        n_steps = len(steps)
        fig, axes = plt.subplots(2, (n_steps + 1) // 2, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, (title, img) in enumerate(steps):
            if len(img.shape) == 2:
                axes[idx].imshow(img, cmap='gray')
            else:
                axes[idx].imshow(img)
            axes[idx].set_title(title, fontsize=10)
            axes[idx].axis('off')
        
        # Hide extra subplots
        for idx in range(len(steps), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('preprocessing_pipeline.png', dpi=150, bbox_inches='tight')
        print("✓ Saved visualization to: preprocessing_pipeline.png")
        plt.close()


def compare_preprocessing_methods(image_path: Path, save_path: str = "preprocessing_comparison.png"):
    """
    Compare different preprocessing approaches side-by-side.
    
    Args:
        image_path: Path to test image
        save_path: Where to save comparison figure
    """
    # Load original image
    img_original = cv2.imread(str(image_path))
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
    # Method 1: Basic (current approach)
    processor_basic = RetinalImagePreprocessor(
        use_green_channel=False,
        apply_illumination_correction=False,
        apply_vessel_enhancement=False,
        clahe_clip_limit=2.0
    )
    img_basic = processor_basic.process(image_path)
    if img_basic is None:
        print("Warning: Basic preprocessing failed, skipping comparison")
        return
    
    # Method 2: Green channel + CLAHE
    processor_green = RetinalImagePreprocessor(
        use_green_channel=True,
        apply_illumination_correction=False,
        apply_vessel_enhancement=False,
        clahe_clip_limit=3.0
    )
    img_green = processor_green.process(image_path)
    if img_green is None:
        print("Warning: Green channel preprocessing failed, skipping comparison")
        return
    
    # Method 3: Full pipeline
    processor_full = RetinalImagePreprocessor(
        use_green_channel=True,
        apply_illumination_correction=True,
        apply_vessel_enhancement=False,
        clahe_clip_limit=3.0
    )
    img_full = processor_full.process(image_path)
    if img_full is None:
        print("Warning: Full pipeline preprocessing failed, skipping comparison")
        return
    
    # Method 4: Full + vessel enhancement
    processor_vessel = RetinalImagePreprocessor(
        use_green_channel=True,
        apply_illumination_correction=True,
        apply_vessel_enhancement=True,
        clahe_clip_limit=3.0
    )
    img_vessel = processor_vessel.process(image_path)
    if img_vessel is None:
        print("Warning: Vessel enhancement preprocessing failed, skipping comparison")
        return
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(img_original)
    axes[0, 0].set_title("Original Image", fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_basic)
    axes[0, 1].set_title("Basic CLAHE\n(Current Method)", fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_green)
    axes[0, 2].set_title("Green Channel + CLAHE", fontsize=12)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_full)
    axes[1, 0].set_title("Green + Illumination Correction\n(Recommended)", fontsize=12, weight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_vessel)
    axes[1, 1].set_title("Full Pipeline + Vessel Enhancement", fontsize=12)
    axes[1, 1].axis('off')
    
    # Add text comparison
    axes[1, 2].axis('off')
    comparison_text = """
COMPARISON:

Basic (Current):
✓ Fast processing
✗ Retains lighting artifacts
✗ Less vessel contrast

Green Channel:
✓ Better vessel visibility
✓ Reduced noise
~ Moderate improvement

Full Pipeline:
✓ Best contrast
✓ Removes artifacts
✓ Most consistent
⭐ RECOMMENDED

Vessel Enhancement:
✓ Maximum detail
✗ May amplify noise
~ Use for vessel diseases
"""
    axes[1, 2].text(0.1, 0.5, comparison_text, 
                    fontsize=10, family='monospace',
                    verticalalignment='center')
    
    plt.suptitle("Preprocessing Methods Comparison", fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison to: {save_path}")
    plt.close()


def test_preprocessing_on_sample(image_path: Path):
    """
    Test advanced preprocessing on a sample image with visualization.
    
    Args:
        image_path: Path to test image
    """
    print("\n" + "="*70)
    print("TESTING ADVANCED PREPROCESSING")
    print("="*70)
    
    print(f"\nProcessing: {image_path.name}")
    
    # Test with visualization
    processor = RetinalImagePreprocessor(
        use_green_channel=True,
        apply_illumination_correction=True,
        apply_vessel_enhancement=False,
        clahe_clip_limit=3.0
    )
    
    result = processor.process(image_path, visualize=True)
    
    if result is not None:
        print(f"\n✓ Preprocessing successful!")
        print(f"  Output shape: {result.shape}")
        print(f"  Value range: [{result.min():.3f}, {result.max():.3f}]")
        print(f"  Mean: {result.mean():.3f}, Std: {result.std():.3f}")
    else:
        print("\n✗ Preprocessing failed!")
    
    # Compare methods
    print("\nComparing preprocessing methods...")
    compare_preprocessing_methods(image_path)
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test on provided image
        test_image = Path(sys.argv[1])
        if test_image.exists():
            test_preprocessing_on_sample(test_image)
        else:
            print(f"Error: Image not found: {test_image}")
    else:
        # Test on sample from dataset
        sample_images = list(Path("ODIR-5K/Training Images").glob("*.jpg"))
        if sample_images:
            test_preprocessing_on_sample(sample_images[0])
        else:
            print("No sample images found. Provide an image path:")
            print("  python advanced_preprocessing.py path/to/image.jpg")
