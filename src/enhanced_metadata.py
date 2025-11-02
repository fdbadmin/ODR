"""
Enhanced metadata feature engineering for improved disease prediction.
Expands simple age/gender into richer feature set.
"""
import numpy as np
import torch


class EnhancedMetadataFeatures:
    """
    Transform simple metadata (age, gender) into enhanced feature set.
    """
    
    def __init__(self):
        # Age bin thresholds
        self.age_bins = [0, 30, 40, 50, 60, 70, 100]
        self.num_age_bins = len(self.age_bins) - 1
        
    def transform_single(self, age: float, gender: float) -> np.ndarray:
        """
        Transform single metadata sample into enhanced features.
        
        Args:
            age: Age in years (normalized 0-1 or raw)
            gender: Binary gender (0 or 1)
            
        Returns:
            Enhanced feature vector (18 features)
        """
        # Denormalize age if needed (assume normalized between 0-100)
        if age <= 1.0:
            age = age * 100
        
        features = []
        
        # 1. Original features (2)
        features.append(age / 100.0)  # Normalized age
        features.append(gender)
        
        # 2. Age polynomial features (3)
        features.append((age / 100.0) ** 2)  # Age squared
        features.append((age / 100.0) ** 3)  # Age cubed
        features.append(np.sqrt(age / 100.0))  # Age square root
        
        # 3. Age bins (one-hot encoded) (6)
        age_bin_features = self._age_to_bins(age)
        features.extend(age_bin_features)
        
        # 4. Age-gender interactions (4)
        features.append((age / 100.0) * gender)  # Linear interaction
        features.append((age / 100.0) ** 2 * gender)  # Quadratic interaction
        features.append(gender * int(age < 50))  # Young female/male
        features.append(gender * int(age >= 50))  # Old female/male
        
        # 5. Age risk categories (3)
        features.append(float(age < 40))  # Low risk age
        features.append(float(40 <= age < 65))  # Medium risk age
        features.append(float(age >= 65))  # High risk age
        
        return np.array(features, dtype=np.float32)
    
    def _age_to_bins(self, age: float) -> list:
        """Convert age to one-hot encoded bins."""
        bins = np.zeros(self.num_age_bins, dtype=np.float32)
        for i in range(self.num_age_bins):
            if self.age_bins[i] <= age < self.age_bins[i+1]:
                bins[i] = 1.0
                break
        return bins.tolist()
    
    def transform_batch(self, metadata: np.ndarray) -> np.ndarray:
        """
        Transform batch of metadata.
        
        Args:
            metadata: (N, 2) array of [age, gender]
            
        Returns:
            (N, 18) array of enhanced features
        """
        enhanced = []
        for i in range(len(metadata)):
            age, gender = metadata[i]
            features = self.transform_single(age, gender)
            enhanced.append(features)
        return np.array(enhanced, dtype=np.float32)
    
    def get_feature_names(self) -> list:
        """Get names of all enhanced features."""
        names = [
            'age_norm',
            'gender',
            'age_squared',
            'age_cubed',
            'age_sqrt',
            'age_bin_0-30',
            'age_bin_30-40',
            'age_bin_40-50',
            'age_bin_50-60',
            'age_bin_60-70',
            'age_bin_70+',
            'age_gender_linear',
            'age_gender_quadratic',
            'young_gender',
            'old_gender',
            'low_risk_age',
            'medium_risk_age',
            'high_risk_age'
        ]
        return names
    
    @property
    def num_features(self) -> int:
        """Total number of enhanced features."""
        return 18


def create_enhanced_metadata_dataset(images_path: str, labels_path: str, 
                                     metadata_path: str, output_path: str):
    """
    Create enhanced metadata dataset from existing preprocessed data.
    
    Args:
        images_path: Path to images .npy file
        labels_path: Path to labels .npy file
        metadata_path: Path to original metadata .npy file
        output_path: Path to save enhanced metadata .npy file
    """
    print(f"Loading data from {metadata_path}...")
    metadata = np.load(metadata_path)
    
    print(f"Original metadata shape: {metadata.shape}")
    print(f"Transforming to enhanced features...")
    
    enhancer = EnhancedMetadataFeatures()
    enhanced_metadata = enhancer.transform_batch(metadata)
    
    print(f"Enhanced metadata shape: {enhanced_metadata.shape}")
    print(f"Feature names: {enhancer.get_feature_names()}")
    
    # Save enhanced metadata
    np.save(output_path, enhanced_metadata)
    print(f"✓ Saved enhanced metadata to {output_path}")
    
    return enhanced_metadata


if __name__ == "__main__":
    # Test the enhancer
    enhancer = EnhancedMetadataFeatures()
    
    # Test single transformation
    age = 0.55  # 55 years (normalized)
    gender = 1.0  # Male
    
    features = enhancer.transform_single(age, gender)
    print(f"Enhanced features for age={age*100}, gender={gender}:")
    print(f"  Shape: {features.shape}")
    print(f"  Features: {features}")
    
    feature_names = enhancer.get_feature_names()
    print(f"\nFeature breakdown:")
    for name, value in zip(feature_names, features):
        print(f"  {name:25s}: {value:.4f}")
    
    # Test batch transformation
    metadata_batch = np.array([
        [0.25, 0],  # 25 years, female
        [0.55, 1],  # 55 years, male
        [0.75, 0],  # 75 years, female
    ])
    
    enhanced_batch = enhancer.transform_batch(metadata_batch)
    print(f"\nBatch transformation:")
    print(f"  Input shape: {metadata_batch.shape}")
    print(f"  Output shape: {enhanced_batch.shape}")
    print(f"  Total features: {enhancer.num_features}")
