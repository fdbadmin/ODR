"""
Advanced keyword-based label parser for ODIR-5K dataset.
Extracts eye-specific disease labels from diagnostic keywords.
"""
import numpy as np
import re
from typing import List, Dict, Tuple


class DiagnosticKeywordParser:
    """
    Parser to convert diagnostic keywords to multi-label disease vectors.
    
    Uses comprehensive pattern matching with medical terminology to
    accurately identify diseases from free-text diagnostic keywords.
    """
    
    def __init__(self):
        """Initialize with comprehensive disease patterns."""
        
        # Disease code mapping: N, D, G, C, A, H, M, O
        self.disease_codes = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
        
        # Comprehensive pattern matching for each disease
        # Patterns are ordered by specificity (most specific first)
        self.disease_patterns = {
            'N': [  # Normal
                'normal fundus',
                'no fundus abnormalities',
                'fundus normal',
                '^normal$',  # Exact match only
            ],
            'D': [  # Diabetes
                'diabetic retinopathy',
                'diabetes',
                'proliferative.*retinopathy',
                'nonproliferative.*retinopathy',
                'non.*proliferative.*retinopathy',
                'microaneurysms',
                'hard exudates',
                'soft exudates',
                'cotton wool spots',
                'neovascularization',
                'vitreous hemorrhage',
                'laser.*spot',  # Post-laser often indicates diabetic treatment
                'diabetic macular edema',
                'dme',
            ],
            'G': [  # Glaucoma
                'glaucoma',
                'suspected glaucoma',
                'glaucomatous',
                'optic disc cupping',
                'disc cupping',
            ],
            'C': [  # Cataract
                'cataract',
                'lens opacity',
                'lens dust',
                'subcapsular cataract',
                'nuclear cataract',
                'cortical cataract',
                'refractive media opacity',  # Often indicates cataract
            ],
            'A': [  # Age-related Macular Degeneration
                'age.?related macular degeneration',
                'age.?related maculopathy',
                'macular degeneration',
                'dry.*macular degeneration',
                'wet.*macular degeneration',
                'macular drusen',
                'drusen',
                'geographic atrophy',
                'choroidal neovascularization',
            ],
            'H': [  # Hypertension
                'hypertensive retinopathy',
                'retinal arteriosclerosis',
                'arteriovenous.*nicking',
                'copper.*wiring',
                'silver.*wiring',
            ],
            'M': [  # Pathological Myopia
                'pathological myopia',
                'myopic retinopathy',
                'myopia.*retinopathy',
                'high myopia',
                'tessellated fundus',
                'myopic.*degeneration',
                'myopic maculopathy',
            ],
            'O': [  # Other diseases/abnormalities
                # Retinal vein/artery occlusions
                'vein occlusion',
                'artery occlusion',
                'brvo',
                'crvo',
                'brao',
                'crao',
                
                # Membranes and structural
                'epiretinal membrane',
                'macular.*membrane',
                'vitreomacular',
                'macular hole',
                'macular pucker',
                
                # Retinal issues
                'retinal.*detachment',
                'retinal.*tear',
                'retinal.*break',
                'retinitis.*pigmentosa',
                'retinitis',
                'chorioretinitis',
                'chorioretinal.*atrophy',
                'retinal.*pigmentation',
                'myelinated.*nerve.*fiber',
                
                # Macular issues
                'maculopathy',
                'macular.*scar',
                'macular.*atrophy',
                
                # Degenerations
                'asteroid hyalosis',
                'vitreous.*degeneration',
                'lattice.*degeneration',
                
                # Post-surgical
                'post.*laser',
                'laser.*photocoagulation',
                'post.*retinal.*surgery',
                
                # Optic disc abnormalities
                'optic disc.*abnormalit',
                'optic.*atrophy',
                'optic.*drusen',
                'disc.*abnormalit',
                
                # Other
                'wedge.*white.*line',
                'punctate.*choroidopathy',
                'spotted.*membranous',
                'chorioretinal',
                'atrophy',
                'scar',
            ]
        }
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for disease, patterns in self.disease_patterns.items():
            self.compiled_patterns[disease] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        # Special handling for ambiguous terms
        self.ambiguous_terms = {
            'lens dust': 'C',  # Usually indicates early cataract
            'low image quality': None,  # Ignore, not a diagnosis
            'image quality issues': None,
        }
        
        # Quality-related keywords that indicate image should be excluded
        self.quality_exclusion_keywords = [
            'low image quality',
            'image quality issues',
            'poor image quality',
            'unclear fundus',
            'unable to assess',
            'cannot assess',
            'poor quality',
            'insufficient quality',
        ]
    
    def is_low_quality(self, keywords: str) -> bool:
        """
        Check if keywords indicate low image quality that should be excluded.
        
        Args:
            keywords: Diagnostic keywords string
            
        Returns:
            True if image should be excluded due to quality issues
        """
        if pd.isna(keywords) or keywords == '':
            return False
        
        keywords_lower = str(keywords).lower()
        
        # Check if any exclusion keyword is present
        for exclusion_kw in self.quality_exclusion_keywords:
            if exclusion_kw in keywords_lower:
                return True
        
        return False
    
    def parse_keywords(self, keywords: str) -> np.ndarray:
        """
        Parse diagnostic keywords and return multi-label vector.
        
        Args:
            keywords: Comma or Chinese-comma separated diagnostic keywords
            
        Returns:
            Binary label vector [N, D, G, C, A, H, M, O] where 1=present
        """
        if pd.isna(keywords) or keywords == '' or str(keywords).lower() == 'nan':
            # Return empty labels (will use fallback)
            return np.zeros(8, dtype=np.float32)
        
        # Initialize label vector
        labels = np.zeros(8, dtype=np.float32)
        
        # Clean and normalize
        keywords = str(keywords).lower().strip()
        
        # Split by comma (both Western and Chinese)
        keyword_list = []
        for separator in ['，', ',']:
            keyword_list.extend([kw.strip() for kw in keywords.split(separator)])
        keyword_list = [kw for kw in keyword_list if kw]
        
        # Track if we found any diseases
        found_any_disease = False
        
        # Check each keyword against patterns
        for keyword in keyword_list:
            # Handle ambiguous terms first
            if keyword in self.ambiguous_terms:
                disease_code = self.ambiguous_terms[keyword]
                if disease_code is not None:
                    idx = self.disease_codes.index(disease_code)
                    labels[idx] = 1.0
                    found_any_disease = True
                continue
            
            # Match against each disease pattern
            for disease, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(keyword):
                        idx = self.disease_codes.index(disease)
                        labels[idx] = 1.0
                        found_any_disease = True
                        break  # Move to next keyword once matched
        
        # Special case: if "normal fundus" is explicitly mentioned, set N=1
        # But if other diseases are also found, keep both (multi-label)
        if any(re.search(r'normal\s+fundus', kw) for kw in keyword_list):
            labels[0] = 1.0  # N=1
            found_any_disease = True
        
        # If we found diseases but Normal wasn't set, ensure Normal is 0
        if found_any_disease and labels[0] == 1.0 and labels[1:].sum() > 0:
            # Patient has "normal fundus" AND other diseases
            # This is likely an annotation style - keep both
            pass
        
        return labels
    
    def parse_with_fallback(self, 
                           keywords: str, 
                           fallback_labels: np.ndarray,
                           confidence_threshold: float = 0.5) -> Tuple[np.ndarray, str]:
        """
        Parse keywords with fallback to patient-level labels.
        
        Args:
            keywords: Diagnostic keywords string
            fallback_labels: Patient-level labels from columns H-O
            confidence_threshold: If parsed labels are uncertain, use fallback
            
        Returns:
            Tuple of (labels, source) where source is 'parsed', 'fallback', or 'combined'
        """
        parsed_labels = self.parse_keywords(keywords)
        
        # Check if we parsed anything
        if parsed_labels.sum() == 0:
            # No keywords found, use fallback
            return fallback_labels, 'fallback'
        
        # Check if parsed labels conflict with fallback
        # If there's strong agreement, use parsed
        # If there's strong disagreement, combine them
        agreement = (parsed_labels == fallback_labels).sum() / len(parsed_labels)
        
        if agreement >= 0.75:
            # Strong agreement, use parsed
            return parsed_labels, 'parsed'
        elif agreement < 0.5:
            # Strong disagreement, combine (OR operation)
            combined = np.maximum(parsed_labels, fallback_labels)
            return combined, 'combined'
        else:
            # Moderate agreement, prefer parsed
            return parsed_labels, 'parsed'
    
    def parse_patient_both_eyes(self,
                               left_keywords: str,
                               right_keywords: str,
                               patient_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Smart parsing for both eyes of a patient, using patient-level labels intelligently.
        
        This method is MUCH smarter than parsing each eye independently:
        - If one eye has clear keywords and the other doesn't, it allocates remaining 
          patient diseases to the unclear eye
        - Handles unilateral conditions (cataract, injury) more accurately
        - Reduces false positives from blanket patient-level label application
        
        Example:
            Patient labels: [Cataract=1, Diabetes=1]
            Left keywords: "cataract" → [C=1]
            Right keywords: "unclear image" → ???
            
            OLD approach: Right gets [C=1, D=1] (all patient diseases) ❌
            NEW approach: Right gets [D=1] (only remaining diseases) ✓
        
        Args:
            left_keywords: Diagnostic keywords for left eye (column F)
            right_keywords: Diagnostic keywords for right eye (column G)
            patient_labels: Patient-level binary labels (columns H-O)
            
        Returns:
            Tuple of (left_labels, right_labels, stats_dict)
        """
        # Parse both eyes independently first
        left_parsed = self.parse_keywords(left_keywords)
        right_parsed = self.parse_keywords(right_keywords)
        
        left_has_clear = left_parsed.sum() > 0
        right_has_clear = right_parsed.sum() > 0
        
        stats = {
            'left_source': 'unknown',
            'right_source': 'unknown',
            'allocation_used': False
        }
        
        # Case 1: Both eyes have clear keyword parsing
        if left_has_clear and right_has_clear:
            stats['left_source'] = 'parsed'
            stats['right_source'] = 'parsed'
            return left_parsed, right_parsed, stats
        
        # Case 2: Neither eye has clear keywords - use patient labels for both
        if not left_has_clear and not right_has_clear:
            stats['left_source'] = 'fallback'
            stats['right_source'] = 'fallback'
            return patient_labels.copy(), patient_labels.copy(), stats
        
        # Case 3: One eye clear, one eye unclear - SMART ALLOCATION!
        stats['allocation_used'] = True
        
        if left_has_clear and not right_has_clear:
            # Left is clear, right is unclear
            # Allocate remaining patient diseases to right eye
            stats['left_source'] = 'parsed'
            stats['right_source'] = 'allocated'
            
            # Diseases in patient labels but NOT found in left eye
            remaining_diseases = patient_labels * (1 - left_parsed)
            
            # Right eye gets the remaining diseases
            right_labels = remaining_diseases
            
            return left_parsed, right_labels, stats
        
        else:  # right_has_clear and not left_has_clear
            # Right is clear, left is unclear
            # Allocate remaining patient diseases to left eye
            stats['right_source'] = 'parsed'
            stats['left_source'] = 'allocated'
            
            # Diseases in patient labels but NOT found in right eye
            remaining_diseases = patient_labels * (1 - right_parsed)
            
            # Left eye gets the remaining diseases
            left_labels = remaining_diseases
            
            return left_labels, right_parsed, stats
    
    def validate_parsing(self, df, sample_size: int = 100) -> Dict:
        """
        Validate parsing accuracy against patient-level labels.
        
        Args:
            df: DataFrame with diagnostic keywords and labels
            sample_size: Number of samples to validate
            
        Returns:
            Dictionary with validation metrics
        """
        import pandas as pd
        
        results = {
            'total_samples': min(sample_size, len(df)),
            'perfect_match': 0,
            'partial_match': 0,
            'no_match': 0,
            'examples': []
        }
        
        for i in range(min(sample_size, len(df))):
            row = df.iloc[i]
            
            # Parse both eyes
            left_parsed = self.parse_keywords(row['Left-Diagnostic Keywords'])
            right_parsed = self.parse_keywords(row['Right-Diagnostic Keywords'])
            
            # Get patient labels
            patient_labels = row[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].values.astype(np.float32)
            
            # Check left eye match
            left_match = (left_parsed == patient_labels).all()
            right_match = (right_parsed == patient_labels).all()
            
            if left_match and right_match:
                results['perfect_match'] += 1
            elif (left_parsed == patient_labels).sum() >= 6 or (right_parsed == patient_labels).sum() >= 6:
                results['partial_match'] += 1
            else:
                results['no_match'] += 1
                
                # Store example of mismatch
                if len(results['examples']) < 5:
                    results['examples'].append({
                        'id': row['ID'],
                        'left_keywords': row['Left-Diagnostic Keywords'],
                        'right_keywords': row['Right-Diagnostic Keywords'],
                        'left_parsed': left_parsed,
                        'right_parsed': right_parsed,
                        'patient_labels': patient_labels
                    })
        
        return results


# Import pandas at module level for type hints
import pandas as pd


def test_parser():
    """Test the parser on sample data."""
    print("="*80)
    print("TESTING DIAGNOSTIC KEYWORD PARSER")
    print("="*80)
    
    parser = DiagnosticKeywordParser()
    
    # Test cases
    test_cases = [
        ("normal fundus", "Should detect: Normal only"),
        ("cataract", "Should detect: Cataract only"),
        ("moderate non proliferative retinopathy", "Should detect: Diabetes only"),
        ("glaucoma", "Should detect: Glaucoma only"),
        ("pathological myopia", "Should detect: Pathological Myopia only"),
        ("dry age-related macular degeneration", "Should detect: AMD only"),
        ("hypertensive retinopathy", "Should detect: Hypertension only"),
        ("macular epiretinal membrane", "Should detect: Other only"),
        ("laser spot，moderate non proliferative retinopathy", "Should detect: Diabetes only"),
        ("cataract，drusen", "Should detect: Cataract + AMD"),
        ("branch retinal vein occlusion", "Should detect: Other only"),
    ]
    
    disease_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
    
    print("\n🧪 Test Cases:\n")
    for i, (keywords, expected) in enumerate(test_cases, 1):
        labels = parser.parse_keywords(keywords)
        detected = [disease_names[i] for i, val in enumerate(labels) if val == 1.0]
        
        print(f"{i:2d}. Keywords: '{keywords}'")
        print(f"    Expected: {expected}")
        print(f"    Detected: {detected if detected else 'None'}")
        print(f"    Label Vector: {labels.astype(int)}")
        print()
    
    print("="*80)


if __name__ == "__main__":
    test_parser()
