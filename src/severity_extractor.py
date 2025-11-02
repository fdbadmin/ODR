#!/usr/bin/env python3
"""
Extract severity levels from diagnostic keywords.
This adds fine-grained information beyond binary disease labels.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class SeverityExtractor:
    """
    Extract disease severity information from diagnostic keywords.
    
    This provides additional metadata that can:
    1. Be used as auxiliary labels for multi-task learning
    2. Help prioritize cases during inference
    3. Provide clinical context for predictions
    """
    
    def __init__(self):
        # Severity levels (ordered from least to most severe)
        self.severity_levels = {
            'normal': 0,
            'mild': 1,
            'moderate': 2,
            'severe': 3,
            'proliferative': 4  # Most severe for DR
        }
        
        # Patterns for each severity level
        self.severity_patterns = {
            'mild': [
                r'\bmild\b',
                r'\bearly\b',
                r'\binitial\b',
            ],
            'moderate': [
                r'\bmoderate\b',
            ],
            'severe': [
                r'\bsevere\b',
                r'\badvanced\b',
            ],
            'proliferative': [
                r'\bproliferative\b',
            ],
            'non_proliferative': [
                r'\bnon.?proliferative\b',
                r'\bnonproliferative\b',
            ]
        }
        
        # Compile regex patterns
        self.compiled_patterns = {}
        for severity, patterns in self.severity_patterns.items():
            self.compiled_patterns[severity] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def extract_severity(self, keywords: str) -> Dict[str, any]:
        """
        Extract severity information from diagnostic keywords.
        
        Args:
            keywords: Diagnostic keywords string
            
        Returns:
            Dictionary with:
            - 'has_severity': bool (True if any severity indicator found)
            - 'severity_level': str ('mild', 'moderate', 'severe', 'proliferative', 'none')
            - 'severity_score': int (0-4, for ordering)
            - 'is_proliferative': bool (specific to diabetic retinopathy)
            - 'raw_text': str (original keywords)
        """
        if pd.isna(keywords) or keywords == '':
            return {
                'has_severity': False,
                'severity_level': 'none',
                'severity_score': 0,
                'is_proliferative': False,
                'raw_text': ''
            }
        
        keywords_lower = str(keywords).lower()
        
        # Check for non-proliferative FIRST (before proliferative check)
        if any(p.search(keywords_lower) for p in self.compiled_patterns.get('non_proliferative', [])):
            # Non-proliferative can be mild, moderate, or severe
            if any(p.search(keywords_lower) for p in self.compiled_patterns.get('severe', [])):
                return {
                    'has_severity': True,
                    'severity_level': 'severe',
                    'severity_score': 3,
                    'is_proliferative': False,
                    'raw_text': keywords
                }
            elif any(p.search(keywords_lower) for p in self.compiled_patterns.get('moderate', [])):
                return {
                    'has_severity': True,
                    'severity_level': 'moderate',
                    'severity_score': 2,
                    'is_proliferative': False,
                    'raw_text': keywords
                }
            else:  # Assume mild if no modifier
                return {
                    'has_severity': True,
                    'severity_level': 'mild',
                    'severity_score': 1,
                    'is_proliferative': False,
                    'raw_text': keywords
                }
        
        # Check for proliferative (most severe for DR)
        if any(p.search(keywords_lower) for p in self.compiled_patterns.get('proliferative', [])):
            return {
                'has_severity': True,
                'severity_level': 'proliferative',
                'severity_score': 4,
                'is_proliferative': True,
                'raw_text': keywords
            }
        
        # Check for severe (without proliferative)
        if any(p.search(keywords_lower) for p in self.compiled_patterns.get('severe', [])):
            return {
                'has_severity': True,
                'severity_level': 'severe',
                'severity_score': 3,
                'is_proliferative': False,
                'raw_text': keywords
            }
        
        # Check for moderate
        if any(p.search(keywords_lower) for p in self.compiled_patterns.get('moderate', [])):
            return {
                'has_severity': True,
                'severity_level': 'moderate',
                'severity_score': 2,
                'is_proliferative': False,
                'raw_text': keywords
            }
        
        # Check for mild
        if any(p.search(keywords_lower) for p in self.compiled_patterns.get('mild', [])):
            return {
                'has_severity': True,
                'severity_level': 'mild',
                'severity_score': 1,
                'is_proliferative': False,
                'raw_text': keywords
            }
        
        # No severity indicator found
        return {
            'has_severity': False,
            'severity_level': 'none',
            'severity_score': 0,
            'is_proliferative': False,
            'raw_text': keywords
        }
    
    def extract_patient_both_eyes(self, 
                                  left_keywords: str,
                                  right_keywords: str) -> Tuple[Dict, Dict]:
        """
        Extract severity for both eyes of a patient.
        
        Args:
            left_keywords: Left eye diagnostic keywords
            right_keywords: Right eye diagnostic keywords
            
        Returns:
            Tuple of (left_severity_dict, right_severity_dict)
        """
        left_severity = self.extract_severity(left_keywords)
        right_severity = self.extract_severity(right_keywords)
        
        return left_severity, right_severity
    
    def analyze_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze entire dataset and add severity columns.
        
        Args:
            df: DataFrame with 'Left-Diagnostic Keywords' and 'Right-Diagnostic Keywords'
            
        Returns:
            DataFrame with added severity columns
        """
        # Extract severity for all rows
        left_severities = []
        right_severities = []
        
        for _, row in df.iterrows():
            left_sev, right_sev = self.extract_patient_both_eyes(
                row.get('Left-Diagnostic Keywords', ''),
                row.get('Right-Diagnostic Keywords', '')
            )
            left_severities.append(left_sev)
            right_severities.append(right_sev)
        
        # Add columns to dataframe
        df = df.copy()
        df['Left-Severity-Level'] = [s['severity_level'] for s in left_severities]
        df['Left-Severity-Score'] = [s['severity_score'] for s in left_severities]
        df['Right-Severity-Level'] = [s['severity_level'] for s in right_severities]
        df['Right-Severity-Score'] = [s['severity_score'] for s in right_severities]
        
        return df


if __name__ == '__main__':
    # Test the severity extractor
    extractor = SeverityExtractor()
    
    print("="*80)
    print("TESTING SEVERITY EXTRACTOR")
    print("="*80)
    
    test_cases = [
        "mild nonproliferative retinopathy",
        "moderate non proliferative retinopathy",
        "severe nonproliferative retinopathy",
        "proliferative diabetic retinopathy",
        "normal fundus",
        "cataract",
        "advanced glaucoma",
        "early age-related macular degeneration",
    ]
    
    for i, keywords in enumerate(test_cases, 1):
        result = extractor.extract_severity(keywords)
        print(f"\n{i}. '{keywords}'")
        print(f"   Severity: {result['severity_level']} (score: {result['severity_score']})")
        print(f"   Has severity info: {result['has_severity']}")
        if result['is_proliferative']:
            print(f"   ⚠️  PROLIFERATIVE (most severe)")
    
    print("\n" + "="*80)
    print("Testing on real dataset...")
    print("="*80)
    
    # Load and analyze real data
    df = pd.read_excel('ODIR-5K/data.xlsx')
    df_with_severity = extractor.analyze_dataset(df)
    
    # Statistics
    left_severity_counts = df_with_severity['Left-Severity-Level'].value_counts()
    right_severity_counts = df_with_severity['Right-Severity-Level'].value_counts()
    
    print(f"\nLeft eye severity distribution:")
    for level, count in left_severity_counts.items():
        print(f"  {level}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nRight eye severity distribution:")
    for level, count in right_severity_counts.items():
        print(f"  {level}: {count} ({count/len(df)*100:.1f}%)")
    
    # Total with severity info
    total_with_severity = (
        (df_with_severity['Left-Severity-Score'] > 0).sum() +
        (df_with_severity['Right-Severity-Score'] > 0).sum()
    )
    print(f"\nTotal images with severity information: {total_with_severity} / {len(df)*2}")
    print(f"Percentage: {total_with_severity / (len(df)*2) * 100:.1f}%")
    
    print("\n✓ Severity extractor ready for use!")
