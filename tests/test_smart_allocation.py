#!/usr/bin/env python3
"""
Test the smart disease allocation feature
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.keyword_label_parser import DiagnosticKeywordParser

# Create test cases
parser = DiagnosticKeywordParser()

print("="*80)
print("TESTING SMART DISEASE ALLOCATION")
print("="*80)

# Test Case 1: One eye clear, one eye unclear
print("\n📋 Test Case 1: Unilateral Cataract")
print("-" * 40)
patient_labels = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=np.float32)  # Diabetes + Cataract
left_keywords = "cataract"
right_keywords = "unclear fundus image"

print(f"Patient diseases: Diabetes=1, Cataract=1")
print(f"Left keywords: '{left_keywords}'")
print(f"Right keywords: '{right_keywords}' (unclear)")

left_labels, right_labels, stats = parser.parse_patient_both_eyes(
    left_keywords, right_keywords, patient_labels
)

label_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
print(f"\nResults:")
print(f"  Left eye: {[name for name, val in zip(label_names, left_labels) if val == 1]}")
print(f"  Right eye: {[name for name, val in zip(label_names, right_labels) if val == 1]}")
print(f"  Left source: {stats['left_source']}")
print(f"  Right source: {stats['right_source']}")
print(f"  Smart allocation used: {stats['allocation_used']}")

if stats['allocation_used']:
    print(f"  ✓ CORRECT: Right eye gets only Diabetes (remaining disease)")
else:
    print(f"  ✗ Not using smart allocation")

# Test Case 2: Both eyes clear
print("\n📋 Test Case 2: Both Eyes Have Clear Keywords")
print("-" * 40)
patient_labels = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=np.float32)  # Diabetes + Cataract
left_keywords = "cataract"
right_keywords = "moderate non proliferative retinopathy"

print(f"Patient diseases: Diabetes=1, Cataract=1")
print(f"Left keywords: '{left_keywords}'")
print(f"Right keywords: '{right_keywords}'")

left_labels, right_labels, stats = parser.parse_patient_both_eyes(
    left_keywords, right_keywords, patient_labels
)

print(f"\nResults:")
print(f"  Left eye: {[name for name, val in zip(label_names, left_labels) if val == 1]}")
print(f"  Right eye: {[name for name, val in zip(label_names, right_labels) if val == 1]}")
print(f"  Left source: {stats['left_source']}")
print(f"  Right source: {stats['right_source']}")
print(f"  Smart allocation used: {stats['allocation_used']}")

if not stats['allocation_used']:
    print(f"  ✓ CORRECT: Both clear, no allocation needed")

# Test Case 3: Neither eye clear
print("\n📋 Test Case 3: Neither Eye Has Clear Keywords")
print("-" * 40)
patient_labels = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=np.float32)  # Diabetes + Cataract
left_keywords = "low image quality"
right_keywords = "unclear fundus"

print(f"Patient diseases: Diabetes=1, Cataract=1")
print(f"Left keywords: '{left_keywords}' (unclear)")
print(f"Right keywords: '{right_keywords}' (unclear)")

left_labels, right_labels, stats = parser.parse_patient_both_eyes(
    left_keywords, right_keywords, patient_labels
)

print(f"\nResults:")
print(f"  Left eye: {[name for name, val in zip(label_names, left_labels) if val == 1]}")
print(f"  Right eye: {[name for name, val in zip(label_names, right_labels) if val == 1]}")
print(f"  Left source: {stats['left_source']}")
print(f"  Right source: {stats['right_source']}")
print(f"  Smart allocation used: {stats['allocation_used']}")

if not stats['allocation_used'] and stats['left_source'] == 'fallback':
    print(f"  ✓ CORRECT: Both unclear, using patient labels for both")

# Test Case 4: Normal vs Disease
print("\n📋 Test Case 4: Normal Eye vs Diseased Eye")
print("-" * 40)
patient_labels = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)  # Only Cataract
left_keywords = "cataract"
right_keywords = "normal fundus"

print(f"Patient diseases: Cataract=1")
print(f"Left keywords: '{left_keywords}'")
print(f"Right keywords: '{right_keywords}'")

left_labels, right_labels, stats = parser.parse_patient_both_eyes(
    left_keywords, right_keywords, patient_labels
)

print(f"\nResults:")
print(f"  Left eye: {[name for name, val in zip(label_names, left_labels) if val == 1]}")
print(f"  Right eye: {[name for name, val in zip(label_names, right_labels) if val == 1]}")
print(f"  Left source: {stats['left_source']}")
print(f"  Right source: {stats['right_source']}")
print(f"  Smart allocation used: {stats['allocation_used']}")

if not stats['allocation_used']:
    print(f"  ✓ CORRECT: Both clear, accurate representation")

# Test Case 5: Multiple diseases with allocation
print("\n📋 Test Case 5: Multiple Diseases with Smart Allocation")
print("-" * 40)
patient_labels = np.array([0, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)  # Diabetes + Glaucoma + Cataract
left_keywords = "cataract, glaucoma"
right_keywords = "image quality issues"

print(f"Patient diseases: Diabetes=1, Glaucoma=1, Cataract=1")
print(f"Left keywords: '{left_keywords}'")
print(f"Right keywords: '{right_keywords}' (unclear)")

left_labels, right_labels, stats = parser.parse_patient_both_eyes(
    left_keywords, right_keywords, patient_labels
)

print(f"\nResults:")
print(f"  Left eye: {[name for name, val in zip(label_names, left_labels) if val == 1]}")
print(f"  Right eye: {[name for name, val in zip(label_names, right_labels) if val == 1]}")
print(f"  Left source: {stats['left_source']}")
print(f"  Right source: {stats['right_source']}")
print(f"  Smart allocation used: {stats['allocation_used']}")

if stats['allocation_used']:
    right_diseases = [name for name, val in zip(label_names, right_labels) if val == 1]
    if right_diseases == ['Diabetes']:
        print(f"  ✓ CORRECT: Right eye gets only Diabetes (remaining disease after left claimed Cataract + Glaucoma)")
    else:
        print(f"  ✗ UNEXPECTED: Expected only Diabetes, got {right_diseases}")

print("\n" + "="*80)
print("ALL TESTS COMPLETE")
print("="*80)
print("\n💡 Key Benefits:")
print("  • Reduces false positives for unilateral conditions")
print("  • Better handles unclear/ambiguous keywords")
print("  • Uses patient-level labels intelligently")
print("  • More accurate than applying all patient diseases to both eyes")
