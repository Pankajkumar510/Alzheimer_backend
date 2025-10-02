#!/usr/bin/env python3

"""
Test the new fusion confidence calculation logic with simulated data.
"""

import numpy as np
import json
from pathlib import Path

# Add the src directory to the path so we can import the InferenceEngine
import sys
sys.path.append('.')

from src.inference.predict import InferenceEngine

def test_fusion_confidence_calculation():
    """Test that fusion confidence is calculated as average of individual confidences"""
    
    print("=== Testing New Fusion Confidence Calculation ===\n")
    
    # Initialize the inference engine with fusion mapping
    engine = InferenceEngine(
        mri_classes_json="meta/mri_classes.json",
        pet_classes_json="meta/pet_classes.json", 
        fusion_mapping_json="meta/fusion_mapping.json"
    )
    
    # Create a mock prediction result that simulates what we see in the UI
    mock_result = {
        "patient_id": "test_fusion",
        "inputs": {"mri_path": None, "pet_path": None, "cognitive": None},
        "predictions": {
            "mri": {
                "label": "ModerateDemented",
                "probabilities": [0.01, 0.02, 0.02, 0.95]  # 95% confidence on ModerateDemented
            },
            "pet": {
                "label": "CN", 
                "probabilities": [0.918, 0.03, 0.02, 0.02, 0.012]  # 91.8% confidence on CN
            },
            "cognitive": {
                "label": "MCI",
                "probabilities": [0.05, 0.10, 0.20, 0.598, 0.052]  # 59.8% confidence on MCI
            }
        },
        "explainability": {},
        "debug": {}
    }
    
    # Extract individual confidences
    mri_confidence = max(mock_result["predictions"]["mri"]["probabilities"])
    pet_confidence = max(mock_result["predictions"]["pet"]["probabilities"]) 
    cognitive_confidence = max(mock_result["predictions"]["cognitive"]["probabilities"])
    
    print(f"Individual confidences:")
    print(f"  MRI: {mri_confidence:.3f} ({mri_confidence*100:.1f}%)")
    print(f"  PET: {pet_confidence:.3f} ({pet_confidence*100:.1f}%)")
    print(f"  Cognitive: {cognitive_confidence:.3f} ({cognitive_confidence*100:.1f}%)")
    
    # Calculate expected combined confidence (average of MRI and PET only)
    expected_combined = (mri_confidence + pet_confidence) / 2
    print(f"\nExpected combined confidence (MRI + PET avg): {expected_combined:.3f} ({expected_combined*100:.1f}%)")
    
    # Test the fusion mapping
    fusion_mapping = engine.fusion
    if fusion_mapping:
        print(f"\nFusion mapping loaded successfully:")
        print(f"  Fusion classes: {fusion_mapping['fusion_classes']}")
        
        # Apply fusion mappings manually to verify
        mri_to_fusion = np.array(fusion_mapping['mri_to_fusion'])
        pet_to_fusion = np.array(fusion_mapping['pet_to_fusion'])
        
        mri_probs = np.array(mock_result["predictions"]["mri"]["probabilities"])
        pet_probs = np.array(mock_result["predictions"]["pet"]["probabilities"])
        cognitive_probs = np.array(mock_result["predictions"]["cognitive"]["probabilities"])
        
        # Transform to fusion space
        mri_in_fusion = mri_probs @ mri_to_fusion
        pet_in_fusion = pet_probs @ pet_to_fusion
        
        print(f"\nAfter mapping to fusion space:")
        print(f"  MRI in fusion: {mri_in_fusion}")
        print(f"  PET in fusion: {pet_in_fusion}")
        print(f"  Cognitive in fusion: {cognitive_probs}")
        
        # Apply original fusion (for comparison)
        parts = [mri_in_fusion, pet_in_fusion, cognitive_probs]
        fused_original = np.mean(parts, axis=0)
        fused_original = fused_original / (fused_original.sum() + 1e-8)
        original_confidence = fused_original.max()
        predicted_class_idx = int(np.argmax(fused_original))
        
        print(f"\nOriginal fusion result:")
        print(f"  Predicted class: {fusion_mapping['fusion_classes'][predicted_class_idx]}")
        print(f"  Original confidence: {original_confidence:.3f} ({original_confidence*100:.1f}%)")
        
        # Apply new fusion logic
        individual_confidences = [mri_confidence, pet_confidence]
        combined_confidence = np.mean(individual_confidences)
        
        # Create adjusted probabilities 
        fused_adjusted = fused_original.copy()
        fused_adjusted[predicted_class_idx] = combined_confidence
        
        # Redistribute remaining probability
        remaining_prob = 1.0 - combined_confidence
        other_indices = [i for i in range(len(fused_adjusted)) if i != predicted_class_idx]
        if other_indices and remaining_prob > 0:
            other_probs_sum = sum(fused_original[i] for i in other_indices)
            if other_probs_sum > 0:
                for i in other_indices:
                    fused_adjusted[i] = (fused_original[i] / other_probs_sum) * remaining_prob
            else:
                for i in other_indices:
                    fused_adjusted[i] = remaining_prob / len(other_indices)
        
        new_confidence = fused_adjusted.max()
        
        print(f"\nNew fusion result:")
        print(f"  Predicted class: {fusion_mapping['fusion_classes'][predicted_class_idx]}")
        print(f"  New confidence: {new_confidence:.3f} ({new_confidence*100:.1f}%)")
        print(f"  Adjusted probabilities: {fused_adjusted}")
        
        print(f"\nComparison:")
        print(f"  Expected: {expected_combined*100:.1f}%")
        print(f"  Actual: {new_confidence*100:.1f}%")
        print(f"  Match: {'✅ YES' if abs(new_confidence - expected_combined) < 0.001 else '❌ NO'}")
        
    else:
        print("Fusion mapping not loaded!")

if __name__ == "__main__":
    test_fusion_confidence_calculation()