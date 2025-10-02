#!/usr/bin/env python3

"""
Debug script to test the fusion logic and understand why combined results are lower.
"""

import json
import numpy as np
from pathlib import Path

def load_fusion_mapping():
    """Load the fusion mapping file"""
    fusion_path = Path("meta/fusion_mapping.json")
    if fusion_path.exists():
        with open(fusion_path, 'r') as f:
            return json.load(f)
    return None

def simulate_predictions():
    """Simulate the predictions shown in the UI"""
    # Based on your screenshot:
    # MRI: ModerateDementia (95.8% confidence) 
    # PET: CN (91.8% confidence)
    # Cognitive: MCI (59.8% confidence)
    
    # MRI model classes: [NonDemented, VeryMildDemented, MildDemented, ModerateDemented]
    # Prediction: ModerateDemented (index 3) with 95.8% confidence
    mri_probs = np.array([0.01, 0.02, 0.02, 0.95])  # 95% on ModerateDemented
    
    # PET model classes: [CN, EMCI, LMCI, MCI, AD]
    # Prediction: CN (index 0) with 91.8% confidence
    pet_probs = np.array([0.918, 0.03, 0.02, 0.02, 0.012])  # 91.8% on CN
    
    # Cognitive (heuristic): 5-class output with MCI prediction
    # Based on your UI: MCI at 59.8%
    cognitive_probs = np.array([0.05, 0.10, 0.20, 0.598, 0.052])  # 59.8% on MCI (index 3)
    
    return mri_probs, pet_probs, cognitive_probs

def test_fusion():
    """Test the fusion logic step by step"""
    print("=== Fusion Debug Analysis ===\n")
    
    # Load fusion mapping
    fusion_mapping = load_fusion_mapping()
    if not fusion_mapping:
        print("Error: Could not load fusion mapping!")
        return
        
    print("Fusion mapping loaded:")
    print(f"Fusion classes: {fusion_mapping['fusion_classes']}")
    print(f"MRI classes: {fusion_mapping['mri_classes']}")
    print(f"PET classes: {fusion_mapping['pet_classes']}")
    print()
    
    # Get simulated predictions
    mri_probs, pet_probs, cognitive_probs = simulate_predictions()
    
    print("Individual predictions:")
    print(f"MRI probabilities: {mri_probs}")
    print(f"  -> Predicted class: {fusion_mapping['mri_classes'][np.argmax(mri_probs)]} ({mri_probs.max():.3f})")
    print(f"PET probabilities: {pet_probs}")
    print(f"  -> Predicted class: {fusion_mapping['pet_classes'][np.argmax(pet_probs)]} ({pet_probs.max():.3f})")
    print(f"Cognitive probabilities: {cognitive_probs}")
    print(f"  -> Predicted class: {fusion_mapping['fusion_classes'][np.argmax(cognitive_probs)]} ({cognitive_probs.max():.3f})")
    print()
    
    # Apply fusion mappings
    mri_to_fusion = np.array(fusion_mapping['mri_to_fusion'])
    pet_to_fusion = np.array(fusion_mapping['pet_to_fusion'])
    
    print("Mapping matrices:")
    print("MRI to fusion mapping:")
    for i, class_name in enumerate(fusion_mapping['mri_classes']):
        print(f"  {class_name}: {mri_to_fusion[i]}")
    print("PET to fusion mapping:")
    for i, class_name in enumerate(fusion_mapping['pet_classes']):
        print(f"  {class_name}: {pet_to_fusion[i]}")
    print()
    
    # Transform to fusion space
    mri_in_fusion = mri_probs @ mri_to_fusion
    pet_in_fusion = pet_probs @ pet_to_fusion
    
    print("After mapping to fusion space:")
    print(f"MRI in fusion space: {mri_in_fusion}")
    print(f"  -> Distribution: {dict(zip(fusion_mapping['fusion_classes'], mri_in_fusion))}")
    print(f"PET in fusion space: {pet_in_fusion}")
    print(f"  -> Distribution: {dict(zip(fusion_mapping['fusion_classes'], pet_in_fusion))}")
    print(f"Cognitive (already in fusion space): {cognitive_probs}")
    print(f"  -> Distribution: {dict(zip(fusion_mapping['fusion_classes'], cognitive_probs))}")
    print()
    
    # Apply fusion (simple average)
    parts = [mri_in_fusion, pet_in_fusion, cognitive_probs]
    fused = np.mean(parts, axis=0)
    fused = fused / (fused.sum() + 1e-8)  # Normalize
    
    print("Final fused prediction:")
    print(f"Fused probabilities: {fused}")
    print(f"Predicted class: {fusion_mapping['fusion_classes'][np.argmax(fused)]} ({fused.max():.3f})")
    print()
    
    # Analyze the discrepancy
    print("=== Analysis of the fusion discrepancy ===")
    print("The fusion is showing lower confidence because:")
    print(f"1. MRI predicts ModerateDemented -> Maps to AD in fusion space: {mri_in_fusion[4]:.3f}")
    print(f"2. PET predicts CN -> Maps to CN in fusion space: {pet_in_fusion[0]:.3f}")  
    print(f"3. Cognitive predicts MCI -> Already in fusion space: {cognitive_probs[3]:.3f}")
    print()
    print("The models are giving CONFLICTING predictions:")
    print("- MRI says: Severe dementia (AD)")
    print("- PET says: No dementia (CN)")
    print("- Cognitive says: Mild cognitive impairment (MCI)")
    print()
    print("When averaged, these conflicting predictions result in:")
    for i, class_name in enumerate(fusion_mapping['fusion_classes']):
        print(f"  {class_name}: {fused[i]:.3f} = ({mri_in_fusion[i]:.3f} + {pet_in_fusion[i]:.3f} + {cognitive_probs[i]:.3f}) / 3")
    print()
    print(f"The final prediction is {fusion_mapping['fusion_classes'][np.argmax(fused)]} with {fused.max():.1%} confidence")
    print("This lower confidence reflects the disagreement between modalities.")

if __name__ == "__main__":
    test_fusion()