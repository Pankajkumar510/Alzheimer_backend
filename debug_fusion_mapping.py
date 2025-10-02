#!/usr/bin/env python3

"""
Debug script to check if fusion mapping is loaded correctly.
"""

import sys
sys.path.append('.')

from src.inference.predict import InferenceEngine

def debug_fusion_mapping():
    """Debug fusion mapping loading"""
    
    print("=== Debugging Fusion Mapping ===")
    
    # Initialize engine same way as API does
    engine = InferenceEngine(
        mri_classes_json="meta/mri_classes.json",
        pet_classes_json="meta/pet_classes.json", 
        fusion_mapping_json="meta/fusion_mapping.json"
    )
    
    print(f"Fusion mapping exists: {engine.fusion is not None}")
    
    if engine.fusion:
        print("Fusion mapping details:")
        print(f"  Fusion classes: {engine.fusion.get('fusion_classes', [])}")
        print(f"  MRI classes: {engine.fusion.get('mri_classes', [])}")
        print(f"  PET classes: {engine.fusion.get('pet_classes', [])}")
        print(f"  mri_R exists: {engine.fusion.get('mri_R') is not None}")
        print(f"  pet_R exists: {engine.fusion.get('pet_R') is not None}")
        
        if engine.fusion.get('mri_R') is not None:
            print(f"  mri_R shape: {engine.fusion['mri_R'].shape}")
        if engine.fusion.get('pet_R') is not None:
            print(f"  pet_R shape: {engine.fusion['pet_R'].shape}")
    else:
        print("No fusion mapping loaded!")
        
    print(f"\nMRI class to idx: {engine.mri_class_to_idx}")
    print(f"PET class to idx: {engine.pet_class_to_idx}")
    
    # Check the condition that determines which fusion path is taken
    condition1 = engine.fusion and (engine.fusion.get("mri_R") is not None or engine.fusion.get("pet_R") is not None)
    condition2 = engine.fusion_enabled_direct
    
    print(f"\nFusion path conditions:")
    print(f"  Fusion mapping path (condition1): {condition1}")
    print(f"  Direct fusion path (condition2): {condition2}")
    
    if condition1:
        print("  -> Will use fusion mapping path (our modified code)")
    elif condition2:
        print("  -> Will use direct fusion path")
    else:
        print("  -> No fusion will be applied")

if __name__ == "__main__":
    debug_fusion_mapping()