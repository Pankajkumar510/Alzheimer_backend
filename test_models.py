#!/usr/bin/env python3
"""
Test script to verify that the trained models can be loaded correctly.
"""

import sys
from pathlib import Path

# Add the prediction script to path
sys.path.append(str(Path(__file__).parent))

def test_model_loading():
    """Test if both MRI and PET models can be loaded."""
    try:
        from predict_with_trained_models import AlzheimerPredictor
        
        print("Testing model loading...")
        predictor = AlzheimerPredictor()
        
        print(f"\nModel Status:")
        print(f"  MRI Model: {'✓ Loaded' if predictor.mri_model is not None else '✗ Not loaded'}")
        print(f"  PET Model: {'✓ Loaded' if predictor.pet_model is not None else '✗ Not loaded'}")
        
        print(f"\nClass Mappings:")
        print(f"  MRI Classes: {predictor.mri_classes}")
        print(f"  PET Classes: {predictor.pet_classes}")
        print(f"  Fusion Classes: {predictor.fusion_classes}")
        
        print(f"\nModel Paths:")
        print(f"  MRI: {predictor.mri_model_path}")
        print(f"  PET: {predictor.pet_model_path}")
        print(f"  MRI Exists: {predictor.mri_model_path.exists()}")
        print(f"  PET Exists: {predictor.pet_model_path.exists()}")
        
        if predictor.mri_model is not None and predictor.pet_model is not None:
            print("\n✓ SUCCESS: Both models loaded successfully!")
            return True
        else:
            print("\n✗ WARNING: One or both models failed to load")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loading()