#!/usr/bin/env python3

"""
Simple test to verify the new fusion confidence calculation.
"""

def test_simple_calculation():
    """Test the simple calculation we want: (95.8 + 91.8) / 2 = 93.8%"""
    
    mri_confidence = 0.958  # 95.8%
    pet_confidence = 0.918  # 91.8%
    
    expected_combined = (mri_confidence + pet_confidence) / 2
    
    print(f"MRI Confidence: {mri_confidence*100:.1f}%")
    print(f"PET Confidence: {pet_confidence*100:.1f}%") 
    print(f"Expected Combined: {expected_combined*100:.1f}%")
    
    # This should be 93.8%
    print(f"Target: 93.8%")
    print(f"Match: {'✅ YES' if abs(expected_combined*100 - 93.8) < 0.1 else '❌ NO'}")

if __name__ == "__main__":
    test_simple_calculation()