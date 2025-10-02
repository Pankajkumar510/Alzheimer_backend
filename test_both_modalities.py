#!/usr/bin/env python3

"""
Test with both MRI and PET files to demonstrate proper multimodal results.
"""

import requests
import json
from pathlib import Path

def test_with_both_files():
    """Test API with both MRI and PET files"""
    
    print("=== Testing with Both MRI and PET Files ===")
    
    uploads_dir = Path("data/uploads")
    
    # Find an MRI file
    mri_files = [f for f in uploads_dir.glob("patient_*_mri_*.jpg")]
    # Find a PET file  
    pet_files = [f for f in uploads_dir.glob("patient_*_pet_*.jpg")]
    
    if not mri_files:
        print("No MRI files found!")
        return
        
    if not pet_files:
        print("No PET files found!")
        return
    
    # Use different files for MRI and PET
    mri_file = mri_files[0]
    pet_file = pet_files[0]
    
    print(f"Using MRI file: {mri_file.name}")
    print(f"Using PET file: {pet_file.name}")
    print(f"Files are different: {'✓' if mri_file.name != pet_file.name else '✗ SAME FILE!'}")
    
    url = "http://localhost:8000/predict"
    
    try:
        with open(mri_file, 'rb') as mf, open(pet_file, 'rb') as pf:
            files = {
                'mri_file': ('mri_scan.jpg', mf.read(), 'image/jpeg'),
                'pet_file': ('pet_scan.jpg', pf.read(), 'image/jpeg')
            }
            data = {
                'patient_id': 'both_modalities_test',
                'cognitive': json.dumps({
                    "orientation_1": "2025",
                    "orientation_2": "winter", 
                    "orientation_3": "december",
                    "memory_1": "about the same",
                    "memory_2": "sometimes"
                })
            }
            
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"\\n=== Results with Both Files ===")
                print(f"Status: {result['status']}")
                
                if 'result' in result and 'predictions' in result['result']:
                    predictions = result['result']['predictions']
                    
                    for modality, pred in predictions.items():
                        if 'probabilities' in pred:
                            confidence = max(pred['probabilities']) * 100
                            print(f"{modality.upper()}: {pred.get('label', 'N/A')} ({confidence:.1f}%)")
                    
                    # Check fusion calculation debug
                    if 'debug' in result['result'] and 'fusion_calculation' in result['result']['debug']:
                        debug = result['result']['debug']['fusion_calculation']
                        print(f"\\n=== Fusion Debug Info ===")
                        print(f"Fusion method: {debug.get('fusion_method', 'N/A')}")
                        print(f"Modalities used: {debug.get('modalities_used', 0)}")
                        print(f"Individual confidences: {debug.get('individual_confidences', [])}")
                        print(f"Combined confidence: {debug.get('combined_confidence', 'N/A'):.3f}")
                        
                        if debug.get('modalities_used', 0) >= 2:
                            print("✅ SUCCESS: Using both MRI and PET for fusion!")
                            mri_conf = debug.get('individual_confidences', [0, 0])[0]
                            pet_conf = debug.get('individual_confidences', [0, 0])[1] 
                            expected = (mri_conf + pet_conf) / 2
                            print(f"Expected average: {expected:.3f}")
                        else:
                            print("❌ Only using single modality for fusion")
                
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_with_both_files()