#!/usr/bin/env python3

"""
Test the API directly with the uploaded files to see if they can be processed.
"""

import requests
import json
from pathlib import Path

def test_with_uploaded_files():
    """Test API with the most recent uploaded files"""
    
    # Get the most recent uploaded files
    uploads_dir = Path("data/uploads")
    
    # Find most recent MRI and PET files for the same patient
    files = list(uploads_dir.glob("patient_*_mri_*.jpg"))
    
    if not files:
        print("No uploaded MRI files found")
        return
        
    # Get the most recent MRI file
    most_recent_mri = max(files, key=lambda x: x.stat().st_mtime)
    patient_id_from_filename = most_recent_mri.name.split('_')[1]
    
    # Find corresponding PET file for same patient
    pet_files = list(uploads_dir.glob(f"patient_{patient_id_from_filename}_pet_*.jpg"))
    
    if not pet_files:
        print(f"No PET file found for patient {patient_id_from_filename}")
        return
        
    most_recent_pet = pet_files[0]
    
    print(f"Testing with:")
    print(f"  MRI file: {most_recent_mri}")
    print(f"  PET file: {most_recent_pet}")
    print(f"  Patient ID: patient_{patient_id_from_filename}")
    
    # Test the API
    url = "http://localhost:8000/predict"
    
    try:
        with open(most_recent_mri, 'rb') as mri_file, open(most_recent_pet, 'rb') as pet_file:
            files = {
                'mri_file': ('mri_test.jpg', mri_file, 'image/jpeg'),
                'pet_file': ('pet_test.jpg', pet_file, 'image/jpeg')
            }
            data = {
                'patient_id': f'test_patient_{patient_id_from_filename}',
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
                print("\n=== API Response ===")
                print(f"Status: {result['status']}")
                
                if 'result' in result and 'predictions' in result['result']:
                    predictions = result['result']['predictions']
                    
                    print("\nPredictions:")
                    for modality, pred in predictions.items():
                        if 'probabilities' in pred:
                            confidence = max(pred['probabilities']) * 100
                            print(f"  {modality}: {pred['label']} ({confidence:.1f}%)")
                        else:
                            print(f"  {modality}: {pred}")
                            
                    # Check debug info
                    if 'debug' in result['result']:
                        debug = result['result']['debug']
                        print(f"\nDebug info:")
                        print(f"  MRI model loaded: {debug.get('mri', {}).get('loaded', False)}")
                        print(f"  PET model loaded: {debug.get('pet', {}).get('loaded', False)}")
                        print(f"  Cognitive mode: {debug.get('cognitive', {}).get('mode', 'N/A')}")
                        
                        if 'fusion_calculation' in debug:
                            fusion_calc = debug['fusion_calculation']
                            print(f"\nFusion calculation debug:")
                            print(f"  Individual confidences: {fusion_calc.get('individual_confidences', [])}")
                            print(f"  Combined confidence: {fusion_calc.get('combined_confidence', 'N/A')}")
                            print(f"  Original fused max: {fusion_calc.get('original_fused_max', 'N/A')}")
                            print(f"  Fusion method: {fusion_calc.get('fusion_method', 'N/A')}")
                        else:
                            print("  No fusion calculation debug info found")
                else:
                    print("No predictions found in response")
                    print(json.dumps(result, indent=2))
                    
            else:
                print(f"API Error: {response.status_code}")
                print(response.text)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_with_uploaded_files()