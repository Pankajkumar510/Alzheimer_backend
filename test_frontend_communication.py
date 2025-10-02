#!/usr/bin/env python3

"""
Test the frontend-backend communication with browser-like requests.
"""

import requests
import json
from pathlib import Path

def test_frontend_like_request():
    """Test the API exactly like the frontend would call it"""
    
    print("=== Testing Frontend-like Communication ===")
    
    # Test 1: Check if API is accessible with CORS headers like a browser would
    print("1. Testing CORS preflight...")
    try:
        response = requests.options("http://localhost:8000/predict", 
                                   headers={
                                       "Origin": "http://localhost:5173",
                                       "Access-Control-Request-Method": "POST",
                                       "Access-Control-Request-Headers": "content-type"
                                   })
        print(f"   CORS preflight status: {response.status_code}")
        print(f"   CORS headers: {response.headers.get('Access-Control-Allow-Origin', 'Not set')}")
    except Exception as e:
        print(f"   CORS preflight failed: {e}")
    
    # Test 2: Test with actual browser-like headers
    print("\\n2. Testing with browser-like headers...")
    
    # Use one of the uploaded files
    uploads_dir = Path("data/uploads")
    mri_files = list(uploads_dir.glob("patient_*_mri_*.jpg"))
    
    if mri_files:
        most_recent_mri = max(mri_files, key=lambda x: x.stat().st_mtime)
        patient_id_from_filename = most_recent_mri.name.split('_')[1]
        pet_files = list(uploads_dir.glob(f"patient_{patient_id_from_filename}_pet_*.jpg"))
        
        if pet_files:
            most_recent_pet = pet_files[0]
            
            print(f"   Using files: {most_recent_mri.name}, {most_recent_pet.name}")
            
            try:
                with open(most_recent_mri, 'rb') as mri_file, open(most_recent_pet, 'rb') as pet_file:
                    # Mimic browser request
                    files = {
                        'mri_file': ('mri_scan.jpg', mri_file.read(), 'image/jpeg'),
                        'pet_file': ('pet_scan.jpg', pet_file.read(), 'image/jpeg')
                    }
                    data = {
                        'patient_id': f'frontend_test_{patient_id_from_filename}',
                        'cognitive': json.dumps({
                            "orientation_1": "2025",
                            "orientation_2": "winter", 
                            "orientation_3": "december",
                            "memory_1": "about the same",
                            "memory_2": "sometimes",
                            "attention_1": "65",
                            "language_1": "rarely",
                            "language_2": "never"
                        })
                    }
                    headers = {
                        "Origin": "http://localhost:5173",
                        "Referer": "http://localhost:5173/"
                    }
                    
                    response = requests.post("http://localhost:8000/predict", 
                                           files=files, 
                                           data=data,
                                           headers=headers)
                    
                    print(f"   Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"   API Response Status: {result.get('status')}")
                        
                        if 'result' in result:
                            predictions = result['result'].get('predictions', {})
                            print(f"   Predictions found: {list(predictions.keys())}")
                            
                            for modality, pred in predictions.items():
                                if 'probabilities' in pred:
                                    confidence = max(pred['probabilities']) * 100
                                    print(f"     {modality}: {pred.get('label', 'N/A')} ({confidence:.1f}%)")
                        
                        # Test 3: Check the response format that frontend expects
                        print("\\n3. Checking response format for frontend compatibility...")
                        
                        # The frontend likely expects specific fields - let's check what we're returning
                        expected_structure = {
                            "status": "ok", 
                            "result": {
                                "patient_id": "string",
                                "predictions": {
                                    "mri": {"label": "string", "probabilities": []},
                                    "pet": {"label": "string", "probabilities": []},
                                    "fusion": {"label": "string", "probabilities": []}
                                }
                            }
                        }
                        
                        print(f"   Response has 'status': {'✓' if 'status' in result else '✗'}")
                        print(f"   Response has 'result': {'✓' if 'result' in result else '✗'}")
                        if 'result' in result:
                            print(f"   Result has 'predictions': {'✓' if 'predictions' in result['result'] else '✗'}")
                            if 'predictions' in result['result']:
                                preds = result['result']['predictions']
                                print(f"   Has MRI prediction: {'✓' if 'mri' in preds else '✗'}")
                                print(f"   Has PET prediction: {'✓' if 'pet' in preds else '✗'}")
                                print(f"   Has fusion prediction: {'✓' if 'fusion' in preds else '✗'}")
                    
                    else:
                        print(f"   Error: {response.status_code}")
                        print(f"   Response: {response.text}")
                        
            except Exception as e:
                print(f"   Request failed: {e}")
        else:
            print("   No PET file found for testing")
    else:
        print("   No uploaded files found for testing")

    # Test 4: Check if there are any recent prediction requests in logs
    print("\\n4. Checking for recent API activity...")
    try:
        # Check for any recent uploads that might indicate frontend activity
        all_files = list(uploads_dir.glob("patient_*.jpg"))
        recent_files = [f for f in all_files if f.stat().st_mtime > (Path().stat().st_mtime - 3600)]  # Last hour
        print(f"   Recent uploads (last hour): {len(recent_files)}")
        
        if recent_files:
            print("   Most recent frontend uploads:")
            for f in sorted(recent_files, key=lambda x: x.stat().st_mtime)[-3:]:
                print(f"     {f.name} at {f.stat().st_mtime}")
    except Exception as e:
        print(f"   Could not check recent files: {e}")

if __name__ == "__main__":
    test_frontend_like_request()