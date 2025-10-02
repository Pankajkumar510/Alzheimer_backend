#!/usr/bin/env python3

"""
Test the new fusion confidence calculation that averages individual MRI and PET confidences.
"""

import requests
import json

def test_api_with_images():
    """Test the API with actual images to see the new fusion confidence"""
    
    url = "http://localhost:8000/predict"
    
    # You can add actual image files here if needed
    data = {
        'patient_id': 'test_new_fusion',
        'cognitive': json.dumps({
            "orientation_1": "2025",
            "orientation_2": "winter", 
            "orientation_3": "december",
            "memory_1": "about the same",
            "memory_2": "sometimes",
            "attention_1": "65",
            "language_1": "rarely",
            "language_2": "never",
            "visuospatial_1": "sometimes",
            "lifestyle_1": "16",
            "lifestyle_2": "4", 
            "lifestyle_3": "no"
        })
    }
    
    try:
        response = requests.post(url, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            print("=== New Fusion Confidence Test ===")
            print(f"Status: {result['status']}")
            
            if 'result' in result and 'predictions' in result['result']:
                predictions = result['result']['predictions']
                
                # Extract individual confidences
                mri_confidence = None
                pet_confidence = None
                fusion_confidence = None
                
                if 'mri' in predictions:
                    mri_probs = predictions['mri']['probabilities']
                    mri_confidence = max(mri_probs) * 100
                    print(f"MRI Confidence: {mri_confidence:.1f}%")
                    
                if 'pet' in predictions:
                    pet_probs = predictions['pet']['probabilities']
                    pet_confidence = max(pet_probs) * 100
                    print(f"PET Confidence: {pet_confidence:.1f}%")
                    
                if 'cognitive' in predictions:
                    cog_probs = predictions['cognitive']['probabilities']
                    cog_confidence = max(cog_probs) * 100
                    print(f"Cognitive Confidence: {cog_confidence:.1f}%")
                    
                if 'fusion' in predictions:
                    fusion_probs = predictions['fusion']['probabilities']
                    fusion_confidence = max(fusion_probs) * 100
                    print(f"Combined Confidence: {fusion_confidence:.1f}%")
                    print(f"Combined Prediction: {predictions['fusion']['label']}")
                    
                # Calculate expected average
                if mri_confidence and pet_confidence:
                    expected_avg = (mri_confidence + pet_confidence) / 2
                    print(f"Expected Average: {expected_avg:.1f}%")
                    
                    if fusion_confidence:
                        print(f"Difference: {abs(fusion_confidence - expected_avg):.1f}%")
                        
                        if abs(fusion_confidence - expected_avg) < 0.1:
                            print("✅ SUCCESS: Combined confidence matches average of individual confidences!")
                        else:
                            print("❌ ISSUE: Combined confidence doesn't match expected average")
                            
            else:
                print("No predictions found in response")
                print(json.dumps(result, indent=2))
                
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the backend is running on localhost:8000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api_with_images()