#!/usr/bin/env python3
"""
Test script to check the /predict API endpoint
"""

import requests
import json


def test_api_endpoint():
    """Test the prediction API endpoint."""
    
    url = "http://localhost:8000/predict"
    
    print("Testing /predict endpoint with cognitive-only data...")
    
    # Test with minimal cognitive data
    data = {
        'patient_id': 'test_patient_123',
        'cognitive': json.dumps({
            'orientation_1': '2025',
            'orientation_2': 'winter',
            'orientation_3': 'december',
            'memory_1': 'about the same',
            'memory_2': 'sometimes'
        })
    }
    
    try:
        print(f"Sending POST request to {url}")
        print(f"Data: {data}")
        
        response = requests.post(url, data=data, timeout=30)
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                json_response = response.json()
                print("Response JSON:")
                print(json.dumps(json_response, indent=2))
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                print(f"Raw response: {response.text}")
        else:
            print(f"Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("Request timed out - API may be hanging")
    except requests.exceptions.ConnectionError:
        print("Connection error - API server may not be running")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    test_api_endpoint()