"""
Test the FastAPI backend
"""

import requests
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n[1] Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        print("✓ Health check passed")
        print(f"  Response: {response.json()}")
        return True
    else:
        print(f"✗ Health check failed: {response.status_code}")
        return False

def test_synergy():
    """Test synergy calculation"""
    print("\n[2] Testing synergy calculation...")
    
    data = {
        "drug_a_effect": 0.20,
        "drug_b_effect": 0.30,
        "combination_effect": 0.70,
        "model": "bliss"
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/synergy/calculate", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Synergy calculation passed")
        print(f"  Synergy score: {result['synergy_score']:.2f}")
        print(f"  Interpretation: {result['interpretation']}")
        return True
    else:
        print(f"✗ Synergy calculation failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False

def test_models_list():
    """Test models listing"""
    print("\n[3] Testing models list...")
    
    response = requests.get(f"{BASE_URL}/api/v1/segmentation/models")
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Models list passed")
        print(f"  Available models: {result['models']}")
        return True
    else:
        print(f"✗ Models list failed: {response.status_code}")
        return False

def main():
    print("="*50)
    print("  ADDS Backend API Tests")
    print("="*50)
    print(f"\nTesting backend at: {BASE_URL}")
    print("Make sure the backend is running (start_backend.bat)")
    print()
    
    results = []
    
    # Run tests
    results.append(test_health())
    results.append(test_synergy())
    results.append(test_models_list())
    
    # Summary
    print("\n" + "="*50)
    print(f"  Results: {sum(results)}/{len(results)} tests passed")
    print("="*50)
    
    if all(results):
        print("\n✓ All tests passed! Backend is working correctly.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Check the backend logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
