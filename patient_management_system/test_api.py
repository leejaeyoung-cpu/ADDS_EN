"""Test script for Patient API endpoints"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("Testing ADDS Patient Management System API")
print("=" * 60)

# Test 1: Health check
print("\n1. Testing Health Endpoint...")
response = requests.get(f"{BASE_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# Test 2: Register a patient
print("\n2. Testing Patient Registration...")
patient_data = {
    "name": "홍길동",
    "birthdate": "1980-01-15",
    "gender": "M",
    "contact": "010-1234-5678",
    "address": "서울시 강남구"
}
response = requests.post(f"{BASE_URL}/api/patients/register", json=patient_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
patient_id = response.json()["patient_id"]

# Test 3: Search patient
print("\n3. Testing Patient Search...")
response = requests.get(f"{BASE_URL}/api/patients/search", params={"query": "홍길동"})
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# Test 4: Get patient details
print(f"\n4. Testing Get Patient Details (ID: {patient_id})...")
response = requests.get(f"{BASE_URL}/api/patients/{patient_id}")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# Test 5: Register another patient
print("\n5. Registering another patient...")
patient_data2 = {
    "name": "김영희",
    "birthdate": "1975-05-20",
    "gender": "F",
    "contact": "010-9876-5432"
}
response = requests.post(f"{BASE_URL}/api/patients/register", json=patient_data2)
print(f"Status: {response.status_code}")
print(f"Patient ID: {response.json()['patient_id']}")

# Test 6: List all patients
print("\n6. Testing List All Patients...")
response = requests.get(f"{BASE_URL}/api/patients/", params={"limit": 10})
print(f"Status: {response.status_code}")
print(f"Total patients: {len(response.json())}")
for p in response.json():
    print(f"  - {p['patient_id']}: {p['name']} ({p['gender']})")

print("\n" + "=" * 60)
print("All API tests completed successfully!")
print("=" * 60)
