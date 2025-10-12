"""
Test the API endpoints.
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_vocab():
    """Test vocabulary endpoint."""
    print("Testing /vocab...")
    response = requests.get(f"{BASE_URL}/vocab")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_analyze():
    """Test analyze endpoint."""
    print("Testing /analyze...")
    payload = {
        "sequence": "MVHLTPEEKS"
    }
    response = requests.post(f"{BASE_URL}/analyze", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_generate():
    """Test generate endpoint."""
    print("Testing /generate...")
    payload = {
        "max_length": 50,
        "temperature": 1.0,
        "num_sequences": 2
    }
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Generated {len(result['sequences'])} sequences:")
    for i, seq in enumerate(result['sequences']):
        print(f"  {i+1}. {seq}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Protogen API")
    print("=" * 50 + "\n")
    
    try:
        test_health()
        test_vocab()
        test_analyze()
        test_generate()
        
        print("All tests passed!")
    
    except Exception as e:
        print(f"Error: {e}")

