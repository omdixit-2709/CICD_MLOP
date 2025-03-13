import requests
import json
import base64
import os

# Basic authentication helper
def basic_auth(username, password):
    token = base64.b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")
    return {'Authorization': f'Basic {token}'}

# Base URL for the API - can be overridden with environment variable
base_url = os.environ.get("API_URL", "http://localhost:8004")

print(f"Testing Sentiment Analysis API at {base_url}...\n")

# Test the root endpoint
try:
    response = requests.get(f"{base_url}/")
    print(f"Root Endpoint - Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("Root endpoint test successful!\n")
except Exception as e:
    print(f"Error testing root endpoint: {str(e)}\n")

# Test the test endpoint
try:
    response = requests.get(f"{base_url}/test")
    print(f"Test Endpoint - Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("Test endpoint successful!\n")
except Exception as e:
    print(f"Error testing test endpoint: {str(e)}\n")

# Test the health endpoint
try:
    response = requests.get(f"{base_url}/health")
    print(f"Health Endpoint - Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("Health endpoint test successful!\n")
except Exception as e:
    print(f"Error testing health endpoint: {str(e)}\n")

# Test the prediction endpoint
try:
    headers = basic_auth("admin", "password")
    headers['Content-Type'] = 'application/json'
    
    # Test with positive text
    positive_text = "I love this product! It's amazing and works perfectly."
    payload = {"text": positive_text}
    
    response = requests.post(
        f"{base_url}/predict",
        headers=headers,
        data=json.dumps(payload)
    )
    
    print(f"Prediction Endpoint (Positive) - Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("Prediction endpoint test successful (positive example)!\n")
    
    # Test with negative text
    negative_text = "This is terrible. I hate it and it doesn't work at all."
    payload = {"text": negative_text}
    
    response = requests.post(
        f"{base_url}/predict",
        headers=headers,
        data=json.dumps(payload)
    )
    
    print(f"Prediction Endpoint (Negative) - Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("Prediction endpoint test successful (negative example)!\n")
    
except Exception as e:
    print(f"Error testing prediction endpoint: {str(e)}\n")

print("API testing complete!") 