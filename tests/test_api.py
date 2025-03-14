#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the sentiment analysis API.
"""

import unittest
import json
import base64
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add the parent directory to the path so we can import the API
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the API - if it fails, skip the tests
try:
    from direct_api import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Basic authentication helper
def get_auth_header(username, password):
    token = base64.b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")
    return {'Authorization': f'Basic {token}'}

@pytest.mark.skipif(not API_AVAILABLE, reason="API module not available")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.auth_headers = get_auth_header("admin", "password123")
        cls.headers = {**cls.auth_headers, "Content-Type": "application/json"}
    
    def test_root_endpoint(self):
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Welcome" in data["message"]
    
    def test_test_endpoint(self):
        response = self.client.get("/test")
        assert response.status_code == 200
        data = response.json()
        assert "test" in data
        assert "time" in data
        assert data["test"] == "successful"
    
    def test_health_endpoint(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_prediction_positive(self):
        positive_text = "I love this product! It's amazing and works perfectly."
        payload = {"text": positive_text}
        
        response = self.client.post(
            "/predict",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 401:
            print(f"Auth failed: {response.json()}")
            print(f"Headers used: {self.headers}")
        
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert any(key in data for key in ["probability", "confidence"])
        assert data["sentiment"].lower() == "positive"
    
    def test_prediction_negative(self):
        negative_text = "This is terrible. I hate it and it doesn't work at all."
        payload = {"text": negative_text}
        
        response = self.client.post(
            "/predict",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 401:
            print(f"Auth failed: {response.json()}")
            print(f"Headers used: {self.headers}")
        
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert any(key in data for key in ["probability", "confidence"])
        assert data["sentiment"].lower() == "negative"

if __name__ == "__main__":
    unittest.main() 