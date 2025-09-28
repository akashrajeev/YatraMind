#!/usr/bin/env python3
import urllib.request
import json

def test_simple_endpoint():
    try:
        url = "http://localhost:8000/api/dashboard/overview"
        headers = {
            'X-API-Key': 'kmrl_api_key_2024',
            'Content-Type': 'application/json'
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            data = response.read()
            print(f"Dashboard Status: {response.status}")
            return True
            
    except Exception as e:
        print(f"Dashboard Error: {e}")
        return False

def test_optimization_endpoint():
    try:
        url = "http://localhost:8000/api/optimization/latest"
        headers = {
            'X-API-Key': 'kmrl_api_key_2024',
            'Content-Type': 'application/json'
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            data = response.read()
            print(f"Optimization Status: {response.status}")
            return True
            
    except Exception as e:
        print(f"Optimization Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing endpoints...")
    test_simple_endpoint()
    test_optimization_endpoint()

