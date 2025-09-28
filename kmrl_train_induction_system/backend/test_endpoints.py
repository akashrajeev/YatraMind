#!/usr/bin/env python3
import urllib.request
import urllib.parse
import json

def test_endpoint(endpoint, description):
    try:
        url = f"http://localhost:8000/api{endpoint}"
        headers = {
            'X-API-Key': 'kmrl_api_key_2024',
            'Content-Type': 'application/json'
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            data = response.read()
            print(f"[OK] {description}: Status {response.status}")
            return True
            
    except Exception as e:
        print(f"[ERROR] {description}: Error - {e}")
        return False

def main():
    endpoints = [
        ("/dashboard/overview", "Dashboard Overview"),
        ("/dashboard/alerts", "Dashboard Alerts"),
        ("/dashboard/performance", "Dashboard Performance"),
        ("/v1/assignments/", "Assignments List"),
        ("/v1/assignments/summary", "Assignments Summary"),
        ("/optimization/latest", "Optimization Latest"),
        ("/trainsets/", "Trainsets List"),
        ("/ingestion/status", "Ingestion Status"),
    ]
    
    print("Testing API Endpoints:")
    print("=" * 50)
    
    working = 0
    total = len(endpoints)
    
    for endpoint, description in endpoints:
        if test_endpoint(endpoint, description):
            working += 1
    
    print("=" * 50)
    print(f"Results: {working}/{total} endpoints working")

if __name__ == "__main__":
    main()
