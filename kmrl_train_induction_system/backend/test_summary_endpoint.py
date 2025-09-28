#!/usr/bin/env python3
import urllib.request
import json

def test_summary_endpoint():
    try:
        url = "http://localhost:8000/api/v1/assignments/summary"
        headers = {
            'X-API-Key': 'kmrl_api_key_2024',
            'Content-Type': 'application/json'
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            data = response.read()
            result = json.loads(data.decode('utf-8'))
            print(f"Status Code: {response.status}")
            print(f"Response: {json.dumps(result, indent=2)}")
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_summary_endpoint()
