#!/usr/bin/env python3
"""Test if trainsets are updated after fitness upload"""

import urllib.request
import urllib.parse
import json

def test_trainset_data():
    """Test if T-001 has updated fitness data"""
    
    try:
        url = 'http://localhost:8000/api/trainsets/T-001'
        headers = {
            'X-API-Key': 'kmrl_api_key_2024',
            'Content-Type': 'application/json'
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            data = response.read()
            result = json.loads(data.decode('utf-8'))
            print(f"[SUCCESS] Retrieved trainset T-001")
            print(f"Trainset ID: {result.get('trainset_id')}")
            print(f"Fitness Certificates: {result.get('fitness_certificates', {})}")
            print(f"Branding: {result.get('branding', {})}")
            print(f"Last Updated Sources: {result.get('last_updated_sources', {})}")
            return True
            
    except urllib.error.HTTPError as e:
        print(f"[ERROR] HTTP Error {e.code}: {e.reason}")
        try:
            error_data = e.read().decode('utf-8')
            print(f"Error details: {error_data}")
        except:
            pass
        return False
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

if __name__ == "__main__":
    test_trainset_data()
