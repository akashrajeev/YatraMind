#!/usr/bin/env python3
"""Simple test for fitness certificate upload using urllib"""

import urllib.request
import urllib.parse
import json

def test_fitness_upload():
    """Test fitness certificate upload"""
    
    # Create the CSV content
    csv_content = """trainset_id,dept,certificate,status,valid_from,valid_to,issued_by,certificate_id
T-001,ROLLING_STOCK,ROLLING_STOCK,VALID,2024-01-01,2024-12-31,KMRL_CERT_AUTH,RC-2024-001
T-001,SIGNALLING,SIGNALLING,VALID,2024-01-15,2024-12-31,KMRL_SIG_AUTH,SC-2024-001
T-001,TELECOM,TELECOM,VALID,2024-02-01,2024-12-31,KMRL_TEL_AUTH,TC-2024-001
T-002,ROLLING_STOCK,ROLLING_STOCK,VALID,2024-01-01,2024-12-31,KMRL_CERT_AUTH,RC-2024-002
T-002,SIGNALLING,SIGNALLING,EXPIRED,2023-12-31,2023-12-31,KMRL_SIG_AUTH,SC-2023-002"""
    
    print("Testing fitness certificate upload...")
    print("CSV Content:")
    print(csv_content)
    print("\n" + "="*50)
    
    try:
        # Create multipart form data
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        
        # Build the multipart body
        body_parts = []
        body_parts.append(f'--{boundary}')
        body_parts.append('Content-Disposition: form-data; name="file"; filename="test_fitness.csv"')
        body_parts.append('Content-Type: text/csv')
        body_parts.append('')
        body_parts.append(csv_content)
        body_parts.append(f'--{boundary}--')
        body_parts.append('')
        
        body = '\r\n'.join(body_parts).encode('utf-8')
        
        # Create request
        url = 'http://localhost:8000/api/ingestion/fitness/upload'
        headers = {
            'X-API-Key': 'kmrl_api_key_2024',
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Content-Length': str(len(body))
        }
        
        req = urllib.request.Request(url, data=body, headers=headers, method='POST')
        
        print(f"Making request to: {url}")
        print(f"Headers: {headers}")
        print(f"Body length: {len(body)}")
        
        with urllib.request.urlopen(req) as response:
            data = response.read()
            result = json.loads(data.decode('utf-8'))
            print(f"[SUCCESS] Upload successful! Status: {response.status}")
            print(f"Response: {result}")
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
    test_fitness_upload()
