#!/usr/bin/env python3
import urllib.request
import json

def test_ranking():
    try:
        url = "http://localhost:8000/api/optimization/latest"
        headers = {
            'X-API-Key': 'kmrl_api_key_2024',
            'Content-Type': 'application/json'
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            data = response.read()
            result = json.loads(data.decode('utf-8'))
            print(f"Status Code: {response.status}")
            print(f"Number of ranked trainsets: {len(result)}")
            print(f"\nTop 10 ranked trainsets:")
            print("-" * 60)
            for i, item in enumerate(result[:10]):
                print(f"{i+1:2d}. {item['trainset_id']} - Score: {item['score']:.3f} - Decision: {item['decision']} - Confidence: {item['confidence_score']:.3f}")
            
            print(f"\nBottom 5 trainsets:")
            print("-" * 60)
            for i, item in enumerate(result[-5:]):
                rank = len(result) - 4 + i
                print(f"{rank:2d}. {item['trainset_id']} - Score: {item['score']:.3f} - Decision: {item['decision']} - Confidence: {item['confidence_score']:.3f}")
            
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_ranking()
