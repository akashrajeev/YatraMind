import asyncio
import sys
import os

# Add backend to path so we can import app.config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
import httpx

async def verify():
    print("--- Gemini API Verification ---")
    key = settings.gemini_api_key
    if not key:
        print("[FAIL] No GEMINI_API_KEY found in settings.")
        print("Please ensure GEMINI_API_KEY is set in backend/.env")
        return

    print(f"[PASS] Key found: {key[:5]}...{key[-5:]}")
    
    # First list models to see what is available
    print("Listing available models...")
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(list_url, timeout=10.0)
            if resp.status_code == 200:
                models = resp.json().get('models', [])
                print(f"Found {len(models)} models.")
                for m in models:
                    if 'generateContent' in m.get('supportedGenerationMethods', []):
                        print(f" - {m['name']}")
            else:
                print(f"[FAIL] ListModels Error: {resp.text}")
                return
    except Exception as e:
        print(f"[ERROR] ListModels Connection failed: {e}")
        return

    print("\nAttempting API call with 'gemini-2.0-flash'...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
    payload = {
        "contents": [{
            "parts": [{"text": "Test connection. Reply with 'OK'."}]
        }]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Simple retry logic for 429 warnings
            max_retries = 3
            for attempt in range(max_retries):
                resp = await client.post(url, json=payload, timeout=10.0)
                
                if resp.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"[WARN] Rate limited (429). Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                print(f"Status Code: {resp.status_code}")
                if resp.status_code == 200:
                    print(f"[SUCCESS] Response: {resp.json()}")
                else:
                    print(f"[FAIL] GenerateContent Error after retries: {resp.text}")
                break
            else:
                # Loop completed without break = all attempts failed
                print(f"[FAIL] Rate limit exceeded after {max_retries} retries.")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify())
