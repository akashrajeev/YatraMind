#!/usr/bin/env python3
"""
Test script for KMRL Train Induction System API
Tests all endpoints systematically
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

def test_endpoint(method, endpoint, data=None, description=""):
    """Test a single API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n🧪 Testing: {description}")
    print(f"   {method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=TIMEOUT)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=TIMEOUT)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=TIMEOUT)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                if isinstance(result, list):
                    print(f"   ✅ Success: {len(result)} items returned")
                elif isinstance(result, dict):
                    print(f"   ✅ Success: {len(result)} fields returned")
                else:
                    print(f"   ✅ Success: {type(result).__name__} returned")
            except:
                print(f"   ✅ Success: {len(response.text)} characters returned")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
        return response.status_code == 200
        
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection Error: Server not running on {BASE_URL}")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout: Request took longer than {TIMEOUT} seconds")
        return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 KMRL Train Induction System - API Testing")
    print("=" * 60)
    
    # Test 1: Basic connectivity
    print("\n📡 Step 1: Testing Basic Connectivity")
    if not test_endpoint("GET", "/", description="Root endpoint"):
        print("❌ Server not accessible. Please start the server first.")
        return
    
    test_endpoint("GET", "/health", description="Health check")
    
    # Test 2: Load mock data
    print("\n📊 Step 2: Loading Mock Data")
    test_endpoint("POST", "/api/ingestion/ingest/all", description="Load all mock data")
    
    # Wait for data to be processed
    print("\n⏳ Waiting for data processing...")
    time.sleep(5)
    
    # Test 3: Check data was loaded
    print("\n🔍 Step 3: Verifying Data Load")
    test_endpoint("GET", "/api/trainsets/", description="Get all trainsets")
    test_endpoint("GET", "/api/ingestion/status", description="Check ingestion status")
    
    # Test 4: Test individual trainset operations
    print("\n🚇 Step 4: Testing Trainset Operations")
    test_endpoint("GET", "/api/trainsets/T-001", description="Get specific trainset")
    test_endpoint("GET", "/api/trainsets/T-001/fitness", description="Check fitness certificates")
    
    # Test 5: Test optimization
    print("\n🤖 Step 5: Testing AI/ML Optimization")
    optimization_request = {
        "target_date": datetime.now().isoformat(),
        "required_service_hours": 14
    }
    test_endpoint("POST", "/api/optimization/run", optimization_request, "Run optimization")
    
    # Test 6: Test constraints
    print("\n⚖️ Step 6: Testing Constraint Engine")
    test_endpoint("GET", "/api/optimization/constraints/check", description="Check constraints")
    
    # Test 7: Test stabling geometry
    print("\n🏗️ Step 7: Testing Stabling Geometry")
    test_endpoint("GET", "/api/optimization/stabling-geometry", description="Get stabling optimization")
    test_endpoint("GET", "/api/optimization/shunting-schedule", description="Get shunting schedule")
    
    # Test 8: Test dashboard
    print("\n📈 Step 8: Testing Dashboard")
    test_endpoint("GET", "/api/dashboard/overview", description="Fleet overview")
    test_endpoint("GET", "/api/dashboard/alerts", description="Active alerts")
    test_endpoint("GET", "/api/dashboard/performance", description="Performance metrics")
    
    # Test 9: Test simulation
    print("\n🎯 Step 9: Testing What-if Simulation")
    test_endpoint("GET", "/api/optimization/simulate", description="What-if simulation")
    
    print("\n" + "=" * 60)
    print("🎉 API Testing Complete!")
    print(f"📖 View API Documentation: {BASE_URL}/docs")
    print(f"📚 View ReDoc: {BASE_URL}/redoc")

if __name__ == "__main__":
    main()
