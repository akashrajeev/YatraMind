#!/usr/bin/env python3
"""
Quick Start Script for KMRL Production System
This script automates the entire setup process for production deployment
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "motor",
        "influxdb-client",
        "redis",
        "paho-mqtt",
        "pydantic",
        "pydantic-settings"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"   ✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to install {package}")
                return False
    
    return True

def check_env_file():
    """Check if .env file exists and is configured"""
    print("\n🔍 Checking environment configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("   Please copy env_template.txt to .env and configure your credentials.")
        return False
    
    # Check if .env has real credentials (not template values)
    env_content = env_file.read_text()
    if "your_username" in env_content or "your_password" in env_content:
        print("⚠️  .env file contains template values!")
        print("   Please update .env with your actual cloud service credentials.")
        return False
    
    print("✅ .env file configured")
    return True

async def run_production_setup():
    """Run the production setup process"""
    print("\n🚀 Starting production setup...")
    
    try:
        # Import and run the setup script
        from setup_cloud_services import CloudServiceSetup
        
        setup = CloudServiceSetup()
        
        # Test all connections
        print("\n🔍 Testing cloud service connections...")
        mongodb_ok = await setup.test_mongodb_connection()
        influxdb_ok = await setup.test_influxdb_connection()
        redis_ok = await setup.test_redis_connection()
        mqtt_ok = await setup.test_mqtt_connection()
        
        # Print summary
        setup.print_summary()
        
        # Load production data if all connections successful
        if all([mongodb_ok, influxdb_ok, redis_ok]):
            print("\n📊 Loading production data...")
            await setup.load_production_data()
            return True
        else:
            print("\n❌ Some cloud services failed to connect.")
            print("   Please check your .env configuration and try again.")
            return False
            
    except Exception as e:
        print(f"\n❌ Production setup failed: {e}")
        return False

def start_production_server():
    """Start the production server"""
    print("\n🚀 Starting production server...")
    
    try:
        # Start the server
        cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
        print(f"   Command: {' '.join(cmd)}")
        print("   Server will start in the background...")
        
        # Start server in background
        process = subprocess.Popen(cmd)
        
        print("✅ Production server started!")
        print("   URL: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        print("   Press Ctrl+C to stop the server")
        
        # Wait for user to stop
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            process.terminate()
            process.wait()
            print("✅ Server stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

async def main():
    """Main function"""
    print("🚀 KMRL Train Induction System - Production Quick Start")
    print("=" * 60)
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        return
    
    # Step 2: Check environment configuration
    if not check_env_file():
        print("\n❌ Environment configuration check failed!")
        return
    
    # Step 3: Switch to production mode
    print("\n🔄 Switching to production mode...")
    try:
        from switch_to_production import switch_to_production
        if not switch_to_production():
            print("\n❌ Failed to switch to production mode!")
            return
    except Exception as e:
        print(f"\n❌ Failed to switch to production mode: {e}")
        return
    
    # Step 4: Run production setup
    if not await run_production_setup():
        print("\n❌ Production setup failed!")
        return
    
    # Step 5: Start production server
    print("\n🎉 Production setup complete!")
    start_production_server()

if __name__ == "__main__":
    asyncio.run(main())
