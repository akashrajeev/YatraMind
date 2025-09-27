#!/usr/bin/env python3
"""
KMRL Operations Dashboard - Development Startup Script
This script sets up and starts the development environment for the KMRL Operations Dashboard.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    # Check if Python version is 3.8 or higher
    if sys.version_info.major < 3 or (sys.version_info.major == 3 and sys.version_info.minor < 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import pymongo
        print("✅ Core dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r backend/requirements.txt")
        sys.exit(1)

def setup_environment():
    """Setup environment variables"""
    env_file = Path("backend/.env")
    env_example = Path("backend/env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        with open(env_example, 'r') as f:
            content = f.read()
        with open(env_file, 'w') as f:
            f.write(content)
        print("✅ .env file created. Please update with your actual values.")
    elif env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  No .env file found. Using default values.")

def start_backend():
    """Start the FastAPI backend server"""
    print("\n🚀 Starting KMRL Operations Dashboard Backend...")
    backend_dir = Path("backend")
    os.chdir(backend_dir)
    
    try:
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload",
            "--log-level", "info"
        ], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Backend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start backend: {e}")
        sys.exit(1)

def start_frontend():
    """Start the React frontend development server"""
    print("\n🎨 Starting KMRL Operations Dashboard Frontend...")
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return
    
    os.chdir(frontend_dir)
    
    try:
        # Check if node_modules exists
        if not Path("node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        # Start the development server
        subprocess.run(["npm", "run", "dev"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Frontend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start frontend: {e}")
        print("Please run: cd frontend && npm install && npm run dev")

def main():
    """Main startup function"""
    print("🏗️  KMRL Operations Dashboard - Development Environment")
    print("=" * 60)
    
    # Check requirements
    check_python_version()
    check_dependencies()
    setup_environment()
    
    # Ask user what to start
    print("\nWhat would you like to start?")
    print("1. Backend only (FastAPI)")
    print("2. Frontend only (React)")
    print("3. Both (Backend + Frontend)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        start_backend()
    elif choice == "2":
        start_frontend()
    elif choice == "3":
        print("\n🔄 Starting both backend and frontend...")
        print("Backend will be available at: http://localhost:8000")
        print("Frontend will be available at: http://localhost:3000")
        print("API Documentation: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop both servers")
        
        # Start backend in a separate process
        import threading
        backend_thread = threading.Thread(target=start_backend)
        backend_thread.daemon = True
        backend_thread.start()
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        # Start frontend
        start_frontend()
    elif choice == "4":
        print("👋 Goodbye!")
        sys.exit(0)
    else:
        print("❌ Invalid choice")
        sys.exit(1)

if __name__ == "__main__":
    main()
