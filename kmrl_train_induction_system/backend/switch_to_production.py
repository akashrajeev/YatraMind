#!/usr/bin/env python3
"""
Switch KMRL System to Production Mode
This script switches the system from mock data to real cloud services
"""

import os
import shutil
from pathlib import Path

def switch_to_production():
    """Switch the system to use production cloud services"""
    
    print("🚀 Switching KMRL System to Production Mode")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("   Please copy env_template.txt to .env and configure your credentials.")
        return False
    
    # Backup current cloud_database.py
    cloud_db_file = Path("app/utils/cloud_database.py")
    cloud_db_backup = Path("app/utils/cloud_database_mock.py")
    
    if cloud_db_file.exists():
        shutil.copy2(cloud_db_file, cloud_db_backup)
        print(f"✅ Backed up mock database manager to {cloud_db_backup}")
    
    # Replace with production version
    production_file = Path("app/utils/cloud_database_production.py")
    if production_file.exists():
        shutil.copy2(production_file, cloud_db_file)
        print("✅ Switched to production cloud database manager")
    else:
        print("❌ Production database manager not found!")
        return False
    
    # Update imports in cloud_database.py
    update_imports_in_cloud_database()
    
    print("\n🎉 System switched to production mode!")
    print("\nNext steps:")
    print("1. Run: py -3.11 setup_cloud_services.py")
    print("2. Start server: py -3.11 -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("3. Test: py -3.11 test_api.py")
    
    return True

def switch_to_mock():
    """Switch the system back to mock mode"""
    
    print("🔄 Switching KMRL System to Mock Mode")
    print("=" * 50)
    
    # Restore mock version
    cloud_db_file = Path("app/utils/cloud_database.py")
    cloud_db_backup = Path("app/utils/cloud_database_mock.py")
    
    if cloud_db_backup.exists():
        shutil.copy2(cloud_db_backup, cloud_db_file)
        print("✅ Switched back to mock database manager")
    else:
        print("❌ Mock database manager backup not found!")
        return False
    
    print("\n🎉 System switched to mock mode!")
    print("\nYou can now run the server without cloud service credentials.")
    
    return True

def update_imports_in_cloud_database():
    """Update imports in the cloud_database.py file"""
    cloud_db_file = Path("app/utils/cloud_database.py")
    
    if cloud_db_file.exists():
        content = cloud_db_file.read_text()
        
        # Replace the cloud_db_manager instantiation
        old_line = "cloud_db_manager = CloudDatabaseManager()"
        new_line = "cloud_db_manager = ProductionCloudDatabaseManager()"
        
        if old_line in content:
            content = content.replace(old_line, new_line)
            cloud_db_file.write_text(content)
            print("✅ Updated cloud database manager instantiation")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "production":
            switch_to_production()
        elif mode == "mock":
            switch_to_mock()
        else:
            print("Usage: python switch_to_production.py [production|mock]")
    else:
        print("🚀 KMRL System Mode Switcher")
        print("=" * 30)
        print("1. Switch to Production Mode")
        print("2. Switch to Mock Mode")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            switch_to_production()
        elif choice == "2":
            switch_to_mock()
        elif choice == "3":
            print("Goodbye!")
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
