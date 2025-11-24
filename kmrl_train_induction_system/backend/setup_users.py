#!/usr/bin/env python3
"""
Script to seed initial users into the database
Run this script once to create the 6 predefined admin users
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.auth_service import auth_service
from app.models.user import UserRole
from app.utils.cloud_database import cloud_db_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Predefined users: username - password
USERS = [
    {"username": "adithkp", "password": "Adith@123", "name": "Adith KP", "role": UserRole.OPERATIONS_MANAGER},
    {"username": "akashrajeevkv", "password": "Akash@123", "name": "Akash Rajeev KV", "role": UserRole.OPERATIONS_MANAGER},
    {"username": "abindasp", "password": "Abindas@123", "name": "Abindas P", "role": UserRole.OPERATIONS_MANAGER},
    {"username": "alanb", "password": "Alan2123", "name": "Alan B", "role": UserRole.OPERATIONS_MANAGER},
    {"username": "pradyodhp", "password": "Pradyodh@123", "name": "Pradyodh P", "role": UserRole.OPERATIONS_MANAGER},
    {"username": "poojacv", "password": "Pooja@123", "name": "Pooja CV", "role": UserRole.OPERATIONS_MANAGER},
]


async def setup_users():
    """Create all predefined users in the database"""
    try:
        # Connect to MongoDB
        await cloud_db_manager.connect_mongodb()
        logger.info("Connected to MongoDB")
        
        collection = await cloud_db_manager.get_collection("users")
        
        # Check if users already exist
        existing_count = await collection.count_documents({})
        if existing_count > 0:
            logger.warning(f"Found {existing_count} existing users. Skipping user creation.")
            logger.info("To recreate users, delete existing users from the database first.")
            return
        
        # Create each user
        created_count = 0
        for user_data in USERS:
            try:
                # Check if user already exists
                existing = await collection.find_one({"username": user_data["username"]})
                if existing:
                    logger.info(f"User {user_data['username']} already exists, skipping...")
                    continue
                
                # Create user
                user = await auth_service.create_user(
                    username=user_data["username"],
                    password=user_data["password"],
                    name=user_data["name"],
                    role=user_data["role"].value,
                    email=f"{user_data['username']}@kmrl.in"  # Generate email from username
                )
                logger.info(f"✓ Created user: {user_data['username']} ({user_data['name']})")
                created_count += 1
                
            except Exception as e:
                logger.error(f"Error creating user {user_data['username']}: {e}")
        
        logger.info(f"\n✓ Successfully created {created_count} users")
        logger.info("Users can now login with their username and password")
        
    except Exception as e:
        logger.error(f"Error setting up users: {e}")
        raise
    finally:
        await cloud_db_manager.close_all()


if __name__ == "__main__":
    asyncio.run(setup_users())

