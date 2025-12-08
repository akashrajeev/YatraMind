#!/usr/bin/env python3
"""
Script to seed initial users into the database
Run this script once to create the 6 predefined admin users
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.auth_service import auth_service
from app.models.user import UserRole
from app.utils.cloud_database import cloud_db_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Predefined users: username - password (order as provided by client)
USERS = [
    {"username": "adithkp", "password": "Adith@123", "name": "Adith KP", "role": UserRole.ADMIN},
    {"username": "akashrajeevkv", "password": "Akash@123", "name": "Akash Rajeev KV", "role": UserRole.ADMIN},
    {"username": "pradyodhp", "password": "Pradyodh P", "name": "Pradyodh P", "role": UserRole.ADMIN},
    {"username": "abindasp", "password": "Abindas@123", "name": "Abindas P", "role": UserRole.ADMIN},
    {"username": "alanb", "password": "Alan B", "name": "Alan B", "role": UserRole.ADMIN},
    {"username": "poojacv", "password": "Pooja@123", "name": "Pooja CV", "role": UserRole.ADMIN},
]


async def setup_users():
    """Create all predefined users in the database"""
    try:
        # Connect to MongoDB
        await cloud_db_manager.connect_mongodb()
        logger.info("Connected to MongoDB")
        
        collection = await cloud_db_manager.get_collection("users")
        
        created_count = 0
        updated_count = 0
        for user_data in USERS:
            try:
                # Hash password using auth_service helper
                hashed_password = auth_service.get_password_hash(user_data["password"])

                # Upsert user (create if not exists, update if exists)
                result = await collection.update_one(
                    {"username": user_data["username"]},
                    {"$set": {
                        "username": user_data["username"],
                        "name": user_data["name"],
                        "email": f"{user_data['username']}@kmrl.in",
                        "role": user_data["role"].value,
                        "permissions": [],
                        "is_active": True,
                        "is_approved": True,  # Pre-approved admins
                        "hashed_password": hashed_password,
                        "updated_at": datetime.utcnow(),
                    },
                    "$setOnInsert": {
                        "id": str(uuid.uuid4()),
                        "created_at": datetime.utcnow(),
                    }},
                    upsert=True
                )

                if result.upserted_id:
                    created_count += 1
                    logger.info(f"✓ Created user: {user_data['username']} ({user_data['name']})")
                elif result.modified_count > 0:
                    updated_count += 1
                    logger.info(f"✓ Updated user: {user_data['username']} ({user_data['name']})")
                else:
                    logger.info(f"User {user_data['username']} already up-to-date")
                
            except Exception as e:
                logger.error(f"Error creating/updating user {user_data['username']}: {e}")
        
        logger.info(f"\n✓ Created: {created_count}, Updated: {updated_count}")
        logger.info("Admins can now login with their username and password")
        
    except Exception as e:
        logger.error(f"Error setting up users: {e}")
        raise
    finally:
        await cloud_db_manager.close_all()


if __name__ == "__main__":
    asyncio.run(setup_users())

