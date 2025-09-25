# api/index.py - Vercel entry point
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vercel_main import app

# Vercel expects the app to be available
# This is a simple wrapper for serverless functions
