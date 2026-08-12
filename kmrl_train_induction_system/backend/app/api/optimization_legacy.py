# Legacy optimization routes preserved for compatibility; primary execution lives in app.api.optimization.
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import json
import logging
import uuid
from pathlib import Path
from math import ceil

from app.models.trainset import OptimizationRequest, InductionDecision, StablingGeometryResponse
from app.services.optimizer import TrainInductionOptimizer
from app.services.stabling_optimizer import StablingGeometryOptimizer
from app.services.optimization_store import get_latest_decisions, get_decisions_from_history
from app.utils.cloud_database import cloud_db_manager
from app.utils.explainability import generate_comprehensive_explanation, render_explanation_html, render_explanation_text
from app.config import settings
from app.security import require_api_key, require_role, require_permission
from app.models.user import UserRole, User
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)
