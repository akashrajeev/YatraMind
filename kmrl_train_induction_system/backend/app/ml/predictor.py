# backend/app/ml/predictor.py
from __future__ import annotations

from typing import Dict, Any, List
import io
import numpy as np
import pandas as pd
import torch
import shap
import random
import logging
import asyncio

from app.utils.cloud_database import cloud_db_manager
from app.config import settings

logger = logging.getLogger(__name__)

# Seed for deterministic ML inference
_ml_seed_initialized = False

def _ensure_deterministic_seeding():
    """Ensure all RNGs are seeded for deterministic ML inference."""
    global _ml_seed_initialized
    if not _ml_seed_initialized:
        seed = settings.ml_deterministic_seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Set PyTorch to deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        _ml_seed_initialized = True
        logger.info(f"ML inference seeded with seed={seed} for determinism")


async def load_latest_model() -> Dict[str, Any]:
    col = await cloud_db_manager.get_collection("models")
    doc = await col.find_one(sort=[("meta.created_at", -1)])
    if not doc:
        raise RuntimeError("No model registered")
    meta = doc.get("meta", {})
    blob = doc.get("blob")
    if isinstance(blob, bytes):
        buf = io.BytesIO(blob)
    else:
        buf = io.BytesIO(bytes(blob))
    model = torch.jit.load(buf)
    model.eval()
    return {"model": model, "meta": meta}


async def batch_predict(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch predict with deterministic inference.
    
    Ensures all RNGs are seeded for reproducible results.
    """
    if not features:
        return []
    
    # Ensure deterministic seeding
    _ensure_deterministic_seeding()
    
    # Log input hash for reproducibility tracking
    import hashlib
    input_hash = hashlib.md5(str(sorted(features)).encode()).hexdigest()[:8]
    logger.debug(f"ML batch_predict called with {len(features)} features, input_hash={input_hash}")
    
    bundle = await load_latest_model()
    model = bundle["model"]
    meta = bundle["meta"]
    feature_cols = meta.get("features", [])
    
    # Log model version if available
    model_version = meta.get("version", "unknown")
    logger.debug(f"Using ML model version: {model_version}")

    df = pd.DataFrame(features)
    # Center with feature means if available
    means = meta.get("feature_means", {})
    df_centered = df.reindex(columns=feature_cols).fillna(0.0)
    for c in feature_cols:
        if c in means:
            df_centered[c] = df_centered[c] - float(means[c])
    X = df_centered.astype(np.float32).values
    
    # Set model to eval mode and disable dropout for deterministic inference
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(X)).squeeze(1).numpy()

    # SHAP kernel approximation for top feature attributions (lightweight)
    # Run in separate thread to avoid blocking event loop
    try:
        top_idx = await asyncio.to_thread(_calculate_shap, model, X, feature_cols)
    except Exception as e:
        logger.warning(f"SHAP calculation failed: {e}")
        top_idx = None

    out: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        item = {
            "trainset_id": row.get("trainset_id"),
            "risk_prob": float(p[i]),
        }
        if top_idx is not None:
            item["top_features"] = [feature_cols[j] for j in top_idx[i]]
        out.append(item)
    return out


def _calculate_shap(model, X, feature_cols):
    """Run SHAP calculation (CPU intensive)."""
    try:
        # Define prediction wrapper for SHAP
        def predict_fn(z):
            with torch.no_grad():
                tensor_z = torch.from_numpy(z.astype(np.float32))
                return model(tensor_z).detach().numpy()
                
        explainer = shap.Explainer(predict_fn, feature_names=feature_cols)
        shap_vals = explainer(X, max_evals=100)
        contrib = np.abs(shap_vals.values)
        return np.argsort(-contrib, axis=1)[:, :3]
    except Exception as e:
        logger.warning(f"SHAP internal error: {e}")
        return None


def predict_maintenance_health(trainset: Dict[str, Any]) -> float:
    """Lightweight heuristic predictor for maintenance health score (0-1).
    
    This provides a quick health assessment when full ML model is unavailable
    or as a fallback. Higher score = better health.
    
    Args:
        trainset: Trainset dictionary with features
        
    Returns:
        Health score between 0.0 and 1.0
    """
    try:
        health = 0.85  # Default baseline
        
        # Factor 1: Job cards (critical cards reduce health significantly)
        job_cards = trainset.get("job_cards", {})
        if isinstance(job_cards, dict):
            critical_cards = int(job_cards.get("critical_cards", 0))
            open_cards = int(job_cards.get("open_cards", 0))
            
            # Critical cards have severe impact
            if critical_cards > 0:
                health -= 0.5  # Major penalty
            # Open cards have moderate impact
            if open_cards > 5:
                health -= 0.1 * min(0.5, (open_cards - 5) / 10)
        
        # Factor 2: Mileage ratio (higher mileage = lower health)
        current_mileage = float(trainset.get("current_mileage", 0))
        max_mileage = float(trainset.get("max_mileage_before_maintenance", 50000))
        if max_mileage > 0:
            mileage_ratio = current_mileage / max_mileage
            if mileage_ratio > 0.9:
                health -= 0.2  # Approaching maintenance limit
            elif mileage_ratio > 0.7:
                health -= 0.1  # Moderate wear
        
        # Factor 3: Fitness certificates
        fitness_certs = trainset.get("fitness_certificates", {})
        if isinstance(fitness_certs, dict):
            expired_count = sum(
                1 for cert in fitness_certs.values()
                if isinstance(cert, dict) and str(cert.get("status", "")).upper() == "EXPIRED"
            )
            if expired_count > 0:
                health -= 0.3  # Expired certs are critical
        
        # Factor 4: Sensor health (if available)
        sensor_health = trainset.get("sensor_health_score")
        if isinstance(sensor_health, (int, float)):
            # Blend sensor health with heuristic (70% sensor, 30% heuristic)
            health = 0.7 * float(sensor_health) + 0.3 * health
        
        # Factor 5: Predicted failure risk (if available from ML)
        predicted_risk = trainset.get("predicted_failure_risk")
        if isinstance(predicted_risk, (int, float)):
            risk = float(predicted_risk)
            # Invert risk to health (high risk = low health)
            health = 0.6 * health + 0.4 * (1.0 - risk)
        
        # Clamp to valid range
        return max(0.0, min(1.0, health))
        
    except Exception as e:
        # Safe fallback on any error
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error in predict_maintenance_health: {e}, using default")
        return 0.85  # Default safe value


