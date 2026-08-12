# backend/app/ml/multi_depot/demand_forecaster.py
"""
Demand & Headway Forecaster
Predicts service demand bands and required service_trains per band.
Uses Gradient Boosting or Sequence model with a deterministic fallback.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available, using fallback")

from app.utils.cloud_database import cloud_db_manager


class DemandForecaster:
    """Demand and headway forecasting service"""
    
    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.is_trained = False
    
    async def load_model(self) -> bool:
        """Load trained model"""
        try:
            collection = await cloud_db_manager.get_collection("demand_forecast_models")
            doc = await collection.find_one(sort=[("meta.created_at", -1)])
            if not doc:
                logger.warning("No demand forecast model found, using fallback")
                return False
            import pickle
            import io
            blob = doc.get("blob")
            if isinstance(blob, bytes):
                buf = io.BytesIO(blob)
            else:
                buf = io.BytesIO(bytes(blob))
            self.model = pickle.load(buf)
            meta = doc.get("meta", {})
            self.feature_names = meta.get("feature_names", [])
            self.is_trained = True
            logger.info(f"Loaded demand forecast model version {meta.get('version', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Error loading demand forecast model: {e}")
            return False
    
    def extract_features(self, date: datetime, historical_demand: Optional[List[Dict[str, Any]]] = None,
                         weather: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Extract features for demand prediction"""
        features = [
            date.weekday() / 6.0,
            date.hour / 23.0,
            date.month / 12.0,
            1.0 if date.weekday() < 5 else 0.0,
        ]
        if historical_demand:
            same_weekday_demand = [d.get("service_trains", 0) for d in historical_demand
                                   if isinstance(d, dict) and d.get("date", date).weekday() == date.weekday()]
            features.append(np.mean(same_weekday_demand) / 20.0 if same_weekday_demand else 13.0 / 20.0)
            recent_demand = [d.get("service_trains", 0) for d in historical_demand[-7:] if isinstance(d, dict)]
            features.append(np.mean(recent_demand) / 20.0 if recent_demand else 13.0 / 20.0)
        else:
            features.extend([13.0 / 20.0, 13.0 / 20.0])
        if weather:
            features.append(weather.get("temperature", 25.0) / 50.0)
            features.append(1.0 if weather.get("rain", False) else 0.0)
        else:
            features.extend([0.5, 0.0])
        return np.array(features, dtype=np.float32)
    
    async def forecast(self, date: datetime, depot_id: Optional[str] = None,
                       historical_demand: Optional[List[Dict[str, Any]]] = None,
                       weather: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        features = self.extract_features(date, historical_demand, weather)
        if self.model and self.is_trained:
            predicted_demand = self.model.predict(features.reshape(1, -1))[0]
            confidence = 0.85
        else:
            predicted_demand = self._heuristic_forecast(date, historical_demand)
            confidence = 0.6
        predicted_demand = max(10, min(20, int(round(predicted_demand))))
        demand_bands = self._generate_demand_bands(date, predicted_demand)
        return {
            "date": date.isoformat(),
            "depot_id": depot_id,
            "required_service_trains": predicted_demand,
            "demand_bands": demand_bands,
            "confidence": confidence,
            "features_used": features.tolist(),
        }
    
    def _heuristic_forecast(self, date: datetime, historical_demand: Optional[List[Dict[str, Any]]]) -> float:
        base_demand = 13.0 * (1.1 if date.weekday() < 5 else 0.9)
        if historical_demand:
            recent = [d.get("service_trains", 13) for d in historical_demand[-7:] if isinstance(d, dict)]
            if recent:
                base_demand = np.mean(recent)
        return base_demand
    
    def _generate_demand_bands(self, date: datetime, total_demand: int) -> List[Dict[str, Any]]:
        return [
            {"time_range": "06:00-10:00", "demand": int(total_demand * 0.4), "headway_minutes": 4},
            {"time_range": "10:00-17:00", "demand": int(total_demand * 0.35), "headway_minutes": 6},
            {"time_range": "17:00-22:00", "demand": int(total_demand * 0.25), "headway_minutes": 5},
        ]
    
    async def train(self, training_data: List[Dict[str, Any]], use_lightgbm: bool = True) -> Dict[str, Any]:
        if not training_data or len(training_data) < 50:
            return {"status": "insufficient_data", "rows": len(training_data)}
        X, y = [], []
        for sample in training_data:
            date_value = sample.get("date")
            date = datetime.fromisoformat(date_value.replace('Z', '+00:00')) if isinstance(date_value, str) else date_value
            X.append(self.extract_features(date, sample.get("historical_demand"), sample.get("weather")))
            y.append(float(sample.get("service_trains", 13)))
        X, y = np.array(X), np.array(y)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        if use_lightgbm and HAS_LIGHTGBM:
            self.model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        else:
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.is_trained = True
        await self._save_model()
        return {"status": "ok", "train_score": float(train_score), "val_score": float(val_score), "feature_names": self.feature_names}
    
    async def _save_model(self):
        try:
            import pickle
            import io
            import hashlib
            buf = io.BytesIO()
            pickle.dump(self.model, buf)
            model_bytes = buf.getvalue()
            meta = {"version": hashlib.sha1(model_bytes).hexdigest()[:12], "created_at": datetime.now().isoformat(), "feature_names": self.feature_names}
            collection = await cloud_db_manager.get_collection("demand_forecast_models")
            await collection.insert_one({"meta": meta, "blob": model_bytes})
            logger.info(f"Saved demand forecast model version {meta['version']}")
        except Exception as e:
            logger.error(f"Error saving demand forecast model: {e}")
