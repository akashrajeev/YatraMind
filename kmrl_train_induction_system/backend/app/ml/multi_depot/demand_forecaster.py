# backend/app/ml/multi_depot/demand_forecaster.py
"""
Demand & Headway Forecaster
Predicts service demand bands and required service_trains per band
Uses Gradient Boosting or Sequence model
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available, using fallback")

from app.utils.cloud_database import cloud_db_manager

logger = logging.getLogger(__name__)


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
            
            # Load model (LightGBM can be pickled)
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
        features = []
        
        # Temporal features
        features.append(date.weekday() / 6.0)  # Day of week (0=Monday, 6=Sunday)
        features.append(date.hour / 23.0)  # Hour of day
        features.append(date.month / 12.0)  # Month
        features.append(1.0 if date.weekday() < 5 else 0.0)  # Is weekday
        
        # Historical demand features (if available)
        if historical_demand:
            # Average demand for same day of week
            same_weekday_demand = [d.get("service_trains", 0) for d in historical_demand 
                                  if isinstance(d, dict) and d.get("date", {}).weekday() == date.weekday()]
            if same_weekday_demand:
                features.append(np.mean(same_weekday_demand) / 20.0)
            else:
                features.append(13.0 / 20.0)  # Default
            
            # Recent trend (last 7 days average)
            recent_demand = [d.get("service_trains", 0) for d in historical_demand[-7:] 
                           if isinstance(d, dict)]
            if recent_demand:
                features.append(np.mean(recent_demand) / 20.0)
            else:
                features.append(13.0 / 20.0)
        else:
            features.extend([13.0 / 20.0, 13.0 / 20.0])
        
        # Weather features (if available)
        if weather:
            features.append(weather.get("temperature", 25.0) / 50.0)
            features.append(1.0 if weather.get("rain", False) else 0.0)
        else:
            features.extend([0.5, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    async def forecast(self, date: datetime, depot_id: Optional[str] = None,
                      historical_demand: Optional[List[Dict[str, Any]]] = None,
                      weather: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Forecast demand for a specific date
        
        Returns:
        - required_service_trains: Predicted number of trains needed
        - demand_bands: List of demand bands with time ranges
        - confidence: Prediction confidence (0-1)
        """
        # Extract features
        features = self.extract_features(date, historical_demand, weather)
        
        if self.model and self.is_trained:
            # Use model
            features_2d = features.reshape(1, -1)
            predicted_demand = self.model.predict(features_2d)[0]
            confidence = 0.85  # Model confidence
        else:
            # Fallback heuristic
            predicted_demand = self._heuristic_forecast(date, historical_demand)
            confidence = 0.6
        
        # Ensure reasonable range
        predicted_demand = max(10, min(20, int(round(predicted_demand))))
        
        # Generate demand bands (simplified - could be more sophisticated)
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
        """Fallback heuristic forecasting"""
        base_demand = 13.0
        
        # Adjust for day of week
        if date.weekday() < 5:  # Weekday
            base_demand *= 1.1
        else:  # Weekend
            base_demand *= 0.9
        
        # Adjust for historical if available
        if historical_demand:
            recent = [d.get("service_trains", 13) for d in historical_demand[-7:] if isinstance(d, dict)]
            if recent:
                base_demand = np.mean(recent)
        
        return base_demand
    
    def _generate_demand_bands(self, date: datetime, total_demand: int) -> List[Dict[str, Any]]:
        """Generate demand bands with time ranges"""
        # Simplified: 3 bands (morning peak, midday, evening peak)
        bands = [
            {
                "time_range": "06:00-10:00",
                "demand": int(total_demand * 0.4),  # 40% in morning peak
                "headway_minutes": 4,
            },
            {
                "time_range": "10:00-17:00",
                "demand": int(total_demand * 0.35),  # 35% midday
                "headway_minutes": 6,
            },
            {
                "time_range": "17:00-22:00",
                "demand": int(total_demand * 0.25),  # 25% evening peak
                "headway_minutes": 5,
            },
        ]
        return bands
    
    async def train(self, training_data: List[Dict[str, Any]], 
                   use_lightgbm: bool = True) -> Dict[str, Any]:
        """
        Train demand forecasting model
        
        training_data: List of dicts with keys:
        - date, service_trains, historical_demand, weather (optional)
        """
        if not training_data or len(training_data) < 50:
            return {"status": "insufficient_data", "rows": len(training_data)}
        
        # Prepare features and labels
        X = []
        y = []
        
        for sample in training_data:
            date_str = sample.get("date")
            if isinstance(date_str, str):
                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                date = date_str
            
            features = self.extract_features(
                date,
                sample.get("historical_demand"),
                sample.get("weather")
            )
            X.append(features)
            y.append(float(sample.get("service_trains", 13)))
        
        X = np.array(X)
        y = np.array(y)
        
        # Split train/val
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        if use_lightgbm and HAS_LIGHTGBM:
            self.model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)
            
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.is_trained = True
        else:
            # Fallback: simple linear regression
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.is_trained = True
        
        # Save model
        await self._save_model()
        
        return {
            "status": "ok",
            "train_score": float(train_score),
            "val_score": float(val_score),
            "feature_names": self.feature_names,
        }
    
    async def _save_model(self):
        """Save trained model"""
        try:
            import pickle
            import io
            import hashlib
            
            # Serialize model
            buf = io.BytesIO()
            pickle.dump(self.model, buf)
            buf.seek(0)
            model_bytes = buf.getvalue()
            
            # Create metadata
            meta = {
                "version": hashlib.sha1(model_bytes).hexdigest()[:12],
                "created_at": datetime.now().isoformat(),
                "feature_names": self.feature_names,
            }
            
            # Save to database
            collection = await cloud_db_manager.get_collection("demand_forecast_models")
            await collection.insert_one({
                "meta": meta,
                "blob": model_bytes,
            })
            
            logger.info(f"Saved demand forecast model version {meta['version']}")
            
        except Exception as e:
            logger.error(f"Error saving demand forecast model: {e}")


