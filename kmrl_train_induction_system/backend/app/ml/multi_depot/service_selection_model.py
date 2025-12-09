# backend/app/ml/multi_depot/service_selection_model.py
"""
Service Selection Model (Policy / Learned Ranker)
Trains a supervised ranker that outputs selection probability for each train being in SERVICE
Uses softmax sampling + top-K selection
"""
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime

from app.ml.multi_depot.config import FleetFeatures
from app.utils.cloud_database import cloud_db_manager

logger = logging.getLogger(__name__)


class ServiceSelectionDataset(Dataset):
    """Dataset for service selection"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, scores: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)  # Binary: selected for service
        self.scores = torch.FloatTensor(scores)  # Continuous score
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.scores[idx]


class ServiceSelectionModel(nn.Module):
    """Neural network for service selection ranking"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Output: service score
        )
    
    def forward(self, x):
        return self.network(x)


class ServiceSelector:
    """Service selection model with softmax sampling"""
    
    def __init__(self):
        self.model: Optional[ServiceSelectionModel] = None
        self.feature_names: List[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    async def load_model(self) -> bool:
        """Load trained model"""
        try:
            collection = await cloud_db_manager.get_collection("service_selection_models")
            doc = await collection.find_one(sort=[("meta.created_at", -1)])
            
            if not doc:
                logger.warning("No service selection model found, using fallback")
                return False
            
            import io
            blob = doc.get("blob")
            if isinstance(blob, bytes):
                buf = io.BytesIO(blob)
            else:
                buf = io.BytesIO(bytes(blob))
            
            meta = doc.get("meta", {})
            self.feature_names = meta.get("feature_names", [])
            
            self.model = torch.jit.load(buf, map_location=self.device)
            self.model.eval()
            
            logger.info(f"Loaded service selection model version {meta.get('version', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading service selection model: {e}")
            return False
    
    def extract_features(self, fleet_features: FleetFeatures, 
                        risk_prediction: Optional[Dict[str, Any]] = None,
                        dead_km_cost: float = 0.0) -> np.ndarray:
        """Extract features for service selection"""
        features = []
        
        # Risk features
        if risk_prediction:
            features.append(risk_prediction.get("risk_24h", 0.1))
            features.append(risk_prediction.get("risk_72h", 0.15))
            features.append(risk_prediction.get("health_score", 0.85))
        else:
            features.extend([0.1, 0.15, 0.85])
        
        # Branding priority
        features.append(1.0 if fleet_features.branding_flag else 0.0)
        features.append(fleet_features.branding_priority)
        
        # Recent uptime (simplified - would need historical data)
        features.append(0.95)  # Placeholder
        
        # Cleaning status
        features.append(0.0)  # Placeholder - would need cleaning schedule
        
        # Last turnout time (normalized)
        features.append(10.0 / 30.0)  # Placeholder - 10 min / 30 min max
        
        # Dead km cost (normalized)
        features.append(min(1.0, dead_km_cost / 50.0))  # Normalize to 50 km max
        
        # Job cards
        features.append(float(fleet_features.job_cards.get("critical_cards", 0)) / 10.0)
        features.append(float(fleet_features.job_cards.get("open_cards", 0)) / 20.0)
        
        # Sensor health
        features.append(fleet_features.sensor_health_score)
        
        # Mileage ratio
        mileage_ratio = fleet_features.current_mileage / fleet_features.max_mileage_before_maintenance
        if fleet_features.max_mileage_before_maintenance > 0:
            features.append(mileage_ratio)
        else:
            features.append(0.5)
        
        return np.array(features, dtype=np.float32)
    
    async def rank_trains(self, fleet_features_list: List[FleetFeatures],
                         risk_predictions: Optional[Dict[str, Dict[str, Any]]] = None,
                         required_count: int = 13,
                         temperature: float = 1.0) -> List[Dict[str, Any]]:
        """
        Rank trains for service selection using softmax sampling
        
        temperature: Controls randomness (lower = more deterministic)
        """
        if not self.model:
            await self.load_model()
        
        scored_trains = []
        
        for fleet_features in fleet_features_list:
            train_id = fleet_features.train_id
            risk_pred = risk_predictions.get(train_id) if risk_predictions else None
            
            # Extract features
            features = self.extract_features(fleet_features, risk_pred, dead_km_cost=0.0)
            
            # Predict service score
            if self.model:
                feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    score = self.model(feature_tensor)
                    service_score = float(score.item())
            else:
                service_score = self._heuristic_score(fleet_features, risk_pred)
            
            scored_trains.append({
                "train_id": train_id,
                "service_score": service_score,
                "features": features,
            })
        
        # Apply softmax to get selection probabilities
        scores = np.array([t["service_score"] for t in scored_trains])
        
        # Softmax with temperature
        exp_scores = np.exp(scores / temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Rank by score
        ranked_indices = np.argsort(-scores)
        
        # Build ranked results
        ranked_results = []
        for rank, idx in enumerate(ranked_indices, 1):
            train = scored_trains[idx]
            ranked_results.append({
                "train_id": train["train_id"],
                "service_score": train["service_score"],
                "selection_probability": float(probabilities[idx]),
                "rank": rank,
                "selected": rank <= required_count,
                "features": train["features"],
            })
        
        return ranked_results
    
    def _heuristic_score(self, fleet_features: FleetFeatures, 
                        risk_prediction: Optional[Dict[str, Any]]) -> float:
        """Fallback heuristic scoring"""
        score = 0.5
        
        if risk_prediction:
            score += (1.0 - risk_prediction.get("risk_24h", 0.1)) * 0.3
            score += risk_prediction.get("health_score", 0.85) * 0.2
        
        if fleet_features.branding_flag:
            score += 0.2
        
        score += fleet_features.sensor_health_score * 0.15
        
        if fleet_features.job_cards.get("critical_cards", 0) > 0:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    async def train(self, training_data: List[Dict[str, Any]], epochs: int = 50,
                   batch_size: int = 32, learning_rate: float = 1e-3) -> Dict[str, Any]:
        """Train the service selection model"""
        if not training_data or len(training_data) < 100:
            return {"status": "insufficient_data", "rows": len(training_data)}
        
        # Prepare data
        feature_vectors = []
        labels = []
        scores = []
        
        for sample in training_data:
            fleet_features = sample.get("fleet_features")
            risk_pred = sample.get("risk_prediction")
            
            if not fleet_features:
                continue
            
            features = self.extract_features(fleet_features, risk_pred)
            feature_vectors.append(features)
            
            labels.append(float(sample.get("selected", 0)))
            scores.append(float(sample.get("service_score", 0.5)))
        
        if not feature_vectors:
            return {"status": "no_valid_data"}
        
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(len(feature_vectors[0]))]
        
        # Convert to numpy
        X = np.array(feature_vectors, dtype=np.float32)
        y_labels = np.array(labels, dtype=np.float32)
        y_scores = np.array(scores, dtype=np.float32)
        
        # Split train/val
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_labels_train, y_labels_val = y_labels[:split_idx], y_labels[split_idx:]
        y_scores_train, y_scores_val = y_scores[:split_idx], y_scores[split_idx:]
        
        # Create datasets
        train_dataset = ServiceSelectionDataset(X_train, y_labels_train, y_scores_train)
        val_dataset = ServiceSelectionDataset(X_val, y_labels_val, y_scores_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_dim = len(self.feature_names)
        self.model = ServiceSelectionModel(input_dim=input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for features, labels_true, scores_true in train_loader:
                features = features.to(self.device)
                scores_true = scores_true.to(self.device)
                
                optimizer.zero_grad()
                scores_pred = self.model(features).squeeze()
                loss = criterion(scores_pred, scores_true)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, labels_true, scores_true in val_loader:
                    features = features.to(self.device)
                    scores_true = scores_true.to(self.device)
                    
                    scores_pred = self.model(features).squeeze()
                    loss = criterion(scores_pred, scores_true)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        # Save model
        await self._save_model()
        
        return {
            "status": "ok",
            "epochs": epochs,
            "best_val_loss": float(best_val_loss),
            "feature_names": self.feature_names,
        }
    
    async def _save_model(self):
        """Save trained model"""
        try:
            import io
            import hashlib
            
            # Convert to TorchScript
            example_input = torch.randn(1, len(self.feature_names)).to(self.device)
            self.model.eval()
            traced_model = torch.jit.trace(self.model, example_input)
            
            # Serialize
            buf = io.BytesIO()
            torch.jit.save(traced_model, buf)
            buf.seek(0)
            model_bytes = buf.getvalue()
            
            # Create metadata
            meta = {
                "version": hashlib.sha1(model_bytes).hexdigest()[:12],
                "created_at": datetime.now().isoformat(),
                "feature_names": self.feature_names,
            }
            
            # Save to database
            collection = await cloud_db_manager.get_collection("service_selection_models")
            await collection.insert_one({
                "meta": meta,
                "blob": model_bytes,
            })
            
            logger.info(f"Saved service selection model version {meta['version']}")
            
        except Exception as e:
            logger.error(f"Error saving service selection model: {e}")


