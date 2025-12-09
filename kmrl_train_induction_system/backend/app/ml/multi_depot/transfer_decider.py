# backend/app/ml/multi_depot/transfer_decider.py
"""
Inter-Depot Transfer Decider (Learned)
Decides when to relocate stock across depots overnight
Uses cost model: transfer_dead_km + downtime vs expected improvement in service reliability
"""
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime

from app.ml.multi_depot.config import DepotConfig, FleetFeatures
from app.utils.cloud_database import cloud_db_manager

logger = logging.getLogger(__name__)


class TransferDecisionModel(nn.Module):
    """Binary classifier for transfer decisions"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Transfer probability
        )
    
    def forward(self, x):
        return self.network(x)


class TransferDecider:
    """Inter-depot transfer decision service"""
    
    def __init__(self):
        self.model: Optional[TransferDecisionModel] = None
        self.feature_names: List[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    async def load_model(self) -> bool:
        """Load trained model"""
        try:
            collection = await cloud_db_manager.get_collection("transfer_decision_models")
            doc = await collection.find_one(sort=[("meta.created_at", -1)])
            
            if not doc:
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
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading transfer decision model: {e}")
            return False
    
    def extract_features(self, train_features: FleetFeatures,
                        from_depot: DepotConfig, to_depot: DepotConfig,
                        predicted_demand: Dict[str, int],
                        risk_prediction: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Extract features for transfer decision"""
        features = []
        
        # Distance between depots (dead km cost)
        if from_depot.coordinates and to_depot.coordinates:
            # Simplified distance calculation
            lat_diff = abs(from_depot.coordinates[0] - to_depot.coordinates[0])
            lon_diff = abs(from_depot.coordinates[1] - to_depot.coordinates[1])
            distance_km = np.sqrt(lat_diff**2 + lon_diff**2) * 111.0  # Approximate
        else:
            distance_km = 20.0  # Default
        
        features.append(distance_km / 100.0)  # Normalize
        
        # Demand difference
        demand_from = predicted_demand.get(from_depot.depot_id, 0)
        demand_to = predicted_demand.get(to_depot.depot_id, 0)
        demand_diff = demand_to - demand_from
        features.append(demand_diff / 20.0)  # Normalize
        
        # Risk at current depot
        if risk_prediction:
            features.append(risk_prediction.get("risk_24h", 0.1))
        else:
            features.append(0.1)
        
        # Depot capacities
        features.append(from_depot.service_bay_capacity / 20.0)
        features.append(to_depot.service_bay_capacity / 20.0)
        
        # Train features
        features.append(train_features.sensor_health_score)
        features.append(1.0 if train_features.branding_flag else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    async def decide_transfer(self, train_features: FleetFeatures,
                             from_depot: DepotConfig, to_depot: DepotConfig,
                             predicted_demand: Dict[str, int],
                             risk_prediction: Optional[Dict[str, Any]] = None,
                             threshold: float = 0.5) -> Dict[str, Any]:
        """
        Decide whether to transfer train between depots
        
        Returns:
        - should_transfer: bool
        - transfer_probability: float
        - cost_estimate: Dict with dead_km, downtime, expected_benefit
        """
        if not self.model:
            await self.load_model()
        
        # Extract features
        features = self.extract_features(train_features, from_depot, to_depot, 
                                         predicted_demand, risk_prediction)
        
        # Predict transfer probability
        if self.model:
            feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            with torch.no_grad():
                transfer_prob = float(self.model(feature_tensor).item())
        else:
            transfer_prob = self._heuristic_transfer(features)
        
        # Calculate costs
        if from_depot.coordinates and to_depot.coordinates:
            lat_diff = abs(from_depot.coordinates[0] - to_depot.coordinates[0])
            lon_diff = abs(from_depot.coordinates[1] - to_depot.coordinates[1])
            dead_km = np.sqrt(lat_diff**2 + lon_diff**2) * 111.0
        else:
            dead_km = 20.0
        
        downtime_hours = 2.0  # Estimated transfer downtime
        expected_benefit = (predicted_demand.get(to_depot.depot_id, 0) - 
                          predicted_demand.get(from_depot.depot_id, 0)) * 0.1
        
        cost_estimate = {
            "dead_km": dead_km,
            "downtime_hours": downtime_hours,
            "expected_benefit": expected_benefit,
            "net_cost": dead_km * 0.5 + downtime_hours * 10.0 - expected_benefit,
        }
        
        should_transfer = transfer_prob >= threshold
        
        return {
            "should_transfer": should_transfer,
            "transfer_probability": transfer_prob,
            "cost_estimate": cost_estimate,
            "from_depot": from_depot.depot_id,
            "to_depot": to_depot.depot_id,
        }
    
    def _heuristic_transfer(self, features: np.ndarray) -> float:
        """Fallback heuristic"""
        # Higher demand difference = higher transfer probability
        demand_diff = features[1] * 20.0  # Denormalize
        if demand_diff > 3:
            return 0.7
        elif demand_diff > 1:
            return 0.4
        else:
            return 0.1
    
    async def train(self, training_data: List[Dict[str, Any]], epochs: int = 50,
                   batch_size: int = 32, learning_rate: float = 1e-3) -> Dict[str, Any]:
        """Train transfer decision model"""
        if not training_data or len(training_data) < 50:
            return {"status": "insufficient_data"}
        
        # Prepare data
        feature_vectors = []
        labels = []
        
        for sample in training_data:
            features = sample.get("features")
            if features is not None:
                feature_vectors.append(features)
                labels.append(float(sample.get("should_transfer", 0)))
        
        if not feature_vectors:
            return {"status": "no_valid_data"}
        
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(len(feature_vectors[0]))]
        
        X = np.array(feature_vectors, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        
        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        class SimpleDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.FloatTensor(X)
                self.y = torch.FloatTensor(y)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        
        train_dataset = SimpleDataset(X_train, y_train)
        val_dataset = SimpleDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_dim = len(self.feature_names)
        self.model = TransferDecisionModel(input_dim=input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training
        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for features_batch, labels_batch in train_loader:
                features_batch = features_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(features_batch).squeeze()
                loss = criterion(pred, labels_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features_batch, labels_batch in val_loader:
                    features_batch = features_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    pred = self.model(features_batch).squeeze()
                    loss = criterion(pred, labels_batch)
                    val_loss += loss.item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss/len(val_loader):.4f}")
        
        # Save
        await self._save_model()
        
        return {
            "status": "ok",
            "best_val_loss": float(best_val_loss),
        }
    
    async def _save_model(self):
        """Save trained model"""
        try:
            import io
            import hashlib
            
            example_input = torch.randn(1, len(self.feature_names)).to(self.device)
            self.model.eval()
            traced_model = torch.jit.trace(self.model, example_input)
            
            buf = io.BytesIO()
            torch.jit.save(traced_model, buf)
            buf.seek(0)
            model_bytes = buf.getvalue()
            
            meta = {
                "version": hashlib.sha1(model_bytes).hexdigest()[:12],
                "created_at": datetime.now().isoformat(),
                "feature_names": self.feature_names,
            }
            
            collection = await cloud_db_manager.get_collection("transfer_decision_models")
            await collection.insert_one({
                "meta": meta,
                "blob": model_bytes,
            })
            
            logger.info(f"Saved transfer decision model version {meta['version']}")
            
        except Exception as e:
            logger.error(f"Error saving transfer decision model: {e}")

