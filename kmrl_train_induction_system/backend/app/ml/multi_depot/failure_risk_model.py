# backend/app/ml/multi_depot/failure_risk_model.py
"""
Failure Risk Model (LSTM/Transformer time-series)
Predicts P(failure in 24h), P(failure in 72h), component-level risk
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime, timedelta

from app.ml.multi_depot.config import FleetFeatures
from app.utils.cloud_database import cloud_db_manager

logger = logging.getLogger(__name__)


class FailureRiskDataset(Dataset):
    """Time-series dataset for failure risk prediction"""
    
    def __init__(self, sequences: np.ndarray, labels_24h: np.ndarray, labels_72h: np.ndarray, 
                 component_labels: Optional[np.ndarray] = None):
        """
        sequences: (N, seq_len, features) - time-series features
        labels_24h: (N,) - binary labels for 24h failure
        labels_72h: (N,) - binary labels for 72h failure
        component_labels: (N, num_components) - component-level failure labels
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels_24h = torch.FloatTensor(labels_24h)
        self.labels_72h = torch.FloatTensor(labels_72h)
        if component_labels is not None:
            self.component_labels = torch.FloatTensor(component_labels)
        else:
            self.component_labels = None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.component_labels is not None:
            return self.sequences[idx], self.labels_24h[idx], self.labels_72h[idx], self.component_labels[idx]
        return self.sequences[idx], self.labels_24h[idx], self.labels_72h[idx]


class FailureRiskLSTM(nn.Module):
    """LSTM-based failure risk predictor"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, num_components: int = 5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism (optional, for better interpretability)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Output heads
        self.head_24h = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self.head_72h = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Component-level risk head
        self.head_components = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_components),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use attention to focus on important time steps
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last time step
        last_output = attn_out[:, -1, :]  # (batch, hidden_dim)
        
        # Predictions
        risk_24h = self.head_24h(last_output)
        risk_72h = self.head_72h(last_output)
        component_risks = self.head_components(last_output)
        
        return risk_24h, risk_72h, component_risks


class FailureRiskTransformer(nn.Module):
    """Transformer-based failure risk predictor (alternative to LSTM)"""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, dropout: float = 0.1, num_components: int = 5):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1000, d_model))  # Max seq_len 1000
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.head_24h = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        self.head_72h = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        self.head_components = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_components),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        seq_len = x.size(1)
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder[:seq_len, :].unsqueeze(0)
        
        # Transformer encoding
        encoded = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Use last time step
        last_output = encoded[:, -1, :]  # (batch, d_model)
        
        # Predictions
        risk_24h = self.head_24h(last_output)
        risk_72h = self.head_72h(last_output)
        component_risks = self.head_components(last_output)
        
        return risk_24h, risk_72h, component_risks


class FailureRiskPredictor:
    """Failure risk prediction service"""
    
    def __init__(self, model_type: str = "LSTM", sequence_length: int = 30):
        """
        model_type: "LSTM" or "Transformer"
        sequence_length: Number of days of history to use
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.model: Optional[nn.Module] = None
        self.feature_names: List[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_components = 5  # bogie, brake, hvac, electrical, signalling
    
    async def load_model(self) -> bool:
        """Load trained model"""
        try:
            collection = await cloud_db_manager.get_collection("failure_risk_models")
            doc = await collection.find_one(sort=[("meta.created_at", -1)])
            
            if not doc:
                logger.warning("No failure risk model found, using fallback")
                return False
            
            import io
            blob = doc.get("blob")
            if isinstance(blob, bytes):
                buf = io.BytesIO(blob)
            else:
                buf = io.BytesIO(bytes(blob))
            
            meta = doc.get("meta", {})
            self.feature_names = meta.get("feature_names", [])
            self.sequence_length = meta.get("sequence_length", 30)
            self.model_type = meta.get("model_type", "LSTM")
            
            self.model = torch.jit.load(buf, map_location=self.device)
            self.model.eval()
            
            logger.info(f"Loaded failure risk model version {meta.get('version', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading failure risk model: {e}")
            return False
    
    def extract_timeseries_features(self, fleet_features: FleetFeatures, 
                                   historical_data: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """Extract time-series features from fleet features and historical data"""
        # Build sequence of daily features
        sequence = []
        
        # Use historical data if available, otherwise use sensor timeseries
        if historical_data:
            for day_data in historical_data[-self.sequence_length:]:
                features = [
                    day_data.get("mileage", 0) / 100000.0,
                    float(day_data.get("critical_cards", 0)) / 10.0,
                    float(day_data.get("open_cards", 0)) / 20.0,
                    day_data.get("sensor_health", 0.85),
                    day_data.get("temperature_avg", 25.0) / 50.0,
                    day_data.get("vibration_avg", 0.0) / 10.0,
                    day_data.get("anomaly_count", 0) / 10.0,
                ]
                sequence.append(features)
        elif fleet_features.sensor_timeseries:
            # Group timeseries by day
            daily_data = {}
            for sensor_data in fleet_features.sensor_timeseries:
                if isinstance(sensor_data, dict):
                    date_str = sensor_data.get("date", datetime.now().date().isoformat())
                    if date_str not in daily_data:
                        daily_data[date_str] = {
                            "temperatures": [],
                            "vibrations": [],
                            "anomalies": [],
                        }
                    daily_data[date_str]["temperatures"].append(sensor_data.get("temperature", 25.0))
                    daily_data[date_str]["vibrations"].append(sensor_data.get("vibration", 0.0))
                    if sensor_data.get("anomaly", False):
                        daily_data[date_str]["anomalies"].append(1)
            
            # Convert to sequence
            for date_str in sorted(daily_data.keys())[-self.sequence_length:]:
                day_data = daily_data[date_str]
                features = [
                    fleet_features.current_mileage / 100000.0,
                    float(fleet_features.job_cards.get("critical_cards", 0)) / 10.0,
                    float(fleet_features.job_cards.get("open_cards", 0)) / 20.0,
                    fleet_features.sensor_health_score,
                    np.mean(day_data["temperatures"]) / 50.0 if day_data["temperatures"] else 0.5,
                    np.mean(day_data["vibrations"]) / 10.0 if day_data["vibrations"] else 0.0,
                    len(day_data["anomalies"]) / 10.0,
                ]
                sequence.append(features)
        
        # Pad if insufficient history
        if len(sequence) < self.sequence_length:
            padding = [[0.0] * 7] * (self.sequence_length - len(sequence))
            sequence = padding + sequence
        
        # Take last sequence_length days
        sequence = sequence[-self.sequence_length:]
        
        return np.array(sequence, dtype=np.float32)
    
    async def predict(self, fleet_features: FleetFeatures,
                     historical_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Predict failure risk for a train
        
        Returns:
        - risk_24h: Probability of failure in 24 hours
        - risk_72h: Probability of failure in 72 hours
        - component_risks: Dict of component-level risks
        """
        if not self.model:
            await self.load_model()
        
        # Extract time-series features
        sequence = self.extract_timeseries_features(fleet_features, historical_data)
        
        if self.model:
            # Use model
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                risk_24h, risk_72h, component_risks = self.model(sequence_tensor)
                risk_24h_val = float(risk_24h.item())
                risk_72h_val = float(risk_72h.item())
                component_risks_val = component_risks.squeeze().cpu().numpy()
        else:
            # Fallback heuristic
            risk_24h_val, risk_72h_val, component_risks_val = self._heuristic_risk(fleet_features)
        
        # Component names
        component_names = ["bogie", "brake_system", "hvac", "electrical", "signalling"]
        component_risks_dict = {
            name: float(risk) for name, risk in zip(component_names, component_risks_val)
        }
        
        return {
            "train_id": fleet_features.train_id,
            "risk_24h": risk_24h_val,
            "risk_72h": risk_72h_val,
            "component_risks": component_risks_dict,
            "health_score": max(0.0, min(1.0, 1.0 - (risk_24h_val * 0.7 + risk_72h_val * 0.3))),
        }
    
    def _heuristic_risk(self, fleet_features: FleetFeatures) -> Tuple[float, float, np.ndarray]:
        """Fallback heuristic risk calculation"""
        risk_24h = 0.1
        risk_72h = 0.15
        
        # Increase risk based on features
        if fleet_features.job_cards.get("critical_cards", 0) > 0:
            risk_24h += 0.3
            risk_72h += 0.4
        
        if fleet_features.sensor_health_score < 0.7:
            risk_24h += 0.2
            risk_72h += 0.25
        
        # Component risks (simplified)
        component_risks = np.array([risk_24h * 0.8, risk_24h * 0.7, risk_24h * 0.6, 
                                    risk_24h * 0.5, risk_24h * 0.4])
        
        return min(1.0, risk_24h), min(1.0, risk_72h), component_risks
    
    async def train(self, training_data: List[Dict[str, Any]], epochs: int = 50,
                   batch_size: int = 32, learning_rate: float = 1e-3) -> Dict[str, Any]:
        """
        Train the failure risk model
        
        training_data: List of dicts with keys:
        - train_id, sequences (list of daily feature dicts),
        - label_24h (0/1), label_72h (0/1), component_labels (list)
        """
        if not training_data or len(training_data) < 100:
            return {"status": "insufficient_data", "rows": len(training_data)}
        
        # Prepare sequences
        sequences_list = []
        labels_24h = []
        labels_72h = []
        component_labels_list = []
        
        for sample in training_data:
            sequences = sample.get("sequences", [])
            if len(sequences) < self.sequence_length:
                continue
            
            # Convert to matrix
            sequence_matrix = []
            for day_data in sequences[-self.sequence_length:]:
                feature_vector = [
                    day_data.get("mileage", 0) / 100000.0,
                    float(day_data.get("critical_cards", 0)) / 10.0,
                    float(day_data.get("open_cards", 0)) / 20.0,
                    day_data.get("sensor_health", 0.85),
                    day_data.get("temperature_avg", 25.0) / 50.0,
                    day_data.get("vibration_avg", 0.0) / 10.0,
                    day_data.get("anomaly_count", 0) / 10.0,
                ]
                sequence_matrix.append(feature_vector)
            
            sequences_list.append(sequence_matrix)
            labels_24h.append(float(sample.get("label_24h", 0)))
            labels_72h.append(float(sample.get("label_72h", 0)))
            
            # Component labels
            comp_labels = sample.get("component_labels", [0.0] * self.num_components)
            component_labels_list.append(comp_labels[:self.num_components])
        
        if not sequences_list:
            return {"status": "insufficient_sequences"}
        
        # Convert to numpy
        X = np.array(sequences_list, dtype=np.float32)
        y_24h = np.array(labels_24h, dtype=np.float32)
        y_72h = np.array(labels_72h, dtype=np.float32)
        y_components = np.array(component_labels_list, dtype=np.float32)
        
        # Split train/val
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_24h_train, y_24h_val = y_24h[:split_idx], y_24h[split_idx:]
        y_72h_train, y_72h_val = y_72h[:split_idx], y_72h[split_idx:]
        y_comp_train, y_comp_val = y_components[:split_idx], y_components[split_idx:]
        
        # Create datasets
        train_dataset = FailureRiskDataset(X_train, y_24h_train, y_72h_train, y_comp_train)
        val_dataset = FailureRiskDataset(X_val, y_24h_val, y_72h_val, y_comp_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_dim = 7  # Number of features per time step
        if self.model_type == "Transformer":
            self.model = FailureRiskTransformer(input_dim=input_dim, num_components=self.num_components).to(self.device)
        else:
            self.model = FailureRiskLSTM(input_dim=input_dim, num_components=self.num_components).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                sequences, label_24h, label_72h, label_comp = batch
                sequences = sequences.to(self.device)
                label_24h = label_24h.to(self.device)
                label_72h = label_72h.to(self.device)
                label_comp = label_comp.to(self.device)
                
                optimizer.zero_grad()
                pred_24h, pred_72h, pred_comp = self.model(sequences)
                loss = (criterion(pred_24h.squeeze(), label_24h) +
                       criterion(pred_72h.squeeze(), label_72h) +
                       criterion(pred_comp, label_comp).mean())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    sequences, label_24h, label_72h, label_comp = batch
                    sequences = sequences.to(self.device)
                    label_24h = label_24h.to(self.device)
                    label_72h = label_72h.to(self.device)
                    label_comp = label_comp.to(self.device)
                    
                    pred_24h, pred_72h, pred_comp = self.model(sequences)
                    loss = (criterion(pred_24h.squeeze(), label_24h) +
                           criterion(pred_72h.squeeze(), label_72h) +
                           criterion(pred_comp, label_comp).mean())
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
            "model_type": self.model_type,
        }
    
    async def _save_model(self):
        """Save trained model"""
        try:
            import io
            import hashlib
            
            # Convert to TorchScript
            example_input = torch.randn(1, self.sequence_length, 7).to(self.device)
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
                "sequence_length": self.sequence_length,
                "model_type": self.model_type,
                "num_components": self.num_components,
            }
            
            # Save to database
            collection = await cloud_db_manager.get_collection("failure_risk_models")
            await collection.insert_one({
                "meta": meta,
                "blob": model_bytes,
            })
            
            logger.info(f"Saved failure risk model version {meta['version']}")
            
        except Exception as e:
            logger.error(f"Error saving failure risk model: {e}")


