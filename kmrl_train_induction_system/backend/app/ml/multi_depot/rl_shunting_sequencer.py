# backend/app/ml/multi_depot/rl_shunting_sequencer.py
"""
RL Shunting Sequencer Agent
Per-depot agent that sequences moves to minimize total shunting time
"""
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging
from datetime import datetime

from app.utils.cloud_database import cloud_db_manager

logger = logging.getLogger(__name__)


class ShuntingSequencerPolicy(nn.Module):
    """Policy network for shunting sequencing"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, state):
        features = self.network[:-1](state)  # All but last layer
        action_logits = self.network[-1](features)
        value = self.value_head(features)
        return action_logits, value


class RLShuntingSequencer:
    """RL agent for shunting sequence optimization"""
    
    def __init__(self, depot_id: str, max_operations: int = 20):
        self.depot_id = depot_id
        self.max_operations = max_operations
        self.action_dim = max_operations  # Actions: operation ordering
        self.state_dim = 50  # Fixed state dimension
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = ShuntingSequencerPolicy(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        
        # Training buffers
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        
        self.gamma = 0.95
    
    async def load_model(self) -> bool:
        """Load trained policy"""
        try:
            collection = await cloud_db_manager.get_collection("rl_shunting_policies")
            doc = await collection.find_one(
                {"meta.depot_id": self.depot_id},
                sort=[("meta.created_at", -1)]
            )
            
            if not doc:
                return False
            
            import io
            blob = doc.get("blob")
            if isinstance(blob, bytes):
                buf = io.BytesIO(blob)
            else:
                buf = io.BytesIO(bytes(blob))
            
            state_dict = torch.load(buf, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.policy_net.eval()
            
            logger.info(f"Loaded shunting sequencer for {self.depot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading shunting sequencer: {e}")
            return False
    
    def state_to_vector(self, operations: List[Dict[str, Any]], 
                       remaining_window_min: int = 120) -> np.ndarray:
        """Convert shunting state to feature vector"""
        if not operations:
            return np.zeros(self.state_dim)
        
        # Aggregate features
        total_ops = len(operations)
        total_time = sum(op.get("estimated_time_min", 10) for op in operations)
        avg_distance = np.mean([op.get("distance_m", 0) for op in operations])
        high_complexity = sum(1 for op in operations if op.get("complexity") == "HIGH")
        
        features = np.array([
            total_ops / 20.0,
            total_time / 120.0,
            avg_distance / 1000.0,
            high_complexity / 10.0,
            remaining_window_min / 120.0,
            max(0.0, (total_time - remaining_window_min) / 120.0),  # Overrun
        ])
        
        # Pad to fixed size
        if len(features) < self.state_dim:
            features = np.pad(features, (0, self.state_dim - len(features)))
        else:
            features = features[:self.state_dim]
        
        return features.astype(np.float32)
    
    def optimize_sequence(self, operations: List[Dict[str, Any]],
                         available_window_min: int = 120) -> List[Dict[str, Any]]:
        """Optimize shunting sequence using RL policy"""
        if not operations:
            return []
        
        # Convert to state
        state_vec = self.state_to_vector(operations, available_window_min)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        
        # Get action (operation ordering)
        with torch.no_grad():
            action_logits, _ = self.policy_net(state_tensor)
            # Select top operations
            _, top_indices = action_logits.topk(min(len(operations), self.max_operations))
            top_indices = top_indices.squeeze().cpu().numpy()
        
        # Reorder operations
        optimized_ops = []
        for idx in top_indices:
            if idx < len(operations):
                op = operations[int(idx)].copy()
                op["sequence"] = len(optimized_ops) + 1
                optimized_ops.append(op)
        
        # Calculate cumulative time
        cumulative_time = 0.0
        for op in optimized_ops:
            cumulative_time += op.get("estimated_time_min", 10)
            op["cumulative_time_min"] = cumulative_time
            op["feasible"] = cumulative_time <= available_window_min
        
        return optimized_ops
    
    def calculate_reward(self, schedule: List[Dict[str, Any]], 
                        available_window_min: int = 120) -> float:
        """Calculate reward for schedule"""
        if not schedule:
            return -100.0
        
        total_time = sum(op.get("estimated_time_min", 10) for op in schedule)
        
        # Time penalty
        time_penalty = total_time / available_window_min
        
        # Overrun penalty (severe)
        overrun = max(0.0, total_time - available_window_min)
        overrun_penalty = overrun / available_window_min * 10.0
        
        # Reward (negative of penalties)
        reward = -(time_penalty + overrun_penalty)
        
        return reward
    
    def train_step(self) -> float:
        """Perform one training step"""
        if len(self.episode_states) < 2:
            return 0.0
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Normalize returns
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Forward pass
        action_logits, values = self.policy_net(states)
        probs = torch.distributions.Categorical(logits=action_logits)
        log_probs = probs.log_prob(actions)
        
        # Loss
        policy_loss = -(log_probs * returns_tensor).mean()
        value_loss = nn.MSELoss()(values.squeeze(), returns_tensor)
        total_loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        
        return float(total_loss.item())
    
    async def save_model(self):
        """Save trained policy"""
        try:
            import io
            import hashlib
            
            buf = io.BytesIO()
            torch.save(self.policy_net.state_dict(), buf)
            buf.seek(0)
            model_bytes = buf.getvalue()
            
            meta = {
                "version": hashlib.sha1(model_bytes).hexdigest()[:12],
                "created_at": datetime.now().isoformat(),
                "depot_id": self.depot_id,
            }
            
            collection = await cloud_db_manager.get_collection("rl_shunting_policies")
            await collection.insert_one({
                "meta": meta,
                "blob": model_bytes,
            })
            
            logger.info(f"Saved shunting sequencer for {self.depot_id}")
            
        except Exception as e:
            logger.error(f"Error saving shunting sequencer: {e}")


