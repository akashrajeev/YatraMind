# backend/app/ml/multi_depot/rl_stabling_agent.py
"""
Reinforcement Learning - Multi-Depot Stabling Agent (Core)
Uses PPO or A2C to learn optimal stabling assignments
State: vectorized view of all depots
Action: assign train → {depot_id, location_type, bay_slot}
Reward: weighted sum (service_coverage, -failure_risk, -dead_km, -shunting_time)
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging
from datetime import datetime

from app.ml.multi_depot.config import DepotConfig, FleetFeatures, MultiDepotState
from app.utils.cloud_database import cloud_db_manager

logger = logging.getLogger(__name__)


class StablingPolicyNetwork(nn.Module):
    """Policy network for stabling allocation (PPO/A2C compatible)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
        # Value head (state value estimate)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, state):
        features = self.feature_extractor(state)
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        return action_logits, value
    
    def get_action_and_value(self, state, action=None):
        """Get action and value for PPO"""
        action_logits, value = self.forward(state)
        probs = torch.distributions.Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        action_log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return action, action_log_prob, entropy, value


class RLStablingAgent:
    """RL agent for multi-depot stabling allocation"""
    
    def __init__(self, depot_configs: List[DepotConfig], algorithm: str = "PPO"):
        """
        algorithm: "PPO" or "A2C"
        """
        self.depot_configs = depot_configs
        self.algorithm = algorithm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate action space
        # Action: (depot_idx, location_type, bay_id)
        # Flattened: depot_idx * (max_locations) + location_type * max_bays + bay_id
        self.max_bays = max(depot_config.total_bays for depot_config in depot_configs) if depot_configs else 12
        self.max_locations = 3  # bay, terminal, yard
        self.action_dim = len(depot_configs) * self.max_locations * self.max_bays
        
        # State dimension (will be calculated from MultiDepotState)
        self.state_dim = None  # Set during first state conversion
        
        # Policy network
        self.policy_net: Optional[StablingPolicyNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Training buffers
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_dones = []
        
        # Experience replay (for A2C)
        self.replay_buffer = deque(maxlen=10000)
        
    async def load_model(self) -> bool:
        """Load trained policy"""
        try:
            collection = await cloud_db_manager.get_collection("rl_stabling_policies")
            doc = await collection.find_one(sort=[("meta.created_at", -1)])
            
            if not doc:
                logger.warning("No RL stabling policy found, using random policy")
                return False
            
            import io
            blob = doc.get("blob")
            if isinstance(blob, bytes):
                buf = io.BytesIO(blob)
            else:
                buf = io.BytesIO(bytes(blob))
            
            state_dict = torch.load(buf, map_location=self.device)
            
            meta = doc.get("meta", {})
            self.state_dim = meta.get("state_dim")
            self.action_dim = meta.get("action_dim")
            
            # Initialize network if not already
            if self.policy_net is None and self.state_dim:
                self.policy_net = StablingPolicyNetwork(self.state_dim, self.action_dim).to(self.device)
                self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
            
            if self.policy_net:
                self.policy_net.load_state_dict(state_dict)
                self.policy_net.eval()
            
            logger.info(f"Loaded RL stabling policy version {meta.get('version', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading RL stabling policy: {e}")
            return False
    
    def select_action(self, state: MultiDepotState, training: bool = False) -> Dict[str, Any]:
        """
        Select action using policy network
        
        Returns:
        - action_dict with depot_id, location_type, bay_id
        """
        if self.policy_net is None:
            # Initialize network on first call
            state_vec = state.to_vector(self.depot_configs, [])
            self.state_dim = len(state_vec)
            self.policy_net = StablingPolicyNetwork(self.state_dim, self.action_dim).to(self.device)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        
        # Convert state to vector
        # Note: Need fleet_features for full state, simplified here
        state_vec = state.to_vector(self.depot_configs, [])
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        
        if training:
            # Training: sample from policy
            action, log_prob, entropy, value = self.policy_net.get_action_and_value(state_tensor)
            action_idx = action.item()
            
            # Store for training
            self.episode_states.append(state_vec)
            self.episode_actions.append(action_idx)
            self.episode_log_probs.append(log_prob.item())
            self.episode_values.append(value.item())
        else:
            # Inference: use deterministic (greedy)
            with torch.no_grad():
                action_logits, value = self.policy_net(state_tensor)
                action_idx = action_logits.argmax().item()
        
        # Decode action
        action_dict = self._decode_action(action_idx)
        
        return action_dict
    
    def _decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Decode flattened action index to depot_id, location_type, bay_id"""
        # Action encoding: depot_idx * (max_locations * max_bays) + location_type * max_bays + bay_id
        depot_idx = action_idx // (self.max_locations * self.max_bays)
        remainder = action_idx % (self.max_locations * self.max_bays)
        location_type_idx = remainder // self.max_bays
        bay_id = (remainder % self.max_bays) + 1  # 1-indexed
        
        if depot_idx < len(self.depot_configs):
            depot_config = self.depot_configs[depot_idx]
            location_types = ["bay", "terminal", "yard"]
            location_type = location_types[location_type_idx] if location_type_idx < len(location_types) else "bay"
            
            return {
                "depot_id": depot_config.depot_id,
                "depot_name": depot_config.depot_name,
                "location_type": location_type,
                "bay_id": bay_id if location_type == "bay" else None,
            }
        else:
            # Fallback to first depot
            return {
                "depot_id": self.depot_configs[0].depot_id,
                "depot_name": self.depot_configs[0].depot_name,
                "location_type": "bay",
                "bay_id": 1,
            }
    
    def calculate_reward(self, state: MultiDepotState, action: Dict[str, Any],
                        next_state: Optional[MultiDepotState],
                        outcomes: Dict[str, Any]) -> float:
        """
        Calculate reward for action
        
        Reward = service_coverage - failure_risk_penalty - dead_km_cost - shunting_time_penalty
        """
        reward = 0.0
        
        # Service coverage (positive)
        service_coverage = outcomes.get("service_coverage", 0.0)
        reward += service_coverage * 10.0
        
        # Failure risk penalty (negative)
        failure_risk = outcomes.get("failure_risk", 0.1)
        reward -= failure_risk * 20.0
        
        # Dead km cost (negative)
        dead_km = outcomes.get("dead_km", 0.0)
        reward -= dead_km * 0.5  # 0.5 per km
        
        # Shunting time penalty (negative)
        shunting_time = outcomes.get("shunting_time_min", 0.0)
        reward -= shunting_time * 0.2  # 0.2 per minute
        
        # Infeasible assignment penalty (heavy)
        if outcomes.get("infeasible", False):
            reward -= 50.0
        
        # Bay overflow penalty
        if outcomes.get("bay_overflow", False):
            reward -= 30.0
        
        # Maintenance requirement violation
        if outcomes.get("maintenance_violation", False):
            reward -= 40.0
        
        return reward
    
    def store_transition(self, state: MultiDepotState, action: Dict[str, Any],
                       reward: float, next_state: Optional[MultiDepotState], done: bool):
        """Store transition for training"""
        self.episode_rewards.append(reward)
        self.episode_dones.append(done)
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool], next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        returns = []
        
        gae = 0
        next_value = next_value
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = next_value if not dones[step] else 0.0
            
            delta = rewards[step] + self.gamma * next_value - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]
        
        return advantages, returns
    
    def train_step(self) -> float:
        """Perform one training step (PPO)"""
        if len(self.episode_states) < 2:
            return 0.0
        
        # Compute advantages and returns
        next_value = 0.0  # Terminal state value
        advantages, returns = self.compute_gae(
            self.episode_rewards,
            self.episode_values,
            self.episode_dones,
            next_value
        )
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.episode_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Forward pass
        action_logits, values = self.policy_net(states)
        probs = torch.distributions.Categorical(logits=action_logits)
        new_log_probs = probs.log_prob(actions)
        entropy = probs.entropy().mean()
        
        # PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns_tensor)
        
        # Entropy bonus
        entropy_loss = -entropy
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear episode buffer
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_dones = []
        
        return float(total_loss.item())
    
    async def save_model(self):
        """Save trained policy"""
        try:
            import io
            import hashlib
            
            if self.policy_net is None:
                return
            
            # Serialize state dict
            buf = io.BytesIO()
            torch.save(self.policy_net.state_dict(), buf)
            buf.seek(0)
            model_bytes = buf.getvalue()
            
            # Create metadata
            meta = {
                "version": hashlib.sha1(model_bytes).hexdigest()[:12],
                "created_at": datetime.now().isoformat(),
                "algorithm": self.algorithm,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "num_depots": len(self.depot_configs),
            }
            
            # Save to database
            collection = await cloud_db_manager.get_collection("rl_stabling_policies")
            await collection.insert_one({
                "meta": meta,
                "blob": model_bytes,
            })
            
            logger.info(f"Saved RL stabling policy version {meta['version']}")
            
        except Exception as e:
            logger.error(f"Error saving RL stabling policy: {e}")


