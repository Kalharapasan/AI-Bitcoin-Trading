"""
Reinforcement Learning for Trading
Deep reinforcement learning algorithms for automated trading strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union, Deque
import warnings
import random
from collections import deque, namedtuple
import gym
from gym import spaces
import pytorch_lightning as pl
from torch.optim import Adam, AdamW, RMSprop
from torch.distributions import Normal, Categorical
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============ Trading Environment ============
class TradingEnvironment(gym.Env):
    """Trading environment for reinforcement learning"""
    
    def __init__(self, data: np.ndarray, features: np.ndarray,
                 initial_balance: float = 10000.0, commission: float = 0.001,
                 max_position: float = 0.1, window_size: int = 60):
        super().__init__()
        
        self.data = data  # Price data (n_timesteps,)
        self.features = features  # Feature data (n_timesteps, n_features)
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_position = max_position
        self.window_size = window_size
        
        self.n_features = features.shape[1]
        self.current_step = window_size
        self.max_steps = len(data) - window_size - 1
        
        # Define action space
        # Action format: [position_type, position_size]
        # position_type: 0=hold, 1=long, 2=short
        # position_size: percentage of portfolio (0 to max_position)
        self.action_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([2, max_position]),
            dtype=np.float32
        )
        
        # Define observation space
        # Observation: [price_features, portfolio_state, market_features]
        obs_low = np.concatenate([
            np.min(features, axis=0),  # Price features
            np.array([0, -max_position, 0, 0]),  # Portfolio state
            np.zeros(self.n_features)  # Market features
        ])
        
        obs_high = np.concatenate([
            np.max(features, axis=0),  # Price features
            np.array([np.inf, max_position, np.inf, 1]),  # Portfolio state
            np.ones(self.n_features)  # Market features
        ])
        
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.balance = self.initial_balance
        self.position = 0.0  # Current position size
        self.position_type = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0.0
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.trades = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Get price features for current window
        price_features = self.features[
            self.current_step - self.window_size:self.current_step
        ].mean(axis=0)
        
        # Get portfolio state
        portfolio_state = np.array([
            self.balance,
            self.position * self.position_type,  # Signed position
            self.entry_price if self.position_type != 0 else 0.0,
            len(self.trades) / 100  # Normalized trade count
        ])
        
        # Get market features (technical indicators for current price)
        market_features = self.features[self.current_step]
        
        # Combine all features
        observation = np.concatenate([
            price_features,
            portfolio_state,
            market_features
        ])
        
        return observation.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        
        # Parse action
        position_type = int(action[0])  # 0: hold, 1: long, 2: short
        position_size = float(action[1])
        
        # Get current price
        current_price = self.data[self.current_step]
        
        # Execute trade if needed
        reward, trade_executed = self._execute_trade(
            position_type, position_size, current_price
        )
        
        # Update portfolio value
        self._update_portfolio_value(current_price)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Get next observation
        observation = self._get_observation()
        
        # Calculate additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'position_type': self.position_type,
            'balance': self.balance,
            'current_price': current_price,
            'trade_executed': trade_executed,
            'step': self.current_step
        }
        
        return observation, reward, done, info
    
    def _execute_trade(self, target_position_type: int, target_position_size: float,
                      current_price: float) -> Tuple[float, bool]:
        """Execute trade based on action"""
        trade_executed = False
        
        # Determine target position
        if target_position_type == 0:  # Hold
            target_position = 0.0
            target_position_type = 0
        elif target_position_type == 1:  # Long
            target_position = target_position_size
            target_position_type = 1
        elif target_position_type == 2:  # Short
            target_position = target_position_size
            target_position_type = -1  # Short positions are negative
        else:
            raise ValueError(f"Invalid position type: {target_position_type}")
        
        # Calculate position change
        if self.position_type == 0:  # No current position
            position_change = target_position
        elif self.position_type == target_position_type:  # Same direction
            position_change = target_position - self.position
        else:  # Different direction (reversal)
            # First close current position
            position_change = -self.position
            target_position_type = 0
            target_position = 0.0
        
        # Execute trade if position change is significant
        if abs(position_change) > 0.001:
            trade_executed = True
            
            # Calculate trade value
            trade_value = abs(position_change) * self.portfolio_value
            
            # Apply commission
            commission_cost = trade_value * self.commission
            self.balance -= commission_cost
            
            # Update position
            self.position += position_change
            self.position_type = target_position_type
            
            if self.position_type != 0:
                self.entry_price = current_price
            
            # Record trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'position_change': position_change,
                'position_type': target_position_type,
                'commission': commission_cost
            })
        
        # Calculate reward
        reward = self._calculate_reward(current_price, trade_executed)
        
        return reward, trade_executed
    
    def _update_portfolio_value(self, current_price: float):
        """Update portfolio value based on current position"""
        position_value = 0.0
        
        if self.position_type == 1:  # Long position
            if self.entry_price > 0:
                profit_ratio = (current_price - self.entry_price) / self.entry_price
                position_value = self.position * self.portfolio_value * (1 + profit_ratio)
        elif self.position_type == -1:  # Short position
            if self.entry_price > 0:
                profit_ratio = (self.entry_price - current_price) / self.entry_price
                position_value = self.position * self.portfolio_value * (1 + profit_ratio)
        
        self.portfolio_value = self.balance + position_value
    
    def _calculate_reward(self, current_price: float, trade_executed: bool) -> float:
        """Calculate reward for current step"""
        
        # Base reward: portfolio return
        if self.current_step == self.window_size:
            previous_value = self.initial_balance
        else:
            previous_price = self.data[self.current_step - 1]
            self._update_portfolio_value(previous_price)
            previous_value = self.portfolio_value
        
        self._update_portfolio_value(current_price)
        current_value = self.portfolio_value
        
        return_reward = (current_value - previous_value) / previous_value
        
        # Risk penalty
        risk_penalty = 0.0
        if self.position_type != 0:
            # Penalize large positions
            risk_penalty = -0.1 * abs(self.position) / self.max_position
            
            # Penalize holding positions for too long
            if len(self.trades) > 0 and not trade_executed:
                last_trade = self.trades[-1]
                steps_since_trade = self.current_step - last_trade['step']
                risk_penalty -= 0.001 * steps_since_trade
        
        # Trade penalty (to avoid overtrading)
        trade_penalty = -0.001 if trade_executed else 0.0
        
        # Sharpe-like reward (reward per unit risk)
        total_reward = return_reward + risk_penalty + trade_penalty
        
        return total_reward
    
    def render(self, mode: str = 'human'):
        """Render current state"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Position: {self.position:.4f} ({'Long' if self.position_type == 1 else 'Short' if self.position_type == -1 else 'None'})")
            print(f"Entry Price: ${self.entry_price:.2f}")
            print(f"Current Price: ${self.data[self.current_step]:.2f}")
            print(f"Number of Trades: {len(self.trades)}")
            print("-" * 50)
    
    def get_trading_statistics(self) -> Dict[str, float]:
        """Get trading statistics"""
        if len(self.trades) == 0:
            return {}
        
        returns = []
        trade_pnls = []
        
        for i in range(1, len(self.trades)):
            trade_start = self.trades[i-1]
            trade_end = self.trades[i]
            
            if trade_start['position_type'] == 1:  # Long
                pnl = (trade_end['price'] - trade_start['price']) / trade_start['price']
            elif trade_start['position_type'] == -1:  # Short
                pnl = (trade_start['price'] - trade_end['price']) / trade_start['price']
            else:
                continue
            
            trade_pnls.append(pnl - self.commission * 2)  # Entry and exit commissions
        
        if len(trade_pnls) == 0:
            return {}
        
        trade_pnls = np.array(trade_pnls)
        
        stats = {
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'sharpe_ratio': np.mean(trade_pnls) / (np.std(trade_pnls) + 1e-8) * np.sqrt(252),
            'win_rate': np.mean(trade_pnls > 0),
            'profit_factor': np.sum(trade_pnls[trade_pnls > 0]) / abs(np.sum(trade_pnls[trade_pnls < 0])) if np.any(trade_pnls < 0) else np.inf,
            'avg_win': np.mean(trade_pnls[trade_pnls > 0]) if np.any(trade_pnls > 0) else 0,
            'avg_loss': np.mean(trade_pnls[trade_pnls < 0]) if np.any(trade_pnls < 0) else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'num_trades': len(self.trades)
        }
        
        return stats
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.trades) < 2:
            return 0.0
        
        # Simulate portfolio values
        portfolio_values = []
        for i in range(self.window_size, min(self.current_step, len(self.data))):
            price = self.data[i]
            self._update_portfolio_value(price)
            portfolio_values.append(self.portfolio_value)
        
        if len(portfolio_values) == 0:
            return 0.0
        
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        return max_drawdown

# ============ Replay Buffer ============
Transition = namedtuple('Transition', 
                        ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for RL"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: np.ndarray, 
             reward: float, next_state: np.ndarray, done: bool):
        """Add transition to buffer"""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch from buffer"""
        if len(self.buffer) < batch_size:
            return None
        
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.FloatTensor(np.array(batch.action))
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self) -> int:
        return len(self.buffer)

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay"""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: np.ndarray, 
             reward: float, next_state: np.ndarray, done: bool):
        """Add transition with max priority"""
        super().push(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with prioritization"""
        if len(self.buffer) < batch_size:
            return None
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Get transitions
        transitions = [self.buffer[idx] for idx in indices]
        batch = Transition(*zip(*transitions))
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        weights = torch.FloatTensor(weights).unsqueeze(1)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.FloatTensor(np.array(batch.action))
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority.item() + 1e-6
            self.max_priority = max(self.max_priority, priority.item())

# ============ Neural Network Architectures ============
class ActorNetwork(nn.Module):
    """Actor network for policy-based methods"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # Output heads for position type and size
        self.position_type_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 3),
            nn.Softmax(dim=-1)
        )
        
        self.position_size_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        features = self.network(state)
        
        position_type_probs = self.position_type_head(features)
        position_size = self.position_size_head(features)
        
        return position_type_probs, position_size
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action from policy"""
        position_type_probs, position_size = self.forward(state)
        
        if deterministic:
            position_type = torch.argmax(position_type_probs, dim=-1, keepdim=True)
        else:
            dist = Categorical(position_type_probs)
            position_type = dist.sample().unsqueeze(1)
        
        # Combine position type and size
        action = torch.cat([position_type.float(), position_size], dim=-1)
        
        return action
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of action"""
        position_type_probs, position_size = self.forward(state)
        
        position_type = action[:, 0].long()
        position_size_action = action[:, 1].unsqueeze(1)
        
        # Log prob for position type (categorical)
        dist = Categorical(position_type_probs)
        type_log_prob = dist.log_prob(position_type).unsqueeze(1)
        
        # Log prob for position size (continuous, using normal distribution)
        # We use the predicted position size as mean and fixed std
        size_dist = Normal(position_size, 0.1)
        size_log_prob = size_dist.log_prob(position_size_action)
        
        # Total log prob
        total_log_prob = type_log_prob + size_log_prob
        
        return total_log_prob

class CriticNetwork(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.state_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        self.action_network = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4),
            nn.ReLU()
        )
        
        self.combined_network = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        state_features = self.state_network(state)
        action_features = self.action_network(action)
        
        combined = torch.cat([state_features, action_features], dim=-1)
        value = self.combined_network(combined)
        
        return value

class DQNNetwork(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)

class AttentionActorNetwork(nn.Module):
    """Actor network with attention mechanism"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Attention layer for state features
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Positional encoding for sequence
        self.pos_encoder = nn.Parameter(torch.randn(1, 10, hidden_dim // 2) * 0.1)
        
        # Output heads
        self.position_type_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
        
        self.position_size_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention"""
        # Reshape state for attention (batch, sequence, features)
        batch_size = state.size(0)
        state_reshaped = state.view(batch_size, -1, state.size(-1) // 10)
        
        # Extract features
        features = self.feature_extractor(state_reshaped)
        
        # Add positional encoding
        features = features + self.pos_encoder[:, :features.size(1), :]
        
        # Apply attention
        attended, _ = self.attention(features, features, features)
        
        # Global pooling
        pooled = attended.mean(dim=1)
        
        # Get outputs
        position_type_probs = self.position_type_head(pooled)
        position_size = self.position_size_head(pooled)
        
        return position_type_probs, position_size

# ============ RL Algorithms ============
class DQNAgent:
    """Deep Q-Network agent"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks
        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = Adam(self.policy_net.parameters(), 
                             lr=config.get('learning_rate', 0.001))
        
        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=config.get('buffer_size', 100000)
        )
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.tau = config.get('tau', 0.005)  # For soft updates
        self.batch_size = config.get('batch_size', 64)
        
        self.steps_done = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        self.steps_done += 1
        
        if training and random.random() < self.epsilon:
            # Random action
            position_type = random.randint(0, 2)
            position_size = random.random() * self.config.get('max_position', 0.1)
            action = np.array([position_type, position_size], dtype=np.float32)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action_idx = torch.argmax(q_values).item()
                
                # Convert action index to continuous action
                position_type = action_idx // 10  # 0, 1, 2
                position_size = (action_idx % 10) / 10.0 * self.config.get('max_position', 0.1)
                action = np.array([position_type, position_size], dtype=np.float32)
        
        # Decay epsilon
        if training:
            self.epsilon = max(self.epsilon_end, 
                              self.epsilon * self.epsilon_decay)
        
        return action
    
    def train_step(self):
        """Perform one training step"""
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # Sample from replay buffer
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return 0.0
        
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, weights, indices = batch
        else:
            states, actions, rewards, next_states, dones = batch
            weights = torch.ones_like(rewards)
        
        # Convert actions to discrete indices
        action_indices = self._action_to_index(actions)
        
        # Current Q values
        q_values = self.policy_net(states)
        current_q = q_values.gather(1, action_indices.long())
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q = next_q_values.max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            with torch.no_grad():
                td_error = (target_q - current_q).abs().detach().cpu().numpy()
                self.buffer.update_priorities(indices, td_error)
        
        # Soft update target network
        self._soft_update_target_network()
        
        return loss.item()
    
    def _action_to_index(self, action: torch.Tensor) -> torch.Tensor:
        """Convert continuous action to discrete index"""
        position_type = action[:, 0].long()
        position_size = action[:, 1]
        
        # Discretize position size
        size_bin = torch.clamp((position_size * 10).long(), 0, 9)
        
        # Calculate index
        action_idx = position_type * 10 + size_bin
        
        return action_idx.unsqueeze(1)
    
    def _soft_update_target_network(self):
        """Soft update target network parameters"""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                            self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save agent"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load agent"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.config = checkpoint['config']

class PPOAgent:
    """Proximal Policy Optimization agent"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim, action_dim)
        
        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(),
                                   lr=config.get('actor_lr', 0.0003))
        self.critic_optimizer = Adam(self.critic.parameters(),
                                    lr=config.get('critic_lr', 0.001))
        
        # Replay buffer
        self.buffer = ReplayBuffer(capacity=config.get('buffer_size', 10000))
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.lambda_ = config.get('lambda', 0.95)  # For GAE
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        
        self.steps_done = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor.get_action(state_tensor, deterministic=not training)
            action_np = action.squeeze(0).numpy()
        
        self.steps_done += 1
        return action_np
    
    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition in buffer"""
        self.buffer.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train on collected data"""
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0, 0.0
        
        # Get all transitions from buffer
        transitions = list(self.buffer.buffer)
        states = torch.FloatTensor(np.array([t.state for t in transitions]))
        actions = torch.FloatTensor(np.array([t.action for t in transitions]))
        rewards = torch.FloatTensor([t.reward for t in transitions]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions]))
        dones = torch.FloatTensor([t.done for t in transitions]).unsqueeze(1)
        
        # Compute advantages
        with torch.no_grad():
            values = self.critic(states, actions)
            next_values = self.critic(next_states, actions)
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            
            # Compute GAE
            advantages = torch.zeros_like(deltas)
            advantage = 0
            for t in reversed(range(len(deltas))):
                advantage = deltas[t] + self.gamma * self.lambda_ * advantage * (1 - dones[t])
                advantages[t] = advantage
            
            returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        
        # Train for multiple epochs
        for _ in range(self.num_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = self.actor.get_log_prob(batch_states, batch_actions).detach()
                
                # Actor loss
                new_log_probs = self.actor.get_log_prob(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                                  1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy loss (encourage exploration)
                position_type_probs, _ = self.actor(batch_states)
                dist = Categorical(position_type_probs)
                entropy_loss = -dist.entropy().mean()
                
                # Critic loss
                batch_values = self.critic(batch_states, batch_actions)
                critic_loss = F.mse_loss(batch_values, batch_returns)
                
                # Total loss
                total_actor_loss = actor_loss + self.entropy_coef * entropy_loss
                total_critic_loss = critic_loss * self.value_coef
                
                # Update actor
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                total_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Clear buffer after training
        self.buffer.buffer.clear()
        
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_entropy_loss = np.mean(entropy_losses) if entropy_losses else 0.0
        
        return avg_actor_loss, avg_critic_loss, avg_entropy_loss
    
    def save(self, path: str):
        """Save agent"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'steps_done': self.steps_done,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load agent"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.config = checkpoint['config']

class SACAgent:
    """Soft Actor-Critic agent"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic1 = CriticNetwork(state_dim, action_dim)
        self.critic2 = CriticNetwork(state_dim, action_dim)
        self.target_critic1 = CriticNetwork(state_dim, action_dim)
        self.target_critic2 = CriticNetwork(state_dim, action_dim)
        
        # Copy parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(),
                                   lr=config.get('actor_lr', 0.0003))
        self.critic1_optimizer = Adam(self.critic1.parameters(),
                                     lr=config.get('critic_lr', 0.001))
        self.critic2_optimizer = Adam(self.critic2.parameters(),
                                     lr=config.get('critic_lr', 0.001))
        
        # Temperature parameter
        self.log_alpha = torch.tensor(np.log(config.get('alpha', 0.2)), 
                                     requires_grad=True)
        self.alpha_optimizer = Adam([self.log_alpha], 
                                   lr=config.get('alpha_lr', 0.0003))
        
        # Replay buffer
        self.buffer = ReplayBuffer(capacity=config.get('buffer_size', 100000))
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)  # For soft updates
        self.batch_size = config.get('batch_size', 256)
        self.target_entropy = -action_dim  # Heuristic
        
        self.steps_done = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            if training:
                action = self.actor.get_action(state_tensor, deterministic=False)
            else:
                action = self.actor.get_action(state_tensor, deterministic=True)
            
            action_np = action.squeeze(0).numpy()
        
        self.steps_done += 1
        return action_np
    
    def train_step(self):
        """Perform one training step"""
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0, 0.0
        
        # Sample batch
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return 0.0, 0.0, 0.0
        
        states, actions, rewards, next_states, dones = batch
        
        # Update critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        
        # Update actor
        actor_loss, entropy = self._update_actor(states)
        
        # Update temperature
        alpha_loss = self._update_temperature(entropy)
        
        # Soft update target networks
        self._soft_update_target_network()
        
        return critic_loss, actor_loss, alpha_loss.item()
    
    def _update_critic(self, states, actions, rewards, next_states, dones):
        """Update critic networks"""
        with torch.no_grad():
            # Sample next action from current policy
            next_position_type_probs, next_position_size = self.actor(next_states)
            
            # Sample next action
            next_position_type_dist = Categorical(next_position_type_probs)
            next_position_type = next_position_type_dist.sample().unsqueeze(1)
            next_action = torch.cat([next_position_type.float(), next_position_size], dim=-1)
            
            # Get log prob for next action
            next_log_prob = self.actor.get_log_prob(next_states, next_action)
            
            # Target Q values
            alpha = self.log_alpha.exp().detach()
            target_q1 = self.target_critic1(next_states, next_action)
            target_q2 = self.target_critic2(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_value = rewards + self.gamma * (1 - dones) * target_q
        
        # Current Q values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)
        critic_loss = critic1_loss + critic2_loss
        
        # Optimize critics
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, states):
        """Update actor network"""
        # Sample action from current policy
        position_type_probs, position_size = self.actor(states)
        
        # Sample action
        position_type_dist = Categorical(position_type_probs)
        position_type = position_type_dist.sample().unsqueeze(1)
        action = torch.cat([position_type.float(), position_size], dim=-1)
        
        # Get log prob
        log_prob = self.actor.get_log_prob(states, action)
        
        # Q values for sampled action
        q1 = self.critic1(states, action)
        q2 = self.critic2(states, action)
        q = torch.min(q1, q2)
        
        # Actor loss
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_prob - q).mean()
        
        # Entropy
        entropy = -log_prob.mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        return actor_loss.item(), entropy.item()
    
    def _update_temperature(self, entropy):
        """Update temperature parameter"""
        alpha_loss = -self.log_alpha * (entropy - self.target_entropy).detach()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return alpha_loss
    
    def _soft_update_target_network(self):
        """Soft update target networks"""
        for target_param, param in zip(self.target_critic1.parameters(), 
                                      self.critic1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        for target_param, param in zip(self.target_critic2.parameters(), 
                                      self.critic2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save agent"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'steps_done': self.steps_done,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load agent"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.steps_done = checkpoint['steps_done']
        self.config = checkpoint['config']

# ============ PyTorch Lightning Module ============
class RLTraderLightning(pl.LightningModule):
    """PyTorch Lightning module for RL trading"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Extract configuration
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.algorithm = config.get('algorithm', 'ppo')
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Create agent based on algorithm
        if self.algorithm == 'dqn':
            self.agent = DQNAgent(self.state_dim, self.action_dim, config)
        elif self.algorithm == 'ppo':
            self.agent = PPOAgent(self.state_dim, self.action_dim, config)
        elif self.algorithm == 'sac':
            self.agent = SACAgent(self.state_dim, self.action_dim, config)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Training metrics
        self.train_rewards = []
        self.train_losses = []
        self.episode_rewards = []
        
        # Environment (will be set later)
        self.env = None
    
    def set_environment(self, env: TradingEnvironment):
        """Set trading environment"""
        self.env = env
    
    def training_step(self, batch, batch_idx):
        """Training step for RL"""
        if self.env is None:
            raise ValueError("Environment not set. Call set_environment() first.")
        
        # Run one episode
        state = self.env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = self.agent.select_action(state, training=True)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            if isinstance(self.agent, PPOAgent):
                self.agent.store_transition(state, action, reward, next_state, done)
            elif isinstance(self.agent, (DQNAgent, SACAgent)):
                self.agent.buffer.push(state, action, reward, next_state, done)
            
            # Train agent
            if isinstance(self.agent, DQNAgent):
                loss = self.agent.train_step()
                if loss > 0:
                    self.train_losses.append(loss)
            elif isinstance(self.agent, SACAgent):
                critic_loss, actor_loss, alpha_loss = self.agent.train_step()
                if critic_loss > 0:
                    self.train_losses.append(critic_loss)
            
            state = next_state
            episode_reward += reward
        
        # Train PPO after episode
        if isinstance(self.agent, PPOAgent):
            actor_loss, critic_loss, entropy_loss = self.agent.train()
            if actor_loss > 0:
                self.train_losses.append(actor_loss)
        
        # Store episode reward
        self.episode_rewards.append(episode_reward)
        
        # Calculate average reward
        avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else episode_reward
        
        # Log metrics
        self.log('train_episode_reward', episode_reward, prog_bar=True)
        self.log('train_avg_reward', avg_reward, prog_bar=True)
        
        if self.train_losses:
            avg_loss = np.mean(self.train_losses[-100:]) if len(self.train_losses) >= 100 else self.train_losses[-1]
            self.log('train_loss', avg_loss)
        
        # Get trading statistics
        stats = self.env.get_trading_statistics()
        if stats:
            self.log('train_sharpe', stats.get('sharpe_ratio', 0))
            self.log('train_win_rate', stats.get('win_rate', 0))
            self.log('train_profit_factor', min(stats.get('profit_factor', 0), 10))  # Cap at 10
        
        return {'loss': episode_reward, 'avg_reward': avg_reward}
    
    def configure_optimizers(self):
        """Configure optimizers"""
        # RL agents have their own optimizers
        return None
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards[-50:])
            print(f"Epoch {self.current_epoch}: Average Reward = {avg_reward:.4f}")
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step"""
        if self.env is None:
            raise ValueError("Environment not set")
        
        state = self.env.reset()
        done = False
        predictions = []
        
        while not done:
            with torch.no_grad():
                action = self.agent.select_action(state, training=False)
                predictions.append(action)
                
                next_state, _, done, _ = self.env.step(action)
                state = next_state
        
        return np.array(predictions)
    
    def save_agent(self, path: str):
        """Save trained agent"""
        self.agent.save(path)
        print(f"Agent saved to {path}")
    
    def load_agent(self, path: str):
        """Load trained agent"""
        self.agent.load(path)
        print(f"Agent loaded from {path}")
    
    def visualize_training(self):
        """Visualize training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Episode rewards
        ax1 = axes[0, 0]
        if self.episode_rewards:
            ax1.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
            
            # Plot moving average
            window = min(50, len(self.episode_rewards))
            if window > 0:
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(self.episode_rewards)), moving_avg, 
                        'r-', linewidth=2, label=f'MA({window})')
            
            ax1.set_title('Training Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training losses
        ax2 = axes[0, 1]
        if self.train_losses:
            ax2.plot(self.train_losses, alpha=0.6, label='Training Loss')
            
            # Plot moving average
            window = min(50, len(self.train_losses))
            if window > 0:
                moving_avg = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(self.train_losses)), moving_avg, 
                        'r-', linewidth=2, label=f'MA({window})')
            
            ax2.set_title('Training Losses')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Action distribution
        ax3 = axes[1, 0]
        if hasattr(self.agent, 'buffer') and len(self.agent.buffer) > 0:
            actions = np.array([t.action for t in self.agent.buffer.buffer])
            
            if len(actions) > 0:
                position_types = actions[:, 0]
                position_sizes = actions[:, 1]
                
                # Position type distribution
                unique_types, counts = np.unique(position_types, return_counts=True)
                type_labels = ['Hold', 'Long', 'Short']
                type_colors = ['gray', 'green', 'red']
                
                bars = ax3.bar(unique_types, counts, color=[type_colors[int(t)] for t in unique_types])
                ax3.set_xticks(unique_types)
                ax3.set_xticklabels([type_labels[int(t)] for t in unique_types])
                ax3.set_title('Action Type Distribution')
                ax3.set_ylabel('Count')
                ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Position size distribution
        ax4 = axes[1, 1]
        if hasattr(self.agent, 'buffer') and len(self.agent.buffer) > 0:
            actions = np.array([t.action for t in self.agent.buffer.buffer])
            
            if len(actions) > 0:
                position_sizes = actions[:, 1]
                
                ax4.hist(position_sizes, bins=20, alpha=0.7, edgecolor='black')
                ax4.set_title('Position Size Distribution')
                ax4.set_xlabel('Position Size')
                ax4.set_ylabel('Count')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ============ Training Utilities ============
def train_rl_agent(env: TradingEnvironment, agent_config: Dict[str, Any],
                   n_episodes: int = 1000, eval_freq: int = 100,
                   save_path: Optional[str] = None) -> Dict[str, List]:
    """Train RL agent on environment"""
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent_type = agent_config.get('algorithm', 'ppo')
    
    if agent_type == 'dqn':
        agent = DQNAgent(state_dim, action_dim, agent_config)
    elif agent_type == 'ppo':
        agent = PPOAgent(state_dim, action_dim, agent_config)
    elif agent_type == 'sac':
        agent = SACAgent(state_dim, action_dim, agent_config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Training metrics
    episode_rewards = []
    episode_stats = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            if isinstance(agent, PPOAgent):
                agent.store_transition(state, action, reward, next_state, done)
            elif isinstance(agent, (DQNAgent, SACAgent)):
                agent.buffer.push(state, action, reward, next_state, done)
            
            # Train agent
            if isinstance(agent, DQNAgent):
                agent.train_step()
            elif isinstance(agent, SACAgent):
                agent.train_step()
            
            state = next_state
            episode_reward += reward
        
        # Train PPO after episode
        if isinstance(agent, PPOAgent):
            agent.train()
        
        # Store metrics
        episode_rewards.append(episode_reward)
        
        # Get trading statistics
        stats = env.get_trading_statistics()
        if stats:
            episode_stats.append(stats)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"Avg Reward: {avg_reward:.4f}")
            
            if stats:
                print(f"  Sharpe: {stats.get('sharpe_ratio', 0):.2f}, "
                      f"Win Rate: {stats.get('win_rate', 0):.2%}, "
                      f"Trades: {stats.get('num_trades', 0)}")
        
        # Save agent
        if save_path and (episode + 1) % eval_freq == 0:
            agent.save(f"{save_path}_{episode + 1}.pth")
    
    # Final save
    if save_path:
        agent.save(f"{save_path}_final.pth")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_stats': episode_stats,
        'agent': agent
    }

# ============ Example Usage ============
def example_usage():
    """Example usage of RL trading"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 5000
    n_features = 20
    
    # Create synthetic price data
    prices = np.cumprod(1 + np.random.randn(n_samples) * 0.01) * 100
    
    # Create synthetic features
    features = np.random.randn(n_samples, n_features)
    
    # Create environment
    env = TradingEnvironment(
        data=prices,
        features=features,
        initial_balance=10000.0,
        commission=0.001,
        max_position=0.1,
        window_size=60
    )
    
    # Test different agents
    agent_configs = {
        'dqn': {
            'algorithm': 'dqn',
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'buffer_size': 10000,
            'batch_size': 64,
            'tau': 0.005
        },
        'ppo': {
            'algorithm': 'ppo',
            'actor_lr': 0.0003,
            'critic_lr': 0.001,
            'gamma': 0.99,
            'lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'num_epochs': 10,
            'batch_size': 64,
            'buffer_size': 10000
        },
        'sac': {
            'algorithm': 'sac',
            'actor_lr': 0.0003,
            'critic_lr': 0.001,
            'alpha': 0.2,
            'alpha_lr': 0.0003,
            'gamma': 0.99,
            'tau': 0.005,
            'buffer_size': 100000,
            'batch_size': 256
        }
    }
    
    results = {}
    
    for agent_name, config in agent_configs.items():
        print(f"\nTesting {agent_name.upper()} agent:")
        print("-" * 30)
        
        try:
            # Create and train agent
            training_results = train_rl_agent(
                env=env,
                agent_config=config,
                n_episodes=10,  # Small number for testing
                eval_freq=5,
                save_path=f"models/{agent_name}_agent"
            )
            
            results[agent_name] = training_results
            
            print(f"Final average reward: {np.mean(training_results['episode_rewards']):.4f}")
            
            if training_results['episode_stats']:
                final_stats = training_results['episode_stats'][-1]
                print(f"Trading statistics:")
                for key, value in final_stats.items():
                    print(f"  {key}: {value:.4f}")
        
        except Exception as e:
            print(f"Error training {agent_name}: {str(e)}")
    
    return results

if __name__ == "__main__":
    print("Reinforcement Learning for Trading")
    print("=" * 50)
    
    # Test example usage
    results = example_usage()
    
    print("\n" + "=" * 50)
    print("RL trading agents are ready!")
    
    # Create a lightning module example
    print("\nCreating PyTorch Lightning RL trader...")
    config = {
        'state_dim': 84,  # Example dimension
        'action_dim': 2,
        'algorithm': 'ppo',
        'learning_rate': 0.001,
        'actor_lr': 0.0003,
        'critic_lr': 0.001,
        'gamma': 0.99,
        'clip_epsilon': 0.2,
        'buffer_size': 10000
    }
    
    rl_trader = RLTraderLightning(config)
    print(f"RL trader created with algorithm: {config['algorithm'].upper()}")
    
    print("\nAll RL components are ready for training!")