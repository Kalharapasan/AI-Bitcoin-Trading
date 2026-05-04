"""
Stop Loss Manager module for Bitcoin trading AI.
Advanced stop-loss management including trailing stops, volatility-based stops,
time-based stops, and machine learning-based stop loss optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
import json
from pathlib import Path
import hashlib
import asyncio
from collections import deque, defaultdict
import pickle
from functools import lru_cache
import uuid

# Import project modules
from config.settings import TradingSettings, RiskSettings, AppConstants
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.trading.order_manager import Order, OrderSide, OrderType, OrderStatus, OrderTimeInForce
from core.risk_management.risk_analyzer import RiskAnalyzer, RiskMetrics
from core.utils.cache import Cache

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Enums and Types ============
class StopLossType(str, Enum):
    """Types of stop-loss orders"""
    FIXED = "fixed"                      # Fixed price stop
    TRAILING = "trailing"                # Trailing stop (percentage or absolute)
    VOLATILITY = "volatility"            # Volatility-based stop
    TIME = "time"                        # Time-based stop
    MOVING_AVERAGE = "moving_average"    # Moving average stop
    BOLLINGER = "bollinger"              # Bollinger Band stop
    PARABOLIC_SAR = "parabolic_sar"      # Parabolic SAR stop
    CHANDELIER = "chandelier"            # Chandelier Exit stop
    ATR = "atr"                          # Average True Range stop
    MACHINE_LEARNING = "machine_learning" # ML-based stop
    HYBRID = "hybrid"                    # Combination of multiple types

class StopLossActivation(str, Enum):
    """Stop-loss activation conditions"""
    PRICE_BREACH = "price_breach"        # Price hits stop level
    VOLUME_SPIKE = "volume_spike"        # Volume spike detected
    VOLATILITY_SPIKE = "volatility_spike" # Volatility spike
    TIME_EXPIRY = "time_expiry"          # Time limit reached
    TECHNICAL_BREAK = "technical_break"  # Technical indicator break
    CORRELATION_BREAK = "correlation_break" # Correlation break
    NEWS_EVENT = "news_event"            # News event detected

class StopLossPriority(str, Enum):
    """Stop-loss priority levels"""
    LOW = "low"                          # Can be overridden
    MEDIUM = "medium"                    # Standard priority
    HIGH = "high"                        # Must be executed
    CRITICAL = "critical"                # Emergency stop

# ============ Data Structures ============
@dataclass
class StopLossConfig:
    """Configuration for stop-loss orders"""
    
    # Basic configuration
    stop_loss_type: StopLossType = StopLossType.TRAILING
    activation_condition: StopLossActivation = StopLossActivation.PRICE_BREACH
    priority: StopLossPriority = StopLossPriority.MEDIUM
    
    # Price-based parameters
    stop_price: Optional[float] = None            # Fixed stop price
    stop_percentage: float = 0.05                 # 5% stop loss
    trailing_distance: float = 0.03               # 3% trailing distance
    trail_activation_percentage: float = 0.10     # 10% profit before trailing activates
    
    # Volatility-based parameters
    volatility_multiplier: float = 2.0            # ATR multiplier
    volatility_lookback: int = 14                 # Periods for volatility calculation
    max_volatility_stop: float = 0.15             # Max 15% stop based on volatility
    
    # Time-based parameters
    time_horizon_hours: int = 24                  # 24-hour time stop
    time_decay_factor: float = 0.1                # Time decay adjustment
    
    # Moving average parameters
    ma_period: int = 20                           # MA period
    ma_type: str = "sma"                          # sma, ema, wma
    ma_offset: float = 0.02                       # 2% offset from MA
    
    # Bollinger Band parameters
    bb_period: int = 20                           # BB period
    bb_std: float = 2.0                           # BB standard deviations
    bb_offset: float = 0.0                        # BB offset
    
    # Risk management
    max_position_risk: float = 0.02               # Max 2% of portfolio per stop
    max_daily_stops: int = 5                      # Max 5 stops per day
    cooldown_period_minutes: int = 30             # 30 min cooldown after stop
    
    # Advanced features
    enable_dynamic_adjustment: bool = True        # Dynamically adjust stops
    enable_breakeven: bool = True                 # Move to breakeven when profitable
    enable_partial_close: bool = False            # Partial position closing
    partial_close_percentage: float = 0.5         # 50% position close
    
    # Machine learning
    use_ml_model: bool = False                    # Use ML for stop optimization
    ml_model_path: Optional[str] = None          # Path to ML model
    ml_confidence_threshold: float = 0.7         # 70% confidence threshold
    
    # Monitoring and alerts
    enable_alerts: bool = True                    # Enable stop-loss alerts
    alert_channels: List[str] = field(default_factory=lambda: ["log", "email"])
    monitor_frequency_minutes: int = 5           # Check stops every 5 minutes
    
    def __post_init__(self):
        """Validate configuration"""
        if self.stop_percentage <= 0 or self.stop_percentage > 0.5:
            raise ValueError("stop_percentage must be between 0 and 0.5")
        
        if self.trailing_distance <= 0 or self.trailing_distance > 0.2:
            raise ValueError("trailing_distance must be between 0 and 0.2")
        
        if self.volatility_multiplier <= 0:
            raise ValueError("volatility_multiplier must be positive")
        
        # Create ML model if enabled
        if self.use_ml_model and self.ml_model_path:
            self._load_ml_model()
    
    def _load_ml_model(self):
        """Load machine learning model"""
        try:
            import joblib
            self.ml_model = joblib.load(self.ml_model_path)
            logger.info(f"Loaded ML model from {self.ml_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {str(e)}")
            self.ml_model = None

@dataclass
class StopLossOrder:
    """Stop-loss order definition"""
    
    # Identification
    stop_id: str                                # Unique stop ID
    order_id: str                               # Associated order ID
    position_id: str                            # Associated position ID
    
    # Trading details
    symbol: str                                 # Trading symbol
    side: OrderSide                             # BUY or SELL
    quantity: float                             # Quantity to close
    entry_price: float                          # Entry price
    current_price: float                        # Current market price
    
    # Stop parameters
    stop_type: StopLossType                     # Type of stop
    stop_price: float                           # Current stop price
    initial_stop_price: float                   # Initial stop price
    activation_price: float                     # Price that triggers the stop
    
    # Status
    is_active: bool = True                      # Whether stop is active
    is_triggered: bool = False                  # Whether stop is triggered
    trigger_time: Optional[datetime] = None     # Time when triggered
    trigger_price: Optional[float] = None       # Price when triggered
    
    # Performance metrics
    max_profit_price: float = 0.0               # Maximum profit price reached
    max_profit_percentage: float = 0.0          # Maximum profit percentage
    current_drawdown: float = 0.0               # Current drawdown from max profit
    
    # Risk management
    risk_amount: float = 0.0                    # Amount at risk
    risk_percentage: float = 0.0                # Percentage of portfolio at risk
    
    # Configuration
    config: StopLossConfig = field(default_factory=StopLossConfig)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize stop-loss order"""
        if not self.stop_id:
            self.stop_id = f"stop_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Set initial stop price if not provided
        if self.initial_stop_price == 0:
            self.initial_stop_price = self.stop_price
        
        # Calculate initial activation price
        if self.activation_price == 0:
            self.activation_price = self.stop_price
        
        # Calculate current metrics
        self._update_metrics()
    
    def _update_metrics(self):
        """Update performance metrics"""
        if self.side == OrderSide.BUY:
            # For long positions, stop is below entry
            self.current_drawdown = (self.current_price - self.max_profit_price) / self.max_profit_price if self.max_profit_price > 0 else 0.0
            current_profit = (self.current_price - self.entry_price) / self.entry_price
        else:
            # For short positions, stop is above entry
            self.current_drawdown = (self.max_profit_price - self.current_price) / self.max_profit_price if self.max_profit_price > 0 else 0.0
            current_profit = (self.entry_price - self.current_price) / self.entry_price
        
        # Update max profit
        if current_profit > self.max_profit_percentage:
            self.max_profit_percentage = current_profit
            if self.side == OrderSide.BUY:
                self.max_profit_price = self.current_price
            else:
                self.max_profit_price = self.current_price
        
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'stop_id': self.stop_id,
            'order_id': self.order_id,
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_type': self.stop_type.value,
            'stop_price': self.stop_price,
            'initial_stop_price': self.initial_stop_price,
            'activation_price': self.activation_price,
            'is_active': self.is_active,
            'is_triggered': self.is_triggered,
            'trigger_time': self.trigger_time.isoformat() if self.trigger_time else None,
            'trigger_price': self.trigger_price,
            'max_profit_price': self.max_profit_price,
            'max_profit_percentage': self.max_profit_percentage,
            'current_drawdown': self.current_drawdown,
            'risk_amount': self.risk_amount,
            'risk_percentage': self.risk_percentage,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }
    
    def should_trigger(self, current_price: float) -> bool:
        """Check if stop should trigger based on current price"""
        self.current_price = current_price
        self._update_metrics()
        
        if not self.is_active or self.is_triggered:
            return False
        
        if self.side == OrderSide.BUY:
            # For long positions, stop triggers when price goes below stop
            if current_price <= self.activation_price:
                self.is_triggered = True
                self.trigger_time = datetime.now()
                self.trigger_price = current_price
                return True
        else:
            # For short positions, stop triggers when price goes above stop
            if current_price >= self.activation_price:
                self.is_triggered = True
                self.trigger_time = datetime.now()
                self.trigger_price = current_price
                return True
        
        return False
    
    def update_stop_price(self, current_price: float, market_data: Optional[pd.DataFrame] = None):
        """Update stop price based on current market conditions"""
        if not self.is_active or self.is_triggered:
            return
        
        self.current_price = current_price
        
        # Update based on stop type
        if self.stop_type == StopLossType.TRAILING:
            self._update_trailing_stop(current_price)
        elif self.stop_type == StopLossType.VOLATILITY:
            self._update_volatility_stop(current_price, market_data)
        elif self.stop_type == StopLossType.MOVING_AVERAGE:
            self._update_ma_stop(current_price, market_data)
        elif self.stop_type == StopLossType.BOLLINGER:
            self._update_bollinger_stop(current_price, market_data)
        elif self.stop_type == StopLossType.ATR:
            self._update_atr_stop(current_price, market_data)
        elif self.stop_type == StopLossType.CHANDELIER:
            self._update_chandelier_stop(current_price, market_data)
        elif self.stop_type == StopLossType.PARABOLIC_SAR:
            self._update_parabolic_sar_stop(current_price, market_data)
        elif self.stop_type == StopLossType.MACHINE_LEARNING:
            self._update_ml_stop(current_price, market_data)
        
        # Update activation price
        self._update_activation_price()
        
        self._update_metrics()
    
    def _update_trailing_stop(self, current_price: float):
        """Update trailing stop price"""
        if self.side == OrderSide.BUY:
            # For long positions
            if current_price > self.max_profit_price:
                self.max_profit_price = current_price
            
            # Calculate trailing stop
            trail_price = self.max_profit_price * (1 - self.config.trailing_distance)
            
            # Only move stop up, never down
            if trail_price > self.stop_price:
                self.stop_price = trail_price
            
            # Check if trailing should activate
            profit_percentage = (current_price - self.entry_price) / self.entry_price
            if profit_percentage >= self.config.trail_activation_percentage:
                # Activate trailing stop
                self.activation_price = self.stop_price
            else:
                # Use initial stop until trailing activates
                self.activation_price = self.initial_stop_price
        
        else:
            # For short positions
            if current_price < self.max_profit_price or self.max_profit_price == 0:
                self.max_profit_price = current_price
            
            # Calculate trailing stop
            trail_price = self.max_profit_price * (1 + self.config.trailing_distance)
            
            # Only move stop down, never up
            if trail_price < self.stop_price:
                self.stop_price = trail_price
            
            # Check if trailing should activate
            profit_percentage = (self.entry_price - current_price) / self.entry_price
            if profit_percentage >= self.config.trail_activation_percentage:
                # Activate trailing stop
                self.activation_price = self.stop_price
            else:
                # Use initial stop until trailing activates
                self.activation_price = self.initial_stop_price
    
    def _update_volatility_stop(self, current_price: float, market_data: Optional[pd.DataFrame] = None):
        """Update volatility-based stop"""
        if market_data is None or len(market_data) < self.config.volatility_lookback:
            return
        
        # Calculate volatility (standard deviation of returns)
        returns = market_data['close'].pct_change().dropna()
        if len(returns) >= self.config.volatility_lookback:
            volatility = returns.iloc[-self.config.volatility_lookback:].std()
        else:
            volatility = returns.std()
        
        # Calculate stop distance based on volatility
        stop_distance = volatility * self.config.volatility_multiplier
        stop_distance = min(stop_distance, self.config.max_volatility_stop)
        
        if self.side == OrderSide.BUY:
            # For long positions
            new_stop = current_price * (1 - stop_distance)
            
            # Only move stop up, never down (for trailing effect)
            if new_stop > self.stop_price:
                self.stop_price = new_stop
        else:
            # For short positions
            new_stop = current_price * (1 + stop_distance)
            
            # Only move stop down, never up (for trailing effect)
            if new_stop < self.stop_price:
                self.stop_price = new_stop
    
    def _update_ma_stop(self, current_price: float, market_data: Optional[pd.DataFrame] = None):
        """Update moving average stop"""
        if market_data is None or len(market_data) < self.config.ma_period:
            return
        
        # Calculate moving average
        if self.config.ma_type == "sma":
            ma = market_data['close'].rolling(window=self.config.ma_period).mean().iloc[-1]
        elif self.config.ma_type == "ema":
            ma = market_data['close'].ewm(span=self.config.ma_period).mean().iloc[-1]
        elif self.config.ma_type == "wma":
            weights = np.arange(1, self.config.ma_period + 1)
            ma = market_data['close'].rolling(window=self.config.ma_period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            ).iloc[-1]
        else:
            ma = market_data['close'].rolling(window=self.config.ma_period).mean().iloc[-1]
        
        if self.side == OrderSide.BUY:
            # For long positions, stop is below MA
            new_stop = ma * (1 - self.config.ma_offset)
            
            # Only move stop up, never down
            if new_stop > self.stop_price:
                self.stop_price = new_stop
        else:
            # For short positions, stop is above MA
            new_stop = ma * (1 + self.config.ma_offset)
            
            # Only move stop down, never up
            if new_stop < self.stop_price:
                self.stop_price = new_stop
    
    def _update_bollinger_stop(self, current_price: float, market_data: Optional[pd.DataFrame] = None):
        """Update Bollinger Band stop"""
        if market_data is None or len(market_data) < self.config.bb_period:
            return
        
        # Calculate Bollinger Bands
        ma = market_data['close'].rolling(window=self.config.bb_period).mean()
        std = market_data['close'].rolling(window=self.config.bb_period).std()
        
        upper_band = ma + (std * self.config.bb_std)
        lower_band = ma - (std * self.config.bb_std)
        
        upper_band = upper_band.iloc[-1]
        lower_band = lower_band.iloc[-1]
        
        if self.side == OrderSide.BUY:
            # For long positions, use lower band with offset
            new_stop = lower_band * (1 - self.config.bb_offset)
            
            # Only move stop up, never down
            if new_stop > self.stop_price:
                self.stop_price = new_stop
        else:
            # For short positions, use upper band with offset
            new_stop = upper_band * (1 + self.config.bb_offset)
            
            # Only move stop down, never up
            if new_stop < self.stop_price:
                self.stop_price = new_stop
    
    def _update_atr_stop(self, current_price: float, market_data: Optional[pd.DataFrame] = None):
        """Update ATR-based stop"""
        if market_data is None or len(market_data) < self.config.volatility_lookback:
            return
        
        # Calculate ATR
        high = market_data['high']
        low = market_data['low']
        close = market_data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=self.config.volatility_lookback).mean().iloc[-1]
        
        # Calculate stop distance
        stop_distance = atr * self.config.volatility_multiplier
        
        if self.side == OrderSide.BUY:
            # For long positions
            new_stop = current_price - stop_distance
            
            # Only move stop up, never down (for trailing effect)
            if new_stop > self.stop_price:
                self.stop_price = new_stop
        else:
            # For short positions
            new_stop = current_price + stop_distance
            
            # Only move stop down, never up (for trailing effect)
            if new_stop < self.stop_price:
                self.stop_price = new_stop
    
    def _update_chandelier_stop(self, current_price: float, market_data: Optional[pd.DataFrame] = None):
        """Update Chandelier Exit stop"""
        if market_data is None or len(market_data) < self.config.volatility_lookback:
            return
        
        # Calculate highest high or lowest low over lookback period
        high = market_data['high']
        low = market_data['low']
        
        if self.side == OrderSide.BUY:
            # For long positions
            highest_high = high.rolling(window=self.config.volatility_lookback).max().iloc[-1]
            atr = self._calculate_atr(market_data)
            new_stop = highest_high - (atr * self.config.volatility_multiplier)
            
            # Only move stop up, never down
            if new_stop > self.stop_price:
                self.stop_price = new_stop
        else:
            # For short positions
            lowest_low = low.rolling(window=self.config.volatility_lookback).min().iloc[-1]
            atr = self._calculate_atr(market_data)
            new_stop = lowest_low + (atr * self.config.volatility_multiplier)
            
            # Only move stop down, never up
            if new_stop < self.stop_price:
                self.stop_price = new_stop
    
    def _update_parabolic_sar_stop(self, current_price: float, market_data: Optional[pd.DataFrame] = None):
        """Update Parabolic SAR stop"""
        if market_data is None or len(market_data) < 10:
            return
        
        # Simplified Parabolic SAR calculation
        # In production, use a proper SAR implementation
        high = market_data['high']
        low = market_data['low']
        
        # Calculate acceleration factor (simplified)
        acceleration = 0.02
        max_acceleration = 0.2
        
        if self.side == OrderSide.BUY:
            # For long positions
            extreme_point = high.rolling(window=5).max().iloc[-1]
            
            # Calculate SAR
            sar = self.stop_price + acceleration * (extreme_point - self.stop_price)
            
            # Update acceleration
            if current_price > extreme_point:
                acceleration = min(acceleration + 0.02, max_acceleration)
            
            # Update stop
            self.stop_price = sar
        else:
            # For short positions
            extreme_point = low.rolling(window=5).min().iloc[-1]
            
            # Calculate SAR
            sar = self.stop_price - acceleration * (self.stop_price - extreme_point)
            
            # Update acceleration
            if current_price < extreme_point:
                acceleration = min(acceleration + 0.02, max_acceleration)
            
            # Update stop
            self.stop_price = sar
    
    def _update_ml_stop(self, current_price: float, market_data: Optional[pd.DataFrame] = None):
        """Update ML-based stop"""
        if self.config.ml_model is None or market_data is None:
            return
        
        try:
            # Prepare features for ML model
            features = self._prepare_ml_features(current_price, market_data)
            
            # Predict optimal stop distance
            stop_distance = self.config.ml_model.predict([features])[0]
            
            # Apply stop distance
            if self.side == OrderSide.BUY:
                new_stop = current_price * (1 - stop_distance)
                
                # Only move stop up, never down
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
            else:
                new_stop = current_price * (1 + stop_distance)
                
                # Only move stop down, never up
                if new_stop < self.stop_price:
                    self.stop_price = new_stop
        
        except Exception as e:
            logger.warning(f"ML stop update failed: {str(e)}")
    
    def _prepare_ml_features(self, current_price: float, market_data: pd.DataFrame) -> List[float]:
        """Prepare features for ML model"""
        features = []
        
        # Price features
        features.append(current_price)
        features.append(self.entry_price)
        features.append(self.stop_price)
        
        # Return features
        returns = market_data['close'].pct_change().dropna()
        if len(returns) > 0:
            features.append(returns.iloc[-1])  # Latest return
            features.append(returns.mean())    # Mean return
            features.append(returns.std())     # Volatility
        
        # Volume features
        if 'volume' in market_data.columns:
            volume = market_data['volume']
            if len(volume) > 0:
                features.append(volume.iloc[-1])  # Latest volume
                features.append(volume.mean())    # Mean volume
        
        # Technical indicators
        if len(market_data) >= 20:
            # RSI (simplified)
            gains = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            losses = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
            if losses != 0:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
            features.append(rsi)
        
        # Position features
        features.append(self.max_profit_percentage)
        features.append(self.current_drawdown)
        features.append(self.risk_percentage)
        
        return features
    
    def _calculate_atr(self, market_data: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        high = market_data['high']
        low = market_data['low']
        close = market_data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=self.config.volatility_lookback).mean().iloc[-1]
        
        return atr
    
    def _update_activation_price(self):
        """Update activation price based on stop type and configuration"""
        if self.stop_type == StopLossType.TRAILING:
            # For trailing stops, activation depends on profit
            profit_percentage = abs(self.current_price - self.entry_price) / self.entry_price
            if profit_percentage >= self.config.trail_activation_percentage:
                self.activation_price = self.stop_price
            else:
                self.activation_price = self.initial_stop_price
        else:
            # For other stop types, activation price is the stop price
            self.activation_price = self.stop_price

@dataclass
class StopLossTrigger:
    """Stop-loss trigger event"""
    
    trigger_id: str
    stop_id: str
    order_id: str
    position_id: str
    symbol: str
    side: OrderSide
    
    # Trigger details
    trigger_price: float
    trigger_time: datetime
    trigger_type: StopLossActivation
    
    # Position details
    entry_price: float
    exit_price: float
    quantity: float
    
    # Performance metrics
    profit_loss: float
    profit_loss_percentage: float
    holding_period_hours: float
    max_profit_percentage: float
    max_drawdown_percentage: float
    
    # Risk metrics
    risk_amount: float
    risk_percentage: float
    
    # Order details
    exit_order_id: Optional[str] = None
    exit_order_status: Optional[OrderStatus] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trigger_id': self.trigger_id,
            'stop_id': self.stop_id,
            'order_id': self.order_id,
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'trigger_price': self.trigger_price,
            'trigger_time': self.trigger_time.isoformat(),
            'trigger_type': self.trigger_type.value,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'profit_loss': self.profit_loss,
            'profit_loss_percentage': self.profit_loss_percentage,
            'holding_period_hours': self.holding_period_hours,
            'max_profit_percentage': self.max_profit_percentage,
            'max_drawdown_percentage': self.max_drawdown_percentage,
            'risk_amount': self.risk_amount,
            'risk_percentage': self.risk_percentage,
            'exit_order_id': self.exit_order_id,
            'exit_order_status': self.exit_order_status.value if self.exit_order_status else None,
            'metadata': self.metadata
        }

@dataclass
class StopLossPerformance:
    """Stop-loss performance metrics"""
    
    total_stops: int = 0
    triggered_stops: int = 0
    active_stops: int = 0
    
    # Profit/Loss metrics
    total_pnl: float = 0.0
    average_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    average_loss: float = 0.0
    average_win: float = 0.0
    largest_loss: float = 0.0
    largest_win: float = 0.0
    
    # Effectiveness metrics
    stops_that_saved_money: int = 0
    stops_that_lost_money: int = 0
    effectiveness_rate: float = 0.0
    
    # Time metrics
    average_holding_period_hours: float = 0.0
    fastest_stop_hours: float = 0.0
    slowest_stop_hours: float = 0.0
    
    # By stop type
    performance_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Time series
    daily_performance: List[Dict[str, Any]] = field(default_factory=list)
    
    def update(self, trigger: StopLossTrigger):
        """Update performance with new trigger"""
        self.total_stops += 1
        self.triggered_stops += 1
        
        # Update P&L metrics
        self.total_pnl += trigger.profit_loss
        
        if trigger.profit_loss > 0:
            self.average_win = (
                (self.average_win * (self.triggered_stops - 1) + trigger.profit_loss) 
                / self.triggered_stops
            )
        else:
            self.average_loss = (
                (self.average_loss * (self.triggered_stops - 1) + abs(trigger.profit_loss)) 
                / self.triggered_stops
            )
        
        # Update win rate
        if trigger.profit_loss > 0:
            self.win_rate = (
                (self.win_rate * (self.triggered_stops - 1) + 1) 
                / self.triggered_stops
            )
        else:
            self.win_rate = (
                self.win_rate * (self.triggered_stops - 1) 
                / self.triggered_stops
            )
        
        # Update largest win/loss
        if trigger.profit_loss > self.largest_win:
            self.largest_win = trigger.profit_loss
        if trigger.profit_loss < self.largest_loss:
            self.largest_loss = trigger.profit_loss
        
        # Update effectiveness
        if trigger.profit_loss > 0:
            self.stops_that_saved_money += 1
        else:
            self.stops_that_lost_money += 1
        
        self.effectiveness_rate = (
            self.stops_that_saved_money / self.triggered_stops 
            if self.triggered_stops > 0 else 0.0
        )
        
        # Update holding period
        self.average_holding_period_hours = (
            (self.average_holding_period_hours * (self.triggered_stops - 1) + trigger.holding_period_hours) 
            / self.triggered_stops
        )
        
        if trigger.holding_period_hours < self.fastest_stop_hours or self.fastest_stop_hours == 0:
            self.fastest_stop_hours = trigger.holding_period_hours
        
        if trigger.holding_period_hours > self.slowest_stop_hours:
            self.slowest_stop_hours = trigger.holding_period_hours
        
        # Update performance by type
        stop_type = trigger.metadata.get('stop_type', 'unknown')
        if stop_type not in self.performance_by_type:
            self.performance_by_type[stop_type] = {
                'count': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0
            }
        
        self.performance_by_type[stop_type]['count'] += 1
        self.performance_by_type[stop_type]['total_pnl'] += trigger.profit_loss
        
        if trigger.profit_loss > 0:
            self.performance_by_type[stop_type]['win_rate'] = (
                (self.performance_by_type[stop_type]['win_rate'] * 
                 (self.performance_by_type[stop_type]['count'] - 1) + 1) 
                / self.performance_by_type[stop_type]['count']
            )
        else:
            self.performance_by_type[stop_type]['win_rate'] = (
                self.performance_by_type[stop_type]['win_rate'] * 
                (self.performance_by_type[stop_type]['count'] - 1) 
                / self.performance_by_type[stop_type]['count']
            )
        
        # Update average P&L
        self.average_pnl = self.total_pnl / self.triggered_stops if self.triggered_stops > 0 else 0.0
        
        # Calculate profit factor
        total_wins = sum(max(0, pnl) for pnl in [trigger.profit_loss])
        total_losses = sum(abs(min(0, pnl)) for pnl in [trigger.profit_loss])
        if total_losses > 0:
            self.profit_factor = total_wins / total_losses
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_stops': self.total_stops,
            'triggered_stops': self.triggered_stops,
            'active_stops': self.active_stops,
            'total_pnl': self.total_pnl,
            'average_pnl': self.average_pnl,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'average_loss': self.average_loss,
            'average_win': self.average_win,
            'largest_loss': self.largest_loss,
            'largest_win': self.largest_win,
            'stops_that_saved_money': self.stops_that_saved_money,
            'stops_that_lost_money': self.stops_that_lost_money,
            'effectiveness_rate': self.effectiveness_rate,
            'average_holding_period_hours': self.average_holding_period_hours,
            'fastest_stop_hours': self.fastest_stop_hours,
            'slowest_stop_hours': self.slowest_stop_hours,
            'performance_by_type': self.performance_by_type,
            'daily_performance': self.daily_performance
        }

# ============ Stop Loss Algorithms ============
class StopLossCalculator:
    """Calculate optimal stop-loss levels"""
    
    def __init__(self, config: StopLossConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def calculate_stop_price(self,
                           entry_price: float,
                           side: OrderSide,
                           market_data: pd.DataFrame,
                           stop_type: StopLossType,
                           portfolio_risk: float = 0.02) -> float:
        """Calculate stop price based on type and parameters"""
        
        if stop_type == StopLossType.FIXED:
            return self._calculate_fixed_stop(entry_price, side)
        
        elif stop_type == StopLossType.TRAILING:
            return self._calculate_trailing_stop(entry_price, side)
        
        elif stop_type == StopLossType.VOLATILITY:
            return self._calculate_volatility_stop(entry_price, side, market_data)
        
        elif stop_type == StopLossType.ATR:
            return self._calculate_atr_stop(entry_price, side, market_data)
        
        elif stop_type == StopLossType.MOVING_AVERAGE:
            return self._calculate_ma_stop(entry_price, side, market_data)
        
        elif stop_type == StopLossType.BOLLINGER:
            return self._calculate_bollinger_stop(entry_price, side, market_data)
        
        elif stop_type == StopLossType.CHANDELIER:
            return self._calculate_chandelier_stop(entry_price, side, market_data)
        
        elif stop_type == StopLossType.PARABOLIC_SAR:
            return self._calculate_parabolic_sar_stop(entry_price, side, market_data)
        
        elif stop_type == StopLossType.MACHINE_LEARNING:
            return self._calculate_ml_stop(entry_price, side, market_data)
        
        elif stop_type == StopLossType.HYBRID:
            return self._calculate_hybrid_stop(entry_price, side, market_data, portfolio_risk)
        
        else:
            raise ValueError(f"Unknown stop type: {stop_type}")
    
    def _calculate_fixed_stop(self, entry_price: float, side: OrderSide) -> float:
        """Calculate fixed stop price"""
        if side == OrderSide.BUY:
            return entry_price * (1 - self.config.stop_percentage)
        else:
            return entry_price * (1 + self.config.stop_percentage)
    
    def _calculate_trailing_stop(self, entry_price: float, side: OrderSide) -> float:
        """Calculate initial trailing stop price"""
        if side == OrderSide.BUY:
            return entry_price * (1 - self.config.stop_percentage)
        else:
            return entry_price * (1 + self.config.stop_percentage)
    
    def _calculate_volatility_stop(self, entry_price: float, side: OrderSide, market_data: pd.DataFrame) -> float:
        """Calculate volatility-based stop"""
        if len(market_data) < self.config.volatility_lookback:
            return self._calculate_fixed_stop(entry_price, side)
        
        # Calculate volatility
        returns = market_data['close'].pct_change().dropna()
        if len(returns) >= self.config.volatility_lookback:
            volatility = returns.iloc[-self.config.volatility_lookback:].std()
        else:
            volatility = returns.std()
        
        # Calculate stop distance
        stop_distance = volatility * self.config.volatility_multiplier
        stop_distance = min(stop_distance, self.config.max_volatility_stop)
        
        if side == OrderSide.BUY:
            return entry_price * (1 - stop_distance)
        else:
            return entry_price * (1 + stop_distance)
    
    def _calculate_atr_stop(self, entry_price: float, side: OrderSide, market_data: pd.DataFrame) -> float:
        """Calculate ATR-based stop"""
        if len(market_data) < self.config.volatility_lookback:
            return self._calculate_fixed_stop(entry_price, side)
        
        # Calculate ATR
        high = market_data['high']
        low = market_data['low']
        close = market_data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=self.config.volatility_lookback).mean().iloc[-1]
        
        # Calculate stop distance
        stop_distance = (atr * self.config.volatility_multiplier) / entry_price
        
        if side == OrderSide.BUY:
            return entry_price * (1 - stop_distance)
        else:
            return entry_price * (1 + stop_distance)
    
    def _calculate_ma_stop(self, entry_price: float, side: OrderSide, market_data: pd.DataFrame) -> float:
        """Calculate moving average stop"""
        if len(market_data) < self.config.ma_period:
            return self._calculate_fixed_stop(entry_price, side)
        
        # Calculate moving average
        if self.config.ma_type == "sma":
            ma = market_data['close'].rolling(window=self.config.ma_period).mean().iloc[-1]
        elif self.config.ma_type == "ema":
            ma = market_data['close'].ewm(span=self.config.ma_period).mean().iloc[-1]
        elif self.config.ma_type == "wma":
            weights = np.arange(1, self.config.ma_period + 1)
            ma = market_data['close'].rolling(window=self.config.ma_period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            ).iloc[-1]
        else:
            ma = market_data['close'].rolling(window=self.config.ma_period).mean().iloc[-1]
        
        if side == OrderSide.BUY:
            return ma * (1 - self.config.ma_offset)
        else:
            return ma * (1 + self.config.ma_offset)
    
    def _calculate_bollinger_stop(self, entry_price: float, side: OrderSide, market_data: pd.DataFrame) -> float:
        """Calculate Bollinger Band stop"""
        if len(market_data) < self.config.bb_period:
            return self._calculate_fixed_stop(entry_price, side)
        
        # Calculate Bollinger Bands
        ma = market_data['close'].rolling(window=self.config.bb_period).mean()
        std = market_data['close'].rolling(window=self.config.bb_period).std()
        
        upper_band = ma + (std * self.config.bb_std)
        lower_band = ma - (std * self.config.bb_std)
        
        upper_band = upper_band.iloc[-1]
        lower_band = lower_band.iloc[-1]
        
        if side == OrderSide.BUY:
            return lower_band * (1 - self.config.bb_offset)
        else:
            return upper_band * (1 + self.config.bb_offset)
    
    def _calculate_chandelier_stop(self, entry_price: float, side: OrderSide, market_data: pd.DataFrame) -> float:
        """Calculate Chandelier Exit stop"""
        if len(market_data) < self.config.volatility_lookback:
            return self._calculate_fixed_stop(entry_price, side)
        
        # Calculate ATR
        atr = self._calculate_atr_value(market_data)
        
        if side == OrderSide.BUY:
            # For long positions
            highest_high = market_data['high'].rolling(window=self.config.volatility_lookback).max().iloc[-1]
            return highest_high - (atr * self.config.volatility_multiplier)
        else:
            # For short positions
            lowest_low = market_data['low'].rolling(window=self.config.volatility_lookback).min().iloc[-1]
            return lowest_low + (atr * self.config.volatility_multiplier)
    
    def _calculate_parabolic_sar_stop(self, entry_price: float, side: OrderSide, market_data: pd.DataFrame) -> float:
        """Calculate Parabolic SAR stop"""
        # Simplified implementation
        # In production, use a proper SAR calculation
        
        if side == OrderSide.BUY:
            # Initial SAR for long position
            lowest_low = market_data['low'].rolling(window=5).min().iloc[-1]
            return lowest_low
        else:
            # Initial SAR for short position
            highest_high = market_data['high'].rolling(window=5).max().iloc[-1]
            return highest_high
    
    def _calculate_ml_stop(self, entry_price: float, side: OrderSide, market_data: pd.DataFrame) -> float:
        """Calculate ML-based stop"""
        if self.config.ml_model is None:
            return self._calculate_fixed_stop(entry_price, side)
        
        try:
            # Prepare features
            features = self._prepare_features_for_ml(entry_price, side, market_data)
            
            # Predict optimal stop distance
            stop_distance = self.config.ml_model.predict([features])[0]
            stop_distance = min(stop_distance, 0.5)  # Cap at 50%
            
            if side == OrderSide.BUY:
                return entry_price * (1 - stop_distance)
            else:
                return entry_price * (1 + stop_distance)
        
        except Exception as e:
            self.logger.warning(f"ML stop calculation failed: {str(e)}")
            return self._calculate_fixed_stop(entry_price, side)
    
    def _calculate_hybrid_stop(self, entry_price: float, side: OrderSide, market_data: pd.DataFrame, portfolio_risk: float) -> float:
        """Calculate hybrid stop using multiple methods"""
        
        # Calculate stops using different methods
        stops = []
        
        # Fixed stop
        fixed_stop = self._calculate_fixed_stop(entry_price, side)
        stops.append(('fixed', fixed_stop))
        
        # Volatility stop
        volatility_stop = self._calculate_volatility_stop(entry_price, side, market_data)
        stops.append(('volatility', volatility_stop))
        
        # ATR stop
        atr_stop = self._calculate_atr_stop(entry_price, side, market_data)
        stops.append(('atr', atr_stop))
        
        # Weight the stops based on risk tolerance
        # More conservative (tighter stops) for higher risk
        if portfolio_risk > 0.03:  # High risk tolerance
            # Use tighter stops (take the minimum)
            if side == OrderSide.BUY:
                stop_price = max(stop for _, stop in stops)
            else:
                stop_price = min(stop for _, stop in stops)
        else:  # Low risk tolerance
            # Use average of stops
            if side == OrderSide.BUY:
                # For long positions, take the maximum (most conservative)
                stop_price = max(stop for _, stop in stops)
            else:
                # For short positions, take the minimum (most conservative)
                stop_price = min(stop for _, stop in stops)
        
        return stop_price
    
    def _calculate_atr_value(self, market_data: pd.DataFrame) -> float:
        """Calculate ATR value"""
        high = market_data['high']
        low = market_data['low']
        close = market_data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=self.config.volatility_lookback).mean().iloc[-1]
        
        return atr
    
    def _prepare_features_for_ml(self, entry_price: float, side: OrderSide, market_data: pd.DataFrame) -> List[float]:
        """Prepare features for ML model"""
        features = []
        
        # Price features
        features.append(entry_price)
        if len(market_data) > 0:
            features.append(market_data['close'].iloc[-1])
        
        # Return features
        if len(market_data) > 1:
            returns = market_data['close'].pct_change().dropna()
            if len(returns) > 0:
                features.append(returns.iloc[-1])  # Latest return
                features.append(returns.mean())    # Mean return
                features.append(returns.std())     # Volatility
        
        # Volume features
        if 'volume' in market_data.columns and len(market_data) > 0:
            volume = market_data['volume']
            features.append(volume.iloc[-1])
            if len(volume) > 0:
                features.append(volume.mean())
        
        # Market regime features
        if len(market_data) >= 50:
            # Calculate trend
            short_ma = market_data['close'].rolling(window=20).mean().iloc[-1]
            long_ma = market_data['close'].rolling(window=50).mean().iloc[-1]
            features.append(short_ma / long_ma)  # Trend indicator
        
        # Side feature
        features.append(1 if side == OrderSide.BUY else 0)
        
        return features

# ============ Main Stop Loss Manager ============
class StopLossManager:
    """Main stop-loss management engine"""
    
    def __init__(self, config: Optional[StopLossConfig] = None):
        self.config = config or StopLossConfig()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.stop_calculator = StopLossCalculator(self.config)
        self.risk_analyzer = RiskAnalyzer()
        
        # Stop tracking
        self.active_stops: Dict[str, StopLossOrder] = {}
        self.triggered_stops: Dict[str, StopLossTrigger] = {}
        self.stop_performance = StopLossPerformance()
        
        # Risk limits
        self.daily_stop_count = 0
        self.last_stop_reset = datetime.now().date()
        
        # Monitoring
        self.monitoring_task = None
        self.is_monitoring = False
        
        # Cache for market data
        self.cache = Cache(ttl=60)  # 1 minute TTL
        
        # Alert system
        self.alert_handlers = self._initialize_alert_handlers()
        
        # Cooldown tracking
        self.cooldown_stops: Dict[str, datetime] = {}
        
        self.logger.info("Stop Loss Manager initialized")
    
    def _initialize_alert_handlers(self) -> Dict[str, Callable]:
        """Initialize alert handlers"""
        handlers = {
            'log': self._log_alert,
            'email': self._email_alert,
            'webhook': self._webhook_alert
        }
        
        return handlers
    
    def _log_alert(self, message: str, level: str = "INFO"):
        """Log alert message"""
        if level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "CRITICAL":
            self.logger.critical(message)
    
    def _email_alert(self, message: str, level: str = "INFO"):
        """Send email alert"""
        # Implementation would connect to email service
        self.logger.info(f"Email alert ({level}): {message}")
    
    def _webhook_alert(self, message: str, level: str = "INFO"):
        """Send webhook alert"""
        # Implementation would send HTTP request to webhook
        self.logger.info(f"Webhook alert ({level}): {message}")
    
    def create_stop_loss(self,
                        order: Order,
                        position_id: str,
                        market_data: pd.DataFrame,
                        portfolio_value: float,
                        custom_config: Optional[StopLossConfig] = None) -> StopLossOrder:
        """Create a new stop-loss order"""
        
        # Reset daily stop count if new day
        self._reset_daily_stop_count_if_needed()
        
        # Check daily stop limit
        if self.daily_stop_count >= self.config.max_daily_stops:
            raise ValueError(f"Daily stop limit reached: {self.daily_stop_count}/{self.config.max_daily_stops}")
        
        # Use custom config if provided, otherwise use default
        config = custom_config or self.config
        
        # Calculate initial stop price
        stop_price = self.stop_calculator.calculate_stop_price(
            entry_price=order.price or market_data['close'].iloc[-1],
            side=order.side,
            market_data=market_data,
            stop_type=config.stop_loss_type,
            portfolio_risk=config.max_position_risk
        )
        
        # Calculate risk amount
        risk_amount = abs((stop_price - order.price) * order.quantity) if order.price else 0.0
        risk_percentage = risk_amount / portfolio_value if portfolio_value > 0 else 0.0
        
        # Check maximum position risk
        if risk_percentage > config.max_position_risk:
            self.logger.warning(f"Stop loss risk {risk_percentage:.2%} exceeds maximum {config.max_position_risk:.2%}")
            # Adjust quantity to meet risk limit
            if order.price and order.price != stop_price:
                max_risk_amount = portfolio_value * config.max_position_risk
                max_quantity = max_risk_amount / abs(order.price - stop_price)
                order.quantity = min(order.quantity, max_quantity)
                risk_amount = abs((stop_price - order.price) * order.quantity)
                risk_percentage = risk_amount / portfolio_value
        
        # Create stop-loss order
        stop_order = StopLossOrder(
            stop_id=f"stop_{uuid.uuid4().hex[:8]}",
            order_id=order.order_id,
            position_id=position_id,
            symbol=order.trading_pair,
            side=order.side,
            quantity=order.quantity,
            entry_price=order.price or market_data['close'].iloc[-1],
            current_price=market_data['close'].iloc[-1],
            stop_type=config.stop_loss_type,
            stop_price=stop_price,
            initial_stop_price=stop_price,
            activation_price=stop_price,
            risk_amount=risk_amount,
            risk_percentage=risk_percentage,
            config=config,
            metadata={
                'creation_reason': 'new_position',
                'portfolio_value': portfolio_value,
                'volatility': market_data['close'].pct_change().std() if len(market_data) > 1 else 0.0
            }
        )
        
        # Add to active stops
        self.active_stops[stop_order.stop_id] = stop_order
        
        # Update daily stop count
        self.daily_stop_count += 1
        
        # Send alert
        if config.enable_alerts:
            self._send_alert(
                f"Created stop loss: {stop_order.symbol} {stop_order.side.value} "
                f"@{stop_order.entry_price:.2f}, stop @{stop_order.stop_price:.2f} "
                f"({abs(stop_order.stop_price - stop_order.entry_price)/stop_order.entry_price:.1%})",
                "INFO"
            )
        
        self.logger.info(f"Created stop loss {stop_order.stop_id} for order {order.order_id}")
        
        return stop_order
    
    def update_stop_loss(self,
                        stop_id: str,
                        current_price: float,
                        market_data: Optional[pd.DataFrame] = None):
        """Update stop-loss price based on current market conditions"""
        
        if stop_id not in self.active_stops:
            self.logger.warning(f"Stop loss {stop_id} not found")
            return
        
        stop_order = self.active_stops[stop_id]
        
        if not stop_order.is_active or stop_order.is_triggered:
            return
        
        # Check cooldown period
        if stop_id in self.cooldown_stops:
            cooldown_end = self.cooldown_stops[stop_id]
            if datetime.now() < cooldown_end:
                return
        
        # Update stop price
        stop_order.update_stop_price(current_price, market_data)
        
        # Check breakeven activation
        if stop_order.config.enable_breakeven:
            self._check_breakeven(stop_order, current_price)
        
        # Check for trigger
        if stop_order.should_trigger(current_price):
            self._trigger_stop_loss(stop_order, current_price)
        
        # Send alert for significant stop adjustment
        if abs(stop_order.stop_price - stop_order.initial_stop_price) / stop_order.initial_stop_price > 0.01:
            if stop_order.config.enable_alerts:
                self._send_alert(
                    f"Stop loss adjusted: {stop_order.symbol} {stop_order.side.value} "
                    f"stop moved to {stop_order.stop_price:.2f} "
                    f"(from {stop_order.initial_stop_price:.2f})",
                    "INFO"
                )
    
    def _check_breakeven(self, stop_order: StopLossOrder, current_price: float):
        """Check and move stop to breakeven if profitable"""
        
        if stop_order.side == OrderSide.BUY:
            profit_percentage = (current_price - stop_order.entry_price) / stop_order.entry_price
            if profit_percentage >= stop_order.config.trail_activation_percentage:
                # Move stop to entry price (breakeven)
                if current_price > stop_order.stop_price:
                    stop_order.stop_price = stop_order.entry_price
        else:
            profit_percentage = (stop_order.entry_price - current_price) / stop_order.entry_price
            if profit_percentage >= stop_order.config.trail_activation_percentage:
                # Move stop to entry price (breakeven)
                if current_price < stop_order.stop_price:
                    stop_order.stop_price = stop_order.entry_price
    
    def _trigger_stop_loss(self, stop_order: StopLossOrder, trigger_price: float):
        """Trigger a stop-loss order"""
        
        # Calculate P&L
        if stop_order.side == OrderSide.BUY:
            profit_loss = (trigger_price - stop_order.entry_price) * stop_order.quantity
            profit_loss_percentage = (trigger_price - stop_order.entry_price) / stop_order.entry_price
        else:
            profit_loss = (stop_order.entry_price - trigger_price) * stop_order.quantity
            profit_loss_percentage = (stop_order.entry_price - trigger_price) / stop_order.entry_price
        
        # Calculate holding period
        holding_period = (datetime.now() - stop_order.created_at).total_seconds() / 3600
        
        # Create trigger event
        trigger = StopLossTrigger(
            trigger_id=f"trigger_{uuid.uuid4().hex[:8]}",
            stop_id=stop_order.stop_id,
            order_id=stop_order.order_id,
            position_id=stop_order.position_id,
            symbol=stop_order.symbol,
            side=stop_order.side,
            trigger_price=trigger_price,
            trigger_time=datetime.now(),
            trigger_type=StopLossActivation.PRICE_BREACH,
            entry_price=stop_order.entry_price,
            exit_price=trigger_price,
            quantity=stop_order.quantity,
            profit_loss=profit_loss,
            profit_loss_percentage=profit_loss_percentage,
            holding_period_hours=holding_period,
            max_profit_percentage=stop_order.max_profit_percentage,
            max_drawdown_percentage=stop_order.current_drawdown,
            risk_amount=stop_order.risk_amount,
            risk_percentage=stop_order.risk_percentage,
            metadata={
                'stop_type': stop_order.stop_type.value,
                'initial_stop_price': stop_order.initial_stop_price,
                'max_profit_price': stop_order.max_profit_price,
                'config': stop_order.config.__dict__
            }
        )
        
        # Add to triggered stops
        self.triggered_stops[trigger.trigger_id] = trigger
        
        # Update performance metrics
        self.stop_performance.update(trigger)
        
        # Remove from active stops
        del self.active_stops[stop_order.stop_id]
        
        # Add to cooldown
        cooldown_end = datetime.now() + timedelta(minutes=stop_order.config.cooldown_period_minutes)
        self.cooldown_stops[stop_order.stop_id] = cooldown_end
        
        # Send alert
        if stop_order.config.enable_alerts:
            alert_level = "CRITICAL" if profit_loss_percentage < -0.1 else "WARNING"
            self._send_alert(
                f"Stop loss triggered: {stop_order.symbol} {stop_order.side.value} "
                f"@ {trigger_price:.2f}, P&L: {profit_loss:.2f} ({profit_loss_percentage:.2%})",
                alert_level
            )
        
        self.logger.info(f"Stop loss {stop_order.stop_id} triggered at {trigger_price:.2f}, "
                        f"P&L: {profit_loss:.2f} ({profit_loss_percentage:.2%})")
        
        # Generate exit order
        exit_order = self._generate_exit_order(stop_order, trigger_price)
        trigger.exit_order_id = exit_order.order_id if exit_order else None
        
        return trigger, exit_order
    
    def _generate_exit_order(self, stop_order: StopLossOrder, trigger_price: float) -> Optional[Order]:
        """Generate exit order for triggered stop"""
        
        # Check if partial close is enabled
        if stop_order.config.enable_partial_close:
            # Close partial position
            close_quantity = stop_order.quantity * stop_order.config.partial_close_percentage
        else:
            # Close full position
            close_quantity = stop_order.quantity
        
        # Create exit order
        exit_order = Order(
            order_id=f"exit_{stop_order.order_id}",
            trading_pair=stop_order.symbol,
            side=OrderSide.SELL if stop_order.side == OrderSide.BUY else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=close_quantity,
            price=None,  # Market order
            time_in_force=OrderTimeInForce.IOC,
            status=OrderStatus.NEW,
            metadata={
                'stop_id': stop_order.stop_id,
                'trigger_price': trigger_price,
                'original_order_id': stop_order.order_id,
                'is_stop_loss': True
            }
        )
        
        return exit_order
    
    def cancel_stop_loss(self, stop_id: str, reason: str = "manual_cancellation"):
        """Cancel a stop-loss order"""
        
        if stop_id not in self.active_stops:
            self.logger.warning(f"Stop loss {stop_id} not found")
            return False
        
        stop_order = self.active_stops[stop_id]
        
        if stop_order.is_triggered:
            self.logger.warning(f"Stop loss {stop_id} already triggered")
            return False
        
        # Deactivate stop
        stop_order.is_active = False
        
        # Remove from active stops
        del self.active_stops[stop_id]
        
        # Send alert
        if stop_order.config.enable_alerts:
            self._send_alert(
                f"Stop loss cancelled: {stop_order.symbol} {stop_order.side.value} "
                f"@ {stop_order.stop_price:.2f}, Reason: {reason}",
                "INFO"
            )
        
        self.logger.info(f"Stop loss {stop_id} cancelled: {reason}")
        
        return True
    
    def modify_stop_loss(self,
                        stop_id: str,
                        new_stop_price: Optional[float] = None,
                        new_quantity: Optional[float] = None,
                        new_config: Optional[StopLossConfig] = None):
        """Modify an existing stop-loss order"""
        
        if stop_id not in self.active_stops:
            self.logger.warning(f"Stop loss {stop_id} not found")
            return False
        
        stop_order = self.active_stops[stop_id]
        
        if stop_order.is_triggered:
            self.logger.warning(f"Stop loss {stop_id} already triggered")
            return False
        
        # Update stop price if provided
        if new_stop_price is not None:
            stop_order.stop_price = new_stop_price
            stop_order.activation_price = new_stop_price
        
        # Update quantity if provided
        if new_quantity is not None:
            stop_order.quantity = new_quantity
        
        # Update config if provided
        if new_config is not None:
            stop_order.config = new_config
        
        stop_order.updated_at = datetime.now()
        
        # Send alert
        if stop_order.config.enable_alerts:
            self._send_alert(
                f"Stop loss modified: {stop_order.symbol} {stop_order.side.value} "
                f"new stop @ {stop_order.stop_price:.2f}",
                "INFO"
            )
        
        self.logger.info(f"Stop loss {stop_id} modified")
        
        return True
    
    def batch_update_stops(self,
                          price_updates: Dict[str, float],
                          market_data: Dict[str, pd.DataFrame]):
        """Update multiple stops with new price data"""
        
        triggered_stops = []
        
        for stop_id, stop_order in list(self.active_stops.items()):
            if stop_order.symbol in price_updates:
                current_price = price_updates[stop_order.symbol]
                market_data_for_symbol = market_data.get(stop_order.symbol)
                
                # Update stop
                self.update_stop_loss(stop_id, current_price, market_data_for_symbol)
                
                # Check if triggered
                if stop_id not in self.active_stops:  # Was triggered and removed
                    triggered_stops.append(stop_id)
        
        return triggered_stops
    
    def start_monitoring(self, interval_minutes: int = 5):
        """Start automatic stop-loss monitoring"""
        
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        
        async def monitor_task():
            while self.is_monitoring:
                try:
                    # Check all active stops
                    self._check_all_stops()
                    
                    # Clean up old cooldown entries
                    self._cleanup_cooldown_entries()
                    
                    # Log status
                    self.logger.debug(f"Stop loss monitoring: {len(self.active_stops)} active stops")
                    
                except Exception as e:
                    self.logger.error(f"Error in stop loss monitoring: {str(e)}")
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
        
        self.monitoring_task = asyncio.create_task(monitor_task())
        self.logger.info(f"Started stop loss monitoring with {interval_minutes} minute interval")
    
    def stop_monitoring(self):
        """Stop automatic stop-loss monitoring"""
        
        if not self.is_monitoring:
            self.logger.warning("Monitoring not started")
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
        
        self.logger.info("Stopped stop loss monitoring")
    
    def _check_all_stops(self):
        """Check all active stops"""
        # This would typically fetch current prices and update stops
        # For now, it's a placeholder for the monitoring logic
        pass
    
    def _cleanup_cooldown_entries(self):
        """Clean up expired cooldown entries"""
        current_time = datetime.now()
        expired_stops = [
            stop_id for stop_id, cooldown_end in self.cooldown_stops.items()
            if current_time > cooldown_end
        ]
        
        for stop_id in expired_stops:
            del self.cooldown_stops[stop_id]
    
    def _reset_daily_stop_count_if_needed(self):
        """Reset daily stop count if new day"""
        today = datetime.now().date()
        if today != self.last_stop_reset:
            self.daily_stop_count = 0
            self.last_stop_reset = today
            self.logger.info(f"Reset daily stop count for {today}")
    
    def _send_alert(self, message: str, level: str = "INFO"):
        """Send alert through configured channels"""
        
        for channel in self.config.alert_channels:
            if channel in self.alert_handlers:
                try:
                    self.alert_handlers[channel](message, level)
                except Exception as e:
                    self.logger.error(f"Error sending alert via {channel}: {str(e)}")
    
    def get_stop_loss_status(self, stop_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a stop-loss order"""
        
        if stop_id in self.active_stops:
            stop_order = self.active_stops[stop_id]
            return {
                'status': 'active',
                'details': stop_order.to_dict(),
                'distance_to_stop': self._calculate_distance_to_stop(stop_order)
            }
        elif stop_id in self.triggered_stops:
            trigger = self.triggered_stops[stop_id]
            return {
                'status': 'triggered',
                'details': trigger.to_dict()
            }
        else:
            return None
    
    def _calculate_distance_to_stop(self, stop_order: StopLossOrder) -> Dict[str, float]:
        """Calculate distance to stop in various units"""
        
        if stop_order.side == OrderSide.BUY:
            price_distance = stop_order.current_price - stop_order.activation_price
            percentage_distance = price_distance / stop_order.current_price
        else:
            price_distance = stop_order.activation_price - stop_order.current_price
            percentage_distance = price_distance / stop_order.current_price
        
        # Calculate in volatility units (ATR)
        atr_distance = 0.0
        if 'atr' in stop_order.metadata:
            atr = stop_order.metadata['atr']
            atr_distance = price_distance / atr if atr > 0 else 0.0
        
        return {
            'price_distance': price_distance,
            'percentage_distance': percentage_distance,
            'atr_distance': atr_distance,
            'risk_reward_ratio': abs(price_distance / (stop_order.entry_price - stop_order.stop_price)) if stop_order.entry_price != stop_order.stop_price else 0.0
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get stop-loss performance report"""
        
        report = {
            'performance': self.stop_performance.to_dict(),
            'active_stops_count': len(self.active_stops),
            'triggered_stops_count': len(self.triggered_stops),
            'daily_stop_count': self.daily_stop_count,
            'cooldown_stops_count': len(self.cooldown_stops),
            'monitoring_status': self.is_monitoring
        }
        
        # Add top performing stops
        if self.triggered_stops:
            top_winners = sorted(
                self.triggered_stops.values(),
                key=lambda x: x.profit_loss_percentage,
                reverse=True
            )[:5]
            
            top_losers = sorted(
                self.triggered_stops.values(),
                key=lambda x: x.profit_loss_percentage
            )[:5]
            
            report['top_winners'] = [
                {
                    'symbol': stop.symbol,
                    'pnl_percentage': stop.profit_loss_percentage,
                    'stop_type': stop.metadata.get('stop_type', 'unknown')
                }
                for stop in top_winners
            ]
            
            report['top_losers'] = [
                {
                    'symbol': stop.symbol,
                    'pnl_percentage': stop.profit_loss_percentage,
                    'stop_type': stop.metadata.get('stop_type', 'unknown')
                }
                for stop in top_losers
            ]
        
        return report
    
    def optimize_stop_loss_parameters(self,
                                    historical_data: pd.DataFrame,
                                    stop_type: StopLossType = StopLossType.TRAILING) -> Dict[str, float]:
        """Optimize stop-loss parameters using historical data"""
        
        if len(historical_data) < 100:
            self.logger.warning("Insufficient historical data for optimization")
            return {}
        
        # Define parameter ranges for optimization
        if stop_type == StopLossType.TRAILING:
            param_ranges = {
                'stop_percentage': (0.01, 0.10),      # 1% to 10%
                'trailing_distance': (0.01, 0.10),    # 1% to 10%
                'trail_activation_percentage': (0.02, 0.20)  # 2% to 20%
            }
        elif stop_type == StopLossType.VOLATILITY:
            param_ranges = {
                'volatility_multiplier': (1.0, 3.0),  # 1x to 3x volatility
                'volatility_lookback': (5, 30),       # 5 to 30 periods
                'max_volatility_stop': (0.05, 0.20)   # 5% to 20% max stop
            }
        else:
            self.logger.warning(f"Parameter optimization not implemented for {stop_type}")
            return {}
        
        # Simple grid search
        best_params = {}
        best_sharpe = -np.inf
        
        # Generate parameter combinations
        param_names = list(param_ranges.keys())
        param_values = []
        
        for param_name in param_names:
            min_val, max_val = param_ranges[param_name]
            if param_name.endswith('percentage'):
                values = np.linspace(min_val, max_val, 5)
            else:
                values = np.linspace(min_val, max_val, 5, dtype=int) if param_name.endswith('lookback') else np.linspace(min_val, max_val, 5)
            param_values.append(values)
        
        # Test all combinations
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            
            # Backtest with these parameters
            sharpe = self._backtest_parameters(historical_data, stop_type, params)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        
        self.logger.info(f"Optimized parameters for {stop_type}: {best_params} (Sharpe: {best_sharpe:.3f})")
        
        return best_params
    
    def _backtest_parameters(self,
                            historical_data: pd.DataFrame,
                            stop_type: StopLossType,
                            params: Dict[str, float]) -> float:
        """Backtest stop-loss parameters"""
        
        # Simplified backtest
        # In production, implement full backtesting with position tracking
        
        returns = historical_data['close'].pct_change().dropna()
        
        # Simulate stop losses
        simulated_returns = []
        in_position = False
        entry_price = 0.0
        
        for i in range(1, len(historical_data)):
            current_price = historical_data['close'].iloc[i]
            
            if not in_position:
                # Enter position
                in_position = True
                entry_price = current_price
                
                # Calculate stop price
                if stop_type == StopLossType.TRAILING:
                    stop_price = entry_price * (1 - params['stop_percentage'])
                elif stop_type == StopLossType.VOLATILITY:
                    # Calculate volatility
                    lookback = int(params['volatility_lookback'])
                    if i >= lookback:
                        volatility = returns.iloc[i-lookback:i].std()
                    else:
                        volatility = returns.iloc[:i].std() if i > 0 else 0.01
                    
                    stop_distance = volatility * params['volatility_multiplier']
                    stop_distance = min(stop_distance, params['max_volatility_stop'])
                    stop_price = entry_price * (1 - stop_distance)
                else:
                    stop_price = entry_price * 0.95  # Default 5% stop
            
            else:
                # Check stop loss
                if current_price <= stop_price:
                    # Stop triggered
                    exit_return = (stop_price - entry_price) / entry_price
                    simulated_returns.append(exit_return)
                    in_position = False
                else:
                    # Update trailing stop
                    if stop_type == StopLossType.TRAILING:
                        if current_price > entry_price * (1 + params['trail_activation_percentage']):
                            # Update trailing stop
                            trail_stop = current_price * (1 - params['trailing_distance'])
                            stop_price = max(stop_price, trail_stop)
        
        # Calculate Sharpe ratio
        if simulated_returns:
            mean_return = np.mean(simulated_returns)
            std_return = np.std(simulated_returns)
            if std_return > 0:
                sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
            else:
                sharpe = 0.0
        else:
            sharpe = -np.inf
        
        return sharpe
    
    def save_state(self, filepath: str):
        """Save stop-loss manager state to file"""
        
        try:
            state = {
                'active_stops': {k: v.to_dict() for k, v in self.active_stops.items()},
                'triggered_stops': {k: v.to_dict() for k, v in self.triggered_stops.items()},
                'stop_performance': self.stop_performance.to_dict(),
                'daily_stop_count': self.daily_stop_count,
                'last_stop_reset': self.last_stop_reset.isoformat(),
                'cooldown_stops': {k: v.isoformat() for k, v in self.cooldown_stops.items()},
                'config': self.config.__dict__
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.logger.info(f"Stop loss manager state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
    
    def load_state(self, filepath: str):
        """Load stop-loss manager state from file"""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Load active stops
            self.active_stops = {}
            for stop_id, stop_data in state.get('active_stops', {}).items():
                # Convert string dates back to datetime
                stop_data['created_at'] = datetime.fromisoformat(stop_data['created_at'])
                stop_data['updated_at'] = datetime.fromisoformat(stop_data['updated_at'])
                if stop_data['trigger_time']:
                    stop_data['trigger_time'] = datetime.fromisoformat(stop_data['trigger_time'])
                
                # Create stop order
                config_data = stop_data.pop('config', {})
                config = StopLossConfig(**config_data)
                
                stop_order = StopLossOrder(**stop_data, config=config)
                self.active_stops[stop_id] = stop_order
            
            # Load triggered stops
            self.triggered_stops = {}
            for trigger_id, trigger_data in state.get('triggered_stops', {}).items():
                # Convert string dates back to datetime
                trigger_data['trigger_time'] = datetime.fromisoformat(trigger_data['trigger_time'])
                
                # Create trigger
                trigger = StopLossTrigger(**trigger_data)
                self.triggered_stops[trigger_id] = trigger
            
            # Load performance
            self.stop_performance = StopLossPerformance(**state.get('stop_performance', {}))
            
            # Load other state
            self.daily_stop_count = state.get('daily_stop_count', 0)
            self.last_stop_reset = datetime.fromisoformat(state.get('last_stop_reset', datetime.now().date().isoformat()))
            
            # Load cooldown stops
            self.cooldown_stops = {}
            for stop_id, cooldown_str in state.get('cooldown_stops', {}).items():
                self.cooldown_stops[stop_id] = datetime.fromisoformat(cooldown_str)
            
            self.logger.info(f"Stop loss manager state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")

# ============ Factory Function ============
def create_stop_loss_manager(config: Optional[StopLossConfig] = None) -> StopLossManager:
    """Factory function to create stop loss manager"""
    return StopLossManager(config)

# ============ Main Execution ============
async def main():
    """Main execution for testing"""
    
    # Create stop loss manager
    manager = create_stop_loss_manager()
    
    # Create test order
    test_order = Order(
        order_id="test_order_001",
        trading_pair="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.1,
        price=50000.0,
        time_in_force=OrderTimeInForce.GTC,
        status=OrderStatus.NEW
    )
    
    # Create test market data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
    prices = 50000 + np.random.randn(100).cumsum() * 100
    
    market_data = pd.DataFrame({
        'close': prices,
        'high': prices + np.random.uniform(0, 100, 100),
        'low': prices - np.random.uniform(0, 100, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    }, index=dates)
    
    try:
        # Create stop loss
        stop_order = manager.create_stop_loss(
            order=test_order,
            position_id="test_position_001",
            market_data=market_data,
            portfolio_value=100000.0
        )
        
        print(f"\n=== Stop Loss Created ===")
        print(f"Stop ID: {stop_order.stop_id}")
        print(f"Entry Price: ${stop_order.entry_price:.2f}")
        print(f"Initial Stop Price: ${stop_order.stop_price:.2f}")
        print(f"Stop Distance: {abs(stop_order.stop_price - stop_order.entry_price)/stop_order.entry_price:.2%}")
        print(f"Risk Amount: ${stop_order.risk_amount:.2f}")
        print(f"Risk Percentage: {stop_order.risk_percentage:.2%}")
        
        # Simulate price movements and update stop
        print(f"\n=== Simulating Price Movements ===")
        
        simulated_prices = [51000, 52000, 51500, 50500, 49500, 49000]
        
        for i, price in enumerate(simulated_prices):
            print(f"\nStep {i+1}: Price = ${price:.2f}")
            
            # Update stop
            manager.update_stop_loss(stop_order.stop_id, price, market_data)
            
            # Get stop status
            status = manager.get_stop_loss_status(stop_order.stop_id)
            
            if status and status['status'] == 'active':
                stop_order = manager.active_stops[stop_order.stop_id]
                print(f"  Current Stop: ${stop_order.stop_price:.2f}")
                print(f"  Activation Price: ${stop_order.activation_price:.2f}")
                print(f"  Max Profit: {stop_order.max_profit_percentage:.2%}")
                print(f"  Current Drawdown: {stop_order.current_drawdown:.2%}")
                
                # Calculate distance to stop
                distance = manager._calculate_distance_to_stop(stop_order)
                print(f"  Distance to Stop: {distance['percentage_distance']:.2%}")
            else:
                print("  Stop triggered!")
                break
        
        # Get performance report
        print(f"\n=== Performance Report ===")
        report = manager.get_performance_report()
        print(f"Active Stops: {report['active_stops_count']}")
        print(f"Triggered Stops: {report['triggered_stops_count']}")
        print(f"Daily Stop Count: {report['daily_stop_count']}")
        
        if report['performance']['triggered_stops'] > 0:
            print(f"\nPerformance Metrics:")
            print(f"  Total P&L: ${report['performance']['total_pnl']:.2f}")
            print(f"  Win Rate: {report['performance']['win_rate']:.1%}")
            print(f"  Average P&L: ${report['performance']['average_pnl']:.2f}")
            print(f"  Profit Factor: {report['performance']['profit_factor']:.2f}")
        
        # Test parameter optimization
        print(f"\n=== Parameter Optimization ===")
        optimized_params = manager.optimize_stop_loss_parameters(
            historical_data=market_data,
            stop_type=StopLossType.TRAILING
        )
        
        if optimized_params:
            print(f"Optimized Parameters: {optimized_params}")
        
        # Save state
        manager.save_state("stop_loss_manager_state.json")
        print(f"\nState saved to stop_loss_manager_state.json")
        
    except Exception as e:
        print(f"Error in stop loss test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())