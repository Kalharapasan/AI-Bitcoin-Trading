import os
import yaml
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
from datetime import datetime
import jsonschema
import logging
from pydantic import BaseModel, Field, field_validator, ConfigDict
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

for directory in [CONFIG_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

class TradingMode(str, Enum):
    """Trading modes"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

class TimeFrame(str, Enum):
    """Available timeframes"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

class OrderType(str, Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class PositionSide(str, Enum):
    """Position sides"""
    LONG = "long"
    SHORT = "short"

class ModelType(str, Enum):
    """Model types"""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"
    ATTENTION = "attention"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"
    DEEP_RL = "deep_rl"

class PositionSizing(str, Enum):
    """Position sizing methods"""
    KELLY = "kelly"
    FIXED = "fixed"
    VOLATILITY = "volatility"
    OPTIMAL_F = "optimal_f"
    MARTINGALE = "martingale"
    ANTI_MARTINGALE = "anti_martingale"

class TradingStrategy(str, Enum):
    """Trading strategies"""
    ML_ENSEMBLE = "ml_ensemble"
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    TRANSFORMER_LSTM = "transformer_lstm"
    CNN_LSTM = "cnn_lstm"
    ATTENTION_LSTM = "attention_lstm"
    DEEP_REINFORCEMENT = "deep_rl"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"

class APIConfig(BaseModel):
    """API Configuration Model"""
    exchanges: Dict[str, Dict[str, str]] = Field(
        default_factory=lambda: {
            'binance': {'api_key': '', 'api_secret': '', 'testnet': False},
            'coinbase': {'api_key': '', 'api_secret': ''},
            'kraken': {'api_key': '', 'api_secret': ''},
            'bitstamp': {'api_key': '', 'api_secret': ''}
        },
        description="Exchange API configurations"
    )
    
    data_providers: Dict[str, Dict[str, str]] = Field(
        default_factory=lambda: {
            'cryptocompare': {'api_key': ''},
            'glassnode': {'api_key': ''},
            'coingecko': {'api_key': ''}
        },
        description="Data provider API configurations"
    )
    
    sentiment_apis: Dict[str, Dict[str, str]] = Field(
        default_factory=lambda: {
            'twitter': {'bearer_token': ''},
            'reddit': {'client_id': '', 'client_secret': ''},
            'newsapi': {'api_key': ''}
        },
        description="Sentiment API configurations"
    )
    
    rate_limit_multiplier: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Multiplier for rate limits (0.1-1.0)"
    )
    
    timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="API timeout in seconds"
    )
    
    retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of API retry attempts"
    )
    
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    @field_validator('exchanges')
    @classmethod
    def validate_exchange_credentials(cls, v):
        """Validate exchange credentials"""
        for exchange, creds in v.items():
            if creds.get('api_key') and not creds.get('api_secret'):
                raise ValueError(f"Exchange {exchange} requires both api_key and api_secret")
        return v

class ModelArchitecture(BaseModel):
    lstm_layers: List[int] = Field(
        default=[128, 64, 32],
        description="LSTM layer sizes"
    )
    
    lstm_dropout: float = Field(
        default=0.3,
        ge=0.0,
        le=0.5,
        description="LSTM dropout rate"
    )
    
    lstm_recurrent_dropout: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="LSTM recurrent dropout rate"
    )
    
    lstm_bidirectional: bool = Field(
        default=True,
        description="Use bidirectional LSTM"
    )
    
    transformer_d_model: int = Field(
        default=64,
        ge=32,
        le=512,
        description="Transformer model dimension"
    )
    
    transformer_nhead: int = Field(
        default=8,
        ge=1,
        le=16,
        description="Transformer number of attention heads"
    )
    
    transformer_num_layers: int = Field(
        default=3,
        ge=1,
        le=12,
        description="Transformer number of layers"
    )
    
    transformer_dim_feedforward: int = Field(
        default=256,
        ge=64,
        le=2048,
        description="Transformer feedforward dimension"
    )
    
    transformer_dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Transformer dropout rate"
    )
    
    cnn_filters: List[int] = Field(
        default=[64, 128, 256],
        description="CNN filter sizes"
    )
    
    cnn_kernel_sizes: List[int] = Field(
        default=[3, 5, 3],
        description="CNN kernel sizes"
    )
    
    cnn_pool_sizes: List[int] = Field(
        default=[2, 2, 1],
        description="CNN pool sizes"
    )
    
    attention_heads: int = Field(
        default=8,
        ge=1,
        le=16,
        description="Number of attention heads"
    )
    
    attention_dim: int = Field(
        default=64,
        ge=32,
        le=256,
        description="Attention dimension"
    )
    batch_size: int = Field(
        default=32,
        ge=8,
        le=256,
        description="Training batch size"
    )
    
    epochs: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Training epochs"
    )
    
    learning_rate: float = Field(
        default=0.001,
        ge=1e-5,
        le=0.1,
        description="Learning rate"
    )
    
    weight_decay: float = Field(
        default=0.0001,
        ge=0.0,
        le=0.01,
        description="Weight decay"
    )
    
    patience: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Early stopping patience"
    )
    
    sequence_length: int = Field(
        default=60,
        ge=10,
        le=500,
        description="Input sequence length"
    )
    
    prediction_horizon: int = Field(
        default=24,
        ge=1,
        le=100,
        description="Prediction horizon"
    )
    
    feature_count: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of features"
    )
    
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    @field_validator('lstm_layers', 'cnn_filters', 'cnn_kernel_sizes', 'cnn_pool_sizes')
    @classmethod
    def validate_list_length(cls, v):
        """Validate list lengths"""
        if len(v) == 0:
            raise ValueError("List cannot be empty")
        return v

class ModelConfig(BaseModel):
    enabled_models: List[ModelType] = Field(
        default_factory=lambda: [
            ModelType.LSTM,
            ModelType.TRANSFORMER,
            ModelType.XGBOOST,
            ModelType.ENSEMBLE
        ],
        description="Enabled model types"
    )
    
    architecture: ModelArchitecture = Field(
        default_factory=ModelArchitecture,
        description="Model architecture configuration"
    )
    
    ensemble_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            'lstm': 0.25,
            'transformer': 0.25,
            'attention': 0.20,
            'xgboost': 0.15,
            'lightgbm': 0.15
        },
        description="Ensemble model weights"
    )
    
    retrain_frequency: str = Field(
        default="1w",
        description="Model retraining frequency (1d, 1w, 1m)"
    )
    
    retrain_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Retrain threshold (accuracy drop)"
    )
    
    online_learning: bool = Field(
        default=True,
        description="Enable online learning"
    )
    
    use_gpu: bool = Field(
        default=True,
        description="Use GPU for training"
    )
    
    model_save_path: str = Field(
        default=str(MODELS_DIR),
        description="Path to save models"
    )
    
    @field_validator('ensemble_weights')
    @classmethod
    def validate_ensemble_weights(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ensemble weights must sum to 1, got {total}")
        return v
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = 'forbid'

class TradingConfig(BaseModel):
    mode: TradingMode = Field(
        default=TradingMode.PAPER,
        description="Trading mode"
    )
    
    symbol: str = Field(
        default="BTC/USDT",
        description="Trading symbol"
    )
    
    timeframe: TimeFrame = Field(
        default=TimeFrame.HOUR_1,
        description="Trading timeframe"
    )
    
    lookback_periods: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Lookback periods for analysis"
    )
    
    initial_capital: float = Field(
        default=10000.0,
        ge=100.0,
        le=10000000.0,
        description="Initial capital"
    )
    primary_strategy: TradingStrategy = Field(
        default=TradingStrategy.ML_ENSEMBLE,
        description="Primary trading strategy"
    )
    
    secondary_strategies: List[TradingStrategy] = Field(
        default_factory=lambda: [
            TradingStrategy.TREND_FOLLOWING,
            TradingStrategy.MEAN_REVERSION
        ],
        description="Secondary trading strategies"
    )
    position_sizing: PositionSizing = Field(
        default=PositionSizing.KELLY,
        description="Position sizing method"
    )
    
    max_position_size: float = Field(
        default=0.15,
        ge=0.01,
        le=1.0,
        description="Maximum position size (fraction of portfolio)"
    )
    
    min_position_size: float = Field(
        default=0.01,
        ge=0.001,
        le=0.1,
        description="Minimum position size"
    )
    
    max_open_positions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum open positions"
    )
    order_type: OrderType = Field(
        default=OrderType.LIMIT,
        description="Default order type"
    )
    
    slippage_tolerance: float = Field(
        default=0.001,
        ge=0.0,
        le=0.01,
        description="Slippage tolerance"
    )
    
    take_profit: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="Take profit percentage"
    )
    
    stop_loss: float = Field(
        default=0.02,
        ge=0.005,
        le=0.1,
        description="Stop loss percentage"
    )
    
    trailing_stop: bool = Field(
        default=True,
        description="Enable trailing stop"
    )
    
    trailing_distance: float = Field(
        default=0.01,
        ge=0.001,
        le=0.05,
        description="Trailing stop distance"
    )
    
    trading_hours: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Trading hours by day (e.g., {'mon': ['09:00', '17:00']})"
    )
    
    avoid_high_impact_news: bool = Field(
        default=True,
        description="Avoid trading during high-impact news"
    )
    
    news_cooldown_minutes: int = Field(
        default=30,
        ge=0,
        le=240,
        description="Cooldown period after news (minutes)"
    )
