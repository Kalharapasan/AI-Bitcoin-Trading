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