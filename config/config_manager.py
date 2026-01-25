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