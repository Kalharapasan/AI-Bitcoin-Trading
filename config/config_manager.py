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
    
    @field_validator('max_position_size', 'min_position_size')
    @classmethod
    def validate_position_sizes(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Position size must be between 0 and 1")
        return v

    @field_validator('take_profit', 'stop_loss')
    @classmethod
    def validate_risk_params(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Risk parameters must be between 0 and 1")
        return v
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = 'forbid'
        
class RiskConfig(BaseModel):
    max_daily_loss: float = Field(
        default=0.02,
        ge=0.0,
        le=0.1,
        description="Maximum daily loss (fraction of portfolio)"
    )
    
    max_drawdown: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        description="Maximum drawdown"
    )
    
    max_position_risk: float = Field(
        default=0.02,
        ge=0.001,
        le=0.1,
        description="Maximum risk per position"
    )
    
    max_correlation: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Maximum allowed correlation between positions"
    )
    var_confidence: float = Field(
        default=0.95,
        ge=0.9,
        le=0.99,
        description="Value at Risk confidence level"
    )
    
    cvar_confidence: float = Field(
        default=0.99,
        ge=0.95,
        le=0.999,
        description="Conditional Value at Risk confidence level"
    )
    circuit_breakers: Dict[str, float] = Field(
        default_factory=lambda: {
            "price_change_5min": 0.05,
            "price_change_1hour": 0.10,
            "volume_spike": 5.0,
            "volatility_spike": 3.0
        },
        description="Circuit breaker thresholds"
    )
    
    stress_test_scenarios: List[str] = Field(
        default_factory=lambda: [
            "flash_crash",
            "liquidity_crisis",
            "volatility_spike",
            "exchange_outage"
        ],
        description="Stress test scenarios"
    )
    
    stress_test_frequency: str = Field(
        default="1d",
        description="Stress test frequency"
    )
    
    risk_models: List[str] = Field(
        default_factory=lambda: [
            "value_at_risk",
            "expected_shortfall",
            "monte_carlo",
            "stress_testing"
        ],
        description="Enabled risk models"
    )
    
    @field_validator('circuit_breakers')
    @classmethod
    def validate_circuit_breakers(cls, v):
        for key, value in v.items():
            if value <= 0:
                raise ValueError(f"Circuit breaker {key} must be positive")
        return v
    
    class Config:
        validate_assignment = True
        extra = 'forbid'
    

class DataConfig(BaseModel):
    primary_exchange: str = Field(
        default="binance",
        description="Primary exchange for data"
    )
    
    backup_exchanges: List[str] = Field(
        default_factory=lambda: ["coinbase", "kraken", "bitstamp"],
        description="Backup exchanges"
    )
    
    data_quality_threshold: float = Field(
        default=0.95,
        ge=0.8,
        le=1.0,
        description="Minimum data quality score"
    )
    
    max_data_latency_ms: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum allowed data latency (ms)"
    )
    
    technical_indicators: List[str] = Field(
        default_factory=lambda: [
            "sma_20", "sma_50", "ema_12", "ema_26",
            "rsi", "macd", "bb_upper", "bb_lower",
            "atr", "adx", "stoch_k", "stoch_d",
            "obv", "mfi", "cci", "williams_r"
        ],
        description="Technical indicators to calculate"
    )
    
    onchain_metrics: List[str] = Field(
        default_factory=lambda: [
            "hash_rate", "active_addresses", "transaction_count",
            "miner_revenue", "exchange_flows"
        ],
        description="On-chain metrics to collect"
    )
    
    sentiment_metrics: List[str] = Field(
        default_factory=lambda: [
            "fear_greed_index", "twitter_sentiment",
            "news_sentiment", "social_volume"
        ],
        description="Sentiment metrics to collect"
    )
    
    realtime_enabled: bool = Field(
        default=True,
        description="Enable real-time data collection"
    )
    
    websocket_enabled: bool = Field(
        default=True,
        description="Enable WebSocket connections"
    )
    
    historical_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Days of historical data to collect"
    )
    
    data_save_path: str = Field(
        default=str(DATA_DIR),
        description="Path to save data"
    )
    
    cache_enabled: bool = Field(
        default=True,
        description="Enable data caching"
    )
    
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache time-to-live (seconds)"
    )
    
    @field_validator('backup_exchanges')
    @classmethod
    def validate_backup_exchanges(cls, v):
        if len(v) == 0:
            raise ValueError("At least one backup exchange is required")
        return v
    
    class Config:
        validate_assignment = True
        extra = 'forbid'

class MonitoringConfig(BaseModel):
    metrics_enabled: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Metrics server port"
    )
    
    push_to_prometheus: bool = Field(
        default=False,
        description="Push metrics to Prometheus"
    )
    alert_channels: List[str] = Field(
        default_factory=lambda: ["console", "email"],
        description="Alert channels"
    )
    
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "drawdown": 0.05,
            "loss_streak": 3,
            "low_confidence": 0.7,
            "high_slippage": 0.005,
            "high_latency": 2000
        },
        description="Alert thresholds"
    )
    
    dashboard_enabled: bool = Field(
        default=True,
        description="Enable dashboard"
    )
    
    dashboard_port: int = Field(
        default=8050,
        ge=1024,
        le=65535,
        description="Dashboard port"
    )
    
    dashboard_refresh_interval: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Dashboard refresh interval (seconds)"
    )
    
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    log_format: str = Field(
        default="standard", 
        description="Log format"
    )
    
    log_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Log retention days"
    )
    
    log_file: str = Field(
        default=str(LOGS_DIR / "trading.log"),
        description="Log file path"
    )
    performance_metrics: List[str] = Field(
        default_factory=lambda: [
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "win_rate", "profit_factor", "calmar_ratio"
        ],
        description="Performance metrics to track"
    )
    
    @field_validator('alert_thresholds')
    @classmethod
    def validate_alert_thresholds(cls, v):
        for key, value in v.items():
            if value < 0:
                raise ValueError(f"Alert threshold {key} must be non-negative")
        return v
    
    class Config:
        validate_assignment = True
        extra = 'forbid'

class BacktestConfig(BaseModel):
    start_date: str = Field(
        default="2023-01-01",
        description="Backtest start date (YYYY-MM-DD)"
    )
    
    end_date: str = Field(
        default="2024-01-01",
        description="Backtest end date (YYYY-MM-DD)"
    )
    
    initial_capital: float = Field(
        default=10000.0,
        ge=100.0,
        description="Initial capital for backtest"
    )
    
    commission: float = Field(
        default=0.001,
        ge=0.0,
        le=0.01,
        description="Trading commission"
    )
    
    slippage: float = Field(
        default=0.001,
        ge=0.0,
        le=0.01,
        description="Slippage percentage"
    )
    walk_forward: bool = Field(
        default=True,
        description="Enable walk-forward testing"
    )
    
    training_window: int = Field(
        default=180,
        ge=30,
        le=730,
        description="Training window (days)"
    )
    
    testing_window: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Testing window (days)"
    )
    monte_carlo_simulations: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of Monte Carlo simulations"
    )
    metrics: List[str] = Field(
        default_factory=lambda: [
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "win_rate", "profit_factor", "calmar_ratio",
            "omega_ratio", "value_at_risk"
        ],
        description="Backtest metrics to calculate"
    )
    
    results_save_path: str = Field(
        default=str(RESULTS_DIR),
        description="Path to save backtest results"
    )
    
    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")
        return v

    model_config = ConfigDict(validate_assignment=True, extra='forbid')

@dataclass
class AppConfig:
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    version: str = "1.0.0"
    environment: str = "development"
    config_hash: str = ""
    last_modified: datetime = field(default_factory=datetime.now)
    config_path: str = ""
    
    @classmethod
    def load(cls, config_path: Union[str, Path] = None) -> 'AppConfig':
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            AppConfig instance
        """
        if config_path is None:
            config_path = CONFIG_DIR / "config.yaml"
        
        config_path = Path(config_path)
        if not config_path.exists():
            config = cls()
            config.config_path = str(config_path)
            config.save()
            logger.info(f"Created default configuration at {config_path}")
            return config

        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                config_data = {}
            errors = ConfigValidator.validate_config(config_data)
            if errors:
                logger.error(f"Configuration validation errors: {errors}")
                raise ValueError(f"Configuration validation failed: {errors}")
            
            config = cls(
                api=APIConfig(**config_data.get('api', {})),
                trading=TradingConfig(**config_data.get('trading', {})),
                models=ModelConfig(**config_data.get('models', {})),
                risk=RiskConfig(**config_data.get('risk', {})),
                data=DataConfig(**config_data.get('data', {})),
                monitoring=MonitoringConfig(**config_data.get('monitoring', {})),
                backtest=BacktestConfig(**config_data.get('backtest', {})),
                version=config_data.get('version', '1.0.0'),
                environment=config_data.get('environment', 'development'),
                config_path=str(config_path)
            )
            config.config_hash = config.calculate_hash()
            config.last_modified = datetime.fromtimestamp(config_path.stat().st_mtime)
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
            raise
    
    def save(self, config_path: Union[str, Path] = None) -> None:
        if config_path is None:
            config_path = self.config_path or CONFIG_DIR / "config.yaml"
        
        config_path = Path(config_path)
        config_path.parent.mkdir(exist_ok=True, parents=True)
        config_dict = self.to_dict()
        config_dict.update({
            'version': self.version,
            'environment': self.environment,
            'last_modified': datetime.now().isoformat()
        })
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
        
        self.config_path = str(config_path)
        self.config_hash = self.calculate_hash()
        self.last_modified = datetime.now()
        
        logger.info(f"Configuration saved to {config_path}")