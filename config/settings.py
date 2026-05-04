"""
Application Settings and Constants
Centralized configuration for the Bitcoin Trading AI application
"""

import os
import sys
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging
import json

# ============ Base Paths ============
class Paths:
    """Application paths configuration"""
    
    # Base directory (project root)
    BASE_DIR = Path(__file__).parent.parent
    
    # Configuration directories
    CONFIG_DIR = BASE_DIR / "config"
    SECRETS_DIR = BASE_DIR / "secrets"
    
    # Data directories
    DATA_DIR = BASE_DIR / "data"
    DATA_RAW_DIR = DATA_DIR / "raw"
    DATA_PROCESSED_DIR = DATA_DIR / "processed"
    DATA_CACHE_DIR = DATA_DIR / "cache"
    
    # Model directories
    MODELS_DIR = BASE_DIR / "models"
    MODELS_CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    MODELS_SERIALIZED_DIR = MODELS_DIR / "serialized"
    
    # Logging directories
    LOGS_DIR = BASE_DIR / "logs"
    LOGS_APP_DIR = LOGS_DIR / "app"
    LOGS_TRADING_DIR = LOGS_DIR / "trading"
    LOGS_MODELS_DIR = LOGS_DIR / "models"
    
    # Results directories
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_BACKTEST_DIR = RESULTS_DIR / "backtest"
    RESULTS_LIVE_DIR = RESULTS_DIR / "live"
    RESULTS_OPTIMIZATION_DIR = RESULTS_DIR / "optimization"
    
    # Web directories
    WEB_DIR = BASE_DIR / "web"
    WEB_STATIC_DIR = WEB_DIR / "static"
    WEB_TEMPLATES_DIR = WEB_DIR / "templates"
    
    # Scripts directory
    SCRIPTS_DIR = BASE_DIR / "scripts"
    
    # Tests directory
    TESTS_DIR = BASE_DIR / "tests"
    
    # Database directory
    DATABASE_DIR = BASE_DIR / "database"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.CONFIG_DIR, cls.SECRETS_DIR,
            cls.DATA_DIR, cls.DATA_RAW_DIR, cls.DATA_PROCESSED_DIR, cls.DATA_CACHE_DIR,
            cls.MODELS_DIR, cls.MODELS_CHECKPOINTS_DIR, cls.MODELS_SERIALIZED_DIR,
            cls.LOGS_DIR, cls.LOGS_APP_DIR, cls.LOGS_TRADING_DIR, cls.LOGS_MODELS_DIR,
            cls.RESULTS_DIR, cls.RESULTS_BACKTEST_DIR, cls.RESULTS_LIVE_DIR, cls.RESULTS_OPTIMIZATION_DIR,
            cls.WEB_DIR, cls.WEB_STATIC_DIR, cls.WEB_TEMPLATES_DIR,
            cls.SCRIPTS_DIR, cls.TESTS_DIR, cls.DATABASE_DIR
        ]
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)
            # Add __init__.py to make directories Python packages
            init_file = directory / "__init__.py"
            if not init_file.exists():
                init_file.touch()

# Create all directories
Paths.create_directories()

# ============ Environment Configuration ============
class Environment(str, Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

# Detect current environment
ENVIRONMENT = os.getenv("TRADING_ENV", Environment.DEVELOPMENT.value)

# ============ Application Constants ============
class AppConstants:
    """Application constants"""
    
    # Application metadata
    APP_NAME = "Bitcoin Trading AI"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Advanced AI-powered Bitcoin Trading System"
    
    # Default settings
    DEFAULT_SYMBOL = "BTC/USDT"
    DEFAULT_TIMEFRAME = "1h"
    DEFAULT_INITIAL_CAPITAL = 10000.0
    DEFAULT_COMMISSION = 0.001  # 0.1%
    DEFAULT_SLIPPAGE = 0.001   # 0.1%
    
    # Time constants
    ONE_MINUTE = 60
    FIVE_MINUTES = 300
    FIFTEEN_MINUTES = 900
    ONE_HOUR = 3600
    FOUR_HOURS = 14400
    ONE_DAY = 86400
    ONE_WEEK = 604800
    
    # Data constants
    MAX_DATA_POINTS = 1000000
    DATA_RETENTION_DAYS = 365
    CACHE_TTL = 3600  # 1 hour
    
    # Model constants
    DEFAULT_SEQUENCE_LENGTH = 60
    DEFAULT_PREDICTION_HORIZON = 24
    DEFAULT_TRAIN_TEST_SPLIT = 0.8
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_EPOCHS = 100
    DEFAULT_LEARNING_RATE = 0.001
    
    # Trading constants
    MAX_POSITION_SIZE = 0.15  # 15% of portfolio
    MIN_POSITION_SIZE = 0.01  # 1% of portfolio
    MAX_OPEN_POSITIONS = 5
    MIN_TRADE_AMOUNT = 10.0   # Minimum $10 per trade
    
    # Risk constants
    MAX_DAILY_LOSS = 0.02     # 2%
    MAX_DRAWDOWN = 0.15       # 15%
    VAR_CONFIDENCE = 0.95     # 95%
    
    # Performance thresholds
    MIN_SHARPE_RATIO = 1.0
    MAX_DRAWDOWN_THRESHOLD = 0.20
    MIN_WIN_RATE = 0.55
    MIN_PROFIT_FACTOR = 1.5
    
    # API constants
    API_TIMEOUT = 30
    API_RETRY_ATTEMPTS = 3
    API_RATE_LIMIT_MULTIPLIER = 0.8
    
    # Web constants
    DASHBOARD_PORT = 8050
    API_PORT = 8000
    METRICS_PORT = 9090
    WEBSOCKET_PORT = 8765

# ============ Exchange Configuration ============
class ExchangeSettings:
    """Exchange-specific settings"""
    
    # Supported exchanges
    SUPPORTED_EXCHANGES = [
        "binance",
        "coinbase",
        "kraken",
        "bitstamp",
        "bybit",
        "okx",
        "huobi",
        "gateio",
        "mexc",
        "bitfinex"
    ]
    
    # Exchange rate limits (requests per minute)
    RATE_LIMITS = {
        "binance": 1200,
        "coinbase": 100,
        "kraken": 60,
        "bitstamp": 600,
        "bybit": 120,
        "okx": 180,
        "huobi": 150,
        "gateio": 200,
        "mexc": 120,
        "bitfinex": 90
    }
    
    # Exchange timeouts (seconds)
    TIMEOUTS = {
        "binance": 30,
        "coinbase": 30,
        "kraken": 30,
        "bitstamp": 30,
        "bybit": 30,
        "okx": 30,
        "huobi": 30,
        "gateio": 30,
        "mexc": 30,
        "bitfinex": 30
    }
    
    # Exchange symbols mapping
    SYMBOL_MAPPING = {
        "binance": {
            "BTC/USDT": "BTCUSDT",
            "ETH/USDT": "ETHUSDT",
            "BNB/USDT": "BNBUSDT"
        },
        "coinbase": {
            "BTC/USDT": "BTC-USD",
            "ETH/USDT": "ETH-USD"
        },
        "kraken": {
            "BTC/USDT": "XBT/USD",
            "ETH/USDT": "ETH/USD"
        }
    }
    
    # Exchange minimum trade amounts
    MIN_TRADE_AMOUNTS = {
        "binance": {"BTC": 0.0001, "ETH": 0.001, "USDT": 10},
        "coinbase": {"BTC": 0.0001, "ETH": 0.001, "USD": 10},
        "kraken": {"BTC": 0.0001, "ETH": 0.001, "USD": 10}
    }
    
    # Exchange fees
    FEES = {
        "binance": {"maker": 0.001, "taker": 0.001},
        "coinbase": {"maker": 0.005, "taker": 0.005},
        "kraken": {"maker": 0.0016, "taker": 0.0026},
        "bitstamp": {"maker": 0.005, "taker": 0.005}
    }

# ============ Model Configuration ============
class ModelSettings:
    """Model-related settings"""
    
    # Available model types
    MODEL_TYPES = [
        "lstm", "transformer", "cnn_lstm", "attention", 
        "xgboost", "lightgbm", "catboost", "prophet",
        "ensemble", "deep_rl", "gru", "tcn"
    ]
    
    # Default model parameters
    DEFAULT_PARAMS = {
        "lstm": {
            "units": [128, 64, 32],
            "dropout": 0.3,
            "recurrent_dropout": 0.2,
            "bidirectional": True,
            "activation": "tanh",
            "recurrent_activation": "sigmoid"
        },
        "transformer": {
            "d_model": 64,
            "nhead": 8,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "activation": "gelu"
        },
        "xgboost": {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0
        },
        "lightgbm": {
            "n_estimators": 200,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1
        }
    }
    
    # Training parameters
    TRAINING_PARAMS = {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 20,
        "min_delta": 0.001,
        "validation_split": 0.2,
        "shuffle": True
    }
    
    # Feature engineering
    FEATURE_CONFIG = {
        "technical_indicators": [
            "sma", "ema", "wma", "hma", "vwap",
            "macd", "rsi", "stoch", "williams_r",
            "adx", "cci", "atr", "bollinger",
            "obv", "mfi", "roc", "momentum",
            "ichimoku", "parabolic_sar", "keltner"
        ],
        "statistical_features": [
            "returns", "log_returns", "volatility",
            "skewness", "kurtosis", "zscore",
            "quantiles", "autocorrelation", "hurst"
        ],
        "time_features": [
            "hour", "day_of_week", "day_of_month",
            "month", "quarter", "is_weekend",
            "is_month_end", "is_quarter_end"
        ]
    }
    
    # Sequence configuration
    SEQUENCE_CONFIG = {
        "min_length": 10,
        "max_length": 500,
        "default_length": 60,
        "step_size": 1,
        "overlap": 0
    }

# ============ Trading Configuration ============
class TradingSettings:
    """Trading-related settings"""
    
    # Available strategies
    STRATEGIES = [
        "ml_ensemble", "transformer", "lstm", "transformer_lstm",
        "cnn_lstm", "attention_lstm", "deep_rl", "mean_reversion",
        "trend_following", "breakout", "arbitrage", "market_making",
        "hedging", "scalping", "swing"
    ]
    
    # Position sizing methods
    POSITION_SIZING_METHODS = [
        "kelly", "fixed", "volatility", "optimal_f",
        "martingale", "anti_martingale", "risk_parity", "cvar"
    ]
    
    # Order types
    ORDER_TYPES = [
        "market", "limit", "stop", "stop_limit",
        "trailing_stop", "iceberg", "twap", "vwap"
    ]
    
    # Risk parameters
    RISK_PARAMS = {
        "max_position_risk": 0.02,      # 2% per position
        "max_portfolio_risk": 0.15,     # 15% total portfolio risk
        "max_correlation": 0.7,         # Maximum correlation between positions
        "max_leverage": 3.0,            # Maximum leverage
        "min_volume_ratio": 0.1,        # Minimum volume ratio for entry
        "max_slippage": 0.005,          # Maximum slippage tolerance
    }
    
    # Stop loss and take profit
    STOP_CONFIG = {
        "initial_stop_loss": 0.02,      # 2% initial stop loss
        "trailing_stop_distance": 0.01, # 1% trailing stop distance
        "take_profit_ratio": 2.0,       # Take profit = 2 * stop loss
        "break_even_at": 0.01,          # Move to breakeven at 1% profit
        "partial_exit_at": [0.5, 0.75]  # Partial exits at 50% and 75% of target
    }
    
    # Trading hours (UTC)
    TRADING_HOURS = {
        "monday": ["00:00", "23:59"],
        "tuesday": ["00:00", "23:59"],
        "wednesday": ["00:00", "23:59"],
        "thursday": ["00:00", "23:59"],
        "friday": ["00:00", "23:59"],
        "saturday": ["00:00", "23:59"],
        "sunday": ["00:00", "23:59"]
    }
    
    # Market holidays (YYYY-MM-DD)
    MARKET_HOLIDAYS = [
        "2024-01-01",  # New Year's Day
        "2024-12-25",  # Christmas
    ]

# ============ Data Configuration ============
class DataSettings:
    """Data-related settings"""
    
    # Data collection
    DATA_COLLECTION = {
        "realtime_enabled": True,
        "websocket_enabled": True,
        "historical_days": 365,
        "update_frequency": 60,  # seconds
        "max_retries": 3,
        "retry_delay": 5,        # seconds
        "timeout": 30,           # seconds
        "batch_size": 1000,
        "compress_data": True,
        "encrypt_data": False
    }
    
    # Data validation
    DATA_VALIDATION = {
        "min_data_quality": 0.95,
        "max_data_latency": 1000,  # milliseconds
        "max_missing_pct": 0.05,
        "max_outlier_pct": 0.01,
        "price_change_limit": 0.50,  # Maximum 50% price change
        "volume_spike_limit": 10.0,  # Maximum 10x volume spike
        "correlation_threshold": 0.95
    }
    
    # Feature engineering
    FEATURE_ENGINEERING = {
        "window_sizes": [5, 10, 20, 50, 100],
        "lag_periods": [1, 2, 3, 5, 10],
        "rolling_functions": ["mean", "std", "min", "max", "median"],
        "interaction_degree": 2,
        "polynomial_degree": 3,
        "pca_components": 20,
        "feature_selection_method": "mutual_info",
        "max_features": 100
    }
    
    # Data sources priority
    DATA_SOURCES_PRIORITY = [
        "binance", "coinbase", "kraken", "bitstamp",
        "bybit", "okx", "yfinance", "cryptocompare"
    ]

# ============ Risk Management Configuration ============
class RiskSettings:
    """Risk management settings"""
    
    # Risk metrics
    RISK_METRICS = [
        "value_at_risk",
        "expected_shortfall",
        "maximum_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "omega_ratio",
        "gain_loss_ratio",
        "tail_ratio",
        "information_ratio"
    ]
    
    # Circuit breakers
    CIRCUIT_BREAKERS = {
        "price_change_1min": 0.05,   # 5% in 1 minute
        "price_change_5min": 0.10,   # 10% in 5 minutes
        "price_change_1hour": 0.20,  # 20% in 1 hour
        "volume_spike": 10.0,        # 10x volume spike
        "volatility_spike": 5.0,     # 5x volatility spike
        "order_book_imbalance": 3.0, # 3x order book imbalance
        "liquidity_drop": 0.5,       # 50% liquidity drop
    }
    
    # Stress test scenarios
    STRESS_TEST_SCENARIOS = [
        "flash_crash_2010",
        "bitcoin_2017_crash",
        "covid_crash_2020",
        "luna_crash_2022",
        "exchange_hack",
        "liquidity_crisis",
        "regulatory_shock",
        "network_attack"
    ]
    
    # Risk limits
    RISK_LIMITS = {
        "max_daily_trades": 50,
        "max_consecutive_losses": 5,
        "max_daily_loss_pct": 0.05,
        "max_weekly_loss_pct": 0.10,
        "max_monthly_loss_pct": 0.20,
        "min_confidence_score": 0.70,
        "min_liquidity_score": 0.50,
        "max_position_duration": 30  # days
    }

# ============ Monitoring Configuration ============
class MonitoringSettings:
    """Monitoring and logging settings"""
    
    # Logging configuration
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file_app": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": str(Paths.LOGS_APP_DIR / "app.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10
            },
            "file_trading": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": str(Paths.LOGS_TRADING_DIR / "trading.log"),
                "maxBytes": 10485760,
                "backupCount": 10
            },
            "file_errors": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": str(Paths.LOGS_DIR / "errors.log"),
                "maxBytes": 5242880,  # 5MB
                "backupCount": 5
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "file_app"],
                "level": "INFO",
                "propagate": True
            },
            "trading": {
                "handlers": ["console", "file_trading"],
                "level": "INFO",
                "propagate": False
            },
            "models": {
                "handlers": ["console", "file_app"],
                "level": "DEBUG",
                "propagate": False
            },
            "data": {
                "handlers": ["console", "file_app"],
                "level": "INFO",
                "propagate": False
            },
            "risk": {
                "handlers": ["console", "file_trading"],
                "level": "WARNING",
                "propagate": False
            }
        }
    }
    
    # Alert configuration
    ALERT_CONFIG = {
        "channels": ["console", "email", "slack", "telegram"],
        "thresholds": {
            "error": 3,          # Alert after 3 errors
            "warning": 10,       # Alert after 10 warnings
            "drawdown": 0.05,    # Alert on 5% drawdown
            "loss_streak": 3,    # Alert after 3 consecutive losses
            "low_confidence": 0.6,  # Alert on low model confidence
            "high_slippage": 0.01,  # Alert on high slippage
            "api_error": 5,      # Alert after 5 API errors
            "data_latency": 5000  # Alert on 5-second data latency
        },
        "cooldown_periods": {
            "error": 300,        # 5 minutes
            "warning": 60,       # 1 minute
            "drawdown": 900,     # 15 minutes
            "loss_streak": 1800  # 30 minutes
        }
    }
    
    # Performance metrics
    PERFORMANCE_METRICS = [
        "total_return",
        "annual_return",
        "volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "omega_ratio",
        "win_rate",
        "profit_factor",
        "average_win",
        "average_loss",
        "expectancy",
        "risk_adjusted_return"
    ]
    
    # Dashboard configuration
    DASHBOARD_CONFIG = {
        "refresh_interval": 5,      # seconds
        "max_data_points": 1000,
        "theme": "dark",
        "charts_per_row": 2,
        "enable_live_updates": True,
        "enable_export": True,
        "enable_alerts": True
    }

# ============ Database Configuration ============
class DatabaseSettings:
    """Database settings"""
    
    # Database URLs
    DATABASE_URLS = {
        "development": f"sqlite:///{Paths.DATABASE_DIR}/trading_dev.db",
        "testing": f"sqlite:///{Paths.DATABASE_DIR}/trading_test.db",
        "staging": "postgresql://user:pass@localhost:5432/trading_staging",
        "production": "postgresql://user:pass@localhost:5432/trading_prod"
    }
    
    # Redis URLs
    REDIS_URLS = {
        "development": "redis://localhost:6379/0",
        "testing": "redis://localhost:6379/1",
        "staging": "redis://staging-redis:6379/0",
        "production": "redis://production-redis:6379/0"
    }
    
    # Cache configuration
    CACHE_CONFIG = {
        "type": "redis",  # redis, memory, disk
        "ttl": 3600,      # Time to live in seconds
        "max_size": 10000,  # Maximum cache size
        "compression": True,
        "encryption": False
    }
    
    # Connection pools
    CONNECTION_POOLS = {
        "database": {
            "pool_size": 20,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 3600
        },
        "redis": {
            "max_connections": 50,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            "retry_on_timeout": True
        }
    }

# ============ API Configuration ============
class APISettings:
    """API settings"""
    
    # REST API configuration
    REST_API = {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": ENVIRONMENT == Environment.DEVELOPMENT,
        "reload": ENVIRONMENT == Environment.DEVELOPMENT,
        "workers": 4,
        "timeout": 30,
        "max_requests": 1000,
        "max_requests_jitter": 100,
        "cors_origins": ["*"],
        "rate_limit": "100/minute",
        "enable_docs": True,
        "enable_metrics": True
    }
    
    # WebSocket configuration
    WEBSOCKET = {
        "host": "0.0.0.0",
        "port": 8765,
        "ping_interval": 20,
        "ping_timeout": 30,
        "max_message_size": 10485760,  # 10MB
        "max_queue_size": 100,
        "compression": True
    }
    
    # Endpoints
    ENDPOINTS = {
        "trading": "/api/v1/trading",
        "data": "/api/v1/data",
        "models": "/api/v1/models",
        "risk": "/api/v1/risk",
        "portfolio": "/api/v1/portfolio",
        "backtest": "/api/v1/backtest",
        "monitoring": "/api/v1/monitoring",
        "system": "/api/v1/system"
    }
    
    # Authentication
    AUTHENTICATION = {
        "enabled": True,
        "jwt_secret": os.getenv("JWT_SECRET", "your-secret-key-change-in-production"),
        "jwt_algorithm": "HS256",
        "jwt_expiration": 3600,  # 1 hour
        "api_key_header": "X-API-Key",
        "rate_limit_by_ip": True,
        "require_https": ENVIRONMENT == Environment.PRODUCTION
    }

# ============ Security Configuration ============
class SecuritySettings:
    """Security settings"""
    
    # Encryption
    ENCRYPTION = {
        "algorithm": "AES-256-GCM",
        "key_derivation": "PBKDF2",
        "iterations": 100000,
        "salt_length": 16,
        "nonce_length": 12,
        "tag_length": 16
    }
    
    # API keys storage
    API_KEYS = {
        "encrypt_at_rest": True,
        "key_rotation_days": 90,
        "max_keys_per_user": 5,
        "key_format": "hex",  # hex, base64, base58
        "key_length": 32
    }
    
    # Network security
    NETWORK_SECURITY = {
        "enable_firewall": True,
        "rate_limiting": True,
        "ip_whitelist": [],
        "ip_blacklist": [],
        "require_vpn": False,
        "enable_ssl": ENVIRONMENT == Environment.PRODUCTION,
        "ssl_cert_path": "",
        "ssl_key_path": ""
    }
    
    # Data protection
    DATA_PROTECTION = {
        "encrypt_sensitive_data": True,
        "mask_api_keys": True,
        "log_sanitization": True,
        "data_retention_days": 90,
        "secure_deletion": True
    }

# ============ Performance Configuration ============
class PerformanceSettings:
    """Performance optimization settings"""
    
    # Parallel processing
    PARALLEL_PROCESSING = {
        "enabled": True,
        "max_workers": 4,
        "thread_pool_size": 10,
        "process_pool_size": 4,
        "use_gpu": True,
        "gpu_memory_fraction": 0.8
    }
    
    # Memory management
    MEMORY_MANAGEMENT = {
        "max_memory_usage": 0.8,  # 80% of system memory
        "cache_size": 10000,
        "garbage_collection": True,
        "gc_threshold": (700, 10, 10),
        "memory_monitoring": True,
        "memory_warning_threshold": 0.7
    }
    
    # Optimization
    OPTIMIZATION = {
        "use_jit": True,
        "use_mkl": True,
        "use_cudnn": True,
        "tensorflow_memory_growth": True,
        "pytorch_benchmark": False,
        "numpy_threads": 4
    }
    
    # Batch processing
    BATCH_PROCESSING = {
        "data_batch_size": 1000,
        "training_batch_size": 32,
        "prediction_batch_size": 64,
        "inference_batch_size": 128,
        "stream_buffer_size": 10000
    }

# ============ Environment-specific Settings ============
class EnvironmentSettings:
    """Environment-specific settings"""
    
    @staticmethod
    def get_development_settings():
        """Development environment settings"""
        return {
            "debug": True,
            "testing": False,
            "log_level": "DEBUG",
            "database_url": DatabaseSettings.DATABASE_URLS["development"],
            "redis_url": DatabaseSettings.REDIS_URLS["development"],
            "enable_hot_reload": True,
            "enable_profiling": True,
            "mock_external_apis": True,
            "use_testnet": True,
            "simulate_latency": False
        }
    
    @staticmethod
    def get_testing_settings():
        """Testing environment settings"""
        return {
            "debug": True,
            "testing": True,
            "log_level": "DEBUG",
            "database_url": DatabaseSettings.DATABASE_URLS["testing"],
            "redis_url": DatabaseSettings.REDIS_URLS["testing"],
            "enable_hot_reload": False,
            "enable_profiling": False,
            "mock_external_apis": True,
            "use_testnet": True,
            "simulate_latency": False
        }
    
    @staticmethod
    def get_staging_settings():
        """Staging environment settings"""
        return {
            "debug": False,
            "testing": False,
            "log_level": "INFO",
            "database_url": DatabaseSettings.DATABASE_URLS["staging"],
            "redis_url": DatabaseSettings.REDIS_URLS["staging"],
            "enable_hot_reload": False,
            "enable_profiling": False,
            "mock_external_apis": False,
            "use_testnet": True,
            "simulate_latency": True
        }
    
    @staticmethod
    def get_production_settings():
        """Production environment settings"""
        return {
            "debug": False,
            "testing": False,
            "log_level": "WARNING",
            "database_url": DatabaseSettings.DATABASE_URLS["production"],
            "redis_url": DatabaseSettings.REDIS_URLS["production"],
            "enable_hot_reload": False,
            "enable_profiling": False,
            "mock_external_apis": False,
            "use_testnet": False,
            "simulate_latency": False
        }
    
    @classmethod
    def get_current_settings(cls):
        """Get settings for current environment"""
        env_mapping = {
            Environment.DEVELOPMENT: cls.get_development_settings,
            Environment.TESTING: cls.get_testing_settings,
            Environment.STAGING: cls.get_staging_settings,
            Environment.PRODUCTION: cls.get_production_settings
        }
        
        current_env = Environment(ENVIRONMENT)
        return env_mapping[current_env]()

# ============ Feature Flags ============
class FeatureFlags:
    """Feature flags for gradual rollout"""
    
    # Model features
    MODELS = {
        "enable_transformer": True,
        "enable_lstm": True,
        "enable_ensemble": True,
        "enable_deep_rl": False,  # Experimental
        "enable_online_learning": True,
        "enable_transfer_learning": True,
        "enable_meta_learning": False  # Experimental
    }
    
    # Trading features
    TRADING = {
        "enable_short_selling": True,
        "enable_margin_trading": False,
        "enable_options_trading": False,
        "enable_futures_trading": False,
        "enable_arbitrage": True,
        "enable_market_making": False,
        "enable_hedging": True
    }
    
    # Data features
    DATA = {
        "enable_sentiment_analysis": True,
        "enable_onchain_analysis": True,
        "enable_social_metrics": True,
        "enable_news_analysis": True,
        "enable_alternative_data": False,
        "enable_real_time_streaming": True
    }
    
    # Risk features
    RISK = {
        "enable_stress_testing": True,
        "enable_monte_carlo": True,
        "enable_scenario_analysis": True,
        "enable_correlation_analysis": True,
        "enable_liquidity_analysis": True,
        "enable_volatility_forecasting": True
    }

# ============ Timezone Configuration ============
TIMEZONE = "UTC"
MARKET_OPEN_TIME = "00:00"
MARKET_CLOSE_TIME = "23:59"

# ============ Helper Functions ============
def get_all_settings() -> Dict[str, Any]:
    """Get all settings as a dictionary"""
    settings_classes = [
        AppConstants,
        ExchangeSettings,
        ModelSettings,
        TradingSettings,
        DataSettings,
        RiskSettings,
        MonitoringSettings,
        DatabaseSettings,
        APISettings,
        SecuritySettings,
        PerformanceSettings,
        FeatureFlags
    ]
    
    all_settings = {}
    for settings_class in settings_classes:
        # Get all uppercase attributes from the class
        attrs = {k: v for k, v in settings_class.__dict__.items() 
                if not k.startswith('_') and k.isupper()}
        all_settings[settings_class.__name__] = attrs
    
    # Add environment settings
    all_settings['EnvironmentSettings'] = EnvironmentSettings.get_current_settings()
    
    return all_settings

def print_settings_summary():
    """Print a summary of all settings"""
    print(f"\n{'='*60}")
    print(f"{AppConstants.APP_NAME} v{AppConstants.APP_VERSION}")
    print(f"{'='*60}")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Base Directory: {Paths.BASE_DIR}")
    print(f"Default Symbol: {AppConstants.DEFAULT_SYMBOL}")
    print(f"Default Timeframe: {AppConstants.DEFAULT_TIMEFRAME}")
    print(f"Initial Capital: ${AppConstants.DEFAULT_INITIAL_CAPITAL:,.2f}")
    print(f"{'='*60}\n")

# ============ Export Important Constants ============
# Export frequently used constants
BASE_DIR = Paths.BASE_DIR
CONFIG_DIR = Paths.CONFIG_DIR
DATA_DIR = Paths.DATA_DIR
MODELS_DIR = Paths.MODELS_DIR
LOGS_DIR = Paths.LOGS_DIR
RESULTS_DIR = Paths.RESULTS_DIR

# Export environment
ENVIRONMENT = ENVIRONMENT

# Export default values
DEFAULT_SYMBOL = AppConstants.DEFAULT_SYMBOL
DEFAULT_TIMEFRAME = AppConstants.DEFAULT_TIMEFRAME
DEFAULT_CAPITAL = AppConstants.DEFAULT_INITIAL_CAPITAL

# Export time constants
ONE_MINUTE = AppConstants.ONE_MINUTE
ONE_HOUR = AppConstants.ONE_HOUR
ONE_DAY = AppConstants.ONE_DAY

# Export model constants
DEFAULT_SEQUENCE_LENGTH = AppConstants.DEFAULT_SEQUENCE_LENGTH
DEFAULT_PREDICTION_HORIZON = AppConstants.DEFAULT_PREDICTION_HORIZON

# Export risk constants
MAX_POSITION_SIZE = AppConstants.MAX_POSITION_SIZE
MAX_DAILY_LOSS = AppConstants.MAX_DAILY_LOSS

# Print settings summary on import
if __name__ != "__main__":
    print_settings_summary()