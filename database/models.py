"""
Database models for the Bitcoin Trading AI application.
Defines SQLAlchemy ORM models for storing trading data, models, and results.
"""

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, 
    ForeignKey, Text, JSON, BigInteger, DECIMAL
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class TradeSide(enum.Enum):
    """Enum for trade direction"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(enum.Enum):
    """Enum for order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Timeframe(enum.Enum):
    """Enum for trading timeframes"""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"


class MarketData(Base):
    """Model for storing OHLCV market data"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)  # e.g., "BTC/USDT"
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # From Timeframe enum
    open = Column(DECIMAL(20, 8), nullable=False)
    high = Column(DECIMAL(20, 8), nullable=False)
    low = Column(DECIMAL(20, 8), nullable=False)
    close = Column(DECIMAL(20, 8), nullable=False)
    volume = Column(DECIMAL(30, 8), nullable=False)
    quote_volume = Column(DECIMAL(30, 8))
    trade_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Index for efficient querying
    __table_args__ = {
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }


class Trade(Base):
    """Model for storing executed trades"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(100), unique=True, nullable=False, index=True)  # Exchange trade ID
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # From TradeSide enum
    price = Column(DECIMAL(20, 8), nullable=False)
    quantity = Column(DECIMAL(20, 8), nullable=False)
    commission = Column(DECIMAL(20, 8), default=0)
    commission_asset = Column(String(10))
    timestamp = Column(DateTime, nullable=False, index=True)
    is_maker = Column(Boolean, default=False)
    exchange_order_id = Column(String(100), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to order
    order_id = Column(Integer, ForeignKey('orders.id'))
    order = relationship("Order", back_populates="trades")
    
    __table_args__ = {
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }


class Order(Base):
    """Model for storing trading orders"""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(100), unique=True, nullable=False, index=True)  # Exchange order ID
    client_order_id = Column(String(100), index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # From TradeSide enum
    order_type = Column(String(20), nullable=False)  # "limit", "market", "stop_loss", etc.
    status = Column(String(20), nullable=False)  # From OrderStatus enum
    price = Column(DECIMAL(20, 8))
    stop_price = Column(DECIMAL(20, 8))
    quantity = Column(DECIMAL(20, 8), nullable=False)
    executed_quantity = Column(DECIMAL(20, 8), default=0)
    iceberg_quantity = Column(DECIMAL(20, 8))
    time_in_force = Column(String(10))
    created_time = Column(DateTime, nullable=False, index=True)
    updated_time = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trades = relationship("Trade", back_populates="order")
    trading_session_id = Column(Integer, ForeignKey('trading_sessions.id'))
    trading_session = relationship("TradingSession", back_populates="orders")
    
    __table_args__ = {
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }


class TradingSession(Base):
    """Model for tracking trading sessions/strategies"""
    __tablename__ = "trading_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    initial_capital = Column(DECIMAL(20, 8), nullable=False)
    current_capital = Column(DECIMAL(20, 8), nullable=False)
    status = Column(String(20), nullable=False)  # "active", "paused", "stopped"
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime)
    parameters = Column(JSON)  # Strategy parameters as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    orders = relationship("Order", back_populates="trading_session")
    performance_metrics = relationship("PerformanceMetrics", back_populates="trading_session")
    
    __table_args__ = {
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }


class PerformanceMetrics(Base):
    """Model for storing trading performance metrics"""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trading_session_id = Column(Integer, ForeignKey('trading_sessions.id'), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    total_pnl = Column(DECIMAL(20, 8), nullable=False)
    realized_pnl = Column(DECIMAL(20, 8), nullable=False)
    unrealized_pnl = Column(DECIMAL(20, 8), nullable=False)
    total_fees = Column(DECIMAL(20, 8), nullable=False)
    win_rate = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=False)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    winning_trades = Column(Integer, nullable=False)
    losing_trades = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    trading_session = relationship("TradingSession", back_populates="performance_metrics")
    
    __table_args__ = {
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }


class ModelTraining(Base):
    """Model for tracking AI model training sessions"""
    __tablename__ = "model_trainings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    training_id = Column(String(100), unique=True, nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # "transformer", "lstm", "cnn_lstm", etc.
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    training_start = Column(DateTime, nullable=False)
    training_end = Column(DateTime)
    status = Column(String(20), nullable=False)  # "training", "completed", "failed"
    hyperparameters = Column(JSON)  # Model hyperparameters
    training_metrics = Column(JSON)  # Training loss, accuracy, etc.
    validation_metrics = Column(JSON)  # Validation metrics
    test_metrics = Column(JSON)  # Test metrics
    model_path = Column(String(500))  # Path to saved model
    feature_columns = Column(JSON)  # List of feature columns used
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = {
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }


class ModelPrediction(Base):
    """Model for storing model predictions"""
    __tablename__ = "model_predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(100), unique=True, nullable=False, index=True)
    model_training_id = Column(Integer, ForeignKey('model_trainings.id'), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    prediction = Column(JSON)  # Prediction output (could be probabilities, values, etc.)
    confidence = Column(Float)  # Prediction confidence score
    actual_value = Column(DECIMAL(20, 8))  # Actual value if available
    error = Column(Float)  # Prediction error if actual value is available
    features = Column(JSON)  # Input features used for prediction
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    model_training = relationship("ModelTraining")
    
    __table_args__ = {
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }


class Signal(Base):
    """Model for storing trading signals generated by models/strategies"""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(100), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    signal_type = Column(String(50), nullable=False)  # "buy", "sell", "hold", "strong_buy", etc.
    signal_strength = Column(Float, nullable=False)  # 0.0 to 1.0 or -1.0 to 1.0
    source = Column(String(50), nullable=False)  # "model", "strategy", "hybrid"
    source_id = Column(String(100), nullable=False)  # ID of model or strategy
    confidence = Column(Float)  # Confidence level
    price_at_signal = Column(DECIMAL(20, 8), nullable=False)
    target_price = Column(DECIMAL(20, 8))
    stop_loss = Column(DECIMAL(20, 8))
    timeframe = Column(String(10), nullable=False)
    signal_metadata = Column(JSON)  # Additional signal metadata (renamed to avoid SQLAlchemy reserved name)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = {
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }


class BacktestResult(Base):
    """Model for storing backtesting results"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_id = Column(String(100), unique=True, nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(DECIMAL(20, 8), nullable=False)
    final_capital = Column(DECIMAL(20, 8), nullable=False)
    total_return = Column(Float, nullable=False)
    annual_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    avg_trade = Column(Float)
    parameters = Column(JSON)  # Strategy parameters used
    trades = Column(JSON)  # Serialized trades data
    equity_curve = Column(JSON)  # Equity curve data points
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = {
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }


class Alert(Base):
    """Model for storing system alerts"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(100), unique=True, nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)  # "price", "volume", "technical", "risk", "system"
    severity = Column(String(20), nullable=False)  # "info", "warning", "error", "critical"
    symbol = Column(String(20), index=True)
    message = Column(Text, nullable=False)
    data = Column(JSON)  # Alert data/metadata
    is_read = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    triggered_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = {
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }


class SystemLog(Base):
    """Model for storing system logs"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    level = Column(String(20), nullable=False)  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    logger = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    module = Column(String(100))
    function = Column(String(100))
    line_number = Column(Integer)
    exception = Column(Text)  # Exception traceback if any
    extra_data = Column(JSON)  # Additional log data
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = {
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci',
        'mysql_engine': 'InnoDB'
    }


# Additional utility functions
def create_all_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(engine)


def drop_all_tables(engine):
    """Drop all database tables"""
    Base.metadata.drop_all(engine)