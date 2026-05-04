"""
Machine Learning Trading Strategies for Bitcoin Trading AI System
Implements various ML-based trading strategies using different models and approaches
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import project modules
try:
    from config.config_manager import ConfigManager
    from core.neural_networks.transformer_model import TransformerModel
    from core.neural_networks.lstm_attention import LSTMAttentionModel
    from core.neural_networks.cnn_lstm import CNNLSTMModel
    from core.neural_networks.ensemble_model import EnsembleModel
    from core.neural_networks.reinforcement_learning import RLModel
    from core.data_processing.feature_engineer import FeatureEngineer
    from core.trading.signal_generator import SignalGenerator
    from core.trading.position_sizer import PositionSizer
    from core.risk_management.risk_analyzer import RiskAnalyzer
    from core.monitoring.performance_tracker import PerformanceTracker
    from core.utils.logger import setup_logger
    from core.utils.cache import CacheManager
except ImportError:
    # For testing purposes
    ConfigManager = type('ConfigManager', (), {})
    TransformerModel = type('TransformerModel', (), {})
    LSTMAttentionModel = type('LSTMAttentionModel', (), {})
    CNNLSTMModel = type('CNNLSTMModel', (), {})
    EnsembleModel = type('EnsembleModel', (), {})
    RLModel = type('RLModel', (), {})
    FeatureEngineer = type('FeatureEngineer', (), {})
    SignalGenerator = type('SignalGenerator', (), {})
    PositionSizer = type('PositionSizer', (), {})
    RiskAnalyzer = type('RiskAnalyzer', (), {})
    PerformanceTracker = type('PerformanceTracker', (), {})
    setup_logger = lambda name: logging.getLogger(name)
    CacheManager = type('CacheManager', (), {})

# Initialize logger
logger = setup_logger(__name__)

# Strategy Enums
class MLStrategyType(Enum):
    """Enum for ML strategy types"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    PATTERN_RECOGNITION = "pattern_recognition"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"

class SignalType(Enum):
    """Enum for signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    HEDGE = "hedge"

class ConfidenceLevel(Enum):
    """Enum for confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# Data Classes
@dataclass
class TradingSignal:
    """Data class for trading signals"""
    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    timestamp: datetime
    strategy_name: str
    model_name: str
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "strategy_name": self.strategy_name,
            "model_name": self.model_name,
            "features": self.features,
            "metadata": self.metadata
        }

@dataclass
class StrategyConfig:
    """Data class for strategy configuration"""
    name: str
    strategy_type: MLStrategyType
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    enabled: bool = True
    model_name: Optional[str] = None
    lookback_period: int = 100
    confidence_threshold: float = 0.7
    risk_per_trade: float = 1.0
    max_position_size: float = 0.1
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyPerformance:
    """Data class for strategy performance"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_streak: int = 0
    last_trade_time: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)

# Base Strategy Class
class BaseMLStrategy(ABC):
    """Base class for all ML trading strategies"""
    
    def __init__(self, config_manager: ConfigManager, strategy_type: MLStrategyType):
        self.config_manager = config_manager
        self.strategy_type = strategy_type
        self.name = f"{strategy_type.value}_strategy"
        self.config = None
        self.initialized = False
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(config_manager)
        self.signal_generator = SignalGenerator(config_manager)
        self.position_sizer = PositionSizer(config_manager)
        self.risk_analyzer = RiskAnalyzer(config_manager)
        self.performance_tracker = PerformanceTracker()
        self.cache_manager = CacheManager(config_manager)
        
        # Models
        self.models: Dict[str, Any] = {}
        self.active_model: Optional[Any] = None
        
        # State
        self.performance = StrategyPerformance(self.name)
        self.signals_history: List[TradingSignal] = []
        self.trades_history: List[Dict] = []
        
        logger.info(f"Initialized base ML strategy: {self.name}")
    
    async def initialize(self, config: StrategyConfig) -> bool:
        """Initialize strategy with configuration"""
        try:
            self.config = config
            self.name = config.name
            
            # Load or create models
            await self.load_models()
            
            # Initialize performance tracking
            await self.load_performance_history()
            
            self.initialized = True
            logger.info(f"Strategy '{self.name}' initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize strategy '{self.name}': {e}")
            return False
    
    @abstractmethod
    async def load_models(self):
        """Load or create ML models for the strategy"""
        pass
    
    @abstractmethod
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate trading signal based on market data"""
        pass
    
    async def analyze_market(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Analyze market data for the strategy"""
        try:
            # Get market data (this would come from data collector)
            # For now, generate mock data
            dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)
            prices = np.random.normal(45000, 1000, limit).cumsum() + 45000
            volumes = np.random.randint(100, 1000, limit)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices - np.random.uniform(50, 200, limit),
                'high': prices + np.random.uniform(50, 200, limit),
                'low': prices - np.random.uniform(50, 200, limit),
                'close': prices,
                'volume': volumes
            })
            
            # Calculate features
            df = await self.feature_engineer.calculate_features(df, timeframe)
            
            return df
        
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            raise
    
    async def calculate_position_size(self, signal: TradingSignal, account_balance: float) -> float:
        """Calculate position size based on signal and risk parameters"""
        try:
            position_size = await self.position_sizer.calculate_position_size(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                confidence=signal.confidence,
                current_price=signal.price,
                account_balance=account_balance,
                risk_per_trade=self.config.risk_per_trade if self.config else 1.0,
                max_position_size=self.config.max_position_size if self.config else 0.1
            )
            
            return position_size
        
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def validate_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """Validate trading signal with risk management rules"""
        try:
            # Check confidence threshold
            if signal.confidence < (self.config.confidence_threshold if self.config else 0.7):
                return False, f"Confidence below threshold: {signal.confidence:.2f}"
            
            # Check risk limits
            risk_check = await self.risk_analyzer.check_signal_risk(signal.to_dict())
            if not risk_check.get("allowed", False):
                return False, risk_check.get("reason", "Risk check failed")
            
            # Check market conditions
            market_conditions = await self.analyze_market_conditions(signal.symbol)
            if not market_conditions.get("favorable", True):
                return False, market_conditions.get("reason", "Unfavorable market conditions")
            
            return True, "Signal validated"
        
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False, str(e)
    
    async def analyze_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Analyze current market conditions"""
        # This would check volatility, trend, volume, etc.
        return {
            "favorable": True,
            "volatility": "normal",
            "trend": "neutral",
            "volume": "adequate",
            "liquidity": "high"
        }
    
    async def update_performance(self, trade_result: Dict):
        """Update strategy performance metrics"""
        try:
            self.trades_history.append(trade_result)
            
            # Update performance metrics
            self.performance.total_trades += 1
            
            if trade_result.get("pnl", 0) > 0:
                self.performance.winning_trades += 1
                self.performance.current_streak = max(0, self.performance.current_streak) + 1
            else:
                self.performance.losing_trades += 1
                self.performance.current_streak = min(0, self.performance.current_streak) - 1
            
            self.performance.total_pnl += trade_result.get("pnl", 0)
            self.performance.win_rate = self.performance.winning_trades / self.performance.total_trades
            self.performance.last_trade_time = datetime.now()
            
            # Calculate additional metrics
            await self.calculate_performance_metrics()
            
            # Save performance
            await self.save_performance()
            
            logger.info(f"Updated performance for {self.name}: {self.performance}")
        
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    async def calculate_performance_metrics(self):
        """Calculate advanced performance metrics"""
        if len(self.trades_history) < 2:
            return
        
        # Calculate profit factor
        total_profits = sum(t.get("pnl", 0) for t in self.trades_history if t.get("pnl", 0) > 0)
        total_losses = abs(sum(t.get("pnl", 0) for t in self.trades_history if t.get("pnl", 0) < 0))
        
        if total_losses > 0:
            self.performance.profit_factor = total_profits / total_losses
        
        # Calculate Sharpe ratio (simplified)
        returns = [t.get("pnl_percentage", 0) for t in self.trades_history if "pnl_percentage" in t]
        if returns and np.std(returns) > 0:
            self.performance.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Calculate max drawdown
        cumulative_returns = np.cumsum([t.get("pnl", 0) for t in self.trades_history])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        if len(drawdowns) > 0:
            self.performance.max_drawdown = np.max(drawdowns)
    
    async def save_performance(self):
        """Save performance data to storage"""
        # This would save to database or file
        pass
    
    async def load_performance_history(self):
        """Load performance history from storage"""
        # This would load from database or file
        pass
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "strategy_name": self.name,
            "performance": {
                "total_trades": self.performance.total_trades,
                "winning_trades": self.performance.winning_trades,
                "losing_trades": self.performance.losing_trades,
                "win_rate": self.performance.win_rate,
                "total_pnl": self.performance.total_pnl,
                "profit_factor": self.performance.profit_factor,
                "sharpe_ratio": self.performance.sharpe_ratio,
                "max_drawdown": self.performance.max_drawdown,
                "current_streak": self.performance.current_streak
            },
            "recent_trades": self.trades_history[-10:] if self.trades_history else [],
            "signals_generated": len(self.signals_history),
            "last_updated": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check strategy health"""
        return {
            "healthy": self.initialized,
            "initialized": self.initialized,
            "models_loaded": len(self.models) > 0,
            "active_model": self.active_model is not None,
            "performance_tracking": len(self.trades_history) > 0,
            "last_signal_time": self.signals_history[-1].timestamp.isoformat() if self.signals_history else None,
            "issues": []
        }

# Concrete Strategy Implementations

class MomentumMLStrategy(BaseMLStrategy):
    """Momentum-based ML trading strategy"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, MLStrategyType.MOMENTUM)
        self.momentum_periods = [10, 20, 50]
        self.acceleration_threshold = 0.5
    
    async def load_models(self):
        """Load momentum prediction models"""
        try:
            # Load transformer model for momentum prediction
            self.models["transformer"] = TransformerModel(self.config_manager)
            await self.models["transformer"].load_model("momentum_transformer")
            
            # Load LSTM model for trend analysis
            self.models["lstm"] = LSTMAttentionModel(self.config_manager)
            await self.models["lstm"].load_model("momentum_lstm")
            
            self.active_model = self.models["transformer"]
            logger.info(f"Loaded momentum models for strategy: {self.name}")
        
        except Exception as e:
            logger.error(f"Error loading momentum models: {e}")
            # Create new models if loading fails
            await self.create_models()
    
    async def create_models(self):
        """Create new momentum models"""
        try:
            # Create and train transformer model
            self.models["transformer"] = TransformerModel(self.config_manager)
            # Training would happen here
            
            self.active_model = self.models["transformer"]
            logger.info(f"Created new momentum models for strategy: {self.name}")
        
        except Exception as e:
            logger.error(f"Error creating momentum models: {e}")
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate momentum-based trading signal"""
        try:
            if self.active_model is None:
                logger.error("No active model for momentum strategy")
                return None
            
            # Calculate momentum indicators
            momentum_features = await self.calculate_momentum_features(market_data)
            
            # Prepare input for model
            model_input = self.prepare_model_input(market_data, momentum_features)
            
            # Get prediction from model
            prediction = await self.active_model.predict(model_input)
            
            # Interpret prediction
            signal_type, confidence = self.interpret_momentum_prediction(prediction, momentum_features)
            
            if signal_type == SignalType.HOLD:
                return None
            
            # Create signal
            signal = TradingSignal(
                symbol=self.config.symbol if self.config else "BTCUSDT",
                signal_type=signal_type,
                confidence=confidence,
                price=market_data['close'].iloc[-1],
                timestamp=datetime.now(),
                strategy_name=self.name,
                model_name=self.active_model.__class__.__name__,
                features=momentum_features,
                metadata={
                    "prediction": prediction,
                    "momentum_periods": self.momentum_periods,
                    "acceleration": momentum_features.get('acceleration', 0)
                }
            )
            
            self.signals_history.append(signal)
            return signal
        
        except Exception as e:
            logger.error(f"Error generating momentum signal: {e}")
            return None
    
    async def calculate_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum features"""
        features = {}
        
        for period in self.momentum_periods:
            if len(df) >= period:
                # Price momentum
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period]
                features[f'momentum_{period}'] = price_change
                
                # Volume momentum
                if 'volume' in df.columns:
                    volume_change = (df['volume'].iloc[-1] - df['volume'].iloc[-period]) / df['volume'].iloc[-period]
                    features[f'volume_momentum_{period}'] = volume_change
        
        # Acceleration (rate of change of momentum)
        if 'momentum_10' in features and 'momentum_20' in features:
            features['acceleration'] = features['momentum_10'] - features['momentum_20']
        
        # RSI-based momentum
        if 'rsi' in df.columns:
            features['rsi_momentum'] = df['rsi'].iloc[-1] - df['rsi'].iloc[-5] if len(df) >= 5 else 0
        
        return features
    
    def prepare_model_input(self, df: pd.DataFrame, features: Dict) -> pd.DataFrame:
        """Prepare input data for model"""
        # Combine price data with features
        input_data = df[['close', 'volume']].copy()
        
        for feature_name, feature_value in features.items():
            input_data[feature_name] = feature_value
        
        # Add technical indicators if available
        tech_indicators = ['rsi', 'macd', 'bollinger_upper', 'bollinger_lower']
        for indicator in tech_indicators:
            if indicator in df.columns:
                input_data[indicator] = df[indicator]
        
        return input_data.tail(100)  # Last 100 data points
    
    def interpret_momentum_prediction(self, prediction: Dict, features: Dict) -> Tuple[SignalType, float]:
        """Interpret model prediction into trading signal"""
        try:
            # Extract prediction values
            direction = prediction.get('direction', 0)
            confidence = prediction.get('confidence', 0.5)
            
            # Get momentum values
            momentum_short = features.get('momentum_10', 0)
            momentum_long = features.get('momentum_50', 0)
            acceleration = features.get('acceleration', 0)
            
            # Determine signal based on momentum and prediction
            if direction > 0.5 and momentum_short > 0 and acceleration > 0:
                # Strong bullish momentum
                signal_type = SignalType.BUY
                confidence = min(0.95, confidence + 0.2)
            
            elif direction < -0.5 and momentum_short < 0 and acceleration < 0:
                # Strong bearish momentum
                signal_type = SignalType.SELL
                confidence = min(0.95, confidence + 0.2)
            
            elif direction > 0.3 and momentum_short > momentum_long:
                # Moderate bullish
                signal_type = SignalType.BUY
                confidence = confidence
            
            elif direction < -0.3 and momentum_short < momentum_long:
                # Moderate bearish
                signal_type = SignalType.SELL
                confidence = confidence
            
            else:
                # No clear momentum
                signal_type = SignalType.HOLD
                confidence = 0.0
            
            return signal_type, confidence
        
        except Exception as e:
            logger.error(f"Error interpreting momentum prediction: {e}")
            return SignalType.HOLD, 0.0

class MeanReversionMLStrategy(BaseMLStrategy):
    """Mean reversion ML trading strategy"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, MLStrategyType.MEAN_REVERSION)
        self.reversion_periods = [20, 50, 100]
        self.deviation_threshold = 2.0  # Standard deviations
    
    async def load_models(self):
        """Load mean reversion models"""
        try:
            # Load LSTM model for mean reversion detection
            self.models["lstm"] = LSTMAttentionModel(self.config_manager)
            await self.models["lstm"].load_model("mean_reversion_lstm")
            
            # Load ensemble model for confirmation
            self.models["ensemble"] = EnsembleModel(self.config_manager)
            await self.models["ensemble"].load_model("mean_reversion_ensemble")
            
            self.active_model = self.models["ensemble"]
            logger.info(f"Loaded mean reversion models for strategy: {self.name}")
        
        except Exception as e:
            logger.error(f"Error loading mean reversion models: {e}")
            await self.create_models()
    
    async def create_models(self):
        """Create new mean reversion models"""
        try:
            self.models["lstm"] = LSTMAttentionModel(self.config_manager)
            self.models["ensemble"] = EnsembleModel(self.config_manager)
            self.active_model = self.models["ensemble"]
            logger.info(f"Created new mean reversion models for strategy: {self.name}")
        
        except Exception as e:
            logger.error(f"Error creating mean reversion models: {e}")
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate mean reversion trading signal"""
        try:
            if self.active_model is None:
                logger.error("No active model for mean reversion strategy")
                return None
            
            # Calculate mean reversion features
            reversion_features = await self.calculate_reversion_features(market_data)
            
            # Check for extreme deviations
            deviations = self.detect_deviations(market_data, reversion_features)
            
            if not deviations["extreme_deviation"]:
                return None  # No extreme deviation detected
            
            # Prepare model input
            model_input = self.prepare_reversion_input(market_data, reversion_features, deviations)
            
            # Get prediction
            prediction = await self.active_model.predict(model_input)
            
            # Interpret prediction
            signal_type, confidence = self.interpret_reversion_prediction(prediction, deviations)
            
            if signal_type == SignalType.HOLD:
                return None
            
            # Create signal
            signal = TradingSignal(
                symbol=self.config.symbol if self.config else "BTCUSDT",
                signal_type=signal_type,
                confidence=confidence,
                price=market_data['close'].iloc[-1],
                timestamp=datetime.now(),
                strategy_name=self.name,
                model_name=self.active_model.__class__.__name__,
                features=reversion_features,
                metadata={
                    "prediction": prediction,
                    "deviations": deviations,
                    "reversion_potential": reversion_features.get('reversion_potential', 0)
                }
            )
            
            self.signals_history.append(signal)
            return signal
        
        except Exception as e:
            logger.error(f"Error generating mean reversion signal: {e}")
            return None
    
    async def calculate_reversion_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate mean reversion features"""
        features = {}
        
        for period in self.reversion_periods:
            if len(df) >= period:
                # Calculate moving averages
                ma = df['close'].rolling(window=period).mean().iloc[-1]
                current_price = df['close'].iloc[-1]
                
                # Deviation from mean
                deviation = (current_price - ma) / ma
                features[f'deviation_{period}'] = deviation
                
                # Z-score
                std = df['close'].rolling(window=period).std().iloc[-1]
                if std > 0:
                    z_score = (current_price - ma) / std
                    features[f'zscore_{period}'] = z_score
        
        # Bollinger Bands deviation
        if 'bollinger_upper' in df.columns and 'bollinger_lower' in df.columns:
            bb_width = df['bollinger_upper'].iloc[-1] - df['bollinger_lower'].iloc[-1]
            bb_mid = (df['bollinger_upper'].iloc[-1] + df['bollinger_lower'].iloc[-1]) / 2
            if bb_width > 0:
                bb_deviation = (df['close'].iloc[-1] - bb_mid) / (bb_width / 2)
                features['bb_deviation'] = bb_deviation
        
        # RSI-based reversion
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            features['rsi_reversion'] = abs(rsi - 50) / 50  # Normalized distance from 50
        
        # Calculate reversion potential
        if 'zscore_20' in features:
            features['reversion_potential'] = abs(features['zscore_20'])
        
        return features
    
    def detect_deviations(self, df: pd.DataFrame, features: Dict) -> Dict[str, Any]:
        """Detect extreme deviations from mean"""
        deviations = {
            "extreme_deviation": False,
            "direction": None,  # 'overbought' or 'oversold'
            "strength": 0.0,
            "periods": []
        }
        
        # Check each period for extreme deviations
        for period in self.reversion_periods:
            zscore_key = f'zscore_{period}'
            if zscore_key in features:
                zscore = features[zscore_key]
                
                if abs(zscore) >= self.deviation_threshold:
                    deviations["extreme_deviation"] = True
                    deviations["periods"].append({
                        "period": period,
                        "zscore": zscore,
                        "extreme": True
                    })
                    
                    if zscore > 0:
                        deviations["direction"] = "overbought"
                    else:
                        deviations["direction"] = "oversold"
                    
                    deviations["strength"] = max(deviations["strength"], abs(zscore))
        
        # Check RSI extremes
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if rsi >= 70:
                deviations["extreme_deviation"] = True
                deviations["direction"] = "overbought"
                deviations["strength"] = max(deviations["strength"], (rsi - 50) / 20)
            elif rsi <= 30:
                deviations["extreme_deviation"] = True
                deviations["direction"] = "oversold"
                deviations["strength"] = max(deviations["strength"], (50 - rsi) / 20)
        
        return deviations
    
    def prepare_reversion_input(self, df: pd.DataFrame, features: Dict, deviations: Dict) -> pd.DataFrame:
        """Prepare input data for mean reversion model"""
        input_data = df[['close', 'volume']].copy()
        
        # Add features
        for feature_name, feature_value in features.items():
            input_data[feature_name] = feature_value
        
        # Add deviation information
        input_data['extreme_deviation'] = deviations["extreme_deviation"]
        input_data['deviation_direction'] = 1 if deviations["direction"] == "overbought" else -1 if deviations["direction"] == "oversold" else 0
        input_data['deviation_strength'] = deviations["strength"]
        
        # Add technical indicators
        tech_indicators = ['rsi', 'macd', 'stoch_k', 'stoch_d']
        for indicator in tech_indicators:
            if indicator in df.columns:
                input_data[indicator] = df[indicator]
        
        return input_data.tail(100)
    
    def interpret_reversion_prediction(self, prediction: Dict, deviations: Dict) -> Tuple[SignalType, float]:
        """Interpret model prediction for mean reversion"""
        try:
            direction = prediction.get('direction', 0)
            confidence = prediction.get('confidence', 0.5)
            reversion_strength = prediction.get('reversion_strength', 0)
            
            if not deviations["extreme_deviation"]:
                return SignalType.HOLD, 0.0
            
            # Determine signal based on deviation direction
            if deviations["direction"] == "overbought" and direction < -0.3:
                # Price is overbought, expecting reversion down
                signal_type = SignalType.SELL
                confidence = min(0.95, confidence + reversion_strength)
            
            elif deviations["direction"] == "oversold" and direction > 0.3:
                # Price is oversold, expecting reversion up
                signal_type = SignalType.BUY
                confidence = min(0.95, confidence + reversion_strength)
            
            else:
                # No clear reversion signal
                signal_type = SignalType.HOLD
                confidence = 0.0
            
            return signal_type, confidence
        
        except Exception as e:
            logger.error(f"Error interpreting reversion prediction: {e}")
            return SignalType.HOLD, 0.0

class DeepLearningStrategy(BaseMLStrategy):
    """Deep learning-based trading strategy using multiple architectures"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, MLStrategyType.DEEP_LEARNING)
        self.architectures = ['transformer', 'lstm', 'cnn_lstm']
        self.ensemble_weights = {'transformer': 0.4, 'lstm': 0.3, 'cnn_lstm': 0.3}
    
    async def load_models(self):
        """Load deep learning models"""
        try:
            # Load multiple architectures
            for arch in self.architectures:
                if arch == 'transformer':
                    self.models[arch] = TransformerModel(self.config_manager)
                elif arch == 'lstm':
                    self.models[arch] = LSTMAttentionModel(self.config_manager)
                elif arch == 'cnn_lstm':
                    self.models[arch] = CNNLSTMModel(self.config_manager)
                
                model_name = f"deep_learning_{arch}"
                await self.models[arch].load_model(model_name)
            
            # Load ensemble model
            self.models["ensemble"] = EnsembleModel(self.config_manager)
            await self.models["ensemble"].load_model("deep_learning_ensemble")
            
            self.active_model = self.models["ensemble"]
            logger.info(f"Loaded deep learning models for strategy: {self.name}")
        
        except Exception as e:
            logger.error(f"Error loading deep learning models: {e}")
            await self.create_models()
    
    async def create_models(self):
        """Create new deep learning models"""
        try:
            for arch in self.architectures:
                if arch == 'transformer':
                    self.models[arch] = TransformerModel(self.config_manager)
                elif arch == 'lstm':
                    self.models[arch] = LSTMAttentionModel(self.config_manager)
                elif arch == 'cnn_lstm':
                    self.models[arch] = CNNLSTMModel(self.config_manager)
            
            self.models["ensemble"] = EnsembleModel(self.config_manager)
            self.active_model = self.models["ensemble"]
            logger.info(f"Created new deep learning models for strategy: {self.name}")
        
        except Exception as e:
            logger.error(f"Error creating deep learning models: {e}")
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate signal using deep learning ensemble"""
        try:
            if self.active_model is None:
                logger.error("No active model for deep learning strategy")
                return None
            
            # Calculate advanced features
            dl_features = await self.calculate_dl_features(market_data)
            
            # Get predictions from all models
            predictions = {}
            for arch, model in self.models.items():
                if arch != "ensemble":
                    model_input = self.prepare_arch_input(arch, market_data, dl_features)
                    prediction = await model.predict(model_input)
                    predictions[arch] = prediction
            
            # Get ensemble prediction
            ensemble_input = self.prepare_ensemble_input(predictions, dl_features)
            ensemble_prediction = await self.active_model.predict(ensemble_input)
            
            # Interpret ensemble prediction
            signal_type, confidence = self.interpret_dl_prediction(ensemble_prediction, predictions)
            
            if signal_type == SignalType.HOLD:
                return None
            
            # Create signal
            signal = TradingSignal(
                symbol=self.config.symbol if self.config else "BTCUSDT",
                signal_type=signal_type,
                confidence=confidence,
                price=market_data['close'].iloc[-1],
                timestamp=datetime.now(),
                strategy_name=self.name,
                model_name="DeepLearningEnsemble",
                features=dl_features,
                metadata={
                    "ensemble_prediction": ensemble_prediction,
                    "individual_predictions": predictions,
                    "ensemble_weights": self.ensemble_weights
                }
            )
            
            self.signals_history.append(signal)
            return signal
        
        except Exception as e:
            logger.error(f"Error generating deep learning signal: {e}")
            return None
    
    async def calculate_dl_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced features for deep learning"""
        features = {}
        
        # Price action features
        features['returns_1'] = df['close'].pct_change(1).iloc[-1] if len(df) > 1 else 0
        features['returns_5'] = df['close'].pct_change(5).iloc[-1] if len(df) > 5 else 0
        features['returns_20'] = df['close'].pct_change(20).iloc[-1] if len(df) > 20 else 0
        
        # Volatility features
        if len(df) >= 20:
            returns = df['close'].pct_change().dropna()
            features['volatility_20'] = returns.tail(20).std()
            features['realized_volatility'] = np.sqrt((returns ** 2).sum())
        
        # Volume features
        if 'volume' in df.columns:
            features['volume_ratio'] = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 1
            features['volume_trend'] = df['volume'].iloc[-1] > df['volume'].rolling(5).mean().iloc[-1] if len(df) >= 5 else False
        
        # Market regime features
        features['trend_strength'] = self.calculate_trend_strength(df)
        features['market_regime'] = self.detect_market_regime(df)
        
        # Correlation features (would require multiple symbols in real implementation)
        features['correlation_strength'] = 0.5  # Placeholder
        
        return features
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength"""
        if len(df) < 50:
            return 0.0
        
        # ADX-like trend strength calculation
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate smoothed values
        window = 14
        atr = tr.rolling(window).mean()
        plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime"""
        if len(df) < 100:
            return "unknown"
        
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        mean_return = returns.mean()
        
        if volatility > returns.std() * 1.5:
            return "high_volatility"
        elif volatility < returns.std() * 0.5:
            return "low_volatility"
        elif mean_return > 0:
            return "bullish"
        elif mean_return < 0:
            return "bearish"
        else:
            return "ranging"
    
    def prepare_arch_input(self, architecture: str, df: pd.DataFrame, features: Dict) -> pd.DataFrame:
        """Prepare input for specific architecture"""
        input_data = df.copy()
        
        # Add engineered features
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float)):
                input_data[feature_name] = feature_value
        
        # Architecture-specific preprocessing
        if architecture == 'cnn_lstm':
            # Add image-like features for CNN
            input_data = self.add_cnn_features(input_data)
        
        return input_data.tail(100)
    
    def add_cnn_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features suitable for CNN processing"""
        # Create image-like representation of price action
        # This could be candlestick patterns, price matrices, etc.
        
        # Add price matrix features
        price_matrix = self.create_price_matrix(df)
        for i in range(min(10, price_matrix.shape[0])):
            for j in range(min(10, price_matrix.shape[1])):
                df[f'price_matrix_{i}_{j}'] = price_matrix[i, j] if i < price_matrix.shape[0] and j < price_matrix.shape[1] else 0
        
        return df
    
    def create_price_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Create price matrix for CNN input"""
        # Simple implementation: normalize prices and create matrix
        prices = df['close'].values[-100:]  # Last 100 prices
        if len(prices) < 100:
            prices = np.pad(prices, (0, 100 - len(prices)), 'edge')
        
        # Normalize
        prices_norm = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
        
        # Reshape to 10x10 matrix
        matrix = prices_norm[:100].reshape(10, 10)
        return matrix
    
    def prepare_ensemble_input(self, predictions: Dict[str, Dict], features: Dict) -> pd.DataFrame:
        """Prepare input for ensemble model"""
        input_data = {}
        
        # Add predictions from individual models
        for arch, prediction in predictions.items():
            if 'direction' in prediction:
                input_data[f'{arch}_direction'] = prediction['direction']
            if 'confidence' in prediction:
                input_data[f'{arch}_confidence'] = prediction['confidence']
        
        # Add features
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float)):
                input_data[feature_name] = feature_value
        
        # Convert to DataFrame (single row)
        return pd.DataFrame([input_data])
    
    def interpret_dl_prediction(self, ensemble_prediction: Dict, individual_predictions: Dict) -> Tuple[SignalType, float]:
        """Interpret deep learning ensemble prediction"""
        try:
            direction = ensemble_prediction.get('direction', 0)
            confidence = ensemble_prediction.get('confidence', 0.5)
            
            # Check agreement between models
            agreement_score = self.calculate_model_agreement(individual_predictions)
            
            # Adjust confidence based on agreement
            adjusted_confidence = confidence * (0.5 + 0.5 * agreement_score)
            
            # Determine signal
            if direction > 0.4 and adjusted_confidence > 0.6:
                signal_type = SignalType.BUY
                final_confidence = adjusted_confidence
            
            elif direction < -0.4 and adjusted_confidence > 0.6:
                signal_type = SignalType.SELL
                final_confidence = adjusted_confidence
            
            else:
                signal_type = SignalType.HOLD
                final_confidence = 0.0
            
            return signal_type, final_confidence
        
        except Exception as e:
            logger.error(f"Error interpreting DL prediction: {e}")
            return SignalType.HOLD, 0.0
    
    def calculate_model_agreement(self, predictions: Dict[str, Dict]) -> float:
        """Calculate agreement score between models"""
        directions = []
        confidences = []
        
        for arch, prediction in predictions.items():
            if 'direction' in prediction:
                directions.append(prediction['direction'])
            if 'confidence' in prediction:
                confidences.append(prediction['confidence'])
        
        if not directions:
            return 0.0
        
        # Calculate variance of directions
        direction_variance = np.var(directions)
        
        # Agreement is inverse of variance (normalized)
        agreement = 1.0 / (1.0 + direction_variance)
        
        return float(agreement)

class ReinforcementLearningStrategy(BaseMLStrategy):
    """Reinforcement learning trading strategy"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, MLStrategyType.REINFORCEMENT_LEARNING)
        self.state_size = 10
        self.action_space = ['buy', 'sell', 'hold', 'close']
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.95   # Discount factor
    
    async def load_models(self):
        """Load reinforcement learning model"""
        try:
            self.models["rl"] = RLModel(self.config_manager)
            await self.models["rl"].load_model("trading_rl")
            
            self.active_model = self.models["rl"]
            logger.info(f"Loaded RL model for strategy: {self.name}")
        
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            await self.create_models()
    
    async def create_models(self):
        """Create new RL model"""
        try:
            self.models["rl"] = RLModel(self.config_manager)
            self.active_model = self.models["rl"]
            logger.info(f"Created new RL model for strategy: {self.name}")
        
        except Exception as e:
            logger.error(f"Error creating RL model: {e}")
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate signal using reinforcement learning"""
        try:
            if self.active_model is None:
                logger.error("No active model for RL strategy")
                return None
            
            # Get current state
            state = await self.get_state(market_data)
            
            # Get action from RL agent
            action, q_values = await self.active_model.get_action(state, self.epsilon)
            
            # Map action to signal
            signal_type = self.map_action_to_signal(action)
            
            if signal_type == SignalType.HOLD:
                return None
            
            # Calculate confidence from Q-values
            confidence = self.calculate_confidence(q_values, action)
            
            # Create signal
            signal = TradingSignal(
                symbol=self.config.symbol if self.config else "BTCUSDT",
                signal_type=signal_type,
                confidence=confidence,
                price=market_data['close'].iloc[-1],
                timestamp=datetime.now(),
                strategy_name=self.name,
                model_name="ReinforcementLearning",
                features={"state": state.tolist() if hasattr(state, 'tolist') else state},
                metadata={
                    "action": action,
                    "q_values": q_values.tolist() if hasattr(q_values, 'tolist') else q_values,
                    "epsilon": self.epsilon
                }
            )
            
            self.signals_history.append(signal)
            return signal
        
        except Exception as e:
            logger.error(f"Error generating RL signal: {e}")
            return None
    
    async def get_state(self, df: pd.DataFrame) -> np.ndarray:
        """Get current state for RL agent"""
        # State includes price features, technical indicators, and portfolio info
        state_features = []
        
        # Price features
        if len(df) >= 10:
            prices = df['close'].values[-10:]
            returns = np.diff(prices) / prices[:-1]
            
            state_features.extend([
                prices[-1] / prices[0] - 1,  # Recent return
                np.mean(returns),            # Average return
                np.std(returns),             # Volatility
                prices[-1] > np.mean(prices) # Above average
            ])
        
        # Technical indicators
        if 'rsi' in df.columns and len(df) >= 10:
            state_features.append(df['rsi'].iloc[-1] / 100)  # Normalized RSI
        
        if 'macd' in df.columns and len(df) >= 10:
            state_features.append(df['macd'].iloc[-1])
        
        # Volume indicator
        if 'volume' in df.columns and len(df) >= 10:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(10).mean().iloc[-1]
            state_features.append(current_volume / avg_volume if avg_volume > 0 else 1)
        
        # Pad or truncate to state_size
        if len(state_features) < self.state_size:
            state_features.extend([0] * (self.state_size - len(state_features)))
        else:
            state_features = state_features[:self.state_size]
        
        return np.array(state_features, dtype=np.float32)
    
    def map_action_to_signal(self, action: int) -> SignalType:
        """Map RL action to trading signal"""
        action_map = {
            0: SignalType.BUY,
            1: SignalType.SELL,
            2: SignalType.HOLD,
            3: SignalType.CLOSE
        }
        return action_map.get(action, SignalType.HOLD)
    
    def calculate_confidence(self, q_values: np.ndarray, action: int) -> float:
        """Calculate confidence from Q-values"""
        if not isinstance(q_values, np.ndarray) or len(q_values) == 0:
            return 0.5
        
        # Normalize Q-value for selected action
        q_value = q_values[action]
        q_min = np.min(q_values)
        q_max = np.max(q_values)
        
        if q_max - q_min > 0:
            confidence = (q_value - q_min) / (q_max - q_min)
        else:
            confidence = 0.5
        
        return float(confidence)
    
    async def update_model(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Update RL model with experience"""
        try:
            if self.active_model and hasattr(self.active_model, 'update'):
                await self.active_model.update(state, action, reward, next_state, done)
                logger.debug(f"Updated RL model with reward: {reward}")
        
        except Exception as e:
            logger.error(f"Error updating RL model: {e}")
    
    async def calculate_reward(self, trade_result: Dict) -> float:
        """Calculate reward for RL agent"""
        try:
            pnl = trade_result.get("pnl", 0)
            risk_taken = trade_result.get("risk", 0)
            duration = trade_result.get("duration", 0)
            
            # Base reward on PnL
            reward = pnl
            
            # Adjust for risk (risk-adjusted return)
            if risk_taken > 0:
                reward = pnl / risk_taken
            
            # Penalize long durations (want efficient trades)
            if duration > 3600:  # More than 1 hour
                reward *= 0.9
            
            # Bonus for consecutive wins
            if self.performance.current_streak > 0:
                reward *= (1 + 0.1 * self.performance.current_streak)
            
            return reward
        
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

# Strategy Factory
class MLStrategyFactory:
    """Factory for creating ML trading strategies"""
    
    @staticmethod
    def create_strategy(strategy_type: Union[str, MLStrategyType], config_manager: ConfigManager) -> BaseMLStrategy:
        """Create strategy instance based on type"""
        if isinstance(strategy_type, str):
            try:
                strategy_type = MLStrategyType(strategy_type.lower())
            except ValueError:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        if strategy_type == MLStrategyType.MOMENTUM:
            return MomentumMLStrategy(config_manager)
        
        elif strategy_type == MLStrategyType.MEAN_REVERSION:
            return MeanReversionMLStrategy(config_manager)
        
        elif strategy_type == MLStrategyType.DEEP_LEARNING:
            return DeepLearningStrategy(config_manager)
        
        elif strategy_type == MLStrategyType.REINFORCEMENT_LEARNING:
            return ReinforcementLearningStrategy(config_manager)
        
        elif strategy_type == MLStrategyType.ENSEMBLE:
            # Ensemble combines multiple strategies
            return EnsembleMLStrategy(config_manager)
        
        elif strategy_type == MLStrategyType.TRANSFORMER:
            return TransformerMLStrategy(config_manager)
        
        elif strategy_type == MLStrategyType.HYBRID:
            return HybridMLStrategy(config_manager)
        
        else:
            raise ValueError(f"Strategy type not implemented: {strategy_type}")

# Additional Strategy Classes (simplified implementations)

class EnsembleMLStrategy(BaseMLStrategy):
    """Ensemble of multiple ML strategies"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, MLStrategyType.ENSEMBLE)
        self.sub_strategies: List[BaseMLStrategy] = []
        self.strategy_weights: Dict[str, float] = {}
    
    async def load_models(self):
        """Load models for all sub-strategies"""
        # Implementation would load multiple strategies
        pass
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate signal by combining multiple strategies"""
        # Implementation would ensemble signals from sub-strategies
        pass

class TransformerMLStrategy(BaseMLStrategy):
    """Transformer-based trading strategy"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, MLStrategyType.TRANSFORMER)
    
    async def load_models(self):
        """Load transformer models"""
        # Implementation would load transformer models
        pass
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate signal using transformer"""
        # Implementation would use transformer for prediction
        pass

class HybridMLStrategy(BaseMLStrategy):
    """Hybrid strategy combining ML and traditional techniques"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, MLStrategyType.HYBRID)
    
    async def load_models(self):
        """Load hybrid models"""
        # Implementation would load multiple model types
        pass
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate signal using hybrid approach"""
        # Implementation would combine ML and traditional signals
        pass

# Strategy Manager
class MLStrategyManager:
    """Manages multiple ML trading strategies"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.strategies: Dict[str, BaseMLStrategy] = {}
        self.factory = MLStrategyFactory()
        self.logger = setup_logger(__name__)
    
    async def create_strategy(self, config: StrategyConfig) -> bool:
        """Create and initialize a new strategy"""
        try:
            # Check if strategy already exists
            if config.name in self.strategies:
                self.logger.warning(f"Strategy '{config.name}' already exists")
                return False
            
            # Create strategy instance
            strategy = self.factory.create_strategy(config.strategy_type, self.config_manager)
            
            # Initialize strategy
            success = await strategy.initialize(config)
            
            if success:
                self.strategies[config.name] = strategy
                self.logger.info(f"Created strategy '{config.name}' ({config.strategy_type.value})")
                return True
            else:
                self.logger.error(f"Failed to initialize strategy '{config.name}'")
                return False
        
        except Exception as e:
            self.logger.error(f"Error creating strategy '{config.name}': {e}")
            return False
    
    async def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            self.logger.info(f"Removed strategy '{strategy_name}'")
            return True
        return False
    
    async def get_strategy(self, strategy_name: str) -> Optional[BaseMLStrategy]:
        """Get strategy by name"""
        return self.strategies.get(strategy_name)
    
    async def get_all_strategies(self) -> Dict[str, BaseMLStrategy]:
        """Get all strategies"""
        return self.strategies.copy()
    
    async def generate_signals(self, symbol: str, timeframe: str) -> List[TradingSignal]:
        """Generate signals from all active strategies"""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                if strategy.config and strategy.config.enabled:
                    # Analyze market
                    market_data = await strategy.analyze_market(symbol, timeframe)
                    
                    # Generate signal
                    signal = await strategy.generate_signal(market_data)
                    
                    if signal:
                        signals.append(signal)
                
            except Exception as e:
                self.logger.error(f"Error generating signal for strategy '{strategy_name}': {e}")
        
        return signals
    
    async def get_performance_reports(self) -> Dict[str, Dict]:
        """Get performance reports for all strategies"""
        reports = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                report = await strategy.get_performance_report()
                reports[strategy_name] = report
            except Exception as e:
                self.logger.error(f"Error getting performance for '{strategy_name}': {e}")
                reports[strategy_name] = {"error": str(e)}
        
        return reports
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all strategies"""
        health_status = {
            "total_strategies": len(self.strategies),
            "active_strategies": 0,
            "healthy_strategies": 0,
            "strategies": {}
        }
        
        for strategy_name, strategy in self.strategies.items():
            try:
                health = await strategy.health_check()
                health_status["strategies"][strategy_name] = health
                
                if health.get("healthy", False):
                    health_status["healthy_strategies"] += 1
                
                if strategy.config and strategy.config.enabled:
                    health_status["active_strategies"] += 1
            
            except Exception as e:
                health_status["strategies"][strategy_name] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        health_status["overall_health"] = (
            health_status["healthy_strategies"] / health_status["total_strategies"]
            if health_status["total_strategies"] > 0 else 0
        )
        
        return health_status

# Example usage
async def example_usage():
    """Example of how to use the ML strategies"""
    # Create config manager
    config_manager = ConfigManager()
    
    # Create strategy manager
    strategy_manager = MLStrategyManager(config_manager)
    
    # Create momentum strategy config
    momentum_config = StrategyConfig(
        name="btc_momentum_v1",
        strategy_type=MLStrategyType.MOMENTUM,
        symbol="BTCUSDT",
        timeframe="1h",
        confidence_threshold=0.65,
        risk_per_trade=1.0,
        parameters={
            "momentum_periods": [10, 20, 50],
            "acceleration_threshold": 0.3
        }
    )
    
    # Create and initialize strategy
    await strategy_manager.create_strategy(momentum_config)
    
    # Generate signals
    signals = await strategy_manager.generate_signals("BTCUSDT", "1h")
    
    for signal in signals:
        print(f"Signal: {signal.signal_type.value} with confidence {signal.confidence:.2f}")
    
    # Get performance reports
    reports = await strategy_manager.get_performance_reports()
    print(f"Performance reports: {reports}")
    
    # Health check
    health = await strategy_manager.health_check()
    print(f"Health status: {health}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())