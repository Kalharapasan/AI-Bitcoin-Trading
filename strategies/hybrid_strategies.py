"""
Hybrid Trading Strategies for Bitcoin Trading AI System
Combines ML predictions with technical analysis for enhanced trading decisions
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
import warnings
warnings.filterwarnings('ignore')

# Import project modules
try:
    from config.config_manager import ConfigManager
    from core.neural_networks.transformer_model import TransformerModel
    from core.neural_networks.lstm_attention import LSTMAttentionModel
    from core.neural_networks.ensemble_model import EnsembleModel
    from core.data_processing.feature_engineer import FeatureEngineer
    from core.trading.signal_generator import SignalGenerator
    from core.trading.position_sizer import PositionSizer
    from core.risk_management.risk_analyzer import RiskAnalyzer
    from core.monitoring.performance_tracker import PerformanceTracker
    from core.utils.logger import setup_logger
    from core.utils.cache import CacheManager
    from strategies.ml_strategies import MLStrategyType, TradingSignal, SignalType, StrategyConfig, BaseMLStrategy
    from strategies.technical_strategies import TechnicalStrategyType, BaseTechnicalStrategy, TechnicalAnalysis
except ImportError:
    # For testing purposes
    ConfigManager = type('ConfigManager', (), {})
    TransformerModel = type('TransformerModel', (), {})
    LSTMAttentionModel = type('LSTMAttentionModel', (), {})
    EnsembleModel = type('EnsembleModel', (), {})
    FeatureEngineer = type('FeatureEngineer', (), {})
    SignalGenerator = type('SignalGenerator', (), {})
    PositionSizer = type('PositionSizer', (), {})
    RiskAnalyzer = type('RiskAnalyzer', (), {})
    PerformanceTracker = type('PerformanceTracker', (), {})
    setup_logger = lambda name: logging.getLogger(name)
    CacheManager = type('CacheManager', (), {})
    MLStrategyType = Enum('MLStrategyType', ['MOMENTUM', 'MEAN_REVERSION', 'DEEP_LEARNING', 'REINFORCEMENT_LEARNING'])
    TradingSignal = type('TradingSignal', (), {})
    SignalType = Enum('SignalType', ['BUY', 'SELL', 'HOLD', 'CLOSE', 'HEDGE'])
    StrategyConfig = type('StrategyConfig', (), {})
    BaseMLStrategy = type('BaseMLStrategy', (ABC,), {})
    TechnicalStrategyType = Enum('TechnicalStrategyType', ['RSI_STRATEGY', 'MACD_STRATEGY', 'BOLLINGER_BANDS', 'ICHIMOKU'])
    BaseTechnicalStrategy = type('BaseTechnicalStrategy', (ABC,), {})
    TechnicalAnalysis = type('TechnicalAnalysis', (), {})

# Initialize logger
logger = setup_logger(__name__)

# Hybrid Strategy Enums
class HybridStrategyType(Enum):
    """Enum for hybrid strategy types"""
    ML_TECHNICAL_FUSION = "ml_technical_fusion"
    ENSEMBLE_FUSION = "ensemble_fusion"
    ADAPTIVE_WEIGHTING = "adaptive_weighting"
    REINFORCEMENT_ENHANCED = "reinforcement_enhanced"
    TRANSFORMER_TECHNICAL = "transformer_technical"
    MULTI_MODAL = "multi_modal"
    CONTEXT_AWARE = "context_aware"
    MARKET_REGIME_ADAPTIVE = "market_regime_adaptive"

class FusionMethod(Enum):
    """Enum for fusion methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"
    META_LEARNING = "meta_learning"
    BAYESIAN = "bayesian"
    NEURAL_FUSION = "neural_fusion"

class ConfidenceCalibration(Enum):
    """Enum for confidence calibration methods"""
    HISTORICAL_ACCURACY = "historical_accuracy"
    MARKET_CONDITIONS = "market_conditions"
    MODEL_CERTAINTY = "model_certainty"
    ENSEMBLE_VARIANCE = "ensemble_variance"
    ADAPTIVE_LEARNING = "adaptive_learning"

# Data Classes
@dataclass
class ComponentSignal:
    """Data class for component strategy signals"""
    strategy_type: Union[MLStrategyType, TechnicalStrategyType]
    strategy_name: str
    signal_type: SignalType
    confidence: float
    features: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class FusionWeights:
    """Data class for fusion weights"""
    ml_weight: float = 0.5
    technical_weight: float = 0.5
    component_weights: Dict[str, float] = field(default_factory=dict)
    dynamic_weights: bool = True
    last_updated: Optional[datetime] = None
    
    def get_total_weight(self) -> float:
        """Get total weight sum"""
        total = self.ml_weight + self.technical_weight
        total += sum(self.component_weights.values())
        return total
    
    def normalize(self):
        """Normalize weights to sum to 1"""
        total = self.get_total_weight()
        if total > 0:
            self.ml_weight /= total
            self.technical_weight /= total
            for key in self.component_weights:
                self.component_weights[key] /= total

@dataclass
class MarketContext:
    """Data class for market context"""
    regime: str  # trending, ranging, volatile, news_driven
    volatility: float
    trend_strength: float
    volume_profile: str  # high, normal, low
    liquidity: str  # high, normal, low
    sentiment: str  # bullish, bearish, neutral
    time_of_day: str
    economic_events: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "regime": self.regime,
            "volatility": self.volatility,
            "trend_strength": self.trend_strength,
            "volume_profile": self.volume_profile,
            "liquidity": self.liquidity,
            "sentiment": self.sentiment,
            "time_of_day": self.time_of_day,
            "economic_events": self.economic_events
        }

@dataclass
class FusionResult:
    """Data class for fusion results"""
    final_signal: SignalType
    confidence: float
    component_signals: List[ComponentSignal]
    fusion_weights: FusionWeights
    market_context: MarketContext
    fusion_method: FusionMethod
    metadata: Dict[str, Any] = field(default_factory=dict)

# Base Hybrid Strategy Class
class BaseHybridStrategy(ABC):
    """Base class for all hybrid trading strategies"""
    
    def __init__(self, config_manager: ConfigManager, strategy_type: HybridStrategyType):
        self.config_manager = config_manager
        self.strategy_type = strategy_type
        self.name = f"{strategy_type.value}_hybrid_strategy"
        self.config = None
        self.initialized = False
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(config_manager)
        self.signal_generator = SignalGenerator(config_manager)
        self.position_sizer = PositionSizer(config_manager)
        self.risk_analyzer = RiskAnalyzer(config_manager)
        self.performance_tracker = PerformanceTracker()
        self.cache_manager = CacheManager(config_manager)
        
        # Component strategies
        self.ml_strategies: Dict[str, BaseMLStrategy] = {}
        self.technical_strategies: Dict[str, BaseTechnicalStrategy] = {}
        
        # Fusion configuration
        self.fusion_method = FusionMethod.WEIGHTED_AVERAGE
        self.fusion_weights = FusionWeights()
        self.confidence_calibration = ConfidenceCalibration.ADAPTIVE_LEARNING
        
        # State
        self.signals_history: List[TradingSignal] = []
        self.component_signals_history: List[ComponentSignal] = []
        self.fusion_results_history: List[FusionResult] = []
        self.trades_history: List[Dict] = []
        
        # Performance tracking
        self.component_performance: Dict[str, Dict] = {}
        self.fusion_performance: Dict[str, float] = {}
        
        # Market context
        self.market_context_history: List[MarketContext] = []
        
        logger.info(f"Initialized base hybrid strategy: {self.name}")
    
    async def initialize(self, config: StrategyConfig) -> bool:
        """Initialize strategy with configuration"""
        try:
            self.config = config
            self.name = config.name
            
            # Load component strategies
            await self.load_component_strategies()
            
            # Initialize fusion weights
            await self.initialize_fusion_weights()
            
            # Load historical performance
            await self.load_performance_history()
            
            self.initialized = True
            logger.info(f"Hybrid strategy '{self.name}' initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize hybrid strategy '{self.name}': {e}")
            return False
    
    @abstractmethod
    async def load_component_strategies(self):
        """Load component ML and technical strategies"""
        pass
    
    async def initialize_fusion_weights(self):
        """Initialize fusion weights"""
        # Default equal weights
        num_ml = len(self.ml_strategies)
        num_technical = len(self.technical_strategies)
        total = num_ml + num_technical
        
        if total > 0:
            self.fusion_weights.ml_weight = num_ml / total
            self.fusion_weights.technical_weight = num_technical / total
            
            # Initialize component weights
            for name in self.ml_strategies:
                self.fusion_weights.component_weights[name] = 1.0 / total
            
            for name in self.technical_strategies:
                self.fusion_weights.component_weights[name] = 1.0 / total
            
            self.fusion_weights.normalize()
        
        self.fusion_weights.last_updated = datetime.now()
    
    async def load_performance_history(self):
        """Load historical performance data"""
        # This would load from database or file
        pass
    
    async def analyze_market(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Analyze market data for all component strategies"""
        try:
            # Get market data (this would come from data collector)
            # For now, generate mock data
            dates = pd.date_range(end=datetime.now(), periods=100, freq=timeframe)
            prices = np.random.normal(45000, 1000, 100).cumsum() + 45000
            volumes = np.random.randint(100, 1000, 100)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices - np.random.uniform(50, 200, 100),
                'high': prices + np.random.uniform(50, 200, 100),
                'low': prices - np.random.uniform(50, 200, 100),
                'close': prices,
                'volume': volumes
            })
            
            # Calculate features
            df = await self.feature_engineer.calculate_features(df, timeframe)
            
            return df
        
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            raise
    
    async def analyze_market_context(self, market_data: pd.DataFrame) -> MarketContext:
        """Analyze current market context"""
        try:
            closes = market_data['close'].values
            volumes = market_data['volume'].values if 'volume' in market_data.columns else np.zeros_like(closes)
            
            # Calculate volatility
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
            
            # Determine trend
            if len(closes) >= 20:
                sma_20 = np.mean(closes[-20:])
                sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
                trend_strength = (closes[-1] - sma_50) / sma_50 if sma_50 > 0 else 0
            else:
                trend_strength = 0
            
            # Determine market regime
            regime = self.determine_market_regime(volatility, trend_strength, returns)
            
            # Analyze volume profile
            if len(volumes) > 0:
                avg_volume = np.mean(volumes)
                current_volume = volumes[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                volume_profile = "high" if volume_ratio > 1.5 else "low" if volume_ratio < 0.5 else "normal"
            else:
                volume_profile = "normal"
            
            # Determine sentiment (simplified)
            if trend_strength > 0.02:
                sentiment = "bullish"
            elif trend_strength < -0.02:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            # Determine time of day
            hour = datetime.now().hour
            if 0 <= hour < 4:
                time_of_day = "asia_session"
            elif 4 <= hour < 8:
                time_of_day = "europe_session"
            elif 8 <= hour < 12:
                time_of_day = "london_session"
            elif 12 <= hour < 16:
                time_of_day = "us_session"
            elif 16 <= hour < 20:
                time_of_day = "us_afternoon"
            else:
                time_of_day = "evening_session"
            
            # Check for economic events (this would come from calendar)
            economic_events = []
            
            # Determine liquidity (simplified)
            liquidity = "high" if volume_profile == "high" else "normal"
            
            context = MarketContext(
                regime=regime,
                volatility=float(volatility),
                trend_strength=float(trend_strength),
                volume_profile=volume_profile,
                liquidity=liquidity,
                sentiment=sentiment,
                time_of_day=time_of_day,
                economic_events=economic_events
            )
            
            # Store context history
            self.market_context_history.append(context)
            if len(self.market_context_history) > 1000:
                self.market_context_history = self.market_context_history[-1000:]
            
            return context
        
        except Exception as e:
            logger.error(f"Error analyzing market context: {e}")
            # Return default context
            return MarketContext(
                regime="unknown",
                volatility=0.0,
                trend_strength=0.0,
                volume_profile="normal",
                liquidity="normal",
                sentiment="neutral",
                time_of_day="unknown",
                economic_events=[]
            )
    
    def determine_market_regime(self, volatility: float, trend_strength: float, 
                              returns: np.ndarray) -> str:
        """Determine current market regime"""
        if len(returns) == 0:
            return "unknown"
        
        # Calculate additional metrics
        recent_volatility = np.std(returns[-5:]) if len(returns) >= 5 else volatility
        avg_volatility = np.std(returns[-20:]) if len(returns) >= 20 else recent_volatility
        
        if recent_volatility > avg_volatility * 1.5:
            return "volatile"
        elif abs(trend_strength) > 0.05:
            return "trending"
        elif abs(trend_strength) < 0.01 and volatility < 0.3:
            return "ranging"
        else:
            return "transition"
    
    async def collect_component_signals(self, market_data: pd.DataFrame, 
                                      market_context: MarketContext) -> List[ComponentSignal]:
        """Collect signals from all component strategies"""
        component_signals = []
        
        # Collect ML strategy signals
        for name, strategy in self.ml_strategies.items():
            try:
                if strategy.config and strategy.config.enabled:
                    # Generate signal from ML strategy
                    signal = await strategy.generate_signal(market_data)
                    
                    if signal and signal.signal_type != SignalType.HOLD:
                        component_signal = ComponentSignal(
                            strategy_type=MLStrategyType.MOMENTUM,  # This should come from strategy
                            strategy_name=name,
                            signal_type=signal.signal_type,
                            confidence=signal.confidence,
                            features=signal.features,
                            metadata={
                                "source": "ml",
                                "strategy_type": strategy.strategy_type.value,
                                **signal.metadata
                            },
                            timestamp=datetime.now()
                        )
                        component_signals.append(component_signal)
            
            except Exception as e:
                logger.error(f"Error collecting signal from ML strategy '{name}': {e}")
        
        # Collect technical strategy signals
        for name, strategy in self.technical_strategies.items():
            try:
                if strategy.config and strategy.config.enabled:
                    # Perform technical analysis
                    analysis = await strategy.analyze(market_data)
                    
                    # Generate signal from technical strategy
                    signal = await strategy.generate_signal(analysis)
                    
                    if signal and signal.signal_type != SignalType.HOLD:
                        component_signal = ComponentSignal(
                            strategy_type=TechnicalStrategyType.RSI_STRATEGY,  # This should come from strategy
                            strategy_name=name,
                            signal_type=signal.signal_type,
                            confidence=signal.confidence,
                            features=signal.features,
                            metadata={
                                "source": "technical",
                                "strategy_type": strategy.strategy_type.value,
                                "analysis_summary": analysis.summary
                            },
                            timestamp=datetime.now()
                        )
                        component_signals.append(component_signal)
            
            except Exception as e:
                logger.error(f"Error collecting signal from technical strategy '{name}': {e}")
        
        # Store component signals history
        self.component_signals_history.extend(component_signals)
        if len(self.component_signals_history) > 10000:
            self.component_signals_history = self.component_signals_history[-10000:]
        
        return component_signals
    
    @abstractmethod
    async def fuse_signals(self, component_signals: List[ComponentSignal],
                          market_context: MarketContext) -> FusionResult:
        """Fuse component signals into final trading signal"""
        pass
    
    async def calibrate_confidence(self, fusion_result: FusionResult, 
                                 historical_accuracy: Dict[str, float]) -> float:
        """Calibrate final signal confidence"""
        try:
            base_confidence = fusion_result.confidence
            
            # Apply market context adjustments
            context = fusion_result.market_context
            adjustment = self.calculate_context_adjustment(context)
            
            # Apply component performance weights
            performance_adjustment = self.calculate_performance_adjustment(
                fusion_result.component_signals, historical_accuracy
            )
            
            # Apply volatility adjustment
            volatility_adjustment = self.calculate_volatility_adjustment(context.volatility)
            
            # Combine adjustments
            calibrated_confidence = base_confidence * adjustment * performance_adjustment * volatility_adjustment
            
            # Ensure confidence is in valid range
            calibrated_confidence = max(0.1, min(0.95, calibrated_confidence))
            
            return calibrated_confidence
        
        except Exception as e:
            logger.error(f"Error calibrating confidence: {e}")
            return fusion_result.confidence
    
    def calculate_context_adjustment(self, context: MarketContext) -> float:
        """Calculate confidence adjustment based on market context"""
        adjustment = 1.0
        
        # Adjust based on market regime
        if context.regime == "trending":
            adjustment *= 1.2  # Higher confidence in trending markets
        elif context.regime == "volatile":
            adjustment *= 0.8  # Lower confidence in volatile markets
        elif context.regime == "ranging":
            adjustment *= 0.9  # Slightly lower in ranging markets
        
        # Adjust based on volume
        if context.volume_profile == "high":
            adjustment *= 1.1  # Higher confidence with high volume
        elif context.volume_profile == "low":
            adjustment *= 0.9  # Lower confidence with low volume
        
        # Adjust based on liquidity
        if context.liquidity == "high":
            adjustment *= 1.05  # Higher confidence with high liquidity
        
        # Adjust based on time of day
        if context.time_of_day in ["london_session", "us_session"]:
            adjustment *= 1.05  # Higher confidence during major sessions
        
        return adjustment
    
    def calculate_performance_adjustment(self, component_signals: List[ComponentSignal],
                                       historical_accuracy: Dict[str, float]) -> float:
        """Calculate adjustment based on component historical performance"""
        if not component_signals:
            return 1.0
        
        total_weight = 0.0
        weighted_accuracy = 0.0
        
        for signal in component_signals:
            strategy_name = signal.strategy_name
            accuracy = historical_accuracy.get(strategy_name, 0.5)
            weight = signal.confidence
            
            weighted_accuracy += accuracy * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_accuracy = weighted_accuracy / total_weight
            # Map accuracy to adjustment (0.5 accuracy = 1.0 adjustment)
            adjustment = avg_accuracy * 2.0
            return max(0.5, min(1.5, adjustment))
        
        return 1.0
    
    def calculate_volatility_adjustment(self, volatility: float) -> float:
        """Calculate adjustment based on market volatility"""
        # Normalize volatility (assuming typical range 0-1 for annualized volatility)
        normalized_vol = min(1.0, volatility)
        
        # Higher volatility -> lower confidence adjustment
        # Sigmoid function to smooth adjustment
        adjustment = 1.0 / (1.0 + np.exp(5 * (normalized_vol - 0.5)))
        
        return max(0.5, min(1.5, adjustment))
    
    async def update_fusion_weights(self, trade_result: Dict):
        """Update fusion weights based on trade outcome"""
        try:
            if not self.fusion_weights.dynamic_weights:
                return
            
            success = trade_result.get("success", False)
            pnl = trade_result.get("pnl", 0)
            
            if not success:
                # Failed trade - reduce weights for involved strategies
                component_signals = trade_result.get("component_signals", [])
                for signal in component_signals:
                    strategy_name = signal.strategy_name
                    if strategy_name in self.fusion_weights.component_weights:
                        current_weight = self.fusion_weights.component_weights[strategy_name]
                        # Reduce weight by 10% but keep minimum
                        new_weight = max(0.01, current_weight * 0.9)
                        self.fusion_weights.component_weights[strategy_name] = new_weight
            
            elif pnl > 0:
                # Successful trade - increase weights for involved strategies
                component_signals = trade_result.get("component_signals", [])
                for signal in component_signals:
                    strategy_name = signal.strategy_name
                    if strategy_name in self.fusion_weights.component_weights:
                        current_weight = self.fusion_weights.component_weights[strategy_name]
                        # Increase weight by 5% but keep maximum
                        new_weight = min(0.5, current_weight * 1.05)
                        self.fusion_weights.component_weights[strategy_name] = new_weight
            
            # Re-normalize weights
            self.fusion_weights.normalize()
            self.fusion_weights.last_updated = datetime.now()
            
            logger.info(f"Updated fusion weights: {self.fusion_weights.component_weights}")
        
        except Exception as e:
            logger.error(f"Error updating fusion weights: {e}")
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate hybrid trading signal"""
        try:
            if not self.initialized:
                logger.error("Strategy not initialized")
                return None
            
            # Analyze market context
            market_context = await self.analyze_market_context(market_data)
            
            # Collect component signals
            component_signals = await self.collect_component_signals(market_data, market_context)
            
            if not component_signals:
                logger.info("No component signals generated")
                return None
            
            # Fuse signals
            fusion_result = await self.fuse_signals(component_signals, market_context)
            
            # Calibrate confidence
            calibrated_confidence = await self.calibrate_confidence(
                fusion_result, self.component_performance
            )
            
            # Create final trading signal
            signal = TradingSignal(
                symbol=self.config.symbol if self.config else "BTCUSDT",
                signal_type=fusion_result.final_signal,
                confidence=calibrated_confidence,
                price=market_data['close'].iloc[-1],
                timestamp=datetime.now(),
                strategy_name=self.name,
                model_name=f"Hybrid_{self.fusion_method.value}",
                features={
                    "component_count": len(component_signals),
                    "fusion_confidence": fusion_result.confidence,
                    "calibrated_confidence": calibrated_confidence
                },
                metadata={
                    "fusion_method": fusion_result.fusion_method.value,
                    "market_context": market_context.to_dict(),
                    "component_signals": [
                        {
                            "strategy_name": cs.strategy_name,
                            "signal_type": cs.signal_type.value,
                            "confidence": cs.confidence,
                            "source": cs.metadata.get("source", "unknown")
                        }
                        for cs in component_signals
                    ],
                    "fusion_weights": {
                        "ml_weight": fusion_result.fusion_weights.ml_weight,
                        "technical_weight": fusion_result.fusion_weights.technical_weight,
                        "component_weights": fusion_result.fusion_weights.component_weights
                    }
                }
            )
            
            # Store fusion result
            self.fusion_results_history.append(fusion_result)
            if len(self.fusion_results_history) > 1000:
                self.fusion_results_history = self.fusion_results_history[-1000:]
            
            # Store signal
            self.signals_history.append(signal)
            
            logger.info(f"Generated hybrid signal: {signal.signal_type.value} "
                       f"with confidence {calibrated_confidence:.2f}")
            
            return signal
        
        except Exception as e:
            logger.error(f"Error generating hybrid signal: {e}")
            return None
    
    async def update_performance(self, trade_result: Dict):
        """Update performance metrics"""
        try:
            self.trades_history.append(trade_result)
            
            # Update component performance
            component_signals = trade_result.get("metadata", {}).get("component_signals", [])
            success = trade_result.get("success", False)
            
            for comp_signal in component_signals:
                strategy_name = comp_signal.get("strategy_name")
                if strategy_name:
                    if strategy_name not in self.component_performance:
                        self.component_performance[strategy_name] = {
                            "total_trades": 0,
                            "successful_trades": 0,
                            "accuracy": 0.0
                        }
                    
                    perf = self.component_performance[strategy_name]
                    perf["total_trades"] += 1
                    if success:
                        perf["successful_trades"] += 1
                    
                    perf["accuracy"] = perf["successful_trades"] / perf["total_trades"]
            
            # Update fusion performance
            fusion_method = trade_result.get("metadata", {}).get("fusion_method", "unknown")
            if fusion_method not in self.fusion_performance:
                self.fusion_performance[fusion_method] = 0.0
            
            if success:
                self.fusion_performance[fusion_method] = min(
                    1.0, self.fusion_performance[fusion_method] + 0.01
                )
            else:
                self.fusion_performance[fusion_method] = max(
                    0.0, self.fusion_performance[fusion_method] - 0.02
                )
            
            # Update fusion weights
            await self.update_fusion_weights(trade_result)
            
            logger.info(f"Updated performance metrics")
        
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "strategy_name": self.name,
            "strategy_type": self.strategy_type.value,
            "component_count": {
                "ml": len(self.ml_strategies),
                "technical": len(self.technical_strategies),
                "total": len(self.ml_strategies) + len(self.technical_strategies)
            },
            "fusion_method": self.fusion_method.value,
            "fusion_weights": {
                "ml": self.fusion_weights.ml_weight,
                "technical": self.fusion_weights.technical_weight,
                "dynamic": self.fusion_weights.dynamic_weights
            },
            "component_performance": self.component_performance,
            "fusion_performance": self.fusion_performance,
            "recent_signals": len(self.signals_history),
            "market_context": self.market_context_history[-1].to_dict() if self.market_context_history else None,
            "last_updated": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check strategy health"""
        return {
            "healthy": self.initialized,
            "initialized": self.initialized,
            "component_strategies": {
                "ml": {name: "loaded" for name in self.ml_strategies},
                "technical": {name: "loaded" for name in self.technical_strategies}
            },
            "fusion_weights_valid": self.fusion_weights.get_total_weight() > 0.9,
            "recent_activity": {
                "signals_last_hour": len([s for s in self.signals_history 
                                         if datetime.now() - s.timestamp < timedelta(hours=1)]),
                "fusion_results": len(self.fusion_results_history),
                "market_context_updates": len(self.market_context_history)
            },
            "issues": []
        }

# Concrete Hybrid Strategy Implementations

class MLTechnicalFusionStrategy(BaseHybridStrategy):
    """Fusion of ML predictions with technical analysis"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, HybridStrategyType.ML_TECHNICAL_FUSION)
        self.fusion_method = FusionMethod.WEIGHTED_AVERAGE
        self.confidence_threshold = 0.6
    
    async def load_component_strategies(self):
        """Load ML and technical strategies"""
        try:
            # Load ML strategies
            from strategies.ml_strategies import MLStrategyFactory
            ml_factory = MLStrategyFactory()
            
            # Load momentum strategy
            momentum_config = StrategyConfig(
                name="ml_momentum",
                strategy_type=MLStrategyType.MOMENTUM,
                symbol=self.config.symbol if self.config else "BTCUSDT",
                timeframe=self.config.timeframe if self.config else "1h"
            )
            momentum_strategy = ml_factory.create_strategy(
                MLStrategyType.MOMENTUM, self.config_manager
            )
            await momentum_strategy.initialize(momentum_config)
            self.ml_strategies["ml_momentum"] = momentum_strategy
            
            # Load mean reversion strategy
            mean_reversion_config = StrategyConfig(
                name="ml_mean_reversion",
                strategy_type=MLStrategyType.MEAN_REVERSION,
                symbol=self.config.symbol if self.config else "BTCUSDT",
                timeframe=self.config.timeframe if self.config else "1h"
            )
            mean_reversion_strategy = ml_factory.create_strategy(
                MLStrategyType.MEAN_REVERSION, self.config_manager
            )
            await mean_reversion_strategy.initialize(mean_reversion_config)
            self.ml_strategies["ml_mean_reversion"] = mean_reversion_strategy
            
            # Load technical strategies
            from strategies.technical_strategies import TechnicalStrategyFactory
            tech_factory = TechnicalStrategyFactory()
            
            # Load RSI strategy
            rsi_config = StrategyConfig(
                name="tech_rsi",
                strategy_type=TechnicalStrategyType.RSI_STRATEGY,
                symbol=self.config.symbol if self.config else "BTCUSDT",
                timeframe=self.config.timeframe if self.config else "1h"
            )
            rsi_strategy = tech_factory.create_strategy(
                TechnicalStrategyType.RSI_STRATEGY, self.config_manager
            )
            await rsi_strategy.initialize(rsi_config)
            self.technical_strategies["tech_rsi"] = rsi_strategy
            
            # Load MACD strategy
            macd_config = StrategyConfig(
                name="tech_macd",
                strategy_type=TechnicalStrategyType.MACD_STRATEGY,
                symbol=self.config.symbol if self.config else "BTCUSDT",
                timeframe=self.config.timeframe if self.config else "1h"
            )
            macd_strategy = tech_factory.create_strategy(
                TechnicalStrategyType.MACD_STRATEGY, self.config_manager
            )
            await macd_strategy.initialize(macd_config)
            self.technical_strategies["tech_macd"] = macd_strategy
            
            # Load Bollinger Bands strategy
            bb_config = StrategyConfig(
                name="tech_bollinger",
                strategy_type=TechnicalStrategyType.BOLLINGER_BANDS,
                symbol=self.config.symbol if self.config else "BTCUSDT",
                timeframe=self.config.timeframe if self.config else "1h"
            )
            bb_strategy = tech_factory.create_strategy(
                TechnicalStrategyType.BOLLINGER_BANDS, self.config_manager
            )
            await bb_strategy.initialize(bb_config)
            self.technical_strategies["tech_bollinger"] = bb_strategy
            
            logger.info(f"Loaded {len(self.ml_strategies)} ML and "
                       f"{len(self.technical_strategies)} technical strategies")
        
        except Exception as e:
            logger.error(f"Error loading component strategies: {e}")
            raise
    
    async def fuse_signals(self, component_signals: List[ComponentSignal],
                          market_context: MarketContext) -> FusionResult:
        """Fuse signals using weighted average"""
        try:
            if not component_signals:
                raise ValueError("No component signals to fuse")
            
            # Convert signals to numerical values
            signal_values = []
            confidences = []
            weights = []
            
            for signal in component_signals:
                # Convert signal to numerical value
                if signal.signal_type == SignalType.BUY:
                    value = 1.0
                elif signal.signal_type == SignalType.SELL:
                    value = -1.0
                else:
                    value = 0.0
                
                # Get weight for this strategy
                weight = self.fusion_weights.component_weights.get(
                    signal.strategy_name, 
                    self.fusion_weights.ml_weight if "ml" in signal.metadata.get("source", "") 
                    else self.fusion_weights.technical_weight
                )
                
                signal_values.append(value)
                confidences.append(signal.confidence)
                weights.append(weight)
            
            # Calculate weighted average
            total_weight = sum(weights)
            if total_weight == 0:
                # Default to equal weights
                weights = [1.0 / len(component_signals)] * len(component_signals)
                total_weight = 1.0
            
            weighted_sum = sum(v * w * c for v, w, c in zip(signal_values, weights, confidences))
            weighted_average = weighted_sum / total_weight
            
            # Determine final signal
            if weighted_average > self.confidence_threshold:
                final_signal = SignalType.BUY
                confidence = min(0.95, weighted_average)
            elif weighted_average < -self.confidence_threshold:
                final_signal = SignalType.SELL
                confidence = min(0.95, abs(weighted_average))
            else:
                final_signal = SignalType.HOLD
                confidence = 0.0
            
            # Calculate agreement score
            agreement_score = self.calculate_agreement_score(component_signals)
            
            # Adjust confidence based on agreement
            if final_signal != SignalType.HOLD:
                confidence = confidence * (0.5 + 0.5 * agreement_score)
            
            return FusionResult(
                final_signal=final_signal,
                confidence=confidence,
                component_signals=component_signals,
                fusion_weights=self.fusion_weights,
                market_context=market_context,
                fusion_method=self.fusion_method,
                metadata={
                    "weighted_average": weighted_average,
                    "agreement_score": agreement_score,
                    "component_count": len(component_signals)
                }
            )
        
        except Exception as e:
            logger.error(f"Error fusing signals: {e}")
            # Return neutral signal on error
            return FusionResult(
                final_signal=SignalType.HOLD,
                confidence=0.0,
                component_signals=component_signals,
                fusion_weights=self.fusion_weights,
                market_context=market_context,
                fusion_method=self.fusion_method,
                metadata={"error": str(e)}
            )
    
    def calculate_agreement_score(self, component_signals: List[ComponentSignal]) -> float:
        """Calculate agreement score among component signals"""
        if len(component_signals) < 2:
            return 1.0
        
        buy_signals = [s for s in component_signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in component_signals if s.signal_type == SignalType.SELL]
        
        total_signals = len(component_signals)
        max_agreement = max(len(buy_signals), len(sell_signals))
        
        agreement_score = max_agreement / total_signals if total_signals > 0 else 0.0
        
        # Weight by confidence
        if max_agreement > 0:
            agreeing_signals = buy_signals if len(buy_signals) > len(sell_signals) else sell_signals
            avg_confidence = sum(s.confidence for s in agreeing_signals) / len(agreeing_signals)
            agreement_score *= avg_confidence
        
        return agreement_score

class EnsembleFusionStrategy(BaseHybridStrategy):
    """Ensemble fusion using stacking or voting methods"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, HybridStrategyType.ENSEMBLE_FUSION)
        self.fusion_method = FusionMethod.STACKING
        self.meta_model = None
    
    async def load_component_strategies(self):
        """Load diverse set of strategies for ensemble"""
        try:
            # Load multiple ML strategies
            from strategies.ml_strategies import MLStrategyFactory
            ml_factory = MLStrategyFactory()
            
            ml_strategy_types = [
                (MLStrategyType.MOMENTUM, "ml_momentum"),
                (MLStrategyType.MEAN_REVERSION, "ml_mean_reversion"),
                (MLStrategyType.DEEP_LEARNING, "ml_deep_learning")
            ]
            
            for strategy_type, name in ml_strategy_types:
                config = StrategyConfig(
                    name=name,
                    strategy_type=strategy_type,
                    symbol=self.config.symbol if self.config else "BTCUSDT",
                    timeframe=self.config.timeframe if self.config else "1h"
                )
                strategy = ml_factory.create_strategy(strategy_type, self.config_manager)
                await strategy.initialize(config)
                self.ml_strategies[name] = strategy
            
            # Load multiple technical strategies
            from strategies.technical_strategies import TechnicalStrategyFactory
            tech_factory = TechnicalStrategyFactory()
            
            tech_strategy_types = [
                (TechnicalStrategyType.RSI_STRATEGY, "tech_rsi"),
                (TechnicalStrategyType.MACD_STRATEGY, "tech_macd"),
                (TechnicalStrategyType.BOLLINGER_BANDS, "tech_bollinger"),
                (TechnicalStrategyType.ICHIMOKU, "tech_ichimoku"),
                (TechnicalStrategyType.MOVING_AVERAGES, "tech_ma")
            ]
            
            for strategy_type, name in tech_strategy_types:
                config = StrategyConfig(
                    name=name,
                    strategy_type=strategy_type,
                    symbol=self.config.symbol if self.config else "BTCUSDT",
                    timeframe=self.config.timeframe if self.config else "1h"
                )
                strategy = tech_factory.create_strategy(strategy_type, self.config_manager)
                await strategy.initialize(config)
                self.technical_strategies[name] = strategy
            
            # Initialize meta-model for stacking
            await self.initialize_meta_model()
            
            logger.info(f"Loaded ensemble of {len(self.ml_strategies)} ML and "
                       f"{len(self.technical_strategies)} technical strategies")
        
        except Exception as e:
            logger.error(f"Error loading ensemble strategies: {e}")
            raise
    
    async def initialize_meta_model(self):
        """Initialize meta-model for stacking"""
        try:
            # Use a simple ensemble model as meta-model
            self.meta_model = EnsembleModel(self.config_manager)
            logger.info("Initialized meta-model for stacking")
        
        except Exception as e:
            logger.error(f"Error initializing meta-model: {e}")
            self.meta_model = None
    
    async def fuse_signals(self, component_signals: List[ComponentSignal],
                          market_context: MarketContext) -> FusionResult:
        """Fuse signals using ensemble methods"""
        try:
            if self.fusion_method == FusionMethod.VOTING:
                return await self.voting_fusion(component_signals, market_context)
            elif self.fusion_method == FusionMethod.STACKING:
                return await self.stacking_fusion(component_signals, market_context)
            elif self.fusion_method == FusionMethod.BAYESIAN:
                return await self.bayesian_fusion(component_signals, market_context)
            else:
                # Default to weighted average
                return await self.weighted_average_fusion(component_signals, market_context)
        
        except Exception as e:
            logger.error(f"Error in ensemble fusion: {e}")
            return FusionResult(
                final_signal=SignalType.HOLD,
                confidence=0.0,
                component_signals=component_signals,
                fusion_weights=self.fusion_weights,
                market_context=market_context,
                fusion_method=self.fusion_method,
                metadata={"error": str(e)}
            )
    
    async def voting_fusion(self, component_signals: List[ComponentSignal],
                          market_context: MarketContext) -> FusionResult:
        """Majority voting fusion"""
        if not component_signals:
            raise ValueError("No component signals for voting")
        
        # Count votes
        buy_votes = sum(1 for s in component_signals if s.signal_type == SignalType.BUY)
        sell_votes = sum(1 for s in component_signals if s.signal_type == SignalType.SELL)
        
        # Weight votes by confidence
        weighted_buy = sum(s.confidence for s in component_signals 
                          if s.signal_type == SignalType.BUY)
        weighted_sell = sum(s.confidence for s in component_signals 
                           if s.signal_type == SignalType.SELL)
        
        total_votes = buy_votes + sell_votes
        
        if total_votes == 0:
            return FusionResult(
                final_signal=SignalType.HOLD,
                confidence=0.0,
                component_signals=component_signals,
                fusion_weights=self.fusion_weights,
                market_context=market_context,
                fusion_method=FusionMethod.VOTING,
                metadata={"vote_counts": {"buy": 0, "sell": 0}}
            )
        
        # Determine winner
        if buy_votes > sell_votes:
            final_signal = SignalType.BUY
            confidence = weighted_buy / buy_votes if buy_votes > 0 else 0.0
            margin = (buy_votes - sell_votes) / total_votes
        elif sell_votes > buy_votes:
            final_signal = SignalType.SELL
            confidence = weighted_sell / sell_votes if sell_votes > 0 else 0.0
            margin = (sell_votes - buy_votes) / total_votes
        else:
            # Tie - use weighted sum
            if weighted_buy > weighted_sell:
                final_signal = SignalType.BUY
                confidence = weighted_buy / buy_votes if buy_votes > 0 else 0.0
                margin = 0.0
            elif weighted_sell > weighted_buy:
                final_signal = SignalType.SELL
                confidence = weighted_sell / sell_votes if sell_votes > 0 else 0.0
                margin = 0.0
            else:
                final_signal = SignalType.HOLD
                confidence = 0.0
                margin = 0.0
        
        # Adjust confidence by voting margin
        confidence = confidence * (0.5 + 0.5 * margin)
        
        return FusionResult(
            final_signal=final_signal,
            confidence=confidence,
            component_signals=component_signals,
            fusion_weights=self.fusion_weights,
            market_context=market_context,
            fusion_method=FusionMethod.VOTING,
            metadata={
                "vote_counts": {"buy": buy_votes, "sell": sell_votes},
                "weighted_votes": {"buy": weighted_buy, "sell": weighted_sell},
                "voting_margin": margin
            }
        )
    
    async def stacking_fusion(self, component_signals: List[ComponentSignal],
                            market_context: MarketContext) -> FusionResult:
        """Stacking fusion using meta-model"""
        try:
            if not self.meta_model:
                logger.warning("Meta-model not initialized, falling back to voting")
                return await self.voting_fusion(component_signals, market_context)
            
            # Prepare features for meta-model
            features = self.prepare_stacking_features(component_signals, market_context)
            
            # Get prediction from meta-model
            prediction = await self.meta_model.predict(features)
            
            # Interpret prediction
            if prediction.get("direction", 0) > 0.5:
                final_signal = SignalType.BUY
                confidence = prediction.get("confidence", 0.5)
            elif prediction.get("direction", 0) < -0.5:
                final_signal = SignalType.SELL
                confidence = prediction.get("confidence", 0.5)
            else:
                final_signal = SignalType.HOLD
                confidence = 0.0
            
            return FusionResult(
                final_signal=final_signal,
                confidence=confidence,
                component_signals=component_signals,
                fusion_weights=self.fusion_weights,
                market_context=market_context,
                fusion_method=FusionMethod.STACKING,
                metadata={
                    "meta_prediction": prediction,
                    "feature_count": len(features)
                }
            )
        
        except Exception as e:
            logger.error(f"Error in stacking fusion: {e}")
            # Fall back to voting
            return await self.voting_fusion(component_signals, market_context)
    
    def prepare_stacking_features(self, component_signals: List[ComponentSignal],
                                market_context: MarketContext) -> Dict[str, float]:
        """Prepare features for stacking meta-model"""
        features = {}
        
        # Add component signal features
        for i, signal in enumerate(component_signals):
            features[f"signal_{i}_type"] = 1.0 if signal.signal_type == SignalType.BUY else -1.0 if signal.signal_type == SignalType.SELL else 0.0
            features[f"signal_{i}_confidence"] = signal.confidence
            features[f"signal_{i}_strategy"] = hash(signal.strategy_name) % 100 / 100.0
        
        # Add market context features
        features["volatility"] = market_context.volatility
        features["trend_strength"] = market_context.trend_strength
        features["sentiment"] = 1.0 if market_context.sentiment == "bullish" else -1.0 if market_context.sentiment == "bearish" else 0.0
        
        # Add agreement features
        buy_count = sum(1 for s in component_signals if s.signal_type == SignalType.BUY)
        sell_count = sum(1 for s in component_signals if s.signal_type == SignalType.SELL)
        total = len(component_signals)
        
        features["agreement_ratio"] = abs(buy_count - sell_count) / total if total > 0 else 0.0
        features["signal_diversity"] = len(set(s.signal_type for s in component_signals))
        
        return features
    
    async def bayesian_fusion(self, component_signals: List[ComponentSignal],
                            market_context: MarketContext) -> FusionResult:
        """Bayesian fusion incorporating uncertainties"""
        try:
            # Simple Bayesian fusion assuming independent components
            prior_buy = 0.5  # Prior probability of buy
            prior_sell = 0.5  # Prior probability of sell
            
            likelihood_buy = 1.0
            likelihood_sell = 1.0
            
            for signal in component_signals:
                # Get component accuracy from performance history
                accuracy = self.component_performance.get(
                    signal.strategy_name, {}
                ).get("accuracy", 0.5)
                
                if signal.signal_type == SignalType.BUY:
                    likelihood_buy *= accuracy * signal.confidence
                    likelihood_sell *= (1 - accuracy) * signal.confidence
                elif signal.signal_type == SignalType.SELL:
                    likelihood_buy *= (1 - accuracy) * signal.confidence
                    likelihood_sell *= accuracy * signal.confidence
            
            # Calculate posterior probabilities
            evidence = prior_buy * likelihood_buy + prior_sell * likelihood_sell
            
            if evidence > 0:
                posterior_buy = (prior_buy * likelihood_buy) / evidence
                posterior_sell = (prior_sell * likelihood_sell) / evidence
            else:
                posterior_buy = 0.5
                posterior_sell = 0.5
            
            # Determine signal
            if posterior_buy > 0.6:  # Threshold for buy
                final_signal = SignalType.BUY
                confidence = posterior_buy
            elif posterior_sell > 0.6:  # Threshold for sell
                final_signal = SignalType.SELL
                confidence = posterior_sell
            else:
                final_signal = SignalType.HOLD
                confidence = 0.0
            
            return FusionResult(
                final_signal=final_signal,
                confidence=confidence,
                component_signals=component_signals,
                fusion_weights=self.fusion_weights,
                market_context=market_context,
                fusion_method=FusionMethod.BAYESIAN,
                metadata={
                    "posterior_buy": posterior_buy,
                    "posterior_sell": posterior_sell,
                    "bayesian_evidence": evidence
                }
            )
        
        except Exception as e:
            logger.error(f"Error in Bayesian fusion: {e}")
            return FusionResult(
                final_signal=SignalType.HOLD,
                confidence=0.0,
                component_signals=component_signals,
                fusion_weights=self.fusion_weights,
                market_context=market_context,
                fusion_method=FusionMethod.BAYESIAN,
                metadata={"error": str(e)}
            )
    
    async def weighted_average_fusion(self, component_signals: List[ComponentSignal],
                                    market_context: MarketContext) -> FusionResult:
        """Weighted average fusion as fallback"""
        # Similar to MLTechnicalFusionStrategy but with dynamic weights
        if not component_signals:
            raise ValueError("No component signals to fuse")
        
        signal_values = []
        confidences = []
        weights = []
        
        for signal in component_signals:
            if signal.signal_type == SignalType.BUY:
                value = 1.0
            elif signal.signal_type == SignalType.SELL:
                value = -1.0
            else:
                value = 0.0
            
            # Use performance-based weights
            accuracy = self.component_performance.get(
                signal.strategy_name, {}
            ).get("accuracy", 0.5)
            
            weight = accuracy * signal.confidence
            
            signal_values.append(value)
            confidences.append(signal.confidence)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        weighted_sum = sum(v * w for v, w in zip(signal_values, weights))
        
        # Determine final signal
        if weighted_sum > 0.3:
            final_signal = SignalType.BUY
            confidence = min(0.95, weighted_sum)
        elif weighted_sum < -0.3:
            final_signal = SignalType.SELL
            confidence = min(0.95, abs(weighted_sum))
        else:
            final_signal = SignalType.HOLD
            confidence = 0.0
        
        return FusionResult(
            final_signal=final_signal,
            confidence=confidence,
            component_signals=component_signals,
            fusion_weights=self.fusion_weights,
            market_context=market_context,
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            metadata={
                "weighted_average": weighted_sum,
                "performance_based_weights": True
            }
        )

class AdaptiveWeightingStrategy(BaseHybridStrategy):
    """Adaptive weighting based on market conditions"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, HybridStrategyType.ADAPTIVE_WEIGHTING)
        self.fusion_method = FusionMethod.WEIGHTED_AVERAGE
        self.regime_weights = self.initialize_regime_weights()
    
    def initialize_regime_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialize weights for different market regimes"""
        return {
            "trending": {"ml_weight": 0.7, "technical_weight": 0.3},
            "ranging": {"ml_weight": 0.3, "technical_weight": 0.7},
            "volatile": {"ml_weight": 0.4, "technical_weight": 0.6},
            "news_driven": {"ml_weight": 0.6, "technical_weight": 0.4},
            "default": {"ml_weight": 0.5, "technical_weight": 0.5}
        }
    
    async def load_component_strategies(self):
        """Load strategies with regime-specific configurations"""
        try:
            # Load ML strategies
            from strategies.ml_strategies import MLStrategyFactory
            ml_factory = MLStrategyFactory()
            
            # Momentum works well in trending markets
            momentum_config = StrategyConfig(
                name="adaptive_momentum",
                strategy_type=MLStrategyType.MOMENTUM,
                symbol=self.config.symbol if self.config else "BTCUSDT",
                timeframe=self.config.timeframe if self.config else "1h",
                parameters={"lookback_period": 20}
            )
            momentum_strategy = ml_factory.create_strategy(
                MLStrategyType.MOMENTUM, self.config_manager
            )
            await momentum_strategy.initialize(momentum_config)
            self.ml_strategies["adaptive_momentum"] = momentum_strategy
            
            # Mean reversion works well in ranging markets
            mean_reversion_config = StrategyConfig(
                name="adaptive_mean_reversion",
                strategy_type=MLStrategyType.MEAN_REVERSION,
                symbol=self.config.symbol if self.config else "BTCUSDT",
                timeframe=self.config.timeframe if self.config else "1h",
                parameters={"lookback_period": 50}
            )
            mean_reversion_strategy = ml_factory.create_strategy(
                MLStrategyType.MEAN_REVERSION, self.config_manager
            )
            await mean_reversion_strategy.initialize(mean_reversion_config)
            self.ml_strategies["adaptive_mean_reversion"] = mean_reversion_strategy
            
            # Load technical strategies
            from strategies.technical_strategies import TechnicalStrategyFactory
            tech_factory = TechnicalStrategyFactory()
            
            # Trend-following technical strategies
            trend_config = StrategyConfig(
                name="adaptive_trend",
                strategy_type=TechnicalStrategyType.MOVING_AVERAGES,
                symbol=self.config.symbol if self.config else "BTCUSDT",
                timeframe=self.config.timeframe if self.config else "1h"
            )
            trend_strategy = tech_factory.create_strategy(
                TechnicalStrategyType.MOVING_AVERAGES, self.config_manager
            )
            await trend_strategy.initialize(trend_config)
            self.technical_strategies["adaptive_trend"] = trend_strategy
            
            # Range-bound technical strategies
            range_config = StrategyConfig(
                name="adaptive_range",
                strategy_type=TechnicalStrategyType.BOLLINGER_BANDS,
                symbol=self.config.symbol if self.config else "BTCUSDT",
                timeframe=self.config.timeframe if self.config else "1h"
            )
            range_strategy = tech_factory.create_strategy(
                TechnicalStrategyType.BOLLINGER_BANDS, self.config_manager
            )
            await range_strategy.initialize(range_config)
            self.technical_strategies["adaptive_range"] = range_strategy
            
            logger.info(f"Loaded adaptive strategies: {len(self.ml_strategies)} ML, "
                       f"{len(self.technical_strategies)} technical")
        
        except Exception as e:
            logger.error(f"Error loading adaptive strategies: {e}")
            raise
    
    async def adapt_weights_to_regime(self, regime: str) -> FusionWeights:
        """Adapt fusion weights based on market regime"""
        regime_config = self.regime_weights.get(regime, self.regime_weights["default"])
        
        # Create new fusion weights
        adaptive_weights = FusionWeights(
            ml_weight=regime_config["ml_weight"],
            technical_weight=regime_config["technical_weight"],
            dynamic_weights=True
        )
        
        # Adjust component weights based on regime
        for name in self.ml_strategies:
            if "momentum" in name and regime == "trending":
                adaptive_weights.component_weights[name] = 0.4
            elif "mean_reversion" in name and regime == "ranging":
                adaptive_weights.component_weights[name] = 0.4
            else:
                adaptive_weights.component_weights[name] = 0.1
        
        for name in self.technical_strategies:
            if "trend" in name and regime == "trending":
                adaptive_weights.component_weights[name] = 0.4
            elif "range" in name and regime == "ranging":
                adaptive_weights.component_weights[name] = 0.4
            else:
                adaptive_weights.component_weights[name] = 0.1
        
        adaptive_weights.normalize()
        return adaptive_weights
    
    async def fuse_signals(self, component_signals: List[ComponentSignal],
                          market_context: MarketContext) -> FusionResult:
        """Fuse signals with regime-adaptive weights"""
        try:
            # Adapt weights to current regime
            regime = market_context.regime
            adaptive_weights = await self.adapt_weights_to_regime(regime)
            
            # Use adapted weights for fusion
            self.fusion_weights = adaptive_weights
            
            # Now fuse using the adapted weights (similar to MLTechnicalFusionStrategy)
            if not component_signals:
                raise ValueError("No component signals to fuse")
            
            signal_values = []
            confidences = []
            weights = []
            
            for signal in component_signals:
                if signal.signal_type == SignalType.BUY:
                    value = 1.0
                elif signal.signal_type == SignalType.SELL:
                    value = -1.0
                else:
                    value = 0.0
                
                weight = adaptive_weights.component_weights.get(
                    signal.strategy_name, 
                    adaptive_weights.ml_weight if "ml" in signal.metadata.get("source", "") 
                    else adaptive_weights.technical_weight
                )
                
                signal_values.append(value)
                confidences.append(signal.confidence)
                weights.append(weight)
            
            # Calculate weighted average
            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1.0 / len(component_signals)] * len(component_signals)
                total_weight = 1.0
            
            weighted_sum = sum(v * w * c for v, w, c in zip(signal_values, weights, confidences))
            weighted_average = weighted_sum / total_weight
            
            # Determine final signal
            if weighted_average > 0.3:
                final_signal = SignalType.BUY
                confidence = min(0.95, weighted_average)
            elif weighted_average < -0.3:
                final_signal = SignalType.SELL
                confidence = min(0.95, abs(weighted_average))
            else:
                final_signal = SignalType.HOLD
                confidence = 0.0
            
            return FusionResult(
                final_signal=final_signal,
                confidence=confidence,
                component_signals=component_signals,
                fusion_weights=adaptive_weights,
                market_context=market_context,
                fusion_method=self.fusion_method,
                metadata={
                    "regime": regime,
                    "weighted_average": weighted_average,
                    "regime_adaptive": True
                }
            )
        
        except Exception as e:
            logger.error(f"Error in adaptive fusion: {e}")
            return FusionResult(
                final_signal=SignalType.HOLD,
                confidence=0.0,
                component_signals=component_signals,
                fusion_weights=self.fusion_weights,
                market_context=market_context,
                fusion_method=self.fusion_method,
                metadata={"error": str(e)}
            )

class ReinforcementEnhancedStrategy(BaseHybridStrategy):
    """Hybrid strategy enhanced with reinforcement learning"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, HybridStrategyType.REINFORCEMENT_ENHANCED)
        self.fusion_method = FusionMethod.NEURAL_FUSION
        self.rl_model = None
        self.state_size = 20
        self.action_space = ['buy', 'sell', 'hold', 'adjust_weights']
    
    async def load_component_strategies(self):
        """Load strategies for RL enhancement"""
        try:
            # Load a mix of strategies
            from strategies.ml_strategies import MLStrategyFactory
            from strategies.technical_strategies import TechnicalStrategyFactory
            
            ml_factory = MLStrategyFactory()
            tech_factory = TechnicalStrategyFactory()
            
            # Load core strategies
            strategies_to_load = [
                ("ml_momentum", MLStrategyType.MOMENTUM, ml_factory),
                ("ml_mean_reversion", MLStrategyType.MEAN_REVERSION, ml_factory),
                ("tech_rsi", TechnicalStrategyType.RSI_STRATEGY, tech_factory),
                ("tech_macd", TechnicalStrategyType.MACD_STRATEGY, tech_factory)
            ]
            
            for name, strategy_type, factory in strategies_to_load:
                if isinstance(strategy_type, MLStrategyType):
                    config = StrategyConfig(
                        name=name,
                        strategy_type=strategy_type,
                        symbol=self.config.symbol if self.config else "BTCUSDT",
                        timeframe=self.config.timeframe if self.config else "1h"
                    )
                    strategy = factory.create_strategy(strategy_type, self.config_manager)
                    await strategy.initialize(config)
                    self.ml_strategies[name] = strategy
                else:
                    config = StrategyConfig(
                        name=name,
                        strategy_type=strategy_type,
                        symbol=self.config.symbol if self.config else "BTCUSDT",
                        timeframe=self.config.timeframe if self.config else "1h"
                    )
                    strategy = factory.create_strategy(strategy_type, self.config_manager)
                    await strategy.initialize(config)
                    self.technical_strategies[name] = strategy
            
            # Initialize RL model
            await self.initialize_rl_model()
            
            logger.info(f"Loaded RL-enhanced strategies: {len(self.ml_strategies)} ML, "
                       f"{len(self.technical_strategies)} technical")
        
        except Exception as e:
            logger.error(f"Error loading RL-enhanced strategies: {e}")
            raise
    
    async def initialize_rl_model(self):
        """Initialize reinforcement learning model"""
        try:
            from core.neural_networks.reinforcement_learning import RLModel
            self.rl_model = RLModel(self.config_manager)
            logger.info("Initialized RL model for strategy enhancement")
        
        except Exception as e:
            logger.error(f"Error initializing RL model: {e}")
            self.rl_model = None
    
    async def get_rl_state(self, component_signals: List[ComponentSignal],
                          market_context: MarketContext) -> np.ndarray:
        """Get state representation for RL agent"""
        state_features = []
        
        # Component signal features
        for signal in component_signals:
            if signal.signal_type == SignalType.BUY:
                state_features.append(1.0)
            elif signal.signal_type == SignalType.SELL:
                state_features.append(-1.0)
            else:
                state_features.append(0.0)
            
            state_features.append(signal.confidence)
        
        # Market context features
        state_features.append(market_context.volatility)
        state_features.append(market_context.trend_strength)
        state_features.append(1.0 if market_context.sentiment == "bullish" else 
                             -1.0 if market_context.sentiment == "bearish" else 0.0)
        
        # Performance features
        for name in list(self.ml_strategies.keys()) + list(self.technical_strategies.keys()):
            accuracy = self.component_performance.get(name, {}).get("accuracy", 0.5)
            state_features.append(accuracy)
        
        # Pad or truncate to state_size
        if len(state_features) < self.state_size:
            state_features.extend([0.0] * (self.state_size - len(state_features)))
        else:
            state_features = state_features[:self.state_size]
        
        return np.array(state_features, dtype=np.float32)
    
    async def fuse_signals(self, component_signals: List[ComponentSignal],
                          market_context: MarketContext) -> FusionResult:
        """Fuse signals with RL enhancement"""
        try:
            if not self.rl_model:
                logger.warning("RL model not available, using default fusion")
                return await self.default_fusion(component_signals, market_context)
            
            # Get RL state
            state = await self.get_rl_state(component_signals, market_context)
            
            # Get action from RL agent
            action, q_values = await self.rl_model.get_action(state, epsilon=0.1)
            
            # Execute action
            if action == 0:  # buy
                final_signal = SignalType.BUY
                confidence = self.calculate_rl_confidence(q_values, action)
            elif action == 1:  # sell
                final_signal = SignalType.SELL
                confidence = self.calculate_rl_confidence(q_values, action)
            elif action == 2:  # hold
                final_signal = SignalType.HOLD
                confidence = 0.0
            elif action == 3:  # adjust_weights
                # Adjust weights based on RL recommendation
                await self.adjust_weights_rl(component_signals, q_values)
                # Re-fuse with adjusted weights
                return await self.default_fusion(component_signals, market_context)
            else:
                final_signal = SignalType.HOLD
                confidence = 0.0
            
            return FusionResult(
                final_signal=final_signal,
                confidence=confidence,
                component_signals=component_signals,
                fusion_weights=self.fusion_weights,
                market_context=market_context,
                fusion_method=self.fusion_method,
                metadata={
                    "rl_action": action,
                    "q_values": q_values.tolist() if hasattr(q_values, 'tolist') else q_values,
                    "rl_enhanced": True
                }
            )
        
        except Exception as e:
            logger.error(f"Error in RL-enhanced fusion: {e}")
            return await self.default_fusion(component_signals, market_context)
    
    async def default_fusion(self, component_signals: List[ComponentSignal],
                           market_context: MarketContext) -> FusionResult:
        """Default fusion when RL is not available"""
        # Similar to weighted average fusion
        if not component_signals:
            raise ValueError("No component signals to fuse")
        
        signal_values = []
        confidences = []
        
        for signal in component_signals:
            if signal.signal_type == SignalType.BUY:
                value = 1.0
            elif signal.signal_type == SignalType.SELL:
                value = -1.0
            else:
                value = 0.0
            
            signal_values.append(value)
            confidences.append(signal.confidence)
        
        avg_value = np.mean(signal_values) if signal_values else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        if avg_value > 0.3:
            final_signal = SignalType.BUY
            confidence = min(0.95, avg_confidence)
        elif avg_value < -0.3:
            final_signal = SignalType.SELL
            confidence = min(0.95, avg_confidence)
        else:
            final_signal = SignalType.HOLD
            confidence = 0.0
        
        return FusionResult(
            final_signal=final_signal,
            confidence=confidence,
            component_signals=component_signals,
            fusion_weights=self.fusion_weights,
            market_context=market_context,
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            metadata={"default_fusion": True}
        )
    
    def calculate_rl_confidence(self, q_values: np.ndarray, action: int) -> float:
        """Calculate confidence from RL Q-values"""
        if not isinstance(q_values, np.ndarray) or len(q_values) == 0:
            return 0.5
        
        q_value = q_values[action]
        q_min = np.min(q_values)
        q_max = np.max(q_values)
        
        if q_max - q_min > 0:
            confidence = (q_value - q_min) / (q_max - q_min)
        else:
            confidence = 0.5
        
        return float(confidence)
    
    async def adjust_weights_rl(self, component_signals: List[ComponentSignal],
                              q_values: np.ndarray):
        """Adjust fusion weights based on RL recommendation"""
        try:
            # Use Q-values to adjust weights
            for signal in component_signals:
                strategy_name = signal.strategy_name
                if strategy_name in self.fusion_weights.component_weights:
                    # Adjust weight based on signal confidence and Q-value spread
                    current_weight = self.fusion_weights.component_weights[strategy_name]
                    
                    # Q-value spread indicates uncertainty
                    q_spread = np.max(q_values) - np.min(q_values)
                    
                    if q_spread > 0.5:  # High uncertainty
                        # Reduce weight for this strategy
                        new_weight = max(0.01, current_weight * 0.8)
                    else:  # Low uncertainty
                        # Increase weight for this strategy
                        new_weight = min(0.5, current_weight * 1.2)
                    
                    self.fusion_weights.component_weights[strategy_name] = new_weight
            
            # Re-normalize weights
            self.fusion_weights.normalize()
            self.fusion_weights.last_updated = datetime.now()
            
            logger.info(f"RL-adjusted weights: {self.fusion_weights.component_weights}")
        
        except Exception as e:
            logger.error(f"Error adjusting weights with RL: {e}")
    
    async def update_rl_model(self, trade_result: Dict):
        """Update RL model based on trade outcome"""
        try:
            if not self.rl_model:
                return
            
            # Extract state and action from trade
            state = trade_result.get("metadata", {}).get("rl_state")
            action = trade_result.get("metadata", {}).get("rl_action")
            
            if state is None or action is None:
                return
            
            # Calculate reward
            reward = await self.calculate_rl_reward(trade_result)
            
            # Get next state (would be from next observation)
            next_state = np.zeros_like(state)  # Placeholder
            
            # Update RL model
            done = trade_result.get("closed", False)
            
            if hasattr(self.rl_model, 'update'):
                await self.rl_model.update(state, action, reward, next_state, done)
                logger.debug(f"Updated RL model with reward: {reward}")
        
        except Exception as e:
            logger.error(f"Error updating RL model: {e}")
    
    async def calculate_rl_reward(self, trade_result: Dict) -> float:
        """Calculate reward for RL agent"""
        try:
            pnl = trade_result.get("pnl", 0)
            risk = trade_result.get("risk", 0)
            duration = trade_result.get("duration", 0)
            
            # Base reward on PnL
            reward = pnl
            
            # Adjust for risk
            if risk > 0:
                reward = pnl / risk
            
            # Penalize long durations
            if duration > 3600:  # More than 1 hour
                reward *= 0.9
            
            # Bonus for consistency
            recent_trades = self.trades_history[-10:] if len(self.trades_history) >= 10 else self.trades_history
            if recent_trades:
                win_rate = sum(1 for t in recent_trades if t.get("pnl", 0) > 0) / len(recent_trades)
                if win_rate > 0.7:
                    reward *= 1.2
            
            return reward
        
        except Exception as e:
            logger.error(f"Error calculating RL reward: {e}")
            return 0.0

# Additional Hybrid Strategy Classes

class TransformerTechnicalStrategy(BaseHybridStrategy):
    """Transformer-enhanced technical analysis"""
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, HybridStrategyType.TRANSFORMER_TECHNICAL)
    
    async def load_component_strategies(self):
        """Load transformer and technical strategies"""
        # Implementation
        pass
    
    async def fuse_signals(self, component_signals: List[ComponentSignal],
                          market_context: MarketContext) -> FusionResult:
        """Transformer-based fusion"""
        # Implementation
        pass

class MultiModalStrategy(BaseHybridStrategy):
    """Multi-modal fusion incorporating multiple data sources"""
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, HybridStrategyType.MULTI_MODAL)
    
    async def load_component_strategies(self):
        """Load strategies for different data modalities"""
        # Implementation
        pass
    
    async def fuse_signals(self, component_signals: List[ComponentSignal],
                          market_context: MarketContext) -> FusionResult:
        """Multi-modal fusion"""
        # Implementation
        pass

class ContextAwareStrategy(BaseHybridStrategy):
    """Context-aware strategy adapting to market conditions"""
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, HybridStrategyType.CONTEXT_AWARE)
    
    async def load_component_strategies(self):
        """Load context-aware strategies"""
        # Implementation
        pass
    
    async def fuse_signals(self, component_signals: List[ComponentSignal],
                          market_context: MarketContext) -> FusionResult:
        """Context-aware fusion"""
        # Implementation
        pass

class MarketRegimeAdaptiveStrategy(BaseHybridStrategy):
    """Strategy that adapts to different market regimes"""
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, HybridStrategyType.MARKET_REGIME_ADAPTIVE)
    
    async def load_component_strategies(self):
        """Load regime-specific strategies"""
        # Implementation
        pass
    
    async def fuse_signals(self, component_signals: List[ComponentSignal],
                          market_context: MarketContext) -> FusionResult:
        """Regime-adaptive fusion"""
        # Implementation
        pass

# Hybrid Strategy Factory
class HybridStrategyFactory:
    """Factory for creating hybrid trading strategies"""
    
    @staticmethod
    def create_strategy(strategy_type: Union[str, HybridStrategyType], 
                       config_manager: ConfigManager) -> BaseHybridStrategy:
        """Create strategy instance based on type"""
        if isinstance(strategy_type, str):
            try:
                strategy_type = HybridStrategyType(strategy_type.lower())
            except ValueError:
                raise ValueError(f"Unknown hybrid strategy type: {strategy_type}")
        
        if strategy_type == HybridStrategyType.ML_TECHNICAL_FUSION:
            return MLTechnicalFusionStrategy(config_manager)
        
        elif strategy_type == HybridStrategyType.ENSEMBLE_FUSION:
            return EnsembleFusionStrategy(config_manager)
        
        elif strategy_type == HybridStrategyType.ADAPTIVE_WEIGHTING:
            return AdaptiveWeightingStrategy(config_manager)
        
        elif strategy_type == HybridStrategyType.REINFORCEMENT_ENHANCED:
            return ReinforcementEnhancedStrategy(config_manager)
        
        elif strategy_type == HybridStrategyType.TRANSFORMER_TECHNICAL:
            return TransformerTechnicalStrategy(config_manager)
        
        elif strategy_type == HybridStrategyType.MULTI_MODAL:
            return MultiModalStrategy(config_manager)
        
        elif strategy_type == HybridStrategyType.CONTEXT_AWARE:
            return ContextAwareStrategy(config_manager)
        
        elif strategy_type == HybridStrategyType.MARKET_REGIME_ADAPTIVE:
            return MarketRegimeAdaptiveStrategy(config_manager)
        
        else:
            raise ValueError(f"Hybrid strategy type not implemented: {strategy_type}")

# Hybrid Strategy Manager
class HybridStrategyManager:
    """Manages multiple hybrid trading strategies"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.strategies: Dict[str, BaseHybridStrategy] = {}
        self.factory = HybridStrategyFactory()
        self.logger = setup_logger(__name__)
    
    async def create_strategy(self, config: StrategyConfig) -> bool:
        """Create and initialize a new hybrid strategy"""
        try:
            # Check if strategy already exists
            if config.name in self.strategies:
                self.logger.warning(f"Strategy '{config.name}' already exists")
                return False
            
            # Create strategy instance
            strategy = self.factory.create_strategy(config.strategy_type.value, self.config_manager)
            
            # Initialize strategy
            success = await strategy.initialize(config)
            
            if success:
                self.strategies[config.name] = strategy
                self.logger.info(f"Created hybrid strategy '{config.name}' ({config.strategy_type.value})")
                return True
            else:
                self.logger.error(f"Failed to initialize strategy '{config.name}'")
                return False
        
        except Exception as e:
            self.logger.error(f"Error creating hybrid strategy '{config.name}': {e}")
            return False
    
    async def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            self.logger.info(f"Removed strategy '{strategy_name}'")
            return True
        return False
    
    async def get_strategy(self, strategy_name: str) -> Optional[BaseHybridStrategy]:
        """Get strategy by name"""
        return self.strategies.get(strategy_name)
    
    async def get_all_strategies(self) -> Dict[str, BaseHybridStrategy]:
        """Get all strategies"""
        return self.strategies.copy()
    
    async def generate_signals(self, symbol: str, timeframe: str) -> List[TradingSignal]:
        """Generate signals from all active hybrid strategies"""
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
        """Check health of all hybrid strategies"""
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
    """Example of how to use hybrid strategies"""
    # Create config manager
    config_manager = ConfigManager()
    
    # Create hybrid strategy manager
    strategy_manager = HybridStrategyManager(config_manager)
    
    # Create ML-technical fusion strategy config
    fusion_config = StrategyConfig(
        name="btc_hybrid_fusion_v1",
        strategy_type=HybridStrategyType.ML_TECHNICAL_FUSION,
        symbol="BTCUSDT",
        timeframe="1h",
        confidence_threshold=0.65,
        risk_per_trade=1.0,
        parameters={
            "fusion_method": "weighted_average",
            "dynamic_weights": True,
            "confidence_calibration": "adaptive_learning"
        }
    )
    
    # Create ensemble fusion strategy config
    ensemble_config = StrategyConfig(
        name="btc_ensemble_v1",
        strategy_type=HybridStrategyType.ENSEMBLE_FUSION,
        symbol="BTCUSDT",
        timeframe="4h",
        parameters={
            "fusion_method": "stacking",
            "meta_model_type": "neural_network",
            "component_count": 7
        }
    )
    
    # Create and initialize strategies
    await strategy_manager.create_strategy(fusion_config)
    await strategy_manager.create_strategy(ensemble_config)
    
    # Generate signals
    signals = await strategy_manager.generate_signals("BTCUSDT", "1h")
    
    for signal in signals:
        print(f"\nSignal from {signal.strategy_name}:")
        print(f"  Type: {signal.signal_type.value}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Price: ${signal.price:.2f}")
        
        # Show component information
        components = signal.metadata.get("component_signals", [])
        print(f"  Components: {len(components)}")
        for comp in components[:3]:  # Show first 3 components
            print(f"    - {comp['strategy_name']}: {comp['signal_type']} "
                  f"(conf: {comp['confidence']:.2f})")
    
    # Get performance reports
    reports = await strategy_manager.get_performance_reports()
    print(f"\nPerformance Reports:")
    for name, report in reports.items():
        print(f"  {name}:")
        if "component_count" in report:
            print(f"    Components: {report['component_count']['total']} "
                  f"(ML: {report['component_count']['ml']}, "
                  f"Technical: {report['component_count']['technical']})")
    
    # Health check
    health = await strategy_manager.health_check()
    print(f"\nHealth Status:")
    print(f"  Total Strategies: {health['total_strategies']}")
    print(f"  Healthy Strategies: {health['healthy_strategies']}")
    print(f"  Active Strategies: {health['active_strategies']}")
    print(f"  Overall Health: {health['overall_health']:.2f}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())