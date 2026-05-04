"""
Signal generation module for Bitcoin trading AI.
Generates trading signals based on ML model predictions, technical indicators, and market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
from datetime import datetime, timedelta
from scipy import stats, signal
import talib
from collections import deque
import json
from pathlib import Path

# Import project modules
from config.settings import TradingSettings, ModelSettings, AppConstants
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.data_processing.feature_engineer import FeatureEngineer
from core.models.model_predictor import ModelPredictor

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Signal Types and Enums ============
class SignalType(str, Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    WAIT = "wait"

class SignalSource(str, Enum):
    """Sources of trading signals"""
    ML_MODEL = "ml_model"
    TECHNICAL = "technical"
    HYBRID = "hybrid"
    REINFORCEMENT = "reinforcement"
    ENSEMBLE = "ensemble"

class SignalStrength(str, Enum):
    """Strength of trading signals"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class MarketRegime(str, Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

# ============ Configuration ============
@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    
    # General settings
    signal_source: SignalSource = SignalSource.HYBRID
    confidence_threshold: float = 0.7
    min_signal_strength: float = 0.5
    max_position_size: float = 0.1  # 10% of portfolio
    use_stop_loss: bool = True
    use_take_profit: bool = True
    
    # ML Model settings
    ml_model_path: Optional[str] = None
    prediction_horizon: int = 24  # hours
    probability_threshold: float = 0.65
    
    # Technical indicator settings
    use_rsi: bool = True
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    rsi_period: int = 14
    
    use_macd: bool = True
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    use_bollinger: bool = True
    bb_period: int = 20
    bb_std: float = 2.0
    
    use_stochastic: bool = True
    stoch_overbought: int = 80
    stoch_oversold: int = 20
    
    use_adx: bool = True
    adx_threshold: int = 25
    
    # Market regime detection
    detect_market_regime: bool = True
    regime_lookback: int = 50
    trend_threshold: float = 0.02  # 2% movement
    
    # Risk management
    max_drawdown_limit: float = 0.05  # 5%
    volatility_filter: bool = True
    volatility_threshold: float = 0.03  # 3%
    correlation_filter: bool = True
    
    # Signal confirmation
    require_confirmation: bool = True
    confirmation_periods: int = 3
    min_confirmations: int = 2
    
    # Signal filtering
    filter_weak_signals: bool = True
    filter_contradictory_signals: bool = True
    filter_time_based: bool = True
    trading_hours: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 24)])  # 24/7 for crypto
    
    # Signal enhancement
    use_momentum: bool = True
    momentum_period: int = 10
    use_volume_confirmation: bool = True
    volume_threshold: float = 1.5  # 150% of average
    
    # Signal history
    keep_signal_history: bool = True
    history_window: int = 100
    signal_decay_rate: float = 0.1
    
    # Performance tracking
    track_signal_performance: bool = True
    performance_window: int = 1000
    
    def __post_init__(self):
        """Validate configuration"""
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if self.probability_threshold < 0 or self.probability_threshold > 1:
            raise ValueError("probability_threshold must be between 0 and 1")
        
        if self.min_signal_strength < 0 or self.min_signal_strength > 1:
            raise ValueError("min_signal_strength must be between 0 and 1")
        
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError("max_position_size must be between 0 and 1 (exclusive)")

# ============ Signal Data Structures ============
@dataclass
class TradingSignal:
    """Trading signal with all metadata"""
    timestamp: datetime
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    price: float
    source: SignalSource
    metadata: Dict[str, Any] = field(default_factory=dict)
    signal_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    
    def __post_init__(self):
        """Validate signal"""
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'signal_type': self.signal_type.value,
            'strength': self.strength,
            'confidence': self.confidence,
            'price': self.price,
            'source': self.source.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            signal_type=SignalType(data['signal_type']),
            strength=data['strength'],
            confidence=data['confidence'],
            price=data['price'],
            source=SignalSource(data['source']),
            metadata=data['metadata'],
            signal_id=data['signal_id']
        )

@dataclass
class SignalAnalysis:
    """Analysis of generated signals"""
    signals_generated: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    avg_strength: float
    avg_confidence: float
    signal_distribution: Dict[SignalType, int]
    source_distribution: Dict[SignalSource, int]
    performance_metrics: Optional[Dict[str, float]] = None

@dataclass
class MarketCondition:
    """Current market conditions"""
    regime: MarketRegime
    trend_strength: float
    volatility: float
    volume_ratio: float
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    rsi_value: Optional[float] = None
    macd_value: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

# ============ Base Signal Generator ============
class BaseSignalGenerator:
    """Base class for signal generators"""
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.signal_history: List[TradingSignal] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.market_conditions: List[MarketCondition] = []
        self.current_market_regime: Optional[MarketRegime] = None
        self.feature_engineer = FeatureEngineer()
        self.logger = get_logger(__name__)
        
        # Initialize ML predictor if configured
        self.ml_predictor = None
        if self.config.ml_model_path and Path(self.config.ml_model_path).exists():
            try:
                from core.models.model_predictor import ModelPredictor
                self.ml_predictor = ModelPredictor(self.config.ml_model_path)
                self.logger.info(f"Loaded ML model from {self.config.ml_model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load ML model: {str(e)}")
    
    def generate_signals(self, market_data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals from market data"""
        raise NotImplementedError
    
    def analyze_market(self, market_data: pd.DataFrame) -> MarketCondition:
        """Analyze current market conditions"""
        raise NotImplementedError
    
    def filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals based on various criteria"""
        raise NotImplementedError
    
    def enhance_signals(self, signals: List[TradingSignal], 
                       market_data: pd.DataFrame) -> List[TradingSignal]:
        """Enhance signals with additional analysis"""
        raise NotImplementedError
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate if a signal should be executed"""
        raise NotImplementedError
    
    def track_performance(self, signal: TradingSignal, 
                         outcome: Optional[Dict[str, Any]] = None):
        """Track signal performance"""
        raise NotImplementedError

# ============ Technical Signal Generator ============
class TechnicalSignalGenerator(BaseSignalGenerator):
    """Generates signals based on technical indicators"""
    
    def __init__(self, config: Optional[SignalConfig] = None):
        super().__init__(config)
        self.indicator_cache = {}
    
    def generate_signals(self, market_data: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals from technical indicators"""
        self.logger.info("Generating technical signals...")
        
        signals = []
        
        # Analyze market conditions first
        market_condition = self.analyze_market(market_data)
        self.market_conditions.append(market_condition)
        self.current_market_regime = market_condition.regime
        
        # Generate signals from each indicator
        if self.config.use_rsi:
            signals.extend(self._generate_rsi_signals(market_data, market_condition))
        
        if self.config.use_macd:
            signals.extend(self._generate_macd_signals(market_data, market_condition))
        
        if self.config.use_bollinger:
            signals.extend(self._generate_bollinger_signals(market_data, market_condition))
        
        if self.config.use_stochastic:
            signals.extend(self._generate_stochastic_signals(market_data, market_condition))
        
        # Combine and filter signals
        combined_signals = self._combine_technical_signals(signals, market_data)
        filtered_signals = self.filter_signals(combined_signals)
        enhanced_signals = self.enhance_signals(filtered_signals, market_data)
        
        # Add to history
        if self.config.keep_signal_history:
            self.signal_history.extend(enhanced_signals)
            # Keep only recent history
            if len(self.signal_history) > self.config.history_window:
                self.signal_history = self.signal_history[-self.config.history_window:]
        
        self.logger.info(f"Generated {len(enhanced_signals)} technical signals")
        
        return enhanced_signals
    
    def _generate_rsi_signals(self, data: pd.DataFrame, 
                            market_condition: MarketCondition) -> List[TradingSignal]:
        """Generate signals based on RSI"""
        signals = []
        
        try:
            # Calculate RSI
            rsi = talib.RSI(data['close'], timeperiod=self.config.rsi_period)
            
            if len(rsi) > 0:
                current_rsi = rsi.iloc[-1]
                previous_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi
                
                # Update market condition
                market_condition.rsi_value = current_rsi
                
                # Generate signals
                current_price = data['close'].iloc[-1]
                
                # Oversold condition
                if current_rsi < self.config.rsi_oversold:
                    # Bullish divergence check
                    if previous_rsi < current_rsi:  # RSI rising from oversold
                        strength = self._calculate_rsi_strength(current_rsi, oversold=True)
                        signal = TradingSignal(
                            timestamp=data.index[-1],
                            signal_type=SignalType.BUY,
                            strength=strength,
                            confidence=0.7,
                            price=current_price,
                            source=SignalSource.TECHNICAL,
                            metadata={
                                'indicator': 'RSI',
                                'rsi_value': current_rsi,
                                'condition': 'oversold',
                                'divergence': 'bullish'
                            }
                        )
                        signals.append(signal)
                
                # Overbought condition
                elif current_rsi > self.config.rsi_overbought:
                    # Bearish divergence check
                    if previous_rsi > current_rsi:  # RSI falling from overbought
                        strength = self._calculate_rsi_strength(current_rsi, oversold=False)
                        signal = TradingSignal(
                            timestamp=data.index[-1],
                            signal_type=SignalType.SELL,
                            strength=strength,
                            confidence=0.7,
                            price=current_price,
                            source=SignalSource.TECHNICAL,
                            metadata={
                                'indicator': 'RSI',
                                'rsi_value': current_rsi,
                                'condition': 'overbought',
                                'divergence': 'bearish'
                            }
                        )
                        signals.append(signal)
        
        except Exception as e:
            self.logger.error(f"Error generating RSI signals: {str(e)}")
        
        return signals
    
    def _calculate_rsi_strength(self, rsi_value: float, oversold: bool) -> float:
        """Calculate signal strength based on RSI"""
        if oversold:
            # Further from 30 is stronger (more oversold)
            strength = (self.config.rsi_oversold - rsi_value) / self.config.rsi_oversold
        else:
            # Further from 70 is stronger (more overbought)
            strength = (rsi_value - self.config.rsi_overbought) / (100 - self.config.rsi_overbought)
        
        return min(max(strength, 0), 1)
    
    def _generate_macd_signals(self, data: pd.DataFrame,
                             market_condition: MarketCondition) -> List[TradingSignal]:
        """Generate signals based on MACD"""
        signals = []
        
        try:
            # Calculate MACD
            macd, macd_signal, macd_hist = talib.MACD(
                data['close'],
                fastperiod=self.config.macd_fast,
                slowperiod=self.config.macd_slow,
                signalperiod=self.config.macd_signal
            )
            
            if len(macd) > 1:
                current_macd = macd.iloc[-1]
                current_signal = macd_signal.iloc[-1]
                current_hist = macd_hist.iloc[-1]
                previous_macd = macd.iloc[-2]
                previous_hist = macd_hist.iloc[-2]
                
                # Update market condition
                market_condition.macd_value = current_macd
                
                # Generate signals
                current_price = data['close'].iloc[-1]
                
                # MACD line crossover
                if previous_macd <= previous_hist and current_macd > current_signal:
                    # Bullish crossover
                    signal = TradingSignal(
                        timestamp=data.index[-1],
                        signal_type=SignalType.BUY,
                        strength=0.6,
                        confidence=0.65,
                        price=current_price,
                        source=SignalSource.TECHNICAL,
                        metadata={
                            'indicator': 'MACD',
                            'macd_value': current_macd,
                            'signal_value': current_signal,
                            'histogram': current_hist,
                            'condition': 'bullish_crossover'
                        }
                    )
                    signals.append(signal)
                
                elif previous_macd >= previous_hist and current_macd < current_signal:
                    # Bearish crossover
                    signal = TradingSignal(
                        timestamp=data.index[-1],
                        signal_type=SignalType.SELL,
                        strength=0.6,
                        confidence=0.65,
                        price=current_price,
                        source=SignalSource.TECHNICAL,
                        metadata={
                            'indicator': 'MACD',
                            'macd_value': current_macd,
                            'signal_value': current_signal,
                            'histogram': current_hist,
                            'condition': 'bearish_crossover'
                        }
                    )
                    signals.append(signal)
                
                # Zero line crossover
                if previous_macd <= 0 and current_macd > 0:
                    # Bullish zero line crossover
                    signal = TradingSignal(
                        timestamp=data.index[-1],
                        signal_type=SignalType.STRONG_BUY,
                        strength=0.8,
                        confidence=0.75,
                        price=current_price,
                        source=SignalSource.TECHNICAL,
                        metadata={
                            'indicator': 'MACD',
                            'condition': 'zero_line_crossover_bullish'
                        }
                    )
                    signals.append(signal)
                
                elif previous_macd >= 0 and current_macd < 0:
                    # Bearish zero line crossover
                    signal = TradingSignal(
                        timestamp=data.index[-1],
                        signal_type=SignalType.STRONG_SELL,
                        strength=0.8,
                        confidence=0.75,
                        price=current_price,
                        source=SignalSource.TECHNICAL,
                        metadata={
                            'indicator': 'MACD',
                            'condition': 'zero_line_crossover_bearish'
                        }
                    )
                    signals.append(signal)
        
        except Exception as e:
            self.logger.error(f"Error generating MACD signals: {str(e)}")
        
        return signals
    
    def _generate_bollinger_signals(self, data: pd.DataFrame,
                                  market_condition: MarketCondition) -> List[TradingSignal]:
        """Generate signals based on Bollinger Bands"""
        signals = []
        
        try:
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                data['close'],
                timeperiod=self.config.bb_period,
                nbdevup=self.config.bb_std,
                nbdevdn=self.config.bb_std
            )
            
            if len(upper) > 0:
                current_price = data['close'].iloc[-1]
                current_upper = upper.iloc[-1]
                current_lower = lower.iloc[-1]
                previous_price = data['close'].iloc[-2] if len(data) > 1 else current_price
                
                # Band width and %B
                band_width = (current_upper - current_lower) / middle.iloc[-1]
                percent_b = (current_price - current_lower) / (current_upper - current_lower)
                
                # Support and resistance levels
                market_condition.support_level = current_lower
                market_condition.resistance_level = current_upper
                
                # Generate signals
                
                # Price touches lower band (potential bounce)
                if current_price <= current_lower * 1.01:  # Within 1% of lower band
                    signal = TradingSignal(
                        timestamp=data.index[-1],
                        signal_type=SignalType.BUY,
                        strength=0.7,
                        confidence=0.6,
                        price=current_price,
                        source=SignalSource.TECHNICAL,
                        metadata={
                            'indicator': 'Bollinger',
                            'condition': 'touch_lower_band',
                            'percent_b': percent_b,
                            'band_width': band_width
                        }
                    )
                    signals.append(signal)
                
                # Price touches upper band (potential reversal)
                elif current_price >= current_upper * 0.99:  # Within 1% of upper band
                    signal = TradingSignal(
                        timestamp=data.index[-1],
                        signal_type=SignalType.SELL,
                        strength=0.7,
                        confidence=0.6,
                        price=current_price,
                        source=SignalSource.TECHNICAL,
                        metadata={
                            'indicator': 'Bollinger',
                            'condition': 'touch_upper_band',
                            'percent_b': percent_b,
                            'band_width': band_width
                        }
                    )
                    signals.append(signal)
                
                # Squeeze breakout (band width narrow)
                if band_width < 0.1:  # Narrow bands
                    # Breakout direction
                    if current_price > previous_price:
                        signal = TradingSignal(
                            timestamp=data.index[-1],
                            signal_type=SignalType.BUY,
                            strength=0.8,
                            confidence=0.7,
                            price=current_price,
                            source=SignalSource.TECHNICAL,
                            metadata={
                                'indicator': 'Bollinger',
                                'condition': 'squeeze_breakout_bullish',
                                'band_width': band_width
                            }
                        )
                        signals.append(signal)
                    elif current_price < previous_price:
                        signal = TradingSignal(
                            timestamp=data.index[-1],
                            signal_type=SignalType.SELL,
                            strength=0.8,
                            confidence=0.7,
                            price=current_price,
                            source=SignalSource.TECHNICAL,
                            metadata={
                                'indicator': 'Bollinger',
                                'condition': 'squeeze_breakout_bearish',
                                'band_width': band_width
                            }
                        )
                        signals.append(signal)
        
        except Exception as e:
            self.logger.error(f"Error generating Bollinger signals: {str(e)}")
        
        return signals
    
    def _generate_stochastic_signals(self, data: pd.DataFrame,
                                   market_condition: MarketCondition) -> List[TradingSignal]:
        """Generate signals based on Stochastic Oscillator"""
        signals = []
        
        try:
            # Calculate Stochastic
            slowk, slowd = talib.STOCH(
                data['high'],
                data['low'],
                data['close'],
                fastk_period=self.config.stoch_period,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            
            if len(slowk) > 1:
                current_k = slowk.iloc[-1]
                current_d = slowd.iloc[-1]
                previous_k = slowk.iloc[-2]
                previous_d = slowd.iloc[-2]
                
                current_price = data['close'].iloc[-1]
                
                # Oversold condition
                if current_k < self.config.stoch_oversold and current_d < self.config.stoch_oversold:
                    # Bullish crossover
                    if previous_k <= previous_d and current_k > current_d:
                        signal = TradingSignal(
                            timestamp=data.index[-1],
                            signal_type=SignalType.BUY,
                            strength=0.7,
                            confidence=0.65,
                            price=current_price,
                            source=SignalSource.TECHNICAL,
                            metadata={
                                'indicator': 'Stochastic',
                                'k_value': current_k,
                                'd_value': current_d,
                                'condition': 'oversold_crossover'
                            }
                        )
                        signals.append(signal)
                
                # Overbought condition
                elif current_k > self.config.stoch_overbought and current_d > self.config.stoch_overbought:
                    # Bearish crossover
                    if previous_k >= previous_d and current_k < current_d:
                        signal = TradingSignal(
                            timestamp=data.index[-1],
                            signal_type=SignalType.SELL,
                            strength=0.7,
                            confidence=0.65,
                            price=current_price,
                            source=SignalSource.TECHNICAL,
                            metadata={
                                'indicator': 'Stochastic',
                                'k_value': current_k,
                                'd_value': current_d,
                                'condition': 'overbought_crossover'
                            }
                        )
                        signals.append(signal)
        
        except Exception as e:
            self.logger.error(f"Error generating Stochastic signals: {str(e)}")
        
        return signals
    
    def _combine_technical_signals(self, signals: List[TradingSignal],
                                 data: pd.DataFrame) -> List[TradingSignal]:
        """Combine multiple technical signals"""
        if not signals:
            return []
        
        # Group signals by type and timestamp
        signal_groups = {}
        for signal in signals:
            key = (signal.timestamp, signal.signal_type)
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)
        
        # Combine signals in each group
        combined_signals = []
        for (timestamp, signal_type), signal_list in signal_groups.items():
            if len(signal_list) == 1:
                combined_signals.append(signal_list[0])
            else:
                # Average strength and confidence
                avg_strength = np.mean([s.strength for s in signal_list])
                avg_confidence = np.mean([s.confidence for s in signal_list])
                
                # Combine metadata
                combined_metadata = {
                    'combined_from': len(signal_list),
                    'sources': [s.metadata.get('indicator', 'unknown') for s in signal_list]
                }
                
                # Take the most recent price
                price = signal_list[-1].price
                
                # Create combined signal
                combined_signal = TradingSignal(
                    timestamp=timestamp,
                    signal_type=signal_type,
                    strength=avg_strength,
                    confidence=avg_confidence,
                    price=price,
                    source=SignalSource.TECHNICAL,
                    metadata=combined_metadata
                )
                combined_signals.append(combined_signal)
        
        return combined_signals
    
    def analyze_market(self, market_data: pd.DataFrame) -> MarketCondition:
        """Analyze current market conditions"""
        try:
            # Calculate basic metrics
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(window=20).std().iloc[-1] if len(returns) >= 20 else 0
            
            # Volume analysis
            avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1] if len(market_data) >= 20 else 0
            current_volume = market_data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Trend analysis
            short_ma = market_data['close'].rolling(window=20).mean().iloc[-1] if len(market_data) >= 20 else market_data['close'].iloc[-1]
            long_ma = market_data['close'].rolling(window=50).mean().iloc[-1] if len(market_data) >= 50 else market_data['close'].iloc[-1]
            
            trend_strength = abs(short_ma - long_ma) / long_ma
            
            # Determine market regime
            regime = self._determine_market_regime(market_data, trend_strength, volatility)
            
            # Calculate ADX if configured
            adx_value = None
            if self.config.use_adx and len(market_data) >= self.config.adx_threshold:
                adx = talib.ADX(market_data['high'], market_data['low'], market_data['close'])
                adx_value = adx.iloc[-1] if len(adx) > 0 else None
            
            market_condition = MarketCondition(
                regime=regime,
                trend_strength=trend_strength,
                volatility=volatility,
                volume_ratio=volume_ratio,
                rsi_value=None,  # Will be set by specific indicator methods
                macd_value=None,
                timestamp=market_data.index[-1]
            )
            
            return market_condition
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            # Return default market condition
            return MarketCondition(
                regime=MarketRegime.RANGING,
                trend_strength=0.0,
                volatility=0.0,
                volume_ratio=1.0,
                timestamp=datetime.now()
            )
    
    def _determine_market_regime(self, data: pd.DataFrame, 
                               trend_strength: float, 
                               volatility: float) -> MarketRegime:
        """Determine current market regime"""
        
        # Check for breakout
        recent_high = data['high'].rolling(window=20).max().iloc[-1]
        recent_low = data['low'].rolling(window=20).min().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Breakout detection
        if current_price > recent_high * 1.01:  # 1% above recent high
            return MarketRegime.BREAKOUT
        elif current_price < recent_low * 0.99:  # 1% below recent low
            return MarketRegime.BREAKOUT
        
        # Trend detection
        if trend_strength > self.config.trend_threshold:
            short_ma = data['close'].rolling(window=20).mean().iloc[-1]
            long_ma = data['close'].rolling(window=50).mean().iloc[-1]
            
            if short_ma > long_ma:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        
        # Volatility detection
        if volatility > self.config.volatility_threshold:
            return MarketRegime.VOLATILE
        
        # Default to ranging
        return MarketRegime.RANGING
    
    def filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals based on configuration"""
        if not signals:
            return []
        
        filtered_signals = []
        
        for signal in signals:
            # Apply basic filters
            if self._apply_basic_filters(signal):
                filtered_signals.append(signal)
        
        # Apply advanced filters
        if self.config.filter_contradictory_signals:
            filtered_signals = self._filter_contradictory_signals(filtered_signals)
        
        if self.config.filter_time_based:
            filtered_signals = self._filter_time_based_signals(filtered_signals)
        
        # Filter by strength threshold
        if self.config.filter_weak_signals:
            filtered_signals = [
                s for s in filtered_signals 
                if s.strength >= self.config.min_signal_strength
            ]
        
        self.logger.debug(f"Filtered {len(signals)} -> {len(filtered_signals)} signals")
        
        return filtered_signals
    
    def _apply_basic_filters(self, signal: TradingSignal) -> bool:
        """Apply basic signal filters"""
        # Strength threshold
        if signal.strength < self.config.min_signal_strength:
            return False
        
        # Confidence threshold
        if signal.confidence < self.config.confidence_threshold:
            return False
        
        # Market regime filter
        if self.current_market_regime:
            regime = self.current_market_regime
            
            # Adjust filters based on market regime
            if regime == MarketRegime.TRENDING_UP:
                # Favor buy signals in uptrend
                if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    signal.confidence *= 0.8  # Reduce confidence for counter-trend signals
            elif regime == MarketRegime.TRENDING_DOWN:
                # Favor sell signals in downtrend
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    signal.confidence *= 0.8
        
        return True
    
    def _filter_contradictory_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter out contradictory signals"""
        if len(signals) <= 1:
            return signals
        
        # Group by signal type
        buy_signals = [s for s in signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]]
        sell_signals = [s for s in signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]]
        
        # If we have both buy and sell signals, keep the stronger ones
        if buy_signals and sell_signals:
            avg_buy_strength = np.mean([s.strength * s.confidence for s in buy_signals])
            avg_sell_strength = np.mean([s.strength * s.confidence for s in sell_signals])
            
            if avg_buy_strength > avg_sell_strength:
                return buy_signals
            else:
                return sell_signals
        
        return signals
    
    def _filter_time_based_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals based on trading hours"""
        if not self.config.trading_hours:
            return signals  # 24/7 trading
        
        filtered_signals = []
        current_hour = signals[0].timestamp.hour if signals else datetime.now().hour
        
        for time_range in self.config.trading_hours:
            start_hour, end_hour = time_range
            if start_hour <= current_hour < end_hour:
                return signals  # Within trading hours
        
        # Outside trading hours, only keep strong signals
        filtered_signals = [
            s for s in signals 
            if s.strength >= 0.8 and s.confidence >= 0.8
        ]
        
        return filtered_signals
    
    def enhance_signals(self, signals: List[TradingSignal], 
                       market_data: pd.DataFrame) -> List[TradingSignal]:
        """Enhance signals with additional analysis"""
        if not signals:
            return []
        
        enhanced_signals = []
        
        for signal in signals:
            enhanced_signal = self._enhance_single_signal(signal, market_data)
            enhanced_signals.append(enhanced_signal)
        
        return enhanced_signals
    
    def _enhance_single_signal(self, signal: TradingSignal, 
                             market_data: pd.DataFrame) -> TradingSignal:
        """Enhance a single signal"""
        enhanced_signal = signal
        
        # Add momentum analysis
        if self.config.use_momentum:
            momentum = self._calculate_momentum(market_data, self.config.momentum_period)
            if momentum is not None:
                # Adjust signal based on momentum
                if (signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] and momentum > 0) or \
                   (signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL] and momentum < 0):
                    # Momentum confirms signal
                    enhanced_signal.confidence *= 1.1
                    enhanced_signal.metadata['momentum_confirmation'] = True
                else:
                    # Momentum contradicts signal
                    enhanced_signal.confidence *= 0.9
                    enhanced_signal.metadata['momentum_confirmation'] = False
        
        # Add volume confirmation
        if self.config.use_volume_confirmation:
            volume_confirm = self._check_volume_confirmation(market_data, signal.signal_type)
            if volume_confirm:
                enhanced_signal.confidence *= 1.05
                enhanced_signal.metadata['volume_confirmation'] = True
        
        # Add market regime context
        if self.current_market_regime:
            enhanced_signal.metadata['market_regime'] = self.current_market_regime.value
            
            # Adjust based on regime
            regime = self.current_market_regime
            if regime == MarketRegime.TRENDING_UP and signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                enhanced_signal.confidence *= 1.05
            elif regime == MarketRegime.TRENDING_DOWN and signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                enhanced_signal.confidence *= 1.05
        
        # Cap confidence at 1.0
        enhanced_signal.confidence = min(enhanced_signal.confidence, 1.0)
        
        return enhanced_signal
    
    def _calculate_momentum(self, data: pd.DataFrame, period: int) -> Optional[float]:
        """Calculate price momentum"""
        if len(data) < period:
            return None
        
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-period]
        
        return (current_price - past_price) / past_price
    
    def _check_volume_confirmation(self, data: pd.DataFrame, 
                                 signal_type: SignalType) -> bool:
        """Check if volume confirms the signal"""
        if len(data) < 20:
            return False
        
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # For buy signals, we want above-average volume
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return volume_ratio > self.config.volume_threshold
        
        # For sell signals, volume confirmation is less critical but still helpful
        return volume_ratio > 1.0
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate if a signal should be executed"""
        # Check basic criteria
        if signal.strength < self.config.min_signal_strength:
            return False
        
        if signal.confidence < self.config.confidence_threshold:
            return False
        
        # Check for recent similar signals
        if self.config.keep_signal_history:
            recent_signals = [
                s for s in self.signal_history[-10:]  # Last 10 signals
                if s.signal_type == signal.signal_type
                and abs((s.timestamp - signal.timestamp).total_seconds()) < 3600  # Within 1 hour
            ]
            
            if len(recent_signals) >= 3:
                self.logger.warning(f"Too many similar recent signals: {len(recent_signals)}")
                return False
        
        # Additional validation based on signal type
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return self._validate_buy_signal(signal)
        elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return self._validate_sell_signal(signal)
        
        return True
    
    def _validate_buy_signal(self, signal: TradingSignal) -> bool:
        """Validate buy signal"""
        # Additional buy signal validation logic
        metadata = signal.metadata
        
        # Check for oversold conditions in metadata
        if 'condition' in metadata:
            condition = metadata['condition']
            if 'oversold' in condition.lower() or 'lower' in condition.lower():
                return True
        
        return signal.confidence >= 0.6
    
    def _validate_sell_signal(self, signal: TradingSignal) -> bool:
        """Validate sell signal"""
        # Additional sell signal validation logic
        metadata = signal.metadata
        
        # Check for overbought conditions in metadata
        if 'condition' in metadata:
            condition = metadata['condition']
            if 'overbought' in condition.lower() or 'upper' in condition.lower():
                return True
        
        return signal.confidence >= 0.6

# ============ ML Signal Generator ============
class MLSignalGenerator(BaseSignalGenerator):
    """Generates signals based on ML model predictions"""
    
    def __init__(self, config: Optional[SignalConfig] = None):
        super().__init__(config)
        
        if not self.ml_predictor:
            raise ValueError("ML model path must be configured for MLSignalGenerator")
        
        self.prediction_history = []
        self.model_confidence_history = []
    
    def generate_signals(self, market_data: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals from ML model predictions"""
        self.logger.info("Generating ML signals...")
        
        signals = []
        
        try:
            # Prepare features for prediction
            features = self._prepare_features(market_data)
            
            if features is None or len(features) == 0:
                self.logger.warning("No features prepared for prediction")
                return signals
            
            # Get predictions from ML model
            predictions = self.ml_predictor.predict(features)
            
            if predictions is None:
                return signals
            
            # Convert predictions to signals
            current_price = market_data['close'].iloc[-1]
            timestamp = market_data.index[-1]
            
            # Handle different prediction formats
            if isinstance(predictions, np.ndarray):
                if predictions.ndim == 1:
                    # Single prediction
                    signals.extend(self._process_single_prediction(
                        predictions[0], current_price, timestamp, features
                    ))
                elif predictions.ndim == 2:
                    # Multiple predictions
                    for i, pred in enumerate(predictions):
                        signals.extend(self._process_single_prediction(
                            pred, current_price, timestamp, features, i
                        ))
            elif isinstance(predictions, dict):
                # Dictionary of predictions
                for key, pred in predictions.items():
                    signals.extend(self._process_single_prediction(
                        pred, current_price, timestamp, features, key
                    ))
            
            # Track predictions
            self.prediction_history.append({
                'timestamp': timestamp,
                'predictions': predictions,
                'features': features.shape,
                'price': current_price
            })
            
            # Filter and enhance signals
            filtered_signals = self.filter_signals(signals)
            enhanced_signals = self.enhance_signals(filtered_signals, market_data)
            
            # Add to history
            if self.config.keep_signal_history:
                self.signal_history.extend(enhanced_signals)
                if len(self.signal_history) > self.config.history_window:
                    self.signal_history = self.signal_history[-self.config.history_window:]
            
            self.logger.info(f"Generated {len(enhanced_signals)} ML signals")
            
            return enhanced_signals
            
        except Exception as e:
            self.logger.error(f"Error generating ML signals: {str(e)}")
            return []
    
    def _prepare_features(self, market_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for ML model prediction"""
        try:
            # Use feature engineer to create features
            features = self.feature_engineer.engineer_features(market_data)
            
            # Select relevant features for the model
            # This should be adapted based on the specific model requirements
            required_features = [
                'close', 'volume', 'returns', 'volatility_20',
                'rsi_14', 'macd', 'bb_position'
            ]
            
            # Ensure we have required features
            available_features = [f for f in required_features if f in features.columns]
            
            if len(available_features) == 0:
                self.logger.warning("No required features available")
                return None
            
            # Take only the latest row for prediction
            latest_features = features[available_features].iloc[[-1]]
            
            return latest_features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None
    
    def _process_single_prediction(self, prediction: Any, 
                                 current_price: float,
                                 timestamp: datetime,
                                 features: pd.DataFrame,
                                 prediction_id: Optional[Any] = None) -> List[TradingSignal]:
        """Process a single prediction into trading signals"""
        signals = []
        
        try:
            # Determine prediction type and convert to signal
            if isinstance(prediction, (int, np.integer)):
                # Classification prediction (0: sell, 1: hold, 2: buy)
                signal_type = self._convert_class_to_signal(prediction)
                confidence = 0.7  # Default confidence for classification
                
                signal = TradingSignal(
                    timestamp=timestamp,
                    signal_type=signal_type,
                    strength=0.6,
                    confidence=confidence,
                    price=current_price,
                    source=SignalSource.ML_MODEL,
                    metadata={
                        'prediction_type': 'classification',
                        'prediction_value': int(prediction),
                        'prediction_id': str(prediction_id),
                        'features_shape': features.shape
                    }
                )
                signals.append(signal)
                
            elif isinstance(prediction, (float, np.floating)):
                # Regression prediction (price change or return)
                signal_type, strength = self._convert_regression_to_signal(prediction)
                
                signal = TradingSignal(
                    timestamp=timestamp,
                    signal_type=signal_type,
                    strength=strength,
                    confidence=0.65,
                    price=current_price,
                    source=SignalSource.ML_MODEL,
                    metadata={
                        'prediction_type': 'regression',
                        'prediction_value': float(prediction),
                        'prediction_id': str(prediction_id),
                        'expected_return': float(prediction)
                    }
                )
                signals.append(signal)
                
            elif isinstance(prediction, dict):
                # Dictionary with probability distribution
                signals.extend(self._process_probability_distribution(
                    prediction, current_price, timestamp, prediction_id
                ))
                
            elif isinstance(prediction, np.ndarray) and prediction.ndim == 1:
                # Probability array [sell_prob, hold_prob, buy_prob]
                if len(prediction) >= 3:
                    probs_dict = {
                        'sell': prediction[0],
                        'hold': prediction[1],
                        'buy': prediction[2]
                    }
                    signals.extend(self._process_probability_distribution(
                        probs_dict, current_price, timestamp, prediction_id
                    ))
        
        except Exception as e:
            self.logger.error(f"Error processing prediction: {str(e)}")
        
        return signals
    
    def _convert_class_to_signal(self, prediction_class: int) -> SignalType:
        """Convert classification prediction to signal type"""
        if prediction_class == 0:
            return SignalType.SELL
        elif prediction_class == 1:
            return SignalType.HOLD
        elif prediction_class == 2:
            return SignalType.BUY
        else:
            return SignalType.HOLD
    
    def _convert_regression_to_signal(self, prediction_value: float) -> Tuple[SignalType, float]:
        """Convert regression prediction to signal type and strength"""
        # Assuming prediction_value is expected return
        if prediction_value > 0.01:  # 1% positive return
            strength = min(abs(prediction_value) * 10, 1.0)
            return SignalType.BUY, strength
        elif prediction_value < -0.01:  # 1% negative return
            strength = min(abs(prediction_value) * 10, 1.0)
            return SignalType.SELL, strength
        else:
            return SignalType.HOLD, 0.3
    
    def _process_probability_distribution(self, probs: Dict[str, float],
                                        current_price: float,
                                        timestamp: datetime,
                                        prediction_id: Any) -> List[TradingSignal]:
        """Process probability distribution into signals"""
        signals = []
        
        try:
            # Get highest probability
            max_action = max(probs.items(), key=lambda x: x[1])
            action_name, probability = max_action
            
            # Convert to signal type
            if action_name.lower() == 'buy':
                signal_type = SignalType.BUY
            elif action_name.lower() == 'sell':
                signal_type = SignalType.SELL
            elif action_name.lower() == 'hold':
                signal_type = SignalType.HOLD
            else:
                signal_type = SignalType.HOLD
            
            # Calculate confidence
            confidence = probability
            
            # Only generate signal if probability exceeds threshold
            if confidence >= self.config.probability_threshold:
                # Calculate strength based on probability difference
                other_probs = [p for k, p in probs.items() if k != action_name]
                if other_probs:
                    max_other_prob = max(other_probs)
                    strength = (probability - max_other_prob) / probability
                else:
                    strength = probability
                
                signal = TradingSignal(
                    timestamp=timestamp,
                    signal_type=signal_type,
                    strength=strength,
                    confidence=confidence,
                    price=current_price,
                    source=SignalSource.ML_MODEL,
                    metadata={
                        'prediction_type': 'probability',
                        'probabilities': probs,
                        'max_probability': probability,
                        'prediction_id': str(prediction_id)
                    }
                )
                signals.append(signal)
        
        except Exception as e:
            self.logger.error(f"Error processing probability distribution: {str(e)}")
        
        return signals
    
    def analyze_market(self, market_data: pd.DataFrame) -> MarketCondition:
        """Analyze market conditions using ML model if available"""
        # Use parent class implementation for basic analysis
        return super().analyze_market(market_data)

# ============ Hybrid Signal Generator ============
class HybridSignalGenerator(BaseSignalGenerator):
    """Generates signals by combining multiple sources"""
    
    def __init__(self, config: Optional[SignalConfig] = None):
        super().__init__(config)
        
        # Initialize component generators
        self.technical_generator = TechnicalSignalGenerator(config)
        self.ml_generator = None
        
        if self.config.ml_model_path and Path(self.config.ml_model_path).exists():
            try:
                self.ml_generator = MLSignalGenerator(config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize ML generator: {str(e)}")
        
        # Weighting for different sources
        self.source_weights = {
            SignalSource.TECHNICAL: 0.4,
            SignalSource.ML_MODEL: 0.6
        }
        
        # Performance tracking by source
        self.source_performance = {source: [] for source in self.source_weights.keys()}
    
    def generate_signals(self, market_data: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals by combining multiple sources"""
        self.logger.info("Generating hybrid signals...")
        
        all_signals = []
        
        # Generate signals from each source
        if self.config.signal_source in [SignalSource.HYBRID, SignalSource.TECHNICAL]:
            technical_signals = self.technical_generator.generate_signals(market_data)
            all_signals.extend(technical_signals)
        
        if self.config.signal_source in [SignalSource.HYBRID, SignalSource.ML_MODEL] and self.ml_generator:
            ml_signals = self.ml_generator.generate_signals(market_data)
            all_signals.extend(ml_signals)
        
        # Combine signals from different sources
        combined_signals = self._combine_source_signals(all_signals)
        
        # Filter and enhance
        filtered_signals = self.filter_signals(combined_signals)
        enhanced_signals = self.enhance_signals(filtered_signals, market_data)
        
        # Add to history
        if self.config.keep_signal_history:
            self.signal_history.extend(enhanced_signals)
            if len(self.signal_history) > self.config.history_window:
                self.signal_history = self.signal_history[-self.config.history_window:]
        
        self.logger.info(f"Generated {len(enhanced_signals)} hybrid signals")
        
        return enhanced_signals
    
    def _combine_source_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Combine signals from different sources"""
        if not signals:
            return []
        
        # Group signals by type and timestamp
        signal_groups = {}
        for signal in signals:
            key = (signal.timestamp, signal.signal_type)
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)
        
        # Combine signals in each group
        combined_signals = []
        
        for (timestamp, signal_type), signal_list in signal_groups.items():
            if len(signal_list) == 1:
                combined_signals.append(signal_list[0])
            else:
                # Weighted combination based on source
                total_weight = 0
                weighted_strength = 0
                weighted_confidence = 0
                combined_metadata = {
                    'combined_from': len(signal_list),
                    'sources': []
                }
                
                for signal in signal_list:
                    weight = self.source_weights.get(signal.source, 0.5)
                    total_weight += weight
                    weighted_strength += signal.strength * weight
                    weighted_confidence += signal.confidence * weight
                    combined_metadata['sources'].append(signal.source.value)
                
                if total_weight > 0:
                    avg_strength = weighted_strength / total_weight
                    avg_confidence = weighted_confidence / total_weight
                else:
                    avg_strength = np.mean([s.strength for s in signal_list])
                    avg_confidence = np.mean([s.confidence for s in signal_list])
                
                # Take the most recent price
                price = signal_list[-1].price
                
                # Determine combined source
                if len(set([s.source for s in signal_list])) > 1:
                    combined_source = SignalSource.HYBRID
                else:
                    combined_source = signal_list[0].source
                
                combined_signal = TradingSignal(
                    timestamp=timestamp,
                    signal_type=signal_type,
                    strength=avg_strength,
                    confidence=avg_confidence,
                    price=price,
                    source=combined_source,
                    metadata=combined_metadata
                )
                combined_signals.append(combined_signal)
        
        return combined_signals
    
    def analyze_market(self, market_data: pd.DataFrame) -> MarketCondition:
        """Analyze market using multiple sources"""
        # Use technical generator's market analysis
        return self.technical_generator.analyze_market(market_data)
    
    def track_performance(self, signal: TradingSignal, 
                         outcome: Optional[Dict[str, Any]] = None):
        """Track performance by source"""
        super().track_performance(signal, outcome)
        
        # Track by source
        if signal.source in self.source_performance:
            perf_data = {
                'timestamp': signal.timestamp,
                'signal_type': signal.signal_type.value,
                'strength': signal.strength,
                'confidence': signal.confidence,
                'outcome': outcome
            }
            self.source_performance[signal.source].append(perf_data)
            
            # Keep only recent performance data
            if len(self.source_performance[signal.source]) > self.config.performance_window:
                self.source_performance[signal.source] = self.source_performance[signal.source][-self.config.performance_window:]

# ============ Signal Manager ============
class SignalManager:
    """Manages signal generation, filtering, and execution"""
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.generator = self._create_generator()
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_queue = deque(maxlen=100)
        self.logger = get_logger(__name__)
        
        # Performance tracking
        self.performance_history = []
        self.signal_statistics = defaultdict(int)
    
    def _create_generator(self) -> BaseSignalGenerator:
        """Create appropriate signal generator based on configuration"""
        if self.config.signal_source == SignalSource.TECHNICAL:
            return TechnicalSignalGenerator(self.config)
        elif self.config.signal_source == SignalSource.ML_MODEL:
            return MLSignalGenerator(self.config)
        elif self.config.signal_source == SignalSource.HYBRID:
            return HybridSignalGenerator(self.config)
        else:
            return HybridSignalGenerator(self.config)  # Default to hybrid
    
    def process_market_data(self, market_data: pd.DataFrame) -> List[TradingSignal]:
        """Process market data and generate signals"""
        self.logger.info("Processing market data for signals...")
        
        # Generate signals
        signals = self.generator.generate_signals(market_data)
        
        # Validate signals
        valid_signals = [s for s in signals if self.generator.validate_signal(s)]
        
        # Queue signals for execution
        for signal in valid_signals:
            self.signal_queue.append(signal)
            self.signal_statistics[signal.signal_type.value] += 1
        
        # Update active signals
        self._update_active_signals(valid_signals)
        
        self.logger.info(f"Processed {len(valid_signals)} valid signals")
        
        return valid_signals
    
    def _update_active_signals(self, new_signals: List[TradingSignal]):
        """Update active signals, removing expired ones"""
        current_time = datetime.now()
        
        # Remove signals older than 1 hour
        expired_ids = []
        for signal_id, signal in self.active_signals.items():
            signal_age = (current_time - signal.timestamp).total_seconds()
            if signal_age > 3600:  # 1 hour
                expired_ids.append(signal_id)
        
        for signal_id in expired_ids:
            del self.active_signals[signal_id]
        
        # Add new signals
        for signal in new_signals:
            self.active_signals[signal.signal_id] = signal
    
    def get_next_signal(self) -> Optional[TradingSignal]:
        """Get the next signal from the queue"""
        if self.signal_queue:
            return self.signal_queue.popleft()
        return None
    
    def get_active_signals(self) -> List[TradingSignal]:
        """Get all active signals"""
        return list(self.active_signals.values())
    
    def get_signal_statistics(self) -> Dict[str, int]:
        """Get signal statistics"""
        return dict(self.signal_statistics)
    
    def record_signal_outcome(self, signal_id: str, 
                            outcome: Dict[str, Any]):
        """Record the outcome of a signal execution"""
        if signal_id in self.active_signals:
            signal = self.active_signals[signal_id]
            
            # Track performance
            self.generator.track_performance(signal, outcome)
            
            # Add to performance history
            perf_record = {
                'signal_id': signal_id,
                'timestamp': datetime.now(),
                'signal_type': signal.signal_type.value,
                'signal_strength': signal.strength,
                'signal_confidence': signal.confidence,
                'outcome': outcome
            }
            self.performance_history.append(perf_record)
            
            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
    
    def analyze_signal_performance(self) -> Dict[str, Any]:
        """Analyze signal performance"""
        if not self.performance_history:
            return {'total_signals': 0, 'performance': {}}
        
        performance_by_type = defaultdict(list)
        
        for record in self.performance_history:
            signal_type = record['signal_type']
            if 'outcome' in record and record['outcome']:
                outcome = record['outcome']
                if 'profit' in outcome:
                    performance_by_type[signal_type].append(outcome['profit'])
        
        analysis = {
            'total_signals': len(self.performance_history),
            'performance_by_type': {}
        }
        
        for signal_type, profits in performance_by_type.items():
            if profits:
                analysis['performance_by_type'][signal_type] = {
                    'count': len(profits),
                    'avg_profit': np.mean(profits),
                    'win_rate': sum(1 for p in profits if p > 0) / len(profits),
                    'total_profit': sum(profits)
                }
        
        return analysis
    
    def save_state(self, filepath: Path):
        """Save signal manager state"""
        state = {
            'active_signals': {k: v.to_dict() for k, v in self.active_signals.items()},
            'signal_queue': [s.to_dict() for s in self.signal_queue],
            'signal_statistics': dict(self.signal_statistics),
            'performance_history': self.performance_history,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            self.logger.info(f"Signal manager state saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
    
    def load_state(self, filepath: Path):
        """Load signal manager state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.active_signals = {
                k: TradingSignal.from_dict(v) 
                for k, v in state['active_signals'].items()
            }
            
            self.signal_queue = deque(
                [TradingSignal.from_dict(s) for s in state['signal_queue']],
                maxlen=100
            )
            
            self.signal_statistics = defaultdict(int, state['signal_statistics'])
            self.performance_history = state['performance_history']
            
            self.logger.info(f"Signal manager state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")

# ============ Factory Functions ============
def create_signal_generator(config: Optional[Dict] = None) -> BaseSignalGenerator:
    """Factory function to create a signal generator"""
    if config:
        signal_config = SignalConfig(**config)
    else:
        signal_config = SignalConfig()
    
    if signal_config.signal_source == SignalSource.TECHNICAL:
        return TechnicalSignalGenerator(signal_config)
    elif signal_config.signal_source == SignalSource.ML_MODEL:
        return MLSignalGenerator(signal_config)
    elif signal_config.signal_source == SignalSource.HYBRID:
        return HybridSignalGenerator(signal_config)
    else:
        return HybridSignalGenerator(signal_config)  # Default

def create_signal_manager(config: Optional[Dict] = None) -> SignalManager:
    """Factory function to create a signal manager"""
    if config:
        signal_config = SignalConfig(**config)
    else:
        signal_config = SignalConfig()
    
    return SignalManager(signal_config)

def load_signal_config(config_path: Path) -> SignalConfig:
    """Load signal configuration from YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return SignalConfig(**config_dict.get('signal_generation', {}))
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {str(e)}")
        return SignalConfig()

# ============ Example Usage ============
def example_usage():
    """Example usage of signal generation"""
    print("Signal Generation Example")
    print("=" * 50)
    
    # Create sample market data
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='H')
    np.random.seed(42)
    
    # Generate synthetic price data with some trends
    base_price = 10000
    returns = np.random.randn(len(dates)) * 0.01
    # Add an upward trend
    trend = np.linspace(0, 0.05, len(dates))
    price = base_price * np.exp(np.cumsum(returns + trend))
    
    market_data = pd.DataFrame({
        'open': price * (1 + np.random.randn(len(dates)) * 0.001),
        'high': price * (1 + np.abs(np.random.randn(len(dates)) * 0.002)),
        'low': price * (1 - np.abs(np.random.randn(len(dates)) * 0.002)),
        'close': price,
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    print(f"Created sample market data with {len(market_data)} rows")
    
    # Create signal manager with technical analysis
    config = {
        'signal_source': 'technical',
        'confidence_threshold': 0.6,
        'use_rsi': True,
        'use_macd': True,
        'use_bollinger': True
    }
    
    signal_manager = create_signal_manager(config)
    
    # Process market data
    print("\n1. Generating signals...")
    signals = signal_manager.process_market_data(market_data)
    
    print(f"Generated {len(signals)} signals")
    
    if signals:
        print("\n2. Signal Details:")
        for i, signal in enumerate(signals[:3], 1):  # Show first 3 signals
            print(f"  Signal {i}:")
            print(f"    Type: {signal.signal_type.value}")
            print(f"    Strength: {signal.strength:.2f}")
            print(f"    Confidence: {signal.confidence:.2f}")
            print(f"    Price: ${signal.price:.2f}")
            print(f"    Source: {signal.source.value}")
    
    # Get statistics
    print("\n3. Signal Statistics:")
    stats = signal_manager.get_signal_statistics()
    for signal_type, count in stats.items():
        print(f"  {signal_type}: {count}")
    
    # Simulate some outcomes
    print("\n4. Simulating signal outcomes...")
    for signal in signals[:2]:
        # Simulate profit/loss
        simulated_profit = np.random.uniform(-0.02, 0.03)  # -2% to +3%
        outcome = {
            'profit': simulated_profit,
            'executed_price': signal.price,
            'exit_price': signal.price * (1 + simulated_profit),
            'duration': '1h'
        }
        
        signal_manager.record_signal_outcome(signal.signal_id, outcome)
        print(f"  Signal {signal.signal_type.value}: {simulated_profit:.2%} profit")
    
    # Analyze performance
    print("\n5. Performance Analysis:")
    performance = signal_manager.analyze_signal_performance()
    print(f"  Total signals executed: {performance['total_signals']}")
    
    for signal_type, perf in performance.get('performance_by_type', {}).items():
        print(f"  {signal_type}:")
        print(f"    Count: {perf['count']}")
        print(f"    Avg Profit: {perf['avg_profit']:.2%}")
        print(f"    Win Rate: {perf['win_rate']:.2%}")
    
    return signal_manager, signals

# ============ Main Execution ============
def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bitcoin Trading AI - Signal Generation')
    parser.add_argument('--data', type=str, required=True,
                       help='Market data file path')
    parser.add_argument('--config', type=str, default='config/signal_generation.yaml',
                       help='Signal configuration file')
    parser.add_argument('--source', type=str, choices=['technical', 'ml', 'hybrid'],
                       help='Signal source (overrides config)')
    parser.add_argument('--output', type=str,
                       help='Output directory for signals')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with synthetic data')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running in test mode with synthetic data...")
        signal_manager, signals = example_usage()
        return
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if config_path.exists():
            signal_config = load_signal_config(config_path)
        else:
            signal_config = SignalConfig()
            logger.info(f"Using default configuration, config file not found: {config_path}")
        
        # Override signal source if specified
        if args.source:
            signal_config.signal_source = SignalSource(args.source)
        
        # Load market data
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"Loading market data from {data_path}")
        
        if data_path.suffix == '.parquet':
            market_data = pd.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            market_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        print(f"Loaded market data with shape: {market_data.shape}")
        print(f"Date range: {market_data.index.min()} to {market_data.index.max()}")
        
        # Create signal manager
        signal_manager = create_signal_manager(signal_config.__dict__)
        
        # Generate signals
        print(f"\nGenerating {signal_config.signal_source.value} signals...")
        signals = signal_manager.process_market_data(market_data)
        
        print(f"Generated {len(signals)} signals")
        
        # Display signal summary
        print("\n" + "="*50)
        print("SIGNAL GENERATION SUMMARY")
        print("="*50)
        
        signal_counts = {}
        for signal in signals:
            signal_type = signal.signal_type.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
        
        for signal_type, count in signal_counts.items():
            print(f"{signal_type.upper()}: {count} signals")
        
        # Show top signals
        if signals:
            print("\nTOP SIGNALS:")
            # Sort by strength * confidence
            sorted_signals = sorted(signals, key=lambda s: s.strength * s.confidence, reverse=True)
            
            for i, signal in enumerate(sorted_signals[:5], 1):
                print(f"  {i}. {signal.signal_type.value} at ${signal.price:.2f}")
                print(f"     Strength: {signal.strength:.2f}, Confidence: {signal.confidence:.2f}")
                print(f"     Source: {signal.source.value}")
                if signal.metadata:
                    metadata_str = ', '.join([f"{k}: {v}" for k, v in list(signal.metadata.items())[:2]])
                    print(f"     Metadata: {metadata_str}")
                print()
        
        # Save signals if output directory specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save signals to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            signals_file = output_dir / f"signals_{timestamp}.json"
            
            signals_data = [s.to_dict() for s in signals]
            with open(signals_file, 'w') as f:
                json.dump(signals_data, f, indent=2, default=str)
            
            print(f"Signals saved to: {signals_file}")
            
            # Save signal manager state
            state_file = output_dir / f"signal_manager_state_{timestamp}.json"
            signal_manager.save_state(state_file)
            
            print(f"Signal manager state saved to: {state_file}")
        
        # Performance analysis if we have historical data
        if len(signals) > 0:
            print("\nPERFORMANCE METRICS:")
            stats = signal_manager.get_signal_statistics()
            for signal_type, count in stats.items():
                print(f"  {signal_type}: {count} generated")
        
        print("\n" + "="*50)
        print(f"Signal generation completed successfully")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()