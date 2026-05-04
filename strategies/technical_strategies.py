"""
Technical Analysis Trading Strategies for Bitcoin Trading AI System
Implements classic and advanced technical analysis strategies with ML enhancements
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import pandas as pd
import numpy as np
import talib
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Import project modules
try:
    from config.config_manager import ConfigManager
    from core.data_processing.feature_engineer import FeatureEngineer
    from core.trading.signal_generator import SignalGenerator
    from core.trading.position_sizer import PositionSizer
    from core.risk_management.risk_analyzer import RiskAnalyzer
    from core.monitoring.performance_tracker import PerformanceTracker
    from core.utils.logger import setup_logger
    from core.utils.cache import CacheManager
    from strategies.ml_strategies import TradingSignal, SignalType, StrategyConfig, BaseMLStrategy
except ImportError:
    # For testing purposes
    ConfigManager = type('ConfigManager', (), {})
    FeatureEngineer = type('FeatureEngineer', (), {})
    SignalGenerator = type('SignalGenerator', (), {})
    PositionSizer = type('PositionSizer', (), {})
    RiskAnalyzer = type('RiskAnalyzer', (), {})
    PerformanceTracker = type('PerformanceTracker', (), {})
    setup_logger = lambda name: logging.getLogger(name)
    CacheManager = type('CacheManager', (), {})
    TradingSignal = type('TradingSignal', (), {})
    SignalType = Enum('SignalType', ['BUY', 'SELL', 'HOLD', 'CLOSE', 'HEDGE'])
    StrategyConfig = type('StrategyConfig', (), {})
    BaseMLStrategy = type('BaseMLStrategy', (ABC,), {})

# Initialize logger
logger = setup_logger(__name__)

# Technical Strategy Enums
class TechnicalStrategyType(Enum):
    """Enum for technical strategy types"""
    RSI_STRATEGY = "rsi"
    MACD_STRATEGY = "macd"
    BOLLINGER_BANDS = "bollinger"
    ICHIMOKU = "ichimoku"
    MOVING_AVERAGES = "moving_averages"
    STOCHASTIC = "stochastic"
    PARABOLIC_SAR = "parabolic_sar"
    FIBONACCI = "fibonacci"
    SUPPLY_DEMAND = "supply_demand"
    SUPPORT_RESISTANCE = "support_resistance"
    PRICE_ACTION = "price_action"
    MULTI_TIMEFRAME = "multi_timeframe"
    VOLUME_PROFILE = "volume_profile"
    MARKET_STRUCTURE = "market_structure"

class Timeframe(Enum):
    """Enum for timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

# Data Classes
@dataclass
class TechnicalIndicator:
    """Data class for technical indicator"""
    name: str
    value: float
    signal: str
    level: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CandlestickPattern:
    """Data class for candlestick pattern"""
    name: str
    pattern_type: str  # bullish, bearish, reversal, continuation
    confidence: float
    location: str  # support, resistance, trendline
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketStructure:
    """Data class for market structure analysis"""
    trend: str  # uptrend, downtrend, sideways
    trend_strength: float
    volatility: float
    volume_profile: Dict[str, float]
    key_levels: List[float]
    structure_type: str  # impulse, correction, consolidation
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class TechnicalAnalysis:
    """Comprehensive technical analysis data"""
    timestamp: datetime
    symbol: str
    timeframe: str
    indicators: Dict[str, TechnicalIndicator]
    patterns: List[CandlestickPattern]
    market_structure: MarketStructure
    signals: List[Dict[str, Any]]
    summary: Dict[str, Any]

# Base Technical Strategy Class
class BaseTechnicalStrategy(ABC):
    """Base class for all technical trading strategies"""
    
    def __init__(self, config_manager: ConfigManager, strategy_type: TechnicalStrategyType):
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
        
        # State
        self.analysis_history: List[TechnicalAnalysis] = []
        self.signals_history: List[TradingSignal] = []
        self.trades_history: List[Dict] = []
        
        # Default parameters
        self.default_params = self.get_default_parameters()
        
        logger.info(f"Initialized base technical strategy: {self.name}")
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for the strategy"""
        return {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bollinger_period": 20,
            "bollinger_std": 2,
            "stochastic_k": 14,
            "stochastic_d": 3,
            "stochastic_slow": 3,
            "ma_fast": 9,
            "ma_medium": 21,
            "ma_slow": 50,
            "ma_long": 200,
            "ichimoku_tenkan": 9,
            "ichimoku_kijun": 26,
            "ichimoku_senkou_b": 52,
            "ichimoku_chikou": 26,
            "parabolic_accel": 0.02,
            "parabolic_max": 0.2,
            "volume_sma_period": 20,
            "atr_period": 14
        }
    
    async def initialize(self, config: StrategyConfig) -> bool:
        """Initialize strategy with configuration"""
        try:
            self.config = config
            
            # Update parameters from config
            if config.parameters:
                self.default_params.update(config.parameters)
            
            # Load any required models or data
            await self.load_resources()
            
            self.initialized = True
            logger.info(f"Technical strategy '{self.name}' initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize technical strategy '{self.name}': {e}")
            return False
    
    async def load_resources(self):
        """Load any required resources"""
        pass
    
    @abstractmethod
    async def analyze(self, market_data: pd.DataFrame) -> TechnicalAnalysis:
        """Perform technical analysis on market data"""
        pass
    
    @abstractmethod
    async def generate_signal(self, analysis: TechnicalAnalysis) -> Optional[TradingSignal]:
        """Generate trading signal based on technical analysis"""
        pass
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Calculate all technical indicators"""
        indicators = {}
        
        try:
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Required: {required_cols}")
            
            # Price-based indicators
            indicators.update(await self.calculate_price_indicators(df))
            
            # Volume indicators
            indicators.update(await self.calculate_volume_indicators(df))
            
            # Volatility indicators
            indicators.update(await self.calculate_volatility_indicators(df))
            
            # Momentum indicators
            indicators.update(await self.calculate_momentum_indicators(df))
            
            # Trend indicators
            indicators.update(await self.calculate_trend_indicators(df))
            
            logger.debug(f"Calculated {len(indicators)} technical indicators")
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    async def calculate_price_indicators(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Calculate price-based indicators"""
        indicators = {}
        
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            opens = df['open'].values
            
            # RSI
            rsi_period = self.default_params.get('rsi_period', 14)
            if len(closes) >= rsi_period:
                rsi = talib.RSI(closes, timeperiod=rsi_period)
                if not np.isnan(rsi[-1]):
                    indicators['rsi'] = TechnicalIndicator(
                        name='RSI',
                        value=float(rsi[-1]),
                        signal=self.get_rsi_signal(rsi[-1]),
                        level=rsi[-1],
                        metadata={'period': rsi_period}
                    )
            
            # MACD
            macd_fast = self.default_params.get('macd_fast', 12)
            macd_slow = self.default_params.get('macd_slow', 26)
            macd_signal = self.default_params.get('macd_signal', 9)
            if len(closes) >= macd_slow:
                macd, macd_signal_line, macd_hist = talib.MACD(
                    closes, 
                    fastperiod=macd_fast,
                    slowperiod=macd_slow,
                    signalperiod=macd_signal
                )
                if not np.isnan(macd[-1]):
                    macd_signal_val = self.get_macd_signal(macd[-1], macd_signal_line[-1], macd_hist[-1])
                    indicators['macd'] = TechnicalIndicator(
                        name='MACD',
                        value=float(macd[-1]),
                        signal=macd_signal_val,
                        metadata={
                            'signal_line': float(macd_signal_line[-1]),
                            'histogram': float(macd_hist[-1]),
                            'cross': macd_signal_val
                        }
                    )
            
            # Stochastic
            stoch_k = self.default_params.get('stochastic_k', 14)
            stoch_d = self.default_params.get('stochastic_d', 3)
            stoch_slow = self.default_params.get('stochastic_slow', 3)
            if len(highs) >= stoch_k and len(lows) >= stoch_k:
                slowk, slowd = talib.STOCH(
                    highs, lows, closes,
                    fastk_period=stoch_k,
                    slowk_period=stoch_slow,
                    slowk_matype=0,
                    slowd_period=stoch_d,
                    slowd_matype=0
                )
                if not np.isnan(slowk[-1]):
                    stoch_signal = self.get_stochastic_signal(slowk[-1], slowd[-1])
                    indicators['stochastic'] = TechnicalIndicator(
                        name='Stochastic',
                        value=float(slowk[-1]),
                        signal=stoch_signal,
                        metadata={
                            'k': float(slowk[-1]),
                            'd': float(slowd[-1]),
                            'cross': stoch_signal
                        }
                    )
            
            # Bollinger Bands
            bb_period = self.default_params.get('bollinger_period', 20)
            bb_std = self.default_params.get('bollinger_std', 2)
            if len(closes) >= bb_period:
                upper, middle, lower = talib.BBANDS(
                    closes,
                    timeperiod=bb_period,
                    nbdevup=bb_std,
                    nbdevdn=bb_std,
                    matype=0
                )
                if not np.isnan(upper[-1]):
                    bb_signal = self.get_bollinger_signal(closes[-1], upper[-1], lower[-1])
                    indicators['bollinger'] = TechnicalIndicator(
                        name='Bollinger Bands',
                        value=float((closes[-1] - lower[-1]) / (upper[-1] - lower[-1])),
                        signal=bb_signal,
                        metadata={
                            'upper': float(upper[-1]),
                            'middle': float(middle[-1]),
                            'lower': float(lower[-1]),
                            'width': float((upper[-1] - lower[-1]) / middle[-1])
                        }
                    )
            
            # Moving Averages
            ma_fast = self.default_params.get('ma_fast', 9)
            ma_medium = self.default_params.get('ma_medium', 21)
            ma_slow = self.default_params.get('ma_slow', 50)
            ma_long = self.default_params.get('ma_long', 200)
            
            for period, name in [(ma_fast, 'SMA_9'), (ma_medium, 'SMA_21'), 
                                 (ma_slow, 'SMA_50'), (ma_long, 'SMA_200')]:
                if len(closes) >= period:
                    sma = talib.SMA(closes, timeperiod=period)
                    if not np.isnan(sma[-1]):
                        indicators[f'sma_{period}'] = TechnicalIndicator(
                            name=name,
                            value=float(sma[-1]),
                            signal='neutral',
                            metadata={'period': period}
                        )
            
            # Calculate MA cross signals
            if 'sma_9' in indicators and 'sma_21' in indicators:
                ma_cross = self.get_ma_cross_signal(
                    indicators['sma_9'].value,
                    indicators['sma_21'].value,
                    closes[-2] if len(closes) > 1 else closes[-1]
                )
                indicators['ma_cross'] = TechnicalIndicator(
                    name='MA Cross',
                    value=float(indicators['sma_9'].value - indicators['sma_21'].value),
                    signal=ma_cross,
                    metadata={'fast': 9, 'slow': 21}
                )
            
            # ATR (Average True Range)
            atr_period = self.default_params.get('atr_period', 14)
            if len(highs) >= atr_period and len(lows) >= atr_period:
                atr = talib.ATR(highs, lows, closes, timeperiod=atr_period)
                if not np.isnan(atr[-1]):
                    indicators['atr'] = TechnicalIndicator(
                        name='ATR',
                        value=float(atr[-1]),
                        signal='volatility',
                        metadata={'period': atr_period, 'percent': float(atr[-1] / closes[-1] * 100)}
                    )
            
            # Parabolic SAR
            parabolic_accel = self.default_params.get('parabolic_accel', 0.02)
            parabolic_max = self.default_params.get('parabolic_max', 0.2)
            if len(highs) >= 2 and len(lows) >= 2:
                sar = talib.SAR(highs, lows, 
                               acceleration=parabolic_accel,
                               maximum=parabolic_max)
                if not np.isnan(sar[-1]):
                    sar_signal = self.get_sar_signal(closes[-1], sar[-1])
                    indicators['parabolic_sar'] = TechnicalIndicator(
                        name='Parabolic SAR',
                        value=float(sar[-1]),
                        signal=sar_signal,
                        metadata={'acceleration': parabolic_accel, 'maximum': parabolic_max}
                    )
            
            # CCI (Commodity Channel Index)
            if len(highs) >= 20 and len(lows) >= 20:
                cci = talib.CCI(highs, lows, closes, timeperiod=20)
                if not np.isnan(cci[-1]):
                    cci_signal = self.get_cci_signal(cci[-1])
                    indicators['cci'] = TechnicalIndicator(
                        name='CCI',
                        value=float(cci[-1]),
                        signal=cci_signal,
                        metadata={'period': 20}
                    )
            
            # Williams %R
            if len(highs) >= 14 and len(lows) >= 14:
                willr = talib.WILLR(highs, lows, closes, timeperiod=14)
                if not np.isnan(willr[-1]):
                    willr_signal = self.get_williams_signal(willr[-1])
                    indicators['williams_r'] = TechnicalIndicator(
                        name="Williams %R",
                        value=float(willr[-1]),
                        signal=willr_signal,
                        metadata={'period': 14}
                    )
        
        except Exception as e:
            logger.error(f"Error calculating price indicators: {e}")
        
        return indicators
    
    async def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Calculate volume-based indicators"""
        indicators = {}
        
        try:
            closes = df['close'].values
            volumes = df['volume'].values
            
            if len(volumes) == 0:
                return indicators
            
            # OBV (On Balance Volume)
            obv = talib.OBV(closes, volumes)
            if len(obv) > 0 and not np.isnan(obv[-1]):
                obv_trend = 'bullish' if obv[-1] > obv[-2] else 'bearish' if len(obv) > 1 else 'neutral'
                indicators['obv'] = TechnicalIndicator(
                    name='OBV',
                    value=float(obv[-1]),
                    signal=obv_trend,
                    metadata={'trend': obv_trend}
                )
            
            # Volume SMA
            volume_period = self.default_params.get('volume_sma_period', 20)
            if len(volumes) >= volume_period:
                volume_sma = talib.SMA(volumes, timeperiod=volume_period)
                if not np.isnan(volume_sma[-1]):
                    volume_ratio = volumes[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1
                    volume_signal = 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.5 else 'normal'
                    indicators['volume_sma'] = TechnicalIndicator(
                        name='Volume SMA',
                        value=float(volume_sma[-1]),
                        signal=volume_signal,
                        metadata={'ratio': float(volume_ratio), 'period': volume_period}
                    )
            
            # Volume Price Trend (VPT)
            if len(closes) > 1 and len(volumes) > 1:
                vpt = np.zeros_like(closes)
                vpt[0] = volumes[0]
                for i in range(1, len(closes)):
                    price_change = (closes[i] - closes[i-1]) / closes[i-1]
                    vpt[i] = vpt[i-1] + volumes[i] * price_change
                
                if not np.isnan(vpt[-1]):
                    vpt_trend = 'bullish' if vpt[-1] > vpt[-2] else 'bearish' if len(vpt) > 1 else 'neutral'
                    indicators['vpt'] = TechnicalIndicator(
                        name='VPT',
                        value=float(vpt[-1]),
                        signal=vpt_trend,
                        metadata={'trend': vpt_trend}
                    )
            
            # Money Flow Index (MFI)
            if len(highs := df['high'].values) >= 14 and len(lows := df['low'].values) >= 14:
                mfi = talib.MFI(highs, lows, closes, volumes, timeperiod=14)
                if not np.isnan(mfi[-1]):
                    mfi_signal = self.get_mfi_signal(mfi[-1])
                    indicators['mfi'] = TechnicalIndicator(
                        name='MFI',
                        value=float(mfi[-1]),
                        signal=mfi_signal,
                        metadata={'period': 14}
                    )
        
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
        
        return indicators
    
    async def calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Calculate volatility indicators"""
        indicators = {}
        
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Donchian Channels
            if len(highs) >= 20 and len(lows) >= 20:
                upper = np.max(highs[-20:])
                lower = np.min(lows[-20:])
                middle = (upper + lower) / 2
                
                dc_signal = self.get_donchian_signal(closes[-1], upper, lower)
                indicators['donchian'] = TechnicalIndicator(
                    name='Donchian Channels',
                    value=float((closes[-1] - lower) / (upper - lower)),
                    signal=dc_signal,
                    metadata={
                        'upper': float(upper),
                        'middle': float(middle),
                        'lower': float(lower),
                        'width': float((upper - lower) / middle)
                    }
                )
            
            # Keltner Channels
            if len(closes) >= 20:
                ema = talib.EMA(closes, timeperiod=20)
                atr = talib.ATR(highs, lows, closes, timeperiod=20)
                
                if not np.isnan(ema[-1]) and not np.isnan(atr[-1]):
                    keltner_upper = ema[-1] + 2 * atr[-1]
                    keltner_lower = ema[-1] - 2 * atr[-1]
                    
                    kc_signal = self.get_keltner_signal(closes[-1], keltner_upper, keltner_lower)
                    indicators['keltner'] = TechnicalIndicator(
                        name='Keltner Channels',
                        value=float((closes[-1] - keltner_lower) / (keltner_upper - keltner_lower)),
                        signal=kc_signal,
                        metadata={
                            'upper': float(keltner_upper),
                            'middle': float(ema[-1]),
                            'lower': float(keltner_lower)
                        }
                    )
            
            # Historical Volatility
            if len(closes) >= 20:
                returns = np.diff(closes) / closes[:-1]
                if len(returns) >= 19:
                    hv = np.std(returns[-20:]) * np.sqrt(252)  # Annualized
                    indicators['historical_vol'] = TechnicalIndicator(
                        name='Historical Volatility',
                        value=float(hv),
                        signal='high' if hv > 0.5 else 'low' if hv < 0.2 else 'normal',
                        metadata={'period': 20, 'annualized': True}
                    )
        
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
        
        return indicators
    
    async def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Calculate momentum indicators"""
        indicators = {}
        
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Rate of Change (ROC)
            if len(closes) >= 12:
                roc = talib.ROC(closes, timeperiod=12)
                if not np.isnan(roc[-1]):
                    roc_signal = 'bullish' if roc[-1] > 0 else 'bearish'
                    indicators['roc'] = TechnicalIndicator(
                        name='ROC',
                        value=float(roc[-1]),
                        signal=roc_signal,
                        metadata={'period': 12}
                    )
            
            # Awesome Oscillator
            if len(highs) >= 34 and len(lows) >= 34:
                ao = talib.SMA((highs + lows) / 2, timeperiod=5) - \
                     talib.SMA((highs + lows) / 2, timeperiod=34)
                if len(ao) > 0 and not np.isnan(ao[-1]):
                    ao_signal = self.get_ao_signal(ao[-1], ao[-2] if len(ao) > 1 else ao[-1])
                    indicators['awesome_oscillator'] = TechnicalIndicator(
                        name='Awesome Oscillator',
                        value=float(ao[-1]),
                        signal=ao_signal,
                        metadata={'signal': ao_signal}
                    )
            
            # Chaikin Oscillator
            if len(highs) >= 10 and len(lows) >= 10:
                ad = talib.AD(highs, lows, closes, df['volume'].values)
                if len(ad) >= 10:
                    ad_ema_3 = talib.EMA(ad, timeperiod=3)
                    ad_ema_10 = talib.EMA(ad, timeperiod=10)
                    if len(ad_ema_3) > 0 and len(ad_ema_10) > 0:
                        co = ad_ema_3[-1] - ad_ema_10[-1]
                        co_signal = 'bullish' if co > 0 else 'bearish'
                        indicators['chaikin_oscillator'] = TechnicalIndicator(
                            name='Chaikin Oscillator',
                            value=float(co),
                            signal=co_signal,
                            metadata={'signal': co_signal}
                        )
            
            # Price Rate of Change (PROC)
            if len(closes) >= 12:
                proc = (closes[-1] - closes[-12]) / closes[-12] * 100
                proc_signal = 'bullish' if proc > 0 else 'bearish'
                indicators['proc'] = TechnicalIndicator(
                    name='Price ROC',
                    value=float(proc),
                    signal=proc_signal,
                    metadata={'period': 12}
                )
        
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
        
        return indicators
    
    async def calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Calculate trend indicators"""
        indicators = {}
        
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # ADX (Average Directional Index)
            if len(highs) >= 14 and len(lows) >= 14:
                adx = talib.ADX(highs, lows, closes, timeperiod=14)
                plus_di = talib.PLUS_DI(highs, lows, closes, timeperiod=14)
                minus_di = talib.MINUS_DI(highs, lows, closes, timeperiod=14)
                
                if not np.isnan(adx[-1]):
                    trend_strength = 'strong' if adx[-1] > 25 else 'weak' if adx[-1] < 20 else 'moderate'
                    trend_direction = 'bullish' if plus_di[-1] > minus_di[-1] else 'bearish'
                    
                    indicators['adx'] = TechnicalIndicator(
                        name='ADX',
                        value=float(adx[-1]),
                        signal=f"{trend_direction}_{trend_strength}",
                        metadata={
                            'trend_strength': trend_strength,
                            'trend_direction': trend_direction,
                            'plus_di': float(plus_di[-1]),
                            'minus_di': float(minus_di[-1])
                        }
                    )
            
            # Ichimoku Cloud (simplified)
            tenkan_period = self.default_params.get('ichimoku_tenkan', 9)
            kijun_period = self.default_params.get('ichimoku_kijun', 26)
            senkou_b_period = self.default_params.get('ichimoku_senkou_b', 52)
            
            if len(highs) >= senkou_b_period and len(lows) >= senkou_b_period:
                # Tenkan-sen (Conversion Line)
                tenkan_high = np.max(highs[-tenkan_period:])
                tenkan_low = np.min(lows[-tenkan_period:])
                tenkan_sen = (tenkan_high + tenkan_low) / 2
                
                # Kijun-sen (Base Line)
                kijun_high = np.max(highs[-kijun_period:])
                kijun_low = np.min(lows[-kijun_period:])
                kijun_sen = (kijun_high + kijun_low) / 2
                
                # Senkou Span A (Leading Span A)
                senkou_span_a = (tenkan_sen + kijun_sen) / 2
                
                # Senkou Span B (Leading Span B)
                senkou_b_high = np.max(highs[-senkou_b_period:])
                senkou_b_low = np.min(lows[-senkou_b_period:])
                senkou_span_b = (senkou_b_high + senkou_b_low) / 2
                
                ichimoku_signal = self.get_ichimoku_signal(
                    closes[-1], tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b
                )
                
                indicators['ichimoku'] = TechnicalIndicator(
                    name='Ichimoku Cloud',
                    value=float(senkou_span_a - senkou_span_b),
                    signal=ichimoku_signal,
                    metadata={
                        'tenkan_sen': float(tenkan_sen),
                        'kijun_sen': float(kijun_sen),
                        'senkou_span_a': float(senkou_span_a),
                        'senkou_span_b': float(senkou_span_b),
                        'cloud_top': float(max(senkou_span_a, senkou_span_b)),
                        'cloud_bottom': float(min(senkou_span_a, senkou_span_b))
                    }
                )
            
            # TRIX (Triple Exponential Average)
            if len(closes) >= 15:
                trix = talib.TRIX(closes, timeperiod=15)
                if not np.isnan(trix[-1]):
                    trix_signal = 'bullish' if trix[-1] > 0 else 'bearish'
                    indicators['trix'] = TechnicalIndicator(
                        name='TRIX',
                        value=float(trix[-1]),
                        signal=trix_signal,
                        metadata={'period': 15}
                    )
            
            # Vortex Indicator
            if len(highs) >= 14 and len(lows) >= 14:
                vi_plus = talib.PLUS_DI(highs, lows, closes, timeperiod=14)
                vi_minus = talib.MINUS_DI(highs, lows, closes, timeperiod=14)
                
                if not np.isnan(vi_plus[-1]):
                    vi_signal = 'bullish' if vi_plus[-1] > vi_minus[-1] else 'bearish'
                    indicators['vortex'] = TechnicalIndicator(
                        name='Vortex Indicator',
                        value=float(vi_plus[-1] - vi_minus[-1]),
                        signal=vi_signal,
                        metadata={
                            'vi_plus': float(vi_plus[-1]),
                            'vi_minus': float(vi_minus[-1])
                        }
                    )
        
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
        
        return indicators
    
    async def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect candlestick patterns"""
        patterns = []
        
        try:
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            # Check last 5 candles for patterns
            lookback = min(5, len(opens))
            
            for i in range(-lookback, 0):
                idx = i + len(opens)
                
                # Bullish patterns
                if idx >= 2:
                    # Hammer
                    if self.is_hammer(opens[idx], highs[idx], lows[idx], closes[idx]):
                        patterns.append(CandlestickPattern(
                            name='Hammer',
                            pattern_type='bullish_reversal',
                            confidence=0.7,
                            location='support',
                            metadata={'index': idx}
                        ))
                    
                    # Bullish Engulfing
                    if self.is_bullish_engulfing(opens[idx-1], closes[idx-1], opens[idx], closes[idx]):
                        patterns.append(CandlestickPattern(
                            name='Bullish Engulfing',
                            pattern_type='bullish_reversal',
                            confidence=0.8,
                            location='support',
                            metadata={'index': idx}
                        ))
                    
                    # Morning Star (simplified)
                    if idx >= 2 and self.is_morning_star(
                        opens[idx-2], closes[idx-2], 
                        opens[idx-1], closes[idx-1],
                        opens[idx], closes[idx]
                    ):
                        patterns.append(CandlestickPattern(
                            name='Morning Star',
                            pattern_type='bullish_reversal',
                            confidence=0.85,
                            location='support',
                            metadata={'index': idx}
                        ))
                
                # Bearish patterns
                if idx >= 2:
                    # Shooting Star
                    if self.is_shooting_star(opens[idx], highs[idx], lows[idx], closes[idx]):
                        patterns.append(CandlestickPattern(
                            name='Shooting Star',
                            pattern_type='bearish_reversal',
                            confidence=0.7,
                            location='resistance',
                            metadata={'index': idx}
                        ))
                    
                    # Bearish Engulfing
                    if self.is_bearish_engulfing(opens[idx-1], closes[idx-1], opens[idx], closes[idx]):
                        patterns.append(CandlestickPattern(
                            name='Bearish Engulfing',
                            pattern_type='bearish_reversal',
                            confidence=0.8,
                            location='resistance',
                            metadata={'index': idx}
                        ))
                    
                    # Evening Star (simplified)
                    if idx >= 2 and self.is_evening_star(
                        opens[idx-2], closes[idx-2], 
                        opens[idx-1], closes[idx-1],
                        opens[idx], closes[idx]
                    ):
                        patterns.append(CandlestickPattern(
                            name='Evening Star',
                            pattern_type='bearish_reversal',
                            confidence=0.85,
                            location='resistance',
                            metadata={'index': idx}
                        ))
                
                # Doji patterns
                if self.is_doji(opens[idx], highs[idx], lows[idx], closes[idx]):
                    doji_type = self.get_doji_type(opens[idx], highs[idx], lows[idx], closes[idx])
                    patterns.append(CandlestickPattern(
                        name=f'{doji_type} Doji',
                        pattern_type='reversal',
                        confidence=0.6,
                        location='neutral',
                        metadata={'index': idx, 'type': doji_type}
                    ))
            
            logger.debug(f"Detected {len(patterns)} candlestick patterns")
        
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
        
        return patterns
    
    async def analyze_market_structure(self, df: pd.DataFrame) -> MarketStructure:
        """Analyze market structure"""
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Determine trend
            trend, trend_strength = self.determine_trend(df)
            
            # Calculate volatility
            volatility = self.calculate_volatility(df)
            
            # Identify key levels (support/resistance)
            key_levels = await self.identify_key_levels(df)
            
            # Volume profile analysis
            volume_profile = await self.analyze_volume_profile(df)
            
            # Determine market structure type
            structure_type = self.determine_structure_type(df, trend)
            
            return MarketStructure(
                trend=trend,
                trend_strength=trend_strength,
                volatility=volatility,
                volume_profile=volume_profile,
                key_levels=key_levels,
                structure_type=structure_type,
                metadata={
                    'price_range': float(highs.max() - lows.min()),
                    'current_price': float(closes[-1]),
                    'analysis_date': datetime.now().isoformat()
                }
            )
        
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return MarketStructure(
                trend='unknown',
                trend_strength=0.0,
                volatility=0.0,
                volume_profile={},
                key_levels=[],
                structure_type='unknown'
            )
    
    # Signal interpretation methods
    def get_rsi_signal(self, rsi_value: float) -> str:
        """Get RSI signal"""
        if rsi_value > 70:
            return 'overbought'
        elif rsi_value < 30:
            return 'oversold'
        elif rsi_value > 55:
            return 'bullish'
        elif rsi_value < 45:
            return 'bearish'
        else:
            return 'neutral'
    
    def get_macd_signal(self, macd: float, signal: float, histogram: float) -> str:
        """Get MACD signal"""
        if macd > signal and histogram > 0:
            return 'bullish_cross'
        elif macd < signal and histogram < 0:
            return 'bearish_cross'
        elif histogram > 0:
            return 'bullish'
        elif histogram < 0:
            return 'bearish'
        else:
            return 'neutral'
    
    def get_stochastic_signal(self, k: float, d: float) -> str:
        """Get Stochastic signal"""
        if k > 80 and d > 80:
            return 'overbought'
        elif k < 20 and d < 20:
            return 'oversold'
        elif k > d:
            return 'bullish_cross'
        elif k < d:
            return 'bearish_cross'
        else:
            return 'neutral'
    
    def get_bollinger_signal(self, price: float, upper: float, lower: float) -> str:
        """Get Bollinger Bands signal"""
        if price > upper:
            return 'overbought'
        elif price < lower:
            return 'oversold'
        elif price > (upper + lower) / 2:
            return 'upper_band'
        else:
            return 'lower_band'
    
    def get_ma_cross_signal(self, fast_ma: float, slow_ma: float, prev_price: float) -> str:
        """Get Moving Average cross signal"""
        if fast_ma > slow_ma:
            return 'golden_cross'
        else:
            return 'death_cross'
    
    def get_sar_signal(self, price: float, sar: float) -> str:
        """Get Parabolic SAR signal"""
        if price > sar:
            return 'bullish'
        else:
            return 'bearish'
    
    def get_cci_signal(self, cci: float) -> str:
        """Get CCI signal"""
        if cci > 100:
            return 'overbought'
        elif cci < -100:
            return 'oversold'
        elif cci > 0:
            return 'bullish'
        else:
            return 'bearish'
    
    def get_williams_signal(self, willr: float) -> str:
        """Get Williams %R signal"""
        if willr > -20:
            return 'overbought'
        elif willr < -80:
            return 'oversold'
        elif willr > -50:
            return 'bullish'
        else:
            return 'bearish'
    
    def get_mfi_signal(self, mfi: float) -> str:
        """Get MFI signal"""
        if mfi > 80:
            return 'overbought'
        elif mfi < 20:
            return 'oversold'
        elif mfi > 50:
            return 'bullish'
        else:
            return 'bearish'
    
    def get_donchian_signal(self, price: float, upper: float, lower: float) -> str:
        """Get Donchian Channels signal"""
        if price > upper:
            return 'breakout'
        elif price < lower:
            return 'breakdown'
        else:
            return 'range'
    
    def get_keltner_signal(self, price: float, upper: float, lower: float) -> str:
        """Get Keltner Channels signal"""
        if price > upper:
            return 'overbought'
        elif price < lower:
            return 'oversold'
        else:
            return 'normal'
    
    def get_ao_signal(self, current_ao: float, prev_ao: float) -> str:
        """Get Awesome Oscillator signal"""
        if current_ao > 0 and prev_ao <= 0:
            return 'bullish_cross'
        elif current_ao < 0 and prev_ao >= 0:
            return 'bearish_cross'
        elif current_ao > 0:
            return 'bullish'
        else:
            return 'bearish'
    
    def get_ichimoku_signal(self, price: float, tenkan: float, kijun: float, 
                           senkou_a: float, senkou_b: float) -> str:
        """Get Ichimoku Cloud signal"""
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        if price > cloud_top:
            if tenkan > kijun:
                return 'strong_bullish'
            else:
                return 'bullish'
        elif price < cloud_bottom:
            if tenkan < kijun:
                return 'strong_bearish'
            else:
                return 'bearish'
        elif price > kijun:
            return 'neutral_bullish'
        else:
            return 'neutral_bearish'
    
    # Candlestick pattern detection methods
    def is_hammer(self, open_price: float, high: float, low: float, close: float) -> bool:
        """Detect hammer pattern"""
        body_size = abs(close - open_price)
        lower_shadow = min(open_price, close) - low
        upper_shadow = high - max(open_price, close)
        
        return (lower_shadow > 2 * body_size and 
                upper_shadow < body_size * 0.1 and
                body_size > 0)
    
    def is_shooting_star(self, open_price: float, high: float, low: float, close: float) -> bool:
        """Detect shooting star pattern"""
        body_size = abs(close - open_price)
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        return (upper_shadow > 2 * body_size and 
                lower_shadow < body_size * 0.1 and
                body_size > 0)
    
    def is_bullish_engulfing(self, prev_open: float, prev_close: float, 
                            curr_open: float, curr_close: float) -> bool:
        """Detect bullish engulfing pattern"""
        prev_body = prev_close - prev_open
        curr_body = curr_close - curr_open
        
        return (prev_body < 0 and curr_body > 0 and
                curr_open < prev_close and curr_close > prev_open)
    
    def is_bearish_engulfing(self, prev_open: float, prev_close: float,
                            curr_open: float, curr_close: float) -> bool:
        """Detect bearish engulfing pattern"""
        prev_body = prev_close - prev_open
        curr_body = curr_close - curr_open
        
        return (prev_body > 0 and curr_body < 0 and
                curr_open > prev_close and curr_close < prev_open)
    
    def is_morning_star(self, day1_open: float, day1_close: float,
                       day2_open: float, day2_close: float,
                       day3_open: float, day3_close: float) -> bool:
        """Detect morning star pattern (simplified)"""
        day1_body = day1_close - day1_open
        day2_body = day2_close - day2_open
        day3_body = day3_close - day3_open
        
        return (day1_body < 0 and abs(day2_body) < abs(day1_body) * 0.5 and
                day3_body > 0 and day3_close > (day1_open + day1_close) / 2)
    
    def is_evening_star(self, day1_open: float, day1_close: float,
                       day2_open: float, day2_close: float,
                       day3_open: float, day3_close: float) -> bool:
        """Detect evening star pattern (simplified)"""
        day1_body = day1_close - day1_open
        day2_body = day2_close - day2_open
        day3_body = day3_close - day3_open
        
        return (day1_body > 0 and abs(day2_body) < abs(day1_body) * 0.5 and
                day3_body < 0 and day3_close < (day1_open + day1_close) / 2)
    
    def is_doji(self, open_price: float, high: float, low: float, close: float) -> bool:
        """Detect doji pattern"""
        body_size = abs(close - open_price)
        total_range = high - low
        
        return body_size < total_range * 0.1
    
    def get_doji_type(self, open_price: float, high: float, low: float, close: float) -> str:
        """Get doji type"""
        body_mid = (open_price + close) / 2
        range_mid = (high + low) / 2
        
        if abs(body_mid - range_mid) < (high - low) * 0.1:
            return 'Standard'
        elif body_mid > range_mid:
            return 'Dragonfly'
        else:
            return 'Gravestone'
    
    # Market structure analysis methods
    def determine_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Determine market trend"""
        closes = df['close'].values
        
        if len(closes) < 50:
            return 'unknown', 0.0
        
        # Calculate multiple trend indicators
        sma_20 = talib.SMA(closes, timeperiod=20)
        sma_50 = talib.SMA(closes, timeperiod=50)
        
        # Simple trend determination
        if len(sma_20) > 0 and len(sma_50) > 0:
            if closes[-1] > sma_20[-1] > sma_50[-1]:
                trend = 'uptrend'
                strength = min(1.0, (closes[-1] - sma_50[-1]) / sma_50[-1] * 10)
            elif closes[-1] < sma_20[-1] < sma_50[-1]:
                trend = 'downtrend'
                strength = min(1.0, (sma_50[-1] - closes[-1]) / closes[-1] * 10)
            else:
                trend = 'sideways'
                strength = 0.0
        else:
            trend = 'unknown'
            strength = 0.0
        
        return trend, float(strength)
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate market volatility"""
        closes = df['close'].values
        
        if len(closes) < 20:
            return 0.0
        
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        
        return float(volatility * 100)  # Return as percentage
    
    async def identify_key_levels(self, df: pd.DataFrame) -> List[float]:
        """Identify key support and resistance levels"""
        levels = []
        
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Use pivot points
            if len(highs) >= 1 and len(lows) >= 1:
                pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
                r1 = 2 * pivot - lows[-1]
                s1 = 2 * pivot - highs[-1]
                r2 = pivot + (highs[-1] - lows[-1])
                s2 = pivot - (highs[-1] - lows[-1])
                
                levels.extend([float(s2), float(s1), float(pivot), float(r1), float(r2)])
            
            # Add recent highs and lows
            lookback = min(50, len(highs))
            recent_highs = highs[-lookback:]
            recent_lows = lows[-lookback:]
            
            levels.append(float(np.max(recent_highs)))
            levels.append(float(np.min(recent_lows)))
            
            # Add round numbers near current price
            current_price = closes[-1]
            round_levels = [round(current_price / 100) * 100,
                           round(current_price / 100) * 100 + 100,
                           round(current_price / 100) * 100 - 100]
            levels.extend(round_levels)
            
            # Remove duplicates and sort
            levels = sorted(list(set(round(level, 2) for level in levels)))
        
        except Exception as e:
            logger.error(f"Error identifying key levels: {e}")
        
        return levels
    
    async def analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume profile"""
        profile = {}
        
        try:
            if 'volume' not in df.columns:
                return profile
            
            volumes = df['volume'].values
            closes = df['close'].values
            
            # Simple volume analysis
            if len(volumes) > 0:
                profile['current_volume'] = float(volumes[-1])
                profile['avg_volume_20'] = float(np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes))
                profile['volume_ratio'] = float(profile['current_volume'] / profile['avg_volume_20'] if profile['avg_volume_20'] > 0 else 1)
                
                # Volume trend
                if len(volumes) >= 5:
                    volume_trend = np.polyfit(range(5), volumes[-5:], 1)[0]
                    profile['volume_trend'] = float(volume_trend)
            
            # Volume-price correlation
            if len(closes) >= 20 and len(volumes) >= 20:
                price_returns = np.diff(closes[-20:]) / closes[-21:-1]
                volume_changes = np.diff(volumes[-20:]) / volumes[-21:-1]
                
                if len(price_returns) > 0 and len(volume_changes) > 0:
                    correlation = np.corrcoef(price_returns, volume_changes)[0, 1]
                    profile['volume_price_correlation'] = float(correlation)
        
        except Exception as e:
            logger.error(f"Error analyzing volume profile: {e}")
        
        return profile
    
    def determine_structure_type(self, df: pd.DataFrame, trend: str) -> str:
        """Determine market structure type"""
        closes = df['close'].values
        
        if len(closes) < 10:
            return 'unknown'
        
        # Calculate price action characteristics
        returns = np.diff(closes) / closes[:-1]
        
        if len(returns) < 9:
            return 'unknown'
        
        # Check for consolidation
        recent_volatility = np.std(returns[-5:])
        avg_volatility = np.std(returns[-20:]) if len(returns) >= 20 else recent_volatility
        
        if recent_volatility < avg_volatility * 0.5:
            return 'consolidation'
        
        # Check for impulse moves
        recent_move = abs(closes[-1] - closes[-5]) / closes[-5]
        
        if recent_move > 0.05:  # 5% move in 5 periods
            if trend == 'uptrend' and closes[-1] > closes[-5]:
                return 'impulse_up'
            elif trend == 'downtrend' and closes[-1] < closes[-5]:
                return 'impulse_down'
        
        # Check for corrections
        if trend == 'uptrend' and closes[-1] < closes[-5]:
            return 'correction_down'
        elif trend == 'downtrend' and closes[-1] > closes[-5]:
            return 'correction_up'
        
        return 'normal'
    
    async def generate_signals_from_analysis(self, analysis: TechnicalAnalysis) -> List[Dict[str, Any]]:
        """Generate trading signals from technical analysis"""
        signals = []
        
        try:
            # RSI signals
            if 'rsi' in analysis.indicators:
                rsi = analysis.indicators['rsi']
                if rsi.signal == 'oversold':
                    signals.append({
                        'type': 'BUY',
                        'indicator': 'RSI',
                        'strength': 'medium',
                        'reason': f'RSI oversold at {rsi.value:.1f}'
                    })
                elif rsi.signal == 'overbought':
                    signals.append({
                        'type': 'SELL',
                        'indicator': 'RSI',
                        'strength': 'medium',
                        'reason': f'RSI overbought at {rsi.value:.1f}'
                    })
            
            # MACD signals
            if 'macd' in analysis.indicators:
                macd = analysis.indicators['macd']
                if 'bullish_cross' in macd.signal:
                    signals.append({
                        'type': 'BUY',
                        'indicator': 'MACD',
                        'strength': 'strong',
                        'reason': 'MACD bullish crossover'
                    })
                elif 'bearish_cross' in macd.signal:
                    signals.append({
                        'type': 'SELL',
                        'indicator': 'MACD',
                        'strength': 'strong',
                        'reason': 'MACD bearish crossover'
                    })
            
            # Bollinger Bands signals
            if 'bollinger' in analysis.indicators:
                bb = analysis.indicators['bollinger']
                if bb.signal == 'oversold':
                    signals.append({
                        'type': 'BUY',
                        'indicator': 'Bollinger',
                        'strength': 'medium',
                        'reason': 'Price at lower Bollinger Band'
                    })
                elif bb.signal == 'overbought':
                    signals.append({
                        'type': 'SELL',
                        'indicator': 'Bollinger',
                        'strength': 'medium',
                        'reason': 'Price at upper Bollinger Band'
                    })
            
            # Moving Average signals
            if 'ma_cross' in analysis.indicators:
                ma_cross = analysis.indicators['ma_cross']
                if ma_cross.signal == 'golden_cross':
                    signals.append({
                        'type': 'BUY',
                        'indicator': 'MA Cross',
                        'strength': 'strong',
                        'reason': 'Golden cross detected'
                    })
                elif ma_cross.signal == 'death_cross':
                    signals.append({
                        'type': 'SELL',
                        'indicator': 'MA Cross',
                        'strength': 'strong',
                        'reason': 'Death cross detected'
                    })
            
            # Market structure signals
            market_structure = analysis.market_structure
            if market_structure.trend == 'uptrend' and market_structure.trend_strength > 0.5:
                signals.append({
                    'type': 'BUY',
                    'indicator': 'Trend',
                    'strength': 'strong',
                    'reason': 'Strong uptrend detected'
                })
            elif market_structure.trend == 'downtrend' and market_structure.trend_strength > 0.5:
                signals.append({
                    'type': 'SELL',
                    'indicator': 'Trend',
                    'strength': 'strong',
                    'reason': 'Strong downtrend detected'
                })
            
            # Candlestick pattern signals
            for pattern in analysis.patterns:
                if 'bullish' in pattern.pattern_type:
                    signals.append({
                        'type': 'BUY',
                        'indicator': 'Pattern',
                        'strength': 'medium' if pattern.confidence > 0.7 else 'weak',
                        'reason': f'{pattern.name} pattern detected'
                    })
                elif 'bearish' in pattern.pattern_type:
                    signals.append({
                        'type': 'SELL',
                        'indicator': 'Pattern',
                        'strength': 'medium' if pattern.confidence > 0.7 else 'weak',
                        'reason': f'{pattern.name} pattern detected'
                    })
            
            # Consolidation breakout signals
            if market_structure.structure_type == 'consolidation':
                # Watch for breakout from consolidation
                signals.append({
                    'type': 'WATCH',
                    'indicator': 'Structure',
                    'strength': 'info',
                    'reason': 'Price in consolidation, watch for breakout'
                })
        
        except Exception as e:
            logger.error(f"Error generating signals from analysis: {e}")
        
        return signals
    
    async def create_summary(self, analysis: TechnicalAnalysis) -> Dict[str, Any]:
        """Create analysis summary"""
        summary = {
            'timestamp': analysis.timestamp.isoformat(),
            'symbol': analysis.symbol,
            'timeframe': analysis.timeframe,
            'overall_bias': 'neutral',
            'confidence': 0.0,
            'key_signals': [],
            'risk_level': 'medium',
            'recommendation': 'HOLD'
        }
        
        try:
            # Count buy/sell signals
            buy_signals = [s for s in analysis.signals if s['type'] in ['BUY', 'STRONG_BUY']]
            sell_signals = [s for s in analysis.signals if s['type'] in ['SELL', 'STRONG_SELL']]
            
            # Determine overall bias
            if len(buy_signals) > len(sell_signals):
                summary['overall_bias'] = 'bullish'
                summary['confidence'] = min(1.0, len(buy_signals) / 10)
            elif len(sell_signals) > len(buy_signals):
                summary['overall_bias'] = 'bearish'
                summary['confidence'] = min(1.0, len(sell_signals) / 10)
            
            # Add key signals
            strong_signals = [s for s in analysis.signals if s['strength'] == 'strong']
            if strong_signals:
                summary['key_signals'] = strong_signals[:3]  # Top 3 strong signals
            
            # Determine risk level based on volatility
            volatility = analysis.market_structure.volatility
            if volatility > 2.0:
                summary['risk_level'] = 'high'
            elif volatility < 0.5:
                summary['risk_level'] = 'low'
            
            # Generate recommendation
            if summary['overall_bias'] == 'bullish' and summary['confidence'] > 0.6:
                summary['recommendation'] = 'BUY'
            elif summary['overall_bias'] == 'bearish' and summary['confidence'] > 0.6:
                summary['recommendation'] = 'SELL'
            elif summary['overall_bias'] == 'bullish' and summary['confidence'] > 0.4:
                summary['recommendation'] = 'WEAK_BUY'
            elif summary['overall_bias'] == 'bearish' and summary['confidence'] > 0.4:
                summary['recommendation'] = 'WEAK_SELL'
        
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
        
        return summary

# Concrete Technical Strategy Implementations

class RSIStrategy(BaseTechnicalStrategy):
    """RSI-based trading strategy"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, TechnicalStrategyType.RSI_STRATEGY)
        self.oversold_level = 30
        self.overbought_level = 70
        self.middle_level = 50
    
    async def analyze(self, market_data: pd.DataFrame) -> TechnicalAnalysis:
        """Perform RSI-focused analysis"""
        try:
            # Calculate all indicators
            indicators = await self.calculate_indicators(market_data)
            
            # Focus on RSI analysis
            if 'rsi' not in indicators:
                # Calculate RSI if not already calculated
                closes = market_data['close'].values
                rsi_period = self.default_params.get('rsi_period', 14)
                if len(closes) >= rsi_period:
                    rsi_values = talib.RSI(closes, timeperiod=rsi_period)
                    if not np.isnan(rsi_values[-1]):
                        indicators['rsi'] = TechnicalIndicator(
                            name='RSI',
                            value=float(rsi_values[-1]),
                            signal=self.get_rsi_signal(rsi_values[-1]),
                            level=rsi_values[-1],
                            metadata={'period': rsi_period}
                        )
            
            # Detect patterns
            patterns = await self.detect_candlestick_patterns(market_data)
            
            # Analyze market structure
            market_structure = await self.analyze_market_structure(market_data)
            
            # Generate signals
            signals = await self.generate_signals_from_analysis(
                TechnicalAnalysis(
                    timestamp=datetime.now(),
                    symbol=self.config.symbol if self.config else 'BTCUSDT',
                    timeframe=self.config.timeframe if self.config else '1h',
                    indicators=indicators,
                    patterns=patterns,
                    market_structure=market_structure,
                    signals=[],
                    summary={}
                )
            )
            
            # Create summary
            analysis = TechnicalAnalysis(
                timestamp=datetime.now(),
                symbol=self.config.symbol if self.config else 'BTCUSDT',
                timeframe=self.config.timeframe if self.config else '1h',
                indicators=indicators,
                patterns=patterns,
                market_structure=market_structure,
                signals=signals,
                summary={}
            )
            
            analysis.summary = await self.create_summary(analysis)
            
            # Store analysis
            self.analysis_history.append(analysis)
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error in RSI analysis: {e}")
            raise
    
    async def generate_signal(self, analysis: TechnicalAnalysis) -> Optional[TradingSignal]:
        """Generate RSI-based trading signal"""
        try:
            if 'rsi' not in analysis.indicators:
                return None
            
            rsi = analysis.indicators['rsi']
            current_price = analysis.market_structure.metadata.get('current_price', 0)
            
            # RSI-based signals
            signal_type = SignalType.HOLD
            confidence = 0.0
            
            if rsi.value <= self.oversold_level:
                # Oversold condition
                signal_type = SignalType.BUY
                confidence = min(0.95, (self.oversold_level - rsi.value) / self.oversold_level + 0.3)
                
                # Increase confidence if other indicators confirm
                if 'macd' in analysis.indicators and 'bullish' in analysis.indicators['macd'].signal:
                    confidence = min(0.95, confidence + 0.2)
                if analysis.market_structure.trend == 'uptrend':
                    confidence = min(0.95, confidence + 0.1)
            
            elif rsi.value >= self.overbought_level:
                # Overbought condition
                signal_type = SignalType.SELL
                confidence = min(0.95, (rsi.value - self.overbought_level) / (100 - self.overbought_level) + 0.3)
                
                # Increase confidence if other indicators confirm
                if 'macd' in analysis.indicators and 'bearish' in analysis.indicators['macd'].signal:
                    confidence = min(0.95, confidence + 0.2)
                if analysis.market_structure.trend == 'downtrend':
                    confidence = min(0.95, confidence + 0.1)
            
            elif 45 <= rsi.value <= 55:
                # Neutral zone - look for divergences
                divergence_signal = await self.check_rsi_divergence(analysis)
                if divergence_signal:
                    signal_type = divergence_signal[0]
                    confidence = divergence_signal[1]
            
            if signal_type == SignalType.HOLD or confidence < 0.6:
                return None
            
            # Create trading signal
            signal = TradingSignal(
                symbol=analysis.symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timestamp=analysis.timestamp,
                strategy_name=self.name,
                model_name="RSI_Strategy",
                features={'rsi': rsi.value},
                metadata={
                    'rsi_level': rsi.value,
                    'rsi_signal': rsi.signal,
                    'analysis_summary': analysis.summary
                }
            )
            
            self.signals_history.append(signal)
            return signal
        
        except Exception as e:
            logger.error(f"Error generating RSI signal: {e}")
            return None
    
    async def check_rsi_divergence(self, analysis: TechnicalAnalysis) -> Optional[Tuple[SignalType, float]]:
        """Check for RSI divergences"""
        # This would require historical RSI and price data
        # For now, return None
        return None

class MACDStrategy(BaseTechnicalStrategy):
    """MACD-based trading strategy"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, TechnicalStrategyType.MACD_STRATEGY)
        self.signal_line_cross_confirmation = True
        self.zero_line_cross_confirmation = False
    
    async def analyze(self, market_data: pd.DataFrame) -> TechnicalAnalysis:
        """Perform MACD-focused analysis"""
        try:
            # Calculate all indicators
            indicators = await self.calculate_indicators(market_data)
            
            # Ensure MACD is calculated
            if 'macd' not in indicators:
                closes = market_data['close'].values
                macd_fast = self.default_params.get('macd_fast', 12)
                macd_slow = self.default_params.get('macd_slow', 26)
                macd_signal = self.default_params.get('macd_signal', 9)
                
                if len(closes) >= macd_slow:
                    macd, signal_line, histogram = talib.MACD(
                        closes,
                        fastperiod=macd_fast,
                        slowperiod=macd_slow,
                        signalperiod=macd_signal
                    )
                    if not np.isnan(macd[-1]):
                        macd_signal_val = self.get_macd_signal(macd[-1], signal_line[-1], histogram[-1])
                        indicators['macd'] = TechnicalIndicator(
                            name='MACD',
                            value=float(macd[-1]),
                            signal=macd_signal_val,
                            metadata={
                                'signal_line': float(signal_line[-1]),
                                'histogram': float(histogram[-1]),
                                'cross': macd_signal_val
                            }
                        )
            
            # Detect patterns
            patterns = await self.detect_candlestick_patterns(market_data)
            
            # Analyze market structure
            market_structure = await self.analyze_market_structure(market_data)
            
            # Generate signals
            signals = await self.generate_signals_from_analysis(
                TechnicalAnalysis(
                    timestamp=datetime.now(),
                    symbol=self.config.symbol if self.config else 'BTCUSDT',
                    timeframe=self.config.timeframe if self.config else '1h',
                    indicators=indicators,
                    patterns=patterns,
                    market_structure=market_structure,
                    signals=[],
                    summary={}
                )
            )
            
            # Create summary
            analysis = TechnicalAnalysis(
                timestamp=datetime.now(),
                symbol=self.config.symbol if self.config else 'BTCUSDT',
                timeframe=self.config.timeframe if self.config else '1h',
                indicators=indicators,
                patterns=patterns,
                market_structure=market_structure,
                signals=signals,
                summary={}
            )
            
            analysis.summary = await self.create_summary(analysis)
            
            # Store analysis
            self.analysis_history.append(analysis)
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error in MACD analysis: {e}")
            raise
    
    async def generate_signal(self, analysis: TechnicalAnalysis) -> Optional[TradingSignal]:
        """Generate MACD-based trading signal"""
        try:
            if 'macd' not in analysis.indicators:
                return None
            
            macd_indicator = analysis.indicators['macd']
            current_price = analysis.market_structure.metadata.get('current_price', 0)
            
            # MACD-based signals
            signal_type = SignalType.HOLD
            confidence = 0.0
            
            # Check for MACD crossovers
            if 'bullish_cross' in macd_indicator.signal:
                signal_type = SignalType.BUY
                confidence = 0.7
                
                # Increase confidence if histogram is increasing
                if 'histogram' in macd_indicator.metadata:
                    histogram = macd_indicator.metadata['histogram']
                    if histogram > 0:
                        confidence = min(0.95, confidence + 0.1)
                
                # Check trend alignment
                if analysis.market_structure.trend == 'uptrend':
                    confidence = min(0.95, confidence + 0.15)
            
            elif 'bearish_cross' in macd_indicator.signal:
                signal_type = SignalType.SELL
                confidence = 0.7
                
                # Increase confidence if histogram is decreasing
                if 'histogram' in macd_indicator.metadata:
                    histogram = macd_indicator.metadata['histogram']
                    if histogram < 0:
                        confidence = min(0.95, confidence + 0.1)
                
                # Check trend alignment
                if analysis.market_structure.trend == 'downtrend':
                    confidence = min(0.95, confidence + 0.15)
            
            # Check zero line cross
            if self.zero_line_cross_confirmation:
                macd_value = macd_indicator.value
                if macd_value > 0 and 'bullish' in macd_indicator.signal:
                    confidence = min(0.95, confidence + 0.1)
                elif macd_value < 0 and 'bearish' in macd_indicator.signal:
                    confidence = min(0.95, confidence + 0.1)
            
            if signal_type == SignalType.HOLD or confidence < 0.65:
                return None
            
            # Check for divergences
            divergence_signal = await self.check_macd_divergence(analysis)
            if divergence_signal:
                if divergence_signal[0] == signal_type:
                    confidence = min(0.95, confidence + 0.15)
                else:
                    # Conflicting signals - reduce confidence
                    confidence = max(0.4, confidence - 0.2)
            
            # Create trading signal
            signal = TradingSignal(
                symbol=analysis.symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timestamp=analysis.timestamp,
                strategy_name=self.name,
                model_name="MACD_Strategy",
                features={'macd': macd_indicator.value},
                metadata={
                    'macd_signal': macd_indicator.signal,
                    'macd_histogram': macd_indicator.metadata.get('histogram', 0),
                    'analysis_summary': analysis.summary
                }
            )
            
            self.signals_history.append(signal)
            return signal
        
        except Exception as e:
            logger.error(f"Error generating MACD signal: {e}")
            return None
    
    async def check_macd_divergence(self, analysis: TechnicalAnalysis) -> Optional[Tuple[SignalType, float]]:
        """Check for MACD divergences"""
        # This would require historical MACD and price data
        # For now, return None
        return None

class BollingerBandsStrategy(BaseTechnicalStrategy):
    """Bollinger Bands-based trading strategy"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, TechnicalStrategyType.BOLLINGER_BANDS)
        self.bb_period = 20
        self.bb_std = 2
        self.squeeze_threshold = 0.1
    
    async def analyze(self, market_data: pd.DataFrame) -> TechnicalAnalysis:
        """Perform Bollinger Bands-focused analysis"""
        try:
            # Calculate all indicators
            indicators = await self.calculate_indicators(market_data)
            
            # Ensure Bollinger Bands are calculated
            if 'bollinger' not in indicators:
                closes = market_data['close'].values
                bb_period = self.default_params.get('bollinger_period', 20)
                bb_std = self.default_params.get('bollinger_std', 2)
                
                if len(closes) >= bb_period:
                    upper, middle, lower = talib.BBANDS(
                        closes,
                        timeperiod=bb_period,
                        nbdevup=bb_std,
                        nbdevdn=bb_std,
                        matype=0
                    )
                    if not np.isnan(upper[-1]):
                        bb_signal = self.get_bollinger_signal(closes[-1], upper[-1], lower[-1])
                        indicators['bollinger'] = TechnicalIndicator(
                            name='Bollinger Bands',
                            value=float((closes[-1] - lower[-1]) / (upper[-1] - lower[-1])),
                            signal=bb_signal,
                            metadata={
                                'upper': float(upper[-1]),
                                'middle': float(middle[-1]),
                                'lower': float(lower[-1]),
                                'width': float((upper[-1] - lower[-1]) / middle[-1])
                            }
                        )
            
            # Detect patterns
            patterns = await self.detect_candlestick_patterns(market_data)
            
            # Analyze market structure
            market_structure = await self.analyze_market_structure(market_data)
            
            # Check for Bollinger Band squeeze
            if 'bollinger' in indicators:
                bb_width = indicators['bollinger'].metadata.get('width', 0)
                if bb_width < self.squeeze_threshold:
                    indicators['bollinger_squeeze'] = TechnicalIndicator(
                        name='Bollinger Squeeze',
                        value=bb_width,
                        signal='squeeze',
                        metadata={'threshold': self.squeeze_threshold}
                    )
            
            # Generate signals
            signals = await self.generate_signals_from_analysis(
                TechnicalAnalysis(
                    timestamp=datetime.now(),
                    symbol=self.config.symbol if self.config else 'BTCUSDT',
                    timeframe=self.config.timeframe if self.config else '1h',
                    indicators=indicators,
                    patterns=patterns,
                    market_structure=market_structure,
                    signals=[],
                    summary={}
                )
            )
            
            # Create summary
            analysis = TechnicalAnalysis(
                timestamp=datetime.now(),
                symbol=self.config.symbol if self.config else 'BTCUSDT',
                timeframe=self.config.timeframe if self.config else '1h',
                indicators=indicators,
                patterns=patterns,
                market_structure=market_structure,
                signals=signals,
                summary={}
            )
            
            analysis.summary = await self.create_summary(analysis)
            
            # Store analysis
            self.analysis_history.append(analysis)
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error in Bollinger Bands analysis: {e}")
            raise
    
    async def generate_signal(self, analysis: TechnicalAnalysis) -> Optional[TradingSignal]:
        """Generate Bollinger Bands-based trading signal"""
        try:
            if 'bollinger' not in analysis.indicators:
                return None
            
            bb = analysis.indicators['bollinger']
            current_price = analysis.market_structure.metadata.get('current_price', 0)
            
            # Bollinger Bands signals
            signal_type = SignalType.HOLD
            confidence = 0.0
            
            # Band touch signals
            if bb.signal == 'oversold':
                signal_type = SignalType.BUY
                confidence = 0.7
                
                # Increase confidence if price is at lower band and RSI confirms
                if 'rsi' in analysis.indicators and analysis.indicators['rsi'].signal == 'oversold':
                    confidence = min(0.95, confidence + 0.15)
                
                # Check for squeeze breakout potential
                if 'bollinger_squeeze' in analysis.indicators:
                    confidence = min(0.95, confidence + 0.1)
            
            elif bb.signal == 'overbought':
                signal_type = SignalType.SELL
                confidence = 0.7
                
                # Increase confidence if price is at upper band and RSI confirms
                if 'rsi' in analysis.indicators and analysis.indicators['rsi'].signal == 'overbought':
                    confidence = min(0.95, confidence + 0.15)
                
                # Check for squeeze breakdown potential
                if 'bollinger_squeeze' in analysis.indicators:
                    confidence = min(0.95, confidence + 0.1)
            
            # Middle band as dynamic support/resistance
            bb_position = bb.value  # 0-1 range from lower to upper band
            if 0.4 <= bb_position <= 0.6:
                # Price near middle band - look for momentum
                if 'macd' in analysis.indicators and 'bullish' in analysis.indicators['macd'].signal:
                    signal_type = SignalType.BUY
                    confidence = 0.6
                elif 'macd' in analysis.indicators and 'bearish' in analysis.indicators['macd'].signal:
                    signal_type = SignalType.SELL
                    confidence = 0.6
            
            if signal_type == SignalType.HOLD or confidence < 0.6:
                return None
            
            # Check band width for volatility confirmation
            bb_width = bb.metadata.get('width', 0)
            if bb_width > 0.05:  # Wide bands indicate high volatility
                confidence = min(0.95, confidence + 0.1)
            
            # Create trading signal
            signal = TradingSignal(
                symbol=analysis.symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timestamp=analysis.timestamp,
                strategy_name=self.name,
                model_name="BollingerBands_Strategy",
                features={
                    'bb_position': bb_position,
                    'bb_width': bb_width
                },
                metadata={
                    'bb_signal': bb.signal,
                    'bb_upper': bb.metadata.get('upper', 0),
                    'bb_lower': bb.metadata.get('lower', 0),
                    'bb_middle': bb.metadata.get('middle', 0),
                    'analysis_summary': analysis.summary
                }
            )
            
            self.signals_history.append(signal)
            return signal
        
        except Exception as e:
            logger.error(f"Error generating Bollinger Bands signal: {e}")
            return None

class IchimokuStrategy(BaseTechnicalStrategy):
    """Ichimoku Cloud-based trading strategy"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, TechnicalStrategyType.ICHIMOKU)
        self.tenkan_period = 9
        self.kijun_period = 26
        self.senkou_b_period = 52
        self.chikou_shift = 26
    
    async def analyze(self, market_data: pd.DataFrame) -> TechnicalAnalysis:
        """Perform Ichimoku Cloud analysis"""
        try:
            # Calculate all indicators
            indicators = await self.calculate_indicators(market_data)
            
            # Ensure Ichimoku is calculated
            if 'ichimoku' not in indicators:
                closes = market_data['close'].values
                highs = market_data['high'].values
                lows = market_data['low'].values
                
                tenkan_period = self.default_params.get('ichimoku_tenkan', 9)
                kijun_period = self.default_params.get('ichimoku_kijun', 26)
                senkou_b_period = self.default_params.get('ichimoku_senkou_b', 52)
                chikou_shift = self.default_params.get('ichimoku_chikou', 26)
                
                if len(highs) >= senkou_b_period and len(lows) >= senkou_b_period:
                    # Tenkan-sen (Conversion Line)
                    tenkan_high = np.max(highs[-tenkan_period:])
                    tenkan_low = np.min(lows[-tenkan_period:])
                    tenkan_sen = (tenkan_high + tenkan_low) / 2
                    
                    # Kijun-sen (Base Line)
                    kijun_high = np.max(highs[-kijun_period:])
                    kijun_low = np.min(lows[-kijun_period:])
                    kijun_sen = (kijun_high + kijun_low) / 2
                    
                    # Senkou Span A (Leading Span A)
                    senkou_span_a = (tenkan_sen + kijun_sen) / 2
                    
                    # Senkou Span B (Leading Span B)
                    senkou_b_high = np.max(highs[-senkou_b_period:])
                    senkou_b_low = np.min(lows[-senkou_b_period:])
                    senkou_span_b = (senkou_b_high + senkou_b_low) / 2
                    
                    ichimoku_signal = self.get_ichimoku_signal(
                        closes[-1], tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b
                    )
                    
                    indicators['ichimoku'] = TechnicalIndicator(
                        name='Ichimoku Cloud',
                        value=float(senkou_span_a - senkou_span_b),
                        signal=ichimoku_signal,
                        metadata={
                            'tenkan_sen': float(tenkan_sen),
                            'kijun_sen': float(kijun_sen),
                            'senkou_span_a': float(senkou_span_a),
                            'senkou_span_b': float(senkou_span_b),
                            'cloud_top': float(max(senkou_span_a, senkou_span_b)),
                            'cloud_bottom': float(min(senkou_span_a, senkou_span_b)),
                            'cloud_thickness': float(abs(senkou_span_a - senkou_span_b))
                        }
                    )
            
            # Detect patterns
            patterns = await self.detect_candlestick_patterns(market_data)
            
            # Analyze market structure
            market_structure = await self.analyze_market_structure(market_data)
            
            # Generate signals
            signals = await self.generate_signals_from_analysis(
                TechnicalAnalysis(
                    timestamp=datetime.now(),
                    symbol=self.config.symbol if self.config else 'BTCUSDT',
                    timeframe=self.config.timeframe if self.config else '1h',
                    indicators=indicators,
                    patterns=patterns,
                    market_structure=market_structure,
                    signals=[],
                    summary={}
                )
            )
            
            # Create summary
            analysis = TechnicalAnalysis(
                timestamp=datetime.now(),
                symbol=self.config.symbol if self.config else 'BTCUSDT',
                timeframe=self.config.timeframe if self.config else '1h',
                indicators=indicators,
                patterns=patterns,
                market_structure=market_structure,
                signals=signals,
                summary={}
            )
            
            analysis.summary = await self.create_summary(analysis)
            
            # Store analysis
            self.analysis_history.append(analysis)
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error in Ichimoku analysis: {e}")
            raise
    
    async def generate_signal(self, analysis: TechnicalAnalysis) -> Optional[TradingSignal]:
        """Generate Ichimoku Cloud-based trading signal"""
        try:
            if 'ichimoku' not in analysis.indicators:
                return None
            
            ichimoku = analysis.indicators['ichimoku']
            current_price = analysis.market_structure.metadata.get('current_price', 0)
            
            # Ichimoku signals
            signal_type = SignalType.HOLD
            confidence = 0.0
            
            # Cloud position signals
            cloud_top = ichimoku.metadata.get('cloud_top', 0)
            cloud_bottom = ichimoku.metadata.get('cloud_bottom', 0)
            tenkan_sen = ichimoku.metadata.get('tenkan_sen', 0)
            kijun_sen = ichimoku.metadata.get('kijun_sen', 0)
            
            # Price relative to cloud
            if current_price > cloud_top:
                # Price above cloud - bullish
                if tenkan_sen > kijun_sen:
                    # Tenkan above Kijun - strong bullish
                    signal_type = SignalType.BUY
                    confidence = 0.8
                else:
                    signal_type = SignalType.BUY
                    confidence = 0.6
            
            elif current_price < cloud_bottom:
                # Price below cloud - bearish
                if tenkan_sen < kijun_sen:
                    # Tenkan below Kijun - strong bearish
                    signal_type = SignalType.SELL
                    confidence = 0.8
                else:
                    signal_type = SignalType.SELL
                    confidence = 0.6
            
            else:
                # Price inside cloud - neutral/consolidation
                # Look for Kijun cross or cloud breakout
                if tenkan_sen > kijun_sen and current_price > kijun_sen:
                    # Bullish Kijun cross
                    signal_type = SignalType.BUY
                    confidence = 0.65
                elif tenkan_sen < kijun_sen and current_price < kijun_sen:
                    # Bearish Kijun cross
                    signal_type = SignalType.SELL
                    confidence = 0.65
            
            # Adjust confidence based on cloud thickness
            cloud_thickness = ichimoku.metadata.get('cloud_thickness', 0)
            if cloud_thickness > 0:
                # Thicker cloud provides stronger support/resistance
                confidence = min(0.95, confidence + (cloud_thickness / current_price) * 10)
            
            # Check Tenkan-Kijun cross
            tk_cross = tenkan_sen - kijun_sen
            if abs(tk_cross) > 0:
                if tk_cross > 0 and signal_type == SignalType.BUY:
                    confidence = min(0.95, confidence + 0.1)
                elif tk_cross < 0 and signal_type == SignalType.SELL:
                    confidence = min(0.95, confidence + 0.1)
            
            if signal_type == SignalType.HOLD or confidence < 0.6:
                return None
            
            # Create trading signal
            signal = TradingSignal(
                symbol=analysis.symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timestamp=analysis.timestamp,
                strategy_name=self.name,
                model_name="Ichimoku_Strategy",
                features={
                    'cloud_position': ichimoku.value,
                    'tenkan_kijun_diff': tk_cross
                },
                metadata={
                    'ichimoku_signal': ichimoku.signal,
                    'cloud_top': cloud_top,
                    'cloud_bottom': cloud_bottom,
                    'tenkan_sen': tenkan_sen,
                    'kijun_sen': kijun_sen,
                    'analysis_summary': analysis.summary
                }
            )
            
            self.signals_history.append(signal)
            return signal
        
        except Exception as e:
            logger.error(f"Error generating Ichimoku signal: {e}")
            return None

# Technical Strategy Factory
class TechnicalStrategyFactory:
    """Factory for creating technical trading strategies"""
    
    @staticmethod
    def create_strategy(strategy_type: Union[str, TechnicalStrategyType], 
                       config_manager: ConfigManager) -> BaseTechnicalStrategy:
        """Create strategy instance based on type"""
        if isinstance(strategy_type, str):
            try:
                strategy_type = TechnicalStrategyType(strategy_type.lower())
            except ValueError:
                raise ValueError(f"Unknown technical strategy type: {strategy_type}")
        
        if strategy_type == TechnicalStrategyType.RSI_STRATEGY:
            return RSIStrategy(config_manager)
        
        elif strategy_type == TechnicalStrategyType.MACD_STRATEGY:
            return MACDStrategy(config_manager)
        
        elif strategy_type == TechnicalStrategyType.BOLLINGER_BANDS:
            return BollingerBandsStrategy(config_manager)
        
        elif strategy_type == TechnicalStrategyType.ICHIMOKU:
            return IchimokuStrategy(config_manager)
        
        elif strategy_type == TechnicalStrategyType.MOVING_AVERAGES:
            return MovingAveragesStrategy(config_manager)
        
        elif strategy_type == TechnicalStrategyType.STOCHASTIC:
            return StochasticStrategy(config_manager)
        
        elif strategy_type == TechnicalStrategyType.PARABOLIC_SAR:
            return ParabolicSARStrategy(config_manager)
        
        elif strategy_type == TechnicalStrategyType.MULTI_TIMEFRAME:
            return MultiTimeframeStrategy(config_manager)
        
        else:
            raise ValueError(f"Technical strategy type not implemented: {strategy_type}")

# Additional Strategy Classes (simplified implementations)

class MovingAveragesStrategy(BaseTechnicalStrategy):
    """Moving Averages-based strategy"""
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, TechnicalStrategyType.MOVING_AVERAGES)
    
    async def analyze(self, market_data: pd.DataFrame) -> TechnicalAnalysis:
        """MA-focused analysis"""
        # Implementation
        pass
    
    async def generate_signal(self, analysis: TechnicalAnalysis) -> Optional[TradingSignal]:
        """MA-based signal generation"""
        # Implementation
        pass

class StochasticStrategy(BaseTechnicalStrategy):
    """Stochastic oscillator strategy"""
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, TechnicalStrategyType.STOCHASTIC)
    
    async def analyze(self, market_data: pd.DataFrame) -> TechnicalAnalysis:
        """Stochastic-focused analysis"""
        # Implementation
        pass
    
    async def generate_signal(self, analysis: TechnicalAnalysis) -> Optional[TradingSignal]:
        """Stochastic-based signal generation"""
        # Implementation
        pass

class ParabolicSARStrategy(BaseTechnicalStrategy):
    """Parabolic SAR strategy"""
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, TechnicalStrategyType.PARABOLIC_SAR)
    
    async def analyze(self, market_data: pd.DataFrame) -> TechnicalAnalysis:
        """Parabolic SAR-focused analysis"""
        # Implementation
        pass
    
    async def generate_signal(self, analysis: TechnicalAnalysis) -> Optional[TradingSignal]:
        """Parabolic SAR-based signal generation"""
        # Implementation
        pass

class MultiTimeframeStrategy(BaseTechnicalStrategy):
    """Multi-timeframe analysis strategy"""
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, TechnicalStrategyType.MULTI_TIMEFRAME)
    
    async def analyze(self, market_data: pd.DataFrame) -> TechnicalAnalysis:
        """Multi-timeframe analysis"""
        # Implementation
        pass
    
    async def generate_signal(self, analysis: TechnicalAnalysis) -> Optional[TradingSignal]:
        """Multi-timeframe signal generation"""
        # Implementation
        pass

# Technical Strategy Manager
class TechnicalStrategyManager:
    """Manages multiple technical trading strategies"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.strategies: Dict[str, BaseTechnicalStrategy] = {}
        self.factory = TechnicalStrategyFactory()
        self.logger = setup_logger(__name__)
    
    async def create_strategy(self, config: StrategyConfig) -> bool:
        """Create and initialize a new technical strategy"""
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
                self.logger.info(f"Created technical strategy '{config.name}' ({config.strategy_type.value})")
                return True
            else:
                self.logger.error(f"Failed to initialize strategy '{config.name}'")
                return False
        
        except Exception as e:
            self.logger.error(f"Error creating technical strategy '{config.name}': {e}")
            return False
    
    async def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            self.logger.info(f"Removed strategy '{strategy_name}'")
            return True
        return False
    
    async def get_strategy(self, strategy_name: str) -> Optional[BaseTechnicalStrategy]:
        """Get strategy by name"""
        return self.strategies.get(strategy_name)
    
    async def get_all_strategies(self) -> Dict[str, BaseTechnicalStrategy]:
        """Get all strategies"""
        return self.strategies.copy()
    
    async def analyze_market(self, symbol: str, timeframe: str, 
                           strategy_name: Optional[str] = None) -> Dict[str, TechnicalAnalysis]:
        """Analyze market with strategies"""
        analyses = {}
        
        if strategy_name:
            # Analyze with specific strategy
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                try:
                    # Get market data (this would come from data collector)
                    market_data = pd.DataFrame()  # Placeholder
                    analysis = await strategy.analyze(market_data)
                    analyses[strategy_name] = analysis
                except Exception as e:
                    self.logger.error(f"Error analyzing with '{strategy_name}': {e}")
        else:
            # Analyze with all strategies
            for name, strategy in self.strategies.items():
                try:
                    if strategy.config and strategy.config.enabled:
                        # Get market data (this would come from data collector)
                        market_data = pd.DataFrame()  # Placeholder
                        analysis = await strategy.analyze(market_data)
                        analyses[name] = analysis
                except Exception as e:
                    self.logger.error(f"Error analyzing with '{name}': {e}")
        
        return analyses
    
    async def generate_signals(self, symbol: str, timeframe: str) -> List[TradingSignal]:
        """Generate signals from all active strategies"""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                if strategy.config and strategy.config.enabled:
                    # Analyze market
                    market_data = pd.DataFrame()  # Placeholder
                    analysis = await strategy.analyze(market_data)
                    
                    # Generate signal
                    signal = await strategy.generate_signal(analysis)
                    
                    if signal:
                        signals.append(signal)
            
            except Exception as e:
                self.logger.error(f"Error generating signal for strategy '{strategy_name}': {e}")
        
        return signals
    
    async def get_analysis_reports(self) -> Dict[str, Dict]:
        """Get analysis reports for all strategies"""
        reports = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                if strategy.analysis_history:
                    latest_analysis = strategy.analysis_history[-1]
                    reports[strategy_name] = {
                        'analysis': latest_analysis.summary,
                        'timestamp': latest_analysis.timestamp.isoformat(),
                        'signals_generated': len(latest_analysis.signals)
                    }
            except Exception as e:
                self.logger.error(f"Error getting analysis for '{strategy_name}': {e}")
                reports[strategy_name] = {"error": str(e)}
        
        return reports

# Example usage
async def example_usage():
    """Example of how to use the technical strategies"""
    # Create config manager
    config_manager = ConfigManager()
    
    # Create strategy manager
    strategy_manager = TechnicalStrategyManager(config_manager)
    
    # Create RSI strategy config
    rsi_config = StrategyConfig(
        name="btc_rsi_v1",
        strategy_type=TechnicalStrategyType.RSI_STRATEGY,
        symbol="BTCUSDT",
        timeframe="1h",
        confidence_threshold=0.65,
        risk_per_trade=1.0,
        parameters={
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30
        }
    )
    
    # Create MACD strategy config
    macd_config = StrategyConfig(
        name="btc_macd_v1",
        strategy_type=TechnicalStrategyType.MACD_STRATEGY,
        symbol="BTCUSDT",
        timeframe="4h",
        confidence_threshold=0.7,
        parameters={
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9
        }
    )
    
    # Create and initialize strategies
    await strategy_manager.create_strategy(rsi_config)
    await strategy_manager.create_strategy(macd_config)
    
    # Analyze market
    analyses = await strategy_manager.analyze_market("BTCUSDT", "1h")
    
    for strategy_name, analysis in analyses.items():
        print(f"\n{strategy_name} Analysis:")
        print(f"  Overall Bias: {analysis.summary.get('overall_bias', 'unknown')}")
        print(f"  Confidence: {analysis.summary.get('confidence', 0):.2f}")
        print(f"  Recommendation: {analysis.summary.get('recommendation', 'HOLD')}")
    
    # Generate signals
    signals = await strategy_manager.generate_signals("BTCUSDT", "1h")
    
    for signal in signals:
        print(f"\nSignal from {signal.strategy_name}:")
        print(f"  Type: {signal.signal_type.value}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Price: ${signal.price:.2f}")
    
    # Get analysis reports
    reports = await strategy_manager.get_analysis_reports()
    print(f"\nAnalysis Reports: {reports}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())