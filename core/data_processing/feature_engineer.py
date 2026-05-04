"""
Feature Engineering for Bitcoin Trading AI
Creates technical indicators, statistical features, and transforms for ML models
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path
from scipy import stats, signal
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
import ta  # Technical Analysis Library

# Optional tsfresh imports
try:
    from tsfresh import extract_features
    from tsfresh.feature_extraction import ComprehensiveFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
except ImportError:
    # Create fallback functions
    def extract_features(*args, **kwargs):
        raise NotImplementedError("tsfresh not available")
    
    class ComprehensiveFCParameters:
        pass
    
    def impute(*args, **kwargs):
        raise NotImplementedError("tsfresh not available")
    
    TSFRESH_AVAILABLE = False

# Import project modules
from config.settings import (
    DataSettings, ModelSettings, AppConstants,
    BASE_DIR, DATA_DIR
)
from config.config_manager import get_config

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ============ Feature Types ============
class FeatureType(str, Enum):
    """Types of features"""
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    TIME = "time"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    TRANSFORMED = "transformed"

class Timeframe(str, Enum):
    """Timeframes for feature calculation"""
    SHORT = "short"      # 5-20 periods
    MEDIUM = "medium"    # 20-50 periods
    LONG = "long"        # 50-200 periods
    VERY_LONG = "very_long"  # 200+ periods

# ============ Feature Configurations ============
@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    
    # Technical indicators
    moving_averages: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    rsi_periods: List[int] = field(default_factory=lambda: [14])
    bollinger_periods: List[int] = field(default_factory=lambda: [20])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_periods: List[int] = field(default_factory=lambda: [14])
    stoch_periods: List[int] = field(default_factory=lambda: [14])
    adx_periods: List[int] = field(default_factory=lambda: [14])
    cci_periods: List[int] = field(default_factory=lambda: [20])
    williams_periods: List[int] = field(default_factory=lambda: [14])
    
    # Statistical features
    returns_periods: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    volatility_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    skewness_periods: List[int] = field(default_factory=lambda: [20, 50, 100])
    kurtosis_periods: List[int] = field(default_factory=lambda: [20, 50, 100])
    quantile_periods: List[int] = field(default_factory=lambda: [20, 50, 100])
    
    # Time features
    include_time_features: bool = True
    include_day_of_week: bool = True
    include_hour_of_day: bool = True
    include_month: bool = True
    include_quarter: bool = True
    include_weekend: bool = True
    include_time_since_open: bool = True
    
    # Feature transformations
    use_log_transform: bool = True
    use_power_transform: bool = False
    use_boxcox: bool = True
    interaction_degree: int = 2
    polynomial_degree: int = 2
    
    # Feature selection
    feature_selection_method: str = "mutual_info"  # mutual_info, correlation, variance
    max_features: int = 100
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    
    # Normalization
    normalization_method: str = "standard"  # standard, minmax, robust
    scale_features: bool = True
    separate_scalers: bool = False
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_features < 10:
            logger.warning(f"max_features ({self.max_features}) is very low, minimum recommended is 10")
        if self.correlation_threshold > 0.99:
            logger.warning(f"correlation_threshold ({self.correlation_threshold}) is very high")

# ============ Base Feature Engineering ============
class BaseFeatureEngineer:
    """Base class for feature engineering"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.scalers = {}
        self.pca = None
        self.feature_importance = {}
        self.selected_features = []
        self.feature_groups = {}
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all features from raw data"""
        raise NotImplementedError
    
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features (normalization, scaling, etc.)"""
        raise NotImplementedError
    
    def select_features(self, features: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Select most important features"""
        raise NotImplementedError
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all features"""
        raise NotImplementedError

# ============ Technical Indicators ============
class TechnicalIndicatorEngineer:
    """Engineer technical indicators"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def calculate_moving_averages(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate various moving averages"""
        results = pd.DataFrame(index=df.index)
        
        for period in self.config.moving_averages:
            # Simple Moving Average
            results[f'sma_{period}'] = df[price_col].rolling(window=period).mean()
            
            # Weighted Moving Average
            weights = np.arange(1, period + 1)
            results[f'wma_{period}'] = df[price_col].rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
            
            # Hull Moving Average
            half_period = period // 2
            sqrt_period = int(np.sqrt(period))
            
            wma_half = df[price_col].rolling(window=half_period).apply(
                lambda x: np.dot(x, np.arange(1, half_period + 1)) / np.arange(1, half_period + 1).sum(),
                raw=True
            )
            
            wma_full = df[price_col].rolling(window=period).apply(
                lambda x: np.dot(x, np.arange(1, period + 1)) / np.arange(1, period + 1).sum(),
                raw=True
            )
            
            hma_raw = 2 * wma_half - wma_full
            results[f'hma_{period}'] = hma_raw.rolling(window=sqrt_period).apply(
                lambda x: np.dot(x, np.arange(1, sqrt_period + 1)) / np.arange(1, sqrt_period + 1).sum(),
                raw=True
            )
        
        return results
    
    def calculate_exponential_moving_averages(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate exponential moving averages"""
        results = pd.DataFrame(index=df.index)
        
        for period in self.config.ema_periods:
            results[f'ema_{period}'] = df[price_col].ewm(span=period, adjust=False).mean()
            
            # Double EMA
            ema = df[price_col].ewm(span=period, adjust=False).mean()
            dema = 2 * ema - ema.ewm(span=period, adjust=False).mean()
            results[f'dema_{period}'] = dema
            
            # Triple EMA
            ema1 = df[price_col].ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            tema = 3 * ema1 - 3 * ema2 + ema3
            results[f'tema_{period}'] = tema
        
        return results
    
    def calculate_rsi(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        results = pd.DataFrame(index=df.index)
        
        for period in self.config.rsi_periods:
            # Traditional RSI
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            results[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Wilder's RSI
            avg_gain = delta.where(delta > 0, 0).rolling(window=period).apply(
                lambda x: x.mean() if len(x) > 0 else 0, raw=True
            )
            avg_loss = (-delta.where(delta < 0, 0)).rolling(window=period).apply(
                lambda x: x.mean() if len(x) > 0 else 0, raw=True
            )
            
            # Handle division by zero
            avg_loss = avg_loss.replace(0, np.nan)
            rs_wilder = avg_gain / avg_loss
            results[f'rsi_wilder_{period}'] = 100 - (100 / (1 + rs_wilder))
            
            # RSI Smoothed
            results[f'rsi_smooth_{period}'] = results[f'rsi_{period}'].ewm(span=period//2).mean()
        
        return results
    
    def calculate_macd(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate MACD indicators"""
        results = pd.DataFrame(index=df.index)
        
        # Calculate MACD
        fast_ema = df[price_col].ewm(span=self.config.macd_fast, adjust=False).mean()
        slow_ema = df[price_col].ewm(span=self.config.macd_slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.config.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        results['macd_line'] = macd_line
        results['macd_signal'] = signal_line
        results['macd_histogram'] = histogram
        
        # MACD variations
        results['macd_ratio'] = macd_line / slow_ema  # Normalized MACD
        results['macd_slope'] = macd_line.diff()  # Rate of change
        
        # MACD with different periods
        for fast in [8, 12]:
            for slow in [26, 30]:
                if fast >= slow:
                    continue
                fast_ema_var = df[price_col].ewm(span=fast, adjust=False).mean()
                slow_ema_var = df[price_col].ewm(span=slow, adjust=False).mean()
                macd_var = fast_ema_var - slow_ema_var
                results[f'macd_{fast}_{slow}'] = macd_var
        
        return results
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        results = pd.DataFrame(index=df.index)
        
        for period in self.config.bollinger_periods:
            # Standard Bollinger Bands
            sma = df[price_col].rolling(window=period).mean()
            std = df[price_col].rolling(window=period).std()
            
            results[f'bb_upper_{period}'] = sma + (2 * std)
            results[f'bb_middle_{period}'] = sma
            results[f'bb_lower_{period}'] = sma - (2 * std)
            
            # Bollinger Band Width
            results[f'bb_width_{period}'] = (results[f'bb_upper_{period}'] - results[f'bb_lower_{period}']) / sma
            
            # %B indicator
            results[f'bb_percent_b_{period}'] = (df[price_col] - results[f'bb_lower_{period}']) / \
                                               (results[f'bb_upper_{period}'] - results[f'bb_lower_{period}'])
            
            # Bollinger Band Squeeze
            results[f'bb_squeeze_{period}'] = results[f'bb_width_{period}'].rolling(window=20).mean() / \
                                             (results[f'bb_width_{period}'] + 1e-8)
        
        return results
    
    def calculate_atr(self, df: pd.DataFrame, high_col: str = 'high', 
                     low_col: str = 'low', close_col: str = 'close') -> pd.DataFrame:
        """Calculate Average True Range"""
        results = pd.DataFrame(index=df.index)
        
        for period in self.config.atr_periods:
            # True Range
            high_low = df[high_col] - df[low_col]
            high_close_prev = abs(df[high_col] - df[close_col].shift())
            low_close_prev = abs(df[low_col] - df[close_col].shift())
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            
            # ATR
            results[f'atr_{period}'] = true_range.rolling(window=period).mean()
            
            # Normalized ATR
            results[f'natr_{period}'] = results[f'atr_{period}'] / df[close_col] * 100
            
            # ATR ratio (current vs historical)
            atr_ratio = results[f'atr_{period}'] / results[f'atr_{period}'].rolling(window=period*2).mean()
            results[f'atr_ratio_{period}'] = atr_ratio
        
        return results
    
    def calculate_stochastic(self, df: pd.DataFrame, high_col: str = 'high', 
                           low_col: str = 'low', close_col: str = 'close') -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        results = pd.DataFrame(index=df.index)
        
        for period in self.config.stoch_periods:
            # %K line
            low_min = df[low_col].rolling(window=period).min()
            high_max = df[high_col].rolling(window=period).max()
            
            results[f'stoch_k_{period}'] = 100 * (df[close_col] - low_min) / (high_max - low_min + 1e-8)
            
            # %D line (signal line)
            results[f'stoch_d_{period}'] = results[f'stoch_k_{period}'].rolling(window=3).mean()
            
            # Slow Stochastic
            results[f'stoch_slow_k_{period}'] = results[f'stoch_k_{period}'].rolling(window=3).mean()
            results[f'stoch_slow_d_{period}'] = results[f'stoch_slow_k_{period}'].rolling(window=3).mean()
            
            # Stochastic RSI
            rsi = 100 - (100 / (1 + df[close_col].diff().where(lambda x: x > 0, 0).rolling(window=period).mean() /
                               df[close_col].diff().where(lambda x: x < 0, 0).abs().rolling(window=period).mean()))
            stoch_rsi_k = (rsi - rsi.rolling(window=period).min()) / \
                         (rsi.rolling(window=period).max() - rsi.rolling(window=period).min() + 1e-8) * 100
            results[f'stoch_rsi_k_{period}'] = stoch_rsi_k
            results[f'stoch_rsi_d_{period}'] = stoch_rsi_k.rolling(window=3).mean()
        
        return results
    
    def calculate_adx(self, df: pd.DataFrame, high_col: str = 'high', 
                     low_col: str = 'low', close_col: str = 'close') -> pd.DataFrame:
        """Calculate Average Directional Index"""
        results = pd.DataFrame(index=df.index)
        
        for period in self.config.adx_periods:
            # Calculate +DM and -DM
            high_diff = df[high_col].diff()
            low_diff = df[low_col].diff().abs()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            # Calculate True Range
            tr1 = df[high_col] - df[low_col]
            tr2 = abs(df[high_col] - df[close_col].shift())
            tr3 = abs(df[low_col] - df[close_col].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Smooth the values
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / atr
            minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / atr
            
            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            results[f'adx_{period}'] = dx.rolling(window=period).mean()
            
            # Store DI values
            results[f'plus_di_{period}'] = plus_di
            results[f'minus_di_{period}'] = minus_di
            
            # ADX ratio
            results[f'adx_ratio_{period}'] = results[f'adx_{period}'] / results[f'adx_{period}'].rolling(window=period*2).mean()
        
        return results
    
    def calculate_cci(self, df: pd.DataFrame, high_col: str = 'high', 
                     low_col: str = 'low', close_col: str = 'close') -> pd.DataFrame:
        """Calculate Commodity Channel Index"""
        results = pd.DataFrame(index=df.index)
        
        for period in self.config.cci_periods:
            # Typical Price
            tp = (df[high_col] + df[low_col] + df[close_col]) / 3
            
            # Simple Moving Average of TP
            sma_tp = tp.rolling(window=period).mean()
            
            # Mean Deviation
            mean_dev = tp.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
            )
            
            # CCI
            results[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mean_dev + 1e-8)
            
            # CCI variations
            results[f'cci_sma_{period}'] = results[f'cci_{period}'].rolling(window=period//2).mean()
            results[f'cci_ema_{period}'] = results[f'cci_{period}'].ewm(span=period//2).mean()
        
        return results
    
    def calculate_williams_r(self, df: pd.DataFrame, high_col: str = 'high', 
                           low_col: str = 'low', close_col: str = 'close') -> pd.DataFrame:
        """Calculate Williams %R"""
        results = pd.DataFrame(index=df.index)
        
        for period in self.config.williams_periods:
            highest_high = df[high_col].rolling(window=period).max()
            lowest_low = df[low_col].rolling(window=period).min()
            
            results[f'williams_r_{period}'] = -100 * (highest_high - df[close_col]) / (highest_high - lowest_low + 1e-8)
            
            # Smoothed Williams %R
            results[f'williams_r_smooth_{period}'] = results[f'williams_r_{period}'].ewm(span=period//2).mean()
        
        return results
    
    def calculate_obv(self, df: pd.DataFrame, close_col: str = 'close', 
                     volume_col: str = 'volume') -> pd.DataFrame:
        """Calculate On-Balance Volume"""
        results = pd.DataFrame(index=df.index)
        
        # Basic OBV
        price_diff = df[close_col].diff()
        obv = pd.Series(0, index=df.index)
        
        for i in range(1, len(df)):
            if price_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + df[volume_col].iloc[i]
            elif price_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - df[volume_col].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        results['obv'] = obv
        
        # OBV variations
        results['obv_sma'] = obv.rolling(window=20).mean()
        results['obv_ema'] = obv.ewm(span=20).mean()
        
        # OBV ratio
        results['obv_ratio'] = obv / obv.rolling(window=50).mean()
        
        # Price-OBV divergence
        price_roc = df[close_col].pct_change(periods=5)
        obv_roc = obv.pct_change(periods=5)
        results['obv_divergence'] = price_roc - obv_roc
        
        return results
    
    def calculate_mfi(self, df: pd.DataFrame, high_col: str = 'high', 
                     low_col: str = 'low', close_col: str = 'close',
                     volume_col: str = 'volume') -> pd.DataFrame:
        """Calculate Money Flow Index"""
        results = pd.DataFrame(index=df.index)
        
        # Typical Price
        typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
        
        # Money Flow
        money_flow = typical_price * df[volume_col]
        
        # Positive and Negative Money Flow
        price_diff = typical_price.diff()
        positive_mf = money_flow.where(price_diff > 0, 0)
        negative_mf = money_flow.where(price_diff < 0, 0)
        
        # 14-period MFI
        period = 14
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()
        
        money_ratio = positive_mf_sum / (negative_mf_sum + 1e-8)
        results['mfi_14'] = 100 - (100 / (1 + money_ratio))
        
        # MFI variations
        for period in [10, 20, 30]:
            positive_mf_sum = positive_mf.rolling(window=period).sum()
            negative_mf_sum = negative_mf.rolling(window=period).sum()
            money_ratio = positive_mf_sum / (negative_mf_sum + 1e-8)
            results[f'mfi_{period}'] = 100 - (100 / (1 + money_ratio))
        
        return results
    
    def calculate_all_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        logger.info("Calculating technical indicators...")
        
        all_indicators = pd.DataFrame(index=df.index)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for technical indicators: {missing_cols}")
            # Create dummy columns
            for col in missing_cols:
                if col == 'close' and 'price' in df.columns:
                    df = df.rename(columns={'price': 'close'})
                else:
                    df[col] = df['close'] if 'close' in df.columns else df.iloc[:, 0]
        
        # Calculate indicators
        try:
            # Moving averages
            ma_features = self.calculate_moving_averages(df)
            all_indicators = pd.concat([all_indicators, ma_features], axis=1)
            
            # Exponential moving averages
            ema_features = self.calculate_exponential_moving_averages(df)
            all_indicators = pd.concat([all_indicators, ema_features], axis=1)
            
            # RSI
            rsi_features = self.calculate_rsi(df)
            all_indicators = pd.concat([all_indicators, rsi_features], axis=1)
            
            # MACD
            macd_features = self.calculate_macd(df)
            all_indicators = pd.concat([all_indicators, macd_features], axis=1)
            
            # Bollinger Bands
            bb_features = self.calculate_bollinger_bands(df)
            all_indicators = pd.concat([all_indicators, bb_features], axis=1)
            
            # ATR
            atr_features = self.calculate_atr(df)
            all_indicators = pd.concat([all_indicators, atr_features], axis=1)
            
            # Stochastic
            stoch_features = self.calculate_stochastic(df)
            all_indicators = pd.concat([all_indicators, stoch_features], axis=1)
            
            # ADX
            adx_features = self.calculate_adx(df)
            all_indicators = pd.concat([all_indicators, adx_features], axis=1)
            
            # CCI
            cci_features = self.calculate_cci(df)
            all_indicators = pd.concat([all_indicators, cci_features], axis=1)
            
            # Williams %R
            williams_features = self.calculate_williams_r(df)
            all_indicators = pd.concat([all_indicators, williams_features], axis=1)
            
            # OBV
            obv_features = self.calculate_obv(df)
            all_indicators = pd.concat([all_indicators, obv_features], axis=1)
            
            # MFI
            mfi_features = self.calculate_mfi(df)
            all_indicators = pd.concat([all_indicators, mfi_features], axis=1)
            
            # Additional TA-Lib indicators
            try:
                # Rate of Change
                for period in [5, 10, 20]:
                    all_indicators[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
                
                # Momentum
                for period in [5, 10, 20]:
                    all_indicators[f'momentum_{period}'] = talib.MOM(df['close'], timeperiod=period)
                
                # Parabolic SAR
                all_indicators['sar'] = talib.SAR(df['high'], df['low'])
                
                # Aroon
                aroon_down, aroon_up = talib.AROON(df['high'], df['low'])
                all_indicators['aroon_down'] = aroon_down
                all_indicators['aroon_up'] = aroon_up
                all_indicators['aroon_oscillator'] = aroon_up - aroon_down
                
                # Chaikin Oscillator
                all_indicators['chaikin_oscillator'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
                
            except Exception as e:
                logger.warning(f"Could not calculate TA-Lib indicators: {str(e)}")
            
            logger.info(f"Generated {len(all_indicators.columns)} technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
        
        return all_indicators

# ============ Statistical Features ============
class StatisticalFeatureEngineer:
    """Engineer statistical features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def calculate_returns(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate returns and log returns"""
        results = pd.DataFrame(index=df.index)
        
        for period in self.config.returns_periods:
            # Simple returns
            results[f'return_{period}'] = df[price_col].pct_change(periods=period)
            
            # Log returns
            results[f'log_return_{period}'] = np.log(df[price_col] / df[price_col].shift(period))
            
            # Cumulative returns
            results[f'cum_return_{period}'] = (1 + results[f'return_{period}']).cumprod() - 1
            
            # Rolling Sharpe ratio (using returns as proxy)
            if period >= 20:  # Need enough data
                returns = df[price_col].pct_change()
                results[f'sharpe_{period}'] = returns.rolling(window=period).mean() / \
                                            (returns.rolling(window=period).std() + 1e-8) * np.sqrt(252)
        
        return results
    
    def calculate_volatility(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate volatility measures"""
        results = pd.DataFrame(index=df.index)
        
        returns = df[price_col].pct_change()
        
        for period in self.config.volatility_periods:
            # Standard deviation (volatility)
            results[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
            
            # Historical volatility (using close-to-close)
            log_returns = np.log(df[price_col] / df[price_col].shift())
            results[f'hist_vol_{period}'] = log_returns.rolling(window=period).std() * np.sqrt(252)
            
            # Parkinson volatility (using high-low range)
            if 'high' in df.columns and 'low' in df.columns:
                parkinson = (1 / (4 * np.log(2))) * \
                           (np.log(df['high'] / df['low']) ** 2).rolling(window=period).sum()
                results[f'parkinson_vol_{period}'] = np.sqrt(parkinson / period) * np.sqrt(252)
            
            # Garman-Klass volatility
            if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
                gk = 0.5 * (np.log(df['high'] / df['low']) ** 2) - \
                     (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2)
                results[f'gk_vol_{period}'] = np.sqrt(gk.rolling(window=period).sum() / period) * np.sqrt(252)
            
            # Volatility ratio (current vs historical)
            if period >= 20:
                vol_ratio = results[f'volatility_{period}'] / \
                           results[f'volatility_{period}'].rolling(window=period*2).mean()
                results[f'vol_ratio_{period}'] = vol_ratio
        
        return results
    
    def calculate_skewness_kurtosis(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate skewness and kurtosis"""
        results = pd.DataFrame(index=df.index)
        
        returns = df[price_col].pct_change()
        
        for period in self.config.skewness_periods:
            # Skewness
            results[f'skewness_{period}'] = returns.rolling(window=period).skew()
            
            # Kurtosis
            results[f'kurtosis_{period}'] = returns.rolling(window=period).kurt()
            
            # Modified skewness/kurtosis
            if period in self.config.kurtosis_periods:
                # Fisher transformation of skewness
                results[f'skewness_fisher_{period}'] = 0.5 * np.log(
                    (1 + results[f'skewness_{period}']) / (1 - results[f'skewness_{period}'] + 1e-8)
                )
                
                # Excess kurtosis
                results[f'excess_kurtosis_{period}'] = results[f'kurtosis_{period}'] - 3
        
        return results
    
    def calculate_quantile_features(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate quantile-based features"""
        results = pd.DataFrame(index=df.index)
        
        for period in self.config.quantile_periods:
            # Rolling quantiles
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                results[f'q{int(q*100)}_{period}'] = df[price_col].rolling(window=period).quantile(q)
            
            # Interquartile range (IQR)
            q75 = df[price_col].rolling(window=period).quantile(0.75)
            q25 = df[price_col].rolling(window=period).quantile(0.25)
            results[f'iqr_{period}'] = q75 - q25
            
            # Price position within range
            current_price = df[price_col]
            rolling_min = df[price_col].rolling(window=period).min()
            rolling_max = df[price_col].rolling(window=period).max()
            results[f'price_position_{period}'] = (current_price - rolling_min) / (rolling_max - rolling_min + 1e-8)
            
            # Z-score (price relative to rolling mean/std)
            rolling_mean = df[price_col].rolling(window=period).mean()
            rolling_std = df[price_col].rolling(window=period).std()
            results[f'zscore_{period}'] = (current_price - rolling_mean) / (rolling_std + 1e-8)
        
        return results
    
    def calculate_autocorrelation(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate autocorrelation features"""
        results = pd.DataFrame(index=df.index)
        
        returns = df[price_col].pct_change()
        
        for lag in [1, 5, 10, 20]:
            results[f'autocorr_{lag}'] = returns.rolling(window=50).apply(
                lambda x: pd.Series(x).autocorr(lag=lag), raw=False
            )
        
        # Hurst exponent (roughness measure)
        results['hurst'] = returns.rolling(window=100).apply(
            self._calculate_hurst, raw=False
        )
        
        return results
    
    def _calculate_hurst(self, series):
        """Calculate Hurst exponent for a series"""
        if len(series) < 20:
            return np.nan
        
        try:
            # Calculate R/S statistic
            lags = range(2, len(series) // 2)
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return np.nan
    
    def calculate_all_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all statistical features"""
        logger.info("Calculating statistical features...")
        
        all_features = pd.DataFrame(index=df.index)
        
        # Ensure required columns exist
        if 'close' not in df.columns:
            logger.warning("'close' column not found for statistical features")
            if len(df.columns) > 0:
                df = df.rename(columns={df.columns[0]: 'close'})
            else:
                return all_features
        
        try:
            # Returns
            returns_features = self.calculate_returns(df)
            all_features = pd.concat([all_features, returns_features], axis=1)
            
            # Volatility
            volatility_features = self.calculate_volatility(df)
            all_features = pd.concat([all_features, volatility_features], axis=1)
            
            # Skewness and Kurtosis
            skewness_features = self.calculate_skewness_kurtosis(df)
            all_features = pd.concat([all_features, skewness_features], axis=1)
            
            # Quantile features
            quantile_features = self.calculate_quantile_features(df)
            all_features = pd.concat([all_features, quantile_features], axis=1)
            
            # Autocorrelation
            autocorr_features = self.calculate_autocorrelation(df)
            all_features = pd.concat([all_features, autocorr_features], axis=1)
            
            # Additional statistical features
            try:
                # Rolling statistics
                for period in [5, 10, 20, 50]:
                    all_features[f'mean_{period}'] = df['close'].rolling(window=period).mean()
                    all_features[f'std_{period}'] = df['close'].rolling(window=period).std()
                    all_features[f'min_{period}'] = df['close'].rolling(window=period).min()
                    all_features[f'max_{period}'] = df['close'].rolling(window=period).max()
                    all_features[f'median_{period}'] = df['close'].rolling(window=period).median()
                    all_features[f'mad_{period}'] = df['close'].rolling(window=period).apply(
                        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
                    )
                
                # Efficiency ratio
                for period in [10, 20, 50]:
                    price_change = abs(df['close'].diff(period))
                    volatility = df['close'].diff().abs().rolling(window=period).sum()
                    all_features[f'efficiency_ratio_{period}'] = price_change / (volatility + 1e-8)
                
                # Fractal dimension (approximate)
                for period in [20, 50, 100]:
                    all_features[f'fractal_dim_{period}'] = df['close'].rolling(window=period).apply(
                        lambda x: 1 + np.log(np.sum(np.abs(np.diff(x)))) / np.log(period - 1) if len(x) > 1 else np.nan,
                        raw=True
                    )
                
            except Exception as e:
                logger.warning(f"Could not calculate additional statistical features: {str(e)}")
            
            logger.info(f"Generated {len(all_features.columns)} statistical features")
            
        except Exception as e:
            logger.error(f"Error calculating statistical features: {str(e)}")
        
        return all_features

# ============ Time Features ============
class TimeFeatureEngineer:
    """Engineer time-based features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def calculate_time_features(self, df_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Calculate time-based features"""
        results = pd.DataFrame(index=df_index)
        
        if not isinstance(df_index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, cannot calculate time features")
            return results
        
        try:
            # Basic time features
            results['hour'] = df_index.hour
            results['day'] = df_index.day
            results['day_of_week'] = df_index.dayofweek
            results['day_of_year'] = df_index.dayofyear
            results['week'] = df_index.isocalendar().week
            results['month'] = df_index.month
            results['quarter'] = df_index.quarter
            results['year'] = df_index.year
            
            # Cyclical encoding for periodic features
            results['hour_sin'] = np.sin(2 * np.pi * results['hour'] / 24)
            results['hour_cos'] = np.cos(2 * np.pi * results['hour'] / 24)
            
            results['day_of_week_sin'] = np.sin(2 * np.pi * results['day_of_week'] / 7)
            results['day_of_week_cos'] = np.cos(2 * np.pi * results['day_of_week'] / 7)
            
            results['month_sin'] = np.sin(2 * np.pi * results['month'] / 12)
            results['month_cos'] = np.cos(2 * np.pi * results['month'] / 12)
            
            # Time of day categories
            results['is_morning'] = results['hour'].between(6, 12).astype(int)
            results['is_afternoon'] = results['hour'].between(12, 18).astype(int)
            results['is_evening'] = results['hour'].between(18, 24).astype(int)
            results['is_night'] = results['hour'].between(0, 6).astype(int)
            
            # Day categories
            results['is_weekend'] = results['day_of_week'].isin([5, 6]).astype(int)
            results['is_monday'] = (results['day_of_week'] == 0).astype(int)
            results['is_friday'] = (results['day_of_week'] == 4).astype(int)
            
            # Month categories
            results['is_q1'] = results['quarter'].eq(1).astype(int)
            results['is_q4'] = results['quarter'].eq(4).astype(int)
            results['is_year_end'] = results['month'].isin([11, 12]).astype(int)
            
            # Time since market events (approximate for crypto)
            # Assuming 24/7 market with potential lower liquidity on weekends
            results['hours_since_week_start'] = (results['day_of_week'] * 24 + results['hour'])
            results['hours_since_month_start'] = ((results['day'] - 1) * 24 + results['hour'])
            
            # Seasonality features (sine/cosine with yearly period)
            day_of_year = df_index.dayofyear
            results['seasonality_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
            results['seasonality_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
            
            # Business day features (crypto is 24/7 but has patterns)
            results['is_business_hour'] = results['hour'].between(9, 17).astype(int)  # 9am-5pm UTC
            
            # Time elapsed features
            if len(df_index) > 1:
                time_diff = df_index.to_series().diff().dt.total_seconds()
                results['time_since_last'] = time_diff.fillna(0)
                results['is_regular_interval'] = (time_diff == time_diff.mode().iloc[0]).astype(int) \
                    if not time_diff.mode().empty else 0
            
            logger.info(f"Generated {len(results.columns)} time features")
            
        except Exception as e:
            logger.error(f"Error calculating time features: {str(e)}")
        
        return results

# ============ Feature Transformations ============
class FeatureTransformer:
    """Transform features for better model performance"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scalers = {}
        self.transforms = {}
        self.fitted = False
    
    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features"""
        self.fitted = True
        return self.transform(features, fit=True)
    
    def transform(self, features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Transform features"""
        transformed = features.copy()
        
        # Handle missing values
        transformed = self._handle_missing_values(transformed)
        
        # Apply transformations
        if self.config.use_log_transform:
            transformed = self._apply_log_transform(transformed)
        
        if self.config.use_boxcox:
            transformed = self._apply_boxcox_transform(transformed, fit)
        
        # Apply scaling/normalization
        if self.config.scale_features:
            transformed = self._apply_scaling(transformed, fit)
        
        # Create interaction features
        if self.config.interaction_degree > 1:
            transformed = self._create_interaction_features(transformed)
        
        # Create polynomial features
        if self.config.polynomial_degree > 1:
            transformed = self._create_polynomial_features(transformed)
        
        return transformed
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # Forward fill then backward fill
        features_ffill = features.ffill()
        features_filled = features_ffill.bfill()
        
        # If still missing, fill with column mean
        for col in features_filled.columns:
            if features_filled[col].isnull().any():
                features_filled[col] = features_filled[col].fillna(features_filled[col].mean())
        
        return features_filled
    
    def _apply_log_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply log transform to appropriate features"""
        transformed = features.copy()
        
        # Identify columns suitable for log transform
        log_cols = []
        for col in transformed.columns:
            # Skip if column contains negative values or zeros
            if transformed[col].min() > 0:
                # Check if distribution is right-skewed
                skewness = transformed[col].skew()
                if skewness > 0.5:  # Right-skewed
                    log_cols.append(col)
        
        # Apply log transform
        for col in log_cols:
            transformed[f'log_{col}'] = np.log(transformed[col] + 1e-8)
        
        return transformed
    
    def _apply_boxcox_transform(self, features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Apply Box-Cox transform to appropriate features"""
        transformed = features.copy()
        
        # Identify columns suitable for Box-Cox
        boxcox_cols = []
        for col in transformed.columns:
            # Box-Cox requires positive values
            if transformed[col].min() > 0:
                boxcox_cols.append(col)
        
        # Apply Box-Cox
        for col in boxcox_cols:
            try:
                if fit:
                    # Fit and transform
                    transformed_col, lambda_val = stats.boxcox(transformed[col] + 1e-8)
                    self.transforms[f'boxcox_{col}'] = lambda_val
                    transformed[f'boxcox_{col}'] = transformed_col
                else:
                    # Transform using fitted lambda
                    if f'boxcox_{col}' in self.transforms:
                        lambda_val = self.transforms[f'boxcox_{col}']
                        transformed[f'boxcox_{col}'] = stats.boxcox(
                            transformed[col] + 1e-8, lmbda=lambda_val
                        )
            except Exception as e:
                logger.warning(f"Could not apply Box-Cox to {col}: {str(e)}")
        
        return transformed
    
    def _apply_scaling(self, features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Apply scaling/normalization to features"""
        scaled = features.copy()
        
        # Separate numeric and non-numeric columns
        numeric_cols = scaled.select_dtypes(include=[np.number]).columns
        non_numeric_cols = scaled.select_dtypes(exclude=[np.number]).columns
        
        if self.config.separate_scalers:
            # Scale each column separately
            for col in numeric_cols:
                if fit:
                    if self.config.normalization_method == 'standard':
                        scaler = StandardScaler()
                    elif self.config.normalization_method == 'minmax':
                        scaler = MinMaxScaler()
                    elif self.config.normalization_method == 'robust':
                        scaler = RobustScaler()
                    else:
                        scaler = StandardScaler()
                    
                    scaled_values = scaler.fit_transform(scaled[[col]])
                    self.scalers[col] = scaler
                else:
                    if col in self.scalers:
                        scaler = self.scalers[col]
                        scaled_values = scaler.transform(scaled[[col]])
                    else:
                        scaled_values = scaled[[col]].values
                
                scaled[col] = scaled_values.flatten()
        else:
            # Scale all numeric columns together
            if len(numeric_cols) > 0:
                if fit:
                    if self.config.normalization_method == 'standard':
                        scaler = StandardScaler()
                    elif self.config.normalization_method == 'minmax':
                        scaler = MinMaxScaler()
                    elif self.config.normalization_method == 'robust':
                        scaler = RobustScaler()
                    else:
                        scaler = StandardScaler()
                    
                    scaled_values = scaler.fit_transform(scaled[numeric_cols])
                    self.scalers['_all_numeric'] = scaler
                else:
                    if '_all_numeric' in self.scalers:
                        scaler = self.scalers['_all_numeric']
                        scaled_values = scaler.transform(scaled[numeric_cols])
                    else:
                        scaled_values = scaled[numeric_cols].values
                
                scaled[numeric_cols] = scaled_values
        
        return scaled
    
    def _create_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        interaction_features = pd.DataFrame(index=features.index)
        
        # Get numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return features
        
        # Create pairwise interactions up to specified degree
        degree = min(self.config.interaction_degree, len(numeric_cols))
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j and i < degree and j < degree:
                    # Multiplication interaction
                    interaction_features[f'{col1}_x_{col2}'] = features[col1] * features[col2]
                    
                    # Ratio interaction (avoid division by zero)
                    if (features[col2] != 0).all():
                        interaction_features[f'{col1}_div_{col2}'] = features[col1] / (features[col2] + 1e-8)
                    
                    # Difference interaction
                    interaction_features[f'{col1}_minus_{col2}'] = features[col1] - features[col2]
                    
                    # Sum interaction
                    interaction_features[f'{col1}_plus_{col2}'] = features[col1] + features[col2]
        
        # Combine with original features
        result = pd.concat([features, interaction_features], axis=1)
        
        return result
    
    def _create_polynomial_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features"""
        polynomial_features = pd.DataFrame(index=features.index)
        
        # Get numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return features
        
        degree = min(self.config.polynomial_degree, 3)  # Limit to reasonable degree
        
        # Create polynomial features
        for col in numeric_cols:
            for d in range(2, degree + 1):
                polynomial_features[f'{col}^{d}'] = features[col] ** d
        
        # Combine with original features
        result = pd.concat([features, polynomial_features], axis=1)
        
        return result

# ============ Feature Selection ============
class FeatureSelector:
    """Select most important features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.selected_features = []
        self.feature_importance = {}
        self.correlation_matrix = None
        self.selector = None
    
    def select_features(self, features: pd.DataFrame, 
                       target: Optional[pd.Series] = None,
                       method: Optional[str] = None) -> pd.DataFrame:
        """Select features using specified method"""
        if method is None:
            method = self.config.feature_selection_method
        
        # Ensure we only use numeric features
        numeric_features = features.select_dtypes(include=[np.number])
        
        if len(numeric_features.columns) == 0:
            logger.warning("No numeric features to select from")
            return features
        
        # Remove features with too many missing values
        numeric_features = self._remove_high_missing(numeric_features)
        
        # Remove low variance features
        numeric_features = self._remove_low_variance(numeric_features)
        
        # Remove highly correlated features
        numeric_features = self._remove_high_correlation(numeric_features)
        
        # Apply feature selection method
        if method == 'mutual_info' and target is not None:
            selected = self._select_by_mutual_info(numeric_features, target)
        elif method == 'correlation' and target is not None:
            selected = self._select_by_correlation(numeric_features, target)
        elif method == 'variance':
            selected = self._select_by_variance(numeric_features)
        else:
            # Default: select top features by variance
            selected = self._select_by_variance(numeric_features)
        
        # Limit to max_features
        if len(selected.columns) > self.config.max_features:
            # Sort by importance if available
            if self.feature_importance:
                sorted_features = sorted(
                    self.feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:self.config.max_features]
                selected_cols = [col for col, _ in sorted_features]
                selected = selected[selected_cols]
            else:
                # Otherwise select first max_features
                selected = selected.iloc[:, :self.config.max_features]
        
        self.selected_features = selected.columns.tolist()
        logger.info(f"Selected {len(self.selected_features)} features")
        
        return selected
    
    def _remove_high_missing(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove features with too many missing values"""
        missing_ratio = features.isnull().sum() / len(features)
        valid_features = features.columns[missing_ratio < 0.5]  # Keep if < 50% missing
        return features[valid_features]
    
    def _remove_low_variance(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove features with low variance"""
        variances = features.var()
        threshold = self.config.variance_threshold * features.var().max()
        valid_features = features.columns[variances > threshold]
        return features[valid_features]
    
    def _remove_high_correlation(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        if len(features.columns) < 2:
            return features
        
        # Calculate correlation matrix
        self.correlation_matrix = features.corr().abs()
        
        # Select features to keep
        features_to_keep = []
        features_to_remove = []
        
        for i, col_i in enumerate(features.columns):
            if col_i in features_to_remove:
                continue
            
            features_to_keep.append(col_i)
            
            for j, col_j in enumerate(features.columns[i+1:], i+1):
                if col_j in features_to_remove:
                    continue
                
                # Check correlation
                if self.correlation_matrix.iloc[i, j] > self.config.correlation_threshold:
                    # Keep the feature with higher variance
                    var_i = features[col_i].var()
                    var_j = features[col_j].var()
                    
                    if var_i >= var_j:
                        features_to_remove.append(col_j)
                    else:
                        features_to_remove.append(col_i)
                        features_to_keep.remove(col_i)
                        break
        
        result = features[features_to_keep]
        logger.info(f"Removed {len(features_to_remove)} highly correlated features")
        
        return result
    
    def _select_by_mutual_info(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select features using mutual information"""
        try:
            # Calculate mutual information
            mi_scores = mutual_info_regression(features, target)
            
            # Store importance scores
            self.feature_importance = dict(zip(features.columns, mi_scores))
            
            # Select features
            k = min(self.config.max_features, len(features.columns))
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
            selector.fit(features, target)
            
            selected_cols = features.columns[selector.get_support()]
            return features[selected_cols]
            
        except Exception as e:
            logger.error(f"Error in mutual info selection: {str(e)}")
            return features
    
    def _select_by_correlation(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select features using correlation with target"""
        try:
            # Calculate correlation with target
            correlations = features.apply(lambda x: x.corr(target))
            
            # Store importance scores
            self.feature_importance = correlations.abs().to_dict()
            
            # Select features with highest absolute correlation
            k = min(self.config.max_features, len(features.columns))
            selected_cols = correlations.abs().nlargest(k).index
            
            return features[selected_cols]
            
        except Exception as e:
            logger.error(f"Error in correlation selection: {str(e)}")
            return features
    
    def _select_by_variance(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select features by variance"""
        try:
            # Calculate variance
            variances = features.var()
            
            # Store importance scores
            self.feature_importance = variances.to_dict()
            
            # Select features with highest variance
            k = min(self.config.max_features, len(features.columns))
            selected_cols = variances.nlargest(k).index
            
            return features[selected_cols]
            
        except Exception as e:
            logger.error(f"Error in variance selection: {str(e)}")
            return features

# ============ Main Feature Engineer ============
class BitcoinFeatureEngineer(BaseFeatureEngineer):
    """Main feature engineer for Bitcoin trading"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config)
        
        # Initialize sub-engineers
        self.technical_engineer = TechnicalIndicatorEngineer(self.config)
        self.statistical_engineer = StatisticalFeatureEngineer(self.config)
        self.time_engineer = TimeFeatureEngineer(self.config)
        self.transformer = FeatureTransformer(self.config)
        self.selector = FeatureSelector(self.config)
        
        # Feature tracking
        self.feature_groups = {
            FeatureType.TECHNICAL: [],
            FeatureType.STATISTICAL: [],
            FeatureType.TIME: [],
            FeatureType.TRANSFORMED: []
        }
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all features from raw data"""
        logger.info("Starting feature engineering...")
        
        # Ensure we have a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
                data.index = pd.to_datetime(data.index)
            else:
                data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='H')
        
        all_features = pd.DataFrame(index=data.index)
        
        try:
            # Technical indicators
            technical_features = self.technical_engineer.calculate_all_technical_indicators(data)
            if not technical_features.empty:
                all_features = pd.concat([all_features, technical_features], axis=1)
                self.feature_groups[FeatureType.TECHNICAL] = technical_features.columns.tolist()
            
            # Statistical features
            statistical_features = self.statistical_engineer.calculate_all_statistical_features(data)
            if not statistical_features.empty:
                all_features = pd.concat([all_features, statistical_features], axis=1)
                self.feature_groups[FeatureType.STATISTICAL] = statistical_features.columns.tolist()
            
            # Time features
            time_features = self.time_engineer.calculate_time_features(data.index)
            if not time_features.empty:
                all_features = pd.concat([all_features, time_features], axis=1)
                self.feature_groups[FeatureType.TIME] = time_features.columns.tolist()
            
            logger.info(f"Created {len(all_features.columns)} raw features")
            
            # Handle missing values in raw features
            all_features = all_features.ffill().bfill()
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            raise
    
    def transform_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Transform features (normalization, scaling, etc.)"""
        logger.info("Transforming features...")
        
        try:
            if fit:
                transformed = self.transformer.fit_transform(features)
            else:
                transformed = self.transformer.transform(features, fit=False)
            
            self.feature_groups[FeatureType.TRANSFORMED] = [
                col for col in transformed.columns 
                if col not in sum(self.feature_groups.values(), [])
            ]
            
            logger.info(f"Transformed to {len(transformed.columns)} features")
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}")
            raise
    
    def select_features(self, features: pd.DataFrame, 
                       target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Select most important features"""
        logger.info("Selecting features...")
        
        try:
            selected = self.selector.select_features(features, target)
            
            # Update feature importance
            self.feature_importance = self.selector.feature_importance
            
            logger.info(f"Selected {len(selected.columns)} features")
            
            return selected
            
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            raise
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all features"""
        descriptions = {
            # Technical indicators
            'sma_*': 'Simple Moving Average over * periods',
            'ema_*': 'Exponential Moving Average over * periods',
            'rsi_*': 'Relative Strength Index over * periods',
            'macd_line': 'MACD line (fast EMA - slow EMA)',
            'macd_signal': 'MACD signal line',
            'macd_histogram': 'MACD histogram',
            'bb_upper_*': 'Bollinger Band upper band (* periods)',
            'bb_lower_*': 'Bollinger Band lower band (* periods)',
            'atr_*': 'Average True Range over * periods',
            'stoch_k_*': 'Stochastic %K over * periods',
            'stoch_d_*': 'Stochastic %D over * periods',
            'adx_*': 'Average Directional Index over * periods',
            'cci_*': 'Commodity Channel Index over * periods',
            'williams_r_*': 'Williams %R over * periods',
            'obv': 'On-Balance Volume',
            'mfi_*': 'Money Flow Index over * periods',
            
            # Statistical features
            'return_*': 'Price return over * periods',
            'log_return_*': 'Logarithmic return over * periods',
            'volatility_*': 'Volatility (std of returns) over * periods',
            'skewness_*': 'Skewness of returns over * periods',
            'kurtosis_*': 'Kurtosis of returns over * periods',
            'zscore_*': 'Z-score (price relative to rolling mean/std)',
            'hurst': 'Hurst exponent (market memory)',
            
            # Time features
            'hour': 'Hour of day (0-23)',
            'day_of_week': 'Day of week (0=Monday)',
            'month': 'Month (1-12)',
            'quarter': 'Quarter (1-4)',
            'is_weekend': 'Is weekend (1=yes, 0=no)',
            'hour_sin': 'Sine encoding of hour',
            'hour_cos': 'Cosine encoding of hour',
            'seasonality_sin': 'Sine encoding of yearly seasonality',
            'seasonality_cos': 'Cosine encoding of yearly seasonality',
            
            # Transformed features
            'log_*': 'Logarithmic transform of *',
            'boxcox_*': 'Box-Cox transform of *',
            '*_x_*': 'Interaction (multiplication) between features',
            '*_div_*': 'Interaction (division) between features',
            '*^2': 'Polynomial feature (squared)',
            '*^3': 'Polynomial feature (cubed)'
        }
        
        return descriptions
    
    def create_label_features(self, prices: pd.Series, horizon: int = 24) -> pd.DataFrame:
        """Create label features for supervised learning"""
        logger.info(f"Creating label features with horizon {horizon}")
        
        labels = pd.DataFrame(index=prices.index)
        
        try:
            # Future returns
            labels[f'future_return_{horizon}'] = prices.pct_change(horizon).shift(-horizon)
            
            # Future log returns
            labels[f'future_log_return_{horizon}'] = np.log(prices.shift(-horizon) / prices)
            
            # Direction (binary classification)
            labels[f'direction_{horizon}'] = (labels[f'future_return_{horizon}'] > 0).astype(int)
            
            # Volatility prediction
            future_volatility = prices.pct_change().rolling(horizon).std().shift(-horizon)
            labels[f'future_volatility_{horizon}'] = future_volatility
            
            # Price levels
            labels[f'future_price_{horizon}'] = prices.shift(-horizon)
            
            # Remove NaN values at the end
            labels = labels.dropna()
            
            logger.info(f"Created {len(labels.columns)} label features")
            
            return labels
            
        except Exception as e:
            logger.error(f"Error creating label features: {str(e)}")
            raise
    
    def create_sequence_features(self, features: pd.DataFrame, 
                               sequence_length: int = 60) -> np.ndarray:
        """Create sequences for time series models"""
        logger.info(f"Creating sequences of length {sequence_length}")
        
        try:
            # Ensure features are numeric
            numeric_features = features.select_dtypes(include=[np.number])
            
            # Create sequences
            sequences = []
            for i in range(len(numeric_features) - sequence_length):
                sequence = numeric_features.iloc[i:i + sequence_length].values
                sequences.append(sequence)
            
            sequences_array = np.array(sequences)
            
            logger.info(f"Created {len(sequences_array)} sequences")
            
            return sequences_array
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise
    
    def save_features(self, features: pd.DataFrame, path: Path):
        """Save engineered features to disk"""
        try:
            # Save as parquet for efficiency
            parquet_path = path.with_suffix('.parquet')
            features.to_parquet(parquet_path)
            
            # Also save as CSV for inspection
            csv_path = path.with_suffix('.csv')
            features.head(1000).to_csv(csv_path)  # Save first 1000 rows
            
            logger.info(f"Saved features to {parquet_path}")
            
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise
    
    def load_features(self, path: Path) -> pd.DataFrame:
        """Load engineered features from disk"""
        try:
            # Try parquet first
            parquet_path = path.with_suffix('.parquet')
            if parquet_path.exists():
                features = pd.read_parquet(parquet_path)
            else:
                # Fall back to CSV
                csv_path = path.with_suffix('.csv')
                features = pd.read_csv(csv_path, index_col=0)
            
            logger.info(f"Loaded {len(features)} features with {len(features.columns)} columns")
            
            return features
            
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise

# ============ Feature Pipeline ============
class FeaturePipeline:
    """Complete feature engineering pipeline"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.engineer = BitcoinFeatureEngineer(self.config)
        self.is_fitted = False
    
    def fit_transform(self, data: pd.DataFrame, 
                     target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform data through entire pipeline"""
        logger.info("Running feature pipeline...")
        
        try:
            # Step 1: Create raw features
            raw_features = self.engineer.create_features(data)
            
            # Step 2: Transform features
            transformed_features = self.engineer.transform_features(raw_features, fit=True)
            
            # Step 3: Select features (if target provided)
            if target is not None:
                selected_features = self.engineer.select_features(transformed_features, target)
            else:
                selected_features = transformed_features
            
            self.is_fitted = True
            
            logger.info(f"Pipeline completed: {len(selected_features.columns)} final features")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in feature pipeline: {str(e)}")
            raise
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        try:
            # Create raw features
            raw_features = self.engineer.create_features(data)
            
            # Transform features (without fitting)
            transformed_features = self.engineer.transform_features(raw_features, fit=False)
            
            # Select features (using previously selected features)
            selected_features = transformed_features[self.engineer.selected_features]
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def get_feature_report(self) -> Dict[str, Any]:
        """Get report on engineered features"""
        report = {
            'total_features': len(self.engineer.selected_features),
            'feature_groups': {
                group_type.value: len(features)
                for group_type, features in self.engineer.feature_groups.items()
                if features
            },
            'feature_importance': dict(
                sorted(self.engineer.feature_importance.items(), 
                      key=lambda x: abs(x[1]), reverse=True)[:20]
            ) if self.engineer.feature_importance else {},
            'config': {
                'moving_averages': self.config.moving_averages,
                'returns_periods': self.config.returns_periods,
                'max_features': self.config.max_features,
                'normalization_method': self.config.normalization_method
            }
        }
        
        return report

# ============ Example Usage ============
def example_usage():
    """Example usage of feature engineering"""
    
    print("Feature Engineering Example")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    n_samples = len(dates)
    
    # Generate synthetic price data
    np.random.seed(42)
    returns = np.random.randn(n_samples) * 0.01
    price = 10000 * np.exp(np.cumsum(returns))
    
    # Create sample DataFrame
    data = pd.DataFrame({
        'open': price * (1 + np.random.randn(n_samples) * 0.001),
        'high': price * (1 + np.abs(np.random.randn(n_samples)) * 0.002),
        'low': price * (1 - np.abs(np.random.randn(n_samples)) * 0.002),
        'close': price,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    print(f"Created sample data with {len(data)} rows")
    print(f"Columns: {list(data.columns)}")
    
    # Create feature pipeline
    pipeline = FeaturePipeline()
    
    # Create target for supervised feature selection
    target = data['close'].pct_change().shift(-1)  # Next period return
    
    # Run feature pipeline
    print("\n1. Running feature pipeline...")
    features = pipeline.fit_transform(data, target)
    
    print(f"Generated {len(features.columns)} features")
    print(f"Feature shape: {features.shape}")
    
        # Get feature report
    print("\n2. Generating feature report...")
    report = pipeline.get_feature_report()
    
    print(f"Total features selected: {report['total_features']}")
    print("\nFeature groups:")
    for group, count in report['feature_groups'].items():
        print(f"  {group}: {count} features")
    
    print("\nTop 10 important features:")
    for i, (feature, importance) in enumerate(list(report['feature_importance'].items())[:10], 1):
        print(f"  {i:2}. {feature}: {importance:.6f}")
    
    # Create label features for supervised learning
    print("\n3. Creating label features...")
    labels = pipeline.engineer.create_label_features(data['close'], horizon=24)
    print(f"Created {len(labels.columns)} label features")
    
    # Create sequences for time series models
    print("\n4. Creating sequences for time series models...")
    sequences = pipeline.engineer.create_sequence_features(features, sequence_length=60)
    print(f"Created {len(sequences)} sequences of shape {sequences[0].shape}")
    
    # Test transform on new data
    print("\n5. Testing transform on new data...")
    new_data = data.iloc[-100:].copy()  # Last 100 rows
    new_features = pipeline.transform(new_data)
    print(f"Transformed new data: {new_features.shape}")
    
    return pipeline, features, labels, sequences


# ============ Factory Functions ============
def create_feature_engineer(config: Optional[Dict] = None) -> BitcoinFeatureEngineer:
    """Factory function to create a feature engineer"""
    if config:
        feature_config = FeatureConfig(**config)
    else:
        feature_config = FeatureConfig()
    
    return BitcoinFeatureEngineer(feature_config)


def create_feature_pipeline(config: Optional[Dict] = None) -> FeaturePipeline:
    """Factory function to create a feature pipeline"""
    if config:
        feature_config = FeatureConfig(**config)
    else:
        feature_config = FeatureConfig()
    
    return FeaturePipeline(feature_config)


def load_feature_config(config_path: Path) -> FeatureConfig:
    """Load feature configuration from YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return FeatureConfig(**config_dict.get('feature_engineering', {}))
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {str(e)}")
        return FeatureConfig()


# ============ Feature Registry ============
class FeatureRegistry:
    """Registry for managing and tracking features"""
    
    def __init__(self):
        self.features = {}
        self.feature_metadata = {}
        self.feature_categories = {}
        self.last_updated = None
    
    def register_feature(self, name: str, description: str, 
                        category: FeatureType, source: str,
                        formula: Optional[str] = None):
        """Register a feature with metadata"""
        self.features[name] = {
            'description': description,
            'category': category.value,
            'source': source,
            'formula': formula,
            'created_at': pd.Timestamp.now(),
            'usage_count': 0
        }
        
        # Track by category
        if category not in self.feature_categories:
            self.feature_categories[category] = []
        self.feature_categories[category].append(name)
        
        self.last_updated = pd.Timestamp.now()
    
    def increment_usage(self, feature_name: str):
        """Increment usage count for a feature"""
        if feature_name in self.features:
            self.features[feature_name]['usage_count'] += 1
    
    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """Get information about a specific feature"""
        return self.features.get(feature_name, {})
    
    def get_features_by_category(self, category: FeatureType) -> List[str]:
        """Get all features in a category"""
        return self.feature_categories.get(category, [])
    
    def get_most_used_features(self, n: int = 20) -> List[Tuple[str, int]]:
        """Get most frequently used features"""
        features_with_usage = [
            (name, info['usage_count'])
            for name, info in self.features.items()
        ]
        return sorted(features_with_usage, key=lambda x: x[1], reverse=True)[:n]
    
    def save_registry(self, path: Path):
        """Save feature registry to disk"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        logger.info(f"Saved feature registry to {path}")
    
    def load_registry(self, path: Path):
        """Load feature registry from disk"""
        import pickle
        with open(path, 'rb') as f:
            self.__dict__ = pickle.load(f)
        logger.info(f"Loaded feature registry from {path}")


# ============ On-chain Feature Engineering ============
class OnChainFeatureEngineer:
    """Engineer on-chain Bitcoin features"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.registry = FeatureRegistry()
    
    def calculate_onchain_features(self, onchain_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate on-chain features"""
        features = pd.DataFrame(index=onchain_data.index)
        
        # Ensure we have required columns
        required_cols = [
            'hash_rate', 'difficulty', 'transaction_count',
            'active_addresses', 'miners_revenue', 'mempool_size'
        ]
        
        available_cols = [col for col in required_cols if col in onchain_data.columns]
        
        if not available_cols:
            logger.warning("No on-chain data available")
            return features
        
        try:
            # Hash rate features
            if 'hash_rate' in available_cols:
                features['hash_rate'] = onchain_data['hash_rate']
                features['hash_rate_ma_7'] = onchain_data['hash_rate'].rolling(7).mean()
                features['hash_rate_ma_30'] = onchain_data['hash_rate'].rolling(30).mean()
                features['hash_rate_growth'] = onchain_data['hash_rate'].pct_change()
                
                self.registry.register_feature(
                    name='hash_rate',
                    description='Bitcoin network hash rate',
                    category=FeatureType.ONCHAIN,
                    source='hash_rate'
                )
            
            # Difficulty features
            if 'difficulty' in available_cols:
                features['difficulty'] = onchain_data['difficulty']
                features['difficulty_change'] = onchain_data['difficulty'].pct_change()
                features['hash_rate_difficulty_ratio'] = (
                    onchain_data['hash_rate'] / onchain_data['difficulty']
                    if 'hash_rate' in available_cols else np.nan
                )
            
            # Transaction features
            if 'transaction_count' in available_cols:
                features['transaction_count'] = onchain_data['transaction_count']
                features['transaction_count_ma_7'] = onchain_data['transaction_count'].rolling(7).mean()
                features['transaction_growth'] = onchain_data['transaction_count'].pct_change()
                features['transactions_per_second'] = onchain_data['transaction_count'] / 86400
            
            # Address features
            if 'active_addresses' in available_cols:
                features['active_addresses'] = onchain_data['active_addresses']
                features['active_addresses_ma_7'] = onchain_data['active_addresses'].rolling(7).mean()
                features['active_addresses_growth'] = onchain_data['active_addresses'].pct_change()
            
            # Miner revenue features
            if 'miners_revenue' in available_cols:
                features['miners_revenue'] = onchain_data['miners_revenue']
                features['miners_revenue_ma_7'] = onchain_data['miners_revenue'].rolling(7).mean()
                features['miners_revenue_growth'] = onchain_data['miners_revenue'].pct_change()
            
            # Mempool features
            if 'mempool_size' in available_cols:
                features['mempool_size'] = onchain_data['mempool_size']
                features['mempool_size_ma_7'] = onchain_data['mempool_size'].rolling(7).mean()
                features['mempool_growth'] = onchain_data['mempool_size'].pct_change()
            
            # Network value features
            if all(col in available_cols for col in ['transaction_count', 'active_addresses']):
                features['nvts_ratio'] = (
                    features['transaction_count'] / features['active_addresses']
                )
            
            # Network momentum
            features['network_momentum'] = (
                features.get('transaction_growth', 0) +
                features.get('active_addresses_growth', 0) +
                features.get('hash_rate_growth', 0)
            ) / 3
            
            logger.info(f"Generated {len(features.columns)} on-chain features")
            
        except Exception as e:
            logger.error(f"Error calculating on-chain features: {str(e)}")
        
        return features


# ============ Sentiment Feature Engineering ============
class SentimentFeatureEngineer:
    """Engineer sentiment features from news and social media"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.registry = FeatureRegistry()
    
    def calculate_sentiment_features(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate sentiment features"""
        features = pd.DataFrame(index=sentiment_data.index)
        
        # Ensure we have required columns
        required_cols = ['sentiment_score', 'volume', 'subjectivity']
        available_cols = [col for col in required_cols if col in sentiment_data.columns]
        
        if not available_cols:
            logger.warning("No sentiment data available")
            return features
        
        try:
            # Basic sentiment features
            if 'sentiment_score' in available_cols:
                features['sentiment_score'] = sentiment_data['sentiment_score']
                features['sentiment_ma_7'] = sentiment_data['sentiment_score'].rolling(7).mean()
                features['sentiment_ma_30'] = sentiment_data['sentiment_score'].rolling(30).mean()
                features['sentiment_momentum'] = sentiment_data['sentiment_score'].diff()
                
                # Sentiment extremes
                features['sentiment_extreme_positive'] = (
                    sentiment_data['sentiment_score'] > 0.7
                ).astype(int)
                features['sentiment_extreme_negative'] = (
                    sentiment_data['sentiment_score'] < -0.7
                ).astype(int)
            
            # Volume features
            if 'volume' in available_cols:
                features['sentiment_volume'] = sentiment_data['volume']
                features['sentiment_volume_ma_7'] = sentiment_data['volume'].rolling(7).mean()
                features['sentiment_volume_ratio'] = (
                    sentiment_data['volume'] / sentiment_data['volume'].rolling(30).mean()
                )
            
            # Subjectivity features
            if 'subjectivity' in available_cols:
                features['subjectivity'] = sentiment_data['subjectivity']
                features['subjectivity_ma_7'] = sentiment_data['subjectivity'].rolling(7).mean()
            
            # Sentiment volatility
            if 'sentiment_score' in available_cols:
                features['sentiment_volatility'] = sentiment_data['sentiment_score'].rolling(7).std()
            
            # Sentiment divergence
            if 'sentiment_score' in available_cols and 'sentiment_ma_7' in features.columns:
                features['sentiment_divergence'] = (
                    sentiment_data['sentiment_score'] - features['sentiment_ma_7']
                )
            
            # Sentiment trends
            if 'sentiment_score' in available_cols:
                # Linear regression slope over last 7 days
                features['sentiment_trend'] = sentiment_data['sentiment_score'].rolling(7).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan
                )
            
            logger.info(f"Generated {len(features.columns)} sentiment features")
            
        except Exception as e:
            logger.error(f"Error calculating sentiment features: {str(e)}")
        
        return features


# ============ Multi-timeframe Feature Engineering ============
class MultiTimeframeFeatureEngineer:
    """Engineer features across multiple timeframes"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.resample_methods = {
            '1H': '1H',
            '4H': '4H',
            '1D': '1D',
            '1W': '1W'
        }
    
    def create_multi_timeframe_features(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create features for multiple timeframes"""
        timeframe_features = {}
        
        try:
            for tf_name, tf_period in self.resample_methods.items():
                logger.info(f"Resampling to {tf_name} timeframe...")
                
                # Resample OHLCV data
                resampled = data.resample(tf_period).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                if len(resampled) < 100:  # Need sufficient data
                    logger.warning(f"Insufficient data for {tf_name} timeframe: {len(resampled)} rows")
                    continue
                
                # Create features for this timeframe
                engineer = BitcoinFeatureEngineer()
                timeframe_features[tf_name] = engineer.create_features(resampled)
                
                logger.info(f"Created {len(timeframe_features[tf_name].columns)} features for {tf_name}")
            
            # Align all timeframes to highest frequency
            if timeframe_features:
                # Get highest frequency (smallest timeframe)
                min_tf = min(timeframe_features.keys(), key=lambda x: pd.Timedelta(self.resample_methods[x]))
                
                # Align all features to this timeframe
                aligned_features = {}
                for tf_name, features in timeframe_features.items():
                    if tf_name == min_tf:
                        aligned_features[tf_name] = features
                    else:
                        # Forward fill higher timeframe features
                        aligned_features[tf_name] = features.reindex(
                            timeframe_features[min_tf].index, method='ffill'
                        )
                
                return aligned_features
            
        except Exception as e:
            logger.error(f"Error creating multi-timeframe features: {str(e)}")
        
        return timeframe_features


# ============ Main Execution ============
def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bitcoin Trading AI - Feature Engineering')
    parser.add_argument('--input', type=str, default='data/raw/bitcoin_data.csv',
                       help='Input data file path')
    parser.add_argument('--output', type=str, default='data/processed/features.parquet',
                       help='Output features file path')
    parser.add_argument('--config', type=str, default='config/feature_config.yaml',
                       help='Feature configuration file')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with synthetic data')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running in test mode with synthetic data...")
        pipeline, features, labels, sequences = example_usage()
        return
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if config_path.exists():
            feature_config = load_feature_config(config_path)
        else:
            feature_config = FeatureConfig()
            logger.info(f"Using default configuration, config file not found: {config_path}")
        
        # Load data
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Loading data from {input_path}")
        
        # Try different file formats
        if input_path.suffix == '.parquet':
            data = pd.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            data = pd.read_csv(input_path, index_col=0, parse_dates=True)
        elif input_path.suffix == '.json':
            data = pd.read_json(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        print(f"Loaded data with shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        
        # Create feature pipeline
        pipeline = FeaturePipeline(feature_config)
        
        # Create target for feature selection
        if 'close' in data.columns:
            target = data['close'].pct_change().shift(-1)  # Next period return
        else:
            target = None
            logger.warning("No 'close' column found for target creation")
        
        # Run pipeline
        print("\nRunning feature engineering pipeline...")
        features = pipeline.fit_transform(data, target)
        
        print(f"Generated {len(features.columns)} features")
        print(f"Final feature shape: {features.shape}")
        
        # Save features
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.parquet':
            features.to_parquet(output_path)
        else:
            features.to_csv(output_path)
        
        print(f"\nFeatures saved to: {output_path}")
        
        # Generate report
        report = pipeline.get_feature_report()
        report_path = output_path.parent / 'feature_report.json'
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Feature report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*50)
        print(f"Input data shape: {data.shape}")
        print(f"Output features shape: {features.shape}")
        print(f"Total features generated: {len(features.columns)}")
        print("\nFeature categories:")
        for category, count in report['feature_groups'].items():
            print(f"  {category}: {count} features")
        
        if report['feature_importance']:
            print("\nTop 5 most important features:")
            for feature, importance in list(report['feature_importance'].items())[:5]:
                print(f"  {feature}: {importance:.6f}")
        
        print("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run main function
    main()