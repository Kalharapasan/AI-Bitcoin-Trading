"""
Model Predictor module for Bitcoin trading AI.
Handles model inference, prediction post-processing, confidence scoring,
and trading signal generation based on model predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
import json
import pickle
import joblib
from pathlib import Path
import hashlib
import asyncio
from collections import deque, defaultdict
import uuid
import time
import traceback

# Import project modules
from config.settings import PredictionSettings, TradingSettings
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.utils.cache import Cache
from core.models.model_manager import ModelManager, ModelMetadata, ModelType, ModelStatus
from core.data_processing.data_preprocessor import DataPreprocessor
from core.data_processing.feature_engineer import FeatureEngineer

# Import ML libraries
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Enums and Types ============
class PredictionType(str, Enum):
    """Types of predictions"""
    PRICE = "price"
    DIRECTION = "direction"
    VOLATILITY = "volatility"
    SIGNAL = "signal"
    CONFIDENCE = "confidence"
    ANOMALY = "anomaly"

class SignalStrength(str, Enum):
    """Strength of trading signals"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class ConfidenceLevel(str, Enum):
    """Confidence levels for predictions"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

# ============ Data Structures ============
@dataclass
class PredictionConfig:
    """Configuration for model predictions"""
    
    # Model settings
    model_id: str
    model_version: str = "latest"
    model_type: Optional[ModelType] = None
    
    # Prediction settings
    prediction_type: PredictionType = PredictionType.PRICE
    forecast_horizon: int = 1
    sequence_length: int = 60
    lookback_window: int = 100
    
    # Confidence settings
    enable_confidence_scoring: bool = True
    confidence_method: str = "ensemble"  # ensemble, monte_carlo, bayesian
    confidence_thresholds: Dict[ConfidenceLevel, float] = field(default_factory=lambda: {
        ConfidenceLevel.VERY_HIGH: 0.9,
        ConfidenceLevel.HIGH: 0.7,
        ConfidenceLevel.MEDIUM: 0.5,
        ConfidenceLevel.LOW: 0.3,
        ConfidenceLevel.VERY_LOW: 0.1
    })
    
    # Signal generation
    enable_signal_generation: bool = True
    signal_thresholds: Dict[SignalStrength, float] = field(default_factory=lambda: {
        SignalStrength.STRONG_BUY: 0.8,
        SignalStrength.BUY: 0.6,
        SignalStrength.NEUTRAL: 0.4,
        SignalStrength.SELL: 0.2,
        SignalStrength.STRONG_SELL: 0.0
    })
    
    # Risk management
    enable_risk_scoring: bool = True
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    
    # Post-processing
    enable_smoothing: bool = True
    smoothing_window: int = 3
    enable_outlier_filtering: bool = True
    outlier_threshold: float = 3.0  # Standard deviations
    
    # Ensemble settings
    enable_ensemble: bool = False
    ensemble_models: List[str] = field(default_factory=list)
    ensemble_method: str = "weighted_average"  # weighted_average, majority_vote, stacking
    
    # Performance settings
    batch_size: int = 32
    use_gpu: bool = True
    parallel_predictions: bool = True
    max_workers: int = 4
    
    # Monitoring
    log_predictions: bool = True
    prediction_log_path: str = "logs/predictions/"
    cache_predictions: bool = True
    cache_ttl: int = 300  # 5 minutes
    
    def __post_init__(self):
        """Validate configuration"""
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        
        # Create log directory
        Path(self.prediction_log_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_id': self.model_id,
            'model_version': self.model_version,
            'model_type': self.model_type.value if self.model_type else None,
            'prediction_type': self.prediction_type.value,
            'forecast_horizon': self.forecast_horizon,
            'sequence_length': self.sequence_length,
            'lookback_window': self.lookback_window,
            'enable_confidence_scoring': self.enable_confidence_scoring,
            'confidence_method': self.confidence_method,
            'confidence_thresholds': {k.value: v for k, v in self.confidence_thresholds.items()},
            'enable_signal_generation': self.enable_signal_generation,
            'signal_thresholds': {k.value: v for k, v in self.signal_thresholds.items()},
            'enable_risk_scoring': self.enable_risk_scoring,
            'max_position_size': self.max_position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'enable_smoothing': self.enable_smoothing,
            'smoothing_window': self.smoothing_window,
            'enable_outlier_filtering': self.enable_outlier_filtering,
            'outlier_threshold': self.outlier_threshold,
            'enable_ensemble': self.enable_ensemble,
            'ensemble_models': self.ensemble_models,
            'ensemble_method': self.ensemble_method,
            'batch_size': self.batch_size,
            'use_gpu': self.use_gpu,
            'parallel_predictions': self.parallel_predictions,
            'max_workers': self.max_workers,
            'log_predictions': self.log_predictions,
            'prediction_log_path': self.prediction_log_path,
            'cache_predictions': self.cache_predictions,
            'cache_ttl': self.cache_ttl
        }

@dataclass
class PredictionResult:
    """Result of model prediction"""
    
    # Prediction data
    timestamp: datetime
    prediction_id: str
    model_id: str
    model_version: str
    
    # Raw predictions
    raw_predictions: np.ndarray
    prediction_type: PredictionType
    
    # Processed predictions
    predicted_value: float
    predicted_values: Optional[np.ndarray] = None
    prediction_interval: Optional[Tuple[float, float]] = None
    
    # Confidence scores
    confidence_score: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Trading signals
    trading_signal: Optional[SignalStrength] = None
    signal_strength: float = 0.0
    position_size: float = 0.0
    
    # Risk metrics
    risk_score: float = 0.0
    volatility_estimate: float = 0.0
    drawdown_risk: float = 0.0
    
    # Features
    input_features: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    # Model info
    model_metadata: Optional[Dict[str, Any]] = None
    inference_time: float = 0.0
    
    # Metadata
    creation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize result"""
        if self.confidence_score >= 0.9:
            self.confidence_level = ConfidenceLevel.VERY_HIGH
        elif self.confidence_score >= 0.7:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence_score >= 0.5:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.confidence_score >= 0.3:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.VERY_LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'timestamp': self.timestamp.isoformat(),
            'prediction_id': self.prediction_id,
            'model_id': self.model_id,
            'model_version': self.model_version,
            'prediction_type': self.prediction_type.value,
            'predicted_value': self.predicted_value,
            'confidence_score': self.confidence_score,
            'confidence_level': self.confidence_level.value,
            'risk_score': self.risk_score,
            'volatility_estimate': self.volatility_estimate,
            'drawdown_risk': self.drawdown_risk,
            'inference_time': self.inference_time,
            'creation_time': self.creation_time.isoformat(),
            'metadata': self.metadata
        }
        
        if self.predicted_values is not None:
            result['predicted_values'] = self.predicted_values.tolist()
        
        if self.prediction_interval is not None:
            result['prediction_interval'] = list(self.prediction_interval)
        
        if self.confidence_interval is not None:
            result['confidence_interval'] = list(self.confidence_interval)
        
        if self.trading_signal is not None:
            result['trading_signal'] = self.trading_signal.value
            result['signal_strength'] = self.signal_strength
            result['position_size'] = self.position_size
        
        if self.feature_importance is not None:
            result['feature_importance'] = self.feature_importance
        
        if self.model_metadata is not None:
            result['model_metadata'] = self.model_metadata
        
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        data = self.to_dict()
        return pd.DataFrame([data])
    
    def save(self, filepath: str):
        """Save prediction result to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.debug(f"Prediction result saved to {filepath}")

@dataclass
class BatchPredictionResult:
    """Result of batch predictions"""
    
    timestamp: datetime
    batch_id: str
    model_id: str
    
    # Batch data
    predictions: List[PredictionResult]
    input_data_shape: Tuple[int, ...]
    
    # Batch statistics
    mean_prediction: float
    std_prediction: float
    min_prediction: float
    max_prediction: float
    
    # Performance metrics
    total_inference_time: float
    avg_inference_time: float
    predictions_per_second: float
    
    # Quality metrics
    avg_confidence: float
    avg_risk_score: float
    
    def __post_init__(self):
        """Initialize batch result"""
        if self.predictions:
            self.avg_confidence = np.mean([p.confidence_score for p in self.predictions])
            self.avg_risk_score = np.mean([p.risk_score for p in self.predictions])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'batch_id': self.batch_id,
            'model_id': self.model_id,
            'input_data_shape': self.input_data_shape,
            'num_predictions': len(self.predictions),
            'mean_prediction': self.mean_prediction,
            'std_prediction': self.std_prediction,
            'min_prediction': self.min_prediction,
            'max_prediction': self.max_prediction,
            'total_inference_time': self.total_inference_time,
            'avg_inference_time': self.avg_inference_time,
            'predictions_per_second': self.predictions_per_second,
            'avg_confidence': self.avg_confidence,
            'avg_risk_score': self.avg_risk_score,
            'predictions': [p.to_dict() for p in self.predictions]
        }

@dataclass 
class ModelState:
    """State of a loaded model"""
    
    model_id: str
    model: Any
    metadata: ModelMetadata
    preprocessor: Optional[Any] = None
    feature_selector: Optional[Any] = None
    scaler: Optional[Any] = None
    last_used: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    average_inference_time: float = 0.0
    
    def update_usage(self, inference_time: float):
        """Update usage statistics"""
        self.last_used = datetime.now()
        self.usage_count += 1
        
        # Update average inference time
        if self.average_inference_time == 0:
            self.average_inference_time = inference_time
        else:
            alpha = 0.1  # Smoothing factor
            self.average_inference_time = (alpha * inference_time + 
                                         (1 - alpha) * self.average_inference_time)

# ============ Signal Generator ============
class SignalGenerator:
    """Generates trading signals from predictions"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.SignalGenerator")
        
    def generate_signal(self, 
                       prediction: float,
                       confidence: float,
                       current_price: float,
                       historical_prices: np.ndarray) -> Tuple[SignalStrength, float, float]:
        """Generate trading signal from prediction"""
        
        # Calculate price change percentage
        price_change_pct = (prediction - current_price) / current_price
        
        # Adjust signal strength based on confidence
        adjusted_change = price_change_pct * confidence
        
        # Generate signal based on thresholds
        signal, strength = self._calculate_signal(adjusted_change)
        
        # Calculate position size based on confidence and risk
        position_size = self._calculate_position_size(signal, strength, confidence, historical_prices)
        
        return signal, strength, position_size
    
    def _calculate_signal(self, adjusted_change: float) -> Tuple[SignalStrength, float]:
        """Calculate signal from adjusted price change"""
        
        # Get thresholds
        strong_buy_threshold = self.config.signal_thresholds[SignalStrength.STRONG_BUY]
        buy_threshold = self.config.signal_thresholds[SignalStrength.BUY]
        sell_threshold = self.config.signal_thresholds[SignalStrength.SELL]
        strong_sell_threshold = self.config.signal_thresholds[SignalStrength.STRONG_SELL]
        
        # Determine signal
        if adjusted_change >= strong_buy_threshold:
            signal = SignalStrength.STRONG_BUY
            strength = min(1.0, (adjusted_change - strong_buy_threshold) / (1.0 - strong_buy_threshold))
        
        elif adjusted_change >= buy_threshold:
            signal = SignalStrength.BUY
            strength = (adjusted_change - buy_threshold) / (strong_buy_threshold - buy_threshold)
        
        elif adjusted_change <= strong_sell_threshold:
            signal = SignalStrength.STRONG_SELL
            strength = min(1.0, (strong_sell_threshold - adjusted_change) / (strong_sell_threshold + 1.0))
        
        elif adjusted_change <= sell_threshold:
            signal = SignalStrength.SELL
            strength = (sell_threshold - adjusted_change) / (sell_threshold - strong_sell_threshold)
        
        else:
            signal = SignalStrength.NEUTRAL
            strength = 0.0
        
        return signal, strength
    
    def _calculate_position_size(self,
                                signal: SignalStrength,
                                strength: float,
                                confidence: float,
                                historical_prices: np.ndarray) -> float:
        """Calculate position size based on signal and risk"""
        
        # Base position size from config
        base_size = self.config.max_position_size
        
        # Adjust by signal strength
        if signal in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL]:
            size_multiplier = 1.0
        elif signal in [SignalStrength.BUY, SignalStrength.SELL]:
            size_multiplier = 0.7
        else:
            size_multiplier = 0.0
        
        # Adjust by confidence
        confidence_multiplier = confidence
        
        # Adjust by market volatility (reduce position in high volatility)
        volatility = self._calculate_volatility(historical_prices)
        volatility_multiplier = max(0.5, 1.0 - volatility)
        
        # Calculate final position size
        position_size = base_size * size_multiplier * confidence_multiplier * volatility_multiplier
        
        return min(position_size, self.config.max_position_size)
    
    def _calculate_volatility(self, prices: np.ndarray, lookback: int = 20) -> float:
        """Calculate market volatility"""
        
        if len(prices) < lookback:
            lookback = len(prices)
        
        recent_prices = prices[-lookback:]
        
        if len(recent_prices) < 2:
            return 0.0
        
        # Calculate returns
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Calculate volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)
        
        return min(volatility, 1.0)  # Cap at 100%

# ============ Confidence Scorer ============
class ConfidenceScorer:
    """Calculates confidence scores for predictions"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.ConfidenceScorer")
        
    def calculate_confidence(self,
                           prediction: float,
                           model: Any,
                           input_data: np.ndarray,
                           historical_predictions: List[float] = None) -> Tuple[float, ConfidenceLevel, Optional[Tuple[float, float]]]:
        """Calculate confidence score for prediction"""
        
        if self.config.confidence_method == "ensemble":
            return self._ensemble_confidence(prediction, model, input_data)
        
        elif self.config.confidence_method == "monte_carlo":
            return self._monte_carlo_confidence(prediction, model, input_data)
        
        elif self.config.confidence_method == "bayesian":
            return self._bayesian_confidence(prediction, model, input_data)
        
        else:
            # Default confidence based on historical accuracy
            return self._historical_confidence(prediction, historical_predictions)
    
    def _ensemble_confidence(self,
                           prediction: float,
                           model: Any,
                           input_data: np.ndarray) -> Tuple[float, ConfidenceLevel, Optional[Tuple[float, float]]]:
        """Calculate confidence using ensemble methods"""
        
        # This would require multiple models or ensemble techniques
        # For now, return a simplified version
        
        # Check if model supports uncertainty estimation
        if hasattr(model, 'predict_proba'):
            # For classification models
            proba = model.predict_proba(input_data.reshape(1, -1))
            confidence = np.max(proba[0])
        
        elif hasattr(model, 'predict_with_confidence'):
            # Custom method
            result = model.predict_with_confidence(input_data)
            confidence = result['confidence']
        
        else:
            # Default confidence based on input data characteristics
            confidence = self._data_quality_confidence(input_data)
        
        # Get confidence interval (simplified)
        if confidence > 0.7:
            interval = (prediction * 0.95, prediction * 1.05)
        elif confidence > 0.5:
            interval = (prediction * 0.9, prediction * 1.1)
        else:
            interval = (prediction * 0.8, prediction * 1.2)
        
        return confidence, self._get_confidence_level(confidence), interval
    
    def _monte_carlo_confidence(self,
                              prediction: float,
                              model: Any,
                              input_data: np.ndarray,
                              n_samples: int = 100) -> Tuple[float, ConfidenceLevel, Optional[Tuple[float, float]]]:
        """Calculate confidence using Monte Carlo dropout"""
        
        if not TORCH_AVAILABLE or not isinstance(model, nn.Module):
            # Fall back to ensemble method
            return self._ensemble_confidence(prediction, model, input_data)
        
        # Enable dropout at inference time
        model.train()
        
        predictions = []
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
            
            for _ in range(n_samples):
                pred = model(input_tensor).cpu().numpy().flatten()[0]
                predictions.append(pred)
        
        # Disable dropout
        model.eval()
        
        # Calculate statistics
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Confidence based on variance
        if std_pred > 0:
            confidence = 1.0 / (1.0 + std_pred / mean_pred) if mean_pred != 0 else 0.5
        else:
            confidence = 0.9
        
        # Confidence interval (95%)
        interval = (mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred)
        
        return confidence, self._get_confidence_level(confidence), interval
    
    def _bayesian_confidence(self,
                           prediction: float,
                           model: Any,
                           input_data: np.ndarray) -> Tuple[float, ConfidenceLevel, Optional[Tuple[float, float]]]:
        """Calculate confidence using Bayesian methods"""
        
        # Simplified Bayesian confidence
        # In practice, this would require Bayesian neural networks
        
        # For now, return medium confidence
        confidence = 0.5
        interval = (prediction * 0.9, prediction * 1.1)
        
        return confidence, self._get_confidence_level(confidence), interval
    
    def _historical_confidence(self,
                             prediction: float,
                             historical_predictions: List[float]) -> Tuple[float, ConfidenceLevel, Optional[Tuple[float, float]]]:
        """Calculate confidence based on historical accuracy"""
        
        if not historical_predictions or len(historical_predictions) < 10:
            # Not enough historical data
            confidence = 0.5
            interval = (prediction * 0.9, prediction * 1.1)
        
        else:
            # Calculate prediction stability
            recent_predictions = historical_predictions[-10:]
            std_dev = np.std(recent_predictions)
            mean_pred = np.mean(recent_predictions)
            
            if mean_pred != 0:
                coefficient_of_variation = std_dev / mean_pred
                confidence = 1.0 / (1.0 + coefficient_of_variation)
            else:
                confidence = 0.5
            
            # Confidence interval based on historical variation
            interval = (prediction - std_dev, prediction + std_dev)
        
        return confidence, self._get_confidence_level(confidence), interval
    
    def _data_quality_confidence(self, input_data: np.ndarray) -> float:
        """Calculate confidence based on input data quality"""
        
        # Check for missing values
        if np.any(np.isnan(input_data)):
            return 0.3
        
        # Check for outliers
        mean = np.mean(input_data)
        std = np.std(input_data)
        
        if std == 0:
            # No variation in data
            return 0.4
        
        # Count outliers
        z_scores = np.abs((input_data - mean) / std)
        outlier_count = np.sum(z_scores > self.config.outlier_threshold)
        outlier_ratio = outlier_count / len(input_data)
        
        # Confidence decreases with more outliers
        confidence = 1.0 - outlier_ratio
        
        return max(0.1, min(1.0, confidence))
    
    def _get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Get confidence level from score"""
        
        if confidence_score >= self.config.confidence_thresholds[ConfidenceLevel.VERY_HIGH]:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= self.config.confidence_thresholds[ConfidenceLevel.HIGH]:
            return ConfidenceLevel.HIGH
        elif confidence_score >= self.config.confidence_thresholds[ConfidenceLevel.MEDIUM]:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= self.config.confidence_thresholds[ConfidenceLevel.LOW]:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

# ============ Risk Assessor ============
class RiskAssessor:
    """Assesses risk for trading predictions"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.RiskAssessor")
        
    def assess_risk(self,
                   prediction: float,
                   confidence: float,
                   historical_prices: np.ndarray,
                   market_conditions: Dict[str, Any] = None) -> Dict[str, float]:
        """Assess risk for a prediction"""
        
        risk_metrics = {
            'risk_score': 0.0,
            'volatility_estimate': 0.0,
            'drawdown_risk': 0.0,
            'market_risk': 0.0,
            'liquidity_risk': 0.0
        }
        
        # Calculate volatility
        volatility = self._calculate_volatility(historical_prices)
        risk_metrics['volatility_estimate'] = volatility
        
        # Calculate drawdown risk
        drawdown_risk = self._calculate_drawdown_risk(historical_prices)
        risk_metrics['drawdown_risk'] = drawdown_risk
        
        # Calculate market risk based on conditions
        if market_conditions:
            market_risk = self._calculate_market_risk(market_conditions)
            risk_metrics['market_risk'] = market_risk
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(
            confidence, volatility, drawdown_risk, risk_metrics.get('market_risk', 0.0)
        )
        risk_metrics['risk_score'] = risk_score
        
        # Calculate stop loss and take profit levels
        if self.config.stop_loss_pct > 0:
            stop_loss = prediction * (1 - self.config.stop_loss_pct)
            take_profit = prediction * (1 + self.config.take_profit_pct)
            risk_metrics['stop_loss'] = stop_loss
            risk_metrics['take_profit'] = take_profit
        
        return risk_metrics
    
    def _calculate_volatility(self, prices: np.ndarray, lookback: int = 20) -> float:
        """Calculate historical volatility"""
        
        if len(prices) < 2:
            return 0.0
        
        # Use recent data
        if len(prices) > lookback:
            prices = prices[-lookback:]
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)
        
        return min(volatility, 2.0)  # Cap at 200%
    
    def _calculate_drawdown_risk(self, prices: np.ndarray, lookback: int = 50) -> float:
        """Calculate risk of significant drawdown"""
        
        if len(prices) < lookback:
            lookback = len(prices)
        
        recent_prices = prices[-lookback:]
        
        if len(recent_prices) < 2:
            return 0.0
        
        # Calculate maximum drawdown
        peak = recent_prices[0]
        max_drawdown = 0
        
        for price in recent_prices[1:]:
            if price > peak:
                peak = price
            
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return min(max_drawdown, 1.0)  # Cap at 100%
    
    def _calculate_market_risk(self, market_conditions: Dict[str, Any]) -> float:
        """Calculate risk based on market conditions"""
        
        risk = 0.0
        
        # Check for high volume (lower risk)
        if 'volume' in market_conditions:
            volume = market_conditions['volume']
            # Normalize volume risk (higher volume = lower risk)
            volume_risk = 1.0 / (1.0 + volume / 1e6)  # Scale with volume
            risk += volume_risk * 0.3
        
        # Check for volatility index
        if 'vix' in market_conditions:
            vix = market_conditions['vix']
            vix_risk = min(vix / 50.0, 1.0)  # Normalize to 0-1
            risk += vix_risk * 0.4
        
        # Check for market sentiment
        if 'sentiment' in market_conditions:
            sentiment = market_conditions['sentiment']
            # Sentiment from -1 (bearish) to 1 (bullish)
            sentiment_risk = (1.0 - sentiment) / 2.0  # Convert to 0-1 risk
            risk += sentiment_risk * 0.3
        
        return min(risk, 1.0)
    
    def _calculate_risk_score(self,
                            confidence: float,
                            volatility: float,
                            drawdown_risk: float,
                            market_risk: float) -> float:
        """Calculate overall risk score"""
        
        # Weights for different risk factors
        weights = {
            'confidence': 0.4,      # Higher confidence = lower risk
            'volatility': 0.3,      # Higher volatility = higher risk
            'drawdown': 0.2,        # Higher drawdown risk = higher risk
            'market': 0.1           # Higher market risk = higher risk
        }
        
        # Normalize confidence to risk (higher confidence = lower risk)
        confidence_risk = 1.0 - confidence
        
        # Calculate weighted risk score
        risk_score = (
            confidence_risk * weights['confidence'] +
            volatility * weights['volatility'] +
            drawdown_risk * weights['drawdown'] +
            market_risk * weights['market']
        )
        
        return min(risk_score, 1.0)

# ============ Post-Processor ============
class PredictionPostProcessor:
    """Post-processes model predictions"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.PostProcessor")
        
    def process(self, predictions: np.ndarray) -> np.ndarray:
        """Process predictions with various techniques"""
        
        processed = predictions.copy()
        
        # Apply smoothing if enabled
        if self.config.enable_smoothing and len(predictions) > self.config.smoothing_window:
            processed = self._apply_smoothing(processed)
        
        # Filter outliers if enabled
        if self.config.enable_outlier_filtering:
            processed = self._filter_outliers(processed)
        
        return processed
    
    def _apply_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """Apply smoothing to predictions"""
        
        if len(predictions) < self.config.smoothing_window:
            return predictions
        
        # Apply moving average
        smoothed = np.zeros_like(predictions)
        
        for i in range(len(predictions)):
            start_idx = max(0, i - self.config.smoothing_window + 1)
            window = predictions[start_idx:i+1]
            smoothed[i] = np.mean(window)
        
        return smoothed
    
    def _filter_outliers(self, predictions: np.ndarray) -> np.ndarray:
        """Filter outlier predictions"""
        
        if len(predictions) < 3:
            return predictions
        
        # Calculate z-scores
        mean = np.mean(predictions)
        std = np.std(predictions)
        
        if std == 0:
            return predictions
        
        z_scores = np.abs((predictions - mean) / std)
        
        # Replace outliers with median
        outlier_mask = z_scores > self.config.outlier_threshold
        
        if np.any(outlier_mask):
            median = np.median(predictions)
            filtered = predictions.copy()
            filtered[outlier_mask] = median
            return filtered
        
        return predictions

# ============ Main Model Predictor ============
class ModelPredictor:
    """Main model prediction engine"""
    
    def __init__(self, 
                 config: PredictionConfig,
                 model_manager: ModelManager):
        
        self.config = config
        self.model_manager = model_manager
        self.logger = get_logger(__name__)
        
        # Model cache
        self.model_cache: Dict[str, ModelState] = {}
        self.max_cache_size = 10
        
        # Prediction cache
        self.prediction_cache = Cache(ttl=config.cache_ttl)
        
        # Component instances
        self.signal_generator = SignalGenerator(config)
        self.confidence_scorer = ConfidenceScorer(config)
        self.risk_assessor = RiskAssessor(config)
        self.post_processor = PredictionPostProcessor(config)
        
        # Statistics
        self.total_predictions = 0
        self.average_inference_time = 0.0
        
        # Device
        self.device = self._get_device()
        
        # Performance monitoring
        self.inference_times = deque(maxlen=1000)
        self.prediction_errors = deque(maxlen=1000)
        
        self.logger.info(f"Model Predictor initialized for model {config.model_id}")
        self.logger.info(f"Using device: {self.device}")
    
    def _get_device(self) -> str:
        """Get available device (CPU or GPU)"""
        
        if self.config.use_gpu:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            elif TF_AVAILABLE and tf.config.list_physical_devices('GPU'):
                return "gpu"
        
        return "cpu"
    
    def load_model(self, model_id: Optional[str] = None) -> ModelState:
        """Load model into cache"""
        
        model_id = model_id or self.config.model_id
        
        # Check if model is already cached
        if model_id in self.model_cache:
            self.logger.debug(f"Model {model_id} found in cache")
            self.model_cache[model_id].last_used = datetime.now()
            return self.model_cache[model_id]
        
        # Load model from model manager
        self.logger.info(f"Loading model {model_id}")
        
        try:
            model_data = self.model_manager.load_model(model_id, self.config.model_version)
            
            if not model_data:
                raise ValueError(f"Model {model_id} not found")
            
            model, metadata, preprocessor, feature_selector, scaler = model_data
            
            # Update config with model type if not set
            if not self.config.model_type and metadata.model_type:
                self.config.model_type = metadata.model_type
            
            # Create model state
            model_state = ModelState(
                model_id=model_id,
                model=model,
                metadata=metadata,
                preprocessor=preprocessor,
                feature_selector=feature_selector,
                scaler=scaler
            )
            
            # Add to cache
            self.model_cache[model_id] = model_state
            
            # Manage cache size
            if len(self.model_cache) > self.max_cache_size:
                # Remove least recently used model
                oldest = min(self.model_cache.values(), key=lambda x: x.last_used)
                del self.model_cache[oldest.model_id]
                self.logger.debug(f"Removed model {oldest.model_id} from cache")
            
            self.logger.info(f"Model {model_id} loaded successfully")
            return model_state
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise
    
    def predict(self,
                data: pd.DataFrame,
                timestamp: Optional[datetime] = None,
                current_price: Optional[float] = None,
                market_conditions: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Make prediction for given data"""
        
        start_time = time.time()
        
        # Generate prediction ID
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Use current timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Load model
            model_state = self.load_model()
            
            # Preprocess data
            processed_data = self._preprocess_data(data, model_state)
            
            # Check cache for same input
            cache_key = self._get_cache_key(processed_data)
            if self.config.cache_predictions:
                cached_result = self.prediction_cache.get(cache_key)
                if cached_result:
                    self.logger.debug(f"Using cached prediction for key: {cache_key[:20]}...")
                    cached_result.prediction_id = prediction_id
                    cached_result.timestamp = timestamp
                    return cached_result
            
            # Make prediction
            raw_prediction = self._make_prediction(processed_data, model_state)
            
            # Post-process prediction
            processed_prediction = self._post_process_prediction(raw_prediction)
            
            # Calculate confidence
            confidence_score, confidence_level, confidence_interval = self._calculate_confidence(
                processed_prediction, model_state.model, processed_data
            )
            
            # Get current price if not provided
            if current_price is None and 'close' in data.columns:
                current_price = data['close'].iloc[-1]
            
            # Generate trading signal if enabled
            trading_signal = None
            signal_strength = 0.0
            position_size = 0.0
            
            if self.config.enable_signal_generation and current_price is not None:
                historical_prices = data['close'].values if 'close' in data.columns else np.array([current_price])
                trading_signal, signal_strength, position_size = self.signal_generator.generate_signal(
                    processed_prediction, confidence_score, current_price, historical_prices
                )
            
            # Assess risk if enabled
            risk_metrics = {}
            if self.config.enable_risk_scoring and current_price is not None:
                historical_prices = data['close'].values if 'close' in data.columns else np.array([current_price])
                risk_metrics = self.risk_assessor.assess_risk(
                    processed_prediction, confidence_score, historical_prices, market_conditions
                )
            
            # Calculate inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.average_inference_time = np.mean(self.inference_times) if self.inference_times else inference_time
            
            # Update model usage statistics
            model_state.update_usage(inference_time)
            
            # Create prediction result
            result = PredictionResult(
                timestamp=timestamp,
                prediction_id=prediction_id,
                model_id=self.config.model_id,
                model_version=self.config.model_version,
                raw_predictions=np.array([raw_prediction]),
                prediction_type=self.config.prediction_type,
                predicted_value=float(processed_prediction),
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                confidence_interval=confidence_interval,
                trading_signal=trading_signal,
                signal_strength=signal_strength,
                position_size=position_size,
                risk_score=risk_metrics.get('risk_score', 0.0),
                volatility_estimate=risk_metrics.get('volatility_estimate', 0.0),
                drawdown_risk=risk_metrics.get('drawdown_risk', 0.0),
                model_metadata=model_state.metadata.to_dict() if model_state.metadata else None,
                inference_time=inference_time,
                metadata={
                    'cache_key': cache_key,
                    'data_shape': processed_data.shape,
                    'model_type': model_state.metadata.model_type.value if model_state.metadata else 'unknown'
                }
            )
            
            # Cache result
            if self.config.cache_predictions:
                self.prediction_cache.set(cache_key, result)
            
            # Log prediction
            if self.config.log_predictions:
                self._log_prediction(result)
            
            self.total_predictions += 1
            
            self.logger.debug(
                f"Prediction {prediction_id}: value={processed_prediction:.4f}, "
                f"confidence={confidence_score:.2f}, signal={trading_signal.value if trading_signal else 'none'}, "
                f"time={inference_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Create error result
            return PredictionResult(
                timestamp=timestamp,
                prediction_id=prediction_id,
                model_id=self.config.model_id,
                model_version=self.config.model_version,
                raw_predictions=np.array([0]),
                prediction_type=self.config.prediction_type,
                predicted_value=0.0,
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                inference_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def predict_batch(self,
                     data_list: List[pd.DataFrame],
                     timestamps: Optional[List[datetime]] = None,
                     current_prices: Optional[List[float]] = None,
                     market_conditions_list: Optional[List[Dict[str, Any]]] = None) -> BatchPredictionResult:
        """Make batch predictions"""
        
        batch_start_time = time.time()
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        if timestamps is None:
            timestamps = [datetime.now()] * len(data_list)
        
        if current_prices is None:
            current_prices = [None] * len(data_list)
        
        if market_conditions_list is None:
            market_conditions_list = [None] * len(data_list)
        
        predictions = []
        inference_times = []
        
        self.logger.info(f"Starting batch prediction with {len(data_list)} items")
        
        # Process in parallel if enabled
        if self.config.parallel_predictions and len(data_list) > 1:
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all prediction tasks
                future_to_idx = {
                    executor.submit(
                        self.predict,
                        data,
                        timestamp,
                        current_price,
                        market_conditions
                    ): idx for idx, (data, timestamp, current_price, market_conditions) in enumerate(zip(
                        data_list, timestamps, current_prices, market_conditions_list
                    ))
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result(timeout=30)
                        predictions.append(result)
                        inference_times.append(result.inference_time)
                    except Exception as e:
                        self.logger.error(f"Batch prediction failed for item {idx}: {str(e)}")
                        # Create error result
                        predictions.append(PredictionResult(
                            timestamp=timestamps[idx],
                            prediction_id=f"error_{idx}",
                            model_id=self.config.model_id,
                            model_version=self.config.model_version,
                            raw_predictions=np.array([0]),
                            prediction_type=self.config.prediction_type,
                            predicted_value=0.0,
                            confidence_score=0.0,
                            confidence_level=ConfidenceLevel.VERY_LOW,
                            inference_time=0.0,
                            metadata={'error': str(e), 'index': idx}
                        ))
        else:
            # Process sequentially
            for idx, (data, timestamp, current_price, market_conditions) in enumerate(zip(
                data_list, timestamps, current_prices, market_conditions_list
            )):
                try:
                    result = self.predict(data, timestamp, current_price, market_conditions)
                    predictions.append(result)
                    inference_times.append(result.inference_time)
                except Exception as e:
                    self.logger.error(f"Batch prediction failed for item {idx}: {str(e)}")
                    # Create error result
                    predictions.append(PredictionResult(
                        timestamp=timestamp,
                        prediction_id=f"error_{idx}",
                        model_id=self.config.model_id,
                        model_version=self.config.model_version,
                        raw_predictions=np.array([0]),
                        prediction_type=self.config.prediction_type,
                        predicted_value=0.0,
                        confidence_score=0.0,
                        confidence_level=ConfidenceLevel.VERY_LOW,
                        inference_time=0.0,
                        metadata={'error': str(e), 'index': idx}
                    ))
        
        # Calculate batch statistics
        predicted_values = [p.predicted_value for p in predictions if p.predicted_value != 0]
        
        if predicted_values:
            mean_prediction = np.mean(predicted_values)
            std_prediction = np.std(predicted_values)
            min_prediction = np.min(predicted_values)
            max_prediction = np.max(predicted_values)
        else:
            mean_prediction = 0.0
            std_prediction = 0.0
            min_prediction = 0.0
            max_prediction = 0.0
        
        total_inference_time = sum(inference_times)
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        predictions_per_second = len(predictions) / total_inference_time if total_inference_time > 0 else 0.0
        
        # Create batch result
        batch_result = BatchPredictionResult(
            timestamp=datetime.now(),
            batch_id=batch_id,
            model_id=self.config.model_id,
            predictions=predictions,
            input_data_shape=data_list[0].shape if data_list else (0, 0),
            mean_prediction=mean_prediction,
            std_prediction=std_prediction,
            min_prediction=min_prediction,
            max_prediction=max_prediction,
            total_inference_time=total_inference_time,
            avg_inference_time=avg_inference_time,
            predictions_per_second=predictions_per_second,
            avg_confidence=0.0,  # Will be calculated in __post_init__
            avg_risk_score=0.0   # Will be calculated in __post_init__
        )
        
        total_batch_time = time.time() - batch_start_time
        self.logger.info(
            f"Batch prediction completed: {len(predictions)} predictions, "
            f"total time={total_batch_time:.2f}s, "
            f"avg time={avg_inference_time:.3f}s, "
            f"throughput={predictions_per_second:.1f} preds/s"
        )
        
        return batch_result
    
    def _preprocess_data(self, data: pd.DataFrame, model_state: ModelState) -> np.ndarray:
        """Preprocess data for model input"""
        
        # Extract features (excluding target if present)
        if 'target' in data.columns:
            X = data.drop(columns=['target']).values
        elif 'close' in data.columns:
            # For time series, we might predict the next close price
            X = data.values
        else:
            X = data.values
        
        # Apply preprocessing if available
        if model_state.scaler:
            X = model_state.scaler.transform(X)
        
        if model_state.feature_selector:
            X = model_state.feature_selector.transform(X)
        
        # Reshape for sequence models
        if self.config.model_type in [ModelType.LSTM_ATTENTION, ModelType.CNN_LSTM, ModelType.TRANSFORMER]:
            # Ensure we have enough data for sequence
            if len(X) < self.config.sequence_length:
                # Pad with zeros or repeat first value
                padding = np.zeros((self.config.sequence_length - len(X), X.shape[1]))
                X = np.vstack([padding, X])
            
            # Take the last sequence_length data points
            X = X[-self.config.sequence_length:]
            
            # Reshape for sequence models (batch, sequence, features)
            X = X.reshape(1, self.config.sequence_length, -1)
        
        else:
            # For non-sequence models, use the most recent data point
            X = X[-1:].reshape(1, -1)
        
        return X
    
    def _make_prediction(self, data: np.ndarray, model_state: ModelState) -> float:
        """Make prediction using loaded model"""
        
        model = model_state.model
        
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            # PyTorch model
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(data)
                if self.device == "cuda":
                    input_tensor = input_tensor.cuda()
                    model = model.cuda()
                
                output = model(input_tensor)
                prediction = output.cpu().numpy().flatten()[0]
        
        elif TF_AVAILABLE and isinstance(model, keras.Model):
            # TensorFlow model
            prediction = model.predict(data, verbose=0).flatten()[0]
        
        else:
            # Scikit-learn or other models
            try:
                prediction = model.predict(data)[0]
            except:
                # Try predict_proba for classification
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(data)[0]
                    # For binary classification, use probability of positive class
                    if len(proba) == 2:
                        prediction = proba[1]  # Probability of positive class
                    else:
                        prediction = np.max(proba)
                else:
                    raise ValueError("Model doesn't support predict or predict_proba")
        
        return float(prediction)
    
    def _post_process_prediction(self, prediction: float) -> float:
        """Post-process prediction"""
        
        predictions_array = np.array([prediction])
        processed_array = self.post_processor.process(predictions_array)
        
        return float(processed_array[0])
    
    def _calculate_confidence(self,
                            prediction: float,
                            model: Any,
                            input_data: np.ndarray) -> Tuple[float, ConfidenceLevel, Optional[Tuple[float, float]]]:
        """Calculate confidence for prediction"""
        
        if self.config.enable_confidence_scoring:
            return self.confidence_scorer.calculate_confidence(prediction, model, input_data)
        else:
            # Default medium confidence
            return 0.5, ConfidenceLevel.MEDIUM, (prediction * 0.9, prediction * 1.1)
    
    def _get_cache_key(self, data: np.ndarray) -> str:
        """Generate cache key for data"""
        
        # Create hash from data
        data_hash = hashlib.md5(data.tobytes()).hexdigest()
        
        # Include model and config in key
        key_parts = [
            self.config.model_id,
            self.config.model_version,
            str(self.config.forecast_horizon),
            data_hash
        ]
        
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()
    
    def _log_prediction(self, result: PredictionResult):
        """Log prediction result"""
        
        log_file = Path(self.config.prediction_log_path) / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Load existing logs
        logs = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Add new prediction
        logs.append(result.to_dict())
        
        # Keep only last 1000 predictions per day
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        # Save logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
    
    def predict_ensemble(self,
                        data: pd.DataFrame,
                        model_ids: Optional[List[str]] = None,
                        timestamp: Optional[datetime] = None) -> PredictionResult:
        """Make prediction using ensemble of models"""
        
        if not self.config.enable_ensemble:
            raise ValueError("Ensemble predictions are not enabled")
        
        model_ids = model_ids or self.config.ensemble_models
        
        if not model_ids:
            model_ids = self.model_manager.list_models()[:3]  # Use first 3 models
        
        predictions = []
        confidences = []
        
        self.logger.info(f"Making ensemble prediction with {len(model_ids)} models")
        
        # Get predictions from each model
        for model_id in model_ids:
            try:
                # Temporarily change config to use this model
                original_model_id = self.config.model_id
                self.config.model_id = model_id
                
                # Make prediction
                result = self.predict(data, timestamp)
                
                # Store results
                predictions.append(result.predicted_value)
                confidences.append(result.confidence_score)
                
                # Restore original model ID
                self.config.model_id = original_model_id
                
            except Exception as e:
                self.logger.warning(f"Failed to get prediction from model {model_id}: {str(e)}")
        
        if not predictions:
            raise ValueError("No models successfully made predictions")
        
        # Combine predictions based on ensemble method
        if self.config.ensemble_method == "weighted_average":
            # Weight by confidence
            weights = np.array(confidences)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            ensemble_prediction = np.average(predictions, weights=weights)
            ensemble_confidence = np.mean(confidences)
        
        elif self.config.ensemble_method == "majority_vote":
            # For classification or direction predictions
            # Convert to buy/sell signals and take majority
            buy_signals = sum(1 for p in predictions if p > 0)
            sell_signals = len(predictions) - buy_signals
            
            if buy_signals > sell_signals:
                ensemble_prediction = 1.0  # Buy signal
            else:
                ensemble_prediction = -1.0  # Sell signal
            
            ensemble_confidence = max(buy_signals, sell_signals) / len(predictions)
        
        elif self.config.ensemble_method == "stacking":
            # Would require a meta-model
            # For now, use average
            ensemble_prediction = np.mean(predictions)
            ensemble_confidence = np.mean(confidences)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.ensemble_method}")
        
        # Create ensemble result
        ensemble_result = PredictionResult(
            timestamp=timestamp or datetime.now(),
            prediction_id=f"ensemble_{uuid.uuid4().hex[:8]}",
            model_id="ensemble",
            model_version="1.0.0",
            raw_predictions=np.array(predictions),
            prediction_type=self.config.prediction_type,
            predicted_value=float(ensemble_prediction),
            confidence_score=float(ensemble_confidence),
            confidence_level=self.confidence_scorer._get_confidence_level(ensemble_confidence),
            inference_time=0.0,  # Would need to track total time
            metadata={
                'ensemble_method': self.config.ensemble_method,
                'component_models': model_ids,
                'component_predictions': predictions,
                'component_confidences': confidences
            }
        )
        
        return ensemble_result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the predictor"""
        
        return {
            'total_predictions': self.total_predictions,
            'average_inference_time': self.average_inference_time,
            'cache_hit_rate': self.prediction_cache.get_hit_rate() if hasattr(self.prediction_cache, 'get_hit_rate') else 0.0,
            'model_cache_size': len(self.model_cache),
            'recent_inference_times': list(self.inference_times)[-10:] if self.inference_times else [],
            'device': self.device
        }
    
    def clear_cache(self):
        """Clear model and prediction caches"""
        
        self.model_cache.clear()
        if hasattr(self.prediction_cache, 'clear'):
            self.prediction_cache.clear()
        
        self.logger.info("Caches cleared")

# ============ Helper Functions ============
def create_model_predictor(model_id: str,
                          model_manager: ModelManager,
                          **kwargs) -> ModelPredictor:
    """Factory function to create model predictor"""
    
    # Create base config
    config = PredictionConfig(
        model_id=model_id,
        **kwargs
    )
    
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return ModelPredictor(config, model_manager)


def load_predictor_from_config(config_path: str,
                              model_manager: ModelManager) -> ModelPredictor:
    """Load predictor from configuration file"""
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert string enums back to Enum types
    if 'prediction_type' in config_dict:
        config_dict['prediction_type'] = PredictionType(config_dict['prediction_type'])
    
    if 'model_type' in config_dict and config_dict['model_type']:
        config_dict['model_type'] = ModelType(config_dict['model_type'])
    
    # Convert confidence thresholds
    if 'confidence_thresholds' in config_dict:
        config_dict['confidence_thresholds'] = {
            ConfidenceLevel(k): v for k, v in config_dict['confidence_thresholds'].items()
        }
    
    # Convert signal thresholds
    if 'signal_thresholds' in config_dict:
        config_dict['signal_thresholds'] = {
            SignalStrength(k): v for k, v in config_dict['signal_thresholds'].items()
        }
    
    config = PredictionConfig(**config_dict)
    return ModelPredictor(config, model_manager)


def save_prediction_results(results: Union[PredictionResult, BatchPredictionResult, List[PredictionResult]],
                           filepath: str,
                           format: str = 'json'):
    """Save prediction results to file"""
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        if isinstance(results, PredictionResult):
            data = results.to_dict()
        elif isinstance(results, BatchPredictionResult):
            data = results.to_dict()
        elif isinstance(results, list):
            data = [r.to_dict() for r in results]
        else:
            raise ValueError(f"Unsupported results type: {type(results)}")
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    elif format == 'csv':
        if isinstance(results, PredictionResult):
            df = results.to_dataframe()
        elif isinstance(results, BatchPredictionResult):
            # Convert batch results to DataFrame
            data = []
            for pred in results.predictions:
                pred_dict = pred.to_dict()
                pred_dict['batch_id'] = results.batch_id
                data.append(pred_dict)
            df = pd.DataFrame(data)
        elif isinstance(results, list):
            data = [r.to_dict() for r in results]
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported results type: {type(results)}")
        
        df.to_csv(filepath, index=False)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Prediction results saved to {filepath}")


def validate_prediction_result(result: PredictionResult,
                             min_confidence: float = 0.3,
                             max_risk: float = 0.7) -> Dict[str, bool]:
    """Validate prediction result meets quality criteria"""
    
    validation = {
        'passed': True,
        'checks': {}
    }
    
    # Check confidence
    validation['checks']['confidence'] = result.confidence_score >= min_confidence
    
    # Check risk
    validation['checks']['risk'] = result.risk_score <= max_risk
    
    # Check for valid prediction
    validation['checks']['valid_prediction'] = not np.isnan(result.predicted_value) and not np.isinf(result.predicted_value)
    
    # Check inference time (should be reasonable)
    validation['checks']['inference_time'] = result.inference_time < 5.0  # Less than 5 seconds
    
    # Check model metadata
    validation['checks']['has_metadata'] = result.model_metadata is not None
    
    # Overall pass/fail
    validation['passed'] = all(validation['checks'].values())
    
    return validation


# ============ Example Usage ============
if __name__ == "__main__":
    # Example usage
    print("Model Predictor Module")
    
    # Create a sample config
    config = PredictionConfig(
        model_id="sample_model",
        prediction_type=PredictionType.PRICE,
        forecast_horizon=1
    )
    
    # Note: In real usage, you would need a ModelManager instance
    print(f"Predictor would be created for model {config.model_id}")
    print(f"Prediction type: {config.prediction_type.value}")