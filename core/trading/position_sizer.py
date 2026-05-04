"""
Position sizing module for Bitcoin trading AI.
Determines optimal position sizes based on risk management, market conditions, and strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
from datetime import datetime, timedelta
from scipy import stats, optimize
import math
from collections import deque
import json
from pathlib import Path

# Import project modules
from config.settings import TradingSettings, RiskSettings, AppConstants
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.trading.signal_generator import SignalType, SignalStrength, TradingSignal
from core.risk_management.risk_analyzer import RiskAnalyzer
from core.risk_management.portfolio_optimizer import PortfolioOptimizer

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Position Sizing Methods ============
class PositionSizingMethod(str, Enum):
    """Methods for position sizing"""
    KELLY = "kelly"                    # Kelly Criterion
    FIXED_FRACTIONAL = "fixed_fractional"  # Fixed fractional
    FIXED_RATIO = "fixed_ratio"        # Fixed ratio
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # Volatility adjusted
    SHARPE_OPTIMIZED = "sharpe_optimized"  # Sharpe ratio optimized
    RISK_PARITY = "risk_parity"        # Risk parity
    EQUAL_WEIGHT = "equal_weight"      # Equal weight
    MARTINGALE = "martingale"          # Martingale (anti)
    ANTI_MARTINGALE = "anti_martingale"  # Anti-martingale
    CUSTOM = "custom"                  # Custom method

class PositionSizeUnit(str, Enum):
    """Units for position size"""
    PERCENTAGE = "percentage"          # % of portfolio
    DOLLAR = "dollar"                  # Fixed dollar amount
    UNITS = "units"                    # Number of units
    BTC = "btc"                        # Bitcoin amount
    CONTRACTS = "contracts"            # Number of contracts

class RiskApproach(str, Enum):
    """Risk management approaches"""
    CONSERVATIVE = "conservative"      # Low risk
    MODERATE = "moderate"              # Medium risk
    AGGRESSIVE = "aggressive"          # High risk
    DYNAMIC = "dynamic"                # Adaptive risk
    CUSTOM = "custom"                  # Custom risk

# ============ Configuration ============
@dataclass
class PositionSizingConfig:
    """Configuration for position sizing"""
    
    # General settings
    sizing_method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_ADJUSTED
    size_unit: PositionSizeUnit = PositionSizeUnit.PERCENTAGE
    risk_approach: RiskApproach = RiskApproach.MODERATE
    max_portfolio_risk: float = 0.02  # 2% max risk per trade
    max_position_size: float = 0.1    # 10% max position size
    
    # Kelly Criterion settings
    kelly_fraction: float = 0.5       # Fractional Kelly (0.5 = half Kelly)
    min_win_rate: float = 0.5         # Minimum win rate for Kelly
    min_avg_win_loss_ratio: float = 1.2  # Minimum win/loss ratio
    
    # Volatility settings
    volatility_lookback: int = 20     # Periods for volatility calculation
    volatility_scaling: bool = True   # Scale position by volatility
    target_volatility: float = 0.15   # Target annualized volatility (15%)
    volatility_floor: float = 0.05    # Minimum volatility for scaling
    volatility_cap: float = 0.5       # Maximum volatility for scaling
    
    # Drawdown protection
    max_drawdown: float = 0.1         # 10% maximum drawdown
    drawdown_reduction: float = 0.5   # Reduce position by 50% during drawdown
    recovery_multiplier: float = 1.2  # Increase position after recovery
    
    # Account settings
    initial_capital: float = 10000.0  # Initial capital in USD
    current_capital: Optional[float] = None
    risk_free_rate: float = 0.02      # 2% risk-free rate
    
    # Leverage settings
    use_leverage: bool = False
    max_leverage: float = 3.0         # Maximum leverage
    leverage_decay: float = 0.1       # Leverage decay during drawdown
    
    # Position constraints
    min_position_size: float = 0.01   # 1% minimum position
    max_positions: int = 5            # Maximum concurrent positions
    position_concentration: float = 0.3  # Max 30% in single position
    
    # Market condition adjustments
    adjust_for_volatility: bool = True
    adjust_for_trend: bool = True
    adjust_for_liquidity: bool = True
    adjust_for_correlation: bool = True
    
    # Signal-based adjustments
    use_signal_confidence: bool = True
    use_signal_strength: bool = True
    confidence_multiplier: Dict[str, float] = field(default_factory=lambda: {
        'weak': 0.5,
        'moderate': 0.75,
        'strong': 1.0,
        'very_strong': 1.25
    })
    
    # Performance tracking
    track_performance: bool = True
    performance_window: int = 100
    adapt_to_performance: bool = True
    learning_rate: float = 0.01       # How quickly to adapt
    
    # Advanced settings
    use_machine_learning: bool = False
    ml_model_path: Optional[str] = None
    monte_carlo_simulations: int = 1000
    value_at_risk_confidence: float = 0.95
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_portfolio_risk <= 0 or self.max_portfolio_risk > 1:
            raise ValueError("max_portfolio_risk must be between 0 and 1")
        
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError("max_position_size must be between 0 and 1")
        
        if self.kelly_fraction <= 0 or self.kelly_fraction > 1:
            raise ValueError("kelly_fraction must be between 0 and 1")
        
        if self.current_capital is None:
            self.current_capital = self.initial_capital
        
        # Set risk parameters based on approach
        self._set_risk_parameters()
    
    def _set_risk_parameters(self):
        """Set risk parameters based on risk approach"""
        if self.risk_approach == RiskApproach.CONSERVATIVE:
            self.max_portfolio_risk = 0.01
            self.max_position_size = 0.05
            self.kelly_fraction = 0.25
            self.target_volatility = 0.1
        elif self.risk_approach == RiskApproach.MODERATE:
            self.max_portfolio_risk = 0.02
            self.max_position_size = 0.1
            self.kelly_fraction = 0.5
            self.target_volatility = 0.15
        elif self.risk_approach == RiskApproach.AGGRESSIVE:
            self.max_portfolio_risk = 0.05
            self.max_position_size = 0.2
            self.kelly_fraction = 0.75
            self.target_volatility = 0.25
        elif self.risk_approach == RiskApproach.DYNAMIC:
            # Parameters will be adjusted dynamically
            pass

# ============ Position Size Result ============
@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    position_size: float                     # Size in the configured unit
    size_unit: PositionSizeUnit              # Unit of size
    position_value: float                    # Value in USD
    risk_amount: float                       # Amount at risk in USD
    risk_percentage: float                   % of portfolio at risk
    sizing_method: PositionSizingMethod      # Method used
    confidence: float                        # Confidence in size (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate result"""
        if self.position_size < 0:
            raise ValueError("position_size cannot be negative")
        
        if self.risk_percentage < 0 or self.risk_percentage > 1:
            raise ValueError("risk_percentage must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'position_size': self.position_size,
            'size_unit': self.size_unit.value,
            'position_value': self.position_value,
            'risk_amount': self.risk_amount,
            'risk_percentage': self.risk_percentage,
            'sizing_method': self.sizing_method.value,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionSizeResult':
        """Create from dictionary"""
        return cls(
            position_size=data['position_size'],
            size_unit=PositionSizeUnit(data['size_unit']),
            position_value=data['position_value'],
            risk_amount=data['risk_amount'],
            risk_percentage=data['risk_percentage'],
            sizing_method=PositionSizingMethod(data['sizing_method']),
            confidence=data['confidence'],
            metadata=data['metadata'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

# ============ Portfolio State ============
@dataclass
class PortfolioState:
    """Current state of the portfolio"""
    total_capital: float
    used_capital: float
    available_capital: float
    open_positions: int
    total_risk: float
    current_drawdown: float
    max_drawdown: float
    win_rate: float
    sharpe_ratio: float
    volatility: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_capital': self.total_capital,
            'used_capital': self.used_capital,
            'available_capital': self.available_capital,
            'open_positions': self.open_positions,
            'total_risk': self.total_risk,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'volatility': self.volatility,
            'timestamp': self.timestamp.isoformat()
        }

# ============ Base Position Sizer ============
class BasePositionSizer:
    """Base class for position sizing strategies"""
    
    def __init__(self, config: Optional[PositionSizingConfig] = None):
        self.config = config or PositionSizingConfig()
        self.portfolio_state: Optional[PortfolioState] = None
        self.position_history: List[PositionSizeResult] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.risk_analyzer = RiskAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.logger = get_logger(__name__)
        
        # Performance tracking
        self.win_history = deque(maxlen=self.config.performance_window)
        self.return_history = deque(maxlen=self.config.performance_window)
        
        # Initialize ML model if configured
        self.ml_model = None
        if self.config.use_machine_learning and self.config.ml_model_path:
            self._load_ml_model()
    
    def _load_ml_model(self):
        """Load machine learning model for position sizing"""
        try:
            import joblib
            self.ml_model = joblib.load(self.config.ml_model_path)
            self.logger.info(f"Loaded ML model from {self.config.ml_model_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load ML model: {str(e)}")
    
    def calculate_position_size(self, 
                               signal: TradingSignal,
                               market_data: pd.DataFrame,
                               current_price: float,
                               stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None) -> PositionSizeResult:
        """Calculate position size based on signal and market conditions"""
        raise NotImplementedError
    
    def update_portfolio_state(self, state: PortfolioState):
        """Update portfolio state for position sizing"""
        self.portfolio_state = state
        
        # Update performance metrics
        if state.win_rate > 0:
            self.win_history.append(state.win_rate)
        
        if state.volatility > 0:
            self.return_history.append(state.volatility)
    
    def adjust_for_market_conditions(self, 
                                    base_size: float,
                                    market_data: pd.DataFrame,
                                    signal: TradingSignal) -> float:
        """Adjust position size based on market conditions"""
        adjusted_size = base_size
        
        # Volatility adjustment
        if self.config.adjust_for_volatility:
            adjusted_size *= self._volatility_adjustment(market_data)
        
        # Trend adjustment
        if self.config.adjust_for_trend:
            adjusted_size *= self._trend_adjustment(market_data, signal)
        
        # Liquidity adjustment
        if self.config.adjust_for_liquidity:
            adjusted_size *= self._liquidity_adjustment(market_data)
        
        # Signal confidence adjustment
        if self.config.use_signal_confidence:
            adjusted_size *= self._confidence_adjustment(signal)
        
        # Signal strength adjustment
        if self.config.use_signal_strength:
            adjusted_size *= self._strength_adjustment(signal)
        
        # Drawdown adjustment
        if self.portfolio_state and self.portfolio_state.current_drawdown > 0:
            drawdown_factor = 1 - (self.portfolio_state.current_drawdown * 
                                 self.config.drawdown_reduction)
            adjusted_size *= max(drawdown_factor, 0.1)  # Minimum 10% size
        
        # Ensure within bounds
        adjusted_size = self._apply_constraints(adjusted_size)
        
        return adjusted_size
    
    def _volatility_adjustment(self, market_data: pd.DataFrame) -> float:
        """Adjust position size based on volatility"""
        if len(market_data) < self.config.volatility_lookback:
            return 1.0
        
        # Calculate volatility (standard deviation of returns)
        returns = market_data['close'].pct_change().dropna()
        if len(returns) < self.config.volatility_lookback:
            return 1.0
        
        volatility = returns.rolling(window=self.config.volatility_lookback).std().iloc[-1]
        annualized_vol = volatility * np.sqrt(252)  # Annualize
        
        # Scale inversely with volatility
        if annualized_vol > 0:
            adjustment = self.config.target_volatility / annualized_vol
            # Apply floor and cap
            adjustment = max(self.config.volatility_floor, 
                           min(adjustment, self.config.volatility_cap))
            return adjustment
        
        return 1.0
    
    def _trend_adjustment(self, market_data: pd.DataFrame, signal: TradingSignal) -> float:
        """Adjust position size based on trend strength and direction"""
        if len(market_data) < 50:
            return 1.0
        
        # Calculate trend using moving averages
        short_ma = market_data['close'].rolling(window=20).mean().iloc[-1]
        long_ma = market_data['close'].rolling(window=50).mean().iloc[-1]
        trend_strength = abs(short_ma - long_ma) / long_ma
        
        # Determine if signal aligns with trend
        price_trend = 1 if short_ma > long_ma else -1
        signal_direction = 1 if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else -1
        
        if price_trend * signal_direction > 0:
            # Signal aligns with trend - increase size
            return 1.0 + trend_strength
        else:
            # Signal goes against trend - reduce size
            return 1.0 - trend_strength
    
    def _liquidity_adjustment(self, market_data: pd.DataFrame) -> float:
        """Adjust position size based on liquidity"""
        if 'volume' not in market_data.columns:
            return 1.0
        
        if len(market_data) < 20:
            return 1.0
        
        # Calculate average volume
        avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1]
        current_volume = market_data['volume'].iloc[-1]
        
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            
            # Adjust based on liquidity
            if volume_ratio > 2.0:
                return 1.2  # High liquidity - increase size
            elif volume_ratio > 1.0:
                return 1.0  # Normal liquidity
            elif volume_ratio > 0.5:
                return 0.8  # Low liquidity - reduce size
            else:
                return 0.5  # Very low liquidity - significantly reduce size
        
        return 1.0
    
    def _confidence_adjustment(self, signal: TradingSignal) -> float:
        """Adjust position size based on signal confidence"""
        confidence = signal.confidence
        
        # Map confidence to multiplier
        if confidence >= 0.8:
            return self.config.confidence_multiplier['very_strong']
        elif confidence >= 0.7:
            return self.config.confidence_multiplier['strong']
        elif confidence >= 0.6:
            return self.config.confidence_multiplier['moderate']
        else:
            return self.config.confidence_multiplier['weak']
    
    def _strength_adjustment(self, signal: TradingSignal) -> float:
        """Adjust position size based on signal strength"""
        return signal.strength  # Direct multiplier
    
    def _apply_constraints(self, size: float) -> float:
        """Apply position size constraints"""
        # Minimum position size
        size = max(size, self.config.min_position_size)
        
        # Maximum position size
        size = min(size, self.config.max_position_size)
        
        # Portfolio concentration constraint
        if self.portfolio_state and self.portfolio_state.open_positions > 0:
            avg_position_size = self.portfolio_state.used_capital / self.portfolio_state.open_positions
            max_concentration = self.config.position_concentration * self.portfolio_state.total_capital
            
            if size * self.portfolio_state.total_capital > max_concentration:
                size = max_concentration / self.portfolio_state.total_capital
        
        return size
    
    def record_performance(self, 
                          position_result: PositionSizeResult,
                          outcome: Dict[str, Any]):
        """Record position performance for learning"""
        perf_record = {
            'timestamp': datetime.now(),
            'position_result': position_result.to_dict(),
            'outcome': outcome,
            'portfolio_state': self.portfolio_state.to_dict() if self.portfolio_state else None
        }
        
        self.performance_history.append(perf_record)
        
        # Keep only recent history
        if len(self.performance_history) > self.config.performance_window:
            self.performance_history = self.performance_history[-self.config.performance_window:]
        
        # Update win/loss history for Kelly if needed
        if 'profit' in outcome and self.config.sizing_method == PositionSizingMethod.KELLY:
            profit = outcome['profit']
            self.win_history.append(profit > 0)
            self.return_history.append(profit)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for position sizing"""
        if not self.performance_history:
            return {}
        
        metrics = {
            'total_trades': len(self.performance_history),
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'largest_win': -np.inf,
            'largest_loss': np.inf,
            'profit_factor': 0
        }
        
        wins = []
        losses = []
        
        for record in self.performance_history:
            outcome = record['outcome']
            if 'profit' in outcome:
                profit = outcome['profit']
                metrics['total_profit'] += profit
                
                if profit > 0:
                    metrics['winning_trades'] += 1
                    wins.append(profit)
                    metrics['largest_win'] = max(metrics['largest_win'], profit)
                else:
                    metrics['losing_trades'] += 1
                    losses.append(profit)
                    metrics['largest_loss'] = min(metrics['largest_loss'], profit)
        
        if wins:
            metrics['avg_profit'] = np.mean(wins)
        if losses:
            metrics['avg_loss'] = np.mean(losses)
        
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        
        if total_losses > 0:
            metrics['profit_factor'] = total_wins / total_losses
        
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
        
        return metrics

# ============ Kelly Criterion Sizer ============
class KellyPositionSizer(BasePositionSizer):
    """Position sizing using Kelly Criterion"""
    
    def calculate_position_size(self, 
                               signal: TradingSignal,
                               market_data: pd.DataFrame,
                               current_price: float,
                               stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None) -> PositionSizeResult:
        """Calculate position size using Kelly Criterion"""
        
        # Get historical performance for win rate and win/loss ratio
        performance = self.get_performance_metrics()
        
        if len(self.win_history) < 10:  # Need sufficient history
            # Use default values if insufficient history
            win_rate = 0.55
            avg_win_loss_ratio = 1.5
        else:
            win_rate = performance.get('win_rate', 0.55)
            
            if performance.get('avg_profit', 0) > 0 and abs(performance.get('avg_loss', 0)) > 0:
                avg_win_loss_ratio = performance['avg_profit'] / abs(performance['avg_loss'])
            else:
                avg_win_loss_ratio = 1.5
        
        # Ensure minimum thresholds
        win_rate = max(win_rate, self.config.min_win_rate)
        avg_win_loss_ratio = max(avg_win_loss_ratio, self.config.min_avg_win_loss_ratio)
        
        # Calculate Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction(win_rate, avg_win_loss_ratio)
        
        # Apply fractional Kelly
        position_percentage = kelly_fraction * self.config.kelly_fraction
        
        # Adjust for market conditions
        adjusted_percentage = self.adjust_for_market_conditions(
            position_percentage, market_data, signal
        )
        
        # Calculate position value
        if self.portfolio_state:
            capital = self.portfolio_state.available_capital
        else:
            capital = self.config.current_capital or self.config.initial_capital
        
        position_value = adjusted_percentage * capital
        
        # Calculate risk
        if stop_loss:
            risk_per_trade = abs(current_price - stop_loss) / current_price
            risk_amount = position_value * risk_per_trade
        else:
            # Default to 2% risk per trade
            risk_per_trade = 0.02
            risk_amount = position_value * risk_per_trade
        
        # Convert to appropriate unit
        position_size = self._convert_to_unit(adjusted_percentage, position_value, current_price)
        
        # Calculate confidence based on historical performance
        confidence = min(win_rate * avg_win_loss_ratio / 2, 1.0)
        
        result = PositionSizeResult(
            position_size=position_size,
            size_unit=self.config.size_unit,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage=risk_per_trade * adjusted_percentage,
            sizing_method=PositionSizingMethod.KELLY,
            confidence=confidence,
            metadata={
                'kelly_fraction': kelly_fraction,
                'win_rate': win_rate,
                'win_loss_ratio': avg_win_loss_ratio,
                'fractional_kelly': self.config.kelly_fraction,
                'calculated_percentage': position_percentage,
                'adjusted_percentage': adjusted_percentage
            }
        )
        
        self.position_history.append(result)
        return result
    
    def _calculate_kelly_fraction(self, win_rate: float, win_loss_ratio: float) -> float:
        """Calculate Kelly Criterion fraction"""
        # Kelly formula: f* = (bp - q) / b
        # where b = win/loss ratio, p = win rate, q = loss rate = 1-p
        b = win_loss_ratio
        p = win_rate
        q = 1 - p
        
        if b <= 0:
            return 0
        
        kelly = (b * p - q) / b
        
        # Ensure Kelly is between 0 and 1
        kelly = max(0, min(kelly, 1))
        
        return kelly

# ============ Fixed Fractional Sizer ============
class FixedFractionalPositionSizer(BasePositionSizer):
    """Position sizing using fixed fractional method"""
    
    def calculate_position_size(self, 
                               signal: TradingSignal,
                               market_data: pd.DataFrame,
                               current_price: float,
                               stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None) -> PositionSizeResult:
        """Calculate position size using fixed fractional method"""
        
        # Calculate risk per trade
        if stop_loss:
            risk_per_trade = abs(current_price - stop_loss) / current_price
        else:
            # Default risk based on volatility
            if len(market_data) >= 20:
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.rolling(window=20).std().iloc[-1]
                risk_per_trade = min(volatility * 2, 0.05)  # Max 5% risk
            else:
                risk_per_trade = 0.02  # 2% default risk
        
        # Base position percentage from config
        base_percentage = self.config.max_portfolio_risk / risk_per_trade
        
        # Ensure maximum position size constraint
        base_percentage = min(base_percentage, self.config.max_position_size)
        
        # Adjust for market conditions
        adjusted_percentage = self.adjust_for_market_conditions(
            base_percentage, market_data, signal
        )
        
        # Calculate position value
        if self.portfolio_state:
            capital = self.portfolio_state.available_capital
        else:
            capital = self.config.current_capital or self.config.initial_capital
        
        position_value = adjusted_percentage * capital
        
        # Calculate risk amount
        risk_amount = position_value * risk_per_trade
        
        # Convert to appropriate unit
        position_size = self._convert_to_unit(adjusted_percentage, position_value, current_price)
        
        # Calculate confidence
        confidence = min(signal.confidence * 0.8 + 0.2, 1.0)  # Base confidence on signal
        
        result = PositionSizeResult(
            position_size=position_size,
            size_unit=self.config.size_unit,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage=risk_per_trade,
            sizing_method=PositionSizingMethod.FIXED_FRACTIONAL,
            confidence=confidence,
            metadata={
                'risk_per_trade': risk_per_trade,
                'base_percentage': base_percentage,
                'adjusted_percentage': adjusted_percentage,
                'available_capital': capital
            }
        )
        
        self.position_history.append(result)
        return result

# ============ Volatility Adjusted Sizer ============
class VolatilityAdjustedPositionSizer(BasePositionSizer):
    """Position sizing adjusted for volatility"""
    
    def calculate_position_size(self, 
                               signal: TradingSignal,
                               market_data: pd.DataFrame,
                               current_price: float,
                               stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None) -> PositionSizeResult:
        """Calculate position size adjusted for volatility"""
        
        # Calculate current volatility
        if len(market_data) < self.config.volatility_lookback:
            volatility = 0.02  # Default 2% volatility
        else:
            returns = market_data['close'].pct_change().dropna()
            if len(returns) < self.config.volatility_lookback:
                volatility = 0.02
            else:
                volatility = returns.rolling(window=self.config.volatility_lookback).std().iloc[-1]
                if np.isnan(volatility):
                    volatility = 0.02
        
        # Annualize volatility
        annualized_vol = volatility * np.sqrt(252)
        
        # Calculate base position size based on target volatility
        if annualized_vol > 0:
            # Scale position inversely with volatility to maintain constant risk
            volatility_ratio = self.config.target_volatility / annualized_vol
            volatility_ratio = max(self.config.volatility_floor,
                                 min(volatility_ratio, self.config.volatility_cap))
        else:
            volatility_ratio = 1.0
        
        # Base position percentage
        base_percentage = self.config.max_position_size * volatility_ratio
        
        # Calculate stop loss if not provided
        if stop_loss is None:
            # Set stop loss based on volatility
            stop_loss_distance = volatility * 2  # 2 standard deviations
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = current_price * (1 - stop_loss_distance)
            else:
                stop_loss = current_price * (1 + stop_loss_distance)
        
        # Calculate risk per trade
        risk_per_trade = abs(current_price - stop_loss) / current_price
        
        # Adjust position for risk
        risk_adjusted_percentage = min(base_percentage, 
                                      self.config.max_portfolio_risk / risk_per_trade)
        
        # Further adjust for market conditions
        adjusted_percentage = self.adjust_for_market_conditions(
            risk_adjusted_percentage, market_data, signal
        )
        
        # Calculate position value
        if self.portfolio_state:
            capital = self.portfolio_state.available_capital
        else:
            capital = self.config.current_capital or self.config.initial_capital
        
        position_value = adjusted_percentage * capital
        
        # Calculate risk amount
        risk_amount = position_value * risk_per_trade
        
        # Convert to appropriate unit
        position_size = self._convert_to_unit(adjusted_percentage, position_value, current_price)
        
        # Calculate confidence based on volatility stability
        volatility_stability = self._calculate_volatility_stability(market_data)
        confidence = min(signal.confidence * 0.7 + volatility_stability * 0.3, 1.0)
        
        result = PositionSizeResult(
            position_size=position_size,
            size_unit=self.config.size_unit,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage=risk_per_trade,
            sizing_method=PositionSizingMethod.VOLATILITY_ADJUSTED,
            confidence=confidence,
            metadata={
                'volatility': volatility,
                'annualized_volatility': annualized_vol,
                'volatility_ratio': volatility_ratio,
                'risk_per_trade': risk_per_trade,
                'adjusted_percentage': adjusted_percentage,
                'stop_loss': stop_loss,
                'volatility_stability': volatility_stability
            }
        )
        
        self.position_history.append(result)
        return result
    
    def _calculate_volatility_stability(self, market_data: pd.DataFrame) -> float:
        """Calculate volatility stability score (0-1)"""
        if len(market_data) < 50:
            return 0.5
        
        returns = market_data['close'].pct_change().dropna()
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20).std()
        
        # Remove NaN values
        rolling_vol = rolling_vol.dropna()
        
        if len(rolling_vol) < 20:
            return 0.5
        
        # Calculate coefficient of variation (std/mean)
        if rolling_vol.mean() > 0:
            cv = rolling_vol.std() / rolling_vol.mean()
            # Convert to stability score (lower CV = more stable)
            stability = 1 / (1 + cv)
            return min(stability, 1.0)
        
        return 0.5

# ============ Sharpe Optimized Sizer ============
class SharpeOptimizedPositionSizer(BasePositionSizer):
    """Position sizing optimized for Sharpe ratio"""
    
    def calculate_position_size(self, 
                               signal: TradingSignal,
                               market_data: pd.DataFrame,
                               current_price: float,
                               stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None) -> PositionSizeResult:
        """Calculate position size optimized for Sharpe ratio"""
        
        # Get historical returns for optimization
        if len(market_data) < 50:
            # Insufficient data, use simpler method
            return self._fallback_calculation(signal, market_data, current_price, stop_loss, take_profit)
        
        returns = market_data['close'].pct_change().dropna()
        
        if len(returns) < 30:
            return self._fallback_calculation(signal, market_data, current_price, stop_loss, take_profit)
        
        # Calculate expected return based on signal
        expected_return = self._calculate_expected_return(signal, market_data)
        
        # Calculate portfolio metrics
        portfolio_return = expected_return
        portfolio_volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Risk-free rate
        risk_free_rate = self.config.risk_free_rate
        
        # Calculate Sharpe ratio for different position sizes
        position_sizes = np.linspace(0.01, self.config.max_position_size, 50)
        sharpe_ratios = []
        
        for size in position_sizes:
            # Adjust return and risk for position size
            adjusted_return = portfolio_return * size
            adjusted_risk = portfolio_volatility * size
            
            # Calculate Sharpe ratio
            if adjusted_risk > 0:
                sharpe = (adjusted_return - risk_free_rate) / adjusted_risk
            else:
                sharpe = 0
            
            sharpe_ratios.append(sharpe)
        
        # Find position size with maximum Sharpe ratio
        if sharpe_ratios:
            optimal_idx = np.argmax(sharpe_ratios)
            optimal_percentage = position_sizes[optimal_idx]
            max_sharpe = sharpe_ratios[optimal_idx]
        else:
            optimal_percentage = self.config.max_position_size * 0.5
            max_sharpe = 0
        
        # Adjust for market conditions
        adjusted_percentage = self.adjust_for_market_conditions(
            optimal_percentage, market_data, signal
        )
        
        # Calculate position value
        if self.portfolio_state:
            capital = self.portfolio_state.available_capital
        else:
            capital = self.config.current_capital or self.config.initial_capital
        
        position_value = adjusted_percentage * capital
        
        # Calculate stop loss if not provided
        if stop_loss is None:
            # Set stop loss at 1.5 standard deviations
            stop_loss_distance = returns.std() * 1.5
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = current_price * (1 - stop_loss_distance)
            else:
                stop_loss = current_price * (1 + stop_loss_distance)
        
        # Calculate risk
        risk_per_trade = abs(current_price - stop_loss) / current_price
        risk_amount = position_value * risk_per_trade
        
        # Convert to appropriate unit
        position_size = self._convert_to_unit(adjusted_percentage, position_value, current_price)
        
        # Calculate confidence based on Sharpe ratio
        confidence = min(max_sharpe / 2, 1.0) if max_sharpe > 0 else 0.5
        
        result = PositionSizeResult(
            position_size=position_size,
            size_unit=self.config.size_unit,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage=risk_per_trade,
            sizing_method=PositionSizingMethod.SHARPE_OPTIMIZED,
            confidence=confidence,
            metadata={
                'expected_return': expected_return,
                'portfolio_volatility': portfolio_volatility,
                'optimal_sharpe': max_sharpe,
                'optimal_percentage': optimal_percentage,
                'adjusted_percentage': adjusted_percentage,
                'risk_free_rate': risk_free_rate
            }
        )
        
        self.position_history.append(result)
        return result
    
    def _calculate_expected_return(self, signal: TradingSignal, 
                                 market_data: pd.DataFrame) -> float:
        """Calculate expected return based on signal and market data"""
        if len(market_data) < 20:
            return 0.1  # Default 10% annual return
        
        # Get historical returns
        returns = market_data['close'].pct_change().dropna()
        
        # Base expected return on signal strength and confidence
        base_return = 0.15  # 15% annual base return
        
        # Adjust based on signal
        signal_multiplier = signal.strength * signal.confidence
        
        # Adjust based on market trend
        short_ma = market_data['close'].rolling(window=20).mean().iloc[-1]
        long_ma = market_data['close'].rolling(window=50).mean().iloc[-1]
        
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            trend_multiplier = 1.5 if short_ma > long_ma else 0.8
        else:
            trend_multiplier = 1.5 if short_ma < long_ma else 0.8
        
        expected_return = base_return * signal_multiplier * trend_multiplier
        
        return expected_return
    
    def _fallback_calculation(self, 
                             signal: TradingSignal,
                             market_data: pd.DataFrame,
                             current_price: float,
                             stop_loss: Optional[float] = None,
                             take_profit: Optional[float] = None) -> PositionSizeResult:
        """Fallback calculation when insufficient data"""
        # Use volatility adjusted method as fallback
        fallback_sizer = VolatilityAdjustedPositionSizer(self.config)
        fallback_sizer.portfolio_state = self.portfolio_state
        
        return fallback_sizer.calculate_position_size(
            signal, market_data, current_price, stop_loss, take_profit
        )

# ============ Risk Parity Sizer ============
class RiskParityPositionSizer(BasePositionSizer):
    """Position sizing using risk parity approach"""
    
    def calculate_position_size(self, 
                               signal: TradingSignal,
                               market_data: pd.DataFrame,
                               current_price: float,
                               stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None) -> PositionSizeResult:
        """Calculate position size using risk parity"""
        
        # If we have multiple positions, allocate risk equally
        if self.portfolio_state and self.portfolio_state.open_positions > 0:
            # Calculate target risk per position
            target_risk_per_position = self.config.max_portfolio_risk / (self.portfolio_state.open_positions + 1)
        else:
            target_risk_per_position = self.config.max_portfolio_risk
        
        # Calculate position volatility
        if len(market_data) < 20:
            position_volatility = 0.02  # Default 2%
        else:
            returns = market_data['close'].pct_change().dropna()
            position_volatility = returns.std()
        
        # Calculate position size to achieve target risk contribution
        # In risk parity, each position contributes equally to total risk
        position_percentage = target_risk_per_position / position_volatility
        
        # Adjust for market conditions
        adjusted_percentage = self.adjust_for_market_conditions(
            position_percentage, market_data, signal
        )
        
        # Calculate position value
        if self.portfolio_state:
            capital = self.portfolio_state.available_capital
        else:
            capital = self.config.current_capital or self.config.initial_capital
        
        position_value = adjusted_percentage * capital
        
        # Calculate stop loss if not provided
        if stop_loss is None:
            # Set stop loss at 1 standard deviation
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = current_price * (1 - position_volatility)
            else:
                stop_loss = current_price * (1 + position_volatility)
        
        # Calculate risk
        risk_per_trade = abs(current_price - stop_loss) / current_price
        risk_amount = position_value * risk_per_trade
        
        # Convert to appropriate unit
        position_size = self._convert_to_unit(adjusted_percentage, position_value, current_price)
        
        # Calculate confidence
        confidence = min(signal.confidence * 0.9, 1.0)
        
        result = PositionSizeResult(
            position_size=position_size,
            size_unit=self.config.size_unit,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage=risk_per_trade,
            sizing_method=PositionSizingMethod.RISK_PARITY,
            confidence=confidence,
            metadata={
                'target_risk_per_position': target_risk_per_position,
                'position_volatility': position_volatility,
                'open_positions': self.portfolio_state.open_positions if self.portfolio_state else 0,
                'adjusted_percentage': adjusted_percentage
            }
        )
        
        self.position_history.append(result)
        return result

# ============ Adaptive Position Sizer ============
class AdaptivePositionSizer(BasePositionSizer):
    """Adaptive position sizing that learns from performance"""
    
    def __init__(self, config: Optional[PositionSizingConfig] = None):
        super().__init__(config)
        self.learning_rate = self.config.learning_rate
        self.performance_model = {}
        self.feature_importance = {}
        
    def calculate_position_size(self, 
                               signal: TradingSignal,
                               market_data: pd.DataFrame,
                               current_price: float,
                               stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None) -> PositionSizeResult:
        """Calculate position size using adaptive learning"""
        
        # Extract features for ML model if enabled
        features = self._extract_features(signal, market_data, current_price)
        
        # Get base position size from one of the standard methods
        base_sizer = self._get_base_sizer()
        base_result = base_sizer.calculate_position_size(
            signal, market_data, current_price, stop_loss, take_profit
        )
        
        # Adjust based on learned performance
        if self.config.adapt_to_performance and len(self.performance_history) > 10:
            adjustment = self._calculate_performance_adjustment(features)
            adjusted_percentage = base_result.risk_percentage * adjustment
            
            # Recalculate with adjusted percentage
            if self.portfolio_state:
                capital = self.portfolio_state.available_capital
            else:
                capital = self.config.current_capital or self.config.initial_capital
            
            position_value = adjusted_percentage * capital
            risk_amount = position_value * base_result.risk_percentage
            
            # Convert to appropriate unit
            position_size = self._convert_to_unit(adjusted_percentage, position_value, current_price)
            
            # Update result
            result = PositionSizeResult(
                position_size=position_size,
                size_unit=base_result.size_unit,
                position_value=position_value,
                risk_amount=risk_amount,
                risk_percentage=base_result.risk_percentage,
                sizing_method=PositionSizingMethod.CUSTOM,
                confidence=base_result.confidence * adjustment,
                metadata={
                    **base_result.metadata,
                    'adaptive_adjustment': adjustment,
                    'base_method': base_result.sizing_method.value,
                    'features': features
                }
            )
        else:
            result = base_result
            result.metadata['adaptive_adjustment'] = 1.0
        
        self.position_history.append(result)
        return result
    
    def _get_base_sizer(self) -> BasePositionSizer:
        """Get base position sizer based on configuration"""
        if self.config.sizing_method == PositionSizingMethod.KELLY:
            return KellyPositionSizer(self.config)
        elif self.config.sizing_method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            return VolatilityAdjustedPositionSizer(self.config)
        elif self.config.sizing_method == PositionSizingMethod.SHARPE_OPTIMIZED:
            return SharpeOptimizedPositionSizer(self.config)
        elif self.config.sizing_method == PositionSizingMethod.RISK_PARITY:
            return RiskParityPositionSizer(self.config)
        else:
            return VolatilityAdjustedPositionSizer(self.config)  # Default
    
    def _extract_features(self, signal: TradingSignal, 
                         market_data: pd.DataFrame, 
                         current_price: float) -> Dict[str, float]:
        """Extract features for adaptive learning"""
        features = {
            'signal_strength': signal.strength,
            'signal_confidence': signal.confidence,
            'signal_type_buy': 1 if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else 0,
            'signal_type_sell': 1 if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL] else 0,
            'current_price': current_price
        }
        
        # Add market features if available
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().dropna()
            features['volatility'] = returns.std() if len(returns) > 0 else 0.02
            features['volume_ratio'] = self._calculate_volume_ratio(market_data)
            features['trend_strength'] = self._calculate_trend_strength(market_data)
        
        # Add portfolio features if available
        if self.portfolio_state:
            features['current_drawdown'] = self.portfolio_state.current_drawdown
            features['win_rate'] = self.portfolio_state.win_rate
            features['open_positions'] = self.portfolio_state.open_positions
        
        return features
    
    def _calculate_volume_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate volume ratio"""
        if 'volume' not in market_data.columns or len(market_data) < 20:
            return 1.0
        
        avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1]
        current_volume = market_data['volume'].iloc[-1]
        
        if avg_volume > 0:
            return current_volume / avg_volume
        
        return 1.0
    
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength"""
        if len(market_data) < 50:
            return 0.5
        
        short_ma = market_data['close'].rolling(window=20).mean().iloc[-1]
        long_ma = market_data['close'].rolling(window=50).mean().iloc[-1]
        
        if long_ma > 0:
            return abs(short_ma - long_ma) / long_ma
        
        return 0.5
    
    def _calculate_performance_adjustment(self, features: Dict[str, float]) -> float:
        """Calculate performance-based adjustment"""
        if not self.performance_history:
            return 1.0
        
        # Simple reinforcement learning: increase size after wins, decrease after losses
        recent_performance = self.performance_history[-10:]  # Last 10 trades
        
        if not recent_performance:
            return 1.0
        
        # Calculate win rate in recent trades
        recent_wins = 0
        recent_total = 0
        
        for record in recent_performance:
            outcome = record.get('outcome', {})
            if 'profit' in outcome:
                recent_total += 1
                if outcome['profit'] > 0:
                    recent_wins += 1
        
        if recent_total > 0:
            recent_win_rate = recent_wins / recent_total
        else:
            recent_win_rate = 0.5
        
        # Adjust based on recent performance
        if recent_win_rate > 0.6:
            # Good performance - increase size slightly
            adjustment = 1.0 + (recent_win_rate - 0.6) * 0.5
        elif recent_win_rate < 0.4:
            # Poor performance - decrease size
            adjustment = 1.0 - (0.4 - recent_win_rate) * 0.5
        else:
            # Average performance - no adjustment
            adjustment = 1.0
        
        # Apply learning rate
        adjustment = 1.0 + self.learning_rate * (adjustment - 1.0)
        
        return max(0.5, min(adjustment, 1.5))  # Bound adjustments
    
    def record_performance(self, 
                          position_result: PositionSizeResult,
                          outcome: Dict[str, Any]):
        """Record performance and update learning"""
        super().record_performance(position_result, outcome)
        
        # Update learning rate based on performance
        if 'profit' in outcome:
            profit = outcome['profit']
            
            # Adjust learning rate: increase if winning, decrease if losing
            if profit > 0:
                self.learning_rate = min(self.learning_rate * 1.01, 0.1)
            else:
                self.learning_rate = max(self.learning_rate * 0.99, 0.001)

# ============ Position Size Converter ============
class PositionSizeConverter:
    """Converts position sizes between different units"""
    
    @staticmethod
    def convert(size: float, 
                from_unit: PositionSizeUnit,
                to_unit: PositionSizeUnit,
                current_price: float,
                portfolio_value: float) -> float:
        """Convert position size from one unit to another"""
        
        # First convert to base value (USD)
        if from_unit == PositionSizeUnit.PERCENTAGE:
            base_value = size * portfolio_value
        elif from_unit == PositionSizeUnit.DOLLAR:
            base_value = size
        elif from_unit == PositionSizeUnit.UNITS:
            base_value = size * current_price
        elif from_unit == PositionSizeUnit.BTC:
            base_value = size * current_price
        elif from_unit == PositionSizeUnit.CONTRACTS:
            # Assuming contract size = 1 BTC
            base_value = size * current_price
        else:
            raise ValueError(f"Unknown from_unit: {from_unit}")
        
        # Convert from base value to target unit
        if to_unit == PositionSizeUnit.PERCENTAGE:
            return base_value / portfolio_value if portfolio_value > 0 else 0
        elif to_unit == PositionSizeUnit.DOLLAR:
            return base_value
        elif to_unit == PositionSizeUnit.UNITS:
            return base_value / current_price if current_price > 0 else 0
        elif to_unit == PositionSizeUnit.BTC:
            return base_value / current_price if current_price > 0 else 0
        elif to_unit == PositionSizeUnit.CONTRACTS:
            return base_value / current_price if current_price > 0 else 0
        else:
            raise ValueError(f"Unknown to_unit: {to_unit}")
    
    @staticmethod
    def format_size(size: float, unit: PositionSizeUnit) -> str:
        """Format position size for display"""
        if unit == PositionSizeUnit.PERCENTAGE:
            return f"{size*100:.2f}%"
        elif unit == PositionSizeUnit.DOLLAR:
            return f"${size:,.2f}"
        elif unit == PositionSizeUnit.UNITS:
            return f"{size:.4f} units"
        elif unit == PositionSizeUnit.BTC:
            return f"{size:.6f} BTC"
        elif unit == PositionSizeUnit.CONTRACTS:
            return f"{size:.2f} contracts"
        else:
            return f"{size:.4f}"

# ============ Main Position Sizer ============
class BitcoinPositionSizer:
    """Main position sizer for Bitcoin trading"""
    
    def __init__(self, config: Optional[PositionSizingConfig] = None):
        self.config = config or PositionSizingConfig()
        self.sizer = self._create_sizer()
        self.converter = PositionSizeConverter()
        self.portfolio_state: Optional[PortfolioState] = None
        self.logger = get_logger(__name__)
        
        # History and tracking
        self.sizing_history: List[PositionSizeResult] = []
        self.execution_history: List[Dict[str, Any]] = []
        
    def _create_sizer(self) -> BasePositionSizer:
        """Create appropriate position sizer based on configuration"""
        if self.config.sizing_method == PositionSizingMethod.KELLY:
            return KellyPositionSizer(self.config)
        elif self.config.sizing_method == PositionSizingMethod.FIXED_FRACTIONAL:
            return FixedFractionalPositionSizer(self.config)
        elif self.config.sizing_method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            return VolatilityAdjustedPositionSizer(self.config)
        elif self.config.sizing_method == PositionSizingMethod.SHARPE_OPTIMIZED:
            return SharpeOptimizedPositionSizer(self.config)
        elif self.config.sizing_method == PositionSizingMethod.RISK_PARITY:
            return RiskParityPositionSizer(self.config)
        elif self.config.sizing_method == PositionSizingMethod.ADAPTIVE:
            return AdaptivePositionSizer(self.config)
        else:
            return VolatilityAdjustedPositionSizer(self.config)  # Default
    
    def calculate_position(self,
                          signal: TradingSignal,
                          market_data: pd.DataFrame,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> PositionSizeResult:
        """Calculate position size for a trading signal"""
        
        current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else signal.price
        
        # Update portfolio state in sizer
        if self.portfolio_state:
            self.sizer.update_portfolio_state(self.portfolio_state)
        
        # Calculate position size
        result = self.sizer.calculate_position_size(
            signal, market_data, current_price, stop_loss, take_profit
        )
        
        # Convert to configured unit if different
        if result.size_unit != self.config.size_unit:
            if self.portfolio_state:
                portfolio_value = self.portfolio_state.total_capital
            else:
                portfolio_value = self.config.current_capital or self.config.initial_capital
            
            converted_size = self.converter.convert(
                result.position_size,
                result.size_unit,
                self.config.size_unit,
                current_price,
                portfolio_value
            )
            
            result.position_size = converted_size
            result.size_unit = self.config.size_unit
        
        # Add to history
        self.sizing_history.append(result)
        
        # Keep only recent history
        if len(self.sizing_history) > 100:
            self.sizing_history = self.sizing_history[-100:]
        
        self.logger.info(f"Calculated position size: {self.converter.format_size(result.position_size, result.size_unit)}")
        
        return result
    
    def update_portfolio(self, state: PortfolioState):
        """Update portfolio state"""
        self.portfolio_state = state
        self.sizer.update_portfolio_state(state)
    
    def record_execution(self, 
                        position_result: PositionSizeResult,
                        execution_price: float,
                        fees: float = 0.0):
        """Record position execution"""
        execution_record = {
            'timestamp': datetime.now(),
            'position_result': position_result.to_dict(),
            'execution_price': execution_price,
            'fees': fees,
            'portfolio_state': self.portfolio_state.to_dict() if self.portfolio_state else None
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only recent history
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def record_outcome(self,
                      position_result: PositionSizeResult,
                      exit_price: float,
                      fees: float = 0.0):
        """Record position outcome for performance tracking"""
        if not self.portfolio_state:
            return
        
        # Calculate profit/loss
        position_value = position_result.position_value
        if position_value <= 0:
            return
        
        # For simplicity, assume we traded the full position
        # In reality, you'd track entry price and partial fills
        profit_pct = (exit_price - position_result.metadata.get('entry_price', 0)) / position_result.metadata.get('entry_price', 1)
        profit_amount = position_value * profit_pct - fees
        
        outcome = {
            'profit': profit_amount,
            'profit_percentage': profit_pct,
            'exit_price': exit_price,
            'fees': fees,
            'duration': (datetime.now() - position_result.timestamp).total_seconds() / 3600  # hours
        }
        
        # Record in sizer for learning
        self.sizer.record_performance(position_result, outcome)
    
    def get_sizing_statistics(self) -> Dict[str, Any]:
        """Get statistics on position sizing"""
        if not self.sizing_history:
            return {}
        
        sizes = [r.position_value for r in self.sizing_history if r.position_value > 0]
        risks = [r.risk_amount for r in self.sizing_history]
        confidences = [r.confidence for r in self.sizing_history]
        
        stats = {
            'total_positions': len(self.sizing_history),
            'avg_position_size': np.mean(sizes) if sizes else 0,
            'std_position_size': np.std(sizes) if len(sizes) > 1 else 0,
            'avg_risk_amount': np.mean(risks) if risks else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'min_position_size': min(sizes) if sizes else 0,
            'max_position_size': max(sizes) if sizes else 0,
            'recent_positions': len(self.sizing_history[-10:])
        }
        
        return stats
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        sizer_performance = self.sizer.get_performance_metrics()
        sizing_stats = self.get_sizing_statistics()
        
        report = {
            'sizing_method': self.config.sizing_method.value,
            'risk_approach': self.config.risk_approach.value,
            'portfolio_state': self.portfolio_state.to_dict() if self.portfolio_state else None,
            'sizer_performance': sizer_performance,
            'sizing_statistics': sizing_stats,
            'config': {
                'max_portfolio_risk': self.config.max_portfolio_risk,
                'max_position_size': self.config.max_position_size,
                'target_volatility': self.config.target_volatility,
                'current_capital': self.config.current_capital
            }
        }
        
        return report
    
    def save_state(self, filepath: Path):
        """Save position sizer state"""
        state = {
            'config': self.config.__dict__,
            'portfolio_state': self.portfolio_state.to_dict() if self.portfolio_state else None,
            'sizing_history': [r.to_dict() for r in self.sizing_history],
            'execution_history': self.execution_history,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            self.logger.info(f"Position sizer state saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
    
    def load_state(self, filepath: Path):
        """Load position sizer state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Reload config
            self.config = PositionSizingConfig(**state['config'])
            self.sizer = self._create_sizer()
            
            # Load portfolio state
            if state['portfolio_state']:
                portfolio_dict = state['portfolio_state']
                self.portfolio_state = PortfolioState(
                    total_capital=portfolio_dict['total_capital'],
                    used_capital=portfolio_dict['used_capital'],
                    available_capital=portfolio_dict['available_capital'],
                    open_positions=portfolio_dict['open_positions'],
                    total_risk=portfolio_dict['total_risk'],
                    current_drawdown=portfolio_dict['current_drawdown'],
                    max_drawdown=portfolio_dict['max_drawdown'],
                    win_rate=portfolio_dict['win_rate'],
                    sharpe_ratio=portfolio_dict['sharpe_ratio'],
                    volatility=portfolio_dict['volatility'],
                    timestamp=datetime.fromisoformat(portfolio_dict['timestamp'])
                )
                self.sizer.update_portfolio_state(self.portfolio_state)
            
            # Load history
            self.sizing_history = [
                PositionSizeResult.from_dict(r) for r in state['sizing_history']
            ]
            self.execution_history = state['execution_history']
            
            self.logger.info(f"Position sizer state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")

# ============ Factory Functions ============
def create_position_sizer(config: Optional[Dict] = None) -> BitcoinPositionSizer:
    """Factory function to create a position sizer"""
    if config:
        sizing_config = PositionSizingConfig(**config)
    else:
        sizing_config = PositionSizingConfig()
    
    return BitcoinPositionSizer(sizing_config)

def load_position_config(config_path: Path) -> PositionSizingConfig:
    """Load position sizing configuration from YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return PositionSizingConfig(**config_dict.get('position_sizing', {}))
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {str(e)}")
        return PositionSizingConfig()

# ============ Utility Functions ============
def create_portfolio_state(total_capital: float,
                          used_capital: float,
                          open_positions: int,
                          current_drawdown: float = 0.0,
                          win_rate: float = 0.5,
                          sharpe_ratio: float = 1.0,
                          volatility: float = 0.15) -> PortfolioState:
    """Create a portfolio state object"""
    return PortfolioState(
        total_capital=total_capital,
        used_capital=used_capital,
        available_capital=total_capital - used_capital,
        open_positions=open_positions,
        total_risk=used_capital * 0.02,  # Estimate
        current_drawdown=current_drawdown,
        max_drawdown=max(current_drawdown, 0.1),
        win_rate=win_rate,
        sharpe_ratio=sharpe_ratio,
        volatility=volatility
    )

def calculate_max_position_size(capital: float,
                              risk_per_trade: float,
                              stop_loss_distance: float,
                              max_risk_percentage: float = 0.02) -> float:
    """Calculate maximum position size based on risk parameters"""
    # Risk in dollars
    max_risk_dollars = capital * max_risk_percentage
    
    # Position size that risks max_risk_dollars
    if stop_loss_distance > 0:
        position_size = max_risk_dollars / stop_loss_distance
    else:
        position_size = capital * 0.1  # Default 10%
    
    # Convert to percentage of capital
    position_percentage = position_size / capital if capital > 0 else 0
    
    return min(position_percentage, 0.5)  # Cap at 50%

# ============ Example Usage ============
def example_usage():
    """Example usage of position sizing"""
    print("Position Sizing Example")
    print("=" * 50)
    
    # Create a sample signal
    from core.trading.signal_generator import TradingSignal, SignalType, SignalSource
    from datetime import datetime
    
    signal = TradingSignal(
        timestamp=datetime.now(),
        signal_type=SignalType.BUY,
        strength=0.8,
        confidence=0.75,
        price=45000.0,
        source=SignalSource.TECHNICAL,
        metadata={'indicator': 'RSI', 'rsi_value': 30}
    )
    
    # Create sample market data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    np.random.seed(42)
    
    price = 45000 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
    market_data = pd.DataFrame({
        'open': price * (1 + np.random.randn(100) * 0.001),
        'high': price * (1 + np.abs(np.random.randn(100)) * 0.002),
        'low': price * (1 - np.abs(np.random.randn(100)) * 0.002),
        'close': price,
        'volume': np.random.lognormal(10, 1, 100)
    }, index=dates)
    
    print(f"Created sample signal at ${signal.price:.2f}")
    
    # Create position sizer with different methods
    methods = [
        ('Volatility Adjusted', PositionSizingMethod.VOLATILITY_ADJUSTED),
        ('Kelly Criterion', PositionSizingMethod.KELLY),
        ('Fixed Fractional', PositionSizingMethod.FIXED_FRACTIONAL),
        ('Sharpe Optimized', PositionSizingMethod.SHARPE_OPTIMIZED)
    ]
    
    # Create portfolio state
    portfolio_state = create_portfolio_state(
        total_capital=100000,
        used_capital=20000,
        open_positions=2,
        current_drawdown=0.03,
        win_rate=0.55,
        sharpe_ratio=1.2,
        volatility=0.18
    )
    
    print(f"\nPortfolio State:")
    print(f"  Total Capital: ${portfolio_state.total_capital:,.2f}")
    print(f"  Available Capital: ${portfolio_state.available_capital:,.2f}")
    print(f"  Open Positions: {portfolio_state.open_positions}")
    print(f"  Current Drawdown: {portfolio_state.current_drawdown:.2%}")
    print(f"  Win Rate: {portfolio_state.win_rate:.2%}")
    
    print("\nPosition Sizing Results:")
    print("-" * 50)
    
    for method_name, method in methods:
        config = {
            'sizing_method': method,
            'initial_capital': 100000,
            'risk_approach': RiskApproach.MODERATE
        }
        
        sizer = create_position_sizer(config)
        sizer.update_portfolio(portfolio_state)
        
        # Calculate position size
        result = sizer.calculate_position(
            signal=signal,
            market_data=market_data,
            stop_loss=44000.0,  # 2.2% stop loss
            take_profit=48000.0  # 6.7% take profit
        )
        
        # Format output
        size_str = PositionSizeConverter.format_size(
            result.position_size, 
            result.size_unit
        )
        
        print(f"\n{method_name}:")
        print(f"  Position Size: {size_str}")
        print(f"  Position Value: ${result.position_value:,.2f}")
        print(f"  Risk Amount: ${result.risk_amount:,.2f}")
        print(f"  Risk Percentage: {result.risk_percentage:.2%}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Method: {result.sizing_method.value}")
    
    # Test adaptive sizing
    print("\n" + "=" * 50)
    print("Adaptive Position Sizing Example:")
    print("-" * 50)
    
    adaptive_config = {
        'sizing_method': PositionSizingMethod.ADAPTIVE,
        'risk_approach': RiskApproach.DYNAMIC,
        'initial_capital': 100000,
        'adapt_to_performance': True,
        'learning_rate': 0.05
    }
    
    adaptive_sizer = create_position_sizer(adaptive_config)
    adaptive_sizer.update_portfolio(portfolio_state)
    
    # Simulate some performance history
    for i in range(20):
        mock_result = PositionSizeResult(
            position_size=0.05,
            size_unit=PositionSizeUnit.PERCENTAGE,
            position_value=5000,
            risk_amount=100,
            risk_percentage=0.02,
            sizing_method=PositionSizingMethod.VOLATILITY_ADJUSTED,
            confidence=0.7
        )
        
        # Alternate wins and losses
        profit = 100 if i % 2 == 0 else -50
        outcome = {'profit': profit}
        
        adaptive_sizer.sizer.record_performance(mock_result, outcome)
    
    # Calculate adaptive position
    adaptive_result = adaptive_sizer.calculate_position(
        signal=signal,
        market_data=market_data,
        stop_loss=44000.0,
        take_profit=48000.0
    )
    
    size_str = PositionSizeConverter.format_size(
        adaptive_result.position_size, 
        adaptive_result.size_unit
    )
    
    print(f"\nAdaptive Sizing Result:")
    print(f"  Position Size: {size_str}")
    print(f"  Position Value: ${adaptive_result.position_value:,.2f}")
    print(f"  Adaptive Adjustment: {adaptive_result.metadata.get('adaptive_adjustment', 1.0):.2f}")
    print(f"  Base Method: {adaptive_result.metadata.get('base_method', 'N/A')}")
    
    # Get performance report
    print("\n" + "=" * 50)
    print("Performance Report:")
    print("-" * 50)
    
    report = adaptive_sizer.get_performance_report()
    print(f"Sizing Method: {report['sizing_method']}")
    print(f"Risk Approach: {report['risk_approach']}")
    
    if report['sizer_performance']:
        perf = report['sizer_performance']
        print(f"\nPerformance Metrics:")
        print(f"  Total Trades: {perf['total_trades']}")
        print(f"  Win Rate: {perf['win_rate']:.2%}")
        print(f"  Profit Factor: {perf['profit_factor']:.2f}")
        print(f"  Average Profit: {perf['avg_profit']:.2f}")
    
    return adaptive_sizer, adaptive_result

# ============ Main Execution ============
def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bitcoin Trading AI - Position Sizing')
    parser.add_argument('--signal', type=str, required=True,
                       help='Signal file path (JSON)')
    parser.add_argument('--market_data', type=str, required=True,
                       help='Market data file path')
    parser.add_argument('--config', type=str, default='config/position_sizing.yaml',
                       help='Position sizing configuration file')
    parser.add_argument('--portfolio', type=str,
                       help='Portfolio state file path (JSON)')
    parser.add_argument('--method', type=str, 
                       choices=['kelly', 'fixed_fractional', 'volatility_adjusted', 
                                'sharpe_optimized', 'risk_parity', 'adaptive'],
                       help='Position sizing method (overrides config)')
    parser.add_argument('--output', type=str,
                       help='Output directory for results')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with synthetic data')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running in test mode with synthetic data...")
        sizer, result = example_usage()
        return
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if config_path.exists():
            sizing_config = load_position_config(config_path)
        else:
            sizing_config = PositionSizingConfig()
            logger.info(f"Using default configuration, config file not found: {config_path}")
        
        # Override sizing method if specified
        if args.method:
            sizing_config.sizing_method = PositionSizingMethod(args.method)
        
        # Load signal
        signal_path = Path(args.signal)
        if not signal_path.exists():
            raise FileNotFoundError(f"Signal file not found: {signal_path}")
        
        with open(signal_path, 'r') as f:
            signal_data = json.load(f)
        
        # Convert to TradingSignal object
        from core.trading.signal_generator import TradingSignal
        signal = TradingSignal.from_dict(signal_data)
        
        # Load market data
        data_path = Path(args.market_data)
        if not data_path.exists():
            raise FileNotFoundError(f"Market data file not found: {data_path}")
        
        if data_path.suffix == '.parquet':
            market_data = pd.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            market_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        print(f"Loaded signal: {signal.signal_type.value} at ${signal.price:.2f}")
        print(f"Loaded market data with shape: {market_data.shape}")
        print(f"Current price: ${market_data['close'].iloc[-1]:.2f}")
        
        # Load portfolio state if provided
        portfolio_state = None
        if args.portfolio:
            portfolio_path = Path(args.portfolio)
            if portfolio_path.exists():
                with open(portfolio_path, 'r') as f:
                    portfolio_data = json.load(f)
                
                portfolio_state = PortfolioState(
                    total_capital=portfolio_data['total_capital'],
                    used_capital=portfolio_data['used_capital'],
                    available_capital=portfolio_data['available_capital'],
                    open_positions=portfolio_data['open_positions'],
                    total_risk=portfolio_data['total_risk'],
                    current_drawdown=portfolio_data['current_drawdown'],
                    max_drawdown=portfolio_data['max_drawdown'],
                    win_rate=portfolio_data['win_rate'],
                    sharpe_ratio=portfolio_data['sharpe_ratio'],
                    volatility=portfolio_data['volatility'],
                    timestamp=datetime.fromisoformat(portfolio_data['timestamp'])
                )
        
        # Create position sizer
        sizer = create_position_sizer(sizing_config.__dict__)
        
        if portfolio_state:
            sizer.update_portfolio(portfolio_state)
            print(f"\nPortfolio State:")
            print(f"  Total Capital: ${portfolio_state.total_capital:,.2f}")
            print(f"  Available Capital: ${portfolio_state.available_capital:,.2f}")
            print(f"  Open Positions: {portfolio_state.open_positions}")
            print(f"  Current Drawdown: {portfolio_state.current_drawdown:.2%}")
        
        # Calculate position size
        print(f"\nCalculating position size using {sizing_config.sizing_method.value} method...")
        
        result = sizer.calculate_position(
            signal=signal,
            market_data=market_data,
            stop_loss=signal.price * 0.98,  # 2% stop loss
            take_profit=signal.price * 1.06  # 6% take profit
        )
        
        # Display results
        print("\n" + "="*50)
        print("POSITION SIZING RESULTS")
        print("="*50)
        
        size_str = PositionSizeConverter.format_size(
            result.position_size, 
            result.size_unit
        )
        
        print(f"\nSignal: {signal.signal_type.value.upper()} at ${signal.price:.2f}")
        print(f"Sizing Method: {result.sizing_method.value}")
        print(f"Risk Approach: {sizing_config.risk_approach.value}")
        print(f"\nPosition Details:")
        print(f"  Size: {size_str}")
        print(f"  Value: ${result.position_value:,.2f}")
        print(f"  Risk Amount: ${result.risk_amount:,.2f}")
        print(f"  Risk Percentage: {result.risk_percentage:.2%}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        # Show metadata if available
        if result.metadata:
            print(f"\nCalculation Details:")
            for key, value in result.metadata.items():
                if isinstance(value, float):
                    if 'rate' in key.lower() or 'ratio' in key.lower() or 'percentage' in key.lower():
                        print(f"  {key}: {value:.4f}")
                    elif 'price' in key.lower() or 'value' in key.lower() or 'amount' in key.lower():
                        print(f"  {key}: ${value:,.2f}")
                    else:
                        print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # Get performance report
        report = sizer.get_performance_report()
        if report['sizer_performance']:
            perf = report['sizer_performance']
            if perf['total_trades'] > 0:
                print(f"\nPerformance History:")
                print(f"  Total Trades: {perf['total_trades']}")
                print(f"  Win Rate: {perf['win_rate']:.2%}")
                print(f"  Profit Factor: {perf['profit_factor']:.2f}")
                print(f"  Average Profit: ${perf['avg_profit']:.2f}")
        
        # Save results if output directory specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save position result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = output_dir / f"position_result_{timestamp}.json"
            
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            print(f"\nPosition result saved to: {result_file}")
            
            # Save full report
            report_file = output_dir / f"position_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"Full report saved to: {report_file}")
            
            # Save sizer state
            state_file = output_dir / f"position_sizer_state_{timestamp}.json"
            sizer.save_state(state_file)
            
            print(f"Sizer state saved to: {state_file}")
        
        print("\n" + "="*50)
        print("Position sizing completed successfully")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()