"""
Risk Management and Trading Controls module for Bitcoin Trading Application.
Provides position sizing, risk limits, and trading controls for safe automated trading.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import warnings
from decimal import Decimal, ROUND_DOWN
import threading
import time
from collections import deque, defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')

# Import project modules
from logger import get_logger
from cache import TradingCache

logger = get_logger(__name__)

class ControlType(Enum):
    """Types of trading controls."""
    POSITION_SIZE = "position_size"
    RISK_LIMIT = "risk_limit"
    VOLUME_LIMIT = "volume_limit"
    VELOCITY_LIMIT = "velocity_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    CUSTOM = "custom"

class RiskLevel(Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"

class OrderStatus(Enum):
    """Order status for control validation."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"

@dataclass
class ControlConfig:
    """Configuration for trading controls."""
    # Position sizing
    position_sizing_method: str = "kelly"  # kelly, fixed, volatility, percent
    max_position_size_pct: float = 0.10  # Maximum 10% of capital per trade
    min_position_size: float = 0.001  # Minimum position size in BTC
    
    # Risk limits
    max_daily_loss_pct: float = 0.05  # Maximum 5% daily loss
    max_consecutive_losses: int = 5  # Maximum consecutive losing trades
    max_drawdown_pct: float = 0.20  # Maximum 20% drawdown
    
    # Volume limits
    max_daily_volume: float = 100.0  # Maximum daily volume in BTC
    max_order_volume: float = 10.0  # Maximum single order volume in BTC
    
    # Velocity limits
    max_trades_per_hour: int = 10  # Maximum trades per hour
    max_orders_per_minute: int = 5  # Maximum orders per minute
    
    # Concentration limits
    max_symbol_exposure_pct: float = 0.30  # Maximum 30% exposure to single symbol
    max_correlation_exposure: float = 0.70  # Maximum correlation exposure
    
    # Risk level
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    # Circuit breakers
    enable_circuit_breakers: bool = True
    volatility_break_threshold: float = 0.10  # 10% price move triggers break
    volume_spike_threshold: float = 5.0  # 5x average volume
    
    # Slippage control
    max_slippage_pct: float = 0.01  # Maximum 1% slippage
    min_liquidity_requirement: float = 10000.0  # Minimum $10k liquidity
    
    # Time controls
    trading_hours_start: str = "00:00"  # 24-hour format
    trading_hours_end: str = "23:59"
    exclude_weekends: bool = True
    exclude_holidays: bool = True
    
    # Advanced controls
    enable_stress_testing: bool = True
    enable_scenario_analysis: bool = True
    enable_real_time_monitoring: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        # Ensure percentages are between 0 and 1
        for field_name in ['max_position_size_pct', 'max_daily_loss_pct', 
                          'max_drawdown_pct', 'max_symbol_exposure_pct',
                          'max_slippage_pct']:
            value = getattr(self, field_name)
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")

@dataclass
class ControlState:
    """Current state of trading controls."""
    # Daily tracking
    daily_pnl: float = 0.0
    daily_volume: float = 0.0
    daily_trades: int = 0
    
    # Hourly tracking
    hourly_trades: int = 0
    last_hour_reset: datetime = field(default_factory=datetime.now)
    
    # Minute tracking
    minute_orders: int = 0
    last_minute_reset: datetime = field(default_factory=datetime.now)
    
    # Loss tracking
    consecutive_losses: int = 0
    current_drawdown: float = 0.0
    peak_capital: float = 0.0
    
    # Position tracking
    current_positions: Dict[str, float] = field(default_factory=dict)
    symbol_exposure: Dict[str, float] = field(default_factory=dict)
    
    # Circuit breaker state
    circuit_breaker_active: bool = False
    circuit_breaker_until: Optional[datetime] = None
    
    # Risk metrics
    current_volatility: float = 0.0
    current_correlation: float = 0.0
    
    # Control flags
    trading_enabled: bool = True
    warnings_issued: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)

@dataclass
class ControlResult:
    """Result of control validation."""
    status: OrderStatus
    order_id: str
    symbol: str
    requested_size: float
    approved_size: float
    reason: str
    warnings: List[str]
    violations: List[str]
    risk_score: float
    metadata: Dict[str, Any]

@dataclass
class PositionSizingResult:
    """Result of position sizing calculation."""
    recommended_size: float
    max_allowed_size: float
    risk_per_trade: float
    position_value: float
    sizing_method: str
    confidence: float
    warnings: List[str]

class TradingControls:
    """
    Comprehensive trading controls and risk management system.
    """
    
    def __init__(self, config: ControlConfig = None):
        """
        Initialize trading controls.
        
        Args:
            config: Control configuration
        """
        self.config = config or ControlConfig()
        self.state = ControlState()
        self.cache = TradingCache(cache_type="memory", max_size=1000)
        
        # Initialize risk parameters based on risk level
        self._apply_risk_level()
        
        # Historical data for calculations
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Control validators
        self.validators = self._setup_validators()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_controls, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Initialized TradingControls with risk level: {self.config.risk_level.value}")
    
    def _apply_risk_level(self):
        """Apply risk level to configuration."""
        risk_adjustments = {
            RiskLevel.CONSERVATIVE: {
                'max_position_size_pct': 0.05,
                'max_daily_loss_pct': 0.02,
                'max_drawdown_pct': 0.10,
                'max_trades_per_hour': 5,
                'volatility_break_threshold': 0.05
            },
            RiskLevel.MODERATE: {
                'max_position_size_pct': 0.10,
                'max_daily_loss_pct': 0.05,
                'max_drawdown_pct': 0.20,
                'max_trades_per_hour': 10,
                'volatility_break_threshold': 0.10
            },
            RiskLevel.AGGRESSIVE: {
                'max_position_size_pct': 0.20,
                'max_daily_loss_pct': 0.10,
                'max_drawdown_pct': 0.30,
                'max_trades_per_hour': 20,
                'volatility_break_threshold': 0.15
            },
            RiskLevel.VERY_AGGRESSIVE: {
                'max_position_size_pct': 0.30,
                'max_daily_loss_pct': 0.20,
                'max_drawdown_pct': 0.40,
                'max_trades_per_hour': 30,
                'volatility_break_threshold': 0.20
            }
        }
        
        adjustments = risk_adjustments.get(self.config.risk_level, {})
        for key, value in adjustments.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _setup_validators(self) -> Dict[str, Callable]:
        """Setup control validation functions."""
        return {
            'position_size': self._validate_position_size,
            'daily_loss': self._validate_daily_loss,
            'consecutive_losses': self._validate_consecutive_losses,
            'drawdown': self._validate_drawdown,
            'daily_volume': self._validate_daily_volume,
            'order_volume': self._validate_order_volume,
            'trade_velocity': self._validate_trade_velocity,
            'order_velocity': self._validate_order_velocity,
            'symbol_exposure': self._validate_symbol_exposure,
            'correlation': self._validate_correlation,
            'circuit_breaker': self._validate_circuit_breaker,
            'slippage': self._validate_slippage,
            'liquidity': self._validate_liquidity,
            'trading_hours': self._validate_trading_hours,
            'market_conditions': self._validate_market_conditions
        }
    
    def update_market_data(self, symbol: str, price: float, volume: float):
        """
        Update market data for controls.
        
        Args:
            symbol: Trading symbol
            price: Current price
            volume: Current volume
        """
        with self.lock:
            self.price_history[symbol].append(price)
            self.volume_history[symbol].append(volume)
            
            # Update volatility
            if len(self.price_history[symbol]) > 20:
                prices = list(self.price_history[symbol])
                returns = np.diff(prices) / prices[:-1]
                self.state.current_volatility = np.std(returns) if len(returns) > 0 else 0.0
    
    def update_portfolio_state(self, capital: float, positions: Dict[str, float]):
        """
        Update portfolio state for controls.
        
        Args:
            capital: Current capital
            positions: Current positions
        """
        with self.lock:
            # Update peak capital for drawdown calculation
            if capital > self.state.peak_capital:
                self.state.peak_capital = capital
            
            # Calculate current drawdown
            if self.state.peak_capital > 0:
                self.state.current_drawdown = (self.state.peak_capital - capital) / self.state.peak_capital
            
            # Update positions
            self.state.current_positions = positions.copy()
            
            # Calculate symbol exposure
            total_value = capital + sum(abs(size * self._get_current_price(sym)) 
                                      for sym, size in positions.items())
            
            for symbol, size in positions.items():
                price = self._get_current_price(symbol)
                position_value = abs(size * price)
                exposure = position_value / total_value if total_value > 0 else 0
                self.state.symbol_exposure[symbol] = exposure
    
    def update_trade_result(self, trade_result: Dict[str, Any]):
        """
        Update controls with trade result.
        
        Args:
            trade_result: Dictionary with trade information
        """
        with self.lock:
            # Update daily P&L
            pnl = trade_result.get('pnl', 0)
            self.state.daily_pnl += pnl
            
            # Update daily volume
            volume = trade_result.get('volume', 0)
            self.state.daily_volume += volume
            
            # Update trade counts
            self.state.daily_trades += 1
            self.state.hourly_trades += 1
            self.state.minute_orders += 1
            
            # Update consecutive losses
            if pnl < 0:
                self.state.consecutive_losses += 1
            else:
                self.state.consecutive_losses = 0
            
            # Check for circuit breaker triggers
            if abs(pnl) > self.config.max_daily_loss_pct * self.state.peak_capital:
                warning = f"Large trade P&L: {pnl:.2f}"
                self.state.warnings_issued.append(warning)
                logger.warning(warning)
    
    def calculate_position_size(self,
                              symbol: str,
                              entry_price: float,
                              stop_loss: Optional[float] = None,
                              confidence: float = 0.5,
                              capital: Optional[float] = None) -> PositionSizingResult:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price (optional)
            confidence: Trade confidence (0-1)
            capital: Available capital (uses current if None)
        
        Returns:
            PositionSizingResult: Position sizing recommendations
        """
        with self.lock:
            if capital is None:
                # Estimate capital from positions
                capital = self.state.peak_capital
            
            method = self.config.position_sizing_method
            warnings = []
            
            if method == "kelly":
                size = self._kelly_position_size(symbol, entry_price, stop_loss, confidence, capital)
            elif method == "fixed":
                size = self._fixed_position_size(capital)
            elif method == "volatility":
                size = self._volatility_position_size(symbol, capital)
            elif method == "percent":
                size = self._percent_position_size(capital)
            else:
                size = self._default_position_size(capital)
                warnings.append(f"Unknown sizing method: {method}, using default")
            
            # Apply maximum position size limit
            max_size = self._calculate_max_position_size(symbol, entry_price, capital)
            size = min(size, max_size)
            
            # Apply minimum position size
            size = max(size, self.config.min_position_size)
            
            # Calculate risk per trade
            if stop_loss:
                risk_per_trade = abs(entry_price - stop_loss) * size
            else:
                risk_per_trade = 0.0
            
            position_value = size * entry_price
            
            return PositionSizingResult(
                recommended_size=size,
                max_allowed_size=max_size,
                risk_per_trade=risk_per_trade,
                position_value=position_value,
                sizing_method=method,
                confidence=confidence,
                warnings=warnings
            )
    
    def _kelly_position_size(self,
                           symbol: str,
                           entry_price: float,
                           stop_loss: Optional[float],
                           confidence: float,
                           capital: float) -> float:
        """Calculate position size using Kelly Criterion."""
        # Simplified Kelly calculation
        win_probability = confidence
        win_loss_ratio = 2.0  # Default 2:1 reward:risk
        
        if stop_loss:
            # Calculate actual reward:risk ratio
            take_profit = entry_price * (1 + self.config.max_position_size_pct)
            potential_profit = take_profit - entry_price
            potential_loss = entry_price - stop_loss
            win_loss_ratio = potential_profit / potential_loss if potential_loss > 0 else 2.0
        
        # Kelly fraction
        kelly_f = win_probability - ((1 - win_probability) / win_loss_ratio)
        
        # Conservative Kelly (half-Kelly)
        kelly_f = kelly_f * 0.5
        
        # Position size in BTC
        position_value = capital * kelly_f * self.config.max_position_size_pct
        size = position_value / entry_price
        
        return size
    
    def _fixed_position_size(self, capital: float) -> float:
        """Calculate fixed position size."""
        position_value = capital * self.config.max_position_size_pct
        # Use average price for size calculation
        avg_price = 50000  # BTC average price
        return position_value / avg_price
    
    def _volatility_position_size(self, symbol: str, capital: float) -> float:
        """Calculate position size based on volatility."""
        if len(self.price_history[symbol]) < 20:
            return self._default_position_size(capital)
        
        # Get recent volatility
        prices = list(self.price_history[symbol])[-20:]
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.02
        
        # Adjust position size inversely to volatility
        volatility_scaling = 0.02 / max(volatility, 0.01)  # Scale relative to 2% volatility
        
        position_value = capital * self.config.max_position_size_pct * volatility_scaling
        avg_price = prices[-1] if prices else 50000
        
        return position_value / avg_price
    
    def _percent_position_size(self, capital: float) -> float:
        """Calculate position size as percentage of capital."""
        position_value = capital * self.config.max_position_size_pct
        avg_price = 50000  # BTC average price
        return position_value / avg_price
    
    def _default_position_size(self, capital: float) -> float:
        """Default position size calculation."""
        return self._percent_position_size(capital)
    
    def _calculate_max_position_size(self, symbol: str, price: float, capital: float) -> float:
        """Calculate maximum allowed position size."""
        # Maximum by percentage of capital
        max_by_capital = capital * self.config.max_position_size_pct / price
        
        # Maximum by symbol exposure
        current_exposure = self.state.symbol_exposure.get(symbol, 0)
        exposure_available = max(0, self.config.max_symbol_exposure_pct - current_exposure)
        max_by_exposure = capital * exposure_available / price
        
        # Maximum by order volume limit
        max_by_volume = self.config.max_order_volume
        
        # Take the minimum of all limits
        return min(max_by_capital, max_by_exposure, max_by_volume)
    
    def validate_order(self,
                      order_id: str,
                      symbol: str,
                      order_type: str,
                      side: str,
                      size: float,
                      price: float,
                      metadata: Dict[str, Any] = None) -> ControlResult:
        """
        Validate order against all controls.
        
        Args:
            order_id: Order identifier
            symbol: Trading symbol
            order_type: Type of order
            side: Buy or sell
            size: Order size
            price: Order price
            metadata: Additional order metadata
        
        Returns:
            ControlResult: Validation result
        """
        with self.lock:
            metadata = metadata or {}
            warnings = []
            violations = []
            approved_size = size
            
            # Check if trading is enabled
            if not self.state.trading_enabled:
                return ControlResult(
                    status=OrderStatus.REJECTED,
                    order_id=order_id,
                    symbol=symbol,
                    requested_size=size,
                    approved_size=0,
                    reason="Trading disabled by controls",
                    warnings=warnings,
                    violations=violations,
                    risk_score=1.0,
                    metadata=metadata
                )
            
            # Run all validators
            for validator_name, validator in self.validators.items():
                try:
                    result = validator(symbol, size, price, side, metadata)
                    if result['status'] == 'warning':
                        warnings.append(result['message'])
                    elif result['status'] == 'violation':
                        violations.append(result['message'])
                        if result.get('reject_order', False):
                            return ControlResult(
                                status=OrderStatus.REJECTED,
                                order_id=order_id,
                                symbol=symbol,
                                requested_size=size,
                                approved_size=0,
                                reason=f"Control violation: {result['message']}",
                                warnings=warnings,
                                violations=violations,
                                risk_score=self._calculate_risk_score(violations),
                                metadata=metadata
                            )
                        elif result.get('reduce_size', False) and 'reduced_size' in result:
                            approved_size = min(approved_size, result['reduced_size'])
                except Exception as e:
                    logger.error(f"Validator {validator_name} failed: {e}")
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(violations, warnings)
            
            # Determine final status
            if violations:
                status = OrderStatus.REJECTED
                approved_size = 0
                reason = f"Order rejected due to {len(violations)} control violations"
            elif approved_size < size:
                status = OrderStatus.MODIFIED
                reason = f"Order size reduced from {size:.4f} to {approved_size:.4f}"
            else:
                status = OrderStatus.APPROVED
                reason = "Order approved"
            
            # Add warnings to state
            for warning in warnings:
                self.state.warnings_issued.append(warning)
            
            # Add violations to state
            for violation in violations:
                self.state.violations.append(violation)
            
            result = ControlResult(
                status=status,
                order_id=order_id,
                symbol=symbol,
                requested_size=size,
                approved_size=approved_size,
                reason=reason,
                warnings=warnings,
                violations=violations,
                risk_score=risk_score,
                metadata=metadata
            )
            
            logger.info(f"Order validation: {status.value} for {order_id} - {reason}")
            return result
    
    def _validate_position_size(self, symbol: str, size: float, price: float, 
                              side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate position size limits."""
        # Get current capital estimate
        capital = self.state.peak_capital
        
        # Calculate maximum allowed size
        max_size = self._calculate_max_position_size(symbol, price, capital)
        
        if size > max_size:
            return {
                'status': 'violation',
                'message': f"Position size {size:.4f} exceeds maximum {max_size:.4f}",
                'reduce_size': True,
                'reduced_size': max_size
            }
        
        if size < self.config.min_position_size:
            return {
                'status': 'violation',
                'message': f"Position size {size:.4f} below minimum {self.config.min_position_size:.4f}",
                'reject_order': True
            }
        
        return {'status': 'ok'}
    
    def _validate_daily_loss(self, symbol: str, size: float, price: float,
                           side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate daily loss limits."""
        daily_loss_pct = abs(self.state.daily_pnl) / self.state.peak_capital if self.state.peak_capital > 0 else 0
        
        if daily_loss_pct > self.config.max_daily_loss_pct:
            return {
                'status': 'violation',
                'message': f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.config.max_daily_loss_pct:.2%}",
                'reject_order': True
            }
        
        # Check if this trade could exceed daily loss limit
        potential_loss = size * price * 0.01  # Assume 1% loss for validation
        potential_daily_loss = (abs(self.state.daily_pnl) + potential_loss) / self.state.peak_capital
        
        if potential_daily_loss > self.config.max_daily_loss_pct:
            warning_msg = f"Potential trade could exceed daily loss limit"
            return {
                'status': 'warning',
                'message': warning_msg
            }
        
        return {'status': 'ok'}
    
    def _validate_consecutive_losses(self, symbol: str, size: float, price: float,
                                   side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consecutive losses limit."""
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            return {
                'status': 'violation',
                'message': f"Consecutive losses {self.state.consecutive_losses} exceeds limit {self.config.max_consecutive_losses}",
                'reject_order': True
            }
        
        return {'status': 'ok'}
    
    def _validate_drawdown(self, symbol: str, size: float, price: float,
                         side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate drawdown limits."""
        if self.state.current_drawdown > self.config.max_drawdown_pct:
            return {
                'status': 'violation',
                'message': f"Current drawdown {self.state.current_drawdown:.2%} exceeds limit {self.config.max_drawdown_pct:.2%}",
                'reject_order': True
            }
        
        return {'status': 'ok'}
    
    def _validate_daily_volume(self, symbol: str, size: float, price: float,
                             side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate daily volume limits."""
        trade_volume = size * price
        total_daily_volume = self.state.daily_volume + trade_volume
        
        if total_daily_volume > self.config.max_daily_volume:
            return {
                'status': 'violation',
                'message': f"Daily volume would exceed limit {self.config.max_daily_volume:.2f} BTC",
                'reduce_size': True,
                'reduced_size': max(0, (self.config.max_daily_volume - self.state.daily_volume) / price)
            }
        
        return {'status': 'ok'}
    
    def _validate_order_volume(self, symbol: str, size: float, price: float,
                             side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate single order volume limits."""
        if size > self.config.max_order_volume:
            return {
                'status': 'violation',
                'message': f"Order volume {size:.4f} exceeds maximum {self.config.max_order_volume:.4f}",
                'reduce_size': True,
                'reduced_size': self.config.max_order_volume
            }
        
        return {'status': 'ok'}
    
    def _validate_trade_velocity(self, symbol: str, size: float, price: float,
                               side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade velocity limits."""
        # Reset hourly counter if needed
        current_time = datetime.now()
        if (current_time - self.state.last_hour_reset).seconds >= 3600:
            self.state.hourly_trades = 0
            self.state.last_hour_reset = current_time
        
        if self.state.hourly_trades >= self.config.max_trades_per_hour:
            return {
                'status': 'violation',
                'message': f"Hourly trades {self.state.hourly_trades} exceeds limit {self.config.max_trades_per_hour}",
                'reject_order': True
            }
        
        return {'status': 'ok'}
    
    def _validate_order_velocity(self, symbol: str, size: float, price: float,
                               side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate order velocity limits."""
        # Reset minute counter if needed
        current_time = datetime.now()
        if (current_time - self.state.last_minute_reset).seconds >= 60:
            self.state.minute_orders = 0
            self.state.last_minute_reset = current_time
        
        if self.state.minute_orders >= self.config.max_orders_per_minute:
            return {
                'status': 'violation',
                'message': f"Minute orders {self.state.minute_orders} exceeds limit {self.config.max_orders_per_minute}",
                'reject_order': True
            }
        
        return {'status': 'ok'}
    
    def _validate_symbol_exposure(self, symbol: str, size: float, price: float,
                                side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate symbol concentration limits."""
        current_exposure = self.state.symbol_exposure.get(symbol, 0)
        trade_value = size * price
        total_capital = self.state.peak_capital
        
        if total_capital > 0:
            additional_exposure = trade_value / total_capital
            new_exposure = current_exposure + additional_exposure
            
            if new_exposure > self.config.max_symbol_exposure_pct:
                max_additional = max(0, self.config.max_symbol_exposure_pct - current_exposure)
                max_size = max_additional * total_capital / price
                
                return {
                    'status': 'violation',
                    'message': f"Symbol exposure would exceed {self.config.max_symbol_exposure_pct:.2%} limit",
                    'reduce_size': True,
                    'reduced_size': max_size
                }
        
        return {'status': 'ok'}
    
    def _validate_correlation(self, symbol: str, size: float, price: float,
                            side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate correlation exposure limits."""
        # Simplified correlation check
        # In production, this would use actual correlation matrix
        if self.state.current_correlation > self.config.max_correlation_exposure:
            return {
                'status': 'warning',
                'message': f"High correlation exposure detected: {self.state.current_correlation:.2f}"
            }
        
        return {'status': 'ok'}
    
    def _validate_circuit_breaker(self, symbol: str, size: float, price: float,
                                side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate circuit breaker conditions."""
        if self.state.circuit_breaker_active:
            if self.state.circuit_breaker_until and datetime.now() < self.state.circuit_breaker_until:
                return {
                    'status': 'violation',
                    'message': "Circuit breaker active - trading temporarily suspended",
                    'reject_order': True
                }
            else:
                # Circuit breaker expired
                self.state.circuit_breaker_active = False
                self.state.circuit_breaker_until = None
        
        # Check volatility break
        if self.state.current_volatility > self.config.volatility_break_threshold:
            self._activate_circuit_breaker(
                reason=f"High volatility: {self.state.current_volatility:.2%}",
                duration_minutes=5
            )
            return {
                'status': 'violation',
                'message': f"High volatility triggered circuit breaker",
                'reject_order': True
            }
        
        # Check volume spike
        if len(self.volume_history[symbol]) > 10:
            recent_volume = list(self.volume_history[symbol])[-10:]
            avg_volume = np.mean(recent_volume[:-1])
            current_volume = recent_volume[-1]
            
            if avg_volume > 0 and current_volume / avg_volume > self.config.volume_spike_threshold:
                self._activate_circuit_breaker(
                    reason=f"Volume spike: {current_volume/avg_volume:.1f}x average",
                    duration_minutes=2
                )
                return {
                    'status': 'violation',
                    'message': "Volume spike triggered circuit breaker",
                    'reject_order': True
                }
        
        return {'status': 'ok'}
    
    def _validate_slippage(self, symbol: str, size: float, price: float,
                          side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate slippage limits."""
        # Estimate slippage based on order size and liquidity
        # This is a simplified estimation
        estimated_slippage = size / 1000  # Simplified slippage model
        
        if estimated_slippage > self.config.max_slippage_pct:
            return {
                'status': 'warning',
                'message': f"Estimated slippage {estimated_slippage:.2%} exceeds limit {self.config.max_slippage_pct:.2%}"
            }
        
        return {'status': 'ok'}
    
    def _validate_liquidity(self, symbol: str, size: float, price: float,
                          side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate liquidity requirements."""
        order_value = size * price
        
        if order_value > self.config.min_liquidity_requirement:
            # Check if we have enough liquidity data
            if len(self.volume_history[symbol]) > 0:
                avg_volume = np.mean(list(self.volume_history[symbol]))
                if order_value > avg_volume * 0.1:  # Don't exceed 10% of average volume
                    return {
                        'status': 'warning',
                        'message': f"Order size {order_value:.0f} may exceed available liquidity"
                    }
        
        return {'status': 'ok'}
    
    def _validate_trading_hours(self, symbol: str, size: float, price: float,
                              side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trading hours."""
        current_time = datetime.now()
        
        # Check weekends
        if self.config.exclude_weekends and current_time.weekday() >= 5:
            return {
                'status': 'violation',
                'message': "Trading not allowed on weekends",
                'reject_order': True
            }
        
        # Check holidays (simplified)
        if self.config.exclude_holidays and self._is_holiday(current_time):
            return {
                'status': 'violation',
                'message': "Trading not allowed on holiday",
                'reject_order': True
            }
        
        # Check trading hours
        current_hour = current_time.hour + current_time.minute / 60
        start_hour = float(self.config.trading_hours_start.replace(':', '.'))
        end_hour = float(self.config.trading_hours_end.replace(':', '.'))
        
        if not (start_hour <= current_hour <= end_hour):
            return {
                'status': 'violation',
                'message': f"Outside trading hours ({self.config.trading_hours_start}-{self.config.trading_hours_end})",
                'reject_order': True
            }
        
        return {'status': 'ok'}
    
    def _validate_market_conditions(self, symbol: str, size: float, price: float,
                                  side: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate overall market conditions."""
        warnings = []
        
        # Check for extreme market conditions
        if self.state.current_volatility > 0.15:
            warnings.append("Extreme market volatility detected")
        
        # Check for news events (simplified)
        if self._has_recent_news(symbol):
            warnings.append("Recent news events may affect market")
        
        if warnings:
            return {
                'status': 'warning',
                'message': "; ".join(warnings)
            }
        
        return {'status': 'ok'}
    
    def _activate_circuit_breaker(self, reason: str, duration_minutes: int = 5):
        """Activate circuit breaker."""
        self.state.circuit_breaker_active = True
        self.state.circuit_breaker_until = datetime.now() + timedelta(minutes=duration_minutes)
        
        logger.warning(f"Circuit breaker activated: {reason}. Duration: {duration_minutes} minutes")
        
        # Notify monitoring system
        self.state.warnings_issued.append(f"Circuit breaker: {reason}")
    
    def _calculate_risk_score(self, violations: List[str], warnings: List[str] = None) -> float:
        """Calculate risk score for order."""
        warnings = warnings or []
        
        # Base score
        score = 1.0
        
        # Deduct for violations
        score -= len(violations) * 0.2
        
        # Deduct for warnings
        score -= len(warnings) * 0.05
        
        # Consider market conditions
        if self.state.current_volatility > 0.1:
            score -= 0.1
        
        # Consider drawdown
        if self.state.current_drawdown > 0.1:
            score -= 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        if self.price_history[symbol]:
            return self.price_history[symbol][-1]
        return 50000  # Default BTC price
    
    def _is_holiday(self, date: datetime) -> bool:
        """Check if date is a holiday (simplified)."""
        # Major US holidays (simplified)
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25), # Christmas
        ]
        
        return (date.month, date.day) in holidays
    
    def _has_recent_news(self, symbol: str) -> bool:
        """Check for recent news (simplified - would integrate with news API)."""
        # This would typically connect to a news API
        # For now, return False
        return False
    
    def _monitor_controls(self):
        """Monitor controls in background thread."""
        logger.info("Starting control monitoring thread")
        
        while self.monitoring_active:
            try:
                with self.lock:
                    self._reset_counters_if_needed()
                    self._check_control_limits()
                    self._log_control_state()
                
                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in control monitoring: {e}")
                time.sleep(10)
    
    def _reset_counters_if_needed(self):
        """Reset daily counters if new day."""
        current_time = datetime.now()
        
        # Reset daily counters at midnight
        if current_time.date() > self.state.last_hour_reset.date():
            self.state.daily_pnl = 0.0
            self.state.daily_volume = 0.0
            self.state.daily_trades = 0
            logger.info("Daily counters reset")
    
    def _check_control_limits(self):
        """Check control limits and trigger warnings."""
        # Check daily loss limit
        daily_loss_pct = abs(self.state.daily_pnl) / self.state.peak_capital if self.state.peak_capital > 0 else 0
        if daily_loss_pct > self.config.max_daily_loss_pct * 0.8:  # 80% of limit
            warning = f"Approaching daily loss limit: {daily_loss_pct:.2%}"
            if warning not in self.state.warnings_issued:
                self.state.warnings_issued.append(warning)
                logger.warning(warning)
        
        # Check drawdown limit
        if self.state.current_drawdown > self.config.max_drawdown_pct * 0.8:
            warning = f"Approaching drawdown limit: {self.state.current_drawdown:.2%}"
            if warning not in self.state.warnings_issued:
                self.state.warnings_issued.append(warning)
                logger.warning(warning)
        
        # Check consecutive losses
        if self.state.consecutive_losses >= self.config.max_consecutive_losses - 1:
            warning = f"Approaching consecutive loss limit: {self.state.consecutive_losses}"
            if warning not in self.state.warnings_issued:
                self.state.warnings_issued.append(warning)
                logger.warning(warning)
    
    def _log_control_state(self):
        """Log current control state (periodically)."""
        current_time = datetime.now()
        
        # Log state every 15 minutes
        if hasattr(self, '_last_state_log'):
            if (current_time - self._last_state_log).seconds < 900:
                return
        
        self._last_state_log = current_time
        
        logger.info(f"Control State: "
                   f"Daily P&L: ${self.state.daily_pnl:.2f}, "
                   f"Daily Volume: {self.state.daily_volume:.4f} BTC, "
                   f"Drawdown: {self.state.current_drawdown:.2%}, "
                   f"Consecutive Losses: {self.state.consecutive_losses}")
    
    def get_control_summary(self) -> Dict[str, Any]:
        """Get summary of current control state."""
        with self.lock:
            return {
                'trading_enabled': self.state.trading_enabled,
                'daily_pnl': self.state.daily_pnl,
                'daily_pnl_pct': self.state.daily_pnl / self.state.peak_capital if self.state.peak_capital > 0 else 0,
                'daily_volume': self.state.daily_volume,
                'current_drawdown': self.state.current_drawdown,
                'consecutive_losses': self.state.consecutive_losses,
                'circuit_breaker_active': self.state.circuit_breaker_active,
                'circuit_breaker_until': self.state.circuit_breaker_until,
                'warnings_count': len(self.state.warnings_issued),
                'violations_count': len(self.state.violations),
                'position_counts': len(self.state.current_positions),
                'risk_score': self._calculate_risk_score(self.state.violations, self.state.warnings_issued)
            }
    
    def enable_trading(self):
        """Enable trading."""
        with self.lock:
            self.state.trading_enabled = True
            logger.info("Trading enabled by controls")
    
    def disable_trading(self, reason: str = "Manual intervention"):
        """Disable trading."""
        with self.lock:
            self.state.trading_enabled = False
            logger.warning(f"Trading disabled: {reason}")
    
    def reset_daily_counters(self):
        """Reset daily counters."""
        with self.lock:
            self.state.daily_pnl = 0.0
            self.state.daily_volume = 0.0
            self.state.daily_trades = 0
            logger.info("Daily counters manually reset")
    
    def get_recent_warnings(self, limit: int = 10) -> List[str]:
        """Get recent warnings."""
        with self.lock:
            return self.state.warnings_issued[-limit:] if self.state.warnings_issued else []
    
    def get_recent_violations(self, limit: int = 10) -> List[str]:
        """Get recent violations."""
        with self.lock:
            return self.state.violations[-limit:] if self.state.violations else []
    
    def stress_test_position(self, symbol: str, size: float, price: float,
                           scenarios: List[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Stress test a position under various scenarios.
        
        Args:
            symbol: Trading symbol
            size: Position size
            price: Entry price
            scenarios: List of scenario definitions
        
        Returns:
            Dict[str, Any]: Stress test results
        """
        if scenarios is None:
            scenarios = [
                {'name': 'Mild', 'price_change': -0.05, 'volatility_change': 0.02},
                {'name': 'Moderate', 'price_change': -0.10, 'volatility_change': 0.05},
                {'name': 'Severe', 'price_change': -0.20, 'volatility_change': 0.10},
                {'name': 'Extreme', 'price_change': -0.30, 'volatility_change': 0.20}
            ]
        
        results = []
        position_value = size * price
        
        for scenario in scenarios:
            # Calculate scenario impact
            scenario_price = price * (1 + scenario['price_change'])
            scenario_value = size * scenario_price
            loss = position_value - scenario_value
            
            # Calculate risk metrics
            loss_pct = loss / position_value if position_value > 0 else 0
            
            # Check if scenario violates any limits
            violations = []
            
            # Daily loss limit
            potential_daily_loss = abs(self.state.daily_pnl + loss)
            daily_loss_pct = potential_daily_loss / self.state.peak_capital if self.state.peak_capital > 0 else 0
            if daily_loss_pct > self.config.max_daily_loss_pct:
                violations.append(f"Daily loss limit: {daily_loss_pct:.2%}")
            
            # Drawdown limit
            potential_drawdown = (self.state.peak_capital - (self.state.peak_capital + self.state.daily_pnl - loss)) / self.state.peak_capital
            if potential_drawdown > self.config.max_drawdown_pct:
                violations.append(f"Drawdown limit: {potential_drawdown:.2%}")
            
            results.append({
                'scenario_name': scenario['name'],
                'price_change': scenario['price_change'],
                'scenario_price': scenario_price,
                'position_value': scenario_value,
                'loss': loss,
                'loss_pct': loss_pct,
                'violations': violations,
                'risk_level': 'HIGH' if violations else 'LOW'
            })
        
        return {
            'position_value': position_value,
            'scenarios': results,
            'recommendation': self._generate_stress_test_recommendation(results)
        }
    
    def _generate_stress_test_recommendation(self, results: List[Dict[str, Any]]) -> str:
        """Generate recommendation based on stress test results."""
        # Check for violations in any scenario
        has_violations = any(len(scenario['violations']) > 0 for scenario in results)
        
        if has_violations:
            return "Position size may be too large for current risk limits. Consider reducing position size."
        else:
            return "Position size appears acceptable under stress test scenarios."
    
    def scenario_analysis(self, portfolio: Dict[str, float], 
                         market_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze portfolio under various market scenarios.
        
        Args:
            portfolio: Dictionary of symbol to position size
            market_scenarios: List of market scenario definitions
        
        Returns:
            Dict[str, Any]: Scenario analysis results
        """
        if not market_scenarios:
            market_scenarios = [
                {'name': 'Bull Market', 'price_changes': {'BTC/USDT': 0.20, 'ETH/USDT': 0.15}},
                {'name': 'Bear Market', 'price_changes': {'BTC/USDT': -0.20, 'ETH/USDT': -0.25}},
                {'name': 'High Volatility', 'price_changes': {'BTC/USDT': 0.00, 'ETH/USDT': 0.00}, 'volatility': 0.30},
                {'name': 'Flash Crash', 'price_changes': {'BTC/USDT': -0.40, 'ETH/USDT': -0.45}}
            ]
        
        results = []
        base_portfolio_value = sum(abs(size * self._get_current_price(symbol)) 
                                  for symbol, size in portfolio.items())
        
        for scenario in market_scenarios:
            scenario_value = 0
            price_changes = scenario.get('price_changes', {})
            
            for symbol, size in portfolio.items():
                current_price = self._get_current_price(symbol)
                price_change = price_changes.get(symbol, 0)
                scenario_price = current_price * (1 + price_change)
                scenario_value += size * scenario_price
            
            pnl = scenario_value - base_portfolio_value
            pnl_pct = pnl / base_portfolio_value if base_portfolio_value > 0 else 0
            
            # Check risk limits
            warnings = []
            if abs(pnl) > self.config.max_daily_loss_pct * self.state.peak_capital:
                warnings.append("Exceeds daily loss limit")
            
            results.append({
                'scenario_name': scenario['name'],
                'portfolio_value': scenario_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'warnings': warnings,
                'risk_assessment': 'HIGH' if warnings else 'LOW'
            })
        
        return {
            'base_portfolio_value': base_portfolio_value,
            'scenarios': results,
            'diversification_score': self._calculate_diversification_score(portfolio),
            'worst_case_scenario': min(results, key=lambda x: x['pnl']),
            'best_case_scenario': max(results, key=lambda x: x['pnl'])
        }
    
    def _calculate_diversification_score(self, portfolio: Dict[str, float]) -> float:
        """Calculate diversification score for portfolio."""
        if not portfolio:
            return 0.0
        
        # Simple Herfindahl-Hirschman Index (HHI) calculation
        total_value = sum(abs(size * self._get_current_price(symbol)) 
                         for symbol, size in portfolio.items())
        
        if total_value == 0:
            return 0.0
        
        hhi = 0
        for symbol, size in portfolio.items():
            value = abs(size * self._get_current_price(symbol))
            share = value / total_value
            hhi += share ** 2
        
        # Convert HHI to diversification score (0-100)
        diversification_score = max(0, 100 - (hhi * 100))
        
        return diversification_score
    
    def shutdown(self):
        """Shutdown controls and monitoring."""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Trading controls shut down")

# Risk Manager for higher-level risk management
class RiskManager:
    """
    Higher-level risk management with portfolio-level controls.
    """
    
    def __init__(self, controls: TradingControls, config: Dict[str, Any] = None):
        """
        Initialize risk manager.
        
        Args:
            controls: TradingControls instance
            config: Risk manager configuration
        """
        self.controls = controls
        self.config = config or {}
        
        # Risk limits
        self.max_portfolio_var = self.config.get('max_portfolio_var', 0.05)  # 5% portfolio VaR
        self.max_beta = self.config.get('max_beta', 1.5)
        self.min_sharpe = self.config.get('min_sharpe', 0.5)
        
        # Historical data
        self.portfolio_history = deque(maxlen=1000)
        self.correlation_matrix = {}
        
        logger.info("Initialized RiskManager")
    
    def assess_portfolio_risk(self, portfolio: Dict[str, float], 
                             prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess portfolio-level risk.
        
        Args:
            portfolio: Dictionary of symbol to position size
            prices: Dictionary of symbol to current price
        
        Returns:
            Dict[str, Any]: Portfolio risk assessment
        """
        # Calculate portfolio metrics
        portfolio_value = sum(abs(size * prices.get(symbol, 0)) 
                            for symbol, size in portfolio.items())
        
        # Calculate portfolio VaR (simplified)
        portfolio_var = self._calculate_portfolio_var(portfolio, prices)
        
        # Calculate portfolio beta (simplified)
        portfolio_beta = self._calculate_portfolio_beta(portfolio, prices)
        
        # Calculate concentration risk
        concentration_risk = self._calculate_concentration_risk(portfolio, prices)
        
        # Calculate liquidity risk
        liquidity_risk = self._calculate_liquidity_risk(portfolio)
        
        # Overall risk score
        risk_score = self._calculate_overall_risk_score(
            portfolio_var, portfolio_beta, concentration_risk, liquidity_risk
        )
        
        # Check against limits
        warnings = []
        if portfolio_var > self.max_portfolio_var:
            warnings.append(f"Portfolio VaR {portfolio_var:.2%} exceeds limit {self.max_portfolio_var:.2%}")
        
        if abs(portfolio_beta) > self.max_beta:
            warnings.append(f"Portfolio beta {portfolio_beta:.2f} exceeds limit {self.max_beta:.2f}")
        
        if concentration_risk > 0.7:
            warnings.append(f"High concentration risk: {concentration_risk:.2f}")
        
        if liquidity_risk > 0.5:
            warnings.append(f"High liquidity risk: {liquidity_risk:.2f}")
        
        return {
            'portfolio_value': portfolio_value,
            'portfolio_var': portfolio_var,
            'portfolio_beta': portfolio_beta,
            'concentration_risk': concentration_risk,
            'liquidity_risk': liquidity_risk,
            'risk_score': risk_score,
            'warnings': warnings,
            'recommendations': self._generate_portfolio_recommendations(warnings, portfolio)
        }
    
    def _calculate_portfolio_var(self, portfolio: Dict[str, float], 
                                prices: Dict[str, float]) -> float:
        """Calculate portfolio Value at Risk (simplified)."""
        # Simplified VaR calculation
        # In production, use historical simulation or parametric methods
        total_value = sum(abs(size * prices.get(symbol, 0)) 
                         for symbol, size in portfolio.items())
        
        if total_value == 0:
            return 0.0
        
        # Assume 5% daily VaR for crypto portfolio
        base_var = 0.05
        
        # Adjust for concentration
        concentration = self._calculate_concentration_risk(portfolio, prices)
        adjusted_var = base_var * (1 + concentration)
        
        return min(adjusted_var, 0.20)  # Cap at 20%
    
    def _calculate_portfolio_beta(self, portfolio: Dict[str, float],
                                 prices: Dict[str, float]) -> float:
        """Calculate portfolio beta relative to BTC (simplified)."""
        # Simplified beta calculation
        # In production, use regression against benchmark
        if 'BTC/USDT' not in portfolio:
            return 0.0
        
        btc_value = abs(portfolio.get('BTC/USDT', 0) * prices.get('BTC/USDT', 0))
        total_value = sum(abs(size * prices.get(symbol, 0)) 
                         for symbol, size in portfolio.items())
        
        if total_value == 0:
            return 0.0
        
        # Beta is roughly proportional to BTC exposure
        beta = btc_value / total_value
        
        # Adjust for correlation (simplified)
        return beta * 1.5  # Assume crypto assets have beta > 1
    
    def _calculate_concentration_risk(self, portfolio: Dict[str, float],
                                     prices: Dict[str, float]) -> float:
        """Calculate concentration risk using HHI."""
        total_value = sum(abs(size * prices.get(symbol, 0)) 
                         for symbol, size in portfolio.items())
        
        if total_value == 0:
            return 0.0
        
        hhi = 0
        for symbol, size in portfolio.items():
            value = abs(size * prices.get(symbol, 0))
            share = value / total_value
            hhi += share ** 2
        
        return hhi
    
    def _calculate_liquidity_risk(self, portfolio: Dict[str, float]) -> float:
        """Calculate liquidity risk (simplified)."""
        # Simplified liquidity risk based on position sizes
        # In production, use actual liquidity data
        
        if not portfolio:
            return 0.0
        
        # Assume BTC is most liquid, altcoins less liquid
        liquidity_scores = {
            'BTC/USDT': 1.0,
            'ETH/USDT': 0.8,
            'default': 0.5
        }
        
        total_risk = 0
        for symbol in portfolio:
            score = liquidity_scores.get(symbol, liquidity_scores['default'])
            total_risk += (1 - score) * abs(portfolio[symbol])
        
        # Normalize to 0-1 scale
        max_position = max(abs(size) for size in portfolio.values()) if portfolio else 0
        return total_risk / (len(portfolio) * max_position) if max_position > 0 else 0.0
    
    def _calculate_overall_risk_score(self, portfolio_var: float, portfolio_beta: float,
                                     concentration_risk: float, liquidity_risk: float) -> float:
        """Calculate overall portfolio risk score (0-1, higher is riskier)."""
        weights = {
            'var': 0.4,
            'beta': 0.2,
            'concentration': 0.3,
            'liquidity': 0.1
        }
        
        # Normalize metrics to 0-1 scale
        var_score = min(portfolio_var / 0.20, 1.0)  # Cap at 20% VaR
        beta_score = min(abs(portfolio_beta) / 2.0, 1.0)  # Cap at beta = 2
        concentration_score = concentration_risk  # Already 0-1
        liquidity_score = liquidity_risk  # Already 0-1
        
        risk_score = (
            weights['var'] * var_score +
            weights['beta'] * beta_score +
            weights['concentration'] * concentration_score +
            weights['liquidity'] * liquidity_score
        )
        
        return min(risk_score, 1.0)
    
    def _generate_portfolio_recommendations(self, warnings: List[str],
                                           portfolio: Dict[str, float]) -> List[str]:
        """Generate portfolio recommendations based on risk assessment."""
        recommendations = []
        
        if not warnings:
            recommendations.append("Portfolio risk within acceptable limits.")
            return recommendations
        
        # Generate specific recommendations
        for warning in warnings:
            if "VaR" in warning:
                recommendations.append("Consider reducing position sizes or adding hedging instruments.")
            if "beta" in warning:
                recommendations.append("Consider reducing exposure to high-beta assets.")
            if "concentration" in warning:
                recommendations.append("Consider diversifying across more assets.")
            if "liquidity" in warning:
                recommendations.append("Consider reducing positions in illiquid assets.")
        
        # General recommendations
        if len(portfolio) < 3:
            recommendations.append("Portfolio is under-diversified. Consider adding more assets.")
        
        if len(warnings) > 2:
            recommendations.append("Multiple risk warnings. Consider significant portfolio rebalancing.")
        
        return recommendations
    
    def optimize_portfolio_risk(self, portfolio: Dict[str, float],
                               prices: Dict[str, float],
                               target_risk: float = 0.3) -> Dict[str, float]:
        """
        Optimize portfolio for target risk level.
        
        Args:
            portfolio: Current portfolio
            prices: Current prices
            target_risk: Target risk score (0-1)
        
        Returns:
            Dict[str, float]: Optimized portfolio weights
        """
        # Simplified portfolio optimization
        # In production, use Mean-Variance Optimization or Black-Litterman
        
        current_risk = self._calculate_overall_risk_score(
            self._calculate_portfolio_var(portfolio, prices),
            self._calculate_portfolio_beta(portfolio, prices),
            self._calculate_concentration_risk(portfolio, prices),
            self._calculate_liquidity_risk(portfolio)
        )
        
        if current_risk <= target_risk:
            # Already at or below target risk
            return portfolio
        
        # Reduce positions proportionally to achieve target risk
        reduction_factor = target_risk / current_risk
        
        optimized_portfolio = {}
        for symbol, size in portfolio.items():
            optimized_portfolio[symbol] = size * reduction_factor
        
        return optimized_portfolio

# Example usage
if __name__ == "__main__":
    print("Testing Trading Controls Module...")
    
    # Create controls configuration
    config = ControlConfig(
        risk_level=RiskLevel.MODERATE,
        max_position_size_pct=0.10,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.20,
        max_trades_per_hour=10
    )
    
    # Create trading controls
    controls = TradingControls(config)
    
    # Update with initial state
    controls.update_portfolio_state(
        capital=10000.0,
        positions={'BTC/USDT': 0.1, 'ETH/USDT': 2.0}
    )
    
    # Update market data
    controls.update_market_data('BTC/USDT', 50000, 1000)
    controls.update_market_data('ETH/USDT', 3000, 5000)
    
    # Test position sizing
    print("\n1. Position Sizing Test:")
    sizing_result = controls.calculate_position_size(
        symbol='BTC/USDT',
        entry_price=51000,
        stop_loss=50000,
        confidence=0.7,
        capital=10000
    )
    
    print(f"   Recommended size: {sizing_result.recommended_size:.4f} BTC")
    print(f"   Max allowed size: {sizing_result.max_allowed_size:.4f} BTC")
    print(f"   Risk per trade: ${sizing_result.risk_per_trade:.2f}")
    print(f"   Position value: ${sizing_result.position_value:.2f}")
    
    # Test order validation
    print("\n2. Order Validation Test:")
    validation_result = controls.validate_order(
        order_id='test_order_001',
        symbol='BTC/USDT',
        order_type='market',
        side='buy',
        size=0.5,  # Large size to trigger controls
        price=51000,
        metadata={'strategy': 'test', 'confidence': 0.7}
    )
    
    print(f"   Order ID: {validation_result.order_id}")
    print(f"   Status: {validation_result.status.value}")
    print(f"   Requested size: {validation_result.requested_size:.4f}")
    print(f"   Approved size: {validation_result.approved_size:.4f}")
    print(f"   Reason: {validation_result.reason}")
    print(f"   Risk score: {validation_result.risk_score:.2f}")
    
    if validation_result.warnings:
        print(f"   Warnings: {len(validation_result.warnings)}")
        for warning in validation_result.warnings:
            print(f"     - {warning}")
    
    if validation_result.violations:
        print(f"   Violations: {len(validation_result.violations)}")
        for violation in validation_result.violations:
            print(f"     - {violation}")
    
    # Test stress testing
    print("\n3. Stress Testing:")
    stress_test = controls.stress_test_position(
        symbol='BTC/USDT',
        size=0.2,
        price=51000
    )
    
    print(f"   Position value: ${stress_test['position_value']:.2f}")
    print(f"   Recommendation: {stress_test['recommendation']}")
    
    for scenario in stress_test['scenarios']:
        print(f"   {scenario['scenario_name']}: Loss ${scenario['loss']:.2f} ({scenario['loss_pct']:.2%}) - {scenario['risk_level']}")
    
    # Test control summary
    print("\n4. Control Summary:")
    summary = controls.get_control_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Test risk manager
    print("\n5. Risk Manager Test:")
    
    risk_manager = RiskManager(controls)
    
    portfolio = {'BTC/USDT': 0.2, 'ETH/USDT': 5.0}
    prices = {'BTC/USDT': 51000, 'ETH/USDT': 3100}
    
    risk_assessment = risk_manager.assess_portfolio_risk(portfolio, prices)
    
    print(f"   Portfolio Value: ${risk_assessment['portfolio_value']:.2f}")
    print(f"   Portfolio VaR: {risk_assessment['portfolio_var']:.2%}")
    print(f"   Portfolio Beta: {risk_assessment['portfolio_beta']:.2f}")
    print(f"   Concentration Risk: {risk_assessment['concentration_risk']:.2f}")
    print(f"   Liquidity Risk: {risk_assessment['liquidity_risk']:.2f}")
    print(f"   Overall Risk Score: {risk_assessment['risk_score']:.2f}")
    
    if risk_assessment['warnings']:
        print(f"   Warnings: {len(risk_assessment['warnings'])}")
        for warning in risk_assessment['warnings']:
            print(f"     - {warning}")
    
    if risk_assessment['recommendations']:
        print(f"   Recommendations:")
        for rec in risk_assessment['recommendations']:
            print(f"     - {rec}")
    
    # Shutdown
    controls.shutdown()
    
    print("\nTrading controls testing completed successfully!")