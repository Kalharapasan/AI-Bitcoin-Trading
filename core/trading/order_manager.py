"""
Order management module for Bitcoin trading AI.
Handles order creation, modification, execution, and tracking for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
from datetime import datetime, timedelta
import asyncio
import json
import hashlib
from pathlib import Path
from collections import deque, defaultdict
import uuid

# Import project modules
from config.settings import TradingSettings, ExchangeSettings, AppConstants
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.trading.signal_generator import TradingSignal, SignalType
from core.trading.position_sizer import PositionSizeResult
from core.risk_management.risk_analyzer import RiskAnalyzer
from core.risk_management.stop_loss_manager import StopLossManager
from core.utils.cache import Cache

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Order Types and Enums ============
class OrderType(str, Enum):
    """Types of trading orders"""
    MARKET = "market"          # Execute immediately at market price
    LIMIT = "limit"            # Execute at specified price or better
    STOP = "stop"              # Becomes market order when price reaches stop
    STOP_LIMIT = "stop_limit"  # Becomes limit order when price reaches stop
    TRAILING_STOP = "trailing_stop"  # Trailing stop order
    TAKE_PROFIT = "take_profit"      # Take profit order
    OCO = "oco"                # One-Cancels-Other (bracket order)
    IOC = "ioc"                # Immediate-Or-Cancel
    FOK = "fok"                # Fill-Or-Kill

class OrderSide(str, Enum):
    """Order side (buy/sell)"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    """Order status"""
    PENDING = "pending"        # Order created but not sent
    SENT = "sent"              # Order sent to exchange
    PARTIALLY_FILLED = "partially_filled"  # Partially executed
    FILLED = "filled"          # Fully executed
    CANCELLED = "cancelled"    # Order cancelled
    REJECTED = "rejected"      # Order rejected by exchange
    EXPIRED = "expired"        # Order expired
    FAILED = "failed"          # Order failed

class OrderTimeInForce(str, Enum):
    """Time in force for orders"""
    GTC = "gtc"                # Good Till Cancelled
    IOC = "ioc"                # Immediate Or Cancel
    FOK = "fok"                # Fill Or Kill
    DAY = "day"                # Valid for the day
    GTD = "gtd"                # Good Till Date

# ============ Configuration ============
@dataclass
class OrderManagerConfig:
    """Configuration for order management"""
    
    # General settings
    default_order_type: OrderType = OrderType.LIMIT
    default_time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    max_order_retries: int = 3
    retry_delay_seconds: float = 1.0
    order_timeout_seconds: int = 60
    
    # Slippage control
    max_slippage_percent: float = 0.1  # 0.1% max slippage
    use_slippage_model: bool = True
    slippage_lookback: int = 100
    
    # Order pricing
    limit_price_offset_percent: float = 0.05  # 0.05% offset for limit orders
    market_order_premium_percent: float = 0.02  # 2% premium for market orders
    price_improvement_target: float = 0.01  # 1% price improvement target
    
    # Order size management
    allow_partial_fills: bool = True
    min_order_size: float = 0.0001  # Minimum BTC order size
    max_order_size: float = 100.0   # Maximum BTC order size
    size_increment: float = 0.0001  # Order size increment
    
    # Risk controls
    max_open_orders: int = 10
    max_order_value: float = 10000.0  # Max value per order
    daily_order_limit: int = 100
    rate_limit_orders_per_second: float = 2.0
    
    # Stop loss and take profit
    use_trailing_stops: bool = True
    trailing_stop_distance: float = 0.02  # 2% trailing stop
    take_profit_distance: float = 0.03    # 3% take profit
    stop_loss_distance: float = 0.02      # 2% stop loss
    
    # Advanced order types
    use_bracket_orders: bool = True
    use_oco_orders: bool = True
    use_twap_orders: bool = False         # Time-Weighted Average Price
    use_vwap_orders: bool = False         # Volume-Weighted Average Price
    
    # Execution optimization
    split_large_orders: bool = True
    max_split_parts: int = 5
    min_split_size: float = 0.001  # 0.001 BTC minimum split size
    
    # Error handling
    auto_cancel_stale_orders: bool = True
    stale_order_threshold_minutes: int = 5
    auto_retry_failed_orders: bool = True
    max_retry_attempts: int = 3
    
    # Monitoring and logging
    log_all_orders: bool = True
    log_execution_details: bool = True
    save_order_history: bool = True
    order_history_path: str = "data/orders/"
    
    # Exchange specific
    exchange_name: str = "binance"
    trading_pair: str = "BTCUSDT"
    use_testnet: bool = False
    api_rate_limit_delay: float = 0.1  # 100ms between API calls
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_slippage_percent < 0 or self.max_slippage_percent > 1:
            raise ValueError("max_slippage_percent must be between 0 and 1")
        
        if self.limit_price_offset_percent < 0:
            raise ValueError("limit_price_offset_percent must be non-negative")
        
        if self.market_order_premium_percent < 0:
            raise ValueError("market_order_premium_percent must be non-negative")
        
        # Create order history directory
        Path(self.order_history_path).mkdir(parents=True, exist_ok=True)

# ============ Order Data Structures ============
@dataclass
class Order:
    """Trading order with all metadata"""
    order_id: str
    client_order_id: str
    order_type: OrderType
    order_side: OrderSide
    trading_pair: str
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    fee: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate order"""
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError(f"{self.order_type.value} orders require price")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} orders require stop_price")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'order_type': self.order_type.value,
            'order_side': self.order_side.value,
            'trading_pair': self.trading_pair,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'fee': self.fee,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create from dictionary"""
        return cls(
            order_id=data['order_id'],
            client_order_id=data['client_order_id'],
            order_type=OrderType(data['order_type']),
            order_side=OrderSide(data['order_side']),
            trading_pair=data['trading_pair'],
            quantity=data['quantity'],
            price=data.get('price'),
            stop_price=data.get('stop_price'),
            time_in_force=OrderTimeInForce(data.get('time_in_force', 'gtc')),
            status=OrderStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            filled_quantity=data['filled_quantity'],
            average_fill_price=data.get('average_fill_price'),
            fee=data['fee'],
            metadata=data['metadata']
        )
    
    @property
    def is_open(self) -> bool:
        """Check if order is open (not filled, cancelled, or rejected)"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SENT, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def is_closed(self) -> bool:
        """Check if order is closed"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                              OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.FAILED]
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill"""
        return max(self.quantity - self.filled_quantity, 0)
    
    @property
    def fill_percentage(self) -> float:
        """Get fill percentage"""
        if self.quantity > 0:
            return self.filled_quantity / self.quantity
        return 0.0

@dataclass
class OrderRequest:
    """Request to create an order"""
    order_side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate client order ID if not provided"""
        if self.client_order_id is None:
            self.client_order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

@dataclass
class OrderExecution:
    """Order execution details"""
    execution_id: str
    order_id: str
    quantity: float
    price: float
    fee: float
    fee_currency: str
    timestamp: datetime
    exchange_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderGroup:
    """Group of related orders (e.g., bracket order)"""
    group_id: str
    orders: List[Order]
    parent_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

# ============ Base Order Manager ============
class BaseOrderManager:
    """Base class for order management"""
    
    def __init__(self, config: Optional[OrderManagerConfig] = None):
        self.config = config or OrderManagerConfig()
        self.orders: Dict[str, Order] = {}
        self.order_groups: Dict[str, OrderGroup] = {}
        self.executions: Dict[str, OrderExecution] = {}
        self.risk_analyzer = RiskAnalyzer()
        self.stop_loss_manager = StopLossManager()
        self.cache = Cache()
        self.logger = get_logger(__name__)
        
        # Statistics
        self.order_statistics = defaultdict(int)
        self.execution_statistics = defaultdict(float)
        
        # Initialize exchange connection
        self.exchange = self._initialize_exchange()
        
    def _initialize_exchange(self):
        """Initialize exchange connection"""
        # This is a placeholder - actual implementation depends on exchange
        # For now, return a mock exchange object
        class MockExchange:
            def __init__(self):
                self.name = self.config.exchange_name
                self.testnet = self.config.use_testnet
            
            async def create_order(self, **kwargs):
                """Mock create order method"""
                await asyncio.sleep(0.01)  # Simulate API delay
                return {'id': f'exch_{uuid.uuid4().hex[:10]}', 'status': 'open'}
            
            async def cancel_order(self, order_id):
                """Mock cancel order method"""
                await asyncio.sleep(0.01)
                return True
            
            async def get_order(self, order_id):
                """Mock get order method"""
                await asyncio.sleep(0.01)
                return {'id': order_id, 'status': 'filled'}
        
        return MockExchange()
    
    async def create_order(self, 
                          request: OrderRequest,
                          trading_pair: Optional[str] = None) -> Order:
        """Create a new order"""
        raise NotImplementedError
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        raise NotImplementedError
    
    async def modify_order(self, 
                          order_id: str,
                          new_price: Optional[float] = None,
                          new_quantity: Optional[float] = None) -> bool:
        """Modify an existing order"""
        raise NotImplementedError
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status"""
        raise NotImplementedError
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        return [order for order in self.orders.values() if order.is_open]
    
    def get_order_history(self, 
                         limit: int = 100,
                         status: Optional[OrderStatus] = None) -> List[Order]:
        """Get order history"""
        orders = list(self.orders.values())
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        # Sort by creation time, most recent first
        orders.sort(key=lambda x: x.created_at, reverse=True)
        
        return orders[:limit]
    
    def record_execution(self, execution: OrderExecution):
        """Record order execution"""
        self.executions[execution.execution_id] = execution
        
        # Update order if it exists
        if execution.order_id in self.orders:
            order = self.orders[execution.order_id]
            order.filled_quantity += execution.quantity
            order.fee += execution.fee
            order.updated_at = datetime.now()
            
            # Update average fill price
            if order.filled_quantity > 0:
                if order.average_fill_price is None:
                    order.average_fill_price = execution.price
                else:
                    total_value = (order.average_fill_price * (order.filled_quantity - execution.quantity) +
                                  execution.price * execution.quantity)
                    order.average_fill_price = total_value / order.filled_quantity
            
            # Update status if fully filled
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
            
            self.orders[execution.order_id] = order
            
        # Update statistics
        self.execution_statistics['total_quantity'] += execution.quantity
        self.execution_statistics['total_value'] += execution.quantity * execution.price
        self.execution_statistics['total_fees'] += execution.fee

# ============ Order Creator ============
class OrderCreator:
    """Creates optimized trading orders"""
    
    def __init__(self, config: OrderManagerConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.slippage_model = SlippageModel(config)
        self.price_calculator = PriceCalculator(config)
    
    def create_order_from_signal(self,
                                signal: TradingSignal,
                                position_size: PositionSizeResult,
                                market_data: pd.DataFrame) -> OrderRequest:
        """Create order request from trading signal and position size"""
        
        # Determine order side
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            order_side = OrderSide.BUY
        elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            order_side = OrderSide.SELL
        else:
            # For hold or close signals, create appropriate order
            # This depends on your strategy - for now, return None
            return None
        
        # Calculate order quantity
        quantity = self._calculate_order_quantity(position_size, signal.price)
        
        # Determine order type
        order_type = self._determine_order_type(signal, market_data)
        
        # Calculate optimal price
        price = self._calculate_order_price(
            order_type, order_side, signal.price, market_data
        )
        
        # Calculate stop price if needed
        stop_price = self._calculate_stop_price(
            order_type, order_side, signal, market_data
        )
        
        # Create order request
        request = OrderRequest(
            order_side=order_side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            time_in_force=self.config.default_time_in_force,
            metadata={
                'signal_id': signal.signal_id,
                'signal_type': signal.signal_type.value,
                'signal_strength': signal.strength,
                'signal_confidence': signal.confidence,
                'position_size_result': position_size.to_dict(),
                'original_signal_price': signal.price
            }
        )
        
        return request
    
    def _calculate_order_quantity(self, 
                                 position_size: PositionSizeResult,
                                 current_price: float) -> float:
        """Calculate order quantity from position size"""
        # Convert position size to quantity
        if position_size.size_unit.value == 'btc':
            quantity = position_size.position_size
        elif position_size.size_unit.value == 'percentage':
            # Need portfolio value to calculate
            # For now, assume it's already calculated
            quantity = position_size.position_value / current_price
        elif position_size.size_unit.value == 'dollar':
            quantity = position_size.position_size / current_price
        else:
            # For units or contracts
            quantity = position_size.position_size
        
        # Apply order size constraints
        quantity = max(quantity, self.config.min_order_size)
        quantity = min(quantity, self.config.max_order_size)
        
        # Round to size increment
        if self.config.size_increment > 0:
            quantity = round(quantity / self.config.size_increment) * self.config.size_increment
        
        return quantity
    
    def _determine_order_type(self, 
                             signal: TradingSignal,
                             market_data: pd.DataFrame) -> OrderType:
        """Determine optimal order type based on signal and market conditions"""
        
        # Default to config setting
        order_type = self.config.default_order_type
        
        # Adjust based on signal strength and confidence
        if signal.strength >= 0.8 and signal.confidence >= 0.8:
            # Strong signal with high confidence - use market order for immediate execution
            if self._is_low_volatility(market_data):
                order_type = OrderType.LIMIT  # Use limit for better price in low vol
            else:
                order_type = OrderType.MARKET  # Use market to ensure execution
        
        elif signal.strength <= 0.4 or signal.confidence <= 0.5:
            # Weak signal - use limit order to control price
            order_type = OrderType.LIMIT
        
        # Check for breakout signals
        if 'breakout' in signal.metadata.get('condition', '').lower():
            # For breakouts, use stop orders
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                order_type = OrderType.STOP
            else:
                order_type = OrderType.STOP
        
        # If using trailing stops is enabled and signal is strong
        if self.config.use_trailing_stops and signal.strength >= 0.7:
            order_type = OrderType.TRAILING_STOP
        
        return order_type
    
    def _is_low_volatility(self, market_data: pd.DataFrame) -> bool:
        """Check if market has low volatility"""
        if len(market_data) < 20:
            return False
        
        returns = market_data['close'].pct_change().dropna()
        if len(returns) < 20:
            return False
        
        volatility = returns.std()
        return volatility < 0.01  # Less than 1% volatility
    
    def _calculate_order_price(self,
                              order_type: OrderType,
                              order_side: OrderSide,
                              signal_price: float,
                              market_data: pd.DataFrame) -> Optional[float]:
        """Calculate optimal order price"""
        
        if order_type == OrderType.MARKET:
            # Market orders don't have a price
            return None
        
        current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else signal_price
        
        if order_type == OrderType.LIMIT:
            # Calculate limit price with offset for price improvement
            if order_side == OrderSide.BUY:
                # For buy orders, price below current for better entry
                offset = -self.config.limit_price_offset_percent / 100
            else:
                # For sell orders, price above current for better exit
                offset = self.config.limit_price_offset_percent / 100
            
            # Adjust offset based on market conditions
            offset = self._adjust_price_offset(offset, market_data)
            
            price = current_price * (1 + offset)
            
            # Apply slippage model if enabled
            if self.config.use_slippage_model:
                expected_slippage = self.slippage_model.estimate_slippage(
                    order_side, price, market_data
                )
                # Adjust price for expected slippage
                if order_side == OrderSide.BUY:
                    price *= (1 - expected_slippage)
                else:
                    price *= (1 + expected_slippage)
            
            return price
        
        elif order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            # For stop orders, price is the trigger price
            # The execution price will be market or limit price after trigger
            return None
        
        elif order_type == OrderType.TRAILING_STOP:
            # Trailing stops don't have a fixed price
            return None
        
        return None
    
    def _adjust_price_offset(self, 
                            base_offset: float,
                            market_data: pd.DataFrame) -> float:
        """Adjust price offset based on market conditions"""
        
        if len(market_data) < 50:
            return base_offset
        
        # Adjust based on volatility
        returns = market_data['close'].pct_change().dropna()
        if len(returns) >= 20:
            volatility = returns.std()
            # Increase offset in high volatility, decrease in low volatility
            volatility_factor = volatility / 0.01  # Normalize to 1% volatility
            adjusted_offset = base_offset * volatility_factor
        
        # Adjust based on trend
        trend_strength = self._calculate_trend_strength(market_data)
        # Increase offset against trend, decrease with trend
        trend_factor = 1.0 + abs(trend_strength) * 0.5
        adjusted_offset = adjusted_offset * trend_factor
        
        # Ensure offset is reasonable
        max_offset = 0.05  # 5% maximum offset
        return max(min(adjusted_offset, max_offset), -max_offset)
    
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength (-1 to 1)"""
        if len(market_data) < 50:
            return 0.0
        
        short_ma = market_data['close'].rolling(window=20).mean().iloc[-1]
        long_ma = market_data['close'].rolling(window=50).mean().iloc[-1]
        
        if long_ma > 0:
            trend = (short_ma - long_ma) / long_ma
            return trend
        
        return 0.0
    
    def _calculate_stop_price(self,
                             order_type: OrderType,
                             order_side: OrderSide,
                             signal: TradingSignal,
                             market_data: pd.DataFrame) -> Optional[float]:
        """Calculate stop price for stop orders"""
        
        if order_type not in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP]:
            return None
        
        current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else signal.price
        
        if order_side == OrderSide.BUY:
            # Buy stop orders trigger when price rises above stop price
            stop_price = current_price * (1 + self.config.stop_loss_distance)
        else:
            # Sell stop orders trigger when price falls below stop price
            stop_price = current_price * (1 - self.config.stop_loss_distance)
        
        # Adjust based on volatility
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            # Increase stop distance in high volatility
            volatility_multiplier = 1.0 + (volatility / 0.01) * 0.5
            if order_side == OrderSide.BUY:
                stop_price = current_price * (1 + self.config.stop_loss_distance * volatility_multiplier)
            else:
                stop_price = current_price * (1 - self.config.stop_loss_distance * volatility_multiplier)
        
        return stop_price
    
    def create_bracket_order(self,
                           main_order: OrderRequest,
                           market_data: pd.DataFrame) -> List[OrderRequest]:
        """Create bracket order (main order with stop loss and take profit)"""
        
        if not self.config.use_bracket_orders:
            return [main_order]
        
        bracket_orders = [main_order]
        
        current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else main_order.price
        
        # Create stop loss order
        stop_loss_request = self._create_stop_loss_order(
            main_order, current_price, market_data
        )
        if stop_loss_request:
            bracket_orders.append(stop_loss_request)
        
        # Create take profit order
        take_profit_request = self._create_take_profit_order(
            main_order, current_price, market_data
        )
        if take_profit_request:
            bracket_orders.append(take_profit_request)
        
        return bracket_orders
    
    def _create_stop_loss_order(self,
                               main_order: OrderRequest,
                               current_price: float,
                               market_data: pd.DataFrame) -> Optional[OrderRequest]:
        """Create stop loss order for bracket"""
        
        # Determine stop loss side (opposite of main order)
        stop_side = OrderSide.SELL if main_order.order_side == OrderSide.BUY else OrderSide.BUY
        
        # Calculate stop loss price
        if main_order.order_side == OrderSide.BUY:
            stop_price = current_price * (1 - self.config.stop_loss_distance)
        else:
            stop_price = current_price * (1 + self.config.stop_loss_distance)
        
        # Adjust based on volatility
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            volatility_adjustment = 1.0 + (volatility / 0.01) * 0.2
            
            if main_order.order_side == OrderSide.BUY:
                stop_price = current_price * (1 - self.config.stop_loss_distance * volatility_adjustment)
            else:
                stop_price = current_price * (1 + self.config.stop_loss_distance * volatility_adjustment)
        
        # Create stop order
        stop_request = OrderRequest(
            order_side=stop_side,
            quantity=main_order.quantity,
            order_type=OrderType.STOP,
            price=None,  # Market order after trigger
            stop_price=stop_price,
            time_in_force=OrderTimeInForce.GTC,
            client_order_id=f"{main_order.client_order_id}_stop",
            metadata={
                'parent_order_id': main_order.client_order_id,
                'order_type': 'stop_loss',
                'main_order_side': main_order.order_side.value
            }
        )
        
        return stop_request
    
    def _create_take_profit_order(self,
                                 main_order: OrderRequest,
                                 current_price: float,
                                 market_data: pd.DataFrame) -> Optional[OrderRequest]:
        """Create take profit order for bracket"""
        
        # Determine take profit side (same as main order for profit taking)
        tp_side = main_order.order_side
        
        # Calculate take profit price
        if main_order.order_side == OrderSide.BUY:
            tp_price = current_price * (1 + self.config.take_profit_distance)
        else:
            tp_price = current_price * (1 - self.config.take_profit_distance)
        
        # Adjust based on volatility
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            volatility_adjustment = 1.0 + (volatility / 0.01) * 0.2
            
            if main_order.order_side == OrderType.BUY:
                tp_price = current_price * (1 + self.config.take_profit_distance * volatility_adjustment)
            else:
                tp_price = current_price * (1 - self.config.take_profit_distance * volatility_adjustment)
        
        # Create take profit order (limit order)
        tp_request = OrderRequest(
            order_side=tp_side,
            quantity=main_order.quantity,
            order_type=OrderType.LIMIT,
            price=tp_price,
            stop_price=None,
            time_in_force=OrderTimeInForce.GTC,
            client_order_id=f"{main_order.client_order_id}_tp",
            metadata={
                'parent_order_id': main_order.client_order_id,
                'order_type': 'take_profit',
                'main_order_side': main_order.order_side.value
            }
        )
        
        return tp_request

# ============ Slippage Model ============
class SlippageModel:
    """Models and estimates slippage for orders"""
    
    def __init__(self, config: OrderManagerConfig):
        self.config = config
        self.slippage_history = deque(maxlen=config.slippage_lookback)
        self.logger = get_logger(__name__)
    
    def estimate_slippage(self,
                         order_side: OrderSide,
                         target_price: float,
                         market_data: pd.DataFrame) -> float:
        """Estimate slippage for an order"""
        
        if not self.config.use_slippage_model:
            return 0.0
        
        # Base slippage estimation
        base_slippage = self._calculate_base_slippage(market_data)
        
        # Adjust for order size
        size_adjustment = self._adjust_for_order_size(order_side, target_price, market_data)
        
        # Adjust for market conditions
        market_adjustment = self._adjust_for_market_conditions(market_data)
        
        # Adjust for time of day
        time_adjustment = self._adjust_for_time_of_day()
        
        # Combine adjustments
        estimated_slippage = base_slippage * size_adjustment * market_adjustment * time_adjustment
        
        # Cap at maximum allowed slippage
        estimated_slippage = min(estimated_slippage, self.config.max_slippage_percent)
        
        return estimated_slippage
    
    def _calculate_base_slippage(self, market_data: pd.DataFrame) -> float:
        """Calculate base slippage based on historical data"""
        
        if len(self.slippage_history) > 10:
            # Use historical average
            return np.mean(list(self.slippage_history)[-10:])
        
        # Default base slippage based on market conditions
        if len(market_data) < 20:
            return 0.001  # 0.1% default
        
        # Estimate based on bid-ask spread if available
        if 'bid' in market_data.columns and 'ask' in market_data.columns:
            spread = (market_data['ask'].iloc[-1] - market_data['bid'].iloc[-1]) / market_data['bid'].iloc[-1]
            return spread * 0.5  # Assume half the spread as slippage
        
        # Estimate based on volatility
        returns = market_data['close'].pct_change().dropna()
        if len(returns) >= 20:
            volatility = returns.std()
            return volatility * 0.1  # 10% of volatility as slippage estimate
        
        return 0.001  # 0.1% default
    
    def _adjust_for_order_size(self,
                              order_side: OrderSide,
                              target_price: float,
                              market_data: pd.DataFrame) -> float:
        """Adjust slippage for order size"""
        
        # This is a simplified model
        # In reality, you'd need order book data for accurate size adjustment
        
        # For now, use volume as proxy for liquidity
        if 'volume' not in market_data.columns or len(market_data) < 20:
            return 1.0
        
        avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1]
        if avg_volume <= 0:
            return 1.0
        
        # Assume order size affects slippage logarithmically
        # Larger orders relative to average volume = more slippage
        order_size_btc = 0.01  # Default assumption - should be actual order size
        size_ratio = order_size_btc / avg_volume
        
        # Logarithmic adjustment
        if size_ratio > 0:
            adjustment = 1.0 + np.log1p(size_ratio) * 0.5
        else:
            adjustment = 1.0
        
        return adjustment
    
    def _adjust_for_market_conditions(self, market_data: pd.DataFrame) -> float:
        """Adjust slippage for market conditions"""
        
        if len(market_data) < 50:
            return 1.0
        
        # High volatility increases slippage
        returns = market_data['close'].pct_change().dropna()
        if len(returns) >= 20:
            volatility = returns.std()
            volatility_adjustment = 1.0 + (volatility / 0.01) * 0.3
        else:
            volatility_adjustment = 1.0
        
        # Low liquidity increases slippage
        if 'volume' in market_data.columns:
            volume_trend = self._calculate_volume_trend(market_data)
            liquidity_adjustment = 1.0 + (1 - volume_trend) * 0.2
        else:
            liquidity_adjustment = 1.0
        
        return volatility_adjustment * liquidity_adjustment
    
    def _calculate_volume_trend(self, market_data: pd.DataFrame) -> float:
        """Calculate volume trend (0-1, higher = more volume)"""
        if len(market_data) < 20:
            return 0.5
        
        current_volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1]
        
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            # Normalize to 0-1 range
            return min(max(volume_ratio, 0), 2) / 2
        
        return 0.5
    
    def _adjust_for_time_of_day(self) -> float:
        """Adjust slippage for time of day"""
        current_hour = datetime.now().hour
        
        # Higher slippage during low activity periods
        # Crypto markets are 24/7, but there are still patterns
        
        if 0 <= current_hour < 4:  # Early morning UTC (Asian session)
            return 1.2  # 20% higher slippage
        elif 8 <= current_hour < 12:  # European session
            return 0.9  # 10% lower slippage
        elif 13 <= current_hour < 17:  # US session overlap
            return 0.8  # 20% lower slippage (highest liquidity)
        else:
            return 1.0  # Normal slippage
    
    def record_slippage(self, 
                       expected_price: float,
                       actual_price: float,
                       order_side: OrderSide):
        """Record actual slippage for model improvement"""
        
        if expected_price > 0:
            slippage = abs(actual_price - expected_price) / expected_price
            self.slippage_history.append(slippage)

# ============ Price Calculator ============
class PriceCalculator:
    """Calculates optimal prices for orders"""
    
    def __init__(self, config: OrderManagerConfig):
        self.config = config
    
    def calculate_limit_price(self,
                            order_side: OrderSide,
                            current_price: float,
                            market_data: pd.DataFrame) -> float:
        """Calculate optimal limit price"""
        
        base_price = current_price
        
        # Apply offset based on order side
        if order_side == OrderSide.BUY:
            offset = -self.config.limit_price_offset_percent / 100
        else:
            offset = self.config.limit_price_offset_percent / 100
        
        # Adjust for market conditions
        offset = self._adjust_offset_for_conditions(offset, market_data)
        
        limit_price = base_price * (1 + offset)
        
        return limit_price
    
    def calculate_market_price(self,
                             order_side: OrderSide,
                             current_price: float,
                             market_data: pd.DataFrame) -> float:
        """Calculate expected market price (with premium)"""
        
        base_price = current_price
        
        # Apply premium for market orders
        if order_side == OrderSide.BUY:
            premium = self.config.market_order_premium_percent / 100
        else:
            premium = -self.config.market_order_premium_percent / 100
        
        # Adjust premium based on market conditions
        premium = self._adjust_premium_for_conditions(premium, market_data)
        
        expected_price = base_price * (1 + premium)
        
        return expected_price
    
    def _adjust_offset_for_conditions(self,
                                    base_offset: float,
                                    market_data: pd.DataFrame) -> float:
        """Adjust limit price offset based on market conditions"""
        
        if len(market_data) < 20:
            return base_offset
        
        # Adjust based on volatility
        returns = market_data['close'].pct_change().dropna()
        if len(returns) >= 20:
            volatility = returns.std()
            # Increase offset in high volatility
            volatility_factor = 1.0 + (volatility / 0.01) * 0.5
            base_offset *= volatility_factor
        
        # Adjust based on trend
        trend = self._calculate_price_trend(market_data)
        if abs(base_offset) > 0:
            # Increase offset when going against trend
            if (base_offset > 0 and trend < 0) or (base_offset < 0 and trend > 0):
                base_offset *= 1.2  # 20% larger offset
        
        return base_offset
    
    def _adjust_premium_for_conditions(self,
                                      base_premium: float,
                                      market_data: pd.DataFrame) -> float:
        """Adjust market order premium based on market conditions"""
        
        if len(market_data) < 20:
            return base_premium
        
        # Adjust based on volatility
        returns = market_data['close'].pct_change().dropna()
        if len(returns) >= 20:
            volatility = returns.std()
            # Increase premium in high volatility
            volatility_factor = 1.0 + (volatility / 0.01) * 0.3
            base_premium *= volatility_factor
        
        # Adjust based on liquidity
        if 'volume' in market_data.columns:
            volume_ratio = self._calculate_volume_ratio(market_data)
            # Decrease premium in high liquidity
            liquidity_factor = 1.0 / (1.0 + volume_ratio)
            base_premium *= liquidity_factor
        
        return base_premium
    
    def _calculate_price_trend(self, market_data: pd.DataFrame) -> float:
        """Calculate short-term price trend"""
        if len(market_data) < 10:
            return 0.0
        
        recent_prices = market_data['close'].iloc[-10:]
        if len(recent_prices) >= 2:
            # Simple linear regression for trend
            x = np.arange(len(recent_prices))
            y = recent_prices.values
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize by average price
            avg_price = np.mean(y)
            if avg_price > 0:
                return slope / avg_price
        
        return 0.0
    
    def _calculate_volume_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate current volume relative to average"""
        if len(market_data) < 20 or 'volume' not in market_data.columns:
            return 1.0
        
        current_volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1]
        
        if avg_volume > 0:
            return current_volume / avg_volume
        
        return 1.0

# ============ Order Execution Engine ============
class OrderExecutionEngine:
    """Handles order execution and monitoring"""
    
    def __init__(self, config: OrderManagerConfig):
        self.config = config
        self.order_creator = OrderCreator(config)
        self.active_orders: Dict[str, Order] = {}
        self.order_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.execution_queue = asyncio.Queue()
        self.monitoring_tasks = {}
        self.logger = get_logger(__name__)
        
    async def execute_order(self,
                          order_request: OrderRequest,
                          trading_pair: str = "BTCUSDT") -> Order:
        """Execute an order request"""
        
        # Validate order request
        self._validate_order_request(order_request)
        
        # Create order object
        order = self._create_order_from_request(order_request, trading_pair)
        
        # Store order
        self.active_orders[order.order_id] = order
        
        # Send to exchange (simulated for now)
        exchange_order = await self._send_to_exchange(order)
        
        # Update order with exchange response
        order.status = OrderStatus.SENT
        order.metadata['exchange_order_id'] = exchange_order.get('id')
        order.updated_at = datetime.now()
        
        # Start monitoring
        await self._start_order_monitoring(order)
        
        self.logger.info(f"Order {order.order_id} sent to exchange: {order.order_side.value} "
                        f"{order.quantity} {trading_pair} at {order.price}")
        
        return order
    
    def _validate_order_request(self, request: OrderRequest):
        """Validate order request"""
        if request.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and request.price is None:
            raise ValueError(f"{request.order_type.value} orders require price")
        
        if request.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and request.stop_price is None:
            raise ValueError(f"{request.order_type.value} orders require stop_price")
        
        # Check order size limits
        if request.quantity < self.config.min_order_size:
            raise ValueError(f"Order quantity below minimum: {self.config.min_order_size}")
        
        if request.quantity > self.config.max_order_size:
            raise ValueError(f"Order quantity above maximum: {self.config.max_order_size}")
    
    def _create_order_from_request(self,
                                  request: OrderRequest,
                                  trading_pair: str) -> Order:
        """Create Order object from OrderRequest"""
        
        return Order(
            order_id=str(uuid.uuid4()),
            client_order_id=request.client_order_id,
            order_type=request.order_type,
            order_side=request.order_side,
            trading_pair=trading_pair,
            quantity=request.quantity,
            price=request.price,
            stop_price=request.stop_price,
            time_in_force=request.time_in_force,
            status=OrderStatus.PENDING,
            metadata=request.metadata
        )
    
    async def _send_to_exchange(self, order: Order) -> Dict[str, Any]:
        """Send order to exchange (simulated)"""
        # Simulate API call delay
        await asyncio.sleep(self.config.api_rate_limit_delay)
        
        # Simulate exchange response
        exchange_response = {
            'id': f'exch_{uuid.uuid4().hex[:10]}',
            'clientOrderId': order.client_order_id,
            'symbol': order.trading_pair,
            'side': order.order_side.value,
            'type': order.order_type.value,
            'quantity': order.quantity,
            'price': order.price,
            'stopPrice': order.stop_price,
            'timeInForce': order.time_in_force.value,
            'status': 'new'
        }
        
        return exchange_response
    
    async def _start_order_monitoring(self, order: Order):
        """Start monitoring order status"""
        task = asyncio.create_task(self._monitor_order(order))
        self.monitoring_tasks[order.order_id] = task
    
    async def _monitor_order(self, order: Order):
        """Monitor order status until completion"""
        
        try:
            while order.is_open:
                # Check order status
                status = await self._check_order_status(order)
                
                if status != order.status:
                    # Status changed
                    order.status = status
                    order.updated_at = datetime.now()
                    
                    # Execute callbacks
                    await self._execute_callbacks(order)
                    
                    # Log status change
                    self.logger.info(f"Order {order.order_id} status changed to {status.value}")
                
                # Check if order is stale
                if self.config.auto_cancel_stale_orders:
                    await self._check_stale_order(order)
                
                # Wait before next check
                await asyncio.sleep(5)  # Check every 5 seconds
            
            # Order is closed, clean up
            if order.order_id in self.monitoring_tasks:
                del self.monitoring_tasks[order.order_id]
            
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
            
        except Exception as e:
            self.logger.error(f"Error monitoring order {order.order_id}: {str(e)}")
    
    async def _check_order_status(self, order: Order) -> OrderStatus:
        """Check order status from exchange"""
        # Simulate status check
        await asyncio.sleep(1)
        
        # Simulate random status updates for demonstration
        # In reality, this would query the exchange API
        
        if order.status == OrderStatus.SENT:
            # Simulate order being filled
            if np.random.random() < 0.3:  # 30% chance of fill
                return OrderStatus.FILLED
            elif np.random.random() < 0.1:  # 10% chance of partial fill
                return OrderStatus.PARTIALLY_FILLED
        
        return order.status
    
    async def _check_stale_order(self, order: Order):
        """Check if order is stale and cancel if needed"""
        time_since_update = (datetime.now() - order.updated_at).total_seconds() / 60
        
        if time_since_update > self.config.stale_order_threshold_minutes:
            self.logger.warning(f"Order {order.order_id} is stale, cancelling")
            await self.cancel_order(order.order_id)
    
    async def _execute_callbacks(self, order: Order):
        """Execute registered callbacks for order"""
        if order.order_id in self.order_callbacks:
            for callback in self.order_callbacks[order.order_id]:
                try:
                    await callback(order)
                except Exception as e:
                    self.logger.error(f"Error executing callback for order {order.order_id}: {str(e)}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.active_orders:
            self.logger.warning(f"Order {order_id} not found for cancellation")
            return False
        
        order = self.active_orders[order_id]
        
        if order.is_closed:
            self.logger.warning(f"Order {order_id} is already closed")
            return False
        
        # Simulate exchange cancellation
        await asyncio.sleep(self.config.api_rate_limit_delay)
        
        # Update order status
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        
        # Execute callbacks
        await self._execute_callbacks(order)
        
        self.logger.info(f"Order {order_id} cancelled")
        
        return True
    
    async def modify_order(self,
                          order_id: str,
                          new_price: Optional[float] = None,
                          new_quantity: Optional[float] = None) -> bool:
        """Modify an existing order"""
        if order_id not in self.active_orders:
            self.logger.warning(f"Order {order_id} not found for modification")
            return False
        
        order = self.active_orders[order_id]
        
        if order.is_closed:
            self.logger.warning(f"Order {order_id} is closed, cannot modify")
            return False
        
        # Cancel old order
        await self.cancel_order(order_id)
        
        # Create new order with modifications
        new_request = OrderRequest(
            order_side=order.order_side,
            quantity=new_quantity if new_quantity else order.quantity,
            order_type=order.order_type,
            price=new_price if new_price else order.price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force,
            client_order_id=f"{order.client_order_id}_mod",
            metadata={**order.metadata, 'modified_from': order.order_id}
        )
        
        # Execute new order
        await self.execute_order(new_request, order.trading_pair)
        
        self.logger.info(f"Order {order_id} modified")
        
        return True
    
    def register_callback(self,
                         order_id: str,
                         callback: Callable):
        """Register callback for order status changes"""
        self.order_callbacks[order_id].append(callback)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.active_orders.get(order_id)
    
    def get_all_orders(self) -> List[Order]:
        """Get all active orders"""
        return list(self.active_orders.values())

# ============ Order Risk Manager ============
class OrderRiskManager:
    """Manages risk for orders and trading"""
    
    def __init__(self, config: OrderManagerConfig):
        self.config = config
        self.risk_analyzer = RiskAnalyzer()
        self.stop_loss_manager = StopLossManager()
        self.logger = get_logger(__name__)
        
        # Risk limits
        self.daily_order_count = 0
        self.daily_order_value = 0.0
        self.last_reset_date = datetime.now().date()
        
    def check_order_risk(self,
                        order_request: OrderRequest,
                        current_positions: List[Any],
                        market_data: pd.DataFrame) -> Tuple[bool, str]:
        """Check if order meets risk requirements"""
        
        # Reset daily counters if new day
        self._reset_daily_counters()
        
        # Check daily order limit
        if self.daily_order_count >= self.config.daily_order_limit:
            return False, f"Daily order limit reached: {self.config.daily_order_limit}"
        
        # Check order size limits
        if not self._check_order_size(order_request):
            return False, "Order size exceeds limits"
        
        # Check position concentration
        if not self._check_position_concentration(order_request, current_positions):
            return False, "Position concentration too high"
        
        # Check market conditions
        if not self._check_market_conditions(order_request, market_data):
            return False, "Market conditions not favorable"
        
        # Check volatility
        if not self._check_volatility(order_request, market_data):
            return False, "Market volatility too high"
        
        # Update daily counters
        self.daily_order_count += 1
        # Estimate order value for daily limit
        order_value = self._estimate_order_value(order_request, market_data)
        self.daily_order_value += order_value
        
        return True, "Order passed risk checks"
    
    def _reset_daily_counters(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_order_count = 0
            self.daily_order_value = 0.0
            self.last_reset_date = today
    
    def _check_order_size(self, order_request: OrderRequest) -> bool:
        """Check order size against limits"""
        
        # Check against absolute limits
        if order_request.quantity < self.config.min_order_size:
            self.logger.warning(f"Order quantity below minimum: {order_request.quantity} < {self.config.min_order_size}")
            return False
        
        if order_request.quantity > self.config.max_order_size:
            self.logger.warning(f"Order quantity above maximum: {order_request.quantity} > {self.config.max_order_size}")
            return False
        
        # Check against value limit
        # This requires price information
        if order_request.price and order_request.quantity * order_request.price > self.config.max_order_value:
            self.logger.warning(f"Order value above maximum: {order_request.quantity * order_request.price} > {self.config.max_order_value}")
            return False
        
        return True
    
    def _check_position_concentration(self,
                                    order_request: OrderRequest,
                                    current_positions: List[Any]) -> bool:
        """Check position concentration risk"""
        
        # This is a simplified check
        # In reality, you'd calculate total exposure and compare to limits
        
        # Count current positions in same direction
        same_direction_positions = sum(
            1 for pos in current_positions 
            if hasattr(pos, 'side') and pos.side == order_request.order_side.value
        )
        
        # Limit number of positions in same direction
        max_same_direction = self.config.max_open_orders // 2
        if same_direction_positions >= max_same_direction:
            self.logger.warning(f"Too many positions in {order_request.order_side.value} direction: {same_direction_positions}")
            return False
        
        return True
    
    def _check_market_conditions(self,
                                order_request: OrderRequest,
                                market_data: pd.DataFrame) -> bool:
        """Check market conditions for order"""
        
        if len(market_data) < 20:
            return True  # Not enough data, proceed with caution
        
        # Check for extreme volatility
        returns = market_data['close'].pct_change().dropna()
        if len(returns) >= 20:
            volatility = returns.std()
            if volatility > 0.05:  # 5% volatility threshold
                self.logger.warning(f"High volatility detected: {volatility:.2%}")
                # Allow orders but might want to reduce size
                return True  # Still allow, but with caution
        
        # Check for gaps (overnight or weekend gaps)
        price_gap = self._check_price_gaps(market_data)
        if price_gap > 0.02:  # 2% gap threshold
            self.logger.warning(f"Large price gap detected: {price_gap:.2%}")
            # Might want to wait for price to stabilize
            return False
        
        return True
    
    def _check_price_gaps(self, market_data: pd.DataFrame) -> float:
        """Check for price gaps in market data"""
        if len(market_data) < 2:
            return 0.0
        
        # Calculate gap between close and next open
        gaps = []
        for i in range(1, len(market_data)):
            prev_close = market_data['close'].iloc[i-1]
            curr_open = market_data['open'].iloc[i]
            
            if prev_close > 0:
                gap = abs(curr_open - prev_close) / prev_close
                gaps.append(gap)
        
        if gaps:
            return max(gaps)
        
        return 0.0
    
    def _check_volatility(self,
                         order_request: OrderRequest,
                         market_data: pd.DataFrame) -> bool:
        """Check volatility conditions"""
        
        if len(market_data) < 20:
            return True
        
        returns = market_data['close'].pct_change().dropna()
        if len(returns) < 20:
            return True
        
        volatility = returns.std()
        
        # Adjust threshold based on order type
        if order_request.order_type == OrderType.MARKET:
            # Market orders are more sensitive to volatility
            threshold = 0.03  # 3% volatility threshold
        else:
            threshold = 0.05  # 5% volatility threshold for limit orders
        
        if volatility > threshold:
            self.logger.warning(f"High volatility for {order_request.order_type.value} order: {volatility:.2%}")
            return False
        
        return True
    
    def _estimate_order_value(self,
                            order_request: OrderRequest,
                            market_data: pd.DataFrame) -> float:
        """Estimate order value for daily limits"""
        
        if order_request.price:
            return order_request.quantity * order_request.price
        elif len(market_data) > 0:
            current_price = market_data['close'].iloc[-1]
            return order_request.quantity * current_price
        else:
            return 0.0

# ============ Main Order Manager ============
class BitcoinOrderManager(BaseOrderManager):
    """Main order manager for Bitcoin trading"""
    
    def __init__(self, config: Optional[OrderManagerConfig] = None):
        super().__init__(config)
        
        # Initialize components
        self.order_creator = OrderCreator(self.config)
        self.execution_engine = OrderExecutionEngine(self.config)
        self.risk_manager = OrderRiskManager(self.config)
        
        # Order tracking
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.cancelled_orders: Dict[str, Order] = {}
        
        # Statistics
        self.trade_statistics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitor_orders())
        
    async def create_order_from_signal(self,
                                      signal: TradingSignal,
                                      position_size: PositionSizeResult,
                                      market_data: pd.DataFrame) -> Optional[Order]:
        """Create and execute order from trading signal"""
        
        try:
            # Create order request
            order_request = self.order_creator.create_order_from_signal(
                signal, position_size, market_data
            )
            
            if order_request is None:
                self.logger.warning("No order request created from signal")
                return None
            
            # Check risk before proceeding
            risk_ok, risk_message = self.risk_manager.check_order_risk(
                order_request, self.get_open_orders(), market_data
            )
            
            if not risk_ok:
                self.logger.warning(f"Order rejected by risk manager: {risk_message}")
                return None
            
            # Create bracket orders if enabled
            if self.config.use_bracket_orders:
                order_requests = self.order_creator.create_bracket_order(
                    order_request, market_data
                )
            else:
                order_requests = [order_request]
            
            # Execute orders
            orders = []
            for req in order_requests:
                order = await self.execution_engine.execute_order(
                    req, self.config.trading_pair
                )
                
                # Register callback for order updates
                self.execution_engine.register_callback(
                    order.order_id, self._handle_order_update
                )
                
                # Store order
                self.orders[order.order_id] = order
                self.pending_orders[order.order_id] = order
                
                orders.append(order)
                
                # Log order creation
                self.logger.info(f"Created order {order.order_id}: {order.order_side.value} "
                               f"{order.order_type.value} {order.quantity} {self.config.trading_pair}")
            
            # Update statistics
            self.order_statistics['orders_created'] += len(orders)
            
            # Return main order (first in list)
            return orders[0] if orders else None
            
        except Exception as e:
            self.logger.error(f"Error creating order from signal: {str(e)}")
            return None
    
    async def _handle_order_update(self, order: Order):
        """Handle order status updates"""
        
        # Update order in our tracking
        self.orders[order.order_id] = order
        
        # Move between tracking dictionaries based on status
        if order.status == OrderStatus.FILLED:
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]
            self.filled_orders[order.order_id] = order
            
            # Record execution
            await self._record_order_execution(order)
            
        elif order.status == OrderStatus.CANCELLED:
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]
            self.cancelled_orders[order.order_id] = order
        
        # Save order history if configured
        if self.config.save_order_history:
            await self._save_order_to_history(order)
    
    async def _record_order_execution(self, order: Order):
        """Record order execution details"""
        
        # Create execution record
        execution = OrderExecution(
            execution_id=str(uuid.uuid4()),
            order_id=order.order_id,
            quantity=order.filled_quantity,
            price=order.average_fill_price or order.price or 0.0,
            fee=order.fee,
            fee_currency="USD",
            timestamp=order.updated_at,
            exchange_order_id=order.metadata.get('exchange_order_id'),
            metadata=order.metadata
        )
        
        # Record execution
        self.record_execution(execution)
        
        # Update trade statistics
        self._update_trade_statistics(order, execution)
        
        self.logger.info(f"Order {order.order_id} filled: {order.filled_quantity} at "
                       f"{execution.price}, fee: {execution.fee}")
    
    def _update_trade_statistics(self, order: Order, execution: OrderExecution):
        """Update trade statistics"""
        self.trade_statistics['total_trades'] += 1
        
        # Simple profit calculation
        # In reality, you'd track entry and exit prices
        if 'signal_price' in order.metadata:
            signal_price = order.metadata['signal_price']
            if signal_price > 0 and execution.price > 0:
                if order.order_side == OrderSide.BUY:
                    # For buys, profit if execution price < signal price
                    profit_pct = (signal_price - execution.price) / signal_price
                else:
                    # For sells, profit if execution price > signal price
                    profit_pct = (execution.price - signal_price) / signal_price
                
                profit = profit_pct * execution.quantity * execution.price
                
                self.trade_statistics['total_profit'] += profit
                
                if profit > 0:
                    self.trade_statistics['winning_trades'] += 1
                    self.trade_statistics['largest_win'] = max(
                        self.trade_statistics['largest_win'], profit
                    )
                else:
                    self.trade_statistics['losing_trades'] += 1
                    self.trade_statistics['largest_loss'] = min(
                        self.trade_statistics['largest_loss'], profit
                    )
    
    async def _save_order_to_history(self, order: Order):
        """Save order to history file"""
        try:
            # Create directory if it doesn't exist
            history_dir = Path(self.config.order_history_path)
            history_dir.mkdir(parents=True, exist_ok=True)
            
            # Save order to JSON file
            date_str = order.created_at.strftime("%Y%m%d")
            filename = f"orders_{date_str}.json"
            filepath = history_dir / filename
            
            # Load existing orders if file exists
            orders = []
            if filepath.exists():
                with open(filepath, 'r') as f:
                    orders = json.load(f)
            
            # Add new order
            orders.append(order.to_dict())
            
            # Save back to file
            with open(filepath, 'w') as f:
                json.dump(orders, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error saving order to history: {str(e)}")
    
    async def cancel_all_orders(self) -> List[str]:
        """Cancel all open orders"""
        cancelled_ids = []
        
        for order_id, order in list(self.pending_orders.items()):
            if order.is_open:
                success = await self.execution_engine.cancel_order(order_id)
                if success:
                    cancelled_ids.append(order_id)
        
        self.logger.info(f"Cancelled {len(cancelled_ids)} orders")
        return cancelled_ids
    
    async def modify_order(self,
                          order_id: str,
                          new_price: Optional[float] = None,
                          new_quantity: Optional[float] = None) -> bool:
        """Modify an existing order"""
        return await self.execution_engine.modify_order(
            order_id, new_price, new_quantity
        )
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status"""
        order = self.orders.get(order_id)
        if order:
            return order.status
        
        # Check with execution engine
        order = self.execution_engine.get_order(order_id)
        if order:
            return order.status
        
        return OrderStatus.FAILED
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order statistics"""
        stats = {
            'total_orders': len(self.orders),
            'pending_orders': len(self.pending_orders),
            'filled_orders': len(self.filled_orders),
            'cancelled_orders': len(self.cancelled_orders),
            'order_statistics': dict(self.order_statistics),
            'execution_statistics': dict(self.execution_statistics),
            'trade_statistics': self.trade_statistics
        }
        
        return stats
    
    async def _monitor_orders(self):
        """Monitor all orders for timeouts and other issues"""
        while True:
            try:
                # Check for stale orders
                await self._check_stale_orders()
                
                # Check rate limits
                await self._check_rate_limits()
                
                # Update statistics
                self._update_statistics()
                
                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in order monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_stale_orders(self):
        """Check for stale orders and cancel if needed"""
        if not self.config.auto_cancel_stale_orders:
            return
        
        current_time = datetime.now()
        stale_threshold = timedelta(minutes=self.config.stale_order_threshold_minutes)
        
        for order_id, order in list(self.pending_orders.items()):
            if order.is_open:
                time_since_update = current_time - order.updated_at
                if time_since_update > stale_threshold:
                    self.logger.warning(f"Order {order_id} is stale, cancelling")
                    await self.execution_engine.cancel_order(order_id)
    
    async def _check_rate_limits(self):
        """Check and enforce rate limits"""
        # This would track API call rate and delay if needed
        # For now, just log if we're approaching limits
        pass
    
    def _update_statistics(self):
        """Update order statistics"""
        # Update various statistics
        self.order_statistics['open_orders'] = len(self.pending_orders)
        self.order_statistics['total_executions'] = len(self.executions)
        
        # Calculate fill rates
        total_orders = len(self.orders)
        if total_orders > 0:
            filled_orders = len(self.filled_orders)
            self.order_statistics['fill_rate'] = filled_orders / total_orders
        else:
            self.order_statistics['fill_rate'] = 0.0
    
    async def shutdown(self):
        """Shutdown order manager"""
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all open orders
        await self.cancel_all_orders()
        
        self.logger.info("Order manager shutdown complete")

# ============ Factory Functions ============
def create_order_manager(config: Optional[Dict] = None) -> BitcoinOrderManager:
    """Factory function to create an order manager"""
    if config:
        order_config = OrderManagerConfig(**config)
    else:
        order_config = OrderManagerConfig()
    
    return BitcoinOrderManager(order_config)

def load_order_config(config_path: Path) -> OrderManagerConfig:
    """Load order management configuration from YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return OrderManagerConfig(**config_dict.get('order_management', {}))
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {str(e)}")
        return OrderManagerConfig()

# ============ Utility Functions ============
async def execute_trade_signal(signal: TradingSignal,
                              position_size: PositionSizeResult,
                              market_data: pd.DataFrame,
                              order_manager: Optional[BitcoinOrderManager] = None) -> Optional[Order]:
    """Utility function to execute a trade signal"""
    
    if order_manager is None:
        order_manager = create_order_manager()
    
    try:
        order = await order_manager.create_order_from_signal(
            signal, position_size, market_data
        )
        
        if order:
            logger.info(f"Trade executed: {order.order_side.value} {order.quantity} "
                       f"{order.trading_pair} at {order.price}")
        
        return order
        
    except Exception as e:
        logger.error(f"Error executing trade signal: {str(e)}")
        return None

async def cancel_all_open_orders(order_manager: Optional[BitcoinOrderManager] = None):
    """Cancel all open orders"""
    if order_manager is None:
        order_manager = create_order_manager()
    
    cancelled = await order_manager.cancel_all_orders()
    logger.info(f"Cancelled {len(cancelled)} orders")
    return cancelled

def get_order_summary(order_manager: BitcoinOrderManager) -> Dict[str, Any]:
    """Get summary of order manager state"""
    stats = order_manager.get_order_statistics()
    
    summary = {
        'orders': {
            'total': stats['total_orders'],
            'open': stats['pending_orders'],
            'filled': stats['filled_orders'],
            'cancelled': stats['cancelled_orders']
        },
        'performance': {
            'fill_rate': stats['order_statistics'].get('fill_rate', 0),
            'total_trades': stats['trade_statistics']['total_trades'],
            'win_rate': (stats['trade_statistics']['winning_trades'] / 
                        stats['trade_statistics']['total_trades'] 
                        if stats['trade_statistics']['total_trades'] > 0 else 0),
            'total_profit': stats['trade_statistics']['total_profit']
        }
    }
    
    return summary

# ============ Example Usage ============
async def example_usage():
    """Example usage of order manager"""
    print("Order Manager Example")
    print("=" * 50)
    
    # Create a sample signal
    from core.trading.signal_generator import TradingSignal, SignalType, SignalSource
    from core.trading.position_sizer import PositionSizeResult, PositionSizeUnit, PositionSizingMethod
    
    signal = TradingSignal(
        timestamp=datetime.now(),
        signal_type=SignalType.BUY,
        strength=0.8,
        confidence=0.75,
        price=45000.0,
        source=SignalSource.TECHNICAL,
        metadata={'indicator': 'RSI', 'rsi_value': 30}
    )
    
    # Create sample position size
    position_size = PositionSizeResult(
        position_size=0.02,  # 2% of portfolio
        size_unit=PositionSizeUnit.PERCENTAGE,
        position_value=2000.0,  # $2000
        risk_amount=40.0,  # $40 risk
        risk_percentage=0.02,  # 2% risk
        sizing_method=PositionSizingMethod.VOLATILITY_ADJUSTED,
        confidence=0.7,
        metadata={'portfolio_value': 100000.0}
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
    
    print(f"Created sample signal: {signal.signal_type.value} at ${signal.price:.2f}")
    print(f"Position size: {position_size.position_size:.2%} (${position_size.position_value:.2f})")
    
    # Create order manager
    config = {
        'default_order_type': OrderType.LIMIT,
        'use_bracket_orders': True,
        'trading_pair': 'BTCUSDT'
    }
    
    order_manager = create_order_manager(config)
    
    # Execute order
    print("\n1. Creating order from signal...")
    order = await order_manager.create_order_from_signal(
        signal, position_size, market_data
    )
    
    if order:
        print(f"Order created: {order.order_id}")
        print(f"  Type: {order.order_type.value}")
        print(f"  Side: {order.order_side.value}")
        print(f"  Quantity: {order.quantity:.6f} BTC")
        print(f"  Price: ${order.price:.2f}" if order.price else "  Price: Market")
        print(f"  Status: {order.status.value}")
    
    # Wait a bit for order processing
    await asyncio.sleep(2)
    
    # Get order statistics
    print("\n2. Order Statistics:")
    stats = order_manager.get_order_statistics()
    print(f"  Total orders: {stats['total_orders']}")
    print(f"  Pending orders: {stats['pending_orders']}")
    print(f"  Filled orders: {stats['filled_orders']}")
    
    # Get open orders
    print("\n3. Open Orders:")
    open_orders = order_manager.get_open_orders()
    for o in open_orders[:3]:  # Show first 3
        print(f"  {o.order_id}: {o.order_side.value} {o.quantity:.6f} at "
              f"${o.price:.2f if o.price else 'Market'}")
    
    # Simulate some time passing
    print("\n4. Simulating order fills...")
    await asyncio.sleep(1)
    
    # Cancel all orders
    print("\n5. Cancelling all orders...")
    cancelled = await order_manager.cancel_all_orders()
    print(f"  Cancelled {len(cancelled)} orders")
    
    # Get summary
    print("\n6. Order Manager Summary:")
    summary = get_order_summary(order_manager)
    print(f"  Total trades: {summary['performance']['total_trades']}")
    print(f"  Fill rate: {summary['performance']['fill_rate']:.2%}")
    
    # Shutdown
    print("\n7. Shutting down order manager...")
    await order_manager.shutdown()
    
    print("\n" + "="*50)
    print("Example completed")
    print("="*50)
    
    return order_manager, order

# ============ Main Execution ============
async def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bitcoin Trading AI - Order Management')
    parser.add_argument('--signal', type=str, required=True,
                       help='Signal file path (JSON)')
    parser.add_argument('--position', type=str, required=True,
                       help='Position size file path (JSON)')
    parser.add_argument('--market_data', type=str, required=True,
                       help='Market data file path')
    parser.add_argument('--config', type=str, default='config/order_management.yaml',
                       help='Order management configuration file')
    parser.add_argument('--action', type=str, choices=['execute', 'cancel', 'status', 'summary'],
                       default='execute', help='Action to perform')
    parser.add_argument('--order_id', type=str,
                       help='Order ID for cancel/status actions')
    parser.add_argument('--output', type=str,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if config_path.exists():
            order_config = load_order_config(config_path)
        else:
            order_config = OrderManagerConfig()
            logger.info(f"Using default configuration, config file not found: {config_path}")
        
        # Create order manager
        order_manager = create_order_manager(order_config.__dict__)
        
        if args.action == 'execute':
            # Load signal
            signal_path = Path(args.signal)
            if not signal_path.exists():
                raise FileNotFoundError(f"Signal file not found: {signal_path}")
            
            with open(signal_path, 'r') as f:
                signal_data = json.load(f)
            
            from core.trading.signal_generator import TradingSignal
            signal = TradingSignal.from_dict(signal_data)
            
            # Load position size
            position_path = Path(args.position)
            if not position_path.exists():
                raise FileNotFoundError(f"Position file not found: {position_path}")
            
            with open(position_path, 'r') as f:
                position_data = json.load(f)
            
            from core.trading.position_sizer import PositionSizeResult
            position_size = PositionSizeResult.from_dict(position_data)
            
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
            
            print(f"Executing trade signal:")
            print(f"  Signal: {signal.signal_type.value} at ${signal.price:.2f}")
            print(f"  Position size: ${position_size.position_value:.2f}")
            print(f"  Risk amount: ${position_size.risk_amount:.2f}")
            print(f"  Market data: {len(market_data)} periods")
            
            # Execute order
            order = await order_manager.create_order_from_signal(
                signal, position_size, market_data
            )
            
            if order:
                print(f"\nOrder executed successfully:")
                print(f"  Order ID: {order.order_id}")
                print(f"  Type: {order.order_type.value}")
                print(f"  Side: {order.order_side.value}")
                print(f"  Quantity: {order.quantity:.6f} {order_config.trading_pair[:3]}")
                print(f"  Price: ${order.price:.2f}" if order.price else "  Price: Market")
                print(f"  Status: {order.status.value}")
            else:
                print("\nFailed to execute order")
        
        elif args.action == 'cancel':
            if args.order_id:
                # Cancel specific order
                success = await order_manager.execution_engine.cancel_order(args.order_id)
                if success:
                    print(f"Order {args.order_id} cancelled successfully")
                else:
                    print(f"Failed to cancel order {args.order_id}")
            else:
                # Cancel all orders
                cancelled = await order_manager.cancel_all_orders()
                print(f"Cancelled {len(cancelled)} orders")
        
        elif args.action == 'status':
            if args.order_id:
                status = await order_manager.get_order_status(args.order_id)
                print(f"Order {args.order_id} status: {status.value}")
            else:
                print("Please provide --order_id for status check")
        
        elif args.action == 'summary':
            summary = get_order_summary(order_manager)
            
            print("\nOrder Manager Summary:")
            print("="*50)
            
            print("\nOrder Statistics:")
            orders = summary['orders']
            print(f"  Total Orders: {orders['total']}")
            print(f"  Open Orders: {orders['open']}")
            print(f"  Filled Orders: {orders['filled']}")
            print(f"  Cancelled Orders: {orders['cancelled']}")
            
            print("\nPerformance:")
            perf = summary['performance']
            print(f"  Fill Rate: {perf['fill_rate']:.2%}")
            print(f"  Total Trades: {perf['total_trades']}")
            print(f"  Win Rate: {perf['win_rate']:.2%}")
            print(f"  Total Profit: ${perf['total_profit']:.2f}")
        
        # Save results if output directory specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save order manager state
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state_file = output_dir / f"order_manager_state_{timestamp}.json"
            order_manager.save_state(state_file)
            
            print(f"\nOrder manager state saved to: {state_file}")
        
        # Shutdown
        await order_manager.shutdown()
        
        print("\n" + "="*50)
        print("Order management completed")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())