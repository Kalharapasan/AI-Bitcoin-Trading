"""
Execution engine module for Bitcoin trading AI.
Handles order execution, trade routing, slippage management, and execution optimization.
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
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random

# Import project modules
from config.settings import TradingSettings, ExchangeSettings, AppConstants
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.trading.order_manager import Order, OrderType, OrderSide, OrderStatus, OrderTimeInForce
from core.risk_management.risk_analyzer import RiskAnalyzer
from core.utils.cache import Cache
from core.trading.position_sizer import PositionSizeResult

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Execution Types and Enums ============
class ExecutionStrategy(str, Enum):
    """Execution strategies for order filling"""
    AGGRESSIVE = "aggressive"      # Fill immediately at any price
    PASSIVE = "passive"            # Wait for best price, may not fill
    TWAP = "twap"                  # Time-Weighted Average Price
    VWAP = "vwap"                  # Volume-Weighted Average Price
    ICEBERG = "iceberg"            # Large order split into smaller pieces
    DARK_POOL = "dark_pool"        # Execute in dark pools if available
    SMART_ROUTING = "smart_routing" # Route to best available exchange
    HYBRID = "hybrid"              # Combination of strategies

class ExecutionVenue(str, Enum):
    """Execution venues/exchanges"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BITFINEX = "bitfinex"
    HUOBI = "huobi"
    BYBIT = "bybit"
    OKX = "okx"
    DARK_POOL = "dark_pool"
    AGGREGATOR = "aggregator"

class ExecutionStatus(str, Enum):
    """Execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PARTIALLY_FILLED = "partially_filled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REJECTED = "rejected"

class SlippageType(str, Enum):
    """Types of slippage"""
    POSITIVE = "positive"      # Better than expected price
    NEGATIVE = "negative"      # Worse than expected price
    NONE = "none"              # No slippage

# ============ Configuration ============
@dataclass
class ExecutionConfig:
    """Configuration for execution engine"""
    
    # General settings
    default_execution_strategy: ExecutionStrategy = ExecutionStrategy.SMART_ROUTING
    enable_parallel_execution: bool = True
    max_parallel_orders: int = 5
    execution_timeout_seconds: int = 300  # 5 minutes
    
    # Exchange settings
    primary_exchange: ExecutionVenue = ExecutionVenue.BINANCE
    backup_exchanges: List[ExecutionVenue] = field(default_factory=lambda: [
        ExecutionVenue.COINBASE, ExecutionVenue.KRAKEN
    ])
    use_exchange_aggregator: bool = True
    max_exchange_latency_ms: int = 500
    
    # Slippage management
    max_allowed_slippage_percent: float = 0.5  # 0.5%
    target_slippage_percent: float = 0.1       # 0.1%
    use_slippage_forecast: bool = True
    slippage_lookback_periods: int = 100
    
    # Order splitting
    enable_order_splitting: bool = True
    max_split_parts: int = 10
    min_split_size_btc: float = 0.001
    split_time_interval_seconds: int = 5
    
    # Algorithmic execution
    enable_twap: bool = True
    twap_interval_seconds: int = 30
    enable_vwap: bool = True
    vwap_lookback_periods: int = 50
    
    # Iceberg orders
    enable_iceberg_orders: bool = True
    iceberg_display_size_btc: float = 0.1
    iceberg_replenish_rate_seconds: int = 10
    
    # Dark pool execution
    enable_dark_pool: bool = False
    dark_pool_min_size_btc: float = 1.0
    dark_pool_max_slippage_percent: float = 0.2
    
    # Smart routing
    enable_smart_routing: bool = True
    route_by_liquidity: bool = True
    route_by_fees: bool = True
    route_by_slippage: bool = True
    min_liquidity_btc: float = 10.0
    
    # Risk controls
    max_order_size_btc: float = 50.0
    max_daily_volume_btc: float = 1000.0
    max_position_size_btc: float = 200.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold_percent: float = 5.0
    
    # Performance optimization
    enable_pre_trade_analysis: bool = True
    enable_post_trade_analysis: bool = True
    cache_market_data: bool = True
    cache_ttl_seconds: int = 60
    
    # Monitoring and logging
    log_all_executions: bool = True
    save_execution_reports: bool = True
    execution_report_path: str = "data/executions/"
    real_time_monitoring: bool = True
    monitor_interval_seconds: int = 1
    
    # Advanced features
    use_machine_learning: bool = False
    ml_model_path: Optional[str] = None
    enable_predictive_execution: bool = False
    predictive_horizon_seconds: int = 60
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_allowed_slippage_percent < 0 or self.max_allowed_slippage_percent > 5:
            raise ValueError("max_allowed_slippage_percent must be between 0 and 5")
        
        if self.max_parallel_orders < 1 or self.max_parallel_orders > 20:
            raise ValueError("max_parallel_orders must be between 1 and 20")
        
        # Create execution report directory
        Path(self.execution_report_path).mkdir(parents=True, exist_ok=True)

# ============ Execution Data Structures ============
@dataclass
class ExecutionRequest:
    """Request for order execution"""
    request_id: str
    order_id: str
    trading_pair: str
    order_side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SMART_ROUTING
    client_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Generate request ID if not provided"""
        if not self.request_id:
            self.request_id = f"exec_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'request_id': self.request_id,
            'order_id': self.order_id,
            'trading_pair': self.trading_pair,
            'order_side': self.order_side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'execution_strategy': self.execution_strategy.value,
            'client_id': self.client_id,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class ExecutionResult:
    """Result of order execution"""
    execution_id: str
    request_id: str
    order_id: str
    trading_pair: str
    filled_quantity: float
    average_price: float
    total_fees: float
    execution_time: float  # seconds
    slippage_percent: float
    slippage_type: SlippageType
    execution_venue: ExecutionVenue
    execution_strategy: ExecutionStrategy
    status: ExecutionStatus
    fills: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate execution result"""
        if self.filled_quantity <= 0:
            raise ValueError("filled_quantity must be positive")
        
        if self.average_price <= 0:
            raise ValueError("average_price must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'execution_id': self.execution_id,
            'request_id': self.request_id,
            'order_id': self.order_id,
            'trading_pair': self.trading_pair,
            'filled_quantity': self.filled_quantity,
            'average_price': self.average_price,
            'total_fees': self.total_fees,
            'execution_time': self.execution_time,
            'slippage_percent': self.slippage_percent,
            'slippage_type': self.slippage_type.value,
            'execution_venue': self.execution_venue.value,
            'execution_strategy': self.execution_strategy.value,
            'status': self.status.value,
            'fills': self.fills,
            'metadata': self.metadata,
            'completed_at': self.completed_at.isoformat()
        }

@dataclass
class MarketSnapshot:
    """Snapshot of market conditions"""
    timestamp: datetime
    trading_pair: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume_24h: float
    spread_percent: float
    order_book_depth: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def liquidity_score(self) -> float:
        """Calculate liquidity score (0-1)"""
        # Simple liquidity score based on bid/ask sizes
        total_size = self.bid_size + self.ask_size
        if total_size > 100:  # 100 BTC
            return 1.0
        elif total_size > 10:  # 10 BTC
            return 0.8
        elif total_size > 1:   # 1 BTC
            return 0.5
        else:
            return 0.2

@dataclass
class ExecutionMetrics:
    """Execution performance metrics"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_volume_btc: float = 0.0
    total_fees: float = 0.0
    avg_slippage_percent: float = 0.0
    avg_execution_time: float = 0.0
    best_execution: Optional[ExecutionResult] = None
    worst_execution: Optional[ExecutionResult] = None
    venue_distribution: Dict[str, int] = field(default_factory=dict)
    strategy_distribution: Dict[str, int] = field(default_factory=dict)
    
    def update(self, result: ExecutionResult):
        """Update metrics with new execution result"""
        self.total_executions += 1
        
        if result.status == ExecutionStatus.COMPLETED:
            self.successful_executions += 1
            self.total_volume_btc += result.filled_quantity
            self.total_fees += result.total_fees
            self.avg_slippage_percent = (
                (self.avg_slippage_percent * (self.successful_executions - 1) + 
                 result.slippage_percent) / self.successful_executions
            )
            self.avg_execution_time = (
                (self.avg_execution_time * (self.successful_executions - 1) + 
                 result.execution_time) / self.successful_executions
            )
            
            # Update best/worst execution
            if self.best_execution is None or result.slippage_percent < self.best_execution.slippage_percent:
                self.best_execution = result
            if self.worst_execution is None or result.slippage_percent > self.worst_execution.slippage_percent:
                self.worst_execution = result
            
            # Update venue distribution
            venue = result.execution_venue.value
            self.venue_distribution[venue] = self.venue_distribution.get(venue, 0) + 1
            
            # Update strategy distribution
            strategy = result.execution_strategy.value
            self.strategy_distribution[strategy] = self.strategy_distribution.get(strategy, 0) + 1
        else:
            self.failed_executions += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'success_rate': self.successful_executions / self.total_executions if self.total_executions > 0 else 0,
            'total_volume_btc': self.total_volume_btc,
            'total_fees': self.total_fees,
            'avg_slippage_percent': self.avg_slippage_percent,
            'avg_execution_time': self.avg_execution_time,
            'venue_distribution': self.venue_distribution,
            'strategy_distribution': self.strategy_distribution
        }

# ============ Slippage Models ============
class SimpleSlippageModel:
    """Simple slippage estimation model"""
    
    def estimate(self, order_size: float, liquidity: float, spread: float) -> float:
        """Estimate slippage percentage"""
        size_ratio = min(order_size / liquidity, 1.0) if liquidity > 0 else 1.0
        base_slippage = spread * (1 + size_ratio * 2)
        return min(base_slippage, 5.0)  # Cap at 5%

class AdvancedSlippageModel:
    """Advanced slippage estimation model using historical data"""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.slippage_history = deque(maxlen=lookback_periods)
        self.volume_history = deque(maxlen=lookback_periods)
        
    def estimate(self, order_size: float, current_liquidity: float, 
                 spread: float, volatility: float) -> float:
        """Estimate slippage with advanced features"""
        
        if len(self.slippage_history) < 10:
            # Not enough data, use simple model
            simple_model = SimpleSlippageModel()
            return simple_model.estimate(order_size, current_liquidity, spread)
        
        # Calculate average historical slippage
        hist_avg = np.mean(list(self.slippage_history)[-50:]) if len(self.slippage_history) >= 50 else 0.1
        
        # Adjust for current conditions
        size_factor = min(order_size / current_liquidity, 2.0) if current_liquidity > 0 else 2.0
        spread_factor = spread * 100  # Convert to percentage
        volatility_factor = volatility * 2
        
        # Weighted combination
        estimated_slippage = (
            hist_avg * 0.4 +
            size_factor * 0.3 +
            spread_factor * 0.2 +
            volatility_factor * 0.1
        )
        
        return min(estimated_slippage, 10.0)  # Cap at 10%

# ============ Order Splitter ============
class OrderSplitter:
    """Splits large orders into smaller pieces"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def split_order(self,
                   request: ExecutionRequest,
                   market_snapshot: MarketSnapshot) -> List[ExecutionRequest]:
        """Split order into smaller pieces based on execution strategy"""
        
        if not self.config.enable_order_splitting:
            return [request]
        
        # Check if order should be split
        if not self._should_split_order(request, market_snapshot):
            return [request]
        
        # Determine split strategy based on execution strategy
        if request.execution_strategy == ExecutionStrategy.TWAP:
            return self._split_for_twap(request, market_snapshot)
        elif request.execution_strategy == ExecutionStrategy.VWAP:
            return self._split_for_vwap(request, market_snapshot)
        elif request.execution_strategy == ExecutionStrategy.ICEBERG:
            return self._split_for_iceberg(request, market_snapshot)
        elif request.execution_strategy == ExecutionStrategy.AGGRESSIVE:
            return self._split_for_aggressive(request, market_snapshot)
        else:
            return self._split_for_default(request, market_snapshot)
    
    def _should_split_order(self,
                           request: ExecutionRequest,
                           market_snapshot: MarketSnapshot) -> bool:
        """Determine if order should be split"""
        
        # Don't split very small orders
        if request.quantity < self.config.min_split_size_btc * 2:
            return False
        
        # Don't split market orders (they execute immediately)
        if request.order_type == OrderType.MARKET:
            return False
        
        # Split if order size is large relative to available liquidity
        available_liquidity = market_snapshot.bid_size if request.order_side == OrderSide.SELL else market_snapshot.ask_size
        if request.quantity > available_liquidity * 0.5:  # More than 50% of available liquidity
            return True
        
        # Split large orders
        if request.quantity > self.config.max_order_size_btc * 0.2:  # More than 20% of max order size
            return True
        
        return False
    
    def _split_for_twap(self,
                       request: ExecutionRequest,
                       market_snapshot: MarketSnapshot) -> List[ExecutionRequest]:
        """Split order for Time-Weighted Average Price execution"""
        
        # Determine number of splits based on order size and time interval
        num_splits = min(
            int(request.quantity / self.config.min_split_size_btc),
            self.config.max_split_parts,
            max(2, int(self.config.execution_timeout_seconds / self.config.twap_interval_seconds))
        )
        
        # Calculate split sizes (equal sizes for TWAP)
        split_sizes = self._calculate_equal_splits(request.quantity, num_splits)
        
        # Create split requests
        split_requests = []
        for i, size in enumerate(split_sizes):
            split_request = ExecutionRequest(
                request_id=f"{request.request_id}_split_{i}",
                order_id=request.order_id,
                trading_pair=request.trading_pair,
                order_side=request.order_side,
                order_type=request.order_type,
                quantity=size,
                price=request.price,
                stop_price=request.stop_price,
                time_in_force=request.time_in_force,
                execution_strategy=ExecutionStrategy.TWAP,
                client_id=request.client_id,
                metadata={
                    **request.metadata,
                    'split_index': i,
                    'total_splits': len(split_sizes),
                    'parent_request_id': request.request_id,
                    'execution_type': 'twap_split'
                }
            )
            split_requests.append(split_request)
        
        self.logger.info(f"Split order into {len(split_requests)} pieces for TWAP execution")
        
        return split_requests
    
    def _split_for_vwap(self,
                       request: ExecutionRequest,
                       market_snapshot: MarketSnapshot) -> List[ExecutionRequest]:
        """Split order for Volume-Weighted Average Price execution"""
        
        # VWAP splits are typically based on historical volume profile
        # For simplicity, we'll use a similar approach to TWAP but with variable sizes
        
        num_splits = min(
            int(request.quantity / self.config.min_split_size_btc),
            self.config.max_split_parts,
            20  # Reasonable maximum for VWAP
        )
        
        # Create volume-weighted splits (more volume at certain times)
        # This is a simplified model - in reality, you'd use historical volume data
        split_sizes = self._calculate_volume_weighted_splits(request.quantity, num_splits)
        
        split_requests = []
        for i, size in enumerate(split_sizes):
            split_request = ExecutionRequest(
                request_id=f"{request.request_id}_split_{i}",
                order_id=request.order_id,
                trading_pair=request.trading_pair,
                order_side=request.order_side,
                order_type=request.order_type,
                quantity=size,
                price=request.price,
                stop_price=request.stop_price,
                time_in_force=request.time_in_force,
                execution_strategy=ExecutionStrategy.VWAP,
                client_id=request.client_id,
                metadata={
                    **request.metadata,
                    'split_index': i,
                    'total_splits': len(split_sizes),
                    'parent_request_id': request.request_id,
                    'execution_type': 'vwap_split',
                    'volume_weight': split_sizes[i] / sum(split_sizes)
                }
            )
            split_requests.append(split_request)
        
        self.logger.info(f"Split order into {len(split_requests)} pieces for VWAP execution")
        
        return split_requests
    
    def _split_for_iceberg(self,
                          request: ExecutionRequest,
                          market_snapshot: MarketSnapshot) -> List[ExecutionRequest]:
        """Split order for Iceberg execution"""
        
        if not self.config.enable_iceberg_orders:
            return self._split_for_default(request, market_snapshot)
        
        # Iceberg orders show only a small part at a time
        display_size = min(
            self.config.iceberg_display_size_btc,
            request.quantity * 0.1  # Max 10% displayed
        )
        
        # Calculate number of iceberg slices
        num_slices = max(2, int(request.quantity / display_size))
        num_slices = min(num_slices, self.config.max_split_parts)
        
        # Create iceberg slices
        split_requests = []
        remaining_quantity = request.quantity
        
        for i in range(num_slices):
            # Last slice takes remaining quantity
            if i == num_slices - 1:
                slice_size = remaining_quantity
            else:
                slice_size = min(display_size, remaining_quantity)
            
            if slice_size <= 0:
                break
            
            slice_request = ExecutionRequest(
                request_id=f"{request.request_id}_iceberg_{i}",
                order_id=request.order_id,
                trading_pair=request.trading_pair,
                order_side=request.order_side,
                order_type=request.order_type,
                quantity=slice_size,
                price=request.price,
                stop_price=request.stop_price,
                time_in_force=OrderTimeInForce.IOC,  # Immediate or cancel for iceberg
                execution_strategy=ExecutionStrategy.ICEBERG,
                client_id=request.client_id,
                metadata={
                    **request.metadata,
                    'iceberg_index': i,
                    'total_slices': num_slices,
                    'parent_request_id': request.request_id,
                    'execution_type': 'iceberg',
                    'display_size': display_size,
                    'replenish_rate': self.config.iceberg_replenish_rate_seconds
                }
            )
            split_requests.append(slice_request)
            
            remaining_quantity -= slice_size
        
        self.logger.info(f"Split order into {len(split_requests)} iceberg slices")
        
        return split_requests
    
    def _split_for_aggressive(self,
                             request: ExecutionRequest,
                             market_snapshot: MarketSnapshot) -> List[ExecutionRequest]:
        """Split order for aggressive execution"""
        
        # Aggressive execution uses few, large splits
        num_splits = min(3, max(2, int(request.quantity / self.config.min_split_size_btc)))
        
        # Larger first piece for aggressive impact
        split_sizes = []
        if num_splits == 2:
            split_sizes = [request.quantity * 0.7, request.quantity * 0.3]
        elif num_splits == 3:
            split_sizes = [request.quantity * 0.5, request.quantity * 0.3, request.quantity * 0.2]
        else:
            split_sizes = self._calculate_equal_splits(request.quantity, num_splits)
        
        split_requests = []
        for i, size in enumerate(split_sizes):
            split_request = ExecutionRequest(
                request_id=f"{request.request_id}_agg_{i}",
                order_id=request.order_id,
                trading_pair=request.trading_pair,
                order_side=request.order_side,
                order_type=OrderType.MARKET if i == 0 else request.order_type,  # First piece as market
                quantity=size,
                price=request.price,
                stop_price=request.stop_price,
                time_in_force=OrderTimeInForce.IOC,
                execution_strategy=ExecutionStrategy.AGGRESSIVE,
                client_id=request.client_id,
                metadata={
                    **request.metadata,
                    'split_index': i,
                    'total_splits': len(split_sizes),
                    'parent_request_id': request.request_id,
                    'execution_type': 'aggressive_split',
                    'is_market': i == 0
                }
            )
            split_requests.append(split_request)
        
        self.logger.info(f"Split order into {len(split_requests)} pieces for aggressive execution")
        
        return split_requests
    
    def _split_for_default(self,
                          request: ExecutionRequest,
                          market_snapshot: MarketSnapshot) -> List[ExecutionRequest]:
        """Default order splitting"""
        
        num_splits = min(
            max(2, int(request.quantity / self.config.min_split_size_btc)),
            self.config.max_split_parts
        )
        
        split_sizes = self._calculate_equal_splits(request.quantity, num_splits)
        
        split_requests = []
        for i, size in enumerate(split_sizes):
            split_request = ExecutionRequest(
                request_id=f"{request.request_id}_default_{i}",
                order_id=request.order_id,
                trading_pair=request.trading_pair,
                order_side=request.order_side,
                order_type=request.order_type,
                quantity=size,
                price=request.price,
                stop_price=request.stop_price,
                time_in_force=request.time_in_force,
                execution_strategy=request.execution_strategy,
                client_id=request.client_id,
                metadata={
                    **request.metadata,
                    'split_index': i,
                    'total_splits': len(split_sizes),
                    'parent_request_id': request.request_id,
                    'execution_type': 'default_split'
                }
            )
            split_requests.append(split_request)
        
        return split_requests
    
    def _calculate_equal_splits(self, total_quantity: float, num_splits: int) -> List[float]:
        """Calculate equal split sizes"""
        base_size = total_quantity / num_splits
        splits = [base_size] * num_splits
        
        # Adjust for rounding errors
        total = sum(splits)
        if total != total_quantity:
            splits[-1] += total_quantity - total
        
        return splits
    
    def _calculate_volume_weighted_splits(self, total_quantity: float, num_splits: int) -> List[float]:
        """Calculate volume-weighted split sizes"""
        # Simplified volume profile (more volume in middle, less at edges)
        # In reality, use historical volume data
        
        # Create a bell curve-like distribution
        indices = np.arange(num_splits)
        weights = np.exp(-0.5 * ((indices - num_splits/2) / (num_splits/4)) ** 2)
        weights = weights / weights.sum()  # Normalize
        
        splits = weights * total_quantity
        
        return splits.tolist()

# ============ Slippage Estimator ============
class SlippageEstimator:
    """Estimates and manages slippage for executions"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.slippage_history = deque(maxlen=config.slippage_lookback_periods)
        self.logger = get_logger(__name__)
        
        # Initialize slippage models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize slippage estimation models"""
        # Placeholder for advanced models
        self.simple_model = SimpleSlippageModel()
        self.advanced_model = AdvancedSlippageModel() if self.config.use_slippage_forecast else None
    
    def estimate_slippage(self,
                         request: ExecutionRequest,
                         market_snapshot: MarketSnapshot,
                         execution_venue: ExecutionVenue) -> Tuple[float, SlippageType]:
        """Estimate slippage for an execution request"""
        
        if not self.config.use_slippage_forecast:
            # Return conservative estimate
            return self.config.target_slippage_percent, SlippageType.NEGATIVE
        
        # Calculate base slippage
        base_slippage = self._calculate_base_slippage(request, market_snapshot, execution_venue)
        
        # Adjust for order characteristics
        adjusted_slippage = self._adjust_for_order_characteristics(base_slippage, request)
        
        # Adjust for market conditions
        adjusted_slippage = self._adjust_for_market_conditions(adjusted_slippage, market_snapshot)
        
        # Adjust for venue characteristics
        adjusted_slippage = self._adjust_for_venue(adjusted_slippage, execution_venue)
        
        # Determine slippage type
        slippage_type = self._determine_slippage_type(adjusted_slippage)
        
        # Ensure within allowed limits
        adjusted_slippage = min(adjusted_slippage, self.config.max_allowed_slippage_percent)
        
        return adjusted_slippage, slippage_type
    
    def _calculate_base_slippage(self,
                                request: ExecutionRequest,
                                market_snapshot: MarketSnapshot,
                                execution_venue: ExecutionVenue) -> float:
        """Calculate base slippage"""
        
        # Simple model based on order size relative to liquidity
        if request.order_side == OrderSide.BUY:
            available_liquidity = market_snapshot.ask_size
            spread = market_snapshot.ask_price - market_snapshot.bid_price
        else:
            available_liquidity = market_snapshot.bid_size
            spread = market_snapshot.ask_price - market_snapshot.bid_price
        
        if available_liquidity > 0:
            size_ratio = request.quantity / available_liquidity
        else:
            size_ratio = 1.0
        
        # Base slippage formula
        base_slippage = (spread / market_snapshot.mid_price) * (1 + size_ratio)
        
        return base_slippage * 100  # Convert to percent
    
    def _adjust_for_order_characteristics(self,
                                        base_slippage: float,
                                        request: ExecutionRequest) -> float:
        """Adjust slippage for order characteristics"""
        
        adjusted_slippage = base_slippage
        
        # Adjust for order type
        if request.order_type == OrderType.MARKET:
            adjusted_slippage *= 1.5  # Market orders have higher slippage
        elif request.order_type == OrderType.LIMIT:
            adjusted_slippage *= 0.7  # Limit orders have lower slippage
        
        # Adjust for order size
        if request.quantity > self.config.max_order_size_btc * 0.5:
            adjusted_slippage *= 2.0  # Very large orders
        elif request.quantity > self.config.max_order_size_btc * 0.2:
            adjusted_slippage *= 1.5  # Large orders
        
        # Adjust for time in force
        if request.time_in_force == OrderTimeInForce.IOC:
            adjusted_slippage *= 1.2  # IOC may have higher slippage
        elif request.time_in_force == OrderTimeInForce.FOK:
            adjusted_slippage *= 1.3  # FOK may have higher slippage
        
        return adjusted_slippage
    
    def _adjust_for_market_conditions(self,
                                     slippage: float,
                                     market_snapshot: MarketSnapshot) -> float:
        """Adjust slippage for market conditions"""
        
        adjusted_slippage = slippage
        
        # Adjust for volatility (simplified)
        # In reality, use historical volatility data
        spread_percent = market_snapshot.spread_percent
        if spread_percent > 0.1:  # High spread
            adjusted_slippage *= 1.5
        elif spread_percent > 0.05:  # Medium spread
            adjusted_slippage *= 1.2
        
        # Adjust for liquidity
        liquidity_score = market_snapshot.liquidity_score
        if liquidity_score < 0.3:  # Low liquidity
            adjusted_slippage *= 2.0
        elif liquidity_score < 0.6:  # Medium liquidity
            adjusted_slippage *= 1.3
        
        return adjusted_slippage
    
    def _adjust_for_venue(self,
                         slippage: float,
                         execution_venue: ExecutionVenue) -> float:
        """Adjust slippage for execution venue"""
        
        adjusted_slippage = slippage
        
        # Venue-specific adjustments
        venue_adjustments = {
            ExecutionVenue.BINANCE: 0.9,    # High liquidity
            ExecutionVenue.COINBASE: 1.0,   # Standard
            ExecutionVenue.KRAKEN: 1.1,     # Slightly higher
            ExecutionVenue.BITFINEX: 1.2,   # Higher
            ExecutionVenue.HUOBI: 1.0,      # Standard
            ExecutionVenue.BYBIT: 1.0,      # Standard
            ExecutionVenue.OKX: 1.0,        # Standard
            ExecutionVenue.DARK_POOL: 0.7,  # Lower slippage in dark pools
            ExecutionVenue.AGGREGATOR: 0.8  # Better routing
        }
        
        if execution_venue in venue_adjustments:
            adjusted_slippage *= venue_adjustments[execution_venue]
        
        return adjusted_slippage
    
    def _determine_slippage_type(self, slippage_percent: float) -> SlippageType:
        """Determine slippage type based on value"""
        if slippage_percent < -0.01:  # Better than expected
            return SlippageType.POSITIVE
        elif slippage_percent > 0.01:  # Worse than expected
            return SlippageType.NEGATIVE
        else:
            return SlippageType.NONE
    
    def record_slippage(self, actual_slippage: float, estimated_slippage: float):
        """Record actual slippage for model improvement"""
        self.slippage_history.append({
            'actual': actual_slippage,
            'estimated': estimated_slippage,
            'error': actual_slippage - estimated_slippage,
            'timestamp': datetime.now()
        })
        
        # Log if error is significant
        error_pct = abs(actual_slippage - estimated_slippage) / max(abs(estimated_slippage), 0.001)
        if error_pct > 0.5:  # More than 50% error
            self.logger.warning(f"Large slippage estimation error: {error_pct:.2%}")

# ============ Venue Router ============
class VenueRouter:
    """Routes orders to optimal execution venues"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Venue statistics
        self.venue_stats = defaultdict(lambda: {
            'success_count': 0,
            'failure_count': 0,
            'total_volume': 0.0,
            'avg_slippage': 0.0,
            'avg_latency': 0.0,
            'last_updated': datetime.now()
        })
        
        # Exchange fee structures (simplified)
        self.exchange_fees = {
            ExecutionVenue.BINANCE: 0.001,   # 0.1%
            ExecutionVenue.COINBASE: 0.005,  # 0.5%
            ExecutionVenue.KRAKEN: 0.0026,   # 0.26%
            ExecutionVenue.BITFINEX: 0.001,  # 0.1%
            ExecutionVenue.HUOBI: 0.002,     # 0.2%
            ExecutionVenue.BYBIT: 0.001,     # 0.1%
            ExecutionVenue.OKX: 0.001,       # 0.1%
            ExecutionVenue.DARK_POOL: 0.0005,# 0.05%
            ExecutionVenue.AGGREGATOR: 0.0015 # 0.15%
        }
    
    def select_venue(self,
                    request: ExecutionRequest,
                    market_snapshots: Dict[str, MarketSnapshot]) -> ExecutionVenue:
        """Select optimal execution venue"""
        
        if not self.config.enable_smart_routing:
            return self.config.primary_exchange
        
        # Get available venues
        available_venues = self._get_available_venues(request.trading_pair, market_snapshots)
        
        if not available_venues:
            self.logger.warning("No available venues, using primary exchange")
            return self.config.primary_exchange
        
        # Score each venue
        venue_scores = {}
        for venue in available_venues:
            score = self._calculate_venue_score(venue, request, market_snapshots.get(str(venue)))
            venue_scores[venue] = score
        
        # Select best venue
        best_venue = max(venue_scores.items(), key=lambda x: x[1])[0]
        
        self.logger.info(f"Selected venue {best_venue.value} with score {venue_scores[best_venue]:.3f}")
        
        return best_venue
    
    def _get_available_venues(self,
                             trading_pair: str,
                             market_snapshots: Dict[str, MarketSnapshot]) -> List[ExecutionVenue]:
        """Get available venues for trading pair"""
        
        available_venues = []
        
        # Check primary exchange
        if self._is_venue_available(self.config.primary_exchange, trading_pair, market_snapshots):
            available_venues.append(self.config.primary_exchange)
        
        # Check backup exchanges
        for venue in self.config.backup_exchanges:
            if self._is_venue_available(venue, trading_pair, market_snapshots):
                available_venues.append(venue)
        
        # Check dark pool if enabled
        if self.config.enable_dark_pool:
            available_venues.append(ExecutionVenue.DARK_POOL)
        
        return available_venues
    
    def _is_venue_available(self,
                           venue: ExecutionVenue,
                           trading_pair: str,
                           market_snapshots: Dict[str, MarketSnapshot]) -> bool:
        """Check if venue is available for trading"""
        
        # For simulation, assume all venues are available
        # In production, you would check API connectivity, maintenance, etc.
        
        # Check if market data is available
        snapshot_key = f"{venue.value}_{trading_pair}"
        if snapshot_key in market_snapshots:
            snapshot = market_snapshots[snapshot_key]
            # Check liquidity
            if snapshot.bid_size > 0 and snapshot.ask_size > 0:
                return True
        
        return True  # Assume available for simulation
    
    def _calculate_venue_score(self,
                              venue: ExecutionVenue,
                              request: ExecutionRequest,
                              market_snapshot: Optional[MarketSnapshot]) -> float:
        """Calculate score for a venue (0-1)"""
        
        if market_snapshot is None:
            return 0.0
        
        score = 0.0
        weight_sum = 0.0
        
        # 1. Liquidity score
        if self.config.route_by_liquidity:
            liquidity_weight = 0.4
            liquidity_score = market_snapshot.liquidity_score
            score += liquidity_score * liquidity_weight
            weight_sum += liquidity_weight
        
        # 2. Fee score
        if self.config.route_by_fees:
            fee_weight = 0.3
            fee_score = 1.0 - min(self.exchange_fees.get(venue, 0.01) / 0.01, 1.0)
            score += fee_score * fee_weight
            weight_sum += fee_weight
        
        # 3. Slippage score
        if self.config.route_by_slippage:
            slippage_weight = 0.3
            # Estimate slippage (simplified)
            if request.order_side == OrderSide.BUY:
                spread = market_snapshot.ask_price - market_snapshot.bid_price
            else:
                spread = market_snapshot.ask_price - market_snapshot.bid_price
            
            spread_pct = spread / market_snapshot.mid_price
            slippage_score = 1.0 - min(spread_pct * 100 / 1.0, 1.0)  # Normalize to 0-1
            score += slippage_score * slippage_weight
            weight_sum += slippage_weight
        
        # 4. Historical performance score
        hist_weight = 0.2
        hist_score = self._get_historical_score(venue)
        score += hist_score * hist_weight
        weight_sum += hist_weight
        
        # Normalize by total weight
        if weight_sum > 0:
            score /= weight_sum
        
        return score
    
    def _get_historical_score(self, venue: ExecutionVenue) -> float:
        """Get historical performance score for venue"""
        stats = self.venue_stats[venue]
        
        if stats['success_count'] + stats['failure_count'] == 0:
            return 0.5  # Neutral score for no history
        
        success_rate = stats['success_count'] / (stats['success_count'] + stats['failure_count'])
        
        # Normalize average slippage (lower is better)
        avg_slippage_norm = max(0, 1.0 - min(stats['avg_slippage'] / 1.0, 1.0))
        
        # Combine metrics
        historical_score = (success_rate * 0.6 + avg_slippage_norm * 0.4)
        
        return historical_score
    
    def update_venue_stats(self,
                          venue: ExecutionVenue,
                          result: ExecutionResult):
        """Update venue statistics with execution result"""
        stats = self.venue_stats[venue]
        
        if result.status == ExecutionStatus.COMPLETED:
            stats['success_count'] += 1
            stats['total_volume'] += result.filled_quantity
            
            # Update average slippage
            if stats['success_count'] == 1:
                stats['avg_slippage'] = result.slippage_percent
            else:
                stats['avg_slippage'] = (
                    (stats['avg_slippage'] * (stats['success_count'] - 1) + 
                     result.slippage_percent) / stats['success_count']
                )
        else:
            stats['failure_count'] += 1
        
        stats['last_updated'] = datetime.now()

# ============ Strategy Executor ============
class StrategyExecutor:
    """Executes orders based on different strategies"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Strategy executors
        self.twap_executor = TWAPExecutor(config)
        self.vwap_executor = VWAPExecutor(config)
        self.iceberg_executor = IcebergExecutor(config)
        self.aggressive_executor = AggressiveExecutor(config)
        self.passive_executor = PassiveExecutor(config)
    
    async def execute(self,
                     request: ExecutionRequest,
                     venue: ExecutionVenue,
                     market_snapshot: MarketSnapshot) -> ExecutionResult:
        """Execute request using appropriate strategy"""
        
        # Create execution ID
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Select executor based on strategy
        if request.execution_strategy == ExecutionStrategy.TWAP:
            executor = self.twap_executor
        elif request.execution_strategy == ExecutionStrategy.VWAP:
            executor = self.vwap_executor
        elif request.execution_strategy == ExecutionStrategy.ICEBERG:
            executor = self.iceberg_executor
        elif request.execution_strategy == ExecutionStrategy.AGGRESSIVE:
            executor = self.aggressive_executor
        elif request.execution_strategy == ExecutionStrategy.PASSIVE:
            executor = self.passive_executor
        else:
            # Default to aggressive for SMART_ROUTING, HYBRID, etc.
            executor = self.aggressive_executor
        
        # Execute
        start_time = time.time()
        try:
            result = await executor.execute(
                execution_id=execution_id,
                request=request,
                venue=venue,
                market_snapshot=market_snapshot
            )
            execution_time = time.time() - start_time
            
            # Add execution time to result
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            execution_time = time.time() - start_time
            
            # Create failed execution result
            return ExecutionResult(
                execution_id=execution_id,
                request_id=request.request_id,
                order_id=request.order_id,
                trading_pair=request.trading_pair,
                filled_quantity=0.0,
                average_price=0.0,
                total_fees=0.0,
                execution_time=execution_time,
                slippage_percent=0.0,
                slippage_type=SlippageType.NONE,
                execution_venue=venue,
                execution_strategy=request.execution_strategy,
                status=ExecutionStatus.FAILED,
                metadata={'error': str(e)}
            )

# ============ Strategy Implementations ============
class TWAPExecutor:
    """Time-Weighted Average Price executor"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def execute(self,
                     execution_id: str,
                     request: ExecutionRequest,
                     venue: ExecutionVenue,
                     market_snapshot: MarketSnapshot) -> ExecutionResult:
        """Execute using TWAP strategy"""
        
        self.logger.info(f"Starting TWAP execution {execution_id} for {request.quantity} {request.trading_pair}")
        
        # Simulate TWAP execution by splitting over time
        # In production, this would interface with exchange TWAP API
        
        # For simulation, we'll create a simple implementation
        splits = 5  # Number of time slices
        fills = []
        total_filled = 0.0
        total_cost = 0.0
        total_fees = 0.0
        
        for i in range(splits):
            # Wait for time interval
            await asyncio.sleep(self.config.twap_interval_seconds)
            
            # Execute slice (simulated)
            slice_size = request.quantity / splits
            
            # Get current market price (simulated price movement)
            current_price = self._get_current_price(market_snapshot, i, splits)
            
            # Calculate fees
            fees = slice_size * current_price * self._get_fee_rate(venue)
            
            # Record fill
            fill = {
                'timestamp': datetime.now(),
                'quantity': slice_size,
                'price': current_price,
                'fees': fees,
                'slice_index': i
            }
            fills.append(fill)
            
            total_filled += slice_size
            total_cost += slice_size * current_price
            total_fees += fees
        
        # Calculate average price
        average_price = total_cost / total_filled if total_filled > 0 else 0.0
        
        # Calculate slippage
        expected_price = market_snapshot.mid_price
        slippage_percent = ((average_price - expected_price) / expected_price) * 100
        
        # Determine slippage type
        if slippage_percent < 0:
            slippage_type = SlippageType.POSITIVE
        elif slippage_percent > 0:
            slippage_type = SlippageType.NEGATIVE
        else:
            slippage_type = SlippageType.NONE
        
        return ExecutionResult(
            execution_id=execution_id,
            request_id=request.request_id,
            order_id=request.order_id,
            trading_pair=request.trading_pair,
            filled_quantity=total_filled,
            average_price=average_price,
            total_fees=total_fees,
            execution_time=0.0,  # Will be set by caller
            slippage_percent=slippage_percent,
            slippage_type=slippage_type,
            execution_venue=venue,
            execution_strategy=ExecutionStrategy.TWAP,
            status=ExecutionStatus.COMPLETED,
            fills=fills
        )
    
    def _get_current_price(self, snapshot: MarketSnapshot, slice_index: int, total_slices: int) -> float:
        """Get current price with simulated market impact"""
        # Simulate price movement during execution
        base_price = snapshot.mid_price
        # Small random walk
        movement = random.uniform(-0.001, 0.001) * (slice_index + 1)
        return base_price * (1 + movement)
    
    def _get_fee_rate(self, venue: ExecutionVenue) -> float:
        """Get fee rate for venue"""
        fee_rates = {
            ExecutionVenue.BINANCE: 0.001,
            ExecutionVenue.COINBASE: 0.005,
            ExecutionVenue.KRAKEN: 0.0026,
            ExecutionVenue.BITFINEX: 0.001,
            ExecutionVenue.HUOBI: 0.002,
            ExecutionVenue.BYBIT: 0.001,
            ExecutionVenue.OKX: 0.001,
            ExecutionVenue.DARK_POOL: 0.0005,
            ExecutionVenue.AGGREGATOR: 0.0015
        }
        return fee_rates.get(venue, 0.002)

class VWAPExecutor:
    """Volume-Weighted Average Price executor"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def execute(self,
                     execution_id: str,
                     request: ExecutionRequest,
                     venue: ExecutionVenue,
                     market_snapshot: MarketSnapshot) -> ExecutionResult:
        """Execute using VWAP strategy"""
        # Similar to TWAP but with volume-weighted slices
        # Implementation would use historical volume profiles
        # For now, use TWAP as placeholder
        twap_executor = TWAPExecutor(self.config)
        return await twap_executor.execute(execution_id, request, venue, market_snapshot)

class IcebergExecutor:
    """Iceberg order executor"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def execute(self,
                     execution_id: str,
                     request: ExecutionRequest,
                     venue: ExecutionVenue,
                     market_snapshot: MarketSnapshot) -> ExecutionResult:
        """Execute using Iceberg strategy"""
        # Would execute small visible portions over time
        # For now, use simple execution
        return await self._execute_simple(execution_id, request, venue, market_snapshot)
    
    async def _execute_simple(self,
                            execution_id: str,
                            request: ExecutionRequest,
                            venue: ExecutionVenue,
                            market_snapshot: MarketSnapshot) -> ExecutionResult:
        """Simple iceberg execution (placeholder)"""
        # Similar to TWAP but with specific iceberg logic
        twap_executor = TWAPExecutor(self.config)
        return await twap_executor.execute(execution_id, request, venue, market_snapshot)

class AggressiveExecutor:
    """Aggressive order executor"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def execute(self,
                     execution_id: str,
                     request: ExecutionRequest,
                     venue: ExecutionVenue,
                     market_snapshot: MarketSnapshot) -> ExecutionResult:
        """Execute using Aggressive strategy"""
        
        self.logger.info(f"Starting aggressive execution {execution_id}")
        
        # Aggressive execution fills immediately at market price
        # For simulation, we'll use current market price with some impact
        
        if request.order_side == OrderSide.BUY:
            execution_price = market_snapshot.ask_price * 1.001  # Pay slightly above ask
        else:
            execution_price = market_snapshot.bid_price * 0.999  # Sell slightly below bid
        
        # Calculate fees
        fee_rate = self._get_fee_rate(venue)
        fees = request.quantity * execution_price * fee_rate
        
        # Calculate slippage
        expected_price = market_snapshot.mid_price
        slippage_percent = ((execution_price - expected_price) / expected_price) * 100
        
        # Determine slippage type
        if slippage_percent < 0:
            slippage_type = SlippageType.POSITIVE
        elif slippage_percent > 0:
            slippage_type = SlippageType.NEGATIVE
        else:
            slippage_type = SlippageType.NONE
        
        # Create fill record
        fills = [{
            'timestamp': datetime.now(),
            'quantity': request.quantity,
            'price': execution_price,
            'fees': fees
        }]
        
        return ExecutionResult(
            execution_id=execution_id,
            request_id=request.request_id,
            order_id=request.order_id,
            trading_pair=request.trading_pair,
            filled_quantity=request.quantity,
            average_price=execution_price,
            total_fees=fees,
            execution_time=0.0,
            slippage_percent=slippage_percent,
            slippage_type=slippage_type,
            execution_venue=venue,
            execution_strategy=ExecutionStrategy.AGGRESSIVE,
            status=ExecutionStatus.COMPLETED,
            fills=fills
        )
    
    def _get_fee_rate(self, venue: ExecutionVenue) -> float:
        """Get fee rate for venue"""
        fee_rates = {
            ExecutionVenue.BINANCE: 0.001,
            ExecutionVenue.COINBASE: 0.005,
            ExecutionVenue.KRAKEN: 0.0026,
            ExecutionVenue.BITFINEX: 0.001,
            ExecutionVenue.HUOBI: 0.002,
            ExecutionVenue.BYBIT: 0.001,
            ExecutionVenue.OKX: 0.001,
            ExecutionVenue.DARK_POOL: 0.0005,
            ExecutionVenue.AGGREGATOR: 0.0015
        }
        return fee_rates.get(venue, 0.002)

class PassiveExecutor:
    """Passive order executor"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def execute(self,
                     execution_id: str,
                     request: ExecutionRequest,
                     venue: ExecutionVenue,
                     market_snapshot: MarketSnapshot) -> ExecutionResult:
        """Execute using Passive strategy"""
        # Passive execution waits for best price
        # For simulation, we'll wait and potentially get better price
        
        # Wait for potential price improvement
        await asyncio.sleep(random.uniform(1, 5))
        
        # Get improved price (simulated)
        if request.order_side == OrderSide.BUY:
            execution_price = market_snapshot.bid_price * 0.999  # Better than bid
        else:
            execution_price = market_snapshot.ask_price * 1.001  # Better than ask
        
        # But there's a chance order doesn't fill
        fill_probability = 0.8  # 80% chance of fill
        
        if random.random() > fill_probability:
            # Order not filled
            return ExecutionResult(
                execution_id=execution_id,
                request_id=request.request_id,
                order_id=request.order_id,
                trading_pair=request.trading_pair,
                filled_quantity=0.0,
                average_price=0.0,
                total_fees=0.0,
                execution_time=0.0,
                slippage_percent=0.0,
                slippage_type=SlippageType.NONE,
                execution_venue=venue,
                execution_strategy=ExecutionStrategy.PASSIVE,
                status=ExecutionStatus.CANCELLED,
                metadata={'reason': 'Order not filled within timeout'}
            )
        
        # Calculate fees
        fee_rate = self._get_fee_rate(venue)
        fees = request.quantity * execution_price * fee_rate
        
        # Calculate slippage (should be positive for passive)
        expected_price = market_snapshot.mid_price
        slippage_percent = ((execution_price - expected_price) / expected_price) * 100
        
        # Create fill record
        fills = [{
            'timestamp': datetime.now(),
            'quantity': request.quantity,
            'price': execution_price,
            'fees': fees
        }]
        
        return ExecutionResult(
            execution_id=execution_id,
            request_id=request.request_id,
            order_id=request.order_id,
            trading_pair=request.trading_pair,
            filled_quantity=request.quantity,
            average_price=execution_price,
            total_fees=fees,
            execution_time=0.0,
            slippage_percent=slippage_percent,
            slippage_type=SlippageType.POSITIVE,  # Passive should get better prices
            execution_venue=venue,
            execution_strategy=ExecutionStrategy.PASSIVE,
            status=ExecutionStatus.COMPLETED,
            fills=fills
        )
    
    def _get_fee_rate(self, venue: ExecutionVenue) -> float:
        """Get fee rate for venue"""
        fee_rates = {
            ExecutionVenue.BINANCE: 0.001,
            ExecutionVenue.COINBASE: 0.005,
            ExecutionVenue.KRAKEN: 0.0026,
            ExecutionVenue.BITFINEX: 0.001,
            ExecutionVenue.HUOBI: 0.002,
            ExecutionVenue.BYBIT: 0.001,
            ExecutionVenue.OKX: 0.001,
            ExecutionVenue.DARK_POOL: 0.0005,
            ExecutionVenue.AGGREGATOR: 0.0015
        }
        return fee_rates.get(venue, 0.002)

# ============ Main Execution Engine ============
class ExecutionEngine(BaseExecutionEngine):
    """Main execution engine for Bitcoin trading AI"""
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        super().__init__(config)
        self.logger = get_logger(__name__)
        
        # Thread/process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_parallel_orders)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_triggered_at: Optional[datetime] = None
        
        # Daily volume tracking
        self.daily_volume = 0.0
        self.last_volume_reset = datetime.now().date()
        
        self.logger.info("Execution Engine initialized")
    
    async def execute_order(self,
                          order: Order,
                          market_data: Optional[pd.DataFrame] = None) -> ExecutionResult:
        """Execute a trading order"""
        
        # Convert Order to ExecutionRequest
        request = self._order_to_request(order)
        
        # Execute the request
        return await self.execute_request(request, market_data)
    
    async def execute_request(self,
                            request: ExecutionRequest,
                            market_data: Optional[pd.DataFrame] = None) -> ExecutionResult:
        """Execute an execution request"""
        
        # Check circuit breaker
        if self.circuit_breaker_active:
            return self._create_circuit_breaker_result(request)
        
        # Reset daily volume if new day
        self._reset_daily_volume_if_needed()
        
        # Check daily volume limit
        if self.daily_volume + request.quantity > self.config.max_daily_volume_btc:
            self.logger.warning(f"Daily volume limit reached: {self.daily_volume} BTC")
            return ExecutionResult(
                execution_id=f"reject_{uuid.uuid4().hex[:8]}",
                request_id=request.request_id,
                order_id=request.order_id,
                trading_pair=request.trading_pair,
                filled_quantity=0.0,
                average_price=0.0,
                total_fees=0.0,
                execution_time=0.0,
                slippage_percent=0.0,
                slippage_type=SlippageType.NONE,
                execution_venue=self.config.primary_exchange,
                execution_strategy=request.execution_strategy,
                status=ExecutionStatus.REJECTED,
                metadata={'reason': 'Daily volume limit exceeded'}
            )
        
        # Get market snapshot
        market_snapshot = await self._get_market_snapshot(request.trading_pair, market_data)
        
        if market_snapshot is None:
            return ExecutionResult(
                execution_id=f"fail_{uuid.uuid4().hex[:8]}",
                request_id=request.request_id,
                order_id=request.order_id,
                trading_pair=request.trading_pair,
                filled_quantity=0.0,
                average_price=0.0,
                total_fees=0.0,
                execution_time=0.0,
                slippage_percent=0.0,
                slippage_type=SlippageType.NONE,
                execution_venue=self.config.primary_exchange,
                execution_strategy=request.execution_strategy,
                status=ExecutionStatus.FAILED,
                metadata={'reason': 'No market data available'}
            )
        
        # Pre-trade analysis
        if self.config.enable_pre_trade_analysis:
            analysis_result = await self._pre_trade_analysis(request, market_snapshot)
            if not analysis_result.get('proceed', True):
                return self._create_rejected_result(request, analysis_result.get('reason', 'Pre-trade analysis failed'))
        
        # Split order if needed
        order_splitter = OrderSplitter(self.config)
        split_requests = order_splitter.split_order(request, market_snapshot)
        
        # Execute split requests
        results = []
        if self.config.enable_parallel_execution and len(split_requests) > 1:
            # Execute in parallel
            tasks = []
            for split_request in split_requests:
                task = asyncio.create_task(self._execute_single_request(split_request, market_data))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Execute sequentially
            for split_request in split_requests:
                result = await self._execute_single_request(split_request, market_data)
                results.append(result)
        
        # Aggregate results
        aggregated_result = self._aggregate_results(request, results)
        
        # Update daily volume
        if aggregated_result.status == ExecutionStatus.COMPLETED:
            self.daily_volume += aggregated_result.filled_quantity
        
        # Post-trade analysis
        if self.config.enable_post_trade_analysis:
            await self._post_trade_analysis(aggregated_result)
        
        # Record execution
        self._record_execution(aggregated_result)
        
        # Check circuit breaker
        self._check_circuit_breaker(aggregated_result)
        
        return aggregated_result
    
    async def _execute_single_request(self,
                                     request: ExecutionRequest,
                                     market_data: Optional[pd.DataFrame] = None) -> ExecutionResult:
        """Execute a single execution request"""
        
        # Get market snapshot
        market_snapshot = await self._get_market_snapshot(request.trading_pair, market_data)
        if market_snapshot is None:
            return self._create_failed_result(request, 'No market data')
        
        # Select venue
        venue = self.venue_router.select_venue(request, {self.config.primary_exchange.value: market_snapshot})
        
        # Estimate slippage
        estimated_slippage, slippage_type = self.slippage_estimator.estimate_slippage(
            request, market_snapshot, venue
        )
        
        # Execute using strategy
        result = await self.strategy_executor.execute(request, venue, market_snapshot)
        
        # Update venue statistics
        self.venue_router.update_venue_stats(venue, result)
        
        # Record actual slippage for learning
        if result.status == ExecutionStatus.COMPLETED:
            self.slippage_estimator.record_slippage(result.slippage_percent, estimated_slippage)
        
        return result
    
    async def _get_market_snapshot(self,
                                  trading_pair: str,
                                  market_data: Optional[pd.DataFrame] = None) -> Optional[MarketSnapshot]:
        """Get market snapshot for trading pair"""
        
        # Check cache first
        cache_key = f"market_snapshot_{trading_pair}"
        cached = self.cache.get(cache_key)
        if cached and self.config.cache_market_data:
            return cached
        
        # Create snapshot from market data or simulation
        if market_data is not None and not market_data.empty:
            latest = market_data.iloc[-1]
            snapshot = MarketSnapshot(
                timestamp=datetime.now(),
                trading_pair=trading_pair,
                bid_price=latest.get('bid', latest.get('close', 50000) * 0.999),
                ask_price=latest.get('ask', latest.get('close', 50000) * 1.001),
                bid_size=latest.get('bid_size', 10.0),
                ask_size=latest.get('ask_size', 10.0),
                last_price=latest.get('close', 50000),
                volume_24h=latest.get('volume', 1000.0),
                spread_percent=abs(latest.get('ask', 50000) - latest.get('bid', 50000)) / latest.get('close', 50000) * 100
            )
        else:
            # Create simulated snapshot
            snapshot = self._create_simulated_snapshot(trading_pair)
        
        # Update cache
        if self.config.cache_market_data:
            self.cache.set(cache_key, snapshot)
        
        # Update engine's snapshot
        self.update_market_snapshot(snapshot)
        
        return snapshot
    
    def _create_simulated_snapshot(self, trading_pair: str) -> MarketSnapshot:
        """Create simulated market snapshot"""
        # Base price around 50,000 USD
        base_price = 50000 + random.uniform(-1000, 1000)
        
        return MarketSnapshot(
            timestamp=datetime.now(),
            trading_pair=trading_pair,
            bid_price=base_price * 0.999,
            ask_price=base_price * 1.001,
            bid_size=random.uniform(5, 20),
            ask_size=random.uniform(5, 20),
            last_price=base_price,
            volume_24h=random.uniform(500, 2000),
            spread_percent=0.2  # 0.2% spread
        )
    
    async def _pre_trade_analysis(self,
                                 request: ExecutionRequest,
                                 market_snapshot: MarketSnapshot) -> Dict[str, Any]:
        """Perform pre-trade analysis"""
        
        analysis = {
            'proceed': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check order size vs max position
        if request.quantity > self.config.max_position_size_btc:
            analysis['proceed'] = False
            analysis['reason'] = f"Order size {request.quantity} BTC exceeds max position {self.config.max_position_size_btc} BTC"
        
        # Check liquidity
        if market_snapshot.liquidity_score < 0.3:
            analysis['warnings'].append("Low liquidity detected")
            if request.quantity > market_snapshot.bid_size * 0.5:
                analysis['recommendations'].append("Consider reducing order size due to low liquidity")
        
        # Estimate market impact
        if request.order_side == OrderSide.BUY:
            impact_ratio = request.quantity / market_snapshot.ask_size
        else:
            impact_ratio = request.quantity / market_snapshot.bid_size
        
        if impact_ratio > 0.3:
            analysis['warnings'].append(f"High market impact estimated: {impact_ratio:.1%}")
            analysis['recommendations'].append("Consider using TWAP or VWAP execution")
        
        # Check volatility (simplified)
        if market_snapshot.spread_percent > 0.5:
            analysis['warnings'].append("High volatility detected")
        
        return analysis
    
    async def _post_trade_analysis(self, result: ExecutionResult):
        """Perform post-trade analysis"""
        
        if result.status != ExecutionStatus.COMPLETED:
            return
        
        analysis = {
            'execution_id': result.execution_id,
            'slippage': result.slippage_percent,
            'execution_time': result.execution_time,
            'fees': result.total_fees,
            'venue': result.execution_venue.value,
            'strategy': result.execution_strategy.value
        }
        
        # Check if slippage was within target
        if abs(result.slippage_percent) > self.config.target_slippage_percent * 2:
            self.logger.warning(f"High slippage detected: {result.slippage_percent:.3f}%")
            analysis['slippage_warning'] = True
        
        # Check execution time
        if result.execution_time > 30:  # 30 seconds
            self.logger.warning(f"Long execution time: {result.execution_time:.1f}s")
            analysis['execution_time_warning'] = True
        
        # Log analysis
        if self.config.log_all_executions:
            self.logger.info(f"Post-trade analysis: {analysis}")
    
    def _aggregate_results(self,
                          original_request: ExecutionRequest,
                          results: List[Union[ExecutionResult, Exception]]) -> ExecutionResult:
        """Aggregate multiple execution results"""
        
        # Filter out exceptions
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                self.logger.error(f"Execution resulted in exception: {str(r)}")
            elif isinstance(r, ExecutionResult):
                valid_results.append(r)
        
        if not valid_results:
            return self._create_failed_result(original_request, 'All sub-executions failed')
        
        # Check if any succeeded
        successful_results = [r for r in valid_results if r.status == ExecutionStatus.COMPLETED]
        if not successful_results:
            # Return the first failed result
            return valid_results[0]
        
        # Aggregate successful results
        total_filled = sum(r.filled_quantity for r in successful_results)
        total_cost = sum(r.filled_quantity * r.average_price for r in successful_results)
        total_fees = sum(r.total_fees for r in successful_results)
        
        if total_filled > 0:
            avg_price = total_cost / total_filled
        else:
            avg_price = 0.0
        
        # Calculate weighted average slippage
        weighted_slippage = sum(r.slippage_percent * r.filled_quantity for r in successful_results) / total_filled
        
        # Determine overall status
        if total_filled >= original_request.quantity * 0.95:  # 95% filled
            status = ExecutionStatus.COMPLETED
        elif total_filled > 0:
            status = ExecutionStatus.PARTIALLY_FILLED
        else:
            status = ExecutionStatus.FAILED
        
        # Determine slippage type
        if weighted_slippage < -0.01:
            slippage_type = SlippageType.POSITIVE
        elif weighted_slippage > 0.01:
            slippage_type = SlippageType.NEGATIVE
        else:
            slippage_type = SlippageType.NONE
        
        # Combine all fills
        all_fills = []
        for r in successful_results:
            all_fills.extend(r.fills)
        
        # Create aggregated result
        return ExecutionResult(
            execution_id=f"agg_{uuid.uuid4().hex[:8]}",
            request_id=original_request.request_id,
            order_id=original_request.order_id,
            trading_pair=original_request.trading_pair,
            filled_quantity=total_filled,
            average_price=avg_price,
            total_fees=total_fees,
            execution_time=max(r.execution_time for r in successful_results),
            slippage_percent=weighted_slippage,
            slippage_type=slippage_type,
            execution_venue=successful_results[0].execution_venue,  # Use first venue
            execution_strategy=original_request.execution_strategy,
            status=status,
            fills=all_fills,
            metadata={
                'aggregated_from': len(successful_results),
                'partial_fills': len([r for r in valid_results if r.status == ExecutionStatus.PARTIALLY_FILLED]),
                'failed_fills': len([r for r in valid_results if r.status in [ExecutionStatus.FAILED, ExecutionStatus.REJECTED]])
            }
        )
    
    def _check_circuit_breaker(self, result: ExecutionResult):
        """Check and potentially trigger circuit breaker"""
        
        if not self.config.enable_circuit_breaker:
            return
        
        # Check for consecutive failures
        recent_results = list(self.performance_history)[-10:]  # Last 10 executions
        if len(recent_results) >= 5:
            failure_count = sum(1 for r in recent_results 
                              if r.status in [ExecutionStatus.FAILED, ExecutionStatus.REJECTED])
            
            failure_rate = failure_count / len(recent_results)
            
            if failure_rate > self.config.circuit_breaker_threshold_percent / 100:
                self.circuit_breaker_active = True
                self.circuit_breaker_triggered_at = datetime.now()
                self.logger.critical(f"Circuit breaker triggered! Failure rate: {failure_rate:.1%}")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker"""
        if self.circuit_breaker_active:
            # Check if enough time has passed
            if self.circuit_breaker_triggered_at:
                time_since_trigger = datetime.now() - self.circuit_breaker_triggered_at
                if time_since_trigger.total_seconds() > 300:  # 5 minutes
                    self.circuit_breaker_active = False
                    self.circuit_breaker_triggered_at = None
                    self.logger.info("Circuit breaker reset")
    
    def _reset_daily_volume_if_needed(self):
        """Reset daily volume counter if new day"""
        today = datetime.now().date()
        if today != self.last_volume_reset:
            self.daily_volume = 0.0
            self.last_volume_reset = today
            self.logger.info(f"Reset daily volume counter for {today}")
    
    def _order_to_request(self, order: Order) -> ExecutionRequest:
        """Convert Order to ExecutionRequest"""
        
        return ExecutionRequest(
            request_id=f"req_{order.order_id}",
            order_id=order.order_id,
            trading_pair=order.trading_pair,
            order_side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force,
            execution_strategy=self.config.default_execution_strategy,
            client_id=order.client_id,
            metadata={
                'source': 'order_manager',
                'original_order_status': order.status.value
            }
        )
    
    def _create_circuit_breaker_result(self, request: ExecutionRequest) -> ExecutionResult:
        """Create result for circuit breaker active"""
        
        return ExecutionResult(
            execution_id=f"cb_{uuid.uuid4().hex[:8]}",
            request_id=request.request_id,
            order_id=request.order_id,
            trading_pair=request.trading_pair,
            filled_quantity=0.0,
            average_price=0.0,
            total_fees=0.0,
            execution_time=0.0,
            slippage_percent=0.0,
            slippage_type=SlippageType.NONE,
            execution_venue=self.config.primary_exchange,
            execution_strategy=request.execution_strategy,
            status=ExecutionStatus.REJECTED,
            metadata={
                'reason': 'Circuit breaker active',
                'triggered_at': self.circuit_breaker_triggered_at.isoformat() if self.circuit_breaker_triggered_at else None
            }
        )
    
    def _create_rejected_result(self,
                               request: ExecutionRequest,
                               reason: str) -> ExecutionResult:
        """Create rejected execution result"""
        
        return ExecutionResult(
            execution_id=f"reject_{uuid.uuid4().hex[:8]}",
            request_id=request.request_id,
            order_id=request.order_id,
            trading_pair=request.trading_pair,
            filled_quantity=0.0,
            average_price=0.0,
            total_fees=0.0,
            execution_time=0.0,
            slippage_percent=0.0,
            slippage_type=SlippageType.NONE,
            execution_venue=self.config.primary_exchange,
            execution_strategy=request.execution_strategy,
            status=ExecutionStatus.REJECTED,
            metadata={'reason': reason}
        )
    
    def _create_failed_result(self,
                             request: ExecutionRequest,
                             reason: str) -> ExecutionResult:
        """Create failed execution result"""
        
        return ExecutionResult(
            execution_id=f"fail_{uuid.uuid4().hex[:8]}",
            request_id=request.request_id,
            order_id=request.order_id,
            trading_pair=request.trading_pair,
            filled_quantity=0.0,
            average_price=0.0,
            total_fees=0.0,
            execution_time=0.0,
            slippage_percent=0.0,
            slippage_type=SlippageType.NONE,
            execution_venue=self.config.primary_exchange,
            execution_strategy=request.execution_strategy,
            status=ExecutionStatus.FAILED,
            metadata={'reason': reason}
        )
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an ongoing execution"""
        
        if execution_id in self.active_executions:
            # In production, would send cancel to exchange
            self.logger.info(f"Cancelling execution {execution_id}")
            
            # Remove from active executions
            request = self.active_executions.pop(execution_id, None)
            
            # Create cancelled result
            if request:
                cancelled_result = ExecutionResult(
                    execution_id=execution_id,
                    request_id=request.request_id,
                    order_id=request.order_id,
                    trading_pair=request.trading_pair,
                    filled_quantity=0.0,
                    average_price=0.0,
                    total_fees=0.0,
                    execution_time=0.0,
                    slippage_percent=0.0,
                    slippage_type=SlippageType.NONE,
                    execution_venue=self.config.primary_exchange,
                    execution_strategy=request.execution_strategy,
                    status=ExecutionStatus.CANCELLED,
                    metadata={'cancelled_at': datetime.now().isoformat()}
                )
                self._record_execution(cancelled_result)
                return True
        
        return False
    
    async def get_execution_status(self, execution_id: str) -> ExecutionStatus:
        """Get execution status"""
        
        if execution_id in self.active_executions:
            return ExecutionStatus.IN_PROGRESS
        elif execution_id in self.execution_results:
            return self.execution_results[execution_id].status
        else:
            return ExecutionStatus.FAILED
    
    async def get_execution_report(self,
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get execution report for time period"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=1)
        if not end_date:
            end_date = datetime.now()
        
        report = []
        for result in self.execution_results.values():
            if start_date <= result.completed_at <= end_date:
                report.append(result.to_dict())
        
        return report
    
    async def shutdown(self):
        """Shutdown execution engine"""
        self.logger.info("Shutting down execution engine")
        
        # Cancel all active executions
        for execution_id in list(self.active_executions.keys()):
            await self.cancel_execution(execution_id)
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        self.logger.info("Execution engine shutdown complete")

# ============ Factory Function ============
def create_execution_engine(config: Optional[ExecutionConfig] = None) -> ExecutionEngine:
    """Factory function to create execution engine"""
    return ExecutionEngine(config)

# ============ Main Execution ============
async def main():
    """Main execution for testing"""
    
    # Create execution engine
    config = ExecutionConfig(
        default_execution_strategy=ExecutionStrategy.SMART_ROUTING,
        enable_parallel_execution=True,
        max_parallel_orders=3
    )
    
    engine = create_execution_engine(config)
    
    try:
        # Create test execution request
        from core.trading.order_manager import Order, OrderType, OrderSide, OrderTimeInForce
        
        test_order = Order(
            order_id="test_order_001",
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000.0,
            time_in_force=OrderTimeInForce.GTC,
            status=OrderStatus.NEW
        )
        
        # Execute test order
        result = await engine.execute_order(test_order)
        
        print(f"Execution Result:")
        print(f"  Status: {result.status.value}")
        print(f"  Filled: {result.filled_quantity} BTC")
        print(f"  Average Price: ${result.average_price:.2f}")
        print(f"  Slippage: {result.slippage_percent:.3f}%")
        print(f"  Execution Time: {result.execution_time:.2f}s")
        
        # Get metrics
        metrics = engine.get_execution_metrics()
        print(f"\nExecution Metrics:")
        print(f"  Total Executions: {metrics.total_executions}")
        print(f"  Success Rate: {metrics.successful_executions / metrics.total_executions * 100:.1f}%")
        print(f"  Avg Slippage: {metrics.avg_slippage_percent:.3f}%")
        
    finally:
        # Shutdown engine
        await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())