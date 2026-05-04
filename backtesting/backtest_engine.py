"""
Backtesting Engine for Bitcoin Trading Application.
Provides historical simulation of trading strategies with performance metrics.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import warnings
import copy
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import asyncio

# Suppress warnings
warnings.filterwarnings('ignore')

# Import project modules
from logger import get_logger
from cache import TradingCache, cached

logger = get_logger(__name__)

class OrderType(Enum):
    """Order types for backtesting."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionType(Enum):
    """Position types."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class Trade:
    """Represents a completed trade."""
    id: str
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    fees: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_pnl(self, current_price: Optional[float] = None) -> Tuple[float, float]:
        """Calculate P&L for the trade."""
        if current_price is not None:
            self.exit_price = current_price
            self.exit_time = datetime.now()
        
        if self.side == OrderSide.BUY:
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SELL (for short positions)
            self.pnl = (self.entry_price - self.exit_price) * self.quantity
        
        self.pnl_percentage = (self.pnl / (self.entry_price * self.quantity)) * 100
        return self.pnl, self.pnl_percentage

@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]
    
    def fill(self, price: float, quantity: float, fees: float = 0.0) -> None:
        """Fill the order (partially or completely)."""
        if quantity > self.quantity - self.filled_quantity:
            quantity = self.quantity - self.filled_quantity
        
        self.filled_quantity += quantity
        self.filled_price = price if self.filled_price is None else (
            (self.filled_price * (self.filled_quantity - quantity) + price * quantity) / self.filled_quantity
        )
        self.fees += fees
        
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now()
        else:
            self.status = OrderStatus.PARTIAL

@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    position_type: PositionType
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    unrealized_pnl_percentage: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, current_price: float) -> None:
        """Update position with current price."""
        self.current_price = current_price
        
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        elif self.position_type == PositionType.SHORT:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        else:  # FLAT
            self.unrealized_pnl = 0.0
        
        if self.entry_price > 0:
            self.unrealized_pnl_percentage = (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100
    
    def close(self, exit_price: float, fees: float = 0.0) -> Trade:
        """Close position and return trade."""
        if self.position_type == PositionType.FLAT:
            raise ValueError("Cannot close flat position")
        
        side = OrderSide.SELL if self.position_type == PositionType.LONG else OrderSide.BUY
        trade = Trade(
            id=f"trade_{datetime.now().timestamp()}",
            symbol=self.symbol,
            side=side,
            entry_price=self.entry_price,
            exit_price=exit_price,
            quantity=self.quantity,
            entry_time=self.entry_time,
            exit_time=datetime.now(),
            fees=self.total_fees + fees
        )
        
        trade.calculate_pnl()
        self.realized_pnl += trade.pnl
        self.total_fees += fees
        self.position_type = PositionType.FLAT
        self.quantity = 0.0
        
        return trade

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 10000.0
    trading_fee: float = 0.001  # 0.1%
    slippage: float = 0.0005    # 0.05%
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    timeframe: str = "1h"
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    warmup_period: int = 200  # Number of candles for warmup
    position_sizing: str = "fixed"  # fixed, percentage, kelly
    position_size: float = 0.1  # For fixed: absolute amount, for percentage: % of capital
    max_position_size: float = 1.0  # Maximum position size as percentage of capital
    stop_loss: Optional[float] = None  # Percentage stop loss
    take_profit: Optional[float] = None  # Percentage take profit
    max_drawdown: Optional[float] = None  # Maximum allowed drawdown
    allow_shorting: bool = False
    allow_margin: bool = False
    margin_rate: float = 0.02  # Annual margin interest rate
    commission_model: str = "percentage"  # percentage, fixed, tiered
    data_source: str = "local"  # local, database, api
    cache_results: bool = True
    parallel: bool = False  # Parallel processing for multiple symbols
    verbose: bool = True

@dataclass
class PerformanceMetrics:
    """Performance metrics for backtesting results."""
    # Basic metrics
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return: float = 0.0
    total_return_percentage: float = 0.0
    annual_return: float = 0.0
    annual_return_percentage: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percentage: float = 0.0
    volatility: float = 0.0
    value_at_risk: float = 0.0
    expected_shortfall: float = 0.0
    
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Position metrics
    avg_position_holding_period: timedelta = field(default_factory=lambda: timedelta(0))
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_risk_reward_ratio: float = 0.0
    
    # Efficiency metrics
    total_fees: float = 0.0
    slippage_cost: float = 0.0
    net_profit_after_costs: float = 0.0
    efficiency_ratio: float = 0.0  # Net profit / Gross profit
    
    # Time-based metrics
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0
    market_exposure: float = 0.0  # Percentage of time in market
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

class BacktestEngine:
    """Main backtesting engine for trading strategies."""
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.BacktestEngine")
        
        # State variables
        self.current_time: Optional[datetime] = None
        self.current_index: int = 0
        self.is_running: bool = False
        self.is_paused: bool = False
        
        # Data storage
        self.ohlcv_data: Dict[str, pd.DataFrame] = {}
        self.signals_data: Dict[str, pd.DataFrame] = {}
        self.indicators_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Trading state
        self.capital: float = config.initial_capital
        self.portfolio_value: float = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.order_history: List[Order] = []
        self.trade_history: List[Trade] = []
        
        # Performance tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.drawdown_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        
        # Strategy instance
        self.strategy: Optional[Any] = None
        self.strategy_instance: Optional[Any] = None
        
        # Cache for expensive calculations
        self.cache = TradingCache(cache_type="memory")
        
        # Event handlers
        self.on_trade_callbacks: List[Callable] = []
        self.on_order_callbacks: List[Callable] = []
        self.on_update_callbacks: List[Callable] = []
        
        self.logger.info(f"Initialized BacktestEngine with {config.initial_capital} capital")
    
    def load_data(self, 
                  data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                  symbol: Optional[str] = None) -> None:
        """
        Load OHLCV data for backtesting.
        
        Args:
            data: DataFrame or dictionary of DataFrames with OHLCV data
            symbol: Symbol name (if single DataFrame)
        """
        if isinstance(data, pd.DataFrame):
            if symbol is None:
                symbol = self.config.symbols[0]
            self.ohlcv_data[symbol] = data.copy()
            
            # Ensure proper datetime index
            if not isinstance(self.ohlcv_data[symbol].index, pd.DatetimeIndex):
                if 'timestamp' in self.ohlcv_data[symbol].columns:
                    self.ohlcv_data[symbol].index = pd.to_datetime(
                        self.ohlcv_data[symbol]['timestamp'], unit='ms'
                    )
                elif 'date' in self.ohlcv_data[symbol].columns:
                    self.ohlcv_data[symbol].index = pd.to_datetime(
                        self.ohlcv_data[symbol]['date']
                    )
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in self.ohlcv_data[symbol].columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Filter by date range
            if self.config.start_date:
                self.ohlcv_data[symbol] = self.ohlcv_data[symbol].loc[
                    self.ohlcv_data[symbol].index >= self.config.start_date
                ]
            if self.config.end_date:
                self.ohlcv_data[symbol] = self.ohlcv_data[symbol].loc[
                    self.ohlcv_data[symbol].index <= self.config.end_date
                ]
            
            self.logger.info(f"Loaded data for {symbol}: {len(self.ohlcv_data[symbol])} candles "
                           f"from {self.ohlcv_data[symbol].index[0]} to {self.ohlcv_data[symbol].index[-1]}")
        
        elif isinstance(data, dict):
            for sym, df in data.items():
                self.load_data(df, sym)
        else:
            raise ValueError("Data must be DataFrame or dict of DataFrames")
    
    def add_strategy(self, strategy_class: Any, **strategy_params) -> None:
        """
        Add trading strategy to backtest.
        
        Args:
            strategy_class: Strategy class to instantiate
            strategy_params: Parameters for strategy initialization
        """
        self.strategy = strategy_class
        self.strategy_instance = strategy_class(**strategy_params)
        self.logger.info(f"Added strategy: {strategy_class.__name__}")
    
    def add_indicator(self, 
                     symbol: str,
                     indicator_name: str,
                     indicator_data: pd.Series) -> None:
        """
        Add pre-calculated indicator data.
        
        Args:
            symbol: Trading symbol
            indicator_name: Name of indicator
            indicator_data: Indicator values as Series
        """
        if symbol not in self.indicators_data:
            self.indicators_data[symbol] = {}
        
        self.indicators_data[symbol][indicator_name] = indicator_data
        self.logger.debug(f"Added indicator {indicator_name} for {symbol}")
    
    def add_signal(self,
                  symbol: str,
                  signal_name: str,
                  signal_data: pd.Series) -> None:
        """
        Add pre-calculated signal data.
        
        Args:
            symbol: Trading symbol
            signal_name: Name of signal
            signal_data: Signal values as Series
        """
        if symbol not in self.signals_data:
            self.signals_data[symbol] = pd.DataFrame(index=self.ohlcv_data[symbol].index)
        
        self.signals_data[symbol][signal_name] = signal_data
        self.logger.debug(f"Added signal {signal_name} for {symbol}")
    
    def calculate_position_size(self, 
                               symbol: str,
                               price: float,
                               risk_percentage: Optional[float] = None) -> float:
        """
        Calculate position size based on configuration.
        
        Args:
            symbol: Trading symbol
            price: Current price
            risk_percentage: Risk percentage (overrides config)
        
        Returns:
            float: Position size in asset units
        """
        if risk_percentage is None:
            risk_percentage = self.config.position_size
        
        if self.config.position_sizing == "fixed":
            return risk_percentage / price if risk_percentage > 1 else risk_percentage
        
        elif self.config.position_sizing == "percentage":
            position_value = self.portfolio_value * risk_percentage
            return position_value / price
        
        elif self.config.position_sizing == "kelly":
            # Simplified Kelly Criterion
            win_rate = 0.5  # Default, should be calculated from historical data
            avg_win_loss_ratio = 2.0  # Default
            kelly_fraction = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
            position_value = self.portfolio_value * kelly_fraction * risk_percentage
            return position_value / price
        
        else:
            # Default to percentage
            position_value = self.portfolio_value * 0.1
            return position_value / price
    
    def create_order(self,
                    symbol: str,
                    side: OrderSide,
                    order_type: OrderType,
                    quantity: float,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    limit_price: Optional[float] = None,
                    **metadata) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            order_type: Type of order
            quantity: Order quantity
            price: Price for market orders
            stop_price: Stop price for stop orders
            limit_price: Limit price for limit orders
            **metadata: Additional order metadata
        
        Returns:
            Order: Created order object
        """
        order_id = f"order_{datetime.now().timestamp()}_{len(self.orders)}"
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            limit_price=limit_price,
            metadata=metadata
        )
        
        self.orders.append(order)
        self.order_history.append(copy.deepcopy(order))
        
        # Trigger order callbacks
        for callback in self.on_order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Error in order callback: {e}")
        
        self.logger.debug(f"Created order: {order.id} {side.value} {quantity} {symbol} "
                        f"at {price if price else 'market'}")
        
        return order
    
    def execute_order(self, order: Order, current_price: float) -> bool:
        """
        Execute an order at current market conditions.
        
        Args:
            order: Order to execute
            current_price: Current market price
        
        Returns:
            bool: True if order was executed
        """
        if not order.is_active():
            return False
        
        # Calculate execution price with slippage
        slippage_factor = 1 + (self.config.slippage if order.side == OrderSide.BUY else -self.config.slippage)
        execution_price = current_price * slippage_factor
        
        # Check order conditions
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                self.logger.error("Limit order missing limit price")
                order.status = OrderStatus.REJECTED
                return False
            
            if order.side == OrderSide.BUY and execution_price > order.limit_price:
                execution_price = order.limit_price
            elif order.side == OrderSide.SELL and execution_price < order.limit_price:
                execution_price = order.limit_price
            else:
                # Price not reached
                return False
        
        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                self.logger.error("Stop order missing stop price")
                order.status = OrderStatus.REJECTED
                return False
            
            if order.side == OrderSide.BUY and execution_price < order.stop_price:
                return False
            elif order.side == OrderSide.SELL and execution_price > order.stop_price:
                return False
        
        elif order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                self.logger.error("Stop-limit order missing prices")
                order.status = OrderStatus.REJECTED
                return False
            
            # First check stop condition
            if order.side == OrderSide.BUY and execution_price < order.stop_price:
                return False
            elif order.side == OrderSide.SELL and execution_price > order.stop_price:
                return False
            
            # Then check limit condition
            if order.side == OrderSide.BUY and execution_price > order.limit_price:
                execution_price = order.limit_price
            elif order.side == OrderSide.SELL and execution_price < order.limit_price:
                execution_price = order.limit_price
        
        # Calculate fees
        order_value = execution_price * order.quantity
        if self.config.commission_model == "percentage":
            fees = order_value * self.config.trading_fee
        elif self.config.commission_model == "fixed":
            fees = self.config.trading_fee
        else:  # tiered - simplified
            fees = order_value * max(0.0005, self.config.trading_fee * (1 - order_value / 1000000))
        
        # Check if we have enough capital (for buys)
        if order.side == OrderSide.BUY:
            total_cost = order_value + fees
            if total_cost > self.capital:
                self.logger.warning(f"Insufficient capital for order {order.id}: "
                                  f"needed {total_cost}, have {self.capital}")
                order.status = OrderStatus.REJECTED
                return False
        
        # Execute the order
        order.fill(execution_price, order.quantity, fees)
        
        # Update portfolio
        self._update_portfolio(order, execution_price, fees)
        
        # Remove filled order from active orders
        if order.is_filled():
            self.orders.remove(order)
        
        self.logger.info(f"Executed order {order.id}: {order.side.value} {order.quantity} "
                        f"{order.symbol} at {execution_price:.2f} (fees: {fees:.4f})")
        
        return True
    
    def _update_portfolio(self, order: Order, execution_price: float, fees: float) -> None:
        """Update portfolio after order execution."""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                position_type=PositionType.FLAT,
                quantity=0.0,
                entry_price=0.0,
                current_price=execution_price,
                entry_time=self.current_time,
                total_fees=0.0
            )
        
        position = self.positions[symbol]
        
        if order.side == OrderSide.BUY:
            # Update position for buy order
            if position.position_type == PositionType.FLAT:
                position.position_type = PositionType.LONG
                position.quantity = order.quantity
                position.entry_price = execution_price
                position.entry_time = self.current_time
            elif position.position_type == PositionType.LONG:
                # Average entry price
                total_cost = (position.entry_price * position.quantity) + (execution_price * order.quantity)
                position.quantity += order.quantity
                position.entry_price = total_cost / position.quantity
            elif position.position_type == PositionType.SHORT:
                # Closing short position
                if order.quantity >= position.quantity:
                    # Fully close short
                    trade = position.close(execution_price, fees)
                    self.trades.append(trade)
                    self.trade_history.append(trade)
                else:
                    # Partial close
                    partial_trade = Trade(
                        id=f"trade_{datetime.now().timestamp()}",
                        symbol=symbol,
                        side=OrderSide.BUY,
                        entry_price=position.entry_price,
                        exit_price=execution_price,
                        quantity=order.quantity,
                        entry_time=position.entry_time,
                        exit_time=self.current_time,
                        fees=fees
                    )
                    partial_trade.calculate_pnl()
                    
                    # Update position
                    position.quantity -= order.quantity
                    position.total_fees += fees
                    position.realized_pnl += partial_trade.pnl
                    
                    self.trades.append(partial_trade)
                    self.trade_history.append(partial_trade)
        
        elif order.side == OrderSide.SELL:
            # Update position for sell order
            if position.position_type == PositionType.FLAT and self.config.allow_shorting:
                position.position_type = PositionType.SHORT
                position.quantity = order.quantity
                position.entry_price = execution_price
                position.entry_time = self.current_time
            elif position.position_type == PositionType.FLAT and not self.config.allow_shorting:
                # Can't short
                self.logger.warning(f"Shorting not allowed for {symbol}")
                return
            elif position.position_type == PositionType.SHORT:
                # Average entry price for short
                total_cost = (position.entry_price * position.quantity) + (execution_price * order.quantity)
                position.quantity += order.quantity
                position.entry_price = total_cost / position.quantity
            elif position.position_type == PositionType.LONG:
                # Closing long position
                if order.quantity >= position.quantity:
                    # Fully close long
                    trade = position.close(execution_price, fees)
                    self.trades.append(trade)
                    self.trade_history.append(trade)
                else:
                    # Partial close
                    partial_trade = Trade(
                        id=f"trade_{datetime.now().timestamp()}",
                        symbol=symbol,
                        side=OrderSide.SELL,
                        entry_price=position.entry_price,
                        exit_price=execution_price,
                        quantity=order.quantity,
                        entry_time=position.entry_time,
                        exit_time=self.current_time,
                        fees=fees
                    )
                    partial_trade.calculate_pnl()
                    
                    # Update position
                    position.quantity -= order.quantity
                    position.total_fees += fees
                    position.realized_pnl += partial_trade.pnl
                    
                    self.trades.append(partial_trade)
                    self.trade_history.append(partial_trade)
        
        # Update capital
        if order.side == OrderSide.BUY:
            self.capital -= (execution_price * order.quantity + fees)
        else:  # SELL
            self.capital += (execution_price * order.quantity - fees)
        
        position.total_fees += fees
        position.update(execution_price)
        
        # Update portfolio value
        self._update_portfolio_value()
    
    def _update_portfolio_value(self) -> None:
        """Update total portfolio value."""
        total_position_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in self.ohlcv_data and not self.ohlcv_data[symbol].empty:
                current_price = self.ohlcv_data[symbol].iloc[-1]['close']
                position.update(current_price)
                
                if position.position_type != PositionType.FLAT:
                    position_value = position.quantity * current_price
                    
                    if position.position_type == PositionType.SHORT:
                        # For shorts, value is negative (liability)
                        position_value = -position_value
                    
                    total_position_value += position_value
        
        self.portfolio_value = self.capital + total_position_value
        
        # Record equity curve
        if self.current_time:
            self.equity_curve.append((self.current_time, self.portfolio_value))
            
            # Calculate drawdown
            if len(self.equity_curve) > 1:
                peak = max([v for _, v in self.equity_curve])
                drawdown = (peak - self.portfolio_value) / peak if peak > 0 else 0.0
                self.drawdown_curve.append((self.current_time, drawdown))
    
    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> List[Trade]:
        """
        Check and execute stop loss/take profit orders.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        
        Returns:
            List[Trade]: List of closed trades
        """
        closed_trades = []
        
        if symbol not in self.positions:
            return closed_trades
        
        position = self.positions[symbol]
        
        if position.position_type == PositionType.FLAT:
            return closed_trades
        
        # Check stop loss
        if self.config.stop_loss is not None:
            if position.position_type == PositionType.LONG:
                stop_price = position.entry_price * (1 - self.config.stop_loss)
                if current_price <= stop_price:
                    self.logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                    trade = position.close(stop_price)
                    closed_trades.append(trade)
            
            elif position.position_type == PositionType.SHORT:
                stop_price = position.entry_price * (1 + self.config.stop_loss)
                if current_price >= stop_price:
                    self.logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                    trade = position.close(stop_price)
                    closed_trades.append(trade)
        
        # Check take profit
        if self.config.take_profit is not None:
            if position.position_type == PositionType.LONG:
                take_profit_price = position.entry_price * (1 + self.config.take_profit)
                if current_price >= take_profit_price:
                    self.logger.info(f"Take profit triggered for {symbol} at {current_price}")
                    trade = position.close(take_profit_price)
                    closed_trades.append(trade)
            
            elif position.position_type == PositionType.SHORT:
                take_profit_price = position.entry_price * (1 - self.config.take_profit)
                if current_price <= take_profit_price:
                    self.logger.info(f"Take profit triggered for {symbol} at {current_price}")
                    trade = position.close(take_profit_price)
                    closed_trades.append(trade)
        
        # Add closed trades to history
        for trade in closed_trades:
            self.trades.append(trade)
            self.trade_history.append(trade)
        
        return closed_trades
    
    def step(self) -> bool:
        """
        Execute one step (candle) of backtest.
        
        Returns:
            bool: True if there are more steps, False if finished
        """
        if not self.ohlcv_data:
            self.logger.error("No data loaded for backtesting")
            return False
        
        # Get current timestamp
        symbol = self.config.symbols[0]  # For simplicity, use first symbol
        if self.current_index >= len(self.ohlcv_data[symbol]):
            return False
        
        self.current_time = self.ohlcv_data[symbol].index[self.current_index]
        
        # Get current candle data for all symbols
        current_data = {}
        for sym in self.config.symbols:
            if sym in self.ohlcv_data and self.current_index < len(self.ohlcv_data[sym]):
                current_data[sym] = self.ohlcv_data[sym].iloc[self.current_index]
        
        # Update portfolio value with current prices
        for sym in self.config.symbols:
            if sym in current_data:
                current_price = current_data[sym]['close']
                
                # Update positions
                if sym in self.positions:
                    self.positions[sym].update(current_price)
                
                # Check stop loss/take profit
                closed_trades = self.check_stop_loss_take_profit(sym, current_price)
                for trade in closed_trades:
                    # Trigger trade callbacks
                    for callback in self.on_trade_callbacks:
                        try:
                            callback(trade)
                        except Exception as e:
                            self.logger.error(f"Error in trade callback: {e}")
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Execute pending orders
        for order in self.orders[:]:  # Copy list for safe iteration
            if order.symbol in current_data:
                current_price = current_data[order.symbol]['close']
                self.execute_order(order, current_price)
        
        # Check max drawdown limit
        if self.config.max_drawdown is not None:
            if len(self.drawdown_curve) > 0:
                current_drawdown = self.drawdown_curve[-1][1]
                if current_drawdown >= self.config.max_drawdown:
                    self.logger.warning(f"Max drawdown limit reached: {current_drawdown:.2%}")
                    # Close all positions
                    self.close_all_positions()
                    return False
        
        # Run strategy if we have enough data
        if self.strategy_instance and self.current_index >= self.config.warmup_period:
            try:
                # Prepare data for strategy
                strategy_data = {}
                for sym in self.config.symbols:
                    if sym in self.ohlcv_data:
                        # Get data up to current index
                        data_slice = self.ohlcv_data[sym].iloc[:self.current_index + 1]
                        
                        # Add indicators if available
                        if sym in self.indicators_data:
                            for indicator_name, indicator_series in self.indicators_data[sym].items():
                                if indicator_name not in data_slice.columns:
                                    # Align indicator with data
                                    aligned_indicator = indicator_series.reindex(data_slice.index)
                                    data_slice[indicator_name] = aligned_indicator
                        
                        # Add signals if available
                        if sym in self.signals_data:
                            for signal_name, signal_series in self.signals_data[sym].items():
                                if signal_name not in data_slice.columns:
                                    aligned_signal = signal_series.reindex(data_slice.index)
                                    data_slice[signal_name] = aligned_signal
                        
                        strategy_data[sym] = data_slice
                
                # Get strategy signals
                signals = self.strategy_instance.generate_signals(strategy_data)
                
                # Execute signals
                if signals:
                    self._execute_strategy_signals(signals, current_data)
            
            except Exception as e:
                self.logger.error(f"Error in strategy execution: {e}")
                self.logger.error(traceback.format_exc())
        
        # Record portfolio snapshot
        self._record_portfolio_snapshot(current_data)
        
        # Trigger update callbacks
        for callback in self.on_update_callbacks:
            try:
                callback(self.current_time, self.portfolio_value, self.positions)
            except Exception as e:
                self.logger.error(f"Error in update callback: {e}")
        
        self.current_index += 1
        
        if self.config.verbose and self.current_index % 100 == 0:
            self.logger.info(f"Processed {self.current_index}/{len(self.ohlcv_data[symbol])} candles "
                           f"({self.current_time}), Portfolio: ${self.portfolio_value:.2f}")
        
        return True
    
    def _execute_strategy_signals(self, 
                                 signals: Dict[str, Any],
                                 current_data: Dict[str, pd.Series]) -> None:
        """
        Execute strategy signals.
        
        Args:
            signals: Dictionary of signals from strategy
            current_data: Current candle data
        """
        for symbol, signal_info in signals.items():
            if symbol not in current_data:
                continue
            
            current_price = current_data[symbol]['close']
            
            # Handle different signal formats
            if isinstance(signal_info, dict):
                signal_type = signal_info.get('signal', 'HOLD')
                strength = signal_info.get('strength', 1.0)
                metadata = signal_info.get('metadata', {})
            else:
                signal_type = signal_info
                strength = 1.0
                metadata = {}
            
            # Execute based on signal
            if signal_type.upper() == 'BUY':
                # Calculate position size
                position_size = self.calculate_position_size(symbol, current_price)
                
                # Check if we already have a position
                if symbol in self.positions:
                    position = self.positions[symbol]
                    if position.position_type == PositionType.LONG:
                        # Already long, maybe add to position
                        if strength > 0.8:  # Strong signal
                            self.create_order(
                                symbol=symbol,
                                side=OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                quantity=position_size * 0.5,  # Add half position
                                price=current_price,
                                metadata={"signal_strength": strength, **metadata}
                            )
                    elif position.position_type == PositionType.SHORT:
                        # Close short position
                        self.create_order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=position.quantity,
                            price=current_price,
                            metadata={"signal_strength": strength, **metadata}
                        )
                    else:  # FLAT
                        self.create_order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=position_size,
                            price=current_price,
                            metadata={"signal_strength": strength, **metadata}
                        )
                else:
                    # No position, open new
                    self.create_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=position_size,
                        price=current_price,
                        metadata={"signal_strength": strength, **metadata}
                    )
            
            elif signal_type.upper() == 'SELL':
                if symbol in self.positions:
                    position = self.positions[symbol]
                    
                    if position.position_type == PositionType.LONG:
                        # Close long position
                        self.create_order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=position.quantity,
                            price=current_price,
                            metadata={"signal_strength": strength, **metadata}
                        )
                    
                    elif position.position_type == PositionType.FLAT and self.config.allow_shorting:
                        # Open short position
                        position_size = self.calculate_position_size(symbol, current_price)
                        self.create_order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=position_size,
                            price=current_price,
                            metadata={"signal_strength": strength, **metadata}
                        )
            
            elif signal_type.upper() == 'HOLD':
                # Do nothing
                pass
    
    def _record_portfolio_snapshot(self, current_data: Dict[str, pd.Series]) -> None:
        """Record portfolio snapshot for current time."""
        snapshot = {
            'timestamp': self.current_time,
            'portfolio_value': self.portfolio_value,
            'capital': self.capital,
            'positions': {},
            'total_positions_value': 0.0
        }
        
        for symbol, position in self.positions.items():
            if symbol in current_data:
                position_snapshot = asdict(position)
                snapshot['positions'][symbol] = position_snapshot
                
                if position.position_type != PositionType.FLAT:
                    position_value = position.quantity * current_data[symbol]['close']
                    if position.position_type == PositionType.SHORT:
                        position_value = -position_value
                    snapshot['total_positions_value'] += position_value
        
        self.portfolio_history.append(snapshot)
    
    def close_all_positions(self) -> List[Trade]:
        """
        Close all open positions.
        
        Returns:
            List[Trade]: List of closed trades
        """
        closed_trades = []
        
        for symbol, position in list(self.positions.items()):
            if position.position_type != PositionType.FLAT:
                # Get current price
                if symbol in self.ohlcv_data and not self.ohlcv_data[symbol].empty:
                    current_price = self.ohlcv_data[symbol].iloc[-1]['close']
                    
                    # Create market order to close position
                    side = OrderSide.SELL if position.position_type == PositionType.LONG else OrderSide.BUY
                    order = self.create_order(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=position.quantity,
                        price=current_price,
                        metadata={"reason": "force_close_all"}
                    )
                    
                    # Execute immediately
                    if self.execute_order(order, current_price):
                        # Get the trade that was created
                        if self.trades:
                            closed_trades.append(self.trades[-1])
        
        self.logger.info(f"Closed all positions: {len(closed_trades)} trades")
        return closed_trades
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete backtest.
        
        Returns:
            Dict[str, Any]: Backtest results
        """
        self.logger.info("Starting backtest...")
        
        # Reset state
        self.is_running = True
        self.is_paused = False
        self.current_index = 0
        self.equity_curve = []
        self.drawdown_curve = []
        self.daily_returns = []
        self.portfolio_history = []
        
        # Initial portfolio value
        self._update_portfolio_value()
        self.equity_curve.append((self.current_time or datetime.now(), self.portfolio_value))
        
        # Main backtest loop
        try:
            while self.step() and self.is_running:
                if self.is_paused:
                    time.sleep(0.1)  # Small sleep when paused
                    continue
            
            # Close any remaining positions
            self.close_all_positions()
            
            # Calculate final metrics
            results = self.calculate_results()
            
            self.logger.info(f"Backtest completed. Final portfolio: ${self.portfolio_value:.2f} "
                           f"({results['total_return_percentage']:.2f}%)")
            
            return results
            
        except KeyboardInterrupt:
            self.logger.warning("Backtest interrupted by user")
            return self.calculate_results()
        
        except Exception as e:
            self.logger.error(f"Error during backtest: {e}")
            self.logger.error(traceback.format_exc())
            return self.calculate_results()
        
        finally:
            self.is_running = False
    
    def calculate_results(self) -> Dict[str, Any]:
        """
        Calculate comprehensive backtest results.
        
        Returns:
            Dict[str, Any]: Complete results dictionary
        """
        if not self.equity_curve:
            return {"error": "No data available for analysis"}
        
        # Extract equity values and timestamps
        timestamps = [t for t, _ in self.equity_curve]
        equity_values = [v for _, v in self.equity_curve]
        
        # Convert to numpy arrays for calculations
        equity_array = np.array(equity_values)
        
        # Calculate returns
        returns = np.diff(equity_array) / equity_array[:-1]
        self.daily_returns = returns.tolist()
        
        # Calculate basic metrics
        initial_capital = self.config.initial_capital
        final_capital = equity_array[-1] if len(equity_array) > 0 else initial_capital
        total_return = final_capital - initial_capital
        total_return_percentage = (total_return / initial_capital) * 100
        
        # Annual return (simplified)
        if len(timestamps) > 1:
            days = (timestamps[-1] - timestamps[0]).days
            years = max(days / 365.25, 0.001)
            annual_return_percentage = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
            annual_return = initial_capital * (annual_return_percentage / 100)
        else:
            annual_return = 0.0
            annual_return_percentage = 0.0
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
        if len(returns) > 0:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
            
            # Sortino ratio (only downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
            sortino_ratio = avg_return / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
            
            volatility = std_return * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            volatility = 0.0
        
        # Calculate trade metrics
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        total_profit = sum([t.pnl for t in winning_trades])
        total_loss = abs(sum([t.pnl for t in losing_trades]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_trade_return = np.mean([t.pnl for t in self.trade_history]) if self.trade_history else 0.0
        
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0.0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        # Calculate consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in self.trade_history:
            if trade.pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Calculate holding periods
        holding_periods = []
        for trade in self.trade_history:
            holding_period = trade.exit_time - trade.entry_time
            holding_periods.append(holding_period)
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else timedelta(0)
        
        # Calculate costs
        total_fees = sum([t.fees for t in self.trade_history]) if self.trade_history else 0.0
        
        # Estimate slippage cost (simplified)
        slippage_cost = 0.0
        for trade in self.trade_history:
            slippage_cost += trade.quantity * trade.entry_price * self.config.slippage
        
        # Calculate efficiency
        gross_profit = total_profit - total_loss
        net_profit = gross_profit - total_fees - slippage_cost
        efficiency_ratio = net_profit / gross_profit if gross_profit > 0 else 0.0
        
        # Create performance metrics object
        metrics = PerformanceMetrics(
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_percentage=total_return_percentage,
            annual_return=annual_return,
            annual_return_percentage=annual_return_percentage,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_percentage=max_drawdown * 100,
            volatility=volatility * 100,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate * 100,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            avg_position_holding_period=avg_holding_period,
            total_fees=total_fees,
            slippage_cost=slippage_cost,
            net_profit_after_costs=net_profit,
            efficiency_ratio=efficiency_ratio * 100,
            start_date=timestamps[0] if timestamps else None,
            end_date=timestamps[-1] if timestamps else None,
            trading_days=len(set([t.date() for t in timestamps])) if timestamps else 0
        )
        
        # Prepare results dictionary
        results = {
            'config': asdict(self.config),
            'metrics': asdict(metrics),
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve,
            'trades': [asdict(t) for t in self.trade_history],
            'orders': [asdict(o) for o in self.order_history],
            'portfolio_history': self.portfolio_history,
            'daily_returns': self.daily_returns,
            'positions': {k: asdict(v) for k, v in self.positions.items()},
            'summary': {
                'final_portfolio_value': final_capital,
                'total_return': total_return,
                'total_return_percentage': total_return_percentage,
                'max_drawdown': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate * 100,
                'profit_factor': profit_factor,
                'total_trades': total_trades
            }
        }
        
        return results
    
    def save_results(self, filepath: str, format: str = 'json') -> None:
        """
        Save backtest results to file.
        
        Args:
            filepath: Path to save file
            format: File format (json, pickle, csv)
        """
        results = self.calculate_results()
        
        path = Path(filepath)
        path.parent.mkdir(exist_ok=True)
        
        if format.lower() == 'json':
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, timedelta):
                    return str(obj)
                return obj
            
            with open(path, 'w') as f:
                json.dump(results, f, default=convert_datetime, indent=2)
        
        elif format.lower() == 'pickle':
            with open(path, 'wb') as f:
                import pickle
                pickle.dump(results, f)
        
        elif format.lower() == 'csv':
            # Save trades to CSV
            trades_df = pd.DataFrame([asdict(t) for t in self.trade_history])
            trades_path = path.with_suffix('.trades.csv')
            trades_df.to_csv(trades_path, index=False)
            
            # Save equity curve to CSV
            equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'portfolio_value'])
            equity_path = path.with_suffix('.equity.csv')
            equity_df.to_csv(equity_path, index=False)
        
        self.logger.info(f"Saved results to {path}")
    
    def plot_results(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save plot (optional)
        
        Returns:
            plt.Figure: Matplotlib figure
        """
        if not self.equity_curve:
            self.logger.warning("No data to plot")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Backtest Results', fontsize=16)
        
        # Extract data
        timestamps = [t for t, _ in self.equity_curve]
        equity_values = [v for _, v in self.equity_curve]
        
        # 1. Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(timestamps, equity_values, 'b-', linewidth=2, label='Portfolio Value')
        ax1.axhline(y=self.config.initial_capital, color='r', linestyle='--', label='Initial Capital')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = axes[0, 1]
        if self.drawdown_curve:
            drawdown_timestamps = [t for t, _ in self.drawdown_curve]
            drawdown_values = [d * 100 for _, d in self.drawdown_curve]
            ax2.fill_between(drawdown_timestamps, drawdown_values, 0, color='red', alpha=0.3)
            ax2.plot(drawdown_timestamps, drawdown_values, 'r-', linewidth=1)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Daily Returns Distribution
        ax3 = axes[1, 0]
        if self.daily_returns:
            ax3.hist([r * 100 for r in self.daily_returns], bins=50, alpha=0.7, color='green')
            ax3.axvline(x=0, color='r', linestyle='--')
            mean_return = np.mean(self.daily_returns) * 100
            ax3.axvline(x=mean_return, color='b', linestyle='--', label=f'Mean: {mean_return:.2f}%')
        ax3.set_title('Daily Returns Distribution')
        ax3.set_xlabel('Daily Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Trade P&L Distribution
        ax4 = axes[1, 1]
        if self.trade_history:
            trade_pnls = [t.pnl for t in self.trade_history]
            winning_trades = [p for p in trade_pnls if p > 0]
            losing_trades = [p for p in trade_pnls if p <= 0]
            
            ax4.hist(winning_trades, bins=30, alpha=0.7, color='green', label=f'Wins: {len(winning_trades)}')
            ax4.hist(losing_trades, bins=30, alpha=0.7, color='red', label=f'Losses: {len(losing_trades)}')
            ax4.axvline(x=0, color='k', linestyle='--')
            avg_trade = np.mean(trade_pnls) if trade_pnls else 0
            ax4.axvline(x=avg_trade, color='b', linestyle='--', label=f'Avg: ${avg_trade:.2f}')
        
        ax4.set_title('Trade P&L Distribution')
        ax4.set_xlabel('Trade P&L ($)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Cumulative Returns by Trade
        ax5 = axes[2, 0]
        if self.trade_history:
            cumulative_pnl = np.cumsum([t.pnl for t in self.trade_history])
            trade_numbers = range(1, len(self.trade_history) + 1)
            
            ax5.plot(trade_numbers, cumulative_pnl, 'g-', linewidth=2, marker='o', markersize=4)
            ax5.axhline(y=0, color='r', linestyle='--')
        
        ax5.set_title('Cumulative P&L by Trade')
        ax5.set_xlabel('Trade Number')
        ax5.set_ylabel('Cumulative P&L ($)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Monthly Returns Heatmap
        ax6 = axes[2, 1]
        if len(timestamps) > 30:
            # Create DataFrame for monthly returns
            equity_df = pd.DataFrame({
                'timestamp': timestamps,
                'value': equity_values
            })
            equity_df.set_index('timestamp', inplace=True)
            
            # Resample to monthly
            monthly_returns = equity_df.resample('M').last().pct_change() * 100
            
            if not monthly_returns.empty:
                # Create heatmap data
                monthly_returns['year'] = monthly_returns.index.year
                monthly_returns['month'] = monthly_returns.index.month_name()
                
                pivot_table = monthly_returns.pivot_table(
                    index='year', 
                    columns='month', 
                    values='value',
                    aggfunc='sum'
                )
                
                # Reorder months
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                             'July', 'August', 'September', 'October', 'November', 'December']
                pivot_table = pivot_table.reindex(columns=month_order, fill_value=0)
                
                # Plot heatmap
                im = ax6.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto')
                ax6.set_title('Monthly Returns Heatmap (%)')
                ax6.set_xlabel('Month')
                ax6.set_ylabel('Year')
                
                # Set ticks
                ax6.set_xticks(range(len(month_order)))
                ax6.set_xticklabels([m[:3] for m in month_order], rotation=45)
                ax6.set_yticks(range(len(pivot_table.index)))
                ax6.set_yticklabels(pivot_table.index)
                
                # Add colorbar
                plt.colorbar(im, ax=ax6)
        
        else:
            ax6.text(0.5, 0.5, 'Insufficient data\nfor monthly heatmap', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Monthly Returns Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def add_event_handler(self, 
                         event_type: str,
                         callback: Callable) -> None:
        """
        Add event handler for backtest events.
        
        Args:
            event_type: Type of event (trade, order, update)
            callback: Callback function
        """
        if event_type == 'trade':
            self.on_trade_callbacks.append(callback)
        elif event_type == 'order':
            self.on_order_callbacks.append(callback)
        elif event_type == 'update':
            self.on_update_callbacks.append(callback)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current backtest state.
        
        Returns:
            Dict[str, Any]: Current state dictionary
        """
        return {
            'current_time': self.current_time,
            'current_index': self.current_index,
            'portfolio_value': self.portfolio_value,
            'capital': self.capital,
            'positions': {k: asdict(v) for k, v in self.positions.items()},
            'active_orders': [asdict(o) for o in self.orders],
            'total_trades': len(self.trade_history)
        }

# Example strategy for testing
class SampleMovingAverageStrategy:
    """Sample moving average crossover strategy."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = f"MA_Crossover_{fast_period}_{slow_period}"
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signals based on MA crossover.
        
        Args:
            data: Dictionary of symbol to DataFrame
        
        Returns:
            Dict[str, Any]: Trading signals
        """
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < self.slow_period:
                continue
            
            # Calculate moving averages
            fast_ma = df['close'].rolling(window=self.fast_period).mean()
            slow_ma = df['close'].rolling(window=self.slow_period).mean()
            
            # Get latest values
            fast_current = fast_ma.iloc[-1]
            fast_prev = fast_ma.iloc[-2] if len(fast_ma) > 1 else fast_current
            slow_current = slow_ma.iloc[-1]
            slow_prev = slow_ma.iloc[-2] if len(slow_ma) > 1 else slow_current
            
            # Generate signal
            if fast_prev <= slow_prev and fast_current > slow_current:
                # Golden cross - BUY signal
                signals[symbol] = {
                    'signal': 'BUY',
                    'strength': 1.0,
                    'metadata': {
                        'fast_ma': fast_current,
                        'slow_ma': slow_current,
                        'strategy': self.name
                    }
                }
            elif fast_prev >= slow_prev and fast_current < slow_current:
                # Death cross - SELL signal
                signals[symbol] = {
                    'signal': 'SELL',
                    'strength': 1.0,
                    'metadata': {
                        'fast_ma': fast_current,
                        'slow_ma': slow_current,
                        'strategy': self.name
                    }
                }
            else:
                # Hold
                signals[symbol] = {
                    'signal': 'HOLD',
                    'strength': 0.0,
                    'metadata': {'strategy': self.name}
                }
        
        return signals

# Utility functions for backtesting
def optimize_strategy_parameters(strategy_class: Any,
                                parameter_grid: Dict[str, List[Any]],
                                data: Dict[str, pd.DataFrame],
                                config: BacktestConfig,
                                metric: str = 'sharpe_ratio',
                                n_jobs: int = 1) -> Dict[str, Any]:
    """
    Optimize strategy parameters using grid search.
    
    Args:
        strategy_class: Strategy class
        parameter_grid: Dictionary of parameter names to lists of values
        data: OHLCV data
        config: Backtest configuration
        metric: Metric to optimize
        n_jobs: Number of parallel jobs
    
    Returns:
        Dict[str, Any]: Optimization results
    """
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(parameter_grid.keys())
    param_values = list(product(*parameter_grid.values()))
    
    results = []
    
    for i, values in enumerate(param_values):
        params = dict(zip(param_names, values))
        
        try:
            # Create backtest engine
            engine = BacktestEngine(config)
            engine.load_data(data)
            engine.add_strategy(strategy_class, **params)
            
            # Run backtest
            result = engine.run()
            
            # Extract metric
            if metric in result['metrics']:
                score = result['metrics'][metric]
            elif metric in result['summary']:
                score = result['summary'][metric]
            else:
                score = result['metrics']['total_return_percentage']
            
            results.append({
                'params': params,
                'score': score,
                'result': result
            })
            
            print(f"Test {i+1}/{len(param_values)}: {params} -> {metric}: {score:.4f}")
        
        except Exception as e:
            print(f"Error with params {params}: {e}")
            results.append({
                'params': params,
                'score': -float('inf'),
                'error': str(e)
            })
    
    # Find best parameters
    valid_results = [r for r in results if isinstance(r['score'], (int, float))]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['score'])
        
        return {
            'best_params': best_result['params'],
            'best_score': best_result['score'],
            'all_results': results,
            'metric': metric
        }
    
    return {'error': 'No valid results found'}

def run_walk_forward_analysis(strategy_class: Any,
                             strategy_params: Dict[str, Any],
                             data: Dict[str, pd.DataFrame],
                             config: BacktestConfig,
                             train_size: float = 0.7,
                             n_windows: int = 5) -> Dict[str, Any]:
    """
    Run walk-forward analysis for strategy validation.
    
    Args:
        strategy_class: Strategy class
        strategy_params: Strategy parameters
        data: OHLCV data
        config: Backtest configuration
        train_size: Proportion of data for training
        n_windows: Number of walk-forward windows
    
    Returns:
        Dict[str, Any]: Walk-forward analysis results
    """
    # For simplicity, use first symbol
    symbol = list(data.keys())[0]
    df = data[symbol]
    
    # Split data into windows
    total_len = len(df)
    train_len = int(total_len * train_size)
    test_len = total_len - train_len
    window_size = test_len // n_windows
    
    results = []
    
    for i in range(n_windows):
        train_start = i * window_size
        train_end = train_start + train_len
        test_start = train_end
        test_end = min(test_start + window_size, total_len)
        
        if test_end - test_start < 10:  # Minimum test size
            continue
        
        # Split data
        train_data = {symbol: df.iloc[train_start:train_end]}
        test_data = {symbol: df.iloc[test_start:test_end]}
        
        # Run backtest on test data
        test_config = copy.deepcopy(config)
        test_config.start_date = test_data[symbol].index[0]
        test_config.end_date = test_data[symbol].index[-1]
        
        engine = BacktestEngine(test_config)
        engine.load_data(test_data)
        engine.add_strategy(strategy_class, **strategy_params)
        
        result = engine.run()
        
        results.append({
            'window': i,
            'train_period': (train_data[symbol].index[0], train_data[symbol].index[-1]),
            'test_period': (test_data[symbol].index[0], test_data[symbol].index[-1]),
            'result': result
        })
        
        print(f"Window {i+1}/{n_windows}: Test return = {result['metrics']['total_return_percentage']:.2f}%")
    
    # Calculate aggregate metrics
    total_returns = [r['result']['metrics']['total_return_percentage'] for r in results]
    sharpe_ratios = [r['result']['metrics']['sharpe_ratio'] for r in results]
    win_rates = [r['result']['metrics']['win_rate'] for r in results]
    
    return {
        'window_results': results,
        'aggregate_metrics': {
            'avg_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'avg_win_rate': np.mean(win_rates),
            'consistency': len([r for r in total_returns if r > 0]) / len(total_returns)
        }
    }

# Test function
if __name__ == "__main__":
    print("Testing Backtest Engine...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(1000) * 50,
        'high': prices + np.abs(np.random.randn(1000) * 100),
        'low': prices - np.abs(np.random.randn(1000) * 100),
        'close': prices,
        'volume': np.random.rand(1000) * 1000
    }, index=dates)
    
    # Create backtest configuration
    config = BacktestConfig(
        initial_capital=10000.0,
        trading_fee=0.001,
        slippage=0.0005,
        timeframe="1h",
        symbols=["BTC/USDT"],
        warmup_period=50,
        position_sizing="percentage",
        position_size=0.1,  # 10% of capital per trade
        stop_loss=0.02,  # 2% stop loss
        take_profit=0.04,  # 4% take profit
        allow_shorting=True,
        verbose=True
    )
    
    # Create and run backtest
    engine = BacktestEngine(config)
    engine.load_data(df, "BTC/USDT")
    engine.add_strategy(SampleMovingAverageStrategy, fast_period=10, slow_period=30)
    
    # Run backtest
    results = engine.run()
    
    # Print summary
    summary = results['summary']
    print("\n" + "="*50)
    print("BACKTEST SUMMARY")
    print("="*50)
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${summary['final_portfolio_value']:,.2f}")
    print(f"Total Return: {summary['total_return_percentage']:.2f}%")
    print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")
    print(f"Win Rate: {summary['win_rate']:.2f}%")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    print(f"Total Trades: {summary['total_trades']}")
    print("="*50)
    
    # Plot results
    engine.plot_results("backtest_results.png")
    
    # Save results
    engine.save_results("backtest_results.json")
    
    print("\nBacktest completed successfully!")
    print("Results saved to backtest_results.json")
    print("Plot saved to backtest_results.png")