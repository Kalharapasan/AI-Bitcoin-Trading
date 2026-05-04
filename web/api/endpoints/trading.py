"""
Trading API Endpoints for Bitcoin Trading AI System
Handles all trading-related operations: execution, orders, positions, and strategies
"""

from fastapi import APIRouter, HTTPException, Depends, Security, status, Query, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import asyncio
from pydantic import BaseModel, Field, validator

# Import project modules
try:
    from config.config_manager import ConfigManager
    from core.trading.signal_generator import SignalGenerator
    from core.trading.position_sizer import PositionSizer
    from core.trading.order_manager import OrderManager
    from core.trading.execution_engine import ExecutionEngine
    from core.risk_management.risk_analyzer import RiskAnalyzer
    from core.risk_management.stop_loss_manager import StopLossManager
    from core.risk_management.portfolio_optimizer import PortfolioOptimizer
    from strategies.ml_strategies import MLStrategy
    from strategies.technical_strategies import TechnicalStrategy
    from strategies.hybrid_strategies import HybridStrategy
    from database.crud import CRUDOperations
    from database.connection import DatabaseConnection
    from core.utils.logger import setup_logger
    from web.api.rest_api import get_current_user, security
except ImportError:
    # For testing purposes
    ConfigManager = type('ConfigManager', (), {})
    SignalGenerator = type('SignalGenerator', (), {})
    PositionSizer = type('PositionSizer', (), {})
    OrderManager = type('OrderManager', (), {})
    ExecutionEngine = type('ExecutionEngine', (), {})
    RiskAnalyzer = type('RiskAnalyzer', (), {})
    StopLossManager = type('StopLossManager', (), {})
    PortfolioOptimizer = type('PortfolioOptimizer', (), {})
    MLStrategy = type('MLStrategy', (), {})
    TechnicalStrategy = type('TechnicalStrategy', (), {})
    HybridStrategy = type('HybridStrategy', (), {})
    CRUDOperations = type('CRUDOperations', (), {})
    DatabaseConnection = type('DatabaseConnection', (), {})
    setup_logger = lambda name: type('Logger', (), {})()
    get_current_user = lambda: "admin"
    security = HTTPBearer()

# Initialize logger
logger = setup_logger(__name__)

# Create router
router = APIRouter(prefix="/api/trading", tags=["trading"])

# Initialize services
config_manager = None
signal_generator = None
position_sizer = None
order_manager = None
execution_engine = None
risk_analyzer = None
stop_loss_manager = None
portfolio_optimizer = None
crud = None
strategies = {}

# Pydantic Models
class TradeRequest(BaseModel):
    """Model for trade execution request"""
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    side: str = Field(description="Buy or sell direction")
    quantity: float = Field(gt=0, description="Amount to trade")
    order_type: str = Field(default="market", description="Order type: market, limit, stop, etc.")
    price: Optional[float] = Field(None, description="Price for limit/stop orders")
    stop_price: Optional[float] = Field(None, description="Stop price for stop orders")
    time_in_force: str = Field(default="GTC", description="Time in force: GTC, IOC, FOK")
    strategy: Optional[str] = Field(None, description="Strategy to use for the trade")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    reduce_only: bool = Field(default=False, description="Reduce position only")
    post_only: bool = Field(default=False, description="Post only order")
    
    @validator('side')
    def validate_side(cls, v):
        if v.lower() not in ['buy', 'sell']:
            raise ValueError('Side must be either "buy" or "sell"')
        return v.lower()
    
    @validator('order_type')
    def validate_order_type(cls, v):
        valid_types = ['market', 'limit', 'stop', 'stop_limit', 'trailing_stop']
        if v.lower() not in valid_types:
            raise ValueError(f'Order type must be one of: {valid_types}')
        return v.lower()
    
    @validator('time_in_force')
    def validate_time_in_force(cls, v):
        valid_tif = ['GTC', 'IOC', 'FOK']
        if v.upper() not in valid_tif:
            raise ValueError(f'Time in force must be one of: {valid_tif}')
        return v.upper()

class OrderModifyRequest(BaseModel):
    """Model for order modification request"""
    order_id: str = Field(description="Order ID to modify")
    quantity: Optional[float] = Field(None, gt=0, description="New quantity")
    price: Optional[float] = Field(None, gt=0, description="New price")
    stop_price: Optional[float] = Field(None, description="New stop price")

class OrderCancelRequest(BaseModel):
    """Model for order cancellation request"""
    order_id: str = Field(description="Order ID to cancel")
    symbol: Optional[str] = Field(None, description="Symbol for validation")

class StrategyConfig(BaseModel):
    """Model for strategy configuration"""
    strategy_name: str = Field(description="Name of the strategy")
    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    timeframe: str = Field(default="1h", description="Timeframe")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    enabled: bool = Field(default=True, description="Enable/disable strategy")
    max_position_size: Optional[float] = Field(None, description="Maximum position size")
    risk_per_trade: Optional[float] = Field(None, description="Risk per trade percentage")

class PositionAdjustRequest(BaseModel):
    """Model for position adjustment request"""
    symbol: str = Field(description="Trading pair symbol")
    adjustment: str = Field(description="hedge, close, partial, or rebalance")
    percentage: Optional[float] = Field(None, ge=0, le=100, description="Percentage for partial close")
    target_size: Optional[float] = Field(None, description="Target position size")

class TradingModeRequest(BaseModel):
    """Model for trading mode change request"""
    mode: str = Field(description="Trading mode: paper, live, backtest")
    exchange: Optional[str] = Field(None, description="Exchange name for live trading")
    api_key: Optional[str] = Field(None, description="API key for live trading")
    api_secret: Optional[str] = Field(None, description="API secret for live trading")

# Initialize trading services
async def initialize_trading_services():
    """Initialize trading services"""
    global config_manager, signal_generator, position_sizer, order_manager
    global execution_engine, risk_analyzer, stop_loss_manager, portfolio_optimizer, crud
    
    try:
        if not config_manager:
            config_manager = ConfigManager()
        
        if not crud:
            db = DatabaseConnection()
            await db.connect()
            crud = CRUDOperations(db)
        
        # Initialize services if not already initialized
        if not signal_generator:
            signal_generator = SignalGenerator(config_manager)
        
        if not position_sizer:
            position_sizer = PositionSizer(config_manager)
        
        if not order_manager:
            order_manager = OrderManager(config_manager, crud)
        
        if not execution_engine:
            execution_engine = ExecutionEngine(config_manager, order_manager)
        
        if not risk_analyzer:
            risk_analyzer = RiskAnalyzer(config_manager, crud)
        
        if not stop_loss_manager:
            stop_loss_manager = StopLossManager(config_manager, order_manager)
        
        if not portfolio_optimizer:
            portfolio_optimizer = PortfolioOptimizer(config_manager, crud)
        
        # Load strategies
        await load_strategies()
        
        logger.info("Trading services initialized successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize trading services: {e}")
        raise

async def load_strategies():
    """Load available trading strategies"""
    global strategies
    
    try:
        # ML Strategies
        strategies["ml_momentum"] = MLStrategy(config_manager, "momentum")
        strategies["ml_mean_reversion"] = MLStrategy(config_manager, "mean_reversion")
        strategies["ml_deep_learning"] = MLStrategy(config_manager, "deep_learning")
        
        # Technical Strategies
        strategies["rsi_strategy"] = TechnicalStrategy(config_manager, "rsi")
        strategies["macd_strategy"] = TechnicalStrategy(config_manager, "macd")
        strategies["bollinger_bands"] = TechnicalStrategy(config_manager, "bollinger")
        strategies["ichimoku"] = TechnicalStrategy(config_manager, "ichimoku")
        
        # Hybrid Strategies
        strategies["ml_technical_hybrid"] = HybridStrategy(config_manager, "ml_technical")
        strategies["ensemble"] = HybridStrategy(config_manager, "ensemble")
        
        logger.info(f"Loaded {len(strategies)} trading strategies")
    
    except Exception as e:
        logger.error(f"Failed to load strategies: {e}")

@router.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await initialize_trading_services()

# Trade Execution Endpoints
@router.post("/execute", summary="Execute a trade")
async def execute_trade(
    trade: TradeRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Execute a trade with specified parameters.
    
    This endpoint handles trade execution with risk management checks,
    position sizing, and order management.
    """
    try:
        await initialize_trading_services()
        
        # Check trading mode
        trading_mode = config_manager.get("trading.mode", "paper")
        if trading_mode == "backtest":
            raise HTTPException(status_code=400, detail="Cannot execute trades in backtest mode")
        
        # Get current price if not provided for market orders
        if trade.order_type == "market" and not trade.price:
            # This would come from market data service
            trade.price = await get_current_price(trade.symbol)
        
        # Calculate position size if quantity not specified
        if not trade.quantity and trade.strategy:
            # Use strategy-based position sizing
            trade.quantity = await calculate_position_size(
                trade.symbol,
                trade.side,
                trade.strategy,
                trade.price
            )
        
        # Risk management checks
        risk_check = await risk_analyzer.check_trade_risk(
            symbol=trade.symbol,
            side=trade.side,
            quantity=trade.quantity,
            price=trade.price
        )
        
        if not risk_check.get("allowed", False):
            raise HTTPException(
                status_code=400,
                detail=f"Trade rejected by risk management: {risk_check.get('reason', 'Unknown')}"
            )
        
        # Execute trade
        trade_result = await execution_engine.execute_trade(
            symbol=trade.symbol,
            side=trade.side,
            quantity=trade.quantity,
            order_type=trade.order_type,
            price=trade.price,
            stop_price=trade.stop_price,
            time_in_force=trade.time_in_force,
            strategy=trade.strategy,
            take_profit=trade.take_profit,
            stop_loss=trade.stop_loss,
            reduce_only=trade.reduce_only,
            post_only=trade.post_only
        )
        
        # Log trade in database
        trade_record = {
            "user_id": current_user,
            "symbol": trade.symbol,
            "side": trade.side,
            "quantity": trade.quantity,
            "order_type": trade.order_type,
            "price": trade_result.get("executed_price", trade.price),
            "order_id": trade_result.get("order_id"),
            "status": trade_result.get("status", "pending"),
            "strategy": trade.strategy,
            "take_profit": trade.take_profit,
            "stop_loss": trade.stop_loss,
            "timestamp": datetime.now(),
            "metadata": {
                "risk_check": risk_check,
                "execution_details": trade_result
            }
        }
        
        await crud.create_trade(trade_record)
        
        # Update stop loss/take profit if specified
        if trade.stop_loss or trade.take_profit:
            await stop_loss_manager.update_order_protection(
                order_id=trade_result.get("order_id"),
                symbol=trade.symbol,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit
            )
        
        # Send real-time update via WebSocket
        await send_trading_update({
            "type": "trade_executed",
            "user": current_user,
            "trade": trade_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": "Trade executed successfully",
            "trade": trade_result,
            "risk_check": risk_check,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_current_price(symbol: str) -> float:
    """Get current market price for a symbol"""
    # This would connect to market data service
    # For now, return a mock price
    return 45000.0  # Mock BTC price

async def calculate_position_size(
    symbol: str,
    side: str,
    strategy: str,
    price: float
) -> float:
    """Calculate optimal position size based on strategy and risk"""
    try:
        # Get account balance
        account_info = await order_manager.get_account_info()
        balance = account_info.get("total_balance", 10000.0)  # Default 10k
        
        # Get risk parameters
        risk_per_trade = config_manager.get("risk.risk_per_trade", 1.0)  # 1% default
        
        # Calculate position size based on risk
        risk_amount = balance * (risk_per_trade / 100)
        
        # For now, simple position sizing
        # In production, this would use volatility, stop loss distance, etc.
        position_size = risk_amount / price
        
        # Apply strategy-specific adjustments
        if strategy in strategies:
            strategy_obj = strategies[strategy]
            position_size = await strategy_obj.adjust_position_size(
                symbol=symbol,
                side=side,
                base_size=position_size,
                price=price
            )
        
        return round(position_size, 6)  # Round to 6 decimal places
    
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        # Default position size
        return 0.01  # 0.01 BTC default

async def send_trading_update(update_data: Dict):
    """Send trading update via WebSocket"""
    # This would connect to WebSocket server
    # For now, just log
    logger.info(f"Trading update: {update_data}")

# Order Management Endpoints
@router.get("/orders", summary="Get orders")
async def get_orders(
    symbol: Optional[str] = None,
    status: Optional[str] = Query(None, regex="^(open|closed|cancelled|all)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get list of orders with optional filtering.
    
    Parameters:
    - symbol: Filter by trading symbol
    - status: Filter by order status (open, closed, cancelled, all)
    - limit: Number of orders to return
    - offset: Pagination offset
    """
    try:
        await initialize_trading_services()
        
        # Get orders from database
        orders = await crud.get_orders(
            user_id=current_user,
            symbol=symbol,
            status=status if status != "all" else None,
            limit=limit,
            offset=offset
        )
        
        # Get open orders from exchange
        open_orders = await order_manager.get_open_orders(symbol)
        
        # Merge database orders with live open orders
        all_orders = []
        order_ids = set()
        
        # Add open orders from exchange
        for order in open_orders:
            order_ids.add(order.get("order_id"))
            order["source"] = "exchange"
            all_orders.append(order)
        
        # Add historical orders from database
        for order in orders:
            if order.get("order_id") not in order_ids:
                order["source"] = "database"
                all_orders.append(order)
        
        # Sort by timestamp descending
        all_orders.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
        
        return {
            "orders": all_orders[:limit],
            "total": len(all_orders),
            "open_count": len(open_orders),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orders/{order_id}", summary="Get order details")
async def get_order_details(
    order_id: str,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get detailed information for a specific order.
    """
    try:
        await initialize_trading_services()
        
        # Try to get order from exchange first
        order = await order_manager.get_order(order_id)
        
        if not order:
            # Try to get from database
            order = await crud.get_order_by_id(order_id, current_user)
        
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        # Get trade history for this order
        trades = await crud.get_trades_by_order(order_id, current_user)
        
        # Get order modifications if any
        modifications = await crud.get_order_modifications(order_id)
        
        return {
            "order": order,
            "trades": trades,
            "modifications": modifications,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching order details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/orders/modify", summary="Modify an order")
async def modify_order(
    modify_request: OrderModifyRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Modify an existing order (price, quantity, etc.)
    """
    try:
        await initialize_trading_services()
        
        # Check if order exists and is open
        order = await order_manager.get_order(modify_request.order_id)
        if not order or order.get("status") not in ["open", "pending"]:
            raise HTTPException(status_code=404, detail="Order not found or not modifiable")
        
        # Risk check for modification
        if modify_request.quantity:
            risk_check = await risk_analyzer.check_trade_risk(
                symbol=order.get("symbol"),
                side=order.get("side"),
                quantity=modify_request.quantity,
                price=modify_request.price or order.get("price")
            )
            
            if not risk_check.get("allowed", False):
                raise HTTPException(
                    status_code=400,
                    detail=f"Modification rejected by risk management: {risk_check.get('reason', 'Unknown')}"
                )
        
        # Modify order
        result = await order_manager.modify_order(
            order_id=modify_request.order_id,
            quantity=modify_request.quantity,
            price=modify_request.price,
            stop_price=modify_request.stop_price
        )
        
        # Log modification
        modification_record = {
            "order_id": modify_request.order_id,
            "user_id": current_user,
            "old_quantity": order.get("quantity"),
            "new_quantity": modify_request.quantity,
            "old_price": order.get("price"),
            "new_price": modify_request.price,
            "timestamp": datetime.now()
        }
        
        await crud.create_order_modification(modification_record)
        
        # Send update
        await send_trading_update({
            "type": "order_modified",
            "user": current_user,
            "order_id": modify_request.order_id,
            "modification": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": "Order modified successfully",
            "modification": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error modifying order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/orders/cancel", summary="Cancel an order")
async def cancel_order(
    cancel_request: OrderCancelRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Cancel an existing order.
    """
    try:
        await initialize_trading_services()
        
        # Cancel order
        result = await order_manager.cancel_order(
            order_id=cancel_request.order_id,
            symbol=cancel_request.symbol
        )
        
        # Log cancellation
        await crud.update_order_status(
            order_id=cancel_request.order_id,
            status="cancelled",
            updated_at=datetime.now()
        )
        
        # Send update
        await send_trading_update({
            "type": "order_cancelled",
            "user": current_user,
            "order_id": cancel_request.order_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": "Order cancelled successfully",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/orders/cancel-all", summary="Cancel all orders")
async def cancel_all_orders(
    symbol: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Cancel all open orders for the current user.
    """
    try:
        await initialize_trading_services()
        
        # Cancel all orders
        result = await order_manager.cancel_all_orders(symbol)
        
        # Update database
        await crud.cancel_all_orders(current_user, symbol)
        
        # Send update
        await send_trading_update({
            "type": "all_orders_cancelled",
            "user": current_user,
            "symbol": symbol,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": "All orders cancelled successfully",
            "cancelled_count": result.get("cancelled", 0),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error cancelling all orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Position Management Endpoints
@router.get("/positions", summary="Get open positions")
async def get_positions(
    symbol: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get current open positions.
    """
    try:
        await initialize_trading_services()
        
        # Get positions from exchange
        positions = await order_manager.get_open_positions(symbol)
        
        # Calculate position metrics
        for position in positions:
            # Add P&L calculation
            position["unrealized_pnl"] = await calculate_unrealized_pnl(position)
            position["pnl_percentage"] = await calculate_pnl_percentage(position)
            
            # Add risk metrics
            position["risk_metrics"] = await risk_analyzer.calculate_position_risk(position)
        
        # Get portfolio metrics
        portfolio_metrics = await portfolio_optimizer.calculate_portfolio_metrics(positions)
        
        return {
            "positions": positions,
            "portfolio_metrics": portfolio_metrics,
            "total_positions": len(positions),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def calculate_unrealized_pnl(position: Dict) -> float:
    """Calculate unrealized P&L for a position"""
    try:
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", entry_price)
        quantity = position.get("quantity", 0)
        
        if position.get("side") == "buy":
            pnl = (current_price - entry_price) * quantity
        else:  # sell (short)
            pnl = (entry_price - current_price) * quantity
        
        return pnl
    except Exception:
        return 0.0

async def calculate_pnl_percentage(position: Dict) -> float:
    """Calculate P&L percentage for a position"""
    try:
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", entry_price)
        
        if entry_price == 0:
            return 0.0
        
        if position.get("side") == "buy":
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
        else:  # sell (short)
            pnl_percent = ((entry_price - current_price) / entry_price) * 100
        
        return pnl_percent
    except Exception:
        return 0.0

@router.post("/positions/adjust", summary="Adjust a position")
async def adjust_position(
    adjustment: PositionAdjustRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Adjust an existing position (hedge, close, partial close, rebalance).
    """
    try:
        await initialize_trading_services()
        
        # Get current position
        positions = await order_manager.get_open_positions(adjustment.symbol)
        if not positions:
            raise HTTPException(status_code=404, detail="No position found for symbol")
        
        position = positions[0]
        
        if adjustment.adjustment == "close":
            # Close entire position
            result = await order_manager.close_position(
                symbol=adjustment.symbol,
                position=position
            )
            
        elif adjustment.adjustment == "partial":
            # Close partial position
            if not adjustment.percentage:
                raise HTTPException(status_code=400, detail="Percentage required for partial close")
            
            quantity = position["quantity"] * (adjustment.percentage / 100)
            result = await order_manager.close_partial_position(
                symbol=adjustment.symbol,
                quantity=quantity,
                position=position
            )
            
        elif adjustment.adjustment == "hedge":
            # Hedge position (open opposite position)
            result = await order_manager.hedge_position(
                symbol=adjustment.symbol,
                position=position
            )
            
        elif adjustment.adjustment == "rebalance":
            # Rebalance to target size
            if not adjustment.target_size:
                raise HTTPException(status_code=400, detail="Target size required for rebalance")
            
            result = await order_manager.rebalance_position(
                symbol=adjustment.symbol,
                current_position=position,
                target_size=adjustment.target_size
            )
            
        else:
            raise HTTPException(status_code=400, detail="Invalid adjustment type")
        
        # Log adjustment
        adjustment_record = {
            "user_id": current_user,
            "symbol": adjustment.symbol,
            "adjustment_type": adjustment.adjustment,
            "parameters": adjustment.dict(exclude_none=True),
            "result": result,
            "timestamp": datetime.now()
        }
        
        await crud.create_position_adjustment(adjustment_record)
        
        # Send update
        await send_trading_update({
            "type": "position_adjusted",
            "user": current_user,
            "symbol": adjustment.symbol,
            "adjustment": adjustment.adjustment,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": f"Position {adjustment.adjustment} executed",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adjusting position: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/positions/close-all", summary="Close all positions")
async def close_all_positions(
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Close all open positions.
    """
    try:
        await initialize_trading_services()
        
        # Get all positions
        positions = await order_manager.get_open_positions()
        
        if not positions:
            return {
                "success": True,
                "message": "No positions to close",
                "closed_count": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Close each position
        results = []
        for position in positions:
            result = await order_manager.close_position(
                symbol=position["symbol"],
                position=position
            )
            results.append(result)
        
        # Log closure
        await crud.close_all_positions(current_user)
        
        # Send update
        await send_trading_update({
            "type": "all_positions_closed",
            "user": current_user,
            "closed_count": len(positions),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": f"Closed {len(positions)} positions",
            "closed_positions": len(positions),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error closing all positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Strategy Management Endpoints
@router.get("/strategies", summary="Get available strategies")
async def get_strategies(
    strategy_type: Optional[str] = Query(None, regex="^(ml|technical|hybrid|all)$"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get list of available trading strategies.
    """
    try:
        await initialize_trading_services()
        
        # Filter strategies by type
        filtered_strategies = {}
        for name, strategy in strategies.items():
            if strategy_type == "all" or not strategy_type:
                filtered_strategies[name] = strategy
            elif strategy_type == "ml" and isinstance(strategy, MLStrategy):
                filtered_strategies[name] = strategy
            elif strategy_type == "technical" and isinstance(strategy, TechnicalStrategy):
                filtered_strategies[name] = strategy
            elif strategy_type == "hybrid" and isinstance(strategy, HybridStrategy):
                filtered_strategies[name] = strategy
        
        # Get strategy information
        strategy_info = {}
        for name, strategy in filtered_strategies.items():
            info = await strategy.get_info()
            strategy_info[name] = {
                "name": name,
                "type": info.get("type", "unknown"),
                "description": info.get("description", ""),
                "parameters": info.get("parameters", {}),
                "performance": info.get("performance", {}),
                "enabled": info.get("enabled", False)
            }
        
        return {
            "strategies": strategy_info,
            "total_strategies": len(strategy_info),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategies/configure", summary="Configure a strategy")
async def configure_strategy(
    config: StrategyConfig,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Configure and enable a trading strategy.
    """
    try:
        await initialize_trading_services()
        
        if config.strategy_name not in strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy = strategies[config.strategy_name]
        
        # Configure strategy
        result = await strategy.configure(
            symbol=config.symbol,
            timeframe=config.timeframe,
            parameters=config.parameters,
            enabled=config.enabled
        )
        
        # Save configuration
        config_record = {
            "user_id": current_user,
            "strategy_name": config.strategy_name,
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "parameters": config.parameters,
            "enabled": config.enabled,
            "configured_at": datetime.now()
        }
        
        await crud.save_strategy_config(config_record)
        
        # If enabled, start strategy
        if config.enabled:
            await strategy.start()
            await send_trading_update({
                "type": "strategy_started",
                "user": current_user,
                "strategy": config.strategy_name,
                "symbol": config.symbol,
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "success": True,
            "message": "Strategy configured successfully",
            "configuration": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategies/{strategy_name}/start", summary="Start a strategy")
async def start_strategy(
    strategy_name: str,
    symbol: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Start a configured trading strategy.
    """
    try:
        await initialize_trading_services()
        
        if strategy_name not in strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy = strategies[strategy_name]
        
        # Start strategy
        result = await strategy.start(symbol=symbol)
        
        # Update configuration
        await crud.update_strategy_status(strategy_name, current_user, True)
        
        # Send update
        await send_trading_update({
            "type": "strategy_started",
            "user": current_user,
            "strategy": strategy_name,
            "symbol": symbol,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": "Strategy started successfully",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategies/{strategy_name}/stop", summary="Stop a strategy")
async def stop_strategy(
    strategy_name: str,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Stop a running trading strategy.
    """
    try:
        await initialize_trading_services()
        
        if strategy_name not in strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy = strategies[strategy_name]
        
        # Stop strategy
        result = await strategy.stop()
        
        # Update configuration
        await crud.update_strategy_status(strategy_name, current_user, False)
        
        # Send update
        await send_trading_update({
            "type": "strategy_stopped",
            "user": current_user,
            "strategy": strategy_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": "Strategy stopped successfully",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies/{strategy_name}/performance", summary="Get strategy performance")
async def get_strategy_performance(
    strategy_name: str,
    period: str = Query("30d", regex="^(7d|30d|90d|180d|1y|all)$"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get performance metrics for a specific strategy.
    """
    try:
        await initialize_trading_services()
        
        # Get trades for this strategy
        trades = await crud.get_trades_by_strategy(strategy_name, current_user, period)
        
        if not trades:
            return {
                "strategy": strategy_name,
                "period": period,
                "trades": 0,
                "performance": {},
                "message": "No trades found for this strategy",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate performance metrics
        performance = await calculate_strategy_performance(trades)
        
        # Get strategy information
        strategy_info = {}
        if strategy_name in strategies:
            strategy = strategies[strategy_name]
            info = await strategy.get_info()
            strategy_info = info.get("performance", {})
        
        return {
            "strategy": strategy_name,
            "period": period,
            "trades": len(trades),
            "performance": performance,
            "strategy_info": strategy_info,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching strategy performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def calculate_strategy_performance(trades: List[Dict]) -> Dict:
    """Calculate performance metrics for a set of trades"""
    if not trades:
        return {}
    
    # Calculate basic metrics
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) <= 0]
    
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    total_volume = sum(t.get("quantity", 0) * t.get("price", 0) for t in trades)
    
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    
    avg_win = sum(t.get("pnl", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.get("pnl", 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Calculate Sharpe ratio (simplified)
    returns = [t.get("pnl_percentage", 0) for t in trades]
    avg_return = np.mean(returns) if returns else 0
    std_return = np.std(returns) if returns else 0
    sharpe_ratio = avg_return / std_return if std_return != 0 else 0
    
    # Calculate max drawdown
    cumulative_pnl = np.cumsum([t.get("pnl", 0) for t in trades])
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    max_drawdown_percentage = (max_drawdown / (running_max[-1] + max_drawdown)) * 100 if (running_max[-1] + max_drawdown) != 0 else 0
    
    return {
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "total_volume": round(total_volume, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_percentage": round(max_drawdown_percentage, 2),
        "avg_trade_duration": "N/A"  # Would require timestamps
    }

# Trading Mode Management
@router.get("/mode", summary="Get current trading mode")
async def get_trading_mode(
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get current trading mode (paper, live, backtest).
    """
    try:
        await initialize_trading_services()
        
        mode = config_manager.get("trading.mode", "paper")
        exchange = config_manager.get("trading.exchange", "binance")
        
        return {
            "mode": mode,
            "exchange": exchange,
            "connected": await order_manager.is_connected(),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching trading mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/mode", summary="Change trading mode")
async def change_trading_mode(
    mode_request: TradingModeRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Change trading mode (paper, live, backtest).
    """
    try:
        await initialize_trading_services()
        
        # Validate mode
        valid_modes = ["paper", "live", "backtest"]
        if mode_request.mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Mode must be one of: {valid_modes}")
        
        # If switching to live, validate exchange credentials
        if mode_request.mode == "live":
            if not mode_request.exchange:
                raise HTTPException(status_code=400, detail="Exchange required for live trading")
            
            # Validate credentials (this would connect to exchange)
            if not mode_request.api_key or not mode_request.api_secret:
                raise HTTPException(status_code=400, detail="API key and secret required for live trading")
            
            # Test connection
            connection_test = await order_manager.test_connection(
                exchange=mode_request.exchange,
                api_key=mode_request.api_key,
                api_secret=mode_request.api_secret
            )
            
            if not connection_test.get("success", False):
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to connect to exchange: {connection_test.get('error', 'Unknown error')}"
                )
        
        # Update configuration
        config_update = {
            "trading.mode": mode_request.mode,
            "trading.exchange": mode_request.exchange if mode_request.exchange else "binance"
        }
        
        if mode_request.api_key and mode_request.api_secret:
            config_update["trading.api_key"] = mode_request.api_key
            config_update["trading.api_secret"] = mode_request.api_secret
        
        config_manager.update_config(config_update)
        
        # Reinitialize order manager with new mode
        await order_manager.initialize(mode_request.mode)
        
        # Log mode change
        mode_record = {
            "user_id": current_user,
            "old_mode": config_manager.get_previous("trading.mode"),
            "new_mode": mode_request.mode,
            "exchange": mode_request.exchange,
            "timestamp": datetime.now()
        }
        
        await crud.log_trading_mode_change(mode_record)
        
        # Send update
        await send_trading_update({
            "type": "trading_mode_changed",
            "user": current_user,
            "mode": mode_request.mode,
            "exchange": mode_request.exchange,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": f"Trading mode changed to {mode_request.mode}",
            "mode": mode_request.mode,
            "exchange": mode_request.exchange,
            "connected": await order_manager.is_connected(),
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing trading mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Trading Statistics
@router.get("/statistics", summary="Get trading statistics")
async def get_trading_statistics(
    period: str = Query("30d", regex="^(1d|7d|30d|90d|180d|1y|all)$"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get comprehensive trading statistics for the specified period.
    """
    try:
        await initialize_trading_services()
        
        # Get trades for period
        trades = await crud.get_trades_by_period(current_user, period)
        
        if not trades:
            return {
                "period": period,
                "trades": 0,
                "statistics": {},
                "message": "No trades found for this period",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate statistics
        statistics = await calculate_trading_statistics(trades)
        
        # Get risk metrics
        risk_metrics = await risk_analyzer.get_trading_statistics(period)
        
        # Get portfolio performance
        portfolio_performance = await portfolio_optimizer.get_performance_metrics(period)
        
        return {
            "period": period,
            "trades": len(trades),
            "statistics": statistics,
            "risk_metrics": risk_metrics,
            "portfolio_performance": portfolio_performance,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching trading statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def calculate_trading_statistics(trades: List[Dict]) -> Dict:
    """Calculate comprehensive trading statistics"""
    if not trades:
        return {}
    
    # Basic metrics
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) <= 0]
    
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    total_volume = sum(abs(t.get("quantity", 0) * t.get("price", 0)) for t in trades)
    
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    
    # Advanced metrics
    pnls = [t.get("pnl", 0) for t in trades]
    returns = [t.get("pnl_percentage", 0) for t in trades]
    
    avg_return = np.mean(returns) if returns else 0
    std_return = np.std(returns) if returns else 0
    
    # Consecutive wins/losses
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    current_wins = 0
    current_losses = 0
    
    for trade in trades:
        if trade.get("pnl", 0) > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    # Largest win/loss
    largest_win = max(pnls) if pnls else 0
    largest_loss = min(pnls) if pnls else 0
    
    # Average holding time (simplified)
    # This would require entry and exit timestamps
    
    return {
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "total_volume": round(total_volume, 2),
        "largest_win": round(largest_win, 2),
        "largest_loss": round(largest_loss, 2),
        "avg_return": round(avg_return, 2),
        "std_return": round(std_return, 2),
        "sharpe_ratio": round(avg_return / std_return, 2) if std_return != 0 else 0,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "profit_factor": round(
            abs(sum(w.get("pnl", 0) for w in winning_trades) / sum(l.get("pnl", 0) for l in losing_trades)), 2
        ) if losing_trades else float('inf')
    }

# Account Information
@router.get("/account", summary="Get account information")
async def get_account_info(
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get current account information and balances.
    """
    try:
        await initialize_trading_services()
        
        # Get account info from exchange
        account_info = await order_manager.get_account_info()
        
        # Get positions for net asset value
        positions = await order_manager.get_open_positions()
        total_position_value = sum(p.get("market_value", 0) for p in positions)
        
        # Calculate total account value
        total_balance = account_info.get("total_balance", 0)
        total_value = total_balance + total_position_value
        
        # Get performance today
        today_performance = await calculate_daily_performance(current_user)
        
        return {
            "account_info": account_info,
            "positions_summary": {
                "total_positions": len(positions),
                "total_value": round(total_position_value, 2),
                "unrealized_pnl": sum(p.get("unrealized_pnl", 0) for p in positions)
            },
            "total_account_value": round(total_value, 2),
            "today_performance": today_performance,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def calculate_daily_performance(user_id: str) -> Dict:
    """Calculate today's trading performance"""
    try:
        # Get today's trades
        today = datetime.now().date()
        trades = await crud.get_trades_by_date(user_id, today)
        
        if not trades:
            return {
                "trades": 0,
                "pnl": 0,
                "volume": 0,
                "win_rate": 0
            }
        
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        
        return {
            "trades": len(trades),
            "pnl": sum(t.get("pnl", 0) for t in trades),
            "volume": sum(abs(t.get("quantity", 0) * t.get("price", 0)) for t in trades),
            "win_rate": len(winning_trades) / len(trades) * 100
        }
    except Exception:
        return {
            "trades": 0,
            "pnl": 0,
            "volume": 0,
            "win_rate": 0
        }

# WebSocket endpoint for real-time trading updates
@router.websocket("/ws")
async def trading_websocket(websocket):
    """
    WebSocket endpoint for real-time trading updates.
    """
    # This would connect to the WebSocket server
    # For now, just accept and close
    await websocket.accept()
    await websocket.close()

# Health check endpoint
@router.get("/health", summary="Trading system health check")
async def trading_health_check(
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Check health of trading system components.
    """
    try:
        await initialize_trading_services()
        
        health_checks = {
            "order_manager": await order_manager.health_check(),
            "execution_engine": await execution_engine.health_check(),
            "risk_analyzer": await risk_analyzer.health_check(),
            "exchange_connection": await order_manager.is_connected(),
            "database": await crud.health_check(),
            "strategies": {}
        }
        
        # Check each strategy
        for name, strategy in strategies.items():
            health_checks["strategies"][name] = await strategy.health_check()
        
        all_healthy = all(
            check.get("healthy", False) if isinstance(check, dict) else check
            for check in health_checks.values() if not isinstance(check, dict)
        ) and all(
            all(s.get("healthy", False) for s in health_checks["strategies"].values())
        )
        
        return {
            "healthy": all_healthy,
            "checks": health_checks,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }