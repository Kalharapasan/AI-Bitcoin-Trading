"""
REST API for Bitcoin Trading AI System
Provides endpoints for trading operations, data access, and system management
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import jwt
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
import asyncio
from contextlib import asynccontextmanager

# Import project modules
try:
    from config.config_manager import ConfigManager
    from core.data_processing.data_collector import DataCollector
    from core.trading.signal_generator import SignalGenerator
    from core.trading.order_manager import OrderManager
    from core.models.model_predictor import ModelPredictor
    from core.monitoring.performance_tracker import PerformanceTracker
    from database.crud import CRUDOperations
    from database.connection import DatabaseConnection
    from core.utils.logger import setup_logger
except ImportError:
    # For testing purposes
    ConfigManager = type('ConfigManager', (), {})
    DataCollector = type('DataCollector', (), {})
    SignalGenerator = type('SignalGenerator', (), {})
    OrderManager = type('OrderManager', (), {})
    ModelPredictor = type('ModelPredictor', (), {})
    PerformanceTracker = type('PerformanceTracker', (), {})
    CRUDOperations = type('CRUDOperations', (), {})
    DatabaseConnection = type('DatabaseConnection', (), {})
    setup_logger = lambda name: type('Logger', (), {})()

# Initialize logger
logger = setup_logger(__name__)

# Security
security = HTTPBearer()

# Pydantic Models for Request/Response
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class TradeRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    side: str = Field(description="buy or sell")
    quantity: float = Field(gt=0, description="Amount to trade")
    order_type: str = Field(default="market", description="order type: market, limit, etc.")
    price: Optional[float] = Field(None, description="Price for limit orders")
    strategy: Optional[str] = Field(None, description="Strategy to use for the trade")

class PredictionRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Timeframe for prediction")
    model_name: Optional[str] = Field(None, description="Specific model to use")

class BacktestRequest(BaseModel):
    strategy: str
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    start_date: str
    end_date: str
    initial_balance: float = 10000.0
    parameters: Optional[Dict[str, Any]] = None

class ModelTrainRequest(BaseModel):
    model_name: str
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    lookback_period: int = 30
    retrain: bool = False

class AlertConfig(BaseModel):
    alert_type: str
    condition: str
    threshold: float
    enabled: bool = True
    notification_method: Optional[str] = "in_app"

# Global instances
config_manager = None
data_collector = None
signal_generator = None
order_manager = None
model_predictor = None
performance_tracker = None
db = None
crud = None

# JWT Configuration
JWT_SECRET = "your-secret-key-change-in-production"  # Should be in config
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = timedelta(hours=24)

# In-memory cache for frequently accessed data
cache = {
    "market_data": {},
    "predictions": {},
    "performance_metrics": {},
    "last_update": {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI startup/shutdown events
    """
    # Startup
    logger.info("Starting REST API server...")
    await initialize_services()
    yield
    # Shutdown
    logger.info("Shutting down REST API server...")
    await cleanup_services()

# Initialize FastAPI app
app = FastAPI(
    title="Bitcoin Trading AI REST API",
    description="REST API for AI-powered Bitcoin Trading System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication and Authorization
def create_jwt_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create JWT token for authenticated user
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + JWT_EXPIRATION
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_jwt_token(token: str):
    """
    Verify JWT token
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Dependency to get current user from JWT token
    """
    token = credentials.credentials
    payload = verify_jwt_token(token)
    username = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return username

async def initialize_services():
    """
    Initialize all required services
    """
    global config_manager, data_collector, signal_generator
    global order_manager, model_predictor, performance_tracker, db, crud
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        
        # Initialize database
        db = DatabaseConnection()
        await db.connect()
        crud = CRUDOperations(db)
        
        # Initialize services
        data_collector = DataCollector(config_manager)
        signal_generator = SignalGenerator(config_manager)
        order_manager = OrderManager(config_manager, crud)
        model_predictor = ModelPredictor(config_manager)
        performance_tracker = PerformanceTracker(crud)
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

async def cleanup_services():
    """
    Cleanup services on shutdown
    """
    try:
        if db:
            await db.disconnect()
        logger.info("Services cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Authentication Endpoints
@app.post("/api/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT token
    """
    # In production, validate against database
    if request.username != "admin" or request.password != "admin123":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token_data = {"sub": request.username, "role": "admin"}
    access_token = create_jwt_token(token_data)
    
    return TokenResponse(
        access_token=access_token,
        expires_in=int(JWT_EXPIRATION.total_seconds())
    )

@app.post("/api/auth/refresh")
async def refresh_token(current_user: str = Depends(get_current_user)):
    """
    Refresh JWT token
    """
    token_data = {"sub": current_user, "role": "admin"}
    access_token = create_jwt_token(token_data)
    
    return TokenResponse(
        access_token=access_token,
        expires_in=int(JWT_EXPIRATION.total_seconds())
    )

# Market Data Endpoints
@app.get("/api/market/price/{symbol}")
async def get_current_price(
    symbol: str = "BTCUSDT",
    current_user: str = Depends(get_current_user)
):
    """
    Get current price for a symbol
    """
    try:
        cache_key = f"price_{symbol}"
        
        # Check cache
        if cache_key in cache["market_data"]:
            cached_data = cache["market_data"][cache_key]
            if datetime.now() - cached_data["timestamp"] < timedelta(seconds=5):
                return cached_data["data"]
        
        # Fetch fresh data
        price = await data_collector.get_current_price(symbol)
        
        # Update cache
        cache["market_data"][cache_key] = {
            "data": {"symbol": symbol, "price": price, "timestamp": datetime.now().isoformat()},
            "timestamp": datetime.now()
        }
        
        return {"symbol": symbol, "price": price, "timestamp": datetime.now().isoformat()}
    
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/ohlcv/{symbol}")
async def get_ohlcv_data(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    limit: int = Query(100, ge=1, le=1000),
    current_user: str = Depends(get_current_user)
):
    """
    Get OHLCV data for a symbol
    """
    try:
        cache_key = f"ohlcv_{symbol}_{timeframe}_{limit}"
        
        # Check cache
        if cache_key in cache["market_data"]:
            cached_data = cache["market_data"][cache_key]
            if datetime.now() - cached_data["timestamp"] < timedelta(seconds=30):
                return cached_data["data"]
        
        # Fetch fresh data
        ohlcv_data = await data_collector.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        
        # Update cache
        cache["market_data"][cache_key] = {
            "data": ohlcv_data,
            "timestamp": datetime.now()
        }
        
        return ohlcv_data
    
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/orderbook/{symbol}")
async def get_orderbook(
    symbol: str = "BTCUSDT",
    depth: int = Query(20, ge=5, le=100),
    current_user: str = Depends(get_current_user)
):
    """
    Get orderbook data for a symbol
    """
    try:
        orderbook = await data_collector.get_orderbook(symbol, depth)
        return orderbook
    except Exception as e:
        logger.error(f"Error fetching orderbook for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Trading Endpoints
@app.post("/api/trading/execute")
async def execute_trade(
    trade: TradeRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Execute a trade
    """
    try:
        # Validate trade parameters
        if trade.side not in ["buy", "sell"]:
            raise HTTPException(status_code=400, detail="Side must be 'buy' or 'sell'")
        
        # Execute trade through order manager
        trade_result = await order_manager.execute_trade(
            symbol=trade.symbol,
            side=trade.side,
            quantity=trade.quantity,
            order_type=trade.order_type,
            price=trade.price,
            strategy=trade.strategy
        )
        
        # Log the trade
        await crud.create_trade({
            "user_id": current_user,
            "symbol": trade.symbol,
            "side": trade.side,
            "quantity": trade.quantity,
            "price": trade_result.get("executed_price"),
            "order_type": trade.order_type,
            "status": trade_result.get("status"),
            "timestamp": datetime.now()
        })
        
        return trade_result
    
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trading/positions")
async def get_open_positions(
    symbol: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """
    Get all open positions
    """
    try:
        positions = await order_manager.get_open_positions(symbol)
        return positions
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trading/close-all")
async def close_all_positions(
    symbol: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """
    Close all open positions
    """
    try:
        result = await order_manager.close_all_positions(symbol)
        return {"message": "All positions closed", "details": result}
    except Exception as e:
        logger.error(f"Error closing positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prediction Endpoints
@app.post("/api/predict")
async def get_prediction(
    request: PredictionRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Get price prediction from ML models
    """
    try:
        cache_key = f"prediction_{request.symbol}_{request.timeframe}_{request.model_name}"
        
        # Check cache
        if cache_key in cache["predictions"]:
            cached_data = cache["predictions"][cache_key]
            if datetime.now() - cached_data["timestamp"] < timedelta(minutes=5):
                return cached_data["data"]
        
        # Get prediction
        prediction = await model_predictor.predict(
            symbol=request.symbol,
            timeframe=request.timeframe,
            model_name=request.model_name
        )
        
        # Update cache
        cache["predictions"][cache_key] = {
            "data": prediction,
            "timestamp": datetime.now()
        }
        
        return prediction
    
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict/models")
async def get_available_models(current_user: str = Depends(get_current_user)):
    """
    Get list of available ML models
    """
    try:
        models = await model_predictor.get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict/performance")
async def get_model_performance(
    model_name: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    current_user: str = Depends(get_current_user)
):
    """
    Get performance metrics for ML models
    """
    try:
        performance = await performance_tracker.get_model_performance(model_name, days)
        return performance
    except Exception as e:
        logger.error(f"Error fetching model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Signal Endpoints
@app.get("/api/signals/current")
async def get_current_signals(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    current_user: str = Depends(get_current_user)
):
    """
    Get current trading signals
    """
    try:
        signals = await signal_generator.generate_signals(symbol, timeframe)
        return signals
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals/history")
async def get_signal_history(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    limit: int = Query(100, ge=1, le=1000),
    current_user: str = Depends(get_current_user)
):
    """
    Get historical trading signals
    """
    try:
        signals = await crud.get_signals(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        return signals
    except Exception as e:
        logger.error(f"Error fetching signal history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Backtesting Endpoints
@app.post("/api/backtest/run")
async def run_backtest(
    request: BacktestRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Run a backtest for a strategy
    """
    try:
        from backtesting.backtest_engine import BacktestEngine
        
        backtester = BacktestEngine(config_manager)
        
        results = await backtester.run_backtest(
            strategy=request.strategy,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_balance=request.initial_balance,
            parameters=request.parameters
        )
        
        # Store backtest results
        await crud.create_backtest_result({
            "user_id": current_user,
            "strategy": request.strategy,
            "symbol": request.symbol,
            "results": results,
            "timestamp": datetime.now()
        })
        
        return results
    
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtest/results")
async def get_backtest_results(
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    current_user: str = Depends(get_current_user)
):
    """
    Get historical backtest results
    """
    try:
        results = await crud.get_backtest_results(
            user_id=current_user,
            strategy=strategy,
            symbol=symbol,
            limit=limit
        )
        return results
    except Exception as e:
        logger.error(f"Error fetching backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model Management Endpoints
@app.post("/api/models/train")
async def train_model(
    request: ModelTrainRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Train or retrain an ML model
    """
    try:
        from core.models.model_trainer import ModelTrainer
        
        trainer = ModelTrainer(config_manager)
        
        training_result = await trainer.train_model(
            model_name=request.model_name,
            symbol=request.symbol,
            timeframe=request.timeframe,
            lookback_period=request.lookback_period,
            retrain=request.retrain
        )
        
        # Clear prediction cache for this model
        for key in list(cache["predictions"].keys()):
            if request.model_name in key:
                del cache["predictions"][key]
        
        return training_result
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/status")
async def get_model_status(
    model_name: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """
    Get status of ML models
    """
    try:
        status = await model_predictor.get_model_status(model_name)
        return status
    except Exception as e:
        logger.error(f"Error fetching model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk Management Endpoints
@app.get("/api/risk/metrics")
async def get_risk_metrics(current_user: str = Depends(get_current_user)):
    """
    Get current risk metrics
    """
    try:
        from core.risk_management.risk_analyzer import RiskAnalyzer
        
        risk_analyzer = RiskAnalyzer(config_manager, crud)
        metrics = await risk_analyzer.calculate_risk_metrics()
        
        # Update cache
        cache["performance_metrics"]["risk"] = {
            "data": metrics,
            "timestamp": datetime.now()
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/risk/exposure")
async def get_exposure(current_user: str = Depends(get_current_user)):
    """
    Get current exposure levels
    """
    try:
        from core.risk_management.risk_analyzer import RiskAnalyzer
        
        risk_analyzer = RiskAnalyzer(config_manager, crud)
        exposure = await risk_analyzer.get_current_exposure()
        
        return exposure
    
    except Exception as e:
        logger.error(f"Error fetching exposure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance Endpoints
@app.get("/api/performance/summary")
async def get_performance_summary(
    period: str = Query("7d", regex="^(1d|7d|30d|90d|1y|all)$"),
    current_user: str = Depends(get_current_user)
):
    """
    Get performance summary for specified period
    """
    try:
        summary = await performance_tracker.get_performance_summary(period)
        return summary
    except Exception as e:
        logger.error(f"Error fetching performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/performance/trades")
async def get_trade_history(
    symbol: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: str = Depends(get_current_user)
):
    """
    Get trade history
    """
    try:
        trades = await crud.get_trades(
            user_id=current_user,
            symbol=symbol,
            limit=limit,
            offset=offset
        )
        return trades
    except Exception as e:
        logger.error(f"Error fetching trade history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/performance/equity-curve")
async def get_equity_curve(
    period: str = Query("30d", regex="^(7d|30d|90d|1y|all)$"),
    current_user: str = Depends(get_current_user)
):
    """
    Get equity curve data
    """
    try:
        equity_curve = await performance_tracker.get_equity_curve(period)
        return equity_curve
    except Exception as e:
        logger.error(f"Error fetching equity curve: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Management Endpoints
@app.get("/api/system/status")
async def get_system_status(current_user: str = Depends(get_current_user)):
    """
    Get overall system status
    """
    try:
        status = {
            "api": "running",
            "database": "connected" if db and db.is_connected else "disconnected",
            "exchange_connection": "connected" if data_collector else "disconnected",
            "models": [],
            "cache": {
                "market_data": len(cache.get("market_data", {})),
                "predictions": len(cache.get("predictions", {})),
                "performance_metrics": len(cache.get("performance_metrics", {}))
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Get model statuses
        if model_predictor:
            model_statuses = await model_predictor.get_model_status()
            status["models"] = model_statuses
        
        return status
    
    except Exception as e:
        logger.error(f"Error fetching system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/cache/clear")
async def clear_cache(
    cache_type: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """
    Clear API cache
    """
    try:
        if cache_type:
            if cache_type in cache:
                cache[cache_type].clear()
                message = f"Cleared {cache_type} cache"
            else:
                raise HTTPException(status_code=400, detail=f"Invalid cache type: {cache_type}")
        else:
            for key in cache:
                cache[key].clear()
            message = "Cleared all caches"
        
        return {"message": message, "timestamp": datetime.now().isoformat()}
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alert Endpoints
@app.get("/api/alerts")
async def get_alerts(
    active_only: bool = True,
    limit: int = Query(50, ge=1, le=1000),
    current_user: str = Depends(get_current_user)
):
    """
    Get system alerts
    """
    try:
        from core.monitoring.alert_manager import AlertManager
        
        alert_manager = AlertManager(config_manager, crud)
        alerts = await alert_manager.get_alerts(active_only, limit)
        
        return alerts
    
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/alerts/configure")
async def configure_alert(
    config: AlertConfig,
    current_user: str = Depends(get_current_user)
):
    """
    Configure a new alert
    """
    try:
        from core.monitoring.alert_manager import AlertManager
        
        alert_manager = AlertManager(config_manager, crud)
        
        alert_id = await alert_manager.configure_alert(
            alert_type=config.alert_type,
            condition=config.condition,
            threshold=config.threshold,
            enabled=config.enabled,
            notification_method=config.notification_method
        )
        
        return {"message": "Alert configured", "alert_id": alert_id}
    
    except Exception as e:
        logger.error(f"Error configuring alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export Endpoints
@app.get("/api/export/trades/csv")
async def export_trades_csv(
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """
    Export trades to CSV
    """
    try:
        trades = await crud.get_trades(
            user_id=current_user,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        if not trades:
            raise HTTPException(status_code=404, detail="No trades found")
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        # Create CSV
        csv_path = f"/tmp/trades_export_{current_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        
        return FileResponse(
            path=csv_path,
            filename=f"trades_export_{datetime.now().strftime('%Y%m%d')}.csv",
            media_type="text/csv"
        )
    
    except Exception as e:
        logger.error(f"Error exporting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Webhook Endpoints (for external integrations)
@app.post("/api/webhooks/tradingview")
async def tradingview_webhook(
    payload: Dict[str, Any],
    current_user: str = Depends(get_current_user)
):
    """
    Webhook endpoint for TradingView alerts
    """
    try:
        logger.info(f"Received TradingView webhook: {payload}")
        
        # Process TradingView alert
        if payload.get("action") == "buy" or payload.get("action") == "sell":
            # Execute trade based on TradingView signal
            trade_result = await order_manager.execute_trade(
                symbol=payload.get("symbol", "BTCUSDT"),
                side=payload.get("action"),
                quantity=payload.get("quantity", 0.01),
                order_type="market",
                strategy="tradingview_webhook"
            )
            
            return {"message": "Trade executed via TradingView", "result": trade_result}
        
        return {"message": "Webhook received", "payload": payload}
    
    except Exception as e:
        logger.error(f"Error processing TradingView webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoint (no auth required)
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  # Set to False in production
    )