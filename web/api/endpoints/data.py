"""
Data API Endpoints for Bitcoin Trading AI System
Handles market data, historical data, feature engineering, and data management
"""

from fastapi import APIRouter, HTTPException, Depends, Security, status, Query, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List, Dict, Any, Union, Generator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import asyncio
import csv
import io
from pydantic import BaseModel, Field, validator

# Import project modules
try:
    from config.config_manager import ConfigManager
    from core.data_processing.data_collector import DataCollector
    from core.data_processing.feature_engineer import FeatureEngineer
    from core.data_processing.data_preprocessor import DataPreprocessor
    from core.data_processing.data_validator import DataValidator
    from database.crud import CRUDOperations
    from database.connection import DatabaseConnection
    from core.utils.logger import setup_logger
    from core.utils.cache import CacheManager
    from web.api.rest_api import get_current_user, security
except ImportError:
    # For testing purposes
    ConfigManager = type('ConfigManager', (), {})
    DataCollector = type('DataCollector', (), {})
    FeatureEngineer = type('FeatureEngineer', (), {})
    DataPreprocessor = type('DataPreprocessor', (), {})
    DataValidator = type('DataValidator', (), {})
    CRUDOperations = type('CRUDOperations', (), {})
    DatabaseConnection = type('DatabaseConnection', (), {})
    setup_logger = lambda name: type('Logger', (), {})()
    CacheManager = type('CacheManager', (), {})
    get_current_user = lambda: "admin"
    security = HTTPBearer()

# Initialize logger
logger = setup_logger(__name__)

# Create router
router = APIRouter(prefix="/api/data", tags=["data"])

# Initialize services
config_manager = None
data_collector = None
feature_engineer = None
data_preprocessor = None
data_validator = None
crud = None
cache_manager = None

# Pydantic Models
class DataRequest(BaseModel):
    """Model for data retrieval request"""
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Timeframe for data")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    limit: Optional[int] = Field(1000, ge=1, le=10000, description="Number of candles to retrieve")
    include_features: bool = Field(default=False, description="Include engineered features")
    normalize: bool = Field(default=False, description="Normalize the data")
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        if v not in valid_timeframes:
            raise ValueError(f'Timeframe must be one of: {valid_timeframes}')
        return v

class FeatureRequest(BaseModel):
    """Model for feature engineering request"""
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Timeframe for data")
    features: List[str] = Field(default_factory=lambda: ['all'], description="List of features to calculate")
    lookback_period: Optional[int] = Field(None, description="Lookback period for features")
    start_date: Optional[str] = Field(None, description="Start date for feature calculation")

class DataValidationRequest(BaseModel):
    """Model for data validation request"""
    data: List[Dict[str, Any]] = Field(description="Data to validate")
    validation_rules: Optional[Dict[str, Any]] = Field(None, description="Custom validation rules")
    strict_mode: bool = Field(default=False, description="Enable strict validation mode")

class DataUpdateRequest(BaseModel):
    """Model for data update request"""
    symbol: str = Field(description="Trading pair symbol")
    timeframe: str = Field(description="Timeframe for data")
    force_refresh: bool = Field(default=False, description="Force refresh from source")
    update_interval: Optional[str] = Field(None, description="Auto-update interval")

class DataExportRequest(BaseModel):
    """Model for data export request"""
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    timeframe: Optional[str] = Field(None, description="Timeframe for data")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date")
    format: str = Field(default="csv", description="Export format: csv, json, parquet")
    include_features: bool = Field(default=False, description="Include engineered features")
    compression: Optional[str] = Field(None, description="Compression type: gzip, zip")

# Initialize data services
async def initialize_data_services():
    """Initialize data services"""
    global config_manager, data_collector, feature_engineer
    global data_preprocessor, data_validator, crud, cache_manager
    
    try:
        if not config_manager:
            config_manager = ConfigManager()
        
        if not cache_manager:
            cache_manager = CacheManager(config_manager)
        
        if not crud:
            db = DatabaseConnection()
            await db.connect()
            crud = CRUDOperations(db)
        
        # Initialize services if not already initialized
        if not data_collector:
            data_collector = DataCollector(config_manager, cache_manager)
        
        if not feature_engineer:
            feature_engineer = FeatureEngineer(config_manager)
        
        if not data_preprocessor:
            data_preprocessor = DataPreprocessor(config_manager)
        
        if not data_validator:
            data_validator = DataValidator(config_manager)
        
        logger.info("Data services initialized successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize data services: {e}")
        raise

@router.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await initialize_data_services()

# Market Data Endpoints
@router.get("/price/{symbol}", summary="Get current price")
async def get_current_price(
    symbol: str,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get current market price for a symbol.
    """
    try:
        await initialize_data_services()
        
        # Check cache first
        cache_key = f"price_{symbol}"
        cached_price = await cache_manager.get(cache_key)
        
        if cached_price and not is_cache_expired(cached_price.get('timestamp'), seconds=10):
            return {
                "symbol": symbol,
                "price": cached_price['price'],
                "source": "cache",
                "timestamp": cached_price['timestamp']
            }
        
        # Fetch from exchange
        price = await data_collector.get_current_price(symbol)
        
        # Update cache
        price_data = {
            "price": price,
            "timestamp": datetime.now().isoformat()
        }
        await cache_manager.set(cache_key, price_data, ttl=60)
        
        return {
            "symbol": symbol,
            "price": price,
            "source": "exchange",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def is_cache_expired(timestamp_str: str, seconds: int = 300) -> bool:
    """Check if cache entry is expired"""
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        return (datetime.now() - timestamp).total_seconds() > seconds
    except:
        return True

@router.get("/prices", summary="Get multiple prices")
async def get_multiple_prices(
    symbols: List[str] = Query(["BTCUSDT", "ETHUSDT"], description="List of symbols"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get current prices for multiple symbols.
    """
    try:
        await initialize_data_services()
        
        prices = {}
        for symbol in symbols:
            try:
                price_data = await get_current_price(symbol)
                prices[symbol] = price_data
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")
                prices[symbol] = {
                    "symbol": symbol,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        return {
            "prices": prices,
            "total_symbols": len(symbols),
            "successful": sum(1 for p in prices.values() if 'price' in p),
            "failed": sum(1 for p in prices.values() if 'error' in p),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching multiple prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ticker/{symbol}", summary="Get ticker information")
async def get_ticker(
    symbol: str,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get detailed ticker information for a symbol.
    """
    try:
        await initialize_data_services()
        
        # Check cache
        cache_key = f"ticker_{symbol}"
        cached_ticker = await cache_manager.get(cache_key)
        
        if cached_ticker and not is_cache_expired(cached_ticker.get('timestamp'), seconds=30):
            cached_ticker['source'] = 'cache'
            return cached_ticker
        
        # Fetch from exchange
        ticker = await data_collector.get_ticker(symbol)
        
        # Add metadata
        ticker_data = {
            "symbol": symbol,
            "price": ticker.get('lastPrice', 0),
            "price_change": ticker.get('priceChange', 0),
            "price_change_percent": ticker.get('priceChangePercent', 0),
            "volume": ticker.get('volume', 0),
            "quote_volume": ticker.get('quoteVolume', 0),
            "high": ticker.get('highPrice', 0),
            "low": ticker.get('lowPrice', 0),
            "open": ticker.get('openPrice', 0),
            "close": ticker.get('closePrice', 0),
            "bid": ticker.get('bidPrice', 0),
            "ask": ticker.get('askPrice', 0),
            "bid_qty": ticker.get('bidQty', 0),
            "ask_qty": ticker.get('askQty', 0),
            "timestamp": datetime.now().isoformat(),
            "source": "exchange"
        }
        
        # Update cache
        await cache_manager.set(cache_key, ticker_data, ttl=120)
        
        return ticker_data
    
    except Exception as e:
        logger.error(f"Error fetching ticker for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orderbook/{symbol}", summary="Get order book")
async def get_orderbook(
    symbol: str,
    depth: int = Query(20, ge=5, le=100, description="Order book depth"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get order book (market depth) for a symbol.
    """
    try:
        await initialize_data_services()
        
        # Check cache
        cache_key = f"orderbook_{symbol}_{depth}"
        cached_orderbook = await cache_manager.get(cache_key)
        
        if cached_orderbook and not is_cache_expired(cached_orderbook.get('timestamp'), seconds=5):
            cached_orderbook['source'] = 'cache'
            return cached_orderbook
        
        # Fetch from exchange
        orderbook = await data_collector.get_orderbook(symbol, depth)
        
        # Calculate order book metrics
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        orderbook_data = {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "best_bid": bids[0][0] if bids else 0,
            "best_ask": asks[0][0] if asks else 0,
            "spread": (asks[0][0] - bids[0][0]) if bids and asks else 0,
            "spread_percent": ((asks[0][0] - bids[0][0]) / bids[0][0] * 100) if bids and asks and bids[0][0] > 0 else 0,
            "total_bid_volume": sum(bid[1] for bid in bids),
            "total_ask_volume": sum(ask[1] for ask in asks),
            "timestamp": datetime.now().isoformat(),
            "source": "exchange"
        }
        
        # Update cache
        await cache_manager.set(cache_key, orderbook_data, ttl=10)
        
        return orderbook_data
    
    except Exception as e:
        logger.error(f"Error fetching orderbook for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trades/{symbol}", summary="Get recent trades")
async def get_recent_trades(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000, description="Number of trades to retrieve"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get recent trades for a symbol.
    """
    try:
        await initialize_data_services()
        
        # Check cache
        cache_key = f"trades_{symbol}_{limit}"
        cached_trades = await cache_manager.get(cache_key)
        
        if cached_trades and not is_cache_expired(cached_trades.get('timestamp'), seconds=3):
            cached_trades['source'] = 'cache'
            return cached_trades
        
        # Fetch from exchange
        trades = await data_collector.get_recent_trades(symbol, limit)
        
        # Calculate trade metrics
        buy_trades = [t for t in trades if t.get('isBuyerMaker', False)]
        sell_trades = [t for t in trades if not t.get('isBuyerMaker', False)]
        
        trades_data = {
            "symbol": symbol,
            "trades": trades,
            "total_trades": len(trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "total_volume": sum(float(t.get('qty', 0)) for t in trades),
            "buy_volume": sum(float(t.get('qty', 0)) for t in buy_trades),
            "sell_volume": sum(float(t.get('qty', 0)) for t in sell_trades),
            "avg_price": np.mean([float(t.get('price', 0)) for t in trades]) if trades else 0,
            "timestamp": datetime.now().isoformat(),
            "source": "exchange"
        }
        
        # Update cache
        await cache_manager.set(cache_key, trades_data, ttl=5)
        
        return trades_data
    
    except Exception as e:
        logger.error(f"Error fetching trades for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Historical Data Endpoints
@router.post("/historical", summary="Get historical OHLCV data")
async def get_historical_data(
    request: DataRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get historical OHLCV (Open, High, Low, Close, Volume) data.
    """
    try:
        await initialize_data_services()
        
        # Generate cache key
        cache_key = f"historical_{request.symbol}_{request.timeframe}_{request.start_date}_{request.end_date}_{request.limit}"
        
        # Check cache
        cached_data = await cache_manager.get(cache_key)
        if cached_data and not is_cache_expired(cached_data.get('timestamp'), seconds=300):
            cached_data['source'] = 'cache'
            return cached_data
        
        # Fetch historical data
        ohlcv_data = await data_collector.get_historical_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            limit=request.limit
        )
        
        if not ohlcv_data:
            raise HTTPException(status_code=404, detail="No data found for the specified parameters")
        
        # Validate data
        validation_result = await data_validator.validate_ohlcv(ohlcv_data)
        if not validation_result.get('valid', False):
            logger.warning(f"Data validation issues: {validation_result.get('issues', [])}")
        
        # Calculate basic statistics
        statistics = calculate_ohlcv_statistics(ohlcv_data)
        
        # Prepare response
        response_data = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "data": ohlcv_data,
            "statistics": statistics,
            "total_records": len(ohlcv_data),
            "validation": validation_result,
            "timestamp": datetime.now().isoformat(),
            "source": "exchange"
        }
        
        # Add features if requested
        if request.include_features:
            features = await calculate_features(ohlcv_data, request.timeframe)
            response_data["features"] = features
        
        # Normalize if requested
        if request.normalize:
            normalized_data = await data_preprocessor.normalize_data(ohlcv_data)
            response_data["normalized_data"] = normalized_data
        
        # Update cache
        await cache_manager.set(cache_key, response_data, ttl=600)
        
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_ohlcv_statistics(ohlcv_data: List[Dict]) -> Dict:
    """Calculate statistics for OHLCV data"""
    if not ohlcv_data:
        return {}
    
    closes = [candle['close'] for candle in ohlcv_data]
    volumes = [candle['volume'] for candle in ohlcv_data]
    
    return {
        "price": {
            "min": min(closes),
            "max": max(closes),
            "mean": np.mean(closes),
            "median": np.median(closes),
            "std": np.std(closes),
            "last": closes[-1],
            "change": ((closes[-1] - closes[0]) / closes[0] * 100) if closes[0] > 0 else 0
        },
        "volume": {
            "min": min(volumes),
            "max": max(volumes),
            "mean": np.mean(volumes),
            "median": np.median(volumes),
            "std": np.std(volumes),
            "total": sum(volumes)
        },
        "candles": {
            "green": sum(1 for c in ohlcv_data if c['close'] > c['open']),
            "red": sum(1 for c in ohlcv_data if c['close'] < c['open']),
            "doji": sum(1 for c in ohlcv_data if abs(c['close'] - c['open']) / c['open'] < 0.001)
        }
    }

async def calculate_features(ohlcv_data: List[Dict], timeframe: str) -> Dict:
    """Calculate technical features for OHLCV data"""
    try:
        if not feature_engineer or not ohlcv_data:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data)
        
        # Calculate features
        features = await feature_engineer.calculate_features(df, timeframe)
        
        return features.to_dict('records') if isinstance(features, pd.DataFrame) else features
    
    except Exception as e:
        logger.error(f"Error calculating features: {e}")
        return {}

@router.get("/historical/range", summary="Get data within date range")
async def get_data_range(
    symbol: str = Query("BTCUSDT", description="Trading pair symbol"),
    timeframe: str = Query("1h", description="Timeframe for data"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    include_features: bool = Query(False, description="Include engineered features"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get historical data within a specific date range.
    """
    try:
        await initialize_data_services()
        
        # Validate dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            if start_dt >= end_dt:
                raise HTTPException(status_code=400, detail="Start date must be before end date")
            
            if (end_dt - start_dt).days > 365:
                raise HTTPException(status_code=400, detail="Date range cannot exceed 1 year")
        
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Create data request
        data_request = DataRequest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            include_features=include_features
        )
        
        return await get_historical_data(data_request, current_user, credentials)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching data range: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical/latest", summary="Get latest data")
async def get_latest_data(
    symbol: str = Query("BTCUSDT", description="Trading pair symbol"),
    timeframe: str = Query("1h", description="Timeframe for data"),
    limit: int = Query(100, ge=1, le=5000, description="Number of latest candles"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get the latest historical data.
    """
    try:
        await initialize_data_services()
        
        # Create data request for latest data
        data_request = DataRequest(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        
        return await get_historical_data(data_request, current_user, credentials)
    
    except Exception as e:
        logger.error(f"Error fetching latest data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Feature Engineering Endpoints
@router.post("/features/calculate", summary="Calculate features")
async def calculate_features_endpoint(
    request: FeatureRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Calculate technical indicators and features for data.
    """
    try:
        await initialize_data_services()
        
        # Get historical data
        data_request = DataRequest(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            limit=1000  # Default limit for feature calculation
        )
        
        historical_data = await get_historical_data(data_request, current_user, credentials)
        ohlcv_data = historical_data.get('data', [])
        
        if not ohlcv_data:
            raise HTTPException(status_code=404, detail="No data available for feature calculation")
        
        # Calculate features
        features = await feature_engineer.calculate_specific_features(
            ohlcv_data=ohlcv_data,
            feature_list=request.features,
            timeframe=request.timeframe,
            lookback_period=request.lookback_period
        )
        
        # Get feature descriptions
        feature_descriptions = await feature_engineer.get_feature_descriptions(request.features)
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "features": features,
            "descriptions": feature_descriptions,
            "total_features": len(features),
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/features/available", summary="Get available features")
async def get_available_features(
    feature_type: Optional[str] = Query(None, description="Filter by feature type"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get list of available features/indicators.
    """
    try:
        await initialize_data_services()
        
        features = await feature_engineer.get_available_features()
        
        # Filter by type if specified
        if feature_type:
            features = {k: v for k, v in features.items() if v.get('type') == feature_type}
        
        return {
            "features": features,
            "total_features": len(features),
            "feature_types": list(set(f['type'] for f in features.values() if 'type' in f)),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching available features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/features/{feature_name}", summary="Get feature details")
async def get_feature_details(
    feature_name: str,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get detailed information about a specific feature.
    """
    try:
        await initialize_data_services()
        
        feature_info = await feature_engineer.get_feature_info(feature_name)
        
        if not feature_info:
            raise HTTPException(status_code=404, detail=f"Feature '{feature_name}' not found")
        
        return {
            "feature": feature_name,
            "info": feature_info,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching feature details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Processing Endpoints
@router.post("/preprocess", summary="Preprocess data")
async def preprocess_data(
    data: List[Dict[str, Any]],
    operations: List[str] = Query(["clean", "normalize"], description="Preprocessing operations"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Apply preprocessing operations to data.
    """
    try:
        await initialize_data_services()
        
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Validate data first
        validation_result = await data_validator.validate_ohlcv(data)
        if not validation_result.get('valid', False) and 'strict' in operations:
            raise HTTPException(
                status_code=400,
                detail=f"Data validation failed: {validation_result.get('issues', [])}"
            )
        
        # Apply preprocessing operations
        processed_data = data.copy()
        for operation in operations:
            if operation == "clean":
                processed_data = await data_preprocessor.clean_data(processed_data)
            elif operation == "normalize":
                processed_data = await data_preprocessor.normalize_data(processed_data)
            elif operation == "standardize":
                processed_data = await data_preprocessor.standardize_data(processed_data)
            elif operation == "remove_outliers":
                processed_data = await data_preprocessor.remove_outliers(processed_data)
            elif operation == "fill_missing":
                processed_data = await data_preprocessor.fill_missing_values(processed_data)
            elif operation == "smooth":
                processed_data = await data_preprocessor.smooth_data(processed_data)
        
        return {
            "original_records": len(data),
            "processed_records": len(processed_data),
            "operations_applied": operations,
            "processed_data": processed_data,
            "validation_result": validation_result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate", summary="Validate data")
async def validate_data_endpoint(
    request: DataValidationRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Validate data quality and integrity.
    """
    try:
        await initialize_data_services()
        
        if not request.data:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Apply validation
        validation_result = await data_validator.validate_data(
            data=request.data,
            rules=request.validation_rules,
            strict_mode=request.strict_mode
        )
        
        # Generate data quality report
        quality_report = await generate_quality_report(request.data, validation_result)
        
        return {
            "validation_result": validation_result,
            "quality_report": quality_report,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_quality_report(data: List[Dict], validation_result: Dict) -> Dict:
    """Generate data quality report"""
    if not data:
        return {}
    
    # Basic statistics
    closes = [d.get('close', 0) for d in data]
    volumes = [d.get('volume', 0) for d in data]
    
    return {
        "basic_statistics": {
            "total_records": len(data),
            "date_range": {
                "start": data[0].get('timestamp') if data else None,
                "end": data[-1].get('timestamp') if data else None
            },
            "price_range": {
                "min": min(closes) if closes else 0,
                "max": max(closes) if closes else 0,
                "avg": np.mean(closes) if closes else 0
            },
            "volume_statistics": {
                "total": sum(volumes) if volumes else 0,
                "avg": np.mean(volumes) if volumes else 0
            }
        },
        "quality_metrics": {
            "completeness": calculate_completeness(data),
            "consistency": calculate_consistency(data),
            "accuracy": validation_result.get('accuracy', 0),
            "timeliness": calculate_timeliness(data)
        },
        "issues_summary": {
            "total_issues": len(validation_result.get('issues', [])),
            "critical_issues": sum(1 for i in validation_result.get('issues', []) if i.get('severity') == 'critical'),
            "warning_issues": sum(1 for i in validation_result.get('issues', []) if i.get('severity') == 'warning')
        }
    }

def calculate_completeness(data: List[Dict]) -> float:
    """Calculate data completeness percentage"""
    if not data:
        return 0.0
    
    required_fields = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    total_fields = len(data) * len(required_fields)
    missing_fields = 0
    
    for record in data:
        for field in required_fields:
            if field not in record or record[field] is None:
                missing_fields += 1
    
    return ((total_fields - missing_fields) / total_fields * 100) if total_fields > 0 else 0.0

def calculate_consistency(data: List[Dict]) -> float:
    """Calculate data consistency score"""
    if len(data) < 2:
        return 100.0
    
    inconsistencies = 0
    for i in range(1, len(data)):
        prev = data[i-1]
        curr = data[i]
        
        # Check timestamp sequence
        if 'timestamp' in prev and 'timestamp' in curr:
            try:
                prev_ts = datetime.fromisoformat(prev['timestamp'].replace('Z', '+00:00'))
                curr_ts = datetime.fromisoformat(curr['timestamp'].replace('Z', '+00:00'))
                if curr_ts <= prev_ts:
                    inconsistencies += 1
            except:
                inconsistencies += 1
        
        # Check price consistency
        if not (curr['low'] <= curr['close'] <= curr['high']):
            inconsistencies += 1
    
    total_checks = (len(data) - 1) * 2  # timestamp + price consistency checks
    return ((total_checks - inconsistencies) / total_checks * 100) if total_checks > 0 else 0.0

def calculate_timeliness(data: List[Dict]) -> float:
    """Calculate data timeliness"""
    if not data:
        return 0.0
    
    try:
        latest_timestamp = data[-1].get('timestamp')
        if not latest_timestamp:
            return 0.0
        
        latest_dt = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
        time_diff = (datetime.now() - latest_dt).total_seconds()
        
        # Score based on how recent the data is (within 1 hour = 100%)
        timeliness_score = max(0, 100 - (time_diff / 3600))
        return min(100.0, timeliness_score)
    
    except:
        return 0.0

# Data Management Endpoints
@router.post("/update", summary="Update data")
async def update_data(
    request: DataUpdateRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Update data for a symbol/timeframe.
    """
    try:
        await initialize_data_services()
        
        # Check if data needs update
        cache_key = f"data_status_{request.symbol}_{request.timeframe}"
        last_update = await cache_manager.get(cache_key)
        
        if not request.force_refresh and last_update:
            last_update_time = datetime.fromisoformat(last_update.get('timestamp', ''))
            if (datetime.now() - last_update_time).total_seconds() < 300:  # 5 minutes
                return {
                    "message": "Data recently updated, skipping refresh",
                    "last_update": last_update_time.isoformat(),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Update data
        update_result = await data_collector.update_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            force_refresh=request.force_refresh
        )
        
        # Store update status
        update_status = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "records_updated": update_result.get('records_updated', 0),
            "timestamp": datetime.now().isoformat()
        }
        await cache_manager.set(cache_key, update_status, ttl=3600)
        
        # Set up auto-update if requested
        if request.update_interval:
            await setup_auto_update(request.symbol, request.timeframe, request.update_interval)
        
        return {
            "success": True,
            "message": "Data updated successfully",
            "update_result": update_result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def setup_auto_update(symbol: str, timeframe: str, interval: str):
    """Set up automatic data updates"""
    # This would set up a scheduled task
    # For now, just log the intention
    logger.info(f"Setting up auto-update for {symbol}/{timeframe} every {interval}")

@router.get("/status", summary="Get data status")
async def get_data_status(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get status of data collection and updates.
    """
    try:
        await initialize_data_services()
        
        status_info = await data_collector.get_data_status(symbol, timeframe)
        
        # Add cache statistics
        cache_stats = await cache_manager.get_stats()
        
        # Add database statistics
        db_stats = await crud.get_data_statistics(symbol, timeframe)
        
        return {
            "data_status": status_info,
            "cache_statistics": cache_stats,
            "database_statistics": db_stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching data status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sources", summary="Get data sources")
async def get_data_sources(
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get available data sources and their status.
    """
    try:
        await initialize_data_services()
        
        sources = await data_collector.get_available_sources()
        
        # Check source connectivity
        source_status = []
        for source in sources:
            status = await data_collector.check_source_connectivity(source)
            source_status.append({
                "source": source,
                "status": status,
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "sources": source_status,
            "total_sources": len(sources),
            "available_sources": [s['source'] for s in source_status if s['status'].get('connected', False)],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching data sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Export Endpoints
@router.post("/export", summary="Export data")
async def export_data(
    request: DataExportRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Export data in various formats.
    """
    try:
        await initialize_data_services()
        
        # Get data
        data_request = DataRequest(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            include_features=request.include_features
        )
        
        data_response = await get_historical_data(data_request, current_user, credentials)
        data = data_response.get('data', [])
        
        if not data:
            raise HTTPException(status_code=404, detail="No data to export")
        
        # Export based on format
        if request.format.lower() == 'csv':
            return await export_to_csv(data, request)
        elif request.format.lower() == 'json':
            return await export_to_json(data, request)
        elif request.format.lower() == 'parquet':
            return await export_to_parquet(data, request)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def export_to_csv(data: List[Dict], request: DataExportRequest):
    """Export data to CSV format"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # Prepare filename
        filename = f"{request.symbol}_{request.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/csv; charset=utf-8"
            }
        )
    
    except Exception as e:
        logger.error(f"Error creating CSV export: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def export_to_json(data: List[Dict], request: DataExportRequest):
    """Export data to JSON format"""
    try:
        # Prepare response
        export_data = {
            "metadata": {
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "export_date": datetime.now().isoformat(),
                "total_records": len(data)
            },
            "data": data
        }
        
        # Convert to JSON string
        json_str = json.dumps(export_data, indent=2, default=str)
        
        # Prepare filename
        filename = f"{request.symbol}_{request.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        return StreamingResponse(
            iter([json_str]),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/json; charset=utf-8"
            }
        )
    
    except Exception as e:
        logger.error(f"Error creating JSON export: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def export_to_parquet(data: List[Dict], request: DataExportRequest):
    """Export data to Parquet format"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create Parquet in memory
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        # Prepare filename
        filename = f"{request.symbol}_{request.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        
        return StreamingResponse(
            iter([parquet_buffer.getvalue()]),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/octet-stream"
            }
        )
    
    except Exception as e:
        logger.error(f"Error creating Parquet export: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/template", summary="Get export template")
async def get_export_template(
    format: str = Query("csv", description="Template format"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get export template for data.
    """
    try:
        # Create template based on format
        if format.lower() == 'csv':
            template = "timestamp,open,high,low,close,volume\n"
            template += "2024-01-01T00:00:00Z,45000.0,45200.0,44800.0,45100.0,100.5\n"
            
            return StreamingResponse(
                iter([template]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=data_template.csv",
                    "Content-Type": "text/csv; charset=utf-8"
                }
            )
        
        elif format.lower() == 'json':
            template = {
                "metadata": {
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                    "description": "Data export template"
                },
                "data": [
                    {
                        "timestamp": "2024-01-01T00:00:00Z",
                        "open": 45000.0,
                        "high": 45200.0,
                        "low": 44800.0,
                        "close": 45100.0,
                        "volume": 100.5
                    }
                ]
            }
            
            return JSONResponse(
                content=template,
                headers={
                    "Content-Disposition": "attachment; filename=data_template.json"
                }
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported template format")
    
    except Exception as e:
        logger.error(f"Error creating export template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Analysis Endpoints
@router.get("/analysis/summary", summary="Get data analysis summary")
async def get_data_analysis_summary(
    symbol: str = Query("BTCUSDT", description="Trading pair symbol"),
    timeframe: str = Query("1h", description="Timeframe for analysis"),
    period: str = Query("30d", description="Analysis period"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get comprehensive analysis summary for data.
    """
    try:
        await initialize_data_services()
        
        # Get data for analysis
        data_request = DataRequest(
            symbol=symbol,
            timeframe=timeframe,
            limit=1000  # Adjust based on period
        )
        
        data_response = await get_historical_data(data_request, current_user, credentials)
        data = data_response.get('data', [])
        
        if not data:
            raise HTTPException(status_code=404, detail="No data available for analysis")
        
        # Perform analysis
        analysis_results = await perform_data_analysis(data, timeframe)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "period": period,
            "analysis": analysis_results,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing data analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def perform_data_analysis(data: List[Dict], timeframe: str) -> Dict:
    """Perform comprehensive data analysis"""
    if not data:
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Basic statistics
    basic_stats = {
        "price": {
            "mean": float(df['close'].mean()),
            "median": float(df['close'].median()),
            "std": float(df['close'].std()),
            "min": float(df['close'].min()),
            "max": float(df['close'].max()),
            "range": float(df['close'].max() - df['close'].min())
        },
        "volume": {
            "mean": float(df['volume'].mean()),
            "median": float(df['volume'].median()),
            "std": float(df['volume'].std()),
            "total": float(df['volume'].sum())
        },
        "returns": {
            "mean": float(df['close'].pct_change().mean()),
            "std": float(df['close'].pct_change().std()),
            "skew": float(df['close'].pct_change().skew()),
            "kurtosis": float(df['close'].pct_change().kurtosis())
        }
    }
    
    # Volatility analysis
    returns = df['close'].pct_change().dropna()
    volatility = {
        "daily": float(returns.std() * np.sqrt(252)) if timeframe == '1d' else 0,
        "annualized": float(returns.std() * np.sqrt(252)),
        "rolling_20": float(df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
    }
    
    # Trend analysis
    trend_analysis = {
        "overall_trend": "bullish" if df['close'].iloc[-1] > df['close'].iloc[0] else "bearish",
        "trend_strength": float(abs(df['close'].pct_change().mean() / df['close'].pct_change().std()) if df['close'].pct_change().std() != 0 else 0),
        "moving_averages": {
            "sma_20": float(df['close'].rolling(20).mean().iloc[-1]),
            "sma_50": float(df['close'].rolling(50).mean().iloc[-1]),
            "sma_200": float(df['close'].rolling(200).mean().iloc[-1])
        }
    }
    
    # Market regime detection
    regime_analysis = {
        "current_regime": detect_market_regime(df),
        "regime_changes": count_regime_changes(df),
        "regime_durations": calculate_regime_durations(df)
    }
    
    # Seasonality analysis (if enough data)
    seasonality = {}
    if len(df) > 100:
        try:
            # Simple seasonality detection
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hour_avg_returns = df.groupby('hour')['close'].pct_change().mean()
            seasonality['hourly_pattern'] = hour_avg_returns.to_dict()
        except:
            seasonality['hourly_pattern'] = "Not enough data for seasonality analysis"
    
    return {
        "basic_statistics": basic_stats,
        "volatility_analysis": volatility,
        "trend_analysis": trend_analysis,
        "regime_analysis": regime_analysis,
        "seasonality_analysis": seasonality,
        "data_quality": {
            "completeness": calculate_completeness(data),
            "consistency": calculate_consistency(data),
            "timeliness": calculate_timeliness(data)
        }
    }

def detect_market_regime(df: pd.DataFrame) -> str:
    """Detect current market regime"""
    if len(df) < 20:
        return "unknown"
    
    returns = df['close'].pct_change().dropna()
    volatility = returns.std()
    
    if volatility > returns.std() * 1.5:
        return "high_volatility"
    elif volatility < returns.std() * 0.5:
        return "low_volatility"
    elif returns.mean() > 0:
        return "bullish"
    else:
        return "bearish"

def count_regime_changes(df: pd.DataFrame) -> int:
    """Count regime changes in data"""
    # Simplified implementation
    if len(df) < 50:
        return 0
    
    returns = df['close'].pct_change().dropna()
    changes = 0
    
    for i in range(1, len(returns)):
        if returns.iloc[i] * returns.iloc[i-1] < 0:  # Sign change
            changes += 1
    
    return changes

def calculate_regime_durations(df: pd.DataFrame) -> Dict:
    """Calculate durations of different regimes"""
    # Simplified implementation
    return {
        "average_bullish_duration": 10,
        "average_bearish_duration": 8,
        "longest_bullish_run": 25,
        "longest_bearish_run": 20
    }

# Health Check Endpoint
@router.get("/health", summary="Data system health check")
async def data_health_check(
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Check health of data system components.
    """
    try:
        await initialize_data_services()
        
        health_checks = {
            "data_collector": await data_collector.health_check(),
            "feature_engineer": await feature_engineer.health_check(),
            "data_preprocessor": await data_preprocessor.health_check(),
            "data_validator": await data_validator.health_check(),
            "cache_manager": await cache_manager.health_check(),
            "database": await crud.health_check()
        }
        
        # Check exchange connectivity
        exchange_health = await data_collector.check_exchange_connectivity()
        health_checks["exchange_connectivity"] = exchange_health
        
        # Check data freshness
        freshness_check = await check_data_freshness()
        health_checks["data_freshness"] = freshness_check
        
        all_healthy = all(
            check.get("healthy", False) if isinstance(check, dict) else check
            for check in health_checks.values()
        )
        
        return {
            "healthy": all_healthy,
            "checks": health_checks,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in data health check: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def check_data_freshness() -> Dict:
    """Check freshness of cached data"""
    try:
        # Check various cache entries
        cache_keys = [
            "price_BTCUSDT",
            "ticker_BTCUSDT",
            "historical_BTCUSDT_1h"
        ]
        
        freshness_results = {}
        for key in cache_keys:
            cached_data = await cache_manager.get(key)
            if cached_data:
                timestamp = cached_data.get('timestamp', '')
                if timestamp:
                    age = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds()
                    freshness_results[key] = {
                        "age_seconds": age,
                        "fresh": age < 300,  # Less than 5 minutes
                        "timestamp": timestamp
                    }
        
        return {
            "healthy": all(r["fresh"] for r in freshness_results.values()) if freshness_results else False,
            "freshness": freshness_results,
            "total_checked": len(freshness_results)
        }
    
    except Exception as e:
        logger.error(f"Error checking data freshness: {e}")
        return {"healthy": False, "error": str(e)}