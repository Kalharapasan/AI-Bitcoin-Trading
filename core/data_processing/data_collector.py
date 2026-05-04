"""
Data Collector for Bitcoin Trading AI
Handles data collection from multiple exchanges and data providers
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta, timezone
import time
import json
import pickle
import gzip
from pathlib import Path
import requests
from websockets import connect
import ccxt
import ccxt.pro as ccxtpro
from dataclasses import dataclass, field
from enum import Enum
import warnings
import hashlib
import threading
import queue
from collections import defaultdict, deque
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

warnings.filterwarnings('ignore')

# Import project modules
from config.settings import (
    Paths, ExchangeSettings, DataSettings, AppConstants,
    BASE_DIR, DATA_DIR, CONFIG_DIR
)
from config.config_manager import get_config

logger = logging.getLogger(__name__)

# ============ Data Types ============
class DataType(str, Enum):
    """Types of data that can be collected"""
    OHLCV = "ohlcv"
    TRADES = "trades"
    ORDER_BOOK = "order_book"
    TICKER = "ticker"
    FUNDING_RATE = "funding_rate"
    LIQUIDATION = "liquidation"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    NEWS = "news"
    SOCIAL = "social"

class Exchange(str, Enum):
    """Supported exchanges"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BITSTAMP = "bitstamp"
    BYBIT = "bybit"
    OKX = "okx"
    HUOBI = "huobi"
    GATEIO = "gateio"
    MEXC = "mexc"
    BITFINEX = "bitfinex"

class Timeframe(str, Enum):
    """Timeframes for OHLCV data"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

# ============ Data Structures ============
@dataclass
class OHLCVData:
    """OHLCV data structure"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str
    exchange: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'exchange': self.exchange
        }
    
    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict())

@dataclass
class TradeData:
    """Trade data structure"""
    timestamp: pd.Timestamp
    price: float
    amount: float
    side: str  # 'buy' or 'sell'
    symbol: str
    exchange: str
    trade_id: Optional[str] = None
    maker: Optional[bool] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'price': self.price,
            'amount': self.amount,
            'side': self.side,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'trade_id': self.trade_id,
            'maker': self.maker
        }

@dataclass
class OrderBookData:
    """Order book data structure"""
    timestamp: pd.Timestamp
    bids: List[Tuple[float, float]]  # [(price, amount), ...]
    asks: List[Tuple[float, float]]  # [(price, amount), ...]
    symbol: str
    exchange: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'bids': self.bids,
            'asks': self.asks,
            'symbol': self.symbol,
            'exchange': self.exchange
        }

@dataclass
class TickerData:
    """Ticker data structure"""
    timestamp: pd.Timestamp
    symbol: str
    exchange: str
    last: float
    bid: float
    ask: float
    high: float
    low: float
    volume: float
    quote_volume: float
    change: float
    change_percent: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'last': self.last,
            'bid': self.bid,
            'ask': self.ask,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'quote_volume': self.quote_volume,
            'change': self.change,
            'change_percent': self.change_percent
        }

# ============ Exchange Connectors ============
class ExchangeConnector:
    """Base class for exchange connectors"""
    
    def __init__(self, exchange_name: str, api_key: str = '', api_secret: str = ''):
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = None
        self.pro_exchange = None
        self.rate_limit_multiplier = 0.8
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            # Initialize ccxt exchange
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'rateLimit': int(ExchangeSettings.RATE_LIMITS.get(self.exchange_name, 1000) * 
                               self.rate_limit_multiplier),
                'timeout': ExchangeSettings.TIMEOUTS.get(self.exchange_name, 30000),
                'verbose': False
            })
            
            # Test connection
            self.exchange.load_markets()
            logger.info(f"Connected to {self.exchange_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_name}: {str(e)}")
            raise
    
    async def _initialize_pro_exchange(self):
        """Initialize ccxt.pro exchange for real-time data"""
        try:
            exchange_class = getattr(ccxtpro, self.exchange_name)
            self.pro_exchange = await exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'rateLimit': int(ExchangeSettings.RATE_LIMITS.get(self.exchange_name, 1000) * 
                               self.rate_limit_multiplier),
                'timeout': ExchangeSettings.TIMEOUTS.get(self.exchange_name, 30000),
                'newUpdates': True
            })
            await self.pro_exchange.load_markets()
            logger.info(f"Connected to {self.exchange_name} (pro)")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_name} pro: {str(e)}")
            raise
    
    @retry(
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                         since: Optional[int] = None, limit: int = 1000) -> List[OHLCVData]:
        """Fetch OHLCV data"""
        try:
            # Convert symbol to exchange format
            exchange_symbol = self._format_symbol(symbol)
            
            # Fetch data
            ohlcv_data = await self.pro_exchange.fetch_ohlcv(
                exchange_symbol, timeframe, since, limit
            )
            
            # Convert to OHLCVData objects
            result = []
            for data in ohlcv_data:
                result.append(OHLCVData(
                    timestamp=pd.Timestamp(data[0], unit='ms'),
                    open=float(data[1]),
                    high=float(data[2]),
                    low=float(data[3]),
                    close=float(data[4]),
                    volume=float(data[5]),
                    symbol=symbol,
                    timeframe=timeframe,
                    exchange=self.exchange_name
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} from {self.exchange_name}: {str(e)}")
            raise
    
    @retry(
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def fetch_trades(self, symbol: str, since: Optional[int] = None, 
                          limit: int = 1000) -> List[TradeData]:
        """Fetch recent trades"""
        try:
            exchange_symbol = self._format_symbol(symbol)
            
            trades = await self.pro_exchange.fetch_trades(
                exchange_symbol, since, limit
            )
            
            result = []
            for trade in trades:
                result.append(TradeData(
                    timestamp=pd.Timestamp(trade['timestamp'], unit='ms'),
                    price=float(trade['price']),
                    amount=float(trade['amount']),
                    side=trade['side'],
                    symbol=symbol,
                    exchange=self.exchange_name,
                    trade_id=trade.get('id'),
                    maker=trade.get('takerOrMaker') == 'maker'
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol} from {self.exchange_name}: {str(e)}")
            raise
    
    @retry(
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> OrderBookData:
        """Fetch order book"""
        try:
            exchange_symbol = self._format_symbol(symbol)
            
            orderbook = await self.pro_exchange.fetch_order_book(
                exchange_symbol, limit
            )
            
            return OrderBookData(
                timestamp=pd.Timestamp(orderbook['timestamp'], unit='ms'),
                bids=[(float(bid[0]), float(bid[1])) for bid in orderbook['bids']],
                asks=[(float(ask[0]), float(ask[1])) for ask in orderbook['asks']],
                symbol=symbol,
                exchange=self.exchange_name
            )
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol} from {self.exchange_name}: {str(e)}")
            raise
    
    @retry(
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def fetch_ticker(self, symbol: str) -> TickerData:
        """Fetch ticker data"""
        try:
            exchange_symbol = self._format_symbol(symbol)
            
            ticker = await self.pro_exchange.fetch_ticker(exchange_symbol)
            
            return TickerData(
                timestamp=pd.Timestamp(ticker['timestamp'], unit='ms'),
                symbol=symbol,
                exchange=self.exchange_name,
                last=float(ticker['last']),
                bid=float(ticker['bid']),
                ask=float(ticker['ask']),
                high=float(ticker['high']),
                low=float(ticker['low']),
                volume=float(ticker['baseVolume']),
                quote_volume=float(ticker['quoteVolume']),
                change=float(ticker['change']),
                change_percent=float(ticker['percentage'])
            )
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol} from {self.exchange_name}: {str(e)}")
            raise
    
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for exchange"""
        if self.exchange_name in ExchangeSettings.SYMBOL_MAPPING:
            return ExchangeSettings.SYMBOL_MAPPING[self.exchange_name].get(
                symbol, symbol.replace('/', '')
            )
        return symbol.replace('/', '')
    
    async def close(self):
        """Close exchange connection"""
        if self.pro_exchange:
            await self.pro_exchange.close()
            logger.info(f"Closed connection to {self.exchange_name}")

class BinanceConnector(ExchangeConnector):
    """Binance-specific connector"""
    
    def __init__(self, api_key: str = '', api_secret: str = '', testnet: bool = False):
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        }
        
        if testnet:
            config['urls'] = {
                'api': {
                    'public': 'https://testnet.binance.vision/api',
                    'private': 'https://testnet.binance.vision/api'
                }
            }
        
        super().__init__('binance', api_key, api_secret)
    
    async def fetch_funding_rate(self, symbol: str = 'BTC/USDT') -> Dict:
        """Fetch funding rate (for futures)"""
        try:
            # Switch to futures market
            self.exchange.options['defaultType'] = 'future'
            
            funding_rate = await self.pro_exchange.fetch_funding_rate(
                self._format_symbol(symbol)
            )
            
            return funding_rate
            
        except Exception as e:
            logger.error(f"Error fetching funding rate from Binance: {str(e)}")
            raise
        finally:
            # Switch back to spot
            self.exchange.options['defaultType'] = 'spot'

class CoinbaseConnector(ExchangeConnector):
    """Coinbase-specific connector"""
    
    def __init__(self, api_key: str = '', api_secret: str = ''):
        super().__init__('coinbase', api_key, api_secret)
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                         since: Optional[int] = None, limit: int = 300) -> List[OHLCVData]:
        """Coinbase has different limits"""
        return await super().fetch_ohlcv(symbol, timeframe, since, limit)

# ============ Data Providers ============
class DataProvider:
    """Base class for data providers"""
    
    def __init__(self, name: str, api_key: str = ''):
        self.name = name
        self.api_key = api_key
        self.base_url = ''
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _make_request(self, url: str, method: str = 'GET', 
                           params: Dict = None, headers: Dict = None) -> Dict:
        """Make HTTP request"""
        try:
            if method == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            elif method == 'POST':
                async with self.session.post(url, json=params, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                raise ValueError(f"Unsupported method: {method}")
                
        except Exception as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            raise

class CryptoCompareProvider(DataProvider):
    """CryptoCompare data provider"""
    
    def __init__(self, api_key: str = ''):
        super().__init__('cryptocompare', api_key)
        self.base_url = 'https://min-api.cryptocompare.com/data'
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = 'hour', 
                         limit: int = 2000, exchange: str = 'CCCAGG') -> List[OHLCVData]:
        """Fetch OHLCV data from CryptoCompare"""
        try:
            # Map timeframe
            tf_map = {
                '1m': 'minute', '5m': '5minute', '15m': '15minute',
                '30m': '30minute', '1h': 'hour', '4h': '4hour',
                '12h': '12hour', '1d': 'day', '1w': 'week'
            }
            
            fsym, tsym = symbol.split('/')
            url = f"{self.base_url}/v2/histoday"
            
            params = {
                'fsym': fsym,
                'tsym': tsym,
                'limit': limit,
                'e': exchange,
                'api_key': self.api_key
            }
            
            if timeframe in tf_map:
                if tf_map[timeframe] in ['minute', '5minute', '15minute', '30minute']:
                    url = f"{self.base_url}/v2/histominute"
                elif tf_map[timeframe] in ['hour', '4hour', '12hour']:
                    url = f"{self.base_url}/v2/histohour"
            
            response = await self._make_request(url, params=params)
            
            if response.get('Response') == 'Error':
                raise ValueError(response.get('Message', 'Unknown error'))
            
            data = response.get('Data', {}).get('Data', [])
            
            result = []
            for item in data:
                result.append(OHLCVData(
                    timestamp=pd.Timestamp(item['time'], unit='s'),
                    open=float(item['open']),
                    high=float(item['high']),
                    low=float(item['low']),
                    close=float(item['close']),
                    volume=float(item['volumefrom']),
                    symbol=symbol,
                    timeframe=timeframe,
                    exchange=exchange
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV from CryptoCompare: {str(e)}")
            raise
    
    async def fetch_social_sentiment(self, symbol: str = 'BTC') -> Dict:
        """Fetch social sentiment data"""
        try:
            url = f"{self.base_url}/social/coin/latest"
            params = {
                'coinId': symbol,
                'api_key': self.api_key
            }
            
            response = await self._make_request(url, params=params)
            return response
            
        except Exception as e:
            logger.error(f"Error fetching social sentiment: {str(e)}")
            raise

class GlassnodeProvider(DataProvider):
    """Glassnode on-chain data provider"""
    
    def __init__(self, api_key: str = ''):
        super().__init__('glassnode', api_key)
        self.base_url = 'https://api.glassnode.com/v1'
    
    async def fetch_onchain_metric(self, metric: str, symbol: str = 'BTC', 
                                  timeframe: str = '24h', since: int = None, 
                                  until: int = None) -> pd.DataFrame:
        """Fetch on-chain metric"""
        try:
            url = f"{self.base_url}/metrics/{metric}"
            
            params = {
                'a': symbol,
                'i': timeframe,
                'api_key': self.api_key
            }
            
            if since:
                params['s'] = since
            if until:
                params['u'] = until
            
            response = await self._make_request(url, params=params)
            
            if isinstance(response, list):
                df = pd.DataFrame(response)
                if 't' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['t'], unit='s')
                    df = df.set_index('timestamp')
                return df
            else:
                raise ValueError(f"Unexpected response format: {response}")
            
        except Exception as e:
            logger.error(f"Error fetching {metric} from Glassnode: {str(e)}")
            raise

# ============ WebSocket Streamers ============
class WebSocketStreamer:
    """WebSocket streamer for real-time data"""
    
    def __init__(self, exchange: ExchangeConnector, symbols: List[str], 
                 data_types: List[DataType]):
        self.exchange = exchange
        self.symbols = symbols
        self.data_types = data_types
        self.ws_connections = {}
        self.callbacks = defaultdict(list)
        self.running = False
        self.message_queue = queue.Queue()
    
    async def start(self):
        """Start WebSocket streaming"""
        self.running = True
        
        # Create tasks for each symbol and data type
        tasks = []
        for symbol in self.symbols:
            for data_type in self.data_types:
                task = asyncio.create_task(
                    self._stream_data(symbol, data_type)
                )
                tasks.append(task)
        
        # Start message processor
        processor_task = asyncio.create_task(self._process_messages())
        
        await asyncio.gather(*tasks)
        await processor_task
    
    async def stop(self):
        """Stop WebSocket streaming"""
        self.running = False
        for ws in self.ws_connections.values():
            await ws.close()
    
    async def _stream_data(self, symbol: str, data_type: DataType):
        """Stream data for specific symbol and type"""
        try:
            exchange_symbol = self.exchange._format_symbol(symbol)
            
            if data_type == DataType.OHLCV:
                stream_name = f"{exchange_symbol.lower()}@kline_1m"
            elif data_type == DataType.TRADES:
                stream_name = f"{exchange_symbol.lower()}@trade"
            elif data_type == DataType.ORDER_BOOK:
                stream_name = f"{exchange_symbol.lower()}@depth20"
            elif data_type == DataType.TICKER:
                stream_name = f"{exchange_symbol.lower()}@ticker"
            else:
                logger.warning(f"Unsupported data type for streaming: {data_type}")
                return
            
            # Get WebSocket URL
            ws_url = self.exchange.pro_exchange.urls['api']['ws']
            
            async with connect(ws_url) as websocket:
                self.ws_connections[(symbol, data_type)] = websocket
                
                # Subscribe to stream
                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": 1
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                logger.info(f"Started streaming {data_type.value} for {symbol}")
                
                # Listen for messages
                while self.running:
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(), 
                            timeout=10
                        )
                        self.message_queue.put((symbol, data_type, message))
                        
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.ping()
                    except Exception as e:
                        logger.error(f"Error in WebSocket stream for {symbol}: {str(e)}")
                        break
        
        except Exception as e:
            logger.error(f"Failed to stream {data_type.value} for {symbol}: {str(e)}")
    
    async def _process_messages(self):
        """Process incoming WebSocket messages"""
        while self.running:
            try:
                symbol, data_type, message = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.message_queue.get, True
                    ),
                    timeout=1
                )
                
                # Parse message
                data = json.loads(message)
                
                # Process based on data type
                if data_type == DataType.TRADES:
                    trade_data = self._parse_trade_message(data, symbol)
                    await self._notify_callbacks(data_type, symbol, trade_data)
                
                elif data_type == DataType.OHLCV:
                    ohlcv_data = self._parse_ohlcv_message(data, symbol)
                    await self._notify_callbacks(data_type, symbol, ohlcv_data)
                
                elif data_type == DataType.ORDER_BOOK:
                    orderbook_data = self._parse_orderbook_message(data, symbol)
                    await self._notify_callbacks(data_type, symbol, orderbook_data)
                
                elif data_type == DataType.TICKER:
                    ticker_data = self._parse_ticker_message(data, symbol)
                    await self._notify_callbacks(data_type, symbol, ticker_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
    
    def _parse_trade_message(self, data: Dict, symbol: str) -> TradeData:
        """Parse trade message"""
        if 'e' in data and data['e'] == 'trade':
            return TradeData(
                timestamp=pd.Timestamp(data['T'], unit='ms'),
                price=float(data['p']),
                amount=float(data['q']),
                side='buy' if data['m'] else 'sell',
                symbol=symbol,
                exchange=self.exchange.exchange_name,
                trade_id=str(data['t']),
                maker=data.get('M', False)
            )
        return None
    
    def _parse_ohlcv_message(self, data: Dict, symbol: str) -> OHLCVData:
        """Parse OHLCV message"""
        if 'e' in data and data['e'] == 'kline':
            kline = data['k']
            return OHLCVData(
                timestamp=pd.Timestamp(kline['t'], unit='ms'),
                open=float(kline['o']),
                high=float(kline['h']),
                low=float(kline['l']),
                close=float(kline['c']),
                volume=float(kline['v']),
                symbol=symbol,
                timeframe=kline['i'],
                exchange=self.exchange.exchange_name
            )
        return None
    
    def _parse_orderbook_message(self, data: Dict, symbol: str) -> OrderBookData:
        """Parse order book message"""
        if 'e' in data and data['e'] == 'depthUpdate':
            return OrderBookData(
                timestamp=pd.Timestamp(data['E'], unit='ms'),
                bids=[(float(bid[0]), float(bid[1])) for bid in data.get('b', [])],
                asks=[(float(ask[0]), float(ask[1])) for ask in data.get('a', [])],
                symbol=symbol,
                exchange=self.exchange.exchange_name
            )
        return None
    
    def _parse_ticker_message(self, data: Dict, symbol: str) -> TickerData:
        """Parse ticker message"""
        if 'e' in data and data['e'] == '24hrTicker':
            return TickerData(
                timestamp=pd.Timestamp(data['E'], unit='ms'),
                symbol=symbol,
                exchange=self.exchange.exchange_name,
                last=float(data['c']),
                bid=float(data['b']),
                ask=float(data['a']),
                high=float(data['h']),
                low=float(data['l']),
                volume=float(data['v']),
                quote_volume=float(data['q']),
                change=float(data['p']),
                change_percent=float(data['P'])
            )
        return None
    
    def register_callback(self, data_type: DataType, symbol: str, callback):
        """Register callback for data updates"""
        self.callbacks[(data_type, symbol)].append(callback)
    
    async def _notify_callbacks(self, data_type: DataType, symbol: str, data: Any):
        """Notify registered callbacks"""
        callbacks = self.callbacks.get((data_type, symbol), [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in callback for {symbol}: {str(e)}")

# ============ Data Storage ============
class DataStorage:
    """Handles data storage and retrieval"""
    
    def __init__(self, base_path: Path = DATA_DIR):
        self.base_path = base_path
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        directories = [
            self.base_path / 'raw',
            self.base_path / 'processed',
            self.base_path / 'cache',
            self.base_path / 'raw' / 'ohlcv',
            self.base_path / 'raw' / 'trades',
            self.base_path / 'raw' / 'orderbook',
            self.base_path / 'raw' / 'ticker',
            self.base_path / 'raw' / 'onchain',
            self.base_path / 'raw' / 'sentiment',
            self.base_path / 'processed' / 'features',
            self.base_path / 'processed' / 'labels'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_data(self, data: List, data_type: DataType, symbol: str, 
                 exchange: str, timeframe: str = None, compress: bool = True):
        """Save data to disk"""
        try:
            # Create filename
            if timeframe:
                filename = f"{exchange}_{symbol.replace('/', '_')}_{timeframe}_{data_type.value}.pkl"
            else:
                filename = f"{exchange}_{symbol.replace('/', '_')}_{data_type.value}.pkl"
            
            # Determine path
            if data_type == DataType.OHLCV:
                path = self.base_path / 'raw' / 'ohlcv' / filename
            elif data_type == DataType.TRADES:
                path = self.base_path / 'raw' / 'trades' / filename
            elif data_type == DataType.ORDER_BOOK:
                path = self.base_path / 'raw' / 'orderbook' / filename
            elif data_type == DataType.TICKER:
                path = self.base_path / 'raw' / 'ticker' / filename
            elif data_type == DataType.ONCHAIN:
                path = self.base_path / 'raw' / 'onchain' / filename
            elif data_type == DataType.SENTIMENT:
                path = self.base_path / 'raw' / 'sentiment' / filename
            else:
                path = self.base_path / 'raw' / filename
            
            # Convert to DataFrame if needed
            if isinstance(data, list) and len(data) > 0:
                if hasattr(data[0], 'to_dict'):
                    data = [item.to_dict() for item in data]
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Save data
            if compress:
                with gzip.open(path, 'wb') as f:
                    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                df.to_pickle(path)
            
            logger.info(f"Saved {len(df)} {data_type.value} records for {symbol} to {path}")
            
            # Also save as CSV for easy inspection
            csv_path = path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            
            return path
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def load_data(self, data_type: DataType, symbol: str, exchange: str, 
                 timeframe: str = None, start_date: str = None, 
                 end_date: str = None) -> pd.DataFrame:
        """Load data from disk"""
        try:
            # Create filename
            if timeframe:
                filename = f"{exchange}_{symbol.replace('/', '_')}_{timeframe}_{data_type.value}.pkl"
            else:
                filename = f"{exchange}_{symbol.replace('/', '_')}_{data_type.value}.pkl"
            
            # Determine path
            if data_type == DataType.OHLCV:
                path = self.base_path / 'raw' / 'ohlcv' / filename
            elif data_type == DataType.TRADES:
                path = self.base_path / 'raw' / 'trades' / filename
            elif data_type == DataType.ORDER_BOOK:
                path = self.base_path / 'raw' / 'orderbook' / filename
            elif data_type == DataType.TICKER:
                path = self.base_path / 'raw' / 'ticker' / filename
            elif data_type == DataType.ONCHAIN:
                path = self.base_path / 'raw' / 'onchain' / filename
            elif data_type == DataType.SENTIMENT:
                path = self.base_path / 'raw' / 'sentiment' / filename
            else:
                path = self.base_path / 'raw' / filename
            
            # Check if compressed file exists
            gz_path = path.with_suffix('.pkl.gz')
            if gz_path.exists():
                path = gz_path
            
            if not path.exists():
                logger.warning(f"Data file not found: {path}")
                return pd.DataFrame()
            
            # Load data
            if path.suffix == '.gz':
                with gzip.open(path, 'rb') as f:
                    df = pickle.load(f)
            else:
                df = pd.read_pickle(path)
            
            # Filter by date if requested
            if start_date or end_date:
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    mask = pd.Series(True, index=df.index)
                    
                    if start_date:
                        start_dt = pd.Timestamp(start_date)
                        mask = mask & (df['timestamp'] >= start_dt)
                    
                    if end_date:
                        end_dt = pd.Timestamp(end_date)
                        mask = mask & (df['timestamp'] <= end_dt)
                    
                    df = df[mask]
            
            logger.info(f"Loaded {len(df)} {data_type.value} records for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def get_available_data(self) -> Dict:
        """Get list of available data"""
        available = defaultdict(dict)
        
        for data_type in DataType:
            type_dir = self.base_path / 'raw' / data_type.value
            if type_dir.exists():
                for file_path in type_dir.glob('*.pkl*'):
                    filename = file_path.stem
                    if file_path.suffix == '.gz':
                        filename = file_path.with_suffix('').stem
                    
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        exchange = parts[0]
                        symbol = f"{parts[1]}/{parts[2]}" if len(parts) > 3 else parts[1]
                        timeframe = parts[3] if len(parts) > 3 else None
                        
                        if data_type.value not in available[exchange]:
                            available[exchange][data_type.value] = {}
                        
                        if symbol not in available[exchange][data_type.value]:
                            available[exchange][data_type.value][symbol] = []
                        
                        if timeframe:
                            available[exchange][data_type.value][symbol].append(timeframe)
        
        return dict(available)
    
    def clear_cache(self, older_than_days: int = 7):
        """Clear old cache files"""
        cache_dir = self.base_path / 'cache'
        if not cache_dir.exists():
            return
        
        cutoff_time = time.time() - (older_than_days * 24 * 3600)
        
        for file_path in cache_dir.glob('*'):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                logger.info(f"Removed cache file: {file_path}")

# ============ Main Data Collector ============
class DataCollector:
    """Main data collection orchestrator"""
    
    def __init__(self, config_path: str = None):
        self.config = get_config() if not config_path else self._load_config(config_path)
        self.exchange_connectors = {}
        self.data_providers = {}
        self.storage = DataStorage()
        self.streamers = {}
        self.running = False
        
        self._initialize_connectors()
        self._initialize_providers()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
    
    def _initialize_connectors(self):
        """Initialize exchange connectors"""
        api_config = self.config.api
        
        for exchange_name, credentials in api_config.exchanges.items():
            if credentials.get('api_key') and credentials.get('api_secret'):
                try:
                    if exchange_name == 'binance':
                        connector = BinanceConnector(
                            api_key=credentials['api_key'],
                            api_secret=credentials['api_secret'],
                            testnet=credentials.get('testnet', False)
                        )
                    elif exchange_name == 'coinbase':
                        connector = CoinbaseConnector(
                            api_key=credentials['api_key'],
                            api_secret=credentials['api_secret']
                        )
                    else:
                        connector = ExchangeConnector(
                            exchange_name=exchange_name,
                            api_key=credentials['api_key'],
                            api_secret=credentials['api_secret']
                        )
                    
                    self.exchange_connectors[exchange_name] = connector
                    logger.info(f"Initialized connector for {exchange_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize connector for {exchange_name}: {str(e)}")
    
    def _initialize_providers(self):
        """Initialize data providers"""
        api_config = self.config.api
        
        # CryptoCompare
        if 'cryptocompare' in api_config.data_providers:
            api_key = api_config.data_providers['cryptocompare'].get('api_key', '')
            self.data_providers['cryptocompare'] = CryptoCompareProvider(api_key)
        
        # Glassnode
        if 'glassnode' in api_config.data_providers:
            api_key = api_config.data_providers['glassnode'].get('api_key', '')
            self.data_providers['glassnode'] = GlassnodeProvider(api_key)
    
    async def collect_historical_data(self, symbol: str = 'BTC/USDT', 
                                    timeframe: str = '1h',
                                    start_date: str = '2023-01-01',
                                    end_date: str = None,
                                    exchanges: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Collect historical data from multiple sources"""
        if not exchanges:
            exchanges = [self.config.data.primary_exchange] + self.config.data.backup_exchanges
        
        all_data = {}
        
        for exchange_name in exchanges:
            if exchange_name not in self.exchange_connectors:
                logger.warning(f"Exchange {exchange_name} not available")
                continue
            
            try:
                connector = self.exchange_connectors[exchange_name]
                
                # Initialize pro exchange for async operations
                if not connector.pro_exchange:
                    await connector._initialize_pro_exchange()
                
                # Calculate time range
                end_timestamp = pd.Timestamp(end_date or datetime.now()).timestamp() * 1000
                start_timestamp = pd.Timestamp(start_date).timestamp() * 1000
                
                # Collect data in batches
                data = []
                current_since = start_timestamp
                batch_size = 1000  # Max candles per request
                
                while current_since < end_timestamp:
                    batch_data = await connector.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=int(current_since),
                        limit=batch_size
                    )
                    
                    if not batch_data:
                        break
                    
                    data.extend(batch_data)
                    
                    # Update since to last timestamp + 1 interval
                    last_timestamp = batch_data[-1].timestamp.timestamp() * 1000
                    
                    # Calculate next timestamp (add one timeframe)
                    timeframe_ms = self._timeframe_to_ms(timeframe)
                    current_since = last_timestamp + timeframe_ms
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                
                if data:
                    # Convert to DataFrame
                    df = pd.DataFrame([d.to_dict() for d in data])
                    df = df.sort_values('timestamp').drop_duplicates()
                    
                    # Save to storage
                    self.storage.save_data(
                        df, DataType.OHLCV, symbol, exchange_name, timeframe
                    )
                    
                    all_data[exchange_name] = df
                    logger.info(f"Collected {len(df)} OHLCV records from {exchange_name}")
                
            except Exception as e:
                logger.error(f"Error collecting data from {exchange_name}: {str(e)}")
                continue
        
        return all_data
    
    async def collect_onchain_data(self, metrics: List[str] = None, 
                                 symbol: str = 'BTC',
                                 start_date: str = '2023-01-01'):
        """Collect on-chain data"""
        if not metrics:
            metrics = self.config.data.onchain_metrics
        
        if 'glassnode' not in self.data_providers:
            logger.warning("Glassnode provider not available")
            return {}
        
        all_data = {}
        
        async with self.data_providers['glassnode'] as provider:
            for metric in metrics:
                try:
                    df = await provider.fetch_onchain_metric(
                        metric=metric,
                        symbol=symbol,
                        since=pd.Timestamp(start_date).timestamp()
                    )
                    
                    if not df.empty:
                        self.storage.save_data(
                            df, DataType.ONCHAIN, symbol, 'glassnode'
                        )
                        
                        all_data[metric] = df
                        logger.info(f"Collected {metric} on-chain data")
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error collecting {metric}: {str(e)}")
                    continue
        
        return all_data
    
    async def collect_sentiment_data(self, symbol: str = 'BTC'):
        """Collect sentiment data"""
        if 'cryptocompare' not in self.data_providers:
            logger.warning("CryptoCompare provider not available")
            return {}
        
        try:
            async with self.data_providers['cryptocompare'] as provider:
                sentiment_data = await provider.fetch_social_sentiment(symbol)
                
                if sentiment_data:
                    df = pd.DataFrame([sentiment_data])
                    self.storage.save_data(
                        df, DataType.SENTIMENT, symbol, 'cryptocompare'
                    )
                    
                    logger.info(f"Collected sentiment data for {symbol}")
                    return df
                
        except Exception as e:
            logger.error(f"Error collecting sentiment data: {str(e)}")
        
        return pd.DataFrame()
    
    async def start_realtime_streaming(self, symbols: List[str] = None,
                                     data_types: List[DataType] = None):
        """Start real-time data streaming"""
        if not symbols:
            symbols = [self.config.trading.symbol]
        
        if not data_types:
            data_types = [
                DataType.TRADES,
                DataType.OHLCV,
                DataType.ORDER_BOOK,
                DataType.TICKER
            ]
        
        primary_exchange = self.config.data.primary_exchange
        if primary_exchange not in self.exchange_connectors:
            logger.error(f"Primary exchange {primary_exchange} not available")
            return
        
        connector = self.exchange_connectors[primary_exchange]
        
        # Initialize WebSocket streamer
        streamer = WebSocketStreamer(connector, symbols, data_types)
        self.streamers[primary_exchange] = streamer
        
        # Register callbacks for data storage
        for symbol in symbols:
            for data_type in data_types:
                streamer.register_callback(
                    data_type, symbol,
                    lambda data: self._handle_realtime_data(data, data_type, symbol)
                )
        
        # Start streaming
        self.running = True
        logger.info(f"Starting real-time streaming for {symbols}")
        
        await streamer.start()
    
    def _handle_realtime_data(self, data: Any, data_type: DataType, symbol: str):
        """Handle incoming real-time data"""
        try:
            if data:
                # Convert to DataFrame
                if isinstance(data, (OHLCVData, TradeData, OrderBookData, TickerData)):
                    df = pd.DataFrame([data.to_dict()])
                else:
                    df = pd.DataFrame([data])
                
                # Append to existing data
                existing_data = self.storage.load_data(
                    data_type, symbol, self.config.data.primary_exchange
                )
                
                if not existing_data.empty:
                    df = pd.concat([existing_data, df], ignore_index=True)
                    df = df.drop_duplicates(subset=['timestamp'])
                    df = df.sort_values('timestamp')
                
                # Save updated data
                self.storage.save_data(
                    df, data_type, symbol, self.config.data.primary_exchange
                )
                
                logger.debug(f"Updated {data_type.value} for {symbol}")
                
        except Exception as e:
            logger.error(f"Error handling real-time data: {str(e)}")
    
    async def stop_realtime_streaming(self):
        """Stop real-time data streaming"""
        self.running = False
        
        for streamer in self.streamers.values():
            await streamer.stop()
        
        self.streamers.clear()
        logger.info("Stopped real-time streaming")
    
    def get_data_summary(self) -> Dict:
        """Get summary of available data"""
        available = self.storage.get_available_data()
        summary = {
            'total_exchanges': len(available),
            'total_symbols': 0,
            'data_types': defaultdict(int),
            'timeframes': defaultdict(int)
        }
        
        for exchange, data_types in available.items():
            for data_type, symbols in data_types.items():
                summary['data_types'][data_type] += len(symbols)
                summary['total_symbols'] += len(symbols)
                
                for symbol, timeframes in symbols.items():
                    if timeframes:
                        summary['timeframes'][data_type] += len(timeframes)
        
        return summary
    
    def validate_data_quality(self, symbol: str, exchange: str, 
                            timeframe: str = None) -> Dict[str, float]:
        """Validate data quality"""
        try:
            if timeframe:
                df = self.storage.load_data(
                    DataType.OHLCV, symbol, exchange, timeframe
                )
            else:
                df = self.storage.load_data(DataType.OHLCV, symbol, exchange)
            
            if df.empty:
                return {'quality_score': 0.0, 'missing_data': 1.0}
            
            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            
            # Check for duplicates
            duplicate_ratio = df.duplicated().sum() / len(df)
            
            # Check for outliers (price changes > 50%)
            if 'close' in df.columns:
                returns = df['close'].pct_change().abs()
                outlier_ratio = (returns > 0.5).sum() / len(returns)
            else:
                outlier_ratio = 0.0
            
            # Check time continuity
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                time_diff = df['timestamp'].diff().dt.total_seconds()
                
                if timeframe:
                    expected_interval = self._timeframe_to_seconds(timeframe)
                    time_gap_ratio = (time_diff > expected_interval * 1.5).sum() / len(time_diff)
                else:
                    time_gap_ratio = 0.0
            else:
                time_gap_ratio = 0.0
            
            # Calculate quality score
            quality_score = 1.0 - (missing_ratio * 0.4 + duplicate_ratio * 0.3 + 
                                 outlier_ratio * 0.2 + time_gap_ratio * 0.1)
            quality_score = max(0.0, min(1.0, quality_score))
            
            return {
                'quality_score': quality_score,
                'missing_data': missing_ratio,
                'duplicates': duplicate_ratio,
                'outliers': outlier_ratio,
                'time_gaps': time_gap_ratio,
                'total_records': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            return {'quality_score': 0.0, 'error': str(e)}
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe to milliseconds"""
        tf_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000
        }
        return tf_map.get(timeframe, 60 * 60 * 1000)  # Default to 1h
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe to seconds"""
        return self._timeframe_to_ms(timeframe) // 1000
    
    async def cleanup(self):
        """Cleanup resources"""
        # Stop streaming
        if self.running:
            await self.stop_realtime_streaming()
        
        # Close exchange connections
        for connector in self.exchange_connectors.values():
            await connector.close()
        
        # Clear cache
        self.storage.clear_cache()
        
        logger.info("Data collector cleanup completed")

# ============ Scheduled Collection ============
class ScheduledCollector:
    """Scheduled data collection"""
    
    def __init__(self, collector: DataCollector):
        self.collector = collector
        self.scheduler = None
        self.tasks = []
    
    async def start_daily_collection(self):
        """Start daily data collection"""
        while True:
            try:
                # Wait until 00:05 UTC
                now = datetime.now(timezone.utc)
                target_time = now.replace(hour=0, minute=5, second=0, microsecond=0)
                
                if now > target_time:
                    target_time += timedelta(days=1)
                
                wait_seconds = (target_time - now).total_seconds()
                logger.info(f"Next daily collection in {wait_seconds:.0f} seconds")
                
                await asyncio.sleep(wait_seconds)
                
                # Perform daily collection
                await self._collect_daily_data()
                
            except Exception as e:
                logger.error(f"Error in daily collection: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def start_hourly_collection(self):
        """Start hourly data collection"""
        while True:
            try:
                # Wait until next hour
                now = datetime.now(timezone.utc)
                target_time = now.replace(minute=5, second=0, microsecond=0)
                
                if now > target_time:
                    target_time += timedelta(hours=1)
                
                wait_seconds = (target_time - now).total_seconds()
                
                await asyncio.sleep(wait_seconds)
                
                # Perform hourly collection
                await self._collect_hourly_data()
                
            except Exception as e:
                logger.error(f"Error in hourly collection: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _collect_daily_data(self):
        """Collect daily data"""
        logger.info("Starting daily data collection")
        
        try:
            # Collect on-chain data
            await self.collector.collect_onchain_data()
            
            # Collect sentiment data
            await self.collector.collect_sentiment_data()
            
            # Validate data quality
            symbol = self.collector.config.trading.symbol
            exchange = self.collector.config.data.primary_exchange
            
            quality_metrics = self.collector.validate_data_quality(symbol, exchange)
            logger.info(f"Data quality metrics: {quality_metrics}")
            
            # Send alerts if quality is low
            if quality_metrics.get('quality_score', 0) < 0.8:
                logger.warning(f"Low data quality for {symbol}: {quality_metrics}")
            
            logger.info("Daily data collection completed")
            
        except Exception as e:
            logger.error(f"Error in daily data collection: {str(e)}")
    
    async def _collect_hourly_data(self):
        """Collect hourly data"""
        logger.info("Starting hourly data collection")
        
        try:
            # Collect OHLCV data for last 24 hours
            symbol = self.collector.config.trading.symbol
            timeframe = self.collector.config.trading.timeframe.value
            
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=1)
            
            await self.collector.collect_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            logger.info("Hourly data collection completed")
            
        except Exception as e:
            logger.error(f"Error in hourly data collection: {str(e)}")

# ============ Example Usage ============
async def example_usage():
    """Example usage of data collector"""
    
    print("Data Collector Example")
    print("=" * 50)
    
    # Create data collector
    collector = DataCollector()
    
    try:
        # Collect historical data
        print("\n1. Collecting historical data...")
        historical_data = await collector.collect_historical_data(
            symbol='BTC/USDT',
            timeframe='1h',
            start_date='2024-01-01',
            end_date='2024-01-10'
        )
        
        for exchange, df in historical_data.items():
            print(f"  {exchange}: {len(df)} records")
        
        # Collect on-chain data
        print("\n2. Collecting on-chain data...")
        onchain_data = await collector.collect_onchain_data(
            metrics=['hash_rate', 'active_addresses'],
            symbol='BTC',
            start_date='2024-01-01'
        )
        
        for metric, df in onchain_data.items():
            print(f"  {metric}: {len(df)} records")
        
        # Collect sentiment data
        print("\n3. Collecting sentiment data...")
        sentiment_data = await collector.collect_sentiment_data('BTC')
        print(f"  Sentiment data: {len(sentiment_data)} records")
        
        # Get data summary
        print("\n4. Data summary:")
        summary = collector.get_data_summary()
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
        
        # Validate data quality
        print("\n5. Validating data quality...")
        quality_metrics = collector.validate_data_quality(
            symbol='BTC/USDT',
            exchange=collector.config.data.primary_exchange,
            timeframe='1h'
        )
        
        for metric, value in quality_metrics.items():
            print(f"  {metric}: {value}")
        
        # Start real-time streaming (for demonstration, run for 30 seconds)
        print("\n6. Starting real-time streaming (30 seconds)...")
        streaming_task = asyncio.create_task(
            collector.start_realtime_streaming(
                symbols=['BTC/USDT'],
                data_types=[DataType.TRADES, DataType.OHLCV]
            )
        )
        
        await asyncio.sleep(30)
        
        print("Stopping real-time streaming...")
        await collector.stop_realtime_streaming()
        
        # Cleanup
        print("\n7. Cleaning up...")
        await collector.cleanup()
        
        print("\nData collection example completed successfully!")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
        await collector.cleanup()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Bitcoin Trading AI - Data Collector")
    print("=" * 50)
    
    # Run example
    asyncio.run(example_usage())