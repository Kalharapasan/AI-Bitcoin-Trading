"""
WebSocket Server for Bitcoin Trading AI System
Provides real-time updates for market data, predictions, and trading events
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
import weakref

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
import jwt
from fastapi import HTTPException

# Import project modules
try:
    from config.config_manager import ConfigManager
    from core.data_processing.data_collector import DataCollector
    from core.trading.signal_generator import SignalGenerator
    from core.trading.order_manager import OrderManager
    from core.models.model_predictor import ModelPredictor
    from core.monitoring.performance_tracker import PerformanceTracker
    from core.monitoring.alert_manager import AlertManager
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
    AlertManager = type('AlertManager', (), {})
    CRUDOperations = type('CRUDOperations', (), {})
    DatabaseConnection = type('DatabaseConnection', (), {})
    setup_logger = lambda name: logging.getLogger(name)

# Initialize logger
logger = setup_logger(__name__)

# JWT Configuration
JWT_SECRET = "your-secret-key-change-in-production"  # Should match REST API
JWT_ALGORITHM = "HS256"

class WebSocketManager:
    """
    Manages WebSocket connections and subscriptions
    """
    
    def __init__(self):
        # Active connections grouped by subscription type
        self.connections: Dict[str, Set[WebSocketServerProtocol]] = {
            "market_data": set(),
            "trading_signals": set(),
            "predictions": set(),
            "trades": set(),
            "positions": set(),
            "alerts": set(),
            "system_status": set()
        }
        
        # Connection metadata
        self.connection_metadata: Dict[WebSocketServerProtocol, Dict[str, Any]] = {}
        
        # User connections mapping
        self.user_connections: Dict[str, Set[WebSocketServerProtocol]] = {}
        
        # Initialize services
        self.config_manager = None
        self.data_collector = None
        self.signal_generator = None
        self.order_manager = None
        self.model_predictor = None
        self.performance_tracker = None
        self.alert_manager = None
        self.crud = None
        
        # Market data cache
        self.market_cache: Dict[str, Dict] = {}
        
        # Active tasks
        self.tasks: List[asyncio.Task] = []
        
        # Server stats
        self.stats = {
            "connections_total": 0,
            "connections_active": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "start_time": datetime.now()
        }
    
    async def initialize_services(self):
        """Initialize required services"""
        try:
            self.config_manager = ConfigManager()
            
            # Initialize database
            db = DatabaseConnection()
            await db.connect()
            self.crud = CRUDOperations(db)
            
            # Initialize services
            self.data_collector = DataCollector(self.config_manager)
            self.signal_generator = SignalGenerator(self.config_manager)
            self.order_manager = OrderManager(self.config_manager, self.crud)
            self.model_predictor = ModelPredictor(self.config_manager)
            self.performance_tracker = PerformanceTracker(self.crud)
            self.alert_manager = AlertManager(self.config_manager, self.crud)
            
            logger.info("WebSocket services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket services: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Close all connections
        for subscription in self.connections.values():
            for ws in list(subscription):
                await self.close_connection(ws)
    
    async def authenticate(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def register_connection(self, websocket: WebSocketServerProtocol, user_data: Dict[str, Any]):
        """Register a new connection"""
        connection_id = id(websocket)
        username = user_data.get("sub", "anonymous")
        
        self.connection_metadata[websocket] = {
            "id": connection_id,
            "user": username,
            "role": user_data.get("role", "user"),
            "subscriptions": set(),
            "connected_at": datetime.now(),
            "last_activity": datetime.now()
        }
        
        # Add to user connections mapping
        if username not in self.user_connections:
            self.user_connections[username] = set()
        self.user_connections[username].add(websocket)
        
        # Update stats
        self.stats["connections_total"] += 1
        self.stats["connections_active"] += 1
        
        logger.info(f"New WebSocket connection: {username} (ID: {connection_id})")
    
    async def close_connection(self, websocket: WebSocketServerProtocol):
        """Close a connection and clean up"""
        if websocket not in self.connection_metadata:
            return
        
        metadata = self.connection_metadata[websocket]
        username = metadata["user"]
        
        # Remove from subscriptions
        for subscription in metadata["subscriptions"]:
            if subscription in self.connections:
                self.connections[subscription].discard(websocket)
        
        # Remove from user connections
        if username in self.user_connections:
            self.user_connections[username].discard(websocket)
            if not self.user_connections[username]:
                del self.user_connections[username]
        
        # Remove metadata
        del self.connection_metadata[websocket]
        
        # Update stats
        self.stats["connections_active"] -= 1
        
        # Close WebSocket
        try:
            await websocket.close()
        except Exception:
            pass
        
        logger.info(f"WebSocket connection closed: {username}")
    
    async def subscribe(self, websocket: WebSocketServerProtocol, subscription_type: str, params: Dict = None):
        """Subscribe to a data stream"""
        if websocket not in self.connection_metadata:
            return False
        
        if subscription_type not in self.connections:
            return False
        
        # Add to subscription
        self.connections[subscription_type].add(websocket)
        self.connection_metadata[websocket]["subscriptions"].add(subscription_type)
        
        # Store subscription parameters
        if params:
            if "subscription_params" not in self.connection_metadata[websocket]:
                self.connection_metadata[websocket]["subscription_params"] = {}
            self.connection_metadata[websocket]["subscription_params"][subscription_type] = params
        
        logger.debug(f"Subscription added: {subscription_type} for {self.connection_metadata[websocket]['user']}")
        
        # Send initial data if available
        await self.send_initial_data(websocket, subscription_type, params)
        
        return True
    
    async def unsubscribe(self, websocket: WebSocketServerProtocol, subscription_type: str):
        """Unsubscribe from a data stream"""
        if (websocket not in self.connection_metadata or 
            subscription_type not in self.connections):
            return
        
        self.connections[subscription_type].discard(websocket)
        self.connection_metadata[websocket]["subscriptions"].discard(subscription_type)
        
        # Remove subscription parameters
        if ("subscription_params" in self.connection_metadata[websocket] and
            subscription_type in self.connection_metadata[websocket]["subscription_params"]):
            del self.connection_metadata[websocket]["subscription_params"][subscription_type]
        
        logger.debug(f"Subscription removed: {subscription_type} for {self.connection_metadata[websocket]['user']}")
    
    async def send_initial_data(self, websocket: WebSocketServerProtocol, subscription_type: str, params: Dict = None):
        """Send initial data for a new subscription"""
        try:
            if subscription_type == "market_data" and params and "symbol" in params:
                symbol = params["symbol"]
                # Send latest cached market data
                if symbol in self.market_cache:
                    await self.send_message(websocket, {
                        "type": "market_data",
                        "data": self.market_cache[symbol],
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif subscription_type == "positions":
                # Send current positions
                positions = await self.order_manager.get_open_positions()
                await self.send_message(websocket, {
                    "type": "positions_update",
                    "data": positions,
                    "timestamp": datetime.now().isoformat()
                })
            
            elif subscription_type == "alerts":
                # Send active alerts
                alerts = await self.alert_manager.get_alerts(active_only=True, limit=10)
                await self.send_message(websocket, {
                    "type": "alerts_update",
                    "data": alerts,
                    "timestamp": datetime.now().isoformat()
                })
        
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
    
    async def broadcast(self, subscription_type: str, message: Dict, filter_params: Dict = None):
        """Broadcast message to all subscribers of a type"""
        if subscription_type not in self.connections:
            return
        
        for websocket in list(self.connections[subscription_type]):
            try:
                # Apply filtering based on subscription parameters
                if filter_params and websocket in self.connection_metadata:
                    metadata = self.connection_metadata[websocket]
                    if "subscription_params" in metadata and subscription_type in metadata["subscription_params"]:
                        sub_params = metadata["subscription_params"][subscription_type]
                        # Check if message matches subscription parameters
                        if not self._matches_filter(message, sub_params, filter_params):
                            continue
                
                await self.send_message(websocket, message)
                self.stats["messages_sent"] += 1
                
            except (ConnectionClosedOK, ConnectionClosedError):
                await self.close_connection(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
    
    def _matches_filter(self, message: Dict, subscription_params: Dict, filter_params: Dict) -> bool:
        """Check if message matches subscription filter parameters"""
        for key, value in filter_params.items():
            if key in subscription_params:
                # For market data, check symbol
                if key == "symbol" and message.get("data", {}).get("symbol") != subscription_params["symbol"]:
                    return False
                # For other filters, check exact match
                elif message.get("data", {}).get(key) != subscription_params[key]:
                    return False
        return True
    
    async def send_message(self, websocket: WebSocketServerProtocol, message: Dict):
        """Send a message to a specific WebSocket"""
        try:
            await websocket.send(json.dumps(message))
            self.connection_metadata[websocket]["last_activity"] = datetime.now()
        except (ConnectionClosedOK, ConnectionClosedError):
            await self.close_connection(websocket)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def send_to_user(self, username: str, message: Dict):
        """Send message to all connections of a specific user"""
        if username not in self.user_connections:
            return
        
        for websocket in list(self.user_connections[username]):
            try:
                await self.send_message(websocket, message)
            except Exception as e:
                logger.error(f"Error sending to user {username}: {e}")
    
    async def process_message(self, websocket: WebSocketServerProtocol, message: str):
        """Process incoming WebSocket message"""
        if websocket not in self.connection_metadata:
            return
        
        self.stats["messages_received"] += 1
        self.connection_metadata[websocket]["last_activity"] = datetime.now()
        
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                await self.handle_subscribe(websocket, data)
            elif message_type == "unsubscribe":
                await self.handle_unsubscribe(websocket, data)
            elif message_type == "ping":
                await self.handle_ping(websocket, data)
            elif message_type == "command":
                await self.handle_command(websocket, data)
            else:
                await self.send_message(websocket, {
                    "type": "error",
                    "error": "Unknown message type",
                    "timestamp": datetime.now().isoformat()
                })
        
        except json.JSONDecodeError:
            await self.send_message(websocket, {
                "type": "error",
                "error": "Invalid JSON format",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send_message(websocket, {
                "type": "error",
                "error": "Internal server error",
                "timestamp": datetime.now().isoformat()
            })
    
    async def handle_subscribe(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle subscription request"""
        subscription_type = data.get("subscription")
        params = data.get("params", {})
        
        if not subscription_type:
            await self.send_message(websocket, {
                "type": "error",
                "error": "Missing subscription type",
                "timestamp": datetime.now().isoformat()
            })
            return
        
        success = await self.subscribe(websocket, subscription_type, params)
        
        if success:
            await self.send_message(websocket, {
                "type": "subscription_confirmed",
                "subscription": subscription_type,
                "params": params,
                "timestamp": datetime.now().isoformat()
            })
        else:
            await self.send_message(websocket, {
                "type": "error",
                "error": f"Invalid subscription type: {subscription_type}",
                "timestamp": datetime.now().isoformat()
            })
    
    async def handle_unsubscribe(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle unsubscription request"""
        subscription_type = data.get("subscription")
        
        if not subscription_type:
            await self.send_message(websocket, {
                "type": "error",
                "error": "Missing subscription type",
                "timestamp": datetime.now().isoformat()
            })
            return
        
        await self.unsubscribe(websocket, subscription_type)
        
        await self.send_message(websocket, {
            "type": "unsubscription_confirmed",
            "subscription": subscription_type,
            "timestamp": datetime.now().isoformat()
        })
    
    async def handle_ping(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle ping message"""
        await self.send_message(websocket, {
            "type": "pong",
            "timestamp": datetime.now().isoformat(),
            "data": data.get("data")
        })
    
    async def handle_command(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle command message"""
        command = data.get("command")
        params = data.get("params", {})
        user = self.connection_metadata[websocket]["user"]
        
        if command == "get_stats":
            # Send connection stats
            await self.send_message(websocket, {
                "type": "stats",
                "data": {
                    "user_stats": {
                        "subscriptions": list(self.connection_metadata[websocket]["subscriptions"]),
                        "connected_since": self.connection_metadata[websocket]["connected_at"].isoformat()
                    },
                    "server_stats": self.stats
                },
                "timestamp": datetime.now().isoformat()
            })
        
        elif command == "get_price":
            symbol = params.get("symbol", "BTCUSDT")
            try:
                price = await self.data_collector.get_current_price(symbol)
                await self.send_message(websocket, {
                    "type": "price_response",
                    "data": {
                        "symbol": symbol,
                        "price": price,
                        "timestamp": datetime.now().isoformat()
                    },
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                await self.send_message(websocket, {
                    "type": "error",
                    "error": f"Failed to get price: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
        
        elif command == "place_order":
            # Only allow if user has trading role
            if self.connection_metadata[websocket]["role"] not in ["admin", "trader"]:
                await self.send_message(websocket, {
                    "type": "error",
                    "error": "Insufficient permissions",
                    "timestamp": datetime.now().isoformat()
                })
                return
            
            try:
                trade_result = await self.order_manager.execute_trade(
                    symbol=params.get("symbol", "BTCUSDT"),
                    side=params.get("side"),
                    quantity=params.get("quantity", 0.01),
                    order_type=params.get("order_type", "market"),
                    price=params.get("price"),
                    strategy="websocket_command"
                )
                
                await self.send_message(websocket, {
                    "type": "order_response",
                    "data": trade_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Broadcast trade update
                await self.broadcast("trades", {
                    "type": "trade_executed",
                    "data": trade_result,
                    "user": user,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                await self.send_message(websocket, {
                    "type": "error",
                    "error": f"Failed to place order: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })


class MarketDataStreamer:
    """
    Streams real-time market data
    """
    
    def __init__(self, ws_manager: WebSocketManager):
        self.ws_manager = ws_manager
        self.running = False
        self.symbols = set(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    
    async def start(self):
        """Start streaming market data"""
        self.running = True
        logger.info("Starting market data streaming")
        
        while self.running:
            try:
                for symbol in self.symbols:
                    await self.update_market_data(symbol)
                
                # Update every 5 seconds
                await asyncio.sleep(5)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market data streaming: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    async def stop(self):
        """Stop streaming market data"""
        self.running = False
        logger.info("Stopping market data streaming")
    
    async def update_market_data(self, symbol: str):
        """Update market data for a symbol"""
        try:
            if not self.ws_manager.data_collector:
                return
            
            # Get current price
            price = await self.ws_manager.data_collector.get_current_price(symbol)
            
            # Get 24h stats
            ticker = await self.ws_manager.data_collector.get_ticker(symbol)
            
            # Get orderbook snapshot
            orderbook = await self.ws_manager.data_collector.get_orderbook(symbol, depth=10)
            
            market_data = {
                "symbol": symbol,
                "price": price,
                "change_24h": ticker.get("priceChangePercent", 0),
                "high_24h": ticker.get("highPrice", price),
                "low_24h": ticker.get("lowPrice", price),
                "volume_24h": ticker.get("volume", 0),
                "bid": orderbook.get("bids", [])[0][0] if orderbook.get("bids") else price,
                "ask": orderbook.get("asks", [])[0][0] if orderbook.get("asks") else price,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update cache
            self.ws_manager.market_cache[symbol] = market_data
            
            # Broadcast to subscribers
            await self.ws_manager.broadcast("market_data", {
                "type": "market_update",
                "data": market_data,
                "timestamp": datetime.now().isoformat()
            }, filter_params={"symbol": symbol})
        
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")


class PredictionStreamer:
    """
    Streams real-time predictions
    """
    
    def __init__(self, ws_manager: WebSocketManager):
        self.ws_manager = ws_manager
        self.running = False
    
    async def start(self):
        """Start streaming predictions"""
        self.running = True
        logger.info("Starting prediction streaming")
        
        while self.running:
            try:
                if self.ws_manager.model_predictor:
                    # Get predictions for main symbols
                    symbols = ["BTCUSDT", "ETHUSDT"]
                    timeframes = ["1m", "5m", "15m", "1h"]
                    
                    for symbol in symbols:
                        for timeframe in timeframes:
                            await self.update_predictions(symbol, timeframe)
                
                # Update every minute
                await asyncio.sleep(60)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prediction streaming: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def stop(self):
        """Stop streaming predictions"""
        self.running = False
        logger.info("Stopping prediction streaming")
    
    async def update_predictions(self, symbol: str, timeframe: str):
        """Update predictions for a symbol and timeframe"""
        try:
            predictions = await self.ws_manager.model_predictor.predict(
                symbol=symbol,
                timeframe=timeframe
            )
            
            prediction_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "predictions": predictions,
                "timestamp": datetime.now().isoformat()
            }
            
            # Broadcast to subscribers
            await self.ws_manager.broadcast("predictions", {
                "type": "prediction_update",
                "data": prediction_data,
                "timestamp": datetime.now().isoformat()
            })
        
        except Exception as e:
            logger.error(f"Error updating predictions for {symbol} {timeframe}: {e}")


class TradingSignalStreamer:
    """
    Streams trading signals
    """
    
    def __init__(self, ws_manager: WebSocketManager):
        self.ws_manager = ws_manager
        self.running = False
    
    async def start(self):
        """Start streaming trading signals"""
        self.running = True
        logger.info("Starting trading signal streaming")
        
        while self.running:
            try:
                if self.ws_manager.signal_generator:
                    # Generate signals for main symbols
                    symbols = ["BTCUSDT", "ETHUSDT"]
                    timeframes = ["5m", "15m", "1h"]
                    
                    for symbol in symbols:
                        for timeframe in timeframes:
                            await self.update_signals(symbol, timeframe)
                
                # Update every 30 seconds
                await asyncio.sleep(30)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in signal streaming: {e}")
                await asyncio.sleep(15)  # Wait before retry
    
    async def stop(self):
        """Stop streaming trading signals"""
        self.running = False
        logger.info("Stopping trading signal streaming")
    
    async def update_signals(self, symbol: str, timeframe: str):
        """Update trading signals for a symbol and timeframe"""
        try:
            signals = await self.ws_manager.signal_generator.generate_signals(
                symbol=symbol,
                timeframe=timeframe
            )
            
            signal_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "signals": signals,
                "timestamp": datetime.now().isoformat()
            }
            
            # Broadcast to subscribers
            await self.ws_manager.broadcast("trading_signals", {
                "type": "signal_update",
                "data": signal_data,
                "timestamp": datetime.now().isoformat()
            }, filter_params={"symbol": symbol})
        
        except Exception as e:
            logger.error(f"Error updating signals for {symbol} {timeframe}: {e}")


class AlertMonitor:
    """
    Monitors and streams alerts
    """
    
    def __init__(self, ws_manager: WebSocketManager):
        self.ws_manager = ws_manager
        self.running = False
        self.last_alert_check = datetime.now()
    
    async def start(self):
        """Start monitoring alerts"""
        self.running = True
        logger.info("Starting alert monitoring")
        
        while self.running:
            try:
                if self.ws_manager.alert_manager:
                    await self.check_alerts()
                
                # Check every 10 seconds
                await asyncio.sleep(10)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def stop(self):
        """Stop monitoring alerts"""
        self.running = False
        logger.info("Stopping alert monitoring")
    
    async def check_alerts(self):
        """Check for new alerts"""
        try:
            # Get new alerts since last check
            alerts = await self.ws_manager.alert_manager.get_alerts(
                active_only=True,
                since=self.last_alert_check
            )
            
            if alerts:
                for alert in alerts:
                    # Broadcast alert
                    await self.ws_manager.broadcast("alerts", {
                        "type": "alert_triggered",
                        "data": alert,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Update last check time
                self.last_alert_check = datetime.now()
        
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")


class WebSocketServer:
    """
    Main WebSocket server class
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.ws_manager = WebSocketManager()
        self.market_streamer = MarketDataStreamer(self.ws_manager)
        self.prediction_streamer = PredictionStreamer(self.ws_manager)
        self.signal_streamer = TradingSignalStreamer(self.ws_manager)
        self.alert_monitor = AlertMonitor(self.ws_manager)
        self.server = None
    
    async def start(self):
        """Start the WebSocket server"""
        try:
            # Initialize services
            await self.ws_manager.initialize_services()
            
            # Start streaming tasks
            tasks = [
                asyncio.create_task(self.market_streamer.start()),
                asyncio.create_task(self.prediction_streamer.start()),
                asyncio.create_task(self.signal_streamer.start()),
                asyncio.create_task(self.alert_monitor.start()),
            ]
            self.ws_manager.tasks.extend(tasks)
            
            # Start WebSocket server
            self.server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            
            # Keep server running
            await self.server.wait_closed()
        
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def stop(self):
        """Stop the WebSocket server"""
        logger.info("Stopping WebSocket server...")
        
        # Stop streaming tasks
        await self.market_streamer.stop()
        await self.prediction_streamer.stop()
        await self.signal_streamer.stop()
        await self.alert_monitor.stop()
        
        # Cleanup WebSocket manager
        await self.ws_manager.cleanup()
        
        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("WebSocket server stopped")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connection"""
        logger.debug(f"New connection attempt: {path}")
        
        # Authenticate via query parameters
        token = None
        if path.startswith("/ws?"):
            import urllib.parse
            query = urllib.parse.urlparse(path).query
            params = urllib.parse.parse_qs(query)
            token = params.get("token", [None])[0]
        
        if not token:
            # Try to get token from headers
            try:
                headers = dict(websocket.request_headers)
                token = headers.get("authorization", "").replace("Bearer ", "")
            except Exception:
                pass
        
        # Authenticate
        try:
            user_data = await self.ws_manager.authenticate(token)
        except HTTPException:
            await websocket.close(1008, "Authentication failed")  # Policy Violation
            return
        
        # Register connection
        await self.ws_manager.register_connection(websocket, user_data)
        
        try:
            # Send welcome message
            await self.ws_manager.send_message(websocket, {
                "type": "welcome",
                "message": "Connected to Bitcoin Trading AI WebSocket",
                "user": user_data.get("sub"),
                "timestamp": datetime.now().isoformat(),
                "server_info": {
                    "version": "1.0.0",
                    "supported_subscriptions": list(self.ws_manager.connections.keys())
                }
            })
            
            # Handle messages
            async for message in websocket:
                await self.ws_manager.process_message(websocket, message)
        
        except (ConnectionClosedOK, ConnectionClosedError):
            pass  # Connection closed normally
        except Exception as e:
            logger.error(f"Error in connection handler: {e}")
        finally:
            await self.ws_manager.close_connection(websocket)
    
    async def broadcast_system_status(self):
        """Broadcast system status periodically"""
        while True:
            try:
                system_status = {
                    "connections": self.ws_manager.stats["connections_active"],
                    "messages_sent": self.ws_manager.stats["messages_sent"],
                    "uptime": (datetime.now() - self.ws_manager.stats["start_time"]).total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.ws_manager.broadcast("system_status", {
                    "type": "system_status",
                    "data": system_status,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update every 30 seconds
                await asyncio.sleep(30)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error broadcasting system status: {e}")
                await asyncio.sleep(60)


# FastAPI integration for WebSocket endpoints
async def websocket_endpoint(websocket: WebSocketServerProtocol):
    """FastAPI WebSocket endpoint"""
    server = WebSocketServer()
    await server.handle_connection(websocket, str(websocket.path))


async def start_websocket_server():
    """Start WebSocket server as a separate task"""
    server = WebSocketServer()
    await server.start()


if __name__ == "__main__":
    # Example usage: run standalone WebSocket server
    import asyncio
    
    async def main():
        server = WebSocketServer()
        try:
            await server.start()
        except KeyboardInterrupt:
            await server.stop()
        except Exception as e:
            logger.error(f"Server error: {e}")
            await server.stop()
    
    # Run the server
    asyncio.run(main())