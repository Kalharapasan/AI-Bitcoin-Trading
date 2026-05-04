#!/usr/bin/env python3
"""
Bitcoin Trading AI - Main Entry Point
Main orchestrator for the Bitcoin Trading AI application.
Manages trading, model serving, and system monitoring.
"""

import os
import sys
import argparse
import logging
import signal
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import json
import yaml
from contextlib import asynccontextmanager

# Try to import FastAPI - if it fails, we'll run in simplified mode
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, HTMLResponse
    import websockets
    from websockets.exceptions import ConnectionClosed
    FASTAPI_AVAILABLE = True
except ImportError as e:
    print(f"FastAPI not available: {e}")
    print("Running in simplified mode without web interface")
    FASTAPI_AVAILABLE = False
    # Create dummy classes to prevent errors
    class FastAPI: pass
    class HTTPException: pass
    class BackgroundTasks: pass

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.trading.signal_generator import SignalGenerator
from core.trading.position_sizer import PositionSizer
from core.trading.order_manager import OrderManager
from core.trading.execution_engine import ExecutionEngine
from core.data_processing.data_collector import DataCollector
from core.data_processing.feature_engineer import FeatureEngineer
from core.models.model_manager import ModelManager
from core.models.model_predictor import ModelPredictor
from core.monitoring.performance_tracker import PerformanceTracker
from core.monitoring.alert_manager import AlertManager
from core.monitoring.metrics_collector import MetricsCollector
from backtesting.backtest_engine import BacktestEngine
from database.connection import get_database_manager, init_database
from config.config_manager import ConfigManager


class TradingAISystem:
    """Main system orchestrator for Bitcoin Trading AI"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = self.setup_logger()
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
        
        # Initialize components
        self.initialize_components()
        
    def setup_logger(self) -> logging.Logger:
        """Setup system logger"""
        logger = logging.getLogger('trading_ai')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        log_file = project_root / "logs" / "trading_system.log"
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def initialize_components(self):
        """Initialize all system components"""
        self.logger.info("Initializing system components...")
        
        try:
            # Core trading components
            self.signal_generator = SignalGenerator(self.config)
            self.position_sizer = PositionSizer(self.config)
            self.order_manager = OrderManager(self.config)
            self.execution_engine = ExecutionEngine(self.config)
            
            # Data processing
            self.data_collector = DataCollector(self.config)
            self.feature_engineer = FeatureEngineer(self.config)
            
            # Model management
            self.model_manager = ModelManager(self.config)
            self.model_predictor = ModelPredictor(self.config)
            
            # Monitoring
            self.performance_tracker = PerformanceTracker(self.config)
            self.alert_manager = AlertManager(self.config)
            self.metrics_collector = MetricsCollector(self.config)
            
            # Backtesting
            self.backtest_engine = BacktestEngine(self.config)
            
            # Database
            self.db_manager = get_database_manager(self.config)
            
            # Trading state
            self.trading_state = {
                'active': False,
                'mode': 'paper',  # paper, live
                'current_positions': {},
                'open_orders': [],
                'performance': {},
                'last_signal': None,
                'last_trade': None
            }
            
            self.logger.info("System components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def start_trading(self, mode: str = 'paper'):
        """Start the trading system"""
        self.logger.info(f"Starting trading system in {mode} mode")
        
        if self.is_running:
            self.logger.warning("Trading system is already running")
            return False
        
        try:
            self.trading_state['active'] = True
            self.trading_state['mode'] = mode
            self.is_running = True
            
            # Start background tasks
            tasks = [
                self.collect_market_data(),
                self.monitor_positions(),
                self.generate_signals(),
                self.execute_trades(),
                self.track_performance(),
                self.send_alerts()
            ]
            
            for task_func in tasks:
                task = asyncio.create_task(task_func)
                self.tasks.append(task)
            
            self.logger.info(f"Trading system started successfully in {mode} mode")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start trading system: {e}")
            self.trading_state['active'] = False
            self.is_running = False
            return False
    
    async def stop_trading(self):
        """Stop the trading system"""
        self.logger.info("Stopping trading system...")
        
        self.trading_state['active'] = False
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks.clear()
        
        # Close all open positions if in live mode
        if self.trading_state['mode'] == 'live':
            await self.close_all_positions()
        
        self.logger.info("Trading system stopped successfully")
    
    async def collect_market_data(self):
        """Continuously collect market data"""
        self.logger.info("Starting market data collection")
        
        symbol = self.config.get('trading.symbol', 'BTC/USDT')
        timeframe = self.config.get('trading.timeframe', '1h')
        interval = 60  # seconds
        
        while self.is_running:
            try:
                # Collect latest data
                latest_data = await self.data_collector.get_latest_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=100
                )
                
                if latest_data:
                    # Process and store data
                    processed_data = self.feature_engineer.process_data(latest_data)
                    
                    # Update metrics
                    self.metrics_collector.record_market_data(processed_data)
                    
                    # Broadcast via WebSocket
                    await self.broadcast_market_update(processed_data)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in market data collection: {e}")
                await asyncio.sleep(interval)
    
    async def generate_signals(self):
        """Generate trading signals"""
        self.logger.info("Starting signal generation")
        
        symbol = self.config.get('trading.symbol', 'BTC/USDT')
        interval = 300  # 5 minutes
        
        while self.is_running:
            try:
                # Get latest market data
                market_data = await self.data_collector.get_latest_data(
                    symbol=symbol,
                    timeframe='5m',
                    limit=50
                )
                
                if market_data and len(market_data) > 20:
                    # Generate signal
                    historical_data = {
                        'close': [d['close'] for d in market_data],
                        'volume': [d['volume'] for d in market_data]
                    }
                    
                    current_data = market_data[-1]
                    
                    signal = self.signal_generator.generate_signal(
                        symbol=symbol,
                        market_data=current_data,
                        historical_data=historical_data,
                        strategy='hybrid'  # Combine technical and ML
                    )
                    
                    # Update state
                    self.trading_state['last_signal'] = signal
                    
                    # Log signal
                    self.logger.info(f"Generated signal: {signal}")
                    
                    # Broadcast signal
                    await self.broadcast_signal(signal)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in signal generation: {e}")
                await asyncio.sleep(interval)
    
    async def execute_trades(self):
        """Execute trades based on signals"""
        self.logger.info("Starting trade execution")
        
        symbol = self.config.get('trading.symbol', 'BTC/USDT')
        account_balance = self.config.get('trading.initial_capital', 10000.0)
        interval = 60  # 1 minute
        
        while self.is_running:
            try:
                # Check for new signals
                signal = self.trading_state.get('last_signal')
                
                if signal and signal.get('timestamp') != self.trading_state.get('last_trade_time'):
                    # Get current price
                    current_data = await self.data_collector.get_latest_data(
                        symbol=symbol,
                        timeframe='1m',
                        limit=1
                    )
                    
                    if current_data:
                        current_price = current_data[0]['close']
                        
                        # Execute trade
                        execution_result = await self.execution_engine.execute_trade(
                            symbol=symbol,
                            signal=signal,
                            account_balance=account_balance,
                            current_price=current_price,
                            mode=self.trading_state['mode']
                        )
                        
                        if execution_result.get('success'):
                            # Update state
                            self.trading_state['last_trade'] = execution_result
                            self.trading_state['last_trade_time'] = signal['timestamp']
                            
                            # Update positions
                            if execution_result['action'] == 'buy':
                                self.trading_state['current_positions'][symbol] = {
                                    'quantity': execution_result['quantity'],
                                    'entry_price': execution_result['price'],
                                    'timestamp': datetime.now()
                                }
                            elif execution_result['action'] == 'sell':
                                self.trading_state['current_positions'].pop(symbol, None)
                            
                            # Log trade
                            self.logger.info(f"Trade executed: {execution_result}")
                            
                            # Broadcast trade
                            await self.broadcast_trade(execution_result)
                            
                            # Update performance
                            self.performance_tracker.record_trade(execution_result)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in trade execution: {e}")
                await asyncio.sleep(interval)
    
    async def monitor_positions(self):
        """Monitor and manage open positions"""
        self.logger.info("Starting position monitoring")
        
        interval = 30  # 30 seconds
        
        while self.is_running:
            try:
                for symbol, position in self.trading_state['current_positions'].items():
                    # Get current price
                    current_data = await self.data_collector.get_latest_data(
                        symbol=symbol,
                        timeframe='1m',
                        limit=1
                    )
                    
                    if current_data:
                        current_price = current_data[0]['close']
                        entry_price = position['entry_price']
                        
                        # Calculate P&L
                        pnl = (current_price - entry_price) / entry_price
                        
                        # Check stop loss
                        stop_loss_pct = self.config.get('trading.stop_loss', 0.02)
                        if pnl < -stop_loss_pct:
                            # Trigger stop loss
                            await self.execute_stop_loss(symbol, position, current_price)
                        
                        # Check take profit
                        take_profit_pct = self.config.get('trading.take_profit', 0.05)
                        if pnl > take_profit_pct:
                            # Trigger take profit
                            await self.execute_take_profit(symbol, position, current_price)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(interval)
    
    async def execute_stop_loss(self, symbol: str, position: Dict, current_price: float):
        """Execute stop loss order"""
        try:
            self.logger.info(f"Executing stop loss for {symbol}")
            
            # Create sell order
            order_result = await self.order_manager.create_order(
                symbol=symbol,
                side='sell',
                order_type='market',
                quantity=position['quantity'],
                price=current_price
            )
            
            if order_result.get('success'):
                # Update state
                self.trading_state['current_positions'].pop(symbol, None)
                
                # Record stop loss
                self.performance_tracker.record_stop_loss({
                    'symbol': symbol,
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'pnl': (current_price - position['entry_price']) / position['entry_price'],
                    'timestamp': datetime.now(),
                    'reason': 'stop_loss'
                })
        
        except Exception as e:
            self.logger.error(f"Error executing stop loss: {e}")
    
    async def execute_take_profit(self, symbol: str, position: Dict, current_price: float):
        """Execute take profit order"""
        try:
            self.logger.info(f"Executing take profit for {symbol}")
            
            # Create sell order
            order_result = await self.order_manager.create_order(
                symbol=symbol,
                side='sell',
                order_type='market',
                quantity=position['quantity'],
                price=current_price
            )
            
            if order_result.get('success'):
                # Update state
                self.trading_state['current_positions'].pop(symbol, None)
                
                # Record take profit
                self.performance_tracker.record_take_profit({
                    'symbol': symbol,
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'pnl': (current_price - position['entry_price']) / position['entry_price'],
                    'timestamp': datetime.now(),
                    'reason': 'take_profit'
                })
        
        except Exception as e:
            self.logger.error(f"Error executing take profit: {e}")
    
    async def close_all_positions(self):
        """Close all open positions"""
        self.logger.info("Closing all open positions")
        
        for symbol, position in self.trading_state['current_positions'].items():
            try:
                # Get current price
                current_data = await self.data_collector.get_latest_data(
                    symbol=symbol,
                    timeframe='1m',
                    limit=1
                )
                
                if current_data:
                    current_price = current_data[0]['close']
                    
                    # Create sell order
                    await self.order_manager.create_order(
                        symbol=symbol,
                        side='sell',
                        order_type='market',
                        quantity=position['quantity'],
                        price=current_price
                    )
                    
                    self.logger.info(f"Closed position for {symbol}")
            
            except Exception as e:
                self.logger.error(f"Error closing position for {symbol}: {e}")
    
    async def track_performance(self):
        """Track and report trading performance"""
        self.logger.info("Starting performance tracking")
        
        interval = 300  # 5 minutes
        
        while self.is_running:
            try:
                # Calculate performance metrics
                performance = await self.performance_tracker.calculate_metrics()
                
                # Update state
                self.trading_state['performance'] = performance
                
                # Log performance
                self.logger.info(f"Performance update: {performance}")
                
                # Broadcast performance
                await self.broadcast_performance(performance)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(interval)
    
    async def send_alerts(self):
        """Send system alerts and notifications"""
        self.logger.info("Starting alert system")
        
        interval = 60  # 1 minute
        
        while self.is_running:
            try:
                # Check for alerts
                alerts = await self.alert_manager.check_alerts(
                    trading_state=self.trading_state,
                    performance=self.trading_state.get('performance', {})
                )
                
                # Send alerts
                for alert in alerts:
                    await self.alert_manager.send_alert(alert)
                    
                    # Broadcast alert
                    await self.broadcast_alert(alert)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert system: {e}")
                await asyncio.sleep(interval)
    
    async def broadcast_market_update(self, data: Dict):
        """Broadcast market data update via WebSocket"""
        try:
            message = {
                'type': 'market_update',
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            await self.websocket_broadcast(json.dumps(message, default=str))
        
        except Exception as e:
            self.logger.error(f"Error broadcasting market update: {e}")
    
    async def broadcast_signal(self, signal: Dict):
        """Broadcast trading signal via WebSocket"""
        try:
            message = {
                'type': 'signal',
                'timestamp': datetime.now().isoformat(),
                'data': signal
            }
            
            await self.websocket_broadcast(json.dumps(message, default=str))
        
        except Exception as e:
            self.logger.error(f"Error broadcasting signal: {e}")
    
    async def broadcast_trade(self, trade: Dict):
        """Broadcast trade execution via WebSocket"""
        try:
            message = {
                'type': 'trade_executed',
                'timestamp': datetime.now().isoformat(),
                'data': trade
            }
            
            await self.websocket_broadcast(json.dumps(message, default=str))
        
        except Exception as e:
            self.logger.error(f"Error broadcasting trade: {e}")
    
    async def broadcast_performance(self, performance: Dict):
        """Broadcast performance update via WebSocket"""
        try:
            message = {
                'type': 'performance_update',
                'timestamp': datetime.now().isoformat(),
                'data': performance
            }
            
            await self.websocket_broadcast(json.dumps(message, default=str))
        
        except Exception as e:
            self.logger.error(f"Error broadcasting performance: {e}")
    
    async def broadcast_alert(self, alert: Dict):
        """Broadcast system alert via WebSocket"""
        try:
            message = {
                'type': 'system_alert',
                'timestamp': datetime.now().isoformat(),
                'data': alert
            }
            
            await self.websocket_broadcast(json.dumps(message, default=str))
        
        except Exception as e:
            self.logger.error(f"Error broadcasting alert: {e}")
    
    async def websocket_broadcast(self, message: str):
        """Broadcast message to all connected WebSocket clients"""
        # WebSocket broadcast logic would be implemented here
        # This is a placeholder for the actual implementation
        pass
    
    async def run_backtest(self, config: Dict) -> Dict:
        """Run a backtest with given configuration"""
        try:
            self.logger.info(f"Starting backtest with config: {config}")
            
            # Configure backtest engine
            self.backtest_engine.configure(
                initial_capital=config.get('initial_capital', 10000),
                commission_rate=config.get('commission_rate', 0.001),
                slippage=config.get('slippage', 0.0001)
            )
            
            # Load data
            data = await self.data_collector.load_historical_data(
                symbol=config.get('symbol', 'BTC/USDT'),
                timeframe=config.get('timeframe', '1h'),
                start_date=config.get('start_date'),
                end_date=config.get('end_date')
            )
            
            # Generate signals
            signals = await self.generate_backtest_signals(data, config.get('strategy'))
            
            # Run backtest
            results = await self.backtest_engine.run_backtest(data, signals)
            
            self.logger.info(f"Backtest completed: {results.get('total_trades', 0)} trades")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise
    
    async def generate_backtest_signals(self, data: List[Dict], strategy: str) -> List[Dict]:
        """Generate signals for backtesting"""
        signals = []
        
        for i in range(50, len(data)):  # Start from 50th point for indicators
            historical = data[i-50:i]
            current = data[i]
            
            historical_data = {
                'close': [d['close'] for d in historical],
                'volume': [d['volume'] for d in historical]
            }
            
            signal = self.signal_generator.generate_signal(
                symbol='BTC/USDT',
                market_data=current,
                historical_data=historical_data,
                strategy=strategy
            )
            
            if signal:
                signal['timestamp'] = current['timestamp']
                signals.append(signal)
        
        return signals
    
    async def train_model(self, config: Dict) -> Dict:
        """Train a new model"""
        try:
            self.logger.info(f"Starting model training with config: {config}")
            
            # Load data
            data = await self.data_collector.load_historical_data(
                symbol=config.get('symbol', 'BTC/USDT'),
                timeframe=config.get('timeframe', '1h'),
                start_date=config.get('start_date'),
                end_date=config.get('end_date')
            )
            
            # Train model
            model_id = await self.model_manager.train_model(
                data=data,
                model_type=config.get('model_type', 'transformer'),
                config=config
            )
            
            self.logger.info(f"Model training completed: {model_id}")
            
            return {'model_id': model_id, 'status': 'completed'}
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
    
    async def predict_with_model(self, model_id: str, data: Dict) -> Dict:
        """Make prediction using a trained model"""
        try:
            prediction = await self.model_predictor.predict(
                model_id=model_id,
                data=data
            )
            
            return {'prediction': prediction, 'model_id': model_id}
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            raise
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'status': 'running' if self.is_running else 'stopped',
            'trading_mode': self.trading_state['mode'],
            'active_positions': len(self.trading_state['current_positions']),
            'open_orders': len(self.trading_state['open_orders']),
            'last_signal': self.trading_state.get('last_signal'),
            'last_trade': self.trading_state.get('last_trade'),
            'performance': self.trading_state.get('performance', {}),
            'timestamp': datetime.now().isoformat()
        }
    
    async def cleanup(self):
        """Cleanup system resources"""
        self.logger.info("Cleaning up system resources...")
        
        # Stop trading if running
        if self.is_running:
            await self.stop_trading()
        
        # Close database connections
        if hasattr(self, 'db_manager'):
            self.db_manager.close()
        
        self.logger.info("System cleanup completed")


class TradingAPI:
    """FastAPI application for trading system"""
    
    def __init__(self, trading_system: TradingAISystem):
        self.trading_system = trading_system
        if FASTAPI_AVAILABLE:
            self.app = self.create_app()
            self.websocket_clients = set()
        else:
            self.app = None
            self.websocket_clients = set()
        
    def create_app(self):
        """Create FastAPI application"""
        if not FASTAPI_AVAILABLE:
            return None
        
        @asynccontextmanager
        async def lifespan(app):
            # Startup
            yield
            # Shutdown
            await self.trading_system.cleanup()
        
        app = FastAPI(
            title="Bitcoin Trading AI API",
            description="API for Bitcoin Trading AI System",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self.register_routes(app)
        
        return app
    
    def register_routes(self, app: FastAPI):
        """Register API routes"""
        
        @app.get("/", response_class=HTMLResponse)
        async def root():
            return """
            <html>
                <head>
                    <title>Bitcoin Trading AI</title>
                </head>
                <body>
                    <h1>Bitcoin Trading AI System</h1>
                    <p>API is running. Check <a href="/docs">/docs</a> for API documentation.</p>
                </body>
            </html>
            """
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system": self.trading_system.get_system_status()
            }
        
        @app.get("/api/v1/status")
        async def get_status():
            """Get system status"""
            return self.trading_system.get_system_status()
        
        @app.post("/api/v1/trading/start")
        async def start_trading(mode: str = "paper"):
            """Start trading system"""
            try:
                success = await self.trading_system.start_trading(mode)
                return {"success": success, "mode": mode}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/v1/trading/stop")
        async def stop_trading():
            """Stop trading system"""
            try:
                await self.trading_system.stop_trading()
                return {"success": True}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/v1/backtest")
        async def run_backtest(config: Dict):
            """Run a backtest"""
            try:
                results = await self.trading_system.run_backtest(config)
                return results
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/v1/models/train")
        async def train_model(config: Dict):
            """Train a new model"""
            try:
                results = await self.trading_system.train_model(config)
                return results
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/v1/models/predict")
        async def predict(model_id: str, data: Dict):
            """Make prediction with model"""
            try:
                results = await self.trading_system.predict_with_model(model_id, data)
                return results
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/v1/performance")
        async def get_performance():
            """Get trading performance"""
            try:
                performance = self.trading_system.trading_state.get('performance', {})
                return performance
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/v1/positions")
        async def get_positions():
            """Get current positions"""
            try:
                positions = self.trading_system.trading_state['current_positions']
                return positions
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket):
            """WebSocket endpoint for real-time updates"""
            await self.handle_websocket_connection(websocket)
    
    async def handle_websocket_connection(self, websocket):
        """Handle WebSocket connection"""
        self.websocket_clients.add(websocket)
        try:
            await websocket.send(json.dumps({
                "type": "connected",
                "message": "Connected to Bitcoin Trading AI WebSocket",
                "timestamp": datetime.now().isoformat()
            }))
            
            # Keep connection alive
            async for message in websocket:
                # Handle incoming messages if needed
                pass
                
        except ConnectionClosed:
            pass
        finally:
            self.websocket_clients.remove(websocket)
    
    async def broadcast_to_websockets(self, message: Dict):
        """Broadcast message to all WebSocket clients"""
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send(json.dumps(message))
            except ConnectionClosed:
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.websocket_clients.remove(client)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Bitcoin Trading AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['api', 'trading', 'backtest', 'training', 'all'],
        default='all',
        help='Operation mode (default: all)'
    )
    
    parser.add_argument(
        '--trading-mode',
        type=str,
        choices=['paper', 'live'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='API host (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='API port (default: 8000)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Log level (default: INFO)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes (default: 1)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    return parser.parse_args()


async def main_async():
    """Main async entry point"""
    args = parse_arguments()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = ConfigManager()
        if args.config:
            config.load_config(args.config)
        
        # Initialize trading system
        trading_system = TradingAISystem(config)
        
        # Initialize API only if FastAPI is available
        if FASTAPI_AVAILABLE:
            trading_api = TradingAPI(trading_system)
        else:
            trading_api = None
        
        # Handle signals
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(trading_system.cleanup())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run based on mode
        if args.mode in ['api', 'all'] and FASTAPI_AVAILABLE:
            # Start API server
            config = uvicorn.Config(
                app=trading_api.app,
                host=args.host,
                port=args.port,
                workers=args.workers,
                reload=args.reload
            )
            
            server = uvicorn.Server(config)
            
            if args.mode == 'api':
                await server.serve()
            else:
                # Start API in background
                api_task = asyncio.create_task(server.serve())
        elif args.mode in ['api', 'all'] and not FASTAPI_AVAILABLE:
            logger.warning("FastAPI not available, skipping web interface")
        
        if args.mode in ['trading', 'all']:
            # Start trading
            await trading_system.start_trading(args.trading_mode)
        
        if args.mode == 'all':
            # Keep running
            if FASTAPI_AVAILABLE:
                await asyncio.Event().wait()
            else:
                # In simplified mode, just run for a bit then exit
                logger.info("Running in simplified mode - will exit after demonstration")
                await asyncio.sleep(30)  # Run for 30 seconds then exit
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise
    finally:
        # Cleanup
        if 'trading_system' in locals():
            await trading_system.cleanup()


def main():
    """Main entry point"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()