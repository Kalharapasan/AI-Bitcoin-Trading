"""
Test file for database models in the Bitcoin Trading AI application.
Unit tests for database models and CRUD operations.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import (
    Base, TradeSide, OrderStatus, Timeframe,
    MarketData, Trade, Order, TradingSession,
    PerformanceMetrics, ModelTraining, ModelPrediction,
    Signal, BacktestResult, Alert, SystemLog
)
from database.crud import (
    DatabaseManager, MarketDataCRUD, TradeCRUD,
    OrderCRUD, TradingSessionCRUD, SignalCRUD,
    ModelTrainingCRUD
)
from database.connection import DatabaseConnectionManager
from config.config_manager import ConfigManager


class TestDatabaseModels(unittest.TestCase):
    """Test cases for database models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database"""
        # Create a temporary directory for test database
        cls.test_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.test_dir, 'test_trading_ai.db')
        
        # Create a test config manager
        cls.config_manager = ConfigManager()
        
        # Override database config for testing
        cls.config_manager._config['database'] = {
            'type': 'sqlite',
            'database': cls.db_path,
            'echo': False
        }
        
        # Create database connection
        cls.db_manager = DatabaseConnectionManager(cls.config_manager)
        cls.engine = cls.db_manager.engine
        
        # Create tables
        Base.metadata.create_all(cls.engine)
        
        # Create session
        cls.Session = sessionmaker(bind=cls.engine)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test database"""
        cls.db_manager.close()
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up fresh session for each test"""
        self.session = self.Session()
        self.crud = DatabaseManager(self.session)
    
    def tearDown(self):
        """Clean up session after each test"""
        self.session.rollback()
        self.session.close()
    
    def test_market_data_model(self):
        """Test MarketData model"""
        # Create market data
        market_data = MarketData(
            symbol="BTC/USDT",
            timestamp=datetime.utcnow(),
            timeframe="1h",
            open=Decimal("50000.50"),
            high=Decimal("51000.75"),
            low=Decimal("49500.25"),
            close=Decimal("50500.00"),
            volume=Decimal("100.5"),
            quote_volume=Decimal("5050000.25"),
            trade_count=1500
        )
        
        self.session.add(market_data)
        self.session.commit()
        
        # Retrieve and verify
        retrieved = self.session.query(MarketData).filter_by(symbol="BTC/USDT").first()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.symbol, "BTC/USDT")
        self.assertEqual(retrieved.timeframe, "1h")
        self.assertEqual(float(retrieved.open), 50000.50)
        self.assertEqual(float(retrieved.close), 50500.00)
        self.assertEqual(retrieved.trade_count, 1500)
        
        # Test CRUD operations
        crud_result = self.crud.market_data.create(market_data)
        self.assertIsNotNone(crud_result)
        
        # Test bulk create
        market_data_list = []
        for i in range(5):
            md = MarketData(
                symbol="ETH/USDT",
                timestamp=datetime.utcnow() - timedelta(hours=i),
                timeframe="1h",
                open=Decimal(f"{3000 + i}.50"),
                high=Decimal(f"{3100 + i}.75"),
                low=Decimal(f"{2900 + i}.25"),
                close=Decimal(f"{3050 + i}.00"),
                volume=Decimal(f"{50 + i}.5")
            )
            market_data_list.append(md)
        
        success = self.crud.market_data.bulk_create(market_data_list)
        self.assertTrue(success)
    
    def test_trade_model(self):
        """Test Trade model"""
        # Create a trade
        trade = Trade(
            trade_id="test_trade_123",
            symbol="BTC/USDT",
            side="buy",
            price=Decimal("50000.00"),
            quantity=Decimal("0.5"),
            commission=Decimal("25.00"),
            commission_asset="USDT",
            timestamp=datetime.utcnow(),
            is_maker=False,
            exchange_order_id="order_123"
        )
        
        self.session.add(trade)
        self.session.commit()
        
        # Retrieve and verify
        retrieved = self.session.query(Trade).filter_by(trade_id="test_trade_123").first()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.symbol, "BTC/USDT")
        self.assertEqual(retrieved.side, "buy")
        self.assertEqual(float(retrieved.price), 50000.00)
        self.assertEqual(float(retrieved.quantity), 0.5)
        self.assertEqual(retrieved.commission_asset, "USDT")
        
        # Test CRUD operations
        crud_result = self.crud.trades.create(trade)
        self.assertIsNotNone(crud_result)
        
        # Test get by symbol
        trades = self.crud.trades.get_by_symbol("BTC/USDT")
        self.assertGreaterEqual(len(trades), 1)
    
    def test_order_model(self):
        """Test Order model"""
        # Create a trading session first
        session = TradingSession(
            session_id="test_session_1",
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=Decimal("10000.00"),
            current_capital=Decimal("10000.00"),
            status="active"
        )
        self.session.add(session)
        self.session.commit()
        
        # Create an order
        order = Order(
            order_id="test_order_123",
            client_order_id="client_123",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            status="pending",
            price=Decimal("50000.00"),
            quantity=Decimal("0.2"),
            executed_quantity=Decimal("0.0"),
            time_in_force="GTC",
            created_time=datetime.utcnow(),
            trading_session_id=session.id
        )
        
        self.session.add(order)
        self.session.commit()
        
        # Retrieve and verify
        retrieved = self.session.query(Order).filter_by(order_id="test_order_123").first()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.symbol, "BTC/USDT")
        self.assertEqual(retrieved.side, "buy")
        self.assertEqual(retrieved.order_type, "limit")
        self.assertEqual(retrieved.status, "pending")
        self.assertEqual(retrieved.trading_session_id, session.id)
        
        # Test CRUD operations
        crud_result = self.crud.orders.create(order)
        self.assertIsNotNone(crud_result)
        
        # Test update status
        success = self.crud.orders.update_status(
            order_id="test_order_123",
            status="filled",
            executed_quantity=0.2
        )
        self.assertTrue(success)
        
        # Verify update
        updated = self.session.query(Order).filter_by(order_id="test_order_123").first()
        self.assertEqual(updated.status, "filled")
        self.assertEqual(float(updated.executed_quantity), 0.2)
    
    def test_trading_session_model(self):
        """Test TradingSession model"""
        # Create trading session
        session = TradingSession(
            session_id="test_session_2",
            strategy_name="ml_strategy",
            symbol="ETH/USDT",
            timeframe="15m",
            initial_capital=Decimal("5000.00"),
            current_capital=Decimal("5200.50"),
            status="active",
            parameters={"param1": "value1", "param2": 123}
        )
        
        self.session.add(session)
        self.session.commit()
        
        # Retrieve and verify
        retrieved = self.session.query(TradingSession).filter_by(session_id="test_session_2").first()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.strategy_name, "ml_strategy")
        self.assertEqual(retrieved.symbol, "ETH/USDT")
        self.assertEqual(retrieved.timeframe, "15m")
        self.assertEqual(float(retrieved.initial_capital), 5000.00)
        self.assertEqual(float(retrieved.current_capital), 5200.50)
        self.assertEqual(retrieved.parameters["param1"], "value1")
        self.assertEqual(retrieved.parameters["param2"], 123)
        
        # Test CRUD operations
        crud_result = self.crud.trading_sessions.create(session)
        self.assertIsNotNone(crud_result)
        
        # Test get active sessions
        active_sessions = self.crud.trading_sessions.get_active_sessions()
        self.assertGreaterEqual(len(active_sessions), 1)
        
        # Test update capital
        success = self.crud.trading_sessions.update_capital(
            session_id=retrieved.id,
            new_capital=5300.00
        )
        self.assertTrue(success)
        
        # Test end session
        success = self.crud.trading_sessions.end_session(retrieved.id)
        self.assertTrue(success)
        
        ended = self.session.query(TradingSession).filter_by(id=retrieved.id).first()
        self.assertEqual(ended.status, "stopped")
    
    def test_signal_model(self):
        """Test Signal model"""
        # Create signal
        signal = Signal(
            signal_id="test_signal_123",
            symbol="BTC/USDT",
            timestamp=datetime.utcnow(),
            signal_type="buy",
            signal_strength=0.85,
            source="model",
            source_id="transformer_model_001",
            confidence=0.92,
            price_at_signal=Decimal("50500.50"),
            target_price=Decimal("52000.00"),
            stop_loss=Decimal("49500.00"),
            timeframe="1h",
            metadata={"model_version": "1.2.3", "prediction_time": "2024-01-01T12:00:00"}
        )
        
        self.session.add(signal)
        self.session.commit()
        
        # Retrieve and verify
        retrieved = self.session.query(Signal).filter_by(signal_id="test_signal_123").first()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.symbol, "BTC/USDT")
        self.assertEqual(retrieved.signal_type, "buy")
        self.assertEqual(retrieved.signal_strength, 0.85)
        self.assertEqual(retrieved.source, "model")
        self.assertEqual(retrieved.confidence, 0.92)
        self.assertEqual(float(retrieved.price_at_signal), 50500.50)
        self.assertEqual(retrieved.metadata["model_version"], "1.2.3")
        
        # Test CRUD operations
        crud_result = self.crud.signals.create(signal)
        self.assertIsNotNone(crud_result)
        
        # Test get recent signals
        signals = self.crud.signals.get_recent_signals(symbol="BTC/USDT", limit=10)
        self.assertGreaterEqual(len(signals), 1)
    
    def test_model_training_model(self):
        """Test ModelTraining model"""
        # Create model training record
        training = ModelTraining(
            training_id="test_training_123",
            model_name="transformer_model",
            model_type="transformer",
            symbol="BTC/USDT",
            timeframe="1h",
            training_start=datetime.utcnow() - timedelta(hours=1),
            training_end=datetime.utcnow(),
            status="completed",
            hyperparameters={
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            },
            training_metrics={
                "train_loss": [0.5, 0.3, 0.1],
                "train_accuracy": [0.6, 0.8, 0.95]
            },
            validation_metrics={
                "val_loss": 0.12,
                "val_accuracy": 0.88
            },
            test_metrics={
                "test_loss": 0.15,
                "test_accuracy": 0.85
            },
            model_path="/models/transformer_model_001.pt",
            feature_columns=["close", "volume", "rsi", "macd"]
        )
        
        self.session.add(training)
        self.session.commit()
        
        # Retrieve and verify
        retrieved = self.session.query(ModelTraining).filter_by(training_id="test_training_123").first()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.model_name, "transformer_model")
        self.assertEqual(retrieved.model_type, "transformer")
        self.assertEqual(retrieved.symbol, "BTC/USDT")
        self.assertEqual(retrieved.status, "completed")
        self.assertEqual(retrieved.hyperparameters["learning_rate"], 0.001)
        self.assertEqual(retrieved.validation_metrics["val_accuracy"], 0.88)
        self.assertEqual(len(retrieved.feature_columns), 4)
        
        # Test CRUD operations
        crud_result = self.crud.model_trainings.create(training)
        self.assertIsNotNone(crud_result)
        
        # Test update training results
        success = self.crud.model_trainings.update_training_results(
            training_id="test_training_123",
            status="completed",
            model_path="/new/path/model.pt"
        )
        self.assertTrue(success)
        
        # Test get latest training
        latest = self.crud.model_trainings.get_latest_training(
            model_name="transformer_model",
            symbol="BTC/USDT"
        )
        self.assertIsNotNone(latest)
    
    def test_performance_metrics_model(self):
        """Test PerformanceMetrics model"""
        # Create trading session first
        session = TradingSession(
            session_id="test_session_3",
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=Decimal("10000.00"),
            current_capital=Decimal("11000.00"),
            status="active"
        )
        self.session.add(session)
        self.session.commit()
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            trading_session_id=session.id,
            timestamp=datetime.utcnow(),
            total_pnl=Decimal("1000.00"),
            realized_pnl=Decimal("800.00"),
            unrealized_pnl=Decimal("200.00"),
            total_fees=Decimal("50.00"),
            win_rate=0.65,
            profit_factor=1.5,
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            total_trades=100,
            winning_trades=65,
            losing_trades=35
        )
        
        self.session.add(metrics)
        self.session.commit()
        
        # Retrieve and verify
        retrieved = self.session.query(PerformanceMetrics).filter_by(
            trading_session_id=session.id
        ).first()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.trading_session_id, session.id)
        self.assertEqual(float(retrieved.total_pnl), 1000.00)
        self.assertEqual(retrieved.win_rate, 0.65)
        self.assertEqual(retrieved.profit_factor, 1.5)
        self.assertEqual(retrieved.total_trades, 100)
        self.assertEqual(retrieved.winning_trades, 65)
    
    def test_backtest_result_model(self):
        """Test BacktestResult model"""
        # Create backtest result
        backtest = BacktestResult(
            backtest_id="test_backtest_123",
            strategy_name="moving_average_crossover",
            symbol="BTC/USDT",
            timeframe="1d",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=Decimal("10000.00"),
            final_capital=Decimal("12500.50"),
            total_return=0.25,
            annual_return=0.28,
            sharpe_ratio=1.8,
            max_drawdown=0.12,
            win_rate=0.62,
            profit_factor=1.6,
            total_trades=150,
            avg_trade=16.67,
            parameters={
                "fast_ma": 20,
                "slow_ma": 50,
                "stop_loss": 0.02
            },
            trades=[
                {"timestamp": "2023-01-15", "side": "buy", "price": 20000, "pnl": 500},
                {"timestamp": "2023-01-20", "side": "sell", "price": 20500, "pnl": 500}
            ],
            equity_curve={
                "dates": ["2023-01-01", "2023-12-31"],
                "equity": [10000, 12500.5]
            }
        )
        
        self.session.add(backtest)
        self.session.commit()
        
        # Retrieve and verify
        retrieved = self.session.query(BacktestResult).filter_by(
            backtest_id="test_backtest_123"
        ).first()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.strategy_name, "moving_average_crossover")
        self.assertEqual(retrieved.total_return, 0.25)
        self.assertEqual(retrieved.win_rate, 0.62)
        self.assertEqual(retrieved.parameters["fast_ma"], 20)
        self.assertEqual(len(retrieved.trades), 2)
    
    def test_alert_model(self):
        """Test Alert model"""
        # Create alert
        alert = Alert(
            alert_id="test_alert_123",
            alert_type="price",
            severity="warning",
            symbol="BTC/USDT",
            message="Price dropped by 5% in 1 hour",
            data={
                "price_change": -0.05,
                "timeframe": "1h",
                "current_price": 49500,
                "previous_price": 52105
            },
            is_read=False,
            is_resolved=False
        )
        
        self.session.add(alert)
        self.session.commit()
        
        # Retrieve and verify
        retrieved = self.session.query(Alert).filter_by(alert_id="test_alert_123").first()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.alert_type, "price")
        self.assertEqual(retrieved.severity, "warning")
        self.assertEqual(retrieved.symbol, "BTC/USDT")
        self.assertFalse(retrieved.is_read)
        self.assertFalse(retrieved.is_resolved)
        self.assertEqual(retrieved.data["price_change"], -0.05)
    
    def test_system_log_model(self):
        """Test SystemLog model"""
        # Create system log
        log = SystemLog(
            level="INFO",
            logger="trading_engine",
            message="Trading session started successfully",
            module="core.trading",
            function="start_session",
            line_number=123,
            extra_data={"session_id": "test_session", "symbol": "BTC/USDT"}
        )
        
        self.session.add(log)
        self.session.commit()
        
        # Retrieve and verify
        retrieved = self.session.query(SystemLog).filter_by(
            logger="trading_engine"
        ).first()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.level, "INFO")
        self.assertEqual(retrieved.message, "Trading session started successfully")
        self.assertEqual(retrieved.module, "core.trading")
        self.assertEqual(retrieved.extra_data["session_id"], "test_session")
    
    def test_enum_values(self):
        """Test enum values"""
        # Test TradeSide enum
        self.assertEqual(TradeSide.BUY.value, "buy")
        self.assertEqual(TradeSide.SELL.value, "sell")
        
        # Test OrderStatus enum
        self.assertEqual(OrderStatus.PENDING.value, "pending")
        self.assertEqual(OrderStatus.FILLED.value, "filled")
        self.assertEqual(OrderStatus.CANCELLED.value, "cancelled")
        
        # Test Timeframe enum
        self.assertEqual(Timeframe.ONE_MINUTE.value, "1m")
        self.assertEqual(Timeframe.ONE_HOUR.value, "1h")
        self.assertEqual(Timeframe.ONE_DAY.value, "1d")
    
    def test_database_stats(self):
        """Test database statistics"""
        stats = self.crud.get_db_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('market_data', stats)
        self.assertIn('trades', stats)
        self.assertIn('orders', stats)
        self.assertIn('last_updated', stats)
        
        # All counts should be integers
        for key, value in stats.items():
            if key != 'last_updated':
                self.assertIsInstance(value, int)
    
    def test_relationship_integrity(self):
        """Test model relationships"""
        # Create trading session
        session = TradingSession(
            session_id="rel_test_session",
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=Decimal("10000.00"),
            current_capital=Decimal("10000.00"),
            status="active"
        )
        self.session.add(session)
        self.session.commit()
        
        # Create order with relationship
        order = Order(
            order_id="rel_test_order",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            status="pending",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            created_time=datetime.utcnow(),
            trading_session_id=session.id
        )
        self.session.add(order)
        self.session.commit()
        
        # Create trade with relationship to order
        trade = Trade(
            trade_id="rel_test_trade",
            symbol="BTC/USDT",
            side="buy",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            timestamp=datetime.utcnow(),
            exchange_order_id="rel_test_order",
            order_id=order.id
        )
        self.session.add(trade)
        self.session.commit()
        
        # Test relationships
        session_orders = self.session.query(Order).filter_by(
            trading_session_id=session.id
        ).all()
        self.assertEqual(len(session_orders), 1)
        
        order_trades = self.session.query(Trade).filter_by(
            order_id=order.id
        ).all()
        self.assertEqual(len(order_trades), 1)
        
        # Test that trade is linked to order
        self.assertEqual(trade.order_id, order.id)
        self.assertEqual(order.trades[0].id, trade.id)
    
    def test_cleanup_functionality(self):
        """Test cleanup functionality"""
        # Create old data
        old_date = datetime.utcnow() - timedelta(days=100)
        
        old_market_data = MarketData(
            symbol="BTC/USDT",
            timestamp=old_date,
            timeframe="1h",
            open=Decimal("40000.00"),
            high=Decimal("41000.00"),
            low=Decimal("39000.00"),
            close=Decimal("40500.00"),
            volume=Decimal("100.0")
        )
        
        self.session.add(old_market_data)
        self.session.commit()
        
        # Note: The actual cleanup would be tested in integration tests
        # as it requires a specific cleanup method to be called
        # This just verifies the data was created
        
        old_data = self.session.query(MarketData).filter(
            MarketData.timestamp < old_date + timedelta(days=1)
        ).all()
        
        self.assertGreaterEqual(len(old_data), 1)
    
    def test_error_handling(self):
        """Test error handling in CRUD operations"""
        # Test invalid data
        invalid_trade = Trade(
            trade_id=None,  # This should fail as trade_id is required
            symbol="BTC/USDT",
            side="buy",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            timestamp=datetime.utcnow()
        )
        
        # This should raise an error or return None
        result = self.crud.trades.create(invalid_trade)
        self.assertTrue(result is None or self.session.rollback())
        
        # Test getting non-existent data
        non_existent = self.crud.trades.get_by_trade_id("non_existent_id")
        self.assertIsNone(non_existent)
    
    def test_bulk_operations(self):
        """Test bulk CRUD operations"""
        # Create multiple market data entries
        market_data_list = []
        for i in range(10):
            md = MarketData(
                symbol="LTC/USDT",
                timestamp=datetime.utcnow() - timedelta(hours=i),
                timeframe="1h",
                open=Decimal(f"{100 + i}.00"),
                high=Decimal(f"{105 + i}.00"),
                low=Decimal(f"{95 + i}.00"),
                close=Decimal(f"{102 + i}.00"),
                volume=Decimal(f"{50 + i}.0")
            )
            market_data_list.append(md)
        
        # Test bulk create
        success = self.crud.market_data.bulk_create(market_data_list)
        self.assertTrue(success)
        
        # Verify bulk creation
        count = self.session.query(MarketData).filter_by(symbol="LTC/USDT").count()
        self.assertGreaterEqual(count, 10)


class TestConnectionManager(unittest.TestCase):
    """Test cases for DatabaseConnectionManager"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.test_dir, 'test_connection.db')
        
        # Create test config
        cls.config_manager = ConfigManager()
        cls.config_manager._config['database'] = {
            'type': 'sqlite',
            'database': cls.db_path,
            'echo': False
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_connection_creation(self):
        """Test database connection creation"""
        db_manager = DatabaseConnectionManager(self.config_manager)
        
        self.assertIsNotNone(db_manager.engine)
        self.assertIsNotNone(db_manager.session_factory)
        self.assertIsNotNone(db_manager.ScopedSession)
        
        # Test health check
        health = db_manager.health_check()
        self.assertTrue(health['database_connected'])
        self.assertEqual(health['database_type'], 'sqlite')
        
        db_manager.close()
    
    def test_session_management(self):
        """Test session management"""
        db_manager = DatabaseConnectionManager(self.config_manager)
        
        # Test regular session
        with db_manager.session_scope() as session:
            self.assertIsNotNone(session)
            # Test that we can execute a query
            result = session.execute("SELECT 1").scalar()
            self.assertEqual(result, 1)
        
        # Test scoped session
        with db_manager.scoped_session_scope() as session:
            self.assertIsNotNone(session)
            result = session.execute("SELECT 1").scalar()
            self.assertEqual(result, 1)
        
        db_manager.close()
    
    def test_table_creation(self):
        """Test table creation"""
        db_manager = DatabaseConnectionManager(self.config_manager)
        
        # Create tables
        success = db_manager.create_tables()
        self.assertTrue(success)
        
        # Verify tables exist
        with db_manager.engine.connect() as conn:
            # SQLite specific query to check tables
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            
            table_names = [t[0] for t in tables]
            self.assertIn('market_data', table_names)
            self.assertIn('trades', table_names)
            self.assertIn('orders', table_names)
        
        db_manager.close()
    
    def test_optimization(self):
        """Test database optimization"""
        db_manager = DatabaseConnectionManager(self.config_manager)
        
        # Create tables first
        db_manager.create_tables()
        
        # Test optimization (should not fail)
        success = db_manager.optimize_database()
        self.assertTrue(success)
        
        db_manager.close()


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)