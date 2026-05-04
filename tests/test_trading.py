"""
Test file for trading components in the Bitcoin Trading AI application.
Unit tests for trading logic, signal generation, position sizing, and order management.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.trading.signal_generator import SignalGenerator
from core.trading.position_sizer import PositionSizer
from core.trading.order_manager import OrderManager
from core.trading.execution_engine import ExecutionEngine
from core.risk_management.risk_analyzer import RiskAnalyzer
from core.risk_management.stop_loss_manager import StopLossManager
from database.models import TradeSide, OrderStatus
from config.config_manager import ConfigManager


class TestSignalGenerator(unittest.TestCase):
    """Test cases for SignalGenerator"""
    
    def setUp(self):
        """Set up test environment"""
        self.config_manager = ConfigManager()
        self.signal_generator = SignalGenerator(self.config_manager)
        
        # Mock data for testing
        self.sample_data = {
            'close': [50000, 50500, 50300, 50800, 51000, 50900, 51200, 51500],
            'volume': [100, 150, 120, 180, 200, 190, 210, 220],
            'rsi': [45, 50, 55, 60, 65, 70, 75, 80],
            'macd': [-50, -30, -10, 10, 30, 50, 70, 90],
            'macd_signal': [-60, -40, -20, 0, 20, 40, 60, 80],
            'bb_upper': [51000, 51200, 51400, 51600, 51800, 52000, 52200, 52400],
            'bb_lower': [49000, 49200, 49400, 49600, 49800, 50000, 50200, 50400],
            'sma_20': [49800, 49900, 50000, 50100, 50200, 50300, 50400, 50500],
            'sma_50': [49500, 49600, 49700, 49800, 49900, 50000, 50100, 50200]
        }
    
    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        # Test RSI calculation
        closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42]
        rsi = self.signal_generator._calculate_rsi(closes)
        self.assertIsInstance(rsi, list)
        self.assertEqual(len(rsi), len(closes))
        
        # Test MACD calculation
        macd_line, signal_line, histogram = self.signal_generator._calculate_macd(closes)
        self.assertIsInstance(macd_line, list)
        self.assertIsInstance(signal_line, list)
        self.assertIsInstance(histogram, list)
        
        # Test Bollinger Bands
        upper, middle, lower = self.signal_generator._calculate_bollinger_bands(closes)
        self.assertIsInstance(upper, list)
        self.assertIsInstance(middle, list)
        self.assertIsInstance(lower, list)
        
        # Test moving averages
        sma = self.signal_generator._calculate_sma(closes, period=10)
        ema = self.signal_generator._calculate_ema(closes, period=10)
        self.assertIsInstance(sma, list)
        self.assertIsInstance(ema, list)
    
    def test_moving_average_crossover(self):
        """Test moving average crossover signal"""
        # Bullish crossover (fast MA crosses above slow MA)
        fast_ma = [100, 102, 104, 106, 108]
        slow_ma = [105, 104, 103, 102, 101]
        signal = self.signal_generator._check_ma_crossover(fast_ma, slow_ma)
        self.assertEqual(signal, 'buy')
        
        # Bearish crossover (fast MA crosses below slow MA)
        fast_ma = [105, 104, 103, 102, 101]
        slow_ma = [100, 101, 102, 103, 104]
        signal = self.signal_generator._check_ma_crossover(fast_ma, slow_ma)
        self.assertEqual(signal, 'sell')
        
        # No crossover
        fast_ma = [100, 101, 102, 103, 104]
        slow_ma = [95, 96, 97, 98, 99]
        signal = self.signal_generator._check_ma_crossover(fast_ma, slow_ma)
        self.assertEqual(signal, 'hold')
    
    def test_rsi_signal(self):
        """Test RSI-based signals"""
        # Oversold condition
        rsi_values = [25, 28, 30, 32, 35]
        signal = self.signal_generator._check_rsi_signal(rsi_values)
        self.assertEqual(signal, 'buy')
        
        # Overbought condition
        rsi_values = [75, 78, 80, 82, 85]
        signal = self.signal_generator._check_rsi_signal(rsi_values)
        self.assertEqual(signal, 'sell')
        
        # Neutral condition
        rsi_values = [45, 50, 55, 50, 45]
        signal = self.signal_generator._check_rsi_signal(rsi_values)
        self.assertEqual(signal, 'hold')
    
    def test_macd_signal(self):
        """Test MACD-based signals"""
        # Bullish signal (MACD crosses above signal line)
        macd_line = [10, 20, 30, 40, 50]
        signal_line = [15, 25, 35, 45, 40]  # MACD crosses above at last point
        histogram = [-5, -5, -5, -5, 10]
        
        signal = self.signal_generator._check_macd_signal(macd_line, signal_line, histogram)
        self.assertEqual(signal, 'buy')
        
        # Bearish signal (MACD crosses below signal line)
        macd_line = [50, 45, 40, 35, 30]
        signal_line = [40, 38, 36, 34, 32]  # MACD still above but converging
        histogram = [10, 7, 4, 1, -2]
        
        signal = self.signal_generator._check_macd_signal(macd_line, signal_line, histogram)
        self.assertEqual(signal, 'sell')
        
        # Neutral signal
        macd_line = [10, 11, 12, 13, 14]
        signal_line = [8, 9, 10, 11, 12]  # MACD consistently above
        histogram = [2, 2, 2, 2, 2]
        
        signal = self.signal_generator._check_macd_signal(macd_line, signal_line, histogram)
        self.assertEqual(signal, 'hold')
    
    def test_bollinger_bands_signal(self):
        """Test Bollinger Bands signals"""
        # Price near lower band (buy signal)
        price = 49000
        upper_band = 52000
        middle_band = 50500
        lower_band = 49000
        
        signal = self.signal_generator._check_bollinger_bands_signal(
            price, upper_band, middle_band, lower_band
        )
        self.assertEqual(signal, 'buy')
        
        # Price near upper band (sell signal)
        price = 52000
        upper_band = 52000
        middle_band = 50500
        lower_band = 49000
        
        signal = self.signal_generator._check_bollinger_bands_signal(
            price, upper_band, middle_band, lower_band
        )
        self.assertEqual(signal, 'sell')
        
        # Price in middle (hold signal)
        price = 50500
        upper_band = 52000
        middle_band = 50500
        lower_band = 49000
        
        signal = self.signal_generator._check_bollinger_bands_signal(
            price, upper_band, middle_band, lower_band
        )
        self.assertEqual(signal, 'hold')
    
    def test_generate_signal(self):
        """Test complete signal generation"""
        # Mock market data
        market_data = {
            'timestamp': datetime.utcnow(),
            'open': 50000,
            'high': 51000,
            'low': 49500,
            'close': 50500,
            'volume': 1000
        }
        
        # Mock historical data
        historical_data = {
            'close': [48000, 48500, 49000, 49500, 50000, 50500],
            'volume': [800, 850, 900, 950, 1000, 1050]
        }
        
        # Generate signal
        signal = self.signal_generator.generate_signal(
            symbol="BTC/USDT",
            market_data=market_data,
            historical_data=historical_data,
            strategy="moving_average"
        )
        
        # Verify signal structure
        self.assertIsInstance(signal, dict)
        self.assertIn('signal', signal)
        self.assertIn('strength', signal)
        self.assertIn('confidence', signal)
        self.assertIn('timestamp', signal)
        self.assertIn('symbol', signal)
        
        # Valid signal values
        valid_signals = ['buy', 'sell', 'hold', 'strong_buy', 'strong_sell']
        self.assertIn(signal['signal'], valid_signals)
        
        # Valid strength range
        self.assertGreaterEqual(signal['strength'], 0.0)
        self.assertLessEqual(signal['strength'], 1.0)
        
        # Valid confidence range
        self.assertGreaterEqual(signal['confidence'], 0.0)
        self.assertLessEqual(signal['confidence'], 1.0)
    
    def test_multiple_strategies(self):
        """Test different trading strategies"""
        market_data = {
            'timestamp': datetime.utcnow(),
            'close': 50500,
            'volume': 1000
        }
        
        historical_data = {
            'close': list(range(49000, 51000, 200)),
            'volume': list(range(800, 1200, 80))
        }
        
        strategies = ['moving_average', 'rsi', 'macd', 'bollinger_bands', 'volume']
        
        for strategy in strategies:
            signal = self.signal_generator.generate_signal(
                symbol="BTC/USDT",
                market_data=market_data,
                historical_data=historical_data,
                strategy=strategy
            )
            
            self.assertIsInstance(signal, dict)
            self.assertIn('signal', signal)
            self.assertIn('strategy', signal)
            self.assertEqual(signal['strategy'], strategy)
    
    @patch.object(SignalGenerator, '_calculate_technical_indicators')
    def test_with_mock_indicators(self, mock_indicators):
        """Test signal generation with mocked indicators"""
        # Setup mock
        mock_indicators.return_value = {
            'rsi': [70, 72, 75, 78, 80],  # Overbought
            'macd': [50, 55, 60, 65, 70],
            'macd_signal': [45, 50, 55, 60, 65],
            'sma_20': [50000, 50200, 50400, 50600, 50800],
            'sma_50': [49500, 49600, 49700, 49800, 49900]
        }
        
        market_data = {'close': 51000, 'timestamp': datetime.utcnow()}
        historical_data = {'close': list(range(50000, 51000, 200))}
        
        signal = self.signal_generator.generate_signal(
            symbol="BTC/USDT",
            market_data=market_data,
            historical_data=historical_data,
            strategy="moving_average"
        )
        
        self.assertIsInstance(signal, dict)
        mock_indicators.assert_called_once()


class TestPositionSizer(unittest.TestCase):
    """Test cases for PositionSizer"""
    
    def setUp(self):
        """Set up test environment"""
        self.config_manager = ConfigManager()
        self.position_sizer = PositionSizer(self.config_manager)
        
        # Test parameters
        self.account_balance = 10000.0
        self.current_price = 50000.0
        self.signal_strength = 0.75
        self.risk_per_trade = 0.02  # 2% risk per trade
    
    def test_fixed_fractional_sizing(self):
        """Test fixed fractional position sizing"""
        # Test with default risk percentage
        position_size = self.position_sizer.calculate_position_size(
            account_balance=self.account_balance,
            current_price=self.current_price,
            signal_strength=self.signal_strength,
            sizing_method='fixed_fractional'
        )
        
        self.assertIsInstance(position_size, float)
        self.assertGreater(position_size, 0.0)
        
        # Calculate expected size: (balance * risk% * signal_strength) / price
        expected_size = (self.account_balance * 0.02 * self.signal_strength) / self.current_price
        self.assertAlmostEqual(position_size, expected_size, places=8)
        
        # Test with custom risk percentage
        custom_risk = 0.01  # 1%
        position_size = self.position_sizer.calculate_position_size(
            account_balance=self.account_balance,
            current_price=self.current_price,
            signal_strength=self.signal_strength,
            sizing_method='fixed_fractional',
            risk_percentage=custom_risk
        )
        
        expected_size = (self.account_balance * custom_risk * self.signal_strength) / self.current_price
        self.assertAlmostEqual(position_size, expected_size, places=8)
    
    def test_kelly_criterion(self):
        """Test Kelly Criterion position sizing"""
        # Test with win rate and win/loss ratio
        win_rate = 0.6
        win_loss_ratio = 1.5
        
        position_size = self.position_sizer.calculate_position_size(
            account_balance=self.account_balance,
            current_price=self.current_price,
            signal_strength=self.signal_strength,
            sizing_method='kelly',
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio
        )
        
        self.assertIsInstance(position_size, float)
        self.assertGreaterEqual(position_size, 0.0)
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = 1 - p, b = win/loss ratio
        p = win_rate
        b = win_loss_ratio
        q = 1 - p
        kelly_fraction = (p * b - q) / b
        
        # Apply signal strength and calculate position size
        expected_size = (self.account_balance * kelly_fraction * self.signal_strength) / self.current_price
        
        # Kelly can be negative if edge is negative - position sizer should handle this
        if kelly_fraction > 0:
            self.assertAlmostEqual(position_size, expected_size, places=8)
    
    def test_volatility_based_sizing(self):
        """Test volatility-based position sizing"""
        # Mock volatility data
        volatility = 0.02  # 2% daily volatility
        atr = 1000.0  # Average True Range
        
        position_size = self.position_sizer.calculate_position_size(
            account_balance=self.account_balance,
            current_price=self.current_price,
            signal_strength=self.signal_strength,
            sizing_method='volatility',
            volatility=volatility,
            atr=atr
        )
        
        self.assertIsInstance(position_size, float)
        
        # Volatility-based sizing should consider volatility in calculation
        # Specific formula depends on implementation
        self.assertGreater(position_size, 0.0)
    
    def test_adaptive_sizing(self):
        """Test adaptive position sizing"""
        # Mock market conditions
        market_volatility = 0.015  # 1.5% volatility
        trend_strength = 0.8
        market_regime = 'trending'
        
        position_size = self.position_sizer.calculate_position_size(
            account_balance=self.account_balance,
            current_price=self.current_price,
            signal_strength=self.signal_strength,
            sizing_method='adaptive',
            market_volatility=market_volatility,
            trend_strength=trend_strength,
            market_regime=market_regime
        )
        
        self.assertIsInstance(position_size, float)
        self.assertGreater(position_size, 0.0)
        
        # In trending market with low volatility, position size might be larger
        # This is a qualitative check
        if market_volatility < 0.02 and trend_strength > 0.7:
            # Size should be reasonable but not excessive
            self.assertLess(position_size * self.current_price, self.account_balance * 0.1)
    
    def test_min_max_constraints(self):
        """Test position size constraints"""
        # Test minimum position size
        min_position = 0.001  # Minimum BTC position
        
        # Very small account balance
        small_balance = 100.0
        position_size = self.position_sizer.calculate_position_size(
            account_balance=small_balance,
            current_price=self.current_price,
            signal_strength=1.0,
            sizing_method='fixed_fractional'
        )
        
        # Should be at least minimum position size
        if position_size * self.current_price < small_balance * 0.01:  # If too small
            self.assertEqual(position_size, min_position)
        
        # Test maximum position size
        max_position_pct = 0.1  # Maximum 10% of account per trade
        
        # Very strong signal
        position_size = self.position_sizer.calculate_position_size(
            account_balance=self.account_balance,
            current_price=self.current_price,
            signal_strength=1.0,
            sizing_method='fixed_fractional',
            risk_percentage=0.5  # High risk - should be capped
        )
        
        max_size = (self.account_balance * max_position_pct) / self.current_price
        self.assertLessEqual(position_size, max_size)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Zero or negative account balance
        with self.assertRaises(ValueError):
            self.position_sizer.calculate_position_size(
                account_balance=0.0,
                current_price=self.current_price,
                signal_strength=self.signal_strength
            )
        
        # Zero or negative price
        with self.assertRaises(ValueError):
            self.position_sizer.calculate_position_size(
                account_balance=self.account_balance,
                current_price=0.0,
                signal_strength=self.signal_strength
            )
        
        # Signal strength out of range
        with self.assertRaises(ValueError):
            self.position_sizer.calculate_position_size(
                account_balance=self.account_balance,
                current_price=self.current_price,
                signal_strength=1.5  # > 1.0
            )
    
    def test_position_value_calculation(self):
        """Test position value calculation"""
        position_size = 0.5  # BTC
        position_value = self.position_sizer.calculate_position_value(
            position_size=position_size,
            current_price=self.current_price
        )
        
        expected_value = position_size * self.current_price
        self.assertEqual(position_value, expected_value)
        
        # Test with commission
        commission_rate = 0.001  # 0.1%
        position_value_with_commission = self.position_sizer.calculate_position_value(
            position_size=position_size,
            current_price=self.current_price,
            include_commission=True,
            commission_rate=commission_rate
        )
        
        commission = position_value * commission_rate
        expected_value_with_commission = position_value + commission
        self.assertEqual(position_value_with_commission, expected_value_with_commission)


class TestOrderManager(unittest.TestCase):
    """Test cases for OrderManager"""
    
    def setUp(self):
        """Set up test environment"""
        self.config_manager = ConfigManager()
        self.order_manager = OrderManager(self.config_manager)
        
        # Mock exchange client
        self.mock_exchange = Mock()
        self.order_manager.exchange_client = self.mock_exchange
        
        # Test order parameters
        self.symbol = "BTC/USDT"
        self.side = "buy"
        self.order_type = "limit"
        self.quantity = 0.5
        self.price = 50000.0
        
    def test_create_order(self):
        """Test order creation"""
        # Mock exchange response
        mock_order_response = {
            'id': 'test_order_123',
            'symbol': self.symbol,
            'side': self.side,
            'type': self.order_type,
            'price': str(self.price),
            'amount': str(self.quantity),
            'status': 'open'
        }
        
        self.mock_exchange.create_order.return_value = mock_order_response
        
        # Create order
        order_result = self.order_manager.create_order(
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            quantity=self.quantity,
            price=self.price
        )
        
        # Verify exchange was called
        self.mock_exchange.create_order.assert_called_once_with(
            symbol=self.symbol,
            type=self.order_type,
            side=self.side,
            amount=self.quantity,
            price=self.price
        )
        
        # Verify order result
        self.assertIsInstance(order_result, dict)
        self.assertEqual(order_result['order_id'], 'test_order_123')
        self.assertEqual(order_result['status'], 'open')
        
    def test_market_order(self):
        """Test market order creation"""
        # Mock market order response (no price needed)
        mock_response = {
            'id': 'market_order_123',
            'symbol': self.symbol,
            'side': self.side,
            'type': 'market',
            'amount': str(self.quantity),
            'status': 'filled'
        }
        
        self.mock_exchange.create_order.return_value = mock_response
        
        # Create market order
        order_result = self.order_manager.create_order(
            symbol=self.symbol,
            side=self.side,
            order_type='market',
            quantity=self.quantity
            # No price for market orders
        )
        
        # Verify exchange was called without price
        self.mock_exchange.create_order.assert_called_once_with(
            symbol=self.symbol,
            type='market',
            side=self.side,
            amount=self.quantity
        )
        
        self.assertEqual(order_result['type'], 'market')
        
    def test_stop_loss_order(self):
        """Test stop loss order creation"""
        stop_price = 49000.0
        
        # Mock stop loss order response
        mock_response = {
            'id': 'stop_order_123',
            'symbol': self.symbol,
            'side': 'sell',
            'type': 'stop_loss',
            'amount': str(self.quantity),
            'price': str(self.price),
            'stopPrice': str(stop_price),
            'status': 'open'
        }
        
        self.mock_exchange.create_order.return_value = mock_response
        
        # Create stop loss order
        order_result = self.order_manager.create_stop_loss_order(
            symbol=self.symbol,
            quantity=self.quantity,
            stop_price=stop_price,
            limit_price=self.price
        )
        
        # Verify exchange was called with stop parameters
        self.mock_exchange.create_order.assert_called_once_with(
            symbol=self.symbol,
            type='stop_loss',
            side='sell',
            amount=self.quantity,
            price=self.price,
            stopPrice=stop_price
        )
        
        self.assertEqual(order_result['type'], 'stop_loss')
        
    def test_take_profit_order(self):
        """Test take profit order creation"""
        take_profit_price = 52000.0
        
        # Mock take profit order response
        mock_response = {
            'id': 'tp_order_123',
            'symbol': self.symbol,
            'side': 'sell',
            'type': 'take_profit',
            'amount': str(self.quantity),
            'price': str(take_profit_price),
            'status': 'open'
        }
        
        self.mock_exchange.create_order.return_value = mock_response
        
        # Create take profit order
        order_result = self.order_manager.create_take_profit_order(
            symbol=self.symbol,
            quantity=self.quantity,
            take_profit_price=take_profit_price
        )
        
        # Verify exchange was called
        self.mock_exchange.create_order.assert_called_once_with(
            symbol=self.symbol,
            type='take_profit',
            side='sell',
            amount=self.quantity,
            price=take_profit_price
        )
        
        self.assertEqual(order_result['type'], 'take_profit')
        
    def test_get_order_status(self):
        """Test getting order status"""
        order_id = 'test_order_123'
        
        # Mock order status response
        mock_response = {
            'id': order_id,
            'symbol': self.symbol,
            'side': self.side,
            'type': self.order_type,
            'price': str(self.price),
            'amount': str(self.quantity),
            'filled': str(self.quantity * 0.5),  # 50% filled
            'remaining': str(self.quantity * 0.5),
            'status': 'partially_filled'
        }
        
        self.mock_exchange.fetch_order.return_value = mock_response
        
        # Get order status
        order_status = self.order_manager.get_order_status(
            symbol=self.symbol,
            order_id=order_id
        )
        
        # Verify exchange was called
        self.mock_exchange.fetch_order.assert_called_once_with(
            id=order_id,
            symbol=self.symbol
        )
        
        # Verify status
        self.assertEqual(order_status['order_id'], order_id)
        self.assertEqual(order_status['status'], 'partially_filled')
        self.assertEqual(float(order_status['filled']), self.quantity * 0.5)
        self.assertEqual(float(order_status['remaining']), self.quantity * 0.5)
        
    def test_cancel_order(self):
        """Test order cancellation"""
        order_id = 'test_order_123'
        
        # Mock cancel response
        mock_response = {
            'id': order_id,
            'status': 'canceled'
        }
        
        self.mock_exchange.cancel_order.return_value = mock_response
        
        # Cancel order
        cancel_result = self.order_manager.cancel_order(
            symbol=self.symbol,
            order_id=order_id
        )
        
        # Verify exchange was called
        self.mock_exchange.cancel_order.assert_called_once_with(
            id=order_id,
            symbol=self.symbol
        )
        
        self.assertTrue(cancel_result)
        
    def test_get_open_orders(self):
        """Test getting open orders"""
        # Mock open orders response
        mock_orders = [
            {
                'id': 'order_1',
                'symbol': self.symbol,
                'side': 'buy',
                'type': 'limit',
                'price': '49500.0',
                'amount': '0.1',
                'filled': '0.0',
                'remaining': '0.1',
                'status': 'open'
            },
            {
                'id': 'order_2',
                'symbol': self.symbol,
                'side': 'sell',
                'type': 'limit',
                'price': '50500.0',
                'amount': '0.2',
                'filled': '0.1',
                'remaining': '0.1',
                'status': 'open'
            }
        ]
        
        self.mock_exchange.fetch_open_orders.return_value = mock_orders
        
        # Get open orders
        open_orders = self.order_manager.get_open_orders(symbol=self.symbol)
        
        # Verify exchange was called
        self.mock_exchange.fetch_open_orders.assert_called_once_with(symbol=self.symbol)
        
        # Verify orders
        self.assertEqual(len(open_orders), 2)
        self.assertEqual(open_orders[0]['order_id'], 'order_1')
        self.assertEqual(open_orders[1]['order_id'], 'order_2')
        
    def test_order_validation(self):
        """Test order parameter validation"""
        # Invalid symbol
        with self.assertRaises(ValueError):
            self.order_manager.create_order(
                symbol="INVALID",
                side=self.side,
                order_type=self.order_type,
                quantity=self.quantity,
                price=self.price
            )
        
        # Invalid side
        with self.assertRaises(ValueError):
            self.order_manager.create_order(
                symbol=self.symbol,
                side="invalid_side",
                order_type=self.order_type,
                quantity=self.quantity,
                price=self.price
            )
        
        # Invalid order type
        with self.assertRaises(ValueError):
            self.order_manager.create_order(
                symbol=self.symbol,
                side=self.side,
                order_type="invalid_type",
                quantity=self.quantity,
                price=self.price
            )
        
        # Invalid quantity
        with self.assertRaises(ValueError):
            self.order_manager.create_order(
                symbol=self.symbol,
                side=self.side,
                order_type=self.order_type,
                quantity=0.0,
                price=self.price
            )
        
        # Invalid price for limit order
        with self.assertRaises(ValueError):
            self.order_manager.create_order(
                symbol=self.symbol,
                side=self.side,
                order_type="limit",
                quantity=self.quantity,
                price=0.0
            )
        
    def test_order_limit_calculation(self):
        """Test order limit calculations"""
        # Test calculating order value
        order_value = self.order_manager.calculate_order_value(
            quantity=self.quantity,
            price=self.price
        )
        
        expected_value = self.quantity * self.price
        self.assertEqual(order_value, expected_value)
        
        # Test with commission
        commission_rate = 0.001
        order_value_with_commission = self.order_manager.calculate_order_value(
            quantity=self.quantity,
            price=self.price,
            commission_rate=commission_rate
        )
        
        commission = expected_value * commission_rate
        expected_with_commission = expected_value + commission
        self.assertEqual(order_value_with_commission, expected_with_commission)
        
    @patch('core.trading.order_manager.OrderManager._log_order')
    def test_order_logging(self, mock_log):
        """Test order logging"""
        # Mock order response
        mock_response = {
            'id': 'test_order_123',
            'symbol': self.symbol,
            'side': self.side,
            'type': self.order_type,
            'status': 'open'
        }
        
        self.mock_exchange.create_order.return_value = mock_response
        
        # Create order
        self.order_manager.create_order(
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            quantity=self.quantity,
            price=self.price
        )
        
        # Verify logging was called
        mock_log.assert_called_once()


class TestExecutionEngine(unittest.TestCase):
    """Test cases for ExecutionEngine"""
    
    def setUp(self):
        """Set up test environment"""
        self.config_manager = ConfigManager()
        self.execution_engine = ExecutionEngine(self.config_manager)
        
        # Mock dependencies
        self.execution_engine.order_manager = Mock()
        self.execution_engine.position_sizer = Mock()
        self.execution_engine.risk_analyzer = Mock()
        self.execution_engine.stop_loss_manager = Mock()
        
        # Test parameters
        self.symbol = "BTC/USDT"
        self.account_balance = 10000.0
        self.current_price = 50000.0
        
    def test_execute_buy_order(self):
        """Test executing a buy order"""
        # Mock signals
        signal = {
            'signal': 'buy',
            'strength': 0.8,
            'confidence': 0.85,
            'timestamp': datetime.utcnow()
        }
        
        # Mock position sizing
        self.execution_engine.position_sizer.calculate_position_size.return_value = 0.2
        
        # Mock order creation
        mock_order_result = {
            'order_id': 'buy_order_123',
            'status': 'open',
            'symbol': self.symbol,
            'side': 'buy',
            'quantity': 0.2,
            'price': self.current_price
        }
        
        self.execution_engine.order_manager.create_order.return_value = mock_order_result
        
        # Mock risk check
        self.execution_engine.risk_analyzer.check_trade_risk.return_value = {
            'approved': True,
            'risk_score': 0.3,
            'max_position_size': 0.3
        }
        
        # Execute buy order
        execution_result = self.execution_engine.execute_trade(
            symbol=self.symbol,
            signal=signal,
            account_balance=self.account_balance,
            current_price=self.current_price
        )
        
        # Verify position sizing was called
        self.execution_engine.position_sizer.calculate_position_size.assert_called_once()
        
        # Verify risk check was called
        self.execution_engine.risk_analyzer.check_trade_risk.assert_called_once()
        
        # Verify order was created
        self.execution_engine.order_manager.create_order.assert_called_once()
        
        # Verify result
        self.assertIsInstance(execution_result, dict)
        self.assertEqual(execution_result['order_id'], 'buy_order_123')
        self.assertEqual(execution_result['action'], 'buy')
        
    def test_execute_sell_order(self):
        """Test executing a sell order"""
        # Mock signals
        signal = {
            'signal': 'sell',
            'strength': 0.7,
            'confidence': 0.8,
            'timestamp': datetime.utcnow()
        }
        
        # Mock position (existing holding)
        position_size = 0.5
        
        # Mock order creation
        mock_order_result = {
            'order_id': 'sell_order_123',
            'status': 'open',
            'symbol': self.symbol,
            'side': 'sell',
            'quantity': position_size,
            'price': self.current_price
        }
        
        self.execution_engine.order_manager.create_order.return_value = mock_order_result
        
        # Execute sell order
        execution_result = self.execution_engine.execute_trade(
            symbol=self.symbol,
            signal=signal,
            account_balance=self.account_balance,
            current_price=self.current_price,
            position_size=position_size
        )
        
        # Verify order was created with correct parameters
        call_args = self.execution_engine.order_manager.create_order.call_args
        self.assertEqual(call_args[1]['side'], 'sell')
        self.assertEqual(call_args[1]['quantity'], position_size)
        
        # Verify result
        self.assertEqual(execution_result['action'], 'sell')
        
    def test_execute_hold_signal(self):
        """Test handling hold signal"""
        # Mock hold signal
        signal = {
            'signal': 'hold',
            'strength': 0.5,
            'confidence': 0.6,
            'timestamp': datetime.utcnow()
        }
        
        # Execute with hold signal
        execution_result = self.execution_engine.execute_trade(
            symbol=self.symbol,
            signal=signal,
            account_balance=self.account_balance,
            current_price=self.current_price
        )
        
        # Verify no order was created
        self.execution_engine.order_manager.create_order.assert_not_called()
        
        # Verify result indicates hold
        self.assertEqual(execution_result['action'], 'hold')
        self.assertEqual(execution_result['order_id'], None)
        
    def test_risk_rejection(self):
        """Test trade execution when risk check rejects"""
        # Mock signals
        signal = {
            'signal': 'buy',
            'strength': 0.9,
            'confidence': 0.95,
            'timestamp': datetime.utcnow()
        }
        
        # Mock risk rejection
        self.execution_engine.risk_analyzer.check_trade_risk.return_value = {
            'approved': False,
            'risk_score': 0.8,
            'reason': 'Exceeds maximum risk limit',
            'max_position_size': 0.1
        }
        
        # Execute trade (should be rejected)
        execution_result = self.execution_engine.execute_trade(
            symbol=self.symbol,
            signal=signal,
            account_balance=self.account_balance,
            current_price=self.current_price
        )
        
        # Verify risk check was called
        self.execution_engine.risk_analyzer.check_trade_risk.assert_called_once()
        
        # Verify no order was created
        self.execution_engine.order_manager.create_order.assert_not_called()
        
        # Verify result indicates rejection
        self.assertEqual(execution_result['action'], 'rejected')
        self.assertIn('reason', execution_result)
        
    def test_stop_loss_placement(self):
        """Test automatic stop loss placement"""
        # Mock buy execution
        signal = {'signal': 'buy', 'strength': 0.8, 'confidence': 0.85}
        
        self.execution_engine.position_sizer.calculate_position_size.return_value = 0.2
        self.execution_engine.risk_analyzer.check_trade_risk.return_value = {
            'approved': True,
            'risk_score': 0.2
        }
        
        mock_order_result = {
            'order_id': 'buy_order_123',
            'status': 'filled',
            'symbol': self.symbol,
            'side': 'buy',
            'quantity': 0.2,
            'price': self.current_price
        }
        
        self.execution_engine.order_manager.create_order.return_value = mock_order_result
        
        # Mock stop loss placement
        mock_stop_loss_result = {
            'order_id': 'stop_loss_123',
            'status': 'open'
        }
        
        self.execution_engine.stop_loss_manager.place_stop_loss.return_value = mock_stop_loss_result
        
        # Execute with stop loss
        execution_result = self.execution_engine.execute_trade(
            symbol=self.symbol,
            signal=signal,
            account_balance=self.account_balance,
            current_price=self.current_price,
            use_stop_loss=True,
            stop_loss_percentage=0.02  # 2% stop loss
        )
        
        # Verify stop loss was placed
        self.execution_engine.stop_loss_manager.place_stop_loss.assert_called_once()
        
        # Verify result includes stop loss info
        self.assertIn('stop_loss_order_id', execution_result)
        
    def test_take_profit_placement(self):
        """Test automatic take profit placement"""
        # Mock buy execution
        signal = {'signal': 'buy', 'strength': 0.8, 'confidence': 0.85}
        
        self.execution_engine.position_sizer.calculate_position_size.return_value = 0.2
        self.execution_engine.risk_analyzer.check_trade_risk.return_value = {
            'approved': True,
            'risk_score': 0.2
        }
        
        mock_order_result = {
            'order_id': 'buy_order_123',
            'status': 'filled',
            'symbol': self.symbol,
            'side': 'buy',
            'quantity': 0.2,
            'price': self.current_price
        }
        
        self.execution_engine.order_manager.create_order.return_value = mock_order_result
        
        # Mock take profit placement
        mock_take_profit_result = {
            'order_id': 'take_profit_123',
            'status': 'open'
        }
        
        self.execution_engine.order_manager.create_take_profit_order.return_value = mock_take_profit_result
        
        # Execute with take profit
        execution_result = self.execution_engine.execute_trade(
            symbol=self.symbol,
            signal=signal,
            account_balance=self.account_balance,
            current_price=self.current_price,
            use_take_profit=True,
            take_profit_percentage=0.05  # 5% take profit
        )
        
        # Verify take profit was placed
        self.execution_engine.order_manager.create_take_profit_order.assert_called_once()
        
        # Verify result includes take profit info
        self.assertIn('take_profit_order_id', execution_result)
        
    def test_partial_fill_handling(self):
        """Test handling of partially filled orders"""
        # Mock partial fill scenario
        signal = {'signal': 'buy', 'strength': 0.8, 'confidence': 0.85}
        
        # Mock initial order
        mock_initial_order = {
            'order_id': 'order_123',
            'status': 'open',
            'filled': 0.0,
            'remaining': 0.2
        }
        
        # Mock updated order (partially filled)
        mock_partial_fill = {
            'order_id': 'order_123',
            'status': 'partially_filled',
            'filled': 0.1,
            'remaining': 0.1
        }
        
        self.execution_engine.order_manager.create_order.return_value = mock_initial_order
        self.execution_engine.order_manager.get_order_status.return_value = mock_partial_fill
        
        # Execute trade
        execution_result = self.execution_engine.execute_trade(
            symbol=self.symbol,
            signal=signal,
            account_balance=self.account_balance,
            current_price=self.current_price
        )
        
        # In a real scenario, we might monitor partial fills
        # For this test, we just verify execution started
        self.assertEqual(execution_result['order_id'], 'order_123')
        
    def test_error_handling(self):
        """Test error handling during execution"""
        # Mock signal
        signal = {'signal': 'buy', 'strength': 0.8, 'confidence': 0.85}
        
        # Mock exchange error
        self.execution_engine.order_manager.create_order.side_effect = Exception("Exchange error")
        
        # Execute trade (should handle error gracefully)
        execution_result = self.execution_engine.execute_trade(
            symbol=self.symbol,
            signal=signal,
            account_balance=self.account_balance,
            current_price=self.current_price
        )
        
        # Verify error is captured in result
        self.assertEqual(execution_result['action'], 'error')
        self.assertIn('error', execution_result)
        
    def test_position_monitoring(self):
        """Test position monitoring functionality"""
        # Mock open positions
        mock_positions = [
            {
                'symbol': self.symbol,
                'side': 'buy',
                'quantity': 0.5,
                'entry_price': 49000.0,
                'current_price': 50500.0,
                'unrealized_pnl': 750.0,
                'unrealized_pnl_percent': 0.0306
            }
        ]
        
        self.execution_engine.order_manager.get_positions = Mock(return_value=mock_positions)
        
        # Monitor positions
        positions = self.execution_engine.monitor_positions(symbol=self.symbol)
        
        # Verify positions retrieved
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['symbol'], self.symbol)
        self.assertGreater(positions[0]['unrealized_pnl'], 0)
        
    def test_execution_limits(self):
        """Test execution rate limiting"""
        # Mock multiple rapid executions
        signal = {'signal': 'buy', 'strength': 0.8, 'confidence': 0.85}
        
        # Set up mock responses
        self.execution_engine.position_sizer.calculate_position_size.return_value = 0.1
        self.execution_engine.risk_analyzer.check_trade_risk.return_value = {
            'approved': True,
            'risk_score': 0.2
        }
        
        # Try to execute multiple trades rapidly
        results = []
        for i in range(5):
            result = self.execution_engine.execute_trade(
                symbol=self.symbol,
                signal=signal,
                account_balance=self.account_balance,
                current_price=self.current_price
            )
            results.append(result)
        
        # Execution engine should implement rate limiting
        # This test verifies it doesn't crash on rapid calls
        self.assertEqual(len(results), 5)
        
        # At least some should succeed
        success_count = sum(1 for r in results if r.get('action') in ['buy', 'sell'])
        self.assertGreater(success_count, 0)


class TestRiskManagement(unittest.TestCase):
    """Test cases for risk management components"""
    
    def setUp(self):
        """Set up test environment"""
        self.config_manager = ConfigManager()
        self.risk_analyzer = RiskAnalyzer(self.config_manager)
        self.stop_loss_manager = StopLossManager(self.config_manager)
        
        # Test parameters
        self.symbol = "BTC/USDT"
        self.account_balance = 10000.0
        self.position_size = 0.2
        self.entry_price = 50000.0
        self.current_price = 50500.0
        
    def test_risk_score_calculation(self):
        """Test risk score calculation"""
        # Calculate risk score
        risk_data = {
            'volatility': 0.02,  # 2% daily volatility
            'correlation': 0.8,
            'market_regime': 'trending',
            'position_size': self.position_size,
            'account_balance': self.account_balance
        }
        
        risk_score = self.risk_analyzer.calculate_risk_score(**risk_data)
        
        self.assertIsInstance(risk_score, float)
        self.assertGreaterEqual(risk_score, 0.0)
        self.assertLessEqual(risk_score, 1.0)
        
        # Higher volatility should increase risk score
        high_vol_data = risk_data.copy()
        high_vol_data['volatility'] = 0.05  # 5% volatility
        
        high_vol_score = self.risk_analyzer.calculate_risk_score(**high_vol_data)
        self.assertGreaterEqual(high_vol_score, risk_score)
        
    def test_position_risk_check(self):
        """Test position risk checking"""
        # Check position risk
        position_data = {
            'symbol': self.symbol,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'account_balance': self.account_balance,
            'portfolio_exposure': 0.3  # 30% in BTC
        }
        
        risk_check = self.risk_analyzer.check_position_risk(**position_data)
        
        self.assertIsInstance(risk_check, dict)
        self.assertIn('risk_level', risk_check)
        self.assertIn('recommendation', risk_check)
        self.assertIn('max_position', risk_check)
        
        valid_risk_levels = ['low', 'medium', 'high', 'critical']
        self.assertIn(risk_check['risk_level'], valid_risk_levels)
        
    def test_stop_loss_calculation(self):
        """Test stop loss calculation"""
        # Test fixed percentage stop loss
        stop_loss_price = self.stop_loss_manager.calculate_stop_loss(
            entry_price=self.entry_price,
            stop_loss_type='percentage',
            stop_loss_value=0.02  # 2%
        )
        
        expected_stop = self.entry_price * (1 - 0.02)
        self.assertEqual(stop_loss_price, expected_stop)
        
        # Test ATR-based stop loss
        atr = 1000.0  # Average True Range
        atr_multiplier = 1.5
        
        stop_loss_atr = self.stop_loss_manager.calculate_stop_loss(
            entry_price=self.entry_price,
            stop_loss_type='atr',
            atr=atr,
            atr_multiplier=atr_multiplier
        )
        
        expected_atr_stop = self.entry_price - (atr * atr_multiplier)
        self.assertEqual(stop_loss_atr, expected_atr_stop)
        
        # Test volatility-based stop loss
        volatility = 0.02  # 2% daily volatility
        vol_multiplier = 2.0
        
        stop_loss_vol = self.stop_loss_manager.calculate_stop_loss(
            entry_price=self.entry_price,
            stop_loss_type='volatility',
            volatility=volatility,
            volatility_multiplier=vol_multiplier
        )
        
        expected_vol_stop = self.entry_price * (1 - (volatility * vol_multiplier))
        self.assertEqual(stop_loss_vol, expected_vol_stop)
        
    def test_trailing_stop_loss(self):
        """Test trailing stop loss calculation"""
        # Initial stop loss
        entry_price = 50000.0
        initial_stop = 49000.0  # 2% stop loss
        
        # Price increases
        current_price = 51000.0
        
        # Calculate trailing stop
        trailing_stop = self.stop_loss_manager.calculate_trailing_stop(
            entry_price=entry_price,
            current_price=current_price,
            initial_stop=initial_stop,
            trailing_percentage=0.01  # 1% trailing
        )
        
        # Trailing stop should be current_price * (1 - trailing_percentage)
        expected_trailing = current_price * (1 - 0.01)
        self.assertEqual(trailing_stop, expected_trailing)
        
        # Test that trailing stop only moves up
        price_drops = 50500.0
        trailing_stop_2 = self.stop_loss_manager.calculate_trailing_stop(
            entry_price=entry_price,
            current_price=price_drops,
            initial_stop=initial_stop,
            trailing_percentage=0.01,
            current_trailing_stop=trailing_stop
        )
        
        # Trailing stop should not move down
        self.assertEqual(trailing_stop_2, trailing_stop)
        
    def test_risk_limits(self):
        """Test risk limit enforcement"""
        # Test maximum position size
        max_position_pct = 0.1  # Maximum 10% of account
        
        position_value = self.position_size * self.current_price
        account_exposure = position_value / self.account_balance
        
        within_limits = self.risk_analyzer.check_position_limits(
            position_value=position_value,
            account_balance=self.account_balance,
            max_position_percentage=max_position_pct
        )
        
        if account_exposure > max_position_pct:
            self.assertFalse(within_limits['approved'])
        else:
            self.assertTrue(within_limits['approved'])
        
        # Test maximum daily loss
        daily_pnl = -500.0  # $500 loss
        max_daily_loss = 1000.0  # Max $1000 loss
        
        daily_loss_check = self.risk_analyzer.check_daily_loss_limit(
            daily_pnl=daily_pnl,
            max_daily_loss=max_daily_loss
        )
        
        if abs(daily_pnl) > max_daily_loss:
            self.assertFalse(daily_loss_check['approved'])
        else:
            self.assertTrue(daily_loss_check['approved'])
        
    def test_volatility_adjustment(self):
        """Test volatility-based position adjustment"""
        # High volatility should reduce position size
        high_volatility = 0.05  # 5% daily volatility
        
        adjusted_size = self.risk_analyzer.adjust_position_for_volatility(
            position_size=self.position_size,
            volatility=high_volatility,
            base_volatility=0.02  # Base 2% volatility
        )
        
        # Position should be reduced in high volatility
        self.assertLess(adjusted_size, self.position_size)
        
        # Low volatility might allow larger position
        low_volatility = 0.01  # 1% daily volatility
        
        adjusted_size_low = self.risk_analyzer.adjust_position_for_volatility(
            position_size=self.position_size,
            volatility=low_volatility,
            base_volatility=0.02
        )
        
        # Position might be increased in low volatility
        self.assertGreaterEqual(adjusted_size_low, self.position_size)
        
    def test_correlation_risk(self):
        """Test correlation-based risk assessment"""
        # High correlation with portfolio increases risk
        portfolio_correlation = 0.9
        
        correlation_risk = self.risk_analyzer.assess_correlation_risk(
            symbol=self.symbol,
            portfolio_correlation=portfolio_correlation,
            position_size=self.position_size
        )
        
        self.assertIsInstance(correlation_risk, dict)
        self.assertIn('risk_score', correlation_risk)
        self.assertIn('diversification_score', correlation_risk)
        
        # Higher correlation should increase risk score
        self.assertGreater(correlation_risk['risk_score'], 0.5)
        
    def test_stop_loss_placement(self):
        """Test stop loss order placement"""
        # Mock order manager
        self.stop_loss_manager.order_manager = Mock()
        
        # Mock stop loss order response
        mock_response = {
            'order_id': 'stop_loss_123',
            'status': 'open'
        }
        
        self.stop_loss_manager.order_manager.create_stop_loss_order.return_value = mock_response
        
        # Place stop loss
        stop_loss_price = 49000.0
        result = self.stop_loss_manager.place_stop_loss(
            symbol=self.symbol,
            quantity=self.position_size,
            stop_price=stop_loss_price,
            limit_price=stop_loss_price * 0.99  # Slightly below for limit order
        )
        
        # Verify order was placed
        self.stop_loss_manager.order_manager.create_stop_loss_order.assert_called_once()
        
        # Verify result
        self.assertEqual(result['order_id'], 'stop_loss_123')
        
    def test_stop_loss_monitoring(self):
        """Test stop loss monitoring"""
        # Mock positions with stop losses
        positions = [
            {
                'symbol': self.symbol,
                'quantity': self.position_size,
                'entry_price': self.entry_price,
                'current_price': self.current_price,
                'stop_loss_price': 49500.0,
                'stop_loss_order_id': 'stop_123'
            }
        ]
        
        # Check if stop loss should be triggered
        should_trigger = self.stop_loss_manager.check_stop_losses(
            positions=positions,
            current_prices={self.symbol: 49400.0}  # Below stop loss
        )
        
        self.assertIsInstance(should_trigger, list)
        if should_trigger:
            self.assertIn(self.symbol, [p['symbol'] for p in should_trigger])
        
    def test_risk_reporting(self):
        """Test risk reporting functionality"""
        # Generate risk report
        portfolio_data = {
            'total_value': self.account_balance,
            'positions': [
                {
                    'symbol': self.symbol,
                    'value': self.position_size * self.current_price,
                    'unrealized_pnl': 500.0
                }
            ],
            'daily_pnl': 200.0,
            'max_drawdown': 0.05,
            'volatility': 0.02
        }
        
        risk_report = self.risk_analyzer.generate_risk_report(**portfolio_data)
        
        self.assertIsInstance(risk_report, dict)
        self.assertIn('total_risk_score', risk_report)
        self.assertIn('position_risks', risk_report)
        self.assertIn('recommendations', risk_report)
        
        # Validate report structure
        self.assertGreaterEqual(risk_report['total_risk_score'], 0.0)
        self.assertLessEqual(risk_report['total_risk_score'], 1.0)
        self.assertIsInstance(risk_report['position_risks'], list)
        self.assertIsInstance(risk_report['recommendations'], list)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)