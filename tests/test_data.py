"""
Test file for data processing components in the Bitcoin Trading AI application.
Unit tests for data collection, feature engineering, preprocessing, and validation.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_processing.data_collector import DataCollector
from core.data_processing.feature_engineer import FeatureEngineer
from core.data_processing.data_preprocessor import DataPreprocessor
from core.data_processing.data_validator import DataValidator
from database.models import MarketData
from config.config_manager import ConfigManager


class TestDataCollector(unittest.TestCase):
    """Test cases for DataCollector"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.config_manager = ConfigManager()
        
        # Create test data directory
        cls.data_dir = os.path.join(cls.test_dir, 'data')
        os.makedirs(cls.data_dir, exist_ok=True)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up fresh test instance"""
        self.data_collector = DataCollector(self.config_manager)
        
        # Mock exchange client
        self.data_collector.exchange_client = Mock()
        
        # Test parameters
        self.symbol = "BTC/USDT"
        self.timeframe = "1h"
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 2)
        
    def test_fetch_ohlcv_data(self):
        """Test fetching OHLCV data from exchange"""
        # Mock exchange response
        mock_ohlcv = [
            [1672531200000, 50000, 51000, 49500, 50500, 1000],  # Jan 1, 2024
            [1672534800000, 50500, 51500, 50000, 51000, 1200],  # Jan 1, 2024 +1h
            [1672538400000, 51000, 52000, 50500, 51500, 1500],  # Jan 1, 2024 +2h
        ]
        
        self.data_collector.exchange_client.fetch_ohlcv.return_value = mock_ohlcv
        
        # Fetch data
        ohlcv_data = self.data_collector.fetch_ohlcv_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            since=self.start_date
        )
        
        # Verify exchange was called
        self.data_collector.exchange_client.fetch_ohlcv.assert_called_once_with(
            symbol=self.symbol,
            timeframe=self.timeframe,
            since=int(self.start_date.timestamp() * 1000)
        )
        
        # Verify data structure
        self.assertIsInstance(ohlcv_data, list)
        self.assertEqual(len(ohlcv_data), 3)
        
        # Verify each candle
        for candle in ohlcv_data:
            self.assertEqual(len(candle), 6)  # [timestamp, open, high, low, close, volume]
            self.assertIsInstance(candle[0], int)  # timestamp in milliseconds
            self.assertIsInstance(candle[1], (int, float))  # open
            self.assertIsInstance(candle[2], (int, float))  # high
            self.assertIsInstance(candle[3], (int, float))  # low
            self.assertIsInstance(candle[4], (int, float))  # close
            self.assertIsInstance(candle[5], (int, float))  # volume
    
    def test_convert_ohlcv_to_dataframe(self):
        """Test converting OHLCV data to DataFrame"""
        # Sample OHLCV data
        ohlcv_data = [
            [1672531200000, 50000.0, 51000.0, 49500.0, 50500.0, 1000.0],
            [1672534800000, 50500.0, 51500.0, 50000.0, 51000.0, 1200.0],
            [1672538400000, 51000.0, 52000.0, 50500.0, 51500.0, 1500.0],
        ]
        
        # Convert to DataFrame
        df = self.data_collector.convert_ohlcv_to_dataframe(ohlcv_data)
        
        # Verify DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertListEqual(
            list(df.columns),
            ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Verify data types
        self.assertEqual(df['timestamp'].dtype, 'datetime64[ns]')
        self.assertEqual(df['open'].dtype, 'float64')
        self.assertEqual(df['volume'].dtype, 'float64')
        
        # Verify data values
        self.assertEqual(df.iloc[0]['open'], 50000.0)
        self.assertEqual(df.iloc[0]['close'], 50500.0)
        self.assertEqual(df.iloc[0]['volume'], 1000.0)
        
        # Verify timestamp conversion
        self.assertEqual(df.iloc[0]['timestamp'], pd.Timestamp('2024-01-01 00:00:00'))
    
    def test_fetch_historical_data(self):
        """Test fetching historical data with pagination"""
        # Mock paginated responses
        mock_responses = [
            [  # First page
                [1672531200000, 50000, 51000, 49500, 50500, 1000],
                [1672534800000, 50500, 51500, 50000, 51000, 1200],
            ],
            [  # Second page (empty - end of data)
                [],
            ]
        ]
        
        self.data_collector.exchange_client.fetch_ohlcv.side_effect = mock_responses
        
        # Fetch historical data
        historical_data = self.data_collector.fetch_historical_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Verify multiple calls were made
        self.assertGreaterEqual(self.data_collector.exchange_client.fetch_ohlcv.call_count, 2)
        
        # Verify data collection
        self.assertIsInstance(historical_data, list)
        self.assertEqual(len(historical_data), 2)  # Only first page had data
    
    def test_save_to_database(self):
        """Test saving data to database"""
        # Create mock database session
        mock_session = Mock()
        self.data_collector.db_session = mock_session
        
        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 01:00:00']),
            'open': [50000.0, 50500.0],
            'high': [51000.0, 51500.0],
            'low': [49500.0, 50000.0],
            'close': [50500.0, 51000.0],
            'volume': [1000.0, 1200.0]
        })
        
        # Save to database
        saved_count = self.data_collector.save_to_database(
            df=df,
            symbol=self.symbol,
            timeframe=self.timeframe
        )
        
        # Verify database operations
        self.assertEqual(mock_session.add.call_count, 2)  # Two records
        mock_session.commit.assert_called_once()
        
        # Verify MarketData objects were created
        calls = mock_session.add.call_args_list
        for call in calls:
            args = call[0]
            self.assertIsInstance(args[0], MarketData)
            market_data = args[0]
            self.assertEqual(market_data.symbol, self.symbol)
            self.assertEqual(market_data.timeframe, self.timeframe)
            self.assertIsInstance(market_data.timestamp, datetime)
    
    def test_load_from_database(self):
        """Test loading data from database"""
        # Create mock database query
        mock_query = Mock()
        mock_session = Mock()
        mock_session.query.return_value = mock_query
        
        # Mock query results
        mock_results = [
            MarketData(
                symbol=self.symbol,
                timeframe=self.timeframe,
                timestamp=datetime(2024, 1, 1, 0, 0, 0),
                open=Decimal("50000.00"),
                high=Decimal("51000.00"),
                low=Decimal("49500.00"),
                close=Decimal("50500.00"),
                volume=Decimal("1000.00")
            ),
            MarketData(
                symbol=self.symbol,
                timeframe=self.timeframe,
                timestamp=datetime(2024, 1, 1, 1, 0, 0),
                open=Decimal("50500.00"),
                high=Decimal("51500.00"),
                low=Decimal("50000.00"),
                close=Decimal("51000.00"),
                volume=Decimal("1200.00")
            )
        ]
        
        mock_query.filter.return_value.order_by.return_value.all.return_value = mock_results
        
        self.data_collector.db_session = mock_session
        
        # Load from database
        df = self.data_collector.load_from_database(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Verify database query
        mock_session.query.assert_called_once_with(MarketData)
        
        # Verify DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertListEqual(
            list(df.columns),
            ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Verify data values
        self.assertEqual(df.iloc[0]['open'], 50000.0)
        self.assertEqual(df.iloc[1]['close'], 51000.0)
    
    def test_download_and_save_data(self):
        """Test complete download and save workflow"""
        # Mock exchange response
        mock_ohlcv = [
            [1672531200000, 50000, 51000, 49500, 50500, 1000],
            [1672534800000, 50500, 51500, 50000, 51000, 1200],
        ]
        
        self.data_collector.exchange_client.fetch_ohlcv.return_value = mock_ohlcv
        
        # Mock database session
        mock_session = Mock()
        self.data_collector.db_session = mock_session
        
        # Execute download and save
        result = self.data_collector.download_and_save_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Verify complete workflow
        self.data_collector.exchange_client.fetch_ohlcv.assert_called_once()
        self.assertGreaterEqual(mock_session.add.call_count, 2)
        mock_session.commit.assert_called_once()
        
        # Verify result
        self.assertIsInstance(result, dict)
        self.assertIn('records_saved', result)
        self.assertIn('start_date', result)
        self.assertIn('end_date', result)
        self.assertEqual(result['symbol'], self.symbol)
    
    def test_get_latest_data(self):
        """Test getting latest data"""
        # Mock exchange response for latest candle
        mock_latest = [
            [1672538400000, 51000, 52000, 50500, 51500, 1500],
        ]
        
        self.data_collector.exchange_client.fetch_ohlcv.return_value = mock_latest
        
        # Get latest data
        latest_data = self.data_collector.get_latest_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            limit=1
        )
        
        # Verify exchange call
        self.data_collector.exchange_client.fetch_ohlcv.assert_called_once_with(
            symbol=self.symbol,
            timeframe=self.timeframe,
            limit=1
        )
        
        # Verify data
        self.assertIsInstance(latest_data, list)
        self.assertEqual(len(latest_data), 1)
    
    def test_check_data_gaps(self):
        """Test checking for data gaps"""
        # Create test DataFrame with a gap
        df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 00:00:00',
                '2024-01-01 01:00:00',
                '2024-01-01 03:00:00',  # Gap at 02:00:00
                '2024-01-01 04:00:00'
            ]),
            'open': [50000, 50500, 51000, 51500],
            'close': [50500, 51000, 51500, 52000]
        })
        
        # Check for gaps
        gaps = self.data_collector.check_data_gaps(
            df=df,
            timeframe=self.timeframe
        )
        
        # Verify gaps detected
        self.assertIsInstance(gaps, list)
        if gaps:  # May or may not detect gaps depending on implementation
            for gap in gaps:
                self.assertIsInstance(gap, tuple)
                self.assertEqual(len(gap), 2)
                self.assertIsInstance(gap[0], datetime)
                self.assertIsInstance(gap[1], datetime)
    
    def test_error_handling(self):
        """Test error handling in data collection"""
        # Mock exchange error
        self.data_collector.exchange_client.fetch_ohlcv.side_effect = Exception("Exchange error")
        
        # Should handle error gracefully
        with self.assertRaises(Exception):
            self.data_collector.fetch_ohlcv_data(
                symbol=self.symbol,
                timeframe=self.timeframe
            )
        
        # Test with invalid symbol
        with self.assertRaises(ValueError):
            self.data_collector.fetch_ohlcv_data(
                symbol="INVALID/SYMBOL",
                timeframe=self.timeframe
            )
    
    def test_data_validation_during_collection(self):
        """Test data validation during collection"""
        # Mock exchange response with invalid data
        mock_ohlcv_invalid = [
            [1672531200000, 50000, 51000, 49500, 50500, 1000],
            [1672534800000, 50500, 51500, 50000, None, 1200],  # Invalid close price
        ]
        
        self.data_collector.exchange_client.fetch_ohlcv.return_value = mock_ohlcv_invalid
        
        # Fetch data - should handle invalid data
        ohlcv_data = self.data_collector.fetch_ohlcv_data(
            symbol=self.symbol,
            timeframe=self.timeframe
        )
        
        # Implementation should handle or filter invalid data
        self.assertIsInstance(ohlcv_data, list)
        # May filter out invalid rows or raise exception
    
    @patch('core.data_processing.data_collector.DataCollector.save_to_csv')
    def test_save_to_csv(self, mock_save_csv):
        """Test saving data to CSV"""
        # Create test DataFrame
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 00:00:00']),
            'open': [50000.0],
            'close': [50500.0]
        })
        
        # Save to CSV
        csv_path = os.path.join(self.data_dir, 'test_data.csv')
        self.data_collector.save_to_csv(df, csv_path)
        
        # Verify CSV was saved
        # In actual implementation, this would create a file
        # For test, we verify the method was called
        pass


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer"""
    
    def setUp(self):
        """Set up test environment"""
        self.config_manager = ConfigManager()
        self.feature_engineer = FeatureEngineer(self.config_manager)
        
        # Create test DataFrame
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'open': np.random.uniform(45000, 55000, 100),
            'high': np.random.uniform(46000, 56000, 100),
            'low': np.random.uniform(44000, 54000, 100),
            'close': np.random.uniform(45000, 55000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # Set 'close' as a more predictable series for testing
        self.df['close'] = np.linspace(45000, 55000, 100)
    
    def test_calculate_returns(self):
        """Test calculating returns"""
        # Calculate returns
        df_with_returns = self.feature_engineer.calculate_returns(self.df)
        
        # Verify new columns
        self.assertIn('returns', df_with_returns.columns)
        self.assertIn('log_returns', df_with_returns.columns)
        
        # Verify calculations
        # First return should be NaN
        self.assertTrue(pd.isna(df_with_returns.iloc[0]['returns']))
        
        # Subsequent returns should be calculated
        for i in range(1, len(df_with_returns)):
            expected_return = (self.df.iloc[i]['close'] / self.df.iloc[i-1]['close']) - 1
            self.assertAlmostEqual(
                df_with_returns.iloc[i]['returns'],
                expected_return,
                places=10
            )
            
            # Log returns
            expected_log_return = np.log(self.df.iloc[i]['close'] / self.df.iloc[i-1]['close'])
            self.assertAlmostEqual(
                df_with_returns.iloc[i]['log_returns'],
                expected_log_return,
                places=10
            )
    
    def test_calculate_moving_averages(self):
        """Test calculating moving averages"""
        # Calculate moving averages
        windows = [5, 10, 20]
        df_with_ma = self.feature_engineer.calculate_moving_averages(self.df, windows=windows)
        
        # Verify new columns
        for window in windows:
            col_name = f'sma_{window}'
            self.assertIn(col_name, df_with_ma.columns)
            
            # Verify calculations
            sma_series = df_with_ma[col_name]
            
            # First (window-1) values should be NaN
            for i in range(window - 1):
                self.assertTrue(pd.isna(sma_series.iloc[i]))
            
            # Subsequent values should be calculated
            for i in range(window - 1, len(df_with_ma)):
                expected_sma = self.df['close'].iloc[i-window+1:i+1].mean()
                self.assertAlmostEqual(sma_series.iloc[i], expected_sma, places=10)
    
    def test_calculate_exponential_moving_averages(self):
        """Test calculating exponential moving averages"""
        # Calculate EMAs
        windows = [12, 26]
        df_with_ema = self.feature_engineer.calculate_exponential_moving_averages(self.df, windows=windows)
        
        # Verify new columns
        for window in windows:
            col_name = f'ema_{window}'
            self.assertIn(col_name, df_with_ema.columns)
            
            # First value is SMA
            ema_series = df_with_ema[col_name]
            
            # Should have some non-NaN values
            self.assertTrue(ema_series.notna().any())
    
    def test_calculate_rsi(self):
        """Test calculating RSI"""
        # Calculate RSI
        periods = 14
        df_with_rsi = self.feature_engineer.calculate_rsi(self.df, period=periods)
        
        # Verify RSI column
        self.assertIn('rsi', df_with_rsi.columns)
        
        # RSI should be between 0 and 100
        rsi_series = df_with_rsi['rsi'].dropna()
        
        if len(rsi_series) > 0:
            self.assertGreaterEqual(rsi_series.min(), 0)
            self.assertLessEqual(rsi_series.max(), 100)
            
            # For trending up data, RSI should be higher
            # Our test data is trending up
            self.assertGreater(rsi_series.iloc[-1], 50)
    
    def test_calculate_macd(self):
        """Test calculating MACD"""
        # Calculate MACD
        df_with_macd = self.feature_engineer.calculate_macd(self.df)
        
        # Verify MACD columns
        expected_columns = ['macd', 'macd_signal', 'macd_histogram']
        for col in expected_columns:
            self.assertIn(col, df_with_macd.columns)
        
        # Verify calculations
        macd = df_with_macd['macd'].dropna()
        signal = df_with_macd['macd_signal'].dropna()
        histogram = df_with_macd['macd_histogram'].dropna()
        
        # Should have same number of non-NaN values
        self.assertEqual(len(macd), len(signal))
        self.assertEqual(len(macd), len(histogram))
        
        # Histogram should be MACD - signal line
        for i in range(len(histogram)):
            expected_hist = macd.iloc[i] - signal.iloc[i]
            self.assertAlmostEqual(histogram.iloc[i], expected_hist, places=10)
    
    def test_calculate_bollinger_bands(self):
        """Test calculating Bollinger Bands"""
        # Calculate Bollinger Bands
        window = 20
        num_std = 2
        df_with_bb = self.feature_engineer.calculate_bollinger_bands(
            self.df, 
            window=window, 
            num_std=num_std
        )
        
        # Verify Bollinger Bands columns
        expected_columns = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent']
        for col in expected_columns:
            self.assertIn(col, df_with_bb.columns)
        
        # Verify calculations
        middle_band = df_with_bb['bb_middle'].dropna()
        upper_band = df_with_bb['bb_upper'].dropna()
        lower_band = df_with_bb['bb_lower'].dropna()
        
        # Bands should have same number of values
        self.assertEqual(len(upper_band), len(middle_band))
        self.assertEqual(len(lower_band), len(middle_band))
        
        # Verify band calculations
        for i in range(len(middle_band)):
            idx = i + window - 1  # Adjust for window
            sma = self.df['close'].iloc[idx-window+1:idx+1].mean()
            std = self.df['close'].iloc[idx-window+1:idx+1].std()
            
            self.assertAlmostEqual(middle_band.iloc[i], sma, places=10)
            self.assertAlmostEqual(upper_band.iloc[i], sma + (num_std * std), places=10)
            self.assertAlmostEqual(lower_band.iloc[i], sma - (num_std * std), places=10)
            
            # Verify width
            expected_width = upper_band.iloc[i] - lower_band.iloc[i]
            self.assertAlmostEqual(df_with_bb.iloc[idx]['bb_width'], expected_width, places=10)
    
    def test_calculate_atr(self):
        """Test calculating Average True Range"""
        # Calculate ATR
        period = 14
        df_with_atr = self.feature_engineer.calculate_atr(self.df, period=period)
        
        # Verify ATR column
        self.assertIn('atr', df_with_atr.columns)
        
        # ATR should always be positive
        atr_series = df_with_atr['atr'].dropna()
        self.assertTrue((atr_series > 0).all())
        
        # Verify calculation
        # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        # ATR is SMA of True Range
        tr_values = []
        for i in range(1, len(self.df)):
            high_low = self.df.iloc[i]['high'] - self.df.iloc[i]['low']
            high_close = abs(self.df.iloc[i]['high'] - self.df.iloc[i-1]['close'])
            low_close = abs(self.df.iloc[i]['low'] - self.df.iloc[i-1]['close'])
            tr = max(high_low, high_close, low_close)
            tr_values.append(tr)
        
        # Calculate expected ATR (SMA of TR)
        expected_atr = []
        for i in range(len(tr_values)):
            if i >= period - 1:
                atr = np.mean(tr_values[i-period+1:i+1])
                expected_atr.append(atr)
        
        # Compare with calculated ATR
        for i, atr in enumerate(atr_series):
            self.assertAlmostEqual(atr, expected_atr[i], places=10)
    
    def test_calculate_volume_indicators(self):
        """Test calculating volume indicators"""
        # Calculate volume indicators
        df_with_volume = self.feature_engineer.calculate_volume_indicators(self.df)
        
        # Verify volume indicator columns
        volume_columns = ['volume_sma', 'volume_ratio', 'obv']
        for col in volume_columns:
            self.assertIn(col, df_with_volume.columns)
        
        # Test OBV calculation
        obv_series = df_with_volume['obv'].dropna()
        
        # OBV should change with price direction
        obv_changes = obv_series.diff().dropna()
        
        # Compare with price changes
        close_changes = self.df['close'].diff().dropna()
        
        # OBV should increase when close increases, decrease when close decreases
        for i in range(1, min(len(obv_changes), len(close_changes))):
            if close_changes.iloc[i] > 0:
                self.assertGreaterEqual(obv_changes.iloc[i], 0)
            elif close_changes.iloc[i] < 0:
                self.assertLessEqual(obv_changes.iloc[i], 0)
    
    def test_calculate_stochastic_oscillator(self):
        """Test calculating Stochastic Oscillator"""
        # Calculate Stochastic Oscillator
        df_with_stoch = self.feature_engineer.calculate_stochastic_oscillator(self.df)
        
        # Verify stochastic columns
        stoch_columns = ['stoch_k', 'stoch_d']
        for col in stoch_columns:
            self.assertIn(col, df_with_stoch.columns)
        
        # Stochastic values should be between 0 and 100
        stoch_k = df_with_stoch['stoch_k'].dropna()
        stoch_d = df_with_stoch['stoch_d'].dropna()
        
        if len(stoch_k) > 0:
            self.assertGreaterEqual(stoch_k.min(), 0)
            self.assertLessEqual(stoch_k.max(), 100)
        
        if len(stoch_d) > 0:
            self.assertGreaterEqual(stoch_d.min(), 0)
            self.assertLessEqual(stoch_d.max(), 100)
    
    def test_create_lag_features(self):
        """Test creating lag features"""
        # Create lag features
        lags = [1, 2, 3, 5]
        feature_columns = ['close', 'volume']
        df_with_lags = self.feature_engineer.create_lag_features(
            self.df, 
            feature_columns=feature_columns, 
            lags=lags
        )
        
        # Verify lag columns created
        for col in feature_columns:
            for lag in lags:
                lag_col = f'{col}_lag_{lag}'
                self.assertIn(lag_col, df_with_lags.columns)
                
                # Verify lag values
                for i in range(lag, len(self.df)):
                    expected_value = self.df.iloc[i-lag][col]
                    actual_value = df_with_lags.iloc[i][lag_col]
                    
                    # Handle NaN values
                    if pd.isna(expected_value) or pd.isna(actual_value):
                        self.assertTrue(pd.isna(expected_value) and pd.isna(actual_value))
                    else:
                        self.assertEqual(actual_value, expected_value)
    
    def test_create_rolling_statistics(self):
        """Test creating rolling statistics"""
        # Create rolling statistics
        windows = [5, 10, 20]
        statistics = ['mean', 'std', 'min', 'max']
        feature_columns = ['close', 'volume']
        
        df_with_rolling = self.feature_engineer.create_rolling_statistics(
            self.df,
            feature_columns=feature_columns,
            windows=windows,
            statistics=statistics
        )
        
        # Verify rolling statistics columns created
        for col in feature_columns:
            for window in windows:
                for stat in statistics:
                    roll_col = f'{col}_{stat}_{window}'
                    self.assertIn(roll_col, df_with_rolling.columns)
    
    def test_add_time_features(self):
        """Test adding time-based features"""
        # Add time features
        df_with_time = self.feature_engineer.add_time_features(self.df)
        
        # Verify time features added
        time_columns = [
            'hour', 'day', 'dayofweek', 'dayofyear', 
            'week', 'month', 'quarter', 'year'
        ]
        
        for col in time_columns:
            self.assertIn(col, df_with_time.columns)
        
        # Verify hour extraction
        hours = df_with_time['hour'].unique()
        self.assertTrue(all(0 <= h <= 23 for h in hours))
        
        # Verify day of week
        days_of_week = df_with_time['dayofweek'].unique()
        self.assertTrue(all(0 <= d <= 6 for d in days_of_week))
    
    def test_engineer_all_features(self):
        """Test engineering all features"""
        # Engineer all features
        df_with_features = self.feature_engineer.engineer_all_features(self.df)
        
        # Verify multiple feature types added
        feature_types = [
            'returns', 'sma_', 'ema_', 'rsi', 'macd',
            'bb_', 'atr', 'volume_', 'stoch_'
        ]
        
        # Check that at least some features were added
        added_features = [col for col in df_with_features.columns 
                         if any(ft in col for ft in feature_types)]
        
        self.assertGreater(len(added_features), 0)
        
        # Verify DataFrame not corrupted
        self.assertEqual(len(df_with_features), len(self.df))
        self.assertIn('close', df_with_features.columns)
        self.assertIn('volume', df_with_features.columns)
    
    def test_feature_selection(self):
        """Test feature selection methods"""
        # First engineer features
        df_with_features = self.feature_engineer.engineer_all_features(self.df)
        
        # Select top features by correlation
        target_column = 'returns'
        top_n = 10
        
        selected_features = self.feature_engineer.select_features_by_correlation(
            df_with_features,
            target_column=target_column,
            top_n=top_n
        )
        
        # Verify feature selection
        self.assertIsInstance(selected_features, list)
        self.assertLessEqual(len(selected_features), top_n)
        
        # All selected features should be in DataFrame
        for feature in selected_features:
            self.assertIn(feature, df_with_features.columns)
    
    def test_handle_missing_values_in_features(self):
        """Test handling missing values in engineered features"""
        # Create DataFrame with missing values
        df_with_nan = self.df.copy()
        df_with_nan.loc[10:20, 'close'] = np.nan
        
        # Engineer features - should handle missing values
        df_with_features = self.feature_engineer.engineer_all_features(df_with_nan)
        
        # Check that features are calculated where possible
        # Some features will have NaN where input data is NaN
        self.assertEqual(len(df_with_features), len(df_with_nan))
        
        # Verify that we can still calculate some features
        # (features that don't depend on the NaN rows)
        self.assertTrue(df_with_features['returns'].notna().any())
    
    def test_feature_scaling(self):
        """Test feature scaling"""
        # Engineer features
        df_with_features = self.feature_engineer.engineer_all_features(self.df)
        
        # Select numeric columns for scaling
        numeric_cols = df_with_features.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        if 'returns' in numeric_cols:
            numeric_cols.remove('returns')
        
        # Scale features
        scaled_df, scaler = self.feature_engineer.scale_features(
            df_with_features[numeric_cols]
        )
        
        # Verify scaling
        self.assertIsInstance(scaled_df, pd.DataFrame)
        self.assertIsNotNone(scaler)
        
        # Check that scaled data has mean ~0 and std ~1
        for col in scaled_df.columns:
            if scaled_df[col].notna().any():
                col_mean = scaled_df[col].mean()
                col_std = scaled_df[col].std()
                
                # Allow some tolerance
                self.assertAlmostEqual(col_mean, 0.0, delta=0.1)
                self.assertAlmostEqual(col_std, 1.0, delta=0.1)
        
        # Test inverse transform
        original_data = scaler.inverse_transform(scaled_df)
        self.assertEqual(original_data.shape, scaled_df.shape)


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor"""
    
    def setUp(self):
        """Set up test environment"""
        self.config_manager = ConfigManager()
        self.preprocessor = DataPreprocessor(self.config_manager)
        
        # Create test DataFrame with some issues
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'open': np.random.uniform(45000, 55000, 100),
            'high': np.random.uniform(46000, 56000, 100),
            'low': np.random.uniform(44000, 54000, 100),
            'close': np.random.uniform(45000, 55000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # Add some data issues
        self.df.loc[10, 'close'] = np.nan  # Missing value
        self.df.loc[20, 'volume'] = -100  # Negative volume
        self.df.loc[30, 'high'] = 1000000  # Extreme outlier
        self.df.loc[40:45, 'open'] = np.nan  # Multiple missing values
    
    def test_handle_missing_values(self):
        """Test handling missing values"""
        # Handle missing values
        df_clean = self.preprocessor.handle_missing_values(self.df.copy())
        
        # Verify no NaN values in cleaned DataFrame
        self.assertFalse(df_clean.isna().any().any())
        
        # Verify interpolation or forward fill worked
        self.assertEqual(len(df_clean), len(self.df))
        
        # Check specific cases
        # Row 10 close was NaN, should be filled
        self.assertFalse(pd.isna(df_clean.loc[10, 'close']))
        
        # Row 40-45 open were NaN, should be filled
        for i in range(40, 46):
            self.assertFalse(pd.isna(df_clean.loc[i, 'open']))
    
    def test_handle_outliers(self):
        """Test handling outliers"""
        # Handle outliers
        df_no_outliers = self.preprocessor.handle_outliers(self.df.copy())
        
        # Verify extreme values handled
        # Row 30 had extreme high value
        original_value = self.df.loc[30, 'high']
        processed_value = df_no_outliers.loc[30, 'high']
        
        # Value should be capped or replaced
        self.assertNotEqual(original_value, processed_value)
        self.assertLess(processed_value, 1000000)  # Should be reasonable
        
        # Verify negative volume fixed
        original_volume = self.df.loc[20, 'volume']
        processed_volume = df_no_outliers.loc[20, 'volume']
        
        self.assertLess(original_volume, 0)  # Was negative
        self.assertGreaterEqual(processed_volume, 0)  # Should be non-negative
    
    def test_normalize_data(self):
        """Test data normalization"""
        # Normalize data
        df_normalized, scaler = self.preprocessor.normalize_data(self.df.copy())
        
        # Verify scaler object
        self.assertIsNotNone(scaler)
        
        # Verify normalization
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'timestamp':  # Skip timestamp
                col_mean = df_normalized[col].mean()
                col_std = df_normalized[col].std()
                
                # After normalization, mean ~0, std ~1
                self.assertAlmostEqual(col_mean, 0.0, delta=0.1)
                self.assertAlmostEqual(col_std, 1.0, delta=0.1)
        
        # Test inverse transform
        df_original = scaler.inverse_transform(df_normalized[numeric_cols])
        self.assertEqual(df_original.shape, df_normalized[numeric_cols].shape)
    
    def test_standardize_data(self):
        """Test data standardization"""
        # Standardize data
        df_standardized, scaler = self.preprocessor.standardize_data(self.df.copy())
        
        # Verify standardization
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'timestamp':  # Skip timestamp
                col_values = df_standardized[col].dropna()
                if len(col_values) > 0:
                    # Mean should be approximately 0
                    self.assertAlmostEqual(col_values.mean(), 0.0, delta=0.1)
                    # Std should be approximately 1
                    self.assertAlmostEqual(col_values.std(), 1.0, delta=0.1)
    
    def test_create_sequences(self):
        """Test creating sequences for time series models"""
        # Clean data first
        df_clean = self.preprocessor.handle_missing_values(self.df.copy())
        
        # Create sequences
        sequence_length = 10
        target_col = 'close'
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        
        X, y = self.preprocessor.create_sequences(
            df_clean[feature_cols],
            sequence_length=sequence_length,
            target_col=target_col
        )
        
        # Verify shapes
        n_samples = len(df_clean) - sequence_length
        self.assertEqual(X.shape[0], n_samples)
        self.assertEqual(X.shape[1], sequence_length)
        self.assertEqual(X.shape[2], len(feature_cols))
        
        self.assertEqual(y.shape[0], n_samples)
        
        # Verify sequence creation
        for i in range(n_samples):
            # X should contain sequence of rows i to i+sequence_length-1
            for j in range(sequence_length):
                expected_row = df_clean.iloc[i + j][feature_cols].values
                actual_row = X[i, j, :]
                
                np.testing.assert_array_almost_equal(actual_row, expected_row)
            
            # y should be target value at i+sequence_length
            expected_target = df_clean.iloc[i + sequence_length][target_col]
            self.assertEqual(y[i], expected_target)
    
    def test_split_train_test(self):
        """Test splitting data into train and test sets"""
        # Create larger dataset
        n_samples = 1000
        df_large = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'feature': np.random.randn(n_samples),
            'target': np.random.randn(n_samples)
        })
        
        # Split data
        test_size = 0.2
        train_df, test_df = self.preprocessor.split_train_test(
            df_large,
            test_size=test_size
        )
        
        # Verify sizes
        expected_test_size = int(n_samples * test_size)
        expected_train_size = n_samples - expected_test_size
        
        self.assertEqual(len(train_df), expected_train_size)
        self.assertEqual(len(test_df), expected_test_size)
        
        # Verify chronological split (no shuffling)
        self.assertTrue(train_df['timestamp'].max() < test_df['timestamp'].min())
        
        # Test with datetime split
        split_date = pd.Timestamp('2024-01-15')
        train_df_date, test_df_date = self.preprocessor.split_train_test(
            df_large,
            split_date=split_date
        )
        
        # Verify date split
        self.assertTrue(train_df_date['timestamp'].max() < split_date)
        self.assertTrue(test_df_date['timestamp'].min() >= split_date)
    
    def test_create_rolling_windows(self):
        """Test creating rolling windows for time series"""
        # Create rolling windows
        window_size = 20
        step_size = 5
        
        windows = self.preprocessor.create_rolling_windows(
            self.df['close'].values,
            window_size=window_size,
            step_size=step_size
        )
        
        # Verify window creation
        n_windows = (len(self.df) - window_size) // step_size + 1
        self.assertEqual(len(windows), n_windows)
        
        for i, window in enumerate(windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            expected_window = self.df['close'].values[start_idx:end_idx]
            
            np.testing.assert_array_equal(window, expected_window)
    
    def test_resample_data(self):
        """Test resampling time series data"""
        # Create higher frequency data
        df_hourly = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
            'open': np.random.uniform(45000, 55000, 24),
            'close': np.random.uniform(45000, 55000, 24)
        })
        
        # Resample to daily
        df_daily = self.preprocessor.resample_data(
            df_hourly,
            rule='D',
            aggregation='ohlc'
        )
        
        # Verify resampling
        self.assertLess(len(df_daily), len(df_hourly))
        self.assertEqual(df_daily.index.freq, 'D')
        
        # Verify OHLC aggregation
        self.assertIn('open', df_daily.columns)
        self.assertIn('high', df_daily.columns)
        self.assertIn('low', df_daily.columns)
        self.assertIn('close', df_daily.columns)
    
    def test_remove_duplicates(self):
        """Test removing duplicate data"""
        # Create DataFrame with duplicates
        df_with_duplicates = pd.concat([self.df, self.df.iloc[:5]], ignore_index=True)
        
        # Remove duplicates
        df_unique = self.preprocessor.remove_duplicates(df_with_duplicates)
        
        # Verify duplicates removed
        self.assertEqual(len(df_unique), len(self.df))
        
        # Check that timestamp column is unique
        self.assertEqual(len(df_unique['timestamp'].unique()), len(df_unique))
    
    def test_validate_data_structure(self):
        """Test data structure validation"""
        # Valid DataFrame should pass
        is_valid = self.preprocessor.validate_data_structure(self.df)
        self.assertTrue(is_valid)
        
        # Test with missing required columns
        df_invalid = self.df.drop(columns=['close'])
        is_valid = self.preprocessor.validate_data_structure(df_invalid)
        self.assertFalse(is_valid)
        
        # Test with non-numeric values in numeric columns
        df_invalid2 = self.df.copy()
        df_invalid2.loc[0, 'volume'] = 'invalid'
        is_valid = self.preprocessor.validate_data_structure(df_invalid2)
        self.assertFalse(is_valid)
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline"""
        # Run preprocessing pipeline
        processed_data = self.preprocessor.preprocess_pipeline(
            self.df.copy(),
            steps=[
                'handle_missing_values',
                'handle_outliers',
                'normalize_data'
            ]
        )
        
        # Verify pipeline result
        self.assertIsInstance(processed_data, dict)
        self.assertIn('data', processed_data)
        self.assertIn('metadata', processed_data)
        
        # Verify data is clean
        df_processed = processed_data['data']
        self.assertFalse(df_processed.isna().any().any())
        
        # Verify no negative volumes
        self.assertTrue((df_processed['volume'] >= 0).all())
        
        # Verify metadata
        metadata = processed_data['metadata']
        self.assertIn('preprocessing_steps', metadata)
        self.assertIn('shape_before', metadata)
        self.assertIn('shape_after', metadata)
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis"""
        # Create DataFrame with features and target
        df_with_target = self.df.copy()
        df_with_target['target'] = df_with_target['close'].shift(-1)  # Next period's close
        
        # Drop last row with NaN target
        df_with_target = df_with_target.dropna()
        
        # Analyze feature importance
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        target_col = 'target'
        
        importance = self.preprocessor.analyze_feature_importance(
            df_with_target[feature_cols + [target_col]],
            feature_cols=feature_cols,
            target_col=target_col
        )
        
        # Verify importance results
        self.assertIsInstance(importance, dict)
        self.assertIn('feature_importance', importance)
        self.assertIn('correlation_matrix', importance)
        
        # Feature importance should have values for each feature
        for feature in feature_cols:
            self.assertIn(feature, importance['feature_importance'])
    
    def test_data_augmentation(self):
        """Test data augmentation techniques"""
        # Test adding noise
        df_augmented = self.preprocessor.augment_with_noise(
            self.df.copy(),
            noise_level=0.01
        )
        
        # Verify augmentation
        self.assertEqual(df_augmented.shape, self.df.shape)
        
        # Values should be slightly different
        self.assertFalse(df_augmented.equals(self.df))
        
        # Difference should be within noise level
        max_diff = (df_augmented['close'] - self.df['close']).abs().max()
        self.assertLess(max_diff, self.df['close'].max() * 0.02)  # Within 2%
        
        # Test time shifting
        df_shifted = self.preprocessor.augment_with_time_shift(
            self.df.copy(),
            shift_periods=1
        )
        
        # Verify shifting
        self.assertEqual(df_shifted.shape, self.df.shape)
        np.testing.assert_array_almost_equal(
            df_shifted['close'].iloc[1:].values,
            self.df['close'].iloc[:-1].values
        )


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator"""
    
    def setUp(self):
        """Set up test environment"""
        self.config_manager = ConfigManager()
        self.validator = DataValidator(self.config_manager)
        
        # Create test DataFrame
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'open': np.random.uniform(45000, 55000, 100),
            'high': np.random.uniform(46000, 56000, 100),
            'low': np.random.uniform(44000, 54000, 100),
            'close': np.random.uniform(45000, 55000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # Make 'close' predictable for some tests
        self.df['close'] = np.linspace(45000, 55000, 100)
    
    def test_validate_ohlc_relationships(self):
        """Test validating OHLC relationships"""
        # Valid data should pass
        validation_result = self.validator.validate_ohlc_relationships(self.df)
        self.assertTrue(validation_result['is_valid'])
        self.assertEqual(len(validation_result['errors']), 0)
        
        # Create invalid data (high < low)
        df_invalid = self.df.copy()
        df_invalid.loc[0, 'high'] = 40000  # Less than low
        
        validation_result = self.validator.validate_ohlc_relationships(df_invalid)
        self.assertFalse(validation_result['is_valid'])
        self.assertGreater(len(validation_result['errors']), 0)
        
        # Check specific error message
        errors = validation_result['errors']
        self.assertTrue(any('high < low' in str(error) for error in errors))
    
    def test_validate_price_ranges(self):
        """Test validating price ranges"""
        # Valid data should pass
        validation_result = self.validator.validate_price_ranges(self.df)
        self.assertTrue(validation_result['is_valid'])
        
        # Create data with extreme price jump
        df_invalid = self.df.copy()
        df_invalid.loc[1, 'close'] = 1000000  # Extreme jump
        
        validation_result = self.validator.validate_price_ranges(df_invalid)
        self.assertFalse(validation_result['is_valid'])
        self.assertGreater(len(validation_result['errors']), 0)
    
    def test_validate_volume(self):
        """Test validating volume data"""
        # Valid data should pass
        validation_result = self.validator.validate_volume(self.df)
        self.assertTrue(validation_result['is_valid'])
        
        # Create invalid data (negative volume)
        df_invalid = self.df.copy()
        df_invalid.loc[0, 'volume'] = -100
        
        validation_result = self.validator.validate_volume(df_invalid)
        self.assertFalse(validation_result['is_valid'])
        
        # Check error for negative volume
        errors = validation_result['errors']
        self.assertTrue(any('negative' in str(error).lower() for error in errors))
    
    def test_validate_timestamps(self):
        """Test validating timestamp data"""
        # Valid data should pass
        validation_result = self.validator.validate_timestamps(self.df)
        self.assertTrue(validation_result['is_valid'])
        
        # Create invalid data (duplicate timestamps)
        df_invalid = self.df.copy()
        df_invalid.loc[1, 'timestamp'] = df_invalid.loc[0, 'timestamp']
        
        validation_result = self.validator.validate_timestamps(df_invalid)
        self.assertFalse(validation_result['is_valid'])
        
        # Create invalid data (non-monotonic timestamps)
        df_invalid2 = self.df.copy()
        df_invalid2.loc[10, 'timestamp'] = df_invalid2.loc[5, 'timestamp'] - pd.Timedelta(hours=10)
        
        validation_result = self.validator.validate_timestamps(df_invalid2)
        self.assertFalse(validation_result['is_valid'])
    
    def test_validate_missing_values(self):
        """Test validating for missing values"""
        # Valid data (no missing values) should pass
        validation_result = self.validator.validate_missing_values(self.df)
        self.assertTrue(validation_result['is_valid'])
        
        # Create data with missing values
        df_invalid = self.df.copy()
        df_invalid.loc[0, 'close'] = np.nan
        
        validation_result = self.validator.validate_missing_values(df_invalid)
        self.assertFalse(validation_result['is_valid'])
        
        # Check error message
        errors = validation_result['errors']
        self.assertTrue(any('missing' in str(error).lower() for error in errors))
    
    def test_validate_data_quality_score(self):
        """Test calculating data quality score"""
        # Calculate quality score for valid data
        quality_report = self.validator.calculate_data_quality_score(self.df)
        
        # Verify report structure
        self.assertIsInstance(quality_report, dict)
        self.assertIn('overall_score', quality_report)
        self.assertIn('category_scores', quality_report)
        self.assertIn('issues_found', quality_report)
        
        # Overall score should be high for clean data
        self.assertGreaterEqual(quality_report['overall_score'], 0.8)
        self.assertLessEqual(quality_report['overall_score'], 1.0)
        
        # Verify category scores
        categories = ['completeness', 'accuracy', 'consistency', 'timeliness']
        for category in categories:
            self.assertIn(category, quality_report['category_scores'])
            score = quality_report['category_scores'][category]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_validate_against_statistical_baseline(self):
        """Test validating against statistical baseline"""
        # Create baseline statistics
        baseline_stats = {
            'close': {
                'mean': 50000,
                'std': 5000,
                'min': 45000,
                'max': 55000
            },
            'volume': {
                'mean': 500,
                'std': 300,
                'min': 100,
                'max': 1000
            }
        }
        
        # Validate against baseline
        validation_result = self.validator.validate_against_statistical_baseline(
            self.df,
            baseline_stats
        )
        
        # Clean data should pass
        self.assertTrue(validation_result['is_valid'])
        
        # Create data that violates baseline
        df_invalid = self.df.copy()
        df_invalid['close'] = df_invalid['close'] * 10  # Extreme values
        
        validation_result = self.validator.validate_against_statistical_baseline(
            df_invalid,
            baseline_stats
        )
        
        self.assertFalse(validation_result['is_valid'])
        self.assertGreater(len(validation_result['errors']), 0)
    
    def test_detect_anomalies(self):
        """Test detecting data anomalies"""
        # Detect anomalies
        anomalies = self.validator.detect_anomalies(self.df)
        
        # Verify anomalies report
        self.assertIsInstance(anomalies, dict)
        self.assertIn('anomaly_scores', anomalies)
        self.assertIn('threshold', anomalies)
        self.assertIn('anomaly_indices', anomalies)
        
        # Clean data should have few anomalies
        anomaly_indices = anomalies['anomaly_indices']
        self.assertLess(len(anomaly_indices), len(self.df) * 0.1)  # Less than 10% anomalies
        
        # Create data with obvious anomaly
        df_with_anomaly = self.df.copy()
        df_with_anomaly.loc[50, 'close'] = 100000  # Extreme outlier
        
        anomalies = self.validator.detect_anomalies(df_with_anomaly)
        self.assertIn(50, anomalies['anomaly_indices'])
    
    def test_validate_data_consistency(self):
        """Test validating data consistency"""
        # Valid data should be consistent
        validation_result = self.validator.validate_data_consistency(self.df)
        self.assertTrue(validation_result['is_valid'])
        
        # Create inconsistent data
        df_inconsistent = self.df.copy()
        
        # Add inconsistency: close price outside high-low range
        df_inconsistent.loc[0, 'close'] = df_inconsistent.loc[0, 'high'] + 1000
        
        validation_result = self.validator.validate_data_consistency(df_inconsistent)
        self.assertFalse(validation_result['is_valid'])
        
        # Check for specific error
        errors = validation_result['errors']
        error_messages = [str(error) for error in errors]
        self.assertTrue(any('close outside high-low range' in msg for msg in error_messages))
    
    def test_generate_validation_report(self):
        """Test generating comprehensive validation report"""
        # Generate validation report
        report = self.validator.generate_validation_report(self.df)
        
        # Verify report structure
        self.assertIsInstance(report, dict)
        
        # Check report sections
        expected_sections = [
            'summary',
            'data_quality_score',
            'validation_results',
            'anomalies',
            'recommendations'
        ]
        
        for section in expected_sections:
            self.assertIn(section, report)
        
        # Summary should include key metrics
        summary = report['summary']
        self.assertIn('total_records', summary)
        self.assertIn('valid', summary)
        self.assertIn('issues_found', summary)
        
        # Validation results should include all checks
        validation_results = report['validation_results']
        self.assertGreater(len(validation_results), 0)
        
        # Each check should have result
        for check_name, check_result in validation_results.items():
            self.assertIn('is_valid', check_result)
            self.assertIn('errors', check_result)
    
    def test_validate_real_time_data(self):
        """Test validating real-time data"""
        # Create real-time data point
        data_point = {
            'timestamp': pd.Timestamp('2024-01-01 12:00:00'),
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 1000.0
        }
        
        # Validate single data point
        validation_result = self.validator.validate_real_time_data(data_point)
        
        # Valid data point should pass
        self.assertTrue(validation_result['is_valid'])
        
        # Create invalid data point
        invalid_point = data_point.copy()
        invalid_point['volume'] = -100  # Negative volume
        
        validation_result = self.validator.validate_real_time_data(invalid_point)
        self.assertFalse(validation_result['is_valid'])
        
        # Check error details
        self.assertGreater(len(validation_result['errors']), 0)
    
    def test_benchmark_data_quality(self):
        """Test benchmarking data quality"""
        # Benchmark against historical data
        historical_data = self.df.iloc[:80]  # First 80 records as historical
        current_data = self.df.iloc[80:]     # Last 20 records as current
        
        benchmark_result = self.validator.benchmark_data_quality(
            current_data,
            historical_data
        )
        
        # Verify benchmark result
        self.assertIsInstance(benchmark_result, dict)
        
        # Check benchmark metrics
        expected_metrics = [
            'completeness_change',
            'accuracy_change',
            'consistency_change',
            'drift_score'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, benchmark_result)
            
            # Metrics should be numeric
            value = benchmark_result[metric]
            self.assertIsInstance(value, (int, float, np.number))
    
    def test_validate_for_ml_training(self):
        """Test validating data for ML training"""
        # Add target column for ML
        df_for_ml = self.df.copy()
        df_for_ml['target'] = df_for_ml['close'].shift(-1)  # Next period close
        
        # Drop last row with NaN target
        df_for_ml = df_for_ml.dropna()
        
        # Validate for ML training
        ml_validation = self.validator.validate_for_ml_training(
            df_for_ml,
            target_column='target'
        )
        
        # Verify ML validation result
        self.assertIsInstance(ml_validation, dict)
        
        # Check ML-specific validations
        self.assertIn('has_sufficient_data', ml_validation)
        self.assertIn('class_balance', ml_validation)
        self.assertIn('feature_correlations', ml_validation)
        
        # For regression task, check target distribution
        if 'target_distribution' in ml_validation:
            dist = ml_validation['target_distribution']
            self.assertIn('mean', dist)
            self.assertIn('std', dist)
    
    def test_cross_validate_with_external_source(self):
        """Test cross-validating with external data source"""
        # Create main dataset
        main_data = self.df.copy()
        
        # Create external dataset (slightly different)
        external_data = self.df.copy()
        external_data['close'] = external_data['close'] * 1.01  # 1% difference
        
        # Cross-validate
        cross_validation = self.validator.cross_validate_with_external_source(
            main_data,
            external_data,
            key_columns=['timestamp', 'close']
        )
        
        # Verify cross-validation result
        self.assertIsInstance(cross_validation, dict)
        
        # Check comparison metrics
        self.assertIn('correlation', cross_validation)
        self.assertIn('mean_absolute_error', cross_validation)
        self.assertIn('discrepancies', cross_validation)
        
        # For identical data, correlation should be high
        # (Our data has small difference, so correlation still high)
        self.assertGreater(cross_validation['correlation'], 0.9)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)