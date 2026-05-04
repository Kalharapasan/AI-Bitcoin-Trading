"""
Helpers module for Bitcoin trading AI.
Provides utility functions, common operations, and helper classes
used throughout the trading system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
import json
import pickle
import joblib
from pathlib import Path
import hashlib
import asyncio
from collections import deque, defaultdict
import uuid
import time
import traceback
import random
import string
import math
import inspect
import functools
import itertools
import re
import hashlib
import base64
import csv
import io
import os
import sys
import threading
import concurrent.futures
from contextlib import contextmanager
from decimal import Decimal, ROUND_HALF_UP

# Import project modules
from config.settings import SystemSettings, TradingSettings
from config.config_manager import get_config
from core.utils.logger import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Math & Statistics Helpers ============
class MathHelper:
    """Mathematical and statistical helper functions"""
    
    @staticmethod
    def normalize(values: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize values using various methods"""
        
        if len(values) == 0:
            return values
        
        if method == 'minmax':
            min_val = np.min(values)
            max_val = np.max(values)
            if max_val - min_val == 0:
                return np.zeros_like(values)
            return (values - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean = np.mean(values)
            std = np.std(values)
            if std == 0:
                return np.zeros_like(values)
            return (values - mean) / std
        
        elif method == 'robust':
            median = np.median(values)
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                return np.zeros_like(values)
            return (values - median) / iqr
        
        elif method == 'decimal':
            max_abs = np.max(np.abs(values))
            if max_abs == 0:
                return values
            return values / max_abs
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def denormalize(normalized_values: np.ndarray, 
                   original_values: np.ndarray,
                   method: str = 'minmax') -> np.ndarray:
        """Denormalize values back to original scale"""
        
        if len(normalized_values) == 0:
            return normalized_values
        
        if method == 'minmax':
            min_val = np.min(original_values)
            max_val = np.max(original_values)
            return normalized_values * (max_val - min_val) + min_val
        
        elif method == 'zscore':
            mean = np.mean(original_values)
            std = np.std(original_values)
            return normalized_values * std + mean
        
        elif method == 'robust':
            median = np.median(original_values)
            q75, q25 = np.percentile(original_values, [75, 25])
            iqr = q75 - q25
            return normalized_values * iqr + median
        
        elif method == 'decimal':
            max_abs = np.max(np.abs(original_values))
            return normalized_values * max_abs
        
        else:
            raise ValueError(f"Unknown denormalization method: {method}")
    
    @staticmethod
    def calculate_returns(prices: np.ndarray, method: str = 'log') -> np.ndarray:
        """Calculate returns from price series"""
        
        if len(prices) < 2:
            return np.array([])
        
        if method == 'log':
            returns = np.diff(np.log(prices))
        elif method == 'simple':
            returns = np.diff(prices) / prices[:-1]
        elif method == 'percentage':
            returns = (np.diff(prices) / prices[:-1]) * 100
        else:
            raise ValueError(f"Unknown return calculation method: {method}")
        
        return returns
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, 
                              risk_free_rate: float = 0.0,
                              periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess == 0:
            return 0.0
        
        return (mean_excess / std_excess) * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_max_drawdown(prices: np.ndarray) -> float:
        """Calculate maximum drawdown from price series"""
        
        if len(prices) < 2:
            return 0.0
        
        cumulative = prices
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)
    
    @staticmethod
    def calculate_volatility(returns: np.ndarray, 
                            annualized: bool = True,
                            periods_per_year: int = 252) -> float:
        """Calculate volatility"""
        
        if len(returns) < 2:
            return 0.0
        
        volatility = np.std(returns)
        
        if annualized:
            volatility *= np.sqrt(periods_per_year)
        
        return volatility
    
    @staticmethod
    def calculate_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate correlation between two series"""
        
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        correlation = np.corrcoef(x, y)[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        return correlation
    
    @staticmethod
    def calculate_ema(values: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        
        if len(values) < period:
            return np.full_like(values, np.nan)
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(values, dtype=float)
        ema[:period] = np.nan
        ema[period-1] = np.mean(values[:period])
        
        for i in range(period, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @staticmethod
    def calculate_sma(values: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        
        if len(values) < period:
            return np.full_like(values, np.nan)
        
        sma = np.zeros_like(values, dtype=float)
        sma[:period-1] = np.nan
        
        for i in range(period-1, len(values)):
            sma[i] = np.mean(values[i-period+1:i+1])
        
        return sma
    
    @staticmethod
    def calculate_bollinger_bands(prices: np.ndarray, 
                                 period: int = 20,
                                 num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        
        sma = MathHelper.calculate_sma(prices, period)
        
        upper_band = np.zeros_like(prices)
        lower_band = np.zeros_like(prices)
        
        for i in range(period-1, len(prices)):
            start_idx = i - period + 1
            window = prices[start_idx:i+1]
            std = np.std(window)
            
            upper_band[i] = sma[i] + (num_std * std)
            lower_band[i] = sma[i] - (num_std * std)
        
        upper_band[:period-1] = np.nan
        lower_band[:period-1] = np.nan
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        
        if len(prices) < period + 1:
            return np.full_like(prices, np.nan)
        
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # Initial averages
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        # Calculate remaining averages
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        
        # Calculate RSI
        rs = avg_gain / np.maximum(avg_loss, 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        rsi[:period] = np.nan
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices: np.ndarray, 
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD indicator"""
        
        fast_ema = MathHelper.calculate_ema(prices, fast_period)
        slow_ema = MathHelper.calculate_ema(prices, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = MathHelper.calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

# ============ Data Processing Helpers ============
class DataHelper:
    """Data processing and manipulation helper functions"""
    
    @staticmethod
    def create_lagged_features(data: pd.DataFrame, 
                              columns: List[str],
                              lags: List[int],
                              fill_method: str = 'ffill') -> pd.DataFrame:
        """Create lagged features for time series data"""
        
        result = data.copy()
        
        for column in columns:
            if column not in data.columns:
                continue
            
            for lag in lags:
                lagged_name = f"{column}_lag_{lag}"
                result[lagged_name] = data[column].shift(lag)
        
        # Handle NaN values
        if fill_method == 'ffill':
            result = result.ffill()
        elif fill_method == 'bfill':
            result = result.bfill()
        elif fill_method == 'fillna':
            result = result.fillna(0)
        elif fill_method == 'drop':
            result = result.dropna()
        
        return result
    
    @staticmethod
    def create_rolling_features(data: pd.DataFrame,
                               columns: List[str],
                               windows: List[int],
                               functions: List[str] = ['mean', 'std', 'min', 'max'],
                               fill_method: str = 'ffill') -> pd.DataFrame:
        """Create rolling window features"""
        
        result = data.copy()
        
        for column in columns:
            if column not in data.columns:
                continue
            
            for window in windows:
                for func in functions:
                    feature_name = f"{column}_rolling_{window}_{func}"
                    
                    if func == 'mean':
                        result[feature_name] = data[column].rolling(window=window).mean()
                    elif func == 'std':
                        result[feature_name] = data[column].rolling(window=window).std()
                    elif func == 'min':
                        result[feature_name] = data[column].rolling(window=window).min()
                    elif func == 'max':
                        result[feature_name] = data[column].rolling(window=window).max()
                    elif func == 'median':
                        result[feature_name] = data[column].rolling(window=window).median()
                    elif func == 'sum':
                        result[feature_name] = data[column].rolling(window=window).sum()
        
        # Handle NaN values
        if fill_method == 'ffill':
            result = result.ffill()
        elif fill_method == 'bfill':
            result = result.bfill()
        elif fill_method == 'fillna':
            result = result.fillna(0)
        elif fill_method == 'drop':
            result = result.dropna()
        
        return result
    
    @staticmethod
    def create_time_features(data: pd.DataFrame,
                            datetime_column: str = 'timestamp') -> pd.DataFrame:
        """Create time-based features from datetime column"""
        
        result = data.copy()
        
        if datetime_column not in data.columns:
            logger.warning(f"Datetime column {datetime_column} not found")
            return result
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(result[datetime_column]):
            result[datetime_column] = pd.to_datetime(result[datetime_column])
        
        # Extract time features
        result['hour'] = result[datetime_column].dt.hour
        result['day_of_week'] = result[datetime_column].dt.dayofweek
        result['day_of_month'] = result[datetime_column].dt.day
        result['week_of_year'] = result[datetime_column].dt.isocalendar().week
        result['month'] = result[datetime_column].dt.month
        result['quarter'] = result[datetime_column].dt.quarter
        result['year'] = result[datetime_column].dt.year
        
        # Cyclical encoding for periodic features
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        
        result['day_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        return result
    
    @staticmethod
    def handle_missing_values(data: pd.DataFrame,
                             method: str = 'interpolate',
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle missing values in dataframe"""
        
        result = data.copy()
        
        if columns is None:
            columns = result.columns.tolist()
        
        for column in columns:
            if column not in result.columns:
                continue
            
            if method == 'drop':
                result = result.dropna(subset=[column])
            elif method == 'ffill':
                result[column] = result[column].ffill()
            elif method == 'bfill':
                result[column] = result[column].bfill()
            elif method == 'mean':
                mean_val = result[column].mean()
                result[column] = result[column].fillna(mean_val)
            elif method == 'median':
                median_val = result[column].median()
                result[column] = result[column].fillna(median_val)
            elif method == 'mode':
                mode_val = result[column].mode()[0] if not result[column].mode().empty else 0
                result[column] = result[column].fillna(mode_val)
            elif method == 'interpolate':
                result[column] = result[column].interpolate(method='linear')
            elif method == 'zero':
                result[column] = result[column].fillna(0)
            else:
                raise ValueError(f"Unknown missing value handling method: {method}")
        
        return result
    
    @staticmethod
    def detect_outliers(data: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """Detect outliers in dataframe"""
        
        result = data.copy()
        
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_mask = pd.Series(False, index=result.index)
        
        for column in columns:
            if column not in result.columns:
                continue
            
            if method == 'iqr':
                # Interquartile Range method
                Q1 = result[column].quantile(0.25)
                Q3 = result[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                column_outliers = (result[column] < lower_bound) | (result[column] > upper_bound)
                outlier_mask = outlier_mask | column_outliers
            
            elif method == 'zscore':
                # Z-score method
                mean = result[column].mean()
                std = result[column].std()
                
                if std == 0:
                    continue
                
                z_scores = np.abs((result[column] - mean) / std)
                column_outliers = z_scores > threshold
                outlier_mask = outlier_mask | column_outliers
            
            elif method == 'modified_zscore':
                # Modified Z-score method (more robust)
                median = result[column].median()
                mad = np.median(np.abs(result[column] - median))
                
                if mad == 0:
                    mad = 1.253314 * result[column].std()  # Approximate for normal distribution
                
                modified_z_scores = 0.6745 * (result[column] - median) / mad
                column_outliers = np.abs(modified_z_scores) > threshold
                outlier_mask = outlier_mask | column_outliers
        
        result['is_outlier'] = outlier_mask
        
        return result
    
    @staticmethod
    def remove_outliers(data: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from dataframe"""
        
        data_with_outliers = DataHelper.detect_outliers(data, columns, method, threshold)
        return data_with_outliers[~data_with_outliers['is_outlier']].drop(columns=['is_outlier'])
    
    @staticmethod
    def cap_outliers(data: pd.DataFrame,
                    columns: Optional[List[str]] = None,
                    method: str = 'iqr',
                    threshold: float = 1.5) -> pd.DataFrame:
        """Cap outliers at threshold boundaries"""
        
        result = data.copy()
        
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if column not in result.columns:
                continue
            
            if method == 'iqr':
                Q1 = result[column].quantile(0.25)
                Q3 = result[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                result[column] = result[column].clip(lower=lower_bound, upper=upper_bound)
            
            elif method == 'percentile':
                lower_bound = result[column].quantile(0.01)
                upper_bound = result[column].quantile(0.99)
                result[column] = result[column].clip(lower=lower_bound, upper=upper_bound)
        
        return result
    
    @staticmethod
    def calculate_feature_importance(X: pd.DataFrame,
                                    y: pd.Series,
                                    method: str = 'mutual_info') -> pd.DataFrame:
        """Calculate feature importance scores"""
        
        try:
            if method == 'mutual_info':
                from sklearn.feature_selection import mutual_info_regression
                scores = mutual_info_regression(X, y, random_state=42)
            
            elif method == 'f_regression':
                from sklearn.feature_selection import f_regression
                scores = f_regression(X, y)[0]
            
            elif method == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                scores = rf.feature_importances_
            
            elif method == 'xgb':
                import xgboost as xgb
                xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                xgb_model.fit(X, y)
                scores = xgb_model.feature_importances_
            
            elif method == 'lasso':
                from sklearn.linear_model import Lasso
                lasso = Lasso(alpha=0.01, random_state=42)
                lasso.fit(X, y)
                scores = np.abs(lasso.coef_)
            
            else:
                raise ValueError(f"Unknown feature importance method: {method}")
            
            # Create results dataframe
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': scores
            }).sort_values('importance', ascending=False)
            
            # Normalize importance scores to 0-100
            importance_df['importance_normalized'] = (
                importance_df['importance'] / importance_df['importance'].sum() * 100
            )
            
            return importance_df
            
        except ImportError as e:
            logger.error(f"Failed to calculate feature importance: {str(e)}")
            
            # Return dummy importance scores
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': np.ones(len(X.columns)) / len(X.columns),
                'importance_normalized': 100 / len(X.columns)
            })
            
            return importance_df
    
    @staticmethod
    def split_time_series(data: pd.DataFrame,
                         split_ratio: float = 0.8,
                         datetime_column: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split time series data preserving temporal order"""
        
        if datetime_column not in data.columns:
            raise ValueError(f"Datetime column {datetime_column} not found")
        
        # Sort by datetime
        data_sorted = data.sort_values(datetime_column).reset_index(drop=True)
        
        # Calculate split index
        split_idx = int(len(data_sorted) * split_ratio)
        
        train_data = data_sorted.iloc[:split_idx].reset_index(drop=True)
        test_data = data_sorted.iloc[split_idx:].reset_index(drop=True)
        
        return train_data, test_data
    
    @staticmethod
    def create_time_series_cv_splits(data: pd.DataFrame,
                                    n_splits: int = 5,
                                    test_size: int = 100,
                                    datetime_column: str = 'timestamp') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create time series cross-validation splits"""
        
        if datetime_column not in data.columns:
            raise ValueError(f"Datetime column {datetime_column} not found")
        
        # Sort by datetime
        data_sorted = data.sort_values(datetime_column).reset_index(drop=True)
        
        splits = []
        n_samples = len(data_sorted)
        
        # Ensure we have enough data
        if n_samples < test_size * n_splits:
            raise ValueError(f"Not enough data for {n_splits} splits with test_size {test_size}")
        
        for i in range(n_splits):
            test_end = n_samples - (i * test_size)
            test_start = test_end - test_size
            
            if test_start < 0:
                break
            
            test_data = data_sorted.iloc[test_start:test_end].reset_index(drop=True)
            train_data = data_sorted.iloc[:test_start].reset_index(drop=True)
            
            splits.append((train_data, test_data))
        
        return splits

# ============ Financial Calculation Helpers ============
class FinancialHelper:
    """Financial calculation helper functions"""
    
    @staticmethod
    def calculate_pnl(entry_price: float, 
                     exit_price: float, 
                     quantity: float,
                     side: str = 'long',
                     fees: float = 0.0) -> float:
        """Calculate Profit and Loss for a trade"""
        
        if side == 'long':
            pnl = (exit_price - entry_price) * quantity
        elif side == 'short':
            pnl = (entry_price - exit_price) * quantity
        else:
            raise ValueError(f"Unknown trade side: {side}")
        
        # Apply fees (assuming fees are percentage of trade value)
        entry_fee = entry_price * quantity * fees
        exit_fee = exit_price * quantity * fees
        total_fees = entry_fee + exit_fee
        
        return pnl - total_fees
    
    @staticmethod
    def calculate_returns(initial_value: float,
                         final_value: float,
                         period_days: int = 1) -> Dict[str, float]:
        """Calculate various return metrics"""
        
        returns = {}
        
        # Simple return
        returns['simple'] = (final_value - initial_value) / initial_value
        
        # Logarithmic return
        if initial_value > 0 and final_value > 0:
            returns['log'] = math.log(final_value / initial_value)
        else:
            returns['log'] = 0.0
        
        # Annualized return (if period is provided)
        if period_days > 0:
            returns['annualized'] = ((1 + returns['simple']) ** (365 / period_days)) - 1
        else:
            returns['annualized'] = returns['simple']
        
        return returns
    
    @staticmethod
    def calculate_position_size(account_balance: float,
                               risk_per_trade: float,
                               entry_price: float,
                               stop_loss_price: float,
                               side: str = 'long') -> float:
        """Calculate position size based on risk management"""
        
        if risk_per_trade <= 0 or risk_per_trade > 1:
            raise ValueError(f"Risk per trade must be between 0 and 1, got {risk_per_trade}")
        
        # Calculate risk per unit
        if side == 'long':
            risk_per_unit = entry_price - stop_loss_price
        elif side == 'short':
            risk_per_unit = stop_loss_price - entry_price
        else:
            raise ValueError(f"Unknown trade side: {side}")
        
        # Avoid division by zero or negative risk
        if risk_per_unit <= 0:
            return 0.0
        
        # Calculate position size
        risk_amount = account_balance * risk_per_trade
        position_size = risk_amount / abs(risk_per_unit)
        
        return position_size
    
    @staticmethod
    def calculate_var(returns: np.ndarray,
                     confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """Calculate Value at Risk"""
        
        if len(returns) == 0:
            return 0.0
        
        if method == 'historical':
            # Historical simulation
            var = np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # Parametric (Gaussian) approach
            mean = np.mean(returns)
            std = np.std(returns)
            
            if std == 0:
                return 0.0
            
            # Z-score for confidence level
            from scipy import stats
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + z_score * std
        
        elif method == 'modified':
            # Modified (Cornish-Fisher) approach
            mean = np.mean(returns)
            std = np.std(returns)
            skew = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            if std == 0:
                return 0.0
            
            # Cornish-Fisher expansion
            from scipy import stats
            z = stats.norm.ppf(1 - confidence_level)
            z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurtosis / 24 - (2*z**3 - 5*z) * skew**2 / 36
            var = mean + z_cf * std
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return var
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray,
                      confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        
        if len(returns) == 0:
            return 0.0
        
        var = FinancialHelper.calculate_var(returns, confidence_level, 'historical')
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return np.mean(tail_returns)
    
    @staticmethod
    def calculate_risk_metrics(returns: np.ndarray,
                              risk_free_rate: float = 0.0,
                              periods_per_year: int = 252) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        metrics = {}
        
        if len(returns) < 2:
            return metrics
        
        # Basic statistics
        metrics['mean'] = np.mean(returns)
        metrics['std'] = np.std(returns)
        metrics['skewness'] = stats.skew(returns) if len(returns) > 2 else 0
        metrics['kurtosis'] = stats.kurtosis(returns) if len(returns) > 2 else 0
        
        # Risk metrics
        metrics['sharpe_ratio'] = MathHelper.calculate_sharpe_ratio(
            returns, risk_free_rate, periods_per_year
        )
        
        # Sortino ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                metrics['sortino_ratio'] = (
                    (metrics['mean'] - risk_free_rate / periods_per_year) * np.sqrt(periods_per_year)
                ) / downside_std
        
        # Calmar ratio
        # Note: This requires cumulative returns/max drawdown calculation
        
        # VaR and CVaR
        metrics['var_95'] = FinancialHelper.calculate_var(returns, 0.95)
        metrics['cvar_95'] = FinancialHelper.calculate_cvar(returns, 0.95)
        metrics['var_99'] = FinancialHelper.calculate_var(returns, 0.99)
        metrics['cvar_99'] = FinancialHelper.calculate_cvar(returns, 0.99)
        
        # Information ratio (requires benchmark returns)
        # Would need benchmark returns as input
        
        # Maximum drawdown (requires prices, not returns)
        # Would need prices as input
        
        return metrics
    
    @staticmethod
    def calculate_technical_indicators(prices: pd.DataFrame,
                                      high_col: str = 'high',
                                      low_col: str = 'low',
                                      close_col: str = 'close',
                                      volume_col: str = 'volume') -> pd.DataFrame:
        """Calculate common technical indicators"""
        
        result = prices.copy()
        
        # Ensure we have required columns
        required_cols = [close_col]
        if any(col not in result.columns for col in required_cols):
            logger.warning("Missing required columns for technical indicators")
            return result
        
        # Calculate indicators
        close = result[close_col]
        
        # Moving averages
        result['sma_20'] = MathHelper.calculate_sma(close.values, 20)
        result['sma_50'] = MathHelper.calculate_sma(close.values, 50)
        result['sma_200'] = MathHelper.calculate_sma(close.values, 200)
        
        result['ema_12'] = MathHelper.calculate_ema(close.values, 12)
        result['ema_26'] = MathHelper.calculate_ema(close.values, 26)
        
        # MACD
        macd, signal, histogram = MathHelper.calculate_macd(close.values)
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_histogram'] = histogram
        
        # RSI
        result['rsi_14'] = MathHelper.calculate_rsi(close.values, 14)
        
        # Bollinger Bands
        if high_col in result.columns and low_col in result.columns and close_col in result.columns:
            high = result[high_col]
            low = result[low_col]
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            result['bb_upper'], result['bb_middle'], result['bb_lower'] = MathHelper.calculate_bollinger_bands(
                typical_price.values, 20, 2.0
            )
        
        # Volume indicators
        if volume_col in result.columns:
            volume = result[volume_col]
            
            # Volume moving average
            result['volume_sma_20'] = MathHelper.calculate_sma(volume.values, 20)
            
            # On Balance Volume (simplified)
            obv = np.zeros_like(volume)
            obv[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv[i] = obv[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv[i] = obv[i-1] - volume.iloc[i]
                else:
                    obv[i] = obv[i-1]
            
            result['obv'] = obv
        
        return result

# ============ Date & Time Helpers ============
class DateTimeHelper:
    """Date and time helper functions"""
    
    @staticmethod
    def parse_timestamp(timestamp: Union[str, int, float, datetime],
                       format: Optional[str] = None) -> datetime:
        """Parse timestamp from various formats"""
        
        if isinstance(timestamp, datetime):
            return timestamp
        
        elif isinstance(timestamp, (int, float)):
            # Assume Unix timestamp
            if timestamp > 1e10:  # Milliseconds
                return datetime.fromtimestamp(timestamp / 1000)
            else:  # Seconds
                return datetime.fromtimestamp(timestamp)
        
        elif isinstance(timestamp, str):
            if format:
                return datetime.strptime(timestamp, format)
            
            # Try common formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d',
                '%d/%m/%Y %H:%M:%S',
                '%m/%d/%Y %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
            
            # If all else fails, try dateutil (if available)
            try:
                from dateutil import parser
                return parser.parse(timestamp)
            except ImportError:
                raise ValueError(f"Unable to parse timestamp: {timestamp}")
        
        else:
            raise TypeError(f"Unsupported timestamp type: {type(timestamp)}")
    
    @staticmethod
    def to_unix_timestamp(dt: datetime, milliseconds: bool = True) -> int:
        """Convert datetime to Unix timestamp"""
        
        timestamp = dt.timestamp()
        
        if milliseconds:
            return int(timestamp * 1000)
        else:
            return int(timestamp)
    
    @staticmethod
    def format_timestamp(dt: datetime, format: str = '%Y-%m-%d %H:%M:%S') -> str:
        """Format datetime as string"""
        
        return dt.strftime(format)
    
    @staticmethod
    def get_market_hours(dt: datetime, market: str = 'crypto') -> Tuple[datetime, datetime]:
        """Get market opening and closing hours"""
        
        # Crypto markets are 24/7
        if market == 'crypto':
            start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Traditional market hours (simplified)
        elif market == 'nyse':
            # NYSE: 9:30 AM - 4:00 PM EST
            start = dt.replace(hour=9, minute=30, second=0, microsecond=0)
            end = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        
        elif market == 'forex':
            # Forex: 24/5 (closes Friday 5PM, opens Sunday 5PM EST)
            start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            # Check if it's weekend
            if dt.weekday() >= 5:  # Saturday or Sunday
                # Forex is closed on weekends
                start = None
                end = None
        
        else:
            raise ValueError(f"Unknown market: {market}")
        
        return start, end
    
    @staticmethod
    def is_market_open(dt: datetime, market: str = 'crypto') -> bool:
        """Check if market is open at given datetime"""
        
        if market == 'crypto':
            return True
        
        elif market == 'nyse':
            # NYSE is open Monday-Friday, 9:30 AM - 4:00 PM EST
            if dt.weekday() >= 5:  # Weekend
                return False
            
            hour = dt.hour
            minute = dt.minute
            
            # Before 9:30 AM or after 4:00 PM
            if hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
                return False
            
            return True
        
        elif market == 'forex':
            # Forex is closed on weekends
            if dt.weekday() >= 5:
                return False
            
            return True
        
        else:
            raise ValueError(f"Unknown market: {market}")
    
    @staticmethod
    def get_next_market_open(dt: datetime, market: str = 'crypto') -> datetime:
        """Get next market opening time"""
        
        if market == 'crypto':
            # Crypto markets are always open
            return dt
        
        elif market == 'nyse':
            current_dt = dt
            
            # Move to next day if after market close
            if current_dt.hour >= 16:
                current_dt += timedelta(days=1)
            
            # Skip weekends
            while current_dt.weekday() >= 5:
                current_dt += timedelta(days=1)
            
            # Set to market open time
            next_open = current_dt.replace(hour=9, minute=30, second=0, microsecond=0)
            
            # If we're already past market open today, use tomorrow
            if dt >= next_open and dt.weekday() < 5 and dt.hour < 16:
                return dt  # Market is currently open
            
            return next_open
        
        elif market == 'forex':
            current_dt = dt
            
            # Skip weekends
            while current_dt.weekday() >= 5:
                current_dt += timedelta(days=1)
            
            # Set to midnight
            next_open = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            
            return next_open
        
        else:
            raise ValueError(f"Unknown market: {market}")
    
    @staticmethod
    def calculate_time_difference(start: datetime, 
                                 end: datetime,
                                 unit: str = 'seconds') -> float:
        """Calculate time difference in specified unit"""
        
        diff = end - start
        
        if unit == 'seconds':
            return diff.total_seconds()
        elif unit == 'minutes':
            return diff.total_seconds() / 60
        elif unit == 'hours':
            return diff.total_seconds() / 3600
        elif unit == 'days':
            return diff.total_seconds() / 86400
        elif unit == 'weeks':
            return diff.total_seconds() / (86400 * 7)
        elif unit == 'months':
            return diff.days / 30.44  # Average month length
        elif unit == 'years':
            return diff.days / 365.25
        else:
            raise ValueError(f"Unknown time unit: {unit}")
    
    @staticmethod
    def generate_time_range(start: datetime,
                           end: datetime,
                           interval: str = '1h',
                           inclusive: bool = True) -> List[datetime]:
        """Generate time range with specified interval"""
        
        # Parse interval
        interval_match = re.match(r'(\d+)([smhdwMy])', interval)
        if not interval_match:
            raise ValueError(f"Invalid interval format: {interval}")
        
        quantity = int(interval_match.group(1))
        unit = interval_match.group(2)
        
        # Map unit to timedelta
        unit_map = {
            's': 'seconds',
            'm': 'minutes',
            'h': 'hours',
            'd': 'days',
            'w': 'weeks',
            'M': 'months',  # Approximate
            'y': 'years'    # Approximate
        }
        
        if unit in ['s', 'm', 'h', 'd', 'w']:
            delta_kwargs = {unit_map[unit]: quantity}
            delta = timedelta(**delta_kwargs)
        elif unit == 'M':
            # Approximate month as 30.44 days
            delta = timedelta(days=quantity * 30.44)
        elif unit == 'y':
            # Approximate year as 365.25 days
            delta = timedelta(days=quantity * 365.25)
        
        # Generate range
        current = start
        times = []
        
        while current <= end if inclusive else current < end:
            times.append(current)
            current += delta
        
        return times

# ============ File & I/O Helpers ============
class FileHelper:
    """File and I/O helper functions"""
    
    @staticmethod
    def read_json(filepath: Union[str, Path]) -> Any:
        """Read JSON file"""
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(data: Any, filepath: Union[str, Path], indent: int = 2):
        """Write data to JSON file"""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
    
    @staticmethod
    def read_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Read CSV file"""
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return pd.read_csv(filepath, **kwargs)
    
    @staticmethod
    def write_csv(data: pd.DataFrame, filepath: Union[str, Path], **kwargs):
        """Write DataFrame to CSV file"""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(filepath, **kwargs)
    
    @staticmethod
    def read_parquet(filepath: Union[str, Path]) -> pd.DataFrame:
        """Read Parquet file"""
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return pd.read_parquet(filepath)
    
    @staticmethod
    def write_parquet(data: pd.DataFrame, filepath: Union[str, Path], **kwargs):
        """Write DataFrame to Parquet file"""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_parquet(filepath, **kwargs)
    
    @staticmethod
    def read_pickle(filepath: Union[str, Path]) -> Any:
        """Read pickle file"""
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def write_pickle(data: Any, filepath: Union[str, Path]):
        """Write data to pickle file"""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def list_files(directory: Union[str, Path],
                  pattern: str = "*",
                  recursive: bool = False) -> List[Path]:
        """List files in directory matching pattern"""
        
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))
    
    @staticmethod
    def get_file_hash(filepath: Union[str, Path],
                     algorithm: str = 'md5',
                     chunk_size: int = 8192) -> str:
        """Calculate file hash"""
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        hash_func = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def get_data_hash(data: Any, algorithm: str = 'md5') -> str:
        """Calculate hash of data"""
        
        # Convert data to bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, (dict, list)):
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        elif isinstance(data, pd.DataFrame):
            data_bytes = data.to_csv(index=False).encode('utf-8')
        else:
            data_bytes = pickle.dumps(data)
        
        # Calculate hash
        hash_func = hashlib.new(algorithm)
        hash_func.update(data_bytes)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def backup_file(filepath: Union[str, Path],
                   backup_dir: Optional[Union[str, Path]] = None,
                   max_backups: int = 5):
        """Create backup of file"""
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Cannot backup non-existent file: {filepath}")
            return
        
        if backup_dir is None:
            backup_dir = filepath.parent / 'backups'
        
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        backup_path = backup_dir / backup_name
        
        # Copy file
        import shutil
        shutil.copy2(filepath, backup_path)
        
        # Clean up old backups
        backups = sorted(backup_dir.glob(f"{filepath.stem}_*{filepath.suffix}"))
        if len(backups) > max_backups:
            for old_backup in backups[:-max_backups]:
                old_backup.unlink()
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]):
        """Ensure directory exists"""
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

# ============ String & Encoding Helpers ============
class StringHelper:
    """String manipulation and encoding helper functions"""
    
    @staticmethod
    def generate_id(prefix: str = '', length: int = 8) -> str:
        """Generate unique ID"""
        
        random_part = uuid.uuid4().hex[:length]
        
        if prefix:
            return f"{prefix}_{random_part}"
        else:
            return random_part
    
    @staticmethod
    def generate_random_string(length: int = 10,
                              include_digits: bool = True,
                              include_letters: bool = True,
                              include_symbols: bool = False) -> str:
        """Generate random string"""
        
        characters = ''
        
        if include_letters:
            characters += string.ascii_letters
        
        if include_digits:
            characters += string.digits
        
        if include_symbols:
            characters += string.punctuation
        
        if not characters:
            raise ValueError("At least one character type must be included")
        
        return ''.join(random.choice(characters) for _ in range(length))
    
    @staticmethod
    def truncate_string(text: str, max_length: int, ellipsis: str = '...') -> str:
        """Truncate string to maximum length"""
        
        if len(text) <= max_length:
            return text
        
        if len(ellipsis) >= max_length:
            return ellipsis[:max_length]
        
        return text[:max_length - len(ellipsis)] + ellipsis
    
    @staticmethod
    def camel_to_snake(camel_str: str) -> str:
        """Convert camelCase to snake_case"""
        
        # Insert underscore before uppercase letters
        snake_str = re.sub(r'(?<!^)(?=[A-Z])', '_', camel_str)
        
        # Convert to lowercase
        return snake_str.lower()
    
    @staticmethod
    def snake_to_camel(snake_str: str) -> str:
        """Convert snake_case to camelCase"""
        
        # Split by underscore and capitalize each part except first
        parts = snake_str.split('_')
        
        if not parts:
            return ''
        
        camel_str = parts[0] + ''.join(part.capitalize() for part in parts[1:])
        
        return camel_str
    
    @staticmethod
    def base64_encode(data: Union[str, bytes]) -> str:
        """Encode data to base64"""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return base64.b64encode(data).decode('utf-8')
    
    @staticmethod
    def base64_decode(encoded: str) -> str:
        """Decode base64 encoded string"""
        
        decoded_bytes = base64.b64decode(encoded)
        return decoded_bytes.decode('utf-8')
    
    @staticmethod
    def format_number(number: float,
                     decimal_places: int = 2,
                     thousands_separator: bool = True) -> str:
        """Format number with specified decimal places and separators"""
        
        # Handle NaN and infinite values
        if np.isnan(number) or np.isinf(number):
            return str(number)
        
        # Format with specified decimal places
        if decimal_places >= 0:
            format_str = f".{decimal_places}f"
        else:
            format_str = "f"
        
        formatted = format(number, format_str)
        
        # Add thousands separators
        if thousands_separator and decimal_places >= 0:
            parts = formatted.split('.')
            integer_part = parts[0]
            
            # Add thousands separators
            integer_with_commas = ''
            for i, char in enumerate(reversed(integer_part)):
                if i > 0 and i % 3 == 0:
                    integer_with_commas = ',' + integer_with_commas
                integer_with_commas = char + integer_with_commas
            
            if len(parts) > 1:
                formatted = integer_with_commas + '.' + parts[1]
            else:
                formatted = integer_with_commas
        
        return formatted
    
    @staticmethod
    def format_percentage(value: float,
                         decimal_places: int = 2,
                         include_sign: bool = True) -> str:
        """Format number as percentage"""
        
        percentage = value * 100
        formatted = StringHelper.format_number(percentage, decimal_places, False)
        
        if include_sign:
            return f"{formatted}%"
        else:
            return formatted
    
    @staticmethod
    def format_currency(amount: float,
                       currency: str = 'USD',
                       decimal_places: int = 2) -> str:
        """Format number as currency"""
        
        formatted = StringHelper.format_number(amount, decimal_places, True)
        
        if currency == 'USD':
            return f"${formatted}"
        elif currency == 'EUR':
            return f"€{formatted}"
        elif currency == 'GBP':
            return f"£{formatted}"
        elif currency == 'JPY':
            return f"¥{formatted}"
        else:
            return f"{formatted} {currency}"

# ============ Validation Helpers ============
class ValidationHelper:
    """Validation and verification helper functions"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address format"""
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        
        pattern = r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w._~:/?#[\]@!$&\'()*+,;=]*)?$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_number(value: Any,
                       min_value: Optional[float] = None,
                       max_value: Optional[float] = None,
                       allow_none: bool = False) -> bool:
        """Validate number with optional range"""
        
        if value is None:
            return allow_none
        
        try:
            num = float(value)
            
            if min_value is not None and num < min_value:
                return False
            
            if max_value is not None and num > max_value:
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_string(value: Any,
                       min_length: Optional[int] = None,
                       max_length: Optional[int] = None,
                       allow_empty: bool = True,
                       allow_none: bool = False) -> bool:
        """Validate string with optional length constraints"""
        
        if value is None:
            return allow_none
        
        if not isinstance(value, str):
            return False
        
        if not allow_empty and len(value.strip()) == 0:
            return False
        
        if min_length is not None and len(value) < min_length:
            return False
        
        if max_length is not None and len(value) > max_length:
            return False
        
        return True
    
    @staticmethod
    def validate_list(value: Any,
                     min_length: Optional[int] = None,
                     max_length: Optional[int] = None,
                     allow_empty: bool = True,
                     allow_none: bool = False) -> bool:
        """Validate list with optional length constraints"""
        
        if value is None:
            return allow_none
        
        if not isinstance(value, (list, tuple, set)):
            return False
        
        if not allow_empty and len(value) == 0:
            return False
        
        if min_length is not None and len(value) < min_length:
            return False
        
        if max_length is not None and len(value) > max_length:
            return False
        
        return True
    
    @staticmethod
    def validate_dict(value: Any,
                     required_keys: Optional[List[str]] = None,
                     allow_empty: bool = True,
                     allow_none: bool = False) -> bool:
        """Validate dictionary with optional required keys"""
        
        if value is None:
            return allow_none
        
        if not isinstance(value, dict):
            return False
        
        if not allow_empty and len(value) == 0:
            return False
        
        if required_keys:
            for key in required_keys:
                if key not in value:
                    return False
        
        return True
    
    @staticmethod
    def validate_datetime(value: Any,
                         min_date: Optional[datetime] = None,
                         max_date: Optional[datetime] = None,
                         allow_none: bool = False) -> bool:
        """Validate datetime with optional range"""
        
        if value is None:
            return allow_none
        
        if not isinstance(value, datetime):
            try:
                value = DateTimeHelper.parse_timestamp(value)
            except (ValueError, TypeError):
                return False
        
        if min_date is not None and value < min_date:
            return False
        
        if max_date is not None and value > max_date:
            return False
        
        return True

# ============ Decorators ============
def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    
    return wrapper

def retry(max_attempts: int = 3, 
          delay: float = 1.0,
          backoff: float = 2.0,
          exceptions: Tuple = (Exception,)) -> Callable:
    """Decorator to retry function on exception"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        sleep_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}. "
                            f"Retrying in {sleep_time:.2f} seconds. Error: {str(e)}"
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}. "
                            f"Last error: {str(e)}"
                        )
            
            raise last_exception
        
        return wrapper
    
    return decorator

def cache_result(ttl: Optional[int] = None, 
                 maxsize: Optional[int] = 128) -> Callable:
    """Decorator to cache function results"""
    
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_timestamps = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            
            # Check if result is cached and valid
            if key in cache:
                if ttl is None:
                    return cache[key]
                
                # Check TTL
                current_time = time.time()
                if current_time - cache_timestamps[key] < ttl:
                    return cache[key]
                else:
                    # Cache expired
                    del cache[key]
                    del cache_timestamps[key]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache[key] = result
            cache_timestamps[key] = time.time()
            
            # Apply cache size limit
            if maxsize is not None and len(cache) > maxsize:
                # Remove oldest entry
                oldest_key = min(cache_timestamps, key=cache_timestamps.get)
                del cache[oldest_key]
                del cache_timestamps[oldest_key]
            
            return result
        
        return wrapper
    
    return decorator

def validate_input(validation_rules: Dict[str, Callable]) -> Callable:
    """Decorator to validate function inputs"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each argument
            for param_name, param_value in bound_args.arguments.items():
                if param_name in validation_rules:
                    validator = validation_rules[param_name]
                    
                    if not validator(param_value):
                        raise ValueError(
                            f"Invalid value for parameter '{param_name}': {param_value}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def log_execution(level: str = 'DEBUG') -> Callable:
    """Decorator to log function execution"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_func = getattr(logger, level.lower())
            
            # Log function call
            log_func(f"Executing {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                log_func(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func.__name__} failed with error: {str(e)}")
                raise
        
        return wrapper
    
    return decorator

# ============ Context Managers ============
@contextmanager
def timer_context(name: str = "Operation"):
    """Context manager for timing code blocks"""
    
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        logger.info(f"{name} completed in {end_time - start_time:.4f} seconds")

@contextmanager
def suppress_exceptions(*exceptions):
    """Context manager to suppress specified exceptions"""
    
    try:
        yield
    except exceptions:
        pass

@contextmanager
def change_directory(directory: Union[str, Path]):
    """Context manager to temporarily change working directory"""
    
    original_dir = os.getcwd()
    directory = Path(directory)
    
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(original_dir)

# ============ Main Helper Class ============
class Helper:
    """Main helper class providing access to all helper functions"""
    
    # Sub-helpers
    math = MathHelper()
    data = DataHelper()
    financial = FinancialHelper()
    datetime = DateTimeHelper()
    file = FileHelper()
    string = StringHelper()
    validation = ValidationHelper()
    
    # Decorators
    timer = staticmethod(timer)
    retry = staticmethod(retry)
    cache_result = staticmethod(cache_result)
    validate_input = staticmethod(validate_input)
    log_execution = staticmethod(log_execution)
    
    # Context managers
    timer_context = staticmethod(timer_context)
    suppress_exceptions = staticmethod(suppress_exceptions)
    change_directory = staticmethod(change_directory)
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system information"""
        
        import platform
        import sys
        
        info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'python_compiler': platform.python_compiler(),
            'python_build': platform.python_build(),
        }
        
        # Try to get memory info if psutil is available
        try:
            import psutil
            memory = psutil.virtual_memory()
            info['memory_total_gb'] = memory.total / (1024 ** 3)
            info['memory_available_gb'] = memory.available / (1024 ** 3)
            info['memory_used_percent'] = memory.percent
            
            disk = psutil.disk_usage('/')
            info['disk_total_gb'] = disk.total / (1024 ** 3)
            info['disk_used_gb'] = disk.used / (1024 ** 3)
            info['disk_free_gb'] = disk.free / (1024 ** 3)
            info['disk_used_percent'] = disk.percent
            
        except ImportError:
            pass
        
        return info
    
    @staticmethod
    def format_exception(e: Exception) -> str:
        """Format exception with traceback"""
        
        return ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    
    @staticmethod
    def safe_execute(func: Callable, *args, **kwargs) -> Tuple[bool, Any, Optional[str]]:
        """Safely execute a function with error handling"""
        
        try:
            result = func(*args, **kwargs)
            return True, result, None
        except Exception as e:
            error_msg = Helper.format_exception(e)
            return False, None, error_msg
    
    @staticmethod
    def parallel_execute(funcs: List[Callable],
                        max_workers: Optional[int] = None,
                        timeout: Optional[float] = None) -> List[Tuple[bool, Any, Optional[str]]]:
        """Execute functions in parallel"""
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_func = {executor.submit(func): func for func in funcs}
            
            for future in concurrent.futures.as_completed(future_to_func, timeout=timeout):
                func = future_to_func[future]
                
                try:
                    result = future.result()
                    results.append((True, result, None))
                except Exception as e:
                    error_msg = Helper.format_exception(e)
                    results.append((False, None, error_msg))
        
        return results
    
    @staticmethod
    def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split list into chunks of specified size"""
        
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    
    @staticmethod
    def flatten_list(nested_list: List[Any]) -> List[Any]:
        """Flatten nested list"""
        
        flat_list = []
        
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(Helper.flatten_list(item))
            else:
                flat_list.append(item)
        
        return flat_list
    
    @staticmethod
    def merge_dicts(dicts: List[Dict[str, Any]], 
                   merge_lists: bool = True) -> Dict[str, Any]:
        """Merge multiple dictionaries"""
        
        result = {}
        
        for d in dicts:
            for key, value in d.items():
                if key in result:
                    # Merge based on type
                    if isinstance(value, dict) and isinstance(result[key], dict):
                        result[key] = Helper.merge_dicts([result[key], value], merge_lists)
                    elif isinstance(value, list) and isinstance(result[key], list) and merge_lists:
                        result[key].extend(value)
                    else:
                        # Overwrite with new value
                        result[key] = value
                else:
                    result[key] = value
        
        return result

# ============ Example Usage ============
if __name__ == "__main__":
    # Example usage
    print("Helpers Module")
    
    # Create helper instance
    helper = Helper()
    
    # Math helper example
    values = np.array([1, 2, 3, 4, 5])
    normalized = helper.math.normalize(values, 'minmax')
    print(f"Normalized values: {normalized}")
    
    # Data helper example
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
        'price': np.random.randn(10).cumsum() + 100
    })
    
    lagged_data = helper.data.create_lagged_features(data, ['price'], [1, 2, 3])
    print(f"Lagged data shape: {lagged_data.shape}")
    
    # Financial helper example
    pnl = helper.financial.calculate_pnl(100, 110, 10, 'long', 0.001)
    print(f"Trade P&L: {pnl:.2f}")
    
    # DateTime helper example
    dt = datetime.now()
    formatted = helper.datetime.format_timestamp(dt)
    print(f"Formatted datetime: {formatted}")
    
    # File helper example
    test_data = {'test': 'data', 'value': 42}
    helper.file.write_json(test_data, 'test.json')
    loaded_data = helper.file.read_json('test.json')
    print(f"Loaded data: {loaded_data}")
    
    # String helper example
    random_id = helper.string.generate_id('test', 8)
    print(f"Random ID: {random_id}")
    
    # Validation helper example
    is_valid = helper.validation.validate_number(42, min_value=0, max_value=100)
    print(f"Is 42 valid between 0 and 100? {is_valid}")
    
    # Clean up
    import os
    if os.path.exists('test.json'):
        os.remove('test.json')
    
    print("Helpers module example completed")