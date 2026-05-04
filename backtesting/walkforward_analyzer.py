"""
Walk-Forward Analysis (WFA) module for robust strategy validation.
Implements rolling window analysis for out-of-sample testing and optimization.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import warnings
import copy
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import traceback
import hashlib
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Import project modules
from logger import get_logger
from backtest_engine import BacktestEngine, BacktestConfig, PerformanceMetrics
from cache import TradingCache, cached

logger = get_logger(__name__)

class WFAOptimizationMethod(Enum):
    """Optimization methods for walk-forward analysis."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"

class WFAMetric(Enum):
    """Metrics for walk-forward analysis optimization."""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    EXPECTANCY = "expectancy"
    CUSTOM = "custom"

@dataclass
class WFASplit:
    """Represents a single train-test split in walk-forward analysis."""
    split_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size_days: int
    test_size_days: int
    train_candles: int = 0
    test_candles: int = 0
    
    def __str__(self) -> str:
        return (f"Split {self.split_id}: "
                f"Train [{self.train_start.date()} to {self.train_end.date()}] "
                f"Test [{self.test_start.date()} to {self.test_end.date()}]")

@dataclass
class WFAWindowResult:
    """Results for a single walk-forward window."""
    split: WFASplit
    best_params: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_results: Dict[str, Any]
    test_results: Dict[str, Any]
    optimization_time: float
    backtest_time: float
    optimization_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'split': asdict(self.split),
            'best_params': self.best_params,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'optimization_time': self.optimization_time,
            'backtest_time': self.backtest_time,
            'optimization_notes': self.optimization_notes
        }

@dataclass
class WFAConfig:
    """Configuration for walk-forward analysis."""
    # Data splitting
    initial_train_size: Union[int, float] = 252  # Days or percentage
    test_size: Union[int, float] = 63  # Days or percentage
    step_size: Union[int, float] = 21  # Days or percentage
    splits: int = 10  # Number of splits (alternative to step_size)
    min_train_size: int = 100  # Minimum candles in training set
    min_test_size: int = 20   # Minimum candles in test set
    
    # Optimization
    optimization_method: WFAOptimizationMethod = WFAOptimizationMethod.GRID_SEARCH
    optimization_metric: WFAMetric = WFAMetric.SHARPE_RATIO
    custom_metric_function: Optional[Callable] = None
    maximize_metric: bool = True  # True to maximize, False to minimize
    
    # Parameter search
    param_grid: Optional[Dict[str, List[Any]]] = None
    random_search_iterations: int = 100
    bayesian_iterations: int = 50
    genetic_generations: int = 20
    genetic_population: int = 50
    
    # Backtest
    backtest_config: Optional[BacktestConfig] = None
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "1h"
    
    # Parallel processing
    parallel: bool = True
    max_workers: int = 4
    parallel_backtests: bool = True
    
    # Caching
    use_cache: bool = True
    cache_dir: str = "wfa_cache"
    
    # Output
    verbose: bool = True
    save_results: bool = True
    results_dir: str = "wfa_results"
    save_plots: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.backtest_config is None:
            self.backtest_config = BacktestConfig()
        
        if self.param_grid is None:
            self.param_grid = {
                'fast_period': [5, 10, 20],
                'slow_period': [20, 30, 50]
            }

@dataclass
class WFAFinalResults:
    """Final aggregated results from walk-forward analysis."""
    # Configuration
    config: Dict[str, Any]
    
    # Window results
    window_results: List[WFAWindowResult]
    
    # Aggregate metrics
    aggregate_test_metrics: Dict[str, float]
    aggregate_train_metrics: Dict[str, float]
    
    # Consistency metrics
    consistency_metrics: Dict[str, float]
    
    # Parameter stability
    parameter_stability: Dict[str, Dict[str, float]]
    
    # Performance classification
    performance_classification: Dict[str, Any]
    
    # Timestamps
    created_at: datetime
    total_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'config': self.config,
            'window_results': [r.to_dict() for r in self.window_results],
            'aggregate_test_metrics': self.aggregate_test_metrics,
            'aggregate_train_metrics': self.aggregate_train_metrics,
            'consistency_metrics': self.consistency_metrics,
            'parameter_stability': self.parameter_stability,
            'performance_classification': self.performance_classification,
            'created_at': self.created_at.isoformat(),
            'total_time': self.total_time
        }
    
    def save(self, filepath: str) -> None:
        """Save results to file."""
        path = Path(filepath)
        path.parent.mkdir(exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved WFA results to {path}")

class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis engine for robust strategy validation.
    Implements rolling window optimization and testing.
    """
    
    def __init__(self, config: WFAConfig):
        """
        Initialize walk-forward analyzer.
        
        Args:
            config: WFA configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.WalkForwardAnalyzer")
        
        # Data storage
        self.data: Dict[str, pd.DataFrame] = {}
        self.splits: List[WFASplit] = []
        
        # Results storage
        self.window_results: List[WFAWindowResult] = []
        self.final_results: Optional[WFAFinalResults] = None
        
        # Cache for optimization results
        if config.use_cache:
            self.cache = TradingCache(cache_type="disk", cache_dir=config.cache_dir)
        else:
            self.cache = None
        
        # Strategy information
        self.strategy_class: Optional[Any] = None
        self.strategy_name: str = ""
        
        # Progress tracking
        self.current_split: int = 0
        self.total_splits: int = 0
        
        self.logger.info("Initialized WalkForwardAnalyzer")
    
    def load_data(self, 
                  data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                  symbol: Optional[str] = None) -> None:
        """
        Load OHLCV data for analysis.
        
        Args:
            data: DataFrame or dictionary of DataFrames
            symbol: Symbol name (if single DataFrame)
        """
        if isinstance(data, pd.DataFrame):
            if symbol is None:
                symbol = self.config.symbols[0]
            
            df = data.copy()
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.index = pd.to_datetime(df['timestamp'], unit='ms')
                elif 'date' in df.columns:
                    df.index = pd.to_datetime(df['date'])
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            self.data[symbol] = df
            self.logger.info(f"Loaded data for {symbol}: {len(df)} candles "
                           f"from {df.index[0]} to {df.index[-1]}")
        
        elif isinstance(data, dict):
            for sym, df in data.items():
                self.load_data(df, sym)
        else:
            raise ValueError("Data must be DataFrame or dict of DataFrames")
    
    def generate_splits(self) -> List[WFASplit]:
        """
        Generate walk-forward splits based on configuration.
        
        Returns:
            List[WFASplit]: Generated splits
        """
        if not self.data:
            raise ValueError("No data loaded")
        
        # Use first symbol for split generation
        symbol = list(self.data.keys())[0]
        df = self.data[symbol]
        
        # Convert sizes to candles if percentages
        total_candles = len(df)
        
        if isinstance(self.config.initial_train_size, float):
            initial_train_candles = int(total_candles * self.config.initial_train_size)
        else:
            # Estimate candles from days based on timeframe
            if self.config.timeframe.endswith('m'):
                candles_per_day = 24 * 60 // int(self.config.timeframe[:-1])
            elif self.config.timeframe.endswith('h'):
                candles_per_day = 24 // int(self.config.timeframe[:-1])
            elif self.config.timeframe.endswith('d'):
                candles_per_day = 1
            else:
                candles_per_day = 24  # Default for 1h
            
            initial_train_candles = self.config.initial_train_size * candles_per_day
        
        if isinstance(self.config.test_size, float):
            test_candles = int(total_candles * self.config.test_size)
        else:
            test_candles = self.config.test_size * candles_per_day
        
        if isinstance(self.config.step_size, float):
            step_candles = int(total_candles * self.config.step_size)
        else:
            step_candles = self.config.step_size * candles_per_day
        
        # Ensure minimum sizes
        initial_train_candles = max(initial_train_candles, self.config.min_train_size)
        test_candles = max(test_candles, self.config.min_test_size)
        
        # Generate splits
        splits = []
        split_id = 0
        
        # Start from beginning
        train_start_idx = 0
        train_end_idx = initial_train_candles
        test_end_idx = train_end_idx + test_candles
        
        while test_end_idx <= total_candles:
            # Create split
            split = WFASplit(
                split_id=split_id,
                train_start=df.index[train_start_idx],
                train_end=df.index[train_end_idx - 1],
                test_start=df.index[train_end_idx],
                test_end=df.index[test_end_idx - 1],
                train_size_days=(df.index[train_end_idx - 1] - df.index[train_start_idx]).days,
                test_size_days=(df.index[test_end_idx - 1] - df.index[train_end_idx]).days,
                train_candles=train_end_idx - train_start_idx,
                test_candles=test_end_idx - train_end_idx
            )
            
            splits.append(split)
            
            # Move window
            if self.config.step_size == 0 and self.config.splits > 0:
                # Fixed number of splits
                step_candles = (total_candles - initial_train_candles) // self.config.splits
            
            train_start_idx += step_candles
            train_end_idx = train_start_idx + initial_train_candles
            test_end_idx = train_end_idx + test_candles
            
            split_id += 1
            
            # Check if we've reached requested splits
            if self.config.splits > 0 and len(splits) >= self.config.splits:
                break
        
        self.splits = splits
        self.total_splits = len(splits)
        
        self.logger.info(f"Generated {len(splits)} walk-forward splits")
        for split in splits[:3]:  # Log first 3 splits
            self.logger.info(f"  {split}")
        
        if len(splits) > 3:
            self.logger.info(f"  ... and {len(splits) - 3} more splits")
        
        return splits
    
    def set_strategy(self, strategy_class: Any, strategy_name: Optional[str] = None) -> None:
        """
        Set strategy for analysis.
        
        Args:
            strategy_class: Strategy class
            strategy_name: Optional name for strategy
        """
        self.strategy_class = strategy_class
        self.strategy_name = strategy_name or strategy_class.__name__
        self.logger.info(f"Set strategy: {self.strategy_name}")
    
    def _get_cache_key(self, 
                      split_id: int,
                      params: Dict[str, Any],
                      data_hash: str) -> str:
        """
        Generate cache key for optimization results.
        
        Args:
            split_id: Split ID
            params: Strategy parameters
            data_hash: Hash of data
        
        Returns:
            str: Cache key
        """
        param_str = json.dumps(params, sort_keys=True)
        key_content = f"{self.strategy_name}_{split_id}_{param_str}_{data_hash}"
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def _run_backtest(self,
                     data: Dict[str, pd.DataFrame],
                     params: Dict[str, Any],
                     config: BacktestConfig,
                     split_info: Optional[WFASplit] = None) -> Dict[str, Any]:
        """
        Run a single backtest with given parameters.
        
        Args:
            data: OHLCV data
            params: Strategy parameters
            config: Backtest configuration
            split_info: Split information (for logging)
        
        Returns:
            Dict[str, Any]: Backtest results
        """
        try:
            engine = BacktestEngine(config)
            engine.load_data(data)
            engine.add_strategy(self.strategy_class, **params)
            
            results = engine.run()
            
            if split_info and self.config.verbose:
                self.logger.debug(f"Backtest completed for split {split_info.split_id} "
                                f"with params {params}")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Backtest failed with params {params}: {e}")
            # Return empty results
            return {
                'metrics': {'sharpe_ratio': -float('inf'), 'total_return_percentage': -100},
                'summary': {'total_return_percentage': -100, 'sharpe_ratio': -float('inf')}
            }
    
    def _optimize_grid_search(self,
                            train_data: Dict[str, pd.DataFrame],
                            split: WFASplit) -> Dict[str, Any]:
        """
        Optimize parameters using grid search.
        
        Args:
            train_data: Training data
            split: Current split
        
        Returns:
            Dict[str, Any]: Best parameters and metrics
        """
        from itertools import product
        
        param_names = list(self.config.param_grid.keys())
        param_combinations = list(product(*self.config.param_grid.values()))
        
        best_score = -float('inf') if self.config.maximize_metric else float('inf')
        best_params = None
        best_results = None
        all_results = []
        
        # Prepare backtest config for training
        train_config = copy.deepcopy(self.config.backtest_config)
        train_config.start_date = split.train_start
        train_config.end_date = split.train_end
        train_config.verbose = False
        
        # Hash of training data for caching
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(train_data[list(train_data.keys())[0]]).values.tobytes()
        ).hexdigest()
        
        for i, values in enumerate(param_combinations):
            params = dict(zip(param_names, values))
            
            # Check cache
            cache_key = None
            if self.cache:
                cache_key = self._get_cache_key(split.split_id, params, data_hash)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    results = cached_result
                    self.logger.debug(f"Cache hit for params {params}")
                else:
                    # Run backtest
                    results = self._run_backtest(train_data, params, train_config, split)
                    
                    # Cache results
                    self.cache.set(cache_key, results, ttl=86400)  # 24 hours
            else:
                # Run backtest
                results = self._run_backtest(train_data, params, train_config, split)
            
            # Extract metric
            if self.config.custom_metric_function:
                score = self.config.custom_metric_function(results)
            else:
                metric_name = self.config.optimization_metric.value
                if metric_name in results['metrics']:
                    score = results['metrics'][metric_name]
                elif metric_name in results['summary']:
                    score = results['summary'][metric_name]
                else:
                    # Default to Sharpe ratio
                    score = results['metrics']['sharpe_ratio']
            
            all_results.append({
                'params': params,
                'score': score,
                'results': results
            })
            
            # Update best
            if self.config.maximize_metric:
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_results = results
            else:
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_results = results
            
            if self.config.verbose and i % 10 == 0:
                self.logger.info(f"  Grid search: {i+1}/{len(param_combinations)} "
                               f"params: {params}, score: {score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_results': best_results,
            'all_results': all_results,
            'method': 'grid_search'
        }
    
    def _optimize_random_search(self,
                              train_data: Dict[str, pd.DataFrame],
                              split: WFASplit) -> Dict[str, Any]:
        """
        Optimize parameters using random search.
        
        Args:
            train_data: Training data
            split: Current split
        
        Returns:
            Dict[str, Any]: Best parameters and metrics
        """
        import random
        
        best_score = -float('inf') if self.config.maximize_metric else float('inf')
        best_params = None
        best_results = None
        all_results = []
        
        # Prepare backtest config
        train_config = copy.deepcopy(self.config.backtest_config)
        train_config.start_date = split.train_start
        train_config.end_date = split.train_end
        train_config.verbose = False
        
        # Hash of training data
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(train_data[list(train_data.keys())[0]]).values.tobytes()
        ).hexdigest()
        
        for i in range(self.config.random_search_iterations):
            # Generate random parameters
            params = {}
            for param_name, param_values in self.config.param_grid.items():
                if isinstance(param_values[0], (int, float)):
                    # Continuous parameter
                    min_val = min(param_values)
                    max_val = max(param_values)
                    
                    if all(isinstance(v, int) for v in param_values):
                        # Integer parameter
                        params[param_name] = random.randint(min_val, max_val)
                    else:
                        # Float parameter
                        params[param_name] = random.uniform(min_val, max_val)
                else:
                    # Categorical parameter
                    params[param_name] = random.choice(param_values)
            
            # Check cache
            cache_key = None
            if self.cache:
                cache_key = self._get_cache_key(split.split_id, params, data_hash)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    results = cached_result
                else:
                    results = self._run_backtest(train_data, params, train_config, split)
                    self.cache.set(cache_key, results, ttl=86400)
            else:
                results = self._run_backtest(train_data, params, train_config, split)
            
            # Extract metric
            if self.config.custom_metric_function:
                score = self.config.custom_metric_function(results)
            else:
                metric_name = self.config.optimization_metric.value
                if metric_name in results['metrics']:
                    score = results['metrics'][metric_name]
                elif metric_name in results['summary']:
                    score = results['summary'][metric_name]
                else:
                    score = results['metrics']['sharpe_ratio']
            
            all_results.append({
                'params': params,
                'score': score,
                'results': results
            })
            
            # Update best
            if self.config.maximize_metric:
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_results = results
            else:
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_results = results
            
            if self.config.verbose and i % 20 == 0:
                self.logger.info(f"  Random search: {i+1}/{self.config.random_search_iterations} "
                               f"score: {score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_results': best_results,
            'all_results': all_results,
            'method': 'random_search'
        }
    
    def _optimize_bayesian(self,
                          train_data: Dict[str, pd.DataFrame],
                          split: WFASplit) -> Dict[str, Any]:
        """
        Optimize parameters using Bayesian optimization.
        
        Args:
            train_data: Training data
            split: Current split
        
        Returns:
            Dict[str, Any]: Best parameters and metrics
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
        except ImportError:
            self.logger.warning("scikit-optimize not installed, falling back to random search")
            return self._optimize_random_search(train_data, split)
        
        # Define search space
        dimensions = []
        param_mapping = {}
        
        for param_name, param_values in self.config.param_grid.items():
            if isinstance(param_values[0], int):
                # Integer space
                dimensions.append(Integer(min(param_values), max(param_values), name=param_name))
                param_mapping[param_name] = 'int'
            elif isinstance(param_values[0], float):
                # Real space
                dimensions.append(Real(min(param_values), max(param_values), name=param_name))
                param_mapping[param_name] = 'float'
            else:
                # Categorical space
                dimensions.append(Categorical(param_values, name=param_name))
                param_mapping[param_name] = 'categorical'
        
        # Prepare backtest config
        train_config = copy.deepcopy(self.config.backtest_config)
        train_config.start_date = split.train_start
        train_config.end_date = split.train_end
        train_config.verbose = False
        
        # Hash of training data
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(train_data[list(train_data.keys())[0]]).values.tobytes()
        ).hexdigest()
        
        # Objective function
        @use_named_args(dimensions)
        def objective(**params):
            # Convert parameters to correct types
            for param_name, param_value in params.items():
                if param_mapping[param_name] == 'int':
                    params[param_name] = int(param_value)
            
            # Check cache
            cache_key = None
            if self.cache:
                cache_key = self._get_cache_key(split.split_id, params, data_hash)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    results = cached_result
                else:
                    results = self._run_backtest(train_data, params, train_config, split)
                    self.cache.set(cache_key, results, ttl=86400)
            else:
                results = self._run_backtest(train_data, params, train_config, split)
            
            # Extract metric
            if self.config.custom_metric_function:
                score = self.config.custom_metric_function(results)
            else:
                metric_name = self.config.optimization_metric.value
                if metric_name in results['metrics']:
                    score = results['metrics'][metric_name]
                elif metric_name in results['summary']:
                    score = results['summary'][metric_name]
                else:
                    score = results['metrics']['sharpe_ratio']
            
            # Negate if we're maximizing (gp_minimize minimizes)
            if self.config.maximize_metric:
                return -score
            else:
                return score
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.config.bayesian_iterations,
            random_state=42,
            verbose=self.config.verbose
        )
        
        # Extract best parameters
        best_params = {}
        for i, param_name in enumerate([d.name for d in dimensions]):
            best_params[param_name] = result.x[i]
            
            # Convert to correct type
            if param_mapping[param_name] == 'int':
                best_params[param_name] = int(best_params[param_name])
        
        # Run final backtest with best params
        best_results = self._run_backtest(train_data, best_params, train_config, split)
        
        # Extract actual score (not negated)
        if self.config.custom_metric_function:
            best_score = self.config.custom_metric_function(best_results)
        else:
            metric_name = self.config.optimization_metric.value
            if metric_name in best_results['metrics']:
                best_score = best_results['metrics'][metric_name]
            elif metric_name in best_results['summary']:
                best_score = best_results['summary'][metric_name]
            else:
                best_score = best_results['metrics']['sharpe_ratio']
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_results': best_results,
            'method': 'bayesian_optimization'
        }
    
    def _run_window_optimization(self, split: WFASplit) -> WFAWindowResult:
        """
        Run optimization for a single window.
        
        Args:
            split: Current split
        
        Returns:
            WFAWindowResult: Window results
        """
        self.logger.info(f"Processing split {split.split_id + 1}/{self.total_splits}: {split}")
        
        # Extract training data
        train_data = {}
        for symbol, df in self.data.items():
            train_mask = (df.index >= split.train_start) & (df.index <= split.train_end)
            train_data[symbol] = df[train_mask].copy()
        
        # Extract test data
        test_data = {}
        for symbol, df in self.data.items():
            test_mask = (df.index >= split.test_start) & (df.index <= split.test_end)
            test_data[symbol] = df[test_mask].copy()
        
        # Run optimization on training data
        optimization_start = datetime.now()
        
        if self.config.optimization_method == WFAOptimizationMethod.GRID_SEARCH:
            optimization_result = self._optimize_grid_search(train_data, split)
        
        elif self.config.optimization_method == WFAOptimizationMethod.RANDOM_SEARCH:
            optimization_result = self._optimize_random_search(train_data, split)
        
        elif self.config.optimization_method == WFAOptimizationMethod.BAYESIAN_OPTIMIZATION:
            optimization_result = self._optimize_bayesian(train_data, split)
        
        elif self.config.optimization_method == WFAOptimizationMethod.GENETIC_ALGORITHM:
            # For simplicity, fall back to random search
            self.logger.warning("Genetic algorithm not implemented, using random search")
            optimization_result = self._optimize_random_search(train_data, split)
        
        else:
            raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
        
        optimization_time = (datetime.now() - optimization_start).total_seconds()
        
        # Extract training metrics
        train_metrics = {}
        if optimization_result['best_results']:
            train_metrics = {
                'sharpe_ratio': optimization_result['best_results']['metrics']['sharpe_ratio'],
                'total_return_percentage': optimization_result['best_results']['metrics']['total_return_percentage'],
                'max_drawdown_percentage': optimization_result['best_results']['metrics']['max_drawdown_percentage'],
                'win_rate': optimization_result['best_results']['metrics']['win_rate'],
                'profit_factor': optimization_result['best_results']['metrics']['profit_factor']
            }
        
        # Run backtest on test data with optimized parameters
        backtest_start = datetime.now()
        
        test_config = copy.deepcopy(self.config.backtest_config)
        test_config.start_date = split.test_start
        test_config.end_date = split.test_end
        test_config.verbose = False
        
        test_results = self._run_backtest(
            test_data, 
            optimization_result['best_params'], 
            test_config, 
            split
        )
        
        backtest_time = (datetime.now() - backtest_start).total_seconds()
        
        # Extract test metrics
        test_metrics = {
            'sharpe_ratio': test_results['metrics']['sharpe_ratio'],
            'total_return_percentage': test_results['metrics']['total_return_percentage'],
            'max_drawdown_percentage': test_results['metrics']['max_drawdown_percentage'],
            'win_rate': test_results['metrics']['win_rate'],
            'profit_factor': test_results['metrics']['profit_factor'],
            'total_trades': test_results['metrics']['total_trades']
        }
        
        # Create window result
        window_result = WFAWindowResult(
            split=split,
            best_params=optimization_result['best_params'],
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            train_results=optimization_result['best_results'],
            test_results=test_results,
            optimization_time=optimization_time,
            backtest_time=backtest_time,
            optimization_notes=optimization_result.get('method', 'unknown')
        )
        
        self.logger.info(f"  Split {split.split_id} complete: "
                       f"Train Sharpe: {train_metrics.get('sharpe_ratio', 0):.2f}, "
                       f"Test Sharpe: {test_metrics['sharpe_ratio']:.2f}, "
                       f"Test Return: {test_metrics['total_return_percentage']:.2f}%")
        
        return window_result
    
    def run_analysis(self) -> WFAFinalResults:
        """
        Run complete walk-forward analysis.
        
        Returns:
            WFAFinalResults: Final analysis results
        """
        if not self.strategy_class:
            raise ValueError("No strategy set. Use set_strategy() first.")
        
        if not self.data:
            raise ValueError("No data loaded. Use load_data() first.")
        
        # Generate splits if not already done
        if not self.splits:
            self.generate_splits()
        
        if not self.splits:
            raise ValueError("No splits generated. Check data and configuration.")
        
        self.logger.info(f"Starting walk-forward analysis with {len(self.splits)} splits")
        self.logger.info(f"Strategy: {self.strategy_name}")
        self.logger.info(f"Optimization method: {self.config.optimization_method.value}")
        self.logger.info(f"Optimization metric: {self.config.optimization_metric.value}")
        
        # Run analysis for each split
        start_time = datetime.now()
        
        if self.config.parallel and self.config.max_workers > 1:
            self.window_results = self._run_parallel_analysis()
        else:
            self.window_results = []
            for split in self.splits:
                result = self._run_window_optimization(split)
                self.window_results.append(result)
        
        # Calculate final results
        total_time = (datetime.now() - start_time).total_seconds()
        self.final_results = self._calculate_final_results(total_time)
        
        self.logger.info(f"Walk-forward analysis completed in {total_time:.1f} seconds")
        
        # Save results if configured
        if self.config.save_results:
            self.save_results()
        
        # Generate plots if configured
        if self.config.save_plots:
            self.plot_results()
        
        return self.final_results
    
    def _run_parallel_analysis(self) -> List[WFAWindowResult]:
        """
        Run analysis in parallel.
        
        Returns:
            List[WFAWindowResult]: Window results
        """
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_split = {
                executor.submit(self._run_window_optimization, split): split
                for split in self.splits
            }
            
            # Process completed tasks
            for future in tqdm(as_completed(future_to_split), 
                             total=len(self.splits),
                             desc="Walk-Forward Analysis",
                             disable=not self.config.verbose):
                split = future_to_split[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing split {split.split_id}: {e}")
                    self.logger.error(traceback.format_exc())
        
        # Sort by split ID
        results.sort(key=lambda x: x.split.split_id)
        return results
    
    def _calculate_final_results(self, total_time: float) -> WFAFinalResults:
        """
        Calculate final aggregated results.
        
        Args:
            total_time: Total analysis time in seconds
        
        Returns:
            WFAFinalResults: Final results
        """
        if not self.window_results:
            raise ValueError("No window results available")
        
        # Extract metrics
        train_sharpes = [r.train_metrics.get('sharpe_ratio', 0) for r in self.window_results]
        test_sharpes = [r.test_metrics.get('sharpe_ratio', 0) for r in self.window_results]
        test_returns = [r.test_metrics.get('total_return_percentage', 0) for r in self.window_results]
        test_drawdowns = [r.test_metrics.get('max_drawdown_percentage', 0) for r in self.window_results]
        test_win_rates = [r.test_metrics.get('win_rate', 0) for r in self.window_results]
        test_profit_factors = [r.test_metrics.get('profit_factor', 0) for r in self.window_results]
        
        # Calculate aggregate metrics
        aggregate_test_metrics = {
            'mean_sharpe': np.mean(test_sharpes),
            'std_sharpe': np.std(test_sharpes),
            'mean_return': np.mean(test_returns),
            'std_return': np.std(test_returns),
            'mean_drawdown': np.mean(test_drawdowns),
            'mean_win_rate': np.mean(test_win_rates),
            'mean_profit_factor': np.mean(test_profit_factors),
            'median_sharpe': np.median(test_sharpes),
            'median_return': np.median(test_returns),
            'best_sharpe': np.max(test_sharpes),
            'worst_sharpe': np.min(test_sharpes),
            'best_return': np.max(test_returns),
            'worst_return': np.min(test_returns)
        }
        
        aggregate_train_metrics = {
            'mean_sharpe': np.mean(train_sharpes),
            'std_sharpe': np.std(train_sharpes),
            'mean_return': np.mean([r.train_metrics.get('total_return_percentage', 0) 
                                   for r in self.window_results]),
            'mean_drawdown': np.mean([r.train_metrics.get('max_drawdown_percentage', 0) 
                                     for r in self.window_results])
        }
        
        # Calculate consistency metrics
        positive_returns = sum(1 for r in test_returns if r > 0)
        positive_sharpes = sum(1 for s in test_sharpes if s > 0)
        
        consistency_metrics = {
            'return_consistency': positive_returns / len(test_returns),
            'sharpe_consistency': positive_sharpes / len(test_sharpes),
            'win_rate_consistency': sum(1 for w in test_win_rates if w > 50) / len(test_win_rates),
            'profit_factor_consistency': sum(1 for p in test_profit_factors if p > 1) / len(test_profit_factors),
            'train_test_correlation': np.corrcoef(train_sharpes, test_sharpes)[0, 1],
            'train_test_ratio': np.mean(test_sharpes) / np.mean(train_sharpes) if np.mean(train_sharpes) != 0 else 0
        }
        
        # Calculate parameter stability
        all_params = [r.best_params for r in self.window_results]
        param_stability = {}
        
        if all_params and len(all_params) > 1:
            for param_name in all_params[0].keys():
                param_values = [p.get(param_name) for p in all_params if param_name in p]
                
                if param_values and all(isinstance(v, (int, float)) for v in param_values):
                    param_stability[param_name] = {
                        'mean': np.mean(param_values),
                        'std': np.std(param_values),
                        'cv': np.std(param_values) / np.mean(param_values) if np.mean(param_values) != 0 else 0,
                        'min': np.min(param_values),
                        'max': np.max(param_values),
                        'range': np.max(param_values) - np.min(param_values)
                    }
        
        # Performance classification
        performance_classification = {
            'overall_performance': self._classify_performance(aggregate_test_metrics),
            'stability': self._classify_stability(consistency_metrics),
            'robustness': self._classify_robustness(aggregate_test_metrics, consistency_metrics)
        }
        
        # Create final results
        final_results = WFAFinalResults(
            config=asdict(self.config),
            window_results=self.window_results,
            aggregate_test_metrics=aggregate_test_metrics,
            aggregate_train_metrics=aggregate_train_metrics,
            consistency_metrics=consistency_metrics,
            parameter_stability=param_stability,
            performance_classification=performance_classification,
            created_at=datetime.now(),
            total_time=total_time
        )
        
        return final_results
    
    def _classify_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify overall performance.
        
        Args:
            metrics: Aggregate test metrics
        
        Returns:
            Dict[str, Any]: Performance classification
        """
        sharpe = metrics['mean_sharpe']
        returns = metrics['mean_return']
        drawdown = metrics['mean_drawdown']
        
        # Sharpe classification
        if sharpe >= 1.5:
            sharpe_class = "Excellent"
        elif sharpe >= 1.0:
            sharpe_class = "Good"
        elif sharpe >= 0.5:
            sharpe_class = "Average"
        elif sharpe >= 0:
            sharpe_class = "Poor"
        else:
            sharpe_class = "Very Poor"
        
        # Return classification (annualized equivalent)
        annual_return = returns  # Assuming returns are already annualized
        if annual_return >= 30:
            return_class = "Excellent"
        elif annual_return >= 15:
            return_class = "Good"
        elif annual_return >= 5:
            return_class = "Average"
        elif annual_return >= 0:
            return_class = "Poor"
        else:
            return_class = "Very Poor"
        
        # Drawdown classification
        if drawdown <= 10:
            drawdown_class = "Excellent"
        elif drawdown <= 20:
            drawdown_class = "Good"
        elif drawdown <= 30:
            drawdown_class = "Average"
        elif drawdown <= 40:
            drawdown_class = "Poor"
        else:
            drawdown_class = "Very Poor"
        
        # Overall classification
        classifications = {
            'sharpe': {'value': sharpe, 'class': sharpe_class},
            'return': {'value': returns, 'class': return_class},
            'drawdown': {'value': drawdown, 'class': drawdown_class},
            'overall': self._calculate_overall_class(sharpe_class, return_class, drawdown_class)
        }
        
        return classifications
    
    def _classify_stability(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify strategy stability.
        
        Args:
            metrics: Consistency metrics
        
        Returns:
            Dict[str, Any]: Stability classification
        """
        return_consistency = metrics['return_consistency']
        sharpe_consistency = metrics['sharpe_consistency']
        train_test_ratio = metrics['train_test_ratio']
        
        # Consistency classification
        if return_consistency >= 0.7:
            consistency_class = "Excellent"
        elif return_consistency >= 0.6:
            consistency_class = "Good"
        elif return_consistency >= 0.5:
            consistency_class = "Average"
        elif return_consistency >= 0.4:
            consistency_class = "Poor"
        else:
            consistency_class = "Very Poor"
        
        # Train-test ratio classification
        if train_test_ratio >= 0.8:
            ratio_class = "Excellent"
        elif train_test_ratio >= 0.6:
            ratio_class = "Good"
        elif train_test_ratio >= 0.4:
            ratio_class = "Average"
        elif train_test_ratio >= 0.2:
            ratio_class = "Poor"
        else:
            ratio_class = "Very Poor"
        
        classifications = {
            'return_consistency': {'value': return_consistency * 100, 'class': consistency_class},
            'sharpe_consistency': {'value': sharpe_consistency * 100, 'class': consistency_class},
            'train_test_ratio': {'value': train_test_ratio, 'class': ratio_class},
            'overall_stability': self._calculate_stability_class(consistency_class, ratio_class)
        }
        
        return classifications
    
    def _classify_robustness(self, 
                           test_metrics: Dict[str, float],
                           consistency_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify strategy robustness.
        
        Args:
            test_metrics: Test metrics
            consistency_metrics: Consistency metrics
        
        Returns:
            Dict[str, Any]: Robustness classification
        """
        sharpe_std = test_metrics['std_sharpe']
        return_std = test_metrics['std_return']
        consistency = consistency_metrics['return_consistency']
        
        # Volatility classification (lower is better)
        if sharpe_std <= 0.5:
            volatility_class = "Excellent"
        elif sharpe_std <= 1.0:
            volatility_class = "Good"
        elif sharpe_std <= 1.5:
            volatility_class = "Average"
        elif sharpe_std <= 2.0:
            volatility_class = "Poor"
        else:
            volatility_class = "Very Poor"
        
        # Overall robustness
        if consistency >= 0.7 and sharpe_std <= 1.0:
            robustness_class = "Excellent"
        elif consistency >= 0.6 and sharpe_std <= 1.5:
            robustness_class = "Good"
        elif consistency >= 0.5 and sharpe_std <= 2.0:
            robustness_class = "Average"
        elif consistency >= 0.4:
            robustness_class = "Poor"
        else:
            robustness_class = "Very Poor"
        
        classifications = {
            'sharpe_volatility': {'value': sharpe_std, 'class': volatility_class},
            'return_volatility': {'value': return_std, 'class': volatility_class},
            'overall_robustness': robustness_class
        }
        
        return classifications
    
    def _calculate_overall_class(self, 
                               sharpe_class: str, 
                               return_class: str, 
                               drawdown_class: str) -> str:
        """Calculate overall performance class."""
        class_scores = {
            "Excellent": 5,
            "Good": 4,
            "Average": 3,
            "Poor": 2,
            "Very Poor": 1
        }
        
        avg_score = (class_scores[sharpe_class] + 
                    class_scores[return_class] + 
                    class_scores[drawdown_class]) / 3
        
        if avg_score >= 4.5:
            return "Excellent"
        elif avg_score >= 3.5:
            return "Good"
        elif avg_score >= 2.5:
            return "Average"
        elif avg_score >= 1.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def _calculate_stability_class(self, 
                                 consistency_class: str, 
                                 ratio_class: str) -> str:
        """Calculate overall stability class."""
        class_scores = {
            "Excellent": 5,
            "Good": 4,
            "Average": 3,
            "Poor": 2,
            "Very Poor": 1
        }
        
        avg_score = (class_scores[consistency_class] + class_scores[ratio_class]) / 2
        
        if avg_score >= 4.5:
            return "Excellent"
        elif avg_score >= 3.5:
            return "Good"
        elif avg_score >= 2.5:
            return "Average"
        elif avg_score >= 1.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def save_results(self, filepath: Optional[str] = None) -> None:
        """
        Save analysis results to file.
        
        Args:
            filepath: Optional custom filepath
        """
        if not self.final_results:
            raise ValueError("No results to save. Run analysis first.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wfa_results_{self.strategy_name}_{timestamp}.json"
            filepath = Path(self.config.results_dir) / filename
        
        self.final_results.save(filepath)
    
    def plot_results(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Plot walk-forward analysis results.
        
        Args:
            save_dir: Directory to save plots
        
        Returns:
            Dict[str, plt.Figure]: Dictionary of figures
        """
        if not self.final_results:
            raise ValueError("No results to plot. Run analysis first.")
        
        if save_dir is None:
            save_dir = self.config.results_dir
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        figures = {}
        
        # 1. Performance Comparison Plot
        fig1 = self._plot_performance_comparison()
        figures['performance_comparison'] = fig1
        fig1.savefig(save_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        
        # 2. Parameter Stability Plot
        fig2 = self._plot_parameter_stability()
        if fig2:
            figures['parameter_stability'] = fig2
            fig2.savefig(save_path / 'parameter_stability.png', dpi=300, bbox_inches='tight')
        
        # 3. Equity Curve Comparison
        fig3 = self._plot_equity_curves()
        if fig3:
            figures['equity_curves'] = fig3
            fig3.savefig(save_path / 'equity_curves.png', dpi=300, bbox_inches='tight')
        
        # 4. Metrics Distribution
        fig4 = self._plot_metrics_distribution()
        figures['metrics_distribution'] = fig4
        fig4.savefig(save_path / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
        
        # 5. Train-Test Correlation
        fig5 = self._plot_train_test_correlation()
        figures['train_test_correlation'] = fig5
        fig5.savefig(save_path / 'train_test_correlation.png', dpi=300, bbox_inches='tight')
        
        plt.close('all')
        
        self.logger.info(f"Saved plots to {save_path}")
        return figures
    
    def _plot_performance_comparison(self) -> plt.Figure:
        """Plot performance comparison across splits."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Walk-Forward Analysis: {self.strategy_name}', fontsize=16)
        
        # Extract data
        split_ids = [r.split.split_id for r in self.window_results]
        train_sharpes = [r.train_metrics.get('sharpe_ratio', 0) for r in self.window_results]
        test_sharpes = [r.test_metrics.get('sharpe_ratio', 0) for r in self.window_results]
        train_returns = [r.train_metrics.get('total_return_percentage', 0) for r in self.window_results]
        test_returns = [r.test_metrics.get('total_return_percentage', 0) for r in self.window_results]
        
        # 1. Sharpe Ratio Comparison
        ax1 = axes[0, 0]
        width = 0.35
        x = np.arange(len(split_ids))
        
        ax1.bar(x - width/2, train_sharpes, width, label='Train', alpha=0.7, color='blue')
        ax1.bar(x + width/2, test_sharpes, width, label='Test', alpha=0.7, color='green')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Split ID')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('Sharpe Ratio: Train vs Test')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns Comparison
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, train_returns, width, label='Train', alpha=0.7, color='blue')
        ax2.bar(x + width/2, test_returns, width, label='Test', alpha=0.7, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Split ID')
        ax2.set_ylabel('Return (%)')
        ax2.set_title('Returns: Train vs Test')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Train-Test Ratio
        ax3 = axes[1, 0]
        ratios = [test/train if train != 0 else 0 
                 for train, test in zip(train_sharpes, test_sharpes)]
        ax3.bar(x, ratios, color='orange', alpha=0.7)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Ideal (1.0)')
        ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        ax3.set_xlabel('Split ID')
        ax3.set_ylabel('Test/Train Ratio')
        ax3.set_title('Train-Test Performance Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Consistency Metrics
        ax4 = axes[1, 1]
        metrics_data = {
            'Positive Returns': sum(1 for r in test_returns if r > 0) / len(test_returns) * 100,
            'Positive Sharpe': sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) * 100,
            'Win Rate > 50%': sum(1 for r in self.window_results 
                                 if r.test_metrics.get('win_rate', 0) > 50) / len(self.window_results) * 100
        }
        
        bars = ax4.bar(range(len(metrics_data)), list(metrics_data.values()), 
                      color=['green', 'blue', 'purple'], alpha=0.7)
        ax4.set_xlabel('Metric')
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title('Strategy Consistency')
        ax4.set_xticks(range(len(metrics_data)))
        ax4.set_xticklabels(list(metrics_data.keys()), rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metrics_data.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def _plot_parameter_stability(self) -> Optional[plt.Figure]:
        """Plot parameter stability across splits."""
        if not self.window_results:
            return None
        
        # Extract parameters
        all_params = [r.best_params for r in self.window_results]
        if not all_params:
            return None
        
        # Get numeric parameters only
        numeric_params = {}
        for param_name in all_params[0].keys():
            values = [p.get(param_name) for p in all_params]
            if all(isinstance(v, (int, float)) for v in values):
                numeric_params[param_name] = values
        
        if not numeric_params:
            return None
        
        fig, axes = plt.subplots(len(numeric_params), 1, 
                               figsize=(12, 3 * len(numeric_params)))
        
        if len(numeric_params) == 1:
            axes = [axes]
        
        for idx, (param_name, values) in enumerate(numeric_params.items()):
            ax = axes[idx]
            
            # Plot parameter values
            ax.plot(range(len(values)), values, 'o-', linewidth=2, markersize=6)
            
            # Add mean line
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                      label=f'Mean: {mean_val:.2f}')
            
            # Add std bands
            std_val = np.std(values)
            ax.fill_between(range(len(values)), 
                          mean_val - std_val, 
                          mean_val + std_val, 
                          alpha=0.2, color='red')
            
            ax.set_xlabel('Split ID')
            ax.set_ylabel('Parameter Value')
            ax.set_title(f'Parameter Stability: {param_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_equity_curves(self) -> Optional[plt.Figure]:
        """Plot equity curves for test periods."""
        if not self.window_results:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.window_results)))
        
        for idx, result in enumerate(self.window_results):
            if 'equity_curve' in result.test_results:
                equity_curve = result.test_results['equity_curve']
                
                if equity_curve:
                    dates = [d for d, _ in equity_curve]
                    values = [v for _, v in equity_curve]
                    
                    # Normalize to starting value
                    if values[0] > 0:
                        normalized = [v / values[0] * 100 for v in values]
                        ax.plot(dates, normalized, color=colors[idx], 
                               alpha=0.7, linewidth=1,
                               label=f'Split {result.split.split_id}')
        
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Starting Value')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value (%)')
        ax.set_title('Test Period Equity Curves (Normalized)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_metrics_distribution(self) -> plt.Figure:
        """Plot distribution of key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract metrics
        test_sharpes = [r.test_metrics.get('sharpe_ratio', 0) for r in self.window_results]
        test_returns = [r.test_metrics.get('total_return_percentage', 0) for r in self.window_results]
        test_drawdowns = [r.test_metrics.get('max_drawdown_percentage', 0) for r in self.window_results]
        test_win_rates = [r.test_metrics.get('win_rate', 0) for r in self.window_results]
        
        # 1. Sharpe Ratio Distribution
        ax1 = axes[0, 0]
        ax1.hist(test_sharpes, bins=15, alpha=0.7, color='green', edgecolor='black')
        ax1.axvline(x=np.mean(test_sharpes), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(test_sharpes):.2f}')
        ax1.axvline(x=np.median(test_sharpes), color='blue', linestyle='--',
                   label=f'Median: {np.median(test_sharpes):.2f}')
        ax1.set_xlabel('Sharpe Ratio')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Sharpe Ratio Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns Distribution
        ax2 = axes[0, 1]
        ax2.hist(test_returns, bins=15, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=np.mean(test_returns), color='red', linestyle='--',
                   label=f'Mean: {np.mean(test_returns):.2f}%')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Returns Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown Distribution
        ax3 = axes[1, 0]
        ax3.hist(test_drawdowns, bins=15, alpha=0.7, color='red', edgecolor='black')
        ax3.axvline(x=np.mean(test_drawdowns), color='blue', linestyle='--',
                   label=f'Mean: {np.mean(test_drawdowns):.2f}%')
        ax3.set_xlabel('Max Drawdown (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Drawdown Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Win Rate Distribution
        ax4 = axes[1, 1]
        ax4.hist(test_win_rates, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(x=np.mean(test_win_rates), color='red', linestyle='--',
                   label=f'Mean: {np.mean(test_win_rates):.2f}%')
        ax4.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='50% Threshold')
        ax4.set_xlabel('Win Rate (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Win Rate Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_train_test_correlation(self) -> plt.Figure:
        """Plot train-test correlation."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        train_sharpes = [r.train_metrics.get('sharpe_ratio', 0) for r in self.window_results]
        test_sharpes = [r.test_metrics.get('sharpe_ratio', 0) for r in self.window_results]
        
        # Scatter plot
        scatter = ax.scatter(train_sharpes, test_sharpes, 
                           c=range(len(train_sharpes)), 
                           cmap='viridis', 
                           s=100, 
                           alpha=0.7,
                           edgecolors='black')
        
        # Add labels for each point
        for i, (x, y) in enumerate(zip(train_sharpes, test_sharpes)):
            ax.text(x, y, f'{i}', fontsize=8, ha='center', va='center')
        
        # Add diagonal line (perfect correlation)
        lims = [
            min(min(train_sharpes), min(test_sharpes)),
            max(max(train_sharpes), max(test_sharpes))
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Correlation')
        
        # Add regression line
        if len(train_sharpes) > 1:
            z = np.polyfit(train_sharpes, test_sharpes, 1)
            p = np.poly1d(z)
            ax.plot(train_sharpes, p(train_sharpes), "r--", alpha=0.7, 
                   label=f'Regression (r={np.corrcoef(train_sharpes, test_sharpes)[0,1]:.2f})')
        
        ax.set_xlabel('Train Sharpe Ratio')
        ax.set_ylabel('Test Sharpe Ratio')
        ax.set_title('Train-Test Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Split ID')
        
        plt.tight_layout()
        return fig
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of analysis results.
        
        Returns:
            Dict[str, Any]: Summary dictionary
        """
        if not self.final_results:
            raise ValueError("No results available. Run analysis first.")
        
        summary = {
            'strategy': self.strategy_name,
            'splits': len(self.window_results),
            'overall_performance': self.final_results.performance_classification['overall_performance'],
            'stability': self.final_results.performance_classification['stability']['overall_stability'],
            'robustness': self.final_results.performance_classification['robustness']['overall_robustness'],
            'key_metrics': {
                'mean_test_sharpe': self.final_results.aggregate_test_metrics['mean_sharpe'],
                'mean_test_return': self.final_results.aggregate_test_metrics['mean_return'],
                'mean_test_drawdown': self.final_results.aggregate_test_metrics['mean_drawdown'],
                'return_consistency': self.final_results.consistency_metrics['return_consistency'] * 100,
                'train_test_ratio': self.final_results.consistency_metrics['train_test_ratio']
            },
            'recommendation': self._generate_recommendation()
        }
        
        return summary
    
    def _generate_recommendation(self) -> Dict[str, Any]:
        """
        Generate trading recommendation based on analysis.
        
        Returns:
            Dict[str, Any]: Recommendation
        """
        if not self.final_results:
            return {'status': 'No analysis performed'}
        
        metrics = self.final_results
        performance = metrics.performance_classification['overall_performance']
        stability = metrics.performance_classification['stability']['overall_stability']
        robustness = metrics.performance_classification['robustness']['overall_robustness']
        
        # Decision matrix
        if performance in ['Excellent', 'Good']:
            if stability in ['Excellent', 'Good'] and robustness in ['Excellent', 'Good']:
                status = 'STRONG BUY'
                confidence = 'High'
                reasoning = 'Excellent performance with strong stability and robustness'
            elif stability in ['Excellent', 'Good']:
                status = 'BUY'
                confidence = 'Medium'
                reasoning = 'Good performance with strong stability'
            else:
                status = 'CAUTIOUS BUY'
                confidence = 'Low'
                reasoning = 'Good performance but stability concerns'
        
        elif performance == 'Average':
            if stability in ['Excellent', 'Good']:
                status = 'HOLD'
                confidence = 'Medium'
                reasoning = 'Average performance but good stability'
            else:
                status = 'AVOID'
                confidence = 'Low'
                reasoning = 'Average performance with stability concerns'
        
        else:  # Poor or Very Poor
            status = 'AVOID'
            confidence = 'High'
            reasoning = 'Poor performance metrics'
        
        recommendation = {
            'status': status,
            'confidence': confidence,
            'reasoning': reasoning,
            'suggested_action': self._get_suggested_action(status)
        }
        
        return recommendation
    
    def _get_suggested_action(self, status: str) -> str:
        """Get suggested action based on recommendation status."""
        actions = {
            'STRONG BUY': 'Allocate significant capital, consider aggressive position sizing',
            'BUY': 'Allocate moderate capital, standard position sizing',
            'CAUTIOUS BUY': 'Allocate small capital, reduced position sizing',
            'HOLD': 'Maintain existing positions if any, no new allocations',
            'AVOID': 'Close existing positions, avoid new allocations',
            'SELL': 'Close all positions, consider shorting if strategy allows'
        }
        return actions.get(status, 'No specific recommendation')
    
    def export_report(self, filepath: str, format: str = 'html') -> None:
        """
        Export comprehensive analysis report.
        
        Args:
            filepath: Path to save report
            format: Report format (html, pdf, markdown)
        """
        if not self.final_results:
            raise ValueError("No results to export. Run analysis first.")
        
        if format == 'html':
            self._export_html_report(filepath)
        elif format == 'markdown':
            self._export_markdown_report(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Exported report to {filepath}")
    
    def _export_html_report(self, filepath: str) -> None:
        """Export HTML report."""
        import jinja2
        
        # Prepare template data
        summary = self.get_summary()
        
        template_data = {
            'strategy_name': self.strategy_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': summary,
            'final_results': self.final_results.to_dict() if self.final_results else {},
            'window_results': [r.to_dict() for r in self.window_results] if self.window_results else []
        }
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Walk-Forward Analysis Report: {{ strategy_name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 30px 0; }
                .metric-box { display: inline-block; margin: 10px; padding: 15px; 
                             background: #e8f4f8; border-radius: 5px; }
                .recommendation { padding: 20px; background: #fff3cd; border-radius: 5px; }
                .good { color: green; font-weight: bold; }
                .average { color: orange; font-weight: bold; }
                .poor { color: red; font-weight: bold; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Walk-Forward Analysis Report</h1>
                <h2>Strategy: {{ strategy_name }}</h2>
                <p>Generated: {{ timestamp }}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="recommendation">
                    <h3>Recommendation: {{ summary.recommendation.status }}</h3>
                    <p><strong>Confidence:</strong> {{ summary.recommendation.confidence }}</p>
                    <p><strong>Reasoning:</strong> {{ summary.recommendation.reasoning }}</p>
                    <p><strong>Suggested Action:</strong> {{ summary.recommendation.suggested_action }}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Key Metrics</h2>
                <div>
                    {% for key, value in summary.key_metrics.items() %}
                    <div class="metric-box">
                        <strong>{{ key.replace('_', ' ').title() }}:</strong><br>
                        {{ "%.2f"|format(value) }}
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Classification</h2>
                <table>
                    <tr>
                        <th>Aspect</th>
                        <th>Classification</th>
                        <th>Score</th>
                    </tr>
                    <tr>
                        <td>Overall Performance</td>
                        <td class="{{ summary.overall_performance.lower().replace(' ', '-') }}">
                            {{ summary.overall_performance }}
                        </td>
                        <td>{{ "%.2f"|format(summary.key_metrics.mean_test_sharpe) }}</td>
                    </tr>
                    <tr>
                        <td>Stability</td>
                        <td class="{{ summary.stability.lower().replace(' ', '-') }}">
                            {{ summary.stability }}
                        </td>
                        <td>{{ "%.2f"|format(summary.key_metrics.return_consistency) }}%</td>
                    </tr>
                    <tr>
                        <td>Robustness</td>
                        <td class="{{ summary.robustness.lower().replace(' ', '-') }}">
                            {{ summary.robustness }}
                        </td>
                        <td>{{ "%.2f"|format(summary.key_metrics.train_test_ratio) }}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                <p>Analysis performed over {{ summary.splits }} walk-forward splits.</p>
                <p>See generated plots for detailed visual analysis.</p>
            </div>
            
            <div class="section">
                <h2>Configuration</h2>
                <p><strong>Optimization Method:</strong> {{ final_results.config.optimization_method.value }}</p>
                <p><strong>Optimization Metric:</strong> {{ final_results.config.optimization_metric.value }}</p>
                <p><strong>Splits:</strong> {{ final_results.config.splits }}</p>
            </div>
        </body>
        </html>
        """
        
        # Render template
        template = jinja2.Template(html_template)
        html_content = template.render(**template_data)
        
        # Save to file
        with open(filepath, 'w') as f:
            f.write(html_content)
    
    def _export_markdown_report(self, filepath: str) -> None:
        """Export Markdown report."""
        summary = self.get_summary()
        
        with open(filepath, 'w') as f:
            f.write(f"# Walk-Forward Analysis Report\n\n")
            f.write(f"**Strategy:** {self.strategy_name}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"**Recommendation:** {summary['recommendation']['status']}\n")
            f.write(f"**Confidence:** {summary['recommendation']['confidence']}\n")
            f.write(f"**Reasoning:** {summary['recommendation']['reasoning']}\n")
            f.write(f"**Suggested Action:** {summary['recommendation']['suggested_action']}\n\n")
            
            f.write("## Key Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in summary['key_metrics'].items():
                f.write(f"| {key.replace('_', ' ').title()} | {value:.2f} |\n")
            f.write("\n")
            
            f.write("## Performance Classification\n\n")
            f.write(f"- **Overall Performance:** {summary['overall_performance']}\n")
            f.write(f"- **Stability:** {summary['stability']}\n")
            f.write(f"- **Robustness:** {summary['robustness']}\n\n")
            
            f.write(f"## Analysis Details\n\n")
            f.write(f"- **Number of Splits:** {summary['splits']}\n")
            f.write(f"- **Optimization Method:** {self.config.optimization_method.value}\n")
            f.write(f"- **Optimization Metric:** {self.config.optimization_metric.value}\n\n")

# Example usage
if __name__ == "__main__":
    print("Testing Walk-Forward Analyzer...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=2000, freq='1H')
    prices = 50000 + np.cumsum(np.random.randn(2000) * 100)
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(2000) * 50,
        'high': prices + np.abs(np.random.randn(2000) * 100),
        'low': prices - np.abs(np.random.randn(2000) * 100),
        'close': prices,
        'volume': np.random.rand(2000) * 1000
    }, index=dates)
    
    # Import sample strategy
    from backtest_engine import SampleMovingAverageStrategy
    
    # Create WFA configuration
    wfa_config = WFAConfig(
        initial_train_size=252,  # 252 days training
        test_size=63,           # 63 days testing
        step_size=21,           # 21 days step
        optimization_method=WFAOptimizationMethod.GRID_SEARCH,
        optimization_metric=WFAMetric.SHARPE_RATIO,
        param_grid={
            'fast_period': [5, 10, 20],
            'slow_period': [20, 30, 50]
        },
        backtest_config=BacktestConfig(
            initial_capital=10000,
            trading_fee=0.001,
            timeframe="1h",
            verbose=False
        ),
        parallel=True,
        max_workers=2,
        verbose=True
    )
    
    # Create analyzer
    analyzer = WalkForwardAnalyzer(wfa_config)
    analyzer.load_data(df, "BTC/USDT")
    analyzer.set_strategy(SampleMovingAverageStrategy)
    
    # Run analysis
    print("\nStarting walk-forward analysis...")
    results = analyzer.run_analysis()
    
    # Get summary
    summary = analyzer.get_summary()
    print("\n" + "="*60)
    print("WALK-FORWARD ANALYSIS SUMMARY")
    print("="*60)
    print(f"Strategy: {summary['strategy']}")
    print(f"Splits Analyzed: {summary['splits']}")
    print(f"\nRecommendation: {summary['recommendation']['status']}")
    print(f"Confidence: {summary['recommendation']['confidence']}")
    print(f"\nKey Metrics:")
    for key, value in summary['key_metrics'].items():
        print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
    print(f"\nPerformance: {summary['overall_performance']}")
    print(f"Stability: {summary['stability']}")
    print(f"Robustness: {summary['robustness']}")
    print("="*60)
    
    # Export report
    analyzer.export_report("wfa_report.html", format="html")
    print("\nReport exported to wfa_report.html")
    
    print("\nWalk-forward analysis completed successfully!")