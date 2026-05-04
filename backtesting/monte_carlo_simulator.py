"""
Monte Carlo Simulation module for trading strategy risk analysis.
Provides probabilistic analysis of trading strategies through simulation.
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
from scipy import stats
from scipy.stats import norm, t, skew, kurtosis
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import traceback
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Import project modules
from logger import get_logger
from backtest_engine import BacktestEngine, BacktestConfig, PerformanceMetrics
from cache import TradingCache, cached

logger = get_logger(__name__)

class MonteCarloMethod(Enum):
    """Monte Carlo simulation methods."""
    HISTORICAL_BOOTSTRAP = "historical_bootstrap"
    PARAMETRIC = "parametric"
    GARCH = "garch"
    COPULA = "copula"
    GEOMETRIC_BROWNIAN = "geometric_brownian"

class RiskMetric(Enum):
    """Risk metrics for Monte Carlo analysis."""
    VAR = "var"  # Value at Risk
    CVAR = "cvar"  # Conditional Value at Risk (Expected Shortfall)
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    ULGER_INDEX = "ulger_index"
    TAIL_RATIO = "tail_ratio"

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    # Simulation parameters
    method: MonteCarloMethod = MonteCarloMethod.HISTORICAL_BOOTSTRAP
    simulations: int = 1000
    time_horizon: int = 252  # Trading days
    block_size: int = 5  # For block bootstrap
    
    # Risk parameters
    confidence_level: float = 0.95
    risk_metrics: List[RiskMetric] = field(default_factory=lambda: [
        RiskMetric.VAR, RiskMetric.CVAR, RiskMetric.MAX_DRAWDOWN
    ])
    
    # Distribution parameters
    distribution: str = "normal"  # normal, student-t, skewed-t
    degrees_freedom: int = 5  # For student-t distribution
    
    # GARCH parameters
    garch_p: int = 1
    garch_q: int = 1
    
    # Trading parameters
    initial_capital: float = 10000.0
    trading_fee: float = 0.001
    slippage: float = 0.0005
    
    # Strategy parameters (optional)
    strategy_params: Optional[Dict[str, Any]] = None
    
    # Data parameters
    return_type: str = "log"  # log or simple
    annualization_factor: int = 252
    
    # Parallel processing
    parallel: bool = True
    max_workers: int = 4
    
    # Output
    verbose: bool = True
    save_results: bool = True
    results_dir: str = "monte_carlo_results"
    save_plots: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")

@dataclass
class SimulationResult:
    """Result of a single Monte Carlo simulation."""
    simulation_id: int
    final_value: float
    returns: List[float]
    equity_curve: List[float]
    drawdown_curve: List[float]
    peak_values: List[float]
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'simulation_id': self.simulation_id,
            'final_value': self.final_value,
            'total_return': (self.final_value / self.equity_curve[0] - 1) * 100 if self.equity_curve else 0,
            'max_drawdown': min(self.drawdown_curve) * 100 if self.drawdown_curve else 0,
            'sharpe_ratio': self.metrics.get('sharpe_ratio', 0),
            'metrics': self.metrics,
            'parameters': self.parameters
        }

@dataclass
class MonteCarloResults:
    """Aggregated results from Monte Carlo simulation."""
    # Configuration
    config: Dict[str, Any]
    
    # Simulation results
    simulation_results: List[SimulationResult]
    
    # Aggregate statistics
    aggregate_stats: Dict[str, float]
    
    # Risk metrics
    risk_metrics: Dict[str, float]
    
    # Percentiles
    percentiles: Dict[str, List[float]]
    
    # Distribution parameters
    distribution_params: Dict[str, float]
    
    # Performance classification
    performance_classification: Dict[str, Any]
    
    # Timestamps
    created_at: datetime
    total_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'config': self.config,
            'simulation_results': [r.to_dict() for r in self.simulation_results],
            'aggregate_stats': self.aggregate_stats,
            'risk_metrics': self.risk_metrics,
            'percentiles': self.percentiles,
            'distribution_params': self.distribution_params,
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
        
        logger.info(f"Saved Monte Carlo results to {path}")

class MonteCarloSimulator:
    """
    Monte Carlo simulator for trading strategy risk analysis.
    Provides probabilistic analysis through multiple simulations.
    """
    
    def __init__(self, config: MonteCarloConfig):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            config: Monte Carlo configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.MonteCarloSimulator")
        
        # Data storage
        self.returns: Optional[pd.Series] = None
        self.prices: Optional[pd.Series] = None
        self.volatility: Optional[float] = None
        self.drift: Optional[float] = None
        
        # Results storage
        self.simulation_results: List[SimulationResult] = []
        self.final_results: Optional[MonteCarloResults] = None
        
        # Cache for simulations
        self.cache = TradingCache(cache_type="memory", max_size=1000)
        
        # Strategy information
        self.strategy: Optional[Any] = None
        self.strategy_name: str = ""
        
        # GARCH model (if used)
        self.garch_model: Optional[Any] = None
        
        self.logger.info("Initialized MonteCarloSimulator")
    
    def load_returns(self, 
                    returns: Union[pd.Series, List[float], np.ndarray],
                    prices: Optional[pd.Series] = None) -> None:
        """
        Load historical returns for simulation.
        
        Args:
            returns: Historical returns series
            prices: Historical prices (optional, for GBM)
        """
        if isinstance(returns, (list, np.ndarray)):
            self.returns = pd.Series(returns)
        else:
            self.returns = returns.copy()
        
        if prices is not None:
            self.prices = prices.copy() if isinstance(prices, pd.Series) else pd.Series(prices)
        
        # Calculate statistics
        if self.returns is not None and len(self.returns) > 0:
            self.drift = np.mean(self.returns)
            self.volatility = np.std(self.returns)
            
            self.logger.info(f"Loaded {len(self.returns)} returns")
            self.logger.info(f"Drift: {self.drift:.6f}, Volatility: {self.volatility:.6f}")
            self.logger.info(f"Mean Return: {self.drift * 100:.4f}%, "
                           f"Std Dev: {self.volatility * 100:.4f}%")
    
    def load_price_data(self, 
                       prices: pd.Series,
                       return_type: str = "log") -> None:
        """
        Load price data and calculate returns.
        
        Args:
            prices: Price series
            return_type: Type of returns (log or simple)
        """
        self.prices = prices.copy()
        
        if return_type == "log":
            self.returns = np.log(prices / prices.shift(1)).dropna()
        else:  # simple returns
            self.returns = (prices / prices.shift(1) - 1).dropna()
        
        self.load_returns(self.returns, prices)
    
    def set_strategy(self, 
                    strategy: Optional[Any] = None,
                    strategy_name: str = "") -> None:
        """
        Set trading strategy for simulation.
        
        Args:
            strategy: Trading strategy object
            strategy_name: Strategy name
        """
        self.strategy = strategy
        self.strategy_name = strategy_name or (strategy.__class__.__name__ if strategy else "No Strategy")
        self.logger.info(f"Set strategy: {self.strategy_name}")
    
    def _historical_bootstrap(self, 
                            n_periods: int,
                            block_size: int = 1) -> np.ndarray:
        """
        Generate simulated returns using historical bootstrap.
        
        Args:
            n_periods: Number of periods to simulate
            block_size: Block size for block bootstrap
        
        Returns:
            np.ndarray: Simulated returns
        """
        if self.returns is None:
            raise ValueError("No returns data loaded")
        
        returns_array = self.returns.values
        
        if block_size == 1:
            # Simple bootstrap
            indices = np.random.randint(0, len(returns_array), n_periods)
            simulated_returns = returns_array[indices]
        else:
            # Block bootstrap
            n_blocks = int(np.ceil(n_periods / block_size))
            block_starts = np.random.randint(0, len(returns_array) - block_size + 1, n_blocks)
            
            simulated_returns = []
            for start in block_starts:
                block = returns_array[start:start + block_size]
                simulated_returns.extend(block)
            
            simulated_returns = np.array(simulated_returns[:n_periods])
        
        return simulated_returns
    
    def _parametric_simulation(self, 
                             n_periods: int,
                             distribution: str = "normal") -> np.ndarray:
        """
        Generate simulated returns using parametric method.
        
        Args:
            n_periods: Number of periods to simulate
            distribution: Distribution type
        
        Returns:
            np.ndarray: Simulated returns
        """
        if self.drift is None or self.volatility is None:
            raise ValueError("No statistics calculated from returns")
        
        if distribution == "normal":
            simulated_returns = np.random.normal(
                self.drift, 
                self.volatility, 
                n_periods
            )
        
        elif distribution == "student-t":
            # Fit student-t distribution
            if len(self.returns) > 10:
                df, loc, scale = stats.t.fit(self.returns.values)
                simulated_returns = stats.t.rvs(df, loc, scale, n_periods)
            else:
                # Fall back to normal
                simulated_returns = np.random.normal(self.drift, self.volatility, n_periods)
        
        elif distribution == "skewed-t":
            # Skewed t-distribution (using scipy's skewnorm as approximation)
            if len(self.returns) > 10:
                skewness = skew(self.returns.values)
                simulated_returns = stats.skewnorm.rvs(
                    skewness, 
                    self.drift, 
                    self.volatility, 
                    n_periods
                )
            else:
                simulated_returns = np.random.normal(self.drift, self.volatility, n_periods)
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return simulated_returns
    
    def _garch_simulation(self, n_periods: int) -> np.ndarray:
        """
        Generate simulated returns using GARCH model.
        
        Args:
            n_periods: Number of periods to simulate
        
        Returns:
            np.ndarray: Simulated returns
        """
        try:
            from arch import arch_model
        except ImportError:
            self.logger.warning("ARCH package not installed, falling back to parametric")
            return self._parametric_simulation(n_periods)
        
        if self.returns is None:
            raise ValueError("No returns data loaded")
        
        # Fit GARCH model if not already fitted
        if self.garch_model is None:
            try:
                self.garch_model = arch_model(
                    self.returns * 100,  # Scale for better convergence
                    vol='Garch',
                    p=self.config.garch_p,
                    q=self.config.garch_q,
                    dist='normal'
                )
                garch_fit = self.garch_model.fit(disp='off')
                self.garch_model = garch_fit
            except Exception as e:
                self.logger.error(f"GARCH fitting failed: {e}")
                return self._parametric_simulation(n_periods)
        
        # Forecast volatility
        try:
            forecast = self.garch_model.forecast(horizon=n_periods)
            conditional_volatility = np.sqrt(forecast.variance.values[-1, :]) / 100
            
            # Simulate returns with time-varying volatility
            simulated_returns = np.random.normal(
                self.drift,
                conditional_volatility,
                n_periods
            )
            
            return simulated_returns
        
        except Exception as e:
            self.logger.error(f"GARCH forecast failed: {e}")
            return self._parametric_simulation(n_periods)
    
    def _geometric_brownian_motion(self, n_periods: int) -> np.ndarray:
        """
        Generate simulated prices using Geometric Brownian Motion.
        
        Args:
            n_periods: Number of periods to simulate
        
        Returns:
            np.ndarray: Simulated returns
        """
        if self.prices is None or len(self.prices) == 0:
            raise ValueError("No price data loaded for GBM")
        
        if self.drift is None or self.volatility is None:
            raise ValueError("No statistics calculated")
        
        # Initial price
        S0 = self.prices.iloc[-1]
        
        # Time step (assuming daily)
        dt = 1 / self.config.annualization_factor
        
        # Generate price path
        price_path = [S0]
        
        for _ in range(n_periods):
            # GBM equation: dS = μS dt + σS dW
            dW = np.random.normal(0, np.sqrt(dt))
            dS = self.drift * dt + self.volatility * dW
            
            # Update price
            new_price = price_path[-1] * np.exp(dS)
            price_path.append(new_price)
        
        # Calculate returns from price path
        price_array = np.array(price_path)
        returns = np.diff(price_array) / price_array[:-1]
        
        return returns
    
    def _copula_simulation(self, n_periods: int) -> np.ndarray:
        """
        Generate simulated returns using copula method.
        Simplified implementation using Gaussian copula.
        
        Args:
            n_periods: Number of periods to simulate
        
        Returns:
            np.ndarray: Simulated returns
        """
        if self.returns is None:
            raise ValueError("No returns data loaded")
        
        # Transform to uniform using empirical CDF
        from scipy.stats import rankdata
        
        n = len(self.returns)
        ranks = rankdata(self.returns)
        uniforms = ranks / (n + 1)
        
        # Transform to normal
        normals = norm.ppf(uniforms)
        
        # Generate correlated normals (simplified - single asset)
        simulated_normals = np.random.normal(np.mean(normals), np.std(normals), n_periods)
        
        # Transform back using inverse CDF
        simulated_uniforms = norm.cdf(simulated_normals)
        
        # Transform back to returns using empirical inverse CDF
        sorted_returns = np.sort(self.returns.values)
        simulated_returns = np.percentile(sorted_returns, simulated_uniforms * 100)
        
        return simulated_returns
    
    def generate_returns(self, 
                        n_periods: int,
                        method: Optional[MonteCarloMethod] = None) -> np.ndarray:
        """
        Generate simulated returns using specified method.
        
        Args:
            n_periods: Number of periods to simulate
            method: Simulation method
        
        Returns:
            np.ndarray: Simulated returns
        """
        if method is None:
            method = self.config.method
        
        if method == MonteCarloMethod.HISTORICAL_BOOTSTRAP:
            returns = self._historical_bootstrap(n_periods, self.config.block_size)
        
        elif method == MonteCarloMethod.PARAMETRIC:
            returns = self._parametric_simulation(n_periods, self.config.distribution)
        
        elif method == MonteCarloMethod.GARCH:
            returns = self._garch_simulation(n_periods)
        
        elif method == MonteCarloMethod.GEOMETRIC_BROWNIAN:
            returns = self._geometric_brownian_motion(n_periods)
        
        elif method == MonteCarloMethod.COPULA:
            returns = self._copula_simulation(n_periods)
        
        else:
            raise ValueError(f"Unknown simulation method: {method}")
        
        return returns
    
    def _simulate_trading_strategy(self,
                                 returns: np.ndarray,
                                 simulation_id: int) -> SimulationResult:
        """
        Simulate trading strategy on generated returns.
        
        Args:
            returns: Simulated returns
            simulation_id: Simulation ID
        
        Returns:
            SimulationResult: Simulation results
        """
        # Initial values
        initial_capital = self.config.initial_capital
        capital = initial_capital
        equity_curve = [capital]
        drawdown_curve = [0.0]
        peak = capital
        
        # Store returns
        period_returns = []
        
        # Apply strategy if available
        if self.strategy is not None:
            # This is a simplified implementation
            # In practice, you would apply your actual trading logic
            for ret in returns:
                # Example: Simple buy-and-hold with fees
                capital *= (1 + ret) * (1 - self.config.trading_fee)
                
                # Update equity curve
                equity_curve.append(capital)
                period_returns.append(ret)
                
                # Update drawdown
                if capital > peak:
                    peak = capital
                drawdown = (peak - capital) / peak
                drawdown_curve.append(drawdown)
        else:
            # Simple buy-and-hold
            for ret in returns:
                capital *= (1 + ret)
                equity_curve.append(capital)
                period_returns.append(ret)
                
                if capital > peak:
                    peak = capital
                drawdown = (peak - capital) / peak
                drawdown_curve.append(drawdown)
        
        # Calculate metrics
        equity_array = np.array(equity_curve)
        returns_array = np.array(period_returns)
        
        # Basic metrics
        total_return = (capital - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (self.config.annualization_factor / len(returns)) - 1
        
        # Sharpe ratio (assuming 0 risk-free rate)
        if len(returns_array) > 1:
            excess_returns = returns_array  # Assuming 0 risk-free
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.config.annualization_factor)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        max_drawdown = np.min(drawdown_curve)
        
        # Sortino ratio
        if len(returns_array) > 1:
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino_ratio = np.mean(returns_array) / downside_std * np.sqrt(self.config.annualization_factor)
            else:
                sortino_ratio = float('inf')
        else:
            sortino_ratio = 0.0
        
        # Calmar ratio
        if max_drawdown > 0:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = float('inf')
        
        # Compile metrics
        metrics = {
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown * 100,
            'volatility': np.std(returns_array) * np.sqrt(self.config.annualization_factor) * 100
        }
        
        # Create result
        result = SimulationResult(
            simulation_id=simulation_id,
            final_value=capital,
            returns=period_returns,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            peak_values=[peak] * len(equity_curve),
            metrics=metrics,
            parameters={
                'method': self.config.method.value,
                'distribution': self.config.distribution
            }
        )
        
        return result
    
    def _run_single_simulation(self, 
                             simulation_id: int) -> SimulationResult:
        """
        Run a single Monte Carlo simulation.
        
        Args:
            simulation_id: Simulation ID
        
        Returns:
            SimulationResult: Simulation results
        """
        try:
            # Generate returns
            returns = self.generate_returns(self.config.time_horizon)
            
            # Simulate trading
            result = self._simulate_trading_strategy(returns, simulation_id)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error in simulation {simulation_id}: {e}")
            # Return empty result
            return SimulationResult(
                simulation_id=simulation_id,
                final_value=self.config.initial_capital,
                returns=[],
                equity_curve=[self.config.initial_capital],
                drawdown_curve=[0.0],
                peak_values=[self.config.initial_capital],
                metrics={},
                parameters={}
            )
    
    def run_simulation(self) -> MonteCarloResults:
        """
        Run complete Monte Carlo simulation.
        
        Returns:
            MonteCarloResults: Simulation results
        """
        if self.returns is None or len(self.returns) == 0:
            raise ValueError("No returns data loaded")
        
        self.logger.info(f"Starting Monte Carlo simulation with {self.config.simulations} iterations")
        self.logger.info(f"Method: {self.config.method.value}")
        self.logger.info(f"Time horizon: {self.config.time_horizon} periods")
        self.logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")
        
        # Run simulations
        start_time = datetime.now()
        
        if self.config.parallel and self.config.max_workers > 1:
            self.simulation_results = self._run_parallel_simulations()
        else:
            self.simulation_results = []
            
            for i in tqdm(range(self.config.simulations), 
                         desc="Monte Carlo Simulation",
                         disable=not self.config.verbose):
                result = self._run_single_simulation(i)
                self.simulation_results.append(result)
                
                if self.config.verbose and (i + 1) % 100 == 0:
                    self.logger.info(f"Completed {i + 1}/{self.config.simulations} simulations")
        
        # Calculate final results
        total_time = (datetime.now() - start_time).total_seconds()
        self.final_results = self._calculate_final_results(total_time)
        
        self.logger.info(f"Monte Carlo simulation completed in {total_time:.1f} seconds")
        
        # Save results if configured
        if self.config.save_results:
            self.save_results()
        
        # Generate plots if configured
        if self.config.save_plots:
            self.plot_results()
        
        return self.final_results
    
    def _run_parallel_simulations(self) -> List[SimulationResult]:
        """
        Run simulations in parallel.
        
        Returns:
            List[SimulationResult]: Simulation results
        """
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all simulations
            future_to_id = {
                executor.submit(self._run_single_simulation, i): i
                for i in range(self.config.simulations)
            }
            
            # Process completed simulations
            for future in tqdm(as_completed(future_to_id), 
                             total=self.config.simulations,
                             desc="Parallel Monte Carlo",
                             disable=not self.config.verbose):
                sim_id = future_to_id[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in simulation {sim_id}: {e}")
                    self.logger.error(traceback.format_exc())
        
        # Sort by simulation ID
        results.sort(key=lambda x: x.simulation_id)
        return results
    
    def _calculate_final_results(self, total_time: float) -> MonteCarloResults:
        """
        Calculate final aggregated results.
        
        Args:
            total_time: Total simulation time in seconds
        
        Returns:
            MonteCarloResults: Final results
        """
        if not self.simulation_results:
            raise ValueError("No simulation results available")
        
        # Extract final values and metrics
        final_values = [r.final_value for r in self.simulation_results]
        total_returns = [(v / self.config.initial_capital - 1) * 100 for v in final_values]
        sharpe_ratios = [r.metrics.get('sharpe_ratio', 0) for r in self.simulation_results]
        max_drawdowns = [r.metrics.get('max_drawdown', 0) for r in self.simulation_results]
        
        # Calculate aggregate statistics
        aggregate_stats = {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'mean_total_return': np.mean(total_returns),
            'median_total_return': np.median(total_returns),
            'std_total_return': np.std(total_returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'median_max_drawdown': np.median(max_drawdowns),
            'success_rate': sum(1 for v in final_values if v > self.config.initial_capital) / len(final_values),
            'ruin_rate': sum(1 for v in final_values if v < self.config.initial_capital * 0.5) / len(final_values)
        }
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(final_values, total_returns)
        
        # Calculate percentiles
        percentiles = self._calculate_percentiles(final_values, total_returns)
        
        # Calculate distribution parameters
        distribution_params = self._calculate_distribution_parameters(total_returns)
        
        # Performance classification
        performance_classification = self._classify_performance(aggregate_stats, risk_metrics)
        
        # Create final results
        final_results = MonteCarloResults(
            config=asdict(self.config),
            simulation_results=self.simulation_results,
            aggregate_stats=aggregate_stats,
            risk_metrics=risk_metrics,
            percentiles=percentiles,
            distribution_params=distribution_params,
            performance_classification=performance_classification,
            created_at=datetime.now(),
            total_time=total_time
        )
        
        return final_results
    
    def _calculate_risk_metrics(self, 
                               final_values: List[float],
                               total_returns: List[float]) -> Dict[str, float]:
        """
        Calculate risk metrics from simulation results.
        
        Args:
            final_values: List of final portfolio values
            total_returns: List of total returns
        
        Returns:
            Dict[str, float]: Risk metrics
        """
        initial_capital = self.config.initial_capital
        returns_array = np.array(total_returns) / 100  # Convert to decimal
        
        risk_metrics = {}
        
        # Value at Risk (VaR)
        if RiskMetric.VAR in self.config.risk_metrics:
            var_level = 1 - self.config.confidence_level
            var_historical = np.percentile(total_returns, var_level * 100)
            
            # Parametric VaR (assuming normal distribution)
            if len(returns_array) > 1:
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                var_parametric = norm.ppf(var_level, mean_return, std_return) * initial_capital
            else:
                var_parametric = 0.0
            
            risk_metrics['var_historical'] = var_historical
            risk_metrics['var_parametric'] = var_parametric
            risk_metrics['var'] = var_historical  # Default to historical
        
        # Conditional VaR (Expected Shortfall)
        if RiskMetric.CVAR in self.config.risk_metrics:
            var_level = 1 - self.config.confidence_level
            var_threshold = np.percentile(total_returns, var_level * 100)
            tail_returns = [r for r in total_returns if r <= var_threshold]
            
            if tail_returns:
                cvar = np.mean(tail_returns)
            else:
                cvar = var_threshold
            
            risk_metrics['cvar'] = cvar
        
        # Maximum Drawdown distribution
        if RiskMetric.MAX_DRAWDOWN in self.config.risk_metrics:
            max_drawdowns = [r.metrics.get('max_drawdown', 0) for r in self.simulation_results]
            risk_metrics['avg_max_drawdown'] = np.mean(max_drawdowns)
            risk_metrics['worst_max_drawdown'] = np.max(max_drawdowns)
            risk_metrics['drawdown_95'] = np.percentile(max_drawdowns, 95)
        
        # Sharpe ratio statistics
        if RiskMetric.SHARPE_RATIO in self.config.risk_metrics:
            sharpe_ratios = [r.metrics.get('sharpe_ratio', 0) for r in self.simulation_results]
            risk_metrics['sharpe_ratio_mean'] = np.mean(sharpe_ratios)
            risk_metrics['sharpe_ratio_std'] = np.std(sharpe_ratios)
            risk_metrics['sharpe_ratio_negative_prob'] = sum(1 for s in sharpe_ratios if s < 0) / len(sharpe_ratios)
        
        # Sortino ratio
        if RiskMetric.SORTINO_RATIO in self.config.risk_metrics:
            sortino_ratios = [r.metrics.get('sortino_ratio', 0) for r in self.simulation_results]
            # Filter out infinite values
            finite_sortinos = [s for s in sortino_ratios if np.isfinite(s)]
            if finite_sortinos:
                risk_metrics['sortino_ratio_mean'] = np.mean(finite_sortinos)
                risk_metrics['sortino_ratio_std'] = np.std(finite_sortinos)
        
        # Calmar ratio
        if RiskMetric.CALMAR_RATIO in self.config.risk_metrics:
            calmar_ratios = []
            for r in self.simulation_results:
                calmar = r.metrics.get('calmar_ratio', 0)
                if np.isfinite(calmar):
                    calmar_ratios.append(calmar)
            
            if calmar_ratios:
                risk_metrics['calmar_ratio_mean'] = np.mean(calmar_ratios)
                risk_metrics['calmar_ratio_std'] = np.std(calmar_ratios)
        
        # Tail ratio
        if RiskMetric.TAIL_RATIO in self.config.risk_metrics:
            if len(returns_array) > 1:
                # Tail ratio: 95th percentile / 5th percentile (absolute)
                tail_95 = np.percentile(np.abs(returns_array), 95)
                tail_5 = np.percentile(np.abs(returns_array), 5)
                risk_metrics['tail_ratio'] = tail_95 / tail_5 if tail_5 > 0 else float('inf')
        
        # Ulcer Index
        if RiskMetric.ULGER_INDEX in self.config.risk_metrics:
            # Simplified ulcer index calculation
            ulcer_indices = []
            for result in self.simulation_results:
                if len(result.drawdown_curve) > 0:
                    ulcer = np.sqrt(np.mean(np.array(result.drawdown_curve) ** 2)) * 100
                    ulcer_indices.append(ulcer)
            
            if ulcer_indices:
                risk_metrics['ulcer_index_mean'] = np.mean(ulcer_indices)
                risk_metrics['ulcer_index_std'] = np.std(ulcer_indices)
        
        return risk_metrics
    
    def _calculate_percentiles(self,
                              final_values: List[float],
                              total_returns: List[float]) -> Dict[str, List[float]]:
        """
        Calculate percentiles for key metrics.
        
        Args:
            final_values: List of final portfolio values
            total_returns: List of total returns
        
        Returns:
            Dict[str, List[float]]: Percentiles for different metrics
        """
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        
        # Final value percentiles
        final_value_percentiles = np.percentile(final_values, percentiles).tolist()
        
        # Return percentiles
        return_percentiles = np.percentile(total_returns, percentiles).tolist()
        
        # Sharpe ratio percentiles
        sharpe_ratios = [r.metrics.get('sharpe_ratio', 0) for r in self.simulation_results]
        sharpe_percentiles = np.percentile(sharpe_ratios, percentiles).tolist()
        
        # Max drawdown percentiles
        max_drawdowns = [r.metrics.get('max_drawdown', 0) for r in self.simulation_results]
        drawdown_percentiles = np.percentile(max_drawdowns, percentiles).tolist()
        
        return {
            'percentiles': percentiles,
            'final_values': final_value_percentiles,
            'total_returns': return_percentiles,
            'sharpe_ratios': sharpe_percentiles,
            'max_drawdowns': drawdown_percentiles
        }
    
    def _calculate_distribution_parameters(self, total_returns: List[float]) -> Dict[str, float]:
        """
        Calculate distribution parameters of returns.
        
        Args:
            total_returns: List of total returns
        
        Returns:
            Dict[str, float]: Distribution parameters
        """
        returns_array = np.array(total_returns)
        
        if len(returns_array) < 2:
            return {}
        
        params = {
            'mean': np.mean(returns_array),
            'median': np.median(returns_array),
            'std': np.std(returns_array),
            'skewness': skew(returns_array),
            'kurtosis': kurtosis(returns_array),
            'min': np.min(returns_array),
            'max': np.max(returns_array),
            'range': np.max(returns_array) - np.min(returns_array),
            'jarque_bera_stat': 0,
            'jarque_bera_pvalue': 0
        }
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns_array)
            params['jarque_bera_stat'] = jb_stat
            params['jarque_bera_pvalue'] = jb_pvalue
        except:
            pass
        
        return params
    
    def _classify_performance(self,
                            aggregate_stats: Dict[str, float],
                            risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify performance based on simulation results.
        
        Args:
            aggregate_stats: Aggregate statistics
            risk_metrics: Risk metrics
        
        Returns:
            Dict[str, Any]: Performance classification
        """
        mean_return = aggregate_stats.get('mean_total_return', 0)
        mean_sharpe = aggregate_stats.get('mean_sharpe', 0)
        success_rate = aggregate_stats.get('success_rate', 0)
        ruin_rate = aggregate_stats.get('ruin_rate', 0)
        mean_drawdown = aggregate_stats.get('mean_max_drawdown', 0)
        
        # Return classification
        if mean_return >= 20:
            return_class = "Excellent"
        elif mean_return >= 10:
            return_class = "Good"
        elif mean_return >= 5:
            return_class = "Average"
        elif mean_return >= 0:
            return_class = "Poor"
        else:
            return_class = "Very Poor"
        
        # Sharpe ratio classification
        if mean_sharpe >= 1.5:
            sharpe_class = "Excellent"
        elif mean_sharpe >= 1.0:
            sharpe_class = "Good"
        elif mean_sharpe >= 0.5:
            sharpe_class = "Average"
        elif mean_sharpe >= 0:
            sharpe_class = "Poor"
        else:
            sharpe_class = "Very Poor"
        
        # Success rate classification
        if success_rate >= 0.7:
            success_class = "Excellent"
        elif success_rate >= 0.6:
            success_class = "Good"
        elif success_rate >= 0.5:
            success_class = "Average"
        elif success_rate >= 0.4:
            success_class = "Poor"
        else:
            success_class = "Very Poor"
        
        # Risk classification (based on ruin rate)
        if ruin_rate <= 0.01:
            risk_class = "Excellent"
        elif ruin_rate <= 0.05:
            risk_class = "Good"
        elif ruin_rate <= 0.10:
            risk_class = "Average"
        elif ruin_rate <= 0.20:
            risk_class = "Poor"
        else:
            risk_class = "Very Poor"
        
        # Drawdown classification
        if mean_drawdown <= 10:
            drawdown_class = "Excellent"
        elif mean_drawdown <= 20:
            drawdown_class = "Good"
        elif mean_drawdown <= 30:
            drawdown_class = "Average"
        elif mean_drawdown <= 40:
            drawdown_class = "Poor"
        else:
            drawdown_class = "Very Poor"
        
        # Overall classification
        class_scores = {
            "Excellent": 5,
            "Good": 4,
            "Average": 3,
            "Poor": 2,
            "Very Poor": 1
        }
        
        avg_score = (class_scores[return_class] + 
                    class_scores[sharpe_class] + 
                    class_scores[success_class] + 
                    (6 - class_scores[risk_class]) +  # Invert risk score (lower is better)
                    (6 - class_scores[drawdown_class])) / 5  # Invert drawdown score
        
        if avg_score >= 4.5:
            overall_class = "Excellent"
        elif avg_score >= 3.5:
            overall_class = "Good"
        elif avg_score >= 2.5:
            overall_class = "Average"
        elif avg_score >= 1.5:
            overall_class = "Poor"
        else:
            overall_class = "Very Poor"
        
        classifications = {
            'return': {'value': mean_return, 'class': return_class},
            'sharpe': {'value': mean_sharpe, 'class': sharpe_class},
            'success_rate': {'value': success_rate * 100, 'class': success_class},
            'risk': {'value': ruin_rate * 100, 'class': risk_class},
            'drawdown': {'value': mean_drawdown, 'class': drawdown_class},
            'overall': overall_class
        }
        
        return classifications
    
    def save_results(self, filepath: Optional[str] = None) -> None:
        """
        Save simulation results to file.
        
        Args:
            filepath: Optional custom filepath
        """
        if not self.final_results:
            raise ValueError("No results to save. Run simulation first.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monte_carlo_{self.strategy_name}_{timestamp}.json"
            filepath = Path(self.config.results_dir) / filename
        
        self.final_results.save(filepath)
    
    def plot_results(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Plot Monte Carlo simulation results.
        
        Args:
            save_dir: Directory to save plots
        
        Returns:
            Dict[str, plt.Figure]: Dictionary of figures
        """
        if not self.final_results:
            raise ValueError("No results to plot. Run simulation first.")
        
        if save_dir is None:
            save_dir = self.config.results_dir
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        figures = {}
        
        # 1. Equity Curve Distribution
        fig1 = self._plot_equity_curve_distribution()
        figures['equity_curve_distribution'] = fig1
        fig1.savefig(save_path / 'equity_curve_distribution.png', dpi=300, bbox_inches='tight')
        
        # 2. Final Value Distribution
        fig2 = self._plot_final_value_distribution()
        figures['final_value_distribution'] = fig2
        fig2.savefig(save_path / 'final_value_distribution.png', dpi=300, bbox_inches='tight')
        
        # 3. Risk-Return Scatter
        fig3 = self._plot_risk_return_scatter()
        figures['risk_return_scatter'] = fig3
        fig3.savefig(save_path / 'risk_return_scatter.png', dpi=300, bbox_inches='tight')
        
        # 4. Performance Metrics Distribution
        fig4 = self._plot_metrics_distribution()
        figures['metrics_distribution'] = fig4
        fig4.savefig(save_path / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
        
        # 5. VaR and CVAR Visualization
        fig5 = self._plot_var_cvar()
        figures['var_cvar'] = fig5
        fig5.savefig(save_path / 'var_cvar.png', dpi=300, bbox_inches='tight')
        
        # 6. Drawdown Analysis
        fig6 = self._plot_drawdown_analysis()
        figures['drawdown_analysis'] = fig6
        fig6.savefig(save_path / 'drawdown_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.close('all')
        
        self.logger.info(f"Saved plots to {save_path}")
        return figures
    
    def _plot_equity_curve_distribution(self) -> plt.Figure:
        """Plot distribution of equity curves."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Select a subset of equity curves to plot
        n_curves = min(100, len(self.simulation_results))
        colors = plt.cm.viridis(np.linspace(0, 1, n_curves))
        
        # Plot individual equity curves
        for i, result in enumerate(self.simulation_results[:n_curves]):
            equity_curve = result.equity_curve
            if len(equity_curve) > 1:
                # Normalize to starting value
                normalized = np.array(equity_curve) / equity_curve[0] * 100
                ax.plot(normalized, color=colors[i], alpha=0.1, linewidth=0.5)
        
        # Calculate and plot percentiles
        all_curves = []
        max_len = 0
        
        for result in self.simulation_results:
            equity_curve = result.equity_curve
            if len(equity_curve) > 1:
                normalized = np.array(equity_curve) / equity_curve[0] * 100
                all_curves.append(normalized)
                max_len = max(max_len, len(normalized))
        
        if all_curves:
            # Pad curves to same length
            padded_curves = []
            for curve in all_curves:
                if len(curve) < max_len:
                    padded = np.pad(curve, (0, max_len - len(curve)), 'edge')
                else:
                    padded = curve[:max_len]
                padded_curves.append(padded)
            
            padded_array = np.array(padded_curves)
            
            # Calculate percentiles
            percentiles = [5, 25, 50, 75, 95]
            percentile_values = np.percentile(padded_array, percentiles, axis=0)
            
            # Plot percentiles
            for i, p in enumerate(percentiles):
                ax.plot(percentile_values[i], label=f'{p}th percentile', linewidth=2, 
                       alpha=0.8 if p == 50 else 0.6)
            
            # Plot mean
            mean_curve = np.mean(padded_array, axis=0)
            ax.plot(mean_curve, 'k--', label='Mean', linewidth=3, alpha=0.9)
        
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Starting Value')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Portfolio Value (% of Initial)')
        ax.set_title('Monte Carlo: Equity Curve Distribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_final_value_distribution(self) -> plt.Figure:
        """Plot distribution of final portfolio values."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        final_values = [r.final_value for r in self.simulation_results]
        initial_capital = self.config.initial_capital
        
        # 1. Histogram with KDE
        ax1 = axes[0]
        n, bins, patches = ax1.hist(final_values, bins=50, alpha=0.7, 
                                   color='blue', edgecolor='black', density=True)
        
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(final_values)
        x_range = np.linspace(min(final_values), max(final_values), 1000)
        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Add reference lines
        ax1.axvline(x=initial_capital, color='green', linestyle='--', 
                   linewidth=2, label=f'Initial: ${initial_capital:,.0f}')
        ax1.axvline(x=np.mean(final_values), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: ${np.mean(final_values):,.0f}')
        ax1.axvline(x=np.median(final_values), color='orange', linestyle='--',
                   linewidth=2, label=f'Median: ${np.median(final_values):,.0f}')
        
        ax1.set_xlabel('Final Portfolio Value ($)')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Final Portfolio Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative Distribution Function (CDF)
        ax2 = axes[1]
        sorted_values = np.sort(final_values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        
        ax2.plot(sorted_values, cdf, 'b-', linewidth=2, label='CDF')
        
        # Add reference lines
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Median (50%)')
        ax2.axvline(x=initial_capital, color='g', linestyle='--', alpha=0.5)
        
        # Annotate key percentiles
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        for p in percentiles:
            value = np.percentile(final_values, p)
            ax2.axvline(x=value, color='gray', linestyle=':', alpha=0.3)
            ax2.text(value, 0.02, f'{p}%', rotation=90, fontsize=8, alpha=0.7)
        
        ax2.set_xlabel('Final Portfolio Value ($)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution of Final Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_risk_return_scatter(self) -> plt.Figure:
        """Plot risk-return scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract metrics
        final_values = [r.final_value for r in self.simulation_results]
        returns = [(v / self.config.initial_capital - 1) * 100 for v in final_values]
        sharpe_ratios = [r.metrics.get('sharpe_ratio', 0) for r in self.simulation_results]
        max_drawdowns = [r.metrics.get('max_drawdown', 0) for r in self.simulation_results]
        
        # Create scatter plot
        scatter = ax.scatter(max_drawdowns, returns, 
                           c=sharpe_ratios, 
                           cmap='viridis',
                           s=50, 
                           alpha=0.6,
                           edgecolors='black')
        
        # Add reference lines
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add efficient frontier concept
        if len(returns) > 10:
            # Calculate convex hull for efficient frontier visualization
            from scipy.spatial import ConvexHull
            
            points = np.column_stack([max_drawdowns, returns])
            try:
                hull = ConvexHull(points)
                
                # Plot convex hull
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.2)
                
                # Highlight best risk-adjusted returns (top 10% by Sharpe)
                top_indices = np.argsort(sharpe_ratios)[-len(sharpe_ratios)//10:]
                ax.scatter(np.array(max_drawdowns)[top_indices],
                          np.array(returns)[top_indices],
                          c='red', s=100, marker='*', 
                          edgecolors='gold', linewidth=2,
                          label='Top 10% by Sharpe')
            except:
                pass
        
        ax.set_xlabel('Maximum Drawdown (%)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Risk-Return Scatter Plot')
        
        # Add colorbar for Sharpe ratio
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_metrics_distribution(self) -> plt.Figure:
        """Plot distribution of key performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract metrics
        sharpe_ratios = [r.metrics.get('sharpe_ratio', 0) for r in self.simulation_results]
        total_returns = [(r.final_value / self.config.initial_capital - 1) * 100 
                        for r in self.simulation_results]
        max_drawdowns = [r.metrics.get('max_drawdown', 0) for r in self.simulation_results]
        
        # Filter finite values for sortino
        sortino_ratios = []
        for r in self.simulation_results:
            sortino = r.metrics.get('sortino_ratio', 0)
            if np.isfinite(sortino):
                sortino_ratios.append(sortino)
        
        # 1. Sharpe Ratio Distribution
        ax1 = axes[0, 0]
        ax1.hist(sharpe_ratios, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero')
        ax1.axvline(x=np.mean(sharpe_ratios), color='blue', linestyle='--', 
                   alpha=0.7, label=f'Mean: {np.mean(sharpe_ratios):.2f}')
        ax1.set_xlabel('Sharpe Ratio')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Sharpe Ratio Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Total Return Distribution
        ax2 = axes[0, 1]
        ax2.hist(total_returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax2.axvline(x=np.mean(total_returns), color='green', linestyle='--',
                   alpha=0.7, label=f'Mean: {np.mean(total_returns):.2f}%')
        ax2.set_xlabel('Total Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Total Return Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Maximum Drawdown Distribution
        ax3 = axes[1, 0]
        ax3.hist(max_drawdowns, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax3.axvline(x=np.mean(max_drawdowns), color='blue', linestyle='--',
                   alpha=0.7, label=f'Mean: {np.mean(max_drawdowns):.2f}%')
        
        # Add VaR line if available
        if 'var' in self.final_results.risk_metrics:
            var = self.final_results.risk_metrics['var']
            ax3.axvline(x=var, color='purple', linestyle='--',
                       alpha=0.7, label=f'VaR: {var:.2f}%')
        
        ax3.set_xlabel('Maximum Drawdown (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Maximum Drawdown Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Sortino Ratio Distribution
        ax4 = axes[1, 1]
        if sortino_ratios:
            ax4.hist(sortino_ratios, bins=30, alpha=0.7, color='purple', edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero')
            ax4.axvline(x=np.mean(sortino_ratios), color='blue', linestyle='--',
                       alpha=0.7, label=f'Mean: {np.mean(sortino_ratios):.2f}')
            ax4.set_xlabel('Sortino Ratio')
        else:
            ax4.text(0.5, 0.5, 'No finite Sortino ratios', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        ax4.set_ylabel('Frequency')
        ax4.set_title('Sortino Ratio Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_var_cvar(self) -> plt.Figure:
        """Plot Value at Risk and Conditional VaR."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        final_values = [r.final_value for r in self.simulation_results]
        returns = [(v / self.config.initial_capital - 1) * 100 for v in final_values]
        
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Calculate empirical CDF
        cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
        
        # Plot CDF
        ax.plot(sorted_returns, cdf, 'b-', linewidth=2, label='Empirical CDF')
        
        # Mark VaR and CVaR
        confidence_level = self.config.confidence_level
        var_level = (1 - confidence_level) * 100
        var_index = int(var_level / 100 * len(sorted_returns))
        
        if var_index < len(sorted_returns):
            var = sorted_returns[var_index]
            cvar = np.mean(sorted_returns[:var_index + 1])
            
            # Shade VaR region
            ax.axvspan(sorted_returns[0], var, alpha=0.3, color='red', 
                      label=f'VaR {var_level:.1f}%: {var:.2f}%')
            
            # Mark VaR line
            ax.axvline(x=var, color='red', linestyle='--', linewidth=2,
                      label=f'VaR ({confidence_level*100:.0f}%): {var:.2f}%')
            
            # Mark CVaR line
            ax.axvline(x=cvar, color='darkred', linestyle='--', linewidth=2,
                      label=f'CVaR: {cvar:.2f}%')
            
            # Add text annotations
            ax.text(var, 0.5, f'VaR\n{var:.1f}%', 
                   ha='right', va='center', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
            
            ax.text(cvar, 0.3, f'CVaR\n{cvar:.1f}%', 
                   ha='right', va='center', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='darkred', alpha=0.3))
        
        # Add reference lines
        ax.axhline(y=1-confidence_level, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Portfolio Return (%)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Value at Risk (VaR) and Conditional VaR (CVaR) at {confidence_level*100:.0f}% Confidence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_drawdown_analysis(self) -> plt.Figure:
        """Plot drawdown analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract drawdown curves
        all_drawdowns = []
        for result in self.simulation_results:
            if len(result.drawdown_curve) > 0:
                # Convert to percentage and take absolute value
                drawdowns = np.array(result.drawdown_curve) * 100
                all_drawdowns.append(drawdowns)
        
        if not all_drawdowns:
            return fig
        
        # Pad drawdown curves to same length
        max_len = max(len(d) for d in all_drawdowns)
        padded_drawdowns = []
        for drawdowns in all_drawdowns:
            if len(drawdowns) < max_len:
                padded = np.pad(drawdowns, (0, max_len - len(drawdowns)), 'edge')
            else:
                padded = drawdowns[:max_len]
            padded_drawdowns.append(padded)
        
        drawdown_array = np.array(padded_drawdowns)
        
        # 1. Drawdown distribution over time
        ax1 = axes[0]
        
        # Plot individual drawdown curves (subset)
        n_curves = min(50, len(drawdown_array))
        for i in range(n_curves):
            ax1.plot(drawdown_array[i], alpha=0.1, color='blue', linewidth=0.5)
        
        # Plot percentiles
        percentiles = [50, 75, 90, 95]
        colors = ['green', 'orange', 'red', 'darkred']
        
        for p, color in zip(percentiles, colors):
            percentile_curve = np.percentile(drawdown_array, p, axis=0)
            ax1.plot(percentile_curve, color=color, linewidth=2, 
                    label=f'{p}th percentile', alpha=0.8)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Drawdown (%)')
        ax1.set_title('Drawdown Distribution Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Maximum drawdown analysis
        ax2 = axes[1]
        
        max_drawdowns = [np.max(d) for d in all_drawdowns]
        
        # Histogram of maximum drawdowns
        n, bins, patches = ax2.hist(max_drawdowns, bins=30, alpha=0.7, 
                                   color='red', edgecolor='black', density=True)
        
        # Add KDE
        kde = stats.gaussian_kde(max_drawdowns)
        x_range = np.linspace(min(max_drawdowns), max(max_drawdowns), 1000)
        ax2.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
        
        # Add statistics
        mean_mdd = np.mean(max_drawdowns)
        median_mdd = np.median(max_drawdowns)
        worst_mdd = np.max(max_drawdowns)
        
        ax2.axvline(x=mean_mdd, color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {mean_mdd:.1f}%')
        ax2.axvline(x=median_mdd, color='orange', linestyle='--',
                   linewidth=2, label=f'Median: {median_mdd:.1f}%')
        ax2.axvline(x=worst_mdd, color='darkred', linestyle=':',
                   linewidth=2, label=f'Worst: {worst_mdd:.1f}%')
        
        # Add VaR for drawdown
        if 'drawdown_95' in self.final_results.risk_metrics:
            dd_var = self.final_results.risk_metrics['drawdown_95']
            ax2.axvline(x=dd_var, color='purple', linestyle='--',
                       linewidth=2, label=f'95% VaR: {dd_var:.1f}%')
        
        ax2.set_xlabel('Maximum Drawdown (%)')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Maximum Drawdowns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of simulation results.
        
        Returns:
            Dict[str, Any]: Summary dictionary
        """
        if not self.final_results:
            raise ValueError("No results available. Run simulation first.")
        
        agg_stats = self.final_results.aggregate_stats
        risk_metrics = self.final_results.risk_metrics
        classification = self.final_results.performance_classification
        
        summary = {
            'strategy': self.strategy_name,
            'simulations': len(self.simulation_results),
            'time_horizon': self.config.time_horizon,
            'method': self.config.method.value,
            'overall_performance': classification['overall'],
            'key_metrics': {
                'mean_final_value': agg_stats['mean_final_value'],
                'mean_total_return': agg_stats['mean_total_return'],
                'mean_sharpe': agg_stats['mean_sharpe'],
                'success_rate': agg_stats['success_rate'] * 100,
                'ruin_rate': agg_stats['ruin_rate'] * 100
            },
            'risk_metrics': {
                'var_95': risk_metrics.get('var', 0),
                'cvar_95': risk_metrics.get('cvar', 0),
                'mean_max_drawdown': agg_stats['mean_max_drawdown']
            },
            'recommendation': self._generate_recommendation()
        }
        
        return summary
    
    def _generate_recommendation(self) -> Dict[str, Any]:
        """
        Generate trading recommendation based on simulation.
        
        Returns:
            Dict[str, Any]: Recommendation
        """
        if not self.final_results:
            return {'status': 'No simulation performed'}
        
        classification = self.final_results.performance_classification
        agg_stats = self.final_results.aggregate_stats
        
        overall = classification['overall']
        success_rate = agg_stats['success_rate']
        ruin_rate = agg_stats['ruin_rate']
        mean_sharpe = agg_stats['mean_sharpe']
        
        # Decision matrix
        if overall in ['Excellent', 'Good']:
            if success_rate >= 0.6 and ruin_rate <= 0.05:
                status = 'STRONG BUY'
                confidence = 'High'
                reasoning = 'Excellent performance with low ruin probability'
            elif success_rate >= 0.55 and ruin_rate <= 0.10:
                status = 'BUY'
                confidence = 'Medium'
                reasoning = 'Good performance with acceptable risk'
            else:
                status = 'CAUTIOUS BUY'
                confidence = 'Low'
                reasoning = 'Good returns but elevated risk'
        
        elif overall == 'Average':
            if ruin_rate <= 0.10:
                status = 'HOLD'
                confidence = 'Medium'
                reasoning = 'Average performance with manageable risk'
            else:
                status = 'AVOID'
                confidence = 'Low'
                reasoning = 'Average performance with high risk'
        
        else:  # Poor or Very Poor
            status = 'AVOID'
            confidence = 'High'
            reasoning = 'Poor performance metrics'
        
        recommendation = {
            'status': status,
            'confidence': confidence,
            'reasoning': reasoning,
            'risk_tolerance': self._get_risk_tolerance(ruin_rate, mean_sharpe),
            'suggested_position_size': self._calculate_position_size(ruin_rate, mean_sharpe)
        }
        
        return recommendation
    
    def _get_risk_tolerance(self, ruin_rate: float, sharpe: float) -> str:
        """Get risk tolerance based on simulation results."""
        if ruin_rate <= 0.01 and sharpe >= 1.0:
            return 'Aggressive'
        elif ruin_rate <= 0.05 and sharpe >= 0.5:
            return 'Moderate'
        elif ruin_rate <= 0.10:
            return 'Conservative'
        else:
            return 'Very Conservative'
    
    def _calculate_position_size(self, ruin_rate: float, sharpe: float) -> float:
        """Calculate suggested position size based on risk."""
        # Kelly Criterion inspired position sizing
        if sharpe > 0 and ruin_rate < 0.5:
            # Simplified position sizing formula
            position_size = min(0.5, sharpe * 0.1 / (1 + ruin_rate * 10))
            return round(position_size * 100, 1)  # Return as percentage
        return 10.0  # Default 10%
    
    def sensitivity_analysis(self,
                           parameter_name: str,
                           parameter_values: List[Any],
                           n_simulations: int = 100) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on a parameter.
        
        Args:
            parameter_name: Name of parameter to analyze
            parameter_values: List of parameter values to test
            n_simulations: Number of simulations per parameter value
        
        Returns:
            Dict[str, Any]: Sensitivity analysis results
        """
        results = []
        
        original_config = copy.deepcopy(self.config)
        
        for value in parameter_values:
            self.logger.info(f"Testing {parameter_name} = {value}")
            
            # Update parameter
            if hasattr(self.config, parameter_name):
                setattr(self.config, parameter_name, value)
            elif parameter_name in self.config.strategy_params:
                self.config.strategy_params[parameter_name] = value
            
            # Run reduced simulation
            self.config.simulations = n_simulations
            self.simulation_results = []  # Clear previous results
            
            try:
                sim_results = self.run_simulation()
                
                results.append({
                    'parameter_value': value,
                    'mean_return': sim_results.aggregate_stats['mean_total_return'],
                    'mean_sharpe': sim_results.aggregate_stats['mean_sharpe'],
                    'success_rate': sim_results.aggregate_stats['success_rate'],
                    'ruin_rate': sim_results.aggregate_stats['ruin_rate']
                })
            except Exception as e:
                self.logger.error(f"Error testing {parameter_name}={value}: {e}")
                results.append({
                    'parameter_value': value,
                    'error': str(e)
                })
        
        # Restore original configuration
        self.config = original_config
        
        # Find optimal parameter value
        valid_results = [r for r in results if 'mean_sharpe' in r]
        if valid_results:
            optimal_result = max(valid_results, key=lambda x: x['mean_sharpe'])
            
            return {
                'parameter_name': parameter_name,
                'results': results,
                'optimal_value': optimal_result['parameter_value'],
                'optimal_metrics': {
                    'mean_return': optimal_result['mean_return'],
                    'mean_sharpe': optimal_result['mean_sharpe']
                }
            }
        
        return {'parameter_name': parameter_name, 'results': results}

# Example usage
if __name__ == "__main__":
    print("Testing Monte Carlo Simulator...")
    
    # Create sample returns data
    np.random.seed(42)
    n_periods = 1000
    # Generate returns with some drift and volatility
    drift = 0.0005  # 0.05% daily
    volatility = 0.02  # 2% daily
    
    returns = np.random.normal(drift, volatility, n_periods)
    prices = 50000 * np.cumprod(1 + returns)
    
    # Create Monte Carlo configuration
    mc_config = MonteCarloConfig(
        method=MonteCarloMethod.HISTORICAL_BOOTSTRAP,
        simulations=500,
        time_horizon=252,  # 1 year
        initial_capital=10000,
        confidence_level=0.95,
        parallel=True,
        max_workers=2,
        verbose=True
    )
    
    # Create simulator
    simulator = MonteCarloSimulator(mc_config)
    simulator.load_returns(pd.Series(returns), pd.Series(prices))
    
    # Run simulation
    print("\nStarting Monte Carlo simulation...")
    results = simulator.run_simulation()
    
    # Get summary
    summary = simulator.get_summary()
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION SUMMARY")
    print("="*60)
    print(f"Simulations: {summary['simulations']}")
    print(f"Time Horizon: {summary['time_horizon']} periods")
    print(f"Method: {summary['method']}")
    print(f"\nOverall Performance: {summary['overall_performance']}")
    print(f"\nKey Metrics:")
    print(f"  Mean Final Value: ${summary['key_metrics']['mean_final_value']:,.2f}")
    print(f"  Mean Total Return: {summary['key_metrics']['mean_total_return']:.2f}%")
    print(f"  Mean Sharpe Ratio: {summary['key_metrics']['mean_sharpe']:.2f}")
    print(f"  Success Rate: {summary['key_metrics']['success_rate']:.1f}%")
    print(f"  Ruin Rate: {summary['key_metrics']['ruin_rate']:.1f}%")
    print(f"\nRisk Metrics:")
    print(f"  VaR (95%): {summary['risk_metrics']['var_95']:.2f}%")
    print(f"  CVaR (95%): {summary['risk_metrics']['cvar_95']:.2f}%")
    print(f"  Mean Max Drawdown: {summary['risk_metrics']['mean_max_drawdown']:.2f}%")
    print(f"\nRecommendation: {summary['recommendation']['status']}")
    print(f"Confidence: {summary['recommendation']['confidence']}")
    print(f"Risk Tolerance: {summary['recommendation']['risk_tolerance']}")
    print(f"Suggested Position Size: {summary['recommendation']['suggested_position_size']}%")
    print("="*60)
    
    # Run sensitivity analysis example
    print("\nRunning sensitivity analysis on volatility...")
    sensitivity_results = simulator.sensitivity_analysis(
        parameter_name='volatility',
        parameter_values=[0.01, 0.02, 0.03, 0.04],
        n_simulations=100
    )
    
    if 'optimal_value' in sensitivity_results:
        print(f"\nOptimal volatility: {sensitivity_results['optimal_value']}")
        print(f"Optimal Sharpe: {sensitivity_results['optimal_metrics']['mean_sharpe']:.2f}")
    
    print("\nMonte Carlo simulation completed successfully!")