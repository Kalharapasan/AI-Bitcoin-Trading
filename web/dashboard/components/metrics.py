"""
Comprehensive performance metrics module for trading strategy evaluation.
Provides quantitative analysis of trading performance with statistical rigor.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import warnings
from scipy import stats
from scipy.stats import norm, skew, kurtosis, t
import math
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')

# Import project modules
from logger import get_logger

logger = get_logger(__name__)

class MetricCategory(Enum):
    """Categories of performance metrics."""
    RETURN = "return"
    RISK = "risk"
    RISK_ADJUSTED = "risk_adjusted"
    TRADE = "trade"
    DRAWDOWN = "drawdown"
    EFFICIENCY = "efficiency"
    CUSTOM = "custom"

class RiskFreeRateType(Enum):
    """Types of risk-free rate assumptions."""
    ZERO = "zero"
    TREASURY_BILL = "treasury_bill"
    LIBOR = "libor"
    CUSTOM = "custom"

@dataclass
class MetricsConfig:
    """Configuration for metrics calculation."""
    risk_free_rate: float = 0.02  # Annual risk-free rate (2%)
    risk_free_type: RiskFreeRateType = RiskFreeRateType.TREASURY_BILL
    annualization_factor: int = 252  # Trading days per year
    confidence_level: float = 0.95  # For VaR, CVaR calculations
    benchmark_returns: Optional[pd.Series] = None
    benchmark_name: str = "Benchmark"
    compound_returns: bool = True
    include_trade_metrics: bool = True
    include_drawdown_metrics: bool = True
    include_efficiency_metrics: bool = True
    custom_metrics: Dict[str, Callable] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container."""
    # Basic metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    cumulative_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    annual_volatility: float = 0.0
    downside_volatility: float = 0.0
    value_at_risk: float = 0.0
    conditional_var: float = 0.0
    tail_ratio: float = 0.0
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0
    jensens_alpha: float = 0.0
    appraisal_ratio: float = 0.0
    
    # Drawdown metrics
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0
    pain_index: float = 0.0
    martin_ratio: float = 0.0
    
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    k_ratio: float = 0.0
    payoff_ratio: float = 0.0
    profit_loss_ratio: float = 0.0
    
    # Efficiency metrics
    sharpe_efficiency: float = 0.0
    trading_efficiency: float = 0.0
    diversification_ratio: float = 0.0
    capacity_ratio: float = 0.0
    
    # Statistical metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    jarque_bera_stat: float = 0.0
    jarque_bera_pvalue: float = 0.0
    
    # Benchmark comparison
    alpha: float = 0.0
    beta: float = 0.0
    r_squared: float = 0.0
    tracking_error: float = 0.0
    active_return: float = 0.0
    
    # Time-based metrics
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_days: int = 0
    market_exposure: float = 0.0
    
    # Composite scores
    composite_score: float = 0.0
    risk_adjusted_score: float = 0.0
    consistency_score: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame."""
        data = self.to_dict()
        # Flatten nested dictionaries
        flat_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_data[f"{key}_{subkey}"] = subvalue
            else:
                flat_data[key] = value
        
        return pd.DataFrame([flat_data])
    
    def get_summary(self, categories: Optional[List[MetricCategory]] = None) -> Dict[str, float]:
        """
        Get summary of key metrics.
        
        Args:
            categories: List of metric categories to include
        
        Returns:
            Dict[str, float]: Summary metrics
        """
        if categories is None:
            categories = [MetricCategory.RETURN, MetricCategory.RISK, 
                         MetricCategory.RISK_ADJUSTED, MetricCategory.DRAWDOWN]
        
        summary = {}
        
        if MetricCategory.RETURN in categories:
            summary.update({
                'total_return': self.total_return,
                'annual_return': self.annual_return,
                'cumulative_return': self.cumulative_return
            })
        
        if MetricCategory.RISK in categories:
            summary.update({
                'volatility': self.volatility,
                'max_drawdown': self.max_drawdown,
                'value_at_risk': self.value_at_risk
            })
        
        if MetricCategory.RISK_ADJUSTED in categories:
            summary.update({
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio
            })
        
        if MetricCategory.TRADE in categories:
            summary.update({
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'expectancy': self.expectancy
            })
        
        if MetricCategory.DRAWDOWN in categories:
            summary.update({
                'max_drawdown': self.max_drawdown,
                'avg_drawdown': self.avg_drawdown,
                'ulcer_index': self.ulcer_index
            })
        
        if MetricCategory.EFFICIENCY in categories:
            summary.update({
                'trading_efficiency': self.trading_efficiency,
                'sharpe_efficiency': self.sharpe_efficiency
            })
        
        return summary

class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for trading strategies.
    Calculates a wide range of performance and risk metrics.
    """
    
    def __init__(self, config: MetricsConfig = None):
        """
        Initialize performance analyzer.
        
        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        self.metrics = PerformanceMetrics()
        self.returns = None
        self.equity_curve = None
        self.trades = None
        self.benchmark_returns = self.config.benchmark_returns
        
        logger.info("Initialized PerformanceAnalyzer")
    
    def load_returns(self, returns: pd.Series):
        """
        Load returns data for analysis.
        
        Args:
            returns: Series of returns with datetime index
        """
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Returns must have DatetimeIndex")
        
        self.returns = returns.copy()
        logger.info(f"Loaded returns: {len(self.returns)} periods from {self.returns.index[0]} to {self.returns.index[-1]}")
    
    def load_equity_curve(self, equity_curve: pd.Series):
        """
        Load equity curve data.
        
        Args:
            equity_curve: Series of equity values with datetime index
        """
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            raise ValueError("Equity curve must have DatetimeIndex")
        
        self.equity_curve = equity_curve.copy()
        logger.info(f"Loaded equity curve: {len(self.equity_curve)} points")
    
    def load_trades(self, trades: pd.DataFrame):
        """
        Load trade data for analysis.
        
        Args:
            trades: DataFrame with trade information
        """
        required_cols = ['pnl']
        for col in required_cols:
            if col not in trades.columns:
                raise ValueError(f"Trades must have '{col}' column")
        
        self.trades = trades.copy()
        logger.info(f"Loaded trades: {len(self.trades)} trades")
    
    def calculate_all_metrics(self) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        
        Returns:
            PerformanceMetrics: Comprehensive metrics object
        """
        if self.returns is None:
            raise ValueError("No returns data loaded")
        
        logger.info("Calculating all performance metrics...")
        
        # Calculate basic metrics
        self._calculate_basic_metrics()
        
        # Calculate risk metrics
        self._calculate_risk_metrics()
        
        # Calculate risk-adjusted metrics
        self._calculate_risk_adjusted_metrics()
        
        # Calculate drawdown metrics
        if self.equity_curve is not None and self.config.include_drawdown_metrics:
            self._calculate_drawdown_metrics()
        
        # Calculate trade metrics
        if self.trades is not None and self.config.include_trade_metrics:
            self._calculate_trade_metrics()
        
        # Calculate efficiency metrics
        if self.config.include_efficiency_metrics:
            self._calculate_efficiency_metrics()
        
        # Calculate statistical metrics
        self._calculate_statistical_metrics()
        
        # Calculate benchmark metrics
        if self.benchmark_returns is not None:
            self._calculate_benchmark_metrics()
        
        # Calculate composite scores
        self._calculate_composite_scores()
        
        # Calculate custom metrics
        self._calculate_custom_metrics()
        
        logger.info("All metrics calculated successfully")
        return self.metrics
    
    def _calculate_basic_metrics(self):
        """Calculate basic return metrics."""
        if self.returns is None:
            return
        
        returns = self.returns.dropna()
        n_periods = len(returns)
        
        # Total return (cumulative)
        if self.config.compound_returns:
            cumulative_return = np.prod(1 + returns) - 1
        else:
            cumulative_return = returns.sum()
        
        # Annual return
        if n_periods > 1:
            total_days = (returns.index[-1] - returns.index[0]).days
            years = total_days / 365.25
            if years > 0:
                annual_return = (1 + cumulative_return) ** (1 / years) - 1
            else:
                annual_return = cumulative_return
        else:
            annual_return = cumulative_return
        
        self.metrics.total_return = cumulative_return
        self.metrics.annual_return = annual_return
        self.metrics.cumulative_return = cumulative_return
        self.metrics.start_date = returns.index[0]
        self.metrics.end_date = returns.index[-1]
        self.metrics.total_days = (returns.index[-1] - returns.index[0]).days
    
    def _calculate_risk_metrics(self):
        """Calculate risk metrics."""
        if self.returns is None:
            return
        
        returns = self.returns.dropna()
        n_periods = len(returns)
        
        # Volatility (standard deviation)
        volatility = returns.std()
        annual_volatility = volatility * np.sqrt(self.config.annualization_factor)
        
        # Downside volatility (semi-deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_volatility = downside_returns.std()
        else:
            downside_volatility = 0.0
        
        # Value at Risk (VaR)
        confidence_level = self.config.confidence_level
        var_historical = -np.percentile(returns, (1 - confidence_level) * 100)
        
        # Parametric VaR (assuming normal distribution)
        var_parametric = -norm.ppf(1 - confidence_level, returns.mean(), returns.std())
        
        # Use historical VaR as default
        value_at_risk = var_historical
        
        # Conditional VaR (Expected Shortfall)
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) > 0:
            conditional_var = -tail_returns.mean()
        else:
            conditional_var = -var_threshold
        
        # Tail ratio (95th percentile / 5th percentile)
        tail_95 = np.percentile(np.abs(returns), 95)
        tail_5 = np.percentile(np.abs(returns), 5)
        tail_ratio = tail_95 / tail_5 if tail_5 > 0 else float('inf')
        
        self.metrics.volatility = volatility
        self.metrics.annual_volatility = annual_volatility
        self.metrics.downside_volatility = downside_volatility
        self.metrics.value_at_risk = value_at_risk
        self.metrics.conditional_var = conditional_var
        self.metrics.tail_ratio = tail_ratio
    
    def _calculate_risk_adjusted_metrics(self):
        """Calculate risk-adjusted return metrics."""
        if self.returns is None:
            return
        
        returns = self.returns.dropna()
        n_periods = len(returns)
        
        # Risk-free rate adjustments
        rf_rate_daily = self.config.risk_free_rate / self.config.annualization_factor
        
        # Excess returns
        excess_returns = returns - rf_rate_daily
        
        # Sharpe Ratio
        if returns.std() > 0:
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(self.config.annualization_factor)
        else:
            sharpe_ratio = 0.0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(self.config.annualization_factor)
        else:
            sortino_ratio = float('inf') if excess_returns.mean() > 0 else 0.0
        
        # Calmar Ratio
        if self.metrics.max_drawdown != 0:
            calmar_ratio = self.metrics.annual_return / abs(self.metrics.max_drawdown)
        else:
            calmar_ratio = float('inf') if self.metrics.annual_return > 0 else 0.0
        
        # Omega Ratio
        threshold = rf_rate_daily
        upside_returns = returns[returns > threshold] - threshold
        downside_returns = threshold - returns[returns <= threshold]
        
        if len(downside_returns) > 0 and downside_returns.sum() > 0:
            omega_ratio = upside_returns.sum() / downside_returns.sum()
        else:
            omega_ratio = float('inf') if len(upside_returns) > 0 else 0.0
        
        # Treynor Ratio (requires beta)
        if self.metrics.beta != 0:
            treynor_ratio = excess_returns.mean() * self.config.annualization_factor / self.metrics.beta
        else:
            treynor_ratio = 0.0
        
        # Information Ratio (requires benchmark)
        if self.benchmark_returns is not None:
            aligned_benchmark = self.benchmark_returns.reindex(returns.index).fillna(0)
            active_returns = returns - aligned_benchmark
            tracking_error = active_returns.std() * np.sqrt(self.config.annualization_factor)
            
            if tracking_error > 0:
                information_ratio = active_returns.mean() * self.config.annualization_factor / tracking_error
            else:
                information_ratio = 0.0
        else:
            information_ratio = 0.0
        
        # Jensen's Alpha
        if self.metrics.beta != 0:
            expected_return = rf_rate_daily + self.metrics.beta * (aligned_benchmark.mean() - rf_rate_daily)
            jensens_alpha = returns.mean() - expected_return
            jensens_alpha_annual = jensens_alpha * self.config.annualization_factor
        else:
            jensens_alpha_annual = 0.0
        
        # Appraisal Ratio (Treynor-Black)
        if self.metrics.tracking_error > 0:
            appraisal_ratio = jensens_alpha_annual / self.metrics.tracking_error
        else:
            appraisal_ratio = 0.0
        
        self.metrics.sharpe_ratio = sharpe_ratio
        self.metrics.sortino_ratio = sortino_ratio
        self.metrics.calmar_ratio = calmar_ratio
        self.metrics.omega_ratio = omega_ratio
        self.metrics.treynor_ratio = treynor_ratio
        self.metrics.information_ratio = information_ratio
        self.metrics.jensens_alpha = jensens_alpha_annual
        self.metrics.appraisal_ratio = appraisal_ratio
    
    def _calculate_drawdown_metrics(self):
        """Calculate drawdown-related metrics."""
        if self.equity_curve is None:
            return
        
        equity = self.equity_curve.dropna()
        
        # Calculate drawdown series
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        drawdown_pct = drawdown * 100
        
        # Maximum drawdown
        max_drawdown = drawdown_pct.min()
        
        # Average drawdown
        avg_drawdown = drawdown_pct[drawdown_pct < 0].mean() if len(drawdown_pct[drawdown_pct < 0]) > 0 else 0.0
        
        # Maximum drawdown duration
        drawdown_durations = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown_pct.items():
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_date = date
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                duration = (date - start_date).days
                drawdown_durations.append(duration)
        
        if in_drawdown and start_date:
            duration = (equity.index[-1] - start_date).days
            drawdown_durations.append(duration)
        
        max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0
        
        # Recovery factor
        if max_drawdown < 0:
            recovery_factor = abs(self.metrics.total_return / max_drawdown * 100)
        else:
            recovery_factor = float('inf')
        
        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown_pct[drawdown_pct < 0] ** 2)) if len(drawdown_pct[drawdown_pct < 0]) > 0 else 0.0
        
        # Pain Index (average drawdown)
        pain_index = abs(avg_drawdown)
        
        # Martin Ratio (Ulcer Performance Index)
        if ulcer_index > 0:
            martin_ratio = self.metrics.annual_return / ulcer_index
        else:
            martin_ratio = float('inf') if self.metrics.annual_return > 0 else 0.0
        
        self.metrics.max_drawdown = max_drawdown
        self.metrics.avg_drawdown = avg_drawdown
        self.metrics.max_drawdown_duration = max_drawdown_duration
        self.metrics.recovery_factor = recovery_factor
        self.metrics.ulcer_index = ulcer_index
        self.metrics.pain_index = pain_index
        self.metrics.martin_ratio = martin_ratio
    
    def _calculate_trade_metrics(self):
        """Calculate trade-based metrics."""
        if self.trades is None:
            return
        
        trades = self.trades.copy()
        
        # Basic trade counts
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] <= 0])
        
        # Win/loss rates
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        loss_rate = losing_trades / total_trades if total_trades > 0 else 0.0
        
        # Average win/loss
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0.0
        avg_loss = trades[trades['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0.0
        
        # Largest win/loss
        largest_win = trades['pnl'].max() if total_trades > 0 else 0.0
        largest_loss = trades['pnl'].min() if total_trades > 0 else 0.0
        
        # Profit factor
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] <= 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
        
        # K-Ratio (Persistence of returns)
        if len(trades) >= 10 and 'exit_time' in trades.columns:
            # Sort trades by exit time
            trades_sorted = trades.sort_values('exit_time')
            cumulative_pnl = trades_sorted['pnl'].cumsum()
            
            # Fit linear regression to cumulative P&L
            x = np.arange(len(cumulative_pnl))
            y = cumulative_pnl.values
            
            if len(y) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                residuals = y - (slope * x + intercept)
                std_error = np.std(residuals) / np.sqrt(len(y))
                
                if std_error > 0:
                    k_ratio = slope / std_error
                else:
                    k_ratio = 0.0
            else:
                k_ratio = 0.0
        else:
            k_ratio = 0.0
        
        # Payoff ratio (average win / average loss)
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Profit/Loss ratio
        profit_loss_ratio = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        self.metrics.total_trades = total_trades
        self.metrics.winning_trades = winning_trades
        self.metrics.losing_trades = losing_trades
        self.metrics.win_rate = win_rate
        self.metrics.loss_rate = loss_rate
        self.metrics.avg_win = avg_win
        self.metrics.avg_loss = avg_loss
        self.metrics.largest_win = largest_win
        self.metrics.largest_loss = largest_loss
        self.metrics.profit_factor = profit_factor
        self.metrics.expectancy = expectancy
        self.metrics.k_ratio = k_ratio
        self.metrics.payoff_ratio = payoff_ratio
        self.metrics.profit_loss_ratio = profit_loss_ratio
    
    def _calculate_efficiency_metrics(self):
        """Calculate efficiency metrics."""
        if self.returns is None:
            return
        
        returns = self.returns.dropna()
        
        # Sharpe Efficiency (Actual Sharpe / Maximum Possible Sharpe)
        # Maximum possible Sharpe is sqrt(annualization_factor) for perfect strategy
        max_possible_sharpe = np.sqrt(self.config.annualization_factor)
        sharpe_efficiency = self.metrics.sharpe_ratio / max_possible_sharpe if max_possible_sharpe > 0 else 0.0
        
        # Trading Efficiency (Net Profit / Gross Profit)
        if self.trades is not None:
            gross_profit = self.trades[self.trades['pnl'] > 0]['pnl'].sum()
            net_profit = self.trades['pnl'].sum()
            
            if gross_profit > 0:
                trading_efficiency = net_profit / gross_profit
            else:
                trading_efficiency = 0.0
        else:
            trading_efficiency = 0.0
        
        # Diversification Ratio (if multiple assets)
        # Simplified implementation - would need correlation matrix for full version
        diversification_ratio = 1.0  # Default, implement based on portfolio
        
        # Capacity Ratio (Liquidity efficiency)
        # Simplified - would need trade size and market impact data
        capacity_ratio = 1.0  # Default
        
        self.metrics.sharpe_efficiency = sharpe_efficiency
        self.metrics.trading_efficiency = trading_efficiency
        self.metrics.diversification_ratio = diversification_ratio
        self.metrics.capacity_ratio = capacity_ratio
    
    def _calculate_statistical_metrics(self):
        """Calculate statistical metrics."""
        if self.returns is None:
            return
        
        returns = self.returns.dropna()
        
        # Skewness
        skewness = returns.skew()
        
        # Kurtosis
        kurt = returns.kurtosis()
        
        # Jarque-Bera test for normality
        n = len(returns)
        if n > 0:
            jb_stat = (n / 6) * (skewness**2 + (kurt**2) / 4)
            jb_pvalue = 1 - stats.chi2.cdf(jb_stat, 2)
        else:
            jb_stat = 0.0
            jb_pvalue = 1.0
        
        self.metrics.skewness = skewness
        self.metrics.kurtosis = kurt
        self.metrics.jarque_bera_stat = jb_stat
        self.metrics.jarque_bera_pvalue = jb_pvalue
    
    def _calculate_benchmark_metrics(self):
        """Calculate benchmark comparison metrics."""
        if self.returns is None or self.benchmark_returns is None:
            return
        
        returns = self.returns.dropna()
        benchmark = self.benchmark_returns.reindex(returns.index).dropna()
        
        if len(benchmark) < 2:
            return
        
        # Align returns and benchmark
        aligned_returns = returns.reindex(benchmark.index)
        
        # Calculate alpha and beta (CAPM)
        excess_returns = aligned_returns - self.config.risk_free_rate / self.config.annualization_factor
        excess_benchmark = benchmark - self.config.risk_free_rate / self.config.annualization_factor
        
        # Linear regression
        if len(excess_returns) > 1 and excess_benchmark.std() > 0:
            # Add constant for intercept
            X = sm.add_constant(excess_benchmark) if 'sm' in globals() else np.column_stack([np.ones(len(excess_benchmark)), excess_benchmark])
            y = excess_returns.values
            
            try:
                # Use statsmodels if available
                import statsmodels.api as sm
                model = sm.OLS(y, X).fit()
                alpha = model.params[0] * self.config.annualization_factor
                beta = model.params[1]
                r_squared = model.rsquared
            except:
                # Fallback to numpy
                cov = np.cov(excess_returns, excess_benchmark)[0, 1]
                var = np.var(excess_benchmark)
                beta = cov / var if var > 0 else 0.0
                alpha = excess_returns.mean() - beta * excess_benchmark.mean()
                alpha *= self.config.annualization_factor
                
                # Calculate R-squared
                y_pred = alpha/self.config.annualization_factor + beta * excess_benchmark
                ss_res = np.sum((excess_returns - y_pred) ** 2)
                ss_tot = np.sum((excess_returns - excess_returns.mean()) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            alpha = 0.0
            beta = 0.0
            r_squared = 0.0
        
        # Tracking error
        active_returns = aligned_returns - benchmark
        tracking_error = active_returns.std() * np.sqrt(self.config.annualization_factor)
        
        # Active return
        active_return = (aligned_returns.mean() - benchmark.mean()) * self.config.annualization_factor
        
        self.metrics.alpha = alpha
        self.metrics.beta = beta
        self.metrics.r_squared = r_squared
        self.metrics.tracking_error = tracking_error
        self.metrics.active_return = active_return
    
    def _calculate_composite_scores(self):
        """Calculate composite performance scores."""
        # Composite Score (weighted average of key metrics)
        weights = {
            'sharpe_ratio': 0.25,
            'sortino_ratio': 0.15,
            'calmar_ratio': 0.15,
            'win_rate': 0.15,
            'profit_factor': 0.15,
            'max_drawdown': 0.10,  # Negative weight
            'volatility': 0.05  # Negative weight
        }
        
        # Normalize metrics to 0-1 scale
        normalized = {}
        
        # Sharpe ratio (0-3 scale)
        normalized['sharpe_ratio'] = max(0, min(1, self.metrics.sharpe_ratio / 3))
        
        # Sortino ratio (0-5 scale)
        normalized['sortino_ratio'] = max(0, min(1, self.metrics.sortino_ratio / 5))
        
        # Calmar ratio (0-3 scale)
        normalized['calmar_ratio'] = max(0, min(1, self.metrics.calmar_ratio / 3))
        
        # Win rate (0-100% scale)
        normalized['win_rate'] = self.metrics.win_rate
        
        # Profit factor (1-5 scale)
        normalized['profit_factor'] = max(0, min(1, (self.metrics.profit_factor - 1) / 4))
        
        # Max drawdown (inverted, -50% to 0% scale)
        normalized['max_drawdown'] = max(0, min(1, 1 + self.metrics.max_drawdown / 50))
        
        # Volatility (inverted, 0-50% scale)
        normalized['volatility'] = max(0, min(1, 1 - self.metrics.volatility / 0.5))
        
        # Calculate weighted sum
        composite_score = sum(weights[key] * normalized[key] for key in weights)
        
        # Risk-adjusted score (focus on risk metrics)
        risk_weights = {
            'sharpe_ratio': 0.30,
            'sortino_ratio': 0.25,
            'calmar_ratio': 0.25,
            'max_drawdown': 0.10,
            'volatility': 0.10
        }
        
        risk_adjusted_score = sum(risk_weights[key] * normalized[key] for key in risk_weights)
        
        # Consistency score (based on win rate and k-ratio)
        win_rate_score = self.metrics.win_rate
        k_ratio_score = max(0, min(1, self.metrics.k_ratio / 2))
        consistency_score = 0.6 * win_rate_score + 0.4 * k_ratio_score
        
        self.metrics.composite_score = composite_score
        self.metrics.risk_adjusted_score = risk_adjusted_score
        self.metrics.consistency_score = consistency_score
    
    def _calculate_custom_metrics(self):
        """Calculate custom metrics defined in configuration."""
        if not self.config.custom_metrics:
            return
        
        custom_results = {}
        for name, func in self.config.custom_metrics.items():
            try:
                result = func(self.returns, self.equity_curve, self.trades)
                custom_results[name] = result
            except Exception as e:
                logger.error(f"Error calculating custom metric '{name}': {e}")
                custom_results[name] = None
        
        self.metrics.custom_metrics = custom_results
    
    def get_metrics_by_category(self, category: MetricCategory) -> Dict[str, float]:
        """
        Get metrics filtered by category.
        
        Args:
            category: Metric category
        
        Returns:
            Dict[str, float]: Metrics in specified category
        """
        metrics_dict = self.metrics.to_dict()
        
        # Define category mappings
        category_mappings = {
            MetricCategory.RETURN: ['total_return', 'annual_return', 'cumulative_return'],
            MetricCategory.RISK: ['volatility', 'annual_volatility', 'downside_volatility', 
                                'value_at_risk', 'conditional_var', 'tail_ratio'],
            MetricCategory.RISK_ADJUSTED: ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                                         'omega_ratio', 'treynor_ratio', 'information_ratio',
                                         'jensens_alpha', 'appraisal_ratio'],
            MetricCategory.DRAWDOWN: ['max_drawdown', 'avg_drawdown', 'max_drawdown_duration',
                                    'recovery_factor', 'ulcer_index', 'pain_index', 'martin_ratio'],
            MetricCategory.TRADE: ['total_trades', 'winning_trades', 'losing_trades', 'win_rate',
                                 'loss_rate', 'avg_win', 'avg_loss', 'largest_win', 'largest_loss',
                                 'profit_factor', 'expectancy', 'k_ratio', 'payoff_ratio', 'profit_loss_ratio'],
            MetricCategory.EFFICIENCY: ['sharpe_efficiency', 'trading_efficiency',
                                      'diversification_ratio', 'capacity_ratio'],
            MetricCategory.CUSTOM: list(self.metrics.custom_metrics.keys())
        }
        
        filtered_metrics = {}
        for metric_name in category_mappings.get(category, []):
            if metric_name in metrics_dict:
                filtered_metrics[metric_name] = metrics_dict[metric_name]
        
        return filtered_metrics
    
    def compare_with_benchmark(self) -> pd.DataFrame:
        """
        Compare strategy metrics with benchmark.
        
        Returns:
            pd.DataFrame: Comparison table
        """
        if self.benchmark_returns is None:
            raise ValueError("No benchmark returns provided")
        
        # Calculate benchmark metrics
        benchmark_analyzer = PerformanceAnalyzer(
            MetricsConfig(
                risk_free_rate=self.config.risk_free_rate,
                annualization_factor=self.config.annualization_factor
            )
        )
        benchmark_analyzer.load_returns(self.benchmark_returns)
        benchmark_metrics = benchmark_analyzer.calculate_all_metrics()
        
        # Create comparison table
        comparison_data = []
        
        # Key metrics to compare
        key_metrics = [
            'total_return', 'annual_return', 'volatility', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'win_rate', 'profit_factor'
        ]
        
        for metric in key_metrics:
            strategy_value = getattr(self.metrics, metric, 0)
            benchmark_value = getattr(benchmark_metrics, metric, 0)
            
            # Calculate difference
            if metric in ['total_return', 'annual_return']:
                difference = strategy_value - benchmark_value
                difference_pct = (difference / abs(benchmark_value)) * 100 if benchmark_value != 0 else 0
            else:
                difference = strategy_value - benchmark_value
                difference_pct = 0
            
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Strategy': strategy_value,
                'Benchmark': benchmark_value,
                'Difference': difference,
                'Difference %': difference_pct
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_performance_grade(self) -> Dict[str, Any]:
        """
        Get performance grade based on metrics.
        
        Returns:
            Dict[str, Any]: Performance grade with scores and classification
        """
        grades = {
            'A+': {'min_score': 0.9, 'color': '#00FF00', 'description': 'Excellent'},
            'A': {'min_score': 0.8, 'color': '#7CFC00', 'description': 'Very Good'},
            'B': {'min_score': 0.7, 'color': '#FFFF00', 'description': 'Good'},
            'C': {'min_score': 0.6, 'color': '#FFA500', 'description': 'Average'},
            'D': {'min_score': 0.5, 'color': '#FF4500', 'description': 'Below Average'},
            'F': {'min_score': 0.0, 'color': '#FF0000', 'description': 'Poor'}
        }
        
        # Calculate overall score
        overall_score = self.metrics.composite_score
        
        # Determine grade
        grade = 'F'
        for g, criteria in grades.items():
            if overall_score >= criteria['min_score']:
                grade = g
                break
        
        # Calculate category scores
        category_scores = {}
        
        # Return score
        return_score = min(1.0, max(0.0, (self.metrics.annual_return + 0.2) / 0.4))  # Map -20% to +20% to 0-1
        
        # Risk score (inverted)
        risk_score = min(1.0, max(0.0, 1 - self.metrics.volatility / 0.3))  # Map 0-30% vol to 1-0
        
        # Risk-adjusted score
        risk_adj_score = min(1.0, max(0.0, self.metrics.sharpe_ratio / 2))  # Map 0-2 Sharpe to 0-1
        
        # Drawdown score (inverted)
        drawdown_score = min(1.0, max(0.0, 1 + self.metrics.max_drawdown / 30))  # Map -30% to 0% to 0-1
        
        # Consistency score
        consistency_score = self.metrics.consistency_score
        
        category_scores = {
            'returns': return_score,
            'risk': risk_score,
            'risk_adjusted': risk_adj_score,
            'drawdown': drawdown_score,
            'consistency': consistency_score
        }
        
        return {
            'overall_grade': grade,
            'overall_score': overall_score,
            'grade_info': grades[grade],
            'category_scores': category_scores,
            'recommendation': self._get_recommendation(grade, category_scores)
        }
    
    def _get_recommendation(self, grade: str, category_scores: Dict[str, float]) -> str:
        """Get trading recommendation based on performance grade."""
        recommendations = {
            'A+': 'STRONG BUY: Excellent performance across all metrics. Consider aggressive position sizing.',
            'A': 'BUY: Very good performance. Suitable for core portfolio allocation.',
            'B': 'HOLD: Good performance with some areas for improvement. Maintain current allocation.',
            'C': 'CAUTION: Average performance. Consider reducing position size or waiting for improvement.',
            'D': 'AVOID: Below average performance. Not recommended for new allocations.',
            'F': 'SELL: Poor performance. Consider exiting positions.'
        }
        
        base_recommendation = recommendations.get(grade, 'No recommendation available.')
        
        # Add specific advice based on weak categories
        weak_categories = [cat for cat, score in category_scores.items() if score < 0.6]
        
        if weak_categories:
            advice = f" Weaknesses detected in: {', '.join(weak_categories)}."
            
            # Specific advice for each weak category
            specific_advice = []
            if 'drawdown' in weak_categories:
                specific_advice.append("Consider implementing stop-loss orders.")
            if 'risk' in weak_categories:
                specific_advice.append("Reduce position size to manage volatility.")
            if 'consistency' in weak_categories:
                specific_advice.append("Review trade execution and timing.")
            
            if specific_advice:
                advice += " Suggestions: " + "; ".join(specific_advice)
            
            return base_recommendation + advice
        
        return base_recommendation

# Specialized metric calculators
class MetricCalculator:
    """Static methods for calculating individual metrics."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, 
                              risk_free_rate: float = 0.02,
                              annualization_factor: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            annualization_factor: Trading periods per year
        
        Returns:
            float: Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / annualization_factor
        if returns.std() > 0:
            return excess_returns.mean() / returns.std() * np.sqrt(annualization_factor)
        return 0.0
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series,
                               risk_free_rate: float = 0.02,
                               annualization_factor: int = 252) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            annualization_factor: Trading periods per year
        
        Returns:
            float: Sortino ratio
        """
        excess_returns = returns - risk_free_rate / annualization_factor
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            return excess_returns.mean() / downside_returns.std() * np.sqrt(annualization_factor)
        elif excess_returns.mean() > 0:
            return float('inf')
        else:
            return 0.0
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Series of equity values
        
        Returns:
            float: Maximum drawdown (negative percentage)
        """
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min() * 100  # Return as percentage
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series,
                              equity_curve: pd.Series,
                              annualization_factor: int = 252) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            returns: Series of returns
            equity_curve: Series of equity values
            annualization_factor: Trading periods per year
        
        Returns:
            float: Calmar ratio
        """
        cumulative_return = np.prod(1 + returns) - 1
        
        # Annual return
        total_days = (returns.index[-1] - returns.index[0]).days
        years = total_days / 365.25
        annual_return = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else cumulative_return
        
        # Max drawdown
        max_dd = MetricCalculator.calculate_max_drawdown(equity_curve)
        
        if max_dd != 0:
            return annual_return / abs(max_dd)
        elif annual_return > 0:
            return float('inf')
        else:
            return 0.0
    
    @staticmethod
    def calculate_value_at_risk(returns: pd.Series,
                               confidence_level: float = 0.95,
                               method: str = 'historical') -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Calculation method ('historical', 'parametric', 'modified')
        
        Returns:
            float: Value at Risk (negative number)
        """
        if method == 'historical':
            return -np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # Assuming normal distribution
            mean = returns.mean()
            std = returns.std()
            return -(mean + std * norm.ppf(1 - confidence_level))
        
        elif method == 'modified':
            # Cornish-Fisher expansion for non-normal distributions
            z = norm.ppf(1 - confidence_level)
            s = returns.skew()
            k = returns.kurtosis()
            
            z_cf = (z + (z**2 - 1) * s / 6 + 
                   (z**3 - 3*z) * (k - 3) / 24 - 
                   (2*z**3 - 5*z) * s**2 / 36)
            
            mean = returns.mean()
            std = returns.std()
            return -(mean + std * z_cf)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    @staticmethod
    def calculate_conditional_var(returns: pd.Series,
                                 confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
        
        Returns:
            float: Conditional VaR (negative number)
        """
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) > 0:
            return -tail_returns.mean()
        else:
            return -var_threshold
    
    @staticmethod
    def calculate_omega_ratio(returns: pd.Series,
                             threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.
        
        Args:
            returns: Series of returns
            threshold: Return threshold
        
        Returns:
            float: Omega ratio
        """
        upside = returns[returns > threshold] - threshold
        downside = threshold - returns[returns <= threshold]
        
        if len(downside) > 0 and downside.sum() > 0:
            return upside.sum() / downside.sum()
        elif len(upside) > 0:
            return float('inf')
        else:
            return 0.0
    
    @staticmethod
    def calculate_ulcer_index(equity_curve: pd.Series) -> float:
        """
        Calculate Ulcer Index.
        
        Args:
            equity_curve: Series of equity values
        
        Returns:
            float: Ulcer Index
        """
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        drawdown_pct = drawdown * 100
        
        negative_dd = drawdown_pct[drawdown_pct < 0]
        if len(negative_dd) > 0:
            return np.sqrt(np.mean(negative_dd ** 2))
        else:
            return 0.0
    
    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        """
        Calculate profit factor.
        
        Args:
            trades: DataFrame with 'pnl' column
        
        Returns:
            float: Profit factor
        """
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] <= 0]['pnl'].sum())
        
        if gross_loss > 0:
            return gross_profit / gross_loss
        elif gross_profit > 0:
            return float('inf')
        else:
            return 0.0
    
    @staticmethod
    def calculate_k_ratio(trades: pd.DataFrame) -> float:
        """
        Calculate K-Ratio (measure of return persistence).
        
        Args:
            trades: DataFrame with 'pnl' and 'exit_time' columns
        
        Returns:
            float: K-Ratio
        """
        if 'exit_time' not in trades.columns or len(trades) < 10:
            return 0.0
        
        # Sort trades by exit time
        trades_sorted = trades.sort_values('exit_time')
        cumulative_pnl = trades_sorted['pnl'].cumsum()
        
        # Fit linear regression
        x = np.arange(len(cumulative_pnl))
        y = cumulative_pnl.values
        
        if len(y) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            residuals = y - (slope * x + intercept)
            std_error = np.std(residuals) / np.sqrt(len(y))
            
            if std_error > 0:
                return slope / std_error
        
        return 0.0

# Batch analysis utilities
class BatchAnalyzer:
    """Utilities for batch analysis of multiple strategies."""
    
    @staticmethod
    def analyze_strategies(strategy_results: Dict[str, Dict[str, Any]],
                          config: MetricsConfig = None) -> pd.DataFrame:
        """
        Analyze multiple strategies and return comparison table.
        
        Args:
            strategy_results: Dictionary of strategy names to results dictionaries
            config: Metrics configuration
        
        Returns:
            pd.DataFrame: Comparison table
        """
        all_metrics = []
        
        for strategy_name, results in strategy_results.items():
            analyzer = PerformanceAnalyzer(config)
            
            # Load data from results
            if 'returns' in results:
                analyzer.load_returns(results['returns'])
            
            if 'equity_curve' in results:
                analyzer.load_equity_curve(results['equity_curve'])
            
            if 'trades' in results:
                analyzer.load_trades(results['trades'])
            
            # Calculate metrics
            metrics = analyzer.calculate_all_metrics()
            metrics_dict = metrics.to_dict()
            metrics_dict['strategy'] = strategy_name
            
            all_metrics.append(metrics_dict)
        
        # Create DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Reorder columns
        cols = ['strategy'] + [c for c in df.columns if c != 'strategy']
        return df[cols]
    
    @staticmethod
    def rank_strategies(metrics_df: pd.DataFrame,
                       ranking_metric: str = 'composite_score',
                       ascending: bool = False) -> pd.DataFrame:
        """
        Rank strategies based on specified metric.
        
        Args:
            metrics_df: DataFrame with strategy metrics
            ranking_metric: Metric to use for ranking
            ascending: Sort order
        
        Returns:
            pd.DataFrame: Ranked strategies
        """
        if ranking_metric not in metrics_df.columns:
            raise ValueError(f"Metric '{ranking_metric}' not found in DataFrame")
        
        ranked = metrics_df.sort_values(ranking_metric, ascending=ascending).copy()
        ranked['rank'] = range(1, len(ranked) + 1)
        
        return ranked
    
    @staticmethod
    def create_correlation_matrix(strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Create correlation matrix for multiple strategies.
        
        Args:
            strategy_returns: Dictionary of strategy names to returns series
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Align all returns to common index
        aligned_returns = {}
        for name, returns in strategy_returns.items():
            aligned_returns[name] = returns
        
        # Create DataFrame
        df = pd.DataFrame(aligned_returns)
        
        # Calculate correlation matrix
        return df.corr()
    
    @staticmethod
    def create_performance_heatmap(metrics_df: pd.DataFrame,
                                  metrics: List[str] = None) -> pd.DataFrame:
        """
        Create performance heatmap data.
        
        Args:
            metrics_df: DataFrame with strategy metrics
            metrics: List of metrics to include
        
        Returns:
            pd.DataFrame: Heatmap data
        """
        if metrics is None:
            metrics = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        
        # Filter metrics that exist in DataFrame
        existing_metrics = [m for m in metrics if m in metrics_df.columns]
        
        if not existing_metrics:
            raise ValueError("No valid metrics found")
        
        # Select and normalize metrics
        heatmap_data = metrics_df[['strategy'] + existing_metrics].copy()
        
        # Normalize each metric to 0-1 scale
        for metric in existing_metrics:
            if metric in ['max_drawdown']:  # Negative is bad
                min_val = heatmap_data[metric].min()
                max_val = heatmap_data[metric].max()
                if max_val > min_val:
                    heatmap_data[f'{metric}_norm'] = 1 - (heatmap_data[metric] - min_val) / (max_val - min_val)
                else:
                    heatmap_data[f'{metric}_norm'] = 0.5
            else:  # Positive is good
                min_val = heatmap_data[metric].min()
                max_val = heatmap_data[metric].max()
                if max_val > min_val:
                    heatmap_data[f'{metric}_norm'] = (heatmap_data[metric] - min_val) / (max_val - min_val)
                else:
                    heatmap_data[f'{metric}_norm'] = 0.5
        
        return heatmap_data

# Example usage
if __name__ == "__main__":
    print("Testing Performance Metrics Module...")
    
    # Generate sample data
    np.random.seed(42)
    
    # Generate returns
    dates = pd.date_range('2023-01-01', periods=252, freq='B')  # Business days
    returns = pd.Series(np.random.normal(0.0005, 0.02, len(dates)), index=dates)
    
    # Generate equity curve
    equity_curve = 10000 * np.cumprod(1 + returns)
    
    # Generate trades
    trade_dates = dates[::5]  # Every 5 days
    trades = pd.DataFrame({
        'entry_time': trade_dates,
        'exit_time': trade_dates + pd.Timedelta(days=1),
        'pnl': np.random.normal(50, 200, len(trade_dates))
    })
    
    # Generate benchmark returns
    benchmark_returns = pd.Series(np.random.normal(0.0003, 0.015, len(dates)), index=dates)
    
    # Create configuration
    config = MetricsConfig(
        risk_free_rate=0.02,
        benchmark_returns=benchmark_returns,
        benchmark_name="S&P 500"
    )
    
    # Create analyzer
    analyzer = PerformanceAnalyzer(config)
    analyzer.load_returns(returns)
    analyzer.load_equity_curve(equity_curve)
    analyzer.load_trades(trades)
    
    # Calculate all metrics
    metrics = analyzer.calculate_all_metrics()
    
    # Print key metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*60)
    
    print(f"\nReturn Metrics:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Annual Return: {metrics.annual_return:.2%}")
    print(f"  Cumulative Return: {metrics.cumulative_return:.2%}")
    
    print(f"\nRisk Metrics:")
    print(f"  Volatility: {metrics.volatility:.2%}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2f}%")
    print(f"  Value at Risk (95%): {metrics.value_at_risk:.2%}")
    
    print(f"\nRisk-Adjusted Metrics:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")
    
    print(f"\nTrade Metrics:")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Win Rate: {metrics.win_rate:.2%}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Expectancy: ${metrics.expectancy:.2f}")
    
    print(f"\nStatistical Metrics:")
    print(f"  Skewness: {metrics.skewness:.3f}")
    print(f"  Kurtosis: {metrics.kurtosis:.3f}")
    print(f"  Jarque-Bera p-value: {metrics.jarque_bera_pvalue:.4f}")
    
    print(f"\nComposite Scores:")
    print(f"  Overall Score: {metrics.composite_score:.3f}")
    print(f"  Risk-Adjusted Score: {metrics.risk_adjusted_score:.3f}")
    print(f"  Consistency Score: {metrics.consistency_score:.3f}")
    
    # Get performance grade
    grade_info = analyzer.get_performance_grade()
    print(f"\nPerformance Grade: {grade_info['overall_grade']} ({grade_info['grade_info']['description']})")
    print(f"Recommendation: {grade_info['recommendation']}")
    
    # Compare with benchmark
    print(f"\n" + "="*60)
    print("BENCHMARK COMPARISON")
    print("="*60)
    
    comparison = analyzer.compare_with_benchmark()
    print(comparison.to_string(index=False))
    
    # Test individual metric calculators
    print(f"\n" + "="*60)
    print("INDIVIDUAL METRIC CALCULATIONS")
    print("="*60)
    
    sharpe = MetricCalculator.calculate_sharpe_ratio(returns)
    sortino = MetricCalculator.calculate_sortino_ratio(returns)
    max_dd = MetricCalculator.calculate_max_drawdown(equity_curve)
    var = MetricCalculator.calculate_value_at_risk(returns)
    
    print(f"Sharpe Ratio (individual): {sharpe:.3f}")
    print(f"Sortino Ratio (individual): {sortino:.3f}")
    print(f"Max Drawdown (individual): {max_dd:.2f}%")
    print(f"VaR 95% (individual): {var:.2%}")
    
    print("\nPerformance metrics calculation completed successfully!")