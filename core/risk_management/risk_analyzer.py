"""
Risk Analyzer module for Bitcoin trading AI.
Comprehensive risk management including VaR, CVaR, stress testing, position sizing, and risk limits.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
import json
from pathlib import Path
import hashlib
import asyncio
from collections import deque, defaultdict
import pickle
from functools import lru_cache

# Import project modules
from config.settings import TradingSettings, RiskSettings, AppConstants
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.trading.position_sizer import PositionSizeResult
from core.trading.order_manager import Order, OrderSide, OrderType
from core.utils.cache import Cache

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Enums and Types ============
class RiskLevel(str, Enum):
    """Risk levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

class RiskMetric(str, Enum):
    """Risk metrics"""
    VAR = "var"                    # Value at Risk
    CVAR = "cvar"                  # Conditional Value at Risk
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    BETA = "beta"                  # Market correlation
    CORRELATION = "correlation"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"
    LEVERAGE_RISK = "leverage_risk"

class StressTestScenario(str, Enum):
    """Stress test scenarios"""
    FLASH_CRASH_2010 = "flash_crash_2010"
    BITCOIN_2017_CRASH = "bitcoin_2017_crash"
    COVID_CRASH_2020 = "covid_crash_2020"
    BITCOIN_2021_CORRECTION = "bitcoin_2021_correction"
    FTX_COLLAPSE_2022 = "ftx_collapse_2022"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    EXCHANGE_HACK = "exchange_hack"
    REGULATORY_CRACKDOWN = "regulatory_crackdown"
    BLACK_SWAN = "black_swan"

# ============ Data Structures ============
@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a portfolio or position"""
    
    # Core metrics
    value_at_risk_95: float = 0.0          # 95% confidence VaR
    value_at_risk_99: float = 0.0          # 99% confidence VaR
    conditional_var_95: float = 0.0        # 95% confidence CVaR
    conditional_var_99: float = 0.0        # 99% confidence CVaR
    expected_shortfall: float = 0.0        # Expected shortfall
    max_drawdown: float = 0.0              # Maximum drawdown
    max_drawdown_duration: int = 0         # Duration in days
    
    # Volatility metrics
    daily_volatility: float = 0.0          # Daily volatility (std dev)
    annual_volatility: float = 0.0         # Annualized volatility
    realized_volatility: float = 0.0       # Realized volatility
    implied_volatility: float = 0.0        # Implied volatility (if available)
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0              # Sharpe ratio
    sortino_ratio: float = 0.0             # Sortino ratio (downside risk)
    calmar_ratio: float = 0.0              # Calmar ratio (return/drawdown)
    information_ratio: float = 0.0         # Information ratio
    
    # Correlation and sensitivity
    beta_to_btc: float = 0.0               # Beta to Bitcoin
    beta_to_sp500: float = 0.0             # Beta to S&P 500
    correlation_matrix: Optional[pd.DataFrame] = None
    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)
    
    # Liquidity risk
    bid_ask_spread: float = 0.0            # Average bid-ask spread
    market_depth: float = 0.0              # Market depth at +/- 2%
    liquidity_score: float = 0.0           # 0-100 liquidity score
    slippage_estimate: float = 0.0         # Estimated slippage for medium order
    
    # Concentration risk
    herfindahl_index: float = 0.0          # Concentration index
    position_concentration: float = 0.0    # Largest position percentage
    sector_concentration: float = 0.0      # Sector concentration
    
    # Leverage risk
    leverage_ratio: float = 1.0            # Current leverage
    max_leverage_used: float = 1.0         # Maximum leverage used
    liquidation_price: Optional[float] = None  # Liquidation price
    
    # Stress test results
    stress_test_losses: Dict[str, float] = field(default_factory=dict)
    worst_case_scenario: Optional[str] = None
    worst_case_loss: float = 0.0
    
    # Risk level
    overall_risk_level: RiskLevel = RiskLevel.MODERATE
    risk_score: float = 50.0               # 0-100 risk score
    
    # Metadata
    calculation_time: datetime = field(default_factory=datetime.now)
    confidence_interval: float = 0.95
    lookback_period: int = 252             # Trading days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'value_at_risk_95': self.value_at_risk_95,
            'value_at_risk_99': self.value_at_risk_99,
            'conditional_var_95': self.conditional_var_95,
            'conditional_var_99': self.conditional_var_99,
            'expected_shortfall': self.expected_shortfall,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'daily_volatility': self.daily_volatility,
            'annual_volatility': self.annual_volatility,
            'realized_volatility': self.realized_volatility,
            'implied_volatility': self.implied_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'information_ratio': self.information_ratio,
            'beta_to_btc': self.beta_to_btc,
            'beta_to_sp500': self.beta_to_sp500,
            'bid_ask_spread': self.bid_ask_spread,
            'market_depth': self.market_depth,
            'liquidity_score': self.liquidity_score,
            'slippage_estimate': self.slippage_estimate,
            'herfindahl_index': self.herfindahl_index,
            'position_concentration': self.position_concentration,
            'sector_concentration': self.sector_concentration,
            'leverage_ratio': self.leverage_ratio,
            'max_leverage_used': self.max_leverage_used,
            'liquidation_price': self.liquidation_price,
            'stress_test_losses': self.stress_test_losses,
            'worst_case_scenario': self.worst_case_scenario,
            'worst_case_loss': self.worst_case_loss,
            'overall_risk_level': self.overall_risk_level.value,
            'risk_score': self.risk_score,
            'calculation_time': self.calculation_time.isoformat(),
            'confidence_interval': self.confidence_interval,
            'lookback_period': self.lookback_period
        }
        
        if self.correlation_matrix is not None:
            result['correlation_matrix'] = self.correlation_matrix.to_dict()
        
        return result

@dataclass
class RiskLimit:
    """Risk limit configuration"""
    metric: RiskMetric
    limit_value: float
    warning_threshold: float = 0.8  # 80% of limit
    time_period: str = "daily"      # daily, weekly, monthly
    is_hard_limit: bool = True      # Hard limit (cannot exceed) vs soft limit
    description: str = ""
    
    def is_exceeded(self, current_value: float) -> bool:
        """Check if limit is exceeded"""
        return current_value > self.limit_value
    
    def is_warning(self, current_value: float) -> bool:
        """Check if warning threshold is reached"""
        return current_value > self.limit_value * self.warning_threshold

@dataclass
class RiskReport:
    """Comprehensive risk report"""
    report_id: str
    portfolio_id: str
    timestamp: datetime
    metrics: RiskMetrics
    limits_violated: List[RiskLimit] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    stress_test_summary: Dict[str, Any] = field(default_factory=dict)
    analysis_period: str = "daily"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'report_id': self.report_id,
            'portfolio_id': self.portfolio_id,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics.to_dict(),
            'limits_violated': [
                {
                    'metric': limit.metric.value,
                    'limit_value': limit.limit_value,
                    'current_value': getattr(self.metrics, limit.metric.value, 0.0),
                    'description': limit.description
                }
                for limit in self.limits_violated
            ],
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'stress_test_summary': self.stress_test_summary,
            'analysis_period': self.analysis_period
        }

@dataclass
class PortfolioState:
    """Portfolio state for risk analysis"""
    timestamp: datetime
    positions: Dict[str, float]              # symbol -> quantity
    cash: float
    portfolio_value: float
    leverage: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============ Risk Configuration ============
class RiskConfig:
    """Risk analysis configuration"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config = config_dict or self._default_config()
        self.limits = self._initialize_limits()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default risk configuration"""
        return {
            # VaR parameters
            'var_confidence_level': 0.95,
            'var_lookback_days': 252,
            'var_method': 'historical',  # historical, parametric, monte_carlo
            
            # CVaR parameters
            'cvar_confidence_level': 0.95,
            
            # Volatility calculation
            'volatility_lookback': 30,
            'volatility_annualization_factor': np.sqrt(252),
            
            # Drawdown parameters
            'drawdown_lookback': 252,
            
            # Correlation parameters
            'correlation_lookback': 90,
            'min_correlation_data_points': 30,
            
            # Stress test parameters
            'stress_test_scenarios': [
                StressTestScenario.FLASH_CRASH_2010,
                StressTestScenario.BITCOIN_2017_CRASH,
                StressTestScenario.COVID_CRASH_2020,
                StressTestScenario.FTX_COLLAPSE_2022
            ],
            'stress_test_shock_size': 0.3,  # 30% shock for generic scenarios
            
            # Liquidity risk
            'liquidity_threshold': 0.05,  # 5% of daily volume
            'slippage_model': 'linear',   # linear, square_root
            
            # Concentration limits
            'max_position_concentration': 0.2,  # 20% of portfolio
            'max_sector_concentration': 0.5,    # 50% of portfolio
            
            # Leverage limits
            'max_leverage': 3.0,
            'leverage_warning_threshold': 2.0,
            
            # Risk score calculation
            'risk_score_weights': {
                'volatility': 0.2,
                'drawdown': 0.25,
                'var': 0.15,
                'liquidity': 0.15,
                'concentration': 0.15,
                'leverage': 0.1
            },
            
            # Risk level thresholds
            'risk_level_thresholds': {
                RiskLevel.LOW: 25,
                RiskLevel.MODERATE: 50,
                RiskLevel.HIGH: 75,
                RiskLevel.EXTREME: 100
            },
            
            # Reporting
            'reporting_frequency': 'daily',
            'save_reports': True,
            'report_directory': 'data/risk_reports/'
        }
    
    def _initialize_limits(self) -> List[RiskLimit]:
        """Initialize default risk limits"""
        return [
            # VaR limits
            RiskLimit(
                metric=RiskMetric.VAR,
                limit_value=0.05,  # 5% daily VaR
                warning_threshold=0.8,
                time_period="daily",
                is_hard_limit=False,
                description="Daily Value at Risk limit"
            ),
            
            # Drawdown limits
            RiskLimit(
                metric=RiskMetric.MAX_DRAWDOWN,
                limit_value=0.20,  # 20% max drawdown
                warning_threshold=0.8,
                time_period="monthly",
                is_hard_limit=True,
                description="Maximum drawdown limit"
            ),
            
            # Volatility limits
            RiskLimit(
                metric=RiskMetric.VOLATILITY,
                limit_value=0.80,  # 80% annual volatility
                warning_threshold=0.8,
                time_period="daily",
                is_hard_limit=False,
                description="Annual volatility limit"
            ),
            
            # Leverage limits
            RiskLimit(
                metric=RiskMetric.LEVERAGE_RISK,
                limit_value=3.0,  # 3x leverage
                warning_threshold=0.8,
                time_period="daily",
                is_hard_limit=True,
                description="Maximum leverage limit"
            ),
            
            # Concentration limits
            RiskLimit(
                metric=RiskMetric.CONCENTRATION_RISK,
                limit_value=0.30,  # 30% in single position
                warning_threshold=0.8,
                time_period="daily",
                is_hard_limit=True,
                description="Maximum position concentration"
            ),
            
            # Liquidity risk limits
            RiskLimit(
                metric=RiskMetric.LIQUIDITY_RISK,
                limit_value=0.10,  # 10% slippage
                warning_threshold=0.8,
                time_period="daily",
                is_hard_limit=False,
                description="Maximum estimated slippage"
            )
        ]

# ============ Risk Models ============
class ValueAtRiskModel:
    """Value at Risk calculation models"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def calculate_historical_var(self,
                               returns: pd.Series,
                               confidence_level: float = 0.95,
                               portfolio_value: float = 1.0) -> float:
        """Calculate historical VaR"""
        if len(returns) < 30:
            self.logger.warning(f"Insufficient data for VaR calculation: {len(returns)} observations")
            return 0.0
        
        var_percentile = 1 - confidence_level
        var = np.percentile(returns, var_percentile * 100)
        
        # Convert to dollar amount if portfolio value provided
        if portfolio_value != 1.0:
            var = var * portfolio_value
        
        return var
    
    def calculate_parametric_var(self,
                               returns: pd.Series,
                               confidence_level: float = 0.95,
                               portfolio_value: float = 1.0,
                               distribution: str = 'normal') -> float:
        """Calculate parametric VaR (assuming distribution)"""
        if len(returns) < 30:
            self.logger.warning(f"Insufficient data for parametric VaR: {len(returns)} observations")
            return 0.0
        
        mean = returns.mean()
        std = returns.std()
        
        if distribution == 'normal':
            # Normal distribution
            z_score = stats.norm.ppf(1 - confidence_level)
        elif distribution == 't':
            # Student's t-distribution
            # Fit t-distribution to returns
            params = stats.t.fit(returns)
            z_score = stats.t.ppf(1 - confidence_level, *params)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        var = mean + z_score * std
        
        # Convert to dollar amount
        if portfolio_value != 1.0:
            var = var * portfolio_value
        
        return var
    
    def calculate_monte_carlo_var(self,
                                 returns: pd.Series,
                                 confidence_level: float = 0.95,
                                 portfolio_value: float = 1.0,
                                 simulations: int = 10000,
                                 time_horizon: int = 1) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        if len(returns) < 30:
            self.logger.warning(f"Insufficient data for Monte Carlo VaR: {len(returns)} observations")
            return 0.0
        
        mean = returns.mean()
        std = returns.std()
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean, std, (simulations, time_horizon))
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = np.prod(1 + simulated_returns, axis=1) - 1
        
        # Calculate VaR
        var_percentile = 1 - confidence_level
        var = np.percentile(portfolio_returns, var_percentile * 100)
        
        # Convert to dollar amount
        if portfolio_value != 1.0:
            var = var * portfolio_value
        
        return var

class ExpectedShortfallModel:
    """Expected Shortfall (Conditional VaR) models"""
    
    def calculate_historical_cvar(self,
                                returns: pd.Series,
                                confidence_level: float = 0.95,
                                portfolio_value: float = 1.0) -> float:
        """Calculate historical Expected Shortfall"""
        if len(returns) < 30:
            self.logger.warning(f"Insufficient data for CVaR calculation: {len(returns)} observations")
            return 0.0
        
        # Calculate VaR threshold
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Calculate average of returns worse than VaR
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return var_threshold
        
        cvar = tail_returns.mean()
        
        # Convert to dollar amount
        if portfolio_value != 1.0:
            cvar = cvar * portfolio_value
        
        return cvar
    
    def calculate_parametric_cvar(self,
                                returns: pd.Series,
                                confidence_level: float = 0.95,
                                portfolio_value: float = 1.0,
                                distribution: str = 'normal') -> float:
        """Calculate parametric Expected Shortfall"""
        if len(returns) < 30:
            self.logger.warning(f"Insufficient data for parametric CVaR: {len(returns)} observations")
            return 0.0
        
        mean = returns.mean()
        std = returns.std()
        
        if distribution == 'normal':
            # For normal distribution
            z_score = stats.norm.ppf(1 - confidence_level)
            pdf_z = stats.norm.pdf(z_score)
            cdf_z = stats.norm.cdf(z_score)
            
            if cdf_z > 0:
                cvar = mean - std * (pdf_z / cdf_z)
            else:
                cvar = mean + z_score * std
        else:
            # For other distributions, use historical method on fitted distribution
            raise NotImplementedError(f"CVaR for {distribution} distribution not implemented")
        
        # Convert to dollar amount
        if portfolio_value != 1.0:
            cvar = cvar * portfolio_value
        
        return cvar

class DrawdownAnalyzer:
    """Maximum drawdown analysis"""
    
    def calculate_max_drawdown(self,
                              prices: pd.Series,
                              return_percent: bool = True) -> Tuple[float, int, int, int]:
        """Calculate maximum drawdown and duration"""
        if len(prices) < 2:
            return 0.0, 0, 0, 0
        
        # Calculate cumulative returns
        cumulative_returns = prices / prices.iloc[0] - 1 if return_percent else prices
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = cumulative_returns - running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        max_drawdown_index = drawdown.idxmin()
        
        # Find peak before drawdown
        if max_drawdown_index is not None:
            peak_index = drawdown[:max_drawdown_index].idxmax()
            drawdown_duration = (max_drawdown_index - peak_index).days if hasattr(max_drawdown_index, 'day') else 0
        else:
            peak_index = None
            drawdown_duration = 0
        
        # Recovery period (time to return to previous peak)
        if max_drawdown_index is not None and len(prices) > max_drawdown_index:
            recovery_data = prices[max_drawdown_index:]
            if len(recovery_data) > 0:
                previous_peak = prices.loc[peak_index] if peak_index is not None else prices.iloc[0]
                recovery_mask = recovery_data >= previous_peak
                if recovery_mask.any():
                    recovery_index = recovery_mask.idxmax()
                    recovery_period = (recovery_index - max_drawdown_index).days if hasattr(recovery_index, 'day') else 0
                else:
                    recovery_period = -1  # Never recovered
            else:
                recovery_period = 0
        else:
            recovery_period = 0
        
        max_drawdown_value = abs(max_drawdown) if return_percent else abs(max_drawdown)
        
        return max_drawdown_value, drawdown_duration, recovery_period, max_drawdown_index

class VolatilityModel:
    """Volatility calculation models"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
    
    def calculate_historical_volatility(self,
                                      returns: pd.Series,
                                      annualize: bool = True) -> float:
        """Calculate historical volatility"""
        if len(returns) < 2:
            return 0.0
        
        volatility = returns.std()
        
        if annualize:
            volatility *= np.sqrt(self.config.config['volatility_annualization_factor'])
        
        return volatility
    
    def calculate_realized_volatility(self,
                                    returns: pd.Series,
                                    window: int = 20) -> pd.Series:
        """Calculate realized volatility using rolling window"""
        if len(returns) < window:
            return pd.Series([returns.std()] * len(returns), index=returns.index)
        
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def calculate_garch_volatility(self,
                                 returns: pd.Series,
                                 p: int = 1,
                                 q: int = 1) -> pd.Series:
        """Calculate volatility using GARCH model"""
        try:
            from arch import arch_model
            
            if len(returns) < 100:
                self.logger.warning(f"Insufficient data for GARCH: {len(returns)} observations")
                return self.calculate_realized_volatility(returns)
            
            # Fit GARCH model
            model = arch_model(returns, vol='Garch', p=p, q=q)
            result = model.fit(disp='off')
            
            # Get conditional volatility
            volatility = result.conditional_volatility
            
            return volatility
            
        except ImportError:
            self.logger.warning("ARCH package not installed, using realized volatility")
            return self.calculate_realized_volatility(returns)
        except Exception as e:
            self.logger.error(f"GARCH model failed: {str(e)}")
            return self.calculate_realized_volatility(returns)

class CorrelationAnalyzer:
    """Correlation and beta analysis"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
    
    def calculate_correlation_matrix(self,
                                   returns_df: pd.DataFrame,
                                   min_periods: int = 30) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets"""
        if len(returns_df) < min_periods:
            self.logger.warning(f"Insufficient data for correlation matrix: {len(returns_df)} observations")
            return pd.DataFrame()
        
        return returns_df.corr(min_periods=min_periods)
    
    def calculate_beta(self,
                      asset_returns: pd.Series,
                      market_returns: pd.Series,
                      min_periods: int = 30) -> float:
        """Calculate beta of asset to market"""
        if len(asset_returns) < min_periods or len(market_returns) < min_periods:
            self.logger.warning("Insufficient data for beta calculation")
            return 1.0
        
        # Align indices
        aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
        
        if len(aligned_data) < min_periods:
            return 1.0
        
        asset = aligned_data.iloc[:, 0]
        market = aligned_data.iloc[:, 1]
        
        # Calculate covariance and variance
        covariance = asset.cov(market)
        market_variance = market.var()
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return beta
    
    def calculate_correlation_breakdown(self,
                                      returns_df: pd.DataFrame,
                                      threshold: float = 0.7) -> Dict[str, List[str]]:
        """Identify highly correlated pairs"""
        corr_matrix = self.calculate_correlation_matrix(returns_df)
        
        if corr_matrix.empty:
            return {}
        
        highly_correlated = {}
        
        # Get upper triangle of correlation matrix
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find pairs with correlation above threshold
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = upper_triangle.iloc[i, j]
                if not pd.isna(corr_value) and abs(corr_value) > threshold:
                    pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                    highly_correlated[str(pair)] = corr_value
        
        return highly_correlated

class LiquidityRiskModel:
    """Liquidity risk assessment"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
    
    def calculate_bid_ask_spread(self,
                                bid_prices: pd.Series,
                                ask_prices: pd.Series,
                                mid_prices: Optional[pd.Series] = None) -> pd.Series:
        """Calculate bid-ask spread"""
        if mid_prices is None:
            mid_prices = (bid_prices + ask_prices) / 2
        
        spread = (ask_prices - bid_prices) / mid_prices
        return spread
    
    def estimate_slippage(self,
                         order_size: float,
                         daily_volume: float,
                         volatility: float,
                         model: str = 'linear') -> float:
        """Estimate slippage for given order size"""
        
        # Calculate volume ratio
        volume_ratio = order_size / daily_volume if daily_volume > 0 else 1.0
        
        if model == 'linear':
            # Simple linear model
            slippage = volume_ratio * volatility * 2
        elif model == 'square_root':
            # Square root model (common in market impact models)
            slippage = np.sqrt(volume_ratio) * volatility
        else:
            raise ValueError(f"Unknown slippage model: {model}")
        
        return min(slippage, 0.5)  # Cap at 50%
    
    def calculate_liquidity_score(self,
                                 daily_volume: float,
                                 avg_spread: float,
                                 market_depth: float,
                                 volatility: float) -> float:
        """Calculate liquidity score (0-100)"""
        
        # Normalize each factor to 0-1 scale
        volume_score = min(daily_volume / 1000.0, 1.0)  # 1000 BTC/day = perfect score
        spread_score = max(0, 1 - avg_spread / 0.05)    # 5% spread = 0 score
        depth_score = min(market_depth / 100.0, 1.0)    # 100 BTC depth = perfect score
        volatility_penalty = min(volatility / 2.0, 1.0)  # High volatility reduces score
        
        # Weighted average
        liquidity_score = (
            volume_score * 0.4 +
            spread_score * 0.3 +
            depth_score * 0.3
        ) * (1 - volatility_penalty * 0.5)
        
        return max(0, min(100, liquidity_score * 100))

class ConcentrationAnalyzer:
    """Portfolio concentration analysis"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
    
    def calculate_herfindahl_index(self,
                                  position_values: Dict[str, float],
                                  total_portfolio_value: float) -> float:
        """Calculate Herfindahl-Hirschman Index (HHI) for concentration"""
        if total_portfolio_value <= 0:
            return 0.0
        
        hhi = 0.0
        for value in position_values.values():
            market_share = value / total_portfolio_value
            hhi += market_share ** 2
        
        return hhi
    
    def calculate_position_concentration(self,
                                       position_values: Dict[str, float],
                                       total_portfolio_value: float) -> Dict[str, float]:
        """Calculate concentration metrics for each position"""
        if total_portfolio_value <= 0:
            return {symbol: 0.0 for symbol in position_values.keys()}
        
        concentration = {}
        for symbol, value in position_values.items():
            concentration[symbol] = value / total_portfolio_value
        
        return concentration
    
    def get_top_positions(self,
                         position_values: Dict[str, float],
                         top_n: int = 5) -> List[Tuple[str, float]]:
        """Get top N positions by value"""
        sorted_positions = sorted(position_values.items(), key=lambda x: x[1], reverse=True)
        return sorted_positions[:top_n]

class StressTestEngine:
    """Stress testing engine for various scenarios"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.scenario_data = self._load_scenario_data()
    
    def _load_scenario_data(self) -> Dict[str, Dict[str, Any]]:
        """Load historical stress test scenarios"""
        
        # Historical Bitcoin drawdowns and events
        scenarios = {
            StressTestScenario.FLASH_CRASH_2010.value: {
                'name': 'Flash Crash 2010',
                'date': '2010-05-06',
                'btc_drawdown': -0.15,  # -15%
                'duration_days': 1,
                'recovery_days': 7,
                'description': 'US Stock Market Flash Crash analogy for crypto'
            },
            StressTestScenario.BITCOIN_2017_CRASH.value: {
                'name': 'Bitcoin 2017 Crash',
                'date': '2017-12-17',
                'btc_drawdown': -0.65,  # -65%
                'duration_days': 365,
                'recovery_days': 1095,  # 3 years
                'description': 'Bitcoin bubble burst after 2017 bull run'
            },
            StressTestScenario.COVID_CRASH_2020.value: {
                'name': 'COVID-19 Crash 2020',
                'date': '2020-03-12',
                'btc_drawdown': -0.50,  # -50%
                'duration_days': 3,
                'recovery_days': 60,
                'description': 'Global market crash due to COVID-19 pandemic'
            },
            StressTestScenario.BITCOIN_2021_CORRECTION.value: {
                'name': 'Bitcoin 2021 Correction',
                'date': '2021-05-19',
                'btc_drawdown': -0.53,  # -53%
                'duration_days': 30,
                'recovery_days': 180,
                'description': 'China mining crackdown and environmental concerns'
            },
            StressTestScenario.FTX_COLLAPSE_2022.value: {
                'name': 'FTX Collapse 2022',
                'date': '2022-11-08',
                'btc_drawdown': -0.25,  # -25%
                'duration_days': 7,
                'recovery_days': 90,
                'description': 'Major exchange collapse causing liquidity crisis'
            },
            StressTestScenario.INTEREST_RATE_SHOCK.value: {
                'name': 'Interest Rate Shock',
                'date': '2022-01-01',
                'btc_drawdown': -0.30,  # -30%
                'duration_days': 90,
                'recovery_days': 180,
                'description': 'Rapid interest rate increases affecting risk assets'
            },
            StressTestScenario.LIQUIDITY_CRISIS.value: {
                'name': 'Liquidity Crisis',
                'date': '2023-03-10',
                'btc_drawdown': -0.20,  # -20%
                'duration_days': 5,
                'recovery_days': 30,
                'description': 'Sudden liquidity withdrawal from market'
            },
            StressTestScenario.EXCHANGE_HACK.value: {
                'name': 'Major Exchange Hack',
                'date': '2014-02-28',
                'btc_drawdown': -0.40,  # -40%
                'duration_days': 1,
                'recovery_days': 60,
                'description': 'Mt.Gox hack scenario'
            },
            StressTestScenario.REGULATORY_CRACKDOWN.value: {
                'name': 'Regulatory Crackdown',
                'date': '2021-09-24',
                'btc_drawdown': -0.35,  # -35%
                'duration_days': 30,
                'recovery_days': 90,
                'description': 'Major regulatory announcement affecting crypto'
            },
            StressTestScenario.BLACK_SWAN.value: {
                'name': 'Black Swan Event',
                'date': '2020-03-12',
                'btc_drawdown': -0.60,  # -60%
                'duration_days': 1,
                'recovery_days': 365,
                'description': 'Extreme unexpected event'
            }
        }
        
        return scenarios
    
    def run_stress_test(self,
                       portfolio_state: PortfolioState,
                       market_data: Dict[str, pd.DataFrame],
                       scenarios: Optional[List[str]] = None) -> Dict[str, float]:
        """Run stress tests on portfolio"""
        
        if scenarios is None:
            scenarios = self.config.config['stress_test_scenarios']
        
        stress_test_results = {}
        
        for scenario_name in scenarios:
            if scenario_name not in self.scenario_data:
                self.logger.warning(f"Unknown stress test scenario: {scenario_name}")
                continue
            
            scenario = self.scenario_data[scenario_name]
            
            # Calculate portfolio loss under scenario
            loss = self._calculate_scenario_loss(
                portfolio_state, 
                market_data, 
                scenario
            )
            
            stress_test_results[scenario_name] = loss
        
        return stress_test_results
    
    def _calculate_scenario_loss(self,
                               portfolio_state: PortfolioState,
                               market_data: Dict[str, pd.DataFrame],
                               scenario: Dict[str, Any]) -> float:
        """Calculate portfolio loss for a specific scenario"""
        
        total_loss = 0.0
        
        for symbol, quantity in portfolio_state.positions.items():
            if symbol in market_data:
                # Get current price
                current_price = self._get_current_price(market_data[symbol])
                
                if current_price > 0:
                    # Calculate shock price
                    shock_multiplier = 1 + scenario['btc_drawdown']
                    shock_price = current_price * shock_multiplier
                    
                    # Calculate position loss
                    position_value = quantity * current_price
                    shocked_value = quantity * shock_price
                    position_loss = position_value - shocked_value
                    
                    total_loss += position_loss
        
        # Add correlation effects (simplified)
        # In reality, you would use a correlation matrix
        correlation_factor = 1.2  # Assume 20% additional loss from correlation
        
        total_loss *= correlation_factor
        
        return total_loss
    
    def _get_current_price(self, price_data: pd.DataFrame) -> float:
        """Get current price from market data"""
        if 'close' in price_data.columns:
            return price_data['close'].iloc[-1]
        elif 'price' in price_data.columns:
            return price_data['price'].iloc[-1]
        else:
            return price_data.iloc[-1, 0]  # First column

# ============ Main Risk Analyzer ============
class RiskAnalyzer:
    """Main risk analysis engine"""
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.logger = get_logger(__name__)
        
        # Initialize models
        self.var_model = ValueAtRiskModel(self.config)
        self.es_model = ExpectedShortfallModel()
        self.drawdown_analyzer = DrawdownAnalyzer()
        self.volatility_model = VolatilityModel(self.config)
        self.correlation_analyzer = CorrelationAnalyzer(self.config)
        self.liquidity_model = LiquidityRiskModel(self.config)
        self.concentration_analyzer = ConcentrationAnalyzer(self.config)
        self.stress_test_engine = StressTestEngine(self.config)
        
        # Cache for performance
        self.cache = Cache(ttl=300)  # 5 minutes TTL
        
        # Risk history
        self.risk_history = deque(maxlen=1000)
        
        # Performance tracking
        self.analysis_count = 0
        self.last_analysis_time = None
        
        self.logger.info("Risk Analyzer initialized")
    
    def analyze_portfolio_risk(self,
                             portfolio_state: PortfolioState,
                             market_data: Dict[str, pd.DataFrame],
                             historical_data: Optional[Dict[str, pd.DataFrame]] = None,
                             include_stress_tests: bool = True) -> RiskMetrics:
        """Comprehensive portfolio risk analysis"""
        
        self.analysis_count += 1
        self.last_analysis_time = datetime.now()
        
        # Check cache first
        cache_key = self._generate_cache_key(portfolio_state, market_data)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.logger.debug("Returning cached risk analysis")
            return cached_result
        
        # Calculate basic metrics
        returns_data = self._prepare_returns_data(market_data, historical_data)
        
        # Initialize risk metrics
        risk_metrics = RiskMetrics()
        
        try:
            # Calculate Value at Risk
            risk_metrics.value_at_risk_95, risk_metrics.value_at_risk_99 = self._calculate_var(
                returns_data, portfolio_state.portfolio_value
            )
            
            # Calculate Conditional VaR (Expected Shortfall)
            risk_metrics.conditional_var_95, risk_metrics.conditional_var_99 = self._calculate_cvar(
                returns_data, portfolio_state.portfolio_value
            )
            
            # Calculate volatility
            risk_metrics.daily_volatility, risk_metrics.annual_volatility = self._calculate_volatility(
                returns_data
            )
            
            # Calculate drawdown
            risk_metrics.max_drawdown, risk_metrics.max_drawdown_duration = self._calculate_drawdown(
                market_data
            )
            
            # Calculate risk-adjusted returns
            risk_metrics.sharpe_ratio, risk_metrics.sortino_ratio = self._calculate_risk_adjusted_ratios(
                returns_data
            )
            
            # Calculate correlation and beta
            risk_metrics.beta_to_btc, risk_metrics.correlation_matrix = self._calculate_correlation_metrics(
                returns_data
            )
            
            # Calculate liquidity risk
            risk_metrics.liquidity_score, risk_metrics.slippage_estimate = self._calculate_liquidity_risk(
                portfolio_state, market_data
            )
            
            # Calculate concentration risk
            risk_metrics.herfindahl_index, risk_metrics.position_concentration = self._calculate_concentration_risk(
                portfolio_state
            )
            
            # Calculate leverage risk
            risk_metrics.leverage_ratio = portfolio_state.leverage
            risk_metrics.liquidation_price = self._estimate_liquidation_price(
                portfolio_state, market_data
            )
            
            # Run stress tests if requested
            if include_stress_tests:
                risk_metrics.stress_test_losses = self._run_stress_tests(
                    portfolio_state, market_data
                )
                risk_metrics.worst_case_scenario, risk_metrics.worst_case_loss = self._get_worst_case_stress_test(
                    risk_metrics.stress_test_losses
                )
            
            # Calculate overall risk score and level
            risk_metrics.risk_score = self._calculate_risk_score(risk_metrics)
            risk_metrics.overall_risk_level = self._determine_risk_level(risk_metrics.risk_score)
            
            # Cache the result
            self.cache.set(cache_key, risk_metrics)
            
            # Add to history
            self.risk_history.append({
                'timestamp': datetime.now(),
                'metrics': risk_metrics,
                'portfolio_state': portfolio_state
            })
            
            self.logger.info(f"Risk analysis completed. Risk score: {risk_metrics.risk_score:.1f}")
            
        except Exception as e:
            self.logger.error(f"Error in risk analysis: {str(e)}")
            # Return minimal risk metrics in case of error
            risk_metrics.risk_score = 100  # Maximum risk on error
            risk_metrics.overall_risk_level = RiskLevel.EXTREME
        
        return risk_metrics
    
    def _calculate_var(self,
                      returns_data: Dict[str, pd.Series],
                      portfolio_value: float) -> Tuple[float, float]:
        """Calculate Value at Risk at 95% and 99% confidence"""
        
        # For now, use Bitcoin returns as proxy for portfolio
        # In reality, calculate portfolio returns
        if 'BTC' in returns_data:
            btc_returns = returns_data['BTC']
            
            var_95 = self.var_model.calculate_historical_var(
                btc_returns, confidence_level=0.95, portfolio_value=portfolio_value
            )
            var_99 = self.var_model.calculate_historical_var(
                btc_returns, confidence_level=0.99, portfolio_value=portfolio_value
            )
            
            return var_95, var_99
        
        return 0.0, 0.0
    
    def _calculate_cvar(self,
                       returns_data: Dict[str, pd.Series],
                       portfolio_value: float) -> Tuple[float, float]:
        """Calculate Conditional VaR at 95% and 99% confidence"""
        
        if 'BTC' in returns_data:
            btc_returns = returns_data['BTC']
            
            cvar_95 = self.es_model.calculate_historical_cvar(
                btc_returns, confidence_level=0.95, portfolio_value=portfolio_value
            )
            cvar_99 = self.es_model.calculate_historical_cvar(
                btc_returns, confidence_level=0.99, portfolio_value=portfolio_value
            )
            
            return cvar_95, cvar_99
        
        return 0.0, 0.0
    
    def _calculate_volatility(self,
                            returns_data: Dict[str, pd.Series]) -> Tuple[float, float]:
        """Calculate daily and annual volatility"""
        
        if 'BTC' in returns_data:
            btc_returns = returns_data['BTC']
            
            daily_vol = self.volatility_model.calculate_historical_volatility(
                btc_returns, annualize=False
            )
            annual_vol = self.volatility_model.calculate_historical_volatility(
                btc_returns, annualize=True
            )
            
            return daily_vol, annual_vol
        
        return 0.0, 0.0
    
    def _calculate_drawdown(self,
                          market_data: Dict[str, pd.DataFrame]) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        
        if 'BTC' in market_data:
            btc_prices = market_data['BTC']['close'] if 'close' in market_data['BTC'] else market_data['BTC'].iloc[:, 0]
            
            max_dd, duration, _, _ = self.drawdown_analyzer.calculate_max_drawdown(
                btc_prices, return_percent=True
            )
            
            return max_dd, duration
        
        return 0.0, 0
    
    def _calculate_risk_adjusted_ratios(self,
                                      returns_data: Dict[str, pd.Series]) -> Tuple[float, float]:
        """Calculate Sharpe and Sortino ratios"""
        
        if 'BTC' in returns_data:
            btc_returns = returns_data['BTC']
            
            # Sharpe ratio (assuming 0% risk-free rate for crypto)
            sharpe = self._calculate_sharpe_ratio(btc_returns, risk_free_rate=0.0)
            
            # Sortino ratio (downside deviation)
            sortino = self._calculate_sortino_ratio(btc_returns, risk_free_rate=0.0)
            
            return sharpe, sortino
        
        return 0.0, 0.0
    
    def _calculate_sharpe_ratio(self,
                              returns: pd.Series,
                              risk_free_rate: float = 0.0,
                              periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
        
        return sharpe
    
    def _calculate_sortino_ratio(self,
                               returns: pd.Series,
                               risk_free_rate: float = 0.0,
                               periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio (downside risk only)"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf  # No downside risk
        
        downside_deviation = downside_returns.std()
        
        if downside_deviation == 0:
            return np.inf
        
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation
        
        return sortino
    
    def _calculate_correlation_metrics(self,
                                     returns_data: Dict[str, pd.Series]) -> Tuple[float, Optional[pd.DataFrame]]:
        """Calculate correlation matrix and beta to Bitcoin"""
        
        if len(returns_data) < 2 or 'BTC' not in returns_data:
            return 1.0, None
        
        # Create DataFrame of returns
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 30:
            return 1.0, None
        
        # Calculate correlation matrix
        corr_matrix = self.correlation_analyzer.calculate_correlation_matrix(returns_df)
        
        # Calculate beta to Bitcoin for each asset
        btc_beta = 1.0  # Bitcoin's beta to itself is 1
        
        # For other assets, calculate beta to Bitcoin
        for asset in returns_df.columns:
            if asset != 'BTC':
                # This would calculate beta for each asset
                # For now, just return 1.0 for Bitcoin
                pass
        
        return btc_beta, corr_matrix
    
    def _calculate_liquidity_risk(self,
                                 portfolio_state: PortfolioState,
                                 market_data: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
        """Calculate liquidity score and slippage estimate"""
        
        # Simplified liquidity calculation
        # In reality, use order book data, volume data, etc.
        
        liquidity_score = 70.0  # Placeholder
        
        # Estimate slippage for portfolio
        total_slippage = 0.0
        for symbol, quantity in portfolio_state.positions.items():
            if symbol in market_data:
                # Get daily volume (placeholder)
                daily_volume = 1000.0  # BTC
                
                # Estimate slippage for this position
                position_slippage = self.liquidity_model.estimate_slippage(
                    order_size=quantity,
                    daily_volume=daily_volume,
                    volatility=0.02  # 2% daily volatility
                )
                
                total_slippage += position_slippage * abs(quantity)
        
        # Normalize by portfolio value
        if portfolio_state.portfolio_value > 0:
            slippage_estimate = total_slippage / portfolio_state.portfolio_value
        else:
            slippage_estimate = 0.0
        
        return liquidity_score, slippage_estimate
    
    def _calculate_concentration_risk(self,
                                    portfolio_state: PortfolioState) -> Tuple[float, float]:
        """Calculate concentration metrics"""
        
        # Calculate position values
        position_values = {}
        total_value = portfolio_state.portfolio_value
        
        # For now, use placeholder values
        # In reality, calculate actual position values from market prices
        
        # Calculate Herfindahl Index
        hhi = self.concentration_analyzer.calculate_herfindahl_index(
            position_values, total_value
        )
        
        # Calculate maximum position concentration
        if position_values:
            max_concentration = max(position_values.values()) / total_value if total_value > 0 else 0.0
        else:
            max_concentration = 0.0
        
        return hhi, max_concentration
    
    def _estimate_liquidation_price(self,
                                   portfolio_state: PortfolioState,
                                   market_data: Dict[str, pd.DataFrame]) -> Optional[float]:
        """Estimate liquidation price for leveraged positions"""
        
        if portfolio_state.leverage <= 1.0:
            return None
        
        # Simplified liquidation price calculation
        # In reality, this depends on exchange rules, collateral, etc.
        
        if 'BTC' in market_data:
            current_price = self._get_current_price(market_data['BTC'])
            
            # Assume liquidation at 80% of maintenance margin
            # This is a simplified calculation
            liquidation_multiplier = 0.8
            
            liquidation_price = current_price * (1 - (portfolio_state.leverage - 1) * liquidation_multiplier / portfolio_state.leverage)
            
            return max(liquidation_price, 0)
        
        return None
    
    def _run_stress_tests(self,
                         portfolio_state: PortfolioState,
                         market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Run stress tests and return losses by scenario"""
        
        stress_test_results = self.stress_test_engine.run_stress_test(
            portfolio_state, market_data
        )
        
        return stress_test_results
    
    def _get_worst_case_stress_test(self,
                                   stress_test_results: Dict[str, float]) -> Tuple[Optional[str], float]:
        """Get the worst-case scenario from stress tests"""
        
        if not stress_test_results:
            return None, 0.0
        
        worst_scenario = max(stress_test_results.items(), key=lambda x: abs(x[1]))[0]
        worst_loss = stress_test_results[worst_scenario]
        
        return worst_scenario, worst_loss
    
    def _calculate_risk_score(self, metrics: RiskMetrics) -> float:
        """Calculate overall risk score (0-100)"""
        
        weights = self.config.config['risk_score_weights']
        
        # Calculate component scores (0-100)
        component_scores = {}
        
        # Volatility score (higher volatility = higher risk)
        volatility_score = min(metrics.annual_volatility * 100, 100)
        component_scores['volatility'] = volatility_score
        
        # Drawdown score (higher drawdown = higher risk)
        drawdown_score = min(metrics.max_drawdown * 500, 100)  # 20% drawdown = 100 score
        component_scores['drawdown'] = drawdown_score
        
        # VaR score (higher VaR = higher risk)
        var_score = min(abs(metrics.value_at_risk_95) * 2000, 100)  # 5% VaR = 100 score
        component_scores['var'] = var_score
        
        # Liquidity score (lower liquidity = higher risk)
        liquidity_risk_score = 100 - metrics.liquidity_score
        component_scores['liquidity'] = liquidity_risk_score
        
        # Concentration score (higher concentration = higher risk)
        concentration_score = min(metrics.position_concentration * 333, 100)  # 30% concentration = 100 score
        component_scores['concentration'] = concentration_score
        
        # Leverage score (higher leverage = higher risk)
        leverage_score = min((metrics.leverage_ratio - 1) * 50, 100)  # 3x leverage = 100 score
        component_scores['leverage'] = leverage_score
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for component, weight in weights.items():
            if component in component_scores:
                total_score += component_scores[component] * weight
                total_weight += weight
        
        if total_weight > 0:
            risk_score = total_score / total_weight
        else:
            risk_score = 50.0  # Neutral score
        
        return min(max(risk_score, 0), 100)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score"""
        
        thresholds = self.config.config['risk_level_thresholds']
        
        if risk_score <= thresholds[RiskLevel.LOW]:
            return RiskLevel.LOW
        elif risk_score <= thresholds[RiskLevel.MODERATE]:
            return RiskLevel.MODERATE
        elif risk_score <= thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    def _prepare_returns_data(self,
                             market_data: Dict[str, pd.DataFrame],
                             historical_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.Series]:
        """Prepare returns data for analysis"""
        
        returns_data = {}
        
        # Use historical_data if provided, otherwise use market_data
        data_source = historical_data if historical_data is not None else market_data
        
        for symbol, data in data_source.items():
            if len(data) > 1:
                # Try to find price column
                price_col = None
                for col in ['close', 'price', 'last']:
                    if col in data.columns:
                        price_col = col
                        break
                
                if price_col is None and len(data.columns) > 0:
                    price_col = data.columns[0]
                
                if price_col:
                    prices = data[price_col]
                    returns = prices.pct_change().dropna()
                    returns_data[symbol] = returns
        
        return returns_data
    
    def _generate_cache_key(self,
                          portfolio_state: PortfolioState,
                          market_data: Dict[str, pd.DataFrame]) -> str:
        """Generate cache key for risk analysis"""
        
        # Create hash of relevant data
        data_to_hash = {
            'positions': str(sorted(portfolio_state.positions.items())),
            'portfolio_value': portfolio_state.portfolio_value,
            'leverage': portfolio_state.leverage,
            'timestamp': portfolio_state.timestamp.isoformat(),
            'market_data_keys': sorted(market_data.keys())
        }
        
        data_str = json.dumps(data_to_hash, sort_keys=True)
        cache_key = hashlib.md5(data_str.encode()).hexdigest()
        
        return f"risk_analysis_{cache_key}"
    
    def check_risk_limits(self,
                         risk_metrics: RiskMetrics,
                         limits: Optional[List[RiskLimit]] = None) -> Tuple[List[RiskLimit], List[str]]:
        """Check risk metrics against limits"""
        
        if limits is None:
            limits = self.config.limits
        
        violated_limits = []
        warnings = []
        
        for limit in limits:
            # Get current value for this metric
            current_value = getattr(risk_metrics, limit.metric.value, 0.0)
            
            # Check if limit is violated
            if limit.is_exceeded(current_value):
                violated_limits.append(limit)
                
                warning_msg = f"Risk limit violated: {limit.description}. "
                warning_msg += f"Current: {current_value:.3f}, Limit: {limit.limit_value:.3f}"
                warnings.append(warning_msg)
                
                self.logger.warning(warning_msg)
            
            # Check warning threshold
            elif limit.is_warning(current_value):
                warning_msg = f"Risk limit warning: {limit.description}. "
                warning_msg += f"Current: {current_value:.3f} ({(current_value/limit.limit_value*100):.1f}% of limit)"
                warnings.append(warning_msg)
                
                self.logger.info(warning_msg)
        
        return violated_limits, warnings
    
    def generate_risk_report(self,
                           portfolio_state: PortfolioState,
                           risk_metrics: RiskMetrics,
                           limits_violated: List[RiskLimit],
                           warnings: List[str],
                           analysis_period: str = "daily") -> RiskReport:
        """Generate comprehensive risk report"""
        
        report_id = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_metrics, limits_violated)
        
        # Create stress test summary
        stress_test_summary = {
            'worst_case_scenario': risk_metrics.worst_case_scenario,
            'worst_case_loss': risk_metrics.worst_case_loss,
            'total_scenarios_tested': len(risk_metrics.stress_test_losses)
        }
        
        # Create risk report
        report = RiskReport(
            report_id=report_id,
            portfolio_id=portfolio_state.metadata.get('portfolio_id', 'default'),
            timestamp=datetime.now(),
            metrics=risk_metrics,
            limits_violated=limits_violated,
            warnings=warnings,
            recommendations=recommendations,
            stress_test_summary=stress_test_summary,
            analysis_period=analysis_period
        )
        
        # Save report if configured
        if self.config.config['save_reports']:
            self._save_risk_report(report)
        
        return report
    
    def _generate_recommendations(self,
                                risk_metrics: RiskMetrics,
                                limits_violated: List[RiskLimit]) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        # Check overall risk level
        if risk_metrics.overall_risk_level == RiskLevel.EXTREME:
            recommendations.append("EXTREME RISK LEVEL: Consider reducing position sizes immediately")
        elif risk_metrics.overall_risk_level == RiskLevel.HIGH:
            recommendations.append("HIGH RISK LEVEL: Consider implementing risk reduction measures")
        
        # Check specific metrics
        if risk_metrics.max_drawdown > 0.15:  # 15% drawdown
            recommendations.append(f"Large drawdown detected ({risk_metrics.max_drawdown:.1%}). Consider adding stop-loss orders")
        
        if risk_metrics.annual_volatility > 0.80:  # 80% annual volatility
            recommendations.append(f"High volatility detected ({risk_metrics.annual_volatility:.1%}). Consider volatility targeting")
        
        if risk_metrics.leverage_ratio > 2.0:
            recommendations.append(f"High leverage detected ({risk_metrics.leverage_ratio:.1f}x). Consider deleveraging")
        
        if risk_metrics.position_concentration > 0.25:  # 25% concentration
            recommendations.append(f"High concentration detected ({risk_metrics.position_concentration:.1%}). Consider diversification")
        
        if risk_metrics.liquidity_score < 30:  # Low liquidity
            recommendations.append("Low liquidity detected. Consider reducing position sizes to avoid slippage")
        
        # If no specific issues, provide general recommendation
        if not recommendations and risk_metrics.risk_score < 30:
            recommendations.append("Risk level is low. Current risk management appears adequate")
        
        return recommendations
    
    def _save_risk_report(self, report: RiskReport):
        """Save risk report to file"""
        
        try:
            report_dir = Path(self.config.config['report_directory'])
            report_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{report.report_id}.json"
            filepath = report_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Risk report saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving risk report: {str(e)}")
    
    def analyze_order_risk(self,
                         order: Order,
                         current_positions: Dict[str, float],
                         market_data: Dict[str, pd.DataFrame],
                         portfolio_value: float) -> Dict[str, Any]:
        """Analyze risk impact of a new order"""
        
        analysis = {
            'order_id': order.order_id,
            'symbol': order.trading_pair,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': order.price,
            'risk_impact': 'neutral',
            'risk_score_change': 0.0,
            'warnings': [],
            'recommendations': []
        }
        
        # Simulate new portfolio state
        new_positions = current_positions.copy()
        symbol = order.trading_pair
        
        if order.side == OrderSide.BUY:
            new_positions[symbol] = new_positions.get(symbol, 0) + order.quantity
        else:
            new_positions[symbol] = new_positions.get(symbol, 0) - order.quantity
        
        # Remove zero positions
        new_positions = {k: v for k, v in new_positions.items() if abs(v) > 1e-10}
        
        # Calculate new concentration
        if symbol in market_data:
            current_price = self._get_current_price(market_data[symbol])
            position_value = abs(new_positions.get(symbol, 0)) * current_price
            
            concentration = position_value / portfolio_value if portfolio_value > 0 else 0.0
            
            # Check concentration limits
            max_concentration = self.config.config['max_position_concentration']
            if concentration > max_concentration:
                analysis['risk_impact'] = 'high'
                analysis['warnings'].append(
                    f"Order would increase {symbol} concentration to {concentration:.1%}, "
                    f"exceeding limit of {max_concentration:.1%}"
                )
                analysis['recommendations'].append(f"Reduce order size by at least {(concentration/max_concentration - 1)*100:.0f}%")
            elif concentration > max_concentration * 0.8:
                analysis['risk_impact'] = 'medium'
                analysis['warnings'].append(
                    f"Order would bring {symbol} concentration to {concentration:.1%}, "
                    f"approaching limit of {max_concentration:.1%}"
                )
        
        # Estimate market impact
        if symbol in market_data and 'volume' in market_data[symbol].columns:
            daily_volume = market_data[symbol]['volume'].iloc[-1] if len(market_data[symbol]) > 0 else 0
            
            if daily_volume > 0:
                volume_ratio = order.quantity / daily_volume
                
                if volume_ratio > 0.1:  # More than 10% of daily volume
                    analysis['risk_impact'] = 'high'
                    analysis['warnings'].append(
                        f"Order size ({order.quantity}) is {volume_ratio:.1%} of daily volume, "
                        f"may cause significant market impact"
                    )
                    analysis['recommendations'].append("Consider splitting order into smaller pieces")
                elif volume_ratio > 0.05:  # More than 5% of daily volume
                    if analysis['risk_impact'] == 'neutral':
                        analysis['risk_impact'] = 'medium'
                    analysis['warnings'].append(
                        f"Order size ({order.quantity}) is {volume_ratio:.1%} of daily volume, "
                        f"may cause noticeable market impact"
                    )
        
        return analysis
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of risk analyzer state"""
        
        return {
            'analysis_count': self.analysis_count,
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'risk_history_size': len(self.risk_history),
            'cache_size': len(self.cache._cache) if hasattr(self.cache, '_cache') else 0,
            'config': {
                'var_confidence': self.config.config['var_confidence_level'],
                'stress_test_scenarios': len(self.config.config['stress_test_scenarios']),
                'max_leverage': self.config.config['max_leverage']
            }
        }
    
    def _get_current_price(self, price_data: pd.DataFrame) -> float:
        """Get current price from market data"""
        if 'close' in price_data.columns:
            return price_data['close'].iloc[-1]
        elif 'price' in price_data.columns:
            return price_data['price'].iloc[-1]
        else:
            return price_data.iloc[-1, 0]

# ============ Factory Function ============
def create_risk_analyzer(config: Optional[RiskConfig] = None) -> RiskAnalyzer:
    """Factory function to create risk analyzer"""
    return RiskAnalyzer(config)

# ============ Main Execution ============
async def main():
    """Main execution for testing"""
    
    # Create risk analyzer
    analyzer = create_risk_analyzer()
    
    # Create test portfolio state
    portfolio_state = PortfolioState(
        timestamp=datetime.now(),
        positions={'BTC': 1.0, 'ETH': 10.0},
        cash=10000.0,
        portfolio_value=100000.0,
        leverage=1.5,
        metadata={'portfolio_id': 'test_portfolio'}
    )
    
    # Create test market data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    btc_prices = 50000 + np.random.randn(100).cumsum() * 1000
    eth_prices = 3000 + np.random.randn(100).cumsum() * 100
    
    market_data = {
        'BTC': pd.DataFrame({
            'close': btc_prices,
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates),
        'ETH': pd.DataFrame({
            'close': eth_prices,
            'volume': np.random.uniform(50000, 200000, 100)
        }, index=dates)
    }
    
    try:
        # Analyze portfolio risk
        risk_metrics = analyzer.analyze_portfolio_risk(
            portfolio_state=portfolio_state,
            market_data=market_data,
            include_stress_tests=True
        )
        
        print("\n=== Risk Analysis Results ===")
        print(f"Overall Risk Level: {risk_metrics.overall_risk_level.value}")
        print(f"Risk Score: {risk_metrics.risk_score:.1f}/100")
        print(f"\nKey Metrics:")
        print(f"  95% VaR: {risk_metrics.value_at_risk_95:.2f} USD")
        print(f"  99% VaR: {risk_metrics.value_at_risk_99:.2f} USD")
        print(f"  Max Drawdown: {risk_metrics.max_drawdown:.1%}")
        print(f"  Annual Volatility: {risk_metrics.annual_volatility:.1%}")
        print(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
        print(f"  Leverage Ratio: {risk_metrics.leverage_ratio:.2f}x")
        print(f"  Liquidity Score: {risk_metrics.liquidity_score:.1f}/100")
        
        # Check risk limits
        violated_limits, warnings = analyzer.check_risk_limits(risk_metrics)
        
        if violated_limits:
            print(f"\n⚠️  Risk Limits Violated: {len(violated_limits)}")
            for limit in violated_limits:
                print(f"  - {limit.description}")
        
        if warnings:
            print(f"\n⚠️  Warnings: {len(warnings)}")
            for warning in warnings[:3]:  # Show first 3 warnings
                print(f"  - {warning}")
        
        # Generate risk report
        report = analyzer.generate_risk_report(
            portfolio_state=portfolio_state,
            risk_metrics=risk_metrics,
            limits_violated=violated_limits,
            warnings=warnings
        )
        
        print(f"\n📊 Risk Report Generated: {report.report_id}")
        
        # Get analyzer summary
        summary = analyzer.get_risk_summary()
        print(f"\nAnalyzer Summary:")
        print(f"  Total Analyses: {summary['analysis_count']}")
        print(f"  Risk History: {summary['risk_history_size']} entries")
        
    except Exception as e:
        print(f"Error in risk analysis: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())