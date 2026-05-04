"""
Portfolio Optimizer module for Bitcoin trading AI.
Advanced portfolio optimization including mean-variance optimization, 
risk parity, Black-Litterman model, and hierarchical risk parity.
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
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
import cvxpy as cp
import json
from pathlib import Path
import hashlib
import asyncio
from collections import deque, defaultdict
import pickle
from functools import lru_cache
import itertools
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

# Import project modules
from config.settings import TradingSettings, PortfolioSettings, AppConstants
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.risk_management.risk_analyzer import RiskAnalyzer, RiskMetrics, PortfolioState
from core.utils.cache import Cache

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Enums and Types ============
class OptimizationObjective(str, Enum):
    """Portfolio optimization objectives"""
    MAX_SHARPE = "max_sharpe"              # Maximize Sharpe ratio
    MIN_VARIANCE = "min_variance"          # Minimize portfolio variance
    MAX_RETURN = "max_return"              # Maximize expected return
    RISK_PARITY = "risk_parity"            # Equal risk contribution
    MAX_DIVERSIFICATION = "max_diversification"  # Maximize diversification ratio
    MAX_UTILITY = "max_utility"            # Maximize utility function
    CUSTOM = "custom"                      # Custom objective

class OptimizationMethod(str, Enum):
    """Optimization methods"""
    MEAN_VARIANCE = "mean_variance"        # Markowitz mean-variance
    BLACK_LITTERMAN = "black_litterman"    # Black-Litterman model
    HRP = "hrp"                            # Hierarchical Risk Parity
    CLA = "cla"                            # Critical Line Algorithm
    MONTE_CARLO = "monte_carlo"            # Monte Carlo simulation
    GENETIC_ALGORITHM = "genetic_algorithm" # Genetic algorithm

class ConstraintType(str, Enum):
    """Constraint types"""
    BUDGET = "budget"                      # Sum of weights = 1
    LONG_ONLY = "long_only"                # No short positions
    LEVERAGE = "leverage"                  # Maximum leverage
    CONCENTRATION = "concentration"        # Position concentration limits
    SECTOR = "sector"                      # Sector exposure limits
    TURNOVER = "turnover"                  # Maximum turnover
    CARDINALITY = "cardinality"            # Maximum number of assets

# ============ Data Structures ============
@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""
    
    # Basic allocation
    weights: Dict[str, float]              # Asset -> weight (0-1)
    expected_return: float                 # Expected portfolio return
    expected_volatility: float             # Expected portfolio volatility
    sharpe_ratio: Optional[float] = None   # Sharpe ratio (if applicable)
    
    # Risk metrics
    marginal_risk_contributions: Dict[str, float] = field(default_factory=dict)
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    diversification_ratio: Optional[float] = None
    
    # Optimization details
    optimization_objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE
    optimization_method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    constraints: List[str] = field(default_factory=list)
    
    # Performance attribution
    return_contributions: Dict[str, float] = field(default_factory=dict)
    tracking_error: Optional[float] = None  # vs benchmark
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    calculation_time: float = 0.0          # Seconds
    
    def __post_init__(self):
        """Validate allocation"""
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Portfolio weights must sum to 1, got {total_weight}")
        
        for asset, weight in self.weights.items():
            if weight < -1e-6:  # Allow small negative for numerical precision
                raise ValueError(f"Negative weight for {asset}: {weight}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'weights': self.weights,
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'marginal_risk_contributions': self.marginal_risk_contributions,
            'risk_contributions': self.risk_contributions,
            'diversification_ratio': self.diversification_ratio,
            'optimization_objective': self.optimization_objective.value,
            'optimization_method': self.optimization_method.value,
            'constraints': self.constraints,
            'return_contributions': self.return_contributions,
            'tracking_error': self.tracking_error,
            'timestamp': self.timestamp.isoformat(),
            'calculation_time': self.calculation_time
        }
    
    @property
    def concentrated_assets(self) -> List[Tuple[str, float]]:
        """Get assets with weight > 5%"""
        return [(asset, weight) for asset, weight in self.weights.items() if weight > 0.05]
    
    @property
    def effective_number_of_assets(self) -> float:
        """Calculate effective number of assets (diversification measure)"""
        weights = np.array(list(self.weights.values()))
        weights = weights[weights > 0]  # Only positive weights
        
        if len(weights) == 0:
            return 0.0
        
        # Herfindahl-based measure
        hhi = np.sum(weights ** 2)
        return 1.0 / hhi if hhi > 0 else 0.0

@dataclass
class OptimizationInputs:
    """Inputs for portfolio optimization"""
    
    # Assets and returns
    assets: List[str]                      # Asset symbols
    returns: pd.DataFrame                  # Historical returns (assets x time)
    
    # Expected returns (optional)
    expected_returns: Optional[pd.Series] = None  # Prior expected returns
    
    # Covariance matrix
    covariance_matrix: Optional[pd.DataFrame] = None
    
    # Market data
    market_prices: Optional[Dict[str, pd.DataFrame]] = None
    risk_free_rate: float = 0.0            # Annual risk-free rate
    
    # Views for Black-Litterman
    views: Optional[pd.Series] = None      # Absolute or relative views
    view_confidences: Optional[pd.Series] = None  # Confidence in views (0-1)
    
    # Benchmark
    benchmark_returns: Optional[pd.Series] = None
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate inputs"""
        if len(self.assets) != len(self.returns.columns):
            raise ValueError("Number of assets doesn't match returns columns")
        
        if self.expected_returns is not None:
            if len(self.expected_returns) != len(self.assets):
                raise ValueError("Expected returns length doesn't match assets")
        
        if self.covariance_matrix is not None:
            if self.covariance_matrix.shape != (len(self.assets), len(self.assets)):
                raise ValueError("Covariance matrix shape doesn't match assets")

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    
    # Weight constraints
    min_weight: Union[float, Dict[str, float]] = 0.0      # Minimum weight per asset
    max_weight: Union[float, Dict[str, float]] = 1.0      # Maximum weight per asset
    
    # Portfolio constraints
    leverage_limit: float = 1.0                           # Maximum leverage (sum abs weights)
    turnover_limit: Optional[float] = None                # Maximum turnover
    cardinality_limit: Optional[int] = None               # Maximum number of assets
    
    # Risk constraints
    max_volatility: Optional[float] = None               # Maximum portfolio volatility
    max_var: Optional[float] = None                      # Maximum Value at Risk
    max_drawdown: Optional[float] = None                 # Maximum expected drawdown
    
    # Sector/group constraints
    group_limits: Dict[str, float] = field(default_factory=dict)  # Group -> max weight
    asset_groups: Dict[str, List[str]] = field(default_factory=dict)  # Group -> assets
    
    # Custom constraints
    custom_constraints: List[Callable] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'leverage_limit': self.leverage_limit,
            'turnover_limit': self.turnover_limit,
            'cardinality_limit': self.cardinality_limit,
            'max_volatility': self.max_volatility,
            'max_var': self.max_var,
            'max_drawdown': self.max_drawdown,
            'group_limits': self.group_limits,
            'asset_groups': self.asset_groups
        }

@dataclass
class EfficientFrontier:
    """Efficient frontier points"""
    
    returns: np.ndarray                    # Expected returns
    volatilities: np.ndarray               # Portfolio volatilities
    sharpe_ratios: np.ndarray              # Sharpe ratios
    weights: np.ndarray                    # Weight matrix (points x assets)
    
    # Special points
    max_sharpe_weights: np.ndarray         # Max Sharpe portfolio weights
    min_variance_weights: np.ndarray       # Min variance portfolio weights
    
    # Metadata
    risk_free_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'returns': self.returns.tolist(),
            'volatilities': self.volatilities.tolist(),
            'sharpe_ratios': self.sharpe_ratios.tolist(),
            'weights': self.weights.tolist(),
            'max_sharpe_weights': self.max_sharpe_weights.tolist(),
            'min_variance_weights': self.min_variance_weights.tolist(),
            'risk_free_rate': self.risk_free_rate,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class RebalancingDecision:
    """Portfolio rebalancing decision"""
    
    # Current vs target allocation
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    
    # Required trades
    trades: Dict[str, float]               # Asset -> trade amount (positive = buy)
    trade_costs: Dict[str, float]          # Estimated trading costs per asset
    
    # Rebalancing metrics
    tracking_error_reduction: float        # Expected reduction in tracking error
    turnover: float                        # Portfolio turnover (%)
    implementation_shortfall: float        # Estimated implementation cost
    
    # Decision
    should_rebalance: bool                 # Whether to execute rebalancing
    rebalancing_score: float               # 0-100 score for rebalancing urgency
    reason: str = ""                       # Reason for decision
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'current_weights': self.current_weights,
            'target_weights': self.target_weights,
            'trades': self.trades,
            'trade_costs': self.trade_costs,
            'tracking_error_reduction': self.tracking_error_reduction,
            'turnover': self.turnover,
            'implementation_shortfall': self.implementation_shortfall,
            'should_rebalance': self.should_rebalance,
            'rebalancing_score': self.rebalancing_score,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat()
        }

# ============ Configuration ============
class PortfolioOptimizerConfig:
    """Portfolio optimizer configuration"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config = config_dict or self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default portfolio optimizer configuration"""
        return {
            # Optimization settings
            'default_objective': OptimizationObjective.MAX_SHARPE,
            'default_method': OptimizationMethod.MEAN_VARIANCE,
            
            # Return estimation
            'return_estimation_method': 'historical',  # historical, exponential, shrinkage
            'return_lookback_periods': 252,           # Trading days
            'shrinkage_intensity': 0.5,               # Shrinkage intensity for covariance
            
            # Covariance estimation
            'covariance_estimation_method': 'ledoit_wolf',  # sample, ledoit_wolf, oracle_approx
            'covariance_lookback_periods': 252,
            
            # Black-Litterman parameters
            'tau': 0.05,                              # Scaling factor for views
            'view_confidence_method': 'idzorek',      # idzorek, relative
            'market_cap_weights': True,               # Use market cap for equilibrium
            
            # HRP parameters
            'hrp_linkage_method': 'ward',
            'hrp_metric': 'euclidean',
            
            # Constraints
            'default_min_weight': 0.0,
            'default_max_weight': 0.3,                # 30% max per asset
            'max_leverage': 1.0,                      # No leverage by default
            'max_turnover': 0.2,                      # 20% max turnover
            'max_cardinality': 20,                    # Max 20 assets
            
            # Risk constraints
            'max_volatility': 0.5,                    # 50% annual volatility
            'max_var_confidence': 0.95,
            'max_var_limit': 0.1,                     # 10% daily VaR
            
            # Rebalancing
            'rebalancing_frequency': 'monthly',       # daily, weekly, monthly
            'rebalancing_threshold': 0.05,            # 5% deviation triggers rebalancing
            'min_rebalancing_size': 0.01,             # 1% minimum trade size
            
            # Transaction costs
            'transaction_cost_model': 'proportional', # proportional, fixed, tiered
            'transaction_cost_rate': 0.001,           # 0.1% per trade
            'fixed_transaction_cost': 0.0,
            
            # Numerical optimization
            'optimization_tolerance': 1e-8,
            'max_iterations': 1000,
            'random_seed': 42,
            
            # Performance
            'cache_enabled': True,
            'cache_ttl_seconds': 300,                 # 5 minutes
            'parallel_processing': True,
            'max_workers': 4,
            
            # Reporting
            'save_optimizations': True,
            'optimization_report_path': 'data/portfolio_optimizations/',
            'generate_efficient_frontier': True,
            'frontier_points': 50
        }

# ============ Return and Covariance Estimation ============
class ReturnEstimator:
    """Estimate expected returns"""
    
    def __init__(self, config: PortfolioOptimizerConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def estimate_historical_returns(self,
                                  returns: pd.DataFrame,
                                  method: str = 'mean',
                                  lookback: Optional[int] = None) -> pd.Series:
        """Estimate expected returns from historical data"""
        
        if lookback is not None:
            returns = returns.iloc[-lookback:]
        
        if method == 'mean':
            expected_returns = returns.mean()
        elif method == 'median':
            expected_returns = returns.median()
        elif method == 'exponential':
            # Exponentially weighted moving average
            expected_returns = returns.ewm(span=lookback or len(returns)).mean().iloc[-1]
        elif method == 'shrinkage':
            # Shrink towards grand mean
            grand_mean = returns.mean().mean()
            asset_means = returns.mean()
            shrinkage_factor = self.config.config.get('shrinkage_intensity', 0.5)
            expected_returns = shrinkage_factor * grand_mean + (1 - shrinkage_factor) * asset_means
        else:
            raise ValueError(f"Unknown return estimation method: {method}")
        
        return expected_returns
    
    def estimate_capm_returns(self,
                            returns: pd.DataFrame,
                            market_returns: pd.Series,
                            risk_free_rate: float = 0.0) -> pd.Series:
        """Estimate returns using CAPM"""
        
        expected_returns = {}
        
        for asset in returns.columns:
            asset_returns = returns[asset]
            
            # Calculate beta
            covariance = asset_returns.cov(market_returns)
            market_variance = market_returns.var()
            
            if market_variance == 0:
                beta = 1.0
            else:
                beta = covariance / market_variance
            
            # CAPM formula: E(R) = Rf + beta * (E(Rm) - Rf)
            market_premium = market_returns.mean() - risk_free_rate
            expected_return = risk_free_rate + beta * market_premium
            
            expected_returns[asset] = expected_return
        
        return pd.Series(expected_returns)
    
    def estimate_black_litterman_returns(self,
                                       historical_returns: pd.DataFrame,
                                       market_weights: pd.Series,
                                       views: pd.Series,
                                       view_confidences: Optional[pd.Series] = None,
                                       risk_aversion: float = 2.5,
                                       tau: float = 0.05) -> pd.Series:
        """Estimate returns using Black-Litterman model"""
        
        # 1. Calculate equilibrium returns (prior)
        sigma = historical_returns.cov()  # Covariance matrix
        n = len(market_weights)
        
        # Implied equilibrium returns: Pi = delta * Sigma * w
        equilibrium_returns = risk_aversion * sigma @ market_weights
        
        # 2. Process views
        # Create pick matrix P and view vector Q
        # For simplicity, assume absolute views on specific assets
        P = np.zeros((len(views), n))
        Q = np.zeros(len(views))
        
        # Map views to assets
        asset_to_idx = {asset: i for i, asset in enumerate(historical_returns.columns)}
        
        for i, (asset, view_return) in enumerate(views.items()):
            if asset in asset_to_idx:
                idx = asset_to_idx[asset]
                P[i, idx] = 1.0
                Q[i] = view_return
        
        # 3. Calculate uncertainty of views (Omega)
        if view_confidences is None:
            # Use Idzorek method or simple diagonal
            Omega = np.diag(np.diag(P @ (tau * sigma) @ P.T))
        else:
            # Use provided confidences
            Omega = np.diag(1.0 / view_confidences.values)
        
        # 4. Combine prior and views
        # Posterior returns: E[R] = [(tau*Sigma)^{-1} + P'*Omega^{-1}*P]^{-1} * [(tau*Sigma)^{-1}*Pi + P'*Omega^{-1}*Q]
        
        tau_sigma_inv = np.linalg.inv(tau * sigma)
        omega_inv = np.linalg.inv(Omega)
        
        # Calculate posterior mean
        M = tau_sigma_inv + P.T @ omega_inv @ P
        V = tau_sigma_inv @ equilibrium_returns + P.T @ omega_inv @ Q
        
        posterior_returns = np.linalg.solve(M, V)
        
        return pd.Series(posterior_returns, index=historical_returns.columns)

class CovarianceEstimator:
    """Estimate covariance matrix"""
    
    def __init__(self, config: PortfolioOptimizerConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def estimate_sample_covariance(self,
                                 returns: pd.DataFrame,
                                 frequency: str = 'daily') -> pd.DataFrame:
        """Estimate sample covariance matrix"""
        
        if frequency == 'daily':
            ann_factor = 252
        elif frequency == 'weekly':
            ann_factor = 52
        elif frequency == 'monthly':
            ann_factor = 12
        else:
            ann_factor = 1
        
        # Annualize covariance
        cov_matrix = returns.cov() * ann_factor
        
        return cov_matrix
    
    def estimate_shrinkage_covariance(self,
                                    returns: pd.DataFrame,
                                    method: str = 'ledoit_wolf') -> pd.DataFrame:
        """Estimate covariance matrix using shrinkage methods"""
        
        try:
            from sklearn.covariance import LedoitWolf, OAS
            
            if method == 'ledoit_wolf':
                estimator = LedoitWolf()
            elif method == 'oas':
                estimator = OAS()
            else:
                raise ValueError(f"Unknown shrinkage method: {method}")
            
            estimator.fit(returns.values)
            cov_matrix = pd.DataFrame(
                estimator.covariance_,
                index=returns.columns,
                columns=returns.columns
            )
            
            return cov_matrix
            
        except ImportError:
            self.logger.warning("scikit-learn not installed, using sample covariance")
            return self.estimate_sample_covariance(returns)
    
    def estimate_exponential_covariance(self,
                                      returns: pd.DataFrame,
                                      span: int = 60) -> pd.DataFrame:
        """Estimate covariance using exponential weighting"""
        
        # Exponentially weighted covariance
        cov_matrix = returns.ewm(span=span).cov().iloc[-len(returns.columns):]
        
        return cov_matrix
    
    def estimate_robust_covariance(self,
                                 returns: pd.DataFrame,
                                 method: str = 'minimum_covariance_determinant') -> pd.DataFrame:
        """Estimate robust covariance matrix"""
        
        try:
            from sklearn.covariance import MinCovDet
            
            if method == 'minimum_covariance_determinant':
                estimator = MinCovDet()
            else:
                raise ValueError(f"Unknown robust method: {method}")
            
            estimator.fit(returns.values)
            cov_matrix = pd.DataFrame(
                estimator.covariance_,
                index=returns.columns,
                columns=returns.columns
            )
            
            return cov_matrix
            
        except ImportError:
            self.logger.warning("scikit-learn not installed, using sample covariance")
            return self.estimate_sample_covariance(returns)
    
    def denoise_covariance(self,
                          cov_matrix: pd.DataFrame,
                          method: str = 'random_matrix') -> pd.DataFrame:
        """Denoise covariance matrix"""
        
        if method == 'random_matrix':
            # Random matrix theory denoising
            n, t = cov_matrix.shape[0], 100  # t = time periods
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Theoretical maximum eigenvalue for random matrix
            q = n / t
            lambda_max = (1 + np.sqrt(q)) ** 2
            
            # Keep eigenvalues above theoretical maximum, shrink others
            denoised_eigenvalues = np.zeros_like(eigenvalues)
            for i, eigenvalue in enumerate(eigenvalues):
                if eigenvalue > lambda_max:
                    denoised_eigenvalues[i] = eigenvalue
                else:
                    # Shrink towards average of small eigenvalues
                    denoised_eigenvalues[i] = np.mean(eigenvalues[eigenvalues <= lambda_max])
            
            # Reconstruct covariance matrix
            denoised_cov = eigenvectors @ np.diag(denoised_eigenvalues) @ eigenvectors.T
            
            return pd.DataFrame(
                denoised_cov,
                index=cov_matrix.index,
                columns=cov_matrix.columns
            )
        
        else:
            raise ValueError(f"Unknown denoising method: {method}")

# ============ Optimization Models ============
class MeanVarianceOptimizer:
    """Mean-Variance Optimization (Markowitz)"""
    
    def __init__(self, config: PortfolioOptimizerConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def optimize(self,
                expected_returns: pd.Series,
                covariance_matrix: pd.DataFrame,
                constraints: OptimizationConstraints,
                objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
                risk_free_rate: float = 0.0) -> PortfolioAllocation:
        """Perform mean-variance optimization"""
        
        assets = expected_returns.index.tolist()
        n_assets = len(assets)
        
        # Convert to numpy arrays
        mu = expected_returns.values
        Sigma = covariance_matrix.values
        
        # Define optimization variables
        w = cp.Variable(n_assets)
        
        # Define objective
        if objective == OptimizationObjective.MAX_SHARPE:
            # Maximize Sharpe ratio: (mu'w - rf) / sqrt(w'Σw)
            portfolio_return = mu.T @ w - risk_free_rate
            portfolio_volatility = cp.quad_form(w, Sigma)
            
            # Use epigraph form to handle Sharpe ratio
            t = cp.Variable()
            constraints_cp = self._create_cvxpy_constraints(w, constraints, assets)
            constraints_cp += [cp.quad_form(w, Sigma) <= t**2, t >= 0]
            
            problem = cp.Problem(
                cp.Maximize(mu.T @ w - risk_free_rate * t),
                constraints_cp
            )
            
        elif objective == OptimizationObjective.MIN_VARIANCE:
            # Minimize variance: w'Σw
            portfolio_variance = cp.quad_form(w, Sigma)
            
            constraints_cp = self._create_cvxpy_constraints(w, constraints, assets)
            problem = cp.Problem(cp.Minimize(portfolio_variance), constraints_cp)
            
        elif objective == OptimizationObjective.MAX_RETURN:
            # Maximize return: μ'w
            portfolio_return = mu.T @ w
            
            constraints_cp = self._create_cvxpy_constraints(w, constraints, assets)
            problem = cp.Problem(cp.Maximize(portfolio_return), constraints_cp)
            
        elif objective == OptimizationObjective.MAX_UTILITY:
            # Maximize utility: μ'w - (risk_aversion/2) * w'Σw
            risk_aversion = self.config.config.get('risk_aversion', 2.5)
            utility = mu.T @ w - (risk_aversion / 2) * cp.quad_form(w, Sigma)
            
            constraints_cp = self._create_cvxpy_constraints(w, constraints, assets)
            problem = cp.Problem(cp.Maximize(utility), constraints_cp)
            
        else:
            raise ValueError(f"Unsupported objective: {objective}")
        
        # Solve optimization problem
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise RuntimeError(f"Optimization failed with status: {problem.status}")
            
            # Extract solution
            weights = w.value
            weights = np.maximum(weights, 0)  # Ensure non-negative
            weights = weights / np.sum(weights)  # Re-normalize
            
            # Calculate portfolio metrics
            portfolio_return = mu @ weights
            portfolio_volatility = np.sqrt(weights.T @ Sigma @ weights)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
            
            # Calculate risk contributions
            marginal_risk = Sigma @ weights
            total_risk = portfolio_volatility ** 2
            risk_contributions = weights * marginal_risk / total_risk if total_risk > 0 else np.zeros_like(weights)
            
            # Create allocation
            allocation = PortfolioAllocation(
                weights=dict(zip(assets, weights)),
                expected_return=portfolio_return,
                expected_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                marginal_risk_contributions=dict(zip(assets, marginal_risk)),
                risk_contributions=dict(zip(assets, risk_contributions)),
                optimization_objective=objective,
                optimization_method=OptimizationMethod.MEAN_VARIANCE
            )
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Mean-variance optimization failed: {str(e)}")
            raise
    
    def _create_cvxpy_constraints(self,
                                 w: cp.Variable,
                                 constraints: OptimizationConstraints,
                                 assets: List[str]) -> List:
        """Create CVXPY constraints"""
        
        n_assets = len(assets)
        constraints_cp = []
        
        # Budget constraint: sum of weights = 1
        constraints_cp.append(cp.sum(w) == 1)
        
        # Long-only constraint
        constraints_cp.append(w >= 0)
        
        # Minimum and maximum weight constraints
        if isinstance(constraints.min_weight, dict):
            for i, asset in enumerate(assets):
                if asset in constraints.min_weight:
                    constraints_cp.append(w[i] >= constraints.min_weight[asset])
        elif constraints.min_weight > 0:
            constraints_cp.append(w >= constraints.min_weight)
        
        if isinstance(constraints.max_weight, dict):
            for i, asset in enumerate(assets):
                if asset in constraints.max_weight:
                    constraints_cp.append(w[i] <= constraints.max_weight[asset])
        elif constraints.max_weight < 1:
            constraints_cp.append(w <= constraints.max_weight)
        
        # Leverage constraint
        if constraints.leverage_limit is not None:
            constraints_cp.append(cp.norm(w, 1) <= constraints.leverage_limit)
        
        # Group constraints
        if constraints.asset_groups and constraints.group_limits:
            for group, limit in constraints.group_limits.items():
                if group in constraints.asset_groups:
                    group_indices = [i for i, asset in enumerate(assets) 
                                   if asset in constraints.asset_groups[group]]
                    if group_indices:
                        group_sum = sum(w[i] for i in group_indices)
                        constraints_cp.append(group_sum <= limit)
        
        # Volatility constraint
        if constraints.max_volatility is not None:
            # This would require the covariance matrix, handled separately
            pass
        
        return constraints_cp
    
    def calculate_efficient_frontier(self,
                                   expected_returns: pd.Series,
                                   covariance_matrix: pd.DataFrame,
                                   constraints: OptimizationConstraints,
                                   risk_free_rate: float = 0.0,
                                   n_points: int = 50) -> EfficientFrontier:
        """Calculate efficient frontier"""
        
        assets = expected_returns.index.tolist()
        n_assets = len(assets)
        
        mu = expected_returns.values
        Sigma = covariance_matrix.values
        
        # Find minimum variance portfolio
        w = cp.Variable(n_assets)
        variance = cp.quad_form(w, Sigma)
        
        min_var_constraints = self._create_cvxpy_constraints(w, constraints, assets)
        min_var_problem = cp.Problem(cp.Minimize(variance), min_var_constraints)
        min_var_problem.solve()
        
        if min_var_problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError("Minimum variance optimization failed")
        
        min_var_weights = w.value
        min_var_return = mu @ min_var_weights
        min_var_vol = np.sqrt(min_var_weights.T @ Sigma @ min_var_weights)
        
        # Find maximum return portfolio
        max_return_constraints = self._create_cvxpy_constraints(w, constraints, assets)
        max_return_problem = cp.Problem(cp.Maximize(mu.T @ w), max_return_constraints)
        max_return_problem.solve()
        
        if max_return_problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError("Maximum return optimization failed")
        
        max_return_weights = w.value
        max_return = mu @ max_return_weights
        max_return_vol = np.sqrt(max_return_weights.T @ Sigma @ max_return_weights)
        
        # Generate points along efficient frontier
        target_returns = np.linspace(min_var_return, max_return, n_points)
        
        frontier_returns = []
        frontier_volatilities = []
        frontier_sharpe_ratios = []
        frontier_weights = []
        
        for target_return in target_returns:
            w = cp.Variable(n_assets)
            variance = cp.quad_form(w, Sigma)
            
            frontier_constraints = self._create_cvxpy_constraints(w, constraints, assets)
            frontier_constraints.append(mu.T @ w == target_return)
            
            frontier_problem = cp.Problem(cp.Minimize(variance), frontier_constraints)
            frontier_problem.solve()
            
            if frontier_problem.status in ["optimal", "optimal_inaccurate"]:
                weights = w.value
                vol = np.sqrt(variance.value)
                sharpe = (target_return - risk_free_rate) / vol if vol > 0 else 0.0
                
                frontier_returns.append(target_return)
                frontier_volatilities.append(vol)
                frontier_sharpe_ratios.append(sharpe)
                frontier_weights.append(weights)
        
        # Find maximum Sharpe portfolio
        max_sharpe_idx = np.argmax(frontier_sharpe_ratios)
        max_sharpe_weights = frontier_weights[max_sharpe_idx]
        
        return EfficientFrontier(
            returns=np.array(frontier_returns),
            volatilities=np.array(frontier_volatilities),
            sharpe_ratios=np.array(frontier_sharpe_ratios),
            weights=np.array(frontier_weights),
            max_sharpe_weights=max_sharpe_weights,
            min_variance_weights=min_var_weights,
            risk_free_rate=risk_free_rate
        )

class RiskParityOptimizer:
    """Risk Parity optimization (equal risk contribution)"""
    
    def __init__(self, config: PortfolioOptimizerConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def optimize(self,
                covariance_matrix: pd.DataFrame,
                constraints: OptimizationConstraints) -> PortfolioAllocation:
        """Perform risk parity optimization"""
        
        assets = covariance_matrix.index.tolist()
        n_assets = len(assets)
        
        Sigma = covariance_matrix.values
        
        # Define objective: minimize sum of squared differences in risk contributions
        w = cp.Variable(n_assets, nonneg=True)
        
        # Budget constraint
        constraints_cp = [cp.sum(w) == 1]
        
        # Additional constraints
        constraints_cp = self._add_constraints(w, constraints_cp, constraints, assets)
        
        # Risk contributions: RC_i = w_i * (Σw)_i / sqrt(w'Σw)
        # We'll use a simplified objective: minimize variance with log barrier for diversification
        portfolio_variance = cp.quad_form(w, Sigma)
        
        # Add log barrier for diversification
        log_weights = cp.sum(cp.entr(w))  # -sum(w * log(w)), encourages diversification
        
        # Combined objective
        objective = cp.Minimize(portfolio_variance - 0.1 * log_weights)
        
        problem = cp.Problem(objective, constraints_cp)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise RuntimeError(f"Risk parity optimization failed: {problem.status}")
            
            weights = w.value
            weights = weights / np.sum(weights)  # Ensure normalization
            
            # Calculate metrics
            portfolio_volatility = np.sqrt(weights.T @ Sigma @ weights)
            marginal_risk = Sigma @ weights
            risk_contributions = weights * marginal_risk / (portfolio_volatility ** 2)
            
            # Calculate diversification ratio
            weighted_vol = np.sqrt(np.diag(Sigma))
            avg_vol = np.sum(weights * weighted_vol)
            diversification_ratio = avg_vol / portfolio_volatility if portfolio_volatility > 0 else 1.0
            
            allocation = PortfolioAllocation(
                weights=dict(zip(assets, weights)),
                expected_return=0.0,  # Risk parity doesn't optimize for return
                expected_volatility=portfolio_volatility,
                diversification_ratio=diversification_ratio,
                marginal_risk_contributions=dict(zip(assets, marginal_risk)),
                risk_contributions=dict(zip(assets, risk_contributions)),
                optimization_objective=OptimizationObjective.RISK_PARITY,
                optimization_method=OptimizationMethod.MEAN_VARIANCE
            )
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Risk parity optimization failed: {str(e)}")
            raise
    
    def _add_constraints(self,
                        w: cp.Variable,
                        constraints_cp: List,
                        constraints: OptimizationConstraints,
                        assets: List[str]) -> List:
        """Add constraints to optimization problem"""
        
        # Weight constraints
        if isinstance(constraints.min_weight, dict):
            for i, asset in enumerate(assets):
                if asset in constraints.min_weight:
                    constraints_cp.append(w[i] >= constraints.min_weight[asset])
        elif constraints.min_weight > 0:
            constraints_cp.append(w >= constraints.min_weight)
        
        if isinstance(constraints.max_weight, dict):
            for i, asset in enumerate(assets):
                if asset in constraints.max_weight:
                    constraints_cp.append(w[i] <= constraints.max_weight[asset])
        elif constraints.max_weight < 1:
            constraints_cp.append(w <= constraints.max_weight)
        
        # Group constraints
        if constraints.asset_groups and constraints.group_limits:
            for group, limit in constraints.group_limits.items():
                if group in constraints.asset_groups:
                    group_indices = [i for i, asset in enumerate(assets) 
                                   if asset in constraints.asset_groups[group]]
                    if group_indices:
                        group_sum = sum(w[i] for i in group_indices)
                        constraints_cp.append(group_sum <= limit)
        
        return constraints_cp

class HierarchicalRiskParityOptimizer:
    """Hierarchical Risk Parity optimization"""
    
    def __init__(self, config: PortfolioOptimizerConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def optimize(self,
                returns: pd.DataFrame,
                constraints: Optional[OptimizationConstraints] = None) -> PortfolioAllocation:
        """Perform Hierarchical Risk Parity optimization"""
        
        # 1. Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # 2. Calculate distance matrix
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)
        
        # 3. Perform hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        
        # 4. Quasi-diagonalization
        sort_order = self._quasi_diagonalize(linkage_matrix)
        
        # 5. Recursive bisection
        weights = self._recursive_bisection(
            returns.iloc[:, sort_order], 
            constraints
        )
        
        # Reorder weights to original asset order
        final_weights = np.zeros(len(returns.columns))
        for i, w in enumerate(weights):
            final_weights[sort_order[i]] = w
        
        # Calculate portfolio metrics
        expected_returns = returns.mean()
        covariance_matrix = returns.cov()
        
        portfolio_return = expected_returns @ final_weights
        portfolio_volatility = np.sqrt(final_weights.T @ covariance_matrix.values @ final_weights)
        
        allocation = PortfolioAllocation(
            weights=dict(zip(returns.columns, final_weights)),
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            optimization_objective=OptimizationObjective.MAX_DIVERSIFICATION,
            optimization_method=OptimizationMethod.HRP
        )
        
        return allocation
    
    def _quasi_diagonalize(self, linkage_matrix: np.ndarray) -> List[int]:
        """Quasi-diagonalize the linkage matrix"""
        
        def _get_cluster_order(linkage_matrix):
            """Recursively get cluster order"""
            n = linkage_matrix.shape[0] + 1
            order = []
            
            def _recurse(node):
                if node < n:
                    order.append(node)
                else:
                    left = int(linkage_matrix[node - n, 0])
                    right = int(linkage_matrix[node - n, 1])
                    _recurse(left)
                    _recurse(right)
            
            _recurse(2 * n - 2)  # Start from root
            return order
        
        return _get_cluster_order(linkage_matrix)
    
    def _recursive_bisection(self,
                            returns: pd.DataFrame,
                            constraints: Optional[OptimizationConstraints]) -> np.ndarray:
        """Recursive bisection algorithm"""
        
        n_assets = len(returns.columns)
        
        if n_assets == 1:
            return np.array([1.0])
        
        # Split into two clusters
        split_idx = n_assets // 2
        cluster1 = returns.iloc[:, :split_idx]
        cluster2 = returns.iloc[:, split_idx:]
        
        # Recursively optimize each cluster
        weights1 = self._recursive_bisection(cluster1, constraints)
        weights2 = self._recursive_bisection(cluster2, constraints)
        
        # Calculate inverse variance weights for combining clusters
        var1 = np.var(cluster1 @ weights1) if len(weights1) > 0 else 1.0
        var2 = np.var(cluster2 @ weights2) if len(weights2) > 0 else 1.0
        
        # Avoid division by zero
        if var1 == 0 and var2 == 0:
            alpha = 0.5
        elif var1 == 0:
            alpha = 0.0
        elif var2 == 0:
            alpha = 1.0
        else:
            alpha = var2 / (var1 + var2)
        
        # Combine weights
        combined_weights = np.concatenate([weights1 * alpha, weights2 * (1 - alpha)])
        
        # Apply constraints if provided
        if constraints is not None:
            combined_weights = self._apply_constraints(combined_weights, returns.columns, constraints)
        
        return combined_weights
    
    def _apply_constraints(self,
                          weights: np.ndarray,
                          assets: List[str],
                          constraints: OptimizationConstraints) -> np.ndarray:
        """Apply constraints to weights"""
        
        # Minimum weight constraint
        if isinstance(constraints.min_weight, dict):
            for i, asset in enumerate(assets):
                if asset in constraints.min_weight:
                    weights[i] = max(weights[i], constraints.min_weight[asset])
        elif constraints.min_weight > 0:
            weights = np.maximum(weights, constraints.min_weight)
        
        # Maximum weight constraint
        if isinstance(constraints.max_weight, dict):
            for i, asset in enumerate(assets):
                if asset in constraints.max_weight:
                    weights[i] = min(weights[i], constraints.max_weight[asset])
        elif constraints.max_weight < 1:
            weights = np.minimum(weights, constraints.max_weight)
        
        # Normalize
        weights = weights / np.sum(weights)
        
        return weights

class GeneticAlgorithmOptimizer:
    """Genetic Algorithm for portfolio optimization"""
    
    def __init__(self, config: PortfolioOptimizerConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # GA parameters
        self.population_size = 100
        self.generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elitism_count = 5
    
    def optimize(self,
                expected_returns: pd.Series,
                covariance_matrix: pd.DataFrame,
                constraints: OptimizationConstraints,
                objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
                risk_free_rate: float = 0.0) -> PortfolioAllocation:
        """Optimize using genetic algorithm"""
        
        assets = expected_returns.index.tolist()
        n_assets = len(assets)
        
        mu = expected_returns.values
        Sigma = covariance_matrix.values
        
        # Initialize population
        population = self._initialize_population(n_assets)
        
        # Evolution
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = self._evaluate_population(
                population, mu, Sigma, objective, risk_free_rate, constraints
            )
            
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Crossover
            offspring = self._crossover(selected)
            
            # Mutation
            offspring = self._mutation(offspring)
            
            # Apply constraints
            offspring = self._apply_constraints(offspring, constraints, assets)
            
            # Elitism
            if self.elitism_count > 0:
                elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
                elite = population[elite_indices]
                offspring[-self.elitism_count:] = elite
            
            # New generation
            population = offspring
        
        # Select best solution
        fitness_scores = self._evaluate_population(
            population, mu, Sigma, objective, risk_free_rate, constraints
        )
        best_idx = np.argmax(fitness_scores)
        best_weights = population[best_idx]
        
        # Calculate metrics
        portfolio_return = mu @ best_weights
        portfolio_volatility = np.sqrt(best_weights.T @ Sigma @ best_weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        allocation = PortfolioAllocation(
            weights=dict(zip(assets, best_weights)),
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            optimization_objective=objective,
            optimization_method=OptimizationMethod.GENETIC_ALGORITHM
        )
        
        return allocation
    
    def _initialize_population(self, n_assets: int) -> np.ndarray:
        """Initialize random population"""
        population = np.random.rand(self.population_size, n_assets)
        
        # Normalize each individual (sum to 1)
        population = population / population.sum(axis=1, keepdims=True)
        
        return population
    
    def _evaluate_population(self,
                            population: np.ndarray,
                            mu: np.ndarray,
                            Sigma: np.ndarray,
                            objective: OptimizationObjective,
                            risk_free_rate: float,
                            constraints: OptimizationConstraints) -> np.ndarray:
        """Evaluate fitness of population"""
        
        fitness_scores = np.zeros(len(population))
        
        for i, weights in enumerate(population):
            # Calculate portfolio metrics
            portfolio_return = mu @ weights
            portfolio_volatility = np.sqrt(weights.T @ Sigma @ weights)
            
            # Calculate fitness based on objective
            if objective == OptimizationObjective.MAX_SHARPE:
                if portfolio_volatility > 0:
                    fitness = (portfolio_return - risk_free_rate) / portfolio_volatility
                else:
                    fitness = 0.0
            elif objective == OptimizationObjective.MIN_VARIANCE:
                fitness = -portfolio_volatility  # Negative because we maximize fitness
            elif objective == OptimizationObjective.MAX_RETURN:
                fitness = portfolio_return
            elif objective == OptimizationObjective.MAX_UTILITY:
                risk_aversion = self.config.config.get('risk_aversion', 2.5)
                fitness = portfolio_return - (risk_aversion / 2) * portfolio_volatility ** 2
            else:
                fitness = 0.0
            
            # Penalize constraint violations
            penalty = self._calculate_constraint_penalty(weights, constraints)
            fitness_scores[i] = fitness - penalty
        
        return fitness_scores
    
    def _calculate_constraint_penalty(self,
                                     weights: np.ndarray,
                                     constraints: OptimizationConstraints) -> float:
        """Calculate penalty for constraint violations"""
        
        penalty = 0.0
        
        # Weight constraints
        if isinstance(constraints.min_weight, dict):
            pass  # Hard to handle with dictionary in GA
        elif constraints.min_weight > 0:
            penalty += np.sum(np.maximum(0, constraints.min_weight - weights)) * 100
        
        if isinstance(constraints.max_weight, dict):
            pass
        elif constraints.max_weight < 1:
            penalty += np.sum(np.maximum(0, weights - constraints.max_weight)) * 100
        
        # Leverage constraint
        if constraints.leverage_limit is not None:
            leverage = np.sum(np.abs(weights))
            if leverage > constraints.leverage_limit:
                penalty += (leverage - constraints.leverage_limit) * 100
        
        return penalty
    
    def _selection(self,
                  population: np.ndarray,
                  fitness_scores: np.ndarray) -> np.ndarray:
        """Select individuals for reproduction"""
        
        # Tournament selection
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Random tournament
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return np.array(selected)
    
    def _crossover(self, population: np.ndarray) -> np.ndarray:
        """Perform crossover"""
        
        offspring = []
        
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                parent1 = population[i]
                parent2 = population[i + 1]
                
                if np.random.rand() < self.crossover_rate:
                    # Uniform crossover
                    mask = np.random.rand(len(parent1)) > 0.5
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                    child1[mask] = parent2[mask]
                    child2[mask] = parent1[mask]
                    
                    # Normalize
                    child1 = child1 / np.sum(child1)
                    child2 = child2 / np.sum(child2)
                    
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parent1, parent2])
        
        return np.array(offspring)
    
    def _mutation(self, population: np.ndarray) -> np.ndarray:
        """Perform mutation"""
        
        mutated = population.copy()
        
        for i in range(len(mutated)):
            if np.random.rand() < self.mutation_rate:
                # Random perturbation
                mutation = np.random.randn(len(mutated[i])) * 0.1
                mutated[i] = np.maximum(0, mutated[i] + mutation)
                mutated[i] = mutated[i] / np.sum(mutated[i])
        
        return mutated
    
    def _apply_constraints(self,
                          population: np.ndarray,
                          constraints: OptimizationConstraints,
                          assets: List[str]) -> np.ndarray:
        """Apply constraints to population"""
        
        constrained_population = population.copy()
        
        for i in range(len(constrained_population)):
            weights = constrained_population[i]
            
            # Apply minimum weight
            if isinstance(constraints.min_weight, dict):
                for j, asset in enumerate(assets):
                    if asset in constraints.min_weight:
                        weights[j] = max(weights[j], constraints.min_weight[asset])
            elif constraints.min_weight > 0:
                weights = np.maximum(weights, constraints.min_weight)
            
            # Apply maximum weight
            if isinstance(constraints.max_weight, dict):
                for j, asset in enumerate(assets):
                    if asset in constraints.max_weight:
                        weights[j] = min(weights[j], constraints.max_weight[asset])
            elif constraints.max_weight < 1:
                weights = np.minimum(weights, constraints.max_weight)
            
            # Normalize
            weights = weights / np.sum(weights)
            constrained_population[i] = weights
        
        return constrained_population

# ============ Transaction Cost Models ============
class TransactionCostModel:
    """Models for transaction costs"""
    
    def __init__(self, config: PortfolioOptimizerConfig):
        self.config = config
    
    def calculate_costs(self,
                       current_weights: Dict[str, float],
                       target_weights: Dict[str, float],
                       portfolio_value: float,
                       asset_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate transaction costs"""
        
        costs = {}
        total_cost = 0.0
        
        for asset in set(current_weights.keys()) | set(target_weights.keys()):
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)
            
            # Calculate trade amount in dollars
            trade_amount = abs(target_weight - current_weight) * portfolio_value
            
            if trade_amount > 0:
                # Get asset price
                price = asset_prices.get(asset, 1.0)
                
                # Calculate cost based on model
                cost_model = self.config.config.get('transaction_cost_model', 'proportional')
                
                if cost_model == 'proportional':
                    cost_rate = self.config.config.get('transaction_cost_rate', 0.001)
                    cost = trade_amount * cost_rate
                elif cost_model == 'fixed':
                    fixed_cost = self.config.config.get('fixed_transaction_cost', 0.0)
                    cost = fixed_cost
                elif cost_model == 'tiered':
                    # Tiered cost structure
                    cost = self._calculate_tiered_cost(trade_amount, price)
                else:
                    cost = 0.0
                
                costs[asset] = cost
                total_cost += cost
        
        return costs
    
    def _calculate_tiered_cost(self, trade_amount: float, price: float) -> float:
        """Calculate tiered transaction costs"""
        
        # Example tiered structure
        if trade_amount < 10000:
            rate = 0.0010  # 0.10%
        elif trade_amount < 100000:
            rate = 0.0007  # 0.07%
        elif trade_amount < 1000000:
            rate = 0.0005  # 0.05%
        else:
            rate = 0.0003  # 0.03%
        
        return trade_amount * rate

# ============ Main Portfolio Optimizer ============
class PortfolioOptimizer:
    """Main portfolio optimization engine"""
    
    def __init__(self, config: Optional[PortfolioOptimizerConfig] = None):
        self.config = config or PortfolioOptimizerConfig()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.return_estimator = ReturnEstimator(self.config)
        self.covariance_estimator = CovarianceEstimator(self.config)
        self.transaction_cost_model = TransactionCostModel(self.config)
        
        # Initialize optimizers
        self.mean_variance_optimizer = MeanVarianceOptimizer(self.config)
        self.risk_parity_optimizer = RiskParityOptimizer(self.config)
        self.hrp_optimizer = HierarchicalRiskParityOptimizer(self.config)
        self.genetic_algorithm_optimizer = GeneticAlgorithmOptimizer(self.config)
        
        # Cache for performance
        self.cache = Cache(ttl=self.config.config['cache_ttl_seconds'])
        
        # Optimization history
        self.optimization_history = deque(maxlen=100)
        
        # Performance tracking
        self.optimization_count = 0
        self.last_optimization_time = None
        
        self.logger.info("Portfolio Optimizer initialized")
    
    def optimize_portfolio(self,
                          inputs: OptimizationInputs,
                          constraints: Optional[OptimizationConstraints] = None,
                          objective: Optional[OptimizationObjective] = None,
                          method: Optional[OptimizationMethod] = None) -> PortfolioAllocation:
        """Main portfolio optimization method"""
        
        start_time = time.time()
        self.optimization_count += 1
        self.last_optimization_time = datetime.now()
        
        # Use defaults if not specified
        if objective is None:
            objective = OptimizationObjective(self.config.config['default_objective'])
        
        if method is None:
            method = OptimizationMethod(self.config.config['default_method'])
        
        if constraints is None:
            constraints = self._create_default_constraints(inputs.assets)
        
        # Check cache
        cache_key = self._generate_cache_key(inputs, constraints, objective, method)
        cached_result = self.cache.get(cache_key)
        
        if cached_result and self.config.config['cache_enabled']:
            self.logger.debug("Returning cached optimization result")
            return cached_result
        
        # Prepare data
        expected_returns, covariance_matrix = self._prepare_optimization_data(inputs)
        
        # Perform optimization based on method
        if method == OptimizationMethod.MEAN_VARIANCE:
            allocation = self.mean_variance_optimizer.optimize(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                constraints=constraints,
                objective=objective,
                risk_free_rate=inputs.risk_free_rate
            )
            
        elif method == OptimizationMethod.BLACK_LITTERMAN:
            # Estimate Black-Litterman returns first
            if inputs.views is not None:
                # Need market weights for equilibrium returns
                # For simplicity, use equal weights or provided market cap weights
                if 'market_cap_weights' in inputs.metadata:
                    market_weights = inputs.metadata['market_cap_weights']
                else:
                    market_weights = pd.Series(1/len(inputs.assets), index=inputs.assets)
                
                bl_returns = self.return_estimator.estimate_black_litterman_returns(
                    historical_returns=inputs.returns,
                    market_weights=market_weights,
                    views=inputs.views,
                    view_confidences=inputs.view_confidences,
                    tau=self.config.config.get('tau', 0.05)
                )
                
                # Use BL returns with mean-variance optimization
                allocation = self.mean_variance_optimizer.optimize(
                    expected_returns=bl_returns,
                    covariance_matrix=covariance_matrix,
                    constraints=constraints,
                    objective=objective,
                    risk_free_rate=inputs.risk_free_rate
                )
            else:
                # Fall back to mean-variance
                allocation = self.mean_variance_optimizer.optimize(
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    constraints=constraints,
                    objective=objective,
                    risk_free_rate=inputs.risk_free_rate
                )
        
        elif method == OptimizationMethod.HRP:
            allocation = self.hrp_optimizer.optimize(
                returns=inputs.returns,
                constraints=constraints
            )
            
        elif method == OptimizationMethod.GENETIC_ALGORITHM:
            allocation = self.genetic_algorithm_optimizer.optimize(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                constraints=constraints,
                objective=objective,
                risk_free_rate=inputs.risk_free_rate
            )
        
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
        
        # Add optimization method to allocation
        allocation.optimization_method = method
        
        # Calculate additional metrics
        allocation = self._calculate_additional_metrics(allocation, inputs)
        
        # Calculate calculation time
        allocation.calculation_time = time.time() - start_time
        
        # Cache result
        if self.config.config['cache_enabled']:
            self.cache.set(cache_key, allocation)
        
        # Add to history
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'allocation': allocation,
            'inputs': inputs,
            'constraints': constraints.to_dict() if constraints else None,
            'objective': objective.value,
            'method': method.value
        })
        
        self.logger.info(f"Portfolio optimization completed in {allocation.calculation_time:.2f}s")
        self.logger.info(f"Expected return: {allocation.expected_return:.2%}, "
                        f"Volatility: {allocation.expected_volatility:.2%}, "
                        f"Sharpe: {allocation.sharpe_ratio or 0:.2f}")
        
        return allocation
    
    def _prepare_optimization_data(self,
                                 inputs: OptimizationInputs) -> Tuple[pd.Series, pd.DataFrame]:
        """Prepare expected returns and covariance matrix for optimization"""
        
        # Estimate expected returns if not provided
        if inputs.expected_returns is not None:
            expected_returns = inputs.expected_returns
        else:
            expected_returns = self.return_estimator.estimate_historical_returns(
                inputs.returns,
                method=self.config.config['return_estimation_method'],
                lookback=self.config.config['return_lookback_periods']
            )
        
        # Estimate covariance matrix if not provided
        if inputs.covariance_matrix is not None:
            covariance_matrix = inputs.covariance_matrix
        else:
            covariance_method = self.config.config['covariance_estimation_method']
            
            if covariance_method == 'sample':
                covariance_matrix = self.covariance_estimator.estimate_sample_covariance(
                    inputs.returns
                )
            elif covariance_method in ['ledoit_wolf', 'oas']:
                covariance_matrix = self.covariance_estimator.estimate_shrinkage_covariance(
                    inputs.returns,
                    method=covariance_method
                )
            elif covariance_method == 'exponential':
                covariance_matrix = self.covariance_estimator.estimate_exponential_covariance(
                    inputs.returns
                )
            else:
                # Default to sample covariance
                covariance_matrix = inputs.returns.cov()
        
        return expected_returns, covariance_matrix
    
    def _create_default_constraints(self, assets: List[str]) -> OptimizationConstraints:
        """Create default optimization constraints"""
        
        return OptimizationConstraints(
            min_weight=self.config.config['default_min_weight'],
            max_weight=self.config.config['default_max_weight'],
            leverage_limit=self.config.config['max_leverage'],
            turnover_limit=self.config.config['max_turnover'],
            cardinality_limit=self.config.config['max_cardinality'],
            max_volatility=self.config.config['max_volatility'],
            max_var=self.config.config['max_var_limit']
        )
    
    def _calculate_additional_metrics(self,
                                    allocation: PortfolioAllocation,
                                    inputs: OptimizationInputs) -> PortfolioAllocation:
        """Calculate additional portfolio metrics"""
        
        # Calculate diversification ratio if not already calculated
        if allocation.diversification_ratio is None:
            # Get covariance matrix
            if inputs.covariance_matrix is not None:
                cov_matrix = inputs.covariance_matrix
            else:
                cov_matrix = inputs.returns.cov()
            
            # Calculate weighted average volatility
            asset_volatilities = np.sqrt(np.diag(cov_matrix))
            weights = np.array([allocation.weights[asset] for asset in cov_matrix.index])
            avg_vol = np.sum(weights * asset_volatilities)
            
            # Diversification ratio = avg vol / portfolio vol
            allocation.diversification_ratio = avg_vol / allocation.expected_volatility if allocation.expected_volatility > 0 else 1.0
        
        # Calculate tracking error if benchmark provided
        if inputs.benchmark_returns is not None and len(inputs.benchmark_returns) > 0:
            # Calculate portfolio returns using historical returns
            portfolio_returns = inputs.returns @ np.array([allocation.weights[asset] for asset in inputs.returns.columns])
            benchmark_returns = inputs.benchmark_returns
            
            # Align dates
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 0:
                portfolio_aligned = portfolio_returns.loc[common_dates]
                benchmark_aligned = benchmark_returns.loc[common_dates]
                
                # Calculate tracking error
                tracking_error = np.std(portfolio_aligned - benchmark_aligned)
                allocation.tracking_error = tracking_error
        
        return allocation
    
    def _generate_cache_key(self,
                           inputs: OptimizationInputs,
                           constraints: OptimizationConstraints,
                           objective: OptimizationObjective,
                           method: OptimizationMethod) -> str:
        """Generate cache key for optimization"""
        
        data_to_hash = {
            'assets': sorted(inputs.assets),
            'returns_hash': hashlib.md5(inputs.returns.values.tobytes()).hexdigest(),
            'constraints': constraints.to_dict() if constraints else {},
            'objective': objective.value,
            'method': method.value,
            'risk_free_rate': inputs.risk_free_rate
        }
        
        data_str = json.dumps(data_to_hash, sort_keys=True)
        cache_key = hashlib.md5(data_str.encode()).hexdigest()
        
        return f"portfolio_opt_{cache_key}"
    
    def calculate_efficient_frontier(self,
                                   inputs: OptimizationInputs,
                                   constraints: Optional[OptimizationConstraints] = None,
                                   n_points: int = 50) -> EfficientFrontier:
        """Calculate efficient frontier"""
        
        if constraints is None:
            constraints = self._create_default_constraints(inputs.assets)
        
        # Prepare data
        expected_returns, covariance_matrix = self._prepare_optimization_data(inputs)
        
        # Calculate efficient frontier
        frontier = self.mean_variance_optimizer.calculate_efficient_frontier(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            constraints=constraints,
            risk_free_rate=inputs.risk_free_rate,
            n_points=n_points
        )
        
        return frontier
    
    def rebalance_portfolio(self,
                          current_allocation: Dict[str, float],
                          target_allocation: PortfolioAllocation,
                          portfolio_value: float,
                          asset_prices: Dict[str, float],
                          constraints: Optional[OptimizationConstraints] = None) -> RebalancingDecision:
        """Determine rebalancing trades"""
        
        # Calculate required trades
        trades = {}
        for asset in set(current_allocation.keys()) | set(target_allocation.weights.keys()):
            current_weight = current_allocation.get(asset, 0.0)
            target_weight = target_allocation.weights.get(asset, 0.0)
            
            # Calculate trade weight
            trade_weight = target_weight - current_weight
            
            # Apply minimum trade size
            min_trade_size = self.config.config['min_rebalancing_size']
            if abs(trade_weight) < min_trade_size:
                # Skip very small trades
                trades[asset] = 0.0
            else:
                trades[asset] = trade_weight
        
        # Calculate transaction costs
        trade_costs = self.transaction_cost_model.calculate_costs(
            current_weights=current_allocation,
            target_weights=target_allocation.weights,
            portfolio_value=portfolio_value,
            asset_prices=asset_prices
        )
        
        # Calculate turnover
        turnover = sum(abs(trade) for trade in trades.values()) / 2  # Divide by 2 because buys+sells
        
        # Calculate implementation shortfall (estimated cost as % of portfolio)
        total_cost = sum(trade_costs.values())
        implementation_shortfall = total_cost / portfolio_value if portfolio_value > 0 else 0.0
        
        # Calculate tracking error reduction (simplified)
        # In reality, would compare current vs target tracking error
        tracking_error_reduction = 0.0
        
        # Determine if rebalancing is needed
        rebalancing_threshold = self.config.config['rebalancing_threshold']
        
        # Calculate maximum deviation from target
        max_deviation = max(
            abs(current_allocation.get(asset, 0.0) - target_allocation.weights.get(asset, 0.0))
            for asset in set(current_allocation.keys()) | set(target_allocation.weights.keys())
        )
        
        should_rebalance = max_deviation > rebalancing_threshold
        
        # Calculate rebalancing score (0-100)
        rebalancing_score = min(100, max_deviation * 100 / rebalancing_threshold)
        
        # Generate reason
        reason = ""
        if should_rebalance:
            reason = f"Maximum deviation {max_deviation:.1%} exceeds threshold {rebalancing_threshold:.1%}"
        else:
            reason = f"Maximum deviation {max_deviation:.1%} within threshold {rebalancing_threshold:.1%}"
        
        decision = RebalancingDecision(
            current_weights=current_allocation,
            target_weights=target_allocation.weights,
            trades=trades,
            trade_costs=trade_costs,
            tracking_error_reduction=tracking_error_reduction,
            turnover=turnover,
            implementation_shortfall=implementation_shortfall,
            should_rebalance=should_rebalance,
            rebalancing_score=rebalancing_score,
            reason=reason
        )
        
        return decision
    
    def optimize_with_risk_constraints(self,
                                     inputs: OptimizationInputs,
                                     risk_constraints: Dict[str, float],
                                     constraints: Optional[OptimizationConstraints] = None,
                                     objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE) -> PortfolioAllocation:
        """Optimize portfolio with additional risk constraints"""
        
        # Start with standard optimization
        allocation = self.optimize_portfolio(
            inputs=inputs,
            constraints=constraints,
            objective=objective
        )
        
        # Check risk constraints
        risk_analyzer = RiskAnalyzer()
        
        # Create portfolio state for risk analysis
        portfolio_state = PortfolioState(
            timestamp=datetime.now(),
            positions=allocation.weights,
            cash=0.0,
            portfolio_value=1.0,  # Normalized
            leverage=1.0
        )
        
        # Convert returns to market data format
        market_data = {}
        for asset in inputs.assets:
            if asset in inputs.returns.columns:
                market_data[asset] = pd.DataFrame({
                    'close': np.exp(inputs.returns[asset].cumsum())  # Convert to prices
                })
        
        # Analyze portfolio risk
        risk_metrics = risk_analyzer.analyze_portfolio_risk(
            portfolio_state=portfolio_state,
            market_data=market_data,
            include_stress_tests=False
        )
        
        # Check if risk constraints are satisfied
        violations = []
        for risk_metric, limit in risk_constraints.items():
            current_value = getattr(risk_metrics, risk_metric, None)
            
            if current_value is not None and current_value > limit:
                violations.append((risk_metric, current_value, limit))
        
        # If violations, adjust optimization
        if violations:
            self.logger.warning(f"Risk constraints violated: {violations}")
            
            # Adjust constraints and re-optimize
            if constraints is None:
                constraints = self._create_default_constraints(inputs.assets)
            
            # For now, just tighten weight constraints
            # In reality, you would implement a more sophisticated adjustment
            constraints.max_weight = min(constraints.max_weight, 0.2)  # Tighten to 20%
            
            # Re-optimize with tighter constraints
            allocation = self.optimize_portfolio(
                inputs=inputs,
                constraints=constraints,
                objective=objective
            )
        
        return allocation
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of optimizations"""
        return list(self.optimization_history)
    
    def save_optimization_report(self,
                                allocation: PortfolioAllocation,
                                inputs: OptimizationInputs,
                                constraints: OptimizationConstraints,
                                filepath: Optional[str] = None):
        """Save optimization report to file"""
        
        if not self.config.config['save_optimizations']:
            return
        
        try:
            if filepath is None:
                # Create default filepath
                report_dir = Path(self.config.config['optimization_report_path'])
                report_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"optimization_{timestamp}.json"
                filepath = report_dir / filename
            
            report = {
                'allocation': allocation.to_dict(),
                'inputs': {
                    'assets': inputs.assets,
                    'num_returns': len(inputs.returns),
                    'risk_free_rate': inputs.risk_free_rate
                },
                'constraints': constraints.to_dict() if constraints else None,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'optimization_count': self.optimization_count
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Optimization report saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimization report: {str(e)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of optimizer"""
        
        return {
            'optimization_count': self.optimization_count,
            'last_optimization_time': self.last_optimization_time.isoformat() if self.last_optimization_time else None,
            'history_size': len(self.optimization_history),
            'cache_size': len(self.cache._cache) if hasattr(self.cache, '_cache') else 0,
            'config': {
                'default_objective': self.config.config['default_objective'],
                'default_method': self.config.config['default_method'],
                'max_leverage': self.config.config['max_leverage']
            }
        }

# ============ Factory Function ============
def create_portfolio_optimizer(config: Optional[PortfolioOptimizerConfig] = None) -> PortfolioOptimizer:
    """Factory function to create portfolio optimizer"""
    return PortfolioOptimizer(config)

# ============ Main Execution ============
import time

async def main():
    """Main execution for testing"""
    
    # Create portfolio optimizer
    optimizer = create_portfolio_optimizer()
    
    # Create test data
    np.random.seed(42)
    n_assets = 10
    n_periods = 252
    
    # Generate random returns
    assets = [f"Asset_{i}" for i in range(n_assets)]
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')
    
    # Create correlated returns
    base_returns = np.random.randn(n_periods, n_assets) * 0.01
    
    # Add some correlation structure
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            if np.random.rand() < 0.3:  # 30% chance of correlation
                correlation = np.random.uniform(0.3, 0.8)
                base_returns[:, j] = correlation * base_returns[:, i] + np.sqrt(1 - correlation**2) * np.random.randn(n_periods) * 0.01
    
    returns_df = pd.DataFrame(base_returns, index=dates, columns=assets)
    
    # Create optimization inputs
    inputs = OptimizationInputs(
        assets=assets,
        returns=returns_df,
        risk_free_rate=0.02  # 2% risk-free rate
    )
    
    # Create constraints
    constraints = OptimizationConstraints(
        min_weight=0.0,
        max_weight=0.3,  # Max 30% per asset
        leverage_limit=1.0,
        max_volatility=0.3  # Max 30% volatility
    )
    
    try:
        # Optimize portfolio
        print("=== Portfolio Optimization ===")
        
        # Mean-Variance optimization
        allocation = optimizer.optimize_portfolio(
            inputs=inputs,
            constraints=constraints,
            objective=OptimizationObjective.MAX_SHARPE,
            method=OptimizationMethod.MEAN_VARIANCE
        )
        
        print(f"\nOptimization Results:")
        print(f"  Expected Return: {allocation.expected_return:.2%}")
        print(f"  Expected Volatility: {allocation.expected_volatility:.2%}")
        print(f"  Sharpe Ratio: {allocation.sharpe_ratio:.2f}")
        print(f"  Diversification Ratio: {allocation.diversification_ratio:.2f}")
        
        print(f"\nTop 5 Holdings:")
        sorted_weights = sorted(allocation.weights.items(), key=lambda x: x[1], reverse=True)
        for asset, weight in sorted_weights[:5]:
            print(f"  {asset}: {weight:.1%}")
        
        # Calculate efficient frontier
        print(f"\n=== Efficient Frontier ===")
        frontier = optimizer.calculate_efficient_frontier(
            inputs=inputs,
            constraints=constraints,
            n_points=20
        )
        
        print(f"Frontier calculated with {len(frontier.returns)} points")
        print(f"Minimum Volatility: {frontier.volatilities.min():.2%}")
        print(f"Maximum Return: {frontier.returns.max():.2%}")
        
        # Test rebalancing
        print(f"\n=== Rebalancing Test ===")
        
        # Create current allocation (different from target)
        current_allocation = {asset: 1/n_assets for asset in assets}  # Equal weights
        
        asset_prices = {asset: 100.0 for asset in assets}  # Fixed prices
        
        decision = optimizer.rebalance_portfolio(
            current_allocation=current_allocation,
            target_allocation=allocation,
            portfolio_value=100000.0,
            asset_prices=asset_prices
        )
        
        print(f"Should Rebalance: {decision.should_rebalance}")
        print(f"Rebalancing Score: {decision.rebalancing_score:.1f}/100")
        print(f"Turnover: {decision.turnover:.1%}")
        print(f"Implementation Shortfall: {decision.implementation_shortfall:.2%}")
        
        # Get optimizer summary
        summary = optimizer.get_performance_summary()
        print(f"\n=== Optimizer Summary ===")
        print(f"Total Optimizations: {summary['optimization_count']}")
        print(f"History Size: {summary['history_size']}")
        
    except Exception as e:
        print(f"Error in portfolio optimization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())