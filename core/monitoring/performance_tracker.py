"""
Performance Tracker module for Bitcoin trading AI.
Tracks and analyzes model performance, trading performance, risk metrics,
and provides comprehensive reporting and visualization.
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
import math

# Import project modules
from config.settings import PerformanceSettings, TradingSettings
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.utils.cache import Cache
from core.models.model_manager import ModelManager, ModelMetadata, ModelType, ModelStatus

# Import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import style
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Visualization disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Enums and Types ============
class MetricType(str, Enum):
    """Types of performance metrics"""
    MODEL = "model"
    TRADING = "trading"
    RISK = "risk"
    ECONOMIC = "economic"
    TECHNICAL = "technical"

class TimeFrame(str, Enum):
    """Time frames for performance analysis"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL = "all"

class PerformanceStatus(str, Enum):
    """Performance status levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

# ============ Data Structures ============
@dataclass
class PerformanceConfig:
    """Configuration for performance tracking"""
    
    # Tracking settings
    track_model_performance: bool = True
    track_trading_performance: bool = True
    track_risk_metrics: bool = True
    track_economic_indicators: bool = True
    
    # Update frequency
    update_frequency_minutes: int = 5
    real_time_tracking: bool = True
    batch_update_size: int = 100
    
    # Data retention
    max_history_days: int = 365
    data_compression: bool = True
    backup_frequency_hours: int = 24
    
    # Alert thresholds
    model_performance_threshold: float = 0.6
    trading_performance_threshold: float = 0.0  # Negative returns trigger alert
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'max_drawdown': 0.2,      # 20% max drawdown
        'var_95': 0.05,           # 5% daily VaR
        'sharpe_ratio': 1.0,      # Minimum Sharpe ratio
        'calmar_ratio': 1.5       # Minimum Calmar ratio
    })
    
    # Reporting
    generate_daily_report: bool = True
    generate_weekly_report: bool = True
    generate_monthly_report: bool = True
    report_format: str = "html"  # html, pdf, json, csv
    report_path: str = "reports/"
    
    # Visualization
    generate_charts: bool = True
    chart_style: str = "seaborn"  # seaborn, plotly, matplotlib
    chart_resolution: Tuple[int, int] = (1200, 800)
    save_charts: bool = True
    charts_path: str = "charts/"
    
    # Comparison
    benchmark_symbols: List[str] = field(default_factory=lambda: ["BTC-USD", "SPY"])
    compare_with_benchmarks: bool = True
    compare_with_models: bool = True
    
    # Advanced metrics
    calculate_advanced_metrics: bool = True
    monte_carlo_simulations: int = 10000
    stress_test_scenarios: List[str] = field(default_factory=lambda: [
        "crash_2008", "flash_crash_2010", "covid_2020", "bull_market"
    ])
    
    # Notifications
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log", "email"])
    email_recipients: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration"""
        if self.update_frequency_minutes < 1:
            raise ValueError("update_frequency_minutes must be at least 1")
        
        # Create directories
        Path(self.report_path).mkdir(parents=True, exist_ok=True)
        Path(self.charts_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'track_model_performance': self.track_model_performance,
            'track_trading_performance': self.track_trading_performance,
            'track_risk_metrics': self.track_risk_metrics,
            'track_economic_indicators': self.track_economic_indicators,
            'update_frequency_minutes': self.update_frequency_minutes,
            'real_time_tracking': self.real_time_tracking,
            'batch_update_size': self.batch_update_size,
            'max_history_days': self.max_history_days,
            'data_compression': self.data_compression,
            'backup_frequency_hours': self.backup_frequency_hours,
            'model_performance_threshold': self.model_performance_threshold,
            'trading_performance_threshold': self.trading_performance_threshold,
            'risk_thresholds': self.risk_thresholds,
            'generate_daily_report': self.generate_daily_report,
            'generate_weekly_report': self.generate_weekly_report,
            'generate_monthly_report': self.generate_monthly_report,
            'report_format': self.report_format,
            'report_path': self.report_path,
            'generate_charts': self.generate_charts,
            'chart_style': self.chart_style,
            'chart_resolution': self.chart_resolution,
            'save_charts': self.save_charts,
            'charts_path': self.charts_path,
            'benchmark_symbols': self.benchmark_symbols,
            'compare_with_benchmarks': self.compare_with_benchmarks,
            'compare_with_models': self.compare_with_models,
            'calculate_advanced_metrics': self.calculate_advanced_metrics,
            'monte_carlo_simulations': self.monte_carlo_simulations,
            'stress_test_scenarios': self.stress_test_scenarios,
            'enable_alerts': self.enable_alerts,
            'alert_channels': self.alert_channels,
            'email_recipients': self.email_recipients
        }

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    
    # Basic metrics
    model_id: str
    timestamp: datetime
    timeframe: TimeFrame
    
    # Accuracy metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mae: float = 0.0  # Mean Absolute Error
    mse: float = 0.0  # Mean Squared Error
    rmse: float = 0.0  # Root Mean Squared Error
    r2_score: float = 0.0  # R-squared
    
    # Prediction metrics
    prediction_accuracy: float = 0.0
    direction_accuracy: float = 0.0
    volatility_accuracy: float = 0.0
    confidence_score: float = 0.0
    
    # Timing metrics
    inference_time_ms: float = 0.0
    training_time_hours: float = 0.0
    last_retraining: Optional[datetime] = None
    
    # Usage metrics
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    prediction_success_rate: float = 0.0
    
    # Model health
    model_age_days: float = 0.0
    data_drift_score: float = 0.0
    concept_drift_score: float = 0.0
    model_degradation_score: float = 0.0
    
    # Performance status
    status: PerformanceStatus = PerformanceStatus.FAIR
    status_reason: str = ""
    
    def calculate_status(self, config: PerformanceConfig) -> None:
        """Calculate performance status based on metrics"""
        
        # Calculate overall score
        accuracy_weight = 0.3
        r2_weight = 0.25
        prediction_weight = 0.25
        drift_weight = 0.2
        
        overall_score = (
            self.accuracy * accuracy_weight +
            self.r2_score * r2_weight +
            self.prediction_accuracy * prediction_weight +
            (1 - max(self.data_drift_score, self.concept_drift_score)) * drift_weight
        )
        
        # Determine status
        if overall_score >= 0.8:
            self.status = PerformanceStatus.EXCELLENT
            self.status_reason = "Excellent overall performance"
        elif overall_score >= 0.7:
            self.status = PerformanceStatus.GOOD
            self.status_reason = "Good performance"
        elif overall_score >= config.model_performance_threshold:
            self.status = PerformanceStatus.FAIR
            self.status_reason = "Acceptable performance"
        elif overall_score >= 0.4:
            self.status = PerformanceStatus.POOR
            self.status_reason = "Poor performance - consider retraining"
        else:
            self.status = PerformanceStatus.CRITICAL
            self.status_reason = "Critical performance - immediate retraining needed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe.value,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mae': self.mae,
            'mse': self.mse,
            'rmse': self.rmse,
            'r2_score': self.r2_score,
            'prediction_accuracy': self.prediction_accuracy,
            'direction_accuracy': self.direction_accuracy,
            'volatility_accuracy': self.volatility_accuracy,
            'confidence_score': self.confidence_score,
            'inference_time_ms': self.inference_time_ms,
            'training_time_hours': self.training_time_hours,
            'last_retraining': self.last_retraining.isoformat() if self.last_retraining else None,
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'failed_predictions': self.failed_predictions,
            'prediction_success_rate': self.prediction_success_rate,
            'model_age_days': self.model_age_days,
            'data_drift_score': self.data_drift_score,
            'concept_drift_score': self.concept_drift_score,
            'model_degradation_score': self.model_degradation_score,
            'status': self.status.value,
            'status_reason': self.status_reason
        }

@dataclass
class TradingPerformance:
    """Trading performance metrics"""
    
    # Basic information
    portfolio_id: str
    timestamp: datetime
    timeframe: TimeFrame
    
    # Returns
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    daily_return_pct: float = 0.0
    weekly_return_pct: float = 0.0
    monthly_return_pct: float = 0.0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # Risk metrics
    volatility_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    var_95_pct: float = 0.0  # Value at Risk 95%
    cvar_95_pct: float = 0.0  # Conditional VaR 95%
    beta: float = 0.0
    alpha: float = 0.0
    
    # Trading activity
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    average_win_pct: float = 0.0
    average_loss_pct: float = 0.0
    avg_trade_duration_hours: float = 0.0
    
    # Position metrics
    average_position_size: float = 0.0
    max_position_size: float = 0.0
    position_concentration: float = 0.0
    
    # Cost metrics
    total_commission: float = 0.0
    total_slippage: float = 0.0
    commission_per_trade: float = 0.0
    
    # Performance vs benchmarks
    vs_benchmark_alpha: float = 0.0
    vs_benchmark_beta: float = 0.0
    vs_benchmark_sharpe: float = 0.0
    vs_benchmark_return: float = 0.0
    
    # Performance status
    status: PerformanceStatus = PerformanceStatus.FAIR
    status_reason: str = ""
    
    def calculate_status(self, config: PerformanceConfig) -> None:
        """Calculate performance status based on metrics"""
        
        # Check various thresholds
        issues = []
        
        if self.total_return_pct < config.trading_performance_threshold:
            issues.append(f"Negative returns: {self.total_return_pct:.2f}%")
        
        if self.max_drawdown_pct > config.risk_thresholds.get('max_drawdown', 0.2):
            issues.append(f"High drawdown: {self.max_drawdown_pct:.2f}%")
        
        if self.sharpe_ratio < config.risk_thresholds.get('sharpe_ratio', 1.0):
            issues.append(f"Low Sharpe ratio: {self.sharpe_ratio:.2f}")
        
        if self.calmar_ratio < config.risk_thresholds.get('calmar_ratio', 1.5):
            issues.append(f"Low Calmar ratio: {self.calmar_ratio:.2f}")
        
        if self.win_rate_pct < 40.0:
            issues.append(f"Low win rate: {self.win_rate_pct:.1f}%")
        
        # Determine status
        if not issues:
            self.status = PerformanceStatus.EXCELLENT
            self.status_reason = "All metrics within acceptable ranges"
        elif len(issues) == 1:
            self.status = PerformanceStatus.GOOD
            self.status_reason = f"Minor issue: {issues[0]}"
        elif len(issues) <= 2:
            self.status = PerformanceStatus.FAIR
            self.status_reason = f"Multiple issues: {', '.join(issues[:2])}"
        elif len(issues) <= 3:
            self.status = PerformanceStatus.POOR
            self.status_reason = f"Significant issues: {', '.join(issues[:3])}"
        else:
            self.status = PerformanceStatus.CRITICAL
            self.status_reason = f"Critical issues: {', '.join(issues[:4])}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'portfolio_id': self.portfolio_id,
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe.value,
            'total_return_pct': self.total_return_pct,
            'annualized_return_pct': self.annualized_return_pct,
            'daily_return_pct': self.daily_return_pct,
            'weekly_return_pct': self.weekly_return_pct,
            'monthly_return_pct': self.monthly_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'omega_ratio': self.omega_ratio,
            'volatility_pct': self.volatility_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'var_95_pct': self.var_95_pct,
            'cvar_95_pct': self.cvar_95_pct,
            'beta': self.beta,
            'alpha': self.alpha,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': self.win_rate_pct,
            'profit_factor': self.profit_factor,
            'average_win_pct': self.average_win_pct,
            'average_loss_pct': self.average_loss_pct,
            'avg_trade_duration_hours': self.avg_trade_duration_hours,
            'average_position_size': self.average_position_size,
            'max_position_size': self.max_position_size,
            'position_concentration': self.position_concentration,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'commission_per_trade': self.commission_per_trade,
            'vs_benchmark_alpha': self.vs_benchmark_alpha,
            'vs_benchmark_beta': self.vs_benchmark_beta,
            'vs_benchmark_sharpe': self.vs_benchmark_sharpe,
            'vs_benchmark_return': self.vs_benchmark_return,
            'status': self.status.value,
            'status_reason': self.status_reason
        }

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    
    # Basic information
    portfolio_id: str
    timestamp: datetime
    
    # Market risk
    market_risk_score: float = 0.0
    volatility_risk: float = 0.0
    correlation_risk: float = 0.0
    liquidity_risk: float = 0.0
    
    # Credit risk
    counterparty_risk: float = 0.0
    settlement_risk: float = 0.0
    
    # Operational risk
    model_risk: float = 0.0
    execution_risk: float = 0.0
    technology_risk: float = 0.0
    
    # Concentration risk
    sector_concentration: float = 0.0
    asset_concentration: float = 0.0
    geographic_concentration: float = 0.0
    
    # Stress test results
    stress_test_results: Dict[str, float] = field(default_factory=dict)
    
    # Scenario analysis
    worst_case_loss: float = 0.0
    expected_loss: float = 0.0
    unexpected_loss: float = 0.0
    
    # Risk limits
    risk_limit_utilization: Dict[str, float] = field(default_factory=dict)
    
    # Composite risk score
    overall_risk_score: float = 0.0
    risk_status: PerformanceStatus = PerformanceStatus.FAIR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'portfolio_id': self.portfolio_id,
            'timestamp': self.timestamp.isoformat(),
            'market_risk_score': self.market_risk_score,
            'volatility_risk': self.volatility_risk,
            'correlation_risk': self.correlation_risk,
            'liquidity_risk': self.liquidity_risk,
            'counterparty_risk': self.counterparty_risk,
            'settlement_risk': self.settlement_risk,
            'model_risk': self.model_risk,
            'execution_risk': self.execution_risk,
            'technology_risk': self.technology_risk,
            'sector_concentration': self.sector_concentration,
            'asset_concentration': self.asset_concentration,
            'geographic_concentration': self.geographic_concentration,
            'stress_test_results': self.stress_test_results,
            'worst_case_loss': self.worst_case_loss,
            'expected_loss': self.expected_loss,
            'unexpected_loss': self.unexpected_loss,
            'risk_limit_utilization': self.risk_limit_utilization,
            'overall_risk_score': self.overall_risk_score,
            'risk_status': self.risk_status.value
        }

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    
    report_id: str
    report_type: str  # daily, weekly, monthly, custom
    period_start: datetime
    period_end: datetime
    generation_time: datetime = field(default_factory=datetime.now)
    
    # Model performance
    model_performance: List[ModelPerformance] = field(default_factory=list)
    best_performing_model: Optional[str] = None
    worst_performing_model: Optional[str] = None
    
    # Trading performance
    trading_performance: Optional[TradingPerformance] = None
    key_trading_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Risk metrics
    risk_metrics: Optional[RiskMetrics] = None
    risk_alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Summary
    executive_summary: str = ""
    overall_status: PerformanceStatus = PerformanceStatus.FAIR
    
    # Metadata
    config_used: Optional[PerformanceConfig] = None
    data_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'report_id': self.report_id,
            'report_type': self.report_type,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'generation_time': self.generation_time.isoformat(),
            'model_performance': [mp.to_dict() for mp in self.model_performance],
            'best_performing_model': self.best_performing_model,
            'worst_performing_model': self.worst_performing_model,
            'trading_performance': self.trading_performance.to_dict() if self.trading_performance else None,
            'key_trading_metrics': self.key_trading_metrics,
            'risk_metrics': self.risk_metrics.to_dict() if self.risk_metrics else None,
            'risk_alerts': self.risk_alerts,
            'recommendations': self.recommendations,
            'executive_summary': self.executive_summary,
            'overall_status': self.overall_status.value,
            'config_used': self.config_used.to_dict() if self.config_used else None,
            'data_sources': self.data_sources
        }
    
    def save(self, filepath: str):
        """Save report to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {filepath}")

# ============ Metric Calculators ============
class ModelMetricCalculator:
    """Calculates model performance metrics"""
    
    @staticmethod
    def calculate_metrics(predictions: pd.DataFrame,
                         actuals: pd.DataFrame,
                         model_id: str,
                         timeframe: TimeFrame) -> ModelPerformance:
        """Calculate comprehensive model metrics"""
        
        performance = ModelPerformance(
            model_id=model_id,
            timestamp=datetime.now(),
            timeframe=timeframe
        )
        
        if predictions.empty or actuals.empty:
            logger.warning("No predictions or actuals data provided")
            return performance
        
        # Align predictions and actuals
        aligned_data = pd.merge(predictions, actuals, left_index=True, right_index=True, how='inner')
        
        if aligned_data.empty:
            logger.warning("No overlapping data between predictions and actuals")
            return performance
        
        pred_col = 'prediction' if 'prediction' in aligned_data.columns else aligned_data.columns[0]
        actual_col = 'actual' if 'actual' in aligned_data.columns else aligned_data.columns[-1]
        
        preds = aligned_data[pred_col].values
        actuals_vals = aligned_data[actual_col].values
        
        # Calculate basic regression metrics
        performance.mae = np.mean(np.abs(preds - actuals_vals))
        performance.mse = np.mean((preds - actuals_vals) ** 2)
        performance.rmse = np.sqrt(performance.mse)
        
        # Calculate R-squared
        ss_res = np.sum((actuals_vals - preds) ** 2)
        ss_tot = np.sum((actuals_vals - np.mean(actuals_vals)) ** 2)
        if ss_tot != 0:
            performance.r2_score = 1 - (ss_res / ss_tot)
        
        # Calculate direction accuracy
        if len(preds) > 1 and len(actuals_vals) > 1:
            pred_changes = np.diff(preds) > 0
            actual_changes = np.diff(actuals_vals) > 0
            performance.direction_accuracy = np.mean(pred_changes == actual_changes)
        
        # Calculate prediction accuracy (within tolerance)
        tolerance = 0.02  # 2% tolerance
        accuracy_mask = np.abs((preds - actuals_vals) / actuals_vals) <= tolerance
        performance.prediction_accuracy = np.mean(accuracy_mask)
        
        # Calculate volatility accuracy if volatility predictions exist
        if 'predicted_volatility' in aligned_data.columns and 'actual_volatility' in aligned_data.columns:
            vol_preds = aligned_data['predicted_volatility'].values
            vol_actuals = aligned_data['actual_volatility'].values
            performance.volatility_accuracy = 1 - np.mean(np.abs(vol_preds - vol_actuals) / vol_actuals)
        
        # For classification metrics (if applicable)
        if 'predicted_class' in aligned_data.columns and 'actual_class' in aligned_data.columns:
            pred_classes = aligned_data['predicted_class'].values
            actual_classes = aligned_data['actual_class'].values
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            performance.accuracy = accuracy_score(actual_classes, pred_classes)
            performance.precision = precision_score(actual_classes, pred_classes, average='weighted', zero_division=0)
            performance.recall = recall_score(actual_classes, pred_classes, average='weighted', zero_division=0)
            performance.f1_score = f1_score(actual_classes, pred_classes, average='weighted', zero_division=0)
        
        # Calculate confidence score (if available)
        if 'confidence' in aligned_data.columns:
            performance.confidence_score = np.mean(aligned_data['confidence'].values)
        
        return performance
    
    @staticmethod
    def calculate_drift_metrics(model_predictions: pd.DataFrame,
                               recent_data: pd.DataFrame,
                               historical_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate data drift and concept drift scores"""
        
        data_drift_score = 0.0
        concept_drift_score = 0.0
        
        if recent_data.empty or historical_data.empty:
            return data_drift_score, concept_drift_score
        
        # Calculate data drift (distribution changes)
        common_columns = set(recent_data.columns) & set(historical_data.columns)
        
        for col in common_columns:
            if recent_data[col].dtype in ['float64', 'int64']:
                # Compare distributions using Kolmogorov-Smirnov test
                from scipy import stats
                
                # Sample for performance
                recent_sample = recent_data[col].dropna().sample(min(1000, len(recent_data)), random_state=42)
                historical_sample = historical_data[col].dropna().sample(min(1000, len(historical_data)), random_state=42)
                
                if len(recent_sample) > 10 and len(historical_sample) > 10:
                    stat, _ = stats.ks_2samp(recent_sample, historical_sample)
                    data_drift_score = max(data_drift_score, stat)
        
        # Calculate concept drift (prediction performance changes)
        if not model_predictions.empty:
            # Split predictions into windows and compare performance
            window_size = min(100, len(model_predictions))
            windows = []
            
            for i in range(0, len(model_predictions) - window_size, window_size // 2):
                window = model_predictions.iloc[i:i + window_size]
                if 'actual' in window.columns and 'prediction' in window.columns:
                    mae = np.mean(np.abs(window['prediction'] - window['actual']))
                    windows.append(mae)
            
            if len(windows) > 1:
                # Calculate trend in MAE
                from scipy import stats
                x = np.arange(len(windows))
                slope, _, _, _, _ = stats.linregress(x, windows)
                concept_drift_score = abs(slope) * 100  # Scale for readability
        
        return data_drift_score, concept_drift_score

class TradingMetricCalculator:
    """Calculates trading performance metrics"""
    
    @staticmethod
    def calculate_metrics(trades: pd.DataFrame,
                         portfolio_value_series: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         timeframe: TimeFrame = TimeFrame.DAILY) -> TradingPerformance:
        """Calculate comprehensive trading metrics"""
        
        performance = TradingPerformance(
            portfolio_id="default",
            timestamp=datetime.now(),
            timeframe=timeframe
        )
        
        if portfolio_value_series.empty:
            logger.warning("No portfolio value data provided")
            return performance
        
        # Calculate returns
        returns = portfolio_value_series.pct_change().dropna()
        
        if returns.empty:
            logger.warning("No returns calculated")
            return performance
        
        # Basic return metrics
        performance.total_return_pct = ((portfolio_value_series.iloc[-1] / portfolio_value_series.iloc[0]) - 1) * 100
        
        # Annualized return
        days = (portfolio_value_series.index[-1] - portfolio_value_series.index[0]).days
        if days > 0:
            performance.annualized_return_pct = ((1 + performance.total_return_pct/100) ** (365/days) - 1) * 100
        
        # Daily, weekly, monthly returns
        performance.daily_return_pct = returns.iloc[-1] * 100 if len(returns) > 0 else 0
        if len(returns) >= 5:
            performance.weekly_return_pct = ((portfolio_value_series.iloc[-1] / portfolio_value_series.iloc[-5]) - 1) * 100
        if len(returns) >= 20:
            performance.monthly_return_pct = ((portfolio_value_series.iloc[-1] / portfolio_value_series.iloc[-20]) - 1) * 100
        
        # Risk metrics
        performance.volatility_pct = returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        performance.max_drawdown_pct = drawdown.min() * 100
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate for crypto)
        if returns.std() > 0:
            performance.sharpe_ratio = (returns.mean() * np.sqrt(252)) / returns.std()
        
        # Calculate Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            performance.sortino_ratio = (returns.mean() * np.sqrt(252)) / downside_returns.std()
        
        # Calculate Calmar ratio
        if abs(performance.max_drawdown_pct) > 0:
            performance.calmar_ratio = performance.annualized_return_pct / abs(performance.max_drawdown_pct)
        
        # Calculate Value at Risk (95%)
        if len(returns) >= 100:
            performance.var_95_pct = np.percentile(returns, 5) * 100
            performance.cvar_95_pct = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        # Calculate Omega ratio
        if returns.std() > 0:
            threshold = returns.mean()
            gains = returns[returns > threshold].sum()
            losses = abs(returns[returns <= threshold].sum())
            if losses > 0:
                performance.omega_ratio = gains / losses
        
        # Trade metrics (if trades data available)
        if not trades.empty:
            performance.total_trades = len(trades)
            
            if 'pnl' in trades.columns:
                winning_trades = trades[trades['pnl'] > 0]
                losing_trades = trades[trades['pnl'] <= 0]
                
                performance.winning_trades = len(winning_trades)
                performance.losing_trades = len(losing_trades)
                performance.win_rate_pct = (performance.winning_trades / performance.total_trades) * 100 if performance.total_trades > 0 else 0
                
                if len(winning_trades) > 0:
                    performance.average_win_pct = winning_trades['pnl'].mean() * 100
                if len(losing_trades) > 0:
                    performance.average_loss_pct = losing_trades['pnl'].mean() * 100
                
                # Profit factor
                total_gains = winning_trades['pnl'].sum() if not winning_trades.empty else 0
                total_losses = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
                if total_losses > 0:
                    performance.profit_factor = total_gains / total_losses
            
            # Trade duration
            if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
                durations = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 3600
                performance.avg_trade_duration_hours = durations.mean()
        
        # Compare with benchmark
        if benchmark_returns is not None and not benchmark_returns.empty:
            # Align returns with benchmark
            aligned_returns = returns.reindex(benchmark_returns.index).dropna()
            aligned_benchmark = benchmark_returns.reindex(aligned_returns.index)
            
            if len(aligned_returns) > 1 and len(aligned_benchmark) > 1:
                # Calculate alpha and beta
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                
                if benchmark_variance > 0:
                    performance.vs_benchmark_beta = covariance / benchmark_variance
                    performance.vs_benchmark_alpha = (aligned_returns.mean() * 252) - (
                        performance.vs_benchmark_beta * aligned_benchmark.mean() * 252
                    )
                
                # Compare returns
                benchmark_total_return = ((1 + aligned_benchmark).prod() - 1) * 100
                performance.vs_benchmark_return = performance.total_return_pct - benchmark_total_return
        
        return performance
    
    @staticmethod
    def calculate_advanced_metrics(returns: pd.Series,
                                  monte_carlo_simulations: int = 10000) -> Dict[str, Any]:
        """Calculate advanced trading metrics using Monte Carlo simulation"""
        
        advanced_metrics = {}
        
        if returns.empty or len(returns) < 20:
            return advanced_metrics
        
        # Monte Carlo simulation for future returns
        np.random.seed(42)
        
        mean_return = returns.mean()
        std_return = returns.std()
        days_to_simulate = 252  # One year
        
        simulated_paths = []
        for _ in range(monte_carlo_simulations):
            simulated_returns = np.random.normal(mean_return, std_return, days_to_simulate)
            simulated_path = (1 + simulated_returns).cumprod()
            simulated_paths.append(simulated_path)
        
        simulated_paths = np.array(simulated_paths)
        
        # Calculate metrics from simulation
        final_values = simulated_paths[:, -1]
        
        advanced_metrics['monte_carlo'] = {
            'expected_final_value': np.mean(final_values),
            'value_at_risk_95': np.percentile(final_values, 5),
            'value_at_risk_99': np.percentile(final_values, 1),
            'expected_shortfall_95': final_values[final_values <= np.percentile(final_values, 5)].mean(),
            'expected_shortfall_99': final_values[final_values <= np.percentile(final_values, 1)].mean(),
            'probability_of_loss': np.mean(final_values < 1),
            'probability_of_20pct_gain': np.mean(final_values > 1.2),
            'best_case': np.max(final_values),
            'worst_case': np.min(final_values)
        }
        
        # Calculate rolling metrics
        rolling_window = min(60, len(returns))
        if len(returns) >= rolling_window:
            rolling_sharpe = returns.rolling(window=rolling_window).apply(
                lambda x: (x.mean() * np.sqrt(252)) / x.std() if x.std() > 0 else 0
            )
            rolling_max_dd = returns.rolling(window=rolling_window).apply(
                lambda x: ModelPredictor._calculate_max_drawdown_from_returns(x)
            )
            
            advanced_metrics['rolling_metrics'] = {
                'sharpe_trend': float(rolling_sharpe.iloc[-1]) if not rolling_sharpe.empty else 0,
                'max_drawdown_trend': float(rolling_max_dd.iloc[-1]) if not rolling_max_dd.empty else 0,
                'sharpe_stability': float(rolling_sharpe.std()) if not rolling_sharpe.empty else 0
            }
        
        return advanced_metrics
    
    @staticmethod
    def _calculate_max_drawdown_from_returns(returns: pd.Series) -> float:
        """Calculate max drawdown from returns series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

class RiskMetricCalculator:
    """Calculates comprehensive risk metrics"""
    
    @staticmethod
    def calculate_metrics(portfolio_data: pd.DataFrame,
                         market_data: pd.DataFrame,
                         trade_data: pd.DataFrame) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        risk_metrics = RiskMetrics(
            portfolio_id="default",
            timestamp=datetime.now()
        )
        
        # Market risk
        if 'returns' in portfolio_data.columns:
            returns = portfolio_data['returns'].dropna()
            if not returns.empty:
                risk_metrics.volatility_risk = returns.std() * np.sqrt(252)
                
                # Calculate VaR and CVaR
                if len(returns) >= 100:
                    risk_metrics.var_95_pct = np.percentile(returns, 5) * 100
                    risk_metrics.cvar_95_pct = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        # Correlation risk
        if not market_data.empty:
            # Calculate correlation with major market indicators
            portfolio_returns = portfolio_data.get('returns', pd.Series())
            if not portfolio_returns.empty:
                correlations = []
                for col in market_data.columns:
                    if market_data[col].dtype in ['float64', 'int64']:
                        market_returns = market_data[col].pct_change().dropna()
                        aligned_data = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
                        if len(aligned_data) > 10:
                            corr = aligned_data.corr().iloc[0, 1]
                            correlations.append(abs(cr))
                
                if correlations:
                    risk_metrics.correlation_risk = np.mean(correlations)
        
        # Liquidity risk (simplified)
        if not trade_data.empty and 'volume' in trade_data.columns and 'price' in trade_data.columns:
            avg_daily_volume = trade_data['volume'].mean()
            avg_price = trade_data['price'].mean()
            
            # Estimate liquidation impact
            if avg_daily_volume > 0:
                position_size = portfolio_data.get('position_size', 0)
                days_to_liquidate = position_size / (avg_daily_volume * 0.1)  # Assuming 10% of daily volume
                risk_metrics.liquidity_risk = min(1.0, days_to_liquidate / 10)  # Scale to 0-1
        
        # Concentration risk
        if 'positions' in portfolio_data.columns:
            positions = portfolio_data['positions']
            if isinstance(positions, dict):
                position_values = list(positions.values())
                if position_values:
                    total_value = sum(position_values)
                    if total_value > 0:
                        herfindahl_index = sum((v/total_value) ** 2 for v in position_values)
                        risk_metrics.asset_concentration = herfindahl_index
        
        # Model risk (simplified)
        if 'model_predictions' in portfolio_data.columns and 'actual_values' in portfolio_data.columns:
            predictions = portfolio_data['model_predictions']
            actuals = portfolio_data['actual_values']
            
            if len(predictions) > 10 and len(actuals) > 10:
                mae = np.mean(np.abs(predictions - actuals))
                risk_metrics.model_risk = min(1.0, mae / np.std(actuals) if np.std(actuals) > 0 else 0)
        
        # Calculate overall risk score
        risk_metrics.overall_risk_score = RiskMetricCalculator._calculate_overall_risk_score(risk_metrics)
        
        # Determine risk status
        if risk_metrics.overall_risk_score < 0.3:
            risk_metrics.risk_status = PerformanceStatus.EXCELLENT
        elif risk_metrics.overall_risk_score < 0.5:
            risk_metrics.risk_status = PerformanceStatus.GOOD
        elif risk_metrics.overall_risk_score < 0.7:
            risk_metrics.risk_status = PerformanceStatus.FAIR
        elif risk_metrics.overall_risk_score < 0.9:
            risk_metrics.risk_status = PerformanceStatus.POOR
        else:
            risk_metrics.risk_status = PerformanceStatus.CRITICAL
        
        return risk_metrics
    
    @staticmethod
    def _calculate_overall_risk_score(risk_metrics: RiskMetrics) -> float:
        """Calculate composite risk score"""
        
        weights = {
            'volatility_risk': 0.25,
            'correlation_risk': 0.15,
            'liquidity_risk': 0.15,
            'asset_concentration': 0.20,
            'model_risk': 0.25
        }
        
        scores = []
        
        # Volatility risk (normalized)
        vol_score = min(1.0, risk_metrics.volatility_risk / 0.5)  # 50% volatility is max
        scores.append(vol_score * weights['volatility_risk'])
        
        # Correlation risk
        scores.append(risk_metrics.correlation_risk * weights['correlation_risk'])
        
        # Liquidity risk
        scores.append(risk_metrics.liquidity_risk * weights['liquidity_risk'])
        
        # Asset concentration
        scores.append(risk_metrics.asset_concentration * weights['asset_concentration'])
        
        # Model risk
        scores.append(risk_metrics.model_risk * weights['model_risk'])
        
        return min(1.0, sum(scores))

# ============ Visualization ============
class PerformanceVisualizer:
    """Creates visualizations for performance metrics"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.Visualizer")
        
    def create_performance_dashboard(self,
                                   model_performance: List[ModelPerformance],
                                   trading_performance: TradingPerformance,
                                   risk_metrics: RiskMetrics) -> Optional[Any]:
        """Create comprehensive performance dashboard"""
        
        if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
            self.logger.warning("No visualization libraries available")
            return None
        
        if self.config.chart_style == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_dashboard(model_performance, trading_performance, risk_metrics)
        else:
            return self._create_matplotlib_dashboard(model_performance, trading_performance, risk_metrics)
    
    def _create_plotly_dashboard(self,
                                model_performance: List[ModelPerformance],
                                trading_performance: TradingPerformance,
                                risk_metrics: RiskMetrics) -> go.Figure:
        """Create Plotly dashboard"""
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Model Accuracy', 'Returns Distribution', 'Risk Metrics',
                          'Trading Performance', 'Drawdown Analysis', 'Correlation Heatmap',
                          'Model Comparison', 'Portfolio Growth', 'Risk Breakdown'),
            specs=[[{'type': 'bar'}, {'type': 'histogram'}, {'type': 'radar'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'heatmap'}],
                   [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'pie'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Model Accuracy
        if model_performance:
            model_ids = [mp.model_id for mp in model_performance]
            accuracies = [mp.accuracy for mp in model_performance]
            
            fig.add_trace(
                go.Bar(x=model_ids, y=accuracies, name='Accuracy'),
                row=1, col=1
            )
        
        # Risk Metrics Radar Chart
        risk_categories = ['Volatility', 'Correlation', 'Liquidity', 'Concentration', 'Model']
        risk_values = [
            risk_metrics.volatility_risk,
            risk_metrics.correlation_risk,
            risk_metrics.liquidity_risk,
            risk_metrics.asset_concentration,
            risk_metrics.model_risk
        ]
        
        fig.add_trace(
            go.Scatterpolar(
                r=risk_values,
                theta=risk_categories,
                fill='toself',
                name='Risk Profile'
            ),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=self.config.chart_resolution[1],
            width=self.config.chart_resolution[0],
            title_text="Performance Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def _create_matplotlib_dashboard(self,
                                   model_performance: List[ModelPerformance],
                                   trading_performance: TradingPerformance,
                                   risk_metrics: RiskMetrics) -> plt.Figure:
        """Create Matplotlib dashboard"""
        
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        plt.style.use('seaborn' if self.config.chart_style == 'seaborn' else 'default')
        
        fig, axes = plt.subplots(3, 3, figsize=(self.config.chart_resolution[0]/100, 
                                               self.config.chart_resolution[1]/100))
        fig.suptitle('Performance Dashboard', fontsize=16)
        
        # Model Accuracy
        if model_performance:
            model_ids = [mp.model_id[:10] for mp in model_performance]  # Truncate IDs
            accuracies = [mp.accuracy for mp in model_performance]
            
            axes[0, 0].bar(model_ids, accuracies)
            axes[0, 0].set_title('Model Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Risk Metrics
        risk_categories = ['Volatility', 'Correlation', 'Liquidity', 'Concentration', 'Model']
        risk_values = [
            risk_metrics.volatility_risk,
            risk_metrics.correlation_risk,
            risk_metrics.liquidity_risk,
            risk_metrics.asset_concentration,
            risk_metrics.model_risk
        ]
        
        angles = np.linspace(0, 2*np.pi, len(risk_categories), endpoint=False).tolist()
        risk_values += risk_values[:1]
        angles += angles[:1]
        
        axes[0, 2].plot(angles, risk_values, 'o-', linewidth=2)
        axes[0, 2].fill(angles, risk_values, alpha=0.25)
        axes[0, 2].set_title('Risk Profile')
        axes[0, 2].set_xticks(angles[:-1])
        axes[0, 2].set_xticklabels(risk_categories)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def save_chart(self, chart: Any, filename: str):
        """Save chart to file"""
        
        filepath = Path(self.config.charts_path) / filename
        
        if isinstance(chart, go.Figure):
            if PLOTLY_AVAILABLE:
                chart.write_html(str(filepath.with_suffix('.html')))
                if self.config.save_charts:
                    chart.write_image(str(filepath.with_suffix('.png')))
        
        elif isinstance(chart, plt.Figure):
            if MATPLOTLIB_AVAILABLE:
                chart.savefig(str(filepath), dpi=300, bbox_inches='tight')
                plt.close(chart)

# ============ Alert System ============
class PerformanceAlertSystem:
    """Monitors performance and generates alerts"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.AlertSystem")
        self.alerts_sent = deque(maxlen=1000)
    
    def check_alerts(self,
                    model_performance: ModelPerformance,
                    trading_performance: TradingPerformance,
                    risk_metrics: RiskMetrics) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        
        alerts = []
        
        # Check model performance alerts
        if model_performance.status in [PerformanceStatus.POOR, PerformanceStatus.CRITICAL]:
            alerts.append({
                'type': 'model_performance',
                'severity': model_performance.status.value,
                'model_id': model_performance.model_id,
                'metric': 'overall_performance',
                'value': model_performance.accuracy,
                'threshold': self.config.model_performance_threshold,
                'message': f"Model {model_performance.model_id} performance is {model_performance.status.value}: {model_performance.status_reason}"
            })
        
        # Check trading performance alerts
        if trading_performance.status in [PerformanceStatus.POOR, PerformanceStatus.CRITICAL]:
            alerts.append({
                'type': 'trading_performance',
                'severity': trading_performance.status.value,
                'portfolio_id': trading_performance.portfolio_id,
                'metric': 'total_return',
                'value': trading_performance.total_return_pct,
                'threshold': self.config.trading_performance_threshold,
                'message': f"Trading performance is {trading_performance.status.value}: {trading_performance.status_reason}"
            })
        
        # Check risk alerts
        for risk_name, threshold in self.config.risk_thresholds.items():
            risk_value = getattr(trading_performance, f"{risk_name}_pct", None)
            if risk_value is not None:
                if risk_name == 'max_drawdown' and risk_value < -threshold * 100:
                    alerts.append({
                        'type': 'risk',
                        'severity': 'critical',
                        'metric': risk_name,
                        'value': risk_value,
                        'threshold': -threshold * 100,
                        'message': f"{risk_name} exceeded threshold: {risk_value:.2f}% vs {threshold*100:.2f}%"
                    })
                elif risk_name == 'sharpe_ratio' and risk_value < threshold:
                    alerts.append({
                        'type': 'risk',
                        'severity': 'warning',
                        'metric': risk_name,
                        'value': risk_value,
                        'threshold': threshold,
                        'message': f"{risk_name} below threshold: {risk_value:.2f} vs {threshold:.2f}"
                    })
        
        # Send alerts
        for alert in alerts:
            self._send_alert(alert)
        
        return alerts
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels"""
        
        alert_id = str(uuid.uuid4())
        alert['alert_id'] = alert_id
        alert['timestamp'] = datetime.now().isoformat()
        
        self.alerts_sent.append(alert)
        
        # Log alert
        if 'log' in self.config.alert_channels:
            self.logger.warning(f"ALERT: {alert['message']}")
        
        # Email alert (simplified - would need email configuration)
        if 'email' in self.config.alert_channels and self.config.email_recipients:
            self._send_email_alert(alert)
        
        # Could add more channels: Slack, SMS, etc.
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert (placeholder)"""
        # In production, integrate with email service
        pass
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return list(self.alerts_sent)[-limit:]

# ============ Main Performance Tracker ============
class PerformanceTracker:
    """Main performance tracking engine"""
    
    def __init__(self, 
                 config: PerformanceConfig,
                 model_manager: Optional[ModelManager] = None):
        
        self.config = config
        self.model_manager = model_manager
        self.logger = get_logger(__name__)
        
        # Data storage
        self.model_performance_history: Dict[str, List[ModelPerformance]] = defaultdict(list)
        self.trading_performance_history: List[TradingPerformance] = []
        self.risk_metrics_history: List[RiskMetrics] = []
        
        # Calculators
        self.model_calculator = ModelMetricCalculator()
        self.trading_calculator = TradingMetricCalculator()
        self.risk_calculator = RiskMetricCalculator()
        
        # Visualizer
        self.visualizer = PerformanceVisualizer(config)
        
        # Alert system
        self.alert_system = PerformanceAlertSystem(config)
        
        # Report generator
        self.reports_generated = []
        
        # Statistics
        self.last_update = datetime.now()
        self.total_updates = 0
        
        # Initialize data storage
        self._initialize_storage()
        
        self.logger.info("Performance Tracker initialized")
    
    def _initialize_storage(self):
        """Initialize data storage"""
        
        # Create data directory
        data_dir = Path("data/performance")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data if available
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical performance data"""
        
        data_dir = Path("data/performance")
        
        # Load model performance
        model_perf_file = data_dir / "model_performance.json"
        if model_perf_file.exists():
            try:
                with open(model_perf_file, 'r') as f:
                    data = json.load(f)
                    for model_id, perf_list in data.items():
                        for perf_dict in perf_list:
                            # Convert string dates back to datetime
                            perf_dict['timestamp'] = datetime.fromisoformat(perf_dict['timestamp'])
                            if perf_dict['last_retraining']:
                                perf_dict['last_retraining'] = datetime.fromisoformat(perf_dict['last_retraining'])
                            perf_dict['timeframe'] = TimeFrame(perf_dict['timeframe'])
                            perf_dict['status'] = PerformanceStatus(perf_dict['status'])
                            
                            performance = ModelPerformance(**perf_dict)
                            self.model_performance_history[model_id].append(performance)
            except Exception as e:
                self.logger.error(f"Failed to load model performance data: {str(e)}")
    
    def _save_historical_data(self):
        """Save historical performance data"""
        
        data_dir = Path("data/performance")
        
        # Save model performance
        model_perf_data = {}
        for model_id, perf_list in self.model_performance_history.items():
            model_perf_data[model_id] = [p.to_dict() for p in perf_list[-1000:]]  # Keep last 1000 entries
        
        with open(data_dir / "model_performance.json", 'w') as f:
            json.dump(model_perf_data, f, indent=2, default=str)
        
        # Save trading performance
        if self.trading_performance_history:
            trading_data = [p.to_dict() for p in self.trading_performance_history[-1000:]]
            with open(data_dir / "trading_performance.json", 'w') as f:
                json.dump(trading_data, f, indent=2, default=str)
    
    def update_model_performance(self,
                                model_id: str,
                                predictions: pd.DataFrame,
                                actuals: pd.DataFrame,
                                timeframe: TimeFrame = TimeFrame.DAILY) -> ModelPerformance:
        """Update model performance metrics"""
        
        try:
            # Calculate metrics
            performance = self.model_calculator.calculate_metrics(
                predictions, actuals, model_id, timeframe
            )
            
            # Calculate drift metrics if we have historical data
            if model_id in self.model_performance_history:
                recent_data = predictions.tail(100)
                historical_data = predictions.head(max(100, len(predictions) - 100))
                
                data_drift, concept_drift = self.model_calculator.calculate_drift_metrics(
                    predictions, recent_data, historical_data
                )
                
                performance.data_drift_score = data_drift
                performance.concept_drift_score = concept_drift
                performance.model_degradation_score = max(data_drift, concept_drift)
            
            # Update model metadata
            if self.model_manager:
                model_info = self.model_manager.get_model_info(model_id)
                if model_info:
                    performance.model_age_days = (datetime.now() - model_info.created_at).days
                    performance.last_retraining = model_info.updated_at
            
            # Calculate status
            performance.calculate_status(self.config)
            
            # Store in history
            self.model_performance_history[model_id].append(performance)
            
            # Keep only recent data
            max_history = self.config.max_history_days * 24  # Assuming hourly updates
            self.model_performance_history[model_id] = self.model_performance_history[model_id][-max_history:]
            
            # Check for alerts
            self.alert_system.check_alerts(performance, None, None)
            
            self.logger.info(f"Updated model performance for {model_id}: accuracy={performance.accuracy:.3f}, status={performance.status.value}")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Failed to update model performance for {model_id}: {str(e)}")
            return ModelPerformance(model_id=model_id, timestamp=datetime.now(), timeframe=timeframe)
    
    def update_trading_performance(self,
                                  trades: pd.DataFrame,
                                  portfolio_value_series: pd.Series,
                                  benchmark_returns: Optional[pd.Series] = None,
                                  timeframe: TimeFrame = TimeFrame.DAILY) -> TradingPerformance:
        """Update trading performance metrics"""
        
        try:
            # Calculate metrics
            performance = self.trading_calculator.calculate_metrics(
                trades, portfolio_value_series, benchmark_returns, timeframe
            )
            
            # Calculate advanced metrics if enabled
            if self.config.calculate_advanced_metrics:
                returns = portfolio_value_series.pct_change().dropna()
                advanced_metrics = self.trading_calculator.calculate_advanced_metrics(
                    returns, self.config.monte_carlo_simulations
                )
                # Store advanced metrics in performance metadata
                performance.metadata = advanced_metrics
            
            # Calculate status
            performance.calculate_status(self.config)
            
            # Store in history
            self.trading_performance_history.append(performance)
            
            # Keep only recent data
            max_history = self.config.max_history_days
            self.trading_performance_history = self.trading_performance_history[-max_history:]
            
            # Check for alerts
            self.alert_system.check_alerts(None, performance, None)
            
            self.logger.info(f"Updated trading performance: return={performance.total_return_pct:.2f}%, sharpe={performance.sharpe_ratio:.2f}")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Failed to update trading performance: {str(e)}")
            return TradingPerformance(portfolio_id="default", timestamp=datetime.now(), timeframe=timeframe)
    
    def update_risk_metrics(self,
                           portfolio_data: pd.DataFrame,
                           market_data: pd.DataFrame,
                           trade_data: pd.DataFrame) -> RiskMetrics:
        """Update risk metrics"""
        
        try:
            # Calculate risk metrics
            risk_metrics = self.risk_calculator.calculate_metrics(
                portfolio_data, market_data, trade_data
            )
            
            # Store in history
            self.risk_metrics_history.append(risk_metrics)
            
            # Keep only recent data
            max_history = self.config.max_history_days
            self.risk_metrics_history = self.risk_metrics_history[-max_history:]
            
            # Check for alerts
            self.alert_system.check_alerts(None, None, risk_metrics)
            
            self.logger.info(f"Updated risk metrics: overall_score={risk_metrics.overall_risk_score:.3f}, status={risk_metrics.risk_status.value}")
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to update risk metrics: {str(e)}")
            return RiskMetrics(portfolio_id="default", timestamp=datetime.now())
    
    def generate_report(self,
                       report_type: str,
                       period_start: datetime,
                       period_end: datetime) -> PerformanceReport:
        """Generate performance report"""
        
        report_id = f"report_{report_type}_{period_start.strftime('%Y%m%d')}_{period_end.strftime('%Y%m%d')}"
        
        # Filter data for report period
        model_perf_in_period = []
        for model_id, perf_list in self.model_performance_history.items():
            period_perf = [p for p in perf_list if period_start <= p.timestamp <= period_end]
            if period_perf:
                # Use the latest performance in period
                model_perf_in_period.append(period_perf[-1])
        
        trading_perf_in_period = None
        if self.trading_performance_history:
            period_trading = [p for p in self.trading_performance_history 
                            if period_start <= p.timestamp <= period_end]
            if period_trading:
                trading_perf_in_period = period_trading[-1]
        
        risk_metrics_in_period = None
        if self.risk_metrics_history:
            period_risk = [r for r in self.risk_metrics_history 
                          if period_start <= r.timestamp <= period_end]
            if period_risk:
                risk_metrics_in_period = period_risk[-1]
        
        # Create report
        report = PerformanceReport(
            report_id=report_id,
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            model_performance=model_perf_in_period,
            trading_performance=trading_perf_in_period,
            risk_metrics=risk_metrics_in_period,
            config_used=self.config
        )
        
        # Determine best and worst performing models
        if model_perf_in_period:
            sorted_models = sorted(model_perf_in_period, key=lambda x: x.accuracy, reverse=True)
            report.best_performing_model = sorted_models[0].model_id
            report.worst_performing_model = sorted_models[-1].model_id
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        # Generate executive summary
        report.executive_summary = self._generate_executive_summary(report)
        
        # Determine overall status
        report.overall_status = self._determine_overall_status(report)
        
        # Save report
        report_filename = f"{report_id}.{self.config.report_format}"
        report_path = Path(self.config.report_path) / report_filename
        report.save(report_path)
        
        # Generate charts if enabled
        if self.config.generate_charts:
            self._generate_report_charts(report)
        
        self.reports_generated.append(report)
        self.logger.info(f"Generated {report_type} report: {report_id}")
        
        return report
    
    def _generate_recommendations(self, report: PerformanceReport) -> List[Dict[str, Any]]:
        """Generate recommendations based on performance"""
        
        recommendations = []
        
        # Model recommendations
        for model_perf in report.model_performance:
            if model_perf.status in [PerformanceStatus.POOR, PerformanceStatus.CRITICAL]:
                recommendations.append({
                    'type': 'model',
                    'priority': 'high' if model_perf.status == PerformanceStatus.CRITICAL else 'medium',
                    'model_id': model_perf.model_id,
                    'action': 'retrain_model',
                    'reason': f"Model performance is {model_perf.status.value}: {model_perf.status_reason}",
                    'details': {
                        'accuracy': model_perf.accuracy,
                        'data_drift': model_perf.data_drift_score,
                        'concept_drift': model_perf.concept_drift_score
                    }
                })
        
        # Trading recommendations
        if report.trading_performance:
            trading_perf = report.trading_performance
            
            if trading_perf.win_rate_pct < 40.0:
                recommendations.append({
                    'type': 'trading',
                    'priority': 'medium',
                    'action': 'review_strategy',
                    'reason': f"Low win rate: {trading_perf.win_rate_pct:.1f}%",
                    'details': {
                        'win_rate': trading_perf.win_rate_pct,
                        'profit_factor': trading_perf.profit_factor,
                        'avg_win': trading_perf.average_win_pct,
                        'avg_loss': trading_perf.average_loss_pct
                    }
                })
            
            if trading_perf.max_drawdown_pct < -20.0:
                recommendations.append({
                    'type': 'risk',
                    'priority': 'high',
                    'action': 'reduce_position_size',
                    'reason': f"High maximum drawdown: {trading_perf.max_drawdown_pct:.2f}%",
                    'details': {
                        'max_drawdown': trading_perf.max_drawdown_pct,
                        'current_position_size': trading_perf.average_position_size
                    }
                })
        
        return recommendations
    
    def _generate_executive_summary(self, report: PerformanceReport) -> str:
        """Generate executive summary for report"""
        
        summary_parts = []
        
        # Model performance summary
        if report.model_performance:
            avg_accuracy = np.mean([mp.accuracy for mp in report.model_performance])
            best_model = report.best_performing_model
            worst_model = report.worst_performing_model
            
            summary_parts.append(
                f"Model Performance: Average accuracy {avg_accuracy:.1%}. "
                f"Best performing model: {best_model}. "
                f"Worst performing model: {worst_model}."
            )
        
        # Trading performance summary
        if report.trading_performance:
            trading = report.trading_performance
            summary_parts.append(
                f"Trading Performance: Return {trading.total_return_pct:.2f}%, "
                f"Sharpe Ratio {trading.sharpe_ratio:.2f}, "
                f"Max Drawdown {trading.max_drawdown_pct:.2f}%, "
                f"Win Rate {trading.win_rate_pct:.1f}%."
            )
        
        # Risk summary
        if report.risk_metrics:
            risk = report.risk_metrics
            summary_parts.append(
                f"Risk Assessment: Overall risk score {risk.overall_risk_score:.3f} "
                f"({risk.risk_status.value})."
            )
        
        # Recommendations summary
        if report.recommendations:
            high_priority = sum(1 for r in report.recommendations if r['priority'] == 'high')
            medium_priority = sum(1 for r in report.recommendations if r['priority'] == 'medium')
            
            if high_priority > 0 or medium_priority > 0:
                summary_parts.append(
                    f"Recommendations: {high_priority} high priority and {medium_priority} medium priority actions recommended."
                )
        
        return " ".join(summary_parts)
    
    def _determine_overall_status(self, report: PerformanceReport) -> PerformanceStatus:
        """Determine overall performance status"""
        
        statuses = []
        
        # Collect model statuses
        for model_perf in report.model_performance:
            statuses.append(model_perf.status)
        
        # Add trading status
        if report.trading_performance:
            statuses.append(report.trading_performance.status)
        
        # Add risk status
        if report.risk_metrics:
            statuses.append(report.risk_metrics.risk_status)
        
        if not statuses:
            return PerformanceStatus.FAIR
        
        # Use worst status
        status_priority = {
            PerformanceStatus.CRITICAL: 5,
            PerformanceStatus.POOR: 4,
            PerformanceStatus.FAIR: 3,
            PerformanceStatus.GOOD: 2,
            PerformanceStatus.EXCELLENT: 1
        }
        
        worst_status = max(statuses, key=lambda s: status_priority[s])
        return worst_status
    
    def _generate_report_charts(self, report: PerformanceReport):
        """Generate charts for report"""
        
        try:
            chart = self.visualizer.create_performance_dashboard(
                report.model_performance,
                report.trading_performance,
                report.risk_metrics
            )
            
            if chart:
                chart_filename = f"chart_{report.report_id}.png"
                self.visualizer.save_chart(chart, chart_filename)
        
        except Exception as e:
            self.logger.error(f"Failed to generate charts: {str(e)}")
    
    def get_model_performance_history(self,
                                     model_id: str,
                                     days_back: int = 30) -> List[ModelPerformance]:
        """Get model performance history"""
        
        if model_id not in self.model_performance_history:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        history = [p for p in self.model_performance_history[model_id] 
                  if p.timestamp >= cutoff_date]
        
        return sorted(history, key=lambda x: x.timestamp)
    
    def get_trading_performance_history(self, days_back: int = 30) -> List[TradingPerformance]:
        """Get trading performance history"""
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        history = [p for p in self.trading_performance_history 
                  if p.timestamp >= cutoff_date]
        
        return sorted(history, key=lambda x: x.timestamp)
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare performance of multiple models"""
        
        comparison = {
            'models': {},
            'summary': {},
            'best_model': None,
            'worst_model': None
        }
        
        valid_models = []
        for model_id in model_ids:
            if model_id in self.model_performance_history:
                recent_perf = self.model_performance_history[model_id][-1] if self.model_performance_history[model_id] else None
                if recent_perf:
                    comparison['models'][model_id] = recent_perf.to_dict()
                    valid_models.append((model_id, recent_perf.accuracy))
        
        if valid_models:
            # Sort by accuracy
            sorted_models = sorted(valid_models, key=lambda x: x[1], reverse=True)
            comparison['best_model'] = sorted_models[0][0]
            comparison['worst_model'] = sorted_models[-1][0]
            
            # Calculate summary statistics
            accuracies = [acc for _, acc in valid_models]
            comparison['summary'] = {
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies)
            }
        
        return comparison
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        
        summary = {
            'last_update': self.last_update.isoformat(),
            'total_updates': self.total_updates,
            'models_tracked': len(self.model_performance_history),
            'trading_performance_updates': len(self.trading_performance_history),
            'risk_updates': len(self.risk_metrics_history),
            'reports_generated': len(self.reports_generated),
            'recent_alerts': self.alert_system.get_recent_alerts(10)
        }
        
        # Add recent performance metrics
        if self.model_performance_history:
            recent_models = []
            for model_id, perf_list in self.model_performance_history.items():
                if perf_list:
                    recent_models.append({
                        'model_id': model_id,
                        'latest_accuracy': perf_list[-1].accuracy,
                        'status': perf_list[-1].status.value
                    })
            summary['recent_model_performance'] = recent_models
        
        if self.trading_performance_history:
            latest_trading = self.trading_performance_history[-1]
            summary['latest_trading_performance'] = {
                'return_pct': latest_trading.total_return_pct,
                'sharpe_ratio': latest_trading.sharpe_ratio,
                'max_drawdown_pct': latest_trading.max_drawdown_pct,
                'status': latest_trading.status.value
            }
        
        return summary
    
    def run_periodic_update(self):
        """Run periodic performance update"""
        
        self.last_update = datetime.now()
        self.total_updates += 1
        
        # Save data
        self._save_historical_data()
        
        # Generate scheduled reports
        now = datetime.now()
        
        if self.config.generate_daily_report and now.hour == 23:  # End of day
            period_start = now - timedelta(days=1)
            self.generate_report("daily", period_start, now)
        
        if self.config.generate_weekly_report and now.weekday() == 6 and now.hour == 23:  # Sunday end of day
            period_start = now - timedelta(days=7)
            self.generate_report("weekly", period_start, now)
        
        if self.config.generate_monthly_report and now.day == 1 and now.hour == 0:  # First day of month
            period_start = now - timedelta(days=30)
            self.generate_report("monthly", period_start, now)
        
        self.logger.debug(f"Periodic update completed at {now.isoformat()}")

# ============ Helper Functions ============
def create_performance_tracker(config: Optional[PerformanceConfig] = None,
                              model_manager: Optional[ModelManager] = None) -> PerformanceTracker:
    """Factory function to create performance tracker"""
    
    if config is None:
        config = PerformanceConfig()
    
    return PerformanceTracker(config, model_manager)


def load_tracker_from_config(config_path: str,
                            model_manager: Optional[ModelManager] = None) -> PerformanceTracker:
    """Load tracker from configuration file"""
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = PerformanceConfig(**config_dict)
    return PerformanceTracker(config, model_manager)


def calculate_performance_score(metrics: Dict[str, float], 
                              weights: Optional[Dict[str, float]] = None) -> float:
    """Calculate overall performance score from metrics"""
    
    if weights is None:
        weights = {
            'accuracy': 0.25,
            'sharpe_ratio': 0.20,
            'win_rate': 0.15,
            'profit_factor': 0.15,
            'max_drawdown': 0.15,
            'volatility': 0.10
        }
    
    score = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            # Normalize different metrics to 0-1 scale
            if metric == 'accuracy':
                normalized = metrics[metric]
            elif metric == 'sharpe_ratio':
                normalized = min(1.0, max(0.0, metrics[metric] / 3.0))  # Cap at 3
            elif metric == 'win_rate':
                normalized = metrics[metric] / 100.0
            elif metric == 'profit_factor':
                normalized = min(1.0, metrics[metric] / 5.0)  # Cap at 5
            elif metric == 'max_drawdown':
                normalized = 1.0 - min(1.0, abs(metrics[metric]) / 50.0)  # 50% is max
            elif metric == 'volatility':
                normalized = 1.0 - min(1.0, metrics[metric] / 100.0)  # 100% is max
            else:
                normalized = 0.0
            
            score += normalized * weight
    
    return min(1.0, max(0.0, score))


def create_performance_dataframe(performance_data: List[Union[ModelPerformance, TradingPerformance]]) -> pd.DataFrame:
    """Convert performance data to pandas DataFrame"""
    
    if not performance_data:
        return pd.DataFrame()
    
    # Convert to dictionary format
    data_dicts = []
    for perf in performance_data:
        data_dicts.append(perf.to_dict())
    
    df = pd.DataFrame(data_dicts)
    
    # Convert timestamp strings to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def visualize_performance_trends(performance_data: pd.DataFrame,
                               metric_column: str,
                               title: str = "Performance Trend") -> Optional[Any]:
    """Visualize performance trends over time"""
    
    if not MATPLOTLIB_AVAILABLE or performance_data.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'timestamp' in performance_data.columns and metric_column in performance_data.columns:
        ax.plot(performance_data['timestamp'], performance_data[metric_column], 
                marker='o', linestyle='-', linewidth=2)
        
        # Add moving average
        if len(performance_data) > 10:
            moving_avg = performance_data[metric_column].rolling(window=7).mean()
            ax.plot(performance_data['timestamp'], moving_avg, 
                    linestyle='--', linewidth=2, alpha=0.7, label='7-day MA')
        
        ax.set_xlabel('Date')
        ax.set_ylabel(metric_column.replace('_', ' ').title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    return None


# ============ Example Usage ============
if __name__ == "__main__":
    # Example usage
    print("Performance Tracker Module")
    
    # Create a sample config
    config = PerformanceConfig(
        track_model_performance=True,
        track_trading_performance=True,
        track_risk_metrics=True
    )
    
    # Create tracker
    tracker = PerformanceTracker(config)
    
    print(f"Performance Tracker initialized")
    print(f"Tracking: Model={config.track_model_performance}, "
          f"Trading={config.track_trading_performance}, "
          f"Risk={config.track_risk_metrics}")
    
    # Example of creating sample performance data
    sample_predictions = pd.DataFrame({
        'prediction': np.random.randn(100),
        'confidence': np.random.uniform(0.5, 0.9, 100)
    }, index=pd.date_range('2024-01-01', periods=100, freq='H'))
    
    sample_actuals = pd.DataFrame({
        'actual': sample_predictions['prediction'] + np.random.randn(100) * 0.1
    }, index=sample_predictions.index)
    
    # Update model performance
    model_perf = tracker.update_model_performance(
        model_id="sample_model",
        predictions=sample_predictions,
        actuals=sample_actuals,
        timeframe=TimeFrame.DAILY
    )
    
    print(f"Model performance updated: accuracy={model_perf.accuracy:.3f}, status={model_perf.status.value}")