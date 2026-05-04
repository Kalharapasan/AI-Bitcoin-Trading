#!/usr/bin/env python3
"""
Backtesting script for Bitcoin Trading AI application.
Run comprehensive backtests on trading strategies and models.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import warnings
import traceback

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.data_processing.data_collector import DataCollector
from core.data_processing.feature_engineer import FeatureEngineer
from core.data_processing.data_preprocessor import DataPreprocessor
from core.trading.signal_generator import SignalGenerator
from core.trading.position_sizer import PositionSizer
from core.trading.order_manager import OrderManager
from core.risk_management.risk_analyzer import RiskAnalyzer
from core.risk_management.stop_loss_manager import StopLossManager
from core.models.model_predictor import ModelPredictor
from backtesting.backtest_engine import BacktestEngine
from backtesting.walkforward_analyzer import WalkforwardAnalyzer
from backtesting.monte_carlo_simulator import MonteCarloSimulator
from database.connection import get_database_manager
from database.crud import BacktestResultCRUD
from config.config_manager import ConfigManager


class BacktestingPipeline:
    """Complete pipeline for backtesting trading strategies"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = self.setup_logger()
        
        # Initialize components
        self.data_collector = DataCollector(config)
        self.feature_engineer = FeatureEngineer(config)
        self.data_preprocessor = DataPreprocessor(config)
        self.signal_generator = SignalGenerator(config)
        self.position_sizer = PositionSizer(config)
        self.order_manager = OrderManager(config)
        self.risk_analyzer = RiskAnalyzer(config)
        self.stop_loss_manager = StopLossManager(config)
        self.model_predictor = ModelPredictor(config)
        
        # Initialize backtesting components
        self.backtest_engine = BacktestEngine(config)
        self.walkforward_analyzer = WalkforwardAnalyzer(config)
        self.monte_carlo_simulator = MonteCarloSimulator(config)
        
        # Get database connection
        self.db_manager = get_database_manager(config)
        
        # Backtesting parameters
        self.backtest_config = config.get_backtesting_config()
        
    def setup_logger(self) -> logging.Logger:
        """Setup logging for backtesting pipeline"""
        logger = logging.getLogger('backtesting')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        
        return logger
    
    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Load data for backtesting"""
        self.logger.info(f"Loading data for {symbol} {timeframe}")
        
        cache_dir = project_root / "data" / "cache"
        cache_file = cache_dir / f"{symbol.replace('/', '_')}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
        
        # Try to load from cache
        if use_cache and cache_file.exists():
            try:
                df = pd.read_pickle(cache_file)
                self.logger.info(f"Loaded data from cache: {len(df)} rows")
                return df
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        
        try:
            # Load from database
            df = self.data_collector.load_from_database(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty or len(df) < 100:
                raise ValueError(f"Insufficient data: {len(df)} rows")
            
            # Cache the data
            cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_pickle(cache_file)
            self.logger.info(f"Cached data: {cache_file}")
            
            self.logger.info(f"Data loaded: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for backtesting"""
        self.logger.info("Preparing features...")
        
        try:
            # Engineer features
            df_features = self.feature_engineer.engineer_all_features(df)
            
            # Remove rows with NaN values
            df_clean = df_features.dropna()
            
            self.logger.info(f"Features prepared: {len(df_clean)} rows, {len(df_clean.columns)} columns")
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            raise
    
    def load_model(self, model_id: str) -> Any:
        """Load trained model for backtesting"""
        self.logger.info(f"Loading model: {model_id}")
        
        try:
            # Try to load from model predictor
            model = self.model_predictor.load_model(model_id)
            
            if model is None:
                raise ValueError(f"Model not found: {model_id}")
            
            self.logger.info(f"Model loaded: {model_id}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        strategy: str,
        model_id: Optional[str] = None,
        model: Optional[Any] = None
    ) -> pd.DataFrame:
        """Generate trading signals"""
        self.logger.info(f"Generating signals using strategy: {strategy}")
        
        signals = []
        
        try:
            if strategy == 'model' and (model_id or model):
                # Use ML model for signal generation
                if model is None:
                    model = self.load_model(model_id)
                
                # Prepare features for model
                features_df = self.prepare_features(df)
                
                # Generate predictions
                predictions = self.model_predictor.predict(
                    model=model,
                    data=features_df
                )
                
                # Convert predictions to signals
                for i, (idx, row) in enumerate(features_df.iterrows()):
                    if i >= len(predictions):
                        break
                    
                    pred = predictions[i]
                    
                    # Simple signal logic (customize based on your model output)
                    if isinstance(pred, (list, np.ndarray)):
                        # Classification model
                        if len(pred) > 1:
                            signal_type = np.argmax(pred)
                            confidence = np.max(pred)
                        else:
                            # Regression model
                            signal_type = 1 if pred[0] > 0 else -1
                            confidence = abs(pred[0])
                    else:
                        # Single value
                        signal_type = 1 if pred > 0 else -1
                        confidence = abs(pred)
                    
                    signal = {
                        'timestamp': row['timestamp'] if 'timestamp' in row else idx,
                        'signal': 'buy' if signal_type == 1 else 'sell',
                        'strength': float(confidence),
                        'price': float(row['close']),
                        'strategy': strategy,
                        'model_id': model_id
                    }
                    signals.append(signal)
                    
            else:
                # Use traditional strategy
                for i, (idx, row) in enumerate(df.iterrows()):
                    # Get historical data for the current point
                    if i >= 50:  # Need enough history for indicators
                        historical_data = {
                            'close': df['close'].iloc[i-50:i].values.tolist(),
                            'volume': df['volume'].iloc[i-50:i].values.tolist()
                        }
                        
                        current_data = {
                            'timestamp': idx,
                            'close': float(row['close']),
                            'volume': float(row['volume'])
                        }
                        
                        # Generate signal
                        signal_info = self.signal_generator.generate_signal(
                            symbol='BTC/USDT',  # Will be overridden
                            market_data=current_data,
                            historical_data=historical_data,
                            strategy=strategy
                        )
                        
                        signal = {
                            'timestamp': idx,
                            'signal': signal_info['signal'],
                            'strength': signal_info['strength'],
                            'confidence': signal_info['confidence'],
                            'price': float(row['close']),
                            'strategy': strategy
                        }
                        signals.append(signal)
        
            signals_df = pd.DataFrame(signals)
            
            if signals_df.empty:
                raise ValueError("No signals generated")
            
            self.logger.info(f"Generated {len(signals_df)} signals")
            return signals_df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            raise
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        signals_df: pd.DataFrame,
        initial_capital: float,
        commission_rate: float = 0.001,
        slippage: float = 0.0001
    ) -> Dict[str, Any]:
        """Run backtest using backtest engine"""
        self.logger.info("Running backtest...")
        
        try:
            # Configure backtest engine
            self.backtest_engine.configure(
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                slippage=slippage
            )
            
            # Run backtest
            results = self.backtest_engine.run(
                data=df,
                signals=signals_df
            )
            
            # Calculate additional metrics
            results = self.calculate_additional_metrics(results)
            
            self.logger.info(f"Backtest completed: {results['total_trades']} trades")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise
    
    def calculate_additional_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional performance metrics"""
        
        # Extract key metrics
        trades = results.get('trades', [])
        equity_curve = results.get('equity_curve', [])
        
        if not trades or len(equity_curve) < 2:
            return results
        
        # Convert to numpy arrays for calculations
        equity = np.array([e['equity'] for e in equity_curve])
        returns = np.diff(equity) / equity[:-1]
        
        # Calculate risk metrics
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Annualized return
        days = len(equity_curve) / 24  # Assuming hourly data
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365 * 24)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = equity[0]
        max_dd = 0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Win rate
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade
        avg_trade = sum(t.get('pnl', 0) for t in trades) / total_trades if total_trades > 0 else 0
        
        # Add metrics to results
        results.update({
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_dd),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor) if profit_factor != float('inf') else None,
            'avg_trade': float(avg_trade),
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades
        })
        
        return results
    
    def run_walkforward_analysis(
        self,
        df: pd.DataFrame,
        strategy: str,
        model_id: Optional[str] = None,
        initial_capital: float = 10000.0,
        window_size: int = 180,  # days
        step_size: int = 30,     # days
        commission_rate: float = 0.001
    ) -> Dict[str, Any]:
        """Run walkforward analysis"""
        self.logger.info("Running walkforward analysis...")
        
        try:
            # Configure walkforward analyzer
            self.walkforward_analyzer.configure(
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                window_size=window_size,
                step_size=step_size
            )
            
            # Run analysis
            results = self.walkforward_analyzer.analyze(
                data=df,
                strategy=strategy,
                model_id=model_id
            )
            
            self.logger.info(f"Walkforward analysis completed: {len(results.get('windows', []))} windows")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running walkforward analysis: {e}")
            raise
    
    def run_monte_carlo_simulation(
        self,
        backtest_results: Dict[str, Any],
        n_simulations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation"""
        self.logger.info("Running Monte Carlo simulation...")
        
        try:
            # Configure Monte Carlo simulator
            self.monte_carlo_simulator.configure(
                n_simulations=n_simulations,
                confidence_level=confidence_level
            )
            
            # Run simulation
            simulation_results = self.monte_carlo_simulator.simulate(
                trades=backtest_results.get('trades', []),
                initial_capital=backtest_results.get('initial_capital', 10000)
            )
            
            self.logger.info(f"Monte Carlo simulation completed: {n_simulations} simulations")
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"Error running Monte Carlo simulation: {e}")
            raise
    
    def create_backtest_report(
        self,
        backtest_results: Dict[str, Any],
        walkforward_results: Optional[Dict[str, Any]] = None,
        monte_carlo_results: Optional[Dict[str, Any]] = None,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Create comprehensive backtest report"""
        self.logger.info("Creating backtest report...")
        
        report = {
            'summary': self.create_summary_section(backtest_results),
            'performance_metrics': self.create_performance_metrics_section(
                backtest_results, walkforward_results, monte_carlo_results
            ),
            'trades': self.create_trades_section(backtest_results),
            'risk_analysis': self.create_risk_analysis_section(
                backtest_results, monte_carlo_results
            ),
            'visualizations': self.create_visualizations_section(
                backtest_results, walkforward_results, monte_carlo_results, output_dir
            ),
            'recommendations': self.create_recommendations_section(backtest_results)
        }
        
        # Save report
        report_path = output_dir / "backtest_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Backtest report saved: {report_path}")
        
        return report
    
    def create_summary_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary section of report"""
        return {
            'strategy': results.get('strategy', 'Unknown'),
            'symbol': results.get('symbol', 'Unknown'),
            'timeframe': results.get('timeframe', 'Unknown'),
            'period': {
                'start': results.get('start_date'),
                'end': results.get('end_date')
            },
            'initial_capital': results.get('initial_capital'),
            'final_capital': results.get('final_capital'),
            'total_return': results.get('total_return'),
            'total_trades': results.get('total_trades', 0),
            'winning_trades': results.get('winning_trades', 0),
            'losing_trades': results.get('losing_trades', 0)
        }
    
    def create_performance_metrics_section(
        self,
        backtest_results: Dict[str, Any],
        walkforward_results: Optional[Dict[str, Any]],
        monte_carlo_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create performance metrics section"""
        metrics = {
            'returns': {
                'total_return': backtest_results.get('total_return'),
                'annual_return': backtest_results.get('annual_return')
            },
            'risk_adjusted': {
                'sharpe_ratio': backtest_results.get('sharpe_ratio'),
                'sortino_ratio': backtest_results.get('sortino_ratio', 0),
                'calmar_ratio': backtest_results.get('calmar_ratio', 0)
            },
            'trading': {
                'win_rate': backtest_results.get('win_rate'),
                'profit_factor': backtest_results.get('profit_factor'),
                'avg_trade': backtest_results.get('avg_trade'),
                'avg_win': backtest_results.get('avg_win', 0),
                'avg_loss': backtest_results.get('avg_loss', 0)
            },
            'risk': {
                'max_drawdown': backtest_results.get('max_drawdown'),
                'max_drawdown_duration': backtest_results.get('max_drawdown_duration', 0),
                'volatility': backtest_results.get('volatility', 0)
            }
        }
        
        # Add walkforward metrics if available
        if walkforward_results:
            metrics['walkforward'] = {
                'consistency': walkforward_results.get('consistency_score', 0),
                'stability': walkforward_results.get('stability_score', 0),
                'best_window': walkforward_results.get('best_window', {}),
                'worst_window': walkforward_results.get('worst_window', {})
            }
        
        # Add Monte Carlo metrics if available
        if monte_carlo_results:
            metrics['monte_carlo'] = {
                'expected_value': monte_carlo_results.get('expected_value'),
                'value_at_risk': monte_carlo_results.get('var'),
                'conditional_var': monte_carlo_results.get('cvar'),
                'success_probability': monte_carlo_results.get('success_probability')
            }
        
        return metrics
    
    def create_trades_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create trades analysis section"""
        trades = results.get('trades', [])
        
        if not trades:
            return {'total_trades': 0}
        
        # Analyze trades
        trade_analysis = {
            'total_trades': len(trades),
            'long_trades': sum(1 for t in trades if t.get('side') == 'buy'),
            'short_trades': sum(1 for t in trades if t.get('side') == 'sell'),
            'trade_duration_stats': {
                'avg': np.mean([t.get('duration', 0) for t in trades]) if trades else 0,
                'median': np.median([t.get('duration', 0) for t in trades]) if trades else 0,
                'std': np.std([t.get('duration', 0) for t in trades]) if trades else 0
            },
            'pnl_distribution': {
                'min': min(t.get('pnl', 0) for t in trades) if trades else 0,
                'max': max(t.get('pnl', 0) for t in trades) if trades else 0,
                'mean': np.mean([t.get('pnl', 0) for t in trades]) if trades else 0,
                'std': np.std([t.get('pnl', 0) for t in trades]) if trades else 0
            }
        }
        
        return trade_analysis
    
    def create_risk_analysis_section(
        self,
        backtest_results: Dict[str, Any],
        monte_carlo_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create risk analysis section"""
        risk_analysis = {
            'drawdown_analysis': {
                'max_drawdown': backtest_results.get('max_drawdown'),
                'avg_drawdown': backtest_results.get('avg_drawdown', 0),
                'drawdown_count': backtest_results.get('drawdown_count', 0)
            },
            'volatility_analysis': {
                'daily_volatility': backtest_results.get('daily_volatility', 0),
                'annual_volatility': backtest_results.get('annual_volatility', 0)
            },
            'risk_limits': {
                'max_position_risk': backtest_results.get('max_position_risk', 0),
                'daily_loss_limit': backtest_results.get('daily_loss_limit', 0)
            }
        }
        
        # Add Monte Carlo risk metrics
        if monte_carlo_results:
            risk_analysis['monte_carlo_risk'] = {
                'value_at_risk': monte_carlo_results.get('var'),
                'expected_shortfall': monte_carlo_results.get('cvar'),
                'ruin_probability': monte_carlo_results.get('ruin_probability', 0)
            }
        
        return risk_analysis
    
    def create_visualizations_section(
        self,
        backtest_results: Dict[str, Any],
        walkforward_results: Optional[Dict[str, Any]],
        monte_carlo_results: Optional[Dict[str, Any]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Create visualizations and return paths"""
        self.logger.info("Creating visualizations...")
        
        visualization_paths = {}
        
        try:
            # 1. Equity Curve
            equity_curve_path = self.plot_equity_curve(backtest_results, output_dir)
            if equity_curve_path:
                visualization_paths['equity_curve'] = str(equity_curve_path)
            
            # 2. Drawdown Chart
            drawdown_path = self.plot_drawdown(backtest_results, output_dir)
            if drawdown_path:
                visualization_paths['drawdown'] = str(drawdown_path)
            
            # 3. Trade Distribution
            trades_path = self.plot_trade_distribution(backtest_results, output_dir)
            if trades_path:
                visualization_paths['trade_distribution'] = str(trades_path)
            
            # 4. Monthly Returns Heatmap
            monthly_path = self.plot_monthly_returns(backtest_results, output_dir)
            if monthly_path:
                visualization_paths['monthly_returns'] = str(monthly_path)
            
            # 5. Walkforward Analysis Plot
            if walkforward_results:
                walkforward_path = self.plot_walkforward_analysis(walkforward_results, output_dir)
                if walkforward_path:
                    visualization_paths['walkforward'] = str(walkforward_path)
            
            # 6. Monte Carlo Simulation Plot
            if monte_carlo_results:
                monte_carlo_path = self.plot_monte_carlo_simulation(monte_carlo_results, output_dir)
                if monte_carlo_path:
                    visualization_paths['monte_carlo'] = str(monte_carlo_path)
            
            # 7. Risk-Return Scatter
            risk_return_path = self.plot_risk_return(backtest_results, output_dir)
            if risk_return_path:
                visualization_paths['risk_return'] = str(risk_return_path)
            
            self.logger.info(f"Created {len(visualization_paths)} visualizations")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
        
        return visualization_paths
    
    def plot_equity_curve(self, results: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Plot equity curve"""
        try:
            equity_curve = results.get('equity_curve', [])
            if not equity_curve:
                return None
            
            dates = [e['timestamp'] for e in equity_curve]
            equity = [e['equity'] for e in equity_curve]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Equity curve
            ax1.plot(dates, equity, 'b-', linewidth=2, label='Equity')
            ax1.fill_between(dates, equity, results.get('initial_capital', equity[0]), 
                           where=[e > results.get('initial_capital', equity[0]) for e in equity],
                           alpha=0.3, color='green', label='Profit')
            ax1.fill_between(dates, equity, results.get('initial_capital', equity[0]),
                           where=[e <= results.get('initial_capital', equity[0]) for e in equity],
                           alpha=0.3, color='red', label='Loss')
            
            ax1.set_title(f"Equity Curve - {results.get('strategy', 'Strategy')}")
            ax1.set_ylabel('Equity ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Daily returns
            returns = np.diff(equity) / equity[:-1]
            ax2.bar(dates[1:], returns, width=1, alpha=0.7, color=np.where(returns > 0, 'green', 'red'))
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_title('Daily Returns')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Return (%)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / "equity_curve.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.warning(f"Could not create equity curve plot: {e}")
            return None
    
    def plot_drawdown(self, results: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Plot drawdown chart"""
        try:
            equity_curve = results.get('equity_curve', [])
            if not equity_curve:
                return None
            
            dates = [e['timestamp'] for e in equity_curve]
            equity = [e['equity'] for e in equity_curve]
            
            # Calculate drawdown
            peak = equity[0]
            drawdown = []
            for value in equity:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                drawdown.append(dd)
            
            fig, ax = plt.subplots(figsize=(12, 4))
            
            ax.fill_between(dates, drawdown, 0, alpha=0.7, color='red')
            ax.plot(dates, drawdown, 'r-', linewidth=1, alpha=0.8)
            
            ax.set_title('Drawdown')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            
            # Add max drawdown line
            max_dd = max(drawdown)
            max_dd_idx = drawdown.index(max_dd)
            ax.axhline(y=max_dd, color='darkred', linestyle='--', alpha=0.7,
                      label=f'Max DD: {max_dd*100:.2f}%')
            
            ax.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / "drawdown.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.warning(f"Could not create drawdown plot: {e}")
            return None
    
    def plot_trade_distribution(self, results: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Plot trade distribution"""
        try:
            trades = results.get('trades', [])
            if not trades:
                return None
            
            pnl_values = [t.get('pnl', 0) for t in trades]
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram of P&L
            axes[0].hist(pnl_values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)
            axes[0].set_title('Distribution of Trade P&L')
            axes[0].set_xlabel('P&L ($)')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            # Box plot of P&L
            axes[1].boxplot(pnl_values, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue'))
            axes[1].set_title('P&L Box Plot')
            axes[1].set_ylabel('P&L ($)')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / "trade_distribution.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.warning(f"Could not create trade distribution plot: {e}")
            return None
    
    def plot_monthly_returns(self, results: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Plot monthly returns heatmap"""
        try:
            equity_curve = results.get('equity_curve', [])
            if not equity_curve:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(equity_curve)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate daily returns
            df['returns'] = df['equity'].pct_change()
            
            # Create monthly returns matrix
            monthly_returns = df['returns'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # Create heatmap data
            years = monthly_returns.index.year.unique()
            months = range(1, 13)
            
            heatmap_data = pd.DataFrame(index=years, columns=months)
            
            for date, ret in monthly_returns.items():
                heatmap_data.loc[date.year, date.month] = ret
            
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            
            im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', 
                          vmin=-0.2, vmax=0.2)
            
            # Add text
            for i in range(len(years)):
                for j in range(len(months)):
                    value = heatmap_data.iloc[i, j]
                    if pd.notna(value):
                        text = f"{value:.1%}"
                        ax.text(j, i, text, ha='center', va='center', 
                               fontsize=8, color='black' if abs(value) < 0.1 else 'white')
            
            ax.set_xticks(range(len(months)))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.set_yticks(range(len(years)))
            ax.set_yticklabels(years)
            
            ax.set_title('Monthly Returns Heatmap')
            plt.colorbar(im, ax=ax, label='Return')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / "monthly_returns.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.warning(f"Could not create monthly returns plot: {e}")
            return None
    
    def plot_walkforward_analysis(self, results: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Plot walkforward analysis results"""
        try:
            windows = results.get('windows', [])
            if not windows:
                return None
            
            # Extract window performance
            window_numbers = list(range(1, len(windows) + 1))
            window_returns = [w.get('return', 0) for w in windows]
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Window returns
            axes[0].bar(window_numbers, window_returns, alpha=0.7, color='steelblue')
            axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[0].set_title('Walkforward Analysis - Window Returns')
            axes[0].set_xlabel('Window Number')
            axes[0].set_ylabel('Return (%)')
            axes[0].grid(True, alpha=0.3)
            
            # Cumulative performance
            cumulative_returns = np.cumprod([1 + r/100 for r in window_returns]) - 1
            axes[1].plot(window_numbers, cumulative_returns, 'b-', linewidth=2, marker='o')
            axes[1].fill_between(window_numbers, 0, cumulative_returns, 
                               where=cumulative_returns >= 0, alpha=0.3, color='green')
            axes[1].fill_between(window_numbers, 0, cumulative_returns,
                               where=cumulative_returns < 0, alpha=0.3, color='red')
            axes[1].set_title('Cumulative Performance')
            axes[1].set_xlabel('Window Number')
            axes[1].set_ylabel('Cumulative Return')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / "walkforward_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.warning(f"Could not create walkforward plot: {e}")
            return None
    
    def plot_monte_carlo_simulation(self, results: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Plot Monte Carlo simulation results"""
        try:
            simulations = results.get('simulations', [])
            if not simulations:
                return None
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Distribution of final equity
            final_equities = [s[-1] for s in simulations]
            
            axes[0].hist(final_equities, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            axes[0].axvline(x=results.get('expected_value', np.mean(final_equities)), 
                          color='red', linestyle='--', linewidth=2, label='Expected Value')
            
            # Add VaR and CVaR lines
            var = results.get('var')
            cvar = results.get('cvar')
            
            if var:
                axes[0].axvline(x=var, color='orange', linestyle='--', linewidth=2, label='VaR (95%)')
            if cvar:
                axes[0].axvline(x=cvar, color='green', linestyle='--', linewidth=2, label='CVaR')
            
            axes[0].set_title('Monte Carlo Simulation - Final Equity Distribution')
            axes[0].set_xlabel('Final Equity ($)')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Sample simulation paths
            n_sample_paths = min(50, len(simulations))
            for i in range(n_sample_paths):
                axes[1].plot(simulations[i], alpha=0.1, color='blue')
            
            # Mean path
            mean_path = np.mean(simulations, axis=0)
            axes[1].plot(mean_path, 'r-', linewidth=2, label='Mean Path')
            
            # Confidence intervals
            if len(simulations) > 1:
                lower_bound = np.percentile(simulations, 2.5, axis=0)
                upper_bound = np.percentile(simulations, 97.5, axis=0)
                axes[1].fill_between(range(len(mean_path)), lower_bound, upper_bound, 
                                   alpha=0.3, color='red', label='95% CI')
            
            axes[1].set_title('Monte Carlo Simulation Paths')
            axes[1].set_xlabel('Time Step')
            axes[1].set_ylabel('Equity ($)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / "monte_carlo_simulation.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.warning(f"Could not create Monte Carlo plot: {e}")
            return None
    
    def plot_risk_return(self, results: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Plot risk-return scatter"""
        try:
            # This would typically compare multiple strategies
            # For now, just plot this strategy's point
            returns = results.get('annual_return', 0)
            risk = results.get('max_drawdown', 0)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot strategy point
            ax.scatter(risk * 100, returns * 100, s=200, color='blue', 
                      edgecolors='black', linewidth=2, label=results.get('strategy', 'Strategy'))
            
            # Add reference lines
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Add efficient frontier reference (simplified)
            x = np.linspace(0, 50, 100)
            y = 10 * (1 - np.exp(-x/20))  # Simplified efficient frontier
            ax.plot(x, y, 'g--', alpha=0.5, label='Efficient Frontier (Reference)')
            
            ax.set_title('Risk-Return Profile')
            ax.set_xlabel('Risk (Max Drawdown %)')
            ax.set_ylabel('Return (Annual %)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / "risk_return.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.warning(f"Could not create risk-return plot: {e}")
            return None
    
    def create_recommendations_section(self, results: Dict[str, Any]) -> List[str]:
        """Create recommendations based on backtest results"""
        recommendations = []
        
        # Analyze results and generate recommendations
        win_rate = results.get('win_rate', 0)
        profit_factor = results.get('profit_factor', 0)
        max_drawdown = results.get('max_drawdown', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        
        # Win rate recommendations
        if win_rate < 0.4:
            recommendations.append("Consider improving entry timing or adding filters to increase win rate")
        elif win_rate > 0.6:
            recommendations.append("High win rate detected. Consider optimizing position sizing for better risk-adjusted returns")
        
        # Profit factor recommendations
        if profit_factor and profit_factor < 1.2:
            recommendations.append("Low profit factor. Review exit strategies and risk management")
        elif profit_factor and profit_factor > 2.0:
            recommendations.append("Excellent profit factor. Strategy shows good risk-reward balance")
        
        # Drawdown recommendations
        if max_drawdown > 0.2:
            recommendations.append(f"High maximum drawdown ({max_drawdown*100:.1f}%). Consider adding stop-losses or reducing position sizes")
        
        # Sharpe ratio recommendations
        if sharpe_ratio < 0.5:
            recommendations.append("Low Sharpe ratio. Consider strategies with better risk-adjusted returns")
        elif sharpe_ratio > 1.5:
            recommendations.append("Good Sharpe ratio. Strategy provides solid risk-adjusted returns")
        
        # General recommendations
        if len(results.get('trades', [])) < 20:
            recommendations.append("Low number of trades. Consider testing over longer period or different market conditions")
        
        if not recommendations:
            recommendations.append("Strategy shows reasonable performance. Monitor closely in live trading.")
        
        return recommendations
    
    def save_results_to_database(
        self,
        results: Dict[str, Any],
        strategy: str,
        symbol: str,
        timeframe: str,
        model_id: Optional[str] = None
    ) -> bool:
        """Save backtest results to database"""
        self.logger.info("Saving results to database...")
        
        try:
            with self.db_manager.session_scope() as session:
                crud = BacktestResultCRUD(session)
                
                # Create backtest record
                backtest_record = {
                    'backtest_id': f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'strategy_name': strategy,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'start_date': results.get('start_date'),
                    'end_date': results.get('end_date'),
                    'initial_capital': results.get('initial_capital'),
                    'final_capital': results.get('final_capital'),
                    'total_return': results.get('total_return'),
                    'annual_return': results.get('annual_return'),
                    'sharpe_ratio': results.get('sharpe_ratio'),
                    'max_drawdown': results.get('max_drawdown'),
                    'win_rate': results.get('win_rate'),
                    'profit_factor': results.get('profit_factor'),
                    'total_trades': results.get('total_trades'),
                    'avg_trade': results.get('avg_trade'),
                    'parameters': {
                        'strategy': strategy,
                        'model_id': model_id,
                        'commission': results.get('commission_rate'),
                        'slippage': results.get('slippage')
                    },
                    'trades': results.get('trades', []),
                    'equity_curve': results.get('equity_curve', [])
                }
                
                # Save to database
                crud.create(backtest_record)
                session.commit()
                
                self.logger.info("Results saved to database")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving to database: {e}")
            return False
    
    def run_complete_backtest(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        strategy: str,
        model_id: Optional[str] = None,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
        slippage: float = 0.0001,
        run_walkforward: bool = False,
        run_monte_carlo: bool = False,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run complete backtesting pipeline"""
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING BACKTESTING PIPELINE")
        self.logger.info(f"Symbol: {symbol}")
        self.logger.info(f"Timeframe: {timeframe}")
        self.logger.info(f"Strategy: {strategy}")
        self.logger.info(f"Model ID: {model_id}")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info("=" * 60)
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = project_root / "results" / "backtesting" / f"{symbol.replace('/', '_')}_{timeframe}_{strategy}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy': strategy,
            'model_id': model_id,
            'start_date': start_date,
            'end_date': end_date,
            'output_dir': str(output_dir)
        }
        
        try:
            # Step 1: Load data
            df = self.load_data(symbol, timeframe, start_date, end_date)
            
            if df.empty or len(df) < 100:
                raise ValueError(f"Insufficient data: {len(df)} rows")
            
            # Step 2: Generate signals
            signals_df = self.generate_signals(df, strategy, model_id)
            
            if signals_df.empty:
                raise ValueError("No signals generated")
            
            # Step 3: Run backtest
            backtest_results = self.run_backtest(
                df, signals_df, initial_capital, commission_rate, slippage
            )
            
            # Add metadata
            backtest_results.update({
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy': strategy,
                'model_id': model_id,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'commission_rate': commission_rate,
                'slippage': slippage
            })
            
            all_results['backtest'] = backtest_results
            
            # Step 4: Run walkforward analysis (optional)
            walkforward_results = None
            if run_walkforward:
                walkforward_results = self.run_walkforward_analysis(
                    df, strategy, model_id, initial_capital
                )
                all_results['walkforward'] = walkforward_results
            
            # Step 5: Run Monte Carlo simulation (optional)
            monte_carlo_results = None
            if run_monte_carlo:
                monte_carlo_results = self.run_monte_carlo_simulation(backtest_results)
                all_results['monte_carlo'] = monte_carlo_results
            
            # Step 6: Create comprehensive report
            report = self.create_backtest_report(
                backtest_results, walkforward_results, monte_carlo_results, output_dir
            )
            all_results['report'] = report
            
            # Step 7: Save results to database
            self.save_results_to_database(
                backtest_results, strategy, symbol, timeframe, model_id
            )
            
            # Step 8: Print summary
            self.print_backtest_summary(backtest_results, walkforward_results, monte_carlo_results)
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("BACKTESTING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"\nBacktesting pipeline failed: {e}")
            if self.logger.level <= logging.DEBUG:
                traceback.print_exc()
            raise
    
    def print_backtest_summary(
        self,
        backtest_results: Dict[str, Any],
        walkforward_results: Optional[Dict[str, Any]] = None,
        monte_carlo_results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Print backtest summary to console"""
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        
        print(f"\nStrategy: {backtest_results.get('strategy', 'Unknown')}")
        print(f"Symbol: {backtest_results.get('symbol', 'Unknown')}")
        print(f"Period: {backtest_results.get('start_date')} to {backtest_results.get('end_date')}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Initial Capital: ${backtest_results.get('initial_capital', 0):.2f}")
        print(f"  Final Capital: ${backtest_results.get('final_capital', 0):.2f}")
        print(f"  Total Return: {backtest_results.get('total_return', 0)*100:.2f}%")
        print(f"  Annual Return: {backtest_results.get('annual_return', 0)*100:.2f}%")
        print(f"  Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {backtest_results.get('max_drawdown', 0)*100:.2f}%")
        
        print(f"\nTrading Statistics:")
        print(f"  Total Trades: {backtest_results.get('total_trades', 0)}")
        print(f"  Winning Trades: {backtest_results.get('winning_trades', 0)}")
        print(f"  Losing Trades: {backtest_results.get('losing_trades', 0)}")
        print(f"  Win Rate: {backtest_results.get('win_rate', 0)*100:.2f}%")
        print(f"  Profit Factor: {backtest_results.get('profit_factor', 0):.2f}")
        print(f"  Average Trade: ${backtest_results.get('avg_trade', 0):.2f}")
        
        if walkforward_results:
            print(f"\nWalkforward Analysis:")
            print(f"  Consistency Score: {walkforward_results.get('consistency_score', 0):.2f}")
            print(f"  Windows Tested: {len(walkforward_results.get('windows', []))}")
        
        if monte_carlo_results:
            print(f"\nMonte Carlo Simulation:")
            print(f"  Simulations: {monte_carlo_results.get('n_simulations', 0)}")
            print(f"  Success Probability: {monte_carlo_results.get('success_probability', 0)*100:.2f}%")
            print(f"  Value at Risk (95%): ${monte_carlo_results.get('var', 0):.2f}")
        
        print(f"\nRecommendations:")
        recommendations = self.create_recommendations_section(backtest_results)
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run backtests for Bitcoin trading strategies"
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading symbol (default: BTC/USDT)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        help='Timeframe for data (default: 1h)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2023-01-01',
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2023-12-31',
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['moving_average', 'rsi', 'macd', 'bollinger_bands', 'model', 'all'],
        default='moving_average',
        help='Trading strategy to test (default: moving_average)'
    )
    
    parser.add_argument(
        '--model-id',
        type=str,
        help='Model ID to use for model-based strategy'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=10000.0,
        help='Initial capital (default: 10000)'
    )
    
    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--slippage',
        type=float,
        default=0.0001,
        help='Slippage rate (default: 0.0001)'
    )
    
    parser.add_argument(
        '--walkforward',
        action='store_true',
        help='Run walkforward analysis'
    )
    
    parser.add_argument(
        '--monte-carlo',
        action='store_true',
        help='Run Monte Carlo simulation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--all-strategies',
        action='store_true',
        help='Test all available strategies'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load configuration
        config = ConfigManager()
        
        if args.config:
            config.load_config(args.config)
        
        # Parse dates
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        # Determine strategies to test
        if args.all_strategies or args.strategy == 'all':
            strategies = ['moving_average', 'rsi', 'macd', 'bollinger_bands']
        else:
            strategies = [args.strategy]
        
        # Create backtesting pipeline
        pipeline = BacktestingPipeline(config)
        
        all_results = {}
        
        # Test each strategy
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Testing strategy: {strategy}")
            print(f"{'='*60}")
            
            try:
                results = pipeline.run_complete_backtest(
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy,
                    model_id=args.model_id if strategy == 'model' else None,
                    initial_capital=args.initial_capital,
                    commission_rate=args.commission,
                    slippage=args.slippage,
                    run_walkforward=args.walkforward,
                    run_monte_carlo=args.monte_carlo,
                    output_dir=Path(args.output_dir) if args.output_dir else None
                )
                
                all_results[strategy] = results
                
            except Exception as e:
                print(f"Failed to test strategy {strategy}: {e}")
                continue
        
        # Print comparative analysis if multiple strategies tested
        if len(all_results) > 1:
            print_comparative_analysis(all_results)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nBacktesting interrupted by user")
        return 1
    except Exception as e:
        print(f"\nBacktesting failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def print_comparative_analysis(results: Dict[str, Dict[str, Any]]) -> None:
    """Print comparative analysis of multiple strategies"""
    print("\n" + "=" * 60)
    print("COMPARATIVE ANALYSIS")
    print("=" * 60)
    
    print(f"\n{'Strategy':<20} {'Return%':>10} {'Sharpe':>10} {'MaxDD%':>10} {'WinRate%':>10}")
    print("-" * 70)
    
    for strategy, result in results.items():
        backtest = result.get('backtest', {})
        print(f"{strategy:<20} "
              f"{backtest.get('total_return', 0)*100:>10.2f} "
              f"{backtest.get('sharpe_ratio', 0):>10.2f} "
              f"{backtest.get('max_drawdown', 0)*100:>10.2f} "
              f"{backtest.get('win_rate', 0)*100:>10.2f}")
    
    # Find best strategy by Sharpe ratio
    best_sharpe = -float('inf')
    best_strategy = None
    
    for strategy, result in results.items():
        sharpe = result.get('backtest', {}).get('sharpe_ratio', -float('inf'))
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_strategy = strategy
    
    if best_strategy:
        print(f"\nBest strategy by Sharpe ratio: {best_strategy} ({best_sharpe:.2f})")
    
    print(f"\nRecommendation: Consider using {best_strategy} for live trading "
          f"after further validation in current market conditions.")


if __name__ == "__main__":
    sys.exit(main())