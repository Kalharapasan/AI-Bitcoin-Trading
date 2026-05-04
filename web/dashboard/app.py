"""
Main application file for Bitcoin Trading Application.
This is the entry point that brings all modules together.
"""

import os
import sys
import json
import signal
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import warnings

# Third-party imports
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_cors import CORS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from logger import get_logger, setup_logger
from cache import TradingCache, CacheType, get_global_cache
from backtest_engine import BacktestEngine, BacktestConfig, SampleMovingAverageStrategy
from walkforward_analyzer import WalkForwardAnalyzer, WFAConfig, WFAOptimizationMethod, WFAMetric
from monte_carlo_simulator import MonteCarloSimulator, MonteCarloConfig, MonteCarloMethod

# Set up logger
logger = setup_logger(name="BitcoinTradingApp", log_dir="logs")
logger.info("Starting Bitcoin Trading Application")

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'bitcoin-trading-app-secret-key-2024')

# Initialize extensions
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configuration
class Config:
    """Application configuration."""
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    TESTING = os.environ.get('TESTING', 'False').lower() == 'true'
    SECRET_KEY = app.secret_key
    DATABASE_URI = os.environ.get('DATABASE_URI', 'sqlite:///trading_app.db')
    CACHE_DIR = 'cache'
    RESULTS_DIR = 'results'
    LOG_DIR = 'logs'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
    
    # Trading defaults
    DEFAULT_INITIAL_CAPITAL = 10000.0
    DEFAULT_TRADING_FEE = 0.001
    DEFAULT_SLIPPAGE = 0.0005
    DEFAULT_TIMEFRAME = '1h'
    DEFAULT_SYMBOLS = ['BTC/USDT', 'ETH/USDT']

app.config.from_object(Config)

# Ensure directories exist
for directory in [Config.CACHE_DIR, Config.RESULTS_DIR, Config.LOG_DIR, Config.UPLOAD_FOLDER]:
    Path(directory).mkdir(exist_ok=True)

# User model for authentication
class User(UserMixin):
    """Simple user model."""
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email
        self.created_at = datetime.now()
        self.last_login = datetime.now()

# Mock user database (in production, use a real database)
users = {
    '1': User('1', 'admin', 'admin@tradingapp.com'),
    '2': User('2', 'trader', 'trader@tradingapp.com')
}

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID."""
    return users.get(user_id)

# Global application state
class AppState:
    """Global application state."""
    def __init__(self):
        self.is_running = False
        self.is_backtesting = False
        self.is_wfa_running = False
        self.is_mc_running = False
        self.current_strategy = None
        self.market_data = {}
        self.active_trades = {}
        self.portfolio_value = Config.DEFAULT_INITIAL_CAPITAL
        self.cache = get_global_cache()
        self.backtest_engine = None
        self.wfa_analyzer = None
        self.mc_simulator = None
        self.strategies = {}
        self.data_sources = {}
        
        # Load available strategies
        self._load_strategies()
        
        logger.info("Application state initialized")

    def _load_strategies(self):
        """Load available trading strategies."""
        # Sample strategies
        self.strategies = {
            'moving_average': {
                'name': 'Moving Average Crossover',
                'description': 'Buys when fast MA crosses above slow MA, sells when crosses below',
                'class': SampleMovingAverageStrategy,
                'parameters': [
                    {'name': 'fast_period', 'type': 'int', 'default': 10, 'min': 5, 'max': 50},
                    {'name': 'slow_period', 'type': 'int', 'default': 30, 'min': 10, 'max': 100}
                ]
            },
            # Add more strategies here
        }
        logger.info(f"Loaded {len(self.strategies)} strategies")

# Initialize application state
app_state = AppState()

# Dashboard routes
@app.route('/')
@login_required
def index():
    """Main dashboard page."""
    return render_template('index.html', 
                         username=current_user.username,
                         portfolio_value=app_state.portfolio_value)

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard with charts and metrics."""
    return render_template('dashboard.html')

@app.route('/backtest')
@login_required
def backtest():
    """Backtesting interface."""
    return render_template('backtest.html', strategies=app_state.strategies)

@app.route('/walkforward')
@login_required
def walkforward():
    """Walk-forward analysis interface."""
    return render_template('walkforward.html', strategies=app_state.strategies)

@app.route('/montecarlo')
@login_required
def montecarlo():
    """Monte Carlo simulation interface."""
    return render_template('montecarlo.html')

@app.route('/live_trading')
@login_required
def live_trading():
    """Live trading interface."""
    return render_template('live_trading.html')

@app.route('/portfolio')
@login_required
def portfolio():
    """Portfolio management interface."""
    return render_template('portfolio.html')

@app.route('/settings')
@login_required
def settings():
    """Application settings."""
    return render_template('settings.html')

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Simple authentication (in production, use proper authentication)
        if username in ['admin', 'trader'] and password == 'password':
            user = users['1'] if username == 'admin' else users['2']
            login_user(user)
            logger.info(f"User {username} logged in")
            return redirect(url_for('index'))
        
        return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout."""
    logout_user()
    logger.info(f"User {current_user.username} logged out")
    return redirect(url_for('login'))

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/market_data', methods=['GET'])
@login_required
def get_market_data():
    """Get market data for symbols."""
    symbols = request.args.get('symbols', 'BTC/USDT').split(',')
    timeframe = request.args.get('timeframe', '1h')
    limit = int(request.args.get('limit', 100))
    
    # Mock market data (in production, fetch from exchange API)
    data = {}
    for symbol in symbols:
        # Generate mock OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)
        prices = 50000 + np.cumsum(np.random.randn(limit) * 1000)
        
        data[symbol] = {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': [{
                'timestamp': d.isoformat(),
                'open': price + np.random.randn() * 50,
                'high': price + abs(np.random.randn() * 100),
                'low': price - abs(np.random.randn() * 100),
                'close': price,
                'volume': np.random.rand() * 1000
            } for d, price in zip(dates, prices)]
        }
    
    return jsonify(data)

@app.route('/api/strategies', methods=['GET'])
@login_required
def get_strategies():
    """Get available trading strategies."""
    strategies_list = []
    for strategy_id, strategy_info in app_state.strategies.items():
        strategies_list.append({
            'id': strategy_id,
            'name': strategy_info['name'],
            'description': strategy_info['description'],
            'parameters': strategy_info['parameters']
        })
    
    return jsonify({'strategies': strategies_list})

@app.route('/api/backtest/run', methods=['POST'])
@login_required
def run_backtest():
    """Run a backtest."""
    try:
        data = request.get_json()
        
        # Extract backtest parameters
        strategy_id = data.get('strategy_id', 'moving_average')
        strategy_params = data.get('strategy_params', {})
        initial_capital = float(data.get('initial_capital', Config.DEFAULT_INITIAL_CAPITAL))
        trading_fee = float(data.get('trading_fee', Config.DEFAULT_TRADING_FEE))
        slippage = float(data.get('slippage', Config.DEFAULT_SLIPPAGE))
        symbols = data.get('symbols', Config.DEFAULT_SYMBOLS)
        timeframe = data.get('timeframe', Config.DEFAULT_TIMEFRAME)
        
        # Create backtest configuration
        config = BacktestConfig(
            initial_capital=initial_capital,
            trading_fee=trading_fee,
            slippage=slippage,
            symbols=symbols,
            timeframe=timeframe,
            verbose=True
        )
        
        # Create and run backtest engine
        engine = BacktestEngine(config)
        
        # Load mock data (in production, load real data)
        for symbol in symbols:
            # Generate mock OHLCV data
            dates = pd.date_range(end=datetime.now(), periods=1000, freq=timeframe)
            prices = 50000 + np.cumsum(np.random.randn(1000) * 1000)
            
            df = pd.DataFrame({
                'open': prices + np.random.randn(1000) * 50,
                'high': prices + abs(np.random.randn(1000) * 100),
                'low': prices - abs(np.random.randn(1000) * 100),
                'close': prices,
                'volume': np.random.rand(1000) * 1000
            }, index=dates)
            
            engine.load_data(df, symbol)
        
        # Add strategy
        strategy_info = app_state.strategies.get(strategy_id)
        if not strategy_info:
            return jsonify({'error': f'Strategy {strategy_id} not found'}), 404
        
        strategy_class = strategy_info['class']
        engine.add_strategy(strategy_class, **strategy_params)
        
        # Run backtest
        app_state.is_backtesting = True
        results = engine.run()
        app_state.is_backtesting = False
        
        # Store results in cache
        cache_key = f"backtest_{datetime.now().timestamp()}"
        app_state.cache.set(cache_key, results, ttl=3600)  # 1 hour
        
        # Prepare response
        response = {
            'success': True,
            'cache_key': cache_key,
            'summary': results['summary'],
            'metrics': results['metrics']
        }
        
        logger.info(f"Backtest completed: {results['summary']['total_return_percentage']:.2f}% return")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        app_state.is_backtesting = False
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest/results/<cache_key>', methods=['GET'])
@login_required
def get_backtest_results(cache_key):
    """Get backtest results from cache."""
    try:
        results = app_state.cache.get(cache_key)
        if not results:
            return jsonify({'error': 'Results not found or expired'}), 404
        
        # Prepare simplified results for frontend
        simplified = {
            'summary': results.get('summary', {}),
            'metrics': results.get('metrics', {}),
            'trades_count': len(results.get('trades', [])),
            'equity_curve': results.get('equity_curve', []),
            'drawdown_curve': results.get('drawdown_curve', [])
        }
        
        return jsonify(simplified)
    
    except Exception as e:
        logger.error(f"Error getting backtest results: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/walkforward/run', methods=['POST'])
@login_required
def run_walkforward():
    """Run walk-forward analysis."""
    try:
        data = request.get_json()
        
        # Extract WFA parameters
        strategy_id = data.get('strategy_id', 'moving_average')
        strategy_params = data.get('strategy_params', {})
        initial_train_size = int(data.get('initial_train_size', 252))
        test_size = int(data.get('test_size', 63))
        step_size = int(data.get('step_size', 21))
        optimization_method = data.get('optimization_method', 'grid_search')
        optimization_metric = data.get('optimization_metric', 'sharpe_ratio')
        
        # Convert string to enum
        method_map = {
            'grid_search': WFAOptimizationMethod.GRID_SEARCH,
            'random_search': WFAOptimizationMethod.RANDOM_SEARCH,
            'bayesian': WFAOptimizationMethod.BAYESIAN_OPTIMIZATION
        }
        
        metric_map = {
            'sharpe_ratio': WFAMetric.SHARPE_RATIO,
            'total_return': WFAMetric.TOTAL_RETURN,
            'max_drawdown': WFAMetric.MAX_DRAWDOWN
        }
        
        # Create WFA configuration
        wfa_config = WFAConfig(
            initial_train_size=initial_train_size,
            test_size=test_size,
            step_size=step_size,
            optimization_method=method_map.get(optimization_method, WFAOptimizationMethod.GRID_SEARCH),
            optimization_metric=metric_map.get(optimization_metric, WFAMetric.SHARPE_RATIO),
            param_grid={
                'fast_period': [5, 10, 20],
                'slow_period': [20, 30, 50]
            },
            backtest_config=BacktestConfig(
                initial_capital=Config.DEFAULT_INITIAL_CAPITAL,
                trading_fee=Config.DEFAULT_TRADING_FEE,
                verbose=False
            ),
            parallel=True,
            max_workers=2
        )
        
        # Create and run WFA analyzer
        analyzer = WalkForwardAnalyzer(wfa_config)
        
        # Load mock data
        symbol = 'BTC/USDT'
        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='1h')
        prices = 50000 + np.cumsum(np.random.randn(len(dates)) * 100)
        
        df = pd.DataFrame({
            'open': prices + np.random.randn(len(dates)) * 50,
            'high': prices + abs(np.random.randn(len(dates)) * 100),
            'low': prices - abs(np.random.randn(len(dates)) * 100),
            'close': prices,
            'volume': np.random.rand(len(dates)) * 1000
        }, index=dates)
        
        analyzer.load_data(df, symbol)
        
        # Set strategy
        strategy_info = app_state.strategies.get(strategy_id)
        if not strategy_info:
            return jsonify({'error': f'Strategy {strategy_id} not found'}), 404
        
        analyzer.set_strategy(strategy_info['class'])
        
        # Run analysis
        app_state.is_wfa_running = True
        results = analyzer.run_analysis()
        app_state.is_wfa_running = False
        
        # Store results in cache
        cache_key = f"wfa_{datetime.now().timestamp()}"
        app_state.cache.set(cache_key, results.to_dict(), ttl=3600)
        
        # Get summary
        summary = analyzer.get_summary()
        
        response = {
            'success': True,
            'cache_key': cache_key,
            'summary': summary,
            'window_count': len(results.window_results)
        }
        
        logger.info(f"Walk-forward analysis completed: {summary['overall_performance']} performance")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Walk-forward analysis error: {e}")
        app_state.is_wfa_running = False
        return jsonify({'error': str(e)}), 500

@app.route('/api/montecarlo/run', methods=['POST'])
@login_required
def run_montecarlo():
    """Run Monte Carlo simulation."""
    try:
        data = request.get_json()
        
        # Extract MC parameters
        simulations = int(data.get('simulations', 1000))
        time_horizon = int(data.get('time_horizon', 252))
        initial_capital = float(data.get('initial_capital', Config.DEFAULT_INITIAL_CAPITAL))
        confidence_level = float(data.get('confidence_level', 0.95))
        method = data.get('method', 'historical_bootstrap')
        
        # Convert string to enum
        method_map = {
            'historical_bootstrap': MonteCarloMethod.HISTORICAL_BOOTSTRAP,
            'parametric': MonteCarloMethod.PARAMETRIC,
            'garch': MonteCarloMethod.GARCH,
            'geometric_brownian': MonteCarloMethod.GEOMETRIC_BROWNIAN
        }
        
        # Create MC configuration
        mc_config = MonteCarloConfig(
            method=method_map.get(method, MonteCarloMethod.HISTORICAL_BOOTSTRAP),
            simulations=simulations,
            time_horizon=time_horizon,
            initial_capital=initial_capital,
            confidence_level=confidence_level,
            parallel=True,
            max_workers=2
        )
        
        # Create and run MC simulator
        simulator = MonteCarloSimulator(mc_config)
        
        # Generate mock returns data
        np.random.seed(42)
        n_periods = 1000
        drift = 0.0005  # 0.05% daily
        volatility = 0.02  # 2% daily
        
        returns = np.random.normal(drift, volatility, n_periods)
        prices = 50000 * np.cumprod(1 + returns)
        
        simulator.load_returns(pd.Series(returns), pd.Series(prices))
        
        # Run simulation
        app_state.is_mc_running = True
        results = simulator.run_simulation()
        app_state.is_mc_running = False
        
        # Store results in cache
        cache_key = f"montecarlo_{datetime.now().timestamp()}"
        app_state.cache.set(cache_key, results.to_dict(), ttl=3600)
        
        # Get summary
        summary = simulator.get_summary()
        
        response = {
            'success': True,
            'cache_key': cache_key,
            'summary': summary,
            'simulation_count': len(results.simulation_results)
        }
        
        logger.info(f"Monte Carlo simulation completed: {summary['overall_performance']} performance")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Monte Carlo simulation error: {e}")
        app_state.is_mc_running = False
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
@login_required
def get_portfolio():
    """Get current portfolio information."""
    # Mock portfolio data (in production, fetch from database or exchange)
    portfolio = {
        'total_value': app_state.portfolio_value,
        'cash': app_state.portfolio_value * 0.3,
        'positions': [
            {
                'symbol': 'BTC/USDT',
                'quantity': 0.5,
                'current_price': 50000,
                'value': 25000,
                'pnl': 2500,
                'pnl_percentage': 10.0
            },
            {
                'symbol': 'ETH/USDT',
                'quantity': 5.0,
                'current_price': 3000,
                'value': 15000,
                'pnl': 1500,
                'pnl_percentage': 10.0
            }
        ],
        'performance': {
            'daily_return': 2.5,
            'weekly_return': 5.8,
            'monthly_return': 12.3,
            'yearly_return': 45.6,
            'sharpe_ratio': 1.8,
            'max_drawdown': 8.7
        }
    }
    
    return jsonify(portfolio)

@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def handle_settings():
    """Handle application settings."""
    if request.method == 'GET':
        # Return current settings
        settings = {
            'trading': {
                'default_capital': Config.DEFAULT_INITIAL_CAPITAL,
                'default_fee': Config.DEFAULT_TRADING_FEE,
                'default_slippage': Config.DEFAULT_SLIPPAGE
            },
            'display': {
                'theme': 'dark',
                'refresh_rate': 5
            },
            'notifications': {
                'email_alerts': True,
                'price_alerts': False,
                'trade_alerts': True
            }
        }
        return jsonify(settings)
    
    else:  # POST
        # Update settings
        data = request.get_json()
        # In production, save to database
        logger.info(f"Settings updated: {data}")
        return jsonify({'success': True, 'message': 'Settings updated'})

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to trading server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('subscribe_market')
def handle_subscribe_market(data):
    """Subscribe to market data updates."""
    symbol = data.get('symbol', 'BTC/USDT')
    timeframe = data.get('timeframe', '1m')
    
    logger.info(f"Client subscribed to {symbol} {timeframe}")
    
    # Start sending mock market data (in production, connect to real data feed)
    def send_market_updates():
        while True:
            time.sleep(1)  # Update every second
            price = 50000 + np.random.randn() * 100
            update = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'price': price,
                'change': np.random.randn() * 0.1,
                'volume': np.random.rand() * 1000
            }
            socketio.emit('market_update', update)
    
    # Start update thread
    thread = threading.Thread(target=send_market_updates, daemon=True)
    thread.start()

@socketio.on('place_order')
@login_required
def handle_place_order(data):
    """Handle order placement."""
    try:
        symbol = data.get('symbol')
        side = data.get('side')
        order_type = data.get('order_type', 'market')
        quantity = float(data.get('quantity', 0))
        price = float(data.get('price', 0))
        
        # Validate order
        if quantity <= 0:
            emit('order_error', {'message': 'Invalid quantity'})
            return
        
        # Mock order execution (in production, send to exchange)
        order_id = f"order_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        # Simulate execution
        executed_price = price if price > 0 else 50000 + np.random.randn() * 50
        
        order_result = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'order_type': order_type,
            'quantity': quantity,
            'price': executed_price,
            'status': 'filled',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Order placed: {order_result}")
        
        # Update portfolio value
        if side == 'buy':
            app_state.portfolio_value -= quantity * executed_price
        else:  # sell
            app_state.portfolio_value += quantity * executed_price
        
        emit('order_confirmation', order_result)
        socketio.emit('portfolio_update', {'value': app_state.portfolio_value})
        
    except Exception as e:
        logger.error(f"Order error: {e}")
        emit('order_error', {'message': str(e)})

# Dashboard visualization routes
@app.route('/api/charts/equity_curve', methods=['GET'])
@login_required
def get_equity_chart():
    """Generate equity curve chart."""
    try:
        # Generate mock equity curve
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                            end=datetime.now(), 
                            periods=100)
        
        # Simulate equity curve with some noise
        base = 10000
        returns = np.random.normal(0.001, 0.02, len(dates) - 1)
        equity = [base]
        
        for ret in returns:
            equity.append(equity[-1] * (1 + ret))
        
        # Create Plotly figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2E86AB', width=2)
        ))
        
        # Add buy/sell markers (mock)
        buy_dates = dates[::10]
        sell_dates = dates[5::10]
        
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=[equity[i] for i in range(0, len(equity), 10)],
            mode='markers',
            name='Buy Signals',
            marker=dict(color='#00FF00', size=10, symbol='triangle-up')
        ))
        
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=[equity[i] for i in range(5, len(equity), 10)],
            mode='markers',
            name='Sell Signals',
            marker=dict(color='#FF0000', size=10, symbol='triangle-down')
        ))
        
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_dark',
            hovermode='x unified'
        )
        
        return jsonify(fig.to_json())
    
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/performance_metrics', methods=['GET'])
@login_required
def get_performance_chart():
    """Generate performance metrics chart."""
    try:
        # Mock performance metrics
        metrics = {
            'Sharpe Ratio': 1.8,
            'Sortino Ratio': 2.1,
            'Calmar Ratio': 1.5,
            'Max Drawdown': -8.7,
            'Win Rate': 62.5,
            'Profit Factor': 1.8
        }
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#386FA4']
        ))
        
        fig.update_layout(
            title='Performance Metrics',
            xaxis_title='Metric',
            yaxis_title='Value',
            template='plotly_dark'
        )
        
        return jsonify(fig.to_json())
    
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/risk_analysis', methods=['GET'])
@login_required
def get_risk_chart():
    """Generate risk analysis chart."""
    try:
        # Mock VaR data
        confidence_levels = [0.90, 0.95, 0.99]
        var_values = [-5.2, -7.8, -12.3]  # Percent
        
        # Create VaR chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f'{cl*100:.0f}%' for cl in confidence_levels],
            y=var_values,
            marker_color=['#6A994E', '#F18F01', '#C73E1D'],
            text=[f'{v:.1f}%' for v in var_values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Value at Risk (VaR) Analysis',
            xaxis_title='Confidence Level',
            yaxis_title='VaR (%)',
            template='plotly_dark'
        )
        
        return jsonify(fig.to_json())
    
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return jsonify({'error': str(e)}), 500

# Data management routes
@app.route('/api/data/upload', methods=['POST'])
@login_required
def upload_data():
    """Upload trading data file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith(('.csv', '.json', '.parquet')):
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Save uploaded file
        filename = f"{int(time.time())}_{file.filename}"
        filepath = Path(Config.UPLOAD_FOLDER) / filename
        file.save(filepath)
        
        # Process file based on format
        if file.filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif file.filename.endswith('.json'):
            df = pd.read_json(filepath)
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'}), 400
        
        # Cache the data
        cache_key = f"uploaded_data_{filename}"
        app_state.cache.set(cache_key, df.to_dict('records'), ttl=86400)  # 24 hours
        
        response = {
            'success': True,
            'filename': filename,
            'cache_key': cache_key,
            'rows': len(df),
            'columns': list(df.columns),
            'date_range': {
                'start': df.index[0].isoformat() if hasattr(df.index[0], 'isoformat') else str(df.index[0]),
                'end': df.index[-1].isoformat() if hasattr(df.index[-1], 'isoformat') else str(df.index[-1])
            }
        }
        
        logger.info(f"Data uploaded: {filename} ({len(df)} rows)")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Data upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/list', methods=['GET'])
@login_required
def list_data():
    """List uploaded data files."""
    try:
        upload_dir = Path(Config.UPLOAD_FOLDER)
        files = []
        
        for filepath in upload_dir.glob('*'):
            if filepath.is_file():
                stat = filepath.stat()
                files.append({
                    'name': filepath.name,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return jsonify({'files': files})
    
    except Exception as e:
        logger.error(f"Data list error: {e}")
        return jsonify({'error': str(e)}), 500

# Utility functions
def cleanup_resources():
    """Clean up resources on shutdown."""
    logger.info("Cleaning up resources...")
    
    # Stop any running processes
    app_state.is_running = False
    app_state.is_backtesting = False
    app_state.is_wfa_running = False
    app_state.is_mc_running = False
    
    # Clear cache
    app_state.cache.clear()
    
    logger.info("Cleanup completed")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    cleanup_resources()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Initialize Dash dashboard
def init_dash_app(server):
    """Initialize Dash application for advanced analytics."""
    dash_app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname='/analytics/',
        external_stylesheets=[dbc.themes.DARKLY]
    )
    
    # Dash layout
    dash_app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Advanced Analytics Dashboard", className="text-center mb-4"), width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Strategy Performance"),
                    dbc.CardBody([
                        dcc.Graph(id='strategy-performance-graph')
                    ])
                ])
            ], width=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Risk Metrics"),
                    dbc.CardBody([
                        dcc.Graph(id='risk-metrics-graph')
                    ])
                ])
            ], width=4)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Portfolio Allocation"),
                    dbc.CardBody([
                        dcc.Graph(id='portfolio-allocation-graph')
                    ])
                ])
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Trade History"),
                    dbc.CardBody([
                        dcc.Graph(id='trade-history-graph')
                    ])
                ])
            ], width=6)
        ])
    ], fluid=True)
    
    return dash_app

# Initialize Dash
dash_app = init_dash_app(app)

# Main entry point
if __name__ == '__main__':
    # Print startup banner
    print("\n" + "="*60)
    print("BITCOIN TRADING APPLICATION")
    print("="*60)
    print(f"Version: 1.0.0")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log directory: {Config.LOG_DIR}")
    print(f"Cache directory: {Config.CACHE_DIR}")
    print(f"Results directory: {Config.RESULTS_DIR}")
    print("="*60)
    print("\nAvailable routes:")
    print("  • http://localhost:5000/ - Main dashboard")
    print("  • http://localhost:5000/backtest - Backtesting interface")
    print("  • http://localhost:5000/walkforward - Walk-forward analysis")
    print("  • http://localhost:5000/montecarlo - Monte Carlo simulation")
    print("  • http://localhost:5000/live_trading - Live trading")
    print("  • http://localhost:5000/analytics/ - Advanced analytics")
    print("  • http://localhost:5000/api/health - Health check")
    print("\nDefault credentials:")
    print("  • Username: admin, Password: password")
    print("  • Username: trader, Password: password")
    print("="*60 + "\n")
    
    # Run the application
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=Config.DEBUG,
            use_reloader=False
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        cleanup_resources()