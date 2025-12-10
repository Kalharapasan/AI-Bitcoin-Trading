# Bitcoin Trading AI - Insider Rate Detection System

A sophisticated AI-powered Bitcoin trading application that uses deep learning models (Transformer, LSTM with Attention) to predict price movements and generate trading signals.

## 🌟 Features

- **Multiple Neural Network Models**: Transformer, LSTM with Attention, and Ensemble predictions
- **Real-time Data Collection**: Fetches live Bitcoin data from major exchanges
- **Advanced Technical Analysis**: 50+ technical indicators and price patterns
- **Risk Management**: Comprehensive risk analysis with position sizing, stop-loss, and portfolio optimization
- **Interactive Dashboard**: Real-time visualization of trades, predictions, and performance metrics
- **Backtesting Engine**: Test strategies on historical data
- **Signal Generation**: Automated buy/sell/hold signal generation with confidence scoring

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL (optional, for data storage)
- Redis (optional, for caching)
- CUDA-capable GPU (recommended for training)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd bitcoin-trading-ai
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install TA-Lib** (Required for technical analysis)

**On Ubuntu/Debian:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

**On macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**On Windows:**
Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

5. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

6. **Create required directories**
```bash
mkdir -p data/{raw,processed,cache} models logs results
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:
- Trading parameters (symbol, timeframes, capital)
- Model hyperparameters
- Risk management settings
- Technical indicators
- Exchange settings

Edit `.env` file with your:
- Exchange API keys (Binance, etc.)
- Database credentials
- Email/Telegram settings for alerts

## 📊 Usage

### 1. Collect Historical Data
```bash
python main.py collect
```

### 2. Train Models
```bash
python main.py train
```

### 3. Run Backtest
```bash
python main.py backtest
```

### 4. Launch Dashboard
```bash
python main.py dashboard
```
Open browser at `http://localhost:8050`

### 5. Start Live Trading
```bash
python main.py live
```

## 🏗️ Project Structure

```
bitcoin-trading-ai/
├── config/              # Configuration files
├── core/
│   ├── neural_networks/ # Deep learning models
│   ├── data_processing/ # Data collection & features
│   ├── trading/         # Trading logic & signals
│   ├── risk_management/ # Risk analysis & optimization
│   ├── models/          # Model management
│   └── utils/           # Helper utilities
├── web/
│   ├── dashboard/       # Interactive web dashboard
│   └── api/             # REST API endpoints
├── backtesting/         # Backtesting engine
├── strategies/          # Trading strategies
├── database/            # Database models
├── tests/               # Unit tests
├── data/                # Data storage
├── models/              # Saved models
└── logs/                # Log files
```

## 🤖 Models

### 1. Transformer Model
- Multi-head attention mechanism
- Positional encoding
- Best for capturing long-term dependencies

### 2. LSTM with Attention
- Bidirectional LSTM layers
- Attention mechanism for focus on important features
- Excellent for sequential data

### 3. Ensemble Model
- Combines predictions from multiple models
- Weighted or averaged predictions
- More robust and accurate

## 📈 Features Engineered

- **Price Indicators**: SMA, EMA, Returns, Log Returns
- **Momentum**: RSI, Stochastic, ROC, Momentum
- **Trend**: MACD, ADX, CCI
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume Ratio
- **Pattern Recognition**: Candlestick patterns
- **Time Features**: Hour, Day, Month (cyclical encoding)

## 🎯 Signal Generation

Signals are generated based on:
1. **Model Predictions**: Price direction and magnitude
2. **Technical Filters**: RSI, MACD, Bollinger Bands
3. **Trend Analysis**: Moving average crossovers
4. **Volume Confirmation**: Volume ratio analysis
5. **Confidence Scoring**: Multi-factor confidence calculation

## 🛡️ Risk Management

- **Position Sizing**: Kelly Criterion-based calculation
- **Stop Loss**: ATR-based or percentage-based
- **Take Profit**: Risk-reward ratio optimization
- **Drawdown Control**: Maximum drawdown limits
- **Portfolio Limits**: Maximum position size restrictions
- **Risk Metrics**: Sharpe ratio, Sortino ratio, VaR

## 📊 Dashboard Features

- Real-time price charts with candlesticks
- AI prediction visualization
- Recent trading signals
- Performance metrics (Win rate, Sharpe ratio, etc.)
- Open positions tracking
- Portfolio value and P&L
- Auto-refresh every 5 seconds

## 🔧 API Endpoints

```python
# Example API usage (when implemented)
GET  /api/price          # Get current price
GET  /api/predictions    # Get AI predictions
GET  /api/signals        # Get trading signals
POST /api/trade          # Execute trade
GET  /api/positions      # Get open positions
GET  /api/performance    # Get performance metrics
```

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

Run specific test:
```bash
pytest tests/test_models.py
```

## 📝 Logging

Logs are stored in `logs/` directory:
- `trading.log`: General application logs
- `trades.log`: All trade executions
- `errors.log`: Error logs

## ⚠️ Disclaimer

This software is for educational purposes only. Cryptocurrency trading carries significant financial risk. Never invest more than you can afford to lose. Past performance does not guarantee future results.

**Always test thoroughly with paper trading before using real funds.**

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check documentation
- Review logs for debugging

## 🔮 Future Enhancements

- [ ] Multi-cryptocurrency support
- [ ] Advanced portfolio optimization
- [ ] Sentiment analysis integration
- [ ] News feed integration
- [ ] Mobile app
- [ ] Cloud deployment
- [ ] Paper trading mode
- [ ] Strategy marketplace
- [ ] Real-time alerts (Telegram/Email)
- [ ] Advanced backtesting with Monte Carlo

## 📚 Resources

- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)
- [Dash Documentation](https://dash.plotly.com/)

---

**Built with ❤️ for crypto traders and AI enthusiasts**