"""Lightweight stub of ccxt for tests and basic offline operation.
This stub provides minimal classes and exceptions used by the project during tests.
Do not use in production — install the real `ccxt` package instead.
"""
class NetworkError(Exception):
    pass

class ExchangeError(Exception):
    pass

class _DummyExchange:
    def __init__(self, opts=None):
        self.opts = opts or {}

    def load_markets(self):
        return {}

    def fetch_ticker(self, symbol):
        return {'timestamp': None, 'last': 0, 'bid': 0, 'ask': 0, 'high': 0, 'low': 0, 'baseVolume': 0, 'quoteVolume': 0, 'change': 0, 'percentage': 0}

    def close(self):
        return None


# Expose common exchange names as simple constructors returning _DummyExchange
def __getattr__(name: str):
    return lambda opts=None: _DummyExchange(opts)
