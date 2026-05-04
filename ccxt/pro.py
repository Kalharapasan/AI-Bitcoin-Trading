"""Stub module to mimic `ccxt.pro` behaviour for tests.
Provides async constructors for exchanges which return a simple object.
"""
import asyncio

class _DummyProExchange:
    def __init__(self, opts=None):
        self.opts = opts or {}

    async def load_markets(self):
        return {}

    async def fetch_ohlcv(self, symbol, timeframe, since, limit):
        return []

    async def fetch_trades(self, symbol, since, limit):
        return []

    async def fetch_order_book(self, symbol, limit):
        return {'timestamp': None, 'bids': [], 'asks': []}

    async def fetch_ticker(self, symbol):
        return {'timestamp': None, 'last': 0, 'bid': 0, 'ask': 0, 'high': 0, 'low': 0, 'baseVolume': 0, 'quoteVolume': 0, 'change': 0, 'percentage': 0}

    async def close(self):
        return None


async def _make_exchange(opts=None):
    await asyncio.sleep(0)  # yield control
    return _DummyProExchange(opts)


def __getattr__(name: str):
    # Return an async callable that creates a dummy exchange instance
    return lambda opts=None: _make_exchange(opts)
