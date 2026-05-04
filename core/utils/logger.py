"""
Logger utilities for Bitcoin Trading AI
Provides logging functionality for trading operations
"""

import logging
from typing import Optional, Dict, Any


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance with proper configuration"""
    if name is None:
        name = __name__
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def trade_signal(symbol: str, signal_type: str, strength: float):
    """Log trade signal"""
    logger = get_logger()
    logger.info(f"SIGNAL: {symbol} - {signal_type} (strength: {strength})")


def trade_execution(symbol: str, order_type: str, side: str, quantity: float, price: float):
    """Log trade execution"""
    logger = get_logger()
    logger.info(f"TRADE: {symbol} - {side} {quantity} at ${price} ({order_type})")


class TradeFormatter(logging.Formatter):
    """Custom formatter for trade logs"""
    
    def format(self, record):
        return super().format(record)