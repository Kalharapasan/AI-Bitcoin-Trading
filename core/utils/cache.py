"""
Cache module for Bitcoin Trading Application.
Provides in-memory and persistent caching with TTL support for market data,
indicators, and trade signals.
"""

import os
import json
import pickle
import time
import threading
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import sqlite3
from abc import ABC, abstractmethod
import asyncio
import inspect
from dataclasses import dataclass, asdict
from enum import Enum

from logger import get_logger

logger = get_logger(__name__)

class CacheType(Enum):
    """Types of cache storage."""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"  # For future Redis implementation
    HYBRID = "hybrid"

class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live

@dataclass
class CacheItem:
    """Represents a cached item with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None  # Time to live in seconds
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def is_expired(self) -> bool:
        """Check if the cache item has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "ttl": self.ttl,
            "tags": self.tags
        }

class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: List[str] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache."""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """Get all cache keys."""
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """Get number of items in cache."""
        pass

class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 10000, policy: CachePolicy = CachePolicy.LRU):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of items to store
            policy: Cache eviction policy
        """
        self._cache: Dict[str, CacheItem] = {}
        self.max_size = max_size
        self.policy = policy
        self._lock = threading.RLock()
        self._access_order: List[str] = []  # For LRU tracking
        self._access_frequency: Dict[str, int] = {}  # For LFU tracking
        
        logger.debug(f"Initialized MemoryCacheBackend with max_size={max_size}, policy={policy}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            item = self._cache[key]
            
            # Check if expired
            if item.is_expired():
                self._delete_item(key)
                return None
            
            # Update access metadata
            item.last_accessed = time.time()
            item.access_count += 1
            
            # Update policy tracking
            if self.policy == CachePolicy.LRU:
                self._update_lru(key)
            elif self.policy == CachePolicy.LFU:
                self._access_frequency[key] = self._access_frequency.get(key, 0) + 1
            
            return item.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: List[str] = None) -> bool:
        """Set value in memory cache."""
        with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict()
            
            # Create cache item
            current_time = time.time()
            item = CacheItem(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=0,
                ttl=ttl,
                tags=tags or []
            )
            
            # Store item
            self._cache[key] = item
            
            # Update policy tracking
            if self.policy == CachePolicy.LRU:
                self._update_lru(key)
            elif self.policy == CachePolicy.LFU:
                self._access_frequency[key] = 0
            
            logger.debug(f"Cached item: {key} (ttl={ttl}, tags={tags})")
            return True
    
    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        with self._lock:
            if key in self._cache:
                self._delete_item(key)
                logger.debug(f"Deleted cache item: {key}")
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        with self._lock:
            if key not in self._cache:
                return False
            
            item = self._cache[key]
            if item.is_expired():
                self._delete_item(key)
                return False
            
            return True
    
    def clear(self) -> bool:
        """Clear all memory cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._access_frequency.clear()
            logger.debug("Cleared all cache items")
            return True
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            # Clean expired items first
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired_keys:
                self._delete_item(key)
            
            return list(self._cache.keys())
    
    def get_size(self) -> int:
        """Get number of items in cache."""
        with self._lock:
            return len(self._cache)
    
    def _delete_item(self, key: str) -> None:
        """Delete item and clean up tracking."""
        if key in self._cache:
            del self._cache[key]
        
        if self.policy == CachePolicy.LRU and key in self._access_order:
            self._access_order.remove(key)
        
        if self.policy == CachePolicy.LFU and key in self._access_frequency:
            del self._access_frequency[key]
    
    def _update_lru(self, key: str) -> None:
        """Update LRU access order."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict(self) -> None:
        """Evict an item based on cache policy."""
        if not self._cache:
            return
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used
            if self._access_order:
                key_to_evict = self._access_order[0]
                self._delete_item(key_to_evict)
        
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            if self._access_frequency:
                key_to_evict = min(self._access_frequency.items(), key=lambda x: x[1])[0]
                self._delete_item(key_to_evict)
        
        elif self.policy == CachePolicy.FIFO:
            # Remove oldest (based on creation time)
            key_to_evict = min(self._cache.items(), key=lambda x: x[1].created_at)[0]
            self._delete_item(key_to_evict)
        
        elif self.policy == CachePolicy.TTL:
            # Remove expired items first, then oldest
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            if expired_keys:
                for key in expired_keys[:1]:  # Remove one expired
                    self._delete_item(key)
            else:
                # Remove oldest
                key_to_evict = min(self._cache.items(), key=lambda x: x[1].created_at)[0]
                self._delete_item(key_to_evict)
        
        logger.debug(f"Evicted item from cache (policy: {self.policy})")

class DiskCacheBackend(CacheBackend):
    """Disk-based cache backend using SQLite."""
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 100):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory to store cache database
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.db_path = self.cache_dir / "cache.db"
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        self._lock = threading.RLock()
        self._init_database()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.debug(f"Initialized DiskCacheBackend at {self.db_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    ttl REAL,
                    tags TEXT,
                    size INTEGER
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache(last_accessed)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags ON cache(tags)')
            
            conn.commit()
            conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT value, created_at, ttl FROM cache WHERE key = ?',
                (key,)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return None
            
            value_blob, created_at, ttl = result
            
            # Check if expired
            if ttl is not None and time.time() > (created_at + ttl):
                self.delete(key)
                return None
            
            # Update access metadata
            self._update_access_metadata(key)
            
            # Deserialize value
            try:
                value = pickle.loads(value_blob)
                return value
            except (pickle.PickleError, EOFError) as e:
                logger.error(f"Failed to deserialize cache value for key {key}: {e}")
                self.delete(key)
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: List[str] = None) -> bool:
        """Set value in disk cache."""
        with self._lock:
            # Check cache size
            self._enforce_size_limit()
            
            # Serialize value
            try:
                value_blob = pickle.dumps(value)
            except (pickle.PickleError, TypeError) as e:
                logger.error(f"Failed to serialize value for caching: {e}")
                return False
            
            current_time = time.time()
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Insert or replace
            cursor.execute('''
                INSERT OR REPLACE INTO cache 
                (key, value, created_at, last_accessed, access_count, ttl, tags, size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                key,
                value_blob,
                current_time,
                current_time,
                0,
                ttl,
                json.dumps(tags or []),
                len(value_blob)
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Cached item to disk: {key} (size: {len(value_blob)} bytes)")
            return True
    
    def delete(self, key: str) -> bool:
        """Delete value from disk cache."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
            deleted = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            if deleted:
                logger.debug(f"Deleted cache item from disk: {key}")
            
            return deleted
    
    def exists(self, key: str) -> bool:
        """Check if key exists in disk cache."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT 1 FROM cache WHERE key = ? AND (ttl IS NULL OR created_at + ttl > ?)',
                (key, time.time())
            )
            
            result = cursor.fetchone() is not None
            conn.close()
            
            return result
    
    def clear(self) -> bool:
        """Clear all disk cache."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM cache')
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Cleared {deleted_count} items from disk cache")
            return True
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Clean expired items first
            self._clean_expired()
            
            cursor.execute('SELECT key FROM cache')
            keys = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return keys
    
    def get_size(self) -> int:
        """Get number of items in cache."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM cache')
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
    
    def get_total_size_bytes(self) -> int:
        """Get total cache size in bytes."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT SUM(size) FROM cache')
            total_size = cursor.fetchone()[0] or 0
            
            conn.close()
            return total_size
    
    def _update_access_metadata(self, key: str) -> None:
        """Update access metadata for a key."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE cache 
            SET last_accessed = ?, access_count = access_count + 1
            WHERE key = ?
        ''', (time.time(), key))
        
        conn.commit()
        conn.close()
    
    def _clean_expired(self) -> None:
        """Clean expired cache items."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            'DELETE FROM cache WHERE ttl IS NOT NULL AND created_at + ttl <= ?',
            (time.time(),)
        )
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            logger.debug(f"Cleaned {deleted_count} expired items from disk cache")
    
    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit by deleting oldest items."""
        current_size = self.get_total_size_bytes()
        
        if current_size > self.max_size_bytes:
            logger.debug(f"Cache size {current_size} bytes exceeds limit {self.max_size_bytes} bytes")
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Delete oldest items until under limit
            cursor.execute('''
                SELECT key, size FROM cache 
                ORDER BY last_accessed ASC
            ''')
            
            deleted_size = 0
            deleted_count = 0
            
            for key, size in cursor.fetchall():
                if current_size - deleted_size <= self.max_size_bytes * 0.9:  # Leave 10% buffer
                    break
                
                cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
                deleted_size += size
                deleted_count += 1
            
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.debug(f"Freed {deleted_size} bytes by deleting {deleted_count} old items")
    
    def _periodic_cleanup(self) -> None:
        """Periodically clean expired items."""
        while True:
            time.sleep(300)  # Clean every 5 minutes
            try:
                self._clean_expired()
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")

class HybridCacheBackend(CacheBackend):
    """Hybrid cache backend with memory and disk layers."""
    
    def __init__(self, 
                 memory_max_size: int = 1000,
                 disk_max_size_mb: int = 100,
                 memory_policy: CachePolicy = CachePolicy.LRU):
        """
        Initialize hybrid cache.
        
        Args:
            memory_max_size: Maximum items in memory cache
            disk_max_size_mb: Maximum size of disk cache in MB
            memory_policy: Memory cache eviction policy
        """
        self.memory_cache = MemoryCacheBackend(
            max_size=memory_max_size,
            policy=memory_policy
        )
        self.disk_cache = DiskCacheBackend(
            cache_dir="cache",
            max_size_mb=disk_max_size_mb
        )
        
        logger.debug("Initialized HybridCacheBackend")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from hybrid cache (memory first, then disk)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: List[str] = None) -> bool:
        """Set value in both memory and disk cache."""
        # Set in memory cache
        memory_success = self.memory_cache.set(key, value, ttl, tags)
        
        # Set in disk cache (for persistence)
        disk_success = self.disk_cache.set(key, value, ttl, tags)
        
        return memory_success and disk_success
    
    def delete(self, key: str) -> bool:
        """Delete value from both memory and disk cache."""
        memory_success = self.memory_cache.delete(key)
        disk_success = self.disk_cache.delete(key)
        
        return memory_success or disk_success
    
    def exists(self, key: str) -> bool:
        """Check if key exists in either cache."""
        return self.memory_cache.exists(key) or self.disk_cache.exists(key)
    
    def clear(self) -> bool:
        """Clear both memory and disk cache."""
        memory_success = self.memory_cache.clear()
        disk_success = self.disk_cache.clear()
        
        return memory_success and disk_success
    
    def keys(self) -> List[str]:
        """Get all keys from both caches (unique)."""
        memory_keys = set(self.memory_cache.keys())
        disk_keys = set(self.disk_cache.keys())
        
        return list(memory_keys.union(disk_keys))
    
    def get_size(self) -> int:
        """Get total number of unique items in cache."""
        return len(self.keys())

class TradingCache:
    """Main trading cache class with specialized methods for trading data."""
    
    def __init__(self, 
                 cache_type: CacheType = CacheType.HYBRID,
                 **kwargs):
        """
        Initialize trading cache.
        
        Args:
            cache_type: Type of cache to use
            **kwargs: Additional arguments for cache backend
        """
        self.cache_type = cache_type
        
        if cache_type == CacheType.MEMORY:
            self.backend = MemoryCacheBackend(**kwargs)
        elif cache_type == CacheType.DISK:
            self.backend = DiskCacheBackend(**kwargs)
        elif cache_type == CacheType.HYBRID:
            self.backend = HybridCacheBackend(**kwargs)
        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")
        
        logger.info(f"Initialized TradingCache with type: {cache_type}")
    
    # Basic cache operations
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        value = self.backend.get(key)
        return value if value is not None else default
    
    def set(self, 
            key: str, 
            value: Any, 
            ttl: Optional[float] = None,
            tags: List[str] = None) -> bool:
        """Set value in cache."""
        return self.backend.set(key, value, ttl, tags)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return self.backend.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.backend.exists(key)
    
    def clear(self) -> bool:
        """Clear all cache."""
        return self.backend.clear()
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        return self.backend.keys()
    
    def get_size(self) -> int:
        """Get number of items in cache."""
        return self.backend.get_size()
    
    # Specialized trading cache methods
    def cache_ohlcv(self, 
                   symbol: str,
                   timeframe: str,
                   ohlcv_data: List[List[float]],
                   source: str = "exchange") -> bool:
        """
        Cache OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            ohlcv_data: List of OHLCV candles
            source: Data source (e.g., "exchange", "database")
        
        Returns:
            bool: Success status
        """
        key = f"ohlcv:{symbol}:{timeframe}:{source}"
        
        # Cache with 1-minute TTL for minute data, longer for higher timeframes
        if timeframe.endswith('m'):
            ttl = 60  # 1 minute
        elif timeframe.endswith('h'):
            ttl = 3600  # 1 hour
        elif timeframe.endswith('d'):
            ttl = 86400  # 1 day
        else:
            ttl = 300  # 5 minutes default
        
        tags = ["ohlcv", symbol, timeframe, source]
        
        success = self.set(key, ohlcv_data, ttl=ttl, tags=tags)
        
        if success:
            logger.debug(f"Cached OHLCV data: {symbol} {timeframe} ({len(ohlcv_data)} candles)")
        
        return success
    
    def get_ohlcv(self, 
                 symbol: str,
                 timeframe: str,
                 source: str = "exchange") -> Optional[List[List[float]]]:
        """
        Get cached OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            source: Data source
        
        Returns:
            Optional OHLCV data
        """
        key = f"ohlcv:{symbol}:{timeframe}:{source}"
        data = self.get(key)
        
        if data:
            logger.debug(f"Retrieved cached OHLCV data: {symbol} {timeframe}")
        
        return data
    
    def cache_indicator(self,
                       symbol: str,
                       timeframe: str,
                       indicator_name: str,
                       indicator_params: Dict[str, Any],
                       indicator_data: Any) -> bool:
        """
        Cache technical indicator data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            indicator_name: Name of indicator (e.g., "RSI", "MACD")
            indicator_params: Parameters used for calculation
            indicator_data: Calculated indicator data
        
        Returns:
            bool: Success status
        """
        # Create hash of parameters for consistent key generation
        params_hash = hashlib.md5(
            json.dumps(indicator_params, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        key = f"indicator:{symbol}:{timeframe}:{indicator_name}:{params_hash}"
        
        # Cache metadata
        metadata = {
            "indicator_name": indicator_name,
            "params": indicator_params,
            "data": indicator_data,
            "cached_at": datetime.now().isoformat()
        }
        
        tags = ["indicator", symbol, timeframe, indicator_name]
        
        # TTL based on timeframe
        ttl = self._get_ttl_for_timeframe(timeframe)
        
        success = self.set(key, metadata, ttl=ttl, tags=tags)
        
        if success:
            logger.debug(f"Cached indicator: {indicator_name} for {symbol} {timeframe}")
        
        return success
    
    def get_indicator(self,
                     symbol: str,
                     timeframe: str,
                     indicator_name: str,
                     indicator_params: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached technical indicator data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            indicator_name: Name of indicator
            indicator_params: Parameters used for calculation
        
        Returns:
            Optional indicator data
        """
        params_hash = hashlib.md5(
            json.dumps(indicator_params, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        key = f"indicator:{symbol}:{timeframe}:{indicator_name}:{params_hash}"
        metadata = self.get(key)
        
        if metadata:
            logger.debug(f"Retrieved cached indicator: {indicator_name} for {symbol} {timeframe}")
            return metadata.get("data")
        
        return None
    
    def cache_trade_signal(self,
                          symbol: str,
                          signal_type: str,
                          signal_data: Dict[str, Any],
                          confidence: float = 0.0) -> bool:
        """
        Cache trade signal.
        
        Args:
            symbol: Trading symbol
            signal_type: Type of signal (BUY, SELL, HOLD)
            signal_data: Signal data including indicators, price, etc.
            confidence: Signal confidence score
        
        Returns:
            bool: Success status
        """
        timestamp = int(time.time())
        key = f"signal:{symbol}:{signal_type}:{timestamp}"
        
        signal_metadata = {
            "symbol": symbol,
            "signal_type": signal_type,
            "data": signal_data,
            "confidence": confidence,
            "timestamp": timestamp,
            "cached_at": datetime.now().isoformat()
        }
        
        tags = ["signal", symbol, signal_type]
        
        # Short TTL for signals (5 minutes)
        success = self.set(key, signal_metadata, ttl=300, tags=tags)
        
        if success:
            logger.debug(f"Cached trade signal: {signal_type} for {symbol} (confidence: {confidence:.2%})")
        
        return success
    
    def get_recent_signals(self,
                          symbol: str,
                          signal_type: Optional[str] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent trade signals.
        
        Args:
            symbol: Trading symbol
            signal_type: Filter by signal type (optional)
            limit: Maximum number of signals to return
        
        Returns:
            List of signal metadata
        """
        all_keys = self.keys()
        signal_keys = []
        
        for key in all_keys:
            if key.startswith(f"signal:{symbol}:"):
                if signal_type and f":{signal_type}:" not in key:
                    continue
                signal_keys.append(key)
        
        # Sort by timestamp (newest first)
        signal_keys.sort(reverse=True)
        
        signals = []
        for key in signal_keys[:limit]:
            signal = self.get(key)
            if signal:
                signals.append(signal)
        
        return signals
    
    def cache_order_book(self,
                        symbol: str,
                        order_book: Dict[str, Any],
                        depth: int = 100) -> bool:
        """
        Cache order book data.
        
        Args:
            symbol: Trading symbol
            order_book: Order book data
            depth: Order book depth
        
        Returns:
            bool: Success status
        """
        key = f"orderbook:{symbol}:{depth}"
        
        metadata = {
            "symbol": symbol,
            "depth": depth,
            "data": order_book,
            "timestamp": time.time(),
            "cached_at": datetime.now().isoformat()
        }
        
        tags = ["orderbook", symbol]
        
        # Very short TTL for order book (10 seconds)
        success = self.set(key, metadata, ttl=10, tags=tags)
        
        if success:
            logger.debug(f"Cached order book for {symbol} (depth: {depth})")
        
        return success
    
    def get_order_book(self, symbol: str, depth: int = 100) -> Optional[Dict[str, Any]]:
        """
        Get cached order book data.
        
        Args:
            symbol: Trading symbol
            depth: Order book depth
        
        Returns:
            Optional order book data
        """
        key = f"orderbook:{symbol}:{depth}"
        metadata = self.get(key)
        
        if metadata:
            logger.debug(f"Retrieved cached order book for {symbol}")
            return metadata.get("data")
        
        return None
    
    def cache_market_summary(self,
                            symbol: str,
                            summary: Dict[str, Any]) -> bool:
        """
        Cache market summary data.
        
        Args:
            symbol: Trading symbol
            summary: Market summary data
        
        Returns:
            bool: Success status
        """
        key = f"marketsummary:{symbol}"
        
        metadata = {
            "symbol": symbol,
            "data": summary,
            "timestamp": time.time(),
            "cached_at": datetime.now().isoformat()
        }
        
        tags = ["marketsummary", symbol]
        
        # 1-minute TTL for market summary
        success = self.set(key, metadata, ttl=60, tags=tags)
        
        if success:
            logger.debug(f"Cached market summary for {symbol}")
        
        return success
    
    def get_market_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached market summary data.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Optional market summary data
        """
        key = f"marketsummary:{symbol}"
        metadata = self.get(key)
        
        if metadata:
            logger.debug(f"Retrieved cached market summary for {symbol}")
            return metadata.get("data")
        
        return None
    
    def cache_strategy_state(self,
                           strategy_name: str,
                           symbol: str,
                           state: Dict[str, Any]) -> bool:
        """
        Cache trading strategy state.
        
        Args:
            strategy_name: Name of trading strategy
            symbol: Trading symbol
            state: Strategy state data
        
        Returns:
            bool: Success status
        """
        key = f"strategy:{strategy_name}:{symbol}:state"
        
        metadata = {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "state": state,
            "timestamp": time.time(),
            "cached_at": datetime.now().isoformat()
        }
        
        tags = ["strategy", strategy_name, symbol, "state"]
        
        # Longer TTL for strategy state (1 hour)
        success = self.set(key, metadata, ttl=3600, tags=tags)
        
        if success:
            logger.debug(f"Cached strategy state for {strategy_name} on {symbol}")
        
        return success
    
    def get_strategy_state(self, strategy_name: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached trading strategy state.
        
        Args:
            strategy_name: Name of trading strategy
            symbol: Trading symbol
        
        Returns:
            Optional strategy state data
        """
        key = f"strategy:{strategy_name}:{symbol}:state"
        metadata = self.get(key)
        
        if metadata:
            logger.debug(f"Retrieved cached strategy state for {strategy_name} on {symbol}")
            return metadata.get("state")
        
        return None
    
    def delete_by_tag(self, tag: str) -> int:
        """
        Delete all cache items with a specific tag.
        
        Args:
            tag: Tag to match
        
        Returns:
            int: Number of items deleted
        """
        deleted_count = 0
        keys_to_delete = []
        
        # Get all keys
        all_keys = self.keys()
        
        # Check each key for the tag
        for key in all_keys:
            # Note: This implementation depends on the backend supporting tag retrieval
            # For simplicity, we'll delete keys that contain the tag in their key
            if tag in key:
                keys_to_delete.append(key)
        
        # Delete matched keys
        for key in keys_to_delete:
            if self.delete(key):
                deleted_count += 1
        
        logger.debug(f"Deleted {deleted_count} items with tag: {tag}")
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        all_keys = self.keys()
        
        stats = {
            "total_items": len(all_keys),
            "cache_type": self.cache_type.value,
            "keys_by_type": {},
            "size_by_type": {}
        }
        
        # Categorize keys
        for key in all_keys:
            key_type = key.split(":")[0] if ":" in key else "other"
            stats["keys_by_type"][key_type] = stats["keys_by_type"].get(key_type, 0) + 1
        
        return stats
    
    def _get_ttl_for_timeframe(self, timeframe: str) -> float:
        """Get appropriate TTL for a given timeframe."""
        timeframe_ttl_map = {
            "1m": 60,      # 1 minute
            "5m": 300,     # 5 minutes
            "15m": 900,    # 15 minutes
            "30m": 1800,   # 30 minutes
            "1h": 3600,    # 1 hour
            "4h": 14400,   # 4 hours
            "1d": 86400,   # 1 day
            "1w": 604800,  # 1 week
        }
        
        return timeframe_ttl_map.get(timeframe, 300)  # Default 5 minutes

# Decorator for caching function results
def cached(ttl: Optional[float] = None, 
           key_prefix: str = "func",
           cache_instance: Optional[TradingCache] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        cache_instance: TradingCache instance (uses global cache if None)
    
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get or create cache instance
            cache = cache_instance or get_global_cache()
            
            # Create cache key from function name and arguments
            args_repr = str(args)
            kwargs_repr = str(sorted(kwargs.items()))
            key_content = f"{func.__module__}.{func.__name__}({args_repr},{kwargs_repr})"
            
            # Hash the key content
            key_hash = hashlib.md5(key_content.encode()).hexdigest()
            cache_key = f"{key_prefix}:{func.__name__}:{key_hash}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            
            logger.debug(f"Cache miss for {func.__name__}, cached result")
            return result
        
        # For async functions
        async def async_wrapper(*args, **kwargs):
            # Get or create cache instance
            cache = cache_instance or get_global_cache()
            
            # Create cache key
            args_repr = str(args)
            kwargs_repr = str(sorted(kwargs.items()))
            key_content = f"{func.__module__}.{func.__name__}({args_repr},{kwargs_repr})"
            
            key_hash = hashlib.md5(key_content.encode()).hexdigest()
            cache_key = f"{key_prefix}:{func.__name__}:{key_hash}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for async {func.__name__}")
                return cached_result
            
            # Execute async function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            
            logger.debug(f"Cache miss for async {func.__name__}, cached result")
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator

# Global cache instance
_global_cache = None

def setup_global_cache(cache_type: CacheType = CacheType.HYBRID, **kwargs) -> TradingCache:
    """
    Setup and return a global cache instance.
    
    Args:
        cache_type: Type of cache to use
        **kwargs: Additional arguments for TradingCache
    
    Returns:
        TradingCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = TradingCache(cache_type=cache_type, **kwargs)
    return _global_cache

def get_global_cache() -> TradingCache:
    """
    Get the global cache instance. Creates one if it doesn't exist.
    
    Returns:
        TradingCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = setup_global_cache()
    return _global_cache

# Test function
if __name__ == "__main__":
    # Test the cache
    cache = setup_global_cache(CacheType.MEMORY, max_size=100)
    
    # Basic operations
    cache.set("test_key", "test_value", ttl=10)
    print(f"Exists test_key: {cache.exists('test_key')}")
    print(f"Get test_key: {cache.get('test_key')}")
    
    # Test OHLCV caching
    ohlcv_data = [
        [1633046400000, 45000, 45500, 44800, 45200, 100.5],
        [1633046460000, 45200, 45400, 45100, 45300, 85.2]
    ]
    
    cache.cache_ohlcv("BTC/USDT", "1m", ohlcv_data)
    cached_ohlcv = cache.get_ohlcv("BTC/USDT", "1m")
    print(f"Cached OHLCV candles: {len(cached_ohlcv) if cached_ohlcv else 0}")
    
    # Test indicator caching
    rsi_data = [65.5, 67.2, 62.8, 70.1]
    cache.cache_indicator(
        symbol="BTC/USDT",
        timeframe="1h",
        indicator_name="RSI",
        indicator_params={"period": 14},
        indicator_data=rsi_data
    )
    
    cached_rsi = cache.get_indicator(
        symbol="BTC/USDT",
        timeframe="1h",
        indicator_name="RSI",
        indicator_params={"period": 14}
    )
    print(f"Cached RSI data: {cached_rsi}")
    
    # Test trade signal caching
    signal_data = {
        "price": 45250.75,
        "indicators": {"RSI": 65.5, "MACD": 12.3},
        "timestamp": time.time()
    }
    
    cache.cache_trade_signal(
        symbol="BTC/USDT",
        signal_type="BUY",
        signal_data=signal_data,
        confidence=0.75
    )
    
    recent_signals = cache.get_recent_signals("BTC/USDT")
    print(f"Recent signals: {len(recent_signals)}")
    
    # Test cache stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Test decorator
    @cached(ttl=60)
    def expensive_calculation(x, y):
        time.sleep(1)  # Simulate expensive operation
        return x * y
    
    # First call (cache miss)
    start = time.time()
    result1 = expensive_calculation(5, 10)
    elapsed1 = time.time() - start
    print(f"First call: {result1}, took {elapsed1:.2f}s")
    
    # Second call (cache hit)
    start = time.time()
    result2 = expensive_calculation(5, 10)
    elapsed2 = time.time() - start
    print(f"Second call: {result2}, took {elapsed2:.2f}s")
    
    print("Cache test completed!")