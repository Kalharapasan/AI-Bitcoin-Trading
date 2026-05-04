"""
Database connection management for the Bitcoin Trading AI application.
Handles database connections, pooling, and session management.
"""

import logging
from typing import Optional, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import redis
import json

from config.config_manager import ConfigManager
from .models import Base, create_all_tables, drop_all_tables

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors"""
    pass


class RedisConnectionManager:
    """Redis connection manager for caching and pub/sub"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub_client: Optional[redis.Redis] = None
        self._connect_redis()
    
    def _connect_redis(self) -> None:
        """Connect to Redis server"""
        try:
            redis_config = self.config_manager.get_redis_config()
            
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                password=redis_config.get('password'),
                db=redis_config.get('db', 0),
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
            
            # Create separate client for pub/sub if needed
            self.pubsub_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                password=redis_config.get('password'),
                db=redis_config.get('db', 0),
                decode_responses=True
            )
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            self.pubsub_client = None
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            self.redis_client = None
            self.pubsub_client = None
    
    def get_client(self) -> Optional[redis.Redis]:
        """Get Redis client"""
        if self.redis_client is None:
            self._connect_redis()
        return self.redis_client
    
    def get_pubsub(self) -> Optional[redis.Redis]:
        """Get Redis pub/sub client"""
        if self.pubsub_client is None:
            self._connect_redis()
        return self.pubsub_client
    
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        if self.redis_client is None:
            return False
        try:
            self.redis_client.ping()
            return True
        except (redis.ConnectionError, AttributeError):
            return False
    
    def set_cached_data(self, key: str, data: any, expire: int = 3600) -> bool:
        """Cache data in Redis"""
        try:
            client = self.get_client()
            if client is None:
                return False
            
            serialized_data = json.dumps(data)
            client.setex(key, expire, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Error caching data: {e}")
            return False
    
    def get_cached_data(self, key: str) -> Optional[any]:
        """Get cached data from Redis"""
        try:
            client = self.get_client()
            if client is None:
                return None
            
            data = client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None
    
    def delete_cached_data(self, key: str) -> bool:
        """Delete cached data"""
        try:
            client = self.get_client()
            if client is None:
                return False
            
            client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting cached data: {e}")
            return False
    
    def publish_message(self, channel: str, message: dict) -> bool:
        """Publish message to Redis channel"""
        try:
            pubsub = self.get_pubsub()
            if pubsub is None:
                return False
            
            serialized_message = json.dumps(message)
            pubsub.publish(channel, serialized_message)
            return True
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    def close(self) -> None:
        """Close Redis connections"""
        try:
            if self.redis_client:
                self.redis_client.close()
            if self.pubsub_client:
                self.pubsub_client.close()
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")


class DatabaseConnectionManager:
    """Main database connection manager"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
        self.ScopedSession: Optional[scoped_session] = None
        self.redis_manager: Optional[RedisConnectionManager] = None
        self._connect_database()
        self._setup_redis()
    
    def _connect_database(self) -> None:
        """Establish database connection"""
        try:
            db_config = self.config_manager.get_database_config()
            
            # Determine database type and build connection string
            db_type = db_config.get('type', 'sqlite').lower()
            
            if db_type == 'sqlite':
                # SQLite connection
                db_path = db_config.get('database', 'trading_ai.db')
                connection_string = f"sqlite:///{db_path}"
                
                engine_config = {
                    'echo': db_config.get('echo', False),
                    'poolclass': StaticPool,  # SQLite doesn't support connection pooling
                    'connect_args': {'check_same_thread': False}
                }
            
            elif db_type == 'postgresql':
                # PostgreSQL connection
                host = db_config.get('host', 'localhost')
                port = db_config.get('port', 5432)
                database = db_config.get('database', 'trading_ai')
                username = db_config.get('username', 'postgres')
                password = db_config.get('password', '')
                
                connection_string = (
                    f"postgresql://{username}:{password}@{host}:{port}/{database}"
                )
                
                engine_config = {
                    'echo': db_config.get('echo', False),
                    'pool_size': db_config.get('pool_size', 10),
                    'max_overflow': db_config.get('max_overflow', 20),
                    'pool_timeout': db_config.get('pool_timeout', 30),
                    'pool_recycle': db_config.get('pool_recycle', 3600),
                    'pool_pre_ping': db_config.get('pool_pre_ping', True),
                    'poolclass': QueuePool
                }
            
            elif db_type == 'mysql':
                # MySQL connection
                host = db_config.get('host', 'localhost')
                port = db_config.get('port', 3306)
                database = db_config.get('database', 'trading_ai')
                username = db_config.get('username', 'root')
                password = db_config.get('password', '')
                
                connection_string = (
                    f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
                    "?charset=utf8mb4"
                )
                
                engine_config = {
                    'echo': db_config.get('echo', False),
                    'pool_size': db_config.get('pool_size', 10),
                    'max_overflow': db_config.get('max_overflow', 20),
                    'pool_timeout': db_config.get('pool_timeout', 30),
                    'pool_recycle': db_config.get('pool_recycle', 3600),
                    'pool_pre_ping': db_config.get('pool_pre_ping', True),
                    'poolclass': QueuePool
                }
            
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            # Create engine
            self.engine = create_engine(connection_string, **engine_config)
            
            # Add connection pool event listeners
            self._setup_connection_pool_events()
            
            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            # Create scoped session for thread safety
            self.ScopedSession = scoped_session(self.session_factory)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            logger.info(f"Database connection established successfully to {db_type}")
            
        except Exception as e:
            logger.error(f"Failed to establish database connection: {e}")
            self.engine = None
            self.session_factory = None
            self.ScopedSession = None
            raise DatabaseConnectionError(f"Database connection failed: {e}")
    
    def _setup_connection_pool_events(self) -> None:
        """Setup connection pool event listeners"""
        if not self.engine:
            return
        
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance"""
            if self.engine and 'sqlite' in self.engine.url.drivername:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA cache_size=-2000")  # 2MB cache
                cursor.close()
        
        @event.listens_for(self.engine, "checkout")
        def ping_connection(dbapi_connection, connection_record, connection_proxy):
            """Ping connection before using it from pool"""
            try:
                dbapi_connection.ping(False)
            except Exception:
                raise OperationalError("Database connection ping failed")
    
    def _setup_redis(self) -> None:
        """Setup Redis connection"""
        try:
            self.redis_manager = RedisConnectionManager(self.config_manager)
        except Exception as e:
            logger.warning(f"Failed to setup Redis: {e}. Caching will be disabled.")
            self.redis_manager = None
    
    def create_tables(self, drop_existing: bool = False) -> bool:
        """Create all database tables"""
        if not self.engine:
            logger.error("Database engine not initialized")
            return False
        
        try:
            if drop_existing:
                logger.info("Dropping existing tables...")
                drop_all_tables(self.engine)
            
            logger.info("Creating database tables...")
            create_all_tables(self.engine)
            logger.info("Database tables created successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            return False
    
    def get_session(self) -> Session:
        """Get a new database session"""
        if not self.session_factory:
            raise DatabaseConnectionError("Database session factory not initialized")
        
        return self.session_factory()
    
    def get_scoped_session(self) -> Session:
        """Get a scoped session (thread-safe)"""
        if not self.ScopedSession:
            raise DatabaseConnectionError("Scoped session not initialized")
        
        return self.ScopedSession()
    
    def remove_scoped_session(self) -> None:
        """Remove scoped session from registry"""
        if self.ScopedSession:
            self.ScopedSession.remove()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error in session: {e}")
            raise
        finally:
            session.close()
    
    @contextmanager
    def scoped_session_scope(self) -> Generator[Session, None, None]:
        """Provide a scoped transactional scope"""
        session = self.get_scoped_session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database scoped session error: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error in scoped session: {e}")
            raise
        finally:
            self.remove_scoped_session()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        health_status = {
            'database_connected': False,
            'redis_connected': False,
            'database_type': 'unknown',
            'pool_status': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check database connection
        if self.engine:
            try:
                with self.engine.connect() as conn:
                    result = conn.execute("SELECT 1").scalar()
                    health_status['database_connected'] = (result == 1)
                    health_status['database_type'] = self.engine.url.drivername
                    
                    # Get pool status
                    pool = self.engine.pool
                    health_status['pool_status'] = {
                        'size': getattr(pool, 'size', None),
                        'checked_in': getattr(pool, 'checkedin', None),
                        'checked_out': getattr(pool, 'checkedout', None),
                        'overflow': getattr(pool, 'overflow', None)
                    }
            except Exception as e:
                health_status['database_error'] = str(e)
        
        # Check Redis connection
        if self.redis_manager:
            health_status['redis_connected'] = self.redis_manager.is_connected()
        
        return health_status
    
    def backup_database(self, backup_path: str = None) -> Optional[str]:
        """Create database backup"""
        if not self.engine:
            return None
        
        try:
            db_type = self.engine.url.drivername
            
            if 'sqlite' in db_type:
                # SQLite backup
                import shutil
                import os
                
                if not backup_path:
                    backup_dir = "backups"
                    os.makedirs(backup_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = os.path.join(backup_dir, f"trading_ai_backup_{timestamp}.db")
                
                # Close connections before backup
                self.engine.dispose()
                
                # Copy database file
                db_path = self.engine.url.database
                shutil.copy2(db_path, backup_path)
                
                logger.info(f"SQLite database backed up to: {backup_path}")
                return backup_path
            
            elif 'postgresql' in db_type:
                # PostgreSQL backup using pg_dump
                import subprocess
                
                db_config = self.config_manager.get_database_config()
                host = db_config.get('host', 'localhost')
                port = db_config.get('port', 5432)
                database = db_config.get('database', 'trading_ai')
                username = db_config.get('username', 'postgres')
                
                if not backup_path:
                    backup_dir = "backups"
                    os.makedirs(backup_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = os.path.join(backup_dir, f"trading_ai_backup_{timestamp}.sql")
                
                # Build pg_dump command
                cmd = [
                    'pg_dump',
                    '-h', host,
                    '-p', str(port),
                    '-U', username,
                    '-d', database,
                    '-f', backup_path,
                    '-F', 'c'  # Custom format
                ]
                
                # Set PGPASSWORD environment variable
                env = os.environ.copy()
                if 'password' in db_config:
                    env['PGPASSWORD'] = db_config['password']
                
                # Execute backup
                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"PostgreSQL database backed up to: {backup_path}")
                    return backup_path
                else:
                    logger.error(f"PostgreSQL backup failed: {result.stderr}")
                    return None
            
            else:
                logger.warning(f"Backup not implemented for database type: {db_type}")
                return None
        
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return None
    
    def optimize_database(self) -> bool:
        """Optimize database performance"""
        if not self.engine:
            return False
        
        try:
            db_type = self.engine.url.drivername
            
            with self.engine.connect() as conn:
                if 'sqlite' in db_type:
                    # SQLite optimization
                    conn.execute("PRAGMA optimize")
                    conn.execute("VACUUM")
                    logger.info("SQLite database optimized")
                
                elif 'postgresql' in db_type:
                    # PostgreSQL optimization
                    conn.execute("VACUUM ANALYZE")
                    logger.info("PostgreSQL database optimized")
                
                elif 'mysql' in db_type:
                    # MySQL optimization
                    conn.execute("OPTIMIZE TABLE market_data, trades, orders")
                    logger.info("MySQL database optimized")
            
            return True
        
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return False
    
    def close(self) -> None:
        """Close all database connections"""
        try:
            # Close Redis connections
            if self.redis_manager:
                self.redis_manager.close()
            
            # Dispose database engine
            if self.engine:
                self.engine.dispose()
            
            logger.info("Database connections closed")
        
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Singleton instance
_database_manager: Optional[DatabaseConnectionManager] = None


def get_database_manager(config_manager: ConfigManager = None) -> DatabaseConnectionManager:
    """Get or create database connection manager (singleton)"""
    global _database_manager
    
    if _database_manager is None:
        if config_manager is None:
            from config.config_manager import ConfigManager
            config_manager = ConfigManager()
        
        _database_manager = DatabaseConnectionManager(config_manager)
    
    return _database_manager


def init_database(config_manager: ConfigManager = None, drop_existing: bool = False) -> bool:
    """Initialize database (create tables, etc.)"""
    try:
        db_manager = get_database_manager(config_manager)
        return db_manager.create_tables(drop_existing)
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def close_database() -> None:
    """Close database connections"""
    global _database_manager
    
    if _database_manager:
        _database_manager.close()
        _database_manager = None


# Cleanup on exit
import atexit
atexit.register(close_database)


if __name__ == "__main__":
    # Test the database connection
    from config.config_manager import ConfigManager
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        config = ConfigManager()
        db_manager = get_database_manager(config)
        
        # Health check
        health = db_manager.health_check()
        print("Database Health Check:")
        print(json.dumps(health, indent=2))
        
        # Create tables if not exist
        if health['database_connected']:
            db_manager.create_tables()
            print("Database tables created/verified")
        
        # Test Redis
        if db_manager.redis_manager and db_manager.redis_manager.is_connected():
            print("Redis connected")
            # Test caching
            db_manager.redis_manager.set_cached_data("test_key", {"test": "data"}, expire=10)
            cached = db_manager.redis_manager.get_cached_data("test_key")
            print(f"Cached data retrieved: {cached}")
        
    except Exception as e:
        print(f"Database test failed: {e}")