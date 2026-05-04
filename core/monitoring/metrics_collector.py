"""
Metrics Collector module for Bitcoin trading AI.
Collects, aggregates, and manages metrics from all system components
for monitoring, analysis, and performance optimization.
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
import threading
import queue
import signal
import sys

# Import project modules
from config.settings import MetricsSettings, SystemSettings
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.utils.cache import Cache
from core.models.model_manager import ModelManager, ModelMetadata, ModelType, ModelStatus
from core.performance.performance_tracker import PerformanceTracker, PerformanceStatus

# Import monitoring libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. System metrics collection disabled.")

try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available. Prometheus metrics disabled.")

try:
    import influxdb
    from influxdb import InfluxDBClient
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    logger.warning("influxdb not available. InfluxDB integration disabled.")

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Enums and Types ============
class MetricCategory(str, Enum):
    """Categories of metrics"""
    SYSTEM = "system"
    MODEL = "model"
    TRADING = "trading"
    MARKET = "market"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    CUSTOM = "custom"

class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"

class AggregationMethod(str, Enum):
    """Methods for aggregating metrics"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"
    RATE = "rate"

class StorageBackend(str, Enum):
    """Storage backends for metrics"""
    MEMORY = "memory"
    DISK = "disk"
    INFLUXDB = "influxdb"
    PROMETHEUS = "prometheus"
    DATABASE = "database"

# ============ Data Structures ============
@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    
    # General settings
    enabled: bool = True
    collection_interval_seconds: int = 60
    aggregation_interval_minutes: int = 5
    retention_days: int = 30
    compression_enabled: bool = True
    
    # Collection settings
    collect_system_metrics: bool = True
    collect_model_metrics: bool = True
    collect_trading_metrics: bool = True
    collect_market_metrics: bool = True
    collect_performance_metrics: bool = True
    
    # System metrics
    system_metrics_interval: int = 30  # seconds
    collect_cpu_metrics: bool = True
    collect_memory_metrics: bool = True
    collect_disk_metrics: bool = True
    collect_network_metrics: bool = True
    collect_process_metrics: bool = True
    
    # Model metrics
    model_metrics_interval: int = 60  # seconds
    collect_inference_metrics: bool = True
    collect_training_metrics: bool = True
    collect_prediction_metrics: bool = True
    collect_model_performance: bool = True
    
    # Trading metrics
    trading_metrics_interval: int = 10  # seconds
    collect_trade_metrics: bool = True
    collect_position_metrics: bool = True
    collect_portfolio_metrics: bool = True
    collect_risk_metrics: bool = True
    
    # Market metrics
    market_metrics_interval: int = 15  # seconds
    collect_price_metrics: bool = True
    collect_volume_metrics: bool = True
    collect_volatility_metrics: bool = True
    collect_order_book_metrics: bool = True
    
    # Storage settings
    storage_backend: StorageBackend = StorageBackend.MEMORY
    max_memory_metrics: int = 100000
    disk_storage_path: str = "data/metrics/"
    database_url: Optional[str] = None
    
    # InfluxDB settings
    influxdb_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'host': 'localhost',
        'port': 8086,
        'database': 'trading_metrics',
        'username': '',
        'password': '',
        'ssl': False,
        'verify_ssl': False
    })
    
    # Prometheus settings
    prometheus_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'port': 9090,
        'endpoint': '/metrics',
        'registry': 'default'
    })
    
    # Aggregation settings
    aggregation_windows: List[str] = field(default_factory=lambda: [
        '1m', '5m', '15m', '1h', '6h', '1d', '7d'
    ])
    percentiles: List[float] = field(default_factory=lambda: [0.5, 0.9, 0.95, 0.99])
    
    # Alerting thresholds
    metric_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'system.cpu.usage': {'warning': 80.0, 'critical': 95.0},
        'system.memory.usage': {'warning': 85.0, 'critical': 95.0},
        'system.disk.usage': {'warning': 90.0, 'critical': 98.0},
        'model.inference.time': {'warning': 1000.0, 'critical': 5000.0},
        'model.accuracy': {'warning': 0.6, 'critical': 0.4},
        'trading.drawdown': {'warning': 0.05, 'critical': 0.1},
        'trading.volatility': {'warning': 0.5, 'critical': 1.0}
    })
    
    # Export settings
    export_enabled: bool = True
    export_format: str = "parquet"  # parquet, csv, json
    export_interval_minutes: int = 60
    export_path: str = "data/metrics_exports/"
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_port: int = 8081
    metrics_dashboard: bool = True
    real_time_updates: bool = True
    
    # Advanced settings
    buffer_size: int = 1000
    batch_size: int = 100
    max_retries: int = 3
    retry_delay_seconds: int = 1
    
    def __post_init__(self):
        """Validate configuration"""
        if self.collection_interval_seconds < 1:
            raise ValueError("collection_interval_seconds must be at least 1")
        
        if self.aggregation_interval_minutes < 1:
            raise ValueError("aggregation_interval_minutes must be at least 1")
        
        # Create directories
        Path(self.disk_storage_path).mkdir(parents=True, exist_ok=True)
        Path(self.export_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'collection_interval_seconds': self.collection_interval_seconds,
            'aggregation_interval_minutes': self.aggregation_interval_minutes,
            'retention_days': self.retention_days,
            'compression_enabled': self.compression_enabled,
            'collect_system_metrics': self.collect_system_metrics,
            'collect_model_metrics': self.collect_model_metrics,
            'collect_trading_metrics': self.collect_trading_metrics,
            'collect_market_metrics': self.collect_market_metrics,
            'collect_performance_metrics': self.collect_performance_metrics,
            'system_metrics_interval': self.system_metrics_interval,
            'collect_cpu_metrics': self.collect_cpu_metrics,
            'collect_memory_metrics': self.collect_memory_metrics,
            'collect_disk_metrics': self.collect_disk_metrics,
            'collect_network_metrics': self.collect_network_metrics,
            'collect_process_metrics': self.collect_process_metrics,
            'model_metrics_interval': self.model_metrics_interval,
            'collect_inference_metrics': self.collect_inference_metrics,
            'collect_training_metrics': self.collect_training_metrics,
            'collect_prediction_metrics': self.collect_prediction_metrics,
            'collect_model_performance': self.collect_model_performance,
            'trading_metrics_interval': self.trading_metrics_interval,
            'collect_trade_metrics': self.collect_trade_metrics,
            'collect_position_metrics': self.collect_position_metrics,
            'collect_portfolio_metrics': self.collect_portfolio_metrics,
            'collect_risk_metrics': self.collect_risk_metrics,
            'market_metrics_interval': self.market_metrics_interval,
            'collect_price_metrics': self.collect_price_metrics,
            'collect_volume_metrics': self.collect_volume_metrics,
            'collect_volatility_metrics': self.collect_volatility_metrics,
            'collect_order_book_metrics': self.collect_order_book_metrics,
            'storage_backend': self.storage_backend.value,
            'max_memory_metrics': self.max_memory_metrics,
            'disk_storage_path': self.disk_storage_path,
            'database_url': self.database_url,
            'influxdb_config': self.influxdb_config,
            'prometheus_config': self.prometheus_config,
            'aggregation_windows': self.aggregation_windows,
            'percentiles': self.percentiles,
            'metric_thresholds': self.metric_thresholds,
            'export_enabled': self.export_enabled,
            'export_format': self.export_format,
            'export_interval_minutes': self.export_interval_minutes,
            'export_path': self.export_path,
            'enable_monitoring': self.enable_monitoring,
            'monitoring_port': self.monitoring_port,
            'metrics_dashboard': self.metrics_dashboard,
            'real_time_updates': self.real_time_updates,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'max_retries': self.max_retries,
            'retry_delay_seconds': self.retry_delay_seconds
        }

@dataclass
class Metric:
    """Individual metric data point"""
    
    # Core fields
    name: str
    value: Union[float, int, str, bool]
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Metadata
    category: MetricCategory = MetricCategory.CUSTOM
    metric_type: MetricType = MetricType.GAUGE
    labels: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Source information
    source: str = "unknown"
    component: str = "unknown"
    instance: str = "default"
    
    # Aggregation info
    is_aggregated: bool = False
    aggregation_window: Optional[str] = None
    aggregation_method: Optional[AggregationMethod] = None
    
    # Quality info
    confidence: float = 1.0
    error_margin: Optional[float] = None
    
    def __post_init__(self):
        """Validate metric"""
        if not isinstance(self.value, (int, float, str, bool)):
            raise ValueError(f"Invalid metric value type: {type(self.value)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category.value,
            'metric_type': self.metric_type.value,
            'labels': self.labels,
            'tags': self.tags,
            'source': self.source,
            'component': self.component,
            'instance': self.instance,
            'is_aggregated': self.is_aggregated,
            'aggregation_window': self.aggregation_window,
            'aggregation_method': self.aggregation_method.value if self.aggregation_method else None,
            'confidence': self.confidence,
            'error_margin': self.error_margin
        }
    
    def to_influxdb_point(self) -> Dict[str, Any]:
        """Convert to InfluxDB point format"""
        return {
            "measurement": self.name,
            "tags": {**self.labels, **self.tags, "category": self.category.value},
            "fields": {"value": float(self.value) if isinstance(self.value, (int, float)) else str(self.value)},
            "time": self.timestamp.isoformat()
        }
    
    def to_prometheus_labels(self) -> Dict[str, str]:
        """Convert to Prometheus labels format"""
        prom_labels = {
            "category": self.category.value,
            "source": self.source,
            "component": self.component,
            "instance": self.instance
        }
        prom_labels.update(self.labels)
        return prom_labels

@dataclass
class AggregatedMetric:
    """Aggregated metric data"""
    
    name: str
    values: List[float]
    timestamps: List[datetime]
    
    # Aggregation results
    count: int
    sum: float
    avg: float
    min: float
    max: float
    std: float
    percentiles: Dict[float, float]
    
    # Metadata
    category: MetricCategory
    labels: Dict[str, str]
    aggregation_window: str
    aggregation_method: AggregationMethod
    
    # Time info
    window_start: datetime
    window_end: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'count': self.count,
            'sum': self.sum,
            'avg': self.avg,
            'min': self.min,
            'max': self.max,
            'std': self.std,
            'percentiles': self.percentiles,
            'category': self.category.value,
            'labels': self.labels,
            'aggregation_window': self.aggregation_window,
            'aggregation_method': self.aggregation_method.value,
            'window_start': self.window_start.isoformat(),
            'window_end': self.window_end.isoformat(),
            'sample_values': self.values[:10] if self.values else [],  # First 10 values
            'sample_timestamps': [t.isoformat() for t in self.timestamps[:10]] if self.timestamps else []
        }

@dataclass
class MetricBatch:
    """Batch of metrics for efficient processing"""
    
    metrics: List[Metric]
    batch_id: str = field(default_factory=lambda: f"batch_{uuid.uuid4().hex[:8]}")
    created_at: datetime = field(default_factory=datetime.now)
    source: str = "collector"
    
    def __len__(self):
        return len(self.metrics)
    
    def append(self, metric: Metric):
        """Add metric to batch"""
        self.metrics.append(metric)
    
    def extend(self, metrics: List[Metric]):
        """Add multiple metrics to batch"""
        self.metrics.extend(metrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'batch_id': self.batch_id,
            'created_at': self.created_at.isoformat(),
            'source': self.source,
            'metric_count': len(self.metrics),
            'metrics': [m.to_dict() for m in self.metrics]
        }
    
    def save(self, filepath: str):
        """Save batch to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

# ============ Metric Collectors ============
class SystemMetricsCollector:
    """Collects system-level metrics"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.SystemCollector")
        self.last_collection = datetime.now()
    
    def collect(self) -> List[Metric]:
        """Collect system metrics"""
        
        metrics = []
        current_time = datetime.now()
        
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available, skipping system metrics")
            return metrics
        
        try:
            # CPU metrics
            if self.config.collect_cpu_metrics:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_count = psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()
                
                metrics.extend([
                    Metric(
                        name="system.cpu.usage",
                        value=cpu_percent,
                        category=MetricCategory.SYSTEM,
                        metric_type=MetricType.GAUGE,
                        labels={"type": "total"},
                        source="system",
                        component="cpu"
                    ),
                    Metric(
                        name="system.cpu.count",
                        value=cpu_count,
                        category=MetricCategory.SYSTEM,
                        metric_type=MetricType.GAUGE,
                        source="system",
                        component="cpu"
                    )
                ])
                
                if cpu_freq:
                    metrics.append(Metric(
                        name="system.cpu.frequency",
                        value=cpu_freq.current,
                        category=MetricCategory.SYSTEM,
                        metric_type=MetricType.GAUGE,
                        labels={"type": "current"},
                        source="system",
                        component="cpu"
                    ))
            
            # Memory metrics
            if self.config.collect_memory_metrics:
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                metrics.extend([
                    Metric(
                        name="system.memory.usage",
                        value=memory.percent,
                        category=MetricCategory.SYSTEM,
                        metric_type=MetricType.GAUGE,
                        labels={"type": "physical"},
                        source="system",
                        component="memory"
                    ),
                    Metric(
                        name="system.memory.available",
                        value=memory.available / (1024 ** 3),  # Convert to GB
                        category=MetricCategory.SYSTEM,
                        metric_type=MetricType.GAUGE,
                        labels={"type": "physical"},
                        source="system",
                        component="memory"
                    ),
                    Metric(
                        name="system.swap.usage",
                        value=swap.percent,
                        category=MetricCategory.SYSTEM,
                        metric_type=MetricType.GAUGE,
                        source="system",
                        component="memory"
                    )
                ])
            
            # Disk metrics
            if self.config.collect_disk_metrics:
                disk_usage = psutil.disk_usage('/')
                disk_io = psutil.disk_io_counters()
                
                metrics.append(Metric(
                    name="system.disk.usage",
                    value=disk_usage.percent,
                    category=MetricCategory.SYSTEM,
                    metric_type=MetricType.GAUGE,
                    labels={"mount": "/"},
                    source="system",
                    component="disk"
                ))
                
                if disk_io:
                    metrics.extend([
                        Metric(
                            name="system.disk.read_bytes",
                            value=disk_io.read_bytes,
                            category=MetricCategory.SYSTEM,
                            metric_type=MetricType.COUNTER,
                            source="system",
                            component="disk"
                        ),
                        Metric(
                            name="system.disk.write_bytes",
                            value=disk_io.write_bytes,
                            category=MetricCategory.SYSTEM,
                            metric_type=MetricType.COUNTER,
                            source="system",
                            component="disk"
                        )
                    ])
            
            # Network metrics
            if self.config.collect_network_metrics:
                net_io = psutil.net_io_counters()
                
                if net_io:
                    metrics.extend([
                        Metric(
                            name="system.network.bytes_sent",
                            value=net_io.bytes_sent,
                            category=MetricCategory.SYSTEM,
                            metric_type=MetricType.COUNTER,
                            source="system",
                            component="network"
                        ),
                        Metric(
                            name="system.network.bytes_recv",
                            value=net_io.bytes_recv,
                            category=MetricCategory.SYSTEM,
                            metric_type=MetricType.COUNTER,
                            source="system",
                            component="network"
                        )
                    ])
            
            # Process metrics
            if self.config.collect_process_metrics:
                process = psutil.Process()
                
                metrics.extend([
                    Metric(
                        name="system.process.cpu_percent",
                        value=process.cpu_percent(),
                        category=MetricCategory.SYSTEM,
                        metric_type=MetricType.GAUGE,
                        source="system",
                        component="process",
                        labels={"pid": str(process.pid)}
                    ),
                    Metric(
                        name="system.process.memory_percent",
                        value=process.memory_percent(),
                        category=MetricCategory.SYSTEM,
                        metric_type=MetricType.GAUGE,
                        source="system",
                        component="process",
                        labels={"pid": str(process.pid)}
                    ),
                    Metric(
                        name="system.process.memory_rss",
                        value=process.memory_info().rss / (1024 ** 2),  # Convert to MB
                        category=MetricCategory.SYSTEM,
                        metric_type=MetricType.GAUGE,
                        source="system",
                        component="process",
                        labels={"pid": str(process.pid)}
                    )
                ])
            
            # Collection timing
            collection_duration = (datetime.now() - current_time).total_seconds() * 1000
            metrics.append(Metric(
                name="system.metrics.collection_time",
                value=collection_duration,
                category=MetricCategory.SYSTEM,
                metric_type=MetricType.GAUGE,
                source="metrics_collector",
                component="system"
            ))
            
            self.last_collection = datetime.now()
            
            self.logger.debug(f"Collected {len(metrics)} system metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {str(e)}")
        
        return metrics

class ModelMetricsCollector:
    """Collects model-related metrics"""
    
    def __init__(self, config: MetricsConfig, model_manager: Optional[ModelManager] = None):
        self.config = config
        self.model_manager = model_manager
        self.logger = get_logger(f"{__name__}.ModelCollector")
        self.last_inference_times = {}
        self.last_training_times = {}
    
    def collect(self, model_data: Optional[Dict[str, Any]] = None) -> List[Metric]:
        """Collect model metrics"""
        
        metrics = []
        
        try:
            # Inference metrics
            if self.config.collect_inference_metrics and model_data:
                metrics.extend(self._collect_inference_metrics(model_data))
            
            # Training metrics (if available)
            if self.config.collect_training_metrics and model_data:
                metrics.extend(self._collect_training_metrics(model_data))
            
            # Model performance metrics
            if self.config.collect_model_performance and self.model_manager:
                metrics.extend(self._collect_model_performance_metrics())
            
            # Prediction metrics
            if self.config.collect_prediction_metrics and model_data:
                metrics.extend(self._collect_prediction_metrics(model_data))
            
            self.logger.debug(f"Collected {len(metrics)} model metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to collect model metrics: {str(e)}")
        
        return metrics
    
    def _collect_inference_metrics(self, model_data: Dict[str, Any]) -> List[Metric]:
        """Collect inference-related metrics"""
        
        metrics = []
        current_time = datetime.now()
        
        for model_id, data in model_data.items():
            if 'inference_time' in data:
                inference_time = data['inference_time']
                model_type = data.get('model_type', 'unknown')
                
                # Calculate inference rate if we have previous timing
                if model_id in self.last_inference_times:
                    last_time, last_count = self.last_inference_times[model_id]
                    time_diff = (current_time - last_time).total_seconds()
                    
                    if time_diff > 0:
                        inference_rate = (data.get('inference_count', 1) - last_count) / time_diff
                        
                        metrics.append(Metric(
                            name="model.inference.rate",
                            value=inference_rate,
                            category=MetricCategory.MODEL,
                            metric_type=MetricType.GAUGE,
                            labels={"model_id": model_id, "model_type": model_type},
                            source="model",
                            component="inference"
                        ))
                
                # Store current timing
                self.last_inference_times[model_id] = (
                    current_time,
                    data.get('inference_count', 1)
                )
                
                # Add inference time metric
                metrics.append(Metric(
                    name="model.inference.time",
                    value=inference_time,
                    category=MetricCategory.MODEL,
                    metric_type=MetricType.GAUGE,
                    labels={"model_id": model_id, "model_type": model_type},
                    source="model",
                    component="inference"
                ))
            
            # Add any additional inference metrics
            if 'inference_metrics' in data:
                for metric_name, metric_value in data['inference_metrics'].items():
                    metrics.append(Metric(
                        name=f"model.inference.{metric_name}",
                        value=metric_value,
                        category=MetricCategory.MODEL,
                        metric_type=MetricType.GAUGE,
                        labels={"model_id": model_id},
                        source="model",
                        component="inference"
                    ))
        
        return metrics
    
    def _collect_training_metrics(self, model_data: Dict[str, Any]) -> List[Metric]:
        """Collect training-related metrics"""
        
        metrics = []
        
        for model_id, data in model_data.items():
            if 'training_metrics' in data:
                training_data = data['training_metrics']
                
                for epoch, epoch_metrics in training_data.items():
                    for metric_name, metric_value in epoch_metrics.items():
                        metrics.append(Metric(
                            name=f"model.training.{metric_name}",
                            value=metric_value,
                            category=MetricCategory.MODEL,
                            metric_type=MetricType.GAUGE,
                            labels={"model_id": model_id, "epoch": str(epoch)},
                            source="model",
                            component="training"
                        ))
            
            # Training time
            if 'training_time' in data:
                metrics.append(Metric(
                    name="model.training.time",
                    value=data['training_time'],
                    category=MetricCategory.MODEL,
                    metric_type=MetricType.GAUGE,
                    labels={"model_id": model_id},
                    source="model",
                    component="training"
                ))
        
        return metrics
    
    def _collect_model_performance_metrics(self) -> List[Metric]:
        """Collect model performance metrics"""
        
        metrics = []
        
        if not self.model_manager:
            return metrics
        
        try:
            # Get all models
            models = self.model_manager.list_models()
            
            for model_id in models:
                model_info = self.model_manager.get_model_info(model_id)
                
                if model_info and model_info.performance_metrics:
                    perf_metrics = model_info.performance_metrics
                    
                    for metric_name, metric_value in perf_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            metrics.append(Metric(
                                name=f"model.performance.{metric_name}",
                                value=metric_value,
                                category=MetricCategory.MODEL,
                                metric_type=MetricType.GAUGE,
                                labels={"model_id": model_id},
                                source="model",
                                component="performance"
                            ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect model performance metrics: {str(e)}")
        
        return metrics
    
    def _collect_prediction_metrics(self, model_data: Dict[str, Any]) -> List[Metric]:
        """Collect prediction metrics"""
        
        metrics = []
        
        for model_id, data in model_data.items():
            if 'prediction_metrics' in data:
                pred_metrics = data['prediction_metrics']
                
                for metric_name, metric_value in pred_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        metrics.append(Metric(
                            name=f"model.prediction.{metric_name}",
                            value=metric_value,
                            category=MetricCategory.MODEL,
                            metric_type=MetricType.GAUGE,
                            labels={"model_id": model_id},
                            source="model",
                            component="prediction"
                        ))
            
            # Prediction count
            if 'prediction_count' in data:
                metrics.append(Metric(
                    name="model.prediction.count",
                    value=data['prediction_count'],
                    category=MetricCategory.MODEL,
                    metric_type=MetricType.COUNTER,
                    labels={"model_id": model_id},
                    source="model",
                    component="prediction"
                ))
        
        return metrics

class TradingMetricsCollector:
    """Collects trading-related metrics"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.TradingCollector")
        self.last_trade_time = datetime.now()
    
    def collect(self, trading_data: Optional[Dict[str, Any]] = None) -> List[Metric]:
        """Collect trading metrics"""
        
        metrics = []
        
        try:
            # Trade metrics
            if self.config.collect_trade_metrics and trading_data:
                metrics.extend(self._collect_trade_metrics(trading_data))
            
            # Position metrics
            if self.config.collect_position_metrics and trading_data:
                metrics.extend(self._collect_position_metrics(trading_data))
            
            # Portfolio metrics
            if self.config.collect_portfolio_metrics and trading_data:
                metrics.extend(self._collect_portfolio_metrics(trading_data))
            
            # Risk metrics
            if self.config.collect_risk_metrics and trading_data:
                metrics.extend(self._collect_risk_metrics(trading_data))
            
            self.logger.debug(f"Collected {len(metrics)} trading metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to collect trading metrics: {str(e)}")
        
        return metrics
    
    def _collect_trade_metrics(self, trading_data: Dict[str, Any]) -> List[Metric]:
        """Collect trade-related metrics"""
        
        metrics = []
        
        if 'trades' in trading_data:
            trades = trading_data['trades']
            
            if trades and len(trades) > 0:
                # Calculate trade statistics
                trade_pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
                trade_durations = [t.get('duration', 0) for t in trades if 'duration' in t]
                
                if trade_pnls:
                    metrics.extend([
                        Metric(
                            name="trading.trades.pnl.total",
                            value=sum(trade_pnls),
                            category=MetricCategory.TRADING,
                            metric_type=MetricType.GAUGE,
                            source="trading",
                            component="trades"
                        ),
                        Metric(
                            name="trading.trades.pnl.avg",
                            value=np.mean(trade_pnls),
                            category=MetricCategory.TRADING,
                            metric_type=MetricType.GAUGE,
                            source="trading",
                            component="trades"
                        ),
                        Metric(
                            name="trading.trades.winning",
                            value=sum(1 for pnl in trade_pnls if pnl > 0),
                            category=MetricCategory.TRADING,
                            metric_type=MetricType.GAUGE,
                            source="trading",
                            component="trades"
                        ),
                        Metric(
                            name="trading.trades.losing",
                            value=sum(1 for pnl in trade_pnls if pnl <= 0),
                            category=MetricCategory.TRADING,
                            metric_type=MetricType.GAUGE,
                            source="trading",
                            component="trades"
                        )
                    ])
                
                if trade_durations:
                    metrics.append(Metric(
                        name="trading.trades.duration.avg",
                        value=np.mean(trade_durations),
                        category=MetricCategory.TRADING,
                        metric_type=MetricType.GAUGE,
                        source="trading",
                        component="trades"
                    ))
            
            # Trade count
            metrics.append(Metric(
                name="trading.trades.count",
                value=len(trades),
                category=MetricCategory.TRADING,
                metric_type=MetricType.COUNTER,
                source="trading",
                component="trades"
            ))
        
        # Trade rate (trades per minute)
        current_time = datetime.now()
        time_diff = (current_time - self.last_trade_time).total_seconds() / 60  # minutes
        
        if 'trade_count' in trading_data and time_diff > 0:
            trade_rate = trading_data['trade_count'] / time_diff
            metrics.append(Metric(
                name="trading.trades.rate",
                value=trade_rate,
                category=MetricCategory.TRADING,
                metric_type=MetricType.GAUGE,
                source="trading",
                component="trades"
            ))
        
        self.last_trade_time = current_time
        
        return metrics
    
    def _collect_position_metrics(self, trading_data: Dict[str, Any]) -> List[Metric]:
        """Collect position metrics"""
        
        metrics = []
        
        if 'positions' in trading_data:
            positions = trading_data['positions']
            
            if positions:
                # Calculate position statistics
                position_sizes = [p.get('size', 0) for p in positions]
                position_pnls = [p.get('unrealized_pnl', 0) for p in positions]
                
                if position_sizes:
                    metrics.extend([
                        Metric(
                            name="trading.positions.count",
                            value=len(positions),
                            category=MetricCategory.TRADING,
                            metric_type=MetricType.GAUGE,
                            source="trading",
                            component="positions"
                        ),
                        Metric(
                            name="trading.positions.size.total",
                            value=sum(position_sizes),
                            category=MetricCategory.TRADING,
                            metric_type=MetricType.GAUGE,
                            source="trading",
                            component="positions"
                        ),
                        Metric(
                            name="trading.positions.size.avg",
                            value=np.mean(position_sizes),
                            category=MetricCategory.TRADING,
                            metric_type=MetricType.GAUGE,
                            source="trading",
                            component="positions"
                        )
                    ])
                
                if position_pnls:
                    metrics.extend([
                        Metric(
                            name="trading.positions.pnl.total",
                            value=sum(position_pnls),
                            category=MetricCategory.TRADING,
                            metric_type=MetricType.GAUGE,
                            source="trading",
                            component="positions"
                        ),
                        Metric(
                            name="trading.positions.pnl.avg",
                            value=np.mean(position_pnls),
                            category=MetricCategory.TRADING,
                            metric_type=MetricType.GAUGE,
                            source="trading",
                            component="positions"
                        )
                    ])
        
        return metrics
    
    def _collect_portfolio_metrics(self, trading_data: Dict[str, Any]) -> List[Metric]:
        """Collect portfolio metrics"""
        
        metrics = []
        
        if 'portfolio' in trading_data:
            portfolio = trading_data['portfolio']
            
            for metric_name, metric_value in portfolio.items():
                if isinstance(metric_value, (int, float)):
                    metrics.append(Metric(
                        name=f"trading.portfolio.{metric_name}",
                        value=metric_value,
                        category=MetricCategory.TRADING,
                        metric_type=MetricType.GAUGE,
                        source="trading",
                        component="portfolio"
                    ))
        
        return metrics
    
    def _collect_risk_metrics(self, trading_data: Dict[str, Any]) -> List[Metric]:
        """Collect risk metrics"""
        
        metrics = []
        
        if 'risk_metrics' in trading_data:
            risk_metrics = trading_data['risk_metrics']
            
            for metric_name, metric_value in risk_metrics.items():
                if isinstance(metric_value, (int, float)):
                    metrics.append(Metric(
                        name=f"trading.risk.{metric_name}",
                        value=metric_value,
                        category=MetricCategory.TRADING,
                        metric_type=MetricType.GAUGE,
                        source="trading",
                        component="risk"
                    ))
        
        # Calculate volatility if we have returns data
        if 'returns' in trading_data and trading_data['returns']:
            returns = trading_data['returns']
            if len(returns) > 1:
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                metrics.append(Metric(
                    name="trading.risk.volatility",
                    value=volatility,
                    category=MetricCategory.TRADING,
                    metric_type=MetricType.GAUGE,
                    source="trading",
                    component="risk"
                ))
        
        return metrics

class MarketMetricsCollector:
    """Collects market-related metrics"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.MarketCollector")
        self.last_prices = {}
    
    def collect(self, market_data: Optional[Dict[str, Any]] = None) -> List[Metric]:
        """Collect market metrics"""
        
        metrics = []
        
        try:
            # Price metrics
            if self.config.collect_price_metrics and market_data:
                metrics.extend(self._collect_price_metrics(market_data))
            
            # Volume metrics
            if self.config.collect_volume_metrics and market_data:
                metrics.extend(self._collect_volume_metrics(market_data))
            
            # Volatility metrics
            if self.config.collect_volatility_metrics and market_data:
                metrics.extend(self._collect_volatility_metrics(market_data))
            
            # Order book metrics
            if self.config.collect_order_book_metrics and market_data:
                metrics.extend(self._collect_order_book_metrics(market_data))
            
            self.logger.debug(f"Collected {len(metrics)} market metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to collect market metrics: {str(e)}")
        
        return metrics
    
    def _collect_price_metrics(self, market_data: Dict[str, Any]) -> List[Metric]:
        """Collect price metrics"""
        
        metrics = []
        current_time = datetime.now()
        
        for symbol, data in market_data.items():
            if 'price' in data:
                current_price = data['price']
                
                # Store current price
                metrics.append(Metric(
                    name="market.price.current",
                    value=current_price,
                    category=MetricCategory.MARKET,
                    metric_type=MetricType.GAUGE,
                    labels={"symbol": symbol},
                    source="market",
                    component="price"
                ))
                
                # Calculate price change if we have previous price
                if symbol in self.last_prices:
                    last_price, last_time = self.last_prices[symbol]
                    time_diff = (current_time - last_time).total_seconds() / 3600  # hours
                    
                    if time_diff > 0 and last_price > 0:
                        price_change = ((current_price - last_price) / last_price) * 100
                        price_change_per_hour = price_change / time_diff
                        
                        metrics.extend([
                            Metric(
                                name="market.price.change",
                                value=price_change,
                                category=MetricCategory.MARKET,
                                metric_type=MetricType.GAUGE,
                                labels={"symbol": symbol, "period": "since_last"},
                                source="market",
                                component="price"
                            ),
                            Metric(
                                name="market.price.change_rate",
                                value=price_change_per_hour,
                                category=MetricCategory.MARKET,
                                metric_type=MetricType.GAUGE,
                                labels={"symbol": symbol, "unit": "per_hour"},
                                source="market",
                                component="price"
                            )
                        ])
                
                # Update last price
                self.last_prices[symbol] = (current_price, current_time)
            
            # Add OHLC data if available
            for ohlc_type in ['open', 'high', 'low', 'close']:
                if ohlc_type in data:
                    metrics.append(Metric(
                        name=f"market.price.{ohlc_type}",
                        value=data[ohlc_type],
                        category=MetricCategory.MARKET,
                        metric_type=MetricType.GAUGE,
                        labels={"symbol": symbol},
                        source="market",
                        component="price"
                    ))
        
        return metrics
    
    def _collect_volume_metrics(self, market_data: Dict[str, Any]) -> List[Metric]:
        """Collect volume metrics"""
        
        metrics = []
        
        for symbol, data in market_data.items():
            if 'volume' in data:
                volume = data['volume']
                
                metrics.append(Metric(
                    name="market.volume.current",
                    value=volume,
                    category=MetricCategory.MARKET,
                    metric_type=MetricType.GAUGE,
                    labels={"symbol": symbol},
                    source="market",
                    component="volume"
                ))
            
            # Volume statistics if available
            if 'volume_24h' in data:
                metrics.append(Metric(
                    name="market.volume.24h",
                    value=data['volume_24h'],
                    category=MetricCategory.MARKET,
                    metric_type=MetricType.GAUGE,
                    labels={"symbol": symbol},
                    source="market",
                    component="volume"
                ))
            
            if 'volume_change_24h' in data:
                metrics.append(Metric(
                    name="market.volume.change_24h",
                    value=data['volume_change_24h'],
                    category=MetricCategory.MARKET,
                    metric_type=MetricType.GAUGE,
                    labels={"symbol": symbol},
                    source="market",
                    component="volume"
                ))
        
        return metrics
    
    def _collect_volatility_metrics(self, market_data: Dict[str, Any]) -> List[Metric]:
        """Collect volatility metrics"""
        
        metrics = []
        
        for symbol, data in market_data.items():
            # Historical volatility if we have price history
            if 'price_history' in data and len(data['price_history']) > 1:
                prices = data['price_history']
                returns = np.diff(prices) / prices[:-1]
                
                if len(returns) > 0:
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
                    metrics.append(Metric(
                        name="market.volatility.historical",
                        value=volatility,
                        category=MetricCategory.MARKET,
                        metric_type=MetricType.GAUGE,
                        labels={"symbol": symbol, "period": "annualized"},
                        source="market",
                        component="volatility"
                    ))
            
            # Implied volatility if available
            if 'implied_volatility' in data:
                metrics.append(Metric(
                    name="market.volatility.implied",
                    value=data['implied_volatility'],
                    category=MetricCategory.MARKET,
                    metric_type=MetricType.GAUGE,
                    labels={"symbol": symbol},
                    source="market",
                    component="volatility"
                ))
        
        return metrics
    
    def _collect_order_book_metrics(self, market_data: Dict[str, Any]) -> List[Metric]:
        """Collect order book metrics"""
        
        metrics = []
        
        for symbol, data in market_data.items():
            if 'order_book' in data:
                order_book = data['order_book']
                
                # Bid/ask spread
                if 'best_bid' in order_book and 'best_ask' in order_book:
                    spread = order_book['best_ask'] - order_book['best_bid']
                    spread_percent = (spread / order_book['best_bid']) * 100 if order_book['best_bid'] > 0 else 0
                    
                    metrics.extend([
                        Metric(
                            name="market.order_book.spread",
                            value=spread,
                            category=MetricCategory.MARKET,
                            metric_type=MetricType.GAUGE,
                            labels={"symbol": symbol},
                            source="market",
                            component="order_book"
                        ),
                        Metric(
                            name="market.order_book.spread_percent",
                            value=spread_percent,
                            category=MetricCategory.MARKET,
                            metric_type=MetricType.GAUGE,
                            labels={"symbol": symbol},
                            source="market",
                            component="order_book"
                        )
                    ])
                
                # Order book depth
                if 'bid_depth' in order_book and 'ask_depth' in order_book:
                    total_depth = order_book['bid_depth'] + order_book['ask_depth']
                    depth_ratio = order_book['bid_depth'] / order_book['ask_depth'] if order_book['ask_depth'] > 0 else 0
                    
                    metrics.extend([
                        Metric(
                            name="market.order_book.depth.total",
                            value=total_depth,
                            category=MetricCategory.MARKET,
                            metric_type=MetricType.GAUGE,
                            labels={"symbol": symbol},
                            source="market",
                            component="order_book"
                        ),
                        Metric(
                            name="market.order_book.depth.ratio",
                            value=depth_ratio,
                            category=MetricCategory.MARKET,
                            metric_type=MetricType.GAUGE,
                            labels={"symbol": symbol},
                            source="market",
                            component="order_book"
                        )
                    ])
        
        return metrics

# ============ Storage Backends ============
class MemoryStorageBackend:
    """In-memory storage backend for metrics"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.MemoryStorage")
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.aggregated_metrics: Dict[str, List[AggregatedMetric]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def store(self, metrics: List[Metric]) -> bool:
        """Store metrics in memory"""
        
        with self.lock:
            for metric in metrics:
                key = self._get_metric_key(metric)
                self.metrics[key].append(metric)
                
                # Apply retention policy
                if len(self.metrics[key]) > self.config.max_memory_metrics:
                    self.metrics[key] = self.metrics[key][-self.config.max_memory_metrics:]
            
            return True
    
    def retrieve(self, 
                name: str,
                start_time: Optional[datetime] = None,
                end_time: Optional[datetime] = None,
                labels: Optional[Dict[str, str]] = None) -> List[Metric]:
        """Retrieve metrics from memory"""
        
        with self.lock:
            key = self._build_key(name, labels)
            
            if key not in self.metrics:
                return []
            
            metrics = self.metrics[key]
            
            # Apply time filters
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
            
            return sorted(metrics, key=lambda x: x.timestamp)
    
    def aggregate(self, 
                 name: str,
                 window: str,
                 aggregation_method: AggregationMethod,
                 labels: Optional[Dict[str, str]] = None) -> Optional[AggregatedMetric]:
        """Aggregate metrics in memory"""
        
        metrics = self.retrieve(name, labels=labels)
        
        if not metrics:
            return None
        
        # Filter to numeric values
        numeric_metrics = [m for m in metrics if isinstance(m.value, (int, float))]
        
        if not numeric_metrics:
            return None
        
        values = [float(m.value) for m in numeric_metrics]
        timestamps = [m.timestamp for m in numeric_metrics]
        
        # Calculate aggregation
        count = len(values)
        total = sum(values)
        avg = total / count if count > 0 else 0
        min_val = min(values) if values else 0
        max_val = max(values) if values else 0
        std_val = np.std(values) if len(values) > 1 else 0
        
        # Calculate percentiles
        percentiles = {}
        for p in self.config.percentiles:
            if values:
                percentiles[p] = np.percentile(values, p * 100)
        
        # Determine window
        window_end = max(timestamps) if timestamps else datetime.now()
        window_start = window_end - self._parse_window(window)
        
        # Create aggregated metric
        aggregated = AggregatedMetric(
            name=name,
            values=values,
            timestamps=timestamps,
            count=count,
            sum=total,
            avg=avg,
            min=min_val,
            max=max_val,
            std=std_val,
            percentiles=percentiles,
            category=numeric_metrics[0].category if numeric_metrics else MetricCategory.CUSTOM,
            labels=labels or {},
            aggregation_window=window,
            aggregation_method=aggregation_method,
            window_start=window_start,
            window_end=window_end
        )
        
        # Store aggregated result
        agg_key = f"{name}_{window}_{aggregation_method.value}"
        self.aggregated_metrics[agg_key].append(aggregated)
        
        return aggregated
    
    def _get_metric_key(self, metric: Metric) -> str:
        """Get storage key for metric"""
        label_str = "_".join(f"{k}={v}" for k, v in sorted(metric.labels.items()))
        return f"{metric.name}_{label_str}"
    
    def _build_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Build storage key"""
        if not labels:
            return name
        label_str = "_".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}_{label_str}"
    
    def _parse_window(self, window: str) -> timedelta:
        """Parse time window string to timedelta"""
        
        if window.endswith('m'):
            minutes = int(window[:-1])
            return timedelta(minutes=minutes)
        elif window.endswith('h'):
            hours = int(window[:-1])
            return timedelta(hours=hours)
        elif window.endswith('d'):
            days = int(window[:-1])
            return timedelta(days=days)
        elif window.endswith('w'):
            weeks = int(window[:-1])
            return timedelta(weeks=weeks)
        else:
            # Default to minutes
            return timedelta(minutes=int(window))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        with self.lock:
            total_metrics = sum(len(metrics) for metrics in self.metrics.values())
            total_aggregated = sum(len(agg) for agg in self.aggregated_metrics.values())
            
            return {
                'backend': 'memory',
                'metric_count': total_metrics,
                'aggregated_count': total_aggregated,
                'unique_metrics': len(self.metrics),
                'unique_aggregated': len(self.aggregated_metrics),
                'max_memory_metrics': self.config.max_memory_metrics
            }

class DiskStorageBackend:
    """Disk-based storage backend for metrics"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.DiskStorage")
        self.storage_path = Path(config.disk_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
    
    def store(self, metrics: List[Metric]) -> bool:
        """Store metrics to disk"""
        
        try:
            with self.lock:
                # Group metrics by date and category
                metrics_by_date = defaultdict(lambda: defaultdict(list))
                
                for metric in metrics:
                    date_str = metric.timestamp.strftime('%Y-%m-%d')
                    category = metric.category.value
                    metrics_by_date[date_str][category].append(metric)
                
                # Save each date and category
                for date_str, categories in metrics_by_date.items():
                    for category, category_metrics in categories.items():
                        filepath = self.storage_path / f"metrics_{date_str}_{category}.json"
                        
                        # Load existing metrics
                        existing_metrics = []
                        if filepath.exists():
                            try:
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                    existing_metrics = [Metric(**m) for m in data]
                            except Exception as e:
                                self.logger.warning(f"Failed to load existing metrics from {filepath}: {str(e)}")
                        
                        # Add new metrics
                        all_metrics = existing_metrics + category_metrics
                        
                        # Apply retention (keep only last N days worth of data)
                        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
                        filtered_metrics = [m for m in all_metrics if m.timestamp >= cutoff_date]
                        
                        # Save back to file
                        with open(filepath, 'w') as f:
                            json.dump([m.to_dict() for m in filtered_metrics], f, indent=2, default=str)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store metrics to disk: {str(e)}")
            return False
    
    def retrieve(self, 
                name: str,
                start_time: Optional[datetime] = None,
                end_time: Optional[datetime] = None,
                labels: Optional[Dict[str, str]] = None) -> List[Metric]:
        """Retrieve metrics from disk"""
        
        metrics = []
        
        try:
            with self.lock:
                # Determine date range
                if start_time is None:
                    start_time = datetime.now() - timedelta(days=self.config.retention_days)
                
                if end_time is None:
                    end_time = datetime.now()
                
                # Iterate through date files
                current_date = start_time.date()
                end_date = end_time.date()
                
                while current_date <= end_date:
                    date_str = current_date.strftime('%Y-%m-%d')
                    
                    # Check all category files for this date
                    for category in MetricCategory:
                        filepath = self.storage_path / f"metrics_{date_str}_{category.value}.json"
                        
                        if filepath.exists():
                            try:
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                    
                                    for metric_data in data:
                                        metric = Metric(**metric_data)
                                        metric.timestamp = datetime.fromisoformat(metric_data['timestamp'])
                                        
                                        # Apply filters
                                        if metric.name != name:
                                            continue
                                        
                                        if metric.timestamp < start_time or metric.timestamp > end_time:
                                            continue
                                        
                                        if labels and not self._labels_match(metric.labels, labels):
                                            continue
                                        
                                        metrics.append(metric)
                                        
                            except Exception as e:
                                self.logger.warning(f"Failed to load metrics from {filepath}: {str(e)}")
                    
                    current_date += timedelta(days=1)
                
                return sorted(metrics, key=lambda x: x.timestamp)
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve metrics from disk: {str(e)}")
            return []
    
    def _labels_match(self, metric_labels: Dict[str, str], filter_labels: Dict[str, str]) -> bool:
        """Check if metric labels match filter labels"""
        
        for key, value in filter_labels.items():
            if key not in metric_labels or metric_labels[key] != value:
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        try:
            with self.lock:
                files = list(self.storage_path.glob("metrics_*.json"))
                total_size = sum(f.stat().st_size for f in files)
                
                return {
                    'backend': 'disk',
                    'file_count': len(files),
                    'total_size_bytes': total_size,
                    'storage_path': str(self.storage_path),
                    'retention_days': self.config.retention_days
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get disk storage statistics: {str(e)}")
            return {'backend': 'disk', 'error': str(e)}

class InfluxDBStorageBackend:
    """InfluxDB storage backend for metrics"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.InfluxDBStorage")
        self.client = None
        
        if INFLUXDB_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize InfluxDB client"""
        
        try:
            influx_config = self.config.influxdb_config
            
            self.client = InfluxDBClient(
                host=influx_config.get('host', 'localhost'),
                port=influx_config.get('port', 8086),
                username=influx_config.get('username', ''),
                password=influx_config.get('password', ''),
                database=influx_config.get('database', 'trading_metrics'),
                ssl=influx_config.get('ssl', False),
                verify_ssl=influx_config.get('verify_ssl', False)
            )
            
            # Create database if it doesn't exist
            databases = self.client.get_list_database()
            db_name = influx_config.get('database', 'trading_metrics')
            
            if not any(db['name'] == db_name for db in databases):
                self.client.create_database(db_name)
            
            self.logger.info(f"InfluxDB client initialized for database: {db_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize InfluxDB client: {str(e)}")
            self.client = None
    
    def store(self, metrics: List[Metric]) -> bool:
        """Store metrics to InfluxDB"""
        
        if not self.client:
            self.logger.warning("InfluxDB client not initialized")
            return False
        
        try:
            # Convert metrics to InfluxDB points
            points = []
            for metric in metrics:
                point = metric.to_influxdb_point()
                points.append(point)
            
            # Write points in batches
            batch_size = self.config.batch_size
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.write_points(batch)
            
            self.logger.debug(f"Stored {len(points)} metrics to InfluxDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics to InfluxDB: {str(e)}")
            return False
    
    def retrieve(self, 
                name: str,
                start_time: Optional[datetime] = None,
                end_time: Optional[datetime] = None,
                labels: Optional[Dict[str, str]] = None) -> List[Metric]:
        """Retrieve metrics from InfluxDB"""
        
        if not self.client:
            self.logger.warning("InfluxDB client not initialized")
            return []
        
        try:
            # Build query
            query = f'SELECT * FROM "{name}"'
            
            conditions = []
            if start_time:
                conditions.append(f'time >= \'{start_time.isoformat()}\'')
            if end_time:
                conditions.append(f'time <= \'{end_time.isoformat()}\'')
            if labels:
                for key, value in labels.items():
                    conditions.append(f'"{key}" = \'{value}\'')
            
            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)
            
            query += ' ORDER BY time'
            
            # Execute query
            result = self.client.query(query)
            
            # Convert to Metric objects
            metrics = []
            for point in result.get_points():
                metric = Metric(
                    name=name,
                    value=point['value'],
                    timestamp=datetime.fromisoformat(point['time'].replace('Z', '+00:00')),
                    category=MetricCategory(point.get('category', 'custom')),
                    labels={k: v for k, v in point.items() 
                           if k not in ['time', 'value', 'category', 'measurement']}
                )
                metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve metrics from InfluxDB: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        if not self.client:
            return {'backend': 'influxdb', 'status': 'not_initialized'}
        
        try:
            # Get database statistics
            influx_config = self.config.influxdb_config
            db_name = influx_config.get('database', 'trading_metrics')
            
            # Query for some basic statistics
            query = f'SHOW MEASUREMENTS ON "{db_name}"'
            result = self.client.query(query)
            measurements = list(result.get_points())
            
            return {
                'backend': 'influxdb',
                'database': db_name,
                'measurement_count': len(measurements),
                'status': 'connected',
                'host': influx_config.get('host', 'localhost')
            }
            
        except Exception as e:
            return {'backend': 'influxdb', 'status': 'error', 'error': str(e)}

# ============ Main Metrics Collector ============
class MetricsCollector:
    """Main metrics collection engine"""
    
    def __init__(self, 
                 config: MetricsConfig,
                 model_manager: Optional[ModelManager] = None,
                 performance_tracker: Optional[PerformanceTracker] = None):
        
        self.config = config
        self.model_manager = model_manager
        self.performance_tracker = performance_tracker
        self.logger = get_logger(__name__)
        
        # Initialize collectors
        self.collectors: Dict[str, Any] = {}
        self._initialize_collectors()
        
        # Initialize storage backend
        self.storage_backend = self._initialize_storage_backend()
        
        # Metrics buffers
        self.metrics_buffer: deque = deque(maxlen=config.buffer_size)
        self.batch_queue = queue.Queue(maxsize=1000)
        
        # Statistics
        self.collection_count = 0
        self.storage_count = 0
        self.error_count = 0
        self.last_collection_time = datetime.now()
        self.start_time = datetime.now()
        
        # Thread management
        self.collection_thread: Optional[threading.Thread] = None
        self.storage_thread: Optional[threading.Thread] = None
        self.aggregation_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Prometheus metrics (if enabled)
        self.prometheus_metrics: Dict[str, Any] = {}
        if PROMETHEUS_AVAILABLE and config.storage_backend == StorageBackend.PROMETHEUS:
            self._initialize_prometheus()
        
        # Scheduled tasks
        self.scheduled_tasks: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Metrics Collector initialized")
    
    def _initialize_collectors(self):
        """Initialize metric collectors"""
        
        if self.config.collect_system_metrics:
            self.collectors['system'] = SystemMetricsCollector(self.config)
        
        if self.config.collect_model_metrics:
            self.collectors['model'] = ModelMetricsCollector(self.config, self.model_manager)
        
        if self.config.collect_trading_metrics:
            self.collectors['trading'] = TradingMetricsCollector(self.config)
        
        if self.config.collect_market_metrics:
            self.collectors['market'] = MarketMetricsCollector(self.config)
    
    def _initialize_storage_backend(self) -> Any:
        """Initialize storage backend based on config"""
        
        if self.config.storage_backend == StorageBackend.MEMORY:
            return MemoryStorageBackend(self.config)
        
        elif self.config.storage_backend == StorageBackend.DISK:
            return DiskStorageBackend(self.config)
        
        elif self.config.storage_backend == StorageBackend.INFLUXDB:
            return InfluxDBStorageBackend(self.config)
        
        elif self.config.storage_backend == StorageBackend.PROMETHEUS:
            # Prometheus is handled separately
            return None
        
        else:
            self.logger.warning(f"Unknown storage backend: {self.config.storage_backend}")
            return MemoryStorageBackend(self.config)
    
    def _initialize_prometheus(self):
        """Initialize Prometheus metrics"""
        
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Create common Prometheus metrics
        self.prometheus_metrics['metrics_collected'] = Counter(
            'metrics_collector_collected_total',
            'Total number of metrics collected',
            ['category', 'source']
        )
        
        self.prometheus_metrics['metrics_stored'] = Counter(
            'metrics_collector_stored_total',
            'Total number of metrics stored',
            ['backend']
        )
        
        self.prometheus_metrics['collection_errors'] = Counter(
            'metrics_collector_errors_total',
            'Total number of collection errors',
            ['collector']
        )
        
        self.prometheus_metrics['collection_duration'] = Histogram(
            'metrics_collector_collection_duration_seconds',
            'Duration of metric collection',
            ['collector']
        )
        
        self.logger.info("Prometheus metrics initialized")
    
    def start(self):
        """Start metrics collection"""
        
        if not self.config.enabled:
            self.logger.warning("Metrics collection is disabled")
            return
        
        if self.running:
            self.logger.warning("Metrics collector is already running")
            return
        
        self.running = True
        
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            name="MetricsCollectionThread",
            daemon=True
        )
        self.collection_thread.start()
        
        # Start storage thread
        self.storage_thread = threading.Thread(
            target=self._storage_loop,
            name="MetricsStorageThread",
            daemon=True
        )
        self.storage_thread.start()
        
        # Start aggregation thread
        self.aggregation_thread = threading.Thread(
            target=self._aggregation_loop,
            name="MetricsAggregationThread",
            daemon=True
        )
        self.aggregation_thread.start()
        
        # Start Prometheus server if enabled
        if (PROMETHEUS_AVAILABLE and 
            self.config.storage_backend == StorageBackend.PROMETHEUS and
            self.config.enable_monitoring):
            
            prometheus_config = self.config.prometheus_config
            port = prometheus_config.get('port', 9090)
            
            try:
                prometheus_client.start_http_server(port)
                self.logger.info(f"Prometheus metrics server started on port {port}")
            except Exception as e:
                self.logger.error(f"Failed to start Prometheus server: {str(e)}")
        
        self.logger.info("Metrics collection started")
    
    def stop(self):
        """Stop metrics collection"""
        
        if not self.running:
            return
        
        self.running = False
        
        # Wait for threads to finish
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        if self.storage_thread:
            self.storage_thread.join(timeout=5)
        
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5)
        
        # Export final metrics if enabled
        if self.config.export_enabled:
            self.export_metrics()
        
        self.logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        
        self.logger.info("Collection loop started")
        
        # Schedule collection tasks
        self._schedule_collection_tasks()
        
        while self.running:
            try:
                # Check for scheduled tasks
                current_time = datetime.now()
                
                for task_name, task_info in list(self.scheduled_tasks.items()):
                    if current_time >= task_info['next_run']:
                        self._execute_collection_task(task_name, task_info)
                        task_info['last_run'] = current_time
                        task_info['next_run'] = current_time + task_info['interval']
                
                # Sleep to prevent tight loop
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {str(e)}")
                self.error_count += 1
                time.sleep(5)
    
    def _schedule_collection_tasks(self):
        """Schedule collection tasks based on intervals"""
        
        # System metrics
        if self.config.collect_system_metrics:
            self.scheduled_tasks['system'] = {
                'interval': timedelta(seconds=self.config.system_metrics_interval),
                'last_run': datetime.now(),
                'next_run': datetime.now(),
                'collector': 'system',
                'data': None
            }
        
        # Model metrics
        if self.config.collect_model_metrics:
            self.scheduled_tasks['model'] = {
                'interval': timedelta(seconds=self.config.model_metrics_interval),
                'last_run': datetime.now(),
                'next_run': datetime.now() + timedelta(seconds=10),  # Stagger start
                'collector': 'model',
                'data': self._get_model_data()
            }
        
        # Trading metrics
        if self.config.collect_trading_metrics:
            self.scheduled_tasks['trading'] = {
                'interval': timedelta(seconds=self.config.trading_metrics_interval),
                'last_run': datetime.now(),
                'next_run': datetime.now() + timedelta(seconds=20),  # Stagger start
                'collector': 'trading',
                'data': self._get_trading_data()
            }
        
        # Market metrics
        if self.config.collect_market_metrics:
            self.scheduled_tasks['market'] = {
                'interval': timedelta(seconds=self.config.market_metrics_interval),
                'last_run': datetime.now(),
                'next_run': datetime.now() + timedelta(seconds=30),  # Stagger start
                'collector': 'market',
                'data': self._get_market_data()
            }
    
    def _execute_collection_task(self, task_name: str, task_info: Dict[str, Any]):
        """Execute a collection task"""
        
        collector_name = task_info['collector']
        data = task_info['data']
        
        if collector_name not in self.collectors:
            return
        
        collector = self.collectors[collector_name]
        
        try:
            start_time = time.time()
            
            # Collect metrics
            metrics = collector.collect(data)
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and self.config.storage_backend == StorageBackend.PROMETHEUS:
                if 'metrics_collected' in self.prometheus_metrics:
                    category = collector_name
                    self.prometheus_metrics['metrics_collected'].labels(
                        category=category, source='collector'
                    ).inc(len(metrics))
                
                if 'collection_duration' in self.prometheus_metrics:
                    duration = time.time() - start_time
                    self.prometheus_metrics['collection_duration'].labels(
                        collector=collector_name
                    ).observe(duration)
            
            # Add to buffer
            self.metrics_buffer.extend(metrics)
            self.collection_count += len(metrics)
            
            # Update last collection time
            self.last_collection_time = datetime.now()
            
            self.logger.debug(f"Collected {len(metrics)} metrics from {collector_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute collection task {task_name}: {str(e)}")
            
            # Update error metrics
            if PROMETHEUS_AVAILABLE and 'collection_errors' in self.prometheus_metrics:
                self.prometheus_metrics['collection_errors'].labels(
                    collector=collector_name
                ).inc()
    
    def _storage_loop(self):
        """Storage processing loop"""
        
        self.logger.info("Storage loop started")
        
        while self.running:
            try:
                # Check if we have metrics to store
                if len(self.metrics_buffer) > 0:
                    # Take metrics from buffer
                    metrics_to_store = []
                    while len(self.metrics_buffer) > 0 and len(metrics_to_store) < self.config.batch_size:
                        try:
                            metric = self.metrics_buffer.popleft()
                            metrics_to_store.append(metric)
                        except IndexError:
                            break
                    
                    # Store metrics
                    if metrics_to_store:
                        success = self._store_metrics(metrics_to_store)
                        
                        if success:
                            self.storage_count += len(metrics_to_store)
                            
                            # Update Prometheus metrics
                            if PROMETHEUS_AVAILABLE and 'metrics_stored' in self.prometheus_metrics:
                                backend = self.config.storage_backend.value
                                self.prometheus_metrics['metrics_stored'].labels(
                                    backend=backend
                                ).inc(len(metrics_to_store))
                
                # Sleep to prevent tight loop
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in storage loop: {str(e)}")
                self.error_count += 1
                time.sleep(5)
    
    def _store_metrics(self, metrics: List[Metric]) -> bool:
        """Store metrics using configured backend"""
        
        if self.config.storage_backend == StorageBackend.PROMETHEUS:
            # Store to Prometheus
            return self._store_to_prometheus(metrics)
        elif self.storage_backend:
            # Store to configured backend
            return self.storage_backend.store(metrics)
        else:
            self.logger.warning("No storage backend configured")
            return False
    
    def _store_to_prometheus(self, metrics: List[Metric]) -> bool:
        """Store metrics to Prometheus"""
        
        if not PROMETHEUS_AVAILABLE:
            return False
        
        try:
            for metric in metrics:
                # Get or create Prometheus metric
                prom_metric = self._get_prometheus_metric(metric)
                
                if prom_metric:
                    # Update metric value
                    if isinstance(prom_metric, Counter):
                        prom_metric.labels(**metric.labels).inc(metric.value if isinstance(metric.value, (int, float)) else 1)
                    elif isinstance(prom_metric, Gauge):
                        if isinstance(metric.value, (int, float)):
                            prom_metric.labels(**metric.labels).set(metric.value)
                    elif isinstance(prom_metric, Histogram):
                        if isinstance(metric.value, (int, float)):
                            prom_metric.labels(**metric.labels).observe(metric.value)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics to Prometheus: {str(e)}")
            return False
    
    def _get_prometheus_metric(self, metric: Metric) -> Optional[Any]:
        """Get or create Prometheus metric"""
        
        if not PROMETHEUS_AVAILABLE:
            return None
        
        metric_name = f"trading_{metric.name.replace('.', '_')}"
        
        if metric_name in self.prometheus_metrics:
            return self.prometheus_metrics[metric_name]
        
        # Create new Prometheus metric
        try:
            if metric.metric_type == MetricType.COUNTER:
                prom_metric = Counter(
                    metric_name,
                    f"Metric: {metric.name}",
                    list(metric.labels.keys())
                )
            elif metric.metric_type == MetricType.GAUGE:
                prom_metric = Gauge(
                    metric_name,
                    f"Metric: {metric.name}",
                    list(metric.labels.keys())
                )
            elif metric.metric_type == MetricType.HISTOGRAM:
                prom_metric = Histogram(
                    metric_name,
                    f"Metric: {metric.name}",
                    list(metric.labels.keys())
                )
            else:
                prom_metric = Gauge(
                    metric_name,
                    f"Metric: {metric.name}",
                    list(metric.labels.keys())
                )
            
            self.prometheus_metrics[metric_name] = prom_metric
            return prom_metric
            
        except Exception as e:
            self.logger.error(f"Failed to create Prometheus metric {metric_name}: {str(e)}")
            return None
    
    def _aggregation_loop(self):
        """Aggregation processing loop"""
        
        self.logger.info("Aggregation loop started")
        
        while self.running:
            try:
                # Check if it's time to run aggregations
                current_time = datetime.now()
                aggregation_interval = timedelta(minutes=self.config.aggregation_interval_minutes)
                
                # Run aggregations at configured interval
                if current_time.minute % self.config.aggregation_interval_minutes == 0 and current_time.second < 10:
                    self._run_aggregations()
                
                # Sleep to prevent tight loop
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {str(e)}")
                self.error_count += 1
                time.sleep(30)
    
    def _run_aggregations(self):
        """Run metric aggregations"""
        
        self.logger.info("Running metric aggregations")
        
        # Get common metric names to aggregate
        common_metrics = [
            'system.cpu.usage',
            'system.memory.usage',
            'system.disk.usage',
            'model.inference.time',
            'model.accuracy',
            'trading.trades.pnl.total',
            'market.price.current'
        ]
        
        for metric_name in common_metrics:
            for window in self.config.aggregation_windows:
                try:
                    # Aggregate using different methods
                    for method in [AggregationMethod.AVG, AggregationMethod.MAX, AggregationMethod.MIN]:
                        aggregated = self.aggregate_metric(
                            name=metric_name,
                            window=window,
                            aggregation_method=method
                        )
                        
                        if aggregated:
                            # Store aggregated metric
                            aggregated_metric = Metric(
                                name=f"{metric_name}.{window}.{method.value}",
                                value=aggregated.avg if method == AggregationMethod.AVG else
                                      aggregated.max if method == AggregationMethod.MAX else
                                      aggregated.min,
                                category=MetricCategory.PERFORMANCE,
                                metric_type=MetricType.GAUGE,
                                labels=aggregated.labels,
                                source="aggregator",
                                component="aggregation",
                                is_aggregated=True,
                                aggregation_window=window,
                                aggregation_method=method
                            )
                            
                            self._store_metrics([aggregated_metric])
                            
                except Exception as e:
                    self.logger.error(f"Failed to aggregate {metric_name} for window {window}: {str(e)}")
    
    def _get_model_data(self) -> Dict[str, Any]:
        """Get model data for collection"""
        
        model_data = {}
        
        if self.model_manager:
            try:
                models = self.model_manager.list_models()
                
                for model_id in models[:10]:  # Limit to first 10 models
                    model_info = self.model_manager.get_model_info(model_id)
                    
                    if model_info:
                        model_data[model_id] = {
                            'model_type': model_info.model_type.value,
                            'performance_metrics': model_info.performance_metrics,
                            'created_at': model_info.created_at.isoformat(),
                            'updated_at': model_info.updated_at.isoformat()
                        }
            except Exception as e:
                self.logger.error(f"Failed to get model data: {str(e)}")
        
        return model_data
    
    def _get_trading_data(self) -> Dict[str, Any]:
        """Get trading data for collection"""
        
        trading_data = {}
        
        # This would be populated from your trading system
        # For now, return empty data
        return trading_data
    
    def _get_market_data(self) -> Dict[str, Any]:
        """Get market data for collection"""
        
        market_data = {}
        
        # This would be populated from your market data sources
        # For now, return empty data
        return market_data
    
    def collect_custom_metric(self,
                             name: str,
                             value: Union[float, int, str, bool],
                             category: MetricCategory = MetricCategory.CUSTOM,
                             metric_type: MetricType = MetricType.GAUGE,
                             labels: Optional[Dict[str, str]] = None,
                             source: str = "custom",
                             component: str = "custom") -> bool:
        """Collect a custom metric"""
        
        if not self.config.enabled:
            return False
        
        try:
            metric = Metric(
                name=name,
                value=value,
                category=category,
                metric_type=metric_type,
                labels=labels or {},
                source=source,
                component=component
            )
            
            # Add to buffer
            self.metrics_buffer.append(metric)
            self.collection_count += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to collect custom metric {name}: {str(e)}")
            return False
    
    def get_metric(self,
                  name: str,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  labels: Optional[Dict[str, str]] = None) -> List[Metric]:
        """Get metrics by name and optional filters"""
        
        if not self.storage_backend:
            return []
        
        return self.storage_backend.retrieve(name, start_time, end_time, labels)
    
    def aggregate_metric(self,
                        name: str,
                        window: str,
                        aggregation_method: AggregationMethod,
                        labels: Optional[Dict[str, str]] = None) -> Optional[AggregatedMetric]:
        """Aggregate metrics over a time window"""
        
        if not self.storage_backend:
            return None
        
        if hasattr(self.storage_backend, 'aggregate'):
            return self.storage_backend.aggregate(name, window, aggregation_method, labels)
        
        # Manual aggregation for backends that don't support it natively
        metrics = self.get_metric(name, labels=labels)
        
        if not metrics:
            return None
        
        # Parse window
        window_end = datetime.now()
        if window.endswith('m'):
            window_start = window_end - timedelta(minutes=int(window[:-1]))
        elif window.endswith('h'):
            window_start = window_end - timedelta(hours=int(window[:-1]))
        elif window.endswith('d'):
            window_start = window_end - timedelta(days=int(window[:-1]))
        else:
            window_start = window_end - timedelta(minutes=5)  # Default 5 minutes
        
        # Filter metrics by window
        window_metrics = [m for m in metrics if window_start <= m.timestamp <= window_end]
        
        if not window_metrics:
            return None
        
        # Filter to numeric values
        numeric_metrics = [m for m in window_metrics if isinstance(m.value, (int, float))]
        
        if not numeric_metrics:
            return None
        
        values = [float(m.value) for m in numeric_metrics]
        timestamps = [m.timestamp for m in numeric_metrics]
        
        # Calculate aggregation
        count = len(values)
        total = sum(values)
        avg = total / count if count > 0 else 0
        min_val = min(values) if values else 0
        max_val = max(values) if values else 0
        std_val = np.std(values) if len(values) > 1 else 0
        
        # Calculate percentiles
        percentiles = {}
        for p in self.config.percentiles:
            if values:
                percentiles[p] = np.percentile(values, p * 100)
        
        # Create aggregated metric
        aggregated = AggregatedMetric(
            name=name,
            values=values,
            timestamps=timestamps,
            count=count,
            sum=total,
            avg=avg,
            min=min_val,
            max=max_val,
            std=std_val,
            percentiles=percentiles,
            category=numeric_metrics[0].category,
            labels=labels or {},
            aggregation_window=window,
            aggregation_method=aggregation_method,
            window_start=window_start,
            window_end=window_end
        )
        
        return aggregated
    
    def export_metrics(self, 
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      format: Optional[str] = None) -> bool:
        """Export metrics to file"""
        
        if not self.config.export_enabled:
            return False
        
        export_format = format or self.config.export_format
        export_path = Path(self.config.export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get all metrics within time range
            if not self.storage_backend:
                self.logger.warning("No storage backend for export")
                return False
            
            # This is a simplified export - in practice, you'd want to export
            # specific metrics or use database export features
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = export_path / f"metrics_export_{timestamp}.{export_format}"
            
            # For now, just create an empty file
            with open(filename, 'w') as f:
                f.write("Metrics export placeholder\n")
            
            self.logger.info(f"Metrics exported to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics"""
        
        stats = {
            'running': self.running,
            'collection_count': self.collection_count,
            'storage_count': self.storage_count,
            'error_count': self.error_count,
            'buffer_size': len(self.metrics_buffer),
            'last_collection_time': self.last_collection_time.isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'collectors': list(self.collectors.keys()),
            'scheduled_tasks': len(self.scheduled_tasks)
        }
        
        # Add storage backend statistics
        if self.storage_backend and hasattr(self.storage_backend, 'get_statistics'):
            storage_stats = self.storage_backend.get_statistics()
            stats['storage_backend'] = storage_stats
        
        return stats
    
    def check_thresholds(self) -> List[Dict[str, Any]]:
        """Check metric thresholds and return violations"""
        
        violations = []
        
        for metric_name, thresholds in self.config.metric_thresholds.items():
            try:
                # Get latest metric value
                metrics = self.get_metric(
                    metric_name,
                    start_time=datetime.now() - timedelta(minutes=5)
                )
                
                if not metrics:
                    continue
                
                latest_metric = max(metrics, key=lambda x: x.timestamp)
                
                if not isinstance(latest_metric.value, (int, float)):
                    continue
                
                value = float(latest_metric.value)
                
                # Check thresholds
                warning_threshold = thresholds.get('warning')
                critical_threshold = thresholds.get('critical')
                
                if critical_threshold is not None:
                    # Determine if we're checking for high or low values
                    # For most metrics (cpu, memory, disk), high is bad
                    # For accuracy, low is bad
                    if metric_name == 'model.accuracy':
                        if value < critical_threshold:
                            violations.append({
                                'metric': metric_name,
                                'value': value,
                                'threshold': critical_threshold,
                                'severity': 'critical',
                                'message': f"{metric_name} ({value:.2f}) below critical threshold ({critical_threshold:.2f})"
                            })
                            continue
                    else:
                        if value > critical_threshold:
                            violations.append({
                                'metric': metric_name,
                                'value': value,
                                'threshold': critical_threshold,
                                'severity': 'critical',
                                'message': f"{metric_name} ({value:.2f}) above critical threshold ({critical_threshold:.2f})"
                            })
                            continue
                
                if warning_threshold is not None:
                    if metric_name == 'model.accuracy':
                        if value < warning_threshold:
                            violations.append({
                                'metric': metric_name,
                                'value': value,
                                'threshold': warning_threshold,
                                'severity': 'warning',
                                'message': f"{metric_name} ({value:.2f}) below warning threshold ({warning_threshold:.2f})"
                            })
                    else:
                        if value > warning_threshold:
                            violations.append({
                                'metric': metric_name,
                                'value': value,
                                'threshold': warning_threshold,
                                'severity': 'warning',
                                'message': f"{metric_name} ({value:.2f}) above warning threshold ({warning_threshold:.2f})"
                            })
                            
            except Exception as e:
                self.logger.error(f"Failed to check threshold for {metric_name}: {str(e)}")
        
        return violations

# ============ Helper Functions ============
def create_metrics_collector(config: Optional[MetricsConfig] = None,
                           model_manager: Optional[ModelManager] = None,
                           performance_tracker: Optional[PerformanceTracker] = None) -> MetricsCollector:
    """Factory function to create metrics collector"""
    
    if config is None:
        config = MetricsConfig()
    
    return MetricsCollector(config, model_manager, performance_tracker)


def load_metrics_collector_from_config(config_path: str,
                                      model_manager: Optional[ModelManager] = None,
                                      performance_tracker: Optional[PerformanceTracker] = None) -> MetricsCollector:
    """Load metrics collector from configuration file"""
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert string enums back to Enum types
    if 'storage_backend' in config_dict:
        config_dict['storage_backend'] = StorageBackend(config_dict['storage_backend'])
    
    config = MetricsConfig(**config_dict)
    return MetricsCollector(config, model_manager, performance_tracker)


def metrics_to_dataframe(metrics: List[Metric]) -> pd.DataFrame:
    """Convert metrics to pandas DataFrame"""
    
    if not metrics:
        return pd.DataFrame()
    
    data = []
    for metric in metrics:
        metric_dict = metric.to_dict()
        data.append(metric_dict)
    
    df = pd.DataFrame(data)
    
    # Convert timestamp strings to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def aggregated_metrics_to_dataframe(aggregated_metrics: List[AggregatedMetric]) -> pd.DataFrame:
    """Convert aggregated metrics to pandas DataFrame"""
    
    if not aggregated_metrics:
        return pd.DataFrame()
    
    data = []
    for agg in aggregated_metrics:
        agg_dict = agg.to_dict()
        data.append(agg_dict)
    
    df = pd.DataFrame(data)
    
    # Convert timestamp strings to datetime
    timestamp_cols = ['window_start', 'window_end']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df


def visualize_metric_trend(metrics_df: pd.DataFrame,
                          metric_name: str,
                          title: str = "Metric Trend") -> Optional[Any]:
    """Visualize metric trend over time"""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if metrics_df.empty:
            return None
        
        # Filter for specific metric if needed
        if 'name' in metrics_df.columns:
            filtered_df = metrics_df[metrics_df['name'] == metric_name]
        else:
            filtered_df = metrics_df
        
        if filtered_df.empty:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot metric values
        if 'timestamp' in filtered_df.columns and 'value' in filtered_df.columns:
            ax.plot(filtered_df['timestamp'], filtered_df['value'], marker='o', linestyle='-', linewidth=2)
            
            # Add moving average
            if len(filtered_df) > 10:
                moving_avg = filtered_df['value'].rolling(window=10).mean()
                ax.plot(filtered_df['timestamp'], moving_avg, linestyle='--', linewidth=2, alpha=0.7, label='10-point MA')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.set_title(f"{title}: {metric_name}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
        
        return None
        
    except ImportError:
        logger.warning("Matplotlib not available for visualization")
        return None
    except Exception as e:
        logger.error(f"Failed to visualize metric trend: {str(e)}")
        return None


# ============ Example Usage ============
if __name__ == "__main__":
    # Example usage
    print("Metrics Collector Module")
    
    # Create a sample config
    config = MetricsConfig(
        enabled=True,
        collect_system_metrics=True,
        collect_model_metrics=True,
        storage_backend=StorageBackend.MEMORY
    )
    
    # Create metrics collector
    collector = MetricsCollector(config)
    
    print(f"Metrics Collector initialized")
    print(f"Storage backend: {config.storage_backend.value}")
    print(f"Collection interval: {config.collection_interval_seconds} seconds")
    
    # Start collection
    collector.start()
    
    # Collect a custom metric
    collector.collect_custom_metric(
        name="custom.test_metric",
        value=42.0,
        category=MetricCategory.CUSTOM,
        source="example",
        component="test"
    )
    
    print("Custom metric collected")
    
    # Get statistics
    stats = collector.get_statistics()
    print(f"Collection count: {stats['collection_count']}")
    print(f"Buffer size: {stats['buffer_size']}")
    
    # Stop collection
    collector.stop()
    
    print("Metrics collection stopped")