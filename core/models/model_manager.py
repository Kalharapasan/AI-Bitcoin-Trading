"""
Model Manager module for Bitcoin trading AI.
Centralized management for machine learning models including training, 
evaluation, versioning, deployment, and monitoring.
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
import yaml
from pathlib import Path
import hashlib
import shutil
import asyncio
from collections import deque, defaultdict
import uuid
import time
import traceback
import zipfile
import tempfile
import inspect
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import project modules
from config.settings import ModelSettings, AppConstants
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.utils.cache import Cache
from core.neural_networks.transformer_model import TransformerModel
from core.neural_networks.lstm_attention import LSTMAttentionModel
from core.neural_networks.cnn_lstm import CNNLSTMModel
from core.neural_networks.ensemble_model import EnsembleModel
from core.neural_networks.reinforcement_learning import RLModel

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Enums and Types ============
class ModelType(str, Enum):
    """Types of machine learning models"""
    TRANSFORMER = "transformer"
    LSTM_ATTENTION = "lstm_attention"
    CNN_LSTM = "cnn_lstm"
    ENSEMBLE = "ensemble"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    LINEAR_REGRESSION = "linear_regression"
    CUSTOM = "custom"

class ModelStatus(str, Enum):
    """Model lifecycle status"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"
    PENDING = "pending"

class ModelMetric(str, Enum):
    """Model evaluation metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MAE = "mae"                    # Mean Absolute Error
    MSE = "mse"                    # Mean Squared Error
    RMSE = "rmse"                  # Root Mean Squared Error
    MAPE = "mape"                  # Mean Absolute Percentage Error
    SHARPE_RATIO = "sharpe_ratio"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"

class DeploymentStage(str, Enum):
    """Model deployment stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    SHADOW = "shadow"

# ============ Data Structures ============
@dataclass
class ModelMetadata:
    """Metadata for machine learning models"""
    
    # Basic information
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    description: str = ""
    
    # Creation info
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Training info
    training_data_hash: Optional[str] = None
    training_features: List[str] = field(default_factory=list)
    target_variable: str = "price"
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Deployment info
    deployment_stage: DeploymentStage = DeploymentStage.DEVELOPMENT
    deployment_region: str = "local"
    deployment_resources: Dict[str, Any] = field(default_factory=dict)
    
    # Model artifacts
    model_path: Optional[str] = None
    preprocessing_path: Optional[str] = None
    feature_engineering_path: Optional[str] = None
    
    # Dependencies
    dependencies: Dict[str, str] = field(default_factory=dict)
    
    # Monitoring
    inference_count: int = 0
    last_inference_time: Optional[datetime] = None
    average_inference_time: float = 0.0
    
    # Status
    status: ModelStatus = ModelStatus.PENDING
    error_message: Optional[str] = None
    
    # Tags and labels
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metadata"""
        if not self.model_id:
            self.model_id = f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        if not self.version:
            self.version = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_type': self.model_type.value,
            'version': self.version,
            'description': self.description,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'training_data_hash': self.training_data_hash,
            'training_features': self.training_features,
            'target_variable': self.target_variable,
            'feature_importance': self.feature_importance,
            'hyperparameters': self.hyperparameters,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'test_metrics': self.test_metrics,
            'deployment_stage': self.deployment_stage.value,
            'deployment_region': self.deployment_region,
            'deployment_resources': self.deployment_resources,
            'model_path': self.model_path,
            'preprocessing_path': self.preprocessing_path,
            'feature_engineering_path': self.feature_engineering_path,
            'dependencies': self.dependencies,
            'inference_count': self.inference_count,
            'last_inference_time': self.last_inference_time.isoformat() if self.last_inference_time else None,
            'average_inference_time': self.average_inference_time,
            'status': self.status.value,
            'error_message': self.error_message,
            'tags': self.tags,
            'labels': self.labels
        }
    
    def update_inference_stats(self, inference_time: float):
        """Update inference statistics"""
        self.inference_count += 1
        self.last_inference_time = datetime.now()
        
        # Update average inference time
        self.average_inference_time = (
            (self.average_inference_time * (self.inference_count - 1) + inference_time) 
            / self.inference_count
        )
    
    def calculate_model_score(self) -> float:
        """Calculate overall model score (0-100)"""
        
        if not self.validation_metrics:
            return 0.0
        
        score_components = []
        weights = {
            'accuracy': 0.2,
            'f1_score': 0.3,
            'precision': 0.2,
            'recall': 0.2,
            'sharpe_ratio': 0.1
        }
        
        for metric, weight in weights.items():
            if metric in self.validation_metrics:
                value = self.validation_metrics[metric]
                if metric == 'sharpe_ratio':
                    # Normalize Sharpe ratio (typically -1 to 3)
                    normalized = max(0, min(100, (value + 1) * 25))
                elif metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    normalized = value * 100
                else:
                    normalized = value
                
                score_components.append(normalized * weight)
        
        if score_components:
            total_score = sum(score_components) / sum(weights.values())
        else:
            total_score = 0.0
        
        return total_score

@dataclass
class ModelVersion:
    """Version information for models"""
    
    version: str                    # Semantic version (e.g., 1.2.3)
    model_id: str                   # Parent model ID
    metadata: ModelMetadata         # Version-specific metadata
    
    # Version relationships
    parent_version: Optional[str] = None
    base_model_id: Optional[str] = None
    
    # Change information
    changes: List[str] = field(default_factory=list)
    breaking_changes: bool = False
    
    # Quality gates
    passed_tests: bool = False
    performance_threshold: float = 0.7
    
    # Release info
    released: bool = False
    release_notes: str = ""
    released_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize version"""
        if not self.version:
            self.version = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version': self.version,
            'model_id': self.model_id,
            'metadata': self.metadata.to_dict(),
            'parent_version': self.parent_version,
            'base_model_id': self.base_model_id,
            'changes': self.changes,
            'breaking_changes': self.breaking_changes,
            'passed_tests': self.passed_tests,
            'performance_threshold': self.performance_threshold,
            'released': self.released,
            'release_notes': self.release_notes,
            'released_at': self.released_at.isoformat() if self.released_at else None
        }

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # Data configuration
    train_test_split: float = 0.8
    validation_split: float = 0.1
    time_series_split: bool = True
    lookback_window: int = 50
    forecast_horizon: int = 5
    
    # Feature engineering
    feature_scaling: bool = True
    scale_method: str = "standard"  # standard, minmax, robust
    feature_selection: bool = True
    feature_selection_method: str = "mutual_info"  # mutual_info, f_regression, recursive
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_frequency: int = 5
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = False
    tuning_method: str = "grid_search"  # grid_search, random_search, bayesian
    tuning_iterations: int = 20
    
    # Model-specific parameters
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Cross-validation
    cross_validation_folds: int = 5
    cross_validation_method: str = "timeseries"  # timeseries, kfold, stratified
    
    # Performance metrics
    primary_metric: ModelMetric = ModelMetric.SHARPE_RATIO
    secondary_metrics: List[ModelMetric] = field(default_factory=lambda: [
        ModelMetric.ACCURACY, ModelMetric.PRECISION, ModelMetric.RECALL
    ])
    
    # Resource management
    use_gpu: bool = False
    gpu_memory_fraction: float = 0.8
    parallel_training: bool = True
    max_workers: int = 4
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    keep_best_checkpoints: int = 3
    
    # Logging
    log_training_progress: bool = True
    tensorboard_logging: bool = False
    wandb_integration: bool = False
    
    def __post_init__(self):
        """Validate configuration"""
        if self.train_test_split <= 0 or self.train_test_split >= 1:
            raise ValueError("train_test_split must be between 0 and 1")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'train_test_split': self.train_test_split,
            'validation_split': self.validation_split,
            'time_series_split': self.time_series_split,
            'lookback_window': self.lookback_window,
            'forecast_horizon': self.forecast_horizon,
            'feature_scaling': self.feature_scaling,
            'scale_method': self.scale_method,
            'feature_selection': self.feature_selection,
            'feature_selection_method': self.feature_selection_method,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'early_stopping_patience': self.early_stopping_patience,
            'validation_frequency': self.validation_frequency,
            'enable_hyperparameter_tuning': self.enable_hyperparameter_tuning,
            'tuning_method': self.tuning_method,
            'tuning_iterations': self.tuning_iterations,
            'model_parameters': self.model_parameters,
            'cross_validation_folds': self.cross_validation_folds,
            'cross_validation_method': self.cross_validation_method,
            'primary_metric': self.primary_metric.value,
            'secondary_metrics': [m.value for m in self.secondary_metrics],
            'use_gpu': self.use_gpu,
            'gpu_memory_fraction': self.gpu_memory_fraction,
            'parallel_training': self.parallel_training,
            'max_workers': self.max_workers,
            'save_checkpoints': self.save_checkpoints,
            'checkpoint_frequency': self.checkpoint_frequency,
            'keep_best_checkpoints': self.keep_best_checkpoints,
            'log_training_progress': self.log_training_progress,
            'tensorboard_logging': self.tensorboard_logging,
            'wandb_integration': self.wandb_integration
        }

@dataclass
class ModelEvaluation:
    """Comprehensive model evaluation results"""
    
    # Basic info
    evaluation_id: str
    model_id: str
    evaluation_timestamp: datetime
    
    # Dataset info
    dataset_size: int
    train_size: int
    validation_size: int
    test_size: int
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict[str, Any]] = None
    
    # Time series metrics
    forecast_errors: Optional[np.ndarray] = None
    residual_analysis: Optional[Dict[str, float]] = None
    
    # Trading metrics
    trading_metrics: Dict[str, float] = field(default_factory=dict)
    equity_curve: Optional[np.ndarray] = None
    drawdown_curve: Optional[np.ndarray] = None
    
    # Statistical tests
    statistical_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Feature analysis
    feature_importance: Dict[str, float] = field(default_factory=dict)
    shap_values: Optional[np.ndarray] = None
    
    # Error analysis
    error_distribution: Optional[Dict[str, float]] = None
    error_by_feature: Optional[Dict[str, float]] = None
    
    # Benchmark comparison
    benchmark_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    issues_found: List[str] = field(default_factory=list)
    
    # Metadata
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    compute_resources: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize evaluation"""
        if not self.evaluation_id:
            self.evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'evaluation_id': self.evaluation_id,
            'model_id': self.model_id,
            'evaluation_timestamp': self.evaluation_timestamp.isoformat(),
            'dataset_size': self.dataset_size,
            'train_size': self.train_size,
            'validation_size': self.validation_size,
            'test_size': self.test_size,
            'metrics': self.metrics,
            'trading_metrics': self.trading_metrics,
            'statistical_tests': self.statistical_tests,
            'feature_importance': self.feature_importance,
            'error_distribution': self.error_distribution,
            'error_by_feature': self.error_by_feature,
            'benchmark_comparison': self.benchmark_comparison,
            'recommendations': self.recommendations,
            'issues_found': self.issues_found,
            'evaluation_config': self.evaluation_config,
            'compute_resources': self.compute_resources
        }
        
        if self.confusion_matrix is not None:
            result['confusion_matrix'] = self.confusion_matrix.tolist()
        
        if self.forecast_errors is not None:
            result['forecast_errors'] = self.forecast_errors.tolist()
        
        if self.equity_curve is not None:
            result['equity_curve'] = self.equity_curve.tolist()
        
        if self.drawdown_curve is not None:
            result['drawdown_curve'] = self.drawdown_curve.tolist()
        
        if self.shap_values is not None:
            result['shap_values_shape'] = self.shap_values.shape
        
        return result
    
    def get_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall evaluation score"""
        
        if not weights:
            weights = {
                'accuracy': 0.15,
                'f1_score': 0.20,
                'sharpe_ratio': 0.25,
                'profit_factor': 0.20,
                'win_rate': 0.20
            }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in self.metrics:
                value = self.metrics[metric]
                
                # Normalize different metrics to 0-1 scale
                if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'win_rate']:
                    normalized = value
                elif metric == 'sharpe_ratio':
                    normalized = max(0, min(1, (value + 1) / 4))  # -1 to 3 -> 0 to 1
                elif metric == 'profit_factor':
                    normalized = min(1, value / 5)  # Cap at 5
                elif metric == 'mae':
                    normalized = max(0, 1 - value)  # Lower is better
                else:
                    normalized = value
                
                total_score += normalized * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_score = total_score / total_weight
        else:
            overall_score = 0.0
        
        return overall_score

@dataclass
class ModelDeployment:
    """Model deployment configuration"""
    
    deployment_id: str
    model_id: str
    version: str
    
    # Deployment configuration
    deployment_stage: DeploymentStage = DeploymentStage.DEVELOPMENT
    deployment_region: str = "local"
    deployment_resources: Dict[str, Any] = field(default_factory=dict)
    
    # Scaling configuration
    min_instances: int = 1
    max_instances: int = 5
    scaling_metric: str = "inference_latency"
    scaling_threshold: float = 100  # milliseconds
    
    # Monitoring
    health_check_endpoint: Optional[str] = None
    monitoring_interval: int = 60  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Traffic management
    canary_percentage: float = 0.0
    shadow_traffic: bool = False
    
    # Security
    authentication_required: bool = False
    rate_limiting: bool = True
    max_requests_per_minute: int = 1000
    
    # Version management
    auto_rollback: bool = True
    rollback_threshold: float = 0.7  # Performance threshold for rollback
    
    # Status
    status: str = "pending"
    deployed_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize deployment"""
        if not self.deployment_id:
            self.deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'deployment_id': self.deployment_id,
            'model_id': self.model_id,
            'version': self.version,
            'deployment_stage': self.deployment_stage.value,
            'deployment_region': self.deployment_region,
            'deployment_resources': self.deployment_resources,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'scaling_metric': self.scaling_metric,
            'scaling_threshold': self.scaling_threshold,
            'health_check_endpoint': self.health_check_endpoint,
            'monitoring_interval': self.monitoring_interval,
            'alert_thresholds': self.alert_thresholds,
            'canary_percentage': self.canary_percentage,
            'shadow_traffic': self.shadow_traffic,
            'authentication_required': self.authentication_required,
            'rate_limiting': self.rate_limiting,
            'max_requests_per_minute': self.max_requests_per_minute,
            'auto_rollback': self.auto_rollback,
            'rollback_threshold': self.rollback_threshold,
            'status': self.status,
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'metadata': self.metadata
        }

# ============ Model Factory ============
class ModelFactory:
    """Factory for creating different types of models"""
    
    @staticmethod
    def create_model(model_type: ModelType, **kwargs) -> Any:
        """Create a model instance based on type"""
        
        if model_type == ModelType.TRANSFORMER:
            return TransformerModel(**kwargs)
        elif model_type == ModelType.LSTM_ATTENTION:
            return LSTMAttentionModel(**kwargs)
        elif model_type == ModelType.CNN_LSTM:
            return CNNLSTMModel(**kwargs)
        elif model_type == ModelType.ENSEMBLE:
            return EnsembleModel(**kwargs)
        elif model_type == ModelType.REINFORCEMENT_LEARNING:
            return RLModel(**kwargs)
        elif model_type == ModelType.GRADIENT_BOOSTING:
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
            if kwargs.get('task') == 'classification':
                return GradientBoostingClassifier(**kwargs.get('params', {}))
            else:
                return GradientBoostingRegressor(**kwargs.get('params', {}))
        elif model_type == ModelType.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            if kwargs.get('task') == 'classification':
                return RandomForestClassifier(**kwargs.get('params', {}))
            else:
                return RandomForestRegressor(**kwargs.get('params', {}))
        elif model_type == ModelType.SVM:
            from sklearn.svm import SVC, SVR
            if kwargs.get('task') == 'classification':
                return SVC(**kwargs.get('params', {}))
            else:
                return SVR(**kwargs.get('params', {}))
        elif model_type == ModelType.LINEAR_REGRESSION:
            from sklearn.linear_model import LinearRegression, LogisticRegression
            if kwargs.get('task') == 'classification':
                return LogisticRegression(**kwargs.get('params', {}))
            else:
                return LinearRegression(**kwargs.get('params', {}))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

# ============ Model Registry ============
class ModelRegistry:
    """Central registry for managing model metadata and versions"""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelMetadata] = {}
        self.versions: Dict[str, List[ModelVersion]] = defaultdict(list)
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk"""
        registry_file = self.registry_path / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                # Load models
                for model_id, model_data in registry_data.get('models', {}).items():
                    # Convert string dates back to datetime
                    for date_field in ['created_at', 'updated_at', 'last_inference_time']:
                        if model_data.get(date_field):
                            model_data[date_field] = datetime.fromisoformat(model_data[date_field])
                    
                    # Create model metadata
                    model_data['model_type'] = ModelType(model_data['model_type'])
                    model_data['status'] = ModelStatus(model_data['status'])
                    model_data['deployment_stage'] = DeploymentStage(model_data['deployment_stage'])
                    
                    self.models[model_id] = ModelMetadata(**model_data)
                
                # Load versions
                for model_id, version_list in registry_data.get('versions', {}).items():
                    for version_data in version_list:
                        # Convert string dates
                        if version_data.get('released_at'):
                            version_data['released_at'] = datetime.fromisoformat(version_data['released_at'])
                        
                        # Convert nested metadata
                        metadata_data = version_data['metadata']
                        for date_field in ['created_at', 'updated_at', 'last_inference_time']:
                            if metadata_data.get(date_field):
                                metadata_data[date_field] = datetime.fromisoformat(metadata_data[date_field])
                        
                        metadata_data['model_type'] = ModelType(metadata_data['model_type'])
                        metadata_data['status'] = ModelStatus(metadata_data['status'])
                        metadata_data['deployment_stage'] = DeploymentStage(metadata_data['deployment_stage'])
                        
                        version_data['metadata'] = ModelMetadata(**metadata_data)
                        self.versions[model_id].append(ModelVersion(**version_data))
                
                logger.info(f"Loaded registry with {len(self.models)} models")
                
            except Exception as e:
                logger.error(f"Error loading registry: {str(e)}")
    
    def _save_registry(self):
        """Save registry to disk"""
        try:
            registry_data = {
                'models': {model_id: model.to_dict() for model_id, model in self.models.items()},
                'versions': {model_id: [version.to_dict() for version in versions] 
                           for model_id, versions in self.versions.items()}
            }
            
            registry_file = self.registry_path / "registry.json"
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
            
            logger.debug("Registry saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")
    
    def register_model(self, metadata: ModelMetadata) -> str:
        """Register a new model in the registry"""
        
        if metadata.model_id in self.models:
            raise ValueError(f"Model {metadata.model_id} already exists")
        
        # Update timestamps
        metadata.created_at = datetime.now()
        metadata.updated_at = datetime.now()
        
        # Add to registry
        self.models[metadata.model_id] = metadata
        
        # Create initial version
        version = ModelVersion(
            version=metadata.version,
            model_id=metadata.model_id,
            metadata=metadata
        )
        
        self.versions[metadata.model_id].append(version)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered model {metadata.model_id} ({metadata.model_name})")
        
        return metadata.model_id
    
    def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update model metadata"""
        
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found in registry")
            return False
        
        model = self.models[model_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(model, key):
                setattr(model, key, value)
            else:
                logger.warning(f"Invalid update field: {key}")
        
        model.updated_at = datetime.now()
        
        # Update corresponding version if it exists
        if model_id in self.versions:
            for version in self.versions[model_id]:
                if version.version == model.version:
                    version.metadata = model
                    break
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Updated model {model_id}")
        
        return True
    
    def add_version(self, version: ModelVersion) -> bool:
        """Add a new version to a model"""
        
        if version.model_id not in self.models:
            logger.warning(f"Model {version.model_id} not found in registry")
            return False
        
        # Check if version already exists
        existing_versions = self.versions.get(version.model_id, [])
        for existing_version in existing_versions:
            if existing_version.version == version.version:
                logger.warning(f"Version {version.version} already exists for model {version.model_id}")
                return False
        
        # Add version
        self.versions[version.model_id].append(version)
        
        # Update model metadata if this is the latest version
        if version.version > max([v.version for v in existing_versions], default="0.0.0"):
            self.models[version.model_id] = version.metadata
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Added version {version.version} to model {version.model_id}")
        
        return True
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        return self.models.get(model_id)
    
    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        return self.versions.get(model_id, [])
    
    def get_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the latest version of a model"""
        versions = self.versions.get(model_id, [])
        if not versions:
            return None
        
        # Sort by version number (simplified)
        return max(versions, key=lambda v: [int(x) for x in v.version.split('.')])
    
    def search_models(self, 
                     criteria: Dict[str, Any],
                     limit: int = 10) -> List[ModelMetadata]:
        """Search models based on criteria"""
        
        results = []
        
        for model in self.models.values():
            match = True
            
            for key, value in criteria.items():
                if key == 'model_type' and isinstance(value, str):
                    if model.model_type.value != value:
                        match = False
                        break
                elif key == 'status' and isinstance(value, str):
                    if model.status.value != value:
                        match = False
                        break
                elif key == 'tags' and isinstance(value, list):
                    if not all(tag in model.tags for tag in value):
                        match = False
                        break
                elif key == 'min_score':
                    if model.calculate_model_score() < value:
                        match = False
                        break
                elif hasattr(model, key):
                    if getattr(model, key) != value:
                        match = False
                        break
            
            if match:
                results.append(model)
            
            if len(results) >= limit:
                break
        
        return results
    
    def delete_model(self, model_id: str, archive: bool = True) -> bool:
        """Delete or archive a model"""
        
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found in registry")
            return False
        
        if archive:
            # Archive instead of delete
            model = self.models[model_id]
            model.status = ModelStatus.ARCHIVED
            model.updated_at = datetime.now()
            logger.info(f"Archived model {model_id}")
        else:
            # Actually delete
            del self.models[model_id]
            if model_id in self.versions:
                del self.versions[model_id]
            logger.info(f"Deleted model {model_id}")
        
        # Save registry
        self._save_registry()
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        
        total_models = len(self.models)
        total_versions = sum(len(versions) for versions in self.versions.values())
        
        # Count by type
        by_type = defaultdict(int)
        for model in self.models.values():
            by_type[model.model_type.value] += 1
        
        # Count by status
        by_status = defaultdict(int)
        for model in self.models.values():
            by_status[model.status.value] += 1
        
        # Count by deployment stage
        by_stage = defaultdict(int)
        for model in self.models.values():
            by_stage[model.deployment_stage.value] += 1
        
        return {
            'total_models': total_models,
            'total_versions': total_versions,
            'models_by_type': dict(by_type),
            'models_by_status': dict(by_status),
            'models_by_stage': dict(by_stage),
            'average_versions_per_model': total_versions / total_models if total_models > 0 else 0
        }

# ============ Model Storage ============
class ModelStorage:
    """Storage backend for model artifacts"""
    
    def __init__(self, storage_path: str = "models/storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, 
                  model: Any, 
                  model_id: str, 
                  version: str,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save model to storage"""
        
        # Create model directory
        model_dir = self.storage_path / model_id / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        
        try:
            # Try different serialization methods
            if hasattr(model, 'save'):
                # Model has save method (e.g., Keras, PyTorch)
                model.save(str(model_path))
            else:
                # Use joblib for scikit-learn models
                joblib.dump(model, model_path)
            
            # Save metadata
            if metadata:
                metadata_path = model_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved model {model_id} version {version} to {model_path}")
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_id: str, version: str) -> Any:
        """Load model from storage"""
        
        model_path = self.storage_path / model_id / version / "model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            # Try different loading methods
            if model_path.suffix == '.pkl':
                model = joblib.load(model_path)
            elif model_path.suffix == '.h5':
                # Keras model
                from tensorflow import keras
                model = keras.models.load_model(model_path)
            else:
                # Try pickle as fallback
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            logger.info(f"Loaded model {model_id} version {version} from {model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def save_preprocessor(self, 
                         preprocessor: Any,
                         model_id: str,
                         version: str) -> str:
        """Save preprocessing pipeline"""
        
        preprocessor_dir = self.storage_path / model_id / version
        preprocessor_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessor_path = preprocessor_dir / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        
        logger.info(f"Saved preprocessor for {model_id} version {version}")
        
        return str(preprocessor_path)
    
    def load_preprocessor(self, model_id: str, version: str) -> Any:
        """Load preprocessing pipeline"""
        
        preprocessor_path = self.storage_path / model_id / version / "preprocessor.pkl"
        
        if not preprocessor_path.exists():
            return None
        
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Loaded preprocessor for {model_id} version {version}")
        
        return preprocessor
    
    def save_evaluation(self,
                       evaluation: ModelEvaluation,
                       model_id: str,
                       version: str):
        """Save evaluation results"""
        
        eval_dir = self.storage_path / model_id / version / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        eval_path = eval_dir / f"{evaluation.evaluation_id}.json"
        
        with open(eval_path, 'w') as f:
            json.dump(evaluation.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved evaluation {evaluation.evaluation_id} for {model_id} version {version}")
    
    def load_evaluation(self, 
                       evaluation_id: str,
                       model_id: str,
                       version: str) -> Optional[ModelEvaluation]:
        """Load evaluation results"""
        
        eval_path = self.storage_path / model_id / version / "evaluations" / f"{evaluation_id}.json"
        
        if not eval_path.exists():
            return None
        
        try:
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            
            # Convert string dates back to datetime
            eval_data['evaluation_timestamp'] = datetime.fromisoformat(eval_data['evaluation_timestamp'])
            
            # Create evaluation object
            evaluation = ModelEvaluation(**eval_data)
            
            # Convert lists back to numpy arrays
            if 'confusion_matrix' in eval_data:
                evaluation.confusion_matrix = np.array(eval_data['confusion_matrix'])
            if 'forecast_errors' in eval_data:
                evaluation.forecast_errors = np.array(eval_data['forecast_errors'])
            if 'equity_curve' in eval_data:
                evaluation.equity_curve = np.array(eval_data['equity_curve'])
            if 'drawdown_curve' in eval_data:
                evaluation.drawdown_curve = np.array(eval_data['drawdown_curve'])
            
            logger.info(f"Loaded evaluation {evaluation_id} for {model_id} version {version}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error loading evaluation: {str(e)}")
            return None
    
    def delete_model(self, model_id: str, version: Optional[str] = None):
        """Delete model from storage"""
        
        if version:
            # Delete specific version
            model_dir = self.storage_path / model_id / version
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info(f"Deleted model {model_id} version {version}")
        else:
            # Delete all versions
            model_dir = self.storage_path / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info(f"Deleted all versions of model {model_id}")
    
    def list_models(self) -> List[str]:
        """List all models in storage"""
        
        models = []
        for item in self.storage_path.iterdir():
            if item.is_dir():
                models.append(item.name)
        
        return models
    
    def list_versions(self, model_id: str) -> List[str]:
        """List all versions of a model"""
        
        model_dir = self.storage_path / model_id
        if not model_dir.exists():
            return []
        
        versions = []
        for item in model_dir.iterdir():
            if item.is_dir():
                versions.append(item.name)
        
        return sorted(versions)
    
    def export_model(self, 
                    model_id: str, 
                    version: str,
                    export_path: str) -> str:
        """Export model to a portable format"""
        
        model_dir = self.storage_path / model_id / version
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_id} version {version} not found")
        
        # Create export directory
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create zip file
        zip_path = export_dir / f"{model_id}_v{version}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in model_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.storage_path)
                    zipf.write(file_path, arcname)
        
        logger.info(f"Exported model {model_id} version {version} to {zip_path}")
        
        return str(zip_path)
    
    def import_model(self, zip_path: str):
        """Import model from exported zip file"""
        
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Export file not found: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            # Extract all files
            zipf.extractall(self.storage_path)
        
        logger.info(f"Imported model from {zip_path}")

# ============ Model Trainer ============
class ModelTrainer:
    """Model training and hyperparameter tuning"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Training state
        self.training_history = []
        self.best_model = None
        self.best_score = -np.inf
        
        # Resource management
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_workers)
    
    def train_model(self,
                   model_type: ModelType,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None,
                   model_params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Train a model with given data"""
        
        start_time = time.time()
        
        # Prepare training data
        if X_val is None or y_val is None:
            # Split training data for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=self.config.validation_split,
                random_state=42
            )
        
        # Create model
        model_params = model_params or {}
        model_params.update(self.config.model_parameters)
        
        model = ModelFactory.create_model(
            model_type=model_type,
            params=model_params
        )
        
        # Determine task type
        task_type = self._determine_task_type(y_train)
        
        # Train model
        training_metrics = {}
        
        try:
            if model_type in [ModelType.TRANSFORMER, ModelType.LSTM_ATTENTION, 
                            ModelType.CNN_LSTM, ModelType.ENSEMBLE]:
                # Neural network training
                training_history = self._train_neural_network(
                    model, X_train, y_train, X_val, y_val, task_type
                )
                training_metrics['training_history'] = training_history
                
            elif model_type == ModelType.REINFORCEMENT_LEARNING:
                # RL training
                training_history = self._train_reinforcement_learning(
                    model, X_train, y_train
                )
                training_metrics['training_history'] = training_history
                
            else:
                # Traditional ML training
                model.fit(X_train, y_train)
                
                # Calculate training metrics
                y_train_pred = model.predict(X_train)
                training_metrics.update(
                    self._calculate_metrics(y_train, y_train_pred, task_type)
                )
        
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        
        # Calculate validation metrics
        y_val_pred = model.predict(X_val)
        validation_metrics = self._calculate_metrics(y_val, y_val_pred, task_type)
        
        # Calculate training time
        training_time = time.time() - start_time
        training_metrics['training_time'] = training_time
        
        # Update best model
        score = validation_metrics.get(self.config.primary_metric.value, 0.0)
        if score > self.best_score:
            self.best_score = score
            self.best_model = model
        
        # Log training results
        self.logger.info(
            f"Training completed in {training_time:.2f}s. "
            f"Validation {self.config.primary_metric.value}: {score:.4f}"
        )
        
        # Save training history
        self.training_history.append({
            'model_type': model_type.value,
            'training_metrics': training_metrics,
            'validation_metrics': validation_metrics,
            'training_time': training_time,
            'model_params': model_params
        })
        
        return model, {
            'training_metrics': training_metrics,
            'validation_metrics': validation_metrics,
            'training_time': training_time
        }
    
    def _train_neural_network(self,
                             model: Any,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray,
                             task_type: str) -> Dict[str, List[float]]:
        """Train neural network model"""
        
        # Prepare data for neural network
        if len(X_train.shape) == 2:
            # Add sequence dimension for time series models
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        
        # Train model
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            early_stopping_patience=self.config.early_stopping_patience
        )
        
        return history
    
    def _train_reinforcement_learning(self,
                                    model: Any,
                                    X_train: np.ndarray,
                                    y_train: np.ndarray) -> Dict[str, List[float]]:
        """Train reinforcement learning model"""
        
        # RL models need environment and episodes
        # This is a simplified implementation
        history = model.train(
            X_train, y_train,
            episodes=self.config.epochs,
            batch_size=self.config.batch_size
        )
        
        return history
    
    def _determine_task_type(self, y: np.ndarray) -> str:
        """Determine if task is classification or regression"""
        
        # Check if target is categorical
        unique_values = np.unique(y)
        if len(unique_values) <= 10:
            # Likely classification
            return "classification"
        else:
            # Likely regression
            return "regression"
    
    def _calculate_metrics(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          task_type: str) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_absolute_error, mean_squared_error, r2_score
        )
        
        metrics = {}
        
        if task_type == "classification":
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        else:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Calculate MAPE (handle zero values)
            mask = y_true != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
                metrics['mape'] = mape
        
        return metrics
    
    def hyperparameter_tuning(self,
                            model_type: ModelType,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        
        if not self.config.enable_hyperparameter_tuning:
            self.logger.warning("Hyperparameter tuning is disabled")
            return {}
        
        self.logger.info(f"Starting hyperparameter tuning for {model_type.value}")
        
        # Split data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=self.config.validation_split,
            random_state=42
        )
        
        best_params = {}
        best_score = -np.inf
        
        if self.config.tuning_method == "grid_search":
            # Generate all parameter combinations
            from itertools import product
            
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            combinations = list(product(*param_values))
            
            for combination in combinations:
                params = dict(zip(param_names, combination))
                
                try:
                    # Train model with these parameters
                    model, results = self.train_model(
                        model_type=model_type,
                        X_train=X_train_split,
                        y_train=y_train_split,
                        X_val=X_val,
                        y_val=y_val,
                        model_params=params
                    )
                    
                    score = results['validation_metrics'].get(
                        self.config.primary_metric.value, 0.0
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                        self.logger.debug(
                            f"New best params: {params}, "
                            f"score: {score:.4f}"
                        )
                
                except Exception as e:
                    self.logger.warning(f"Failed training with params {params}: {str(e)}")
                    continue
        
        elif self.config.tuning_method == "random_search":
            # Random search
            import random
            
            for _ in range(self.config.tuning_iterations):
                params = {}
                for param_name, param_values in param_grid.items():
                    params[param_name] = random.choice(param_values)
                
                try:
                    model, results = self.train_model(
                        model_type=model_type,
                        X_train=X_train_split,
                        y_train=y_train_split,
                        X_val=X_val,
                        y_val=y_val,
                        model_params=params
                    )
                    
                    score = results['validation_metrics'].get(
                        self.config.primary_metric.value, 0.0
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                        self.logger.debug(
                            f"New best params: {params}, "
                            f"score: {score:.4f}"
                        )
                
                except Exception as e:
                    self.logger.warning(f"Failed training with params {params}: {str(e)}")
                    continue
        
        else:
            raise ValueError(f"Unsupported tuning method: {self.config.tuning_method}")
        
        self.logger.info(
            f"Hyperparameter tuning completed. "
            f"Best score: {best_score:.4f}, "
            f"Best params: {best_params}"
        )
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'tuning_method': self.config.tuning_method
        }
    
    def cross_validate(self,
                      model_type: ModelType,
                      X: np.ndarray,
                      y: np.ndarray,
                      model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform cross-validation"""
        
        from sklearn.model_selection import (
            KFold, StratifiedKFold, TimeSeriesSplit
        )
        
        # Select cross-validation method
        if self.config.cross_validation_method == "timeseries":
            cv = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)
        elif self.config.cross_validation_method == "stratified":
            cv = StratifiedKFold(n_splits=self.config.cross_validation_folds)
        else:
            cv = KFold(n_splits=self.config.cross_validation_folds)
        
        # Perform cross-validation
        fold_scores = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            self.logger.info(f"Cross-validation fold {fold + 1}/{self.config.cross_validation_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            try:
                # Train model
                model, results = self.train_model(
                    model_type=model_type,
                    X_train=X_train_fold,
                    y_train=y_train_fold,
                    X_val=X_val_fold,
                    y_val=y_val_fold,
                    model_params=model_params
                )
                
                # Get score
                score = results['validation_metrics'].get(
                    self.config.primary_metric.value, 0.0
                )
                fold_scores.append(score)
                fold_metrics.append(results['validation_metrics'])
                
                self.logger.debug(f"Fold {fold + 1} score: {score:.4f}")
            
            except Exception as e:
                self.logger.error(f"Fold {fold + 1} failed: {str(e)}")
                continue
        
        # Calculate cross-validation statistics
        cv_results = {
            'fold_scores': fold_scores,
            'fold_metrics': fold_metrics,
            'mean_score': np.mean(fold_scores) if fold_scores else 0.0,
            'std_score': np.std(fold_scores) if fold_scores else 0.0,
            'min_score': np.min(fold_scores) if fold_scores else 0.0,
            'max_score': np.max(fold_scores) if fold_scores else 0.0
        }
        
        self.logger.info(
            f"Cross-validation completed. "
            f"Mean score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}"
        )
        
        return cv_results

# ============ Model Evaluator ============
class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def evaluate_model(self,
                      model: Any,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      X_train: Optional[np.ndarray] = None,
                      y_train: Optional[np.ndarray] = None,
                      evaluation_config: Optional[Dict[str, Any]] = None) -> ModelEvaluation:
        """Evaluate model performance"""
        
        start_time = time.time()
        
        # Default configuration
        config = evaluation_config or {}
        
        # Create evaluation object
        evaluation = ModelEvaluation(
            evaluation_id=f"eval_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}",
            model_id=config.get('model_id', 'unknown'),
            evaluation_timestamp=datetime.now(),
            dataset_size=len(X_test),
            train_size=len(X_train) if X_train is not None else 0,
            test_size=len(X_test),
            evaluation_config=config
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        evaluation.metrics = self._calculate_all_metrics(y_test, y_pred)
        
        # Calculate trading metrics if applicable
        if config.get('calculate_trading_metrics', False):
            evaluation.trading_metrics = self._calculate_trading_metrics(y_test, y_pred)
        
        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            evaluation.feature_importance = self._extract_feature_importance(model)
        
        # Statistical tests
        evaluation.statistical_tests = self._perform_statistical_tests(y_test, y_pred)
        
        # Error analysis
        evaluation.error_distribution = self._analyze_errors(y_test, y_pred)
        
        # Generate recommendations
        evaluation.recommendations = self._generate_recommendations(evaluation)
        
        # Calculate compute resources
        evaluation_time = time.time() - start_time
        evaluation.compute_resources = {
            'evaluation_time': evaluation_time,
            'samples_per_second': len(X_test) / evaluation_time if evaluation_time > 0 else 0
        }
        
        self.logger.info(
            f"Model evaluation completed in {evaluation_time:.2f}s. "
            f"Primary metric: {evaluation.metrics.get('accuracy', 0.0):.4f}"
        )
        
        return evaluation
    
    def _calculate_all_metrics(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_absolute_error, mean_squared_error, r2_score,
            mean_absolute_percentage_error, explained_variance_score
        )
        
        metrics = {}
        
        # Determine task type
        unique_values = np.unique(y_true)
        task_type = "classification" if len(unique_values) <= 10 else "regression"
        
        if task_type == "classification":
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            
            # For binary classification
            if len(unique_values) == 2:
                metrics['precision_binary'] = precision_score(y_true, y_pred, average='binary')
                metrics['recall_binary'] = recall_score(y_true, y_pred, average='binary')
                metrics['f1_binary'] = f1_score(y_true, y_pred, average='binary')
        else:
            # Regression metrics
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
            
            # MAPE (handle zero values)
            mask = y_true != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
                metrics['mape'] = mape
        
        # Additional metrics
        metrics['mean_prediction'] = np.mean(y_pred)
        metrics['std_prediction'] = np.std(y_pred)
        
        return metrics
    
    def _calculate_trading_metrics(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        
        # Simplified trading metrics
        # In production, you would use actual trading signals and returns
        
        # Generate trading signals (simple threshold)
        signals = np.where(y_pred > np.percentile(y_pred, 60), 1, 
                          np.where(y_pred < np.percentile(y_pred, 40), -1, 0))
        
        # Calculate returns (simulated)
        returns = np.diff(y_true) / y_true[:-1]
        signal_returns = returns * signals[:-1]
        
        # Calculate metrics
        metrics = {}
        
        if len(signal_returns) > 0:
            metrics['total_return'] = np.prod(1 + signal_returns) - 1
            metrics['sharpe_ratio'] = np.mean(signal_returns) / np.std(signal_returns) * np.sqrt(252) if np.std(signal_returns) > 0 else 0.0
            metrics['win_rate'] = np.mean(signal_returns > 0)
            metrics['profit_factor'] = np.sum(signal_returns[signal_returns > 0]) / abs(np.sum(signal_returns[signal_returns < 0])) if np.sum(signal_returns[signal_returns < 0]) < 0 else np.inf
        
        return metrics
    
    def _extract_feature_importance(self, model: Any) -> Dict[str, float]:
        """Extract feature importance from model"""
        
        feature_importance = {}
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            for i, importance in enumerate(importances):
                feature_importance[f"feature_{i}"] = importance
        
        elif hasattr(model, 'coef_'):
            # Linear models
            coefs = model.coef_
            if len(coefs.shape) == 1:
                for i, coef in enumerate(coefs):
                    feature_importance[f"feature_{i}"] = abs(coef)
            else:
                for i in range(coefs.shape[1]):
                    feature_importance[f"feature_{i}"] = np.mean(abs(coefs[:, i]))
        
        # Normalize to sum to 1
        if feature_importance:
            total = sum(feature_importance.values())
            if total > 0:
                feature_importance = {k: v/total for k, v in feature_importance.items()}
        
        return feature_importance
    
    def _perform_statistical_tests(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Perform statistical tests on predictions"""
        
        from scipy import stats
        
        tests = {}
        
        # Normality test on residuals
        residuals = y_true - y_pred
        if len(residuals) > 3:
            shapiro_test = stats.shapiro(residuals)
            tests['normality'] = {
                'statistic': shapiro_test.statistic,
                'p_value': shapiro_test.pvalue,
                'is_normal': shapiro_test.pvalue > 0.05
            }
        
        # Stationarity test (Augmented Dickey-Fuller)
        if len(residuals) > 10:
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_test = adfuller(residuals)
                tests['stationarity'] = {
                    'adf_statistic': adf_test[0],
                    'p_value': adf_test[1],
                    'is_stationary': adf_test[1] < 0.05
                }
            except ImportError:
                pass
        
        # Correlation test
        correlation, p_value = stats.pearsonr(y_true, y_pred)
        tests['correlation'] = {
            'correlation': correlation,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }
        
        return tests
    
    def _analyze_errors(self, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze error distribution"""
        
        errors = y_true - y_pred
        
        error_analysis = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'skewness': stats.skew(errors) if len(errors) > 2 else 0.0,
            'kurtosis': stats.kurtosis(errors) if len(errors) > 3 else 0.0,
            'mae': np.mean(np.abs(errors)),
            'mse': np.mean(errors ** 2)
        }
        
        # Error quantiles
        quantiles = np.percentile(errors, [10, 25, 50, 75, 90])
        for i, q in enumerate([10, 25, 50, 75, 90]):
            error_analysis[f'percentile_{q}'] = quantiles[i]
        
        return error_analysis
    
    def _generate_recommendations(self, evaluation: ModelEvaluation) -> List[str]:
        """Generate recommendations based on evaluation results"""
        
        recommendations = []
        
        # Check metrics
        if 'accuracy' in evaluation.metrics:
            accuracy = evaluation.metrics['accuracy']
            if accuracy < 0.7:
                recommendations.append("Model accuracy is below 70%. Consider improving feature engineering or trying different algorithms.")
            elif accuracy > 0.9:
                recommendations.append("Excellent accuracy achieved. Consider monitoring for overfitting.")
        
        if 'r2' in evaluation.metrics:
            r2 = evaluation.metrics['r2']
            if r2 < 0.5:
                recommendations.append("R² score is low. The model explains less than 50% of variance. Consider feature selection or non-linear models.")
        
        # Check trading metrics
        if 'sharpe_ratio' in evaluation.trading_metrics:
            sharpe = evaluation.trading_metrics['sharpe_ratio']
            if sharpe < 1.0:
                recommendations.append("Sharpe ratio is below 1.0. Trading performance may not justify risk.")
        
        # Check statistical tests
        if 'normality' in evaluation.statistical_tests:
            if not evaluation.statistical_tests['normality']['is_normal']:
                recommendations.append("Residuals are not normally distributed. Consider transforming target variable.")
        
        if 'stationarity' in evaluation.statistical_tests:
            if not evaluation.statistical_tests['stationarity']['is_stationary']:
                recommendations.append("Residuals are not stationary. Consider differencing or detrending.")
        
        return recommendations

# ============ Main Model Manager ============
class ModelManager:
    """Main model management system"""
    
    def __init__(self, 
                 registry_path: str = "models/registry",
                 storage_path: str = "models/storage"):
        
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.registry = ModelRegistry(registry_path)
        self.storage = ModelStorage(storage_path)
        self.trainer = None
        self.evaluator = ModelEvaluator()
        
        # Model cache for quick access
        self.model_cache = Cache(ttl=300)  # 5 minutes TTL
        
        # Deployment manager
        self.deployments: Dict[str, ModelDeployment] = {}
        
        # Monitoring
        self.monitoring_task = None
        self.is_monitoring = False
        
        self.logger.info("Model Manager initialized")
    
    def create_model(self,
                    model_name: str,
                    model_type: ModelType,
                    description: str = "",
                    tags: Optional[List[str]] = None,
                    labels: Optional[Dict[str, str]] = None) -> ModelMetadata:
        """Create a new model"""
        
        # Generate model ID
        model_id = f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version="1.0.0",
            description=description,
            tags=tags or [],
            labels=labels or {},
            status=ModelStatus.PENDING
        )
        
        # Register model
        self.registry.register_model(metadata)
        
        self.logger.info(f"Created model {model_id} ({model_name})")
        
        return metadata
    
    def train_new_model(self,
                       model_name: str,
                       model_type: ModelType,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       training_config: TrainingConfig,
                       description: str = "",
                       tags: Optional[List[str]] = None) -> Tuple[ModelMetadata, Any, Dict[str, Any]]:
        """Train a new model from scratch"""
        
        start_time = time.time()
        
        # Create model
        metadata = self.create_model(
            model_name=model_name,
            model_type=model_type,
            description=description,
            tags=tags
        )
        
        # Update status
        metadata.status = ModelStatus.TRAINING
        self.registry.update_model(metadata.model_id, {'status': ModelStatus.TRAINING})
        
        # Initialize trainer
        self.trainer = ModelTrainer(training_config)
        
        try:
            # Train model
            model, training_results = self.trainer.train_model(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                model_params=training_config.model_parameters
            )
            
            # Update metadata
            metadata.status = ModelStatus.TRAINED
            metadata.training_metrics = training_results['training_metrics']
            metadata.validation_metrics = training_results['validation_metrics']
            metadata.hyperparameters = training_config.model_parameters
            
            # Calculate hash of training data
            data_hash = hashlib.md5(
                np.concatenate([X_train.flatten(), y_train.flatten()])
            ).hexdigest()
            metadata.training_data_hash = data_hash
            
            # Save model
            model_path = self.storage.save_model(
                model=model,
                model_id=metadata.model_id,
                version=metadata.version,
                metadata=metadata.to_dict()
            )
            metadata.model_path = model_path
            
            # Update registry
            self.registry.update_model(metadata.model_id, {
                'status': ModelStatus.TRAINED,
                'training_metrics': metadata.training_metrics,
                'validation_metrics': metadata.validation_metrics,
                'hyperparameters': metadata.hyperparameters,
                'training_data_hash': metadata.training_data_hash,
                'model_path': metadata.model_path,
                'updated_at': datetime.now()
            })
            
            training_time = time.time() - start_time
            self.logger.info(
                f"Model training completed in {training_time:.2f}s. "
                f"Model ID: {metadata.model_id}"
            )
            
            return metadata, model, training_results
            
        except Exception as e:
            # Update status to failed
            metadata.status = ModelStatus.FAILED
            metadata.error_message = str(e)
            self.registry.update_model(metadata.model_id, {
                'status': ModelStatus.FAILED,
                'error_message': str(e)
            })
            
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def evaluate_model(self,
                      model_id: str,
                      version: str,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      evaluation_config: Optional[Dict[str, Any]] = None) -> ModelEvaluation:
        """Evaluate a trained model"""
        
        # Load model
        model = self.storage.load_model(model_id, version)
        
        if model is None:
            raise ValueError(f"Model {model_id} version {version} not found")
        
        # Load metadata
        metadata = self.registry.get_model(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Evaluate model
        evaluation = self.evaluator.evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            evaluation_config={
                'model_id': model_id,
                'model_version': version,
                'model_type': metadata.model_type.value,
                'calculate_trading_metrics': True
            }
        )
        
        # Update model metadata with test metrics
        metadata.test_metrics = evaluation.metrics
        metadata.status = ModelStatus.VALIDATED
        self.registry.update_model(model_id, {
            'test_metrics': evaluation.metrics,
            'status': ModelStatus.VALIDATED,
            'updated_at': datetime.now()
        })
        
        # Save evaluation results
        self.storage.save_evaluation(evaluation, model_id, version)
        
        self.logger.info(
            f"Model evaluation completed. "
            f"Overall score: {evaluation.get_overall_score():.4f}"
        )
        
        return evaluation
    
    def deploy_model(self,
                    model_id: str,
                    version: str,
                    deployment_stage: DeploymentStage = DeploymentStage.DEVELOPMENT,
                    deployment_config: Optional[Dict[str, Any]] = None) -> ModelDeployment:
        """Deploy a model to a specific stage"""
        
        # Load model metadata
        metadata = self.registry.get_model(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found")
        
        # Check if model is trained
        if metadata.status != ModelStatus.VALIDATED:
            raise ValueError(f"Model {model_id} is not validated. Current status: {metadata.status}")
        
        # Create deployment
        deployment = ModelDeployment(
            deployment_id=f"deploy_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}",
            model_id=model_id,
            version=version,
            deployment_stage=deployment_stage,
            deployment_resources=deployment_config or {},
            deployed_at=datetime.now(),
            status="active"
        )
        
        # Store deployment
        self.deployments[deployment.deployment_id] = deployment
        
        # Update model metadata
        metadata.deployment_stage = deployment_stage
        metadata.status = ModelStatus.DEPLOYED
        self.registry.update_model(model_id, {
            'deployment_stage': deployment_stage.value,
            'status': ModelStatus.DEPLOYED,
            'updated_at': datetime.now()
        })
        
        # Cache model for faster inference
        model = self.storage.load_model(model_id, version)
        cache_key = f"{model_id}_{version}"
        self.model_cache.set(cache_key, model)
        
        self.logger.info(
            f"Model {model_id} version {version} deployed to {deployment_stage.value}. "
            f"Deployment ID: {deployment.deployment_id}"
        )
        
        return deployment
    
    def predict(self,
               model_id: str,
               version: str,
               X: np.ndarray,
               preprocess: bool = True) -> np.ndarray:
        """Make predictions using a deployed model"""
        
        start_time = time.time()
        
        # Try to get model from cache
        cache_key = f"{model_id}_{version}"
        model = self.model_cache.get(cache_key)
        
        if model is None:
            # Load model from storage
            model = self.storage.load_model(model_id, version)
            
            if model is None:
                raise ValueError(f"Model {model_id} version {version} not found")
            
            # Cache model
            self.model_cache.set(cache_key, model)
        
        # Load preprocessor if needed
        if preprocess:
            preprocessor = self.storage.load_preprocessor(model_id, version)
            if preprocessor is not None:
                X = preprocessor.transform(X)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Update inference statistics
        inference_time = time.time() - start_time
        self._update_inference_stats(model_id, version, inference_time)
        
        self.logger.debug(
            f"Inference completed in {inference_time:.4f}s. "
            f"Samples: {len(X)}, "
            f"Throughput: {len(X)/inference_time:.2f} samples/s"
        )
        
        return predictions
    
    def _update_inference_stats(self, model_id: str, version: str, inference_time: float):
        """Update inference statistics for a model"""
        
        metadata = self.registry.get_model(model_id)
        if metadata is None:
            return
        
        metadata.update_inference_stats(inference_time)
        self.registry.update_model(model_id, {
            'inference_count': metadata.inference_count,
            'last_inference_time': metadata.last_inference_time,
            'average_inference_time': metadata.average_inference_time,
            'updated_at': datetime.now()
        })
    
    def retrain_model(self,
                     model_id: str,
                     new_data: Tuple[np.ndarray, np.ndarray],
                     training_config: TrainingConfig,
                     version_increment: str = "minor") -> Tuple[ModelMetadata, Any]:
        """Retrain a model with new data"""
        
        # Load existing model metadata
        metadata = self.registry.get_model(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found")
        
        # Load existing model
        old_model = self.storage.load_model(model_id, metadata.version)
        
        # Prepare training data
        X_new, y_new = new_data
        
        # Determine new version
        new_version = self._increment_version(metadata.version, version_increment)
        
        # Create new metadata
        new_metadata = ModelMetadata(
            model_id=model_id,
            model_name=metadata.model_name,
            model_type=metadata.model_type,
            version=new_version,
            description=f"Retrained from version {metadata.version}",
            tags=metadata.tags.copy(),
            labels=metadata.labels.copy(),
            parent_version=metadata.version
        )
        
        # Update status
        new_metadata.status = ModelStatus.TRAINING
        
        # Initialize trainer
        self.trainer = ModelTrainer(training_config)
        
        try:
            # Train new model (could use transfer learning from old model)
            model, training_results = self.trainer.train_model(
                model_type=metadata.model_type,
                X_train=X_new,
                y_train=y_new,
                model_params=training_config.model_parameters
            )
            
            # Update metadata
            new_metadata.status = ModelStatus.TRAINED
            new_metadata.training_metrics = training_results['training_metrics']
            new_metadata.validation_metrics = training_results['validation_metrics']
            new_metadata.hyperparameters = training_config.model_parameters
            
            # Calculate hash of new training data
            data_hash = hashlib.md5(
                np.concatenate([X_new.flatten(), y_new.flatten()])
            ).hexdigest()
            new_metadata.training_data_hash = data_hash
            
            # Save new model
            model_path = self.storage.save_model(
                model=model,
                model_id=model_id,
                version=new_version,
                metadata=new_metadata.to_dict()
            )
            new_metadata.model_path = model_path
            
            # Create new version
            version = ModelVersion(
                version=new_version,
                model_id=model_id,
                metadata=new_metadata,
                parent_version=metadata.version,
                changes=["Retrained with new data"]
            )
            
            # Add to registry
            self.registry.add_version(version)
            
            self.logger.info(
                f"Model retraining completed. "
                f"New version: {new_version}, "
                f"Old version: {metadata.version}"
            )
            
            return new_metadata, model
            
        except Exception as e:
            new_metadata.status = ModelStatus.FAILED
            new_metadata.error_message = str(e)
            self.logger.error(f"Model retraining failed: {str(e)}")
            raise
    
    def _increment_version(self, current_version: str, increment_type: str) -> str:
        """Increment version number"""
        
        parts = current_version.split('.')
        if len(parts) != 3:
            return f"{current_version}.1"
        
        major, minor, patch = map(int, parts)
        
        if increment_type == "major":
            return f"{major + 1}.0.0"
        elif increment_type == "minor":
            return f"{major}.{minor + 1}.0"
        else:  # patch
            return f"{major}.{minor}.{patch + 1}"
    
    def compare_models(self,
                      model_ids: List[str],
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict[str, Any]:
        """Compare multiple models"""
        
        comparison_results = {}
        
        for model_id in model_ids:
            # Get latest version
            latest_version = self.registry.get_latest_version(model_id)
            if latest_version is None:
                continue
            
            # Load model
            model = self.storage.load_model(model_id, latest_version.version)
            if model is None:
                continue
            
            # Evaluate model
            evaluation = self.evaluator.evaluate_model(
                model=model,
                X_test=X_test,
                y_test=y_test
            )
            
            # Store results
            comparison_results[model_id] = {
                'version': latest_version.version,
                'model_type': latest_version.metadata.model_type.value,
                'metrics': evaluation.metrics,
                'overall_score': evaluation.get_overall_score(),
                'trading_metrics': evaluation.trading_metrics
            }
        
        # Rank models by overall score
        ranked_models = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        results = {
            'comparison_results': comparison_results,
            'ranked_models': ranked_models,
            'best_model': ranked_models[0] if ranked_models else None
        }
        
        self.logger.info(
            f"Model comparison completed. "
            f"Best model: {results['best_model'][0] if results['best_model'] else 'None'}"
        )
        
        return results
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive status of a model"""
        
        metadata = self.registry.get_model(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found")
        
        # Get all versions
        versions = self.registry.get_model_versions(model_id)
        
        # Get deployment status
        deployment_status = None
        for deployment in self.deployments.values():
            if deployment.model_id == model_id and deployment.status == "active":
                deployment_status = deployment
                break
        
        # Calculate model score
        model_score = metadata.calculate_model_score()
        
        status = {
            'model_id': model_id,
            'metadata': metadata.to_dict(),
            'versions': [v.to_dict() for v in versions],
            'deployment_status': deployment_status.to_dict() if deployment_status else None,
            'model_score': model_score,
            'inference_stats': {
                'inference_count': metadata.inference_count,
                'average_inference_time': metadata.average_inference_time,
                'last_inference_time': metadata.last_inference_time.isoformat() if metadata.last_inference_time else None
            },
            'storage_info': {
                'versions_stored': self.storage.list_versions(model_id),
                'total_size': self._calculate_model_size(model_id)
            }
        }
        
        return status
    
    def _calculate_model_size(self, model_id: str) -> float:
        """Calculate total storage size for a model (in MB)"""
        
        model_dir = Path(self.storage.storage_path) / model_id
        if not model_dir.exists():
            return 0.0
        
        total_size = 0.0
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def cleanup_models(self, 
                      max_versions_per_model: int = 5,
                      max_age_days: int = 30):
        """Clean up old model versions"""
        
        cleaned_count = 0
        
        for model_id in self.storage.list_models():
            versions = self.storage.list_versions(model_id)
            
            if len(versions) <= max_versions_per_model:
                continue
            
            # Sort versions by timestamp (assuming version naming includes timestamp)
            sorted_versions = sorted(versions, reverse=True)
            
            # Keep only the newest versions
            versions_to_keep = sorted_versions[:max_versions_per_model]
            versions_to_delete = sorted_versions[max_versions_per_model:]
            
            # Delete old versions
            for version in versions_to_delete:
                self.storage.delete_model(model_id, version)
                cleaned_count += 1
        
        self.logger.info(f"Cleaned up {cleaned_count} old model versions")
    
    def export_model_registry(self, export_path: str):
        """Export entire model registry"""
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export registry data
        registry_data = self.registry.get_statistics()
        registry_file = export_dir / "registry_summary.json"
        
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2, default=str)
        
        # Export all models
        models_file = export_dir / "models_list.csv"
        models_data = []
        
        for model_id, metadata in self.registry.models.items():
            models_data.append({
                'model_id': model_id,
                'model_name': metadata.model_name,
                'model_type': metadata.model_type.value,
                'version': metadata.version,
                'status': metadata.status.value,
                'deployment_stage': metadata.deployment_stage.value,
                'created_at': metadata.created_at.isoformat(),
                'model_score': metadata.calculate_model_score()
            })
        
        df = pd.DataFrame(models_data)
        df.to_csv(models_file, index=False)
        
        self.logger.info(f"Exported model registry to {export_path}")
    
    def start_monitoring(self, interval_minutes: int = 5):
        """Start model monitoring"""
        
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        
        async def monitor_task():
            while self.is_monitoring:
                try:
                    # Monitor model performance
                    self._monitor_model_performance()
                    
                    # Monitor deployment health
                    self._monitor_deployment_health()
                    
                    # Clean up cache
                    self.model_cache.cleanup()
                    
                    # Log status
                    self.logger.debug("Model monitoring completed")
                    
                except Exception as e:
                    self.logger.error(f"Error in model monitoring: {str(e)}")
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
        
        self.monitoring_task = asyncio.create_task(monitor_task())
        self.logger.info(f"Started model monitoring with {interval_minutes} minute interval")
    
    def stop_monitoring(self):
        """Stop model monitoring"""
        
        if not self.is_monitoring:
            self.logger.warning("Monitoring not started")
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
        
        self.logger.info("Stopped model monitoring")
    
    def _monitor_model_performance(self):
        """Monitor model performance degradation"""
        # Implementation would check for performance degradation
        # and trigger retraining if needed
        pass
    
    def _monitor_deployment_health(self):
        """Monitor deployment health"""
        # Implementation would check deployment health endpoints
        # and trigger alerts if needed
        pass

# ============ Factory Function ============
def create_model_manager(registry_path: str = "models/registry",
                        storage_path: str = "models/storage") -> ModelManager:
    """Factory function to create model manager"""
    return ModelManager(registry_path, storage_path)

# ============ Main Execution ============
async def main():
    """Main execution for testing"""
    
    # Create model manager
    manager = create_model_manager()
    
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        print("=== Model Manager Test ===")
        
        # Create training config
        training_config = TrainingConfig(
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            train_test_split=0.8,
            validation_split=0.1,
            enable_hyperparameter_tuning=False
        )
        
        # Train a new model
        print("\n1. Training new model...")
        metadata, model, training_results = manager.train_new_model(
            model_name="Test Regression Model",
            model_type=ModelType.RANDOM_FOREST,
            X_train=X_train,
            y_train=y_train,
            training_config=training_config,
            description="Test model for regression task",
            tags=["test", "regression", "random_forest"]
        )
        
        print(f"   Model ID: {metadata.model_id}")
        print(f"   Model Name: {metadata.model_name}")
        print(f"   Model Type: {metadata.model_type.value}")
        print(f"   Version: {metadata.version}")
        print(f"   Status: {metadata.status.value}")
        print(f"   Training Metrics: {metadata.training_metrics}")
        
        # Evaluate model
        print("\n2. Evaluating model...")
        evaluation = manager.evaluate_model(
            model_id=metadata.model_id,
            version=metadata.version,
            X_test=X_test,
            y_test=y_test
        )
        
        print(f"   Evaluation ID: {evaluation.evaluation_id}")
        print(f"   Test Metrics: {evaluation.metrics}")
        print(f"   Overall Score: {evaluation.get_overall_score():.4f}")
        
        # Deploy model
        print("\n3. Deploying model...")
        deployment = manager.deploy_model(
            model_id=metadata.model_id,
            version=metadata.version,
            deployment_stage=DeploymentStage.DEVELOPMENT
        )
        
        print(f"   Deployment ID: {deployment.deployment_id}")
        print(f"   Deployment Stage: {deployment.deployment_stage.value}")
        print(f"   Status: {deployment.status}")
        
        # Make predictions
        print("\n4. Making predictions...")
        test_samples = X_test[:5]
        predictions = manager.predict(
            model_id=metadata.model_id,
            version=metadata.version,
            X=test_samples
        )
        
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[:3]}")
        
        # Get model status
        print("\n5. Getting model status...")
        status = manager.get_model_status(metadata.model_id)
        
        print(f"   Model Score: {status['model_score']:.2f}/100")
        print(f"   Inference Count: {status['inference_stats']['inference_count']}")
        print(f"   Average Inference Time: {status['inference_stats']['average_inference_time']:.4f}s")
        
        # Get registry statistics
        print("\n6. Registry statistics...")
        stats = manager.registry.get_statistics()
        
        print(f"   Total Models: {stats['total_models']}")
        print(f"   Total Versions: {stats['total_versions']}")
        print(f"   Models by Type: {stats['models_by_type']}")
        print(f"   Models by Status: {stats['models_by_status']}")
        
        # Export registry
        print("\n7. Exporting registry...")
        manager.export_model_registry("exported_registry")
        print("   Registry exported to 'exported_registry/'")
        
        # Start monitoring
        print("\n8. Starting monitoring...")
        manager.start_monitoring(interval_minutes=1)
        print("   Monitoring started (will run for 2 minutes)")
        
        # Wait for monitoring to run
        await asyncio.sleep(120)
        
        # Stop monitoring
        print("\n9. Stopping monitoring...")
        manager.stop_monitoring()
        print("   Monitoring stopped")
        
        print("\n=== Test Completed Successfully ===")
        
    except Exception as e:
        print(f"Error in model manager test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())