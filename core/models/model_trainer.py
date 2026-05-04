"""
Model Trainer module for Bitcoin trading AI.
Handles training, validation, hyperparameter tuning, and model optimization
for various machine learning models used in trading.
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
import shutil
import asyncio
from collections import deque, defaultdict
import uuid
import time
import traceback
import tempfile
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Import project modules
from config.settings import ModelSettings, TrainingSettings
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.utils.cache import Cache
from core.models.model_manager import ModelManager, ModelMetadata, ModelType, ModelStatus
from core.data_processing.data_preprocessor import DataPreprocessor
from core.data_processing.feature_engineer import FeatureEngineer

# Import ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Enums and Types ============
class TrainingPhase(str, Enum):
    """Phases of model training"""
    DATA_PREPARATION = "data_preparation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    VALIDATION = "validation"
    EVALUATION = "evaluation"
    SAVING = "saving"

class TrainingStatus(str, Enum):
    """Training process status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OPTIMIZING = "optimizing"

class ModelTask(str, Enum):
    """Types of modeling tasks"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    MULTI_CLASSIFICATION = "multi_classification"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ANOMALY_DETECTION = "anomaly_detection"

# ============ Data Structures ============
@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # General settings
    model_type: ModelType
    model_task: ModelTask = ModelTask.REGRESSION
    random_seed: int = 42
    
    # Data settings
    train_test_split: float = 0.8
    validation_split: float = 0.1
    time_series_split: bool = True
    lookback_window: int = 50
    forecast_horizon: int = 1
    sequence_length: int = 60
    
    # Feature engineering
    feature_scaling: bool = True
    scale_method: str = "standard"  # standard, minmax, robust, quantile
    feature_selection: bool = True
    feature_selection_method: str = "mutual_info"  # mutual_info, f_regression, recursive, lasso
    feature_selection_top_k: int = 50
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    validation_frequency: int = 1
    
    # Optimization
    optimizer: str = "adam"  # adam, sgd, rmsprop, adagrad
    loss_function: str = "mse"  # mse, mae, huber, binary_crossentropy, categorical_crossentropy
    metrics: List[str] = field(default_factory=lambda: ["mae", "mse", "rmse"])
    
    # Regularization
    dropout_rate: float = 0.2
    l1_regularization: float = 0.0
    l2_regularization: float = 0.0
    batch_normalization: bool = True
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = True
    tuning_method: str = "bayesian"  # grid, random, bayesian, genetic
    tuning_iterations: int = 50
    cv_folds: int = 5
    cv_method: str = "timeseries"  # timeseries, kfold, stratified
    
    # Neural network specific
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    activation_function: str = "relu"  # relu, tanh, sigmoid, leaky_relu, elu
    output_activation: str = "linear"  # linear, sigmoid, softmax
    use_batch_norm: bool = True
    use_residual_connections: bool = False
    
    # Tree-based models
    n_estimators: int = 100
    max_depth: int = 6
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    learning_rate_gbm: float = 0.1
    subsample: float = 0.8
    
    # Model ensemble
    enable_ensemble: bool = False
    ensemble_method: str = "stacking"  # stacking, blending, voting, averaging
    base_models: List[ModelType] = field(default_factory=list)
    
    # Checkpointing and saving
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5
    checkpoint_path: str = "models/checkpoints/"
    keep_best_checkpoints: int = 3
    save_final_model: bool = True
    
    # Monitoring and logging
    log_training_progress: bool = True
    tensorboard_logging: bool = False
    wandb_integration: bool = False
    wandb_project: str = "bitcoin-trading-ai"
    log_interval: int = 10
    
    # Resource management
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    parallel_training: bool = True
    max_workers: int = 4
    mixed_precision: bool = True
    
    # Advanced features
    enable_transfer_learning: bool = False
    transfer_model_path: Optional[str] = None
    enable_data_augmentation: bool = False
    augmentation_factor: float = 1.5
    
    def __post_init__(self):
        """Validate configuration"""
        if self.train_test_split <= 0 or self.train_test_split >= 1:
            raise ValueError("train_test_split must be between 0 and 1")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        
        # Create checkpoint directory
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_type': self.model_type.value,
            'model_task': self.model_task.value,
            'random_seed': self.random_seed,
            'train_test_split': self.train_test_split,
            'validation_split': self.validation_split,
            'time_series_split': self.time_series_split,
            'lookback_window': self.lookback_window,
            'forecast_horizon': self.forecast_horizon,
            'sequence_length': self.sequence_length,
            'feature_scaling': self.feature_scaling,
            'scale_method': self.scale_method,
            'feature_selection': self.feature_selection,
            'feature_selection_method': self.feature_selection_method,
            'feature_selection_top_k': self.feature_selection_top_k,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_delta': self.early_stopping_delta,
            'validation_frequency': self.validation_frequency,
            'optimizer': self.optimizer,
            'loss_function': self.loss_function,
            'metrics': self.metrics,
            'dropout_rate': self.dropout_rate,
            'l1_regularization': self.l1_regularization,
            'l2_regularization': self.l2_regularization,
            'batch_normalization': self.batch_normalization,
            'enable_hyperparameter_tuning': self.enable_hyperparameter_tuning,
            'tuning_method': self.tuning_method,
            'tuning_iterations': self.tuning_iterations,
            'cv_folds': self.cv_folds,
            'cv_method': self.cv_method,
            'hidden_layers': self.hidden_layers,
            'activation_function': self.activation_function,
            'output_activation': self.output_activation,
            'use_batch_norm': self.use_batch_norm,
            'use_residual_connections': self.use_residual_connections,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'learning_rate_gbm': self.learning_rate_gbm,
            'subsample': self.subsample,
            'enable_ensemble': self.enable_ensemble,
            'ensemble_method': self.ensemble_method,
            'base_models': [m.value for m in self.base_models],
            'save_checkpoints': self.save_checkpoints,
            'checkpoint_frequency': self.checkpoint_frequency,
            'checkpoint_path': self.checkpoint_path,
            'keep_best_checkpoints': self.keep_best_checkpoints,
            'save_final_model': self.save_final_model,
            'log_training_progress': self.log_training_progress,
            'tensorboard_logging': self.tensorboard_logging,
            'wandb_integration': self.wandb_integration,
            'wandb_project': self.wandb_project,
            'log_interval': self.log_interval,
            'use_gpu': self.use_gpu,
            'gpu_memory_fraction': self.gpu_memory_fraction,
            'parallel_training': self.parallel_training,
            'max_workers': self.max_workers,
            'mixed_precision': self.mixed_precision,
            'enable_transfer_learning': self.enable_transfer_learning,
            'transfer_model_path': self.transfer_model_path,
            'enable_data_augmentation': self.enable_data_augmentation,
            'augmentation_factor': self.augmentation_factor
        }

@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    
    # Loss metrics
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    test_loss: Optional[float] = None
    
    # Accuracy metrics
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    test_accuracy: Optional[float] = None
    
    # Regression metrics
    train_mae: List[float] = field(default_factory=list)
    train_mse: List[float] = field(default_factory=list)
    train_rmse: List[float] = field(default_factory=list)
    train_r2: List[float] = field(default_factory=list)
    
    val_mae: List[float] = field(default_factory=list)
    val_mse: List[float] = field(default_factory=list)
    val_rmse: List[float] = field(default_factory=list)
    val_r2: List[float] = field(default_factory=list)
    
    test_mae: Optional[float] = None
    test_mse: Optional[float] = None
    test_rmse: Optional[float] = None
    test_r2: Optional[float] = None
    
    # Classification metrics
    train_precision: List[float] = field(default_factory=list)
    train_recall: List[float] = field(default_factory=list)
    train_f1: List[float] = field(default_factory=list)
    
    val_precision: List[float] = field(default_factory=list)
    val_recall: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_f1: Optional[float] = None
    
    # Timing metrics
    epoch_times: List[float] = field(default_factory=list)
    total_training_time: Optional[float] = None
    inference_time: Optional[float] = None
    
    # Resource metrics
    memory_usage: List[float] = field(default_factory=list)
    gpu_utilization: List[float] = field(default_factory=list)
    
    # Convergence metrics
    learning_rate_history: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    
    # Best metrics
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    best_val_metric: float = 0.0
    
    def update_epoch(self,
                    epoch: int,
                    train_loss: float,
                    val_loss: float,
                    train_metrics: Dict[str, float],
                    val_metrics: Dict[str, float],
                    epoch_time: float):
        """Update metrics for an epoch"""
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.epoch_times.append(epoch_time)
        
        # Update best metrics
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
        
        # Update other metrics
        for key, value in train_metrics.items():
            if key.startswith('train_'):
                getattr(self, key, []).append(value)
        
        for key, value in val_metrics.items():
            if key.startswith('val_'):
                getattr(self, key, []).append(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'test_loss': self.test_loss,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy,
            'test_accuracy': self.test_accuracy,
            'train_mae': self.train_mae,
            'train_mse': self.train_mse,
            'train_rmse': self.train_rmse,
            'train_r2': self.train_r2,
            'val_mae': self.val_mae,
            'val_mse': self.val_mse,
            'val_rmse': self.val_rmse,
            'val_r2': self.val_r2,
            'test_mae': self.test_mae,
            'test_mse': self.test_mse,
            'test_rmse': self.test_rmse,
            'test_r2': self.test_r2,
            'train_precision': self.train_precision,
            'train_recall': self.train_recall,
            'train_f1': self.train_f1,
            'val_precision': self.val_precision,
            'val_recall': self.val_recall,
            'val_f1': self.val_f1,
            'test_precision': self.test_precision,
            'test_recall': self.test_recall,
            'test_f1': self.test_f1,
            'epoch_times': self.epoch_times,
            'total_training_time': self.total_training_time,
            'inference_time': self.inference_time,
            'memory_usage': self.memory_usage,
            'gpu_utilization': self.gpu_utilization,
            'learning_rate_history': self.learning_rate_history,
            'gradient_norms': self.gradient_norms,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric
        }

@dataclass
class TrainingResult:
    """Result of model training"""
    
    # Model information
    model_id: str
    model_name: str
    model_type: ModelType
    model_task: ModelTask
    version: str
    
    # Model artifacts
    model: Any
    preprocessor: Optional[Any] = None
    feature_selector: Optional[Any] = None
    scaler: Optional[Any] = None
    
    # Training information
    config: TrainingConfig
    metrics: TrainingMetrics
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Data information
    training_data_hash: Optional[str] = None
    feature_names: List[str] = field(default_factory=list)
    target_name: str = "target"
    
    # Performance
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    final_test_loss: Optional[float] = None
    model_score: float = 0.0
    
    # Metadata
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    training_duration: Optional[float] = None
    status: TrainingStatus = TrainingStatus.PENDING
    error_message: Optional[str] = None
    
    # Checkpoints
    checkpoint_paths: List[str] = field(default_factory=list)
    best_checkpoint_path: Optional[str] = None
    
    def __post_init__(self):
        """Initialize result"""
        if self.end_time and self.start_time:
            self.training_duration = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_type': self.model_type.value,
            'model_task': self.model_task.value,
            'version': self.version,
            'config': self.config.to_dict(),
            'metrics': self.metrics.to_dict(),
            'hyperparameters': self.hyperparameters,
            'training_data_hash': self.training_data_hash,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'final_train_loss': self.final_train_loss,
            'final_val_loss': self.final_val_loss,
            'final_test_loss': self.final_test_loss,
            'model_score': self.model_score,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'training_duration': self.training_duration,
            'status': self.status.value,
            'error_message': self.error_message,
            'checkpoint_paths': self.checkpoint_paths,
            'best_checkpoint_path': self.best_checkpoint_path
        }

@dataclass
class HyperparameterSearchSpace:
    """Search space for hyperparameter tuning"""
    
    # Neural network parameters
    learning_rate: Tuple[float, float] = (1e-5, 1e-2)  # log scale
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    dropout_rate: Tuple[float, float] = (0.0, 0.5)
    hidden_units: List[List[int]] = field(default_factory=lambda: [
        [32], [64], [128], [32, 16], [64, 32], [128, 64], [64, 32, 16]
    ])
    activation: List[str] = field(default_factory=lambda: ['relu', 'tanh', 'sigmoid', 'leaky_relu'])
    
    # Tree-based parameters
    n_estimators: List[int] = field(default_factory=lambda: [50, 100, 200, 500])
    max_depth: List[int] = field(default_factory=lambda: [3, 5, 7, 10, 15])
    min_samples_split: List[int] = field(default_factory=lambda: [2, 5, 10])
    min_samples_leaf: List[int] = field(default_factory=lambda: [1, 2, 4])
    learning_rate_gbm: Tuple[float, float] = (0.01, 0.3)
    subsample: Tuple[float, float] = (0.5, 1.0)
    
    # Regularization
    l1_ratio: Tuple[float, float] = (0.0, 1.0)
    alpha: Tuple[float, float] = (1e-5, 1.0)  # log scale
    
    # Optimizer
    optimizer: List[str] = field(default_factory=lambda: ['adam', 'sgd', 'rmsprop', 'adagrad'])
    momentum: Tuple[float, float] = (0.8, 0.99)
    
    # Learning rate schedule
    lr_schedule: List[str] = field(default_factory=lambda: ['constant', 'exponential', 'step', 'cosine'])
    
    def get_search_space(self, model_type: ModelType) -> Dict[str, Any]:
        """Get search space for specific model type"""
        
        if model_type in [ModelType.TRANSFORMER, ModelType.LSTM_ATTENTION, 
                         ModelType.CNN_LSTM, ModelType.ENSEMBLE]:
            # Neural network parameters
            return {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'dropout_rate': self.dropout_rate,
                'hidden_layers': self.hidden_units,
                'activation': self.activation,
                'optimizer': self.optimizer,
                'l1_regularization': self.alpha,
                'l2_regularization': self.alpha
            }
        
        elif model_type in [ModelType.GRADIENT_BOOSTING, ModelType.RANDOM_FOREST]:
            # Tree-based parameters
            return {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'learning_rate': self.learning_rate_gbm,
                'subsample': self.subsample
            }
        
        elif model_type in [ModelType.SVM, ModelType.LINEAR_REGRESSION]:
            # Linear model parameters
            return {
                'C': self.alpha,
                'epsilon': self.learning_rate,
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        
        else:
            # Default parameters
            return {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'dropout_rate': self.dropout_rate
            }

# ============ Custom Datasets ============
class TimeSeriesDataset(Dataset):
    """Dataset for time series data"""
    
    def __init__(self, 
                 data: np.ndarray, 
                 targets: np.ndarray, 
                 sequence_length: int = 60,
                 forecast_horizon: int = 1):
        """
        Args:
            data: Time series data of shape (n_samples, n_features)
            targets: Target values of shape (n_samples,)
            sequence_length: Length of input sequences
            forecast_horizon: How many steps ahead to predict
        """
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Precompute indices
        self.indices = []
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        # Get sequence
        sequence = self.data[actual_idx:actual_idx + self.sequence_length]
        
        # Get target (forecast_horizon steps ahead)
        target_idx = actual_idx + self.sequence_length + self.forecast_horizon - 1
        target = self.targets[target_idx]
        
        return sequence, target

class TradingDataset(Dataset):
    """Dataset for trading data with multiple features"""
    
    def __init__(self, 
                 features: Dict[str, np.ndarray],
                 targets: np.ndarray,
                 sequence_length: int = 60,
                 include_metadata: bool = False):
        """
        Args:
            features: Dictionary of feature arrays
            targets: Target values
            sequence_length: Length of input sequences
            include_metadata: Whether to include metadata in batches
        """
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.include_metadata = include_metadata
        
        # Validate all features have same length
        lengths = [len(arr) for arr in features.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All feature arrays must have the same length")
        
        if len(targets) != lengths[0]:
            raise ValueError("Targets must have same length as features")
        
        self.n_samples = lengths[0] - sequence_length + 1
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Get sequence for each feature
        sequence_data = {}
        for feature_name, feature_array in self.features.items():
            sequence = feature_array[idx:idx + self.sequence_length]
            sequence_data[feature_name] = sequence
        
        # Get target
        target = self.targets[idx + self.sequence_length - 1]
        
        if self.include_metadata:
            # Include additional metadata
            metadata = {
                'sequence_idx': idx,
                'sequence_length': self.sequence_length,
                'timestamp': idx + self.sequence_length - 1
            }
            return sequence_data, target, metadata
        else:
            return sequence_data, target

# ============ Model Builders ============
class ModelBuilder:
    """Builds different types of models"""
    
    @staticmethod
    def build_neural_network(config: TrainingConfig, 
                            input_shape: Tuple[int, ...],
                            output_dim: int = 1) -> Any:
        """Build neural network model"""
        
        if TORCH_AVAILABLE:
            return ModelBuilder._build_pytorch_model(config, input_shape, output_dim)
        elif TF_AVAILABLE:
            return ModelBuilder._build_tensorflow_model(config, input_shape, output_dim)
        else:
            raise ImportError("Neither PyTorch nor TensorFlow is available")
    
    @staticmethod
    def _build_pytorch_model(config: TrainingConfig,
                            input_shape: Tuple[int, ...],
                            output_dim: int = 1) -> nn.Module:
        """Build PyTorch model"""
        
        class TradingModel(nn.Module):
            def __init__(self, config, input_shape, output_dim):
                super().__init__()
                self.config = config
                self.input_shape = input_shape
                
                # Determine input size
                if len(input_shape) == 3:  # (batch, sequence, features)
                    input_size = input_shape[-1]
                    self.is_sequence = True
                else:  # (batch, features)
                    input_size = input_shape[-1]
                    self.is_sequence = False
                
                # Build layers
                layers = []
                prev_size = input_size
                
                # Sequence processing
                if self.is_sequence and config.model_type == ModelType.LSTM_ATTENTION:
                    # LSTM with attention
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=config.hidden_layers[0],
                        batch_first=True,
                        dropout=config.dropout_rate if len(config.hidden_layers) > 1 else 0.0
                    )
                    self.attention = nn.MultiheadAttention(
                        embed_dim=config.hidden_layers[0],
                        num_heads=4,
                        dropout=config.dropout_rate
                    )
                    prev_size = config.hidden_layers[0]
                
                elif self.is_sequence and config.model_type == ModelType.CNN_LSTM:
                    # CNN-LSTM
                    self.conv1d = nn.Conv1d(
                        in_channels=input_size,
                        out_channels=config.hidden_layers[0],
                        kernel_size=3,
                        padding=1
                    )
                    self.lstm = nn.LSTM(
                        input_size=config.hidden_layers[0],
                        hidden_size=config.hidden_layers[1],
                        batch_first=True
                    )
                    prev_size = config.hidden_layers[1]
                
                else:
                    # Standard feedforward or sequence flattening
                    if self.is_sequence:
                        prev_size = input_size * input_shape[1]  # Flatten sequence
                
                # Hidden layers
                for i, hidden_size in enumerate(config.hidden_layers):
                    layers.append(nn.Linear(prev_size, hidden_size))
                    
                    if config.use_batch_norm:
                        layers.append(nn.BatchNorm1d(hidden_size))
                    
                    layers.append(ModelBuilder._get_activation(config.activation_function))
                    
                    if config.dropout_rate > 0:
                        layers.append(nn.Dropout(config.dropout_rate))
                    
                    prev_size = hidden_size
                
                # Output layer
                layers.append(nn.Linear(prev_size, output_dim))
                output_activation = ModelBuilder._get_activation(config.output_activation)
                if output_activation:
                    layers.append(output_activation)
                
                self.layers = nn.Sequential(*layers)
            
            def forward(self, x):
                # Handle sequence data
                if self.is_sequence and hasattr(self, 'lstm'):
                    if hasattr(self, 'conv1d'):
                        # CNN-LSTM: (batch, sequence, features) -> (batch, features, sequence)
                        x = x.transpose(1, 2)
                        x = self.conv1d(x)
                        x = x.transpose(1, 2)  # Back to (batch, sequence, features)
                    
                    # LSTM processing
                    lstm_out, _ = self.lstm(x)
                    
                    if hasattr(self, 'attention'):
                        # Apply attention
                        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                        x = attn_out[:, -1, :]  # Use last timestep
                    else:
                        x = lstm_out[:, -1, :]  # Use last timestep
                
                elif self.is_sequence:
                    # Flatten sequence
                    x = x.reshape(x.size(0), -1)
                
                # Feedforward layers
                return self.layers(x)
        
        return TradingModel(config, input_shape, output_dim)
    
    @staticmethod
    def _build_tensorflow_model(config: TrainingConfig,
                               input_shape: Tuple[int, ...],
                               output_dim: int = 1) -> keras.Model:
        """Build TensorFlow/Keras model"""
        
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # Sequence processing
        if len(input_shape) == 2:  # (sequence, features)
            if config.model_type == ModelType.LSTM_ATTENTION:
                # LSTM with attention
                x = keras.layers.LSTM(
                    config.hidden_layers[0],
                    return_sequences=True,
                    dropout=config.dropout_rate
                )(x)
                
                # Attention layer
                query = keras.layers.Dense(config.hidden_layers[0])(x)
                key = keras.layers.Dense(config.hidden_layers[0])(x)
                value = keras.layers.Dense(config.hidden_layers[0])(x)
                
                attention = keras.layers.Attention()([query, key, value])
                x = keras.layers.GlobalAveragePooling1D()(attention)
                
            elif config.model_type == ModelType.CNN_LSTM:
                # CNN-LSTM
                x = keras.layers.Conv1D(
                    filters=config.hidden_layers[0],
                    kernel_size=3,
                    padding='same',
                    activation=config.activation_function
                )(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.LSTM(
                    config.hidden_layers[1],
                    dropout=config.dropout_rate
                )(x)
            
            else:
                # Simple LSTM
                x = keras.layers.LSTM(
                    config.hidden_layers[0],
                    dropout=config.dropout_rate
                )(x)
        
        elif len(input_shape) == 1:  # (features,)
            # Flatten if needed
            x = keras.layers.Flatten()(x)
        
        # Hidden layers
        for i, hidden_size in enumerate(config.hidden_layers):
            x = keras.layers.Dense(hidden_size)(x)
            
            if config.use_batch_norm:
                x = keras.layers.BatchNormalization()(x)
            
            x = keras.layers.Activation(config.activation_function)(x)
            
            if config.dropout_rate > 0:
                x = keras.layers.Dropout(config.dropout_rate)(x)
        
        # Output layer
        outputs = keras.layers.Dense(
            output_dim, 
            activation=config.output_activation
        )(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    @staticmethod
    def build_tree_model(config: TrainingConfig) -> Any:
        """Build tree-based model"""
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available")
        
        if config.model_type == ModelType.RANDOM_FOREST:
            model = RandomForestRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf,
                random_state=config.random_seed,
                n_jobs=-1
            )
        
        elif config.model_type == ModelType.GRADIENT_BOOSTING:
            model = GradientBoostingRegressor(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate_gbm,
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf,
                subsample=config.subsample,
                random_state=config.random_seed
            )
        
        elif config.model_type == ModelType.XGBOOST:
            model = xgb.XGBRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate_gbm,
                subsample=config.subsample,
                random_state=config.random_seed,
                n_jobs=-1
            )
        
        elif config.model_type == ModelType.LIGHTGBM:
            model = lgb.LGBMRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate_gbm,
                subsample=config.subsample,
                random_state=config.random_seed,
                n_jobs=-1
            )
        
        else:
            raise ValueError(f"Unsupported tree model type: {config.model_type}")
        
        return model
    
    @staticmethod
    def build_linear_model(config: TrainingConfig) -> Any:
        """Build linear model"""
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available")
        
        if config.model_type == ModelType.LINEAR_REGRESSION:
            model = LinearRegression()
        
        elif config.model_type == ModelType.RIDGE:
            model = Ridge(
                alpha=config.l2_regularization,
                random_state=config.random_seed
            )
        
        elif config.model_type == ModelType.LASSO:
            model = Lasso(
                alpha=config.l1_regularization,
                random_state=config.random_seed
            )
        
        elif config.model_type == ModelType.SVM:
            model = SVR(
                C=1.0 / config.l2_regularization if config.l2_regularization > 0 else 1.0,
                epsilon=0.1
            )
        
        else:
            raise ValueError(f"Unsupported linear model type: {config.model_type}")
        
        return model
    
    @staticmethod
    def _get_activation(activation_name: str) -> nn.Module:
        """Get PyTorch activation function"""
        
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'prelu': nn.PReLU()
        }
        
        return activations.get(activation_name, nn.ReLU())

# ============ Training Utilities ============
class TrainingUtils:
    """Utility functions for training"""
    
    @staticmethod
    def prepare_data(data: pd.DataFrame,
                    target_column: str,
                    config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, 
                                                     np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray,
                                                     Any, Any]:
        """Prepare data for training"""
        
        # Separate features and target
        X = data.drop(columns=[target_column]).values
        y = data[target_column].values
        
        # Feature scaling
        scaler = None
        if config.feature_scaling:
            scaler = TrainingUtils._create_scaler(config.scale_method)
            X = scaler.fit_transform(X)
        
        # Feature selection
        feature_selector = None
        selected_indices = None
        
        if config.feature_selection and X.shape[1] > config.feature_selection_top_k:
            feature_selector, selected_indices = TrainingUtils._select_features(
                X, y, config.feature_selection_method, config.feature_selection_top_k
            )
            X = X[:, selected_indices]
        
        # Split data
        if config.time_series_split:
            # Time series split
            split_idx = int(len(X) * config.train_test_split)
            
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            
            # Further split training data for validation
            val_split_idx = int(len(X_train) * (1 - config.validation_split))
            X_train, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
            y_train, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=1 - config.train_test_split,
                random_state=config.random_seed
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=config.validation_split,
                random_state=config.random_seed
            )
        
        # Reshape for sequence models
        if config.model_type in [ModelType.LSTM_ATTENTION, ModelType.CNN_LSTM, 
                                ModelType.TRANSFORMER]:
            X_train = TrainingUtils._create_sequences(
                X_train, config.sequence_length
            )
            X_val = TrainingUtils._create_sequences(
                X_val, config.sequence_length
            )
            X_test = TrainingUtils._create_sequences(
                X_test, config.sequence_length
            )
        
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_selector
    
    @staticmethod
    def _create_scaler(method: str) -> Any:
        """Create scaler based on method"""
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
        
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'quantile': QuantileTransformer(output_distribution='normal')
        }
        
        return scalers.get(method, StandardScaler())
    
    @staticmethod
    def _select_features(X: np.ndarray, 
                        y: np.ndarray, 
                        method: str, 
                        top_k: int) -> Tuple[Any, np.ndarray]:
        """Select top features"""
        
        from sklearn.feature_selection import (SelectKBest, mutual_info_regression, 
                                              f_regression, RFE)
        from sklearn.linear_model import Lasso
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_regression, k=top_k)
        elif method == 'f_regression':
            selector = SelectKBest(f_regression, k=top_k)
        elif method == 'lasso':
            lasso = Lasso(alpha=0.01, random_state=42)
            selector = RFE(lasso, n_features_to_select=top_k)
        elif method == 'recursive':
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            selector = RFE(rf, n_features_to_select=top_k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)
        
        return selector, selected_indices
    
    @staticmethod
    def _create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for time series data"""
        
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        
        return np.array(sequences)
    
    @staticmethod
    def create_optimizer(model: Any, config: TrainingConfig) -> Any:
        """Create optimizer for model"""
        
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            # PyTorch optimizer
            if config.optimizer == 'adam':
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.l2_regularization
                )
            elif config.optimizer == 'sgd':
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=config.learning_rate,
                    momentum=0.9,
                    weight_decay=config.l2_regularization
                )
            elif config.optimizer == 'rmsprop':
                optimizer = optim.RMSprop(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.l2_regularization
                )
            else:
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=config.learning_rate
                )
            
            return optimizer
        
        elif TF_AVAILABLE and isinstance(model, keras.Model):
            # TensorFlow optimizer
            if config.optimizer == 'adam':
                optimizer = keras.optimizers.Adam(
                    learning_rate=config.learning_rate
                )
            elif config.optimizer == 'sgd':
                optimizer = keras.optimizers.SGD(
                    learning_rate=config.learning_rate,
                    momentum=0.9
                )
            elif config.optimizer == 'rmsprop':
                optimizer = keras.optimizers.RMSprop(
                    learning_rate=config.learning_rate
                )
            else:
                optimizer = keras.optimizers.Adam(
                    learning_rate=config.learning_rate
                )
            
            return optimizer
        
        else:
            # For scikit-learn models, no optimizer needed
            return None
    
    @staticmethod
    def create_loss_function(config: TrainingConfig) -> Any:
        """Create loss function"""
        
        if TORCH_AVAILABLE:
            if config.loss_function == 'mse':
                return nn.MSELoss()
            elif config.loss_function == 'mae':
                return nn.L1Loss()
            elif config.loss_function == 'huber':
                return nn.HuberLoss()
            elif config.loss_function == 'binary_crossentropy':
                return nn.BCELoss()
            elif config.loss_function == 'categorical_crossentropy':
                return nn.CrossEntropyLoss()
            else:
                return nn.MSELoss()
        
        elif TF_AVAILABLE:
            if config.loss_function == 'mse':
                return keras.losses.MeanSquaredError()
            elif config.loss_function == 'mae':
                return keras.losses.MeanAbsoluteError()
            elif config.loss_function == 'huber':
                return keras.losses.Huber()
            elif config.loss_function == 'binary_crossentropy':
                return keras.losses.BinaryCrossentropy()
            elif config.loss_function == 'categorical_crossentropy':
                return keras.losses.CategoricalCrossentropy()
            else:
                return keras.losses.MeanSquaredError()
        
        else:
            # For scikit-learn, loss is built-in
            return None

# ============ Main Model Trainer ============
class ModelTrainer:
    """Main model training engine"""
    
    def __init__(self, 
                 config: TrainingConfig,
                 model_manager: Optional[ModelManager] = None):
        
        self.config = config
        self.model_manager = model_manager
        self.logger = get_logger(__name__)
        
        # Training state
        self.current_phase = TrainingPhase.DATA_PREPARATION
        self.training_metrics = TrainingMetrics()
        self.best_model_state = None
        self.early_stopping_counter = 0
        
        # Resource management
        self.device = self._get_device()
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_workers)
        
        # Checkpoint management
        self.checkpoints = deque(maxlen=config.keep_best_checkpoints)
        
        # Hyperparameter tuning
        self.hyperparameter_results = []
        
        # Monitoring
        self.start_time = None
        self.current_epoch = 0
        
        # Data augmentation
        self.augmenter = None
        if config.enable_data_augmentation:
            self.augmenter = self._create_data_augmenter()
        
        self.logger.info(f"Model Trainer initialized for {config.model_type.value}")
        self.logger.info(f"Using device: {self.device}")
    
    def _get_device(self) -> str:
        """Get available device (CPU or GPU)"""
        
        if self.config.use_gpu:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            elif TF_AVAILABLE and tf.config.list_physical_devices('GPU'):
                return "gpu"
        
        return "cpu"
    
    def _create_data_augmenter(self) -> Any:
        """Create data augmenter"""
        # Simple data augmentation for time series
        # In production, implement more sophisticated augmentation
        return None
    
    def train(self,
              data: pd.DataFrame,
              target_column: str,
              model_name: str,
              description: str = "") -> TrainingResult:
        """Train a model with given data"""
        
        self.start_time = datetime.now()
        
        # Generate model ID
        model_id = f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        version = "1.0.0"
        
        # Create training result
        result = TrainingResult(
            model_id=model_id,
            model_name=model_name,
            model_type=self.config.model_type,
            model_task=self.config.model_task,
            version=version,
            config=self.config,
            metrics=self.training_metrics,
            start_time=self.start_time,
            status=TrainingStatus.RUNNING
        )
        
        try:
            # Phase 1: Data Preparation
            self.current_phase = TrainingPhase.DATA_PREPARATION
            self.logger.info("Phase 1: Data Preparation")
            
            X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_selector = \
                TrainingUtils.prepare_data(data, target_column, self.config)
            
            # Store data hash
            data_hash = hashlib.md5(
                np.concatenate([X_train.flatten(), y_train.flatten()])
            ).hexdigest()
            result.training_data_hash = data_hash
            
            # Phase 2: Feature Engineering
            self.current_phase = TrainingPhase.FEATURE_ENGINEERING
            self.logger.info("Phase 2: Feature Engineering")
            
            # Store preprocessors
            result.scaler = scaler
            result.feature_selector = feature_selector
            
            # Phase 3: Hyperparameter Tuning (if enabled)
            if self.config.enable_hyperparameter_tuning:
                self.current_phase = TrainingPhase.HYPERPARAMETER_TUNING
                self.logger.info("Phase 3: Hyperparameter Tuning")
                
                best_params = self._tune_hyperparameters(
                    X_train, y_train, X_val, y_val
                )
                
                # Update config with best parameters
                self.config = self._update_config_with_params(best_params)
                result.hyperparameters = best_params
                
                self.logger.info(f"Best hyperparameters: {best_params}")
            
            # Phase 4: Model Training
            self.current_phase = TrainingPhase.MODEL_TRAINING
            self.logger.info("Phase 4: Model Training")
            
            # Build model
            model = self._build_model(X_train.shape)
            
            # Train model
            trained_model, training_history = self._train_model(
                model, X_train, y_train, X_val, y_val
            )
            
            result.model = trained_model
            
            # Phase 5: Model Evaluation
            self.current_phase = TrainingPhase.EVALUATION
            self.logger.info("Phase 5: Model Evaluation")
            
            # Evaluate on test set
            test_metrics = self._evaluate_model(trained_model, X_test, y_test)
            self.training_metrics.test_loss = test_metrics.get('loss', 0.0)
            
            # Calculate model score
            result.model_score = self._calculate_model_score(test_metrics)
            result.final_test_loss = self.training_metrics.test_loss
            
            # Phase 6: Saving
            self.current_phase = TrainingPhase.SAVING
            self.logger.info("Phase 6: Saving Model")
            
            # Save model if enabled
            if self.config.save_final_model and self.model_manager:
                self._save_model(result, trained_model)
            
            # Update result
            result.end_time = datetime.now()
            result.status = TrainingStatus.COMPLETED
            result.training_duration = (result.end_time - result.start_time).total_seconds()
            result.metrics.total_training_time = result.training_duration
            
            self.logger.info(
                f"Training completed in {result.training_duration:.2f}s. "
                f"Model score: {result.model_score:.4f}"
            )
            
            return result
            
        except Exception as e:
            # Handle training failure
            result.end_time = datetime.now()
            result.status = TrainingStatus.FAILED
            result.error_message = str(e)
            result.training_duration = (result.end_time - result.start_time).total_seconds()
            
            self.logger.error(f"Training failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return result
    
    def _build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """Build model based on configuration"""
        
        if self.config.model_type in [ModelType.TRANSFORMER, ModelType.LSTM_ATTENTION,
                                     ModelType.CNN_LSTM, ModelType.ENSEMBLE]:
            # Neural network models
            model = ModelBuilder.build_neural_network(
                self.config, input_shape
            )
            
            # Move to device if PyTorch
            if TORCH_AVAILABLE and isinstance(model, nn.Module):
                model = model.to(self.device)
            
            return model
        
        elif self.config.model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING,
                                       ModelType.XGBOOST, ModelType.LIGHTGBM]:
            # Tree-based models
            return ModelBuilder.build_tree_model(self.config)
        
        elif self.config.model_type in [ModelType.LINEAR_REGRESSION, ModelType.RIDGE,
                                       ModelType.LASSO, ModelType.SVM]:
            # Linear models
            return ModelBuilder.build_linear_model(self.config)
        
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _train_model(self,
                    model: Any,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray) -> Tuple[Any, Dict[str, List[float]]]:
        """Train the model"""
        
        if self.config.model_type in [ModelType.TRANSFORMER, ModelType.LSTM_ATTENTION,
                                     ModelType.CNN_LSTM, ModelType.ENSEMBLE]:
            # Neural network training
            return self._train_neural_network(model, X_train, y_train, X_val, y_val)
        
        else:
            # Traditional ML training
            return self._train_traditional_ml(model, X_train, y_train, X_val, y_val)
    
    def _train_neural_network(self,
                             model: Any,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray) -> Tuple[Any, Dict[str, List[float]]]:
        """Train neural network model"""
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            # PyTorch training
            return self._train_pytorch_model(model, X_train, y_train, X_val, y_val)
        
        elif TF_AVAILABLE and isinstance(model, keras.Model):
            # TensorFlow training
            return self._train_tensorflow_model(model, X_train, y_train, X_val, y_val)
        
        else:
            raise ValueError("Neural network framework not available")
    
    def _train_pytorch_model(self,
                            model: nn.Module,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Train PyTorch model"""
        
        # Create optimizer and loss function
        optimizer = TrainingUtils.create_optimizer(model, self.config)
        criterion = TrainingUtils.create_loss_function(self.config)
        
        # Create datasets and dataloaders
        if len(X_train.shape) == 3:  # Sequence data
            train_dataset = TimeSeriesDataset(
                X_train, y_train,
                sequence_length=self.config.sequence_length,
                forecast_horizon=self.config.forecast_horizon
            )
            val_dataset = TimeSeriesDataset(
                X_val, y_val,
                sequence_length=self.config.sequence_length,
                forecast_horizon=self.config.forecast_horizon
            )
        else:
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=not self.config.time_series_split
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions.squeeze(), batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(predictions.squeeze() - batch_y)).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_train_mae = train_mae / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    predictions = model(batch_X)
                    loss = criterion(predictions.squeeze(), batch_y)
                    
                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(predictions.squeeze() - batch_y)).item()
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mae = val_mae / len(val_loader)
            
            # Update training history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['train_mae'].append(avg_train_mae)
            training_history['val_mae'].append(avg_val_mae)
            
            # Update metrics
            self.training_metrics.update_epoch(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                train_metrics={'train_mae': avg_train_mae},
                val_metrics={'val_mae': avg_val_mae},
                epoch_time=0.0  # Would calculate actual time
            )
            
            # Checkpointing
            if self.config.save_checkpoints and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_checkpoint(model, optimizer, epoch, avg_val_loss)
            
            # Early stopping
            if self._check_early_stopping(avg_val_loss):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Log progress
            if epoch % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch}/{self.config.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Train MAE: {avg_train_mae:.4f}, "
                    f"Val MAE: {avg_val_mae:.4f}"
                )
        
        # Load best model
        if self.best_model_state:
            model.load_state_dict(self.best_model_state['model_state_dict'])
        
        return model, training_history
    
    def _train_tensorflow_model(self,
                               model: keras.Model,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_val: np.ndarray,
                               y_val: np.ndarray) -> Tuple[keras.Model, Dict[str, List[float]]]:
        """Train TensorFlow model"""
        
        # Compile model
        optimizer = TrainingUtils.create_optimizer(model, self.config)
        loss = TrainingUtils.create_loss_function(self.config)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=self.config.metrics
        )
        
        # Callbacks
        callbacks = []
        
        if self.config.early_stopping_patience > 0:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_delta,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        if self.config.save_checkpoints:
            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=self.config.checkpoint_path + '/best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
            callbacks.append(checkpoint)
        
        if self.config.tensorboard_logging:
            tensorboard = keras.callbacks.TensorBoard(
                log_dir='logs/tensorboard',
                histogram_freq=1
            )
            callbacks.append(tensorboard)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=self.config.log_interval
        )
        
        # Convert history to dictionary
        training_history = {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
        
        # Add metrics
        for metric in self.config.metrics:
            if metric in history.history:
                training_history[f'train_{metric}'] = history.history[metric]
                training_history[f'val_{metric}'] = history.history[f'val_{metric}']
        
        return model, training_history
    
    def _train_traditional_ml(self,
                             model: Any,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray) -> Tuple[Any, Dict[str, List[float]]]:
        """Train traditional ML model"""
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        training_history = {
            'train_loss': [train_mse],
            'val_loss': [val_mse],
            'train_mae': [train_mae],
            'val_mae': [val_mae],
            'train_r2': [train_r2],
            'val_r2': [val_r2]
        }
        
        # Update metrics
        self.training_metrics.update_epoch(
            epoch=0,
            train_loss=train_mse,
            val_loss=val_mse,
            train_metrics={'train_mae': train_mae, 'train_r2': train_r2},
            val_metrics={'val_mae': val_mae, 'val_r2': val_r2},
            epoch_time=0.0
        )
        
        self.training_metrics.total_training_time = (datetime.now() - self.start_time).total_seconds()
        
        return model, training_history
    
    def _tune_hyperparameters(self,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        
        search_space = HyperparameterSearchSpace()
        param_space = search_space.get_search_space(self.config.model_type)
        
        if self.config.tuning_method == 'grid':
            return self._grid_search(param_space, X_train, y_train, X_val, y_val)
        
        elif self.config.tuning_method == 'random':
            return self._random_search(param_space, X_train, y_train, X_val, y_val)
        
        elif self.config.tuning_method == 'bayesian':
            return self._bayesian_optimization(param_space, X_train, y_train, X_val, y_val)
        
        elif self.config.tuning_method == 'genetic':
            return self._genetic_algorithm(param_space, X_train, y_train, X_val, y_val)
        
        else:
            self.logger.warning(f"Unknown tuning method: {self.config.tuning_method}")
            return {}
    
    def _grid_search(self,
                    param_space: Dict[str, Any],
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray) -> Dict[str, Any]:
        """Grid search hyperparameter tuning"""
        
        best_params = {}
        best_score = -float('inf')
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        # Convert ranges to lists
        for i, values in enumerate(param_values):
            if isinstance(values, tuple) and len(values) == 2:
                # Convert range to list of values
                start, end = values
                if isinstance(start, int) and isinstance(end, int):
                    param_values[i] = list(range(start, end + 1))
                else:
                    # Float range
                    param_values[i] = [start, (start + end) / 2, end]
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"Grid search with {len(combinations)} combinations")
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            try:
                # Update config with these parameters
                temp_config = self._update_config_with_params(params)
                
                # Build and train model
                model = self._build_model(X_train.shape)
                
                if self.config.model_type in [ModelType.TRANSFORMER, ModelType.LSTM_ATTENTION,
                                            ModelType.CNN_LSTM, ModelType.ENSEMBLE]:
                    # Neural network - quick training
                    model, _ = self._train_neural_network(
                        model, X_train, y_train, X_val, y_val
                    )
                    
                    # Evaluate
                    val_pred = self._predict_neural_network(model, X_val)
                    score = -mean_squared_error(y_val, val_pred)  # Negative because lower is better
                
                else:
                    # Traditional ML
                    model.fit(X_train, y_train)
                    val_pred = model.predict(X_val)
                    score = -mean_squared_error(y_val, val_pred)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params
                
                self.logger.debug(f"Combination {i+1}/{len(combinations)}: Score={score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Failed with params {params}: {str(e)}")
                continue
        
        self.logger.info(f"Grid search best score: {best_score:.4f}")
        return best_params
    
    def _random_search(self,
                      param_space: Dict[str, Any],
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray) -> Dict[str, Any]:
        """Random search hyperparameter tuning"""
        
        import random
        
        best_params = {}
        best_score = -float('inf')
        
        self.logger.info(f"Random search with {self.config.tuning_iterations} iterations")
        
        for i in range(self.config.tuning_iterations):
            # Sample random parameters
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    params[param_name] = random.choice(param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    start, end = param_values
                    if isinstance(start, int) and isinstance(end, int):
                        params[param_name] = random.randint(start, end)
                    else:
                        params[param_name] = random.uniform(start, end)
            
            try:
                # Update config with these parameters
                temp_config = self._update_config_with_params(params)
                
                # Build and train model
                model = self._build_model(X_train.shape)
                
                if self.config.model_type in [ModelType.TRANSFORMER, ModelType.LSTM_ATTENTION,
                                            ModelType.CNN_LSTM, ModelType.ENSEMBLE]:
                    # Neural network - quick training
                    model, _ = self._train_neural_network(
                        model, X_train, y_train, X_val, y_val
                    )
                    
                    # Evaluate
                    val_pred = self._predict_neural_network(model, X_val)
                    score = -mean_squared_error(y_val, val_pred)
                
                else:
                    # Traditional ML
                    model.fit(X_train, y_train)
                    val_pred = model.predict(X_val)
                    score = -mean_squared_error(y_val, val_pred)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params
                
                self.logger.debug(f"Iteration {i+1}/{self.config.tuning_iterations}: Score={score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Failed with params {params}: {str(e)}")
                continue
        
        self.logger.info(f"Random search best score: {best_score:.4f}")
        return best_params
    
    def _bayesian_optimization(self,
                              param_space: Dict[str, Any],
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_val: np.ndarray,
                              y_val: np.ndarray) -> Dict[str, Any]:
        """Bayesian optimization hyperparameter tuning"""
        
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
            
            # Define search space for skopt
            dimensions = []
            param_names = []
            
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    dimensions.append(Categorical(param_values, name=param_name))
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    start, end = param_values
                    if isinstance(start, int) and isinstance(end, int):
                        dimensions.append(Integer(start, end, name=param_name))
                    else:
                        dimensions.append(Real(start, end, name=param_name))
                
                param_names.append(param_name)
            
            @use_named_args(dimensions=dimensions)
            def objective(**params):
                try:
                    # Update config with these parameters
                    temp_config = self._update_config_with_params(params)
                    
                    # Build and train model
                    model = self._build_model(X_train.shape)
                    
                    if self.config.model_type in [ModelType.TRANSFORMER, ModelType.LSTM_ATTENTION,
                                                ModelType.CNN_LSTM, ModelType.ENSEMBLE]:
                        # Neural network - quick training
                        model, _ = self._train_neural_network(
                            model, X_train, y_train, X_val, y_val
                        )
                        
                        # Evaluate
                        val_pred = self._predict_neural_network(model, X_val)
                        score = mean_squared_error(y_val, val_pred)  # Minimize MSE
                    
                    else:
                        # Traditional ML
                        model.fit(X_train, y_train)
                        val_pred = model.predict(X_val)
                        score = mean_squared_error(y_val, val_pred)
                    
                    return score
                    
                except Exception as e:
                    self.logger.warning(f"Failed in Bayesian optimization: {str(e)}")
                    return float('inf')
            
            # Run optimization
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=self.config.tuning_iterations,
                random_state=self.config.random_seed,
                verbose=False
            )
            
            # Get best parameters
            best_params = dict(zip(param_names, result.x))
            best_score = result.fun
            
            self.logger.info(f"Bayesian optimization best score: {best_score:.4f}")
            return best_params
            
        except ImportError:
            self.logger.warning("scikit-optimize not installed. Falling back to random search.")
            return self._random_search(param_space, X_train, y_train, X_val, y_val)
    
    def _genetic_algorithm(self,
                          param_space: Dict[str, Any],
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray) -> Dict[str, Any]:
        """Genetic algorithm hyperparameter tuning"""
        
        try:
            from deap import base, creator, tools, algorithms
            
            # Create fitness function
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
            # Define parameter types
            param_types = []
            param_bounds = []
            
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    # Categorical parameter
                    param_types.append('categorical')
                    param_bounds.append(param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    start, end = param_values
                    if isinstance(start, int) and isinstance(end, int):
                        # Integer parameter
                        param_types.append('int')
                        param_bounds.append((start, end))
                    else:
                        # Float parameter
                        param_types.append('float')
                        param_bounds.append((start, end))
            
            def evaluate_individual(individual):
                """Evaluate a set of hyperparameters"""
                params = {}
                for i, (param_name, param_type) in enumerate(zip(param_space.keys(), param_types)):
                    if param_type == 'categorical':
                        params[param_name] = param_bounds[i][int(individual[i]) % len(param_bounds[i])]
                    else:
                        params[param_name] = individual[i]
                
                try:
                    # Update config with these parameters
                    temp_config = self._update_config_with_params(params)
                    
                    # Build and train model
                    model = self._build_model(X_train.shape)
                    
                    if self.config.model_type in [ModelType.TRANSFORMER, ModelType.LSTM_ATTENTION,
                                                ModelType.CNN_LSTM, ModelType.ENSEMBLE]:
                        # Neural network - quick training
                        model, _ = self._train_neural_network(
                            model, X_train, y_train, X_val, y_val
                        )
                        
                        # Evaluate
                        val_pred = self._predict_neural_network(model, X_val)
                        score = mean_squared_error(y_val, val_pred)
                    
                    else:
                        # Traditional ML
                        model.fit(X_train, y_train)
                        val_pred = model.predict(X_val)
                        score = mean_squared_error(y_val, val_pred)
                    
                    return (score,)
                    
                except Exception as e:
                    self.logger.warning(f"Failed in genetic algorithm: {str(e)}")
                    return (float('inf'),)
            
            # Initialize genetic algorithm
            toolbox = base.Toolbox()
            
            # Register individual and population creation
            for i, (param_type, bounds) in enumerate(zip(param_types, param_bounds)):
                if param_type == 'categorical':
                    toolbox.register(f"attr_{i}", np.random.randint, 0, len(bounds))
                elif param_type == 'int':
                    toolbox.register(f"attr_{i}", np.random.randint, bounds[0], bounds[1] + 1)
                else:  # float
                    toolbox.register(f"attr_{i}", np.random.uniform, bounds[0], bounds[1])
            
            toolbox.register("individual", tools.initCycle, creator.Individual,
                           [getattr(toolbox, f"attr_{i}") for i in range(len(param_types))], n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Register genetic operators
            toolbox.register("evaluate", evaluate_individual)
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Create population
            population = toolbox.population(n=50)
            
            # Run evolution
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            
            population, logbook = algorithms.eaSimple(
                population, toolbox,
                cxpb=0.5, mutpb=0.2,
                ngen=self.config.tuning_iterations // 10,
                stats=stats, halloffame=hof,
                verbose=False
            )
            
            # Get best individual
            best_individual = hof[0]
            
            # Convert back to parameters
            best_params = {}
            for i, (param_name, param_type) in enumerate(zip(param_space.keys(), param_types)):
                if param_type == 'categorical':
                    best_params[param_name] = param_bounds[i][int(best_individual[i]) % len(param_bounds[i])]
                else:
                    best_params[param_name] = best_individual[i]
            
            best_score = best_individual.fitness.values[0]
            
            self.logger.info(f"Genetic algorithm best score: {best_score:.4f}")
            return best_params
            
        except ImportError:
            self.logger.warning("DEAP not installed. Falling back to random search.")
            return self._random_search(param_space, X_train, y_train, X_val, y_val)
    
    def _update_config_with_params(self, params: Dict[str, Any]) -> TrainingConfig:
        """Update training config with new parameters"""
        
        # Create a copy of the config
        import copy
        new_config = copy.deepcopy(self.config)
        
        # Update parameters
        for key, value in params.items():
            if hasattr(new_config, key):
                # Handle special cases
                if key == 'learning_rate' and hasattr(new_config, 'learning_rate_gbm'):
                    # Update both learning rates
                    setattr(new_config, key, value)
                    setattr(new_config, 'learning_rate_gbm', value)
                else:
                    setattr(new_config, key, value)
        
        return new_config
    
    def _predict_neural_network(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions with neural network"""
        
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                predictions = model(X_tensor).cpu().numpy()
            return predictions.flatten()
        
        elif TF_AVAILABLE and isinstance(model, keras.Model):
            return model.predict(X, verbose=0).flatten()
        
        else:
            raise ValueError("Unsupported model type")
    
    def _evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data"""
        
        # Make predictions
        if self.config.model_type in [ModelType.TRANSFORMER, ModelType.LSTM_ATTENTION,
                                     ModelType.CNN_LSTM, ModelType.ENSEMBLE]:
            y_pred = self._predict_neural_network(model, X_test)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'loss': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        # For classification tasks
        if self.config.model_task in [ModelTask.CLASSIFICATION, ModelTask.MULTI_CLASSIFICATION]:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            if self.config.model_task == ModelTask.CLASSIFICATION:
                y_pred_class = (y_pred > 0.5).astype(int)
            else:
                y_pred_class = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else y_pred
            
            metrics.update({
                'accuracy': accuracy_score(y_test, y_pred_class),
                'precision': precision_score(y_test, y_pred_class, average='weighted'),
                'recall': recall_score(y_test, y_pred_class, average='weighted'),
                'f1': f1_score(y_test, y_pred_class, average='weighted')
            })
        
        # Calculate additional metrics for trading
        if len(y_test) > 1:
            returns_pred = np.diff(y_pred) / y_pred[:-1]
            returns_actual = np.diff(y_test) / y_test[:-1]
            
            # Direction accuracy
            direction_correct = np.sign(returns_pred) == np.sign(returns_actual)
            metrics['direction_accuracy'] = np.mean(direction_correct)
            
            # Sharpe ratio (simplified)
            if len(returns_pred) > 1:
                metrics['sharpe_ratio'] = np.mean(returns_pred) / np.std(returns_pred) * np.sqrt(252)
        
        self.logger.info(f"Test metrics: {metrics}")
        return metrics
    
    def _calculate_model_score(self, test_metrics: Dict[str, float]) -> float:
        """Calculate overall model score"""
        
        # Weight different metrics based on task
        if self.config.model_task == ModelTask.REGRESSION:
            # For regression, prioritize R² and RMSE
            r2_weight = 0.4
            rmse_weight = 0.3
            mae_weight = 0.2
            direction_weight = 0.1
            
            score = (
                test_metrics.get('r2', 0) * r2_weight +
                (1 - test_metrics.get('rmse', 1) / (np.abs(np.mean(test_metrics.get('rmse', 1))) + 1e-10)) * rmse_weight +
                (1 - test_metrics.get('mae', 1) / (np.abs(np.mean(test_metrics.get('mae', 1))) + 1e-10)) * mae_weight +
                test_metrics.get('direction_accuracy', 0.5) * direction_weight
            )
        
        elif self.config.model_task == ModelTask.CLASSIFICATION:
            # For classification, prioritize F1 and accuracy
            f1_weight = 0.5
            accuracy_weight = 0.3
            precision_weight = 0.1
            recall_weight = 0.1
            
            score = (
                test_metrics.get('f1', 0) * f1_weight +
                test_metrics.get('accuracy', 0) * accuracy_weight +
                test_metrics.get('precision', 0) * precision_weight +
                test_metrics.get('recall', 0) * recall_weight
            )
        
        else:
            # Default score
            score = test_metrics.get('r2', 0) if 'r2' in test_metrics else test_metrics.get('accuracy', 0)
        
        return max(0, min(1, score))  # Clip to [0, 1]
    
    def _save_checkpoint(self, model: Any, optimizer: Any, epoch: int, val_loss: float):
        """Save model checkpoint"""
        
        checkpoint_path = Path(self.config.checkpoint_path) / f"checkpoint_epoch_{epoch}.pt"
        
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'val_loss': val_loss,
                'config': self.config.to_dict()
            }
            torch.save(checkpoint, checkpoint_path)
        
        elif TF_AVAILABLE and isinstance(model, keras.Model):
            model.save(checkpoint_path)
        
        else:
            # For scikit-learn models
            import joblib
            joblib.dump(model, checkpoint_path)
        
        # Add to checkpoints list
        self.checkpoints.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'val_loss': val_loss
        })
        
        # Keep only the best checkpoints
        self.checkpoints = deque(
            sorted(self.checkpoints, key=lambda x: x['val_loss'])[:self.config.keep_best_checkpoints],
            maxlen=self.config.keep_best_checkpoints
        )
        
        # Update best model state
        if val_loss == min(c['val_loss'] for c in self.checkpoints):
            self.best_model_state = checkpoint if TORCH_AVAILABLE and isinstance(model, nn.Module) else None
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping criteria are met"""
        
        if self.config.early_stopping_patience <= 0:
            return False
        
        if val_loss < self.training_metrics.best_val_loss - self.config.early_stopping_delta:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def _save_model(self, result: TrainingResult, model: Any):
        """Save trained model to model manager"""
        
        if not self.model_manager:
            return
        
        try:
            # Create model metadata
            metadata = ModelMetadata(
                model_id=result.model_id,
                name=result.model_name,
                model_type=result.model_type,
                version=result.version,
                description=f"Trained on {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                created_at=result.start_time,
                updated_at=result.end_time,
                status=ModelStatus.TRAINED,
                performance_metrics=result.metrics.to_dict(),
                hyperparameters=result.hyperparameters,
                training_config=result.config.to_dict(),
                feature_columns=result.feature_names,
                target_column=result.target_name,
                training_data_hash=result.training_data_hash,
                model_score=result.model_score
            )
            
            # Save model
            self.model_manager.save_model(
                model=model,
                metadata=metadata,
                preprocessor=result.scaler,
                feature_selector=result.feature_selector
            )
            
            result.best_checkpoint_path = self.model_manager.get_model_path(result.model_id)
            self.logger.info(f"Model saved with ID: {result.model_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
    
    def train_ensemble(self,
                      data: pd.DataFrame,
                      target_column: str,
                      model_name: str,
                      base_models: List[ModelType] = None) -> TrainingResult:
        """Train an ensemble model"""
        
        if not self.config.enable_ensemble:
            raise ValueError("Ensemble training is not enabled")
        
        if base_models is None:
            base_models = [
                ModelType.RANDOM_FOREST,
                ModelType.GRADIENT_BOOSTING,
                ModelType.LSTM_ATTENTION
            ]
        
        self.logger.info(f"Training ensemble with {len(base_models)} base models")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_selector = \
            TrainingUtils.prepare_data(data, target_column, self.config)
        
        # Train base models
        base_predictions_train = []
        base_predictions_val = []
        base_predictions_test = []
        
        for i, model_type in enumerate(base_models):
            self.logger.info(f"Training base model {i+1}/{len(base_models)}: {model_type.value}")
            
            # Create config for base model
            base_config = TrainingConfig(
                model_type=model_type,
                model_task=self.config.model_task,
                random_seed=self.config.random_seed + i
            )
            
            # Train base model
            trainer = ModelTrainer(base_config, self.model_manager)
            result = trainer.train(
                data=data,
                target_column=target_column,
                model_name=f"{model_name}_base_{i}"
            )
            
            if result.status == TrainingStatus.COMPLETED:
                # Get predictions
                if model_type in [ModelType.TRANSFORMER, ModelType.LSTM_ATTENTION,
                                ModelType.CNN_LSTM, ModelType.ENSEMBLE]:
                    pred_train = trainer._predict_neural_network(result.model, X_train)
                    pred_val = trainer._predict_neural_network(result.model, X_val)
                    pred_test = trainer._predict_neural_network(result.model, X_test)
                else:
                    pred_train = result.model.predict(X_train)
                    pred_val = result.model.predict(X_val)
                    pred_test = result.model.predict(X_test)
                
                base_predictions_train.append(pred_train.reshape(-1, 1))
                base_predictions_val.append(pred_val.reshape(-1, 1))
                base_predictions_test.append(pred_test.reshape(-1, 1))
        
        if not base_predictions_train:
            raise ValueError("No base models were successfully trained")
        
        # Stack predictions
        X_meta_train = np.hstack(base_predictions_train)
        X_meta_val = np.hstack(base_predictions_val)
        X_meta_test = np.hstack(base_predictions_test)
        
        # Train meta-model
        self.logger.info("Training meta-model")
        
        if self.config.ensemble_method == 'stacking':
            # Use linear regression as meta-model
            from sklearn.linear_model import LinearRegression
            meta_model = LinearRegression()
            meta_model.fit(X_meta_train, y_train)
            
            # Evaluate
            train_score = meta_model.score(X_meta_train, y_train)
            val_score = meta_model.score(X_meta_val, y_val)
            test_score = meta_model.score(X_meta_test, y_test)
            
            self.logger.info(f"Meta-model scores - Train: {train_score:.4f}, Val: {val_score:.4f}, Test: {test_score:.4f}")
            
            # Create ensemble result
            ensemble_result = TrainingResult(
                model_id=f"ensemble_{uuid.uuid4().hex[:8]}",
                model_name=model_name,
                model_type=ModelType.ENSEMBLE,
                model_task=self.config.model_task,
                version="1.0.0",
                model=meta_model,
                config=self.config,
                metrics=TrainingMetrics(),
                start_time=self.start_time,
                end_time=datetime.now(),
                status=TrainingStatus.COMPLETED
            )
            
            return ensemble_result
        
        else:
            raise ValueError(f"Unsupported ensemble method: {self.config.ensemble_method}")
    
    def cancel_training(self):
        """Cancel ongoing training"""
        
        self.logger.info("Cancelling training...")
        self.current_phase = TrainingPhase.SAVING
        self.status = TrainingStatus.CANCELLED
        
        # Stop thread and process pools
        self.thread_pool.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress"""
        
        return {
            'phase': self.current_phase.value,
            'epoch': self.current_epoch,
            'total_epochs': self.config.epochs,
            'metrics': self.training_metrics.to_dict(),
            'elapsed_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'checkpoints': list(self.checkpoints)
        }


# ============ Helper Functions ============
def create_model_trainer(model_type: ModelType,
                        model_manager: Optional[ModelManager] = None,
                        **kwargs) -> ModelTrainer:
    """Factory function to create model trainer"""
    
    # Create base config
    config = TrainingConfig(
        model_type=model_type,
        **kwargs
    )
    
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return ModelTrainer(config, model_manager)


def load_trained_model(model_path: str) -> Tuple[Any, Dict[str, Any]]:
    """Load a trained model from disk"""
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Try different loading methods
    try:
        if model_path.suffix == '.pt':
            # PyTorch model
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available")
            
            checkpoint = torch.load(model_path, map_location='cpu')
            return checkpoint['model_state_dict'], checkpoint.get('config', {})
        
        elif model_path.suffix == '.keras' or model_path.suffix == '.h5':
            # TensorFlow/Keras model
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not available")
            
            model = keras.models.load_model(model_path)
            # Try to load config
            config_path = model_path.with_suffix('.json')
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            return model, config
        
        elif model_path.suffix == '.joblib' or model_path.suffix == '.pkl':
            # Scikit-learn model
            import joblib
            model = joblib.load(model_path)
            
            # Try to load config
            config_path = model_path.with_suffix('.json')
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            return model, config
        
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def validate_model_performance(model: Any,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              threshold: float = 0.5) -> Dict[str, bool]:
    """Validate model performance meets minimum requirements"""
    
    results = {
        'passed': True,
        'checks': {}
    }
    
    try:
        # Make predictions
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                predictions = model(X_tensor).numpy().flatten()
        elif TF_AVAILABLE and isinstance(model, keras.Model):
            predictions = model.predict(X_test, verbose=0).flatten()
        else:
            predictions = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Check metrics against thresholds
        results['checks']['mse'] = mse < 0.1  # MSE should be less than 0.1
        results['checks']['mae'] = mae < 0.3  # MAE should be less than 0.3
        results['checks']['r2'] = r2 > 0.5   # R² should be greater than 0.5
        
        # Check for NaN predictions
        results['checks']['no_nan'] = not np.any(np.isnan(predictions))
        
        # Check for extreme predictions
        pred_range = np.max(predictions) - np.min(predictions)
        results['checks']['reasonable_range'] = pred_range < 10 * np.std(y_test)
        
        # Overall pass/fail
        results['passed'] = all(results['checks'].values())
        
        results['metrics'] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'prediction_range': pred_range
        }
        
    except Exception as e:
        results['passed'] = False
        results['error'] = str(e)
    
    return results


# ============ Example Usage ============
if __name__ == "__main__":
    # Example usage
    print("Model Trainer Module")
    
    # Create a sample config
    config = TrainingConfig(
        model_type=ModelType.LSTM_ATTENTION,
        epochs=10,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Create trainer
    trainer = ModelTrainer(config)
    
    print(f"Trainer created for {config.model_type.value}")
    print(f"Device: {trainer.device}")