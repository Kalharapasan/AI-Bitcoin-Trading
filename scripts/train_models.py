#!/usr/bin/env python3
"""
Model training script for Bitcoin Trading AI application.
Train and evaluate various neural network models for price prediction.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.data_processing.data_collector import DataCollector
from core.data_processing.feature_engineer import FeatureEngineer
from core.data_processing.data_preprocessor import DataPreprocessor
from core.neural_networks.transformer_model import TransformerModel
from core.neural_networks.lstm_attention import LSTMAttentionModel
from core.neural_networks.cnn_lstm import CNNLSTMModel
from core.neural_networks.ensemble_model import EnsembleModel
from core.models.model_trainer import ModelTrainer
from core.models.model_manager import ModelManager
from database.connection import get_database_manager
from database.crud import ModelTrainingCRUD
from config.config_manager import ConfigManager


class ModelTrainingPipeline:
    """Complete pipeline for training trading models"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = self.setup_logger()
        
        # Initialize components
        self.data_collector = DataCollector(config)
        self.feature_engineer = FeatureEngineer(config)
        self.data_preprocessor = DataPreprocessor(config)
        self.model_manager = ModelManager(config)
        self.model_trainer = ModelTrainer(config)
        
        # Get database connection
        self.db_manager = get_database_manager(config)
        
        # Training parameters
        self.training_config = config.get_training_config()
        
    def setup_logger(self) -> logging.Logger:
        """Setup logging for training pipeline"""
        logger = logging.getLogger('model_training')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        
        return logger
    
    def collect_data(
        self, 
        symbol: str, 
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Collect and prepare data for training"""
        self.logger.info(f"Collecting data for {symbol} {timeframe}")
        
        try:
            # Try to load from database first
            self.logger.info("Loading data from database...")
            df = self.data_collector.load_from_database(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty or len(df) < 100:
                self.logger.warning(f"Insufficient data in database: {len(df)} rows")
                self.logger.info("Downloading data from exchange...")
                
                # Download from exchange
                df = self.data_collector.download_and_save_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
            
            self.logger.info(f"Data collected: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting data: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data"""
        self.logger.info("Engineering features...")
        
        try:
            # Calculate technical indicators
            df_features = self.feature_engineer.engineer_all_features(df)
            
            # Add target variable (future price change)
            lookahead = self.training_config.get('lookahead_periods', 1)
            df_features['target'] = self._create_target_variable(
                df_features['close'].values, 
                lookahead=lookahead
            )
            
            # Remove rows with NaN values
            df_clean = df_features.dropna()
            
            self.logger.info(f"Features engineered: {len(df_clean)} rows, {len(df_clean.columns)} columns")
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {e}")
            raise
    
    def _create_target_variable(self, prices: np.ndarray, lookahead: int = 1) -> np.ndarray:
        """Create target variable for prediction"""
        # Predict percentage change over lookahead periods
        future_prices = np.roll(prices, -lookahead)
        returns = (future_prices - prices) / prices
        
        # Set last lookahead values to NaN (no future data)
        returns[-lookahead:] = np.nan
        
        # Convert to classification if needed
        if self.training_config.get('task_type', 'regression') == 'classification':
            # Classify as up (1), down (-1), or neutral (0)
            threshold = self.training_config.get('classification_threshold', 0.001)
            targets = np.zeros_like(returns)
            targets[returns > threshold] = 1  # Up
            targets[returns < -threshold] = -1  # Down
            return targets
        else:
            return returns
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame,
        sequence_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for sequence models"""
        self.logger.info(f"Preparing training data with sequence length {sequence_length}")
        
        # Select features and target
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'log_returns',
            'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'atr', 'obv'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) < 5:
            self.logger.warning(f"Limited features available: {available_cols}")
            # Use basic columns
            available_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Separate features and target
        X = df[available_cols].values
        y = df['target'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences
        X_sequences, y_sequences = self.data_preprocessor.create_sequences(
            X_scaled, 
            y,
            sequence_length=sequence_length,
            target_offset=0  # Target is already aligned
        )
        
        self.logger.info(f"Created {len(X_sequences)} sequences")
        
        return X_sequences, y_sequences, available_cols, scaler
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets"""
        self.logger.info(f"Splitting data: test={test_size}, validation={val_size}")
        
        # First split: test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Second split: validation set from training data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, shuffle=False
        )
        
        self.logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_transformer_model(
        self, 
        input_shape: Tuple[int, int],
        model_config: Dict[str, Any]
    ) -> TransformerModel:
        """Create Transformer model"""
        self.logger.info("Creating Transformer model")
        
        config = {
            'd_model': model_config.get('d_model', 64),
            'nhead': model_config.get('nhead', 4),
            'num_layers': model_config.get('num_layers', 3),
            'dim_feedforward': model_config.get('dim_feedforward', 256),
            'dropout': model_config.get('dropout', 0.1),
            'num_features': input_shape[1],
            'sequence_length': input_shape[0],
            'output_size': 1 if self.training_config.get('task_type') == 'regression' else 3
        }
        
        return TransformerModel(config)
    
    def create_lstm_attention_model(
        self, 
        input_shape: Tuple[int, int],
        model_config: Dict[str, Any]
    ) -> LSTMAttentionModel:
        """Create LSTM with Attention model"""
        self.logger.info("Creating LSTM with Attention model")
        
        config = {
            'input_size': input_shape[1],
            'hidden_size': model_config.get('hidden_size', 50),
            'num_layers': model_config.get('num_layers', 2),
            'dropout': model_config.get('dropout', 0.2),
            'bidirectional': model_config.get('bidirectional', True),
            'use_attention': model_config.get('use_attention', True),
            'output_size': 1 if self.training_config.get('task_type') == 'regression' else 3
        }
        
        return LSTMAttentionModel(config)
    
    def create_cnn_lstm_model(
        self, 
        input_shape: Tuple[int, int],
        model_config: Dict[str, Any]
    ) -> CNNLSTMModel:
        """Create CNN-LSTM model"""
        self.logger.info("Creating CNN-LSTM model")
        
        config = {
            'input_size': input_shape[1],
            'cnn_filters': model_config.get('cnn_filters', [64, 128]),
            'cnn_kernel_size': model_config.get('cnn_kernel_size', 3),
            'lstm_units': model_config.get('lstm_units', 50),
            'dropout': model_config.get('dropout', 0.2),
            'output_size': 1 if self.training_config.get('task_type') == 'regression' else 3
        }
        
        return CNNLSTMModel(config)
    
    def train_model(
        self,
        model: Any,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        training_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train a single model"""
        self.logger.info(f"Training {model_name} model")
        
        try:
            # Get training parameters
            epochs = training_params.get('epochs', 100)
            batch_size = training_params.get('batch_size', 32)
            learning_rate = training_params.get('learning_rate', 0.001)
            early_stopping_patience = training_params.get('early_stopping_patience', 10)
            
            # Train model
            history = self.model_trainer.train(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
                model_name=model_name
            )
            
            # Get training metrics
            train_metrics = self.model_trainer.evaluate_model(
                model=model,
                X=X_train,
                y=y_train,
                dataset_name='train'
            )
            
            val_metrics = self.model_trainer.evaluate_model(
                model=model,
                X=X_val,
                y=y_val,
                dataset_name='validation'
            )
            
            self.logger.info(f"{model_name} training completed")
            self.logger.info(f"Train metrics: {train_metrics}")
            self.logger.info(f"Validation metrics: {val_metrics}")
            
            return {
                'model': model,
                'history': history,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error training {model_name}: {e}")
            raise
    
    def evaluate_model(
        self,
        model: Any,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler: Optional[StandardScaler] = None
    ) -> Dict[str, Any]:
        """Evaluate model on test set"""
        self.logger.info(f"Evaluating {model_name} on test set")
        
        # Get predictions
        predictions = self.model_trainer.predict(model, X_test)
        
        # Calculate metrics
        test_metrics = self.model_trainer.evaluate_model(
            model=model,
            X=X_test,
            y=y_test,
            dataset_name='test'
        )
        
        # Calculate additional metrics
        if self.training_config.get('task_type') == 'regression':
            # For regression: calculate MAE, MSE, RMSE
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            
            test_metrics.update({
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse)
            })
        else:
            # For classification: calculate accuracy, precision, recall, F1
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            
            # Convert to class predictions
            y_pred_class = np.argmax(predictions, axis=1) if len(predictions.shape) > 1 else (predictions > 0).astype(int)
            y_test_class = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else (y_test > 0).astype(int)
            
            accuracy = accuracy_score(y_test_class, y_pred_class)
            precision = precision_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
            recall = recall_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
            f1 = f1_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
            
            test_metrics.update({
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            })
        
        self.logger.info(f"Test metrics for {model_name}: {test_metrics}")
        
        return {
            'predictions': predictions,
            'metrics': test_metrics
        }
    
    def create_ensemble_model(
        self,
        models: Dict[str, Any],
        X_val: np.ndarray,
        y_val: np.ndarray,
        ensemble_config: Dict[str, Any]
    ) -> EnsembleModel:
        """Create ensemble of trained models"""
        self.logger.info("Creating ensemble model")
        
        # Get model predictions
        model_predictions = {}
        for name, model_info in models.items():
            model = model_info['model']
            predictions = self.model_trainer.predict(model, X_val)
            model_predictions[name] = predictions
        
        # Create ensemble
        ensemble_method = ensemble_config.get('method', 'weighted_average')
        
        if ensemble_method == 'weighted_average':
            # Weight by validation performance
            weights = {}
            for name, model_info in models.items():
                # Use inverse of validation loss as weight
                val_loss = model_info['val_metrics'].get('loss', 1.0)
                weights[name] = 1.0 / (val_loss + 1e-8)
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            
            ensemble = EnsembleModel(
                models=models,
                method='weighted_average',
                weights=weights
            )
            
        elif ensemble_method == 'stacking':
            # Use stacking with meta-learner
            ensemble = EnsembleModel(
                models=models,
                method='stacking',
                meta_learner_type=ensemble_config.get('meta_learner', 'linear')
            )
            
            # Train meta-learner
            ensemble.fit_meta_learner(
                base_predictions=model_predictions,
                y_true=y_val
            )
        
        else:
            # Simple averaging
            ensemble = EnsembleModel(
                models=models,
                method='average'
            )
        
        self.logger.info(f"Ensemble created using {ensemble_method}")
        return ensemble
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        metrics: Dict[str, Any],
        feature_columns: List[str],
        scaler: StandardScaler,
        symbol: str,
        timeframe: str
    ) -> str:
        """Save trained model and metadata"""
        self.logger.info(f"Saving {model_name} model")
        
        # Generate model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_name}_{symbol.replace('/', '_')}_{timeframe}_{timestamp}"
        
        # Save model
        model_path = self.model_manager.save_model(
            model=model,
            model_name=model_name,
            model_id=model_id,
            metrics=metrics,
            feature_columns=feature_columns,
            scaler=scaler,
            symbol=symbol,
            timeframe=timeframe
        )
        
        self.logger.info(f"Model saved to: {model_path}")
        return model_id, model_path
    
    def log_training_to_database(
        self,
        training_id: str,
        model_name: str,
        model_type: str,
        symbol: str,
        timeframe: str,
        hyperparameters: Dict[str, Any],
        training_metrics: Dict[str, Any],
        validation_metrics: Dict[str, Any],
        test_metrics: Dict[str, Any],
        model_path: str,
        feature_columns: List[str]
    ) -> bool:
        """Log training session to database"""
        self.logger.info("Logging training session to database")
        
        try:
            with self.db_manager.session_scope() as session:
                crud = ModelTrainingCRUD(session)
                
                # Create training record
                training_record = {
                    'training_id': training_id,
                    'model_name': model_name,
                    'model_type': model_type,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'training_start': datetime.utcnow() - timedelta(hours=1),  # Approximation
                    'training_end': datetime.utcnow(),
                    'status': 'completed',
                    'hyperparameters': hyperparameters,
                    'training_metrics': training_metrics,
                    'validation_metrics': validation_metrics,
                    'test_metrics': test_metrics,
                    'model_path': model_path,
                    'feature_columns': feature_columns
                }
                
                # Save to database
                crud.create(training_record)
                session.commit()
                
                self.logger.info("Training session logged to database")
                return True
                
        except Exception as e:
            self.logger.error(f"Error logging to database: {e}")
            return False
    
    def plot_training_history(
        self, 
        history: Dict[str, List[float]], 
        model_name: str,
        output_dir: Path
    ) -> None:
        """Plot training history"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            axes[0].plot(history.get('train_loss', []), label='Train Loss')
            axes[0].plot(history.get('val_loss', []), label='Validation Loss')
            axes[0].set_title(f'{model_name} - Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot metric
            if 'train_mae' in history:
                axes[1].plot(history.get('train_mae', []), label='Train MAE')
                axes[1].plot(history.get('val_mae', []), label='Validation MAE')
                axes[1].set_title(f'{model_name} - MAE')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('MAE')
            elif 'train_accuracy' in history:
                axes[1].plot(history.get('train_accuracy', []), label='Train Accuracy')
                axes[1].plot(history.get('val_accuracy', []), label='Validation Accuracy')
                axes[1].set_title(f'{model_name} - Accuracy')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
            
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / f"{model_name}_training_history.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Training plot saved: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not create training plot: {e}")
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_dir: Path
    ) -> None:
        """Plot predictions vs actual values"""
        try:
            plt.figure(figsize=(10, 6))
            
            # For regression tasks
            if len(y_true.shape) == 1:
                # Scatter plot
                plt.scatter(y_true, y_pred, alpha=0.5, s=10)
                plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predictions')
                plt.title(f'{model_name} - Predictions vs Actual')
                
                # Add correlation text
                correlation = np.corrcoef(y_true, y_pred)[0, 1]
                plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=plt.gca().transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / f"{model_name}_predictions.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Predictions plot saved: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not create predictions plot: {e}")
    
    def run_training_pipeline(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        models_to_train: List[str] = None,
        create_ensemble: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run complete training pipeline"""
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING TRAINING PIPELINE")
        self.logger.info(f"Symbol: {symbol}")
        self.logger.info(f"Timeframe: {timeframe}")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info("=" * 60)
        
        # Set default models to train
        if models_to_train is None:
            models_to_train = ['transformer', 'lstm_attention', 'cnn_lstm']
        
        # Create output directory
        if output_dir is None:
            output_dir = project_root / "results" / "model_training"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'models_trained': [],
            'best_model': None,
            'ensemble_model': None
        }
        
        try:
            # Step 1: Collect data
            df_raw = self.collect_data(symbol, timeframe, start_date, end_date)
            
            if df_raw.empty or len(df_raw) < 100:
                raise ValueError(f"Insufficient data: {len(df_raw)} rows")
            
            # Step 2: Engineer features
            df_features = self.engineer_features(df_raw)
            
            if df_features.empty or len(df_features) < 50:
                raise ValueError(f"Insufficient features: {len(df_features)} rows")
            
            # Step 3: Prepare training data
            sequence_length = self.training_config.get('sequence_length', 60)
            X, y, feature_columns, scaler = self.prepare_training_data(
                df_features, sequence_length
            )
            
            # Step 4: Split data
            test_size = self.training_config.get('test_size', 0.2)
            val_size = self.training_config.get('val_size', 0.1)
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
                X, y, test_size=test_size, val_size=val_size
            )
            
            trained_models = {}
            model_results = {}
            
            # Step 5: Train individual models
            for model_type in models_to_train:
                try:
                    self.logger.info(f"\n{'='*40}")
                    self.logger.info(f"Training {model_type} model")
                    self.logger.info(f"{'='*40}")
                    
                    # Create model
                    input_shape = (sequence_length, len(feature_columns))
                    model_config = self.training_config.get('model_configs', {}).get(model_type, {})
                    
                    if model_type == 'transformer':
                        model = self.create_transformer_model(input_shape, model_config)
                    elif model_type == 'lstm_attention':
                        model = self.create_lstm_attention_model(input_shape, model_config)
                    elif model_type == 'cnn_lstm':
                        model = self.create_cnn_lstm_model(input_shape, model_config)
                    else:
                        self.logger.warning(f"Unknown model type: {model_type}")
                        continue
                    
                    # Train model
                    training_params = self.training_config.get('training_params', {})
                    training_result = self.train_model(
                        model, model_type, X_train, y_train, X_val, y_val, training_params
                    )
                    
                    # Evaluate on test set
                    eval_result = self.evaluate_model(
                        training_result['model'], model_type, X_test, y_test, scaler
                    )
                    
                    # Save model
                    all_metrics = {
                        'train': training_result['train_metrics'],
                        'validation': training_result['val_metrics'],
                        'test': eval_result['metrics']
                    }
                    
                    model_id, model_path = self.save_model(
                        model=training_result['model'],
                        model_name=model_type,
                        model_type=model_type,
                        metrics=all_metrics,
                        feature_columns=feature_columns,
                        scaler=scaler,
                        symbol=symbol,
                        timeframe=timeframe
                    )
                    
                    # Log to database
                    self.log_log_training_to_database(
                        training_id=model_id,
                        model_name=model_type,
                        model_type=model_type,
                        symbol=symbol,
                        timeframe=timeframe,
                        hyperparameters=model_config,
                        training_metrics=training_result['train_metrics'],
                        validation_metrics=training_result['val_metrics'],
                        test_metrics=eval_result['metrics'],
                        model_path=model_path,
                        feature_columns=feature_columns
                    )
                    
                    # Store results
                    trained_models[model_type] = training_result['model']
                    model_results[model_type] = {
                        'model_id': model_id,
                        'model_path': model_path,
                        'metrics': all_metrics,
                        'predictions': eval_result['predictions']
                    }
                    
                    results['models_trained'].append(model_type)
                    
                    # Create plots
                    self.plot_training_history(
                        training_result['history'], model_type, output_dir
                    )
                    
                    self.plot_predictions(
                        y_test, eval_result['predictions'], model_type, output_dir
                    )
                    
                    self.logger.info(f"✓ {model_type} model training completed")
                    
                except Exception as e:
                    self.logger.error(f"Failed to train {model_type} model: {e}")
                    continue
            
            # Step 6: Create ensemble model if multiple models trained
            if create_ensemble and len(trained_models) > 1:
                self.logger.info(f"\n{'='*40}")
                self.logger.info("Creating ensemble model")
                self.logger.info(f"{'='*40}")
                
                try:
                    ensemble_config = self.training_config.get('ensemble_config', {})
                    ensemble_model = self.create_ensemble_model(
                        trained_models, X_val, y_val, ensemble_config
                    )
                    
                    # Evaluate ensemble
                    ensemble_predictions = ensemble_model.predict(X_test)
                    ensemble_metrics = self.model_trainer.evaluate_model(
                        ensemble_model, X_test, y_test, dataset_name='ensemble_test'
                    )
                    
                    # Save ensemble
                    ensemble_id, ensemble_path = self.save_model(
                        model=ensemble_model,
                        model_name='ensemble',
                        model_type='ensemble',
                        metrics={'test': ensemble_metrics},
                        feature_columns=feature_columns,
                        scaler=scaler,
                        symbol=symbol,
                        timeframe=timeframe
                    )
                    
                    model_results['ensemble'] = {
                        'model_id': ensemble_id,
                        'model_path': ensemble_path,
                        'metrics': ensemble_metrics,
                        'predictions': ensemble_predictions
                    }
                    
                    results['ensemble_model'] = ensemble_id
                    
                    self.logger.info("✓ Ensemble model created and saved")
                    
                except Exception as e:
                    self.logger.error(f"Failed to create ensemble: {e}")
            
            # Step 7: Determine best model
            if model_results:
                # Find best model based on test loss
                best_model = None
                best_loss = float('inf')
                
                for model_type, result in model_results.items():
                    test_loss = result['metrics']['test'].get('loss', float('inf'))
                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_model = model_type
                
                results['best_model'] = best_model
                
                # Save summary
                summary_path = output_dir / "training_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                self.logger.info(f"\nTraining summary saved: {summary_path}")
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            self.logger.error(f"\nTraining pipeline failed: {e}")
            raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train machine learning models for Bitcoin price prediction"
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading symbol (default: BTC/USDT)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        help='Timeframe for data (default: 1h)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2023-01-01',
        help='Start date for training data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2023-12-31',
        help='End date for training data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['transformer', 'lstm_attention', 'cnn_lstm', 'all'],
        default=['all'],
        help='Models to train (default: all)'
    )
    
    parser.add_argument(
        '--no-ensemble',
        action='store_true',
        help='Skip ensemble model creation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retraining even if model exists'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load configuration
        config = ConfigManager()
        
        if args.config:
            config.load_config(args.config)
        
        # Parse dates
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        # Parse models
        if 'all' in args.models:
            models_to_train = ['transformer', 'lstm_attention', 'cnn_lstm']
        else:
            models_to_train = args.models
        
        # Create training pipeline
        pipeline = ModelTrainingPipeline(config)
        
        # Run training
        results = pipeline.run_training_pipeline(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            models_to_train=models_to_train,
            create_ensemble=not args.no_ensemble,
            output_dir=Path(args.output_dir) if args.output_dir else None
        )
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE - SUMMARY")
        print("="*60)
        print(f"Symbol: {results['symbol']}")
        print(f"Timeframe: {results['timeframe']}")
        print(f"Models trained: {', '.join(results['models_trained'])}")
        
        if results.get('best_model'):
            print(f"Best model: {results['best_model']}")
        
        if results.get('ensemble_model'):
            print(f"Ensemble model: {results['ensemble_model']}")
        
        print("\nNext steps:")
        print("1. Use the trained models for prediction:")
        print("   python scripts/run_predictions.py --model-id <model_id>")
        print("2. Run backtesting with the models:")
        print("   python scripts/run_backtest.py --model-id <model_id>")
        print("3. Deploy model for live trading:")
        print("   Update main.py with your model ID")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTraining failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())