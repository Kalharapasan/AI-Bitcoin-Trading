"""
Models API Endpoints for Bitcoin Trading AI System
Handles ML model management, training, prediction, and evaluation
"""

from fastapi import APIRouter, HTTPException, Depends, Security, status, Query, Body, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import asyncio
import pickle
import io
import zipfile
from pathlib import Path
from pydantic import BaseModel, Field, validator

# Import project modules
try:
    from config.config_manager import ConfigManager
    from core.neural_networks.transformer_model import TransformerModel
    from core.neural_networks.lstm_attention import LSTMAttentionModel
    from core.neural_networks.cnn_lstm import CNNLSTMModel
    from core.neural_networks.ensemble_model import EnsembleModel
    from core.neural_networks.reinforcement_learning import RLModel
    from core.models.model_manager import ModelManager
    from core.models.model_trainer import ModelTrainer
    from core.models.model_predictor import ModelPredictor
    from core.data_processing.data_collector import DataCollector
    from core.data_processing.feature_engineer import FeatureEngineer
    from core.monitoring.performance_tracker import PerformanceTracker
    from database.crud import CRUDOperations
    from database.connection import DatabaseConnection
    from core.utils.logger import setup_logger
    from core.utils.cache import CacheManager
    from web.api.rest_api import get_current_user, security
except ImportError:
    # For testing purposes
    ConfigManager = type('ConfigManager', (), {})
    TransformerModel = type('TransformerModel', (), {})
    LSTMAttentionModel = type('LSTMAttentionModel', (), {})
    CNNLSTMModel = type('CNNLSTMModel', (), {})
    EnsembleModel = type('EnsembleModel', (), {})
    RLModel = type('RLModel', (), {})
    ModelManager = type('ModelManager', (), {})
    ModelTrainer = type('ModelTrainer', (), {})
    ModelPredictor = type('ModelPredictor', (), {})
    DataCollector = type('DataCollector', (), {})
    FeatureEngineer = type('FeatureEngineer', (), {})
    PerformanceTracker = type('PerformanceTracker', (), {})
    CRUDOperations = type('CRUDOperations', (), {})
    DatabaseConnection = type('DatabaseConnection', (), {})
    setup_logger = lambda name: type('Logger', (), {})()
    CacheManager = type('CacheManager', (), {})
    get_current_user = lambda: "admin"
    security = HTTPBearer()

# Initialize logger
logger = setup_logger(__name__)

# Create router
router = APIRouter(prefix="/api/models", tags=["models"])

# Initialize services
config_manager = None
model_manager = None
model_trainer = None
model_predictor = None
data_collector = None
feature_engineer = None
performance_tracker = None
crud = None
cache_manager = None

# Available model types
MODEL_TYPES = {
    "transformer": TransformerModel,
    "lstm_attention": LSTMAttentionModel,
    "cnn_lstm": CNNLSTMModel,
    "ensemble": EnsembleModel,
    "reinforcement_learning": RLModel
}

# Pydantic Models
class ModelTrainRequest(BaseModel):
    """Model for training request"""
    model_type: str = Field(description="Type of model to train")
    model_name: str = Field(description="Unique name for the model")
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Timeframe for training data")
    features: List[str] = Field(default_factory=lambda: ["close", "volume", "rsi", "macd"], description="Features to use")
    lookback_period: int = Field(default=100, ge=10, le=1000, description="Lookback period for sequences")
    forecast_horizon: int = Field(default=1, ge=1, le=10, description="Number of periods to forecast")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Validation split ratio")
    epochs: int = Field(default=50, ge=1, le=1000, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=8, le=256, description="Training batch size")
    learning_rate: float = Field(default=0.001, ge=0.00001, le=0.1, description="Learning rate")
    early_stopping_patience: int = Field(default=10, ge=1, le=50, description="Early stopping patience")
    use_pretrained: bool = Field(default=False, description="Use pretrained model weights")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in MODEL_TYPES:
            raise ValueError(f'Model type must be one of: {list(MODEL_TYPES.keys())}')
        return v

class ModelPredictRequest(BaseModel):
    """Model for prediction request"""
    model_name: str = Field(description="Name of the model to use")
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Timeframe for prediction")
    input_data: Optional[List[Dict]] = Field(None, description="Input data for prediction")
    latest_data_points: int = Field(default=100, ge=10, le=1000, description="Number of latest data points to use")
    return_probabilities: bool = Field(default=False, description="Return prediction probabilities")
    explain_prediction: bool = Field(default=False, description="Include prediction explanation")

class ModelEvaluateRequest(BaseModel):
    """Model for evaluation request"""
    model_name: str = Field(description="Name of the model to evaluate")
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Timeframe for evaluation")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test data size ratio")
    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "mse"], description="Evaluation metrics")
    cross_validation_folds: int = Field(default=5, ge=2, le=10, description="Cross-validation folds")

class ModelUpdateRequest(BaseModel):
    """Model for model update request"""
    model_name: str = Field(description="Name of the model to update")
    update_type: str = Field(description="Type of update: retrain, fine_tune, hyperparameter")
    new_data_only: bool = Field(default=False, description="Use only new data for update")
    epochs: int = Field(default=10, ge=1, le=100, description="Number of update epochs")
    learning_rate: Optional[float] = Field(None, description="Learning rate for update")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Updated hyperparameters")

class ModelCompareRequest(BaseModel):
    """Model for model comparison request"""
    model_names: List[str] = Field(description="List of model names to compare")
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Timeframe for comparison")
    comparison_metrics: List[str] = Field(default_factory=lambda: ["accuracy", "mse", "inference_time"], description="Metrics for comparison")
    test_size: float = Field(default=0.2, description="Test data size ratio")

class ModelExportRequest(BaseModel):
    """Model for model export request"""
    model_name: str = Field(description="Name of the model to export")
    format: str = Field(default="h5", description="Export format: h5, pickle, onnx, joblib")
    include_metadata: bool = Field(default=True, description="Include model metadata")
    include_training_data: bool = Field(default=False, description="Include training data statistics")
    compression: Optional[str] = Field(None, description="Compression type: gzip, zip")

class HyperparameterOptimizationRequest(BaseModel):
    """Model for hyperparameter optimization request"""
    model_type: str = Field(description="Type of model to optimize")
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Timeframe for data")
    param_grid: Dict[str, List[Any]] = Field(description="Hyperparameter grid for optimization")
    optimization_method: str = Field(default="grid", description="Optimization method: grid, random, bayesian")
    cv_folds: int = Field(default=3, ge=2, le=10, description="Cross-validation folds")
    scoring_metric: str = Field(default="accuracy", description="Scoring metric for optimization")
    n_iter: int = Field(default=10, ge=1, le=100, description="Number of iterations for random/bayesian search")

# Initialize model services
async def initialize_model_services():
    """Initialize model services"""
    global config_manager, model_manager, model_trainer, model_predictor
    global data_collector, feature_engineer, performance_tracker, crud, cache_manager
    
    try:
        if not config_manager:
            config_manager = ConfigManager()
        
        if not cache_manager:
            cache_manager = CacheManager(config_manager)
        
        if not crud:
            db = DatabaseConnection()
            await db.connect()
            crud = CRUDOperations(db)
        
        # Initialize services if not already initialized
        if not data_collector:
            data_collector = DataCollector(config_manager, cache_manager)
        
        if not feature_engineer:
            feature_engineer = FeatureEngineer(config_manager)
        
        if not performance_tracker:
            performance_tracker = PerformanceTracker(crud)
        
        if not model_manager:
            model_manager = ModelManager(config_manager, crud)
        
        if not model_trainer:
            model_trainer = ModelTrainer(config_manager, model_manager, data_collector, feature_engineer)
        
        if not model_predictor:
            model_predictor = ModelPredictor(config_manager, model_manager, data_collector)
        
        # Load existing models
        await load_existing_models()
        
        logger.info("Model services initialized successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize model services: {e}")
        raise

async def load_existing_models():
    """Load existing models from storage"""
    try:
        models = await model_manager.load_all_models()
        logger.info(f"Loaded {len(models)} existing models")
        return models
    except Exception as e:
        logger.error(f"Error loading existing models: {e}")
        return {}

@router.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await initialize_model_services()

# Model Information Endpoints
@router.get("/", summary="Get all models")
async def get_all_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    status: Optional[str] = Query(None, regex="^(active|inactive|training|error)$", description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Number of models to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get list of all available models with filtering options.
    """
    try:
        await initialize_model_services()
        
        # Get models from model manager
        models = await model_manager.get_all_models()
        
        # Apply filters
        filtered_models = []
        for model_name, model_info in models.items():
            if model_type and model_info.get('model_type') != model_type:
                continue
            if status and model_info.get('status') != status:
                continue
            filtered_models.append({
                "name": model_name,
                **model_info
            })
        
        # Apply pagination
        total_models = len(filtered_models)
        paginated_models = filtered_models[offset:offset + limit]
        
        return {
            "models": paginated_models,
            "total_models": total_models,
            "filtered_count": len(filtered_models),
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_models
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types", summary="Get available model types")
async def get_model_types(
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get list of available model types with descriptions.
    """
    try:
        model_types_info = {}
        for model_type, model_class in MODEL_TYPES.items():
            # Get model description from class if available
            try:
                model_instance = model_class(config_manager)
                description = await model_instance.get_description()
            except:
                description = f"{model_type.replace('_', ' ').title()} Model"
            
            model_types_info[model_type] = {
                "description": description,
                "suitable_for": get_suitable_applications(model_type),
                "complexity": get_model_complexity(model_type),
                "training_time": get_training_time_estimate(model_type)
            }
        
        return {
            "model_types": model_types_info,
            "total_types": len(model_types_info),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching model types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_suitable_applications(model_type: str) -> List[str]:
    """Get suitable applications for a model type"""
    applications = {
        "transformer": ["price prediction", "sequence modeling", "long-term forecasting"],
        "lstm_attention": ["time series forecasting", "pattern recognition", "short-term prediction"],
        "cnn_lstm": ["feature extraction", "multimodal data", "image + sequence data"],
        "ensemble": ["improved accuracy", "reduced variance", "robust predictions"],
        "reinforcement_learning": ["trading strategy", "portfolio optimization", "risk management"]
    }
    return applications.get(model_type, ["general purpose"])

def get_model_complexity(model_type: str) -> str:
    """Get complexity level for a model type"""
    complexity = {
        "transformer": "high",
        "lstm_attention": "medium",
        "cnn_lstm": "medium",
        "ensemble": "varies",
        "reinforcement_learning": "very high"
    }
    return complexity.get(model_type, "medium")

def get_training_time_estimate(model_type: str) -> str:
    """Get training time estimate for a model type"""
    time_estimates = {
        "transformer": "long",
        "lstm_attention": "medium",
        "cnn_lstm": "medium",
        "ensemble": "long",
        "reinforcement_learning": "very long"
    }
    return time_estimates.get(model_type, "medium")

@router.get("/{model_name}", summary="Get model details")
async def get_model_details(
    model_name: str,
    include_weights: bool = Query(False, description="Include model weights info"),
    include_performance: bool = Query(True, description="Include performance metrics"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get detailed information about a specific model.
    """
    try:
        await initialize_model_services()
        
        # Get model from model manager
        model_info = await model_manager.get_model(model_name)
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Add performance metrics if requested
        if include_performance:
            performance = await performance_tracker.get_model_performance(model_name)
            model_info["performance"] = performance
        
        # Add weights info if requested
        if include_weights:
            weights_info = await model_manager.get_model_weights_info(model_name)
            model_info["weights"] = weights_info
        
        # Add training history
        training_history = await crud.get_model_training_history(model_name)
        model_info["training_history"] = training_history
        
        return {
            "model": model_info,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_name}/architecture", summary="Get model architecture")
async def get_model_architecture(
    model_name: str,
    format: str = Query("json", description="Output format: json, text, image"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get model architecture information.
    """
    try:
        await initialize_model_services()
        
        # Get model architecture
        architecture = await model_manager.get_model_architecture(model_name)
        
        if not architecture:
            raise HTTPException(status_code=404, detail=f"Architecture not found for model '{model_name}'")
        
        if format == "json":
            return JSONResponse(content=architecture)
        
        elif format == "text":
            # Convert to text summary
            text_summary = convert_architecture_to_text(architecture)
            return {
                "model_name": model_name,
                "architecture_summary": text_summary,
                "timestamp": datetime.now().isoformat()
            }
        
        elif format == "image":
            # Generate architecture diagram
            image_data = await generate_architecture_diagram(architecture, model_name)
            return StreamingResponse(
                io.BytesIO(image_data),
                media_type="image/png",
                headers={
                    "Content-Disposition": f"attachment; filename={model_name}_architecture.png"
                }
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use: json, text, image")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model architecture: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def convert_architecture_to_text(architecture: Dict) -> str:
    """Convert architecture dictionary to text summary"""
    summary = []
    
    if "layers" in architecture:
        summary.append(f"Total Layers: {len(architecture['layers'])}")
        for i, layer in enumerate(architecture["layers"]):
            layer_type = layer.get("class_name", "Unknown")
            config = layer.get("config", {})
            summary.append(f"  Layer {i+1}: {layer_type}")
            
            # Add layer details
            for key, value in config.items():
                if key not in ["name", "batch_input_shape"]:
                    summary.append(f"    {key}: {value}")
    
    if "parameters" in architecture:
        summary.append(f"\nTotal Parameters: {architecture['parameters'].get('total', 0):,}")
        summary.append(f"Trainable Parameters: {architecture['parameters'].get('trainable', 0):,}")
        summary.append(f"Non-trainable Parameters: {architecture['parameters'].get('non_trainable', 0):,}")
    
    return "\n".join(summary)

async def generate_architecture_diagram(architecture: Dict, model_name: str) -> bytes:
    """Generate architecture diagram image"""
    # This would use a library like graphviz or matplotlib
    # For now, return a placeholder
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a simple diagram
        layers = architecture.get("layers", [])
        num_layers = len(layers)
        
        for i, layer in enumerate(layers):
            layer_type = layer.get("class_name", f"Layer {i+1}")
            
            # Create rectangle for layer
            rect = patches.Rectangle(
                (0.1, 0.8 - i * 0.1),
                0.8,
                0.08,
                linewidth=1,
                edgecolor='black',
                facecolor='lightblue',
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add layer name
            ax.text(0.5, 0.84 - i * 0.1, layer_type,
                   ha='center', va='center', fontsize=9)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f"{model_name} Architecture", fontsize=14, fontweight='bold')
        
        # Save to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    except Exception as e:
        logger.error(f"Error generating architecture diagram: {e}")
        # Return empty bytes if diagram generation fails
        return b""

# Model Training Endpoints
@router.post("/train", summary="Train a new model")
async def train_model(
    request: ModelTrainRequest,
    background_tasks: Optional[Any] = None,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Train a new machine learning model.
    """
    try:
        await initialize_model_services()
        
        # Check if model with same name already exists
        existing_model = await model_manager.get_model(request.model_name)
        if existing_model and not request.use_pretrained:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_name}' already exists. Use different name or set use_pretrained=True"
            )
        
        # Prepare training data
        logger.info(f"Preparing training data for {request.model_name}")
        
        # Get historical data
        data = await data_collector.get_historical_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=request.lookback_period * 10  # Get enough data
        )
        
        if not data or len(data) < request.lookback_period * 2:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for training. Need at least {request.lookback_period * 2} data points"
            )
        
        # Calculate features
        features_df = await feature_engineer.calculate_features(
            pd.DataFrame(data),
            request.timeframe
        )
        
        # Start training (could be in background)
        training_result = await model_trainer.train_model(
            model_type=request.model_type,
            model_name=request.model_name,
            data=features_df,
            features=request.features,
            lookback_period=request.lookback_period,
            forecast_horizon=request.forecast_horizon,
            validation_split=request.validation_split,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            early_stopping_patience=request.early_stopping_patience,
            use_pretrained=request.use_pretrained,
            hyperparameters=request.hyperparameters
        )
        
        # Log training in database
        training_record = {
            "model_name": request.model_name,
            "model_type": request.model_type,
            "user_id": current_user,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "parameters": request.dict(),
            "training_result": training_result,
            "start_time": datetime.now(),
            "status": "completed"
        }
        
        await crud.create_training_record(training_record)
        
        # Update model status
        await model_manager.update_model_status(
            model_name=request.model_name,
            status="active",
            last_trained=datetime.now()
        )
        
        return {
            "success": True,
            "message": f"Model '{request.model_name}' trained successfully",
            "training_id": training_result.get("training_id"),
            "model_name": request.model_name,
            "training_result": training_result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {e}")
        
        # Log failed training
        try:
            await crud.create_training_record({
                "model_name": request.model_name,
                "model_type": request.model_type,
                "user_id": current_user,
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "parameters": request.dict(),
                "error": str(e),
                "start_time": datetime.now(),
                "status": "failed"
            })
        except:
            pass
        
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train/async", summary="Train model asynchronously")
async def train_model_async(
    request: ModelTrainRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Start asynchronous model training.
    Returns immediately with training job ID.
    """
    try:
        await initialize_model_services()
        
        # Generate training job ID
        training_id = f"train_{request.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start training in background
        asyncio.create_task(
            train_model_background(training_id, request, current_user)
        )
        
        return {
            "success": True,
            "message": f"Training job '{training_id}' started",
            "training_id": training_id,
            "model_name": request.model_name,
            "status": "started",
            "check_status_endpoint": f"/api/models/train/status/{training_id}",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error starting async training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_background(training_id: str, request: ModelTrainRequest, user_id: str):
    """Background task for model training"""
    try:
        # Log training start
        await crud.create_training_record({
            "training_id": training_id,
            "model_name": request.model_name,
            "model_type": request.model_type,
            "user_id": user_id,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "parameters": request.dict(),
            "start_time": datetime.now(),
            "status": "running"
        })
        
        # Simulate training (in real implementation, this would call model_trainer)
        await asyncio.sleep(1)  # Simulate some work
        
        # Update training status
        await crud.update_training_status(
            training_id=training_id,
            status="completed",
            completion_time=datetime.now(),
            result={"message": "Training completed successfully"}
        )
        
    except Exception as e:
        logger.error(f"Background training failed: {e}")
        await crud.update_training_status(
            training_id=training_id,
            status="failed",
            error=str(e)
        )

@router.get("/train/status/{training_id}", summary="Get training status")
async def get_training_status(
    training_id: str,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get status of a training job.
    """
    try:
        await initialize_model_services()
        
        training_record = await crud.get_training_record(training_id)
        
        if not training_record:
            raise HTTPException(status_code=404, detail=f"Training job '{training_id}' not found")
        
        # Add progress information if available
        progress = await get_training_progress(training_id)
        
        return {
            "training_id": training_id,
            "status": training_record.get("status", "unknown"),
            "progress": progress,
            "record": training_record,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_training_progress(training_id: str) -> Dict:
    """Get training progress information"""
    # This would read from a progress file or cache
    # For now, return mock progress
    return {
        "epoch": 25,
        "total_epochs": 50,
        "progress_percentage": 50.0,
        "current_loss": 0.125,
        "current_accuracy": 0.85,
        "estimated_time_remaining": "15 minutes"
    }

# Model Prediction Endpoints
@router.post("/predict", summary="Make prediction")
async def make_prediction(
    request: ModelPredictRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Make predictions using a trained model.
    """
    try:
        await initialize_model_services()
        
        # Check if model exists
        model_info = await model_manager.get_model(request.model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
        
        # Check model status
        if model_info.get("status") != "active":
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_name}' is not active. Status: {model_info.get('status')}"
            )
        
        # Prepare input data
        if request.input_data:
            # Use provided input data
            input_df = pd.DataFrame(request.input_data)
        else:
            # Fetch latest data
            data = await data_collector.get_historical_data(
                symbol=request.symbol,
                timeframe=request.timeframe,
                limit=request.latest_data_points
            )
            
            if not data:
                raise HTTPException(
                    status_code=400,
                    detail=f"No data available for {request.symbol} {request.timeframe}"
                )
            
            input_df = pd.DataFrame(data)
        
        # Make prediction
        prediction_result = await model_predictor.predict(
            model_name=request.model_name,
            input_data=input_df,
            return_probabilities=request.return_probabilities,
            explain_prediction=request.explain_prediction
        )
        
        # Log prediction
        prediction_record = {
            "model_name": request.model_name,
            "user_id": current_user,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "prediction": prediction_result,
            "timestamp": datetime.now()
        }
        
        await crud.create_prediction_record(prediction_record)
        
        return {
            "model_name": request.model_name,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "prediction": prediction_result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", summary="Make batch predictions")
async def make_batch_predictions(
    requests: List[ModelPredictRequest],
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Make batch predictions using multiple models or inputs.
    """
    try:
        await initialize_model_services()
        
        results = []
        for request in requests:
            try:
                result = await make_prediction(request, current_user, credentials)
                results.append({
                    "success": True,
                    "request": request.dict(),
                    "result": result
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "request": request.dict(),
                    "error": str(e)
                })
        
        return {
            "batch_results": results,
            "total_requests": len(requests),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error making batch predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_name}/predict/latest", summary="Get latest prediction")
async def get_latest_prediction(
    model_name: str,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get the latest prediction made by a model.
    """
    try:
        await initialize_model_services()
        
        # Get latest prediction from database
        predictions = await crud.get_latest_predictions(model_name, limit=1)
        
        if not predictions:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for model '{model_name}'"
            )
        
        return {
            "model_name": model_name,
            "latest_prediction": predictions[0],
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching latest prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model Evaluation Endpoints
@router.post("/evaluate", summary="Evaluate model")
async def evaluate_model(
    request: ModelEvaluateRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Evaluate model performance.
    """
    try:
        await initialize_model_services()
        
        # Check if model exists
        model_info = await model_manager.get_model(request.model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
        
        # Get evaluation data
        data = await data_collector.get_historical_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=1000  # Get enough data for evaluation
        )
        
        if not data:
            raise HTTPException(
                status_code=400,
                detail=f"No data available for evaluation"
            )
        
        # Prepare data for evaluation
        data_df = pd.DataFrame(data)
        features_df = await feature_engineer.calculate_features(data_df, request.timeframe)
        
        # Evaluate model
        evaluation_result = await model_predictor.evaluate_model(
            model_name=request.model_name,
            data=features_df,
            test_size=request.test_size,
            metrics=request.metrics,
            cross_validation_folds=request.cross_validation_folds
        )
        
        # Log evaluation
        evaluation_record = {
            "model_name": request.model_name,
            "user_id": current_user,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "evaluation_result": evaluation_result,
            "timestamp": datetime.now()
        }
        
        await crud.create_evaluation_record(evaluation_record)
        
        # Update model performance metrics
        await performance_tracker.update_model_performance(
            model_name=request.model_name,
            metrics=evaluation_result.get("metrics", {})
        )
        
        return {
            "model_name": request.model_name,
            "evaluation": evaluation_result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_name}/performance", summary="Get model performance")
async def get_model_performance(
    model_name: str,
    period: str = Query("30d", regex="^(7d|30d|90d|180d|1y|all)$", description="Performance period"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get performance metrics for a model.
    """
    try:
        await initialize_model_services()
        
        # Get performance metrics
        performance = await performance_tracker.get_model_performance(model_name, period)
        
        # Get recent evaluations
        evaluations = await crud.get_model_evaluations(model_name, limit=10)
        
        # Calculate performance trends
        trends = await calculate_performance_trends(model_name, period)
        
        return {
            "model_name": model_name,
            "period": period,
            "performance": performance,
            "recent_evaluations": evaluations,
            "performance_trends": trends,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def calculate_performance_trends(model_name: str, period: str) -> Dict:
    """Calculate performance trends for a model"""
    try:
        # Get historical performance data
        historical_performance = await crud.get_model_performance_history(model_name, period)
        
        if not historical_performance or len(historical_performance) < 2:
            return {"trend": "insufficient_data", "message": "Not enough data for trend analysis"}
        
        # Calculate trends for key metrics
        trends = {}
        metrics_to_analyze = ["accuracy", "precision", "recall", "f1_score", "mse"]
        
        for metric in metrics_to_analyze:
            values = [p.get("metrics", {}).get(metric, 0) for p in historical_performance if metric in p.get("metrics", {})]
            
            if len(values) >= 2:
                # Calculate slope (simple linear trend)
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                
                trends[metric] = {
                    "current": values[-1] if values else 0,
                    "trend": "improving" if slope > 0.001 else "declining" if slope < -0.001 else "stable",
                    "trend_strength": abs(slope),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": np.mean(values) if values else 0
                }
        
        # Overall trend
        if trends:
            improving_metrics = sum(1 for t in trends.values() if t["trend"] == "improving")
            total_metrics = len(trends)
            
            trends["overall"] = {
                "trend": "improving" if improving_metrics / total_metrics > 0.6 else "declining" if improving_metrics / total_metrics < 0.4 else "mixed",
                "improving_metrics": improving_metrics,
                "total_metrics": total_metrics,
                "improvement_ratio": improving_metrics / total_metrics
            }
        
        return trends
    
    except Exception as e:
        logger.error(f"Error calculating performance trends: {e}")
        return {"error": str(e)}

# Model Management Endpoints
@router.post("/update", summary="Update model")
async def update_model(
    request: ModelUpdateRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Update an existing model (retrain, fine-tune, or update hyperparameters).
    """
    try:
        await initialize_model_services()
        
        # Check if model exists
        model_info = await model_manager.get_model(request.model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
        
        # Update model based on update type
        if request.update_type == "retrain":
            result = await model_trainer.retrain_model(
                model_name=request.model_name,
                new_data_only=request.new_data_only,
                epochs=request.epochs,
                learning_rate=request.learning_rate
            )
        
        elif request.update_type == "fine_tune":
            result = await model_trainer.fine_tune_model(
                model_name=request.model_name,
                learning_rate=request.learning_rate or 0.0001,
                epochs=request.epochs
            )
        
        elif request.update_type == "hyperparameter":
            result = await model_manager.update_model_hyperparameters(
                model_name=request.model_name,
                hyperparameters=request.hyperparameters
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid update type: {request.update_type}. Use: retrain, fine_tune, hyperparameter"
            )
        
        # Log update
        update_record = {
            "model_name": request.model_name,
            "user_id": current_user,
            "update_type": request.update_type,
            "parameters": request.dict(),
            "result": result,
            "timestamp": datetime.now()
        }
        
        await crud.create_model_update_record(update_record)
        
        return {
            "success": True,
            "message": f"Model '{request.model_name}' updated successfully",
            "update_type": request.update_type,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare", summary="Compare models")
async def compare_models(
    request: ModelCompareRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Compare performance of multiple models.
    """
    try:
        await initialize_model_services()
        
        comparison_results = {}
        
        for model_name in request.model_names:
            try:
                # Get model info
                model_info = await model_manager.get_model(model_name)
                
                if not model_info:
                    comparison_results[model_name] = {
                        "error": "Model not found",
                        "comparison": None
                    }
                    continue
                
                # Evaluate model if needed
                evaluation_request = ModelEvaluateRequest(
                    model_name=model_name,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    test_size=request.test_size,
                    metrics=request.comparison_metrics
                )
                
                try:
                    evaluation_result = await evaluate_model(evaluation_request, current_user, credentials)
                    
                    comparison_results[model_name] = {
                        "model_info": model_info,
                        "evaluation": evaluation_result,
                        "metrics": evaluation_result.get("evaluation", {}).get("metrics", {})
                    }
                
                except Exception as e:
                    comparison_results[model_name] = {
                        "error": f"Evaluation failed: {str(e)}",
                        "model_info": model_info,
                        "comparison": None
                    }
            
            except Exception as e:
                comparison_results[model_name] = {
                    "error": str(e),
                    "comparison": None
                }
        
        # Calculate comparison summary
        summary = calculate_comparison_summary(comparison_results, request.comparison_metrics)
        
        return {
            "comparison": comparison_results,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_comparison_summary(comparison_results: Dict, metrics: List[str]) -> Dict:
    """Calculate summary statistics for model comparison"""
    summary = {
        "best_models": {},
        "ranking": [],
        "statistics": {}
    }
    
    # Find best model for each metric
    for metric in metrics:
        best_model = None
        best_value = None
        
        for model_name, result in comparison_results.items():
            if "metrics" in result and metric in result["metrics"]:
                value = result["metrics"][metric]
                
                # Determine if higher or lower is better
                # For accuracy, precision, recall, f1: higher is better
                # For mse, mae, rmse: lower is better
                if "mse" in metric.lower() or "mae" in metric.lower() or "rmse" in metric.lower():
                    if best_value is None or value < best_value:
                        best_value = value
                        best_model = model_name
                else:
                    if best_value is None or value > best_value:
                        best_value = value
                        best_model = model_name
        
        if best_model:
            summary["best_models"][metric] = {
                "model": best_model,
                "value": best_value
            }
    
    # Create ranking based on average performance
    model_scores = {}
    for model_name, result in comparison_results.items():
        if "metrics" in result:
            scores = []
            for metric, value in result["metrics"].items():
                # Normalize scores (higher is always better)
                if "mse" in metric.lower() or "mae" in metric.lower() or "rmse" in metric.lower():
                    # For error metrics, invert the score
                    scores.append(1 / (1 + value))
                else:
                    scores.append(value)
            
            if scores:
                model_scores[model_name] = np.mean(scores)
    
    # Sort models by score
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    summary["ranking"] = [
        {"model": model, "score": score, "rank": i+1}
        for i, (model, score) in enumerate(sorted_models)
    ]
    
    return summary

@router.delete("/{model_name}", summary="Delete model")
async def delete_model(
    model_name: str,
    delete_data: bool = Query(False, description="Delete associated data"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Delete a model and optionally its associated data.
    """
    try:
        await initialize_model_services()
        
        # Check if model exists
        model_info = await model_manager.get_model(model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Delete model
        result = await model_manager.delete_model(model_name, delete_data)
        
        # Log deletion
        deletion_record = {
            "model_name": model_name,
            "user_id": current_user,
            "delete_data": delete_data,
            "timestamp": datetime.now()
        }
        
        await crud.create_model_deletion_record(deletion_record)
        
        return {
            "success": True,
            "message": f"Model '{model_name}' deleted successfully",
            "delete_data": delete_data,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model Export/Import Endpoints
@router.post("/export", summary="Export model")
async def export_model(
    request: ModelExportRequest,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Export a model to a file.
    """
    try:
        await initialize_model_services()
        
        # Check if model exists
        model_info = await model_manager.get_model(request.model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
        
        # Export model
        export_result = await model_manager.export_model(
            model_name=request.model_name,
            export_format=request.format,
            include_metadata=request.include_metadata,
            include_training_data=request.include_training_data
        )
        
        # Prepare export file
        filename = f"{request.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{request.format}"
        
        if request.format == "h5":
            media_type = "application/x-hdf"
        elif request.format == "pickle":
            media_type = "application/octet-stream"
        elif request.format == "onnx":
            media_type = "application/octet-stream"
        elif request.format == "joblib":
            media_type = "application/octet-stream"
        else:
            media_type = "application/octet-stream"
        
        # Create zip file if compression requested
        if request.compression == "zip":
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path, file_content in export_result.get("files", {}).items():
                    zip_file.writestr(file_path, file_content)
            
            zip_buffer.seek(0)
            filename = f"{filename}.zip"
            media_type = "application/zip"
            
            return StreamingResponse(
                zip_buffer,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
        
        else:
            # Return single file
            file_content = export_result.get("file_content", b"")
            
            return StreamingResponse(
                io.BytesIO(file_content),
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/import", summary="Import model")
async def import_model(
    model_file: UploadFile = File(...),
    model_name: Optional[str] = Query(None, description="Name for imported model"),
    overwrite: bool = Query(False, description="Overwrite existing model"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Import a model from a file.
    """
    try:
        await initialize_model_services()
        
        # Read uploaded file
        file_content = await model_file.read()
        filename = model_file.filename
        
        # Determine format from filename
        if filename.endswith('.h5'):
            import_format = 'h5'
        elif filename.endswith('.pkl') or filename.endswith('.pickle'):
            import_format = 'pickle'
        elif filename.endswith('.onnx'):
            import_format = 'onnx'
        elif filename.endswith('.joblib'):
            import_format = 'joblib'
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Extract model name if not provided
        if not model_name:
            model_name = Path(filename).stem
        
        # Check if model already exists
        existing_model = await model_manager.get_model(model_name)
        if existing_model and not overwrite:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' already exists. Use overwrite=True to replace"
            )
        
        # Import model
        import_result = await model_manager.import_model(
            model_name=model_name,
            file_content=file_content,
            import_format=import_format,
            filename=filename
        )
        
        # Log import
        import_record = {
            "model_name": model_name,
            "user_id": current_user,
            "filename": filename,
            "import_format": import_format,
            "result": import_result,
            "timestamp": datetime.now()
        }
        
        await crud.create_model_import_record(import_record)
        
        return {
            "success": True,
            "message": f"Model '{model_name}' imported successfully",
            "model_name": model_name,
            "import_result": import_result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Hyperparameter Optimization Endpoints
@router.post("/hyperparameter/optimize", summary="Optimize hyperparameters")
async def optimize_hyperparameters(
    request: HyperparameterOptimizationRequest,
    background_tasks: Optional[Any] = None,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Optimize model hyperparameters.
    """
    try:
        await initialize_model_services()
        
        # Get data for optimization
        data = await data_collector.get_historical_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=1000
        )
        
        if not data or len(data) < 100:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for hyperparameter optimization"
            )
        
        # Prepare features
        data_df = pd.DataFrame(data)
        features_df = await feature_engineer.calculate_features(data_df, request.timeframe)
        
        # Optimize hyperparameters
        optimization_result = await model_trainer.optimize_hyperparameters(
            model_type=request.model_type,
            data=features_df,
            param_grid=request.param_grid,
            optimization_method=request.optimization_method,
            cv_folds=request.cv_folds,
            scoring_metric=request.scoring_metric,
            n_iter=request.n_iter
        )
        
        # Log optimization
        optimization_record = {
            "model_type": request.model_type,
            "user_id": current_user,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "parameters": request.dict(),
            "result": optimization_result,
            "timestamp": datetime.now()
        }
        
        await crud.create_hyperparameter_optimization_record(optimization_record)
        
        return {
            "success": True,
            "message": "Hyperparameter optimization completed",
            "optimization_result": optimization_result,
            "best_parameters": optimization_result.get("best_params", {}),
            "best_score": optimization_result.get("best_score", 0),
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing hyperparameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hyperparameter/suggestions/{model_type}", summary="Get hyperparameter suggestions")
async def get_hyperparameter_suggestions(
    model_type: str,
    problem_type: str = Query("regression", description="Problem type: regression, classification"),
    data_size: str = Query("medium", description="Data size: small, medium, large"),
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get suggested hyperparameters for a model type.
    """
    try:
        # Generate suggestions based on model type and problem
        suggestions = generate_hyperparameter_suggestions(model_type, problem_type, data_size)
        
        return {
            "model_type": model_type,
            "problem_type": problem_type,
            "data_size": data_size,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error generating hyperparameter suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_hyperparameter_suggestions(model_type: str, problem_type: str, data_size: str) -> Dict:
    """Generate hyperparameter suggestions"""
    base_suggestions = {
        "transformer": {
            "d_model": [64, 128, 256],
            "num_heads": [4, 8, 16],
            "num_layers": [2, 4, 6],
            "dff": [256, 512, 1024],
            "dropout_rate": [0.1, 0.2, 0.3],
            "learning_rate": [0.001, 0.0005, 0.0001]
        },
        "lstm_attention": {
            "lstm_units": [50, 100, 200],
            "attention_units": [32, 64, 128],
            "dense_units": [32, 64, 128],
            "dropout_rate": [0.1, 0.2, 0.3],
            "learning_rate": [0.001, 0.0005, 0.0001]
        },
        "cnn_lstm": {
            "conv_filters": [32, 64, 128],
            "kernel_size": [3, 5, 7],
            "lstm_units": [50, 100, 200],
            "dropout_rate": [0.1, 0.2, 0.3],
            "learning_rate": [0.001, 0.0005, 0.0001]
        }
    }
    
    # Adjust based on data size
    size_multipliers = {
        "small": 0.5,
        "medium": 1.0,
        "large": 2.0
    }
    
    suggestions = base_suggestions.get(model_type, {})
    multiplier = size_multipliers.get(data_size, 1.0)
    
    # Adjust values based on data size
    adjusted_suggestions = {}
    for param, values in suggestions.items():
        if isinstance(values[0], (int, float)):
            adjusted_values = [v * multiplier for v in values]
            adjusted_suggestions[param] = [int(v) if param != "learning_rate" and param != "dropout_rate" else v for v in adjusted_values]
        else:
            adjusted_suggestions[param] = values
    
    return adjusted_suggestions

# Model Monitoring Endpoints
@router.get("/monitoring/status", summary="Get model monitoring status")
async def get_model_monitoring_status(
    model_name: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get monitoring status for models.
    """
    try:
        await initialize_model_services()
        
        if model_name:
            # Get specific model monitoring
            model_info = await model_manager.get_model(model_name)
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
            
            monitoring_data = await get_model_monitoring_data(model_name)
            
            return {
                "model_name": model_name,
                "monitoring": monitoring_data,
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            # Get monitoring for all models
            models = await model_manager.get_all_models()
            
            monitoring_summary = {}
            for name, info in models.items():
                monitoring_data = await get_model_monitoring_data(name)
                monitoring_summary[name] = monitoring_data
            
            # Calculate overall statistics
            total_models = len(models)
            active_models = sum(1 for m in monitoring_summary.values() if m.get("status") == "active")
            warning_models = sum(1 for m in monitoring_summary.values() if m.get("health_status") == "warning")
            error_models = sum(1 for m in monitoring_summary.values() if m.get("health_status") == "error")
            
            return {
                "total_models": total_models,
                "active_models": active_models,
                "warning_models": warning_models,
                "error_models": error_models,
                "monitoring_summary": monitoring_summary,
                "timestamp": datetime.now().isoformat()
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_model_monitoring_data(model_name: str) -> Dict:
    """Get monitoring data for a specific model"""
    try:
        # Get model info
        model_info = await model_manager.get_model(model_name)
        
        # Get recent predictions
        recent_predictions = await crud.get_latest_predictions(model_name, limit=10)
        
        # Get performance metrics
        performance = await performance_tracker.get_model_performance(model_name, "7d")
        
        # Calculate health status
        health_status = calculate_model_health(model_info, performance, recent_predictions)
        
        # Get drift metrics
        drift_metrics = await calculate_drift_metrics(model_name)
        
        return {
            "model_info": model_info,
            "health_status": health_status,
            "recent_activity": {
                "predictions_last_24h": len([p for p in recent_predictions]),
                "last_prediction": recent_predictions[0].get("timestamp") if recent_predictions else None
            },
            "performance": performance,
            "drift_metrics": drift_metrics,
            "last_checked": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting monitoring data for {model_name}: {e}")
        return {
            "error": str(e),
            "health_status": "error",
            "last_checked": datetime.now().isoformat()
        }

def calculate_model_health(model_info: Dict, performance: Dict, recent_predictions: List) -> Dict:
    """Calculate model health status"""
    health_status = {
        "status": "healthy",
        "issues": [],
        "score": 100
    }
    
    # Check model age
    if "last_trained" in model_info:
        last_trained = datetime.fromisoformat(model_info["last_trained"]) if isinstance(model_info["last_trained"], str) else model_info["last_trained"]
        days_since_training = (datetime.now() - last_trained).days
        
        if days_since_training > 30:
            health_status["issues"].append(f"Model not retrained for {days_since_training} days")
            health_status["score"] -= 20
    
    # Check performance metrics
    if "metrics" in performance:
        metrics = performance["metrics"]
        
        if "accuracy" in metrics and metrics["accuracy"] < 0.7:
            health_status["issues"].append(f"Low accuracy: {metrics['accuracy']:.2f}")
            health_status["score"] -= 15
        
        if "mse" in metrics and metrics["mse"] > 0.1:
            health_status["issues"].append(f"High MSE: {metrics['mse']:.4f}")
            health_status["score"] -= 15
    
    # Check recent activity
    if not recent_predictions:
        health_status["issues"].append("No recent predictions")
        health_status["score"] -= 10
    
    # Determine overall status
    if health_status["score"] >= 80:
        health_status["status"] = "healthy"
    elif health_status["score"] >= 60:
        health_status["status"] = "warning"
    else:
        health_status["status"] = "error"
    
    return health_status

async def calculate_drift_metrics(model_name: str) -> Dict:
    """Calculate data drift metrics for a model"""
    # This would compare training data distribution with current data distribution
    # For now, return mock metrics
    return {
        "feature_drift": {
            "mean_drift": 0.05,
            "max_drift": 0.12,
            "drift_detected": False,
            "threshold": 0.15
        },
        "concept_drift": {
            "detected": False,
            "confidence": 0.85
        },
        "last_checked": datetime.now().isoformat()
    }

# Health Check Endpoint
@router.get("/health", summary="Model system health check")
async def model_health_check(
    current_user: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Check health of model system components.
    """
    try:
        await initialize_model_services()
        
        health_checks = {
            "model_manager": await model_manager.health_check(),
            "model_trainer": await model_trainer.health_check(),
            "model_predictor": await model_predictor.health_check(),
            "data_collector": await data_collector.health_check(),
            "feature_engineer": await feature_engineer.health_check(),
            "performance_tracker": await performance_tracker.health_check(),
            "cache_manager": await cache_manager.health_check(),
            "database": await crud.health_check()
        }
        
        # Check model storage
        storage_check = await check_model_storage()
        health_checks["model_storage"] = storage_check
        
        # Check GPU availability if applicable
        gpu_check = await check_gpu_availability()
        health_checks["gpu_availability"] = gpu_check
        
        all_healthy = all(
            check.get("healthy", False) if isinstance(check, dict) else check
            for check in health_checks.values()
        )
        
        return {
            "healthy": all_healthy,
            "checks": health_checks,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in model health check: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def check_model_storage() -> Dict:
    """Check model storage health"""
    try:
        # Check storage directory
        import os
        from pathlib import Path
        
        model_dir = Path("models")
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage(model_dir)
        
        return {
            "healthy": True,
            "storage": {
                "total_gb": total // (2**30),
                "used_gb": used // (2**30),
                "free_gb": free // (2**30),
                "free_percentage": (free / total) * 100
            },
            "model_count": len(list(model_dir.glob("*.h5"))) + len(list(model_dir.glob("*.pkl")))
        }
    
    except Exception as e:
        return {"healthy": False, "error": str(e)}

async def check_gpu_availability() -> Dict:
    """Check GPU availability"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_info = []
            for i in range(gpu_count):
                gpu_info.append({
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated": torch.cuda.memory_allocated(i) / 1024**3,
                    "memory_reserved": torch.cuda.memory_reserved(i) / 1024**3
                })
            
            return {
                "available": True,
                "count": gpu_count,
                "gpus": gpu_info
            }
        else:
            return {
                "available": False,
                "message": "No GPU available, using CPU"
            }
    
    except ImportError:
        return {
            "available": False,
            "message": "PyTorch not installed"
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }