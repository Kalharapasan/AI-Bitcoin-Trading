import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
DEFAULT_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

_current_model = None
_current_scaler = None
_current_model_id = None
_model_load_time = None

def list_available_models() -> List[Dict]:
    models = []
    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        return models
    
    for model_file in model_dir.glob("model*.h5"):
        stat = model_file.stat()
        model_id = model_file.stem.replace("model_", "").replace("model", "default")
        models.append({
            "model_id": model_id,
            "filename": model_file.name,
            "path": str(model_file),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        })
    
    models.sort(key=lambda x: x["created"], reverse=True)
    return models

def get_latest_model_path() -> Optional[str]:
    models = list_available_models()
    if not models:
        return None
    return models[0]["path"]

def load_model_from_path(model_path: str, scaler_path: str):
    print(f"[INFO] Loading model: {model_path}")
    model = load_model(model_path)

    print(f"[INFO] Loading scaler: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

def load_latest_model():
    global _current_model, _current_scaler, _model_load_time, _current_model_id