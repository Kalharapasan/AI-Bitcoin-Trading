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
    latest = get_latest_model_path()
    if latest:
        model_path = latest
        model_name = Path(model_path).stem
        scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl")
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    
    else:
        model_path = DEFAULT_MODEL_PATH
        scaler_path = DEFAULT_SCALER_PATH
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    _current_model, _current_scaler = load_model_from_path(model_path, scaler_path)
    _model_load_time = datetime.now()
    _current_model_id = Path(model_path).stem.replace("model_", "").replace("model", "default")
    print(f"[INFO] Model '{_current_model_id}' loaded successfully")
    return _current_model, _current_scaler

try:
    load_latest_model()
except FileNotFoundError as e:
    print(f"[WARN] {e}")
    print("[WARN] Waiting for Colab to send model via /upload-model")


def save_uploaded_model(model_bytes: bytes, scaler_bytes: bytes, model_id: str, loss: float):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = model_id.replace(" ", "_").replace("/", "_")
    model_filename = f"model_{safe_id}_{timestamp}.h5"
    scaler_filename = f"model_{safe_id}_{timestamp}_scaler.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    scaler_path = os.path.join(MODEL_DIR, scaler_filename)
    with open(model_path, "wb") as f:
        f.write(model_bytes)
    print(f"[INFO] Model saved: {model_path}")
    with open(scaler_path, "wb") as f:
        f.write(scaler_bytes)
    print(f"[INFO] Scaler saved: {scaler_path}")
    with open(os.path.join(MODEL_DIR, "model.h5"), "wb") as f:
        f.write(model_bytes)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        f.write(scaler_bytes)
    
    global _current_model, _current_scaler, _model_load_time, _current_model_id
    _current_model, _current_scaler = load_model_from_path(
        os.path.join(MODEL_DIR, "model.h5"),
        os.path.join(MODEL_DIR, "scaler.pkl")
    )
    _current_model_id = safe_id
    _model_load_time = datetime.now()
    
    return {
        "model_file": model_filename,
        "scaler_file": scaler_filename,
        "model_id": safe_id,
        "loss": loss,
        "loaded": True,
    }
    
def fetch_recent_data(days: int = 7) -> np.ndarray:
    ticker = "BTC-USD"
    df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
    if df.empty:
        raise RuntimeError("Failed to fetch Bitcoin data from yfinance.")
    return df[["Close", "Volume"]].values.astype(np.float32)

def prepare_sequence(data: np.ndarray, window: int = 60) -> np.ndarray:
    if _current_scaler is None:
        raise RuntimeError("Scaler not loaded")
    scaled = _current_scaler.transform(data)