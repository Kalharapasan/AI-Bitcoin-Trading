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

