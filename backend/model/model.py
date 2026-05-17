import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model