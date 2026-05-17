import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    APP_NAME: str = "Bitcoin Trading AI"
    VERSION: str = "0.1.0"
    DEBUG: bool = Field(False, env="DEBUG")
    MODEL_DIR: Path = Path(__file__).resolve().parents[2] / "model"
    MODEL_FILE: str = "model.h5"
    SCALER_FILE: str = "scaler.pkl"