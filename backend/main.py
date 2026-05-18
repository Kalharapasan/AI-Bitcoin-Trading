import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from backend.api.config import settings
from backend.model.model import (
    predict_price as _predict_price,
    reload_model as _reload_model,
    get_model_info as _get_model_info,
    list_available_models as _list_models,
    save_uploaded_model as _save_uploaded_model,
)