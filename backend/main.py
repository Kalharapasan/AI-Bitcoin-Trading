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


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    days: int = settings.DEFAULT_PREDICT_DAYS

class PredictResponse(BaseModel):
    current_price: float
    predicted_price: float
    signal: str


@app.get("/")
async def root():
    return {
        "message": "Bitcoin Trading AI API is running",
        "status": "operational",
        "version": settings.VERSION,
    }
    
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        result = _predict_price(days=request.days)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/chart-data")
async def chart_data(days: int = 30):