from pathlib import Path
import json
from typing import Literal
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from joblib import load
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "premium_model.joblib"
METADATA_PATH = ROOT / "model" / "metadata.json"
if not MODEL_PATH.exists():
    sys.path.insert(0, str(ROOT))
    from train_model import train
    train()

model = load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
app = FastAPI(title="Health Insurance Premium Prediction", version="1.0.0",
              description="Estimate annual insurance charges using a Linear Regression model.")
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))

class PredictionInput(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: Literal["male", "female"]
    bmi: float = Field(ge=10, le=80)
    smoker: Literal["yes", "no"]
    children: int = Field(ge=0, le=10)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "metadata": metadata},
    )

@app.get("/api/health")
def health():
    return {"status": "ok", "model_source": metadata.get("source", "unknown"), "demo_mode": metadata.get("demo_mode", True)}

@app.post("/api/predict")
def predict(payload: PredictionInput):
    row = [{"age": payload.age, "sex": payload.sex, "bmi": payload.bmi,
            "smoker": payload.smoker, "children": payload.children}]
    value = float(model.predict(pd.DataFrame(row, columns=["age", "sex", "bmi", "smoker", "children"]))[0])
    return {"predicted_premium": round(max(0, value), 2), "currency_note": "Value is in the same currency units as the model's training dataset.",
            "demo_mode": bool(metadata.get("demo_mode", True)), "model_source": metadata.get("source", "unknown")}

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "An unexpected server error occurred."})
