"""API prediksi risiko sepsis (FastAPI).

Jalankan lokal dari folder sepsis-early-warning:
    uvicorn api.main:app --reload
Dokumentasi interaktif: http://localhost:8000/docs
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import sepsis_ew.models  # noqa: F401  (kelas di dalam artefak di-pickle dari modul ini)
from sepsis_ew.models import FEATURES

MODEL_PATH = Path(os.getenv(
    "MODEL_PATH", Path(__file__).resolve().parent.parent / "artifacts" / "ensemble.joblib"))
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)  # preprocessor + XGBoost + ANN + bobot
    yield
    state.clear()


app = FastAPI(title="Sepsis Early Warning API", version="1.0.0", lifespan=lifespan)


class Patient(BaseModel):
    """Data vital dan lab awal satu pasien UGD. Batas nilai = rentang masuk akal secara fisiologis."""
    heart_rate: float = Field(..., ge=20, le=300, description="Denyut jantung (bpm)")
    respiratory_rate: float = Field(..., ge=4, le=70, description="Laju napas (napas/menit)")
    temperature: float = Field(..., ge=30, le=45, description="Suhu tubuh (°C)")
    wbc_count: float = Field(..., ge=0, le=100, description="Leukosit (10^3/µL)")
    lactate_level: Optional[float] = Field(
        None, ge=0, le=30, description="Laktat darah (mmol/L). Boleh dikosongkan bila tidak diperiksa")
    age: float = Field(..., ge=0, le=120, description="Usia (tahun)")
    num_comorbidities: int = Field(..., ge=0, le=20, description="Jumlah penyakit penyerta")

    model_config = {"json_schema_extra": {"example": {
        "heart_rate": 122, "respiratory_rate": 29, "temperature": 39.2, "wbc_count": 18.1,
        "lactate_level": 4.7, "age": 83, "num_comorbidities": 4}}}


class Prediction(BaseModel):
    sepsis_risk_prediction: int = Field(..., description="1 = berisiko sepsis, 0 = tidak")
    risk_probability: float = Field(..., description="Probabilitas sepsis (0-1) dari ensemble")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    model = state["model"]
    row = pd.DataFrame([patient.model_dump()])[FEATURES]
    prob = float(model.predict_proba(row)[0])
    return Prediction(sepsis_risk_prediction=int(prob >= model.threshold),
                      risk_probability=round(prob, 4))
