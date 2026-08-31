"""
API de inferencia — STARWARS_AUTOCALLS.
Arrancar en local con: uv run juniors-2026-q3 serve
Documentación interactiva: http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from juniors_2026_q3.api.inference import load_resources, predict
from juniors_2026_q3.api.schemas import PredictionResponse, RFQRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield


app = FastAPI(title="STARWARS_AUTOCALLS — duration prediction API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(rfq: RFQRequest):
    try:
        rfq_id, prediction = predict(rfq.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictionResponse(rfq_id=rfq_id, predicted_avg_duration_months=prediction)