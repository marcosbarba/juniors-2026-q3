"""
Carga del artefacto y del contexto de mercado UNA VEZ al arrancar la API.
Ninguna de las dos cosas se recalcula por petición.
"""

from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import uuid

from juniors_2026_q3.integration.build_features import build_features, prepare_volatility_context
from juniors_2026_q3.models.pipeline import select_features
from juniors_2026_q3.preprocessing.cleaning import clean_rfqs

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
DATA_DIR = PROJECT_ROOT / "data"

_pipeline = None
_context = None


def load_resources() -> None:
    global _pipeline, _context
    _pipeline = joblib.load(ARTIFACT_PATH)
    dv_raw = pd.read_csv(DATA_DIR / "daily_volatility.csv")
    ur_raw = pd.read_csv(DATA_DIR / "underlyings_reference.csv")
    _context = prepare_volatility_context(dv_raw, ur_raw)


def predict(rfq: dict) -> float:
    if _pipeline is None or _context is None:
        raise RuntimeError("Recursos no cargados: llama a load_resources() al arrancar la API.")

    rfq = dict(rfq)
    if rfq.get("rfq_id") is None:
        rfq["rfq_id"] = f"live-{uuid.uuid4().hex[:8]}"
    if rfq.get("requested_date") is None:
        rfq["requested_date"] = date.today()

    df = pd.DataFrame([rfq])
    df = clean_rfqs(df)
    df = build_features(df, _context)
    X = select_features(df)

    prediction = float(_pipeline.predict(X)[0])
    return rfq["rfq_id"], prediction