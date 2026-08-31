from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class RFQRequest(BaseModel):
    """Espejo de rfqs.csv, EXCLUYENDO executed/start_date/end_date/
    avg_duration_months — ninguna existe todavía cuando se cotiza una RFQ
    nueva (ver notebooks/README.md, Decisiones de preprocesamiento)."""

    rfq_id: Optional[str] = Field(default=None, description="Si se omite, se genera uno internamente.")
    product_type: str
    basket_type: str
    underlyings: str = Field(..., description='Subyacentes separados por "|", p.ej. "KYBR|TECH"')
    autocall_barrier_pct: float
    protection_barrier_pct: float
    no_call_period_months: int
    observation_frequency: str
    quoted_implied_vol: float
    notional_credits: int
    counterparty: str
    trader_id: str
    requested_date: Optional[date] = Field(
        default=None,
        description="Si se omite, se usa la fecha actual (cotización en vivo).",
    )


class PredictionResponse(BaseModel):
    rfq_id: str
    predicted_avg_duration_months: float