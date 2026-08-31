"""
Integración de rfqs.csv (ya curado/limpio) con daily_volatility.csv y
underlyings_reference.csv, a nivel de cesta.

Se aplica de forma idéntica en entrenamiento (sobre el resultado de
dataset.curate_rfqs) y en la API (sobre el resultado de
preprocessing.cleaning.clean_rfqs) — asume que `rfqs` ya tiene
`requested_date` como datetime y la columna `underlyings` sin modificar.

Decisión de rendimiento: la correlación entre subyacentes se precalcula
UNA VEZ para los 91 pares posibles del universo de 14 subyacentes (rolling
de 63 días sobre toda la serie histórica), en vez de recalcularse por cada
RFQ. Cada RFQ solo consulta, en su `requested_date`, las columnas de los
pares que forma su propia cesta.
"""

import itertools
from dataclasses import dataclass

import pandas as pd

from juniors_2026_q3.preprocessing.cleaning import (
    clean_daily_volatility,
    clean_underlyings_reference,
)

WINDOW = 63


@dataclass
class VolatilityContext:
    """Todo lo que se puede calcular a partir de daily_volatility.csv y
    underlyings_reference.csv SIN conocer todavía ninguna RFQ concreta.
    Se calcula UNA VEZ — en train.py antes de procesar el dataset completo,
    o en la API al arrancar el servicio — y se reutiliza en cada llamada
    a build_features, sin volver a limpiar ni recalcular nada de mercado."""
    dv_clean: pd.DataFrame
    underlyings_ref_clean: pd.DataFrame
    corr_ref: pd.DataFrame


def _pair_key(a: str, b: str) -> str:
    lo, hi = sorted((a, b))
    return f"{lo}__{hi}"


def pivot_daily_volatility(dv_clean: pd.DataFrame) -> pd.DataFrame:
    return dv_clean.pivot(index="date", columns="underlying", values="realized_vol_63d")


def build_pairwise_correlation_reference(dv_wide: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    underlyings = sorted(dv_wide.columns)
    corr_ref = pd.DataFrame(index=dv_wide.index)
    for a, b in itertools.combinations(underlyings, 2):
        corr_ref[_pair_key(a, b)] = dv_wide[a].rolling(window).corr(dv_wide[b])
    return corr_ref


def prepare_volatility_context(
    daily_volatility_raw: pd.DataFrame,
    underlyings_reference_raw: pd.DataFrame,
    window: int = WINDOW,
) -> VolatilityContext:
    """Punto de entrada del precálculo. Llamar UNA VEZ (arranque de train.py
    o de la API); el resultado se pasa a build_features en cada llamada."""
    dv_clean = clean_daily_volatility(daily_volatility_raw)
    underlyings_ref_clean = clean_underlyings_reference(underlyings_reference_raw)
    dv_wide = pivot_daily_volatility(dv_clean)
    corr_ref = build_pairwise_correlation_reference(dv_wide, window=window)
    return VolatilityContext(dv_clean, underlyings_ref_clean, corr_ref)


def _explode_by_underlying(rfqs: pd.DataFrame) -> pd.DataFrame:
    exploded = rfqs[["rfq_id", "requested_date", "underlyings"]].copy()
    exploded["underlying"] = exploded["underlyings"].str.split("|")
    return exploded.explode("underlying").drop(columns="underlyings")


def _attach_volatility_level(exploded: pd.DataFrame, dv_clean: pd.DataFrame) -> pd.DataFrame:
    left = exploded.sort_values("requested_date")
    right = dv_clean[["date", "underlying", "realized_vol_63d"]].sort_values("date")
    return pd.merge_asof(
        left, right, left_on="requested_date", right_on="date",
        by="underlying", direction="backward",
    )


def _attach_structural_vol(exploded: pd.DataFrame, underlyings_ref_clean: pd.DataFrame) -> pd.DataFrame:
    return exploded.merge(
        underlyings_ref_clean[["underlying", "structural_base_vol"]],
        on="underlying", how="left",
    )


def _aggregate_basket_levels(exploded_with_levels: pd.DataFrame) -> pd.DataFrame:
    return exploded_with_levels.groupby("rfq_id").agg(
        vol_level_mean=("realized_vol_63d", "mean"),
        vol_level_max=("realized_vol_63d", "max"),
        structural_vol_mean=("structural_base_vol", "mean"),
        structural_vol_std=("structural_base_vol", lambda s: s.std(ddof=0)),
    )


def _compute_basket_correlation(row: pd.Series) -> pd.Series:
    subyacentes = sorted(row["underlyings"].split("|"))
    if len(subyacentes) == 1:
        return pd.Series({"basket_corr_mean": 1.0, "basket_corr_min": 1.0})
    pair_keys = [_pair_key(a, b) for a, b in itertools.combinations(subyacentes, 2)]
    valores = row[pair_keys]
    return pd.Series({"basket_corr_mean": valores.mean(), "basket_corr_min": valores.min()})


def build_features(rfqs: pd.DataFrame, context: VolatilityContext) -> pd.DataFrame:
    """
    Integra rfqs (ya limpio, salida de clean_rfqs/curate_rfqs) con el
    contexto de mercado ya precalculado. Válida tanto para 13.493 filas de
    entrenamiento como para una única RFQ nueva en la API.
    """
    exploded = _explode_by_underlying(rfqs)
    exploded = _attach_volatility_level(exploded, context.dv_clean)
    exploded = _attach_structural_vol(exploded, context.underlyings_ref_clean)
    niveles_por_rfq = _aggregate_basket_levels(exploded)

    rfqs_sorted = rfqs.sort_values("requested_date")
    rfqs_con_corr_ref = pd.merge_asof(
        rfqs_sorted, context.corr_ref.reset_index(),
        left_on="requested_date", right_on="date", direction="backward",
    )
    correlaciones = rfqs_con_corr_ref.apply(_compute_basket_correlation, axis=1)

    resultado = rfqs.merge(niveles_por_rfq, on="rfq_id", how="left")
    resultado = resultado.merge(
        pd.concat([rfqs_con_corr_ref["rfq_id"], correlaciones], axis=1),
        on="rfq_id", how="left",
    )
    return resultado