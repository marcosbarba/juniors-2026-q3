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

import pandas as pd

from juniors_2026_q3.preprocessing.cleaning import (
    clean_daily_volatility,
    clean_underlyings_reference,
)

WINDOW = 63


def _pair_key(a: str, b: str) -> str:
    """Nombre de columna canónico para un par de subyacentes, sin
    depender del orden en que aparezcan en la cesta."""
    lo, hi = sorted((a, b))
    return f"{lo}__{hi}"


def pivot_daily_volatility(dv_clean: pd.DataFrame) -> pd.DataFrame:
    """daily_volatility en formato ancho: una fila por fecha, una columna
    por subyacente. Requiere dv_clean con `date` ya en datetime."""
    return dv_clean.pivot(index="date", columns="underlying", values="realized_vol_63d")


def build_pairwise_correlation_reference(dv_wide: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    Correlación móvil de `window` días para TODOS los pares posibles de
    subyacentes del universo (91 pares con 14 subyacentes), calculada una
    sola vez sobre toda la serie histórica.

    Nota: las primeras `window` fechas del panel salen NaN (no hay
    suficiente histórico hacia atrás) — no supone un problema con los
    datos actuales, ya que el panel arranca ~2 años antes que la RFQ más
    temprana, pero conviene tenerlo presente si el panel se recortara.
    """
    underlyings = sorted(dv_wide.columns)
    corr_ref = pd.DataFrame(index=dv_wide.index)
    for a, b in itertools.combinations(underlyings, 2):
        corr_ref[_pair_key(a, b)] = dv_wide[a].rolling(window).corr(dv_wide[b])
    return corr_ref


def _explode_by_underlying(rfqs: pd.DataFrame) -> pd.DataFrame:
    """Una fila por combinación (rfq_id, underlying) de la cesta."""
    exploded = rfqs[["rfq_id", "requested_date", "underlyings"]].copy()
    exploded["underlying"] = exploded["underlyings"].str.split("|")
    exploded = exploded.explode("underlying").drop(columns="underlyings")
    return exploded


def _attach_volatility_level(exploded: pd.DataFrame, dv_clean: pd.DataFrame) -> pd.DataFrame:
    """Nivel de realized_vol_63d de cada subyacente en la fecha disponible
    más reciente igual o anterior a requested_date (join asof, robusto a
    que la fecha de la RFQ no coincida exactamente con un día del panel)."""
    left = exploded.sort_values("requested_date")
    right = dv_clean[["date", "underlying", "realized_vol_63d"]].sort_values("date")
    return pd.merge_asof(
        left, right,
        left_on="requested_date", right_on="date",
        by="underlying", direction="backward",
    )


def _attach_structural_vol(exploded: pd.DataFrame, underlyings_ref_clean: pd.DataFrame) -> pd.DataFrame:
    """structural_base_vol no tiene componente temporal: merge directo."""
    return exploded.merge(
        underlyings_ref_clean[["underlying", "structural_base_vol"]],
        on="underlying", how="left",
    )


def _aggregate_basket_levels(exploded_with_levels: pd.DataFrame) -> pd.DataFrame:
    """
    Agregación por cesta de las magnitudes que no requieren pares
    (nivel de volatilidad, volatilidad estructural).

    - vol_level_mean / vol_level_max: ASUNCIÓN — no se fijó explícitamente
      en el diseño. Se usa mean+max (simétrico al criterio de "capturar
      el extremo" usado en la correlación) a falta de confirmación.
    - structural_vol_mean / structural_vol_std: acordado (std con
      ddof=0, para que una cesta de 1 subyacente dé 0 de forma natural
      en vez de NaN, sin necesitar una rama especial en el código).
    """
    return exploded_with_levels.groupby("rfq_id").agg(
        vol_level_mean=("realized_vol_63d", "mean"),
        vol_level_max=("realized_vol_63d", "max"),
        structural_vol_mean=("structural_base_vol", "mean"),
        structural_vol_std=("structural_base_vol", lambda s: s.std(ddof=0)),
    )


def _compute_basket_correlation(row: pd.Series) -> pd.Series:
    """
    Media y mínimo de las correlaciones por pares de la cesta de esta
    fila, consultando las columnas ya precalculadas en corr_ref.
    Cestas de 1 subyacente: 1.0 en ambas (corr(X, X) = 1 por definición).
    """
    subyacentes = sorted(row["underlyings"].split("|"))
    if len(subyacentes) == 1:
        return pd.Series({"basket_corr_mean": 1.0, "basket_corr_min": 1.0})

    pair_keys = [_pair_key(a, b) for a, b in itertools.combinations(subyacentes, 2)]
    valores = row[pair_keys]
    return pd.Series({"basket_corr_mean": valores.mean(), "basket_corr_min": valores.min()})


def build_features(
    rfqs: pd.DataFrame,
    daily_volatility_raw: pd.DataFrame,
    underlyings_reference_raw: pd.DataFrame,
    window: int = WINDOW,
) -> pd.DataFrame:
    """
    Punto de entrada único de integración. `rfqs` debe llegar ya limpio
    (salida de clean_rfqs o curate_rfqs) — esta función no vuelve a
    limpiarlo. Las otras dos tablas se limpian aquí porque son estáticas/
    de mercado y no dependen del flujo de curación de entrenamiento.

    Devuelve `rfqs` con seis columnas nuevas: vol_level_mean,
    vol_level_max, structural_vol_mean, structural_vol_std,
    basket_corr_mean, basket_corr_min.
    """
    dv_clean = clean_daily_volatility(daily_volatility_raw)
    underlyings_ref_clean = clean_underlyings_reference(underlyings_reference_raw)
    dv_wide = pivot_daily_volatility(dv_clean)

    # --- nivel de volatilidad + volatilidad estructural, por cesta ---
    exploded = _explode_by_underlying(rfqs)
    exploded = _attach_volatility_level(exploded, dv_clean)
    exploded = _attach_structural_vol(exploded, underlyings_ref_clean)
    niveles_por_rfq = _aggregate_basket_levels(exploded)

    # --- correlación de cesta ---
    corr_ref = build_pairwise_correlation_reference(dv_wide, window=window)
    rfqs_sorted = rfqs.sort_values("requested_date")
    rfqs_con_corr_ref = pd.merge_asof(
        rfqs_sorted, corr_ref.reset_index(),
        left_on="requested_date", right_on="date",
        direction="backward",
    )
    correlaciones = rfqs_con_corr_ref.apply(_compute_basket_correlation, axis=1)

    resultado = rfqs.merge(niveles_por_rfq, on="rfq_id", how="left")
    resultado = resultado.merge(
        pd.concat([rfqs_con_corr_ref["rfq_id"], correlaciones], axis=1),
        on="rfq_id", how="left",
    )
    return resultado