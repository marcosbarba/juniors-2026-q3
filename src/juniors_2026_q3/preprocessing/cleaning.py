"""
Limpieza reutilizable de las tres tablas fuente del proyecto STARWARS_AUTOCALLS.

Estas funciones se aplican de forma idéntica tanto al construir el conjunto
de entrenamiento como al consumir el modelo, por eso ninguna de ellas debe
referenciar columnas que solo existen en el histórico
(`executed`, `start_date`, `end_date`, `avg_duration_months`). La curación
específica del conjunto de entrenamiento (filtrado de filas, exclusiones)
vive en `dataset.py`, no aquí.
"""

import pandas as pd

OBS_FREQUENCY_MAP = {
    "Monthly": "Monthly", "1M": "Monthly", "M": "Monthly",
    "mensual": "Monthly", "1 month": "Monthly",
    "2M": "Bimonthly",
    "3 months": "Quarterly", "3M": "Quarterly", "Q": "Quarterly",
    "Quarterly": "Quarterly", "trimestral": "Quarterly",
    "6M": "Semiannual",
    "1Y": "Annual", "12M": "Annual", "Y": "Annual",
    "Annual": "Annual", "anual": "Annual",
    "1D": "Daily",
}

# Meses entre observaciones consecutivas, por categoría normalizada.
# Daily se aproxima como 1/21 (~21 días hábiles por mes) para mantener
# la misma unidad (meses) que el resto del dataset (avg_duration_months,
# nominal_term_months, no_call_period_months). No altera el orden relativo
# de las categorías, que es lo que un modelo de árboles aprovecha.
OBS_FREQUENCY_MONTHS_MAP = {
    "Daily": 1 / 21,
    "Monthly": 1,
    "Bimonthly": 2,
    "Quarterly": 3,
    "Semiannual": 6,
    "Annual": 12,
}


def clean_rfqs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza de rfqs.csv, válida tanto para el histórico de entrenamiento
    como para una RFQ nueva servida en producción.

    No toca ni asume la existencia de `executed`, `start_date`, `end_date`
    ni `avg_duration_months` (esas columnas no están disponibles en el
    momento de cotizar y su tratamiento es responsabilidad de `dataset.py`).

    Parameters
    ----------
    df : pd.DataFrame
        Debe contener, como mínimo, `underlyings`, `observation_frequency`
        y `requested_date`.

    Returns
    -------
    pd.DataFrame
        Copia de `df` con:
        - `requested_date` convertida a datetime.
        - `observation_frequency_clean`: categoría normalizada (6 valores).
        - `observation_frequency_months`: la anterior expresada en meses
          entre observaciones (variable ordinal/numérica).
        - `n_underlyings`: número de subyacentes de la cesta.

    Raises
    ------
    ValueError
        Si `observation_frequency` contiene algún valor no cubierto por
        `OBS_FREQUENCY_MAP`.
    """
    df = df.copy()

    df["requested_date"] = pd.to_datetime(df["requested_date"])

    mapped = df["observation_frequency"].map(OBS_FREQUENCY_MAP)
    unknown_mask = mapped.isna() & df["observation_frequency"].notna()
    if unknown_mask.any():
        unknown_values = sorted(df.loc[unknown_mask, "observation_frequency"].unique())
        raise ValueError(
            f"observation_frequency contiene valores no reconocidos: {unknown_values}. "
            "Actualiza OBS_FREQUENCY_MAP para cubrirlos antes de continuar."
        )
    df["observation_frequency_clean"] = mapped
    df["observation_frequency_months"] = df["observation_frequency_clean"].map(OBS_FREQUENCY_MONTHS_MAP)

    df["n_underlyings"] = df["underlyings"].str.count(r"\|") + 1

    for col in ("product_type", "basket_type"):
        df[col] = df[col].astype("category")

    return df


def clean_daily_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza de daily_volatility.csv: solo conversión de tipos.

    El pivote a formato ancho y cualquier agregación por ventana temporal
    son responsabilidad del módulo de integración, no de esta función.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def clean_underlyings_reference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza de underlyings_reference.csv.

    La EDA no encontró nulos, duplicados ni tipos incorrectos en esta
    tabla. Se mantiene como función explícita, en vez de omitirla, por
    consistencia con las otras dos fuentes y como punto de extensión si
    en el futuro aparecen datos que sí necesiten limpieza o validación.
    """
    df = df.copy()

    if (df["structural_base_vol"] <= 0).any():
        raise ValueError("structural_base_vol contiene valores no positivos.")

    return df