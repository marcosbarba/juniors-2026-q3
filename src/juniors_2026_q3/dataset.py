"""
Curación del conjunto de entrenamiento para el proyecto STARWARS_AUTOCALLS.

A diferencia de `preprocessing/cleaning.py`, este módulo se ejecuta una vez,
antes de entrenar el modelo, nunca se importa desde la API de inferencia.
Aquí viven las decisiones que presuponen que se conoce el desenlace de una RFQ
(`executed`, `avg_duration_months`) o columnas (`start_date`, `end_date`)
que, por la decisión documentada en notebooks/README.md, no forman parte
del contrato de entrada de la API.
"""

import pandas as pd

from juniors_2026_q3.preprocessing.cleaning import clean_rfqs

# Columnas que existen en el histórico de entrenamiento pero que NUNCA deben
# llegar a X: o bien no están disponibles en el momento de servir una
# predicción real, o son el propio target. Mantenerla como lista explícita,
# en vez de confiar en acordarse de eliminar cada columna en el punto donde
# se calcula, es la salvaguarda contra fuga de información hacia el modelo.
COLUMNS_NOT_AVAILABLE_AT_INFERENCE = [
    "executed",
    "start_date",
    "end_date",
    "nominal_term_months",  # transitorio: solo se usa para filtrar filas
]


def _compute_nominal_term_months(df: pd.DataFrame) -> pd.Series:
    """
    Plazo nominal del contrato (en meses), calculado únicamente para poder
    aplicar el filtro de integridad avg_duration_months <= nominal_term_months.
    No es una feature del modelo.
    """
    start = pd.to_datetime(df["start_date"])
    end = pd.to_datetime(df["end_date"])
    return (end - start).dt.days / 30.44


def curate_rfqs(raw_rfqs: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra y limpia rfqs.csv a nivel de fila, dejándolo listo para
    integrar con daily_volatility.csv y underlyings_reference.csv.

    Aplica, en este orden:
      1. Limpieza reutilizable (clean_rfqs), la misma que se aplicara
         para consumir el modelo.
      2. Filtrado a RFQs ejecutadas (executed == True): solo estas tienen
         avg_duration_months definido.
      3. Exclusión de las filas que violan avg_duration_months <=
         nominal_term_months (anomalía documentada, concentrada en
         Wretched Hive Digital (ver notebooks/README.md).
      4. Eliminación de columnas no disponibles en inferencia. 
         avg_duration_months se conserva: es el target, y se separará de X
         una vez el dataframe esté completamente integrado.

    Returns
    -------
    pd.DataFrame
        Filas curadas y limpias, con avg_duration_months incluido y sin
        ninguna columna de COLUMNS_NOT_AVAILABLE_AT_INFERENCE.
    """
    df = clean_rfqs(raw_rfqs)
    df["nominal_term_months"] = _compute_nominal_term_months(df)

    executed_mask = df["executed"]
    valid_duration_mask = df["avg_duration_months"] <= df["nominal_term_months"]

    df_curated = df.loc[executed_mask & valid_duration_mask].copy()

    print(
        f"RFQs totales: {len(df)} | "
        f"excluidas por no ejecutadas: {(~executed_mask).sum()} | "
        f"excluidas por violación de plazo nominal: {(executed_mask & ~valid_duration_mask).sum()} | "
        f"conjunto curado final: {len(df_curated)}"
    )

    return df_curated.drop(columns=COLUMNS_NOT_AVAILABLE_AT_INFERENCE)