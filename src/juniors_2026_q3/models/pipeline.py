"""
models/pipeline.py

Define el Pipeline de scikit-learn (preprocesamiento con estado + modelo)
para el proyecto STARWARS_AUTOCALLS. NO incluye la limpieza (clean_rfqs)
ni la integración (build_features): esas se aplican antes, en train.py y
en la API, sobre datos en su forma "ancha" original (underlyings con "|",
requested_date como fecha) que este Pipeline no necesita ver.

Los hiperparámetros del modelo están congelados a partir de la búsqueda
realizada en notebooks/modeling.ipynb (RandomizedSearchCV, 15 iteraciones,
TimeSeriesSplit de 5 folds) — ver notebooks/README.md, sección Modelado.
No se recalculan en cada entrenamiento: la búsqueda de hiperparámetros es
una actividad de investigación ya cerrada, no parte del pipeline de
producción.
"""

from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

CATEGORICAL_HIGH_CARDINALITY = ["counterparty", "trader_id"]

# Columnas de build_features/clean_rfqs que NO son features del modelo:
# identificadores, la fecha (ya usada para el join punto-en-el-tiempo, no
# aporta nada como input directo), y las versiones crudas de columnas ya
# normalizadas. Se centraliza aquí porque tanto train.py como la futura
# API necesitan aplicar exactamente el mismo recorte antes de predecir.
NON_FEATURE_COLUMNS = [
    "rfq_id", "requested_date", "underlyings",
    "observation_frequency", "observation_frequency_clean",
    "avg_duration_months"
]

MODEL_HYPERPARAMETERS = {
    "max_iter": 300,
    "max_depth": 5,
    "learning_rate": 0.03,
    "l2_regularization": 1.0,
    "categorical_features": "from_dtype",
    "random_state": 42,
}


def select_features(df):
    """Recorta un dataframe ya limpio+integrado a las columnas que el
    Pipeline espera como X. avg_duration_months, si está presente, no se
    incluye en NON_FEATURE_COLUMNS a propósito: quien llame a esta función
    sobre datos de entrenamiento debe extraerlo aparte como y antes."""
    cols_to_drop = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [("target_encoding", TargetEncoder(), CATEGORICAL_HIGH_CARDINALITY)],
        remainder="passthrough",
    ).set_output(transform="pandas")
    model = HistGradientBoostingRegressor(**MODEL_HYPERPARAMETERS)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])