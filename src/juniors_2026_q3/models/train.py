"""
models/train.py — entrenamiento de principio a fin.

Ejecutar con: uv run juniors-2026-q3 (o `python -m juniors_2026_q3.models.train`)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

from juniors_2026_q3.dataset import curate_rfqs
from juniors_2026_q3.integration.build_features import build_features, prepare_volatility_context
from juniors_2026_q3.models.model_card import generate_model_card
from juniors_2026_q3.models.pipeline import build_pipeline, select_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs"
TARGET = "avg_duration_months"
TEST_FRACTION = 0.2


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    rfqs_raw = pd.read_csv(DATA_DIR / "rfqs.csv")
    dv_raw = pd.read_csv(DATA_DIR / "daily_volatility.csv")
    ur_raw = pd.read_csv(DATA_DIR / "underlyings_reference.csv")

    rfqs_curated = curate_rfqs(rfqs_raw)
    context = prepare_volatility_context(dv_raw, ur_raw)
    df = build_features(rfqs_curated, context) 

    split_idx = int(len(df) * (1 - TEST_FRACTION))
    fecha_corte = df.iloc[split_idx]["requested_date"]
    train = df[df["requested_date"] < fecha_corte]
    test = df[df["requested_date"] >= fecha_corte]

    X_train, y_train = select_features(train), train[TARGET]
    X_test, y_test = select_features(test), test[TARGET]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    baseline_pred = np.full_like(y_test, fill_value=y_train.median())

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae_baseline": mean_absolute_error(y_test, baseline_pred),
        "rmse_baseline": np.sqrt(mean_squared_error(y_test, baseline_pred)),
    }
    metrics["improvement_pct"] = (1 - metrics["mae"] / metrics["mae_baseline"]) * 100

    perm = permutation_importance(
        pipeline, X_test, y_test, scoring="neg_mean_absolute_error",
        n_repeats=10, random_state=42, n_jobs=-1,
    )
    importances_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=False)

    joblib.dump(pipeline, ARTIFACTS_DIR / "model.joblib")
    generate_model_card(metrics, importances_df, len(train), len(test), DOCS_DIR / "model_card.md")

    print(f"MAE test: {metrics['mae']:.2f} | RMSE test: {metrics['rmse']:.2f}")
    print(f"Fecha de corte: {fecha_corte.date()} | Train: {len(train)} | Test: {len(test)}")
    print(f"Artefacto guardado en {ARTIFACTS_DIR / 'model.joblib'}")
    print(f"Model card generado en {DOCS_DIR / 'model_card.md'}")


if __name__ == "__main__":
    main()