"""
models/model_card.py

Genera docs/model_card.md combinando prosa fija (contexto, alcance,
limitaciones — consistente con notebooks/README.md) con métricas
recalculadas en cada entrenamiento, para que el documento nunca quede
desincronizado del modelo que realmente se sirve.
"""

from datetime import date


def generate_model_card(metrics: dict, importances_df, n_train: int, n_test: int, output_path) -> None:
    top_features = importances_df.head(8).to_markdown(index=False)

    content = f"""# Model Card — STARWARS_AUTOCALLS duration model

**Fecha de generación**: {date.today().isoformat()}
**Modelo**: HistGradientBoostingRegressor (scikit-learn)
**Datos**: {n_train} RFQs train / {n_test} RFQs test (split temporal 80/20)

## Uso previsto

Estima `avg_duration_months` de un autocallable *worst-of* en el momento de
cotizar una RFQ. **Alcance**: duración esperada condicionada a que la RFQ
se ejecute — el modelo no predice si una RFQ se va a ejecutar o no.

## Métricas (evaluación única sobre test, no usado en ajuste)

| Métrica | Modelo | Baseline (mediana de train) |
|---|---|---|
| MAE  | {metrics['mae']:.2f} | {metrics['mae_baseline']:.2f} |
| RMSE | {metrics['rmse']:.2f} | {metrics['rmse_baseline']:.2f} |

Mejora sobre baseline: {metrics['improvement_pct']:.1f}% en MAE.

## Variables más importantes (permutation importance, sobre test)

{top_features}

## Limitaciones conocidas

- Sesgo de selección: entrenado solo con RFQs ejecutadas (`executed = True`).
- No usa `nominal_term_months` como feature (fidelidad al contrato de datos
  documentado en el enunciado), pese a ser un predictor fuerte en la EDA.
- 303 RFQs de `Wretched Hive Digital` excluidas del entrenamiento por
  inconsistencia entre target y plazo nominal, sin causa identificada.
- El efecto de `quoted_implied_vol` documentado en la EDA (signo invertido
  según `product_type`) no se reproduce de forma concluyente en la
  importancia de features de este modelo.

Detalle completo del proceso de análisis y modelado: `notebooks/README.md`.
"""
    output_path.write_text(content, encoding="utf-8")