# Model Card — STARWARS_AUTOCALLS duration model

**Fecha de generación**: 2026-08-30
**Modelo**: HistGradientBoostingRegressor (scikit-learn)
**Datos**: 10789 RFQs train / 2704 RFQs test (split temporal 80/20)

## Uso previsto

Estima `avg_duration_months` de un autocallable *worst-of* en el momento de
cotizar una RFQ. **Alcance**: duración esperada condicionada a que la RFQ
se ejecute — el modelo no predice si una RFQ se va a ejecutar o no.

## Métricas (evaluación única sobre test, no usado en ajuste)

| Métrica | Modelo | Baseline (mediana de train) |
|---|---|---|
| MAE  | 11.46 | 17.51 |
| RMSE | 15.22 | 22.71 |

Mejora sobre baseline: 34.5% en MAE.

## Variables más importantes (permutation importance, sobre test)

| feature                      |   importance |
|:-----------------------------|-------------:|
| product_type                 |     4.09625  |
| observation_frequency_months |     1.36007  |
| structural_vol_mean          |     1.31608  |
| structural_vol_std           |     0.995534 |
| autocall_barrier_pct         |     0.922673 |
| protection_barrier_pct       |     0.846274 |
| vol_level_max                |     0.572142 |
| vol_level_mean               |     0.516944 |

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
