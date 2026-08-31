# Model Card — STARWARS_AUTOCALLS duration model

**Fecha de generación**: 2026-08-30
**Modelo**: HistGradientBoostingRegressor (scikit-learn)
**Datos**: 6563 RFQs train / 6930 RFQs test (split temporal 80/20)

## Uso previsto

Estima `avg_duration_months` de un autocallable *worst-of* en el momento de
cotizar una RFQ. **Alcance**: duración esperada condicionada a que la RFQ
se ejecute — el modelo no predice si una RFQ se va a ejecutar o no.

## Métricas (evaluación única sobre test, no usado en ajuste)

| Métrica | Modelo | Baseline (mediana de train) |
|---|---|---|
| MAE  | 11.20 | 16.79 |
| RMSE | 14.86 | 22.04 |

Mejora sobre baseline: 33.3% en MAE.

## Variables más importantes (permutation importance, sobre test)

| feature                      |   importance |
|:-----------------------------|-------------:|
| product_type                 |     3.74392  |
| autocall_barrier_pct         |     1.79421  |
| observation_frequency_months |     1.43854  |
| structural_vol_mean          |     1.37318  |
| structural_vol_std           |     0.878704 |
| vol_level_mean               |     0.539535 |
| basket_corr_mean             |     0.53682  |
| vol_level_max                |     0.500839 |

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
