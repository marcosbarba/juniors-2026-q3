# EDA y Modelado — STARWARS_AUTOCALLS

Resumen del trabajo sobre las tres tablas fuente (`rfqs.csv`, `daily_volatility.csv`, `underlyings_reference.csv`): hipótesis de negocio de partida, decisiones de preprocesamiento e integración, hallazgos de la EDA, y metodología y resultados del modelado (comparación de algoritmos, ajuste de hiperparámetros, evaluación final e interpretabilidad).

Notebooks de referencia: `rfqs_EDA.ipynb`, `daily_volatility_EDA.ipynb`, `underlyings_reference_EDA.ipynb`, `modeling.ipynb`.

## Hipótesis de negocio de partida

Antes de tocar los datos, se establecieron las siguientes hipótesis a partir del mecanismo de un autocallable *worst-of* descrito en el enunciado:

- La cancelación anticipada depende únicamente del **subyacente más débil** de la cesta en cada fecha de observación. Por tanto, `autocall_barrier_pct` debería tener relación causal directa con la duración (barrera más alta → más difícil de superar → mayor duración esperada).
- `protection_barrier_pct` solo interviene si el producto llega a vencimiento sin cancelarse antes, por lo que no debería tener relación causal directa con la duración, a diferencia de la barrera de autocall.
- La **dispersión entre los subyacentes de la cesta** (poco correlacionados entre sí) aumenta la probabilidad de que uno se quede rezagado por debajo de la barrera, alargando la duración. Cestas muy correlacionadas se comportan como un solo activo, facilitando que el mínimo también supere la barrera antes.
- El **número de subyacentes** de la cesta debería correlacionar positivamente con la duración, por el mismo mecanismo (más nombres, más probabilidad de un rezagado).
- `quoted_implied_vol`, al ser una expectativa de volatilidad *forward-looking* conocida en el momento de cotizar, se consideró la variable de volatilidad potencialmente más relevante — más que cualquier feature derivada de volatilidad histórica.
- Solo la información disponible en `requested_date` es válida como input: es el único momento en el que la mesa cotiza sin conocer el desenlace.

## Decisiones de preprocesamiento

- **`executed`, `start_date`, `end_date` no se usan directamente como features**, pero se investigó si contradecían la premisa del enunciado (ver más abajo, el resultado cambió una decisión de diseño).
- **Sesgo de selección**: el target `avg_duration_months` solo existe para RFQs con `executed = True`. Es un sesgo de selección (no de supervivencia: el target no está definido, no es que el evento se anule) que acota el alcance del modelo a "duración esperada condicionada a la ejecución". Se cuantificó por `product_type` y `counterparty`: la tasa de ejecución ronda 45/55% en todas las categorías sin diferencias relevantes, por lo que el sesgo existe conceptualmente pero no es severo en la práctica. Se documenta además un posible **bucle de retroalimentación**: un modelo que predice mal la duración de un producto poco ejecutado puede llevar a cotizarlo de forma poco competitiva, perpetuando su bajo volumen de ejecución.
- **`start_date`/`end_date` (y por tanto `nominal_term_months`) no se usan como features, pese a evidencia empírica de que están disponibles antes de la ejecución.** Se comprobó que, para `executed = False`, `start_date` coincide siempre con `requested_date` y `end_date` sigue una distribución de plazo nominal prácticamente idéntica a la de las RFQs ejecutadas — sugiriendo que el sistema histórico sí fija una fecha de inicio/vencimiento propuesta en el momento de cotizar. Sin embargo, el enunciado especifica explícitamente que estos campos "solo existen si `executed = True`", y se decidió **respetar la literalidad del contrato de datos documentado** en vez de apoyarse en un comportamiento empírico no garantizado del sistema histórico, evitando que la API de inferencia dependa de columnas que, según la especificación, no tienen por qué estar disponibles en el momento de servir una predicción. Esta decisión tiene un coste real y asumido conscientemente: `nominal_term_months` mostró la segunda correlación más fuerte con el target de toda la tabla (0.53 global, hasta 0.90 en `Wretched Hive Digital`), y no existe un proxy equivalente entre el resto de columnas — el plazo nominal varía en el mismo rango completo (~24-120 meses) dentro de los seis `product_type`, por lo que esta información no queda capturada indirectamente por ninguna otra variable disponible en el momento de cotizar.
- **Target encoding de `counterparty`/`trader_id`** (cardinalidad 8 y 39 respectivamente): calcular la media del target dentro del propio train introduce fuga de información sutil. Mitigación: K-fold target encoding, con la media global de train como *fallback* para categorías no vistas en producción.
- **Sanity checks superados sin excepciones**: `requested_date <= start_date`, `start_date <= end_date`, `avg_duration_months >= no_call_period_months`, y `basket_type == "single"` ⟺ `n_underlyings == 1`.
- **Exclusión de 303 filas (`Wretched Hive Digital`, ver perfil abajo)**: violan la regla de negocio `avg_duration_months <= nominal_term_months`. Se decidió excluirlas del entrenamiento en lugar de recortarlas (*clip*) o dejarlas, porque no se puede determinar si el error está en el target o en las fechas del contrato (recortar asumiría sin evidencia que el target es el dato corrupto). Volumen excluido: 2.2% de las ejecutadas, sin comprometer el tamaño muestral.
- **`observation_frequency`** normalizada de 18 categorías (duplicadas por idioma/formato: `1M`, `Monthly`, `mensual`...) a 6 categóricas limpias (`Monthly`, `Bimonthly`, `Quarterly`, `Semiannual`, `Annual`, `Daily`), y codificada a `observation_frequency_months` (meses entre observaciones) como variable ordinal — coherente con la mediana de duración observada por categoría (`Annual` más alta, `Daily` más baja).
- **`sector` (de `underlyings_reference.csv`) descartado como feature adicional**: hay 14 sectores para 14 subyacentes, es decir, sector y subyacente son la misma variable. La feature planeada de "número de sectores distintos en la cesta" resulta idéntica a `n_underlyings` y no aporta información nueva.

## Hallazgos — `rfqs.csv`

- Target `avg_duration_months` con fuerte asimetría a la derecha, unimodal, pico en 25-30 meses, cola hasta >120, coherente con el mecanismo de negocio (acumulación de cancelaciones tempranas tras el `no_call_period`, cola decreciente de productos que tardan más o llegan a vencimiento).
- **Confusión por `product_type` (Simpson) confirmada y corregida en dos variables**:
  - `autocall_barrier_pct` mostraba correlación global ≈0 con la duración, contra lo esperado por mecanismo. Al calcular la correlación *dentro* de cada `product_type`, aparece positiva y consistente (0.12–0.21) en los cuatro productos con varianza en la barrera — confirma el mecanismo de negocio, enmascarado a nivel global por las diferencias entre productos.
  - `protection_barrier_pct` mostraba correlación global de -0.337. Dentro de cada producto, la correlación es prácticamente nula (-0.03 a 0.01) — confirma que no hay relación causal real; la correlación global era artefacto de que cada producto tiene su propio nivel característico de barrera de protección y de duración, sin relación entre ambos.
- **`quoted_implied_vol` — hallazgo más relevante de la fase bivariante**: correlación global débil (-0.065), pero el signo se invierte por familia de producto: positiva (0.29–0.37) en los productos con `autocall_barrier_pct` ≈1.0, fuertemente negativa (-0.55) en los de barrera ≈1.27, nula en Wretched Hive Digital. No es un caso de "correlación diluida" sino de dos efectos opuestos que casi se cancelan en el agregado. Se descartó `no_call_period_months` como confusor (correlación con `quoted_implied_vol` ≈0 en los tres productos con periodo variable). Queda como hipótesis razonada de teoría de opciones sobre cestas *worst-of* (a igualdad de barrera al dinero, más volatilidad dificulta que el mínimo de la cesta la supere; con barrera muy por encima del nivel inicial, más volatilidad es la única vía de alcanzarla, y tiende a ocurrir pronto cuando ocurre) — no confirmable con los datos disponibles.
- `nominal_term_months` correlaciona positivamente con la duración en todos los productos (actúa como techo físico), con `Wretched Hive Digital` destacando en 0.898 (ver perfil). *Hallazgo documentado a título de EDA — finalmente no se usa como feature de modelado, ver Decisiones de preprocesamiento.*
- **Multicolinealidad**: la relación más fuerte y esperada es la de las dos barreras entre sí (0.629). Las correlaciones aparentes de ambas barreras con `no_call_period_months` (-0.69, -0.55) se confirmaron como artefacto de `product_type` (patrones fijos por producto, no relación de contrato), contrastando dentro de `Death Star Phoenix Note` (única con varianza simultánea en ambas), donde la correlación es prácticamente nula (-0.022). `notional_credits` y `quoted_implied_vol` sin colinealidad relevante con el resto.

## Hallazgos — `daily_volatility.csv`

- Panel perfectamente rectangular: los 14 subyacentes cubren exactamente el mismo rango (2014-04-01 a 2026-06-30) sin huecos. La RFQ más temprana es de 2016-01-04 — casi dos años de histórico disponibles antes de la primera solicitud, sin problema de cobertura insuficiente.
- **Factor de mercado común confirmado**: 12 de los 14 subyacentes correlacionan entre sí en el rango 0.80–0.93 (serie completa 2014–2026). Implica que la feature de dispersión/correlación de cesta saldrá sistemáticamente alta para cualquier cesta compuesta solo por este bloque.
- **Autocorrelación de lag-1 entre 0.978 y 0.998** en los 14 subyacentes — confirma cuantitativamente (no solo en teoría) que `realized_vol_63d`, al ser media móvil de 63 días, tiene redundancia interna severa: cualquier correlación calculada directamente sobre ella está inflada por este solapamiento.
- Sin valores negativos ni repetidos (no hay "regímenes" con valor sostenido, cada observación es distinta).

## Hallazgos — `underlyings_reference.csv`

- 14 sectores para 14 subyacentes — sector y subyacente son, a efectos prácticos, la misma variable (ver decisión de preprocesamiento).
- El ratio `vol_media_realizada / structural_base_vol` es consistente entre 1.20–1.24 para 12 de los 14 subyacentes, salvo `HTTX` (1.75) y sobre todo `REBL` (2.65) — ver perfiles atípicos.

## Perfiles de subyacentes/productos atípicos

### `Wretched Hive Digital` (`product_type`)

Tres piezas de evidencia independientes, encontradas en momentos distintos de la EDA, apuntan a un comportamiento diferenciado de este producto:

1. Concentra el 70% de las 303 violaciones de `avg_duration_months > nominal_term_months` (tasa de violación 9.17%, frente a un máximo de 2.08% en el resto). Se investigaron y descartaron como causa `no_call_period_months`, `autocall_barrier_pct`, `observation_frequency`, `counterparty` y `trader_id` — sin causa identificable con las columnas disponibles. Decisión: exclusión de esas 303 filas del entrenamiento (ver arriba).
2. Mantiene la mediana de duración más alta de los seis productos, incluso tras excluir las filas anómalas.
3. Su correlación con `nominal_term_months` (0.898) es la más alta de los seis productos, sugiriendo que rara vez se cancela anticipadamente y su duración queda determinada casi enteramente por el plazo nominal asignado.

### `REBL` (subyacente — sector "Renta fija especulativa")

- Correlación **negativa** con los otros 12 subyacentes del bloque común (-0.06 a -0.15), frente a positiva y alta entre ellos.
- Autocorrelación de lag-1 más baja de los 14 subyacentes (0.978), junto con `HTTX`.
- Ratio `vol_media_realizada / structural_base_vol` de 2.65, más del doble que cualquier otro subyacente.
- Hipótesis de negocio (no verificable con los datos, pero coherente con las tres observaciones): comportamiento típico de renta fija frente a renta variable/materias primas en episodios de estrés de mercado — cuando la volatilidad del resto de activos sube, la dinámica de tipos de interés puede moverse en sentido contrario.

### `HTTX` (subyacente — sector "Comercio y materias primas")

Posición intermedia en los mismos tres indicadores: volatilidad media varias veces superior al resto pero no invertida (correlación moderada 0.35–0.47 con el bloque común, no negativa), autocorrelación algo menor (0.982) y ratio estructural/realizada de 1.75. Consistente con un activo real pero de dinámica distinta al bloque de renta variable/corporativo dominante.

## Limitaciones documentadas

- Alcance del modelo limitado a duración condicionada a `executed = True` (sesgo de selección, cuantificado como leve).
- `realized_vol_63d` es la única serie temporal por subyacente disponible; su autocorrelación de lag-1 (>0.97 en todos los casos) infla cualquier correlación de cesta calculada directamente sobre ella — limitación estructural del dataset entregado, no evitable sin datos de retornos diarios brutos.
- Factor de mercado común entre 12 de los 14 subyacentes: la feature de correlación de cesta tendrá poco poder discriminante salvo que la cesta incluya `REBL` o `HTTX`.
- El signo invertido de `quoted_implied_vol` según familia de producto queda documentado pero no explicado causalmente con las columnas disponibles, y no se reprodujo de forma concluyente en la importancia de features del modelo final (ver Interpretabilidad).
- 303 filas de `Wretched Hive Digital` excluidas del entrenamiento por inconsistencia entre target y plazo nominal, sin causa identificada.
- Se descarta `nominal_term_months` como feature pese a ser el segundo predictor más correlacionado con el target (0.53), por fidelidad al contrato de datos documentado en el enunciado — coste de poder predictivo asumido conscientemente a cambio de robustez de producción, sin proxy disponible en el resto de columnas.

## Modelado
 
### Metodología: split temporal, métrica y baseline
 
- **Split temporal 80/20** (no aleatorio, por la naturaleza secuencial de `requested_date`): corte en 2022-10-12 → 10.794 filas train / 2.699 test. Validado con datos, no solo visualmente: volatilidad media de mercado en train (0.366) y test (0.380) prácticamente equivalente (+3.6%), y el máximo histórico de toda la serie (2015) queda dentro de train — el modelo no tiene que generalizar hacia un régimen de estrés mayor al ya visto.
- **Métrica** — elegida en función de la forma del target (ver Hallazgos `rfqs.csv`: fuerte asimetría a la derecha, cola hasta 127 meses, mínimo cercano a cero, 0.16):
  - **MAPE descartado**: con valores del target muy próximos a cero, el error porcentual se dispara o queda mal definido — unos pocos casos con duración real de 1-2 meses podrían dominar la métrica agregada sin representar el error típico del modelo.
  - **RMSE/MSE descartados como métrica única (aunque sí se reportan como secundaria)**: al elevar al cuadrado, sobreponderan los errores de la cola larga del target frente al grueso de observaciones — una métrica principal basada solo en RMSE optimizaría el modelo hacia minimizar errores en los casos extremos (pocos, pero de gran magnitud) a costa de la precisión en el rango típico (25-50 meses, donde se concentra la mayoría de las RFQs).
  - **MAE elegida como métrica principal**: robusta frente a la cola larga (no eleva al cuadrado, cada error pesa proporcionalmente a su magnitud), directamente interpretable en meses (una unidad que tiene significado de negocio inmediato para la mesa), y no distorsionada por los valores cercanos a cero que invalidan el MAPE.
  - **RMSE se mantiene como métrica secundaria**, no descartada del todo: para la gestión del libro de riesgo de la mesa, un error de 40 meses no tiene el mismo impacto que uno de 5, aunque el MAE los trate por igual (misma contribución lineal) — RMSE aporta esa sensibilidad adicional a los errores grandes que el MAE no captura, y se reporta junto al MAE en vez de en su lugar.
- **Baseline** (predicción constante = mediana de train, 35.01 meses): MAE 17.51 / RMSE 22.71 — suelo mínimo que cualquier modelo debe batir con margen para justificar su complejidad.
### Comparación de modelos
 
- 5 algoritmos comparados con `TimeSeriesSplit` (5 folds) **dentro de train únicamente** — test se mantuvo intacto hasta la evaluación final, para no contaminar la decisión de qué modelo elegir: RandomForest y GradientBoosting (one-hot + K-fold target encoding vía `category_encoders.TargetEncoder`), HistGradientBoosting, XGBoost y LightGBM (categóricas nativas, sin one-hot).
- Con hiperparámetros por defecto, XGBoost salía claramente peor (MAE 12.21 frente a 11.57–11.70 del resto) — identificado como artefacto de un `learning_rate` por defecto distinto (0.3 frente a 0.1 en los demás), no de una diferencia real de capacidad del algoritmo. Al igualarlo a 0.1, XGBoost sube a 11.56.
- Con `learning_rate` igualado, los 5 modelos quedan en un rango de MAE 11.46–11.70, indistinguible del ruido entre folds (std 0.30–0.43 en cada uno) — **no hay un ganador estadísticamente significativo**.
- Elegido `HistGradientBoostingRegressor` por criterio de simplicidad ante el empate: incluido en scikit-learn sin dependencias adicionales (evita `xgboost`/`lightgbm` en producción), y gestiona categóricas de alta y baja cardinalidad de forma nativa.
### Ajuste de hiperparámetros y evaluación final
 
- `RandomizedSearchCV` (15 iteraciones, mismo `TimeSeriesSplit`): MAE de 11.57 (por defecto) a 11.46 con `max_iter=300, max_depth=5, learning_rate=0.03, l2_regularization=1.0`.
- **Evaluación única sobre test** (primera vez que se toca en todo el proceso): MAE 11.46 / RMSE 15.22 — 34.5% de mejora sobre el baseline. El MAE de test coincide casi exactamente con el de validación cruzada, señal de que el ajuste de hiperparámetros no sobreajustó a los folds y de que el split temporal generaliza bien al periodo más reciente.
### Interpretabilidad — Permutation Importance (sobre test)
 
- `product_type` domina claramente (importancia 4.10, más del triple que la siguiente) — coherente con la EDA: determina de forma casi determinista `autocall_barrier_pct` y `no_call_period_months`.
- Barreras con importancia moderada (`autocall_barrier_pct` 0.92, `protection_barrier_pct` 0.85), por debajo de `product_type` — su señal ya está parcialmente absorbida por la propia categoría de producto.
- **`quoted_implied_vol` con importancia marginal (0.038) a nivel global**, en contraste con el hallazgo de signo invertido por producto documentado en la EDA. Investigado calculando importancia dentro de cada `product_type` por separado: la importancia sube de forma generalizada respecto al agregado, pero el patrón de signos no reproduce limpiamente el de la EDA (solo 2 de 6 productos coinciden en signo; varias estimaciones, con ~450 filas por producto, resultan indistinguibles de cero). Se documenta como **evidencia parcial y no concluyente**, no como confirmación: permutation importance mide el efecto marginal sobre el modelo completo, no la relación bivariada con el target, y no tiene por qué coincidir con la correlación de la EDA cuando el efecto real opera principalmente por interacción con otra variable.
- `n_underlyings` con importancia nula (0.000), pese a tener una base causal razonable en la EDA — posiblemente diluida por `basket_corr_mean`/`basket_corr_min` y `structural_vol_std`, que ya capturan indirectamente estructura de cesta. `counterparty` con importancia ligeramente negativa (-0.0016) — dentro del ruido de muestreo, se interpreta como sin importancia detectable, no como un efecto perjudicial real.
## Estado del proyecto y pendiente
 
- **Resuelto**: limpieza reutilizable (`preprocessing/cleaning.py`), curación de entrenamiento (`dataset.py`), integración de las tres tablas a nivel de cesta (`integration/build_features.py`, validada contra cálculo manual independiente), y modelado completo (comparación, ajuste, evaluación e interpretabilidad, arriba).
- **Pendiente**: `train.py` (orquestación completa: carga → `curate_rfqs` → `build_features` → `Pipeline` con limpieza envuelta en `FunctionTransformer` + `HistGradientBoostingRegressor` con los hiperparámetros finales → `joblib.dump`), model card en `docs/`, API de inferencia, y tests automatizados (deprioritizados conscientemente por tiempo — ver módulos `preprocessing`/`integration` para los casos límite ya identificados y no implementados).