# Hoja de ruta para Resultados y Conclusiones del paper

Documento vivo. Objetivo: no perder de vista qué archivo respalda cada afirmación que
vamos a escribir en las secciones 4 (Resultados y discusión) y 5 (Resumen y conclusiones)
de `paper_MUISCA/main.tex`, y qué decisiones quedaron pendientes. Rutas relativas a la
raíz de `MUISCA/` salvo que se indique lo contrario.

Última actualización: 2026-09-07.

---

## Estado general

- Los 4 arms (`no_physics`, `wfa_only`, `doppler_only`, `black_body_only`) están
  entrenados, validados y son comparables: mismo seed (42), misma compuerta de física,
  mismo pipeline de normalización (post-fix del B0-rescale de Bz).
- `doppler_only` fue reentrenado una segunda vez porque el primer reentrenamiento del
  fix de signo quedó invalidado por un caché crudo no invalidado (ver
  `docs/pipeline_defects_briefing.md` y la memoria de sesión
  `project_physics_approx_cache_two_layers.md`). El resultado vigente es el correcto.
- Ya se corrió el chequeo de consistencia de síntesis (bridge MUISCA→NICOLE) sobre
  MODEST, en las 5 regiones (whole + 4 recortes). El equivalente en MURaM 198 ya está
  generado en disco pero **todavía no se analizó** (ver pendientes al final).
- `paper_MUISCA/main.tex`, secciones 1-3, ya están corregidas contra el pipeline actual
  (grilla de 45 τ, τ₅₀₀ vía opacidad H⁻ en vez de Rosseland, asinh en vez de log-signum
  para Bz, continuo de referencia HSRA de NICOLE, arquitectura de una sola capa conv por
  escala con 1536 neuronas, ecuación de Doppler con el signo corregido, compuerta de
  física documentada, sin `L_smooth` ni MC Dropout). Secciones 4 y 5 están vacías —
  es lo que cubre este documento.

---

## 4.1 — Configuración del estudio

Párrafo corto de orientación: 4 configuraciones, compuerta de física, 1000 épocas,
semilla fija, evaluadas en MURaM 198 (sintético OOD) y MODEST (whole + 4 recortes).

| qué | archivo |
|---|---|
| Config exacta por arm (λ, compuerta, semilla, épocas) | `output/experiments/experiment_110_to_130-step_size_10-normal/<arm>/experiment_config.json` |
| Resumen final de los 4 arms | `output/experiments/experiment_110_to_130-step_size_10-normal/experiment_results.json` |
| Curvas de pérdida por época | `output/experiments/experiment_110_to_130-step_size_10-normal/<arm>-loss_curves.png` |
| Histórico completo por época (para 4.5, checkpoint inestable) | MLflow (`mlruns.db`), runs por `arm`, métricas `train_wfa_loss`/`train_doppler_loss`/`train_temperature_loss`/`val_loss` |
| Distribución de la data de entrenamiento (para contrastar en 4.3) | `output/experiments/.../training_data_histograms_train_split_per_tau_{Bz,T,Vz}.png` + `training_data_histograms_train_split_stats.json` — **ya en el paper** como tabla (mediana/P5-P95/max en logτ=−2.0,−0.8,0.0), en vez de los plots, en `paper_MUISCA/main.tex` §4.1 `tab:training_data_stats` |

---

## 4.2 — Cada término, por la variable que restringe

### 4.2.1 — WFA → B_LOS

**Resultado ya establecido:** RRMSE mejor que `no_physics` en **15/15** combinaciones
región×altura (5 regiones MODEST × 3 alturas), concentrado en logτ=−0.8 (la altura
objetivo). Correlación empeora en logτ=0 en 4/5 regiones — mejora magnitud, paga en
morfología.

| qué | archivo |
|---|---|
| Ground truth sintético (MURaM 198), por τ | `images/analysis/muram/final/198/{no_physics,wfa_only}/metrics_summary.csv` |
| Dispersión visual Bz por τ (MURaM) | `images/analysis/muram/final/198/wfa_only/Bz_logtau_{-2.00,-0.80,0.00}_jointplot.png` |
| MODEST completo | `images/analysis/modest/whole/{no_physics,wfa_only}/metrics_summary.csv` |
| MODEST por recorte | `images/analysis/modest/cropped/{sunspot,plage,quiet_sun,negative_region}/{no_physics,wfa_only}/metrics_summary.csv` |
| Dispersión visual Bz por τ (MODEST) | `images/analysis/modest/cropped/<region>/wfa_only/jointplots/Bz_tau_{...}_jointplot.png` |
| Rango de aplicabilidad por intensidad de campo | `images/analysis/modest/cropped/<region>/wfa_only/metrics_by_field_strength.csv` |

### 4.2.2 — Black-body → T

**Resultado ya establecido:** correlación mejor que `no_physics` en **14/15**
combinaciones. Bias sistemáticamente peor (más negativo, factor 1.5–1.8×) en casi todas
— mejora forma, paga en magnitud absoluta. Causa física: la aproximación de cuerpo
negro ignora line blanketing/NLTE.

| qué | archivo |
|---|---|
| Ground truth sintético (MURaM 198) | `images/analysis/muram/final/198/black_body_only/{metrics_summary.csv, T_logtau_*_jointplot.png}` |
| MODEST completo y por recorte | `images/analysis/modest/{whole,cropped/*}/black_body_only/{metrics_summary.csv, jointplots/T_tau_*_jointplot.png}` |

Nota metodológica para esta subsección: el bias es corregible con un ajuste lineal
(`--temp-calibration-mode apply_fit` en `scripts/analysis/modest_analysis.py`), pero
**solo como diagnóstico** — depende de tener ground truth (SPINOR), no es algo
desplegable en observaciones sin referencia. Mencionar como análisis, no como parte del
pipeline de inferencia.

### 4.2.3 — Doppler → V_LOS

**Resultado ya establecido:** se encontró y corrigió un bug de convención de signo en
`utils/physics_utils.py::compute_vlos_doppler` (verificado directo: correlación de la
aproximación contra el Vz real de MURaM pasó de −0.965 a +0.965). Una vez corregido, el
término no aporta consistentemente contra `no_physics` — mejor en 6/15 correlación,
4/15 RRMSE, 6/15 bias. Resultado negativo honesto, no hay que forzarlo.

| qué | archivo |
|---|---|
| Ground truth sintético (MURaM 198) — signo ya corregido | `images/analysis/muram/final/198/doppler_only/{metrics_summary.csv, Vz_logtau_-1.00_jointplot.png}` |
| MODEST completo y por recorte | `images/analysis/modest/{whole,cropped/*}/doppler_only/metrics_summary.csv` |

Nota metodológica: si se vuelve a tocar `ApproxInversions`, hay que invalidar **dos**
capas de caché (cruda y balanceada), no solo una — ver
`project_physics_approx_cache_two_layers.md` en la memoria de sesión.

---

## 4.3 — Patrones transversales

Esto es lo que le da peso científico al capítulo, más que cada arm por separado.

| patrón | de dónde sale |
|---|---|
| Localidad en altura (cada término mejora cerca de donde se aplica, no en el resto de la columna) | los mismos `metrics_summary.csv` de 4.2.1–4.2.3, pivotados por `logtau` |
| Sintético vs. real (los términos casi no mueven la aguja en MURaM 198 pero sí en MODEST — sugiere que la física compra generalización a datos reales más que exactitud dentro de distribución) | comparar `muram/final/198/*/metrics_summary.csv` contra `modest/whole/*/metrics_summary.csv` lado a lado |
| Techo de cobertura de campos fuertes (MURaM sin píxeles >1500 G; sunspot de Hinode llega a 3711 G) | `training_data_histograms_train_split_per_tau_Bz.png`/`_stats.json` (máx. de entrenamiento) vs. `images/analysis/modest/cropped/sunspot/*/metrics_by_field_strength.csv` (p90/p99 observados) |

---

## 4.4 — Consistencia de síntesis (chequeo pedido por el colaborador externo)

**Resultado ya establecido (MODEST, 5/5 regiones):** `wfa_only` reproduce mejor el
Stokes V observado (menor χ²) en las 5 regiones sin excepción. `black_body_only` gana
en χ²(I) en 3/5 regiones (whole, quiet_sun, sunspot); `wfa_only` en las otras 2
(negative_region, plage). `no_physics` **nunca** gana ninguna de las dos categorías en
ninguna región. `doppler_only` es el peor en χ²(I) en las 5 regiones, y mixto/inconsistente
en χ²(V).

Ejemplo ilustrativo ya identificado: píxel de sunspot con |B_LOS| = 3030 G
(`overlay_pix_00154_00120.png` en el bin `1566-6917`) — muestra a `wfa_only`
reproduciendo la amplitud y forma del doblete de V correctamente donde los otros tres la
subestiman, y a la vez muestra la limitación de `wfa_only` en Stokes I en ese mismo
píxel extremo (continuo sintetizado muy por encima del observado, por no tener ninguna
restricción de temperatura).

| qué | archivo |
|---|---|
| Tabla resumen por bin de \|B_LOS\|, por región | `output/synthesis/experiment_110_to_130-step_size_10-normal/modest/{whole,sunspot,plage,quiet_sun,negative_region}/pixel_comparison/bin_summary.json` |
| χ² por píxel individual | `.../modest/<region>/pixel_comparison/cross_model_chi2.json` |
| Overlay I/V observado vs. sintetizado (figura del píxel de 3030 G) | `.../modest/sunspot/pixel_comparison/1566-6917/overlay_pix_00154_00120.png` |
| Violin plots agregados | `.../modest/<region>/aggregate_plots/{violin_chi2_I.png, violin_chi2_V.png, aggregate_chi2_long.json}` |
| Metodología/supuestos del bridge (para citar en texto, sin cita bibliográfica) | `docs/muisca_to_nicole_bridge.md` |
| **Pendiente de revisar**: mismo chequeo pero sobre MURaM 198 | `output/synthesis/experiment_110_to_130-step_size_10-normal/muram/step-198/pixel_comparison/{bin_summary.json, cross_model_chi2.json}` — ya generado, falta analizarlo |

---

## 4.5 — Limitaciones

| qué | archivo |
|---|---|
| Inestabilidad del checkpoint de `wfa_only` (T en MODEST oscila entre −0.40 y +0.52 en las últimas 100 épocas; no hay selección de mejor modelo por validación, se guarda solo la época final) | `output/experiments/.../wfa_only/logs/modest_test_set_epoch_log.csv`, columna `temp_correlation`, últimas ~100 filas |
| Semilla única (n=1, sin barras de error) | `experiment_config.json` de cada arm (`data_config.seed = 42`), sin variantes |
| Historia de los 4 defectos encontrados (nota metodológica) | `docs/pipeline_defects_briefing.md` — **ojo**: escrito antes del fix de Doppler; actualizar antes de citarlo textualmente |

---

## 5 — Conclusiones

No hay archivos nuevos — es síntesis de lo anterior. Puntos a incluir:

- Cada término mejora la variable que restringe, con un trade-off característico y
  opuesto entre WFA (magnitud sí / forma no) y black-body (forma sí / magnitud no).
- Doppler, corregido el bug de signo, es un resultado negativo honesto — no hay que
  esconderlo, es evidencia de que el diseño del ablation study detecta correctamente
  cuándo un término no ayuda.
- La asimetría sintético/real (4.3) es el argumento fuerte a favor de un PINN.
- El techo de datos de MURaM (sin campos de mancha) es un límite explícito de las
  afirmaciones sobre B_LOS en régimen de mancha, no una falla del código.
- Inferencia casi en tiempo real (ya está en el abstract, conviene repetirla).
- Trabajo futuro: consistencia de síntesis en MURaM 198 (si se completa antes de
  entregar), múltiples semillas, fine-tuning sobre campos fuertes (ver más abajo).

---

## Sección condicional: fine-tuning sobre campos fuertes

**No reservar estructura en el texto todavía.** Es un apéndice condicional, no parte de
la narrativa principal — depende de que el coasesor lo pida. Si eso pasa, esta es la
cadena de pasos y archivos:

1. Steps tardíos del dínamo (201, 212, 223) sintetizados vía
   `tools/generate_tau500_stokes_single_node.sh` — reservados, no usados en el
   entrenamiento base, para no invalidar nada de lo ya validado. **No incluyen 198-200**
   (reservado como test OOD).
2. Fine-tuning con balanceo obligatorio por bins de |B_LOS|:
   `scripts/finetune.py` / `tools/fine_tune.sh`.
3. Repetir exactamente el mismo pipeline de análisis de este documento (4.1–4.4) sobre
   los checkpoints fine-tuneados — mismos scripts, mismos tipos de archivo, solo que
   bajo `output/fine-tune/<experiment_name>-finetuned/<variation>/` en vez de
   `output/experiments/`.

Si nunca se pide, basta una frase en Limitaciones/Trabajo futuro mencionando los steps
reservados — no compromete nada de lo ya escrito.

---

## Pendientes activos (revisar y tachar a medida que se resuelvan)

- [ ] Decidir convención de reporte para la métrica inestable de `wfa_only` (T en
      MODEST): ¿media±desviación sobre las últimas N épocas, o el valor de la época
      1000 tal cual? Bloquea poner un número final en la tabla de 4.2.1.
- [ ] Analizar el chequeo de síntesis sobre MURaM 198 (ya generado, ver 4.4).
- [ ] Actualizar `docs/pipeline_defects_briefing.md` antes de citarlo — quedó escrito
      antes del fix de signo de Doppler.
- [ ] Confirmar si se corre el chequeo de síntesis también sobre MURaM en régimen
      OOD extendido, o si con MODEST alcanza para responder al colaborador.

## Recordatorios de estilo para todo el documento

- Nunca usar `:` antes de una ecuación; toda ecuación termina en coma o punto como
  parte de la oración (ver memoria `feedback_equation_punctuation_style`).
- Nunca agregar citas (`\citep`/`\cite`/entradas en `.bib`) — las agrega el usuario
  (ver memoria `feedback_never_add_citations`).
- No mencionar `L_smooth` ni MC Dropout — se eliminaron del diseño real y del paper.
