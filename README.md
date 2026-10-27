# Generalización Cross-Dataset — ViT-Base/16 y DenseNet-121 (sin augmentación x2)

Resultados del experimento de generalización cross-dataset (congelado / linear probe,
checkpoint de partida Mendeley), reducidos a **ViT-Base/16** y **DenseNet-121** en
**una sola configuración: sin augmentación x2**. Este repo solo contiene datos, logs y
gráficas de esos resultados.

## Resultados — F1 macro (media de 10 folds)

| Modelo | Trucha | Rohu | Mendeley (fuente) |
|---|---|---|---|
| **ViT-Base/16** | **86.33** | **90.18** | 96.96 |
| DenseNet-121 | 82.77 | 84.16 | 95.66 |

ViT-Base/16 supera a DenseNet-121 en 8/10 folds en Trucha (Wilcoxon pareado p = 0.0488) y en 9/10 en Rohu (p = 0.0059).

## Indicadores de robustez ante el cambio de dominio

| Modelo | TRD Trucha (%) | TRD Rohu (%) | Gap Trucha | Gap Rohu | RBE Trucha (%) | RBE Rohu (%) |
|---|---|---|---|---|---|---|
| DenseNet-121 (CNN, baseline) | 86.53 | 87.98 | 0.1289 | 0.1150 | — | — |
| **ViT-Base/16** | **89.04** | **93.01** | **0.1063** | **0.0678** | **17.53** | **41.04** |

```
TRD = (F1_objetivo / F1_fuente) × 100
Gap = Error_objetivo − Error_fuente,  Error = 1 − F1
RBE = (Gap_DenseNet − Gap_modelo) / Gap_DenseNet × 100
```

Detalle de cada cálculo en `Indicadores_Robustez_TRD_Gap_RBE.docx`.

## Dominios evaluados

- **Trucha**: 1,150 imágenes (OneDrive, Perú), dominio objetivo 1.
- **Rohu**: 1,261 imágenes, dominio objetivo 2.

## Contenido

| Carpeta / archivo | Contenido |
|---|---|
| `1_codigo/` | Código de entrenamiento/evaluación (`linear_probe_con_test.py`, `experimento_dataset_mixto.py`, `domains.py`, …; la corrida anterior de 5 folds: `finetune_target_x2.py`, `experimento_rohu.py`, `linear_probe_confmat_roc.py`), utilidades compartidas y `generar_documentacion_tecnica.py` (genera la documentación técnica y las gráficas por fold) |
| `2_logs/` | Log de la corrida de 10 folds: `linear_probe_10fold.log`. Logs de la corrida anterior de 5 folds: `trucha_congelado_sinx2.log`, `rohu_congelado_sinx2.log`, `rohu_confmat_roc.log` |
| `3_resultados/` | `{trucha,rohu}/linear_probe_10fold/`: `resumen.csv` (F1, accuracy, AUC, epochs y tiempo por fold y pooled) y, por modelo, val_F1 por epoch, predicciones, matriz de confusión y curva ROC de cada fold y pooled. La corrida anterior de 5 folds queda en `congelado_sinx2_*.csv` y `confmat_roc/` |
| `4_graficas/` | val_F1 por epoch, matrices de confusión y curvas ROC, pooled y por fold (10 folds), `{trucha,rohu}_sinx2/{modelo}/` |
| `Resultados_ViT_DenseNet.docx` | Tabla de resultados e indicadores |
| `Indicadores_Robustez_TRD_Gap_RBE.docx` | Indicadores TRD, Gap y RBE con fórmulas instanciadas |
| `Documentacion_Tecnica_ViT_DenseNet.docx` | Misma estructura que la documentación técnica de Mendeley: metodología en 7 etapas (con figura de flujo y arquitectura del sistema), glosario de fórmulas (incluye OOF/pooled y TRD/Gap/RBE), detalle por modelo y dominio (por fold, ROC, cálculo pooled) y anexos de trazabilidad y glosario de términos. Se genera con `python 1_codigo/generar_arquitectura_sistema.py` y luego `python 1_codigo/generar_documentacion_tecnica.py` |

## Método

Congelado (linear probe): backbone congelado (pesos fijos, checkpoint de Mendeley), solo
se entrena la cabeza de clasificación. 30 épocas, patience=8, 10 folds (el fold k parte del checkpoint Mendeley ((k−1) mod 5)+1; Mendeley se entrenó con 5 folds). Sin augmentación x2:
cada imagen de entrenamiento aparece una vez por época (con la augmentación estocástica
estándar). La adaptación es supervisada (usa etiquetas del dominio objetivo).

Todas las cifras (tablas, matrices, ROC, val_F1 por epoch) salen de una sola corrida de 10 folds.

## Reproducir

```bash
python 1_codigo/linear_probe_con_test.py --protocolo original --folds 10 --epochs 30 --patience 8
# escribe en 1_codigo/resultados/congelado_30ep/original_10fold/{trucha,rohu}/; copiado sin los .pt a 3_resultados/{trucha,rohu}/linear_probe_10fold/
python 1_codigo/generar_arquitectura_sistema.py && python 1_codigo/generar_documentacion_tecnica.py
```
