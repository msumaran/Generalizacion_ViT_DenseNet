# Generalización Cross-Dataset — ViT-Base/16 y DenseNet-121

Resultados del experimento de generalización cross-dataset (congelado / linear probe,
checkpoint de partida Mendeley), reducidos a **ViT-Base/16** y **DenseNet-121**. Este repo solo contiene datos, logs y
gráficas de esos resultados.

## Resultados — F1 macro (media de 10 folds)

| Modelo | Trucha | Rohu | Mendeley (fuente) |
|---|---|---|---|
| **ViT-Base/16** | **86.33** | **90.18** | 96.96 |
| DenseNet-121 | 82.77 | 84.16 | 95.66 |

ViT-Base/16 supera a DenseNet-121 en 8/10 folds en Trucha (Wilcoxon pareado p = 0.0488) y en 9/10 en Rohu (p = 0.0059).

## Indicadores de robustez ante el cambio de dominio

| Modelo | TRD Trucha (%) | TRD Rohu (%) | Gap Trucha (puntos) | Gap Rohu (puntos) | RBE Trucha (%) | RBE Rohu (%) |
|---|---|---|---|---|---|---|
| DenseNet-121 (CNN, baseline) | 86.53 | 87.98 | 12.89 | 11.50 | — | — |
| **ViT-Base/16** | **89.04** | **93.01** | **10.63** | **6.78** | **17.53** | **41.04** |

```
TRD = (F1_objetivo / F1_fuente) × 100
Gap = (Error_objetivo − Error_fuente) × 100,  Error = 1 − F1/100   (en puntos de F1)
RBE = (Gap_DenseNet − Gap_modelo) / Gap_DenseNet × 100
```

Detalle de cada cálculo en `6_documentacion/Indicadores_Robustez_TRD_Gap_RBE.docx`.

## Dominios evaluados

- **Trucha**: 1,150 imágenes (OneDrive, Perú), dominio objetivo 1.
- **Rohu**: 1,261 imágenes, dominio objetivo 2.

## Contenido

| Carpeta / archivo | Contenido |
|---|---|
| `1_codigo/` | Código del experimento: corrida de 10 folds (`linear_probe_con_test.py`, `experimento_dataset_mixto.py`), `domains.py`, la prueba de Wilcoxon (`wilcoxon_vit_vs_densenet.py`) y la corrida anterior de 5 folds (`finetune_target_x2.py`, `experimento_rohu.py`, `linear_probe_confmat_roc.py`) |
| `2_logs/` | Log de la corrida de 10 folds: `linear_probe_10fold.log`. Logs de la corrida anterior de 5 folds: `trucha_congelado_sinx2.log`, `rohu_congelado_sinx2.log`, `rohu_confmat_roc.log` |
| `3_resultados/` | `{trucha,rohu}/linear_probe_10fold/`: `resumen.csv` (F1, accuracy, AUC, épocas y tiempo por fold y pooled) y, por modelo, val_F1 por época, predicciones, matriz de confusión y curva ROC de cada fold y pooled. La corrida anterior de 5 folds queda en `congelado_sinx2_*.csv` y `confmat_roc/` |
| `4_graficas/` | val_F1 por época, matrices de confusión y curvas ROC, pooled y por fold (10 folds), `{trucha,rohu}/{modelo}/`; `uml/` con los diagramas UML |
| `6_documentacion/` | `Documentacion_Tecnica_ViT_DenseNet.docx` (ver abajo), `Resultados_ViT_DenseNet.docx` (tabla de resultados e indicadores) e `Indicadores_Robustez_TRD_Gap_RBE.docx` (TRD, Gap y RBE con fórmulas instanciadas); `diagramas/` con la arquitectura y la vista simple en SVG editable y PNG en alta resolución (`version_anterior/` conserva los diagramas previos) |
| `utilitarios/` | Scripts que no son parte del experimento: `diagramas/` (genera los diagramas de `6_documentacion/diagramas/` y los UML) y `documentacion/` (`generar_documentacion_tecnica.py`, que crea el documento y las gráficas por fold, y los scripts que lo completan) |
| `6_documentacion/Documentacion_Tecnica_ViT_DenseNet.docx` | Misma estructura que la documentación técnica de Mendeley: metodología por etapas (con arquitectura del sistema), glosario de fórmulas (incluye OOF/pooled y TRD/Gap/RBE), detalle por modelo y dominio (por fold, ROC, cálculo pooled) y anexos de trazabilidad y glosario de términos. Se genera con `python utilitarios/documentacion/generar_documentacion_tecnica.py` y se completa con los pasos de «Regenerar diagramas y documentación» |

## Método

Congelado (linear probe): backbone congelado (pesos fijos, checkpoint de Mendeley), solo
se entrena la cabeza de clasificación. 30 épocas, patience=8, 10 folds (el fold k parte del checkpoint Mendeley ((k−1) mod 5)+1; Mendeley se entrenó con 5 folds).
Cada imagen de entrenamiento aparece una vez por época, con variaciones al azar (recorte, volteo,
color, desenfoque, rotación y borrado). La adaptación es supervisada (usa etiquetas del dominio objetivo).

Todas las cifras (tablas, matrices, ROC, val_F1 por epoch) salen de una sola corrida de 10 folds.

## Reproducir

```bash
python 1_codigo/linear_probe_con_test.py --protocolo original --folds 10 --epochs 30 --patience 8
# escribe en 1_codigo/resultados/congelado_30ep/original_10fold/{trucha,rohu}/; copiado sin los .pt a 3_resultados/{trucha,rohu}/linear_probe_10fold/
python utilitarios/documentacion/generar_documentacion_tecnica.py   # documento base y gráficas por fold
```

## Regenerar diagramas y documentación

Los utilitarios están separados del código del experimento (`1_codigo/`) y no cambian ningún resultado.
Desde la raíz de este repositorio:

```bash
python utilitarios/diagramas/generar_diagrama_7_etapas.py          # diagramas → 6_documentacion/diagramas/
for t in png svg; do PLANTUML_LIMIT_SIZE=16384 plantuml -t$t -o "$PWD/4_graficas/uml" utilitarios/diagramas/uml/*.puml; done   # UML (brew install plantuml) → 4_graficas/uml/
uv run --with python-docx python utilitarios/documentacion/agregar_explicaciones_reunion.py
uv run --with python-docx python ../herramientas_docx/lectura_amigable.py 6_documentacion/Documentacion_Tecnica_ViT_DenseNet.docx
uv run --with python-docx python ../herramientas_docx/formato_formulas_y_enlaces.py 6_documentacion/Documentacion_Tecnica_ViT_DenseNet.docx
```

El paso de `agregar_explicaciones_reunion.py` parte de `6_documentacion/Documentacion_Tecnica_ViT_DenseNet.antes_reunion.docx` (el documento
base que genera `generar_documentacion_tecnica.py`): antes de volver a correrlo, copiar ese archivo sobre `Documentacion_Tecnica_ViT_DenseNet.docx`. `../herramientas_docx/` tiene las piezas
comunes a los dos repositorios (estilo de diagramas, formato de lectura, fórmulas y enlaces).
