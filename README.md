# Generalización Cross-Dataset — ViT-Base/16 y DenseNet-121 (sin augmentación x2)

Resultados del experimento de generalización cross-dataset (congelado / linear probe,
checkpoint de partida Mendeley), reducidos a **ViT-Base/16** y **DenseNet-121** en
**una sola configuración: sin augmentación x2**. Este repo solo contiene datos, logs y
gráficas de esos resultados.

## Resultados — F1 macro (media de 5 folds)

| Modelo | Trucha | Rohu | Mendeley (fuente) |
|---|---|---|---|
| **ViT-Base/16** | **86.11** | **88.17** | 96.96 |
| DenseNet-121 | 80.88 | 82.85 | 95.66 |

## Indicadores de robustez ante el cambio de dominio

| Modelo | TRD Trucha (%) | TRD Rohu (%) | Gap Trucha | Gap Rohu | RBE Trucha (%) | RBE Rohu (%) |
|---|---|---|---|---|---|---|
| DenseNet-121 (CNN, baseline) | 84.55 | 86.61 | 0.1478 | 0.1281 | — | — |
| **ViT-Base/16** | **88.81** | **90.93** | **0.1085** | **0.0879** | **26.59** | **31.38** |

```
TRD = (F1_objetivo / F1_fuente) × 100
Gap = Error_objetivo − Error_fuente,  Error = 1 − F1
RBE = (Gap_DenseNet − Gap_modelo) / Gap_DenseNet × 100
```

Detalle de cada cálculo en `Indicadores_Robustez_TRD_Gap_RBE.docx`.

## Dominios evaluados

- **Trucha**: 1,150 imágenes (OneDrive, Perú), dominio objetivo 1.
- **Rohu**: 1,261 imágenes — dataset combinado 70 % Trucha (805) / 30 % Rohu Kaggle (456),
  porque Rohu puro tiene un atajo de fecha/sesión que infla el F1 a ~99–100 %.

## Contenido

| Carpeta / archivo | Contenido |
|---|---|
| `1_codigo/` | Código de entrenamiento/evaluación (`finetune_target_x2.py`, `experimento_dataset_mixto.py`, `linear_probe_confmat_roc.py`, `domains.py`, …) y utilidades compartidas |
| `2_logs/` | Logs sin x2: `trucha_congelado_sinx2.log`, `rohu_congelado_sinx2.log`, `rohu_confmat_roc.log` |
| `3_resultados/` | CSV sin x2 (resumen y por fold), solo ViT-Base/16 y DenseNet-121 |
| `4_graficas/` | Matrices de confusión y curvas ROC pooled, `{trucha,rohu}_sinx2/{modelo}/` |
| `Resultados_ViT_DenseNet.docx` | Tabla de resultados e indicadores |
| `Indicadores_Robustez_TRD_Gap_RBE.docx` | Indicadores TRD, Gap y RBE con fórmulas instanciadas |
| `Documentacion_Tecnica_ViT_DenseNet.docx` | Metodología, glosario de fórmulas y detalle por modelo (por fold, matrices, ROC) |

## Método

Congelado (linear probe): backbone congelado (pesos fijos, checkpoint de Mendeley), solo
se entrena la cabeza de clasificación. 30 épocas, patience=8, 5 folds. Sin augmentación x2:
cada imagen de entrenamiento aparece una vez por época (con la augmentación estocástica
estándar). La adaptación es supervisada (usa etiquetas del dominio objetivo).

> Las matrices de confusión y ROC (`4_graficas/`, `rohu_confmat_roc.log`) vienen de una
> corrida aparte con la misma configuración; su F1 pooled difiere levemente del F1 oficial
> de arriba (p. ej. ViT Trucha 86.18 vs. 86.11).

## Reproducir

```bash
# Trucha
python finetune_target_x2.py --dominio onedrive_trucha --congelado --epochs 30 --patience 8 --sin-x2

# Rohu
python experimento_dataset_mixto.py --frac-trucha 0.7 --frac-rohu 0.3 --epochs 30 --patience 8 --sin-x2
```
