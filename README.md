# Generalización Cross-Dataset — ViT-Base/16 y DenseNet-121

Resultados actualizados del experimento de generalización cross-dataset (congelado /
linear probe, checkpoint de partida Mendeley), reducidos a los 2 modelos elegidos del
proyecto: **ViT-Base/16** y **DenseNet-121**.

## Dominios evaluados

- **Trucha**: 1,150 imágenes (OneDrive), dominio objetivo del proyecto.
- **Rohu**: dominio objetivo del proyecto, evaluado bajo el mismo protocolo que Trucha.

## Contenido

| Carpeta | Contenido |
|---|---|
| `1_codigo/` | Código de entrenamiento/evaluación: `finetune_target_x2.py`, `experimento_dataset_mixto.py`, `domains.py`, `eval_generalizacion.py`, `detectar_duplicados.py`, `linear_probe_confmat_roc.py`, más las utilidades compartidas del proyecto (`config.py`, `dataset_freshness.py`, `metrics.py`, `models.py`, `run_freshness_v3.py`) |
| `2_logs/` | Logs de entrenamiento (congelado, con y sin augmentación x2) para Trucha y Rohu |
| `3_resultados/` | CSV de resultados (resumen y por fold) para Trucha y Rohu |
| `Resultados_ViT_DenseNet.docx` | Informe con la tabla comparativa final, solo ViT-Base/16 y DenseNet-121 |

## Método

Congelado (linear probe): backbone congelado (pesos fijos, checkpoint de Mendeley), solo
se entrena la cabeza de clasificación. 30 épocas, patience=8, fold aleatorio (5 folds).
Se evalúa con y sin augmentación x2 (cada imagen de entrenamiento aparece 2 veces por
época, con una transformación aleatoria independiente).

## Reproducir

```bash
# Trucha
python finetune_target_x2.py --dominio onedrive_trucha --congelado --epochs 30 --patience 8
python finetune_target_x2.py --dominio onedrive_trucha --congelado --epochs 30 --patience 8 --sin-x2

# Rohu
python experimento_dataset_mixto.py --frac-trucha 0.7 --frac-rohu 0.3 --epochs 30 --patience 8
python experimento_dataset_mixto.py --frac-trucha 0.7 --frac-rohu 0.3 --epochs 30 --patience 8 --sin-x2
```
