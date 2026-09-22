"""
Evaluation metrics using exact thesis formulas:
  Accuracy    = (TP+TN) / (TP+TN+FP+FN)
  Precision   = TP/(TP+FP) * 100
  Recall      = TP/(TP+FN) * 100
  F1-Score    = 2 * (Precision * Recall) / (Precision + Recall)
  Specificity = TN/(TN+FP) * 100

Multiclass: one-vs-rest per class, then macro-average.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


EPS = 1e-10


def _binary_metrics(tp: float, tn: float, fp: float, fn: float) -> Dict[str, float]:
    accuracy    = (tp + tn) / (tp + tn + fp + fn + EPS)
    precision   = (tp / (tp + fp + EPS)) * 100.0
    recall      = (tp / (tp + fn + EPS)) * 100.0
    f1          = 2.0 * (precision * recall) / (precision + recall + EPS)
    specificity = (tn / (tn + fp + EPS)) * 100.0
    return dict(accuracy=accuracy, precision=precision,
                recall=recall, f1=f1, specificity=specificity)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    num_classes: int, probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Return macro-averaged metrics + per-class breakdown.

    probs: opcional, probabilidad softmax de la clase 1 (binario) para cada
    muestra. Si se pasa, se agrega result["auc"] (roc_auc_score). No cambia
    ninguna de las otras métricas (todas se calculan desde y_pred, no desde
    probs) ni el comportamiento de los llamados existentes que no lo pasan
    -- es puramente aditivo, por retrocompatibilidad con los 70+ usos de
    esta funcion en el proyecto."""
    per_class: List[Dict[str, float]] = []
    for c in range(num_classes):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        tn = float(np.sum((y_pred != c) & (y_true != c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        per_class.append(_binary_metrics(tp, tn, fp, fn))

    overall_acc = float(np.sum(y_pred == y_true)) / (len(y_true) + EPS)

    result: Dict = {"accuracy": overall_acc * 100.0}
    for key in ("precision", "recall", "f1", "specificity"):
        result[f"macro_{key}"] = float(np.mean([m[key] for m in per_class]))
    result["per_class"] = per_class
    if probs is not None:
        result["auc"] = float(roc_auc_score(y_true, probs))
    return result


def metrics_to_series(metrics: Dict[str, float], prefix: str = "") -> pd.Series:
    flat = {
        f"{prefix}accuracy":          metrics["accuracy"],
        f"{prefix}macro_precision":   metrics["macro_precision"],
        f"{prefix}macro_recall":      metrics["macro_recall"],
        f"{prefix}macro_f1":          metrics["macro_f1"],
        f"{prefix}macro_specificity": metrics["macro_specificity"],
    }
    return pd.Series(flat)


# ── Dimensión 2: Robustez ante el Cambio de Dominio ──────────────────────────

def tasa_retencion_desempeno(f1_source: float, f1_target: float) -> float:
    """TRD = (F1_target / F1_source) * 100

    f1_source: F1-Score Macro en dataset de entrenamiento (ej. DNC test)
    f1_target: F1-Score Macro en dataset objetivo (ej. ojos trucha)
    Retorna porcentaje de rendimiento retenido (100% = sin caída).
    """
    return (f1_target / (f1_source + EPS)) * 100.0


def brecha_error(f1_source: float, f1_target: float) -> float:
    """Gap = Error_target - Error_source, donde Error = 1 - F1/100"""
    error_source = 1.0 - f1_source / 100.0
    error_target = 1.0 - f1_target / 100.0
    return error_target - error_source


def reduccion_brecha_error(
    f1_source_baseline: float, f1_target_baseline: float,
    f1_source_propuesto: float, f1_target_propuesto: float,
) -> float:
    """RBE = (Gap_baseline - Gap_propuesto) / Gap_baseline * 100

    Mide cuánto reduce el modelo propuesto la degradación respecto al baseline.
    RBE > 0 → propuesto tiene menor brecha (mejor transferencia).
    RBE < 0 → propuesto tiene mayor brecha (peor transferencia).
    """
    gap_baseline  = brecha_error(f1_source_baseline,  f1_target_baseline)
    gap_propuesto = brecha_error(f1_source_propuesto, f1_target_propuesto)
    return (gap_baseline - gap_propuesto) / (gap_baseline + EPS) * 100.0


def compute_robustez(
    f1_source: float, f1_target: float,
    f1_source_baseline: float = None, f1_target_baseline: float = None,
) -> Dict[str, float]:
    """Calcula indicadores de Dimensión 2.

    Args:
        f1_source:          F1 Macro del modelo propuesto en dataset fuente (%)
        f1_target:          F1 Macro del modelo propuesto en dataset objetivo (%)
        f1_source_baseline: F1 Macro del baseline en dataset fuente (%) [opcional]
        f1_target_baseline: F1 Macro del baseline en dataset objetivo (%) [opcional]

    Returns:
        dict con TRD y opcionalmente RBE
    """
    result = {
        "trd": tasa_retencion_desempeno(f1_source, f1_target),
        "gap": brecha_error(f1_source, f1_target),
    }
    if f1_source_baseline is not None and f1_target_baseline is not None:
        result["rbe"] = reduccion_brecha_error(
            f1_source_baseline, f1_target_baseline,
            f1_source, f1_target,
        )
    return result
