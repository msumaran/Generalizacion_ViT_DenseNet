"""Evaluacion cross-dataset (zero-shot) de los 3 modelos elegidos del
proyecto DNC 5-fold, aplicando la metodologia de la seccion 4.2 y del
diseno experimental del plan de tesis (Plan_de_Tesis_Ilan_Horna_v10.doc):

  Dominio FUENTE:    Mendeley Data   (ya usado para entrenar los checkpoints)
  Dominio OBJETIVO 1: OneDrive - Trucha
  Dominio OBJETIVO 2: Kaggle - Rohu (solo ojos)

Para cada uno de los 3 modelos (vit_base_patch16_224, swin_base_patch4_window7_224,
densenet121) se recargan sus 5 checkpoints YA ENTRENADOS (fold 1-5, entrenados
sobre Mendeley en el proyecto anterior) y se evaluan, SIN REENTRENAR, contra
las imagenes completas de Trucha y de Rohu-ojos (evaluacion zero-shot: el
modelo nunca vio estas imagenes).

Metricas calculadas por (modelo, fold, dominio) -- Dimension 1: Eficacia
Predictiva del plan de tesis (Ecuaciones 33-37): Accuracy, Precision, Recall,
F1 (macro), Especificidad; mas Tiempo de Inferencia (ms/imagen, variable
complementaria de Rapidez).

TODO se registra en logs (consola=INFO, archivo=DEBUG con el detalle de cada
imagen evaluada) y se guarda en CSV para trazabilidad total.

Uso: python eval_generalizacion.py
"""
import csv
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from dataset_freshness import FreshnessDataset
from metrics import compute_metrics
from models import build_model, disable_fused_attention, is_vit
from run_freshness_v3 import IMAGE_SIZE_CNN, IMAGE_SIZE_VIT, V3_CFG, get_transforms_v3

from domains import DOMAINS

MODELOS = ["vit_base_patch16_224", "swin_base_patch4_window7_224", "densenet121", "efficientnet_b0"]
# Copia local (Git LFS) de los 15 checkpoints de los 3 modelos elegidos, para
# que el proyecto sea autocontenido al clonar este repo por separado -- si no
# existe (ej. clon sin `git lfs pull`), cae al proyecto padre como respaldo.
_CKPT_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_freshness_v3_4k_5fold")
_CKPT_PADRE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "checkpoints_freshness_v3_4k_5fold")
CKPT_DIR = _CKPT_LOCAL if os.path.isdir(_CKPT_LOCAL) else _CKPT_PADRE
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
NUM_CLASSES = 2
CLASS_NAMES = ["fresco", "no_fresco"]

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

dev = CFG.device


def setup_logger():
    """Crea un logger con 2 salidas: archivo (nivel DEBUG, incluye cada
    imagen evaluada individualmente) y consola (nivel INFO, solo resumen).
    Un archivo de log nuevo por cada corrida (timestamp en el nombre)."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"eval_generalizacion_{ts}.log")
    logger = logging.getLogger("generalizacion")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # archivo: TODO, incluyendo cada imagen (detalle completo)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)  # consola: resumen legible
    ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_path


log, LOG_PATH = setup_logger()


@torch.no_grad()
def evaluate_on_domain(model, model_name, paths, labels, domain_key, fold):
    """Corre el modelo (ya en eval()) sobre TODAS las imagenes del dominio.
    Devuelve trues, preds, probs y tiempo total de inferencia (segundos)."""
    img_size = IMAGE_SIZE_VIT if is_vit(model_name) else IMAGE_SIZE_CNN
    bs = V3_CFG["batch_size_vit"] if is_vit(model_name) else V3_CFG["batch_size_cnn"]
    loader = DataLoader(FreshnessDataset(paths, labels, get_transforms_v3(img_size, False)),
                         batch_size=bs, shuffle=False, num_workers=2)

    trues, preds, probs = [], [], []
    n_done = 0
    t0 = time.perf_counter()
    for bi, (imgs, lbls) in enumerate(loader):
        logits = model(imgs.to(dev))
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pr = logits.argmax(1).cpu().numpy()
        probs.extend(p)
        preds.extend(pr)
        trues.extend(lbls.numpy())

        for j in range(len(lbls)):
            idx = n_done + j
            log.debug(f"  [{model_name}][fold{fold}][{domain_key}] img={paths[idx]}  "
                      f"real={CLASS_NAMES[lbls[j].item()]}  pred={CLASS_NAMES[pr[j]]}  "
                      f"prob_no_fresco={p[j]:.4f}")
        n_done += len(lbls)
        if (bi + 1) % 20 == 0:
            log.info(f"    ... {n_done}/{len(paths)} imagenes procesadas ({domain_key})")
    t_total = time.perf_counter() - t0
    log.info(f"    Completado {domain_key}: {n_done} imagenes en {t_total:.1f}s "
              f"({t_total/n_done*1000:.2f} ms/imagen)")
    return np.array(trues), np.array(preds), np.array(probs), t_total


def main():
    """Flujo completo: por cada uno de los 3 modelos elegidos, recarga sus
    5 checkpoints (entrenados sobre Mendeley en el proyecto DNC 5-fold) y
    evalua cada uno -- SIN reentrenar -- contra Trucha y Rohu completos.
    Guarda resultados por fold Y un resumen (media +/- desv.std de los 5
    folds) por dominio objetivo. No se "poolea" (concatenar) como en el
    proyecto DNC 5-fold, porque aqui los 5 folds evaluan las MISMAS
    imagenes del dominio objetivo repetidamente -- concatenar duplicaria
    el N artificialmente en vez de representar 8,000 imagenes unicas."""
    log.info("=" * 90)
    log.info("EVALUACION CROSS-DATASET (zero-shot) -- 3 modelos elegidos vs Trucha (OneDrive) y Rohu (Kaggle)")
    log.info(f"Dispositivo: {dev}")
    log.info(f"Log completo (incluye cada imagen evaluada, nivel DEBUG): {LOG_PATH}")
    log.info("=" * 90)

    log.info("Cargando dominios objetivo (OneDrive-Trucha, Kaggle-Rohu ojos)...")
    trucha_paths, trucha_labels = DOMAINS["onedrive_trucha"]["loader"]()
    rohu_paths, rohu_labels = DOMAINS["kaggle_rohu"]["loader"]()
    log.info(f"  OneDrive-Trucha: {len(trucha_paths)} imagenes "
             f"(fresco={trucha_labels.count(0)}, no_fresco={trucha_labels.count(1)})")
    log.info(f"  Kaggle-Rohu (ojos): {len(rohu_paths)} imagenes "
             f"(fresco={rohu_labels.count(0)}, no_fresco={rohu_labels.count(1)})")

    targets = {
        "onedrive_trucha": (trucha_paths, trucha_labels),
        "kaggle_rohu": (rohu_paths, rohu_labels),
    }

    fold_rows = []
    resumen_rows = []

    for model_name in MODELOS:
        log.info("-" * 90)
        log.info(f"MODELO: {model_name}")
        net = build_model(model_name, NUM_CLASSES, pretrained=False)
        if is_vit(model_name):
            disable_fused_attention(net)
        net.to(dev)

        # NOTA METODOLOGICA: aqui NO se "poolea" (concatenar) como en el
        # proyecto DNC 5-fold, porque los 5 folds NO son subconjuntos
        # disjuntos de Trucha/Rohu -- son 5 checkpoints DISTINTOS evaluados
        # cada uno contra las MISMAS imagenes completas del dominio objetivo.
        # Concatenar duplicaria artificialmente el N. Lo correcto es
        # reportar promedio +/- desviacion estandar del F1 (y demas
        # metricas) entre los 5 folds -- misma convencion de "estabilidad
        # entre folds" ya usada en el proyecto DNC 5-fold.
        metrics_per_target = {tk: {"accuracy": [], "precision": [], "recall": [], "f1": [],
                                    "specificity": [], "ms_per_img": []} for tk in targets}

        for fold in range(1, 6):
            ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_fold{fold}_best.pt")
            if not os.path.exists(ckpt_path):
                log.warning(f"  Fold {fold}: checkpoint no encontrado ({ckpt_path}), se omite.")
                continue
            log.info(f"  Fold {fold}: cargando checkpoint {ckpt_path}")
            net.load_state_dict(torch.load(ckpt_path, map_location=dev))
            net.eval()

            for target_key, (paths, labels) in targets.items():
                log.info(f"  Fold {fold} -> evaluando contra {target_key} ({len(paths)} imagenes)")
                trues, preds, probs, t_total = evaluate_on_domain(net, model_name, paths, labels, target_key, fold)
                m = compute_metrics(trues, preds, NUM_CLASSES)
                ms_per_img = t_total / len(paths) * 1000

                row = {
                    "model": model_name, "fold": fold, "target_domain": target_key,
                    "n_images": len(paths),
                    "accuracy": round(m["accuracy"], 2), "precision": round(m["macro_precision"], 2),
                    "recall": round(m["macro_recall"], 2), "f1": round(m["macro_f1"], 2),
                    "specificity": round(m["macro_specificity"], 2),
                    "tiempo_inferencia_ms_img": round(ms_per_img, 3),
                }
                fold_rows.append(row)
                log.info(f"    RESULTADO fold{fold} {target_key}: acc={m['accuracy']:.2f} "
                         f"f1={m['macro_f1']:.2f} tiempo={ms_per_img:.2f}ms/img")

                mt = metrics_per_target[target_key]
                mt["accuracy"].append(m["accuracy"]); mt["precision"].append(m["macro_precision"])
                mt["recall"].append(m["macro_recall"]); mt["f1"].append(m["macro_f1"])
                mt["specificity"].append(m["macro_specificity"]); mt["ms_per_img"].append(ms_per_img)

        for target_key in targets:
            mt = metrics_per_target[target_key]
            if not mt["f1"]:
                continue
            n_folds = len(mt["f1"])
            srow = {"model": model_name, "target_domain": target_key, "n_folds": n_folds,
                    "n_imagenes_por_fold": len(targets[target_key][0])}
            for metric in ("accuracy", "precision", "recall", "f1", "specificity", "ms_per_img"):
                vals = np.array(mt[metric])
                srow[f"{metric}_mean"] = round(float(vals.mean()), 2)
                srow[f"{metric}_std"] = round(float(vals.std(ddof=1)) if n_folds > 1 else 0.0, 2)
            resumen_rows.append(srow)
            log.info(f"  RESUMEN (media +/- std de {n_folds} folds) {model_name} -> {target_key}: "
                     f"F1={srow['f1_mean']:.2f}+/-{srow['f1_std']:.2f}  "
                     f"tiempo={srow['ms_per_img_mean']:.2f}ms/img")

    fold_csv = os.path.join(OUT_DIR, "resultados_por_fold_cross_dataset.csv")
    resumen_csv = os.path.join(OUT_DIR, "resultados_resumen_cross_dataset.csv")
    with open(fold_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        w.writeheader(); w.writerows(fold_rows)
    with open(resumen_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen_rows[0].keys()))
        w.writeheader(); w.writerows(resumen_rows)

    log.info("=" * 90)
    log.info(f"Guardado: {fold_csv}  ({len(fold_rows)} filas)")
    log.info(f"Guardado: {resumen_csv}  ({len(resumen_rows)} filas)")
    log.info("EVALUACION COMPLETADA")
    log.info("=" * 90)


if __name__ == "__main__":
    main()
