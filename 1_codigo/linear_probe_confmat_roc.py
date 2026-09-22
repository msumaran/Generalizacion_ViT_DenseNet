"""Matriz de confusion + curva ROC para el metodo del reporte (congelado /
linear probe, 30 epocas, con x2 -- seccion 10 de Informe_Resultados.md),
en Trucha y en el dataset combinado (50% Trucha + 50% Rohu).

A diferencia de finetune_target_x2.py / experimento_dataset_mixto.py (que
solo guardan F1/accuracy agregado por fold), esto vuelve a entrenar cada
fold guardando las predicciones y probabilidades del MEJOR epoch (el que
dio el F1 reportado en la seccion 10) para poder construir:
  - matriz de confusion por fold + pooled (los 5 folds juntos)
  - curva ROC + AUC por fold + pooled

Mismo patron que dnc_5fold_confmat_roc.py (usado para Mendeley), adaptado
a que aca no hay checkpoints ya entrenados que recargar -- el linear probe
se entrena en el momento, así que "pooled" no es literalmente "nunca visto
por el checkpoint" en el mismo sentido (cada fold parte del checkpoint
Mendeley, no de un checkpoint ya fine-tuneado), pero sigue siendo la union
de 5 particiones disjuntas de validacion.

Uso: python linear_probe_confmat_roc.py [--epochs 30] [--patience 8]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectar_duplicados import grupos_casi_duplicados
from domains import DOMAINS
from eval_generalizacion import CKPT_DIR, MODELOS
from experimento_dataset_mixto import construir_dataset_mixto
from finetune_target_x2 import finetune_eval_fold
from metrics import compute_metrics

N_FOLDS = 5
SEED = 42
CLASS_NAMES = ["fresco", "no_fresco"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "resultados", "congelado_30ep", "confmat_roc")


def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(fpr, tpr, auc, title, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def splits_trucha():
    paths, labels = DOMAINS["onedrive_trucha"]["loader"]()
    labels = np.array(labels); paths = np.array(paths, dtype=object)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return paths, labels, list(skf.split(np.zeros(len(labels)), labels))


def splits_combinado():
    paths, labels, origen, grupos = construir_dataset_mixto()
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return paths, labels, list(skf.split(np.zeros(len(labels)), labels, groups=grupos))


def process_model(model_name, domain_key, paths, labels, splits, epochs, patience,
                  usar_x2=True, out_dir=OUT_DIR):
    model_dir = os.path.join(out_dir, domain_key, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Resume barato: si ya existe el pooled de una corrida anterior (p.ej.
    # cortada por un reinicio de sesion), no repetir este modelo/dominio --
    # se reconstruyen las metricas desde los CSV de cm/roc ya guardados.
    pooled_csv = os.path.join(model_dir, "confusion_matrix_pooled.csv")
    if os.path.exists(pooled_csv):
        print(f"  {model_name}/{domain_key}: ya completo (pooled encontrado), se omite.", flush=True)
        rows = []
        for fold_idx in list(range(1, N_FOLDS + 1)) + ["pooled"]:
            suf = f"fold{fold_idx}" if fold_idx != "pooled" else "pooled"
            cm = pd.read_csv(os.path.join(model_dir, f"confusion_matrix_{suf}.csv"), index_col=0).values
            roc = pd.read_csv(os.path.join(model_dir, f"roc_curve_{suf}.csv"))
            n = int(cm.sum())
            acc = (cm[0, 0] + cm[1, 1]) / n * 100
            f1 = np.mean([2 * cm[i, i] / (2 * cm[i, i] + cm[i, 1 - i] + cm[1 - i, i] + 1e-9)
                          for i in (0, 1)]) * 100
            auc = np.trapz(roc["tpr"], roc["fpr"])
            rows.append({"model": model_name, "domain": domain_key, "fold": fold_idx, "n_val": n,
                         "f1": round(f1, 2), "accuracy": round(acc, 2), "auc": round(float(auc), 4),
                         "epochs_run": ""})
        return rows

    fold_rows = []
    all_trues, all_preds, all_probs = [], [], []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits, start=1):
        ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_fold{fold_idx}_best.pt")
        f1, acc, epochs_run, (trues, preds, probs) = finetune_eval_fold(
            model_name, ckpt_path, paths[tr_idx], labels[tr_idx], paths[val_idx], labels[val_idx],
            usar_x2=usar_x2, congelado=True, epochs=epochs, patience_max=patience,
            return_predictions=True)

        auc = roc_auc_score(trues, probs)
        cm = confusion_matrix(trues, preds, labels=[0, 1])
        fpr, tpr, thr = roc_curve(trues, probs)

        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            os.path.join(model_dir, f"confusion_matrix_fold{fold_idx}.csv"))
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
            os.path.join(model_dir, f"roc_curve_fold{fold_idx}.csv"), index=False)

        fold_rows.append({"model": model_name, "domain": domain_key, "fold": fold_idx,
                           "n_val": len(val_idx), "f1": round(f1, 2), "accuracy": round(acc, 2),
                           "auc": round(auc, 4), "epochs_run": epochs_run})
        print(f"  {model_name}/{domain_key} fold{fold_idx}/5: F1={f1:.2f} Acc={acc:.2f} "
              f"AUC={auc:.4f} epochs={epochs_run}", flush=True)

        all_trues.append(trues); all_preds.append(preds); all_probs.append(probs)

    trues = np.concatenate(all_trues); preds = np.concatenate(all_preds); probs = np.concatenate(all_probs)
    m = compute_metrics(trues, preds, 2)
    auc = roc_auc_score(trues, probs)
    cm = confusion_matrix(trues, preds, labels=[0, 1])
    fpr, tpr, thr = roc_curve(trues, probs)

    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(model_dir, "confusion_matrix_pooled.csv"))
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
        os.path.join(model_dir, "roc_curve_pooled.csv"), index=False)
    plot_confusion_matrix(cm, f"{model_name} ({domain_key}) — Matriz de confusión (pooled 5 folds)",
                           os.path.join(model_dir, "confusion_matrix_pooled.png"))
    plot_roc(fpr, tpr, auc, f"{model_name} ({domain_key}) — Curva ROC (pooled 5 folds)",
             os.path.join(model_dir, "roc_curve_pooled.png"))

    pooled_row = {"model": model_name, "domain": domain_key, "fold": "pooled", "n_val": len(trues),
                  "f1": round(m["macro_f1"], 2), "accuracy": round(m["accuracy"], 2),
                  "auc": round(auc, 4), "epochs_run": ""}
    print(f"  {model_name}/{domain_key} POOLED: F1={m['macro_f1']:.2f} Acc={m['accuracy']:.2f} "
          f"AUC={auc:.4f}", flush=True)
    return fold_rows + [pooled_row]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--sin-x2", action="store_true",
                     help="Sin duplicacion x2 (mismo presupuesto de epocas). Guarda en una "
                          "subcarpeta aparte para no mezclar con la corrida con x2.")
    args = ap.parse_args()
    usar_x2 = not args.sin_x2
    out_dir = OUT_DIR if usar_x2 else os.path.join(os.path.dirname(OUT_DIR), "confmat_roc_sinx2")

    os.makedirs(out_dir, exist_ok=True)
    dominios = {
        "trucha": splits_trucha,
        "combinado": splits_combinado,
    }
    all_rows = []
    for domain_key, split_fn in dominios.items():
        paths, labels, splits = split_fn()
        print(f"\n{'=' * 90}\nDOMINIO: {domain_key}  n={len(paths)}  usar_x2={usar_x2}\n{'=' * 90}", flush=True)
        for model_name in MODELOS:
            print("-" * 90, flush=True)
            print(f"MODELO: {model_name}", flush=True)
            all_rows.extend(process_model(model_name, domain_key, paths, labels, splits,
                                          args.epochs, args.patience, usar_x2=usar_x2, out_dir=out_dir))

    out_csv = os.path.join(out_dir, "resumen_confmat_roc.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nGuardado: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
