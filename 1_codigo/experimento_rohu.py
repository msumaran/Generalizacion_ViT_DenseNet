"""Linear probe (backbone congelado) sobre el dominio Rohu.

Muestreo -- fraccion de cada pool de imagenes (--frac-trucha, --frac-rohu;
default 0.7/0.3, los valores del experimento presentado), estratificado
por clase (fresco/no_fresco), semilla fija para reproducibilidad.

Particion de folds -- StratifiedGroupKFold. Los grupos evitan la fuga de
casi-duplicados (ver detectar_duplicados.py): cada imagen del pool Rohu
recibe el id de su grupo de casi-duplicado, cada imagen del pool Trucha un
id propio (su tasa de duplicados es despreciable, 23/1150).

El checkpoint de partida es el de Mendeley (fold-a-fold, igual que el resto
del proyecto).

Uso: python experimento_rohu.py [--epochs 30] [--patience 8] [--sin-x2]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectar_duplicados import grupos_casi_duplicados
from domains import collect_rohu_eyes, collect_trucha
from eval_generalizacion import CKPT_DIR, MODELOS, NUM_CLASSES
from finetune_target_x2 import finetune_eval_fold

N_FOLDS = 5
SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")


def submuestra_estratificada(labels, frac, seed):
    """Indices que conservan la proporcion de clases del array original."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    idx = []
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        n = max(1, int(round(len(idx_c) * frac)))
        idx.extend(rng.choice(idx_c, n, replace=False))
    return np.array(sorted(idx))


def construir_dataset_rohu(frac_trucha=0.7, frac_rohu=0.3):
    t_paths, t_labels = collect_trucha()
    r_paths, r_labels = collect_rohu_eyes()

    t_idx = submuestra_estratificada(t_labels, frac_trucha, SEED)
    r_idx = submuestra_estratificada(r_labels, frac_rohu, SEED + 1)

    t_paths_s = [t_paths[i] for i in t_idx]
    t_labels_s = [t_labels[i] for i in t_idx]
    r_paths_s = [r_paths[i] for i in r_idx]
    r_labels_s = [r_labels[i] for i in r_idx]

    # Grupos para StratifiedGroupKFold: Rohu usa su grupo de casi-duplicado
    # (offset para no chocar con los ids de Trucha), Trucha usa un id propio
    # por imagen (singleton, no hay agrupamiento real).
    r_grupos = grupos_casi_duplicados(r_paths_s)
    r_grupos = r_grupos + 10_000_000  # offset, evita colision con ids de Trucha
    t_grupos = np.arange(len(t_paths_s))

    paths = np.array(t_paths_s + r_paths_s, dtype=object)
    labels = np.array(t_labels_s + r_labels_s)
    origen = np.array(["trucha"] * len(t_paths_s) + ["rohu"] * len(r_paths_s))
    grupos = np.concatenate([t_grupos, r_grupos])

    print(f"Rohu: {len(paths)} imagenes")
    print(f"  fresco={int((labels==0).sum())}  no_fresco={int((labels==1).sum())}")
    return paths, labels, origen, grupos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--sin-x2", action="store_true")
    ap.add_argument("--frac-trucha", type=float, default=0.7,
                     help="Fraccion del pool de Trucha a usar (default 0.7)")
    ap.add_argument("--frac-rohu", type=float, default=0.3,
                     help="Fraccion del pool de Rohu a usar (default 0.3)")
    args = ap.parse_args()
    usar_x2 = not args.sin_x2

    paths, labels, origen, grupos = construir_dataset_rohu(args.frac_trucha, args.frac_rohu)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(labels)), labels, groups=grupos))

    sufijo = "x2" if usar_x2 else "sinx2"
    out_csv = os.path.join(OUT_DIR, f"experimento_rohu_congelado_{sufijo}_por_fold.csv")
    fieldnames = ["model", "fold", "n_train", "n_val", "n_val_trucha", "n_val_rohu",
                  "epochs_run", "f1", "accuracy", "tiempo_s"]
    rows = []
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in MODELOS:
            print("-" * 90, flush=True)
            print(f"MODELO: {model_name}", flush=True)
            for fold_idx, (tr_idx, val_idx) in enumerate(splits, start=1):
                ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_fold{fold_idx}_best.pt")
                if not os.path.exists(ckpt_path):
                    print(f"  fold{fold_idx}: checkpoint no encontrado, se omite.", flush=True)
                    continue
                t0 = time.perf_counter()
                f1, acc, epochs_run = finetune_eval_fold(
                    model_name, ckpt_path,
                    paths[tr_idx], labels[tr_idx], paths[val_idx], labels[val_idx],
                    usar_x2=usar_x2, congelado=True,
                    epochs=args.epochs, patience_max=args.patience)
                dt = time.perf_counter() - t0
                row = {
                    "model": model_name, "fold": fold_idx,
                    "n_train": len(tr_idx) * (2 if usar_x2 else 1), "n_val": len(val_idx),
                    "n_val_trucha": int((origen[val_idx] == "trucha").sum()),
                    "n_val_rohu": int((origen[val_idx] == "rohu").sum()),
                    "epochs_run": epochs_run, "f1": round(f1, 2), "accuracy": round(acc, 2),
                    "tiempo_s": round(dt, 1),
                }
                rows.append(row)
                writer.writerow(row)
                fh.flush()
                print(f"  fold{fold_idx} (n_val={len(val_idx)}, "
                      f"{row['n_val_trucha']} trucha / {row['n_val_rohu']} rohu): "
                      f"F1={f1:.2f}  Acc={acc:.2f}  epochs={epochs_run}  tiempo={dt:.0f}s", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("RESUMEN (media de 5 folds)", flush=True)
    print("=" * 90, flush=True)
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[r["model"]].append(r["f1"])
    summary = []
    for model_name, f1s in agg.items():
        mean_f1 = float(np.mean(f1s))
        print(f"{model_name:30s} F1={mean_f1:.2f} (n_folds={len(f1s)})", flush=True)
        summary.append({"model": model_name, "f1_mean": round(mean_f1, 2), "n_folds": len(f1s)})
    out_summary = os.path.join(OUT_DIR, f"experimento_rohu_congelado_{sufijo}_resumen.csv")
    with open(out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nGuardado: {out_csv}\nGuardado: {out_summary}", flush=True)


if __name__ == "__main__":
    main()
