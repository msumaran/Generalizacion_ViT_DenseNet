"""Fine-tuning COMPLETO (todos los pesos, no solo la cabeza) de los 4 modelos
sobre Trucha y Rohu, partiendo del checkpoint ya entrenado en Mendeley, con
augmentacion x2 en el set de entrenamiento de cada fold.

Esto DEJA DE SER zero-shot: el modelo ve y aprende de imagenes del dominio
objetivo. Es un experimento de adaptacion de dominio, complementario al
resultado zero-shot ya reportado en Informe_Resultados.md -- no lo
reemplaza.

Metodologia:
  - 5-fold CV estratificado DENTRO de cada dominio objetivo (Trucha, Rohu),
    usando las imagenes ya filtradas por domains.py (mismo filtro de
    calidad y mismo recorte ya corregido que en la evaluacion zero-shot).
  - Rohu usa StratifiedGroupKFold agrupando casi-duplicados (fotos en
    rafaga, ver detectar_duplicados.py) para que un fold nunca entrene con
    una foto y valide con su casi-gemela -- la primera corrida (sin esto)
    dio F1~100% en los 4 modelos por exactamente ese motivo (confirmado:
    22,291 pares casi-identicos cruzaban fold de train/val). Trucha sigue
    con StratifiedKFold normal (solo 23 pares asi, de 1,150 imagenes --
    fuga despreciable, no justifica perder el balance de clase exacto).
  - Cada fold arranca desde el checkpoint Mendeley del MISMO indice de fold
    (fold 3 de Trucha parte del checkpoint fold3, etc.) -- consistente con
    el resto del proyecto, que siempre trata los 5 checkpoints como 5
    modelos independientes.
  - "Augmentacion x2": cada imagen de entrenamiento aparece 2 veces por
    epoca en el loader, con la transformacion de entrenamiento YA
    estocastica de run_freshness_v3.get_transforms_v3 (flip/rotacion/color/
    blur/erasing aleatorios en cada llamada) -- es decir, 2 vistas
    aumentadas independientes por imagen por epoca, no una copia estatica
    en disco. El set de validacion (holdout del fold) nunca se aumenta ni
    se duplica, para no inflar la metrica.
  - Todos los pesos se descongelan (fine-tuning completo), con LR
    diferencial: mas bajo para el backbone (ya visto Mendeley, no romperlo
    de golpe) y mas alto para la cabeza (nueva tarea/dominio).

Uso: python finetune_target_x2.py
"""
import csv
import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from detectar_duplicados import grupos_casi_duplicados
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from dataset_freshness import FreshnessDataset
from metrics import compute_metrics
from models import build_model, disable_fused_attention, is_vit
from run_freshness_v3 import IMAGE_SIZE_CNN, IMAGE_SIZE_VIT, V3_CFG, get_transforms_v3

from domains import DOMAINS
from eval_generalizacion import CKPT_DIR, MODELOS, NUM_CLASSES

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
os.makedirs(OUT_DIR, exist_ok=True)
dev = CFG.device

N_FOLDS = 5
SEED = 42
FT_EPOCHS = 12
FT_PATIENCE = 4
LR_BACKBONE = 2e-5
LR_HEAD = 1e-4
OUT_CSV = os.path.join(OUT_DIR, "finetune_x2_por_fold.csv")


def finetune_eval_fold(model_name, base_ckpt_path, tr_paths, tr_labels, val_paths, val_labels,
                       usar_x2=True, congelado=False, epochs=None, patience_max=None,
                       return_predictions=False, normalizar_features=False):
    # Bajo congelado, las features tienen escala mucho mas chica que el
    # gradiente que reciben con fine-tuning completo (confirmado: std de
    # feature de EfficientNet-B0 ~0.17 vs ~0.52 de DenseNet-121) -- con la
    # misma LR_HEAD, converge mucho mas lento y FT_EPOCHS=12 le corta el
    # entrenamiento antes de llegar a su techo real (verificado: a la
    # epoca 40 seguia subiendo, sin haber llegado al ~92 que muestra una
    # regresion logistica sobre las mismas features). epochs/patience_max
    # permiten un presupuesto mayor sin tocar el default usado por
    # fine-tuning completo (que si converge bien en 12 epocas).
    epochs = epochs if epochs is not None else FT_EPOCHS
    patience_max = patience_max if patience_max is not None else FT_PATIENCE
    img_size = IMAGE_SIZE_VIT if is_vit(model_name) else IMAGE_SIZE_CNN
    bs = V3_CFG["batch_size_vit"] if is_vit(model_name) else V3_CFG["batch_size_cnn"]

    model = build_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(base_ckpt_path, map_location="cpu"))
    if is_vit(model_name):
        disable_fused_attention(model)
    model.to(dev)
    for p in model.parameters():
        p.requires_grad = True

    head_ids = {id(p) for p in model.get_classifier().parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params = [p for p in model.parameters() if id(p) in head_ids]

    # normalizar_features: arreglo de raiz para la convergencia lenta de
    # EfficientNet-B0 bajo congelado (seccion 9.8.1 del informe) -- en vez
    # de compensar con mas epocas, se pareja la ESCALA de la feature que
    # llega a la cabeza (std de EfficientNet-B0 ~0.17 vs DenseNet-121
    # ~0.52, confirmado empiricamente), para que el gradiente hacia la
    # cabeza sea comparable entre modelos con el mismo LR_HEAD.
    # affine=False: solo normaliza media/varianza (estadisticas del batch,
    # sin parametros entrenables) -- no agrega grados de libertad nuevos
    # que compitan con la cabeza real, solo reescala lo que ya sale del
    # backbone congelado. Debe estar en train() junto con la cabeza para
    # que sus running stats se calibren con datos del dominio objetivo.
    feat_bn = None
    if congelado and normalizar_features:
        feat_dim = model.get_classifier().in_features
        feat_bn = nn.BatchNorm1d(feat_dim, affine=False).to(dev)

    def forward(imgs):
        if feat_bn is None:
            return model(imgs)
        feat = model.forward_features(imgs)
        pooled = model.forward_head(feat, pre_logits=True)
        return model.get_classifier()(feat_bn(pooled))

    if congelado:
        # Linear probe: el backbone queda tal cual salio de Mendeley, solo
        # se entrena la cabeza de clasificacion. Menos parametros libres =
        # menos capacidad de adaptarse al dominio, pero tambien menos
        # inestabilidad de entrenamiento (ver seccion 9.5 del informe, que
        # documenta la varianza run-a-run del fine-tuning completo).
        for p in backbone_params:
            p.requires_grad = False
        optimizer = torch.optim.AdamW(head_params, lr=LR_HEAD, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    # x2: cada ruta de train aparece 2 veces -> 2 augmentaciones independientes
    # por imagen y por epoca (get_transforms_v3(train=True) ya es estocastico).
    # usar_x2=False: cada imagen aparece 1 sola vez (sigue habiendo augmentacion
    # estocastica de todos modos, solo no se duplica el tamano del set).
    mult = 2 if usar_x2 else 1
    tr_paths_x2 = list(tr_paths) * mult
    tr_labels_x2 = list(tr_labels) * mult

    train_tf = get_transforms_v3(img_size, train=True)
    val_tf = get_transforms_v3(img_size, train=False)

    tr_loader = DataLoader(FreshnessDataset(tr_paths_x2, tr_labels_x2, train_tf),
                            batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(FreshnessDataset(list(val_paths), list(val_labels), val_tf),
                             batch_size=bs, shuffle=False, num_workers=0)

    best_f1, best_acc, patience, epochs_run = -1.0, 0.0, 0, 0
    best_preds = None
    for epoch in range(epochs):
        if congelado:
            # Backbone en eval(): sin esto, BatchNorm (DenseNet/EfficientNet)
            # seguiria actualizando sus running stats con estadisticas del
            # dominio objetivo aunque los pesos no reciban gradiente -- eso
            # ya no seria "backbone congelado", seria adaptacion parcial.
            model.eval()
            model.get_classifier().train()
            if feat_bn is not None:
                feat_bn.train()
        else:
            model.train()
        for imgs, lbls in tr_loader:
            imgs, lbls = imgs.to(dev), lbls.to(dev)
            optimizer.zero_grad()
            loss = loss_fn(forward(imgs), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()
        epochs_run = epoch + 1

        model.eval()
        if feat_bn is not None:
            feat_bn.eval()
        all_p, all_l, all_prob = [], [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                logits = forward(imgs.to(dev))
                all_prob.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                all_p.extend(logits.argmax(1).cpu().numpy())
                all_l.extend(lbls.numpy())
        m = compute_metrics(np.array(all_l), np.array(all_p), NUM_CLASSES)
        if m["macro_f1"] > best_f1:
            best_f1, best_acc, patience = m["macro_f1"], m["accuracy"], 0
            if return_predictions:
                best_preds = (np.array(all_l), np.array(all_p), np.array(all_prob))
        else:
            patience += 1
            if patience >= patience_max:
                break

    # el allocator de MPS cachea memoria y no la devuelve al SO al terminar
    # esta funcion -- sin este cleanup explicito, 40 llamadas seguidas
    # (4 modelos x 2 dominios x 5 folds) acumulan swap hasta trabar la
    # maquina (visto en la corrida anterior: 11.4/12GB de swap en uso).
    del model, optimizer, scheduler, tr_loader, val_loader
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if return_predictions:
        return best_f1, best_acc, epochs_run, best_preds
    return best_f1, best_acc, epochs_run


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-x2", action="store_true",
                     help="Desactiva la duplicacion x2 del set de entrenamiento (sigue habiendo "
                          "augmentacion estocastica normal, solo no se dobla el tamano del set). "
                          "Usado para comprobar si el F1~100%% en Rohu lo causa la duplicacion x2 "
                          "o la fuga de casi-duplicados entre folds (ver detectar_duplicados.py) "
                          "-- la hipotesis es que es lo segundo, y sin x2 el resultado deberia "
                          "seguir siendo ~100%% en Rohu.")
    ap.add_argument("--dominio", choices=["onedrive_trucha", "kaggle_rohu", "todos"], default="todos")
    ap.add_argument("--congelado", action="store_true",
                     help="Linear probe: congela el backbone (requires_grad=False, y en eval() "
                          "durante el entrenamiento para no mover BatchNorm), entrena solo la "
                          "cabeza de clasificacion. Contraste con el fine-tuning completo -- menos "
                          "capacidad de adaptarse, pero sin la inestabilidad de mover todos los "
                          "pesos (seccion 9.5 del informe).")
    ap.add_argument("--modelo", choices=MODELOS, default=None,
                     help="Corre un solo modelo en vez de los 4. Util para re-correr un modelo "
                          "puntual con --epochs/--patience distintos sin pisar el resto del CSV.")
    ap.add_argument("--epochs", type=int, default=None,
                     help="Override de FT_EPOCHS. Uso tipico: EfficientNet-B0 bajo --congelado "
                          "converge mucho mas lento que los demas modelos (features de escala "
                          "~3x mas chica -> gradientes mas chicos a la cabeza) y con FT_EPOCHS=12 "
                          "queda muy por debajo de su techo real (confirmado: a la epoca 40 seguia "
                          "subiendo). Sin este flag se usa FT_EPOCHS como siempre.")
    ap.add_argument("--patience", type=int, default=None,
                     help="Override de FT_PATIENCE (subir junto con --epochs para no cortar el "
                          "entrenamiento largo por una meseta transitoria).")
    ap.add_argument("--normalizar-features", action="store_true",
                     help="Arreglo de raiz alternativo a --epochs para la convergencia lenta de "
                          "EfficientNet-B0 bajo congelado: inserta un BatchNorm1d(affine=False) "
                          "entre las features del backbone (pre_logits) y la cabeza, para que "
                          "todos los modelos le entreguen features en una escala comparable a la "
                          "cabeza (en vez de compensar con mas epocas). Solo tiene efecto con "
                          "--congelado.")
    args = ap.parse_args()
    usar_x2 = not args.sin_x2
    dominios = ["onedrive_trucha", "kaggle_rohu"] if args.dominio == "todos" else [args.dominio]
    # kaggle_rohu ahora usa StratifiedGroupKFold (dedup) -- folds distintos a
    # los de finetune_x2_por_fold.csv/finetune_sinx2_por_fold.csv (que usaban
    # StratifiedKFold y por eso tuvieron fuga de casi-duplicados). Se guarda
    # aparte para no mezclar dos metodologias de split bajo el mismo nombre.
    sufijo_dedup = "_dedup" if dominios == ["kaggle_rohu"] else ""
    sufijo_congelado = "_congelado" if args.congelado else ""
    sufijo_norm = "_normfeat" if args.normalizar_features else ""
    base_name = "finetune_x2" if usar_x2 else "finetune_sinx2"
    out_csv = os.path.join(OUT_DIR, f"{base_name}{sufijo_congelado}{sufijo_norm}{sufijo_dedup}_por_fold.csv")

    print("=" * 90, flush=True)
    modo = "LINEAR PROBE (backbone congelado)" if args.congelado else "FINE-TUNING COMPLETO"
    print(f"{modo} {'+ AUGMENTACION x2' if usar_x2 else '(SIN augmentacion x2)'} "
          f"sobre {', '.join(dominios)} (NO zero-shot)", flush=True)
    print(f"Dispositivo: {dev}", flush=True)
    print("=" * 90, flush=True)

    rows = []
    done = set()
    write_header = not os.path.exists(out_csv)
    if not write_header:
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                r["fold"] = int(r["fold"])
                r["f1"] = float(r["f1"])
                rows.append(r)
                done.add((r["model"], r["target_domain"], r["fold"]))
        print(f"Reanudando: {len(done)} combinaciones (modelo,dominio,fold) ya calculadas, se omiten.", flush=True)
    fcsv = open(out_csv, "a", newline="")
    writer = None

    for domain_key in dominios:
        paths, labels = DOMAINS[domain_key]["loader"]()
        labels = np.array(labels)
        paths = np.array(paths, dtype=object)
        print(f"\n{domain_key}: n={len(paths)}  fresco={int((labels==0).sum())}  "
              f"no_fresco={int((labels==1).sum())}", flush=True)

        if domain_key == "kaggle_rohu":
            grupos = grupos_casi_duplicados(paths)
            skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            splits = list(skf.split(np.zeros(len(labels)), labels, groups=grupos))
        else:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            splits = list(skf.split(np.zeros(len(labels)), labels))

        for model_name in (MODELOS if args.modelo is None else [args.modelo]):
            print("-" * 90, flush=True)
            print(f"MODELO: {model_name}  DOMINIO: {domain_key}", flush=True)
            for fold_idx, (tr_idx, val_idx) in enumerate(splits, start=1):
                if (model_name, domain_key, fold_idx) in done:
                    print(f"  fold{fold_idx}: ya calculado (resumen), se omite.", flush=True)
                    continue
                ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_fold{fold_idx}_best.pt")
                if not os.path.exists(ckpt_path):
                    print(f"  fold{fold_idx}: checkpoint no encontrado, se omite.", flush=True)
                    continue
                t0 = time.perf_counter()
                f1, acc, epochs_run = finetune_eval_fold(
                    model_name, ckpt_path,
                    paths[tr_idx], labels[tr_idx], paths[val_idx], labels[val_idx],
                    usar_x2=usar_x2, congelado=args.congelado,
                    epochs=args.epochs, patience_max=args.patience,
                    normalizar_features=args.normalizar_features)
                dt = time.perf_counter() - t0
                row = {
                    "model": model_name, "target_domain": domain_key, "fold": fold_idx,
                    "n_train": len(tr_idx) * (2 if usar_x2 else 1), "n_val": len(val_idx),
                    "epochs_run": epochs_run, "f1": round(f1, 2), "accuracy": round(acc, 2),
                    "tiempo_s": round(dt, 1),
                }
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(fcsv, fieldnames=list(row.keys()))
                    if write_header:
                        writer.writeheader()
                writer.writerow(row)
                fcsv.flush()
                print(f"  fold{fold_idx}: F1={f1:.2f}  Acc={acc:.2f}  "
                      f"epochs={epochs_run}  tiempo={dt:.0f}s", flush=True)

    fcsv.close()
    print(f"\nGuardado incremental en: {out_csv}  ({len(rows)} filas)", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("RESUMEN (media de 5 folds)", flush=True)
    print("=" * 90, flush=True)
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["model"], r["target_domain"])].append(r["f1"])
    summary_rows = []
    for (model_name, domain_key), f1s in agg.items():
        mean_f1 = float(np.mean(f1s))
        print(f"{model_name:30s} {domain_key:16s} F1={mean_f1:.2f} (n_folds={len(f1s)})", flush=True)
        summary_rows.append({"model": model_name, "target_domain": domain_key,
                              "f1_mean": round(mean_f1, 2), "n_folds": len(f1s)})
    out_summary = os.path.join(OUT_DIR, f"{base_name}{sufijo_congelado}{sufijo_dedup}_resumen.csv")
    with open(out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nGuardado: {out_summary}", flush=True)


if __name__ == "__main__":
    main()
