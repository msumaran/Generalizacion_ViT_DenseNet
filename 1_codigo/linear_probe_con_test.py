"""Linear probe (backbone congelado, sin augmentacion x2) con conjunto de PRUEBA separado.

Corrida APARTE: no reemplaza ni modifica los resultados ya presentados
(resultados/congelado_sinx2_*, resultados/congelado_30ep/confmat_roc_*). Sirve para medir
el desempeno con el epoch elegido en una validacion distinta del fold que se reporta.

Particion -- los mismos 5 folds de la corrida original (semilla 42):
  - Trucha: StratifiedKFold(5, shuffle=True, random_state=42)
  - Rohu (mixto 70% Trucha / 30% Rohu): StratifiedGroupKFold(5, ...) con grupos de casi-duplicados
El fold k es la PRUEBA (se evalua una sola vez). De los otros 4 folds se separa ~15% como
VALIDACION, que es la que elige el epoch:
  - Trucha: train_test_split(test_size=0.15, stratify, random_state=42), igual que Mendeley
  - Rohu: primer split de StratifiedGroupKFold(7, ...) sobre esos 4 folds (~14%), para que las
    fotos casi identicas no queden repartidas entre entrenamiento y validacion

Entrenamiento identico a finetune_target_x2.finetune_eval_fold(congelado=True): checkpoint
Mendeley del mismo fold, solo la cabeza entrena (AdamW lr 1e-4, wd 1e-4, CosineAnnealingLR,
CrossEntropyLoss), hasta 30 epochs, paciencia 8 sobre val_F1.

Guarda por modelo/dominio/fold (todo lo que la corrida original no guardaba):
  epochs_fold{k}.csv        val_F1 y val_accuracy de cada epoch
  head_fold{k}.pt           pesos de la cabeza elegida
  predicciones_fold{k}.csv  etiqueta real, prediccion y probabilidad de no_fresco en la prueba
  confusion_matrix_fold{k}.csv, roc_curve_fold{k}.csv, y los pooled
Reanuda: un fold con predicciones_fold{k}.csv ya guardado no se vuelve a entrenar.

Uso: python linear_probe_con_test.py [--epochs 30] [--patience 8] [--folds 5] [--protocolo con_test|original]
--protocolo original: mismo metodo que los resultados presentados (sin validacion aparte; el fold de evaluacion
elige el epoch y se reporta), pero guardando todo lo de arriba. Salida en original[_Nfold]/.
--folds 10: 10 folds de prueba (salida en con_test_10fold/). Solo hay 5 checkpoints Mendeley, asi que el
fold k parte del checkpoint ((k-1) mod 5)+1; ninguno vio Trucha ni Rohu, no hay fuga.
"""
import argparse
import copy
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from dataset_freshness import FreshnessDataset
from domains import DOMAINS
from eval_generalizacion import CKPT_DIR, NUM_CLASSES
from experimento_dataset_mixto import construir_dataset_mixto
from metrics import compute_metrics
from models import build_model, disable_fused_attention, is_vit
from run_freshness_v3 import IMAGE_SIZE_CNN, IMAGE_SIZE_VIT, V3_CFG, get_transforms_v3

MODELOS = ["vit_base_patch16_224", "densenet121"]
N_FOLDS, SEED, VAL_FRAC = 5, 42, 0.15
LR_HEAD = 1e-4
CLASES = ["fresco", "no_fresco"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", "congelado_30ep", "con_test")
dev = CFG.device


def particiones(dominio, protocolo="con_test"):
    """(paths, labels, [(train, val, test)] x 5) con los mismos folds de prueba que la corrida original."""
    if dominio == "trucha":
        paths, labels = DOMAINS["onedrive_trucha"]["loader"]()
        paths, labels = np.array(paths, dtype=object), np.array(labels)
        grupos = np.arange(len(labels))  # cada imagen es su propio grupo
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels)
    else:
        paths, labels, _, grupos = construir_dataset_mixto(0.7, 0.3)
        paths, labels = np.array(paths, dtype=object), np.array(labels)
        folds = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels)), labels, groups=grupos)
    out = []
    for resto, test in folds:
        if protocolo == "original":
            # protocolo de los resultados presentados: el fold de evaluacion tambien elige el epoch
            assert len(set(resto) | set(test)) == len(resto) + len(test) == len(labels)
            assert not set(grupos[resto]) & set(grupos[test])
            out.append((resto, test, test))
            continue
        if dominio == "trucha":
            train, val = train_test_split(resto, test_size=VAL_FRAC, stratify=labels[resto], random_state=SEED)
        else:
            sgkf = StratifiedGroupKFold(round(1 / VAL_FRAC), shuffle=True, random_state=SEED)
            i_tr, i_val = next(sgkf.split(np.zeros(len(resto)), labels[resto], groups=grupos[resto]))
            train, val = resto[i_tr], resto[i_val]
        # train / val / test disjuntos, cubren todo el fold, y ningun grupo cruza entre ellos
        assert len(train) + len(val) + len(test) == len(set(train) | set(val) | set(test)) == len(labels)
        g_tr, g_val, g_te = set(grupos[train]), set(grupos[val]), set(grupos[test])
        assert not (g_tr & g_val or g_tr & g_te or g_val & g_te)
        out.append((train, val, test))
    return paths, labels, out


def predecir(model, loader):
    trues, preds, probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            logits = model(imgs.to(dev))
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(lbls.numpy())
    return np.array(trues), np.array(preds), np.array(probs)


def entrenar_fold(model_name, fold, paths, labels, train, val, test, epochs, patience_max, carpeta):
    img = IMAGE_SIZE_VIT if is_vit(model_name) else IMAGE_SIZE_CNN
    bs = V3_CFG["batch_size_vit"] if is_vit(model_name) else V3_CFG["batch_size_cnn"]
    model = build_model(model_name, NUM_CLASSES, pretrained=False)
    ckpt = (fold - 1) % 5 + 1  # 5 checkpoints Mendeley; con 10 folds se reutilizan en ciclo
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{model_name}_fold{ckpt}_best.pt"), map_location="cpu"))
    if is_vit(model_name):
        disable_fused_attention(model)
    model.to(dev)
    cabeza = model.get_classifier()
    head_ids = {id(p) for p in cabeza.parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    optimizer = torch.optim.AdamW(cabeza.parameters(), lr=LR_HEAD, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    def loader(idx, train_tf):
        tf = get_transforms_v3(img, train=train_tf)
        return DataLoader(FreshnessDataset(list(paths[idx]), list(labels[idx]), tf), batch_size=bs,
                          shuffle=train_tf, num_workers=0)
    tr_loader, val_loader, te_loader = loader(train, True), loader(val, False), loader(test, False)

    filas, best_f1, best_state, best_epoch, paciencia = [], -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.eval()  # backbone congelado: BatchNorm no actualiza estadisticas
        cabeza.train()
        for imgs, lbls in tr_loader:
            optimizer.zero_grad()
            loss_fn(model(imgs.to(dev)), lbls.to(dev)).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        t, p, _ = predecir(model, val_loader)
        m = compute_metrics(t, p, NUM_CLASSES)
        if m["macro_f1"] > best_f1:
            best_f1, best_epoch, paciencia = m["macro_f1"], epoch, 0
            best_state = copy.deepcopy(cabeza.state_dict())
        else:
            paciencia += 1
        filas.append({"epoch": epoch, "val_f1": round(m["macro_f1"], 2), "val_accuracy": round(m["accuracy"], 2),
                      "mejor_hasta_ahora": round(best_f1, 2), "paciencia": paciencia})
        print(f"    {model_name} fold{fold} epoch {epoch:2d}: val_F1={m['macro_f1']:.2f} "
              f"mejor={best_f1:.2f} (epoch {best_epoch}) paciencia={paciencia}", flush=True)
        if paciencia >= patience_max:
            break

    cabeza.load_state_dict(best_state)
    torch.save(best_state, os.path.join(carpeta, f"head_fold{fold}.pt"))
    pd.DataFrame(filas).to_csv(os.path.join(carpeta, f"epochs_fold{fold}.csv"), index=False)
    trues, preds, probs = predecir(model, te_loader)  # unica evaluacion sobre la prueba
    pd.DataFrame({"real": trues, "prediccion": preds, "prob_no_fresco": probs}).to_csv(
        os.path.join(carpeta, f"predicciones_fold{fold}.csv"), index=False)

    del model, optimizer, scheduler, tr_loader, val_loader, te_loader
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return best_epoch, best_f1, len(filas)


def guardar_matriz_roc(trues, preds, probs, carpeta, sufijo):
    cm = confusion_matrix(trues, preds, labels=[0, 1])
    pd.DataFrame(cm, index=CLASES, columns=CLASES).to_csv(os.path.join(carpeta, f"confusion_matrix_{sufijo}.csv"))
    fpr, tpr, thr = roc_curve(trues, probs)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
        os.path.join(carpeta, f"roc_curve_{sufijo}.csv"), index=False)
    m = compute_metrics(trues, preds, NUM_CLASSES)
    return round(m["macro_f1"], 2), round(m["accuracy"], 2), round(roc_auc_score(trues, probs), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--protocolo", choices=["con_test", "original"], default="con_test",
                    help="original: sin validacion aparte, el fold de evaluacion elige el epoch (como lo presentado)")
    args = ap.parse_args()
    global N_FOLDS, OUT_DIR
    N_FOLDS = args.folds
    if args.protocolo == "original":
        OUT_DIR = OUT_DIR.replace("con_test", "original")
    if N_FOLDS != 5:
        OUT_DIR = f"{OUT_DIR}_{N_FOLDS}fold"
    print(f"LINEAR PROBE (sin x2)  dispositivo={dev}  epochs={args.epochs} "
          f"paciencia={args.patience} folds={N_FOLDS} protocolo={args.protocolo} salida={OUT_DIR}", flush=True)
    for dominio in ("trucha", "rohu"):
        paths, labels, splits = particiones(dominio, args.protocolo)
        print(f"\n{dominio}: n={len(labels)}  fresco={int((labels == 0).sum())}  no_fresco={int((labels == 1).sum())}",
              flush=True)
        resumen = []
        for model_name in MODELOS:
            carpeta = os.path.join(OUT_DIR, dominio, model_name)
            os.makedirs(carpeta, exist_ok=True)
            print("-" * 90 + f"\nMODELO: {model_name}  DOMINIO: {dominio}", flush=True)
            todos = []
            for fold, (train, val, test) in enumerate(splits, start=1):
                pred_csv = os.path.join(carpeta, f"predicciones_fold{fold}.csv")
                t0 = time.time()
                if os.path.exists(pred_csv):
                    ep = pd.read_csv(os.path.join(carpeta, f"epochs_fold{fold}.csv"))
                    best_epoch = int(ep.loc[ep.val_f1.idxmax(), "epoch"])
                    best_val, n_ep = float(ep.val_f1.max()), len(ep)
                    print(f"  fold{fold}: ya calculado, se reutiliza.", flush=True)
                else:
                    best_epoch, best_val, n_ep = entrenar_fold(model_name, fold, paths, labels, train, val, test,
                                                               args.epochs, args.patience, carpeta)
                pr = pd.read_csv(pred_csv)
                assert (pr.real.values == labels[test]).all()  # mismo orden que el fold de prueba
                f1, acc, auc = guardar_matriz_roc(pr.real, pr.prediccion, pr.prob_no_fresco, carpeta, f"fold{fold}")
                todos.append(pr)
                resumen.append({"model": model_name, "fold": fold, "n_train": len(train), "n_val": len(val),
                                "n_test": len(test), "epoch_elegido": best_epoch, "val_f1_elegido": round(best_val, 2),
                                "epochs_run": n_ep, "test_f1": f1, "test_accuracy": acc, "test_auc": auc,
                                "tiempo_s": round(time.time() - t0, 1)})
                print(f"  fold{fold}: train={len(train)} val={len(val)} test={len(test)}  epoch elegido={best_epoch} "
                      f"(val_F1={best_val:.2f})  TEST: F1={f1:.2f} Acc={acc:.2f} AUC={auc:.4f}", flush=True)
            pr = pd.concat(todos)
            f1, acc, auc = guardar_matriz_roc(pr.real, pr.prediccion, pr.prob_no_fresco, carpeta, "pooled")
            resumen.append({"model": model_name, "fold": "pooled", "n_test": len(pr), "test_f1": f1,
                            "test_accuracy": acc, "test_auc": auc})
            print(f"  POOLED (prueba, {len(pr)} img): F1={f1:.2f} Acc={acc:.2f} AUC={auc:.4f}", flush=True)
            pd.DataFrame(resumen).to_csv(os.path.join(OUT_DIR, dominio, "resumen.csv"), index=False)
    print("LISTO", flush=True)


if __name__ == "__main__":
    main()
