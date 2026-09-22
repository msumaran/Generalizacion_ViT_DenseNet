"""
Módulo 1 v3 — Entrenamiento SOLO en DNC (DataSetNotClean)
         Evaluación de transferencia en TODOS los ojos (1196 imgs)

Ojos nunca vistos durante entrenamiento — test honesto de transferencia.
Resultados en: results_freshness_v3/
Checkpoints:   checkpoints_freshness_v3/
"""

import json
import os
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

from config import CFG
from dataset_freshness import (
    _label_from_folder, FreshnessDataset,
    FRESHNESS_CLASSES_2, FRESHNESS_CLASSES_3,
    DAY_TO_LABEL_3, DAY_TO_LABEL_2, EXTENSIONS,
)
from metrics import compute_metrics
from models import build_model, disable_fused_attention, is_vit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE_SIZE_CNN = 256
IMAGE_SIZE_VIT = 224

DNC_DIR  = "DataSetNotClean_4k"
OJOS_DIR = "ojos"

# Hiperparametros del pipeline DNC 5-fold (dnc_5fold_full_metrics.py y
# dnc_5fold_confmat_roc.py leen este dict directamente).
V3_CFG = dict(
    results_dir        = "results_freshness_v3",
    checkpoints_dir    = "checkpoints_freshness_v3_4k",
    num_epochs         = 30,
    warmup_epochs      = 3,
    batch_size_cnn     = 16,
    batch_size_vit     = 8,
    learning_rate      = 1e-4,
    backbone_lr_factor = 0.1,
    weight_decay       = 0.05,
    label_smoothing    = 0.1,
    focal_gamma        = 2.0,
    early_stop_patience= 7,
)

ALL_MODELS = [
    "densenet121",
    "resnet50",
    "mobilenetv1_100",
    "mobilenetv2_100",
    "mobilenetv3_small_100",
    "efficientnet_b0",
    "efficientnet_lite0",
    "vit_base_patch16_224",
    "swin_base_patch4_window7_224",
    "vit_small_patch14_dinov2",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def collect_dnc_only(two_class: bool):
    """Solo DataSetNotClean, sin ojos."""
    samples, labels, species = [], [], []
    for folder in sorted(os.listdir(DNC_DIR)):
        fp = os.path.join(DNC_DIR, folder)
        if not os.path.isdir(fp):
            continue
        lbl = _label_from_folder(folder, two_class)
        if lbl is None:
            continue
        sp = folder.split(" - ")[0].strip()
        for fname in sorted(os.listdir(fp)):
            if os.path.splitext(fname)[1].lower() in EXTENSIONS:
                samples.append(os.path.join(fp, fname))
                labels.append(lbl)
                species.append(sp)
    return samples, labels, species


def collect_ojos_all(two_class: bool):
    """Todos los ojos — usados solo para eval, nunca para training."""
    day_map = DAY_TO_LABEL_2 if two_class else DAY_TO_LABEL_3
    samples, labels = [], []
    for day_dir in sorted(os.listdir(OJOS_DIR)):
        if day_dir not in day_map:
            continue
        lbl = day_map[day_dir]
        fp = os.path.join(OJOS_DIR, day_dir)
        for fname in sorted(os.listdir(fp)):
            if os.path.splitext(fname)[1].lower() in EXTENSIONS:
                samples.append(os.path.join(fp, fname))
                labels.append(lbl)
    return samples, labels


def _largest_remainder_alloc(target_total: int, weights: np.ndarray) -> np.ndarray:
    """Reparte target_total unidades entre len(weights) grupos, proporcional
    a weights, usando el método de restos mayores (Hamilton) para que la
    suma dé exacto target_total sin sesgar sistemáticamente a ningún grupo."""
    if weights.sum() == 0:
        return np.zeros(len(weights), dtype=int)
    ideal = target_total * weights / weights.sum()
    floor_alloc = np.floor(ideal).astype(int)
    remainder = target_total - floor_alloc.sum()
    frac = ideal - floor_alloc
    order = np.argsort(-frac)  # mayor resto primero
    floor_alloc[order[:remainder]] += 1
    return floor_alloc


def species_stratified_split(species: List[str], labels: List[int], seed: int = 42):
    """Split 70/15/15 que garantiza el total exacto por clase (fresco/no_fresco)
    y reparte esos totales entre especies lo más proporcional posible (en vez
    de dejar el redondeo por grupo a la suerte de StratifiedShuffleSplit)."""
    rng = np.random.RandomState(seed)
    sp_arr  = np.array(species)
    lbl_arr = np.array(labels)
    classes = sorted(set(labels))

    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        c_mask = lbl_arr == c
        n_c = int(c_mask.sum())
        train_c = round(0.70 * n_c)
        val_c   = round(0.15 * n_c)
        test_c  = n_c - train_c - val_c

        sp_list = sorted(set(sp_arr[c_mask]))
        group_idx = {sp: np.where(c_mask & (sp_arr == sp))[0] for sp in sp_list}
        group_sizes = np.array([len(group_idx[sp]) for sp in sp_list])

        train_alloc = _largest_remainder_alloc(train_c, group_sizes)
        remaining_after_train = group_sizes - train_alloc
        val_alloc = _largest_remainder_alloc(val_c, remaining_after_train)

        for sp, n_train, n_val in zip(sp_list, train_alloc, val_alloc):
            idx = group_idx[sp].copy()
            rng.shuffle(idx)
            train_idx.extend(idx[:n_train])
            val_idx.extend(idx[n_train:n_train + n_val])
            test_idx.extend(idx[n_train + n_val:])

    return (np.array(train_idx), np.array(val_idx), np.array(test_idx))


def get_transforms_v3(image_size: int, train: bool):
    """train=True: augmentations (crop/flip/color/blur/rotation/erasing) para
    entrenamiento. train=False: solo resize+normalize, usado para validacion,
    test e inferencia -- misma transformacion en todo el pipeline de auditoria
    para que las imagenes de test se procesen igual que durante el checkpoint
    original."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_dnc_loaders(two_class: bool, image_size: int, batch_size: int, seed: int = 42,
                     augment: bool = True):
    samples, labels, species = collect_dnc_only(two_class)

    # Split 70/15/15 con total EXACTO por clase (fresco/no_fresco) y reparto
    # proporcional por especie dentro de cada clase (método de restos mayores).
    train_idx, val_idx, test_idx = species_stratified_split(species, labels, seed=seed)

    tr_s = [samples[i] for i in train_idx]
    tr_l = [labels[i]  for i in train_idx]
    va_s = [samples[i] for i in val_idx]
    va_l = [labels[i]  for i in val_idx]
    te_s = [samples[i] for i in test_idx]
    te_l = [labels[i]  for i in test_idx]

    class_names  = FRESHNESS_CLASSES_2 if two_class else FRESHNESS_CLASSES_3
    tr_arr       = np.array(tr_l)
    class_counts = np.array([np.sum(tr_arr == c) for c in range(len(class_names))])

    nw = 2  # RAM limitada (8GB): pocos workers para no forzar swap
    kw = dict(batch_size=batch_size, num_workers=nw, pin_memory=torch.cuda.is_available(),
              persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None)
    return (
        DataLoader(FreshnessDataset(tr_s, tr_l, get_transforms_v3(image_size, augment)),
                   shuffle=True,  **kw),
        DataLoader(FreshnessDataset(va_s, va_l, get_transforms_v3(image_size, False)),
                   shuffle=False, **kw),
        DataLoader(FreshnessDataset(te_s, te_l, get_transforms_v3(image_size, False)),
                   shuffle=False, **kw),
        class_names,
        class_counts,
    )


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Optimizer & scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model, model_name):
    lr = V3_CFG["learning_rate"]
    wd = V3_CFG["weight_decay"]
    if is_vit(model_name):
        head_params     = list(model.head.parameters())
        head_ids        = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        return torch.optim.AdamW([
            {"params": backbone_params, "lr": lr * V3_CFG["backbone_lr_factor"]},
            {"params": head_params,     "lr": lr},
        ], weight_decay=wd)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer, num_epochs, warmup_epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_and_evaluate(model_name: str, device: str, two_class: bool = False,
                        augment: bool = True) -> Dict:
    os.makedirs(V3_CFG["results_dir"],     exist_ok=True)
    os.makedirs(V3_CFG["checkpoints_dir"], exist_ok=True)

    num_classes = 2 if two_class else 3
    suffix      = "2c" if two_class else "3c"
    img_size    = IMAGE_SIZE_VIT if is_vit(model_name) else IMAGE_SIZE_CNN
    bs          = V3_CFG["batch_size_vit"] if is_vit(model_name) else V3_CFG["batch_size_cnn"]

    train_loader, val_loader, test_loader, class_names, class_counts = \
        get_dnc_loaders(two_class=two_class, image_size=img_size, batch_size=bs,
                         augment=augment)

    print(f"\n[{model_name}] v3  classes={num_classes}({suffix})  img={img_size}  "
          f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)}  device={device}  bs={bs}", flush=True)
    print(f"  class counts (train): {dict(zip(class_names, class_counts.tolist()))}", flush=True)
    print(f"  NOTE: ojos excluidos del entrenamiento", flush=True)

    model = build_model(model_name, num_classes, pretrained=True)
    if is_vit(model_name):
        disable_fused_attention(model)
    model.to(device)

    raw_w     = 1.0 / (class_counts.astype(float) + 1)
    alpha     = torch.tensor(raw_w / raw_w.sum() * num_classes, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=V3_CFG["focal_gamma"],
                          label_smoothing=V3_CFG["label_smoothing"])
    optimizer = build_optimizer(model, model_name)
    scheduler = build_scheduler(optimizer, V3_CFG["num_epochs"], V3_CFG["warmup_epochs"])

    ckpt_path   = os.path.join(V3_CFG["checkpoints_dir"], f"{model_name}_v3_{suffix}_best.pt")
    best_val_f1 = -1.0
    patience    = 0
    history     = []

    for epoch in range(V3_CFG["num_epochs"]):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            criterion(model(imgs), lbls).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                all_p.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
                all_l.extend(lbls.numpy())
        vm = compute_metrics(np.array(all_l), np.array(all_p), num_classes)

        if device == "mps":
            torch.mps.empty_cache()

        if vm["macro_f1"] > best_val_f1:
            best_val_f1 = vm["macro_f1"]
            torch.save(model.state_dict(), ckpt_path)
            patience = 0
        else:
            patience += 1

        lr_now = optimizer.param_groups[-1]["lr"]
        print(f"  epoch {epoch+1:3d}/{V3_CFG['num_epochs']}  "
              f"val_F1={vm['macro_f1']:.2f}  best={best_val_f1:.2f}  "
              f"pat={patience}  lr={lr_now:.2e}", flush=True)
        history.append({
            "epoch": epoch + 1,
            **{f"val_{k}": round(float(v), 4)
               for k, v in vm.items() if not isinstance(v, (list, dict))}
        })

        if patience >= V3_CFG["early_stop_patience"]:
            print(f"  Early stop at epoch {epoch+1}", flush=True)
            break

    # DNC test eval
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            all_p.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
            all_l.extend(lbls.numpy())
    tm = compute_metrics(np.array(all_l), np.array(all_p), num_classes)
    print(f"\n  DNC TEST → F1={tm['macro_f1']:.2f}  Acc={tm['accuracy']:.2f}", flush=True)

    # Ojos transfer eval (todos los 1196)
    oj_s, oj_l = collect_ojos_all(two_class)
    val_tf = get_transforms_v3(img_size, False)

    nw = 2  # RAM limitada (8GB)
    oj_loader = DataLoader(FreshnessDataset(oj_s, oj_l, val_tf), batch_size=bs,
                           shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available(),
                           persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None)
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, lbls in oj_loader:
            all_p.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
            all_l.extend(lbls.numpy())
    om = compute_metrics(np.array(all_l), np.array(all_p), num_classes)
    print(f"  OJOS TRANSFER → F1={om['macro_f1']:.2f}  Acc={om['accuracy']:.2f}  "
          f"n={len(oj_s)}", flush=True)

    pd.DataFrame(history).to_csv(
        os.path.join(V3_CFG["results_dir"], f"{model_name}_{suffix}_history_v3.csv"),
        index=False)

    return {
        "model":              model_name,
        "type":               "ViT" if is_vit(model_name) else "CNN",
        "version":            "v3",
        "num_classes":        num_classes,
        "checkpoint":         ckpt_path,
        "val_f1":             round(best_val_f1, 4),
        # DNC test
        "dnc_test_f1":        round(float(tm["macro_f1"]),          4),
        "dnc_test_accuracy":  round(float(tm["accuracy"]),           4),
        "dnc_test_precision": round(float(tm["macro_precision"]),    4),
        "dnc_test_recall":    round(float(tm["macro_recall"]),       4),
        # Ojos transfer
        "ojos_f1":            round(float(om["macro_f1"]),           4),
        "ojos_accuracy":      round(float(om["accuracy"]),           4),
        "ojos_precision":     round(float(om["macro_precision"]),    4),
        "ojos_recall":        round(float(om["macro_recall"]),       4),
        "ojos_n":             len(oj_s),
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _load_existing(two_class: bool) -> List[Dict]:
    suffix   = "2c" if two_class else "3c"
    csv_path = os.path.join(V3_CFG["results_dir"], f"freshness_v3_{suffix}.csv")
    return pd.read_csv(csv_path).to_dict("records") if os.path.exists(csv_path) else []


def _save_results(rows: List[Dict], two_class: bool):
    suffix = "2c" if two_class else "3c"
    os.makedirs(V3_CFG["results_dir"], exist_ok=True)
    df = pd.DataFrame(rows).sort_values("dnc_test_f1", ascending=False)
    df.to_csv(os.path.join(V3_CFG["results_dir"], f"freshness_v3_{suffix}.csv"), index=False)
    print(f"\n{'='*75}")
    print(f"RESULTADOS v3 — {suffix}  (DNC-only training | ojos transfer)")
    print(df[["model","type","dnc_test_f1","dnc_test_accuracy",
              "ojos_f1","ojos_accuracy","ojos_n"]].to_string(index=False))
    print(f"{'='*75}\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--two_class",     action="store_true")
    parser.add_argument("--no_augment",    action="store_true",
                         help="Desactiva el data augmentation de entrenamiento")
    args = parser.parse_args()

    two_class  = args.two_class
    augment    = not args.no_augment
    class_list = FRESHNESS_CLASSES_2 if two_class else FRESHNESS_CLASSES_3
    device     = CFG.device

    print(f"Device: {device}", flush=True)
    print(f"Mode: v3 {'2-class' if two_class else '3-class'}  Classes: {class_list}", flush=True)
    print(f"Augment: {augment}", flush=True)
    print(f"Training: DNC ONLY ({DNC_DIR})", flush=True)
    print(f"Transfer eval: todos los ojos ({OJOS_DIR})", flush=True)

    existing       = _load_existing(two_class)
    existing_names = {r["model"] for r in existing}
    models_to_run  = [args.model] if args.model else ALL_MODELS
    all_results    = list(existing)

    for mname in models_to_run:
        if args.skip_existing and mname in existing_names:
            print(f"Skipping {mname} (already done)", flush=True)
            continue
        result = train_and_evaluate(mname, device, two_class=two_class, augment=augment)
        all_results = [r for r in all_results if r["model"] != mname]
        all_results.append(result)
        _save_results(all_results, two_class)

    print("Done.", flush=True)
