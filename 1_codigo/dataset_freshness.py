"""
Combined freshness dataset: DataSetNotClean + ojos (trucha)

Label mapping:
  DataSetNotClean folders:
    "... - Highly Fresh" → 0
    "... - Fresh"        → 1  (only if not Highly Fresh)
    "... - Not Fresh"    → 2

  ojos folders:
    dia-1, dia-3 → 0  (fresco = highly fresh)
    dia-5        → 1  (semifresco = fresh)
    dia-7, dia-9 → 2  (no_fresco = not fresh)

FRESHNESS_CLASSES = ["highly_fresh", "fresh", "not_fresh"]
"""

import os
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader

from dataset import get_transforms


FRESHNESS_CLASSES_3 = ["highly_fresh", "fresh", "not_fresh"]
FRESHNESS_CLASSES_2 = ["fresco", "no_fresco"]
FRESHNESS_CLASSES   = FRESHNESS_CLASSES_3  # default

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DAY_TO_LABEL_3 = {"dia-1": 0, "dia-3": 0, "dia-5": 1, "dia-7": 2, "dia-9": 2}
DAY_TO_LABEL_2 = {"dia-1": 0, "dia-3": 0, "dia-5": 1, "dia-7": 1, "dia-9": 1}


def _label_from_folder(folder_name: str, two_class: bool = False) -> Optional[int]:
    cu = folder_name.strip().upper()
    if two_class:
        if "NOT FRESH" in cu:
            return 1
        if "FRESH" in cu:   # covers both "Fresh" and "Highly Fresh"
            return 0
        return None
    if "HIGHLY FRESH" in cu:
        return 0
    if "NOT FRESH" in cu:
        return 2
    if "FRESH" in cu:
        return 1
    return None


def _collect_samples(dataset_not_clean_dir: str,
                     ojos_dir: str,
                     two_class: bool = False) -> Tuple[List[str], List[int]]:
    samples, labels = [], []
    day_map = DAY_TO_LABEL_2 if two_class else DAY_TO_LABEL_3

    # DataSetNotClean
    for folder in sorted(os.listdir(dataset_not_clean_dir)):
        fp = os.path.join(dataset_not_clean_dir, folder)
        if not os.path.isdir(fp):
            continue
        lbl = _label_from_folder(folder, two_class)
        if lbl is None:
            continue
        for fname in sorted(os.listdir(fp)):
            if os.path.splitext(fname)[1].lower() in EXTENSIONS:
                samples.append(os.path.join(fp, fname))
                labels.append(lbl)

    # ojos
    for day_dir in sorted(os.listdir(ojos_dir)):
        if day_dir not in day_map:
            continue
        lbl = day_map[day_dir]
        fp = os.path.join(ojos_dir, day_dir)
        for fname in sorted(os.listdir(fp)):
            if os.path.splitext(fname)[1].lower() in EXTENSIONS:
                samples.append(os.path.join(fp, fname))
                labels.append(lbl)

    return samples, labels


class FreshnessDataset(Dataset):
    def __init__(self, samples: List[str], labels: List[int], transform=None):
        self.samples   = samples
        self.labels    = labels
        self.transform = transform
        self.classes   = FRESHNESS_CLASSES

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_freshness_loaders(dataset_not_clean_dir: str,
                          ojos_dir: str,
                          image_size: int,
                          batch_size: int,
                          num_workers: int = 0,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15,
                          seed: int = 42,
                          two_class: bool = False):
    """
    Stratified 70/15/15 split of combined DataSetNotClean + ojos.
    Returns (train_loader, val_loader, test_loader, class_names, class_counts).
    two_class=True → binary: fresco(0) vs no_fresco(1)
    """
    all_samples, all_labels = _collect_samples(dataset_not_clean_dir, ojos_dir, two_class)
    labels_arr = np.array(all_labels)
    n = len(labels_arr)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(n), labels_arr))

    val_of_trainval = val_ratio / (1 - test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_of_trainval, random_state=seed)
    train_rel, val_rel = next(sss2.split(np.zeros(len(trainval_idx)),
                                          labels_arr[trainval_idx]))
    train_idx = trainval_idx[train_rel]
    val_idx   = trainval_idx[val_rel]

    train_tf = get_transforms(image_size, train=True)
    infer_tf = get_transforms(image_size, train=False)

    tr_s = [all_samples[i] for i in train_idx]
    tr_l = [all_labels[i]  for i in train_idx]
    va_s = [all_samples[i] for i in val_idx]
    va_l = [all_labels[i]  for i in val_idx]
    te_s = [all_samples[i] for i in test_idx]
    te_l = [all_labels[i]  for i in test_idx]

    class_names  = FRESHNESS_CLASSES_2 if two_class else FRESHNESS_CLASSES_3
    tr_labels    = np.array(tr_l)
    class_counts = np.array([np.sum(tr_labels == c) for c in range(len(class_names))])

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(FreshnessDataset(tr_s, tr_l, train_tf), shuffle=True,  **kw),
        DataLoader(FreshnessDataset(va_s, va_l, infer_tf), shuffle=False, **kw),
        DataLoader(FreshnessDataset(te_s, te_l, infer_tf), shuffle=False, **kw),
        class_names,
        class_counts,
    )
