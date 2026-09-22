"""Configuracion GENERAL del proyecto de tesis (los 4 modulos: entrenamiento
base, adaptacion de dominio, analisis estadistico, etc).

Para el pipeline DNC 5-fold especifico (dnc_5fold_full_metrics.py y
dnc_5fold_confmat_roc.py), de aqui solo se usa `CFG.device` (deteccion
automatica de cuda/mps/cpu) -- el resto de campos (coral_lambda, num_seeds,
etc.) pertenece a otros modulos del proyecto y no aplica a este pipeline."""
from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Config:
    # --- Paths ---
    # Dataset 1: FishImgDataset (multispecies, pre-split into train/val/test)
    dataset1_root: str = "FishImgDataset"
    dataset1_path: str = "FishImgDataset/train"   # used by flat loaders
    dataset1_val_path: str = "FishImgDataset/val"
    dataset1_test_path: str = "FishImgDataset/test"
    # Dataset 2: ojos de trucha clasificados por días → frescura
    dataset2_path: str = "ojos"
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"

    # --- Classes (31 fish species) ---
    num_classes: int = 31
    class_names: List[str] = field(default_factory=lambda: [
        "Bangus", "Big Head Carp", "Black Spotted Barb", "Catfish",
        "Climbing Perch", "Fourfinger Threadfin", "Freshwater Eel",
        "Glass Perchlet", "Goby", "Gold Fish", "Gourami", "Grass Carp",
        "Green Spotted Puffer", "Indian Carp", "Indo-Pacific Tarpon",
        "Jaguar Gapote", "Janitor Fish", "Knifefish", "Long-Snouted Pipefish",
        "Mosquito Fish", "Mudfish", "Mullet", "Pangasius", "Perch",
        "Scat Fish", "Silver Barb", "Silver Carp", "Silver Perch",
        "Snakehead", "Tenpounder", "Tilapia",
    ])

    # --- Image ---
    image_size: int = 224

    # --- Training (Module 1) ---
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    n_folds: int = 5
    early_stopping_patience: int = 10

    # --- CNN model pool ---
    cnn_model_names: List[str] = field(default_factory=lambda: [
        "densenet121",
        "resnet50",
        "mobilenetv1_100",
        "mobilenetv2_100",
        "mobilenetv3_small_100",
        "efficientnet_b0",
        "efficientnet_lite0",
    ])

    # --- ViT model pool (best one is selected automatically) ---
    vit_model_names: List[str] = field(default_factory=lambda: [
        "vit_base_patch16_224",
        "swin_base_patch4_window7_224",
    ])

    # --- How many top CNNs to carry to Module 2 ---
    top_k_cnn: int = 2

    # --- Domain Adaptation (Module 3) ---
    coral_lambda: float = 1.0
    da_num_epochs: int = 30
    da_learning_rate: float = 5e-5
    da_batch_size: int = 16
    da_weight_decay: float = 1e-4

    # --- Statistical Analysis (Module 4) ---
    num_seeds: int = 10
    base_seed: int = 42
    significance_level: float = 0.05

    # --- Device ---
    device: str = field(init=False)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"


CFG = Config()
