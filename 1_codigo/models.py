"""
Model factory using timm.
All models are loaded with ImageNet pre-trained weights and the classification
head replaced to match num_classes.
"""

from typing import List
import timm
import torch.nn as nn


CNN_MODELS: List[str] = [
    "densenet121",
    "resnet50",
    "mobilenetv1_100",
    "mobilenetv2_100",
    "mobilenetv3_small_100",
    "efficientnet_b0",
    "efficientnet_lite0",
]

VIT_MODELS: List[str] = [
    "vit_base_patch16_224",
    "swin_base_patch4_window7_224",
    "vit_small_patch14_dinov2",
]

# DINOv2 default is 518px (fixed); override to 224 for consistency
_DINOV2_MODELS = {"vit_small_patch14_dinov2", "vit_base_patch14_dinov2"}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Crea el modelo via timm con la cabeza de clasificacion ya ajustada a
    num_classes. pretrained=True carga pesos de ImageNet (usado en
    entrenamiento); pretrained=False solo arma la arquitectura, para luego
    cargar un checkpoint propio encima (usado en inferencia/auditoria)."""
    kwargs = {}
    if name in _DINOV2_MODELS:
        kwargs["img_size"] = 224
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model


def is_vit(name: str) -> bool:
    """True si el modelo es un Transformer (ViT/Swin/DINOv2) -> determina
    tamaño de imagen (224) y batch size mas chico frente a las CNN (256)."""
    return name in VIT_MODELS or name.startswith("vit_") or name.startswith("swin_")


def disable_fused_attention(model: nn.Module) -> nn.Module:
    """Required to capture attention weights via hooks (bypasses flash-attn path)."""
    if hasattr(model, "blocks"):
        for block in model.blocks:
            if hasattr(block, "attn") and hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = False
    return model
