"""U-Net RGB con encoder preentrenado (transfer learning).

A diferencia de SingleEncoderUNet (encoder desde cero), este usa un encoder
ya entrenado en ImageNet via segmentation_models_pytorch. Con pocos datos
propios (2 imagenes etiquetadas), arrancar de pesos preentrenados en vez de
aleatorios aprovecha features visuales ya aprendidas (bordes, texturas) y
converge mejor que entrenar todo desde cero.

El encoder NO se congela: se afina (fine-tune) junto con el resto de la red,
que es lo recomendado cuando el dominio (foto aerea) difiere bastante del
dataset de preentrenamiento (fotos a nivel de piso de ImageNet).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PretrainedRGBUNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: str | None = "imagenet",
        rgb_channels: int = 3,
    ):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=rgb_channels,
            classes=2,  # canal 0: mask_logits, canal 1: dist (antes de sigmoid)
            activation=None,
        )

    def forward(self, rgb: torch.Tensor, chm: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = self.net(rgb)
        return {
            "mask_logits": out[:, 0:1],
            "dist_map": torch.sigmoid(out[:, 1:2]),
        }
