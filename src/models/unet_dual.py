"""U-Net de doble encoder para detección de copas de árboles.

Arquitectura (inspirada en el esquema del proyecto: dos ramas CNN que se
concatenan antes de la cabeza de salida, adaptado de detección tipo YOLO a
segmentación U-Net):

  RGB (o multiespectral) ----> Encoder A ----\
                                               >-- Fusion por nivel --> Decoder --> heads
  CHM / nDSM (altura) -------> Encoder B ----/

Salidas (dos cabezas, para poder pasar de semántica a instancias):
  - mask_logits: 1 canal, probabilidad de "copa" por píxel (BCE+Dice).
  - dist_map:    1 canal, transformada de distancia normalizada [0,1] dentro
                 de cada copa (regresión). Sus máximos locales sirven como
                 semillas para separar copas que se tocan mediante watershed
                 (ver src/inference/postprocess.py), dando instancias.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import DoubleConv, Down, FusionBlock, Up


class _Encoder(nn.Module):
    """Encoder U-Net estándar de 4 niveles de downsampling."""

    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.out_channels = [c, c * 2, c * 4, c * 8, c * 16]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]


class DualEncoderUNet(nn.Module):
    """U-Net con dos encoders (imagen + altura) fusionados nivel a nivel.

    Args:
        rgb_channels: canales de la rama óptica (3 para RGB, 4+ para multiespectral).
        chm_channels: canales de la rama de altura (normalmente 1: CHM/nDSM).
        base_channels: ancho base del encoder (se duplica en cada nivel).
        bilinear: usar upsampling bilineal (más liviano) en vez de transpuesta.
    """

    def __init__(
        self,
        rgb_channels: int = 3,
        chm_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = True,
    ):
        super().__init__()
        self.encoder_rgb = _Encoder(rgb_channels, base_channels)
        self.encoder_chm = _Encoder(chm_channels, base_channels)

        fused_channels = self.encoder_rgb.out_channels  # igual en ambos encoders
        self.fusions = nn.ModuleList(
            [FusionBlock(ch, ch, ch) for ch in fused_channels]
        )

        c = base_channels
        factor = 2 if bilinear else 1
        self.up1 = Up(c * 16, c * 8, c * 8 // factor, bilinear)
        self.up2 = Up(c * 8 // factor, c * 4, c * 4 // factor, bilinear)
        self.up3 = Up(c * 4 // factor, c * 2, c * 2 // factor, bilinear)
        self.up4 = Up(c * 2 // factor, c, c, bilinear)

        self.mask_head = nn.Conv2d(c, 1, kernel_size=1)
        self.dist_head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, rgb: torch.Tensor, chm: torch.Tensor) -> dict[str, torch.Tensor]:
        feats_rgb = self.encoder_rgb(rgb)
        feats_chm = self.encoder_chm(chm)

        fused = [fusion(f_rgb, f_chm) for fusion, f_rgb, f_chm in zip(self.fusions, feats_rgb, feats_chm)]
        f1, f2, f3, f4, f5 = fused

        x = self.up1(f5, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)

        return {
            "mask_logits": self.mask_head(x),
            "dist_map": torch.sigmoid(self.dist_head(x)),
        }


def build_model(config: dict) -> DualEncoderUNet:
    model_cfg = config.get("model", {})
    return DualEncoderUNet(
        rgb_channels=model_cfg.get("rgb_channels", 3),
        chm_channels=model_cfg.get("chm_channels", 1),
        base_channels=model_cfg.get("base_channels", 32),
        bilinear=model_cfg.get("bilinear", True),
    )
