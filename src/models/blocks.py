"""Bloques convolucionales reutilizados por el U-Net de doble encoder."""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x2."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling: maxpool seguido de DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upscaling seguido de concatenación con skip connection y DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            conv_in = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv_in = in_channels // 2 + skip_channels
        self.conv = DoubleConv(conv_in, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Alinea tamaños si hay desajuste de 1px por redondeos en pooling.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FusionBlock(nn.Module):
    """Fusiona features de dos encoders (mismo nivel) con concat + conv 1x1."""

    def __init__(self, channels_a: int, channels_b: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels_a + channels_b, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        return self.reduce(torch.cat([feat_a, feat_b], dim=1))
