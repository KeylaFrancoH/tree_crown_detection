"""Funciones de pérdida para el U-Net de doble encoder.

Combina:
  - BCE + Dice sobre la máscara binaria de copa (mask_logits vs mask_gt).
  - MSE sobre el mapa de distancia normalizado (dist_map vs dist_gt), que es
    la señal que permite separar copas contiguas en el post-procesado.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class CrownLoss(nn.Module):
    def __init__(self, mask_weight: float = 1.0, dist_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.mask_weight = mask_weight
        self.dist_weight = dist_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, outputs: dict, targets: dict) -> dict[str, torch.Tensor]:
        mask_logits = outputs["mask_logits"]
        dist_map = outputs["dist_map"]
        mask_gt = targets["mask"].float()
        dist_gt = targets["dist"].float()

        mask_probs = torch.sigmoid(mask_logits)
        bce = self.bce(mask_logits, mask_gt)
        dice = dice_loss(mask_probs, mask_gt)
        mask_loss = bce + self.dice_weight * dice

        dist_loss = self.mse(dist_map, dist_gt)

        total = self.mask_weight * mask_loss + self.dist_weight * dist_loss
        return {
            "loss": total,
            "mask_bce": bce.detach(),
            "mask_dice": dice.detach(),
            "dist_mse": dist_loss.detach(),
        }
