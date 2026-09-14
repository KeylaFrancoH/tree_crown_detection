"""Métricas de validación: IoU semántico (copa vs fondo) sobre la máscara predicha."""
from __future__ import annotations

import torch


@torch.no_grad()
def binary_iou(mask_logits: torch.Tensor, mask_gt: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(mask_logits)
    pred = (probs >= threshold).float()
    intersection = (pred * mask_gt).sum()
    union = pred.sum() + mask_gt.sum() - intersection
    return ((intersection + eps) / (union + eps)).item()
