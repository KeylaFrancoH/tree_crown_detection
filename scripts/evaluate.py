#!/usr/bin/env python
"""Evalúa un checkpoint entrenado sobre el conjunto de test (imágenes nunca vistas
en entrenamiento ni validación).

Ejemplo:
    python scripts/evaluate.py --config configs/cedia.yaml --checkpoint checkpoints_cedia/best.pth
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.data.dataset import TreeCrownTileDataset
from src.models.unet_dual import build_model
from src.training.metrics import binary_iou
from src.training.train import load_config


@torch.no_grad()
def evaluate(config_path: str, checkpoint_path: str) -> float:
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_cfg = config["data"]
    test_ds = TreeCrownTileDataset(
        data_cfg["test_dir"],
        rgb_scale=data_cfg.get("rgb_scale", 255.0),
        chm_max=data_cfg.get("chm_max", 40.0),
    )
    test_loader = DataLoader(test_ds, batch_size=config["training"].get("batch_size", 8), shuffle=False)

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    total_iou = 0.0
    for batch in test_loader:
        rgb = batch["rgb"].to(device)
        chm = batch["chm"].to(device)
        mask_gt = batch["mask"].to(device)
        out = model(rgb, chm)
        total_iou += binary_iou(out["mask_logits"], mask_gt)

    mean_iou = total_iou / max(len(test_loader), 1)
    print(f"Test IoU (semántico, {len(test_ds)} tiles, {len(test_loader)} batches): {mean_iou:.4f}")
    return mean_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cedia.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
