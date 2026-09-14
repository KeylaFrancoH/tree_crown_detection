"""Sanity checks con datos sintéticos: forward/backward del modelo y post-procesado."""
from __future__ import annotations

import numpy as np
import torch

from src.inference.postprocess import instances_from_mask_dist
from src.losses.losses import CrownLoss
from src.models.unet_dual import DualEncoderUNet


def test_forward_shapes():
    model = DualEncoderUNet(rgb_channels=3, chm_channels=1, base_channels=8)
    rgb = torch.rand(2, 3, 64, 64)
    chm = torch.rand(2, 1, 64, 64)
    out = model(rgb, chm)
    assert out["mask_logits"].shape == (2, 1, 64, 64)
    assert out["dist_map"].shape == (2, 1, 64, 64)
    assert torch.all(out["dist_map"] >= 0) and torch.all(out["dist_map"] <= 1)


def test_backward_step():
    model = DualEncoderUNet(rgb_channels=3, chm_channels=1, base_channels=8)
    criterion = CrownLoss()
    rgb = torch.rand(1, 3, 32, 32)
    chm = torch.rand(1, 1, 32, 32)
    targets = {"mask": torch.randint(0, 2, (1, 1, 32, 32)), "dist": torch.rand(1, 1, 32, 32)}

    out = model(rgb, chm)
    losses = criterion(out, targets)
    losses["loss"].backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_watershed_separates_two_crowns():
    # Dos "copas" circulares que se tocan, simuladas directamente en mask/dist.
    size = 40
    mask = np.zeros((size, size), dtype=np.float32)
    dist = np.zeros((size, size), dtype=np.float32)

    centers = [(12, 15), (12, 25)]
    radius = 8
    yy, xx = np.mgrid[0:size, 0:size]
    for cy, cx in centers:
        circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        mask[circle] = 1.0
        d = radius - np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        d = np.clip(d, 0, None) / radius
        dist[circle] = np.maximum(dist[circle], d[circle])

    instances = instances_from_mask_dist(mask, dist, min_seed_distance=3, min_instance_area=3)
    unique_ids = set(np.unique(instances)) - {0}
    assert len(unique_ids) == 2, f"Se esperaban 2 instancias, se obtuvieron {len(unique_ids)}"
