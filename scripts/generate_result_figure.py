#!/usr/bin/env python
"""Genera una figura de resultados estilo paper: por cada tile de ejemplo,
(a) imagen original, (b) regiones segmentadas (instancias coloreadas),
(c) copas aisladas sobre fondo negro (máscara aplicada), (d) máscara real
(ground truth) para comparar.

Inspirado en el estilo de figura de s13369-024-09888-5.pdf (Fig. 6/7/8):
original -> etiquetado de regiones segmentadas -> objeto detectado aislado.
Acá el "objeto" son las copas de árbol y el método es el U-Net + watershed
de este proyecto, no el método de ese paper (Gabor/LBP/K-means, que es para
clasificación de especies de mariposas camufladas, un problema distinto).

Ejemplo:
    python scripts/generate_result_figure.py \
        --config configs/cedia.yaml --checkpoint checkpoints_cedia/best.pth \
        --split test --n 4 --out cedia_dataset/resultado_estilo_paper.png
"""
from __future__ import annotations

import argparse
import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from src.inference.postprocess import instances_from_mask_dist
from src.models.unet_dual import build_model
from src.training.train import load_config


def _rgb_uint8(rgb_bands: np.ndarray) -> np.ndarray:
    """(3,H,W) float -> (H,W,3) uint8 para visualización (estiramiento simple de contraste)."""
    img = rgb_bands.transpose(1, 2, 0)
    p2, p98 = np.percentile(img, [2, 98])
    img = np.clip((img - p2) / max(p98 - p2, 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


def _instance_colormap(n: int) -> ListedColormap:
    rng = np.random.default_rng(0)
    colors = rng.uniform(0.25, 1.0, size=(max(n, 1) + 1, 3))
    colors[0] = [0, 0, 0]  # fondo = negro
    return ListedColormap(colors)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cedia.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=4, help="cantidad de tiles de ejemplo")
    parser.add_argument("--min_crown_fraction", type=float, default=0.15, help="elige tiles con al menos esta fraccion de copa real, para que el ejemplo sea informativo")
    parser.add_argument("--out", default="cedia_dataset/resultado_estilo_paper.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    data_cfg = config["data"]
    files = sorted(glob.glob(f"{data_cfg[args.split + '_dir']}/*.npz"))
    candidates = []
    for f in files:
        d = np.load(f)
        if d["mask"].mean() >= args.min_crown_fraction:
            candidates.append(f)
    if len(candidates) < args.n:
        candidates = files
    random.Random(args.seed).shuffle(candidates)
    chosen = candidates[: args.n]

    inf_cfg = config.get("inference", {})
    fig, axes = plt.subplots(len(chosen), 4, figsize=(16, 4 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]

    col_titles = ["(a) Imagen original", "(b) Instancias segmentadas (predicción)", "(c) Copas aisladas (predicción)", "(d) Etiqueta real (ground truth)"]

    for row, f in enumerate(chosen):
        d = np.load(f)
        rgb_raw = d["rgb"]
        chm_raw = d["chm"]
        gt_instances = d["instances"]

        rgb_t = torch.from_numpy(rgb_raw / data_cfg["rgb_scale"]).unsqueeze(0).to(device)
        chm_t = torch.from_numpy(np.clip(chm_raw, 0, data_cfg["chm_max"]) / data_cfg["chm_max"]).unsqueeze(0).to(device)
        out = model(rgb_t, chm_t)
        mask_prob = torch.sigmoid(out["mask_logits"])[0, 0].cpu().numpy()
        dist_map = out["dist_map"][0, 0].cpu().numpy()

        pred_instances = instances_from_mask_dist(
            mask_prob,
            dist_map,
            mask_threshold=inf_cfg.get("mask_threshold", 0.5),
            min_seed_distance=inf_cfg.get("min_seed_distance", 16),
            min_instance_area=inf_cfg.get("min_instance_area", 60),
        )

        rgb_vis = _rgb_uint8(rgb_raw[:3])
        isolated = rgb_vis.copy()
        isolated[pred_instances == 0] = 0

        n_pred = int(pred_instances.max())
        n_gt = len(np.unique(gt_instances)) - 1
        cmap_pred = _instance_colormap(n_pred)
        cmap_gt = _instance_colormap(max(n_gt, 1))

        axes[row, 0].imshow(rgb_vis)
        axes[row, 1].imshow(pred_instances, cmap=cmap_pred, interpolation="nearest")
        axes[row, 2].imshow(isolated)
        axes[row, 3].imshow(gt_instances, cmap=cmap_gt, interpolation="nearest")

        axes[row, 0].set_ylabel(f.split("/")[-1].replace(".npz", ""), fontsize=8)
        axes[row, 1].set_title(f"{n_pred} copas detectadas", fontsize=9)
        axes[row, 3].set_title(f"{n_gt} copas reales", fontsize=9)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col] + (f"\n{axes[row,col].get_title()}" if axes[row,col].get_title() else ""), fontsize=10)

    fig.suptitle("Segmentación de copas de árbol — U-Net doble encoder (RGB+MS / CHM) + watershed", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figura guardada en {args.out}")


if __name__ == "__main__":
    main()
