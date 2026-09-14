#!/usr/bin/env python
"""CLI de inferencia: corre el modelo entrenado sobre un ortomosaico completo.

Ejemplo:
    python scripts/predict.py \
        --config configs/default.yaml \
        --checkpoint checkpoints/best.pth \
        --rgb data/raw/rgb/sitio1.tif \
        --chm data/raw/chm/sitio1.tif \
        --out results/sitio1_copas.geojson
"""
from __future__ import annotations

import argparse

import torch

from src.inference.predict import predict_orthomosaic
from src.models.unet_dual import build_model
from src.training.train import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--chm", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    inf_cfg = config.get("inference", {})
    data_cfg = config.get("data", {})
    gdf = predict_orthomosaic(
        model,
        rgb_path=args.rgb,
        chm_path=args.chm,
        out_geojson=args.out,
        device=device,
        rgb_scale=data_cfg.get("rgb_scale", 255.0),
        chm_max=data_cfg.get("chm_max", 40.0),
        **inf_cfg,
    )
    print(f"Detectadas {len(gdf)} copas. Guardado en {args.out}")


if __name__ == "__main__":
    main()
