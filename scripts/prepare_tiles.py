#!/usr/bin/env python
"""CLI para tilear un ortomosaico (RGB + CHM + labels) en parches de entrenamiento.

Ejemplo:
    python scripts/prepare_tiles.py \
        --rgb data/raw/rgb/sitio1.tif \
        --chm data/raw/chm/sitio1.tif \
        --labels data/raw/labels/sitio1.tif \
        --out data/processed/train \
        --tile-size 256 --overlap 32
"""
from __future__ import annotations

import argparse

from src.data.tiling import tile_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True, help="GeoTIFF óptico")
    parser.add_argument("--chm", required=True, help="GeoTIFF de altura (CHM/nDSM)")
    parser.add_argument("--labels", default=None, help="GeoTIFF de instancias de copas (opcional)")
    parser.add_argument("--out", required=True, help="Carpeta de salida para los tiles")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--min-valid-fraction", type=float, default=0.0)
    parser.add_argument("--prefix", default="tile")
    args = parser.parse_args()

    count = tile_dataset(
        rgb_path=args.rgb,
        chm_path=args.chm,
        labels_path=args.labels,
        out_dir=args.out,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_valid_fraction=args.min_valid_fraction,
        prefix=args.prefix,
    )
    print(f"Generados {count} tiles en {args.out}")


if __name__ == "__main__":
    main()
