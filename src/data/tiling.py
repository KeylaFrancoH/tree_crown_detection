"""Recorte de ortomosaicos grandes (RGB, CHM, labels) en parches (tiles).

Los tres rasters de entrada (imagen óptica, CHM/nDSM y máscara de instancias)
deben tener la misma grilla (mismo tamaño de píxel, extensión y CRS). Si no
la tienen, primero hay que resamplear/alinear con rasterio (`reproject`)
fuera de este script.

Cada tile se guarda como .npz (liviano, sin depender de georreferenciación
para el entrenamiento) más un .json con la metadata geoespacial, para poder
reconstruir la posición real de cada instancia detectada en inferencia.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from src.data.rasterize import instance_to_targets


def _iter_windows(width: int, height: int, tile_size: int, stride: int):
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            w = min(tile_size, width - col)
            h = min(tile_size, height - row)
            if w <= 0 or h <= 0:
                continue
            yield row, col, h, w


def tile_dataset(
    rgb_path: str,
    chm_path: str,
    labels_path: str | None,
    out_dir: str,
    tile_size: int = 256,
    overlap: int = 32,
    min_valid_fraction: float = 0.0,
    prefix: str = "tile",
) -> int:
    """Genera tiles alineados de RGB + CHM (+ labels si se proveen).

    Args:
        rgb_path: GeoTIFF óptico (N bandas).
        chm_path: GeoTIFF de altura (1 banda), misma grilla que rgb_path.
        labels_path: GeoTIFF de instancias de copas (opcional; None para
            tiles de inferencia sin ground truth).
        out_dir: carpeta de salida para los .npz.
        tile_size: tamaño del parche cuadrado en píxeles.
        overlap: solape entre parches contiguos (en píxeles).
        min_valid_fraction: descarta tiles cuyo % de píxeles válidos (no-nodata
            en CHM) sea menor a este umbral. Útil para saltar zonas vacías.
        prefix: prefijo de los archivos generados.

    Returns:
        Cantidad de tiles escritos.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap debe ser menor que tile_size")

    count = 0
    with rasterio.open(rgb_path) as rgb_ds, rasterio.open(chm_path) as chm_ds:
        if (rgb_ds.width, rgb_ds.height) != (chm_ds.width, chm_ds.height):
            raise ValueError(
                f"RGB ({rgb_ds.width}x{rgb_ds.height}) y CHM ({chm_ds.width}x{chm_ds.height}) "
                "no tienen la misma grilla. Reproyectá/resampleá antes de tilear."
            )
        labels_ds = rasterio.open(labels_path) if labels_path else None
        try:
            for row, col, h, w in _iter_windows(rgb_ds.width, rgb_ds.height, tile_size, stride):
                window = Window(col, row, w, h)
                rgb_tile = rgb_ds.read(window=window)  # (C, h, w)
                chm_tile = chm_ds.read(1, window=window)  # (h, w)

                chm_nodata = chm_ds.nodata
                if chm_nodata is not None:
                    valid_fraction = float(np.mean(chm_tile != chm_nodata))
                    if valid_fraction < min_valid_fraction:
                        continue

                sample = {
                    "rgb": rgb_tile.astype(np.float32),
                    "chm": chm_tile.astype(np.float32)[None, ...],
                }

                if labels_ds is not None:
                    label_tile = labels_ds.read(1, window=window)
                    mask, dist = instance_to_targets(label_tile)
                    sample["mask"] = mask[None, ...]
                    sample["dist"] = dist[None, ...]
                    sample["instances"] = label_tile.astype(np.int32)

                tile_transform = rgb_ds.window_transform(window)
                meta = {
                    "crs": rgb_ds.crs.to_string() if rgb_ds.crs else None,
                    "transform": list(tile_transform)[:6],
                    "row": row,
                    "col": col,
                    "height": h,
                    "width": w,
                }

                tile_name = f"{prefix}_{row}_{col}"
                np.savez_compressed(out_path / f"{tile_name}.npz", **sample)
                with open(out_path / f"{tile_name}.json", "w") as f:
                    json.dump(meta, f)
                count += 1
        finally:
            if labels_ds is not None:
                labels_ds.close()
    return count
