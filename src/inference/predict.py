"""Inferencia sobre un ortomosaico completo: sliding window + stitching +
watershed + export de copas detectadas a GeoJSON.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from src.inference.postprocess import instances_from_mask_dist, polygonize_instances
from src.models.unet_dual import DualEncoderUNet


@torch.no_grad()
def predict_orthomosaic(
    model: DualEncoderUNet,
    rgb_path: str,
    chm_path: str,
    out_geojson: str,
    device: str = "cpu",
    tile_size: int = 256,
    overlap: int = 64,
    rgb_scale: float = 255.0,
    chm_max: float = 40.0,
    mask_threshold: float = 0.5,
    min_seed_distance: int = 4,
    min_instance_area: int = 5,
) -> gpd.GeoDataFrame:
    """Corre el modelo sobre un raster grande con ventanas solapadas.

    El solape evita artefactos de watershed en los bordes de tile: se
    predice sobre una ventana de `tile_size`, pero solo se conserva (y
    polygoniza) el recorte central de tamaño `stride`, descartando el
    margen de `overlap // 2` en cada lado.
    """
    model.eval()
    stride = tile_size - overlap
    margin = overlap // 2

    all_polygons = []
    next_id = 1

    with rasterio.open(rgb_path) as rgb_ds, rasterio.open(chm_path) as chm_ds:
        if (rgb_ds.width, rgb_ds.height) != (chm_ds.width, chm_ds.height):
            raise ValueError("RGB y CHM deben tener la misma grilla (usar tiling/reproject primero).")

        for row in range(0, rgb_ds.height, stride):
            for col in range(0, rgb_ds.width, stride):
                h = min(tile_size, rgb_ds.height - row)
                w = min(tile_size, rgb_ds.width - col)
                if h <= 0 or w <= 0:
                    continue

                window = Window(col, row, w, h)
                rgb_tile = rgb_ds.read(window=window).astype(np.float32) / rgb_scale
                chm_tile = chm_ds.read(1, window=window).astype(np.float32)
                chm_tile = np.clip(chm_tile, 0, chm_max) / chm_max

                rgb_t = torch.from_numpy(rgb_tile).unsqueeze(0).to(device)
                chm_t = torch.from_numpy(chm_tile[None, ...]).unsqueeze(0).to(device)

                outputs = model(rgb_t, chm_t)
                mask_prob = torch.sigmoid(outputs["mask_logits"])[0, 0].cpu().numpy()
                dist_map = outputs["dist_map"][0, 0].cpu().numpy()

                instance_map = instances_from_mask_dist(
                    mask_prob,
                    dist_map,
                    mask_threshold=mask_threshold,
                    min_seed_distance=min_seed_distance,
                    min_instance_area=min_instance_area,
                )

                # Recorta al núcleo del tile (sin margen) para evitar duplicados al pegar tiles.
                top = margin if row > 0 else 0
                left = margin if col > 0 else 0
                bottom = h - margin if row + h < rgb_ds.height else h
                right = w - margin if col + w < rgb_ds.width else w
                core = instance_map[top:bottom, left:right]
                core_transform = rgb_ds.window_transform(Window(col + left, row + top, right - left, bottom - top))

                polygons = polygonize_instances(core, core_transform)
                for poly in polygons:
                    poly["id"] = next_id
                    next_id += 1
                all_polygons.extend(polygons)

        crs = rgb_ds.crs

    if not all_polygons:
        gdf = gpd.GeoDataFrame(columns=["id", "area_px", "geometry"], geometry="geometry", crs=crs)
    else:
        gdf = gpd.GeoDataFrame(all_polygons, geometry="geometry", crs=crs)

    Path(out_geojson).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")
    return gdf
