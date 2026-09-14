"""Test end-to-end con rasters sintéticos: tiling -> dataset -> forward -> polygonize."""
from __future__ import annotations

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from src.data.dataset import TreeCrownTileDataset
from src.data.tiling import tile_dataset
from src.inference.postprocess import instances_from_mask_dist, polygonize_instances
from src.models.unet_dual import DualEncoderUNet


def _write_raster(path, array, transform, crs="EPSG:32633"):
    count = 1 if array.ndim == 2 else array.shape[0]
    height, width = array.shape[-2:]
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype=array.dtype, crs=crs, transform=transform,
    ) as dst:
        if array.ndim == 2:
            dst.write(array, 1)
        else:
            dst.write(array)


def test_tiling_and_pipeline(tmp_path):
    size = 128
    transform = from_origin(500000, 4649000, 0.5, 0.5)

    rng = np.random.default_rng(0)
    rgb = (rng.random((3, size, size)) * 255).astype(np.uint8)
    chm = (rng.random((size, size)) * 20).astype(np.float32)

    labels = np.zeros((size, size), dtype=np.int32)
    yy, xx = np.mgrid[0:size, 0:size]
    for i, (cy, cx) in enumerate([(30, 30), (30, 90), (90, 60)], start=1):
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= 15**2] = i

    rgb_path = tmp_path / "rgb.tif"
    chm_path = tmp_path / "chm.tif"
    labels_path = tmp_path / "labels.tif"
    _write_raster(rgb_path, rgb, transform)
    _write_raster(chm_path, chm, transform)
    _write_raster(labels_path, labels, transform)

    out_dir = tmp_path / "tiles"
    count = tile_dataset(
        str(rgb_path), str(chm_path), str(labels_path), str(out_dir),
        tile_size=64, overlap=16,
    )
    assert count > 0

    ds = TreeCrownTileDataset(str(out_dir))
    sample = ds[0]
    assert sample["rgb"].shape[0] == 3
    assert sample["mask"].shape == sample["dist"].shape

    model = DualEncoderUNet(rgb_channels=3, chm_channels=1, base_channels=8)
    model.eval()
    with torch.no_grad():
        out = model(sample["rgb"].unsqueeze(0), sample["chm"].unsqueeze(0))
    mask_prob = torch.sigmoid(out["mask_logits"])[0, 0].numpy()
    dist_map = out["dist_map"][0, 0].numpy()

    instance_map = instances_from_mask_dist(mask_prob, dist_map, min_instance_area=1)
    polygons = polygonize_instances(instance_map, transform)
    # No aseveramos una cantidad exacta (el modelo no está entrenado), solo que
    # el pipeline geoespacial corre sin errores y produce geometrías válidas.
    assert isinstance(polygons, list)
    for p in polygons:
        assert p["geometry"].is_valid
