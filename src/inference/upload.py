"""Inferencia sobre un archivo subido por el usuario (no pre-tileado), para la
app de prueba (`app.py`). A diferencia de `src/inference/predict.py`
(`predict_orthomosaic`, pensado para pares RGB/CHM ya alineados en la misma
grilla), acá se es tolerante a lo que suba el usuario:

  - El raster de altura puede faltar (se reemplaza por ceros) o venir en una
    grilla distinta a la de la imagen (se realinea vía reproyección).
  - Si se marca como DSM crudo, se le resta una estimación del terreno
    (ver `src/data/height.py`) para aproximar altura de copa.
  - La imagen puede tener menos o más bandas que las que espera el modelo
    (se rellena con ceros o se recorta a las primeras N).

Importante sobre el watershed: se acumulan primero `mask_prob`/`dist_map` de
TODOS los tiles en dos mapas de tamaño completo (promediando el solape), y
recién después se corre `instances_from_mask_dist` una única vez sobre esos
mapas completos. Correr el watershed tile por tile y después pegar los
resultados (como hacía una versión anterior de este archivo) genera un
patrón de grilla visible: cada tile arma sus propias instancias sin ver lo
que hay en el tile vecino, y un mismo árbol que cae a caballo entre dos
tiles termina partido en pedazos con IDs sin relación entre sí.
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from src.data.height import estimate_canopy_height
from src.inference.postprocess import instances_from_mask_dist, polygonize_instances

MAX_PX_FOR_WATERSHED = 2_000_000  # por encima de esto, se corre el watershed a resolución reducida


def _read_rgb_bands(ds: rasterio.DatasetReader, n_channels: int) -> np.ndarray:
    available = ds.count
    n_read = min(available, n_channels)
    bands = ds.read(list(range(1, n_read + 1))).astype(np.float32)
    if n_read < n_channels:
        pad = np.zeros((n_channels - n_read, ds.height, ds.width), dtype=np.float32)
        bands = np.concatenate([bands, pad], axis=0)
    return bands


def _align_chm(chm_bytes: bytes | None, ref_ds: rasterio.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (chm alineado a la grilla de ref_ds, máscara de validez)."""
    if chm_bytes is None:
        zeros = np.zeros((ref_ds.height, ref_ds.width), dtype=np.float32)
        return zeros, np.zeros_like(zeros, dtype=bool)

    with MemoryFile(chm_bytes) as mem:
        with mem.open() as chm_ds:
            same_grid = (
                chm_ds.width == ref_ds.width
                and chm_ds.height == ref_ds.height
                and chm_ds.crs == ref_ds.crs
                and chm_ds.transform.almost_equals(ref_ds.transform, precision=1e-6)
            )
            src_band = chm_ds.read(1).astype(np.float32)
            if same_grid:
                aligned = src_band
            elif chm_ds.crs and ref_ds.crs:
                aligned = np.zeros((ref_ds.height, ref_ds.width), dtype=np.float32)
                reproject(
                    source=src_band,
                    destination=aligned,
                    src_transform=chm_ds.transform,
                    src_crs=chm_ds.crs,
                    dst_transform=ref_ds.transform,
                    dst_crs=ref_ds.crs,
                    src_nodata=chm_ds.nodata,
                    dst_nodata=0.0,
                    resampling=Resampling.bilinear,
                )
            else:
                # Sin CRS en alguno de los dos: lo único que se puede hacer es
                # un resize simple si al menos las dimensiones no coinciden.
                from skimage.transform import resize

                aligned = resize(src_band, (ref_ds.height, ref_ds.width), preserve_range=True).astype(np.float32)

            nodata = chm_ds.nodata if chm_ds.nodata is not None else 0.0
            valid = aligned != nodata
    return aligned, valid


def segment_uploaded(
    rgb_bytes: bytes,
    chm_bytes: bytes | None,
    model,
    config: dict,
    device: str,
    dsm_is_raw: bool = True,
    ground_window_m: float = 15.0,
    tile_size: int = 256,
    overlap: int = 64,
    mask_threshold: float = 0.5,
    min_seed_distance: int = 24,
    min_instance_area: int = 100,
    smooth_sigma: float = 2.0,
    min_seed_prominence: float = 0.15,
    min_saddle_drop: float = 0.03,
    use_height_seeds: bool = False,
    height_weight: float = 0.5,
    progress_callback=None,
) -> dict:
    """Corre el modelo sobre una imagen subida, con sliding window.

    Returns:
        dict con: rgb_true_color (H,W,3 uint8), instance_map (H,W int32),
        polygons (lista, vacía si no hay CRS), crs (str o None), n_instances.
    """
    import torch

    with MemoryFile(rgb_bytes) as mem:
        with mem.open() as rgb_ds:
            n_channels = config["model"]["rgb_channels"]
            pixel_size_m = abs(rgb_ds.transform.a) if rgb_ds.crs else 1.0
            height, width = rgb_ds.height, rgb_ds.width

            rgb_full = _read_rgb_bands(rgb_ds, n_channels)

            chm_aligned, valid_mask = _align_chm(chm_bytes, rgb_ds)
            if chm_bytes is not None:
                if dsm_is_raw:
                    chm_final = estimate_canopy_height(chm_aligned, pixel_size_m, valid_mask, ground_window_m)
                else:
                    chm_final = np.clip(chm_aligned, 0, None).astype(np.float32)
                    chm_final[~valid_mask] = 0.0
            else:
                chm_final = chm_aligned  # ya son ceros

            rgb_scale = config["data"]["rgb_scale"]
            chm_max = config["data"]["chm_max"]

            stride = tile_size - overlap

            mask_accum = np.zeros((height, width), dtype=np.float32)
            dist_accum = np.zeros((height, width), dtype=np.float32)
            weight_accum = np.zeros((height, width), dtype=np.float32)

            ys = list(range(0, max(height - tile_size, 0) + 1, stride)) or [0]
            xs = list(range(0, max(width - tile_size, 0) + 1, stride)) or [0]
            if ys[-1] + tile_size < height:
                ys.append(max(height - tile_size, 0))
            if xs[-1] + tile_size < width:
                xs.append(max(width - tile_size, 0))
            positions = [(y, x) for y in ys for x in xs]
            total = len(positions)

            with torch.no_grad():
                for done, (row, col) in enumerate(positions, start=1):
                    row2 = min(row + tile_size, height)
                    col2 = min(col + tile_size, width)
                    h, w = row2 - row, col2 - col

                    rgb_tile = rgb_full[:, row:row2, col:col2]
                    chm_tile = chm_final[row:row2, col:col2]
                    if h < tile_size or w < tile_size:
                        pad_h, pad_w = tile_size - h, tile_size - w
                        rgb_tile = np.pad(rgb_tile, ((0, 0), (0, pad_h), (0, pad_w)))
                        chm_tile = np.pad(chm_tile, ((0, pad_h), (0, pad_w)))

                    rgb_t = torch.from_numpy(rgb_tile / rgb_scale).unsqueeze(0).to(device)
                    chm_t = torch.from_numpy(np.clip(chm_tile, 0, chm_max)[None, None] / chm_max).to(device)

                    out = model(rgb_t, chm_t)
                    mask_prob = torch.sigmoid(out["mask_logits"])[0, 0].cpu().numpy()[:h, :w]
                    dist_map = out["dist_map"][0, 0].cpu().numpy()[:h, :w]

                    mask_accum[row:row2, col:col2] += mask_prob
                    dist_accum[row:row2, col:col2] += dist_map
                    weight_accum[row:row2, col:col2] += 1

                    if progress_callback is not None:
                        progress_callback(done / total)

            weight_accum[weight_accum == 0] = 1
            mask_full = mask_accum / weight_accum
            dist_full = dist_accum / weight_accum

            # El watershed a resolución nativa es lento en imágenes grandes (varios
            # millones de píxeles): se corre en una versión reducida y se escala el
            # resultado de vuelta con nearest-neighbor, para no mezclar IDs de
            # instancias distintas al interpolar.
            downscale = max(1, int(np.ceil(((height * width) / MAX_PX_FOR_WATERSHED) ** 0.5)))
            if downscale > 1:
                from skimage.transform import resize

                small_shape = (height // downscale, width // downscale)
                mask_small = resize(mask_full, small_shape, preserve_range=True).astype(np.float32)
                dist_small = resize(dist_full, small_shape, preserve_range=True).astype(np.float32)
                height_small = (
                    resize(chm_final, small_shape, preserve_range=True).astype(np.float32)
                    if use_height_seeds
                    else None
                )
                instances_small = instances_from_mask_dist(
                    mask_small,
                    dist_small,
                    mask_threshold=mask_threshold,
                    min_seed_distance=max(1, min_seed_distance // downscale),
                    min_instance_area=max(1, min_instance_area // (downscale * downscale)),
                    smooth_sigma=max(0.5, smooth_sigma / downscale) if smooth_sigma > 0 else 0.0,
                    min_seed_prominence=min_seed_prominence,
                    height_map=height_small,
                    height_weight=height_weight,
                    min_saddle_drop=min_saddle_drop,
                )
                instance_map = resize(
                    instances_small, (height, width), order=0, preserve_range=True, anti_aliasing=False
                ).astype(np.int32)
            else:
                instance_map = instances_from_mask_dist(
                    mask_full,
                    dist_full,
                    mask_threshold=mask_threshold,
                    min_seed_distance=min_seed_distance,
                    min_instance_area=min_instance_area,
                    height_map=chm_final if use_height_seeds else None,
                    height_weight=height_weight,
                    min_saddle_drop=min_saddle_drop,
                    smooth_sigma=smooth_sigma,
                    min_seed_prominence=min_seed_prominence,
                )

            img = rgb_full[:3].transpose(1, 2, 0)
            p2, p98 = np.percentile(img, [2, 98])
            img = np.clip((img - p2) / max(p98 - p2, 1e-6), 0, 1)
            rgb_true_color = (img * 255).astype(np.uint8)

            polygons = []
            crs = None
            if rgb_ds.crs:
                crs = rgb_ds.crs.to_string()
                polygons = polygonize_instances(instance_map, rgb_ds.transform)

    return {
        "rgb_true_color": rgb_true_color,
        "instance_map": instance_map,
        "polygons": polygons,
        "crs": crs,
        "n_instances": int(instance_map.max()),
    }
