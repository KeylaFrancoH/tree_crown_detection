"""Arma el dataset de entrenamiento para el caso CEDIA (copas en ambiente complejo).

Fuente de etiquetas: `proyecto_cedia/segmentacion_visual_verde_v3` (máscaras de
instancia por píxel, revisadas en varias pasadas) — NO
`proyecto_cedia/etiquetado_visual` (solo bounding boxes, sin segmentación).

Para cada una de las 7 imágenes etiquetadas:
  1. Localiza el ortomosaico georreferenciado real (8 bandas: RGB + 4 MS +
     alpha) en `proyecto_cedia/Datos/ortomosaicos/**`.
  2. Alinea el DSM (`.../segmentacion_visual_verde_v3/mde/*.tif`) a la grilla
     exacta del ortomosaico (tenían ~1px de desajuste) vía reproyección.
  3. Convierte el DSM alineado a un CHM aproximado (ver `src/data/height.py`).
  4. Carga la máscara de instancias PNG de 16 bits.
  5. Tilea todo en parches, calculando (mask, dist) por tile.

Divide por imagen completa (no por tile) para evitar fuga de datos entre
train/val/test: los tiles de una misma imagen jamás quedan repartidos entre
conjuntos distintos.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from src.data.height import estimate_canopy_height
from src.data.rasterize import instance_to_targets

try:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Falta Pillow en el entorno: pip install pillow") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CEDIA_DIR = PROJECT_ROOT / "proyecto_cedia"
LABELS_DIR = CEDIA_DIR / "segmentacion_visual_verde_v3"
RGB_SEARCH_ROOT = CEDIA_DIR / "Datos" / "ortomosaicos"

# División por imagen completa. Test = sitio nunca visto en train, para medir
# generalización real a un lugar nuevo; val = una imagen separada para elegir
# el mejor checkpoint durante el entrenamiento.
# (5/1/1: con solo 7 imágenes etiquetadas en total, se prioriza dejarle al
# entrenamiento la mayor cantidad de copas reales posible; el holdout de test
# se redujo a una sola imagen de parcela distinta, el chequeo de
# generalización más informativo que había con 2 imágenes de test.)
SPLIT = {
    "train": [
        "cedia_2-9-25_parcela1_rgb_ms_ze",
        "cedia_28-8-25_parcela2_rgb_ms_ze",
        "cedia_4-9-25_parcela2_rgb_ms_ze",
        "cedia_8-8-25_parcela2_rgb_ms_ze",
        "cedia_31-10-25_parcela2_rgb_ms_ze",
    ],
    "val": [
        "cedia_25-9-25_parcela2_rgb_ms_ze",
    ],
    "test": [
        "cedia_22-12-25_parcela1_rgb_ms_ze",  # parcela distinta -> generalización a sitio nuevo
    ],
}

RGB_BANDS = (1, 2, 3, 4, 5, 6, 7)  # banda 8 es alfa (máscara de validez), no dato espectral
TILE_SIZE = 256
OVERLAP = 32
MIN_VALID_FRACTION = 0.5
GROUND_WINDOW_M = 15.0  # > p99 del diámetro de copa observado (~11.8 m)


def _find_rgb_source(base_name: str) -> Path:
    matches = list(RGB_SEARCH_ROOT.glob(f"**/{base_name}.tif"))
    if not matches:
        raise FileNotFoundError(f"No se encontró el ortomosaico georreferenciado para {base_name}")
    if len(matches) > 1:
        raise RuntimeError(f"Más de un ortomosaico coincide con {base_name}: {matches}")
    return matches[0]


def _align_dsm(mde_path: Path, ref_ds: rasterio.DatasetReader) -> np.ndarray:
    with rasterio.open(mde_path) as src:
        src_band = src.read(1)
        dst = np.zeros((ref_ds.height, ref_ds.width), dtype=np.float32)
        reproject(
            source=src_band,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            src_nodata=src.nodata,
            dst_nodata=0.0,
            resampling=Resampling.bilinear,
        )
    return dst


def _process_image(base_name: str, split: str, out_root: Path) -> int:
    rgb_path = _find_rgb_source(base_name)
    mde_path = LABELS_DIR / "mde" / f"{base_name.replace('rgb_ms_ze', 'mde_ze')}.tif"
    mask_path = LABELS_DIR / "mascaras_instancias" / f"{base_name}.png"

    if not mde_path.exists():
        raise FileNotFoundError(mde_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(rgb_path) as rgb_ds:
        pixel_size_m = abs(rgb_ds.transform.a)
        alpha = rgb_ds.read(8)
        chm_aligned = _align_dsm(mde_path, rgb_ds)
        valid_mask = (alpha > 0) & (chm_aligned != 0.0)
        chm_full = estimate_canopy_height(chm_aligned, pixel_size_m, valid_mask, GROUND_WINDOW_M)

        instances_full = np.array(Image.open(mask_path))
        if instances_full.shape != (rgb_ds.height, rgb_ds.width):
            raise ValueError(
                f"{base_name}: máscara {instances_full.shape} no coincide con RGB "
                f"({rgb_ds.height}, {rgb_ds.width})"
            )

        count = 0
        stride = TILE_SIZE - OVERLAP
        # Solo tiles de tamaño completo (TILE_SIZE x TILE_SIZE): tiles de borde
        # más chicos no se pueden apilar en un batch de tamaño fijo.
        last_row = max(rgb_ds.height - TILE_SIZE, 0)
        last_col = max(rgb_ds.width - TILE_SIZE, 0)
        rows = sorted(set(list(range(0, last_row + 1, stride)) + [last_row]))
        cols = sorted(set(list(range(0, last_col + 1, stride)) + [last_col]))
        for row in rows:
            for col in cols:
                h = w = TILE_SIZE

                valid_crop = valid_mask[row : row + h, col : col + w]
                if valid_crop.mean() < MIN_VALID_FRACTION:
                    continue

                window = Window(col, row, w, h)
                rgb_tile = rgb_ds.read(RGB_BANDS, window=window).astype(np.float32)
                chm_tile = chm_full[row : row + h, col : col + w][None, ...].astype(np.float32)
                label_tile = instances_full[row : row + h, col : col + w]
                mask_tile, dist_tile = instance_to_targets(label_tile)

                sample = {
                    "rgb": rgb_tile,
                    "chm": chm_tile,
                    "mask": mask_tile[None, ...],
                    "dist": dist_tile[None, ...],
                    "instances": label_tile.astype(np.int32),
                }

                tile_transform = rgb_ds.window_transform(window)
                meta = {
                    "source_image": base_name,
                    "crs": rgb_ds.crs.to_string() if rgb_ds.crs else None,
                    "transform": list(tile_transform)[:6],
                    "row": row,
                    "col": col,
                    "height": h,
                    "width": w,
                }

                tile_name = f"{base_name}_{row}_{col}"
                np.savez_compressed(out_dir / f"{tile_name}.npz", **sample)
                with open(out_dir / f"{tile_name}.json", "w") as f:
                    json.dump(meta, f)
                count += 1

    return count


def main() -> None:
    out_root = PROJECT_ROOT / "cedia_dataset"
    summary = {}
    for split, names in SPLIT.items():
        summary[split] = {}
        for base_name in names:
            n = _process_image(base_name, split, out_root)
            summary[split][base_name] = n
            print(f"[{split}] {base_name}: {n} tiles")

    with open(out_root / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    for split, counts in summary.items():
        print(f"{split}: {sum(counts.values())} tiles totales ({len(counts)} imágenes)")


if __name__ == "__main__":
    main()
