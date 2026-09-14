"""Post-procesado: máscara + mapa de distancia -> instancias -> polígonos.

Este es el paso que convierte la salida semántica del U-Net en copas
individuales (segmentación de instancias):

  1. Umbralar `mask` -> máscara binaria de "copa".
  2. Buscar máximos locales de `dist` dentro de la máscara -> semillas
     (aproximadamente, un máximo por copa individual, ya que `dist` es la
     distancia al borde normalizada por instancia).
  3. Watershed sobre `-dist`, restringido a la máscara, usando las semillas
     como marcadores -> cada píxel de copa queda asignado a una instancia.
  4. Polygonizar el raster de instancias resultante a polígonos
     georreferenciados (requiere el `transform` del tile).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def instances_from_mask_dist(
    mask_prob: np.ndarray,
    dist_map: np.ndarray,
    mask_threshold: float = 0.5,
    min_seed_distance: int = 4,
    min_instance_area: int = 5,
) -> np.ndarray:
    """Separa copas contiguas vía watershed marcador-controlado.

    Args:
        mask_prob: (H, W) probabilidad de copa en [0, 1].
        dist_map: (H, W) mapa de distancia normalizado predicho, en [0, 1].
        mask_threshold: umbral de binarización de la máscara.
        min_seed_distance: distancia mínima (px) entre semillas/máximos locales;
            controla cuán juntas pueden estar dos copas detectadas como distintas.
        min_instance_area: instancias más chicas que esto (en px²) se descartan.

    Returns:
        instance_map: (H, W) int32, 0 = fondo, 1..N = ID de copa.
    """
    binary_mask = mask_prob >= mask_threshold
    if not binary_mask.any():
        return np.zeros(mask_prob.shape, dtype=np.int32)

    coords = peak_local_max(
        dist_map,
        min_distance=min_seed_distance,
        labels=binary_mask,
        exclude_border=False,
    )
    seeds = np.zeros(mask_prob.shape, dtype=bool)
    seeds[tuple(coords.T)] = True
    markers, _ = ndi.label(seeds)

    if markers.max() == 0:
        # Sin máximos claros (copa muy chica/plana): tratarla como una sola instancia.
        markers, _ = ndi.label(binary_mask)

    instance_map = watershed(-dist_map, markers=markers, mask=binary_mask)

    if min_instance_area > 0:
        for instance_id in np.unique(instance_map):
            if instance_id == 0:
                continue
            area = int(np.sum(instance_map == instance_id))
            if area < min_instance_area:
                instance_map[instance_map == instance_id] = 0

    return instance_map.astype(np.int32)


def polygonize_instances(instance_map: np.ndarray, transform) -> list[dict]:
    """Convierte un raster de instancias en polígonos georreferenciados.

    Args:
        instance_map: (H, W) int32, salida de `instances_from_mask_dist`.
        transform: affine.Affine del tile (georreferenciación).

    Returns:
        Lista de dicts {"id": int, "geometry": shapely geometry, "area_px": int}.
    """
    from rasterio import features
    from shapely.geometry import shape

    polygons = []
    for geom, value in features.shapes(instance_map, mask=instance_map > 0, transform=transform):
        instance_id = int(value)
        polygons.append(
            {
                "id": instance_id,
                "geometry": shape(geom),
                "area_px": int(np.sum(instance_map == instance_id)),
            }
        )
    return polygons
