"""Conversión de máscaras de instancia (raster) a los targets que consume el modelo.

Se asume que cada imagen de copas segmentadas es un raster GeoTIFF de una
banda donde 0 = fondo y cada copa individual tiene un ID entero distinto
(>0). A partir de eso se generan:

  - mask: máscara binaria (copa / fondo).
  - dist: transformada de distancia euclídea normalizada [0,1] por instancia
    (distancia al borde de la copa, cero fuera de las copas). Los máximos
    locales de este mapa son los "centros" de copa que se usan como semillas
    de watershed en inferencia para separar instancias que se tocan.

Si en cambio se parte de polígonos vectoriales (shapefile/GeoJSON), usar
`rasterize_polygons` para producir primero el raster de instancias con el
mismo grid que la imagen RGB/CHM, y luego pasarlo por `instance_to_targets`.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def instance_to_targets(instance_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convierte una máscara de instancias (H, W) con IDs enteros a (mask, dist).

    Returns:
        mask: array float32 (H, W) en {0, 1}.
        dist: array float32 (H, W) en [0, 1], distancia normalizada por instancia.
    """
    instance_mask = np.asarray(instance_mask)
    mask = (instance_mask > 0).astype(np.float32)
    dist = np.zeros_like(mask, dtype=np.float32)

    for instance_id in np.unique(instance_mask):
        if instance_id == 0:
            continue
        component = instance_mask == instance_id
        component_dist = distance_transform_edt(component)
        max_dist = component_dist.max()
        if max_dist > 0:
            component_dist = component_dist / max_dist
        dist[component] = component_dist[component]

    return mask, dist


def rasterize_polygons(
    geometries,
    out_shape: tuple[int, int],
    transform,
    instance: bool = True,
) -> np.ndarray:
    """Rasteriza polígonos de copas (shapefile/GeoJSON) a un array de instancias.

    Args:
        geometries: lista de geometrías shapely (una por copa), en el CRS del `transform`.
        out_shape: (H, W) del raster de salida, alineado con RGB/CHM.
        transform: affine.Affine del raster de referencia (rasterio).
        instance: si True, cada polígono recibe un ID distinto (1..N);
                  si False, se produce directamente una máscara binaria.
    """
    from rasterio import features

    if instance:
        shapes = [(geom, idx + 1) for idx, geom in enumerate(geometries)]
    else:
        shapes = [(geom, 1) for geom in geometries]

    out = features.rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )
    return out
