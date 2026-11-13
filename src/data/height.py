"""Estimación de altura de copa (CHM aproximado) a partir de un DSM crudo.

El MDE disponible en este proyecto es un DSM (elevación absoluta de
superficie: terreno + dosel), no un nDSM/CHM ya normalizado, y no hay un DTM
(modelo de terreno desnudo) separado para restar. Para evitar que el modelo
confunda pendiente del terreno con altura real de copa, se aproxima el
terreno con una apertura morfológica en escala de grises (min-filter seguido
de max-filter) con una ventana más ancha que la copa más grande observada en
las etiquetas: así el filtro "ve a través" de los árboles y sigue la
tendencia del suelo.

Esto es una aproximación estándar cuando no hay retornos de suelo
clasificados (LiDAR) disponibles; no reemplaza un nDSM real basado en DTM.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter


def _fill_nodata_nearest(array: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Rellena píxeles inválidos con el valor válido más cercano.

    Evita que los 0 de nodata se traten como "suelo muy bajo" y distorsionen
    el filtro de mínimo/máximo cerca de los bordes del vuelo.
    """
    if valid_mask.all():
        return array
    _, (inds_y, inds_x) = distance_transform_edt(~valid_mask, return_indices=True, return_distances=True)
    return array[inds_y, inds_x]


def estimate_canopy_height(
    dsm: np.ndarray,
    pixel_size_m: float,
    valid_mask: np.ndarray,
    ground_window_m: float = 15.0,
) -> np.ndarray:
    """Aproxima altura de copa (CHM) restando una estimación del terreno al DSM.

    Args:
        dsm: (H, W) elevación absoluta en metros.
        pixel_size_m: tamaño de píxel en metros (resolución del DSM).
        valid_mask: (H, W) bool, True donde el DSM tiene dato real.
        ground_window_m: ancho de la ventana de apertura morfológica, en metros.
            Debe ser mayor al diámetro de copa más grande esperado (acá se usa
            15 m, por encima del percentil 99 de diámetro de copa observado
            en las etiquetas de este dataset, ~11.8 m) para que el filtro
            no "coma" el centro de las copas más grandes.

    Returns:
        chm: (H, W) float32, altura estimada en metros (>= 0), 0 en píxeles inválidos.
    """
    dsm_filled = _fill_nodata_nearest(dsm.astype(np.float32), valid_mask)

    size_px = max(3, int(round(ground_window_m / pixel_size_m)))
    if size_px % 2 == 0:
        size_px += 1

    ground = maximum_filter(minimum_filter(dsm_filled, size=size_px), size=size_px)
    chm = np.clip(dsm_filled - ground, 0, None).astype(np.float32)
    chm[~valid_mask] = 0.0
    return chm
