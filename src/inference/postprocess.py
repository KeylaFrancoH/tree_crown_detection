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
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def _normalize01(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normaliza arr a [0,1] usando el min/max dentro de `mask` (resto queda en 0)."""
    out = np.zeros_like(arr, dtype=np.float32)
    if not mask.any():
        return out
    vals = arr[mask]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-6:
        out[mask] = 0.5
    else:
        out[mask] = (arr[mask] - lo) / (hi - lo)
    return out


def _merge_shallow_boundaries(
    instance_map: np.ndarray, elevation: np.ndarray, min_saddle_drop: float
) -> np.ndarray:
    """Fusiona instancias vecinas cuyo "valle" compartido no es lo bastante profundo.

    Dos semillas pueden ser dos bultos de la MISMA copa (textura/sombra) en vez
    de dos copas reales: watershed las separa igual porque cada una es un
    máximo local, aunque no haya un borde real entre ellas. Acá se mira, para
    cada par de instancias vecinas, el valor más alto de `elevation`
    (mask_prob o dist_map ya normalizado) a lo largo del borde que comparten
    — el "paso de montaña" entre sus dos picos. Si ese paso está casi tan alto
    como el más bajo de los dos picos, el "valle" es superficial: no es una
    separación real, y se fusionan.

    Args:
        instance_map: (H, W) int32, salida de watershed (antes de filtrar área).
        elevation: (H, W) float en [0,1] usada para medir picos y valles
            (normalmente el mismo campo sobre el que se buscaron las semillas).
        min_saddle_drop: caída mínima requerida entre el pico más bajo de los
            dos y el valle que los separa para considerarlos instancias
            distintas. 0 desactiva la fusión.

    Returns:
        instance_map remapeado (mismos IDs de fondo=0, instancias renumeradas).
    """
    if min_saddle_drop <= 0:
        return instance_map

    ids = np.unique(instance_map)
    ids = ids[ids != 0]
    if len(ids) <= 1:
        return instance_map

    max_id = int(instance_map.max())
    peaks = ndi.maximum(elevation, labels=instance_map, index=ids)
    peak_lookup = np.zeros(max_id + 1, dtype=np.float32)
    peak_lookup[ids] = peaks

    def _boundary_pairs(a: np.ndarray, b: np.ndarray, va: np.ndarray, vb: np.ndarray):
        diff = (a != b) & (a != 0) & (b != 0)
        lo = np.minimum(a[diff], b[diff])
        hi = np.maximum(a[diff], b[diff])
        val = np.maximum(va[diff], vb[diff])
        return lo, hi, val

    lo_h, hi_h, val_h = _boundary_pairs(
        instance_map[:, :-1], instance_map[:, 1:], elevation[:, :-1], elevation[:, 1:]
    )
    lo_v, hi_v, val_v = _boundary_pairs(
        instance_map[:-1, :], instance_map[1:, :], elevation[:-1, :], elevation[1:, :]
    )
    lo = np.concatenate([lo_h, lo_v]).astype(np.int64)
    hi = np.concatenate([hi_h, hi_v]).astype(np.int64)
    val = np.concatenate([val_h, val_v]).astype(np.float32)
    if lo.size == 0:
        return instance_map

    key = lo * (max_id + 1) + hi
    order = np.argsort(key)
    key_sorted, val_sorted = key[order], val[order]
    unique_keys, start_idx = np.unique(key_sorted, return_index=True)
    saddle_vals = np.maximum.reduceat(val_sorted, start_idx)

    parent = {int(i): int(i) for i in ids}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for k, saddle_val in zip(unique_keys.tolist(), saddle_vals.tolist()):
        a, b = int(k // (max_id + 1)), int(k % (max_id + 1))
        if min(peak_lookup[a], peak_lookup[b]) - saddle_val < min_saddle_drop:
            union(a, b)

    root_to_new: dict[int, int] = {}
    next_id = 1
    remap = np.zeros(max_id + 1, dtype=np.int32)
    for i in ids.tolist():
        root = find(i)
        if root not in root_to_new:
            root_to_new[root] = next_id
            next_id += 1
        remap[i] = root_to_new[root]

    return remap[instance_map]


def instances_from_mask_dist(
    mask_prob: np.ndarray,
    dist_map: np.ndarray,
    mask_threshold: float = 0.5,
    min_seed_distance: int = 4,
    min_instance_area: int = 5,
    smooth_sigma: float = 2.0,
    min_seed_prominence: float = 0.15,
    height_map: np.ndarray | None = None,
    height_weight: float = 0.5,
    min_saddle_drop: float = 0.03,
) -> np.ndarray:
    """Separa copas contiguas vía watershed marcador-controlado.

    Args:
        mask_prob: (H, W) probabilidad de copa en [0, 1].
        dist_map: (H, W) mapa de distancia normalizado predicho, en [0, 1].
        mask_threshold: umbral de binarización de la máscara.
        min_seed_distance: distancia mínima (px) entre semillas/máximos locales;
            controla cuán juntas pueden estar dos copas detectadas como distintas.
        min_instance_area: instancias más chicas que esto (en px²) se descartan.
        smooth_sigma: suavizado gaussiano de `dist_map` antes de buscar picos.
            Sin esto, ruido pixel a pixel en la prediccion del modelo genera
            miles de semillas espurias (sobre-segmentacion severa y watershed
            muy lento). 0 desactiva el suavizado.
        min_seed_prominence: valor minimo de `dist_map` (ya suavizado) para
            que un maximo local cuente como semilla; descarta picos de ruido
            de muy baja confianza.
        height_map: (H, W) opcional, altura real (CHM) alineada con la imagen.
            La textura de hojas/sombras en la foto es ruidosa; la altura real
            de un árbol es una señal más limpia de dónde está su copa. Si se
            pasa, las semillas se buscan en una mezcla de `dist_map` y
            `height_map` (cada uno normalizado 0-1 dentro de la máscara) en
            vez de solo `dist_map`, para evitar que texturas internas de una
            misma copa generen picos espurios que no corresponden a un pico
            real de altura.
        height_weight: peso de `height_map` en esa mezcla (0 = ignora la
            altura, 1 = usa solo altura). Sin efecto si `height_map` es None.
        min_saddle_drop: fusiona dos instancias vecinas si el valle que las
            separa no baja al menos esto respecto del más bajo de sus dos
            picos (ver `_merge_shallow_boundaries`). 0 desactiva la fusión.

    Returns:
        instance_map: (H, W) int32, 0 = fondo, 1..N = ID de copa.
    """
    binary_mask = mask_prob >= mask_threshold
    if not binary_mask.any():
        return np.zeros(mask_prob.shape, dtype=np.int32)

    dist_for_peaks = gaussian_filter(dist_map, sigma=smooth_sigma) if smooth_sigma > 0 else dist_map

    seed_surface = dist_for_peaks
    if height_map is not None and height_weight > 0:
        height_smooth = gaussian_filter(height_map, sigma=smooth_sigma) if smooth_sigma > 0 else height_map
        dist_norm = _normalize01(dist_for_peaks, binary_mask)
        height_norm = _normalize01(height_smooth, binary_mask)
        seed_surface = (1 - height_weight) * dist_norm + height_weight * height_norm

    coords = peak_local_max(
        seed_surface,
        min_distance=min_seed_distance,
        threshold_abs=min_seed_prominence if height_map is None else 0.0,
        labels=binary_mask,
        exclude_border=False,
    )
    seeds = np.zeros(mask_prob.shape, dtype=bool)
    seeds[tuple(coords.T)] = True
    markers, _ = ndi.label(seeds)

    if markers.max() == 0:
        # Sin máximos claros (copa muy chica/plana): tratarla como una sola instancia.
        markers, _ = ndi.label(binary_mask)

    instance_map = watershed(-dist_for_peaks, markers=markers, mask=binary_mask)
    instance_map = _merge_shallow_boundaries(instance_map, dist_for_peaks, min_saddle_drop)

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
