"""App Streamlit simple para probar el detector de copas entrenado.

Uso:
    streamlit run app.py

Sube una imagen RGB (tif georreferenciado, o jpg/png cualquiera) y,
opcionalmente, un modelo de altura (CHM/relieve, tif de 1 banda alineado
con el RGB). Corre el modelo entrenado (checkpoints/best.pth) con una
ventana deslizante sobre toda la imagen y muestra las copas detectadas.

Si no se sube CHM, se usa un canal de altura en cero (el modelo igual
funciona, pero pierde la ayuda de la altura para separar copas que se tocan).
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import rasterio
import streamlit as st
import streamlit.components.v1 as components
import torch
import yaml
from PIL import Image

from src.inference.postprocess import instances_from_mask_dist, polygonize_instances
from src.models.unet_dual import build_model

st.set_page_config(page_title="Detector de copas", layout="wide")
st.title("Detector de copas de arboles")
st.caption("Sube una imagen RGB para detectar copas individuales con el modelo entrenado.")


@st.cache_resource
def load_model(checkpoint_path: str, config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, cfg, device


def read_rgb(file, rgb_channels: int = 3) -> tuple[np.ndarray, float | None, bool]:
    """Devuelve (array (rgb_channels, H, W) float32, tamano de pixel en metros o None,
    si tiene banda NIR real).

    Para un .tif multiespectral del sensor CEDIA (8 bandas: 1-3 RGB,
    4-5 Green/Red del sensor MS, 6 RedEdge, 7 NIR, 8 alpha) con
    rgb_channels=5, extrae directamente [R, G, B, RedEdge, NIR] del mismo
    archivo -- no hace falta subir un CHM aparte para esto, la altura solo
    se usa para generar las etiquetas de entrenamiento, no en inferencia.
    Si el archivo no tiene esas bandas (jpg/png o tif RGB comun), se
    completan los canales faltantes repitiendo la banda roja; en ese caso
    has_real_nir vuelve False, porque esos canales no llevan informacion
    espectral real (no sirven para calcular NDVI).

    El tamano de pixel (metros/pixel, de la georreferenciacion del tif) se
    devuelve para poder expresar los parametros de deteccion (distancia entre
    copas, area minima) en metros reales en vez de pixeles -- una misma
    imagen vista de cerca o con un recorte tiene resoluciones en pixeles
    muy distintas, y un umbral fijo en pixeles no sirve para ambas.
    """
    name = file.name.lower()
    data = file.read()
    pixel_size = None
    has_real_nir = False
    if name.endswith((".tif", ".tiff")):
        import rasterio

        with rasterio.MemoryFile(data) as memfile, memfile.open() as ds:
            if ds.crs is not None:
                pixel_size = abs(ds.transform.a)
            if rgb_channels == 5 and ds.count >= 8:
                arr = ds.read([1, 2, 3, 6, 7]).astype(np.float32)
                return arr, pixel_size, True
            n = min(3, ds.count)
            arr = ds.read(list(range(1, n + 1))).astype(np.float32)
            if n < 3:
                arr = np.repeat(arr[:1], 3, axis=0)
    else:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(im).astype(np.float32).transpose(2, 0, 1)
    if rgb_channels == 5 and arr.shape[0] < 5:
        arr = np.concatenate([arr, np.repeat(arr[:1], 5 - arr.shape[0], axis=0)], axis=0)
    return arr, pixel_size, has_real_nir


def read_chm(file, target_shape: tuple[int, int]) -> np.ndarray:
    """Devuelve un array (H, W) float32 alineado al tamano del RGB (resize simple si hace falta)."""
    import rasterio
    from rasterio.enums import Resampling

    data = file.read()
    with rasterio.MemoryFile(data) as memfile, memfile.open() as ds:
        arr = ds.read(
            1,
            out_shape=target_shape,
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    return arr


def stretch_for_display(rgb: np.ndarray) -> np.ndarray:
    """(3,H,W) numerico -> (H,W,3) uint8 para mostrar, con estiramiento 2-98 pctl si hace falta."""
    if rgb.max() <= 255 and rgb.dtype != np.float64:
        img = np.clip(rgb, 0, 255).astype(np.uint8)
        return img.transpose(1, 2, 0)
    valid = rgb[rgb > 0]
    lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0, 1)
    out = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8).transpose(1, 2, 0)


def polygons_to_pixel_coords(instances: np.ndarray, simplify_tolerance: float = 1.5) -> list[dict]:
    """Instancias -> lista de poligonos en coordenadas de pixel (col, row), sin georreferenciar.

    Se simplifica cada poligono (Douglas-Peucker) porque el raster de instancias
    sale de un watershed corrido en baja resolucion y re-escalado con nearest:
    sin simplificar, los bordes quedan en escalera con miles de vertices por
    copa, inmanejable para editar a mano en el navegador.
    """
    polys = polygonize_instances(instances, transform=rasterio.Affine.identity())
    out = []
    for p in polys:
        geom = p["geometry"].simplify(simplify_tolerance, preserve_topology=True)
        geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for g in geoms:
            if g.is_empty or len(g.exterior.coords) < 4:
                continue
            out.append({"id": int(p["id"]), "coords": list(g.exterior.coords)})
    return out


def build_editor_html(display_img: np.ndarray, polygons: list[dict], max_dim: int = 1600) -> str:
    """Mapa Leaflet + Leaflet-Geoman embebido, con las copas detectadas como
    poligonos editables (arrastrar vertices, cortar, borrar, dibujar nuevos)
    sobre la imagen. "Guardar y descargar" arma el GeoJSON en el navegador
    (en coordenadas de pixel de la imagen original, no hay georreferenciacion
    real todavia) y dispara la descarga -- no hace falta mandar nada de vuelta
    a Streamlit/Python.
    """
    H, W = display_img.shape[:2]
    scale = min(1.0, max_dim / max(H, W))
    disp_h, disp_w = max(1, int(H * scale)), max(1, int(W * scale))

    img_resized = Image.fromarray(display_img).resize((disp_w, disp_h), Image.BILINEAR)
    buf = io.BytesIO()
    img_resized.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    features = []
    for p in polygons:
        pts = [[disp_h - y * scale, x * scale] for x, y in p["coords"]]
        features.append({"id": p["id"], "pts": pts})
    features_json = json.dumps(features)

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<link rel="stylesheet" href="https://unpkg.com/@geoman-io/leaflet-geoman-free@2.15.0/dist/leaflet-geoman.css" />
<style>
  html, body {{ margin:0; padding:0; height:100%; font-family:sans-serif; }}
  #toolbar {{ padding:8px; background:#f0f2f6; display:flex; gap:8px; align-items:center; }}
  #toolbar button {{ padding:6px 14px; border-radius:6px; border:1px solid #ccc; cursor:pointer; background:white; }}
  #save-btn {{ background:#16a34a; color:white; border:none; }}
  #map {{ width:100%; height:700px; background:#111; }}
  #count {{ margin-left:auto; font-size:14px; color:#444; }}
</style>
</head>
<body>
<div id="toolbar">
  <button id="discard-btn">Descartar todo</button>
  <button id="cancel-btn">Cancelar</button>
  <button id="save-btn">Guardar y descargar</button>
  <span id="count"></span>
</div>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/@geoman-io/leaflet-geoman-free@2.15.0/dist/leaflet-geoman.min.js"></script>
<script>
  const H = {disp_h}, W = {disp_w}, SCALE = {scale};
  const initialFeatures = {features_json};

  const map = L.map('map', {{ crs: L.CRS.Simple, minZoom: -5 }});
  const bounds = [[0, 0], [H, W]];
  L.imageOverlay('data:image/png;base64,{img_b64}', bounds).addTo(map);
  map.fitBounds(bounds);

  const layerGroup = L.featureGroup().addTo(map);

  function addPolygon(pts, id) {{
    const poly = L.polygon(pts, {{ color: '#2563eb', weight: 2, fillOpacity: 0.08 }});
    poly.feature = {{ type: 'Feature', properties: {{ id: id }} }};
    poly.addTo(layerGroup);
  }}

  function updateCount() {{
    document.getElementById('count').textContent = layerGroup.getLayers().length + ' copas';
  }}

  function loadInitial() {{
    layerGroup.clearLayers();
    initialFeatures.forEach(f => addPolygon(f.pts, f.id));
    updateCount();
  }}
  loadInitial();

  map.pm.addControls({{
    position: 'topleft',
    drawMarker: false, drawCircleMarker: false, drawPolyline: false,
    drawRectangle: false, drawCircle: false, drawText: false,
    drawPolygon: true, editMode: true, dragMode: true,
    cutPolygon: true, removalMode: true, rotateMode: false,
  }});

  map.on('pm:create', e => {{ e.layer.addTo(layerGroup); updateCount(); }});
  map.on('pm:remove', updateCount);
  layerGroup.on('layerremove', updateCount);

  document.getElementById('discard-btn').addEventListener('click', () => {{
    layerGroup.clearLayers();
    updateCount();
  }});
  document.getElementById('cancel-btn').addEventListener('click', loadInitial);

  document.getElementById('save-btn').addEventListener('click', () => {{
    const feats = [];
    layerGroup.eachLayer(layer => {{
      if (!layer.getLatLngs) return;
      const latlngs = layer.getLatLngs()[0];
      const coords = latlngs.map(ll => [ll.lng / SCALE, (H - ll.lat) / SCALE]);
      coords.push(coords[0]);
      feats.push({{
        type: 'Feature',
        properties: {{ id: layer.feature ? layer.feature.properties.id : null }},
        geometry: {{ type: 'Polygon', coordinates: [coords] }},
      }});
    }});
    const fc = {{ type: 'FeatureCollection', features: feats }};
    const blob = new Blob([JSON.stringify(fc, null, 2)], {{ type: 'application/geo+json' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'copas_editadas.geojson';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }});
</script>
</body>
</html>
"""


def draw_boundaries(rgb_uint8: np.ndarray, instances: np.ndarray) -> np.ndarray:
    """Dibuja bordes rojos entre instancias. Evita skimage.find_boundaries (crashea en Windows)."""
    out = rgb_uint8.copy()
    b = np.zeros(instances.shape, dtype=bool)
    b[1:, :] |= instances[1:, :] != instances[:-1, :]
    b[:-1, :] |= instances[:-1, :] != instances[1:, :]
    b[:, 1:] |= instances[:, 1:] != instances[:, :-1]
    b[:, :-1] |= instances[:, :-1] != instances[:, 1:]
    boundary = b & (instances > 0)
    out[boundary] = [255, 0, 0]
    return out


@torch.no_grad()
def run_inference(model, rgb, chm, cfg, device, tile_size=256, overlap=64, progress=None):
    data_cfg = cfg.get("data", {})
    chm_max = data_cfg.get("chm_max", 10.0)
    # El config asume el sensor de entrenamiento (16 bits -> /65535), pero una
    # foto subida puede venir en 8 bits (jpg/png, u otro tif) -- con el
    # divisor fijo esos valores quedan aplastados cerca de 0 y el modelo ve
    # practicamente una imagen negra. Se detecta el rango real de la imagen
    # en vez de confiar ciegamente en el config.
    peak = float(rgb.max())
    if peak <= 255.0:
        rgb_scale = 255.0
    elif peak <= 4095.0:
        rgb_scale = 4095.0
    else:
        rgb_scale = data_cfg.get("rgb_scale", 65535.0)
    _, H, W = rgb.shape

    mask_full = np.zeros((H, W), dtype=np.float32)
    dist_full = np.zeros((H, W), dtype=np.float32)
    weight_full = np.zeros((H, W), dtype=np.float32)
    stride = tile_size - overlap

    ys = list(range(0, max(H - tile_size, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - tile_size, 0) + 1, stride)) or [0]
    if ys[-1] + tile_size < H:
        ys.append(max(H - tile_size, 0))
    if xs[-1] + tile_size < W:
        xs.append(max(W - tile_size, 0))

    positions = [(y, x) for y in ys for x in xs]
    total = len(positions)
    done = 0
    batch_size = 16

    for i in range(0, total, batch_size):
        batch_positions = positions[i:i + batch_size]
        rgb_batch, chm_batch, sizes = [], [], []
        for y, x in batch_positions:
            y2, x2 = min(y + tile_size, H), min(x + tile_size, W)
            rgb_tile = rgb[:, y:y2, x:x2]
            chm_tile = chm[y:y2, x:x2]
            h, w = rgb_tile.shape[1:]
            sizes.append((h, w))
            pad_h, pad_w = tile_size - h, tile_size - w
            if pad_h or pad_w:
                rgb_tile = np.pad(rgb_tile, ((0, 0), (0, pad_h), (0, pad_w)))
                chm_tile = np.pad(chm_tile, ((0, pad_h), (0, pad_w)))
            rgb_batch.append(rgb_tile / rgb_scale)
            chm_batch.append(np.clip(chm_tile, 0, chm_max) / chm_max)

        rgb_t = torch.from_numpy(np.stack(rgb_batch)).float().to(device)
        chm_t = torch.from_numpy(np.stack(chm_batch)).float().unsqueeze(1).to(device)

        out = model(rgb_t, chm_t)
        mp_batch = torch.sigmoid(out["mask_logits"])[:, 0].cpu().numpy()
        dm_batch = out["dist_map"][:, 0].cpu().numpy()

        for (y, x), (h, w), mp, dm in zip(batch_positions, sizes, mp_batch, dm_batch):
            y2, x2 = y + h, x + w
            mask_full[y:y2, x:x2] += mp[:h, :w]
            dist_full[y:y2, x:x2] += dm[:h, :w]
            weight_full[y:y2, x:x2] += 1

        done += len(batch_positions)
        if progress is not None:
            progress(done / total)

    weight_full[weight_full == 0] = 1
    return mask_full / weight_full, dist_full / weight_full


with st.sidebar:
    st.header("Modelo")
    checkpoint_path = st.text_input("Checkpoint (.pth)", value="checkpoints_rgb_5band/best.pth")
    config_path = st.text_input("Config (.yaml)", value="configs/rgb_pretrained_5band.yaml")
    st.header("Parametros de deteccion")
    mask_threshold = st.slider("Umbral de mascara", 0.0, 1.0, 0.5, 0.05)
    min_seed_distance_m = st.slider(
        "Distancia minima entre copas (m)", 0.2, 10.0, 2.0, 0.1,
        help=(
            "En metros reales, no en pixeles: se convierte a pixeles usando la "
            "resolucion (georreferenciacion) de la imagen subida, asi el mismo "
            "valor sirve tanto para la parcela completa como para un recorte/zoom."
        ),
    )
    min_instance_area_m2 = st.slider("Area minima de copa (m2)", 0.1, 20.0, 1.0, 0.1)
    fallback_pixel_size = st.number_input(
        "Tamano de pixel asumido (m), si la imagen no esta georreferenciada",
        min_value=0.001, max_value=1.0, value=0.05, step=0.005, format="%.3f",
    )
    tile_size = st.selectbox("Tamano de tile", [128, 256, 512], index=1)
    overlap = st.selectbox("Solape entre tiles", [16, 32, 64, 96], index=2)
    ndvi_threshold = st.slider(
        "Umbral NDVI (filtra superficies sin vegetacion)", -1.0, 1.0, 0.2, 0.05,
        help=(
            "Las etiquetas de entrenamiento salen del relieve de altura (CHM), que no "
            "distingue vegetacion de estructuras (veredas, muros, escalones con desnivel): "
            "esto descarta deteciones sobre pixeles con bajo NDVI (poca vegetacion real) "
            "usando las bandas RedEdge/NIR, si estan disponibles. Solo aplica con el "
            "modelo de 5 bandas."
        ),
    )

needs_chm = False
if Path(config_path).exists():
    with open(config_path) as f:
        _cfg_preview = yaml.safe_load(f)
    needs_chm = _cfg_preview.get("model", {}).get("type", "dual") == "dual"

if needs_chm:
    col1, col2 = st.columns(2)
    with col1:
        rgb_file = st.file_uploader("Imagen RGB (.tif/.tiff/.jpg/.png)", type=["tif", "tiff", "jpg", "jpeg", "png"])
    with col2:
        chm_file = st.file_uploader("Modelo de altura / CHM (.tif)", type=["tif", "tiff"])
else:
    st.caption(
        "Solo hace falta subir la foto (no el modelo de altura: eso se usa unicamente "
        "para entrenar). Si el .tif tiene bandas infrarrojas (sensor multiespectral "
        "CEDIA), se usan automaticamente junto al RGB."
    )
    rgb_file = st.file_uploader("Imagen (.tif/.tiff/.jpg/.png)", type=["tif", "tiff", "jpg", "jpeg", "png"])
    chm_file = None

if rgb_file is not None and st.button("Detectar copas", type="primary"):
    if not Path(checkpoint_path).exists():
        st.error(f"No se encontro el checkpoint: {checkpoint_path}")
        st.stop()

    with st.spinner("Cargando modelo..."):
        model, cfg, device = load_model(checkpoint_path, config_path)
    st.caption(f"Usando device: {device}")

    rgb_channels = cfg.get("model", {}).get("rgb_channels", 3)
    with st.spinner("Leyendo imagen..."):
        rgb, pixel_size, has_real_nir = read_rgb(rgb_file, rgb_channels=rgb_channels)
        _, H, W = rgb.shape
    if rgb_channels == 5 and not has_real_nir:
        st.caption(
            "Esta imagen no trae bandas RedEdge/NIR reales (se completaron repitiendo "
            "el rojo) -- el filtro NDVI se desactiva para esta corrida porque no hay "
            "informacion espectral real para calcularlo."
        )
    if pixel_size is None:
        pixel_size = fallback_pixel_size
        st.caption(
            f"La imagen no tiene georreferenciacion -- asumiendo {pixel_size:.3f} m/pixel "
            "para convertir distancia/area minima."
        )
    else:
        st.caption(f"Resolucion detectada: {pixel_size:.4f} m/pixel.")
    min_seed_distance = max(1, round(min_seed_distance_m / pixel_size))
    min_instance_area = max(1, round(min_instance_area_m2 / (pixel_size ** 2)))

    if chm_file is not None:
        with st.spinner("Leyendo modelo de altura..."):
            chm = read_chm(chm_file, target_shape=(H, W))
    else:
        chm = np.zeros((H, W), dtype=np.float32)

    progress_bar = st.progress(0.0, text="Corriendo el modelo sobre la imagen...")
    mask_full, dist_full = run_inference(
        model, rgb, chm, cfg, device,
        tile_size=tile_size, overlap=overlap,
        progress=lambda f: progress_bar.progress(f, text=f"Procesando... {f*100:.0f}%"),
    )
    progress_bar.empty()

    if rgb.shape[0] >= 5 and has_real_nir:
        red = rgb[0].astype(np.float32)
        nir = rgb[4].astype(np.float32)
        ndvi = (nir - red) / (nir + red + 1e-6)
        mask_full = mask_full * (ndvi > ndvi_threshold)

    # Watershed sobre la imagen completa a resolucion nativa es lento en CPU para
    # imagenes grandes (varios millones de pixeles). Lo corremos en una version
    # reducida y despues escalamos el resultado de vuelta (nearest, para no
    # mezclar IDs de instancias distintas).
    max_px_for_watershed = 2_000_000
    downscale = max(1, int(np.ceil(((H * W) / max_px_for_watershed) ** 0.5)))

    # El mapa de distancia predicho por el modelo tiene ruido interno (variacion
    # de luz/sombra dentro del follaje, textura de hojas) que genera picos locales
    # espurios dentro de una misma copa grande -- sin suavizarlo bastante, el
    # watershed las parte en varios pedazos aunque la distancia minima entre
    # semillas ya este bien calibrada. El suavizado se expresa en metros reales
    # (una fraccion de la distancia minima entre copas) para que escale igual de
    # bien que el resto de los parametros, sin agregar otro control manual.
    smooth_sigma_m = min_seed_distance_m / 3.0
    min_seed_prominence = 0.3

    with st.spinner(f"Separando copas individuales (watershed, escala 1/{downscale})..."):
        if downscale > 1:
            small_size = (W // downscale, H // downscale)
            mask_small = np.array(Image.fromarray(mask_full, mode="F").resize(small_size, Image.BILINEAR))
            dist_small = np.array(Image.fromarray(dist_full, mode="F").resize(small_size, Image.BILINEAR))
            instances_small = instances_from_mask_dist(
                mask_small, dist_small,
                mask_threshold=mask_threshold,
                min_seed_distance=max(1, min_seed_distance // downscale),
                min_instance_area=max(1, min_instance_area // (downscale * downscale)),
                smooth_sigma=max(1.0, (smooth_sigma_m / pixel_size) / downscale),
                min_seed_prominence=min_seed_prominence,
            )
            instances = np.array(
                Image.fromarray(instances_small.astype(np.int32), mode="I").resize((W, H), Image.NEAREST)
            )
        else:
            instances = instances_from_mask_dist(
                mask_full, dist_full,
                mask_threshold=mask_threshold,
                min_seed_distance=min_seed_distance,
                min_instance_area=min_instance_area,
                smooth_sigma=max(1.0, smooth_sigma_m / pixel_size),
                min_seed_prominence=min_seed_prominence,
            )

    n_crowns = len(np.unique(instances)) - (1 if instances.min() == 0 else 0)
    st.success(f"Copas detectadas: {n_crowns}")

    display_rgb = stretch_for_display(rgb[:3])
    overlay = draw_boundaries(display_rgb, instances)

    tab0, tab1, tab2, tab3 = st.tabs(
        ["Editor interactivo", "Copas detectadas", "Mascara cruda", "Imagen original"]
    )
    with tab0:
        st.caption(
            "Arrastra los vertices para ajustar una copa, usa las herramientas de la "
            "izquierda para cortar, borrar o dibujar copas nuevas. 'Guardar y descargar' "
            "baja un .geojson con tus correcciones (coordenadas en pixeles de la imagen "
            "subida, todavia sin georreferenciar)."
        )
        with st.spinner("Preparando editor..."):
            pixel_polygons = polygons_to_pixel_coords(instances)
            editor_html = build_editor_html(display_rgb, pixel_polygons)
        components.html(editor_html, height=760, scrolling=False)
    with tab1:
        st.image(overlay, use_container_width=True)
    with tab2:
        st.image((mask_full * 255).astype(np.uint8), use_container_width=True, clamp=True)
    with tab3:
        st.image(display_rgb, use_container_width=True)

    ids = np.unique(instances)
    ids = ids[ids > 0]
    rows = ["id,area_px,centro_x,centro_y"]
    for i in ids:
        ys_i, xs_i = np.where(instances == i)
        rows.append(f"{i},{len(ys_i)},{xs_i.mean():.1f},{ys_i.mean():.1f}")
    csv_data = "\n".join(rows)
    st.download_button("Descargar copas (CSV)", csv_data, file_name="copas_detectadas.csv", mime="text/csv")
else:
    st.info("Subi una imagen RGB y apreta 'Detectar copas' para empezar.")
