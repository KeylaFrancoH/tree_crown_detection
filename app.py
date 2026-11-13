"""App de prueba interactiva para el modelo de copas (RGB+MS / CHM) entrenado con
los datos de CEDIA.

Dos modos:
  - "Explorar tiles de prueba": elegir un tile real (train/val/test) ya
    tileado y comparar la predicción contra la etiqueta real.
  - "Subir imagen nueva": subir un ortomosaico (GeoTIFF) propio y, opcional-
    mente, su mapa de altura, y ver la segmentación de copas sobre esa
    imagen completa.

Nota: la versión anterior de este archivo (con editor interactivo tipo mapa,
filtro NDVI y conversión de parámetros a metros, para los modelos
rgb_only/rgb_pretrained/rgb_5band) quedó guardada en `app_editor_ndvi.py`,
no se perdió.

Correr con:
    streamlit run app.py --server.port 8502
"""
from __future__ import annotations

import glob

import numpy as np
import streamlit as st
import torch

from src.inference.postprocess import instances_from_mask_dist
from src.inference.upload import segment_uploaded
from src.models.unet_dual import build_model
from src.training.metrics import binary_iou
from src.training.train import load_config

st.set_page_config(page_title="Copas CEDIA - prueba de modelo", layout="wide")


@st.cache_resource
def get_model(config_path: str, checkpoint_path: str):
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model, config, device


def rgb_uint8(rgb_bands: np.ndarray) -> np.ndarray:
    img = rgb_bands.transpose(1, 2, 0)
    p2, p98 = np.percentile(img, [2, 98])
    img = np.clip((img - p2) / max(p98 - p2, 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


def instance_rgb(instance_map: np.ndarray, seed: int = 0) -> np.ndarray:
    n = int(instance_map.max())
    rng = np.random.default_rng(seed)
    palette = (rng.uniform(0.25, 1.0, size=(max(n, 1) + 1, 3)) * 255).astype(np.uint8)
    palette[0] = [0, 0, 0]
    return palette[instance_map]


st.title("Segmentación de copas — CEDIA (RGB+MS / CHM)")
st.caption(
    "U-Net de doble encoder + watershed sobre mapa de distancia. "
    "Modelo entrenado en `checkpoints_cedia/best.pth` (test IoU semántico ≈ 0.53)."
)

with st.sidebar:
    st.header("Modelo")
    config_path = st.text_input("Config", "configs/cedia.yaml")
    checkpoint_path = st.text_input("Checkpoint", "checkpoints_cedia/best.pth")

    st.header("Separación de instancias (watershed)")
    mask_threshold = st.slider("Umbral de máscara", 0.0, 1.0, 0.5, 0.05)
    min_seed_distance = st.slider(
        "Distancia mínima entre semillas (px)", 2, 64, 24, 2,
        help="Subir si separa de más (1 árbol detectado como varios). Bajar si junta de más (varios árboles en uno solo).",
    )
    min_seed_prominence = st.slider(
        "Prominencia mínima de un pico para contar como árbol aparte", 0.0, 1.0, 0.15, 0.05,
        help="Qué tan marcado tiene que ser un máximo de altura/distancia para separar un árbol. Bajar si junta de más; subir si separa de más por ruido.",
    )
    smooth_sigma = st.slider(
        "Suavizado del mapa de distancia (sigma)", 0.0, 5.0, 2.0, 0.5,
        help="Más suavizado = menos picos espurios (menos sobre-separación), pero puede fusionar árboles muy cercanos entre sí.",
    )
    min_instance_area = st.slider("Área mínima de instancia (px²)", 0, 500, 100, 10)
    min_saddle_drop = st.slider(
        "Fusionar si el valle entre dos picos es muy chico", 0.0, 0.3, 0.03, 0.01,
        help=(
            "Detecta 'árboles dentro de un árbol': si dos picos vecinos casi no tienen "
            "un valle real entre ellos, se fusionan en una sola copa. Calibrado contra "
            "las etiquetas reales en tiles chicos (0.03 ≈ sin cambios); en fotos grandes "
            "con dosel muy denso puede hacer falta subirlo bastante más — probá y mirá "
            "el resultado."
        ),
    )
    use_height_seeds = st.checkbox(
        "Usar el CHM real (no solo la predicción) para ubicar árboles",
        value=False,
        help="La altura real es una señal más limpia que la textura de la foto, pero el CHM acá es una aproximación (no un nDSM certificado).",
    )
    height_weight = st.slider("Peso del CHM en esa mezcla", 0.0, 1.0, 0.5, 0.1, disabled=not use_height_seeds)

model, config, device = get_model(config_path, checkpoint_path)

tab_explorar, tab_subir = st.tabs(["Explorar tiles de prueba", "Subir imagen nueva"])

with tab_explorar:
    split = st.selectbox("Conjunto", ["test", "val", "train"], index=0)
    files = sorted(glob.glob(f"cedia_dataset/{split}/*.npz"))
    if not files:
        st.error(f"No hay tiles en cedia_dataset/{split}/. Corré scripts/prepare_cedia_dataset.py primero.")
        st.stop()
    labels = [f.split("/")[-1].replace(".npz", "") for f in files]
    idx = st.selectbox("Tile", range(len(files)), format_func=lambda i: labels[i])
    tile_path = files[idx]

    data = np.load(tile_path)
    rgb_raw = data["rgb"]
    chm_raw = data["chm"]
    mask_gt = data["mask"][0]
    instances_gt = data["instances"]

    rgb_t = torch.from_numpy(rgb_raw / config["data"]["rgb_scale"]).unsqueeze(0).to(device)
    chm_t = torch.from_numpy(np.clip(chm_raw, 0, config["data"]["chm_max"]) / config["data"]["chm_max"]).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(rgb_t, chm_t)
    mask_logits = out["mask_logits"]
    mask_prob = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
    dist_map = out["dist_map"][0, 0].cpu().numpy()

    instances_pred = instances_from_mask_dist(
        mask_prob, dist_map, mask_threshold, min_seed_distance, min_instance_area,
        smooth_sigma=smooth_sigma, min_seed_prominence=min_seed_prominence,
        height_map=chm_raw[0] if use_height_seeds else None, height_weight=height_weight,
        min_saddle_drop=min_saddle_drop,
    )

    mask_gt_t = torch.from_numpy(mask_gt[None, None, ...]).to(device)
    tile_iou = binary_iou(mask_logits, mask_gt_t)

    n_pred = int(instances_pred.max())
    n_gt = int(len(np.unique(instances_gt)) - 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("IoU semántico (este tile)", f"{tile_iou:.3f}")
    c2.metric("Copas detectadas", n_pred)
    c3.metric("Copas reales (etiqueta)", n_gt)

    col_a, col_b, col_c, col_d = st.columns(4)
    rgb_vis = rgb_uint8(rgb_raw[:3])
    isolated = rgb_vis.copy()
    isolated[instances_pred == 0] = 0

    col_a.image(rgb_vis, caption="(a) Imagen original (RGB)", width="stretch")
    col_b.image(instance_rgb(instances_pred), caption=f"(b) Instancias predichas ({n_pred})", width="stretch")
    col_c.image(isolated, caption="(c) Copas aisladas (predicción)", width="stretch")
    col_d.image(instance_rgb(instances_gt), caption=f"(d) Etiqueta real ({n_gt})", width="stretch")

    with st.expander("Mapa de altura (CHM aproximado) de este tile"):
        chm_vis = np.clip(chm_raw[0], 0, config["data"]["chm_max"])
        st.image((chm_vis / config["data"]["chm_max"] * 255).astype(np.uint8), width="stretch", clamp=True)

    with st.expander("Probabilidad de máscara cruda (antes del umbral/watershed)"):
        st.image((mask_prob * 255).astype(np.uint8), width="stretch", clamp=True)

with tab_subir:
    st.markdown(
        "El modelo espera **dos entradas**: una imagen (idealmente GeoTIFF con las "
        f"{config['model']['rgb_channels']} bandas RGB+MS con las que se entrenó; si subís "
        "menos bandas, se rellenan con ceros y la calidad baja) y un mapa de altura "
        "(CHM/DSM). Sin mapa de altura igual corre, pero con la rama de altura en cero "
        "— los resultados van a ser menos confiables que con los datos completos."
    )

    rgb_file = st.file_uploader("Ortomosaico / imagen (GeoTIFF recomendado)", type=["tif", "tiff"])
    chm_file = st.file_uploader("Mapa de altura (GeoTIFF, opcional)", type=["tif", "tiff"])

    dsm_is_raw = True
    if chm_file is not None:
        dsm_is_raw = st.checkbox(
            "El mapa de altura es un DSM crudo (elevación absoluta, incluye terreno) "
            "→ aproximar altura de copa restando el terreno estimado",
            value=True,
            help="Desmarcar si el archivo ya es un CHM/nDSM normalizado (altura sobre el suelo).",
        )

    if rgb_file is not None and st.button("Procesar imagen"):
        progress = st.progress(0.0, text="Procesando tiles...")

        def _update(frac: float) -> None:
            progress.progress(min(frac, 1.0), text=f"Procesando tiles... {frac*100:.0f}%")

        with st.spinner("Corriendo el modelo..."):
            result = segment_uploaded(
                rgb_bytes=rgb_file.getvalue(),
                chm_bytes=chm_file.getvalue() if chm_file is not None else None,
                model=model,
                config=config,
                device=device,
                dsm_is_raw=dsm_is_raw,
                mask_threshold=mask_threshold,
                min_seed_distance=min_seed_distance,
                min_instance_area=min_instance_area,
                smooth_sigma=smooth_sigma,
                min_seed_prominence=min_seed_prominence,
                min_saddle_drop=min_saddle_drop,
                use_height_seeds=use_height_seeds,
                height_weight=height_weight,
                progress_callback=_update,
            )
        progress.empty()

        if chm_file is None:
            st.warning("Sin mapa de altura: se usó altura = 0 en toda la imagen. Resultados orientativos solamente.")

        st.metric("Copas detectadas", result["n_instances"])

        col1, col2 = st.columns(2)
        col1.image(result["rgb_true_color"], caption="Imagen original", width="stretch")
        col2.image(instance_rgb(result["instance_map"]), caption="Instancias detectadas", width="stretch")

        if result["polygons"]:
            import geopandas as gpd

            gdf = gpd.GeoDataFrame(result["polygons"], geometry="geometry", crs=result["crs"])
            geojson_bytes = gdf.to_json().encode("utf-8")
            st.download_button(
                "Descargar copas detectadas (GeoJSON)",
                data=geojson_bytes,
                file_name="copas_detectadas.geojson",
                mime="application/geo+json",
            )
        else:
            st.info("La imagen subida no tiene CRS/georreferenciación, así que no se puede exportar un GeoJSON.")
