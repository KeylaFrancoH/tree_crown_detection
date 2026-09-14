# tree_crown_detection

Modelo de detección de copas de árboles combinando imagen óptica (RGB o
multiespectral) y un modelo de altura (CHM/nDSM), usando una **U-Net de
doble encoder** que fusiona ambas ramas y separa copas individuales
(segmentación de instancias) con watershed.

## Idea general

El esquema de partida (dos CNN concatenadas y luego una cabeza de
detección) se adapta así:

```
RGB / multiespectral ---> Encoder A ---\
                                         >-- Fusion por nivel --> Decoder --> [mask, dist]
CHM / nDSM (altura) -----> Encoder B ---/
```

En vez de una cabeza tipo YOLO (bounding boxes), la cabeza es un decoder
U-Net con dos salidas por píxel:

- **mask**: probabilidad de "copa" (semántica, copa vs. fondo).
- **dist**: transformada de distancia normalizada dentro de cada copa.

¿Por qué dos cabezas y no una sola máscara? Con una sola máscara binaria,
copas que se tocan (muy común en dosel cerrado) quedan fusionadas en un solo
blob. El mapa de distancia predicho tiene un máximo en el centro de cada
copa; buscando esos máximos como semillas y corriendo **watershed**
restringido a la máscara, se separan instancias individuales aunque estén
en contacto. El CHM ayuda especialmente acá: los picos de altura suelen
coincidir con el centro de cada árbol, dando una señal fuerte para separar
copas vecinas que en la imagen óptica se ven como una sola mancha verde.

## Estructura del proyecto

```
configs/default.yaml       Hiperparámetros (modelo, tiling, entrenamiento, inferencia)
src/
  data/
    tiling.py               Recorta ortomosaicos grandes en tiles alineados (RGB+CHM+labels)
    rasterize.py             Máscara de instancias -> (mask, dist); rasteriza polígonos si hace falta
    dataset.py                Dataset PyTorch que lee los tiles .npz
    transforms.py             Augmentations (flips/rot90, sincronizados en todas las bandas)
  models/
    unet_dual.py               U-Net de doble encoder (RGB + CHM)
  losses/losses.py               BCE+Dice (máscara) + MSE (distancia)
  training/train.py               Loop de entrenamiento
  inference/
    postprocess.py                Watershed máscara+distancia -> instancias -> polígonos
    predict.py                    Inferencia sobre ortomosaico completo (sliding window)
scripts/
  prepare_tiles.py                 CLI de tiling
  train.py                          CLI de entrenamiento
  predict.py                        CLI de inferencia -> GeoJSON
tests/                                 Tests con datos sintéticos (no requieren datos reales)
notebooks/train_colab.ipynb            Notebook para entrenar en Google Colab (con GPU)
```

## 0. Entrenar en Google Colab (alternativa sin instalar nada local)

Si no tenés GPU local, abrí `notebooks/train_colab.ipynb` en Colab (subiéndolo
o vía `File > Open notebook > GitHub` apuntando a este repo). El notebook
clona el repo, instala dependencias, monta tu Google Drive para leer los
rasters (RGB, CHM, labels) y guardar el modelo entrenado, tilea, entrena y
al final copia `checkpoints/best.pth` a tu Drive (o lo descarga directo).
Los pasos manuales de abajo son los mismos que ejecuta el notebook.

## 1. Preparar los datos

Se necesitan, con **la misma grilla** (mismo tamaño de píxel, extensión y
CRS — reproyectar/resamplear antes si no coinciden):

- `data/raw/rgb/<sitio>.tif`: ortomosaico óptico.
- `data/raw/chm/<sitio>.tif`: modelo de altura (CHM o nDSM), 1 banda.
- `data/raw/labels/<sitio>.tif`: raster de **instancias** de copa — 1 banda,
  0 = fondo, cada copa individual con un entero distinto (1, 2, 3, ...).

Si en cambio las copas segmentadas están como polígonos (shapefile/GeoJSON),
usar `src.data.rasterize.rasterize_polygons(...)` para generar primero ese
raster de instancias con la misma grilla que el RGB/CHM.

Tilear en parches de entrenamiento:

```bash
python scripts/prepare_tiles.py \
  --rgb data/raw/rgb/sitio1.tif \
  --chm data/raw/chm/sitio1.tif \
  --labels data/raw/labels/sitio1.tif \
  --out data/processed/train \
  --tile-size 256 --overlap 32 --min-valid-fraction 0.1
```

Repetir apuntando a `data/processed/val` con un sitio (o subárea) distinto
para no filtrar información entre train y val.

## 2. Entrenar

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
```

Ajustar en `configs/default.yaml`:
- `data.chm_max`: altura máxima esperada en metros (normaliza el CHM a [0,1]).
- `model.rgb_channels`: 3 para RGB, más si es multiespectral.
- `loss.*`: pesos relativos de máscara vs. distancia.

Los checkpoints (modelo entrenado, listo para usar) se guardan como archivos
`.pth` en `checkpoints/last.pth` y `checkpoints/best.pth` (mejor IoU de
máscara en validación). `best.pth` es el archivo que necesitás para
inferencia o para compartir/desplegar el modelo — contiene únicamente los
pesos (`state_dict`), se carga con `build_model(config)` +
`model.load_state_dict(torch.load("checkpoints/best.pth"))`.

## 3. Inferencia -> copas detectadas

```bash
python scripts/predict.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth \
  --rgb data/raw/rgb/sitio_nuevo.tif \
  --chm data/raw/chm/sitio_nuevo.tif \
  --out results/sitio_nuevo_copas.geojson
```

Corre con ventana deslizante sobre el ortomosaico completo y devuelve un
GeoJSON con un polígono por copa individual detectada (instancias), listo
para abrir en QGIS o similar.

## Tests

```bash
python -m pytest tests/ -v
```

Incluyen: forward/backward del modelo, separación de dos copas contiguas
vía watershed, y un pipeline end-to-end (tiling -> dataset -> modelo ->
polygonización) sobre rasters sintéticos generados en el propio test — no
requieren datos reales para correr.

## Notas / próximos pasos

- Si los datos reales tienen resolución de altura menor al RGB (típico:
  dron óptico a pocos cm/px vs. CHM LiDAR a 0.5-1 m/px), resamplear el CHM
  al grid del RGB antes de tilear (`rasterio.warp.reproject`).
- El overlap en inferencia (`inference.overlap`) evita artefactos de
  watershed en los bordes de cada tile, pero copas que caen justo en el
  borde del recorte "núcleo" de un tile pueden no fusionarse con el tile
  vecino; aumentar el overlap si esto es un problema visible.
- Con pocas imágenes segmentadas disponibles, conviene aplicar más
  augmentation (o partir de pesos preentrenados en el encoder RGB, ej.
  ImageNet) y vigilar overfitting con `val_iou`.
