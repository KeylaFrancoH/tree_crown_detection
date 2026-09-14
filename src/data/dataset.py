"""Dataset PyTorch que lee los tiles .npz generados por `src.data.tiling`."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class TreeCrownTileDataset(Dataset):
    """Carga tiles (rgb, chm, mask, dist) desde una carpeta de .npz.

    Espera archivos generados por `tile_dataset(..., labels_path=...)`, es
    decir que cada .npz tiene las claves "rgb", "chm", "mask", "dist".
    Normaliza RGB a [0, 1] (asumiendo uint8/reflectancia 0-255 de entrada)
    y CHM por un `chm_max` fijo (metros) para acotarlo a un rango estable.
    """

    def __init__(
        self,
        tiles_dir: str,
        rgb_scale: float = 255.0,
        chm_max: float = 40.0,
        transform: Callable[[dict], dict] | None = None,
    ):
        self.tiles_dir = Path(tiles_dir)
        self.files = sorted(self.tiles_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No se encontraron tiles .npz en {tiles_dir}")
        self.rgb_scale = rgb_scale
        self.chm_max = chm_max
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        sample = {
            "rgb": data["rgb"] / self.rgb_scale,
            "chm": np.clip(data["chm"], 0, self.chm_max) / self.chm_max,
            "mask": data["mask"],
            "dist": data["dist"],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return {
            "rgb": torch.from_numpy(sample["rgb"].astype(np.float32)),
            "chm": torch.from_numpy(sample["chm"].astype(np.float32)),
            "mask": torch.from_numpy(sample["mask"].astype(np.float32)),
            "dist": torch.from_numpy(sample["dist"].astype(np.float32)),
            "path": str(self.files[idx]),
        }
