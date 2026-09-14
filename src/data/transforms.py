"""Augmentations geométricas simples, sincronizadas entre RGB, CHM y targets.

Se evitan transformaciones que alteren valores radiométricos/de altura de
forma no física (ColorJitter en CHM no tiene sentido). Solo flips/rotaciones
de 90°, que preservan los valores de altura exactamente.
"""
from __future__ import annotations

import random

import numpy as np


class RandomFlipRotate:
    def __init__(self, p_flip: float = 0.5, p_rot90: float = 0.5, seed: int | None = None):
        self.p_flip = p_flip
        self.p_rot90 = p_rot90
        self._rng = random.Random(seed)

    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(sample)

        if self._rng.random() < self.p_flip:
            out = {k: np.flip(v, axis=-1).copy() for k, v in out.items()}
        if self._rng.random() < self.p_flip:
            out = {k: np.flip(v, axis=-2).copy() for k, v in out.items()}
        if self._rng.random() < self.p_rot90:
            k = self._rng.choice([1, 2, 3])
            out = {key: np.rot90(v, k=k, axes=(-2, -1)).copy() for key, v in out.items()}

        return out
