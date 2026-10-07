"""Weather-style image augmentations in NumPy (defined and tested).

Contract: input and output are float32 RGB arrays of shape (H, W, 3) with values in [0, 1]. Each
function returns a new array and never changes its input. The pipeline applies them on the fly to
training images only. Validation and test images are never augmented.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage


def _check(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {img.shape}")
    if img.dtype != np.float32:
        raise TypeError(f"expected float32 in [0, 1], got {img.dtype}")
    if img.min() < 0 or img.max() > 1:
        raise ValueError("pixel values must be in [0, 1]")
    return img


def fog(img: np.ndarray, rng: np.random.Generator, strength: tuple[float, float] = (0.2, 0.5)) -> np.ndarray:
    img = _check(img)
    a = rng.uniform(*strength)
    return np.clip(img * (1 - a) + a * 0.85, 0, 1).astype(np.float32)


def haze(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = _check(img)
    h = img.shape[0]
    gradient = np.linspace(rng.uniform(0.3, 0.5), 0.0, h, dtype=np.float32)[:, None, None]
    return np.clip(img * (1 - gradient) + gradient * np.array([0.8, 0.8, 0.75], np.float32), 0, 1).astype(np.float32)


def rain(img: np.ndarray, rng: np.random.Generator, drops: int | None = None) -> np.ndarray:
    img = _check(img)
    h, w, _ = img.shape
    out = img.copy()
    drops = drops if drops is not None else int(h * w / 150)
    length = max(2, h // 16)
    ys = rng.integers(0, max(1, h - length), drops)
    xs = rng.integers(0, w, drops)
    for k in range(length):
        out[np.clip(ys + k, 0, h - 1), np.clip(xs + k // 3, 0, w - 1)] = 0.8
    return (out * 0.9).astype(np.float32)


def brightness(img: np.ndarray, rng: np.random.Generator, span: float = 0.3) -> np.ndarray:
    img = _check(img)
    return np.clip(img * rng.uniform(1 - span, 1 + span), 0, 1).astype(np.float32)


def shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = _check(img)
    h, w, _ = img.shape
    out = img.copy()
    x0 = int(rng.integers(0, w // 2))
    x1 = int(rng.integers(x0 + 1, w + 1))
    out[:, x0:x1] *= rng.uniform(0.4, 0.7)
    return out.astype(np.float32)


def motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = _check(img)
    size = int(rng.integers(3, 8))
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2, :] = 1.0 / size
    return np.stack([ndimage.convolve(img[:, :, c], kernel, mode="nearest") for c in range(3)], axis=2).astype(
        np.float32)


AUGMENTATIONS: dict[str, Callable] = {
    "fog": fog, "haze": haze, "rain": rain, "brightness": brightness, "shadow": shadow, "motion_blur": motion_blur,
}


class RandomWeather:
    """Apply one random augmentation with probability `p`. Seeded, so a run can be repeated."""

    def __init__(self, p: float = 0.5, seed: int = 0, names: tuple[str, ...] | None = None):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.names = names or tuple(AUGMENTATIONS)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p:
            return img
        name = self.names[int(self.rng.integers(0, len(self.names)))]
        return AUGMENTATIONS[name](img, self.rng)
