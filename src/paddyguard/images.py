"""Image loading, hand-made image features and near-duplicate hashes.

Pixel scale contract: `load_rgb` returns float32 RGB in [0, 1]. The torch models apply the backbone
normalization inside the model code. No other module rescales pixels. Images load one at a time
(streaming), never as one array of the whole dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path, size: int | None = None) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if size is not None:
            img = img.resize((size, size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def thumbnail_signature(path: str | Path, side: int = 16) -> np.ndarray:
    """A side x side gray thumbnail, scaled to mean 0 and standard deviation 1, as a flat vector.

    Near-duplicate shots (a re-save, a resize, a second shot with a small change) give signatures with a
    small mean absolute difference. Downsampling averages out pixel noise.
    """
    with Image.open(path) as img:
        g = np.asarray(img.convert("L").resize((side, side), Image.BILINEAR), dtype=np.float32) / 255.0
    return ((g - g.mean()) / (g.std() + 1e-6)).ravel()


def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx, mn = img.max(axis=2), img.min(axis=2)
    diff = mx - mn + 1e-8
    h = np.where(mx == r, (g - b) / diff % 6, np.where(mx == g, (b - r) / diff + 2, (r - g) / diff + 4)) / 6.0
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0.0)
    return np.stack([h % 1.0, s, mx], axis=2)


def image_features(img: np.ndarray, bins: int = 8) -> np.ndarray:
    """Colour histograms (HSV), lesion-colour shares and texture statistics. 3*bins + 7 values."""
    hsv = _rgb_to_hsv(img)
    hists = [np.histogram(hsv[..., c], bins=bins, range=(0, 1))[0] / hsv[..., c].size for c in range(3)]
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = ((hue > 0.17) & (hue < 0.45) & (sat > 0.25)).mean()
    brown = ((hue < 0.12) & (sat > 0.3) & (val < 0.75)).mean()
    yellow = ((hue >= 0.12) & (hue <= 0.17) & (sat > 0.3)).mean()
    pale = ((sat < 0.2) & (val > 0.6)).mean()
    gray = img.mean(axis=2)
    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)
    extra = [green, brown, yellow, pale, grad.mean(), grad.std(), gray.std()]
    return np.concatenate([*hists, np.asarray(extra, dtype=np.float64)])
