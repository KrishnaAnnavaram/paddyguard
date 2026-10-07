"""Synthetic paddy leaf images with field metadata and a real weather link.

Each field has a location. Each photo has a date. The class of a photo depends on the weather of the
14 days before its date (from `OfflineWeather`), so weather features carry true information here.
Two classes (blast and brown spot) look alike, so the weather can help to separate them. Each field
has its own light and leaf colour, and some photos are near-duplicate shots of the same leaf. The
images are drawings, not photos of real plants.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .weather import OfflineWeather, window_features

CLASSES = ("bacterial_leaf_blight", "blast", "brown_spot", "hispa", "normal", "tungro")


def _leaf(size: int, rng, field: dict) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size, 3), np.float32)
    img[:] = np.array([0.45, 0.35, 0.22]) * field["light"]  # soil background
    centre = size / 2 + rng.normal(0, size * 0.05)
    width = size * rng.uniform(0.18, 0.26)
    leaf = np.abs(yy - centre - (xx - size / 2) * rng.uniform(-0.3, 0.3)) < width * (1 - np.abs(xx - size / 2) / size)
    img[leaf] = np.array([0.25, 0.6, 0.2]) * field["green"] * field["light"]
    return img, leaf


def _spots(img, leaf, rng, n, radius, colour, centre_colour=None):
    size = img.shape[0]
    ys, xs = np.nonzero(leaf)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n):
        k = int(rng.integers(0, len(ys)))
        r = radius * rng.uniform(0.7, 1.3)
        d = ((yy - ys[k]) / (r * 0.6)) ** 2 + ((xx - xs[k]) / r) ** 2
        img[(d <= 1) & leaf] = colour
        if centre_colour is not None:
            img[(d <= 0.35) & leaf] = centre_colour


def draw(label: str, size: int, rng, field: dict) -> np.ndarray:
    img, leaf = _leaf(size, rng, field)
    s = size / 64
    if label == "blast":
        _spots(img, leaf, rng, int(rng.integers(2, 7)), 3.6 * s, (0.43, 0.27, 0.12), (0.55, 0.5, 0.42))
    elif label == "brown_spot":
        _spots(img, leaf, rng, int(rng.integers(3, 8)), 3.2 * s, (0.42, 0.25, 0.1), (0.5, 0.45, 0.35))
    elif label == "bacterial_leaf_blight":
        edge = leaf & ~np.roll(leaf, int(3 * s), axis=0)
        tip = np.mgrid[0:size, 0:size][1] > size * rng.uniform(0.45, 0.65)
        img[edge & tip] = (0.85, 0.8, 0.45)
    elif label == "hispa":
        for _ in range(int(rng.integers(3, 7))):
            y = int(rng.integers(0, size))
            img[max(0, y - 1):y + 1, :][leaf[max(0, y - 1):y + 1, :]] = (0.85, 0.88, 0.8)
    elif label == "tungro":
        img[leaf] = img[leaf] * 0.5 + np.array([0.75, 0.65, 0.15]) * 0.5
    img += rng.normal(0, 0.03, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def _class_weights(w: dict) -> np.ndarray:
    humid = (w["w_humid_days"] - 3) / 3
    rain = (w["w_rainy_days"] - 4) / 3
    warm = (w["w_tmean"] - 27) / 3
    logits = np.array([
        0.8 * rain + 0.5 * warm,    # bacterial_leaf_blight: warm and rainy
        1.4 * humid - 0.6 * warm,   # blast: humid and cool
        -1.0 * humid + 0.4 * warm,  # brown_spot: dry
        -0.6 * rain + 0.6 * warm,   # hispa: dry and warm
        0.3,                        # normal
        0.0,                        # tungro
    ])
    p = np.exp(logits - logits.max())
    return p / p.sum()


def generate(out_dir: str | Path, n_fields: int = 30, photos_per_field: int = 25, size: int = 64, seed: int = 0,
             duplicate_rate: float = 0.1, lag_days: int = 14) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    weather = OfflineWeather()
    rows = []
    start = date(2022, 1, 1)
    for f in range(n_fields):
        lat, lon = float(rng.uniform(10, 25)), float(rng.uniform(75, 90))
        field = {"light": rng.uniform(0.8, 1.15), "green": rng.uniform(0.85, 1.15)}
        days = sorted(int(d) for d in rng.integers(lag_days + 1, 700, photos_per_field))
        daily = weather.daily(lat, lon, start, start + timedelta(days=720))
        previous = None
        for k, d in enumerate(days):
            when = pd.Timestamp(start + timedelta(days=d))
            if previous is not None and rng.random() < duplicate_rate:
                label, img_seed = previous  # a second shot of the same leaf
            else:
                w = window_features(daily, when, lag_days)
                label = CLASSES[int(rng.choice(len(CLASSES), p=_class_weights(w)))]
                img_seed = int(rng.integers(0, 2**31))
            img = draw(label, size, np.random.default_rng(img_seed), field)
            if previous is not None and (label, img_seed) == previous:
                img = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1).astype(np.float32)
            previous = (label, img_seed)
            image_id = f"f{f:03d}_{k:03d}.png"
            path = Path("images") / label / image_id
            (out_dir / path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((img * 255).astype(np.uint8)).save(out_dir / path)
            rows.append({"image_id": image_id, "label": label, "path": path.as_posix(), "field_id": f"F{f:03d}",
                         "latitude": round(lat, 4), "longitude": round(lon, 4), "date": when.date().isoformat()})
    pd.DataFrame(rows).to_csv(out_dir / "metadata.csv", index=False)
    return out_dir / "metadata.csv"
