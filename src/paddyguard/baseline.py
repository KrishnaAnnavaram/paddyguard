"""CPU ablation with no deep learning framework: image-only vs. weather-only vs. image + weather.

All three variants use the same splits, the same seeds and the same model family (a balanced
logistic regression with its C selected on the validation split). The test split is used once for
each variant and seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .augment import AUGMENTATIONS
from .evaluate import group_bootstrap, metrics, paired_bootstrap, seed_summary
from .images import image_features, load_rgb
from .metadata import classes_of, encode
from .splits import assign_splits

C_GRID = (0.1, 1.0, 10.0)


def extract_image_features(meta: pd.DataFrame, data_dir: str | Path, size: int = 64, augment: str | None = None,
                           seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for p in meta["path"]:
        img = load_rgb(Path(data_dir) / p, size)
        if augment is not None:
            img = AUGMENTATIONS[augment](img, rng)
        rows.append(image_features(img))
    return np.vstack(rows)


def _pipe(c: float) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler()),
                     ("model", LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))])


def _proba(pipe: Pipeline, X: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(X), n_classes))
    out[:, pipe.classes_] = pipe.predict_proba(X)
    return out


def fit_select(X: np.ndarray, y: np.ndarray, split: np.ndarray) -> tuple[Pipeline, float]:
    tr, va = split == "train", split == "val"
    best, best_c, best_score = None, None, -1.0
    for c in C_GRID:
        pipe = _pipe(c).fit(X[tr], y[tr])
        score = f1_score(y[va], pipe.predict(X[va]), average="macro", zero_division=0)
        if score > best_score:
            best, best_c, best_score = pipe, c, score
    # Refit on train + val with the selected C. The test rows stay unseen.
    final = _pipe(best_c).fit(X[tr | va], y[tr | va])
    return final, best_c


@dataclass
class AblationResult:
    classes: list[str]
    per_seed: dict[str, list[dict]] = field(default_factory=dict)
    summary: dict[str, dict] = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    robustness: dict = field(default_factory=dict)


def run_ablation(meta: pd.DataFrame, data_dir: str | Path, weather: pd.DataFrame | None, groups: pd.Series,
                 seeds=(0, 1, 2), n_boot: int = 200, image_feats: np.ndarray | None = None,
                 robustness: bool = True) -> AblationResult:
    classes = classes_of(meta)
    y = encode(meta, classes)
    Xi = image_feats if image_feats is not None else extract_image_features(meta, data_dir)
    variants = {"image": Xi}
    if weather is not None:
        Xw = weather.to_numpy(dtype=float)
        variants["weather"] = Xw
        variants["image+weather"] = np.hstack([Xi, Xw])
    result = AblationResult(classes=classes, per_seed={k: [] for k in variants})
    first = {}
    for seed in seeds:
        split = assign_splits(y, groups, seed=seed)
        te = split == "test"
        for name, X in variants.items():
            pipe, c = fit_select(X, y, split)
            proba = _proba(pipe, X[te], len(classes))
            m = metrics(y[te], proba, classes)
            m["macro_f1_ci"] = group_bootstrap(y[te], proba, groups[te], n_boot, seed) if n_boot else None
            m["C"] = c
            result.per_seed[name].append(m)
            if seed == seeds[0]:
                first[name] = (proba, pipe)
        if seed == seeds[0] and weather is not None:
            result.comparison = paired_bootstrap(y[te], first["image+weather"][0], first["image"][0], groups[te],
                                                 n_boot=max(n_boot, 100), seed=seed)
        if seed == seeds[0] and robustness:
            pipe = first["image"][1]
            test_meta = meta[te]
            result.robustness["clean"] = metrics(y[te], first["image"][0], classes)["macro_f1"]
            for aug in AUGMENTATIONS:
                Xa = extract_image_features(test_meta, data_dir, augment=aug, seed=seed)
                result.robustness[aug] = metrics(y[te], _proba(pipe, Xa, len(classes)), classes)["macro_f1"]
    result.summary = {k: seed_summary([m["macro_f1"] for m in v]) for k, v in result.per_seed.items()}
    return result
