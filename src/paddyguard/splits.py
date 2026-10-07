"""Grouped splits. Images of one field and near-duplicate photos stay on one side of each split.

The validation and test parts are made before any augmentation. Augmentation runs on the fly on
training images only, so no augmented copy of a training image can reach validation or test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .images import thumbnail_signature


class LeakageError(AssertionError):
    """A group has images on both sides of a split."""


def duplicate_groups(meta: pd.DataFrame, data_dir: str | Path, max_distance: float = 0.06) -> pd.Series:
    """Union-find over thumbnail signatures. Two images of the same label with a mean absolute
    signature difference <= max_distance share one group: they are near-duplicate shots."""
    sig = np.vstack([thumbnail_signature(Path(data_dir) / p) for p in meta["path"]])
    labels = meta["label"].to_numpy()
    parent = list(range(len(sig)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        block = sig[idx]
        for a in range(len(idx) - 1):
            # Vectorized distance from one image to all later images of the same label.
            dist = np.abs(block[a + 1:] - block[a]).mean(axis=1)
            for b in np.flatnonzero(dist <= max_distance) + a + 1:
                parent[find(int(idx[a]))] = find(int(idx[b]))
    return pd.Series([find(i) for i in range(len(sig))], index=meta.index)


def split_groups(meta: pd.DataFrame, dup: pd.Series | None = None) -> pd.Series:
    """Merge field groups and duplicate groups: two images share a group if they share a field or a duplicate."""
    fields = meta["field_id"].astype(str).to_numpy()
    if dup is None:
        return pd.Series(fields, index=meta.index)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for f, d in zip(fields, dup.to_numpy()):
        a, b = find(f"f:{f}"), find(f"d:{d}")
        if a != b:
            parent[a] = b
    return pd.Series([find(f"f:{f}") for f in fields], index=meta.index)


def assign_splits(y: np.ndarray, groups: pd.Series, seed: int = 42, test_folds: int = 5,
                  val_folds: int = 5) -> np.ndarray:
    """Return an array of 'train', 'val' or 'test'. About 20% test, then about 16% val."""
    split = np.full(len(y), "train", dtype=object)
    outer = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    rest, test = next(outer.split(np.zeros(len(y)), y, groups.to_numpy()))
    split[test] = "test"
    inner = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed + 1)
    _, val = next(inner.split(np.zeros(len(rest)), y[rest], groups.to_numpy()[rest]))
    split[rest[val]] = "val"
    assert_no_overlap(groups, split)
    return split


def assert_no_overlap(groups: pd.Series, split: np.ndarray) -> None:
    frame = pd.DataFrame({"g": groups.to_numpy(), "s": split})
    shared = frame.groupby("g")["s"].nunique()
    if (shared > 1).any():
        raise LeakageError(f"{int((shared > 1).sum())} groups have images in more than one split")
