"""Image metadata: one row for each labelled image.

Required columns: `image_id`, `label`. Optional: `path`, `field_id`, `latitude`, `longitude`, `date`,
`variety`, `age`. The weather branch needs `latitude`, `longitude` and `date` for every image. If
they are missing, paddyguard runs the image-only model and says so.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("image_id", "label")
WEATHER_KEYS = ("latitude", "longitude", "date")


class MetadataError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def classes_of(meta: pd.DataFrame) -> list[str]:
    """The class list is the sorted set of labels. It never depends on folder listing order."""
    return sorted(meta["label"].astype(str).unique())


def validate(meta: pd.DataFrame, data_dir: str | Path | None = None, check_files: bool = True) -> pd.DataFrame:
    problems = [f"missing column {c!r}" for c in REQUIRED if c not in meta.columns]
    if problems:
        raise MetadataError(problems)
    out = meta.copy()
    out["image_id"] = out["image_id"].astype(str)
    out["label"] = out["label"].astype(str)
    if out["image_id"].duplicated().any():
        problems.append(f"{int(out['image_id'].duplicated().sum())} duplicated image_id values")
    if "path" not in out.columns:
        out["path"] = [f"images/{l}/{i}" for l, i in zip(out["label"], out["image_id"])]
    if "field_id" not in out.columns:
        out["field_id"] = out["image_id"]  # no grouping information: each image is its own group
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if out["date"].isna().any():
            problems.append(f"{int(out['date'].isna().sum())} rows have no valid date")
    for col in ("latitude", "longitude"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "latitude" in out.columns and ((out["latitude"].abs() > 90) | out["latitude"].isna()).any():
        problems.append("latitude must be between -90 and 90 in every row")
    if "longitude" in out.columns and ((out["longitude"].abs() > 180) | out["longitude"].isna()).any():
        problems.append("longitude must be between -180 and 180 in every row")
    if check_files and data_dir is not None:
        missing = [p for p in out["path"] if not (Path(data_dir) / p).exists()]
        if missing:
            problems.append(f"{len(missing)} image files do not exist, for example {missing[0]}")
    if problems:
        raise MetadataError(problems)
    return out.reset_index(drop=True)


def weather_available(meta: pd.DataFrame) -> bool:
    return all(c in meta.columns for c in WEATHER_KEYS) and meta[list(WEATHER_KEYS)].notna().all().all()


def load_metadata(data_dir: str | Path, check_files: bool = True) -> pd.DataFrame:
    path = Path(data_dir) / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"No metadata.csv in {data_dir}. Read data/README.md, or run `paddyguard synth`.")
    return validate(pd.read_csv(path), data_dir, check_files)


def encode(meta: pd.DataFrame, classes: list[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    unknown = set(meta["label"]) - set(index)
    if unknown:
        raise MetadataError([f"labels not in the class list: {sorted(unknown)}"])
    return meta["label"].map(index).to_numpy()
