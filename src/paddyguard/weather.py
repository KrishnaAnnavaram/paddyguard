"""Historical weather for each image: provider interface, cache and lagged window features.

Each image gets features from the daily weather at its own location in the `lag_days` days BEFORE
its date (the day of the photo is not in the window). A feature column with one constant value for
all images carries no information. `assert_informative` stops the run in that case.

Providers:
- `OfflineWeather`: a deterministic synthetic climate from latitude, longitude and day. No network.
- `OpenMeteoArchive`: the Open-Meteo historical archive API (no key). Uses only the standard library.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

DAILY_COLUMNS = ("tmean", "tmax", "tmin", "rh", "precip", "wind")
FEATURES = ("w_tmean", "w_tmax", "w_tmin", "w_rh", "w_precip_total", "w_rainy_days", "w_humid_days", "w_wind")
RAINY_MM = 1.0
HUMID_RH = 90.0


class WeatherProvider(Protocol):
    name: str

    def daily(self, latitude: float, longitude: float, start: date, end: date) -> pd.DataFrame:
        """Return one row for each day with `date` and the columns in DAILY_COLUMNS."""


class OfflineWeather:
    name = "offline"

    def daily(self, latitude: float, longitude: float, start: date, end: date) -> pd.DataFrame:
        days = pd.date_range(start, end, freq="D")
        rows = []
        for d in days:
            seed = (int(round(abs(latitude) * 100)) * 1_000_003 + int(round(abs(longitude) * 100)) * 7_919
                    + d.toordinal()) % (2**32)
            rng = np.random.default_rng(seed)
            season = np.sin(2 * np.pi * (d.dayofyear - 100) / 365.25)
            monsoon = max(0.0, np.sin(2 * np.pi * (d.dayofyear - 150) / 365.25))
            tmean = 27 - 0.25 * (abs(latitude) - 15) + 4 * season + rng.normal(0, 1.2)
            rh = float(np.clip(68 + 22 * monsoon + rng.normal(0, 6), 30, 100))
            precip = float(rng.gamma(0.6, 12) if rng.random() < 0.15 + 0.6 * monsoon else 0.0)
            rows.append({"date": d.normalize(), "tmean": tmean, "tmax": tmean + 5 + rng.normal(0, 1),
                         "tmin": tmean - 5 + rng.normal(0, 1), "rh": rh, "precip": precip,
                         "wind": float(abs(rng.normal(10, 4)))})
        return pd.DataFrame(rows)


class OpenMeteoArchive:
    name = "open-meteo"
    URL = "https://archive-api.open-meteo.com/v1/archive"
    VARIABLES = {"temperature_2m_mean": "tmean", "temperature_2m_max": "tmax", "temperature_2m_min": "tmin",
                 "relative_humidity_2m_mean": "rh", "precipitation_sum": "precip", "wind_speed_10m_max": "wind"}

    def __init__(self, timeout: float = 30.0, opener=urllib.request.urlopen):
        self.timeout = timeout
        self.opener = opener

    def daily(self, latitude: float, longitude: float, start: date, end: date) -> pd.DataFrame:
        query = urllib.parse.urlencode({
            "latitude": f"{latitude:.4f}", "longitude": f"{longitude:.4f}", "start_date": start.isoformat(),
            "end_date": end.isoformat(), "daily": ",".join(self.VARIABLES), "timezone": "UTC",
        })
        with self.opener(f"{self.URL}?{query}", timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return parse_open_meteo(payload, self.VARIABLES)


def parse_open_meteo(payload: dict, variables: dict[str, str]) -> pd.DataFrame:
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        raise ValueError(f"unexpected Open-Meteo answer: {str(payload)[:200]}")
    frame = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for source, target in variables.items():
        frame[target] = pd.to_numeric(pd.Series(daily.get(source, [np.nan] * len(frame))), errors="coerce")
    return frame


class CachedProvider:
    """Keep each answer in a CSV file, so a second run needs no network."""

    def __init__(self, inner: WeatherProvider, cache_dir: str | Path):
        self.inner = inner
        self.name = inner.name
        self.cache_dir = Path(cache_dir)

    def daily(self, latitude: float, longitude: float, start: date, end: date) -> pd.DataFrame:
        key = f"{self.inner.name}_{latitude:.3f}_{longitude:.3f}_{start}_{end}.csv"
        path = self.cache_dir / key
        if path.exists():
            return pd.read_csv(path, parse_dates=["date"])
        frame = self.inner.daily(latitude, longitude, start, end)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return frame


def make_provider(name: str, cache_dir: str | Path | None = None) -> WeatherProvider:
    inner = OfflineWeather() if name == "offline" else OpenMeteoArchive() if name == "open-meteo" else None
    if inner is None:
        raise ValueError(f"unknown weather provider {name!r}")
    return CachedProvider(inner, cache_dir) if cache_dir is not None else inner


def window_features(daily: pd.DataFrame, when: pd.Timestamp, lag_days: int) -> dict:
    start = when.normalize() - timedelta(days=lag_days)
    end = when.normalize() - timedelta(days=1)
    w = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    if len(w) < max(1, lag_days // 2):
        return {f: np.nan for f in FEATURES}
    return {
        "w_tmean": float(w["tmean"].mean()), "w_tmax": float(w["tmax"].mean()), "w_tmin": float(w["tmin"].mean()),
        "w_rh": float(w["rh"].mean()), "w_precip_total": float(w["precip"].sum()),
        "w_rainy_days": float((w["precip"] >= RAINY_MM).sum()), "w_humid_days": float((w["rh"] >= HUMID_RH).sum()),
        "w_wind": float(w["wind"].mean()),
    }


def build_features(meta: pd.DataFrame, provider: WeatherProvider, lag_days: int = 14) -> pd.DataFrame:
    """One row of weather features for each image, in the row order of `meta`."""
    out = pd.DataFrame(index=meta.index, columns=list(FEATURES), dtype=float)
    keys = meta[["latitude", "longitude"]].round(3)
    for (lat, lon), rows in meta.groupby([keys["latitude"], keys["longitude"]]):
        dates = pd.to_datetime(rows["date"])
        start = (dates.min() - timedelta(days=lag_days)).date()
        end = (dates.max() - timedelta(days=1)).date()
        daily = provider.daily(float(lat), float(lon), start, end)
        daily["date"] = pd.to_datetime(daily["date"])
        for idx, when in zip(rows.index, dates):
            out.loc[idx] = window_features(daily, when, lag_days)
    return out


def assert_informative(features: pd.DataFrame, min_unique: int = 2) -> None:
    """Stop if a weather feature has one value for all images (for example one API call tiled to all rows)."""
    constant = [c for c in features.columns if features[c].nunique(dropna=True) < min_unique]
    if constant:
        raise ValueError(f"weather features with a constant value carry no information: {constant}")
