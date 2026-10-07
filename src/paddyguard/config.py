"""Runtime settings from environment variables, with safe defaults. No API key is read."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROVIDERS = ("offline", "open-meteo")


def _int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be {minimum} or more, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    output_dir: Path
    seed: int
    image_size: int
    weather_provider: str
    weather_cache: Path
    weather_lag_days: int
    device: str

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.environ.get("PADDYGUARD_WEATHER_PROVIDER", "").strip() or "offline"
        if provider not in PROVIDERS:
            raise ValueError(f"PADDYGUARD_WEATHER_PROVIDER must be one of {', '.join(PROVIDERS)}")
        return cls(
            data_dir=Path(os.environ.get("PADDYGUARD_DATA_DIR", "").strip() or "data/paddy"),
            output_dir=Path(os.environ.get("PADDYGUARD_OUTPUT_DIR", "").strip() or "runs"),
            seed=_int("PADDYGUARD_SEED", 42, 0),
            image_size=_int("PADDYGUARD_IMAGE_SIZE", 224, 16),
            weather_provider=provider,
            weather_cache=Path(os.environ.get("PADDYGUARD_WEATHER_CACHE", "").strip() or "weather_cache"),
            weather_lag_days=_int("PADDYGUARD_WEATHER_LAG_DAYS", 14, 1),
            device=os.environ.get("PADDYGUARD_DEVICE", "").strip() or "auto",
        )
