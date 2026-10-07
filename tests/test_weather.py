"""Problem 1 (constant weather vector) and problem 7 (hard-coded API key)."""
import io
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from paddyguard.config import Settings
from paddyguard.weather import (
    FEATURES,
    CachedProvider,
    OfflineWeather,
    OpenMeteoArchive,
    assert_informative,
    build_features,
    make_provider,
    window_features,
)


def test_offline_weather_is_deterministic_and_local():
    w = OfflineWeather()
    a = w.daily(15.0, 80.0, date(2022, 6, 1), date(2022, 6, 10))
    b = w.daily(15.0, 80.0, date(2022, 6, 1), date(2022, 6, 10))
    pd.testing.assert_frame_equal(a, b)
    c = w.daily(24.0, 88.0, date(2022, 6, 1), date(2022, 6, 10))
    assert not np.allclose(a["tmean"], c["tmean"])
    assert len(a) == 10 and set(["tmean", "rh", "precip"]) <= set(a.columns)


def test_window_excludes_the_photo_day_and_later():
    daily = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=30), "tmean": np.arange(30.0),
                          "tmax": 0.0, "tmin": 0.0, "rh": 95.0, "precip": 2.0, "wind": 1.0})
    f = window_features(daily, pd.Timestamp("2022-01-20"), lag_days=5)
    # Days 14..18 (0-based values 14..18), not day 19 (the photo day).
    assert f["w_tmean"] == pytest.approx(16.0)
    assert f["w_rainy_days"] == 5 and f["w_humid_days"] == 5


def test_short_window_gives_missing_values():
    daily = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=2), "tmean": 1.0, "tmax": 1.0, "tmin": 1.0,
                          "rh": 1.0, "precip": 0.0, "wind": 1.0})
    assert np.isnan(window_features(daily, pd.Timestamp("2022-03-01"), 14)["w_tmean"])


def test_features_vary_per_image_and_constant_is_rejected(meta):
    feats = build_features(meta, OfflineWeather(), lag_days=14)
    assert list(feats.columns) == list(FEATURES)
    assert_informative(feats)
    tiled = pd.DataFrame(np.tile(feats.iloc[[0]].to_numpy(), (len(feats), 1)), columns=feats.columns)
    with pytest.raises(ValueError, match="constant"):
        assert_informative(tiled)


def test_open_meteo_uses_no_key_and_parses_answer():
    seen = {}

    def fake_open(url, timeout):
        seen["url"] = url
        body = {"daily": {"time": ["2022-01-01", "2022-01-02"], "temperature_2m_mean": [25, 26],
                          "temperature_2m_max": [30, 31], "temperature_2m_min": [20, 21],
                          "relative_humidity_2m_mean": [80, 90], "precipitation_sum": [0, 5],
                          "wind_speed_10m_max": [10, 12]}}
        return io.BytesIO(json.dumps(body).encode())

    frame = OpenMeteoArchive(opener=fake_open).daily(15.0, 80.0, date(2022, 1, 1), date(2022, 1, 2))
    assert "key" not in seen["url"].lower() and "appid" not in seen["url"].lower()
    assert frame["precip"].tolist() == [0, 5] and frame["rh"].tolist() == [80, 90]


def test_open_meteo_bad_answer():
    with pytest.raises(ValueError):
        OpenMeteoArchive(opener=lambda url, timeout: io.BytesIO(b'{"error": true}')).daily(
            1, 1, date(2022, 1, 1), date(2022, 1, 2))


def test_cache_returns_the_same_frame_without_a_second_call(tmp_path):
    calls = []

    class Counting(OfflineWeather):
        def daily(self, *a):
            calls.append(a)
            return super().daily(*a)

    cached = CachedProvider(Counting(), tmp_path)
    first = cached.daily(15.0, 80.0, date(2022, 1, 1), date(2022, 1, 5))
    second = cached.daily(15.0, 80.0, date(2022, 1, 1), date(2022, 1, 5))
    assert len(calls) == 1
    np.testing.assert_allclose(first["tmean"], second["tmean"])


def test_provider_names_and_settings(monkeypatch):
    assert make_provider("offline").name == "offline"
    with pytest.raises(ValueError):
        make_provider("openweathermap")
    monkeypatch.setenv("PADDYGUARD_WEATHER_PROVIDER", "openweathermap")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("PADDYGUARD_WEATHER_PROVIDER", "open-meteo")
    monkeypatch.setenv("PADDYGUARD_WEATHER_LAG_DAYS", "7")
    s = Settings.from_env()
    assert s.weather_provider == "open-meteo" and s.weather_lag_days == 7
