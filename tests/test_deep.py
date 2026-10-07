"""Torch model tests (problem 6: two-phase fine-tuning, fusion variants). They skip without torch."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from paddyguard.deep import DeepConfig, FusionNet, TinyCNN, build_backbone, standardize, train_deep  # noqa: E402
from paddyguard.metadata import classes_of, encode  # noqa: E402
from paddyguard.splits import assign_splits, split_groups  # noqa: E402
from paddyguard.weather import OfflineWeather, build_features  # noqa: E402


@pytest.mark.parametrize("fusion", ["none", "late", "film"])
def test_fusion_forward(fusion):
    net = FusionNet(TinyCNN(), 64, 5, weather_dim=0 if fusion == "none" else 8, fusion=fusion)
    out = net(torch.zeros(2, 3, 32, 32), torch.zeros(2, 8))
    assert out.shape == (2, 5)


def test_fusion_needs_weather():
    with pytest.raises(ValueError):
        FusionNet(TinyCNN(), 64, 5, weather_dim=0, fusion="late")
    with pytest.raises(ValueError):
        FusionNet(TinyCNN(), 64, 5, weather_dim=8, fusion="concat")
    with pytest.raises(ValueError):
        build_backbone("vgg16", pretrained=False)


def test_standardize_uses_training_rows():
    w = np.array([[1.0], [3.0], [100.0], [np.nan]])
    out = standardize(w, np.array([True, True, False, False]))
    assert out[0, 0] == pytest.approx(-1.0) and out[3, 0] == pytest.approx(0.0)


def test_train_deep_tiny_with_late_fusion(meta, synth_dir, tmp_path):
    classes = classes_of(meta)
    y = encode(meta, classes)
    split = assign_splits(y, split_groups(meta), seed=0)
    weather = build_features(meta, OfflineWeather()).to_numpy(float)
    cfg = DeepConfig(backbone="tiny_cnn", fusion="late", image_size=32, batch_size=16, epochs=2, pretrained=False,
                     device="cpu", seed=0)
    m = train_deep(meta, synth_dir, y, split, classes, weather, tmp_path, cfg)
    assert 0 <= m["macro_f1"] <= 1 and len(m["history"]) >= 1
    assert json.loads((tmp_path / "config.json").read_text())["fusion"] == "late"
    assert (tmp_path / "best.pt").exists()
