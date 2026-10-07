"""Problem 9 (no baselines, one unseeded run) and the CLI."""
import json

import numpy as np
import pytest

from paddyguard.baseline import run_ablation
from paddyguard.cli import main
from paddyguard.evaluate import ece, group_bootstrap, metrics, paired_bootstrap, seed_summary
from paddyguard.splits import split_groups
from paddyguard.weather import OfflineWeather, build_features


def test_ablation_runs_three_variants_with_seeds(meta, synth_dir):
    weather = build_features(meta, OfflineWeather())
    result = run_ablation(meta, synth_dir, weather, split_groups(meta), seeds=(0, 1), n_boot=20)
    assert set(result.summary) == {"image", "weather", "image+weather"}
    assert all(s["n_seeds"] == 2 for s in result.summary.values())
    assert result.summary["image"]["mean"] > 1 / len(result.classes)
    assert "delta" in result.comparison and len(result.comparison["ci"]) == 2
    assert set(result.robustness) >= {"clean", "fog", "rain"}


def test_ablation_without_weather_is_image_only(meta, synth_dir):
    result = run_ablation(meta, synth_dir, None, split_groups(meta), seeds=(0,), n_boot=0, robustness=False)
    assert set(result.summary) == {"image"}
    assert result.comparison == {}


def test_metric_helpers():
    y = np.array([0, 1, 1, 2])
    proba = np.eye(3)[[0, 1, 2, 2]]
    m = metrics(y, proba, ["a", "b", "c"])
    assert m["confusion_matrix"]["labels"] == ["a", "b", "c"]
    assert m["recall"]["b"] == 0.5
    assert ece(y, np.eye(3)[y]) == 0.0
    lo, hi = group_bootstrap(y, proba, ["g1", "g1", "g2", "g3"], n_boot=20)
    assert 0 <= lo <= hi <= 1
    same = paired_bootstrap(y, proba, proba, ["g1", "g2", "g3", "g4"], n_boot=20)
    assert same["delta"] == 0.0
    assert seed_summary([0.5, 0.7])["mean"] == pytest.approx(0.6)


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PADDYGUARD_WEATHER_CACHE", str(tmp_path / "cache"))
    data = tmp_path / "d"
    assert main(["synth", "--out", str(data), "--fields", "8", "--photos", "10", "--size", "32", "--seed", "1"]) == 0
    assert main(["validate", "--data", str(data)]) == 0
    assert "weather branch: available" in capsys.readouterr().out
    assert main(["weather", "--data", str(data), "--out", str(tmp_path / "w.csv")]) == 0
    assert (tmp_path / "w.csv").exists()
    out = tmp_path / "ablation.json"
    assert main(["ablation", "--data", str(data), "--seeds", "0", "--bootstrap", "10", "--out", str(out)]) == 0
    assert "image+weather" in json.loads(out.read_text(encoding="utf-8"))["summary"]
    first = next((data / "images").rglob("*.png"))
    assert main(["augment-preview", "--image", str(first), "--size", "32", "--out", str(tmp_path / "prev")]) == 0
    assert len(list((tmp_path / "prev").glob("*.png"))) == 6


def test_cli_without_weather_keys(tmp_path, capsys):
    data = tmp_path / "d"
    main(["synth", "--out", str(data), "--fields", "4", "--photos", "6", "--size", "32"])
    path = data / "metadata.csv"
    import pandas as pd

    pd.read_csv(path).drop(columns=["latitude", "longitude", "date"]).to_csv(path, index=False)
    assert main(["validate", "--data", str(data)]) == 0
    assert "NOT available" in capsys.readouterr().out
    assert main(["weather", "--data", str(data)]) == 1


def test_missing_data_is_a_clean_error(tmp_path, capsys):
    assert main(["validate", "--data", str(tmp_path / "none")]) == 1
    assert "paddyguard synth" in capsys.readouterr().err
