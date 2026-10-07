"""Metrics with class names, group bootstrap intervals and a paired comparison of two models."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score


def ece(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(conf[idx == b].mean() - correct[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def metrics(y: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    present = sorted(int(c) for c in np.unique(y))
    pred = proba.argmax(axis=1)
    recall = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "recall": {classes[c]: (float(recall[c]) if c in present else float("nan")) for c in labels},
        "confusion_matrix": {"labels": classes, "matrix": confusion_matrix(y, pred, labels=labels).tolist()},
        "ece": ece(y, proba),
    }


def _members(groups) -> list[np.ndarray]:
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    return [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]


def _macro_f1(y, proba) -> float:
    return float(f1_score(y, proba.argmax(axis=1), labels=np.unique(y), average="macro", zero_division=0))


def group_bootstrap(y: np.ndarray, proba: np.ndarray, groups, n_boot: int = 200, seed: int = 42) -> tuple[float, float]:
    members = _members(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        values.append(_macro_f1(y[idx], proba[idx]))
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def paired_bootstrap(y: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, groups, n_boot: int = 500,
                     seed: int = 42) -> dict:
    """Macro-F1 of A minus B on the same images. Each draw resamples whole groups."""
    members = _members(groups)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        deltas.append(_macro_f1(y[idx], proba_a[idx]) - _macro_f1(y[idx], proba_b[idx]))
    deltas = np.asarray(deltas)
    return {
        "delta": _macro_f1(y, proba_a) - _macro_f1(y, proba_b),
        "ci": (float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))),
        "p_value": float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))),
    }


def seed_summary(values) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"n_seeds": int(len(arr)), "mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}
