"""Torch image models with optional weather fusion (needs the `torch` extra).

- Backbones: `tiny_cnn` (CPU tests and demo), `efficientnet_v2_s`, `convnext_tiny` (torchvision).
- Fusion: `none` (image only), `late` (concatenate image and weather embeddings), `film` (weather
  gives a scale and a shift for each image feature).
- Two-phase fine-tuning for pretrained backbones: the head first with a frozen backbone, then all
  layers at a lower learning rate. Early stopping on the validation macro-F1.
- Images stream from disk. Augmentation runs on training images only.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .augment import RandomWeather
from .evaluate import metrics
from .images import load_rgb

BACKBONES = ("tiny_cnn", "efficientnet_v2_s", "convnext_tiny")
FUSIONS = ("none", "late", "film")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class PaddyDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, data_dir, y: np.ndarray, weather: np.ndarray | None, size: int,
                 augment: RandomWeather | None = None):
        self.paths = [Path(data_dir) / p for p in meta["path"]]
        self.y = y
        self.weather = weather if weather is not None else np.zeros((len(y), 0), np.float32)
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i], self.size)  # float32 RGB in [0, 1]
        if self.augment is not None:
            img = self.augment(img)
        x = (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))) - MEAN) / STD
        return x, torch.tensor(self.weather[i], dtype=torch.float32), int(self.y[i])


class TinyCNN(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.out_dim = width * 4

    def forward(self, x):
        return self.net(x)


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "tiny_cnn":
        net = TinyCNN()
        return net, net.out_dim
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}")
    try:
        import torchvision.models as tvm  # noqa: PLC0415 - optional
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('this backbone needs torchvision: pip install -e ".[torch]"') from exc
    weights = "DEFAULT" if pretrained else None
    if name == "efficientnet_v2_s":
        net = tvm.efficientnet_v2_s(weights=weights)
        dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
    else:
        net = tvm.convnext_tiny(weights=weights)
        dim = net.classifier[2].in_features
        net.classifier[2] = nn.Identity()
    return net, dim


class FusionNet(nn.Module):
    def __init__(self, backbone: nn.Module, dim: int, n_classes: int, weather_dim: int = 0, fusion: str = "none"):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {', '.join(FUSIONS)}")
        if fusion != "none" and weather_dim == 0:
            raise ValueError("a fusion model needs weather features")
        self.backbone, self.fusion = backbone, fusion
        if fusion == "late":
            self.weather = nn.Sequential(nn.Linear(weather_dim, 32), nn.ReLU())
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim + 32, n_classes))
        else:
            if fusion == "film":
                self.film = nn.Sequential(nn.Linear(weather_dim, 32), nn.ReLU(), nn.Linear(32, 2 * dim))
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, n_classes))

    def forward(self, image, weather=None):
        feats = self.backbone(image)
        if self.fusion == "late":
            feats = torch.cat([feats, self.weather(weather)], dim=1)
        elif self.fusion == "film":
            gamma, beta = self.film(weather).chunk(2, dim=1)
            feats = feats * (1 + gamma) + beta
        return self.head(feats)


@dataclass
class DeepConfig:
    backbone: str = "efficientnet_v2_s"
    fusion: str = "none"
    image_size: int = 224
    batch_size: int = 32
    head_epochs: int = 3
    epochs: int = 15
    lr: float = 1e-3
    finetune_lr: float = 1e-4
    patience: int = 3
    pretrained: bool = True
    augment_p: float = 0.5
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, w, _ in loader:
        out.append(torch.softmax(model(x.to(device), w.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out)


def standardize(weather: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Impute and scale weather with training statistics only."""
    mean = np.nanmean(weather[train_mask], axis=0)
    std = np.nanstd(weather[train_mask], axis=0)
    std[std == 0] = 1.0
    filled = np.where(np.isnan(weather), mean, weather)
    return ((filled - mean) / std).astype(np.float32)


def train_deep(meta: pd.DataFrame, data_dir, y: np.ndarray, split: np.ndarray, classes: list[str],
               weather: np.ndarray | None, run_dir, cfg: DeepConfig) -> dict:
    _seed(cfg.seed)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else
                          "cpu" if cfg.device == "auto" else cfg.device)
    w = standardize(weather, split == "train") if (weather is not None and cfg.fusion != "none") else None
    parts = {}
    for s in ("train", "val", "test"):
        mask = split == s
        aug = RandomWeather(cfg.augment_p, cfg.seed) if s == "train" else None
        parts[s] = PaddyDataset(meta[mask], data_dir, y[mask], None if w is None else w[mask], cfg.image_size, aug)
    gen = torch.Generator().manual_seed(cfg.seed)
    loaders = {s: DataLoader(ds, batch_size=cfg.batch_size, shuffle=(s == "train"), generator=gen,
                             num_workers=cfg.num_workers) for s, ds in parts.items()}
    backbone, dim = build_backbone(cfg.backbone, cfg.pretrained)
    model = FusionNet(backbone, dim, len(classes), 0 if w is None else w.shape[1], cfg.fusion).to(device)
    counts = np.bincount(y[split == "train"], minlength=len(classes)).astype(float)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(classes), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best.pt"
    history, best, waited = [], -1.0, 0
    two_phase = cfg.pretrained and cfg.backbone != "tiny_cnn"
    for epoch in range(cfg.epochs):
        frozen = two_phase and epoch < cfg.head_epochs
        for p in model.backbone.parameters():
            p.requires_grad = not frozen
        lr = cfg.lr if (frozen or not two_phase) else cfg.finetune_lr
        if epoch == 0 or (two_phase and epoch == cfg.head_epochs):
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        model.train()
        for x, wx, t in loaders["train"]:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device), wx.to(device)), t.to(device))
            loss.backward()
            opt.step()
        p_val = predict(model, loaders["val"], device)
        score = float(f1_score(y[split == "val"], p_val.argmax(1), average="macro", zero_division=0))
        history.append({"epoch": epoch, "val_macro_f1": score, "frozen_backbone": frozen})
        if score > best:
            best, waited = score, 0
            torch.save(model.state_dict(), ckpt)
        else:
            waited += 1
            if waited >= cfg.patience and not frozen:
                break
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    proba = predict(model, loaders["test"], device)
    result = metrics(y[split == "test"], proba, classes)
    result.update({"best_val_macro_f1": best, "history": history})
    (run_dir / "config.json").write_text(json.dumps({**asdict(cfg), "classes": classes}, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    np.save(run_dir / "test_proba.npy", proba)
    return result
