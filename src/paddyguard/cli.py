"""Command-line interface: `paddyguard <command>`."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from .config import Settings
from .metadata import MetadataError, classes_of, load_metadata, weather_available


def _weather(meta, settings, provider: str | None = None):
    from .weather import assert_informative, build_features, make_provider

    if not weather_available(meta):
        return None
    prov = make_provider(provider or settings.weather_provider, settings.weather_cache)
    feats = build_features(meta, prov, settings.weather_lag_days)
    assert_informative(feats)
    return feats


def _groups(meta, data_dir, no_dedup: bool = False):
    from .splits import duplicate_groups, split_groups

    return split_groups(meta, None if no_dedup else duplicate_groups(meta, data_dir))


def cmd_synth(args, settings) -> int:
    from .synthetic import generate

    path = generate(args.out, n_fields=args.fields, photos_per_field=args.photos, size=args.size, seed=args.seed)
    print(f"wrote {path} and the images under {Path(args.out) / 'images'}")
    return 0


def cmd_validate(args, settings) -> int:
    from .splits import duplicate_groups

    meta = load_metadata(args.data)
    classes = classes_of(meta)
    print(f"images {len(meta)}, classes {len(classes)}: " + ", ".join(
        f"{c} {int((meta['label'] == c).sum())}" for c in classes))
    print(f"fields {meta['field_id'].nunique()}")
    dup = duplicate_groups(meta, args.data)
    print(f"near-duplicate groups with 2+ images: {int((dup.value_counts() > 1).sum())}")
    if weather_available(meta):
        print("weather branch: available (latitude, longitude and date in every row)")
    else:
        print("weather branch: NOT available. paddyguard runs the image-only model. No constant weather is used.")
    return 0


def cmd_weather(args, settings) -> int:
    meta = load_metadata(args.data)
    feats = _weather(meta, settings, args.provider)
    if feats is None:
        print("error: the metadata has no latitude, longitude and date. The weather branch is off.", file=sys.stderr)
        return 1
    out = Path(args.out or settings.output_dir / "weather_features.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.assign(image_id=meta["image_id"]).to_csv(out, index=False)
    print(feats.describe().loc[["mean", "std", "min", "max"]].round(2).to_string())
    print(f"wrote {out}")
    return 0


def _print_ablation(result) -> None:
    for name, s in result.summary.items():
        first = result.per_seed[name][0]
        ci = first.get("macro_f1_ci") or (float("nan"), float("nan"))
        print(f"{name:14s} macro-F1 {s['mean']:.3f} +/- {s['std']:.3f} over {s['n_seeds']} seeds "
              f"(first seed {first['macro_f1']:.3f}, 95% field CI {ci[0]:.3f}-{ci[1]:.3f})")
    if result.comparison:
        c = result.comparison
        print(f"image+weather minus image (first seed): {c['delta']:+.3f}, 95% CI {c['ci'][0]:+.3f} to "
              f"{c['ci'][1]:+.3f}, p = {c['p_value']:.3f}")
    if result.robustness:
        print("image model robustness (test macro-F1): " + ", ".join(f"{k} {v:.3f}" for k, v in result.robustness.items()))


def cmd_ablation(args, settings) -> int:
    from .baseline import run_ablation

    meta = load_metadata(args.data)
    weather = _weather(meta, settings)
    result = run_ablation(meta, args.data, weather, _groups(meta, args.data, args.no_dedup),
                          seeds=tuple(args.seeds), n_boot=args.bootstrap)
    _print_ablation(result)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": result.summary, "per_seed": result.per_seed,
                                              "comparison": result.comparison, "robustness": result.robustness},
                                             indent=2, default=float), encoding="utf-8")
    return 0


def cmd_train(args, settings) -> int:
    try:
        from .deep import DeepConfig, train_deep
    except ImportError as exc:
        print(f'error: the train command needs torch: pip install -e ".[torch]" ({exc})', file=sys.stderr)
        return 1
    from .metadata import encode
    from .splits import assign_splits

    meta = load_metadata(args.data)
    classes = classes_of(meta)
    y = encode(meta, classes)
    weather = _weather(meta, settings)
    if args.fusion != "none" and weather is None:
        print("error: fusion needs weather features, and the metadata has none", file=sys.stderr)
        return 1
    groups = _groups(meta, args.data, args.no_dedup)
    for seed in args.seeds:
        split = assign_splits(y, groups, seed=seed)
        cfg = DeepConfig(backbone=args.backbone, fusion=args.fusion, image_size=args.image_size or settings.image_size,
                         epochs=args.epochs, head_epochs=args.head_epochs, pretrained=not args.no_pretrained,
                         seed=seed, device=settings.device, batch_size=args.batch_size)
        run = Path(args.out or settings.output_dir) / f"{args.backbone}_{args.fusion}_seed{seed}"
        m = train_deep(meta, args.data, y, split, classes, None if weather is None else weather.to_numpy(float), run, cfg)
        print(f"{args.backbone}/{args.fusion} seed {seed}: test macro-F1 {m['macro_f1']:.3f}, "
              f"balanced accuracy {m['balanced_accuracy']:.3f}, run {run}")
    return 0


def cmd_augment_preview(args, settings) -> int:
    from PIL import Image

    from .augment import AUGMENTATIONS
    from .images import load_rgb

    img = load_rgb(args.image, args.size)
    rng = np.random.default_rng(settings.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in AUGMENTATIONS.items():
        Image.fromarray((fn(img, rng) * 255).astype(np.uint8)).save(out / f"{name}.png")
    print(f"wrote {len(AUGMENTATIONS)} images to {out}")
    return 0


def cmd_demo(args, settings) -> int:
    from .baseline import run_ablation
    from .synthetic import generate

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "synthetic"
        generate(data, n_fields=args.fields, photos_per_field=args.photos, seed=args.seed)
        print(f"synthetic data: {args.fields} fields x {args.photos} photos (seed {args.seed})")
        cmd_validate(argparse.Namespace(data=data), settings)
        meta = load_metadata(data)
        local = Settings(**{**settings.__dict__, "weather_provider": "offline", "weather_cache": Path(tmp) / "wc"})
        weather = _weather(meta, local)
        result = run_ablation(meta, data, weather, _groups(meta, data), seeds=(0, 1, 2), n_boot=args.bootstrap)
        _print_ablation(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paddyguard", description="Paddy leaf disease classification with a weather ablation")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synth", help="write synthetic leaf images and metadata")
    s.add_argument("--out", type=Path, default=Path("data/synthetic"))
    s.add_argument("--fields", type=int, default=30)
    s.add_argument("--photos", type=int, default=25)
    s.add_argument("--size", type=int, default=64)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    def with_data(name, func, help_text):
        q = sub.add_parser(name, help=help_text)
        q.add_argument("--data", type=Path, required=True, help="folder with metadata.csv and the images")
        q.set_defaults(func=func)
        return q

    with_data("validate", cmd_validate, "check metadata, files, duplicates and the weather keys")
    q = with_data("weather", cmd_weather, "build lagged historical weather features for each image")
    q.add_argument("--provider", choices=["offline", "open-meteo"])
    q.add_argument("--out", type=Path)
    q = with_data("ablation", cmd_ablation, "image vs. weather vs. image+weather, same splits and seeds (CPU)")
    q.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    q.add_argument("--bootstrap", type=int, default=200)
    q.add_argument("--no-dedup", action="store_true", help="do not group near-duplicate photos")
    q.add_argument("--out", type=Path)
    q = with_data("train", cmd_train, "train a torch model (needs the torch extra)")
    q.add_argument("--backbone", choices=["tiny_cnn", "efficientnet_v2_s", "convnext_tiny"], default="efficientnet_v2_s")
    q.add_argument("--fusion", choices=["none", "late", "film"], default="none")
    q.add_argument("--seeds", type=int, nargs="+", default=[0])
    q.add_argument("--epochs", type=int, default=15)
    q.add_argument("--head-epochs", type=int, default=3)
    q.add_argument("--batch-size", type=int, default=32)
    q.add_argument("--image-size", type=int)
    q.add_argument("--no-pretrained", action="store_true")
    q.add_argument("--no-dedup", action="store_true")
    q.add_argument("--out", type=Path)
    a = sub.add_parser("augment-preview", help="write one image with each weather augmentation")
    a.add_argument("--image", type=Path, required=True)
    a.add_argument("--size", type=int, default=224)
    a.add_argument("--out", type=Path, default=Path("outputs/augment_preview"))
    a.set_defaults(func=cmd_augment_preview)
    d = sub.add_parser("demo", help="synthetic data and the CPU ablation")
    d.add_argument("--fields", type=int, default=30)
    d.add_argument("--photos", type=int, default=25)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--bootstrap", type=int, default=200)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    try:
        return args.func(args, settings)
    except (MetadataError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
