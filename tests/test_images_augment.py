"""Problems 3 (mixed pixel scales), 4 (undefined augmentations) and 5 (wrong preprocessing)."""
import numpy as np
import pytest

from paddyguard.augment import AUGMENTATIONS, RandomWeather
from paddyguard.images import image_features, load_rgb, thumbnail_signature


def leaf_image(seed=0, size=32):
    return np.random.default_rng(seed).random((size, size, 3)).astype(np.float32)


@pytest.mark.parametrize("name", sorted(AUGMENTATIONS))
def test_each_augmentation_keeps_the_contract(name):
    img = leaf_image()
    before = img.copy()
    out = AUGMENTATIONS[name](img, np.random.default_rng(1))
    assert out.shape == img.shape and out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1
    np.testing.assert_array_equal(img, before)  # input unchanged
    assert not np.array_equal(out, img)


@pytest.mark.parametrize("name", sorted(AUGMENTATIONS))
def test_augmentations_are_seeded(name):
    img = leaf_image()
    a = AUGMENTATIONS[name](img, np.random.default_rng(5))
    b = AUGMENTATIONS[name](img, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)


def test_wrong_pixel_scale_is_rejected():
    with pytest.raises(TypeError):
        AUGMENTATIONS["fog"]((leaf_image() * 255).astype(np.uint8), np.random.default_rng(0))
    with pytest.raises(ValueError):
        AUGMENTATIONS["fog"](leaf_image() * 255, np.random.default_rng(0))
    with pytest.raises(ValueError):
        AUGMENTATIONS["fog"](np.zeros((4, 4), np.float32), np.random.default_rng(0))


def test_random_weather_probability():
    img = leaf_image()
    never = RandomWeather(p=0.0, seed=0)
    assert never(img) is img
    always = RandomWeather(p=1.0, seed=0)
    assert not np.array_equal(always(img), img)


def test_load_rgb_scale_and_features(meta, synth_dir):
    img = load_rgb(synth_dir / meta["path"].iloc[0], size=32)
    assert img.dtype == np.float32 and img.shape == (32, 32, 3)
    assert 0 <= img.min() and img.max() <= 1
    feats = image_features(img)
    assert feats.shape == (31,) and np.isfinite(feats).all()


def test_thumbnail_signature_is_normalized(meta, synth_dir):
    sig = thumbnail_signature(synth_dir / meta["path"].iloc[0])
    assert sig.shape == (256,)
    assert abs(sig.mean()) < 1e-4 and abs(sig.std() - 1) < 1e-2
