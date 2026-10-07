"""Problems 2 (validation leak through augmentation) and 8 (near-duplicates across the split)."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from paddyguard.metadata import MetadataError, classes_of, encode, validate, weather_available
from paddyguard.splits import LeakageError, assert_no_overlap, assign_splits, duplicate_groups, split_groups


def test_metadata_defaults_and_checks(tmp_path):
    meta = validate(pd.DataFrame({"image_id": ["a.jpg", "b.jpg"], "label": ["blast", "normal"]}), check_files=False)
    assert meta["path"].tolist() == ["images/blast/a.jpg", "images/normal/b.jpg"]
    assert meta["field_id"].tolist() == ["a.jpg", "b.jpg"]
    assert not weather_available(meta)
    with pytest.raises(MetadataError, match="duplicated"):
        validate(pd.DataFrame({"image_id": ["a", "a"], "label": ["x", "x"]}), check_files=False)
    with pytest.raises(MetadataError, match="latitude"):
        validate(pd.DataFrame({"image_id": ["a"], "label": ["x"], "latitude": [95], "longitude": [1],
                               "date": ["2022-01-01"]}), check_files=False)
    with pytest.raises(MetadataError, match="do not exist"):
        validate(pd.DataFrame({"image_id": ["a"], "label": ["x"]}), tmp_path)


def test_class_list_is_sorted_and_encoding_checks(meta):
    classes = classes_of(meta)
    assert classes == sorted(classes)
    y = encode(meta, classes)
    assert y.min() == 0 and y.max() == len(classes) - 1
    with pytest.raises(MetadataError):
        encode(meta, classes[:-1])


def test_splits_keep_fields_apart_and_are_seeded(meta):
    y = encode(meta, classes_of(meta))
    groups = split_groups(meta)
    split = assign_splits(y, groups, seed=1)
    assert set(split) == {"train", "val", "test"}
    assert_no_overlap(groups, split)
    np.testing.assert_array_equal(split, assign_splits(y, groups, seed=1))


def test_overlap_is_detected():
    with pytest.raises(LeakageError):
        assert_no_overlap(pd.Series(["g1", "g1"]), np.array(["train", "test"]))


def test_near_duplicates_share_a_group(tmp_path):
    rng = np.random.default_rng(0)
    base = (rng.random((40, 40, 3)) * 255).astype(np.uint8)
    other = (rng.random((40, 40, 3)) * 255).astype(np.uint8)
    noisy = np.clip(base.astype(int) + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
    for name, arr in (("a.png", base), ("b.png", noisy), ("c.png", other)):
        Image.fromarray(arr).save(tmp_path / name)
    meta = pd.DataFrame({"path": ["a.png", "b.png", "c.png"], "label": ["x", "x", "x"],
                         "field_id": ["f1", "f2", "f3"]})
    dup = duplicate_groups(meta, tmp_path)
    assert dup[0] == dup[1] and dup[0] != dup[2]
    groups = split_groups(meta, dup)
    assert groups[0] == groups[1] and groups[0] != groups[2]


def test_duplicates_of_other_labels_are_not_merged(tmp_path):
    arr = (np.random.default_rng(1).random((40, 40, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(tmp_path / "a.png")
    Image.fromarray(arr).save(tmp_path / "b.png")
    meta = pd.DataFrame({"path": ["a.png", "b.png"], "label": ["x", "y"], "field_id": ["f1", "f2"]})
    dup = duplicate_groups(meta, tmp_path)
    assert dup[0] != dup[1]
