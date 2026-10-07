import pytest

from paddyguard.metadata import load_metadata
from paddyguard.synthetic import generate


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("paddy")
    generate(out, n_fields=12, photos_per_field=12, size=40, seed=2)
    return out


@pytest.fixture(scope="session")
def meta(synth_dir):
    return load_metadata(synth_dir)
