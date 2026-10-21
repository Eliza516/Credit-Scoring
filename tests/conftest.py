import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, demo, target  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    c = config.load_default(prefer_real=False)
    c["split"]["imbalance"]["method"] = "none"
    c["split"]["min_goods"] = 500
    c["split"]["min_bads"] = 500
    return c


@pytest.fixture(scope="session")
def mapping():
    return dict(demo.DEMO_MAPPING)


@pytest.fixture(scope="session")
def raw():
    return demo.make_demo(n=15000, seed=7)


@pytest.fixture(scope="session")
def tgt(raw, mapping, cfg):
    return target.build_target(raw, mapping, cfg)


@pytest.fixture(scope="session")
def dev(raw, tgt, cfg):
    m = target.development_mask(tgt, cfg)
    return raw[m], tgt.loc[m, "y"]
