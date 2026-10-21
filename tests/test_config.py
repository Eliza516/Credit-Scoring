from core import config


def test_example_loads_and_valid(cfg):
    assert config.validate(cfg) == []
    assert cfg["split"]["imbalance"]["method"] == "none"  # fixture đặt; mặc định file là null


def test_default_imbalance_is_user_choice():
    c = config.load_default(prefer_real=False)
    assert c["split"]["imbalance"]["method"] is None
    assert c["split"]["imbalance"]["prior_correction"] is None


def test_roundtrip_hash_and_diff(cfg):
    c2 = config.loads(config.dumps(cfg))
    assert config.config_hash(cfg) == config.config_hash(c2)
    c3 = config.clone(cfg)
    config.set_(c3, "cleaning.max_missing_pct", 0.5)
    d = config.diff(cfg, c3)
    assert d == [("cleaning.max_missing_pct", 0.3, 0.5)]


def test_validate_catches_errors(cfg):
    c = config.clone(cfg)
    c["split"]["train_pct"] = 0.9
    c["scorecard"]["pdo"] = 0
    c["selection"]["iv"]["weak_below"] = 0.001
    errs = config.validate(c)
    assert len(errs) == 3
