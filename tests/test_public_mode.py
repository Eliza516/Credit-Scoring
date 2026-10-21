from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import config as cfgmod

ROOT = Path(__file__).resolve().parent.parent


def test_public_ignores_real_config(tmp_path, monkeypatch):
    real = tmp_path / "business.yaml"
    c = cfgmod.load_yaml(cfgmod.EXAMPLE_PATH)
    c["scorecard"]["base_score"] = 777
    cfgmod.save(c, real)
    monkeypatch.setattr(cfgmod, "REAL_PATH", real)
    assert cfgmod.load_default(prefer_real=True)["scorecard"]["base_score"] == 777
    assert cfgmod.load_default(prefer_real=False)["scorecard"]["base_score"] != 777

    monkeypatch.setenv("APP_MODE", "public")
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "3_docs.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.session_state["cfg"]["scorecard"]["base_score"] != 777      # public không đọc business.yaml

    monkeypatch.setenv("APP_MODE", "private")
    at2 = AppTest.from_file(str(ROOT / "app" / "pages" / "3_docs.py"), default_timeout=60)
    at2.run()
    assert at2.session_state["cfg"]["scorecard"]["base_score"] == 777


def test_public_limit_blocks_large_demo(monkeypatch):
    monkeypatch.setenv("APP_MODE", "public")
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "1_analysis.py"), default_timeout=120)
    at.run()
    at.session_state["cfg"]["public_mode"]["max_rows"] = 1000
    [b for b in at.button if "dữ liệu demo" in b.label][0].click().run()
    assert any("giới hạn" in e.value for e in at.error)
    assert "raw" not in at.session_state


def test_gitignore_protects_secrets():
    ig = (ROOT / ".gitignore").read_text()
    for item in ("data/", "artifacts/", "config/business.yaml", ".env"):
        assert item in ig
