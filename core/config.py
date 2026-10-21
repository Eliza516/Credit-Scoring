"""Đọc/ghi cấu hình nghiệp vụ. Không phụ thuộc Streamlit."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
EXAMPLE_PATH = CONFIG_DIR / "business.example.yaml"
REAL_PATH = CONFIG_DIR / "business.yaml"

IMBALANCE_METHODS = ["none", "oversample_random", "oversample_smote", "undersample_random", "sample_weight"]


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_default(prefer_real: bool = True) -> dict:
    """business.yaml nếu có (và được phép), ngược lại business.example.yaml."""
    if prefer_real and REAL_PATH.exists():
        return load_yaml(REAL_PATH)
    return load_yaml(EXAMPLE_PATH)


def loads(text: str) -> dict:
    return yaml.safe_load(text) or {}


def dumps(cfg: dict) -> str:
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)


def save(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(dumps(cfg), encoding="utf-8")


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def set_(cfg: dict, dotted: str, value) -> dict:
    cur = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
    return cfg


def diff(old: dict, new: dict, prefix: str = "") -> list[tuple[str, object, object]]:
    """Danh sách (đường dẫn, giá trị cũ, giá trị mới) khác nhau."""
    out: list[tuple[str, object, object]] = []
    keys = list(dict.fromkeys(list(old) + list(new)))
    for k in keys:
        path = f"{prefix}{k}"
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            out += diff(a, b, path + ".")
        elif a != b:
            out.append((path, a, b))
    return out


def validate(cfg: dict) -> list[str]:
    """Trả về danh sách lỗi/cảnh báo (rỗng = hợp lệ)."""
    errs: list[str] = []
    sp = cfg.get("split", {})
    if abs(sp.get("train_pct", 0) + sp.get("test_pct", 0) - 1.0) > 1e-6:
        errs.append("split: train_pct + test_pct phải bằng 1.")
    m = sp.get("imbalance", {}).get("method")
    if m is not None and m not in IMBALANCE_METHODS:
        errs.append(f"split.imbalance.method '{m}' không hợp lệ.")
    sc = cfg.get("scorecard", {})
    if sc.get("pdo", 0) <= 0:
        errs.append("scorecard.pdo phải > 0.")
    if sc.get("base_odds_good_to_bad", 0) <= 0:
        errs.append("scorecard.base_odds_good_to_bad phải > 0.")
    st = cfg.get("modeling", {}).get("stepwise", {})
    if st.get("p_enter", 0) > st.get("p_remove", 1):
        errs.append("stepwise: p_enter không được lớn hơn p_remove.")
    iv = cfg.get("selection", {}).get("iv", {})
    if not (iv.get("useless_below", 0) <= iv.get("weak_below", 0) <= iv.get("medium_below", 0) <= iv.get("strong_below", 0)):
        errs.append("selection.iv: các ngưỡng phải tăng dần.")
    return errs


def clone(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
