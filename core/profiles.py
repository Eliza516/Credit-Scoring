"""Hồ sơ dữ liệu (dataset profile): ánh xạ cột, định dạng ngày, cột rò rỉ, cột suy ra… cho từng bộ dữ liệu cụ thể.

Hồ sơ nằm ở config/profiles/<tên>.yaml; core chỉ đọc và áp dụng, không chứa quy tắc riêng của bộ dữ liệu nào.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from . import config as cfgmod, derive

PROFILE_DIR = cfgmod.CONFIG_DIR / "profiles"


def list_profiles() -> dict[str, str]:
    """{tên: tiêu đề}"""
    out = {}
    for p in sorted(PROFILE_DIR.glob("*.yaml")):
        d = cfgmod.load_yaml(p)
        out[d.get("name", p.stem)] = d.get("title", p.stem)
    return out


def load_profile(name: str) -> dict:
    p = PROFILE_DIR / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Không có hồ sơ '{name}'")
    return cfgmod.load_yaml(p)


def deep_merge(base: dict, over: dict) -> dict:
    """Gộp over vào bản sao của base (dict lồng nhau được gộp, giá trị khác thì ghi đè)."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_to_config(cfg: dict, profile: dict) -> dict:
    """Cấu hình mới = cfg + config_overrides của hồ sơ + định dạng ngày."""
    new = deep_merge(cfg, profile.get("config_overrides", {}))
    new.setdefault("columns", {})["date_formats"] = dict(profile.get("date_formats", {}))
    return new


def mapping_for(df_columns, profile: dict) -> dict:
    """Ánh xạ vai trò → cột theo hồ sơ, chỉ giữ cột thật sự có trong dữ liệu."""
    cols = set(df_columns)
    return {k: v for k, v in profile.get("mapping", {}).items() if v in cols}


def column_notes(profile: dict) -> dict[str, str]:
    """{cột: nghĩa} từ từ điển của hồ sơ (để hiện chú thích trong dashboard)."""
    return {c: v.get("meaning", "") for c, v in (profile.get("dictionary") or {}).items()}


def excluded_features(profile: dict) -> dict[str, str]:
    """{cột: lý do} các cột không được làm biến mô hình theo hồ sơ."""
    out = {}
    for c in profile.get("identifier_columns", []):
        out[c] = "định danh (hồ sơ)"
    for c in profile.get("leakage_columns", []):
        out.setdefault(c, "chỉ có sau khi vay / không rõ thời điểm (hồ sơ)")
    return out


def parse_dates(df: pd.DataFrame, formats: dict[str, str]) -> pd.DataFrame:
    """Chuyển các cột ngày khai báo trong hồ sơ sang datetime (bản sao nếu có thay đổi)."""
    todo = {c: f for c, f in (formats or {}).items() if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c])}
    if not todo:
        return df
    out = df.copy()
    for c, f in todo.items():
        out[c] = pd.to_datetime(out[c], format=f, errors="coerce")
    return out


def apply_derived(df: pd.DataFrame, profile: dict) -> tuple[pd.DataFrame, list[str]]:
    """Tạo cột suy ra + bỏ cột gốc theo hồ sơ. Trả (df mới, danh sách cột đã tạo). Cột nguồn thiếu → bỏ qua."""
    specs = profile.get("derived", [])
    out = df.copy()
    made = []
    for s in specs:
        try:
            out[s["name"]] = derive.run(out, s)
            made.append(s["name"])
        except KeyError:
            continue
    ok = len(made) == len(specs)                      # chỉ bỏ cột gốc khi mọi cột suy ra đều tạo được
    drop = [c for c in profile.get("drop_after_derive", []) if c in out.columns] if ok else []
    return out.drop(columns=drop), made


def row_filter(profile: dict, cfg: dict, mapping: dict | None = None):
    """Hàm lọc dòng sớm khi đọc file lớn (chỉ giữ dòng có nhãn, từ năm tối thiểu). Cột thiếu → bỏ qua bộ lọc đó."""
    load = profile.get("load", {}) or {}
    mp = mapping or profile.get("mapping", {})
    st_col, open_col = mp.get("status_col"), mp.get("open_date")
    labels = list(cfg["target"]["status"].get("bad", [])) + list(cfg["target"]["status"].get("good", []))
    min_year = load.get("min_open_year")
    fmt = (profile.get("date_formats") or {}).get(open_col)

    def keep(chunk: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=chunk.index)
        if load.get("keep_status") and st_col in chunk.columns and labels:
            m &= chunk[st_col].isin(labels)
        if min_year and open_col in chunk.columns:
            d = pd.to_datetime(chunk[open_col], format=fmt, errors="coerce")
            m &= d.dt.year >= min_year
        return m
    return keep
