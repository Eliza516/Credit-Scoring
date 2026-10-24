"""Phép suy ra cột (tổng quát, khai báo trong hồ sơ dữ liệu)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def _months_between(end: pd.Series, start: pd.Series) -> pd.Series:
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)


def run(df: pd.DataFrame, spec: dict) -> pd.Series:
    """spec.op: extract_number | months_between | mean_of. Ném KeyError nếu thiếu cột nguồn."""
    op = spec["op"]
    if op == "extract_number":
        s = df[spec["col"]].astype("string")
        ov = spec.get("overrides") or {}
        num = pd.to_numeric(s.str.extract(_NUM)[0], errors="coerce")
        for k, v in ov.items():
            num = num.mask((s.str.strip() == k).fillna(False).astype(bool), v)
        return num.astype(float)
    if op == "months_between":
        end, start = df[spec["end"]], df[spec["start"]]
        for c in (end, start):
            if not pd.api.types.is_datetime64_any_dtype(c):
                raise KeyError("cột ngày chưa được chuyển sang datetime")
        return _months_between(end, start).astype(float)
    if op == "mean_of":
        return df[list(spec["cols"])].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    raise ValueError(f"Phép suy ra '{op}' chưa hỗ trợ")
