"""Xuất báo cáo Excel/PDF, lưu mô hình, chấm điểm file mới."""
from __future__ import annotations

import io
import json
import pickle

import numpy as np
import pandas as pd

from . import binning, config as cfgmod, profiles, scorecard


def build_bundle(cfg: dict, model: dict, specs: dict, points: pd.DataFrame, cutoff: dict | None, cleaning_log: dict | None = None,
                 segment: dict | None = None, profile: dict | None = None) -> dict:
    """Gói mô hình đủ để chấm điểm file mới (cần specs có 'woe'). profile: hồ sơ dữ liệu để tái tạo cột ngày/cột suy ra khi chấm file thô."""
    slim = None
    if profile:
        slim = {k: profile[k] for k in ("name", "date_formats", "derived", "drop_after_derive") if k in profile}
    return {"config": cfg, "config_hash": cfgmod.config_hash(cfg), "profile": slim, "model": model,
            "specs": {v: specs[v] for v in model["variables"]}, "points": points, "cutoff": cutoff,
            "cleaning_log": cleaning_log, "segment": segment}


def bundle_to_bytes(bundle: dict) -> bytes:
    return pickle.dumps(bundle)


def bundle_from_bytes(blob: bytes) -> dict:
    """Chỉ nạp bundle do chính hệ thống này xuất (pickle không an toàn với file lạ)."""
    return pickle.loads(blob)  # noqa: S301


def score_new_data(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Chấm điểm file mới: thêm score, pd, zone (nếu có cutoff). Thiếu cột → lỗi rõ ràng."""
    need = bundle["model"]["variables"]
    prof = bundle.get("profile")
    if prof:                                                   # file thô: parse ngày + tạo cột suy ra như lúc huấn luyện
        df = profiles.parse_dates(df, prof.get("date_formats"))
        df, _ = profiles.apply_derived(df, prof)
    miss = [v for v in need if v not in df.columns]
    if miss:
        raise ValueError("File thiếu cột: " + ", ".join(miss))
    cfg = bundle["config"]
    s = scorecard.score(df, bundle["model"], bundle["specs"], bundle["points"], cfg)
    W = binning.transform(df, bundle["specs"], need)
    z = bundle["model"]["intercept"] + bundle["model"]["prior_offset"] + sum(bundle["model"]["coef"][v] * W[v] for v in need)
    out = df.copy()
    out["score"] = s
    out["pd"] = 1 / (1 + np.exp(-z))
    c = bundle.get("cutoff")
    if c and c.get("cutoff") is not None:
        out["zone"] = scorecard.apply_zones(s, c["reject_below"], c["approve_from"])
    return out


def to_excel(sheets: dict[str, pd.DataFrame], cfg: dict | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            if df is None:
                continue
            d = df.reset_index() if df.index.name else df
            d.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            for i, col in enumerate(d.columns, 1):
                width = min(max(len(str(col)), *(len(str(v)) for v in d[col].head(200))) + 2, 60) if len(d) else len(str(col)) + 2
                ws.column_dimensions[ws.cell(1, i).column_letter].width = width
        if cfg is not None:
            pd.DataFrame({"config_yaml": cfgmod.dumps(cfg).splitlines()}).to_excel(xw, sheet_name="config", index=False)
    return buf.getvalue()


def to_pdf(title: str, sections: list[tuple[str, pd.DataFrame]], notes: list[str] | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    st = getSampleStyleSheet()
    els = [Paragraph(title, st["Title"])]
    for n in notes or []:
        els.append(Paragraph(n.replace("&", "&amp;").replace("<", "&lt;"), st["Normal"]))
    for name, df in sections:
        els += [Spacer(1, 10), Paragraph(name, st["Heading2"])]
        d = df.head(40).copy()
        d = d.map(lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v))
        data = [list(map(str, d.columns))] + d.values.tolist()
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
        els.append(t)
    doc.build(els)
    return buf.getvalue()


def config_json(cfg: dict) -> str:
    return json.dumps(cfg, ensure_ascii=False, indent=2, default=str)
