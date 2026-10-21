"""Đọc file, ánh xạ cột, kiểm tra schema."""
from __future__ import annotations

import io

import pandas as pd

MAPPING_KEYS = [
    "id", "open_date", "target", "dpd_max", "first_bad_mob", "months_observed", "n_payments",
    "charge_off_flag", "write_off_flag", "bankruptcy_flag", "fraud_flag", "deceased_flag",
    "staff_vip_flag", "policy_flag",
]
DATE_KEYS = {"open_date"}


class IngestError(ValueError):
    pass


def read_file(file_obj, name: str, limits: dict | None = None, size_bytes: int | None = None) -> pd.DataFrame:
    """Đọc CSV/Excel; ném IngestError nếu vượt giới hạn (limits: max_upload_mb, max_rows, max_columns)."""
    limits = limits or {}
    max_mb = limits.get("max_upload_mb")
    if max_mb and size_bytes is not None and size_bytes > max_mb * 1024 * 1024:
        raise IngestError(f"File {size_bytes/1e6:.1f}MB vượt giới hạn {max_mb}MB.")
    lname = name.lower()
    try:
        if lname.endswith((".xlsx", ".xlsm", ".xls")):
            df = pd.read_excel(file_obj)
        elif lname.endswith((".csv", ".txt", ".gz")):
            raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
            df = None
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else raw, encoding=enc, sep=None, engine="python")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise IngestError("Không đọc được encoding của file CSV.")
        else:
            raise IngestError("Chỉ hỗ trợ CSV hoặc Excel (.xlsx).")
    except IngestError:
        raise
    except Exception as e:  # noqa: BLE001
        raise IngestError(f"Không đọc được file: {e}") from e
    return check_limits(df, limits)


def check_limits(df: pd.DataFrame, limits: dict | None) -> pd.DataFrame:
    limits = limits or {}
    if limits.get("max_rows") and len(df) > limits["max_rows"]:
        raise IngestError(f"{len(df)} dòng vượt giới hạn {limits['max_rows']}.")
    if limits.get("max_columns") and df.shape[1] > limits["max_columns"]:
        raise IngestError(f"{df.shape[1]} cột vượt giới hạn {limits['max_columns']}.")
    if df.empty:
        raise IngestError("File không có dữ liệu.")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise IngestError("Tên cột bị trùng: " + ", ".join(df.columns[df.columns.duplicated()]))
    return df


def profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_pct": df.isna().mean().round(4),
        "n_unique": df.nunique(dropna=True),
        "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })


def validate_mapping(df: pd.DataFrame, mapping: dict) -> list[str]:
    """Danh sách lỗi (rỗng = ok)."""
    errs = []
    for k, col in mapping.items():
        if col and col not in df.columns:
            errs.append(f"Cột '{col}' (gán cho '{k}') không có trong dữ liệu.")
    if not mapping.get("target") and not mapping.get("dpd_max"):
        errs.append("Cần cột target có sẵn hoặc cột dpd_max để dựng target.")
    used = [c for c in mapping.values() if c]
    if len(used) != len(set(used)):
        errs.append("Một cột đang được gán cho nhiều vai trò.")
    return errs


def apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Chuyển cột ngày sang datetime; trả bản sao."""
    out = df.copy()
    for k in DATE_KEYS:
        c = mapping.get(k)
        if c:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def feature_columns(df: pd.DataFrame, mapping: dict, extra_exclude: list[str] | None = None) -> list[str]:
    """Cột ứng viên làm biến mô hình = mọi cột không có vai trò đặc biệt."""
    used = {c for c in mapping.values() if c} | set(extra_exclude or [])
    return [c for c in df.columns if c not in used]
