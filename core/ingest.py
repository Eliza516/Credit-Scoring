"""Đọc file, ánh xạ cột, kiểm tra schema."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

MAPPING_KEYS = [
    "id", "open_date", "target", "status_col", "dpd_max", "first_bad_mob", "months_observed", "n_payments",
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
        "example": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "" for c in df.columns],
    })


def validate_mapping(df: pd.DataFrame, mapping: dict) -> list[str]:
    """Danh sách lỗi (rỗng = ok). Nguồn dựng target (target/dpd_max hoặc performance) được kiểm ở bước 3."""
    errs = []
    for k, col in mapping.items():
        if col and col not in df.columns:
            errs.append(f"Cột '{col}' (gán cho '{k}') không có trong dữ liệu.")
    used = [c for c in mapping.values() if c]
    if len(used) != len(set(used)):
        errs.append("Một cột đang được gán cho nhiều vai trò.")
    return errs


def apply_mapping(df: pd.DataFrame, mapping: dict, date_formats: dict | None = None) -> pd.DataFrame:
    """Chuyển cột ngày sang datetime (theo định dạng khai báo nếu có); trả bản sao."""
    out = df.copy()
    fm = date_formats or {}
    for k in DATE_KEYS:
        c = mapping.get(k)
        if c and not pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = pd.to_datetime(out[c], format=fm.get(c), errors="coerce")
    for c, f in fm.items():                                        # các cột ngày khác khai báo trong hồ sơ
        if c in out.columns and not pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = pd.to_datetime(out[c], format=f, errors="coerce")
    return out


def feature_columns(df: pd.DataFrame, mapping: dict, extra_exclude: list[str] | None = None) -> list[str]:
    """Cột ứng viên làm biến mô hình = mọi cột không có vai trò đặc biệt."""
    used = {c for c in mapping.values() if c} | set(extra_exclude or [])
    return [c for c in df.columns if c not in used]


# ================================================================== nạp file lớn theo khối
def _compression(name: str) -> str | None:
    n = name.lower()
    return "gzip" if n.endswith(".gz") else ("zip" if n.endswith(".zip") else None)


def read_csv_chunked(source, name: str = "", *, skip_columns: list[str] | None = None, chunk_rows: int = 200_000,
                     keep=None, max_rows: int = 0, seed: int = 42, progress=None) -> tuple[pd.DataFrame, dict]:
    """Đọc CSV (kể cả .gz) theo khối; lọc sớm bằng `keep(chunk)->mask`; lấy mẫu đều tối đa ~max_rows dòng (0 = lấy hết).

    Lấy mẫu giữ bộ nhớ ≤ ~2×max_rows: mỗi dòng giữ với xác suất p; khi vượt 2×max_rows thì giảm p một nửa và
    làm thưa phần đã giữ → mẫu cuối cùng đều ngẫu nhiên, rồi cắt đúng max_rows.
    Trả (df, thông tin {rows_read, rows_after_filter, rows_kept, sampled}).
    """
    rng = np.random.default_rng(seed)
    comp = _compression(name)
    head = pd.read_csv(source, nrows=0, compression=comp)
    if hasattr(source, "seek"):
        source.seek(0)
    skip = set(skip_columns or [])
    usecols = [c for c in head.columns if c not in skip] if skip else None
    parts: list[pd.DataFrame] = []
    p, kept_rows, read_rows, after_filter = 1.0, 0, 0, 0
    reader = pd.read_csv(source, chunksize=chunk_rows, low_memory=False, usecols=usecols, compression=comp)
    for chunk in reader:
        read_rows += len(chunk)
        if keep is not None:
            chunk = chunk[keep(chunk)]
        after_filter += len(chunk)
        if max_rows and p < 1.0:
            chunk = chunk[rng.random(len(chunk)) < p]
        parts.append(chunk)
        kept_rows += len(chunk)
        if max_rows and kept_rows > 2 * max_rows:
            p /= 2
            parts = [c[rng.random(len(c)) < 0.5] for c in parts]
            kept_rows = sum(len(c) for c in parts)
        if progress:
            progress(read_rows, after_filter)
    df = pd.concat(parts, ignore_index=True) if parts else head.iloc[0:0]
    sampled = bool(max_rows and after_filter > max_rows)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    for c in [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]:   # cột bị đọc thành chữ ở vài khối (vd dòng tổng kết cuối file) → ép về số nếu được
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        if s.notna().any() and num.notna().sum() == s.notna().sum():
            df[c] = num
    return df, {"rows_read": read_rows, "rows_after_filter": after_filter, "rows_kept": len(df), "sampled": sampled}


# ================================================================== nhiều file một lúc
def _sig(cols) -> frozenset:
    return frozenset(str(c).strip().lower() for c in cols)


def group_by_schema(columns_by_file: list[list[str]], min_overlap: float = 0.9) -> list[list[int]]:
    """Gom file có cấu trúc giống nhau (Jaccard của tập tên cột ≥ min_overlap) thành nhóm. Trả danh sách chỉ số file theo nhóm."""
    groups: list[tuple[frozenset, list[int]]] = []
    for i, cols in enumerate(columns_by_file):
        s = _sig(cols)
        for k, (gs, members) in enumerate(groups):
            if len(s & gs) / max(len(s | gs), 1) >= min_overlap:
                groups[k] = (gs | s, members + [i])
                break
        else:
            groups.append((s, [i]))
    return [m for _, m in groups]


def combine_frames(frames: list[pd.DataFrame], names: list[str]) -> pd.DataFrame:
    """Nối dòng các file cùng cấu trúc (hợp các cột, cột thiếu = NaN) và thêm cột _source_file."""
    if len(frames) == 1:
        out = frames[0].copy()
    else:
        low = [{c: str(c).strip() for c in f.columns} for f in frames]
        canon: dict[str, str] = {}
        norm = []
        for f, m in zip(frames, low):
            g = f.rename(columns={c: canon.setdefault(m[c].lower(), m[c]) for c in f.columns})
            norm.append(g)
        out = pd.concat(norm, ignore_index=True, sort=False)
    out["_source_file"] = np.repeat(names, [len(f) for f in frames]) if len(frames) > 1 else names[0]
    return out


def suggest_group_role(df: pd.DataFrame, is_largest: bool) -> str:
    """'main' | 'performance' | 'ignore' gợi ý cho một nhóm file."""
    from . import performance
    if performance.detect_layout(df) == "wide":
        return "performance"
    g = performance.guess_columns(df)
    if g["id"] and g["dpd"] and g["period"] and df.shape[1] <= 8:
        return "performance"
    return "main" if is_largest else "ignore"


def duplicate_id_report(df: pd.DataFrame, id_col: str) -> dict:
    """Số ID trùng (kể cả giữa các file khi có _source_file)."""
    dup = df[id_col].duplicated(keep=False)
    out = {"duplicate_rows": int(df[id_col].duplicated().sum()), "duplicate_ids": int(df.loc[dup, id_col].nunique())}
    if "_source_file" in df.columns and dup.any():
        out["cross_file_ids"] = int((df.loc[dup].groupby(id_col)["_source_file"].nunique() > 1).sum())
    return out
