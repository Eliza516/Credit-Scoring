"""Lưu kết quả trung gian theo phiên (kèm phiên bản cấu hình). Không tự xoá."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

from . import config as cfgmod

STEPS = ["ingest", "cleaning", "target", "segmentation", "binning", "selection", "modeling", "evaluation", "scorecard"]
ROOT = Path(__file__).resolve().parent.parent / "artifacts"


class Store:
    def __init__(self, session_id: str, root: Path | None = None):
        self.dir = (root or ROOT) / session_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, step: str) -> Path:
        return self.dir / f"{step}.pkl"

    def save(self, step: str, payload, cfg: dict) -> None:
        rec = {"payload": payload, "config_hash": cfgmod.config_hash(cfg), "config": cfg, "ts": time.time(), "step": step}
        tmp = self._path(step).with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(rec, f)
        tmp.replace(self._path(step))

    def load(self, step: str):
        p = self._path(step)
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)  # file do chính phiên này ghi

    def has(self, step: str) -> bool:
        return self._path(step).exists()

    def status(self, cfg: dict | None = None) -> list[dict]:
        """Trạng thái từng bước: done / stale (bước trước chạy lại sau nó) / config_changed."""
        out, prev_ts = [], 0.0
        h = cfgmod.config_hash(cfg) if cfg else None
        for step in STEPS:
            rec = self.load(step)
            if rec is None:
                out.append({"step": step, "state": "pending", "ts": None, "config_hash": None})
                continue
            state = "done"
            if rec["ts"] < prev_ts:
                state = "stale"
            elif h and rec["config_hash"] != h:
                state = "config_changed"
            out.append({"step": step, "state": state, "ts": rec["ts"], "config_hash": rec["config_hash"]})
            prev_ts = max(prev_ts, rec["ts"])
        return out

    def save_config_version(self, cfg: dict, note: str = "") -> str:
        d = self.dir / "config_history"
        d.mkdir(exist_ok=True)
        h = cfgmod.config_hash(cfg)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (d / f"{stamp}_{h}.yaml").write_text(f"# {note}\n" + cfgmod.dumps(cfg), encoding="utf-8")
        return h

    def config_history(self) -> list[Path]:
        d = self.dir / "config_history"
        return sorted(d.glob("*.yaml")) if d.exists() else []

    def log(self, line: str) -> None:
        with open(self.dir / "run.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
