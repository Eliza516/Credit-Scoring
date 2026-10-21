"""Xuất dữ liệu demo tổng hợp ra CSV: python scripts/make_demo.py demo/demo_loans.csv"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import demo  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo/demo_loans.csv")
out.parent.mkdir(parents=True, exist_ok=True)
demo.make_demo().to_csv(out, index=False)
print("đã ghi", out)
