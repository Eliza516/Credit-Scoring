"""Hướng dẫn & tài liệu."""
from pathlib import Path

import streamlit as st

from app import common

common.init()
st.title("📚 Hướng dẫn & tài liệu")
root = Path(__file__).resolve().parents[2]
t1, t2, t3 = st.tabs(["Cách dùng", "Quy tắc nghiệp vụ & chỉ số", "Quyền riêng tư"])
with t1:
    st.markdown("""
### Quy trình 9 bước (trang *Phân tích & xây mô hình*)
1. **Dữ liệu** – upload CSV/Excel hoặc dùng *dữ liệu demo tổng hợp*; ánh xạ cột (ID, ngày mở, DPD, MOB…).
2. **Làm sạch** – xem tỷ lệ missing/outlier/giá trị đặc biệt; hệ thống *gợi ý* loại biến theo ngưỡng, bạn tick/bỏ.
3. **Target & Vintage** – đường vintage giúp chọn performance window; dựng bad/good/indeterminate/exclusion.
4. **Chia mẫu & Phân tầng** – chia Train/Test (không K-fold); gợi ý tầng từ dữ liệu, bạn quyết định có tách hay không.
5. **Binning** – binning tự động rồi gộp/tách bin thủ công, xem WoE/IV, bad rate train vs test.
6. **Chọn biến** – phễu IV → tương quan → ổn định → VIF → tuân thủ; bạn duyệt danh sách cuối.
7. **Huấn luyện** – *bạn chọn* xử lý mất cân bằng (oversample/undersample/trọng số/không), stepwise, logistic scorecard, mô hình đối chứng DT/RF/LightGBM.
8. **Đánh giá** – AUC/Gini/KS, PR-AUC, precision/recall/F1/MCC, lift, PSI/CSI, so sánh mô hình.
9. **Bảng điểm & xuất** – quy đổi điểm, tranche, cutoff, vùng Approve/Manual/Reject, xuất Excel/PDF/mô hình, chấm điểm file mới.

Mỗi bước lưu kết quả kèm mã phiên bản cấu hình. Chạy lại một bước thì các bước sau được đánh dấu cần chạy lại.

### Trang *Quy tắc & ngưỡng nghiệp vụ*
Chỉnh mọi ngưỡng bằng widget, xem **tác động trực tiếp** lên dữ liệu đã nạp, lưu/tải/so sánh phiên bản `business.yaml`.
""")
with t2:
    p = root / "docs" / "business_rules.md"
    st.markdown(p.read_text(encoding="utf-8") if p.exists() else "Chưa có docs/business_rules.md.")
with t3:
    st.markdown("""
- Mã nguồn nằm trong repo **private**; `data/`, `artifacts/`, `config/business.yaml`, `.env` không được commit.
- Bản dùng thử công khai (`APP_MODE=public`) dùng cấu hình mẫu, giới hạn kích thước upload, và **không nên dùng dữ liệu thật/nhạy cảm**.
- Kết quả trung gian được lưu theo phiên trên máy chủ chạy ứng dụng (thư mục `artifacts/<mã phiên>`).
- Thuộc tính được bảo vệ (giới tính, tôn giáo, hôn nhân, quốc tịch) bị chặn khỏi mô hình; biến proxy (zip, thiết bị, GPS…) bị cảnh báo.
""")
