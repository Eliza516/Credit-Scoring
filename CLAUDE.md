# Credit Scoring App

## Mục tiêu
Ứng dụng web tự xây mô hình credit scoring, pipeline đi từ upload dữ liệu → làm sạch → target & vintage → chia mẫu/phân tầng → binning → chọn biến → huấn luyện (stepwise) → đánh giá → bảng điểm & cutoff → xuất kết quả.
**Mỗi bước người dùng đều xem được kết quả, phân tích + gợi ý theo ngưỡng, và tinh chỉnh theo nghiệp vụ trước khi sang bước sau.**
Có chế độ dùng thử công khai (`APP_MODE=public`) để deploy lên Streamlit Community Cloud.

## Stack
- Python 3.11+ (đã chạy thử 3.13), Streamlit (giao diện)
- pandas, numpy, scikit-learn, optbinning (WoE), lightgbm, statsmodels, imbalanced-learn, shap
- plotly, openpyxl, reportlab (Excel/PDF), pytest

## Cấu trúc thư mục
```
app/
  main.py             # st.navigation: 3 trang
  common.py           # session, chế độ public, invalidate các bước sau
  pages/              # 1_analysis (9 bước, mỗi bước 1 tab) · 2_rules (dashboard ngưỡng) · 3_docs
  steps/              # s1_data … s9_score: UI mỏng của từng bước
core/                 # Logic thuần Python, KHÔNG phụ thuộc Streamlit
  config.py ingest.py cleaning.py target.py performance.py vintage.py split.py imbalance.py segmentation.py
  binning.py selection.py modeling.py evaluation.py scorecard.py export.py store.py demo.py
config/
  business.example.yaml   # Mẫu quy tắc nghiệp vụ (commit) – mỗi giá trị ghi nguồn slide [p.N]/[ĐỀ XUẤT]/[CẦN CHỐT]
  business.yaml           # Cấu hình thật (KHÔNG commit)
docs/business_rules.md    # Quy tắc để duyệt (tiếng Việt)
docs/deploy.md            # Hướng dẫn deploy
tests/  scripts/  data/(không commit)  artifacts/(không commit)
```

## Nguyên tắc bắt buộc
1. **Bảo vệ sở hữu trí tuệ:** repo private. `data/`, `artifacts/`, `config/business.yaml`, `.env`, `.streamlit/secrets.toml` trong `.gitignore`. Không hard-code quy tắc nghiệp vụ – mọi tham số đọc từ cấu hình. Chế độ public KHÔNG đọc `business.yaml`.
2. **Tách logic và giao diện:** xử lý trong `core/`, `app/` chỉ gọi hàm và hiển thị.
3. **Mỗi bước lưu kết quả trung gian** (kèm mã phiên bản cấu hình; không tự xoá) để quay lại sửa rồi chạy lại các bước sau (`core/store.py`, `app/common.invalidate_from`).
4. **Tái lập được:** `seed` cố định trong config, ghi log cấu hình mỗi lần chạy (`artifacts/<phiên>/run.log`, `config_history/`).
5. **Test:** mỗi module `core/` có test; chạy `pytest` trước khi báo xong.
6. **Quyết định nghiệp vụ do người dùng** (định nghĩa bad, chia bin, cutoff, **oversample/undersample – không có mặc định**…) – hỏi, không tự quyết. Ngưỡng chỉ kích hoạt *gợi ý*, không tự áp dụng.
7. Chia mẫu chỉ **Train/Test** (OOT tuỳ chọn), không dùng K-fold/bootstrap. Không dùng chỉ số xác suất (log loss/Brier/ECE).
8. Trao đổi với người dùng bằng tiếng Việt.

## Lộ trình (đã hoàn thành & có test)
- [x] 0. Khung dự án, `.gitignore`, `requirements.txt`, config mẫu, app 3 trang
- [x] 1. Upload & ánh xạ cột
- [x] 2. Làm sạch (gợi ý theo ngưỡng)
- [x] 3. Target & Vintage (chọn performance window; nguồn tuỳ chọn: không dùng / cột có sẵn / upload performance dọc-ngang; ngưỡng DPD chọn được)
- [x] 4. Chia Train/Test & phân tầng gợi ý
- [x] 5. Binning editor (WoE/IV, gộp/tách bin thủ công)
- [x] 6. Chọn biến (phễu + tuân thủ)
- [x] 7. Huấn luyện: mất cân bằng (người dùng chọn), stepwise, logistic, DT/RF/LightGBM đối chứng
- [x] 8. Đánh giá (AUC/Gini/KS + PR-AUC/precision/recall/F1/MCC/lift, PSI/CSI)
- [x] 9. Bảng điểm, tranche, cutoff, xuất Excel/PDF/mô hình, chấm điểm file mới
- [ ] Deploy Streamlit Community Cloud (xem `docs/deploy.md`)
- [ ] Người dùng chốt các mục "CẦN CHỐT" trong `docs/business_rules.md`
