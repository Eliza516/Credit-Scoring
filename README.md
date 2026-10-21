# Credit Scoring App

Ứng dụng web (Streamlit) tự xây mô hình credit scoring: upload → làm sạch → target & vintage → chia Train/Test & phân tầng → binning WoE → chọn biến → huấn luyện (stepwise) → đánh giá → bảng điểm & cutoff → xuất kết quả. Mỗi bước có phân tích + gợi ý theo ngưỡng; bạn duyệt rồi mới sang bước sau.

## Chạy local
```bash
pip install -r requirements-dev.txt
streamlit run app/main.py
pytest
```
Bấm **"Dùng dữ liệu demo"** ở tab 1 để chạy thử ngay (dữ liệu tổng hợp, không phải dữ liệu thật).

## Cấu hình nghiệp vụ
- Mẫu: `config/business.example.yaml` (có ghi nguồn từng quy tắc). Bản thật: sao chép thành `config/business.yaml` (không commit).
- Giải thích quy tắc: `docs/business_rules.md`. Chỉnh tương tác ở trang **Quy tắc & ngưỡng nghiệp vụ**.

## Dùng thử công khai
Xem `docs/deploy.md` (Streamlit Community Cloud, HTTPS sẵn). Đặt `APP_MODE=public` để bật giới hạn upload và banner cảnh báo.
