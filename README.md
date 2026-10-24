# Credit Scoring App

Ứng dụng web (Streamlit) tự xây mô hình credit scoring: upload → làm sạch → target & vintage → chia Train/Test & phân tầng → binning WoE → chọn biến → huấn luyện (stepwise) → đánh giá → bảng điểm & cutoff → xuất kết quả. Mỗi bước có phân tích + gợi ý theo ngưỡng; bạn duyệt rồi mới sang bước sau.

## Chạy local
```bash
pip install -r requirements-dev.txt
streamlit run app/main.py
pytest
```
Bấm **"Dùng dữ liệu demo"** ở tab 1 để chạy thử ngay (dữ liệu tổng hợp, không phải dữ liệu thật).

## Dữ liệu: nhiều file, file lớn, hồ sơ Lending Club
- **Nhiều file một lúc:** chọn nhiều file ở bước 1; file cùng cấu trúc tự gộp, file khác cấu trúc gán vai trò (dữ liệu chính / performance / bỏ qua).
- **File lớn** (vd `accepted_2007_to_2018Q4.csv.gz`, 2,26 triệu dòng): nhập đường dẫn ở tab *Đường dẫn trên máy chủ*; app đọc theo khối, lọc sớm và lấy mẫu tối đa N dòng (mặc định 300.000).
- **Hồ sơ dữ liệu:** chọn *Lending Club* để tự ánh xạ cột, đọc ngày `Dec-2015`, lấy nhãn từ `loan_status` (Charged Off = bad, Fully Paid = good), lọc khoản đã đáo hạn từ 2010, loại cột chỉ có sau khi vay, chia 70/15/15 theo thời gian. Hồ sơ nằm ở `config/profiles/`.
- Mỗi bước có **dashboard** (bad rate theo biến/nhóm, theo thời gian, phân phối, IV, tương quan, hệ số, điểm…).

## Cấu hình nghiệp vụ
- Mẫu: `config/business.example.yaml` (có ghi nguồn từng quy tắc). Bản thật: sao chép thành `config/business.yaml` (không commit).
- Giải thích quy tắc: `docs/business_rules.md`. Chỉnh tương tác ở trang **Quy tắc & ngưỡng nghiệp vụ**.

## Dùng thử công khai
Xem `docs/deploy.md` (Streamlit Community Cloud, HTTPS sẵn). Đặt `APP_MODE=public` để bật giới hạn upload và banner cảnh báo.
