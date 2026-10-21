# Deploy bản dùng thử công khai (Streamlit Community Cloud)

Repo giữ **private**, app **public** (người dùng thấy web, không thấy code). HTTPS có sẵn (`https://<tên>.streamlit.app`).

## Các bước
1. Đảm bảo nhánh deploy (ví dụ `main`) có `app/main.py`, `requirements.txt`, `packages.txt` (libgomp1 cho LightGBM).
2. Vào https://share.streamlit.io → đăng nhập GitHub → **Create app** → chọn repo `Eliza516/Credit-Scoring` (cấp quyền cho repo private khi được hỏi), nhánh, **Main file path: `app/main.py`**.
3. **Advanced settings**: chọn Python 3.12 (hoặc 3.11–3.13), và thêm Secrets:
   ```toml
   APP_MODE = "public"
   ```
4. Deploy. Lần build đầu mất vài phút (optbinning/ortools/lightgbm/shap).
5. Mở URL ở cửa sổ ẩn danh, bấm **Dùng dữ liệu demo** và chạy hết 9 bước.

## Chế độ public
- Chỉ dùng `config/business.example.yaml` (không đọc `business.yaml`), giới hạn upload theo mục `public_mode` (≤20MB, ≤100k dòng, ≤200 cột), banner cảnh báo không upload dữ liệu nhạy cảm.
- Gói free ~1GB RAM: dữ liệu demo 40k dòng chạy được; hạ `max_rows` nếu bị out-of-memory.
- Kết quả trung gian lưu trong `artifacts/<mã phiên>` trên máy chủ (không tự xoá; ổ đĩa Cloud tạm thời).

## Dự phòng
Nếu thiếu RAM: Hugging Face Spaces (SDK Streamlit hoặc Docker, 16GB), dùng cùng `requirements.txt`.
