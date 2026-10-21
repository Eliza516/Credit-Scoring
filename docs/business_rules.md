# Quy tắc nghiệp vụ – bản để duyệt

Nguồn: slide *DATANITY – Data-Driven Financial Risk Management, Section 5A* (PDF trong repo, 54 trang; `p.N` = số trang).
File cấu hình tương ứng: `config/business.example.yaml`.

Ký hiệu: ✅ lấy từ slide · 🟡 đề xuất của Claude (slide không nêu) · ❓ **cần bạn chốt**.
Mọi ngưỡng chỉ **kích hoạt gợi ý** trên UI; người dùng chấp nhận/bỏ/sửa, hệ thống không tự áp dụng.

## 1. Định nghĩa target (p.20, p.29)
| Mục | Quy tắc | |
|---|---|---|
| Cửa sổ quan sát | ~36 tháng trước T0 | ✅ |
| Performance window | 12–24 tháng, tuỳ sản phẩm, **chốt bằng vintage** | ✅ mặc định 12 ❓ |
| **Bad** | Từng quá hạn 91+ ngày (nhóm nợ 3–5), hoặc charge-off / write-off / phá sản trong performance window | ✅ |
| **Good** | Luôn nhóm 1, không charge-off/write-off | ✅ (max DPD = 0 là 🟡 ❓) |
| **Indeterminate** | Từng 10–90 DPD (nhóm 2) nhưng chưa 91+ → loại khỏi mẫu phát triển | ✅ |
| **Exclusion** | < 6 kỳ thanh toán, đã mất, fraud, nhân viên/VIP, tài khoản chính sách → loại | ✅ |

Lưu ý slide: đổi định nghĩa bad = đổi toàn bộ mô hình; bad rate mẫu dev ảnh hưởng calibration và cutoff.

## 2. Vintage analysis (p.20) – *thêm theo yêu cầu của bạn*
- Dùng để (a) chọn performance window: MOB mà bad rate tích lũy cohort "chín"; (b) kiểm tra ổn định chất lượng cohort theo tháng mở.
- Ngưỡng "chín" (tăng < 0,2 điểm % trong 3 MOB liên tiếp), z-score cohort bất thường 2,0, cỡ cohort tối thiểu 200: 🟡 ❓.

## 3. Chia mẫu (p.34–37)
- Dev 70–80% / Validation 20–30% ✅; thêm tập OOT theo thời gian 🟡 ❓ (số tháng).
- Tối thiểu **2.000 good + 2.000 bad** ✅ (thực tế VN bad rate 10–20% đôi khi chấp nhận).
- Chênh lệch dev–val lớn = overfit ✅ (ngưỡng 5 điểm Gini 🟡 ❓).
- K-fold CV và bootstrap: tổng hợp mọi lần, **không chọn lần đẹp nhất** ✅.
- Mất cân bằng (p.37): các phương án gồm oversample ngẫu nhiên, SMOTE, **undersample**, trọng số mẫu, hoặc không can thiệp. **Không có mặc định – bạn quyết định** (`method: null`, app bắt buộc chọn trước khi huấn luyện). Chỉ áp dụng cho tập train; validation/OOT giữ phân bố thật. UI so sánh các phương án cạnh nhau (Gini, KS, PR-AUC, calibration) để bạn chọn. Nếu đổi bad rate thì cần hiệu chỉnh prior `ln(bad_dev/bad_true)` để PD đúng ✅ (bật/tắt cũng do bạn chọn). ❓

## 4. Phân tầng (segmentation) – dữ liệu gợi ý
Slide chỉ nêu "segmentation basis (for uniform risk profiling)" (p.28). Thiết kế: chế độ `suggest` → hệ thống so sánh bad rate/IV theo biến ứng viên + cây nông (depth 2); gợi ý tách tầng khi: tầng ≥ 10% mẫu, ≥ 500 bad, bad rate lệch ≥ 1,5 lần, Gini tổng tăng ≥ 3 điểm. Toàn bộ là 🟡 ❓; bạn duyệt rồi mới chốt (mặc định không tách).

## 5. Làm sạch dữ liệu (p.32, p.41)
- Loại biến missing > 30% (không mang tính cấu trúc), hằng số, không có tại T0 ✅.
- Missing/"không áp dụng" **≠ 0** → cho bin riêng (mặc định); impute chỉ khi missing ngẫu nhiên; bỏ bản ghi nếu < 5% ✅.
- Outlier: coarse binning (mặc định) hoặc cap/floor P1/P99 ✅. Kiểm tra giá trị bất khả (vd tuổi 150) ✅.
- ❓ Danh sách giá trị đặc biệt (-999, 9999…) và khoảng hợp lệ từng cột.

## 6. Binning & WoE/IV (p.33, p.43)
- ~10 fine bin → gộp coarse bin; mỗi bin ≥ 5% mẫu; bad rate đơn điệu/hợp logic; missing bin riêng ✅. Tối đa 6 bin 🟡 ❓.
- WoE = ln(%Good/%Bad); WoE>0 = rủi ro thấp hơn ✅.
- IV: < 0,02 loại · 0,02–0,10 yếu · 0,10–0,30 trung bình · 0,30–0,50 mạnh · > 0,50 nghi ngờ (kiểm tra rò rỉ) ✅.
- ❓ Biến nào cho phép không đơn điệu (vd tuổi).

## 7. Phễu chọn biến (p.41)
Data quality → tương quan thô |r| > 0,8 → IV & xu hướng bad rate → ổn định dev/val → tương quan sau binning |r| > 0,6 hoặc VIF > 5 → model fit (p > 0,05, dấu hệ số ngược xu hướng) → logic nghiệp vụ. **Kết quả cuối 10–20 biến** ✅. ❓ Biến bắt buộc giữ/loại.

**Tuân thủ (p.24–25):** mỗi biến qua 6 câu hỏi (đồng thuận, thuộc tính được bảo vệ, proxy, nhân quả/danh tiếng, nguồn tin cậy, giải thích được). Loại: giới tính, tôn giáo, hôn nhân, quốc tịch ✅. Cảnh báo proxy: zip, thiết bị, GPS, mạng xã hội ✅. ❓ Bổ sung theo luật tại tổ chức bạn.

## 8. Mô hình hoá (p.41, 46, 51)
- Chính: Logistic trên WoE (diễn giải được, cơ quan quản lý chấp nhận) ✅.
- **Stepwise** (thêm theo yêu cầu): hướng both, vào/ra p = 0,05, tối đa 20 biến, bắt buộc dấu hệ số đúng ✅/🟡; có thể đổi sang AIC/BIC ❓.
- Challenger (chỉ đối chứng, không thay mô hình chính): **Decision Tree** (cây nông, dễ diễn giải) 🟡, **Random Forest** ✅ p.51, **LightGBM** ✅ p.51, kèm SHAP và hiệu chỉnh xác suất (isotonic/Platt) vì ML thường lệch xác suất 🟡 ❓. Cảnh báo nếu ML hơn logistic quá 5 điểm Gini (có thể bỏ sót phi tuyến/tương tác) 🟡 ❓.
- Reject inference (AGB) tuỳ chọn, mặc định tắt ❓.

## 9. Đánh giá (p.47–50)
| Chỉ số | Yếu | Chấp nhận | Tốt | Xuất sắc |
|---|---|---|---|---|
| KS | < 20 | 30–40 | > 40 (mạnh) | application thường 40–60 |
| Gini | < 35% | 35–45% | 45–60% | > 60% |
| AUC | < 0,675 | 0,675–0,725 | 0,725–0,80 | > 0,80 |

✅ Rank-ordering: bad rate giảm đơn điệu qua ~10 tranche A–J. 🟡 ❓ PSI/CSI (slide có nhắc nhưng không nêu ngưỡng): xanh < 0,10, đỏ > 0,25. 🟡 ❓ Sai số calibration cho phép.

### 9b. Chỉ số kiểu ML (bổ sung) 🟡 ❓
Chấm cùng một bộ cho logistic, DT, RF, LightGBM trên validation/OOT:
- **Phân biệt:** ROC-AUC, **PR-AUC** (đáng tin hơn khi bad hiếm; đạt khi ≥ 2× bad rate nền), lift ở decile rủi ro cao nhất (≥ 2,5).
- **Xác suất:** log loss, Brier, ECE (≤ 0,05), đường calibration.
- **Theo ngưỡng quyết định (mặc định theo cutoff điểm):** confusion matrix, precision, recall, F1, specificity, balanced accuracy, MCC.
- **Ổn định:** CV 5 fold (độ lệch chuẩn AUC ≤ 0,02), bootstrap khoảng tin cậy 95% ✅ p.35–36.
- **Overfit:** chênh AUC train–val ≤ 0,05, learning curve.
- **So sánh mô hình:** xếp hạng theo Gini/PR-AUC/KS, permutation importance, partial dependence.
Các ngưỡng trên đều là đề xuất, bạn chốt.

## 10. Bảng điểm & cutoff (p.15, 18, 50)
- Ví dụ slide: 600 điểm = odds 50:1; điểm cao = rủi ro thấp ✅. ❓ Base score, base odds, **PDO**, khoảng điểm.
- Quyết định **Approve / Manual review / Reject** là chính sách (Layer 3), tách khỏi mô hình ✅. ❓ Chiến lược cutoff (approval rate mục tiêu / bad rate chấp nhận / lợi nhuận) và độ rộng vùng xét tay.

## Danh sách ❓ cần bạn chốt (tóm tắt)
1. Performance window (12 hay 24 tháng) và định nghĩa good "sạch".
2. Số tháng OOT; chênh lệch dev–val cho phép.
3. Biến ứng viên phân tầng và các ngưỡng gợi ý tầng.
4. Giá trị đặc biệt, khoảng hợp lệ từng cột, biến cho phép phi đơn điệu.
5. Biến bắt buộc giữ/loại; danh sách biến cấm/cảnh báo theo luật nội bộ.
6. Tiêu chí stepwise (p-value hay AIC/BIC), có dùng regularization không.
7. Ngưỡng PSI/CSI, sai số calibration.
8. Base score / odds / PDO / khoảng điểm.
9. Chiến lược cutoff và độ rộng vùng manual review.
10. Có dùng reject inference không.
11. **Mất cân bằng: oversample hay undersample hay không can thiệp (không có mặc định) + có hiệu chỉnh prior không.**
12. Ngưỡng các chỉ số ML (PR-AUC, lift, ECE, precision/recall tại cutoff) và có dùng DT/RF/LightGBM làm challenger nào.
