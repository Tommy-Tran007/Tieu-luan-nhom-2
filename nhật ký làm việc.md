# Nhật ký Làm việc Dự án

Tệp này tự động lưu vết toàn bộ các phiên làm việc của **Nhóm 02** trong Antigravity IDE theo chuẩn P-D-M-V (*Prompt & Produce – Deconstruct & Read – Mutate & Break – Validate & Refine*).  
- **Tên đề tài:** `PRJ-EXT-01` — Phân tích Biểu hiện Định giá Động (*Dynamic Surge Pricing*) và Dự báo Giá báo API trên Nền tảng Gọi xe Công nghệ (Uber & Lyft Boston, MA)  
- **Nhóm sinh viên thực hiện:** Nhóm 02 — **Trần Thanh Huy** (MSSV: `2521001060`, STT `1`) & **Lê Huy Thắng** (MSSV: `2521001092`, STT `2`) — Lớp `25DDS01`  
- **Phân công & Hệ số đóng góp (Mục 5.4):** Đồng đều **50% / 50%** (`25` phiên `2026-10-01` → `2026-11-28`: **Trần Thanh Huy** `10` phiên độc lập `1, 3, 5, 7, 9, 12, 13, 15, 17, 19`, **Lê Huy Thắng** `11` phiên độc lập `2, 4, 6, 8, 10, 11, 14, 16, 18, 20, 22`, và `4` phiên đồng thực hiện `21, 23, 24, 25`; $\alpha_1 = 1.00$, $\alpha_2 = 1.00$; $\sum \alpha_i = 2.00$).

---

## [2026-11-28 10:55] - Phiên 25: Tinh Gọn Tiểu Luận Về Đúng `45` Trang (Giữ Nguyên Định Dạng), Chuẩn Hóa Cấu Trúc Nộp `25210010XX_Nhom02_TieuLuan/` & Chốt Đồng Bộ Hồ Sơ

- **Thành viên phụ trách chính:** **Đồng thực hiện — Trần Thanh Huy (`2521001060`) & Lê Huy Thắng (`2521001092`)**
- **Giai đoạn:** Hoàn thiện ấn phẩm cuối & chuẩn hóa đóng gói bàn giao (`GĐ8`; khung thực thi sáng `2026-11-28 09:35 → 10:55`).
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`pandas 3.1.1`, `pyarrow 26.0.1`, `numpy 2.4.6`, `scipy 1.17.1`, `statsmodels 0.15.0`, `scikit-learn 1.9.1`, `pytest 9.2.0`, `ruff 0.17.6`, `pymupdf`) | `Quarto 1.10.19` (`Pandoc 3.10.0`, `Typst 0.15.1`)
- **Mục tiêu phiên:**
  - Rút gọn diễn đạt câu chữ có chọn lọc tại nguồn sinh Tiểu luận để giảm tổng độ dài bản PDF từ **`46` trang xuống đúng `45` trang** (`43/45` nội dung ngoài 2 trang bìa, `27` trang Chương 1–5) mà **tuyệt đối không thay đổi định dạng** (khổ A4, lề `2-2-3-2 cm`, font `Times New Roman`/`Consolas`, cỡ chữ, giãn dòng, bảng biểu, biểu đồ, mục lục) và bảo toàn nguyên vẹn toàn bộ số liệu, phát hiện cốt lõi.
  - Chuẩn hóa tên thư mục gốc theo đúng định dạng nộp bài `25210010XX_Nhom02_TieuLuan/`, cập nhật thống nhất trạng thái đồng thực hiện của `Phiên 23`, đồng bộ `README.md` và tái xuất trọn bộ hai hồ sơ `2521001060` & `2521001092` (PDF rời, thư mục làm việc chuẩn `128` tệp và tệp nén `.zip`).
- **Nội dung đã hoàn thành:**
  - **1. Tinh gọn văn bản về đúng `45` trang & khóa đối soát Chương 3:**
    + [scripts/thesis_content.py](scripts/thesis_content.py): Cô đọng các câu diễn đạt dài/trùng lặp tại `Mục 3.4–3.6` (không xóa bảng, hình hay số liệu) kéo tổng số trang PDF từ `46` xuống đúng **`45` trang**; đồng thời hiệu chỉnh chính xác số bậc surge thực tế trên `1.0x` tại `Mục 3.4` (`6 bậc 1.25x–3.00x`, khớp 6 mức `{1.25, 1.50, 1.75, 2.00, 2.50, 3.00}` trong `master_rideshare.parquet`) và đọc động `max_tost_010_p` tại `Mục 3.6` (`TOST ±$0.10: p ≤ 0.0004`, khớp `Bảng 3.6` ngay phía trên).
    + [scripts/thesis_content.py](scripts/thesis_content.py), [scripts/build_final_thesis.py](scripts/build_final_thesis.py) & [scripts/check_thesis_audit.py](scripts/check_thesis_audit.py): Tham số hóa `student_id` cho sơ đồ thư mục và phần giới thiệu Phụ lục A (`25210010XX_Nhom02_TieuLuan/`), đồng thời chuyển sang đếm động số phiên độc lập/đồng thực hiện từ `nhật ký làm việc.md` (`10 + 11 + 4`) cho Phụ lục A và Phụ lục B.
  - **2. Đồng bộ tài liệu dự án & nhật ký:**
    + [README.md](README.md), [dashboard/README.md](dashboard/README.md), [dashboard/app.py](dashboard/app.py): Chuẩn hóa tên thư mục gốc dự án sang `25210010XX_Nhom02_TieuLuan/`, cập nhật thông số ấn phẩm (`45` trang tổng / `43/45` ngoài bìa / `27` Ch.1–5; `187.1 KiB` QMD → `9.35 MiB` PDF / `8.59 MiB` DOCX), đồng nhất làm tròn điểm cắt Hạng sang SUV (`0.85 mi`, `0.605 mi < 1.0`) khớp `Bảng 4.8` & `Hình 4.8`, và đồng bộ phân công `25` phiên (`10 + 11 + 4`).
    + Cập nhật thống nhất `Phiên 23` thành đồng thực hiện và bổ sung `Phiên 25` nguyên văn trên cả 3 tệp nhật ký (`nhật ký làm việc.md`, bản cá nhân Trần Thanh Huy và bản cá nhân Lê Huy Thắng).
  - **3. Tái xuất bản & Đóng gói chuẩn nộp bài (`2521001060` & `2521001092`):**
    + Biên dịch nguyên bản từ Quarto/Typst sang `2521001060_Nhom02_TieuLuan.pdf` và `2521001092_Nhom02_TieuLuan.pdf` (đều đúng **`45` trang**, chỉ khác nhau duy nhất tại trang `39` ở tên thư mục gốc MSSV), đồng bộ bản dự phòng `.docx`, build đầy đủ hai thư mục chuẩn `2521001060_Nhom02_TieuLuan/` & `2521001092_Nhom02_TieuLuan/` (`128` tệp sạch, bảo toàn EOL CRLF, `0` cache, `0` liên kết gãy) cùng hai gói `2521001060_Nhom02_MaNguon.zip` & `2521001092_Nhom02_MaNguon.zip` trùng khớp SHA-256 100%.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Rút gọn diễn đạt câu chữ ở nguồn tạo tiểu luận để giảm tổng số trang từ 46 xuống đúng 45 trang mà tuyệt đối không thay đổi định dạng, giữ nguyên nội dung/số liệu và liên kết giữa toàn bộ tệp nộp, cập nhật Phiên 23 đồng thực hiện và build lại trọn bộ định dạng chuẩn 25210010XX_Nhom02_TieuLuan/."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Khi rút gọn câu chữ Chương 3, rà soát phát hiện và chặn kịp hai sai lệch ngữ nghĩa phát sinh từ việc gộp câu (`7` vs `6` bậc surge trên `1.0x` và gộp `p < 0.0001` trái với `p = 0.0004` tại `Bảng 3.6` Hàng 4) bằng cách tính động `max_tost_010_p`; đồng thời phát hiện và xóa tệp cache ẩn `reports/.quarto/typst/available-fonts.json` sau khi render bản `1092` để giữ đúng `128` tệp sạch cho cả hai gói.
  - *Xác minh tự động:* `check_thesis_audit.py` **100% PASS** trên cả `2521001060` và `2521001092` (`45` trang, `27` trang Ch.1–5, `17/17` hình `≥ 300 DPI`, `23/23` APA 7th, `0` lỗi TOC/font/trang thưa); `ruff check` & `ruff format --check` **0 lỗi (`60/60` tệp)**; `dashboard.validation` **18/18 PASS**; `pytest dashboard/tests` **40/40 PASS**; đối soát SHA-256 `100%` giữa PDF rời, thư mục nộp và `.zip`.
  - *Quyết định kiến trúc:* Giữ bất biến hoàn toàn tham số dàn trang Typst/Quarto (chỉ tối ưu câu chữ ở nguồn nội dung `thesis_content.py`), đọc động toàn bộ thống kê phiên làm việc từ `nhật ký làm việc.md` và giữ quy trình đóng gói nhị phân bảo toàn nguyên vẹn định dạng dòng (`CRLF`) cho cả hai mã số sinh viên.
- **Tồn đọng & Bước tiếp theo:**
  - Hoàn tất toàn bộ bộ sản phẩm nộp bài chuẩn `45` trang cho cả hai thành viên (`2521001060` và `2521001092`), không còn tồn đọng kỹ thuật.

---

## [2026-11-28 03:20] - Phiên 24: Kiểm Toán Toàn Diện Theo Rubric `projects.pdf`, Đồng Bộ OLS `nobs` / `distance_bucket` / Typst / Notebooks / Dashboard & Chốt Gói Nộp Cuối

- **Thành viên phụ trách chính:** **Đồng thực hiện — Trần Thanh Huy (`2521001060`) & Lê Huy Thắng (`2521001092`)**
- **Giai đoạn:** Kiểm toán toàn diện độc lập theo rubric `projects.pdf`, khắc phục triệt để các sai lệch đối soát cuối cùng & tái lập đóng gói nghiệm thu (`GĐ8`).
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`pandas 3.1.1`, `pyarrow 26.0.1`, `numpy 2.4.6`, `scipy 1.17.1`, `statsmodels 0.15.0`, `scikit-learn 1.9.1`, `pytest 9.2.0`, `ruff 0.17.6`, `pymupdf`, `nbclient`) | `Quarto 1.10.19` (`Pandoc 3.10.0`, `Typst 0.15.1`)
- **Mục tiêu phiên:**
  - Đối chiếu độc lập toàn bộ mã nguồn, báo cáo Tiểu luận, `4` notebooks, Dashboard, `README.md`, `nhật ký làm việc.md` và cả `2` gói nộp `.zip` (`2521001060` & `2521001092`) với từng tiêu chí trong `projects.pdf`.
  - Đồng bộ chính xác cỡ mẫu `nobs` thực tế của các hồi quy OLS đa biến có biến `distance` (`wm_dist`: `139,555` / `2,794` cụm; `wm_obs_dist`: `119,903` / `2,793` cụm; `H4c`: `95,925`), bổ sung nhóm `5. Khuyết distance (NaN thô)` (`n = 3,830`) vào `stage6_segment_errors.csv` để đủ `46,573/46,573` dòng Test trên cả `4` chiều, sửa các lỗi văn phong/APA 7th/typography Typst trong Tiểu luận, cập nhật đồng bộ toàn bộ Markdown trong `4` notebooks, loại bỏ các giá trị fallback/chuỗi tĩnh cũ trong Dashboard, hoàn thiện các mục Template `§8.2` còn thiếu ở `Phiên 23` và sửa cơ chế đóng gói `2521001092_Nhom02_MaNguon.zip` sạch 100% liên kết.
- **Nội dung đã hoàn thành:**
  - **1. Đồng bộ cỡ mẫu OLS `nobs` & phân rã `distance_bucket` đủ `46,573` dòng Test:**
    + [src/evaluation.py](src/evaluation.py): Bổ sung hạng mục `"5. Khuyết distance (NaN thô)"` (`3,830` dòng Test) cho `distance_bucket` trong `run_multidimensional_segment_errors()`, chạy lại `python main.py --stage evaluate` xuất [reports/tables/stage6_segment_errors.csv](reports/tables/stage6_segment_errors.csv) khớp đủ `46,573` dòng ở cả `4/4` chiều phân rã.
    + [scripts/thesis_v5_verification.py](scripts/thesis_v5_verification.py) & [scripts/prepare_thesis_figures.py](scripts/prepare_thesis_figures.py): Xuất trực tiếp `n_obs`, `n_loc_hr_clusters`, `n_hour_clusters` và các khóa `n_valid_rain_dist_obs` (`139,555`, `2,794` cụm, `329` giờ; sửa bậc tự do `n_dist_hr - 1`), `n_valid_rain_and_surge_dist_obs` (`119,903`, `2,793` cụm, `329` giờ) vào [reports/results/thesis_v5_verification.json](reports/results/thesis_v5_verification.json) và cập nhật `Hình 3.6A` (`thesis_fig_3_5_hypotheses_clean.png`).
  - **2. Khắc phục triệt để lỗi học thuật, APA 7th & Typst trong Tiểu luận:**
    + [scripts/thesis_content.py](scripts/thesis_content.py) & [scripts/build_final_thesis.py](scripts/build_final_thesis.py): Đồng bộ `Mục 3.6` & `Bảng 3.6` về `119,903` và `139,555`; sửa dải lặp `"14.8%–14.8%"` (`tr. 26` / `p.35`) bằng `_fmt_pct_range()` và dải `0.0293–0.0152` ở `Bảng 5.2`; đọc động tổng XAI Thời tiết & Giờ EST (`0.02%` ở M7, `0.00%` ở M2) khớp `Bảng 4.9`; hiệu chỉnh chi tiết sai số `> $15` chế độ Uber/Lyft (`57/58` > `$15`, `0/18,440 > $16` ở Lyft, Kurtosis `29.46` vs `3.47`); đổi `(EC '17)` / `(IMC '15)` sang `(EC 2017)` / `(IMC 2015)` tránh nháy đơn mở `‘`; chuyển toàn bộ `6` ký tự `∈` sang chế độ toán học Typst (`0` glyph fallback `LibertinusSerif-Regular`); và bổ sung tiêu đề `Bảng B.3:` phía trên kèm `Nguồn:` phía dưới cho bảng đối soát trong `Phụ lục B.3` (giữ nguyên văn 100% `Phiên 12`).
  - **3. Đồng bộ `4` Notebooks, Dashboard, `README.md`, Nhật ký & Gói nộp `.zip`:**
    + Cập nhật toàn bộ cell Markdown còn số liệu cũ trên [notebooks/03_Giai_Doan_3_Data_Prep_EDA_Visualization.ipynb](notebooks/03_Giai_Doan_3_Data_Prep_EDA_Visualization.ipynb) → [notebooks/06_Evaluation_Interpretation_Policy.ipynb](notebooks/06_Evaluation_Interpretation_Policy.ipynb) (`Python 3.11.9 | pandas 3.1.1`, hệ số OLS, siêu tham số M4/M7, `R3 88.85%` / `240,689`, thí nghiệm R4 `price_per_mile` `MAE = $0.8030, MAPE = 6.29%, R² = 0.9176`, chẩn đoán phần dư, `Ex-Ante`, `CQR`) và tái thực thi `4/4` notebook qua `nbclient`; đồng thời chỉnh câu chữ mục R4 `price_per_mile` trong [scripts/thesis_content.py](scripts/thesis_content.py) khớp bản chất số liệu thực nghiệm.
    + Cập nhật [dashboard/app.py](dashboard/app.py), [dashboard/components/tab_overview.py](dashboard/components/tab_overview.py) (đọc động `Ex-Ante M7 R² = 0.9286` thay cho `"0.9292"`), [dashboard/components/tab_models_rq3.py](dashboard/components/tab_models_rq3.py), [dashboard/components/tab_tariff_rq2.py](dashboard/components/tab_tariff_rq2.py), [dashboard/components/tab_simulator_rq4.py](dashboard/components/tab_simulator_rq4.py), [dashboard/charts.py](dashboard/charts.py), [dashboard/export_static_html.py](dashboard/export_static_html.py) và [dashboard/tests/test_business_logic.py](dashboard/tests/test_business_logic.py); tái xuất [reports/dashboard_interactive.html](reports/dashboard_interactive.html).
    + Hoàn thiện đủ 2 trường Template `§8.2` cho `Phiên 23` (và chuẩn hóa đề mục `Phiên 8`), bổ sung `Phiên 24` đồng đều 50/50 ở cả 3 tệp nhật ký (gỡ các liên kết lịch sử `src/` trước Phiên 22 trên hai tệp cá nhân để đạt `0` liên kết gãy), đối soát sạch mọi số liệu trong [README.md](README.md) (`119,903`, `95,925`, `87,733`, tách `CI_HC3` vs `CI_cl`, `XAI 14 nhóm vs 6 cụm`, `CQR`), hỗ trợ `--student-id` trong [scripts/build_final_thesis.py](scripts/build_final_thesis.py) & [scripts/check_thesis_audit.py](scripts/check_thesis_audit.py) để biên dịch nguyên bản `.qmd/.pdf/.docx` cho cả `2521001060` và `2521001092`, tái chạy trọn `python main.py --stage all` từ `data/raw/*`, nén lại sạch hai gói `2521001060_Nhom02_MaNguon.zip` và `2521001092_Nhom02_MaNguon.zip` (không biến đổi hồi tố `Phiên 1–20`, không lặp tên tệp ở `L76` và `0` liên kết gãy ở cả 2 gói) kèm kiểm thử khói trên thư mục giải nén sạch.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Kiểm tra toàn diện theo rubric projects.pdf một cách độc lập, sửa triệt để sai lệch cỡ mẫu OLS nobs, khuyết nhóm NaN trên distance_bucket, lỗi typography/APA/XAI trong Tiểu luận, markdown cũ ở 4 notebooks, fallback/hardcode ở Dashboard, hoàn thiện cấu trúc Template 8.2 nhật ký và sửa cơ chế đóng gói 2521001092 rồi kiểm thử khói sạch cả 2 gói."
  - *Kinh nghiệm bóc tách code & Mutate/Break:*
    1. **Cắt nhóm ngầm của `pd.cut()` trên `distance_bucket`:** `pd.cut()` tạo `NaN` cho `3,830` dòng khuyết `distance` và bị `groupby(..., observed=False)` bỏ qua lặng lẽ, làm tổng `distance_bucket` chỉ đạt `42,743` thay vì `46,573` → bổ sung nhãn phân loại tường minh `"5. Khuyết distance (NaN thô)"` giúp đối soát `100.00%` tập Test.
    2. **Phân biệt cỡ mẫu lọc `is_raining.notna()` vs `nobs` OLS có `distance`:** Các mô hình kiểm soát đa biến tự động loại `12,488` / `10,706` dòng khuyết `distance` trước khi fit (`152,043 → 139,555` và `130,609 → 119,903`), do đó mọi bảng/hình/văn bản cần đọc trực tiếp `m_fit.nobs` thay vì dùng biến đếm trước hồi quy.
    3. **Smartquote & font fallback trong Typst:** Pandoc/Typst tự đổi `'17` thành nháy mở `‘17` (`U+2018`) và rơi ký tự `∈` sang `LibertinusSerif-Regular` → chuẩn hóa năm đầy đủ `(EC 2017)` / `(IMC 2015)` và chuyển `∈` vào chế độ toán học `$in$`.
    4. **Đóng gói `1092` an toàn không phá lịch sử nhật ký:** Không thay chuỗi `1060 -> 1092` trên các phiên lịch sử `1–20`; chỉ chuyển các liên kết báo cáo cũ `[reports/2521001060_...](...)` sang đường dẫn lịch sử dạng mã nội dòng `` `reports/2521001060_...` `` trong gói `1092` để vừa bảo toàn lịch sử nguyên gốc vừa bảo đảm `0` liên kết gãy.
  - *Quyết định kiến trúc:* Khóa nguyên tắc đối soát `100%` số liệu trực tiếp từ `m_fit.nobs`, JSON/CSV kết quả và `summary` động trên cả Tiểu luận, Notebooks, Dashboard lẫn `README.md`; áp dụng quy trình kiểm toán hậu đóng gói trên thư mục giải nén sạch cho cả hai mã số sinh viên `2521001060` và `2521001092`.
- **Tồn đọng & Bước tiếp theo:**
  - Không còn tồn đọng kỹ thuật hoặc sai lệch rubric; toàn bộ hồ sơ bàn giao (`2521001060` và `2521001092`) đã vượt qua đầy đủ các cổng kiểm định tự động (`ruff`, `dashboard.validation` `18/18`, `pytest` `40/40`, `check_thesis_audit.py`, `4/4` notebooks và smoke test giải nén sạch) và sẵn sàng nộp bài.

---

## [2026-11-28 02:35] - Phiên 23: Khử Triệt Để Rò Rỉ `distance` (`RouteDistanceImputer`), Chuẩn Hóa Mục Tiêu SMART (`T` & `≥ 50%` RAM), Khóa Giới Hạn `≤ 45` Trang Ngoài Bìa & Tái Lập Chốt Gói Nộp

- **Thành viên phụ trách chính:** **Đồng thực hiện — Trần Thanh Huy (`2521001060`) & Lê Huy Thắng (`2521001092`)**
- **Giai đoạn:** Khắc phục phản biện rò rỉ dữ liệu & chuẩn hóa báo cáo/hồ sơ nghiệm thu (`GĐ8`; sau khi rà soát chéo toàn bộ kết quả Phiên 22).
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`pandas 3.1.1`, `pyarrow 26.0.1`, `numpy 2.4.6`, `scipy 1.17.1`, `statsmodels 0.15.0`, `scikit-learn 1.9.1`, `joblib 1.6.0`, `matplotlib 3.11.2`, `seaborn 0.13.2`, `pytest 9.2.0`, `ruff 0.17.6`) | `Quarto 1.10.19` (`Pandoc 3.10.0`, `Typst 0.15.1`)
- **Mục tiêu phiên:**
  - Chặn triệt để `100%` rò rỉ dữ liệu ở biến `distance` trước khi chia train/test: loại bỏ trung vị theo tuyến toàn cục khi khôi phục quy mô bậc 10 và ngừng điền khuyết `22,262` dòng thiếu `distance` trên bảng `master_rideshare.parquet`; đóng gói `RouteDistanceImputer` strictly-on-train ngay đầu `sklearn.pipeline.Pipeline` cho toàn bộ các mô hình (`M1–M7`, `Ex-Ante`, `Surge Classifier`, `Quantile CQR`).
  - Đồng bộ hóa động siêu tham số `M7` từ `models/best_model.joblib` và bật `preserve_decimals=2` trong kiểm chứng độ nhạy `price` (`scripts/thesis_v5_verification.py`, `scripts/prepare_thesis_figures.py`); bổ sung yếu tố Time-bound (`T`) cho cả 5 mục tiêu SMART, giải minh tường minh mức tiết kiệm RAM `64.37%` (CSV → Parquet) vs `37.75%` (bước downcast hậu ghép nối), sửa định dạng APA 7th (`SciPy.`, thứ tự chữ cái) và khóa cứng ngưỡng `≤ 45` trang nội dung không tính 2 trang bìa trong `scripts/check_thesis_audit.py`.
  - Tái chạy toàn bộ pipeline `main.py --stage all`, `thesis_v5_verification.py`, `prepare_thesis_figures.py`, `export_static_html.py`, `4/4` notebooks (`03 → 06`), xuất bản Quarto `.qmd → Typst PDF` + `.docx`, cập nhật đồng bộ `dashboard/tests/`, `README.md`, `nhật ký làm việc.md` và kiểm định khói hai gói nộp `.zip`.
- **Nội dung đã hoàn thành:**
  - **1. Kiến trúc Zero-Leakage `RouteDistanceImputer` & bảo vệ `NaN` toàn pipeline:**
    + [src/clean_raw_data.py](src/clean_raw_data.py): Thay `route_ref` toàn cục bằng hằng số vật lý cố định `2.15` dặm khi khôi phục bậc `10^m` (`11,262` dòng `distance` được hiệu chỉnh), đồng thời bảo toàn nguyên trạng `22,262` giá trị `distance = NaN` trong [data/processed/master_rideshare.parquet](data/processed/master_rideshare.parquet) (`distance_imputed_by_route_median = 0`).
    + [src/features.py](src/features.py), [src/stage5_modeling.py](src/stage5_modeling.py), [src/evaluation.py](src/evaluation.py), [src/interpretation.py](src/interpretation.py), [src/stage3_pipeline_and_audit.py](src/stage3_pipeline_and_audit.py), [src/stats_tests.py](src/stats_tests.py): Xây dựng `RouteDistanceImputer` (học `route_medians_` & `global_median_` chỉ trên `X_train`, tự động tính lại `log_distance`, `distance_x_surge`, `distance_x_lyft_surge_excess`, giữ nguyên `category` dtypes và đặt `feature_names_in_`/`get_feature_names_out`), gắn vào toàn bộ 4 nhánh `build_stage4_preprocessors()`, `EconomicInteractionPreprocessor`, `M7_ExAnte`, 4 bộ `Surge Classifier`, 3 ống `Quantile` và `Grouped Permutation Importance`; bổ sung bộ lọc `distance.notna()` trước `np.polyfit`, `linregress`, `sm.OLS`, `het_breuschpagan` và trong [dashboard/config.py](dashboard/config.py), [dashboard/validation.py](dashboard/validation.py), [dashboard/business_logic.py](dashboard/business_logic.py).
    + Kết quả sạch rò rỉ mới (`n_test = 46,573`): **`M7`** `CV RMSE = $1.7656`, `Test R² = 0.9626`, `RMSE = $1.7830`, `MAE = $1.1338`, `MAPE = 8.19%`; **`M2 OLS`** đạt Full Column Rank SVD `82/82` (`cond = 122.42`, `M1` `49/49` `cond = 29.27`), `Test R² = 0.9539`, `RMSE = $1.9802`, `MAPE = 9.33%`; **`UberX vs. Lyft`** `d* = 3.21` dặm (Đặc tả A) / `5.17` dặm (Đặc tả B); **`Cap 2.00x`** `$5.53` (`-18.84%`, Proxy `-0.018%`); **`Cap 1.75x`** `$4.61` (`-11.64%`, Proxy `-0.203%`); **`Monotonic Split-CQR`** `PICP = 79.89%` (`[78.62%, 81.99%]`, `0` cắt ngược); **`route_group_split`** phản ánh trung thực chi phí không rò rỉ trên 15 tuyến mới (`M7 R² = 0.9253, MAPE = 11.50%`; trên `42,823` dòng nguyên bản có `distance`: `R² = 0.9538, MAPE = 9.37%`).
  - **2. Đồng bộ kiểm chứng, SMART, APA 7th & giới hạn `≤ 45` trang nội dung:**
    + [scripts/thesis_v5_verification.py](scripts/thesis_v5_verification.py) & [scripts/prepare_thesis_figures.py](scripts/prepare_thesis_figures.py): Đọc động siêu tham số `M7` từ [models/best_model.joblib](models/best_model.joblib), bật `preserve_decimals=2` cho `distance` & `price`, lọc `wm_dist`/`wm_obs_dist`/`wm_hv` khớp `groups` trong `cov_type="cluster"` & WCR, và cập nhật động nhãn biểu đồ `Hình 4.3` & `Hình 4.8`.
    + [scripts/thesis_content.py](scripts/thesis_content.py), [scripts/build_final_thesis.py](scripts/build_final_thesis.py) & [scripts/check_thesis_audit.py](scripts/check_thesis_audit.py): Bổ sung mốc `Time-bound (T)` cho cả 5 mục tiêu SMART; sửa diễn giải tiết kiệm RAM (`64.37%` CSV → Parquet vs `37.75%` ép kiểu hậu ghép nối) không còn gán nhầm nhãn `≥ 50%` cho `37.75%`; sửa APA 7th (`SciPy.` cho McKinney 2010 & Seabold & Perktold 2010, xếp `Castillo et al., 2017; Hall et al., 2015`); đọc động phiên bản `pandas 3.1.1 / pyarrow 26.0.1` và `parquet_disk_mb`; thêm ràng buộc cứng `n_pages - 2 <= 45` vào `CHECK 02` (kết xuất thực tế **`46` trang** tổng = **`44/45` trang nội dung ngoài 2 trang bìa**, trong đó Chương 1–5 chiếm `28` trang).
  - **3. Đồng bộ Dashboard, Notebooks, `README.md` & đóng gói kiểm định khói:**
    + Cập nhật [dashboard/export_static_html.py](dashboard/export_static_html.py), [dashboard/README.md](dashboard/README.md) và [dashboard/tests/test_business_logic.py](dashboard/tests/test_business_logic.py), [dashboard/tests/test_model.py](dashboard/tests/test_model.py), [dashboard/tests/test_export_and_perf.py](dashboard/tests/test_export_and_perf.py); chạy lại `4/4` notebooks `03 → 06` bằng `nbclient` (`ipykernel`), chuẩn hóa `ruff format` & `ruff check` sạch lỗi trên toàn bộ `50` tệp `.py`/`.ipynb` trong gói nộp chuẩn (`46 .py + 4 .ipynb`), đồng bộ [README.md](README.md) và đóng gói lại sạch hai bộ hồ sơ `2521001060` & `2521001092`.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Khử triệt để rò rỉ dữ liệu ở biến distance trước khi chia train/test bằng RouteDistanceImputer strictly-on-train trong Pipeline, đồng bộ động siêu tham số M7 và preserve_decimals=2, bổ sung Time-bound và sửa tuyên bố RAM SMART, khóa cứng tối đa 45 trang nội dung ngoài bìa, tái lập toàn trình 4 notebooks/PDF/Dashboard và kiểm định khói hai gói ZIP."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* (3 lỗi phát sinh thực tế đã bóc tách và xử lý triệt để khi mở khóa `NaN` của `distance`)
    1. **Lệch `NaN` chéo trong `Grouped Permutation Importance`:** Hoán vị nhóm `distance & log_distance` khiến dòng vốn thiếu `distance` nhận giá trị hữu hạn nhưng vẫn giữ `distance_x_lyft_surge_excess = NaN`, làm `LinearRegression` của `M2` lỗi `Input X contains NaN` → sửa cả `RouteDistanceImputer.transform()` (tự vá mọi cột dẫn xuất chưa hữu hạn) lẫn tiền chuẩn hóa `X_sub` trong `src/interpretation.py`.
    2. **Lệch độ dài vector `groups` trong `statsmodels cov_type="cluster"`:** `smf.ols` tự loại dòng `NaN` của `distance` nhưng `cov_kwds={"groups": ...}` nhận Series nguyên vẹn → lọc trước `wm_dist`, `wm_obs_dist`, `wm_hv`, `wm_obs_clean_any` theo `distance.notna()`.
    3. **Lọc mặc định Dashboard làm rơi `22,262` dòng khuyết `distance`:** Cập nhật `apply_global_filters()` trong `dashboard/business_logic.py` để giữ `distance.isna()` ở dải mặc định `[0.02, 7.86]` tương tự quy tắc CRIT-03 cho `surge_multiplier`, giữ đủ `270,898` quan sát.
  - *Xác minh tự động:* `dashboard.validation` `18/18 PASS`; `pytest dashboard/tests -v` `40/40 PASS`; `check_thesis_audit.py` `100% PASS` (`46` trang = `44/45` nội dung ngoài bìa, `Typst 0.15.1`, `17/17` ảnh `≥ 300 DPI`, `23/23` APA 7th); `ruff check` & `ruff format --check` `0` lỗi trên `50` tệp `.py`/`.ipynb` (cả tại chỗ và trên bản giải nén sạch của hai `.zip`).
  - *Quyết định kiến trúc:* Khóa triệt để zero-leakage bằng `RouteDistanceImputer` strictly-on-train bên trong `Pipeline`, bảo toàn nguyên trạng `NaN` ở `distance` và `surge` trên bảng `master_rideshare.parquet`, tách bạch mức tiết kiệm RAM `64.37%` (CSV → Parquet) vs `37.75%` (ép kiểu hậu ghép nối) trong SMART, và áp dụng cổng kiểm toán cứng `n_pages - 2 <= 45`.
- **Tồn đọng & Bước tiếp theo:**
  - Tiến hành kiểm toán toàn diện độc lập đối chiếu từng mục rubric `projects.pdf` (cỡ mẫu `nobs` OLS thực tế, phân rã `distance_bucket` đủ `46,573` dòng, typography Typst/APA 7th, markdown `4` notebooks, fallback Dashboard và cơ chế đóng gói `2521001092`) trong Phiên 24 trước khi chốt bàn giao.

---

## [2026-11-28 00:22] - Phiên 22: Đại Tu Kỹ Thuật–Toán Học Toàn Diện, Tách `scripts/`, Siết Chuẩn PEP 8 (`≤ 79`), Tái Lập Toàn Trình & Chốt Hồ Sơ Nộp Bài

- **Thành viên phụ trách chính:** **Lê Huy Thắng** (MSSV: `2521001092`)
- **Giai đoạn:** Rà soát phản biện sâu, đại tu kỹ thuật–toán học, tái cấu trúc thư mục & tái lập toàn trình nghiệm thu cuối (`GĐ8`; khung thực thi liên tục `2026-11-27 17:55 → 2026-11-28 00:22`).
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`pandas 3.0.6`, `numpy 2.4.6`, `scipy 1.17.1`, `statsmodels 0.15.0`, `scikit-learn 1.9.1`, `pytest`, `pymupdf`, `ruff`) | `Quarto 1.10.19` (`Pandoc 3.10.0`, `Typst 0.15.1`)
- **Mục tiêu phiên:**
  - Xử lý triệt để toàn bộ tồn đọng kỹ thuật và toán học phát hiện khi kiểm toán sâu hồ sơ sau Phiên 21: đồng bộ `--stage clean` vs `--stage eda/all`, bảo toàn 2 chữ số thập phân cự ly `≤ 8.0` dặm, khử khuyết hạng đồ thị hai phía $K_{6,6}$ cho `M1`/`M2` đạt Full Column Rank SVD, sửa dấu `Rank-Biserial r`, nâng cấp mô phỏng `Surge Cap` sang hàm cung nhạy giá $\varepsilon_S \in \{0.30, 0.60\}$ kèm dư cầu đầy đủ `Σ max(0, Q_D − Q_realized)`, hiệu chỉnh Switchback về đúng `18` ngày có dữ liệu thực, sửa công thức JS Offline HTML và đồng bộ toàn bộ Dashboard.
  - Tách `5` script kiểm chứng mở rộng & xuất bản Tiểu luận ra khỏi `src/` sang thư mục độc lập `scripts/`, siết chặt cấu hình PEP 8 tại `pyproject.toml` (`line-length = 79`, gỡ ngoại lệ `E501`, bao phủ cả `scripts/` và `notebooks/`) và chạy `ruff format` đồng loạt trên toàn bộ `60` tệp `.py`/`.ipynb`.
  - Tái thực thi toàn trình từ dữ liệu gốc chỉ-đọc `data/raw/*` trên một máy tổng hợp (`Python 3.11.9`) — gồm `main.py --stage all`, `thesis_v5_verification.py`, `prepare_thesis_figures.py`, `export_static_html.py`, trọn bộ `4` notebooks `03 → 06` và xuất bản Quarto `.qmd → Typst PDF` chuẩn UFM — rồi rà soát tính xác thực nhật ký, đóng gói & kiểm định khói hai bộ hồ sơ `2521001060` & `2521001092`.
- **Nội dung đã hoàn thành:**
  - **1. Tái cấu trúc `src/` → `scripts/` & Siết chuẩn PEP 8 `≤ 79` ký tự (`27/11 17:55–18:56`):**
    + Khởi tạo gói [scripts/__init__.py](scripts/__init__.py) và di dời `5` script xuất bản/kiểm chứng ([scripts/thesis_v5_verification.py](scripts/thesis_v5_verification.py), [scripts/prepare_thesis_figures.py](scripts/prepare_thesis_figures.py), [scripts/thesis_content.py](scripts/thesis_content.py), [scripts/build_final_thesis.py](scripts/build_final_thesis.py), [scripts/check_thesis_audit.py](scripts/check_thesis_audit.py)) ra khỏi `src/` để giữ `src/` thuần `11` module lõi pipeline DS.
    + Cập nhật [pyproject.toml](pyproject.toml) (`line-length = 79`, gỡ `ignore = ["E501"]`, mở rộng phạm vi sang `scripts/` và `notebooks/`) và chạy `ruff check --fix` + `ruff format` đồng loạt trên toàn bộ `60` tệp `.py`/`.ipynb` ([main.py](main.py), `src/`, `scripts/`, `dashboard/`, `notebooks/`), đạt `0` lỗi lint/format (làm mới đồng bộ dấu thời gian toàn bộ mã nguồn trong gói nộp).
  - **2. Vá lỗi dữ liệu, toán học & kiểm chứng (`27/11 20:10–22:32`):**
    + [src/clean_raw_data.py](src/clean_raw_data.py) & [src/stage3_pipeline_and_audit.py](src/stage3_pipeline_and_audit.py): Đồng bộ `pre_filter_valid=False` giữa `main.py --stage clean` và `--stage eda/all` (`0` dòng sai khác ở `distance`/`price`/`surge_multiplier`), bật `preserve_decimals=2` bảo toàn 2 chữ số thập phân hợp lệ cho `distance ≤ 8.0` dặm, và downcast [data/processed/cab_rides_cleaned.parquet](data/processed/cab_rides_cleaned.parquet) xuống `11.54 MiB` (gọn hơn [data/processed/master_rideshare.parquet](data/processed/master_rideshare.parquet) `15.99 MiB`).
    + [src/features.py](src/features.py), [src/stats_tests.py](src/stats_tests.py), [src/models.py](src/models.py), [src/stage5_modeling.py](src/stage5_modeling.py): Loại 2 cột `Shared × surge` đồng nhất `0` và khử đa cộng tuyến đồ thị hai phía $K_{6,6}$ (`destination_Boston University`) trong `EconomicInteractionPreprocessor` (`M2`) và `BipartiteCategoryEncoder` (`M1`), xác minh bằng SVD đạt **Full Column Rank** (`M1`: `50/50`, `cond = 29.17`; `M2`: `83/83`, `cond = 128.17`); chuẩn hóa dấu `Rank-Biserial r = (2U)/(n₁n₂) − 1`; mở rộng lưới `LassoCV` xuống `1e-5`; tính lại `Permutation Importance` trên `M2` (`distance × surge`: `+$0.3257`). Kết quả Test (`n = 46,573`): **`M7`** `R² = 0.9629`, `RMSE = $1.7769`, `MAPE = 8.14%` (vượt `M6` `0.9623 / $1.7910` trên `5/5` fold CV, `p = 0.0423`); **`M2 OLS`** `R² = 0.9571`, `RMSE = $1.9115`, `MAPE = 9.09%`.
    + [src/simulation.py](src/simulation.py), [src/stage6_evaluation.py](src/stage6_evaluation.py) & [scripts/thesis_v5_verification.py](scripts/thesis_v5_verification.py): Nâng cấp mô phỏng `Surge Cap` bên ngắn `Q_realized = min(Q_D, Q_S)` sang hàm cung nhạy giá (`ε_S ∈ {0.30, 0.60}`, `q_S = clip(1 + ε_S · Δp/p, sàn, 1.0)`) với dư cầu đầy đủ `Σ max(0, Q_D − Q_realized)`; hiệu chỉnh mẫu số Switchback Power Analysis về đúng `18` ngày có dữ liệu quan sát thực (trần `1.75x` tại 5 điểm nóng: `38.5` ngày ở `25×` cho `MDE = 3.0 pp`); bổ sung `strict_train_only_cleaning` cho `route_group_split` (`M7 R² = 0.9379 / MAPE = 10.67%`, tập nguyên bản `0.9538 / 9.37%`).
  - **3. Tái lập toàn trình từ dữ liệu gốc, đồng bộ Dashboard & `4` Notebooks (`27/11 22:33–23:42`):**
    + Thực thi lại trọn gói `main.py --stage all` từ dữ liệu thô chỉ-đọc `data/raw/*` (tái tạo đồng bộ toàn bộ `data/processed/*`, `models/*.joblib`, `16` bảng CSV, `4` tệp JSON, `21` ảnh `stage3..6`), chạy lại [scripts/thesis_v5_verification.py](scripts/thesis_v5_verification.py) và tái xuất `17` biểu đồ `≥ 300 DPI` qua [scripts/prepare_thesis_figures.py](scripts/prepare_thesis_figures.py).
    + Cập nhật [dashboard/config.py](dashboard/config.py), [dashboard/charts.py](dashboard/charts.py), [dashboard/business_logic.py](dashboard/business_logic.py), [dashboard/components/tab_overview.py](dashboard/components/tab_overview.py), [dashboard/components/tab_tariff_rq2.py](dashboard/components/tab_tariff_rq2.py), [dashboard/components/tab_models_rq3.py](dashboard/components/tab_models_rq3.py), [dashboard/components/tab_simulator_rq4.py](dashboard/components/tab_simulator_rq4.py), [dashboard/export_static_html.py](dashboard/export_static_html.py) (sửa công thức JS Estimator về đúng `l.b0 + l.b1 * d * s`, xóa chuỗi hardcode cũ `~15–34 ngày`, đồng bộ `Monotonic Split-CQR` `PICP = 79.98%` với `12/12` gói `[78.47%, 82.16%]`), [dashboard/README.md](dashboard/README.md) và [dashboard/tests/test_business_logic.py](dashboard/tests/test_business_logic.py), [dashboard/tests/test_model.py](dashboard/tests/test_model.py), [dashboard/tests/test_ui_stage6.py](dashboard/tests/test_ui_stage6.py) → tái xuất [reports/dashboard_interactive.html](reports/dashboard_interactive.html) (`5.27 MiB`, offline).
    + Chạy lại toàn bộ `4` notebooks [notebooks/03_Giai_Doan_3_Data_Prep_EDA_Visualization.ipynb](notebooks/03_Giai_Doan_3_Data_Prep_EDA_Visualization.ipynb) → [notebooks/06_Evaluation_Interpretation_Policy.ipynb](notebooks/06_Evaluation_Interpretation_Policy.ipynb) trên cùng máy tổng hợp `Python 3.11.9` để khóa khớp số liệu sau đại tu (`0` lỗi runtime, `0` lỗi `ruff`; toàn bộ 4 notebook trước đó đã được hai thành viên chạy chéo trên `Python 3.14.4` & `3.11.9` qua các phiên `8–19`).
  - **4. Hoàn thiện ấn phẩm Quarto `.qmd → .pdf`, đồng bộ nhật ký & đóng gói (`27/11 23:38 → 28/11 00:22`):**
    + Cập nhật [scripts/thesis_content.py](scripts/thesis_content.py) (bổ sung phân tích *driver multi-homing*, khung pháp lý Massachusetts DPU TNC, *Geographic Coverage Bias* ngoài 12 quận lõi, ẩn danh không gian *k-anonymity* / H3 Pha 3, giải thích hiện tượng ngoại suy phẳng vùng đuôi hiếm `n = 53` của `HistGB` tại `Cap 2.00x`, cập nhật mốc `22` phiên), [scripts/build_final_thesis.py](scripts/build_final_thesis.py) (hiệu chỉnh giãn đoạn chuẩn `After = 6pt`, nâng cỡ chữ 4 danh mục mở đầu lên `11.5pt–11.8pt` với chấm dẫn nét thường thoáng & thụt lề treo `4.5em/4.6em`, dẫn chiếu tường minh `Bảng B.1` & `Bảng B.2`, đổi `B.3` sang *"Trích đoạn mục tiêu Phiên 12"*) và [scripts/check_thesis_audit.py](scripts/check_thesis_audit.py).
    + Xuất [reports/2521001060_Nhom02_TieuLuan.qmd](reports/2521001060_Nhom02_TieuLuan.qmd) → [reports/2521001060_Nhom02_TieuLuan.pdf](reports/2521001060_Nhom02_TieuLuan.pdf) (`46` trang tổng cộng gồm `2` bìa + `7` mở đầu `i–vii` + `37` thân bài & phụ lục `1–37`, `28` trang Ch.1–5 chuẩn khung `25–45`, `17/17` biểu đồ `≥ 300 DPI`, `21` bảng, `17` hình, `0` lỗi TOC, `0` trang trống/thưa, `0` mồ côi `Nguồn:`, `23/23` APA 7th) và bản Word dự phòng [reports/2521001060_Nhom02_TieuLuan.docx](reports/2521001060_Nhom02_TieuLuan.docx).
    + Rà soát tính nhất quán thời gian của [nhật ký làm việc.md](nh%E1%BA%ADt%20k%C3%BD%20l%C3%A0m%20vi%E1%BB%87c.md) (tách riêng toàn bộ đợt đại tu tối `27/11–28/11` thành **Phiên 22**, khôi phục phạm vi gốc của Phiên 21 `27/11 14:10`, chuẩn hóa đường dẫn lịch sử ở các phiên cũ về đúng trạng thái trước khi tách `scripts/` để tránh tham chiếu hồi tố mà vẫn giữ `0` liên kết gãy), đồng bộ [README.md](README.md) và đóng gói đúng `128` tệp chọn lọc cho cả `2521001060_Nhom02_MaNguon.zip` và `2521001092_Nhom02_MaNguon.zip` cùng `2` bản PDF tiểu luận rời tại thư mục nộp bài.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Tách 5 script xuất bản/kiểm chứng sang scripts/, siết chuẩn PEP 8 <= 79 trên toàn bộ 60 tệp, vá triệt để các lỗi toán học/dữ liệu (preserve_decimals=2, SVD full rank K_{6,6}, cung nhạy giá ε_S, Switchback 18 ngày, JS Estimator), chạy tái lập toàn trình từ data/raw/* và 4 notebooks, hoàn thiện Quarto PDF chuẩn UFM, tách biên bản đại tu sang Phiên 22 và đóng gói kiểm định khói hai bộ hồ sơ."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* (6 lỗi thực tế đã phát hiện và khắc phục triệt để trong phiên)
    1. **Sai lệch `324` dòng giữa `--stage clean` vs `--stage all` & chia nhầm `10×` cự ly `≤ 8.0` dặm:** Đồng bộ `pre_filter_valid=False` và bật `preserve_decimals=2` tránh phát sinh cự ly 3 chữ số thập phân bất khả thi.
    2. **Khuyết hạng ma trận thiết kế đồ thị hai phía $K_{6,6}$ & XAI trên `M2`:** Loại 2 cột `Shared × surge` đồng nhất `0` và bỏ thêm `destination_Boston University` đạt SVD full rank (`M1` `50/50`, `M2` `83/83`), đồng thời đọc trực tiếp `distance_x_lyft_surge_excess` giúp `M2` phản hồi đúng khi hoán vị cụm `distance × surge` (`+$0.3257` thay vì `$0.0000`).
    3. **Công thức dư cầu mô phỏng `Surge Cap` & lỗi JS Offline Estimator:** Sửa `src/simulation.py` tính cung nhạy giá theo `Δp/p` kèm phần cầu tăng thêm `Q_D − Q_realized`; sửa `dashboard/export_static_html.py` từ `(l.b0 + l.b1 * d) * s` về đúng `l.b0 + l.b1 * d * s`.
    4. **Mẫu số thời gian Switchback (`22.65` → `18` ngày thực):** Chuyển từ khoảng lịch liên tục sang đúng `18` ngày có dữ liệu quan sát thực để đo chính xác lưu lượng kích hoạt hàng ngày.
    5. **Giãn đoạn `After = 6pt` & thụt lề treo Typst 0.15:** Hiệu chỉnh tham số `spacing` Typst đạt chuẩn `6pt`, dùng `pad(left: 4.5em/4.6em)` kết hợp `#h(-4.5em/-4.6em)` và chấm dẫn nét thường cho `outline.entry`, nâng bảng Chữ viết tắt lên `11.5pt` cho phép ngắt trang tự nhiên có lặp tiêu đề ở trang `iv`.
    6. **Rà soát đóng gói `2521001092_Nhom02_MaNguon.zip`:** Thay cơ chế thay chuỗi rộng bằng ánh xạ chính xác các đường dẫn tệp tiểu luận của MSSV `2521001092`, tránh lỗi lặp tên tệp trong nhật ký và bảo toàn SHA-256 của `data/raw/*`.
  - *Quyết định kiến trúc:* Tách biệt hoàn toàn `src/` (`11` module lõi DS) khỏi `scripts/` (`5` script xuất bản & kiểm chứng + `__init__.py`), siết `pyproject.toml` ở ngưỡng PEP 8 nghiêm ngặt `≤ 79` ký tự phủ cả `notebooks/`, và xác lập luồng tái lập khép kín từ `data/raw/*` → `main.py --stage all` → `thesis_v5_verification.py` → `prepare_thesis_figures.py` → `export_static_html.py` → `4` notebooks → Quarto `.qmd → Typst PDF` trên cùng một máy tổng hợp trước khi chốt gói nộp.
  - *Xác minh tự động:* `dashboard.validation` đạt `18/18 PASS`, `pytest dashboard/tests -v` đạt `40/40 PASS`, `check_thesis_audit.py` đạt `100% PASS` (`[CHECK 04C]` `Typst 0.15.1`, `[CHECK 04D]` `0` lỗi dàn trang/font, `Ch.1–5 = 28` trang), `ruff check` & `ruff format --check` đạt `0` lỗi trên `60` tệp `.py`/`.ipynb` (cả trên workspace chính lẫn bản giải nén sạch của hai gói `.zip`).
- **Tồn đọng & Bước tiếp theo:**
  - Hồ sơ bàn giao cuối cùng (`2521001060_Nhom02_TieuLuan.pdf`, `2521001092_Nhom02_TieuLuan.pdf`, `2521001060_Nhom02_MaNguon.zip`, `2521001092_Nhom02_MaNguon.zip`, `README.md` và `nhật ký làm việc.md`) đã được đối soát đồng bộ thời gian thực, vượt qua toàn bộ cổng kiểm thử tự động và sẵn sàng bảo vệ Vòng 3.

---

## [2026-11-27 14:10] - Phiên 21: Chuẩn Hóa Xuất Bản Chính Thức Từ Quarto (`.qmd → Typst PDF`), Kiểm Soát `≤ 45` Trang & Đóng Gói Lại Hồ Sơ Nộp Bài

- **Thành viên phụ trách chính:** **Đồng thực hiện — Trần Thanh Huy (`2521001060`) & Lê Huy Thắng (`2521001092`)**
- **Giai đoạn:** Chuẩn hóa định dạng xuất bản Quarto `.qmd` bắt buộc theo yêu cầu giảng viên & Nghiệm thu đóng gói (`GĐ8`).
- **Môi trường:** Windows 64-bit | `Python 3.11.9` | `Quarto 1.10.19` (`Pandoc 3.10.0`, `Typst 0.15.1`) | `pytest` / `pymupdf` / `ruff`
- **Mục tiêu phiên:**
  - Đáp ứng triệt để 3 yêu cầu bắt buộc của giảng viên: **(1)** Duy trì đầy đủ nhật ký làm việc `nhật ký làm việc.md`; **(2)** Giữ số trang nội dung chặt chẽ trong khung `25–45` trang; **(3)** Xuất bản Tiểu luận PDF trực tiếp từ tệp nguồn **Quarto Markdown (`.qmd`)** thay vì xuất từ Microsoft Word.
  - Thiết kế lại cấu trúc `reports/2521001060_Nhom02_TieuLuan.qmd` chuẩn học thuật UFM (tự chứa hoàn toàn): đặt `typst` làm định dạng mặc định, bổ sung Trang Bìa chính/phụ khung viền kép UFM, đánh số trang La Mã (`i–vi`) cho phần Mở đầu và Ả Rập (`1–34`) kèm header/footer cho phần Nội dung, tự động sinh `MỤC LỤC`, `DANH MỤC BẢNG BIỂU` (`21` mục) và `DANH MỤC HÌNH VẼ / BIỂU ĐỒ` (`17` mục).
  - Nâng cấp cổng kiểm toán `src/check_thesis_audit.py` bổ sung `[CHECK 04C]` (xác minh nguồn Quarto/Typst + Danh mục Bảng/Hình) và `[CHECK 04D]` (chặn mồ côi `Nguồn:`, chặn lặp `q̂`, khóa font mã nguồn `Consolas`), đồng bộ `README.md` và đóng gói lại `2521001060_Nhom02_MaNguon.zip`.
- **Nội dung đã hoàn thành:**
  - **Nâng cấp trình xuất bản Quarto & kiểm toán:** Cập nhật `src/build_final_thesis.py` (`export_thesis_qmd()`, `_render_pdf_via_quarto()`, `_update_docx_toc_via_word()`), `src/thesis_content.py` (mở rộng cột `1` của `Bảng 3.1` sang `4.2 cm` chống tràn `distance_x_lyft_surge_excess`, cập nhật động mốc `21` phiên) và `src/check_thesis_audit.py` (bổ sung `[CHECK 04C]` & `[CHECK 04D]`).
  - **Ấn phẩm Quarto `.qmd → .pdf` chuẩn UFM:** Xuất [reports/2521001060_Nhom02_TieuLuan.qmd](reports/2521001060_Nhom02_TieuLuan.qmd) (`153.44 KiB`), biên dịch trực tiếp sang [reports/2521001060_Nhom02_TieuLuan.pdf](reports/2521001060_Nhom02_TieuLuan.pdf) (`9.21 MiB`, `42` trang tổng cộng, **`27` trang nội dung chính Ch.1–5** gọn trong khung `25–45`, `17/17` biểu đồ `697.5–990.2 DPI`, `21` bảng, `17` hình, `0` lỗi TOC, `0` trang trống/thưa, `0` mồ côi `Nguồn:`, `23/23` APA 7th) và đồng bộ bản Word dự phòng [reports/2521001060_Nhom02_TieuLuan.docx](reports/2521001060_Nhom02_TieuLuan.docx) (`8.54 MiB`).
  - **Đồng bộ hồ sơ & đóng gói mã nguồn:** Cập nhật [README.md](README.md) và đóng gói lại `2521001060_Nhom02_MaNguon.zip` (`126` tệp sạch, bảo toàn SHA-256 của `data/raw/*`).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Chuyển xuất bản Tiểu luận chính thức sang Quarto (.qmd -> Typst PDF) chuẩn UFM có đầy đủ Bìa, Mục lục, Danh mục Bảng/Hình, bảo đảm khung <= 45 trang nội dung, kiểm duyệt trực quan từng trang và đóng gói lại gói mã nguồn .zip."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* (5 lỗi trình bày thực tế đã phát hiện và xử lý khi kiểm duyệt từng trang PDF)
    1. **Ký tự `_` kích hoạt in nghiêng Pandoc & `q̂` nhân đôi trong PDF:** Bổ sung `s.replace("_", "\\_")` và chuyển `q̂` sang toán học `$hat(q)$` để hiển thị sắc nét và không nhân đôi khi trích xuất văn bản.
    2. **Lỗi vỡ dấu tiếng Việt trong khối mã nguồn (`DejaVuSansMono` ghi đè `Consolas`):** Khóa `monofont: "Consolas"` và `show raw: set text(font: ("Consolas", "Courier New", "DejaVu Sans Mono"))` khôi phục hiển thị dấu tiếng Việt sắc nét.
    3. **Cắt đôi dòng bảng giữa hai trang (`Bảng 2.1`) & tràn ô (`Bảng 3.1`):** Thêm `set table.cell(breakable: false)` và mở rộng cột `1` của `Bảng 3.1` (`3.5 → 4.2 cm`).
    4. **Mồ côi dòng `Nguồn:` đầu trang (`Bảng 3.5`) & sát dòng Mục lục/Danh mục:** Tinh chỉnh giãn cách `show outline.entry`, tắt `justify` ở danh sách đường dẫn bìa phụ và thêm `[CHECK 04D]` tự động chặn tái diễn.
    5. **Rò rỉ phạm vi `#set text` sau Tài liệu tham khảo:** Dùng phạm vi cục bộ Typst `#[ ... ]` cho Bảng chữ viết tắt & APA 7th, giữ thụt treo `1.27 cm` chuẩn APA 7th không ảnh hưởng Phụ lục A/B.
  - *Quyết định kiến trúc:* Chuyển nguồn biên dịch PDF chính thức từ Microsoft Word COM sang Quarto/Typst (`.qmd` tự chứa hoàn toàn) để bảo toàn ảnh `≥ 300 DPI` và tái lập đa nền tảng bằng một lệnh `quarto render`, giữ `.docx` làm bản dự phòng.
- **Tồn đọng & Bước tiếp theo:**
  - Tiến hành kiểm toán phản biện độc lập trên gói `.zip` vừa đóng gói để phát hiện và khắc phục mọi tồn đọng kỹ thuật/học thuật còn sót trước hạn nộp cuối.

---

## [2026-11-25 20:27] - Phiên 20: Kiểm Toán Tổng Thể Ấn Phẩm Tiểu Luận (`v5`), Xuất Bản PDF 300 DPI & Hoàn Tất Đóng Gói Hồ Sơ Nộp Bài

- **Thành viên phụ trách chính:** **Lê Huy Thắng** (MSSV: `2521001092`)
- **Giai đoạn:** Nghiệm thu & Kiểm toán Tổng thể Hồ sơ Nộp bài Cuối cùng (`GĐ8`).
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`python-docx 1.2.0`, `pypdf 6.21.1`, `pywin32 312`, `Microsoft Word Office 16 COM`; `pytest`/`pymupdf`/`ruff` qua `uv`)
- **Mục tiêu phiên:**
  - Đối chiếu nhất quán toàn bộ thông số giữa báo cáo Tiểu luận, `README.md`, bảng phân công thành viên và kết quả định lượng `v5`.
  - Tái biên dịch Tiểu luận Word/PDF/Quarto, cập nhật tự động Mục lục, Danh mục Bảng/Hình, khôi phục `17` biểu đồ `300 DPI` trong PDF và tinh gọn Phụ lục.
  - Thực thi kiểm toán toàn văn (`check_thesis_audit.py`), kiểm tra dữ liệu Dashboard (`dashboard.validation`), kiểm thử đóng gói và nén đồng bộ gói mã nguồn nộp bài.
- **Nội dung đã hoàn thành:**
  - **Đồng bộ tài liệu & cấu hình xuất bản:** [README.md](README.md), [requirements.txt](requirements.txt), `src/thesis_content.py`, `src/build_final_thesis.py`.
  - **Ấn phẩm & Gói nộp cuối:** Tái xuất [reports/2521001060_Nhom02_TieuLuan.qmd](reports/2521001060_Nhom02_TieuLuan.qmd), [reports/2521001060_Nhom02_TieuLuan.docx](reports/2521001060_Nhom02_TieuLuan.docx) (`8.54 MiB`), [reports/2521001060_Nhom02_TieuLuan.pdf](reports/2521001060_Nhom02_TieuLuan.pdf) (`17.11 MiB`, `54` trang sau tinh gọn trang trống Phụ lục A/B, giữ `38` trang Ch.1–5, `0` lỗi TOC, `23/23` APA 7th) và đóng gói lại `2521001060_Nhom02_MaNguon.zip` (`126` tệp, tinh gọn từ `130` tệp của Phiên 19).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Đối soát toàn bộ số liệu Tiểu luận với README và kết quả v5, tái xuất bản Word/PDF 300 DPI, chạy toàn bộ kiểm toán tự động và đóng gói lại tệp nguồn .zip."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Xác minh tự động `check_thesis_audit.py` đạt `12/12 PASS` (khớp `5/5` đoạn mã Phụ lục A, `323` từ Executive Summary), `dashboard.validation` đạt `18/18 PASS` (`0` CRITICAL), `test_export_and_perf.py` `3/3 PASS` và `0` sai lệch SHA-256 giữa `.zip` với ổ đĩa.
  - *Quyết định kiến trúc:* Chốt gói nộp tinh gọn `126` tệp (tách các tệp nội bộ và nháp ngoài danh mục nộp) kèm bản PDF `300 DPI` đồng bộ từ Word/Quarto.
- **Tồn đọng & Bước tiếp theo:**
  - Toàn bộ sản phẩm bàn giao (`2521001060_Nhom02_TieuLuan.pdf`, `README.md` và `2521001060_Nhom02_MaNguon.zip`) đã đồng bộ 100% và sẵn sàng bảo vệ Vòng 3.

---

## [2026-11-25 17:47] - Phiên 19: Đồng Bộ Workspace Nộp Bài, Rà Soát Tài Liệu Kỹ Thuật, Dàn Trang Tiểu Luận `54` Trang & Đóng Gói Mã Nguồn `.zip`

- **Thành viên phụ trách chính:** **Trần Thanh Huy** (MSSV: `2521001060`)
- **Giai đoạn:** Kiểm tra & Đóng gói hồ sơ nghiệm thu (`2026-11-24` → `2026-11-25 17:47`).
- **Môi trường:** Windows 64-bit | `Python 3.14.4` (`python-docx`, `PyMuPDF`, `Microsoft Word Office 16 COM`)
- **Mục tiêu phiên:**
  - Đồng bộ toàn bộ cây thư mục dự án sạch (`2026-11-24 06:17`) sang môi trường đóng gói nộp bài.
  - Rà soát nhất quán hệ thống tài liệu Dashboard, báo cáo Giai đoạn 7 và đồng bộ số trang thực tế của Tiểu luận (`54` trang Word/PDF, giữ `37` trang Ch.1–5, `0` trang thưa) vào `README.md`.
  - Biên dịch lại đồng bộ `.qmd`, `.docx`, `.pdf`, ảnh xem trước `preview_pages/` và nén gói mã nguồn `2521001060_Nhom02_MaNguon.zip`.
- **Nội dung đã hoàn thành:**
  - **Tài liệu rà soát nội bộ (`16:28–16:37`, không đưa vào gói `.zip` nộp):** [dashboard/README.md](dashboard/README.md) (có trong `.zip`), cùng các bản lưu nội bộ `dashboard/docs/FILE_INVENTORY.md`, `dashboard/docs/IMPLEMENTATION_ROADMAP.md`, `dashboard/docs/RISK_REGISTER.md`, `reports/Giai_Doan_7_Bao_Cao_Chi_Tiet_Dashboard_Va_Kiem_Dinh.md`.
  - **Xuất bản & Đóng gói (`17:46–17:47`):** [README.md](README.md), `src/thesis_content.py`, `src/build_final_thesis.py`, [reports/2521001060_Nhom02_TieuLuan.qmd](reports/2521001060_Nhom02_TieuLuan.qmd), [reports/2521001060_Nhom02_TieuLuan.docx](reports/2521001060_Nhom02_TieuLuan.docx), [reports/2521001060_Nhom02_TieuLuan.pdf](reports/2521001060_Nhom02_TieuLuan.pdf) (`54` trang) và `2521001060_Nhom02_MaNguon.zip` (`130` tệp).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Rà soát đồng bộ toàn bộ tài liệu kỹ thuật, cập nhật chính xác số trang bản in cuối cùng vào README và đóng gói bộ mã nguồn nộp giảng viên."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Kiểm tra đối soát chéo số trang bản in thực tế (`54` trang, `37` trang Ch.1–5, `0` trang thưa) qua `PyMuPDF` và dọn các tệp phụ trước khi nén `.zip`.
  - *Quyết định kiến trúc:* Đóng gói chọn lọc `130` tệp sạch vào `.zip` (tách riêng `archive/`, `audit_v4/`, `scratch/` ngoài gói nộp chính thức để giữ cấu trúc tinh gọn đúng chuẩn Mục 4.1).
- **Tồn đọng & Bước tiếp theo:**
  - Đối chiếu chéo toàn bộ hồ sơ tiểu luận, mã nguồn và ấn phẩm hoàn chỉnh trước khi chốt nghiệm thu.

---

## [2026-11-23 20:45] - Phiên 18: Đại Tu Học Thuật v5 — Kiểm Chứng Hợp Nhất `WCR`/`Q_realized`/Switchback, `17` Biểu Đồ `300 DPI` & Tiểu Luận `v5`

- **Thành viên phụ trách chính:** **Lê Huy Thắng** (MSSV: `2521001092`)
- **Giai đoạn:** Khắc phục toàn diện Báo cáo Kiểm toán Độc lập (`v4 → v5`, phần Học thuật, Thống kê, Hình ảnh, Tiểu luận & Đồng bộ Hồ sơ `GĐ3–GĐ5` & `GĐ8`).
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`scipy 1.17.1`, `statsmodels 0.15.0`, `matplotlib 3.11.2`, `pillow 13.0.0`, `python-docx 1.2.0`, `pywin32 312`, `Microsoft Word Office 16 COM`; `pymupdf` qua `uv`).
- **Mục tiêu phiên:**
  - Hợp nhất toàn bộ kiểm chứng thống kê & kinh tế lượng phản biện vào một module chuẩn duy nhất (`src/thesis_v5_verification.py`), xử lý triệt để `C2` (`Q_realized = min(Q_D, Q_S)`), `H1`, `H3`, `H4c` đa cộng tuyến và `Exact WCR` đủ bậc tự do (`B = 1,999`).
  - Vẽ lại toàn bộ `17` biểu đồ tiểu luận `300 DPI` (`src/prepare_thesis_figures.py`), biên tập nội dung `v5` trong `src/thesis_content.py`, chạy kiểm toán `12/12`, đồng bộ `README.md` và rà soát `23/23` trích dẫn APA 7th.
- **Nội dung đã hoàn thành:**
  1. **GĐ3 — Hợp nhất Kiểm chứng `v5`, `C2`, `H3`, `H4c` & `M1`:** Xóa lỗi đa cộng tuyến `H4c` (hạng đầy đủ `36/36`, `F = 0.4993, p = 0.8094, ΔR² = 2.1e-06`, `0` cảnh báo); triển khai phân phối bên ngắn `Q_realized = min(Q_D, Q_S)` cho `simulation.py` kèm `assert supply_drop10 < supply_drop5 < static`; hợp nhất kiểm chứng vào `src/thesis_v5_verification.py` → [reports/results/thesis_v5_verification.json](reports/results/thesis_v5_verification.json): Exact WCR (`B = 1,999`, `8` giờ mưa lớn: `p_wild = 0.8205 / 0.8340`); RQ2 `UberX–Lyft` `3.28 / 5.22 mi`, `UberXL–Lyft XL` Spec B **`8.65 mi > 7.86 mi`**; Monotonic Split-CQR `79.84%` (`12/12` gói `[78.66%, 82.16%]`, `0` cắt ngược); Grouped XAI `97.43%` (`name 69.85% + dist/surge 27.58%`); RQ4 Pha 1 **Trần `2.00x`** (`53` lượt, `-0.018% / -0.022% / -0.026%`, duy nhất đạt `τ ≤ 0.10%`) & Pha 2 **Trần `1.75x`** (`704` lượt, `-0.201% / -0.277% / -0.352%`, Triggered Switchback tại 5 quận điểm nóng); Fair Same-Table I/O (`270,898 × 32`): Parquet `15.87 MB / 0.098s` vs CSV `64.55 MB / 2.880s` (`29.3×` nhanh hơn, `4.07×` nhỏ hơn).
  2. **GĐ4–GĐ5 — `17` Biểu đồ `300 DPI` & Xuất bản Tiểu luận `v5`:** Tái tạo ảnh GĐ3–6 và `17` hình tiểu luận qua `src/prepare_thesis_figures.py`; cập nhật `src/thesis_content.py` & `src/check_thesis_audit.py` (`23/23` APA 7th, `5/5` parity Phụ lục A, `12/12 PASS`, `0` trang thưa trên [reports/2521001060_Nhom02_TieuLuan.pdf](reports/2521001060_Nhom02_TieuLuan.pdf)).
  3. **GĐ8 — Đồng bộ `README.md` & Kiểm chứng Toàn trình:** Đồng bộ toàn bộ chỉ số `v5` sang [README.md](README.md) khớp bảng kết quả CSV/JSON và xác minh `test_root_readme_synchronized_post_audit` PASS.
  - **Bảng Đối soát Trước (`v4`) → Sau (`v5`):**

| Hạng mục kiểm toán                               | Bản `v4`                                                               | Bản `v5`                                                                                                                                                         |
| ------------------------------------------------ | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Schema & RAM `master_rideshare.parquet`          | `270,898 × 30` (`41.08 MB`, `location` object)                         | **`270,898 × 32`** (**`37.46 MB` RAM / `15.87 MB` đĩa**, `8/8` `category`, `29.3×` nhanh hơn CSV)                                                                |
| Bảo toàn nguồn gốc Surge (`CRIT-03`)             | `fillna(1.0)` mù trên `38,037` Lyft → Surge `4.82%`                    | **`0` điền mù**, giữ `NaN` + `is_surge_original` → Surge Lyft thực **`6.80%`** (`92,384` lượt)                                                                   |
| Làm sạch khí tượng (`H4`)                        | Điền khuyết lệch trạm/ngày, gộp sentinel mưa                           | Trung vị cùng giờ toàn Boston (`0` fallback), tách `is_rain_sentinel` → **`152,043`** lượt mưa/khô sạch                                                          |
| RQ2 `UberX–Lyft` / `UberXL–Lyft XL` (Spec A / B) | `3.10 / 4.20 mi` · `3.75 / 5.87 mi`                                    | **`3.28 / 5.22 mi`** · **`3.88 mi` / Ngoài miền (`8.65 > 7.86 mi`: Lyft XL rẻ hơn toàn miền)**                                                                   |
| Dự báo `M7` (Có điều kiện) & Ex-Ante             | Chỉ có Conditional (`R² = 0.9531, RMSE = $2.0213`, `n_test = 54,180`)  | Conditional **`R² = 0.9629, RMSE = $1.7764, MAPE = 8.17%`** (`n_test = 46,573`) + **Ex-Ante `R² = 0.9292`, Surge Classifier `AUC = 0.7404`**                     |
| Monotonic Split-CQR & Grouped XAI                | CQR `OrdinalEncoder` (`80.34%`); XAI `97.95%`                          | CQR **`pandas category` gốc (`79.84%`, `12/12` trong `[78.66%, 82.16%]`, `0` cắt ngược)**; XAI `97.43%`                                                          |
| RQ4 Surge Cap & Power Analysis (`C2/H3`)         | `Q_D × Q_S` (lệch dấu cung), ITT giả định `m=50`                       | **`Q_realized = min(Q_D, Q_S)`** đơn điệu chặt: Pha 1 **`2.00x`** (`τ ≤ 0.10%`), Pha 2 **`1.75x`** **Triggered Switchback**                                      |
| Tiểu luận, Dashboard, PEP 8 & Kiểm thử           | `59` trang (`41` Ch.1–5), `22` `src/`, `17/17` Audit, `38/38` `pytest` | **`54` trang (`37` Ch.1–5, `0` trang thưa)**, **`16` `src/`**, `ruff` **`0` lỗi / `45` tệp `.py`**, **`18/18` Audit**, **`40/40` `pytest`**, `4/4` Notebook PASS |

- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Hợp nhất kiểm chứng v5 vào src/thesis_v5_verification.py, triển khai Q_realized = min(Q_D, Q_S), vẽ 17 hình 300 DPI và hoàn thiện Tiểu luận v5."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Sửa dương tính giả `[CHECK 05]` (`'7.52%'` trùng chuỗi con `(-27.52%)`) bằng negative lookbehind `(?<!\d)`; đồng bộ nguyên văn Phụ lục A Trích đoạn 2 từ `step3_merge_and_eda` (`5/5` parity PASS); tinh chỉnh kích thước `Hình 4.9–4.10` xóa `4` trang cuối chương thưa.
  - *Quyết định kiến trúc:* Xóa đa cộng tuyến `H4c` (hạng `36/36`), chạy Exact WCR `B = 1,999`, CQR `pandas category` `79.84%` và khóa Triggered Switchback.
- **Tồn đọng & Bước tiếp theo:**
  - Hai thành viên rà soát toàn bộ hồ sơ xuất bản (`PDF`, `DOCX`, `README.md`, `.zip`) ở phiên tiếp theo; ở Pha 1 triển khai thực tế, đóng gói bộ 3 tệp Quantile `Monotonic Split-CQR` vào `models/`.

---

## [2026-11-23 14:20] - Phiên 17: Đại Tu Kỹ Thuật v5 — Pipeline `CRIT-03`/`H4`, Mô Hình Có Điều Kiện & `Ex-Ante`, Dashboard `18/18`, `40/40` `pytest` & PEP 8 `ruff`

- **Thành viên phụ trách chính:** **Trần Thanh Huy** (MSSV: `2521001060`)
- **Giai đoạn:** Khắc phục toàn diện Báo cáo Kiểm toán Độc lập (`v4 → v5`, phần Kiến trúc Pipeline, Mô hình hóa ML, Dashboard, Kiểm thử tự động & Chuẩn hóa PEP 8 `GĐ0–GĐ2` & `GĐ6–GĐ7`).
- **Môi trường:** Windows 64-bit | `Python 3.14.4` (`pandas 3.0.3`, `numpy 2.5.0`, `scikit-learn 1.9.1`, `streamlit`, `plotly`, `pytest 9.2.0`, `ruff 0.17.6`).
- **Mục tiêu phiên:**
  - Giải quyết triệt để **100%** các lỗi kỹ thuật pipeline & mô hình (`C1/CRIT-03`, `C3` Ex-Ante, `H4`, `M1`, `M5`) so với bản `v4` mà **tuyệt đối không chỉnh sửa `data/raw/*`** (xác minh SHA-256 trước và sau).
  - Khóa hai quyết định kiến trúc cùng Lê Huy Thắng: **(1)** Giữ `270,898` dòng ở bảng sạch kèm cờ `is_surge_original`, giữ nguyên `NaN` cho `38,037` dòng Lyft khuyết/lính canh surge (`0` `fillna(1.0)` mù) và huấn luyện trên `232,861` lượt quan sát surge thực; **(2)** Định khung hai tầng Mô hình Có điều kiện (`M0a–M7`) + Thực nghiệm Ex-Ante (`23` biến không surge + Bộ phân loại Surge Lyft); hoàn thiện Dashboard `18/18` Audit, `4/4` Notebook PASS, `40/40` `pytest` và `0` lỗi PEP 8 `ruff`.
- **Nội dung đã hoàn thành:**
  1. **GĐ0 — Dọn mã chết & phụ thuộc:** Cập nhật [requirements.txt](requirements.txt) (`+pymupdf>=1.24.0`, `Python >= 3.11`), xóa `4` tệp chết `src/build_stage3..6_deliverables.py` để giữ đúng `16` module trong `src/`.
  2. **GĐ1 — Khắc phục `CRIT-03`, `H4` & `M1`:** Cập nhật [src/clean_raw_data.py](src/clean_raw_data.py) bỏ hoàn toàn `fillna(1.0)` mù trên Lyft (`38,037` dòng = `29.16%` giữ `NaN`, thêm `is_surge_original`; Surge Lyft thực = **`6.80%`** = `6,280 / 92,384`); khôi phục bậc 10 và điền khuyết 5 biến khí tượng theo **trung vị cùng giờ toàn Boston** (`0` fallback lệch cục bộ), thêm cờ `is_rain_sentinel` cô lập `27,237` lượt ô lỗi cảm biến mưa (`152,043` lượt mưa/khô sạch); ép kiểu `location` sang `category` (`8/8` cột phân loại, RAM giảm `55.79%`: `84.80 → 37.49 MB`, đọc Parquet `37.46 MB`); nâng cấp [dashboard/validation.py](dashboard/validation.py) lên **`18/18 PASS`**.
  3. **GĐ2 — FE, Mô hình Có điều kiện, Ex-Ante & `M5`:** Chạy lại [src/features.py](src/features.py) (`41` cột, 3 chiến lược chia trên `232,861` lượt surge thực → `186,288 / 46,573`) & [src/models.py](src/models.py) (sửa deadlock đa luồng RF Windows, `precompute=True` cho `LassoCV`, xác minh khớp 100% siêu tham số `M5`, nâng cấp Quantile `[P10, P90]` sang `pandas category` gốc). Kết quả: **`M7` Có điều kiện** `R² = 0.9629, RMSE = $1.7764, MAE = $1.1307, MAPE = 8.17%` (Lyft `R² = 0.9795, RMSE = $1.4318, 0` lỗi `> $15` vs Uber `R² = 0.9478, RMSE = $1.9698, 59/59` lỗi `> $15`); **`M2` OLS** `R² = 0.9553, RMSE = $1.9503, MAPE = 9.12%`; **Ex-Ante `23` biến** `R² = 0.9292 / RMSE = $2.4551` (Lyft `> 1.0x`: `0.4717 / $10.25`; Holdout `38,037` Lyft khuyết surge: `0.9065 / $3.0780`); **Lyft Surge Classifier** `HistGB ROC-AUC = 0.7404, PR-AUC = 0.1445`.
  4. **GĐ6–GĐ7 — Dashboard, Offline HTML, `4` Notebook, `40/40` `pytest` & PEP 8 (`ruff`):** Đồng bộ `dashboard/` & [dashboard/export_static_html.py](dashboard/export_static_html.py) (`5.27 MB`, tích hợp `Q_realized = min(Q_D, Q_S)`), chạy sống `4/4` Notebook PASS, bổ sung 2 test hồi quy → **`18/18` Audit** & **`40/40` `pytest` PASS**; thêm [pyproject.toml](pyproject.toml), xử lý `32` lỗi lint (`F401`, `I001`, `F841`, `E741`, `F541`) và định dạng **`45/45` tệp `.py`** ([main.py](main.py), `src/`, `dashboard/`: `ruff check` `0` lỗi, `ruff format --check` sạch).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Khắc phục triệt để CRIT-03 (0 fillna mù trên 38,037 Lyft), H4 trung vị khí tượng toàn Boston, Ex-Ante 23 biến, M5 parity và PEP 8 ruff."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Sửa `dashboard/model_loader.py` gây `ValueError: Input X contains NaN` ở `M2` do chưa lọc `is_surge_original == 1`, và sửa `export_static_html.py` tra nhầm `H2c_Price_by_TimeOfDay` thay vì `H2c_Price_All`; xác minh `data/raw/*` SHA-256 bất biến 100%.
  - *Quyết định kiến trúc:* Giữ nguyên `NaN` kèm cờ `is_surge_original` (huấn luyện trên `232,861` surge thực), tách Ex-Ante & Surge Classifier, đạt `0` lỗi `ruff` trên `45` tệp `.py` và `40/40` `pytest`.
- **Tồn đọng & Bước tiếp theo:**
  - Bàn giao toàn bộ dữ liệu sạch `v5`, mô hình `.joblib`, `thesis_v5_verification` đầu vào và Dashboard kiểm thử `40/40` sang **Phiên 18** để Lê Huy Thắng hoàn thiện kiểm chứng hợp nhất học thuật, `17` hình `300 DPI` và xuất bản Tiểu luận `v5`.

---

## [2026-11-22 13:30] - Phiên 16: Khôi Phục Tái Lập Độc Lập 100%, Đồng Bộ Độ Nhạy Cung Tab 5B & Tinh Gọn Hồ Sơ (`v4`)

- **Thành viên phụ trách chính:** **Lê Huy Thắng** (MSSV: `2521001092`)
- **Giai đoạn:** Khóa tái lập & tinh gọn nghiệm thu cuối (*Reproducibility Closure & Final Polish* sau thẩm định Phiên 15: `8.60/10`).
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`pandas 3.0.6`, `numpy 2.4.6`, `scipy 1.17.1`, `statsmodels 0.15.0`, `scikit-learn 1.9.1`, `python-docx 1.2.0`, `pywin32 312`, `Microsoft Word Office 16 COM`; `pyarrow`/`streamlit`/`plotly`/`pytest`/`pymupdf` qua `uv`).
- **Mục tiêu phiên:**
  - Khắc phục triệt để lỗ hổng tái lập mới phát sinh sau khi cô lập `audit_v4/` (tạo module độc lập `src/thesis_v4_verification.py` tái tạo 100% `reports/results/thesis_v4_verification.json`).
  - Sửa sạch các tồn đọng nhỏ hiển hiện: 2 unused imports `RidgeCV/LassoCV` trong `src/models.py`, lỗi định dạng backtick trong nhật ký và tràn 10 dòng mồ côi ở trang cuối PDF.
  - Bổ sung độ nhạy mất cung tài xế `-5%/-10%` vào Dashboard Tab 5B (khớp Bảng 5.1 Tiểu luận), đổi tên hàm biểu đồ `build_surge_cap_tradeoff_fig`, nâng cấp `load_cab_rides_chunked()` nén từng chunk và tinh gọn các tệp nội bộ phụ ngoài danh mục nộp chính.
- **Nội dung đã hoàn thành:**
  1. **Tái lập độc lập `v4` (`src/thesis_v4_verification.py`):** Tích hợp đủ 9 kiểm chứng thực nghiệm (`weather_sentinel_sensitivity`, `spec_b_train80_parity`, `clean_monotonic_cqr`, `strict_train_only_cleaning`, `exact_wcr_wild_bootstrap`, `lyft_surge_all_12_districts`, `stage6_100pct_boundary_misses`, `api_batch_group_split` có cô lập dòng khuyết TS theo `id` cho kết quả `32.35%` trùng / `M7 R² = 0.9503, RMSE = $2.0929, MAPE = 8.51%`, và `oos_surge_cap_175_test_only` `$4.75 / -12.26%`).
  2. **Clean Code & Tab 5B:** Xóa `LassoCV`, `RidgeCV` thừa ở [src/models.py](src/models.py); nâng cấp `load_cab_rides_chunked()` trong [src/clean_raw_data.py](src/clean_raw_data.py) nén `product_id` sang `category` và lọc sớm ~61% dòng từng chunk; tích hợp bộ chọn độ nhạy mất cung `-5% / -10%` trên Tab 5B ([dashboard/components/tab_simulator_rq4.py](dashboard/components/tab_simulator_rq4.py)).
  3. **Tinh gọn hồ sơ & kiểm chứng:** Tách khỏi danh mục nộp chính đối với `dashboard/docs/`, `reports/Giai_Doan_*.{md,pdf}`, `reports/Nhat_Ky_Khac_Phuc_Va_Final_Audit_Lan_2.md`, `data/raw/*_xem_excel.xlsx`, `.vscode/`; co gọn trang cuối [reports/2521001060_Nhom02_TieuLuan.pdf](reports/2521001060_Nhom02_TieuLuan.pdf) xóa trang mồ côi; đồng bộ [README.md](README.md) và xác minh `dashboard.validation` (`17/17 PASS`), `pytest` (`38/38 PASS`).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Xây dựng module tái lập độc lập cho toàn bộ kiểm chứng v4 chỉ đọc từ dữ liệu sạch, đồng bộ độ nhạy mất cung -5%/-10% vào Dashboard Tab 5B, xóa import thừa và tinh gọn danh mục hồ sơ nộp bài."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Cô lập riêng dòng thiếu timestamp theo `id` khi chia `API-Batch GroupShuffleSplit` tránh gom nhầm `62,864` dòng `NaN` vào cùng một cụm; lọc sớm `pre_filter_valid=True` theo từng chunk giúp giảm mạnh đỉnh RAM khi đọc `cab_rides.csv`.
  - *Quyết định kiến trúc:* Tách toàn bộ tài liệu nội bộ/tham khảo phụ khỏi gói nộp chính để bảo đảm kho mã nguồn gọn sạch chuẩn Mục 4.1 và tự tái lập 100% không phụ thuộc `audit_v4/`.
- **Tồn đọng & Bước tiếp theo:**
  - Đã chốt mốc tái lập `v4` (`17/17` kiểm tra Dashboard và `38/38` `pytest` PASS).
  - Rà soát lại toàn bộ quy chuẩn dữ liệu/mô hình (CRISP-DM, XAI, kiểm định thống kê) và chuẩn hóa mã nguồn `PEP 8`.
  - Đối chiếu chéo số liệu giữa báo cáo Tiểu luận, `README.md` và Dashboard trước khi đóng gói hồ sơ nghiệm thu.

---

## [2026-11-21 16:30] - Phiên 15: Kiểm Toán Độc Lập 4 Giai Đoạn (v3 → v4) & Khắc Phục Gatekeeper P0/P1/P2 (CV Pipeline, `API-Batch`, Typography PDF; Ca chiều, cùng Lê Huy Thắng)

- **Thành viên phụ trách chính:** **Trần Thanh Huy** (MSSV: `2521001060`)
- **Giai đoạn:** Kiểm toán Phản biện 4 giai đoạn (`v4`) & Khắc phục toàn diện Gatekeeper `P0/P1/P2` (ca chiều).
- **Môi trường:** Windows 64-bit | `Python 3.14.4` (`pandas 3.0.3`, `scikit-learn 1.9.1`, `statsmodels 0.15.0`, `pytest 9.2.0`, `python-docx`, `PyMuPDF`, `Microsoft Word Office 16 COM`)
- **Mục tiêu phiên:**
  - *(Phần I — Lê Huy Thắng)* Kiểm toán trực tiếp trên dữ liệu thô/sạch và sửa 15 hạng mục (`CRITICAL #1` lỗi biên `$10^{-8}` CQR → `Monotonic Split-CQR 80.34%`; `CRITICAL #2` Top 4 Surge `Theatre District 6.86%`; `HIGH #1` `Strict Train-Only Refit`; `HIGH #2` tách `1,331` sentinel mưa & `Exact WCR B = 1,999`; đồng bộ `23/23` APA 7th).
  - *(Phần II — Trần Thanh Huy)* Khóa Gatekeeper `0` đường dẫn cứng và `0` ký tự lạ (`[CHECK 12B]`), bọc `Pipeline` bên trong `GridSearchCV`/`RandomizedSearchCV` (`src/models.py`), bổ sung `load_cab_rides_chunked()`, chạy `API-Batch GroupShuffleSplit` (`M7 R² = 0.9528`) & `OOS Test-Only Surge Cap 1.75x`, tinh gọn thư mục nộp.
- **Nội dung đã hoàn thành:**
  - **Tài liệu & Mã nguồn cập nhật:** [src/models.py](src/models.py), [src/clean_raw_data.py](src/clean_raw_data.py), [src/stats_tests.py](src/stats_tests.py), [src/simulation.py](src/simulation.py), `src/prepare_thesis_figures.py`, `src/thesis_content.py`, `src/build_final_thesis.py`, `src/check_thesis_audit.py`.
  - **Kết quả v4:** `audit_v4/` (`8` tài liệu đối soát), [reports/2521001060_Nhom02_TieuLuan.pdf](reports/2521001060_Nhom02_TieuLuan.pdf), [reports/dashboard_interactive.html](reports/dashboard_interactive.html), `17/17` Dashboard Audit & `38/38 pytest PASS`.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Thẩm định độc lập bản v3 trên dữ liệu thô, sửa triệt để 15 hạng mục P0/P1/P2 (Gatekeeper 0 đường dẫn cứng, Pipeline trong CV, API-Batch GroupShuffleSplit, chuẩn hóa 23/23 APA 7th) và tái xuất bản toàn bộ sản phẩm."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Xử lý lỗi biên `$10^{-8}` ở dải phân vị `Split-CQR` (`Monotonic Split-CQR 80.34%`), cô lập nhóm `NaN` timestamp theo `id` khi chia `API-Batch GroupShuffleSplit` và kiểm định OOS trên phân khúc trần `1.75x`.
  - *Quyết định kiến trúc:* Chặn rò rỉ nội bộ CV bằng cách bọc `ColumnTransformer` trực tiếp trong `GridSearchCV`; tách tài liệu nháp vào `audit_v4/` để giữ sạch cấu trúc kho chính.
- **Tồn đọng & Bước tiếp theo:**
  - Đóng gói module tái lập độc lập cho `v4`, đồng bộ độ nhạy cung Tab 5B và tối ưu trang cuối PDF ở phiên kế tiếp.

---

## [2026-11-21 12:55] - Phiên 14: Xuất Bản Tiểu Luận 5 Chương Chuẩn UFM (Word/PDF) & Kiểm Định Phản Biện 2 Vòng (`v2 → v3`; Ca trưa, cùng Trần Thanh Huy)

- **Thành viên phụ trách chính:** **Lê Huy Thắng** (MSSV: `2521001092`)
- **Giai đoạn:** Xuất bản Tiểu luận & Thẩm định Phản biện Độc lập 2 vòng (`v2 → v3`, ca trưa).
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`statsmodels 0.15.0`, `scipy 1.17.1`, `scikit-learn 1.9.1`, `matplotlib 3.11.2`, `pillow 13.0.0`, `python-docx 1.2.0`, `pywin32 312`, `Microsoft Word Office 16 COM`; `pymupdf` qua `uv`)
- **Mục tiêu phiên:**
  - *(Phần I — Trần Thanh Huy)* Thiết kế bộ tạo tài liệu động 100% chuẩn UFM (`src/thesis_content.py`, `src/build_final_thesis.py`, `src/check_thesis_audit.py`), chạy Word COM xuất bản Word/PDF có TOC tự động (`53` trang, `0` lỗi trường Word).
  - *(Phần II — Lê Huy Thắng)* Tiếp thu 2 vòng phản biện độc lập, bổ sung kiểm định cụm hai cấp (`3,354` cụm quận×giờ & `329` giờ Boston), `Wild Cluster Bootstrap`, `TOST ±$0.10`, `Holm/BH-FDR`, `Split-CQR` (`80.35%`), `Grouped XAI`, Power Analysis Switchback và kết xuất `17` hình tiểu luận `300 DPI`.
- **Nội dung đã hoàn thành:**
  - **Mã nguồn xuất bản & Kiểm định bổ sung:** `src/thesis_content.py`, `src/build_final_thesis.py`, `src/check_thesis_audit.py`, `src/prepare_thesis_figures.py` (`17` ảnh `300 DPI` tại `reports/figures/thesis/`), [reports/assets/ufm_logo.jpeg](reports/assets/ufm_logo.jpeg).
  - **Ấn phẩm Tiểu luận v3:** [reports/2521001060_Nhom02_TieuLuan.docx](reports/2521001060_Nhom02_TieuLuan.docx) và [reports/2521001060_Nhom02_TieuLuan.pdf](reports/2521001060_Nhom02_TieuLuan.pdf) (`38` trang Ch.1–5 gọn khung `25–45`, `Executive Summary` `315` từ, `0` tiêu đề bảng mồ côi, `12/12` audit PASS).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Xuất bản Tiểu luận chuẩn UFM có Mục lục tự động Word COM; thực thi kiểm định phản biện bổ sung (Cluster SE, WCR, TOST, Split-CQR, Grouped XAI, Power Switchback) chỉ-đọc không sửa dữ liệu sạch và xuất 17 biểu đồ 300 DPI."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Gỡ lỗi treo modal Word COM bằng `$word.AutomationSecurity = 3` và chỉ update TOC; thay `add_page_break()` bằng `page_break_before = True` để xóa trang trắng; chứng minh `M7` thắng `M6` trên `5/5` fold CV (`p = 0.0043`).
  - *Quyết định kiến trúc:* Đóng gói trình sinh báo cáo động từ CSV/JSON (`thesis_content.py` + `build_final_thesis.py`) và chạy cụm kiểm định phản biện chỉ-đọc không can thiệp dữ liệu sạch.
- **Tồn đọng & Bước tiếp theo:**
  - Tiến hành vòng kiểm toán độc lập toàn kho lưu trữ (v4) và rà soát Gatekeeper P0/P1/P2 ở ca chiều.

---

## [2026-11-21 08:18] - Phiên 13: Xây Dựng Trọn Bộ Dashboard 5 Tab (Streamlit + HTML Offline), Tab 1 Executive, `38/38` `pytest` & Hệ Thống P-D-M-V (Ca sáng, cùng Lê Huy Thắng)

- **Thành viên phụ trách chính:** **Trần Thanh Huy** (MSSV: `2521001060`)
- **Giai đoạn:** Dashboard 7 Giai đoạn (`GĐ0 → GĐ7`, ca sáng).
- **Môi trường:** Windows 64-bit | `Python 3.14.4` (`pandas 3.0.3`, `scikit-learn 1.9.1`, `streamlit 1.67.0`, `plotly 7.1.0`, `pytest 9.2.0`), `Node.js v24.16.0`
- **Mục tiêu phiên:**
  - *(GĐ0–GĐ3 — Trần Thanh Huy)* Kiểm toán tiền Dashboard (`PROJECT_AUDIT`, `RISK_REGISTER`), xây dựng Data Layer + `17/17` Validation + `15/15` test, Business Logic + Model Loader (`27/27` test), nền tảng UI + Tab 1.
  - *(GĐ4 & GĐ6 — Lê Huy Thắng)* Xây dựng Tab 2 (`RQ1`), Tab 3 (`RQ2`) và Tab 5 (`Live Predictor` + `Surge Cap Policy Simulator`, `HIGH-03`).
  - *(GĐ5 & GĐ7 — Trần Thanh Huy)* Xây dựng Tab 4 (`ML & XAI`), tối ưu cache `@st.cache_resource`, xuất bản HTML Offline (`5.24 MB`, `38/38 pytest PASS`).
  - *(Hoàn thiện Tab 1 Executive & P-D-M-V — Lê Huy Thắng & Trần Thanh Huy)* Thiết kế lại Tab 1 theo chuẩn Executive 5 khối (`5.26 MB` HTML), lập `Giai_Doan_7_Bao_Cao_Chi_Tiet...md`, viết lại `README.md` và hệ thống hóa 12 khối *Deconstruct & Read*, 12 thí nghiệm *Mutate & Break* cùng sổ *Bug Hunting*.
- **Nội dung đã hoàn thành:**
  - **Tầng lõi & Kiểm toán Dashboard:** [dashboard/config.py](dashboard/config.py), [dashboard/data_loader.py](dashboard/data_loader.py), [dashboard/validation.py](dashboard/validation.py) (`17/17 PASS`), [dashboard/business_logic.py](dashboard/business_logic.py), [dashboard/model_loader.py](dashboard/model_loader.py), [dashboard/charts.py](dashboard/charts.py).
  - **Giao diện 5 Tab & Offline HTML:** [dashboard/app.py](dashboard/app.py), [dashboard/components/tab_overview.py](dashboard/components/tab_overview.py), [dashboard/components/tab_pricing_rq1.py](dashboard/components/tab_pricing_rq1.py), [dashboard/components/tab_tariff_rq2.py](dashboard/components/tab_tariff_rq2.py), [dashboard/components/tab_models_rq3.py](dashboard/components/tab_models_rq3.py), [dashboard/components/tab_simulator_rq4.py](dashboard/components/tab_simulator_rq4.py), [dashboard/export_static_html.py](dashboard/export_static_html.py) → [reports/dashboard_interactive.html](reports/dashboard_interactive.html) (`5.26 MB`, `0` CDN ngoài).
  - **Tài liệu & Bộ kiểm thử:** `dashboard/docs/` (`PROJECT_AUDIT.md`, `FINAL_DASHBOARD_AUDIT.md`, `RISK_REGISTER.md`), `reports/Giai_Doan_7_Bao_Cao_Chi_Tiet_Dashboard_Va_Kiem_Dinh.md`, [README.md](README.md), và 9 tệp `pytest` tại `dashboard/tests/` (**`38/38 PASSED`**).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Triển khai đầy đủ Dashboard Python + Streamlit + Plotly theo quy trình tách lớp 7 giai đoạn, xuất HTML offline chạy không cần Internet, thiết kế Tab 1 Executive 5 khối và hệ thống hóa P-D-M-V phục vụ Vòng 3."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Tổng hợp 12 khối *Deconstruct & Read* (Bảng A), 12 thí nghiệm *Mutate & Break* (Bảng B) và 14 lỗi *Bug Hunting* (Bảng C) ở các mục A–C ngay dưới đây.
  - *Quyết định kiến trúc:* Tách 100% logic khỏi Streamlit UI; nhúng `plotly.js` nội tuyến đúng 1 lần vào HTML; gắn nhãn `API Quote Proxy` & `EXTRAPOLATION WARNING` chống hiểu nhầm ngoại suy; dùng `@st.cache_resource` giảm thời gian rerun `21.15s → 15.67s`.
### A. Tổng hợp Chuỗi *Deconstruct & Read* — Bản đồ 12 Khối Mã nguồn Trọng yếu (Mốc `v3/v4` ngày `21/11` — số dòng và một số chỉ số đã thay đổi ở `v5`)

|  STT | Tệp mã nguồn & Vị trí dòng (Mốc `21/11`)            | Hàm / Lớp trọng tâm                                           | Bản chất thuật toán & Lý do thiết kế (Mốc `v3/v4`)                                                                                                                                                                                                                                                                             |
| ---: | --------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    1 | `src/clean_raw_data.py` (`L99`, `L192`)             | `recover_weather_rain_scaled()`, `is_timestamp_original`      | Khôi phục lỗi nhân bậc $10^m$ của `rain > 1.0` bằng trung vị mưa thực cùng khung giờ (thay vì chia cứng `/10`), giảm `max(rain)` từ `1.000` về đúng `0.7807` in/h; đồng thời gắn cờ `is_timestamp_original` (`1/0`) và giữ `pd.NA` cho `62,864` dòng thiếu TS (`CRIT-02`), tuyệt đối không dùng `.ffill()`.                     |
|    2 | `src/stage3_pipeline_and_audit.py` (`L412`, `L450`) | Chuẩn hóa `America/New_York` & cờ `weather_matched`           | Quy đổi Unix timestamp ms từ UTC sang giờ địa phương Boston (`EST = UTC-5`) để khôi phục đúng nhịp đô thị (đáy `03:00`, đỉnh `19:00` EST); chỉ ghép `LEFT JOIN` thời tiết trên TS gốc và chỉ gán `is_raining` khi `weather_matched == 1` (`179,280` dòng, triệt tiêu `37,536` nhãn mưa giả `CRIT-01`).                          |
|    3 | `src/features.py` (`L7`, `L767`)                    | `lyft_surge_excess`, `EconomicInteractionPreprocessor`        | Khử đa cộng tuyến cấu trúc (`VIF(distance): 177.05 → 1.02`) bằng biến tâm hóa `(surge - 1.0) * is_lyft`; tạo ma trận tương tác kinh tế lượng `C(name):distance`, `C(name):log_distance`, `C(name):lyft_surge_excess` giúp `M2 OLS` nâng `R²` từ `0.9166 → 0.9456` và kéo `MAPE` từ `13.38% → 9.46%` (đạt SMART `≤ 10%`).        |
|    4 | `src/stats_tests.py` (`L282`, `L434`, `L495`)       | `test_h4_*`, `H4a_Wald_UberX`, `H4c_Wald_Joint_Weather`       | Thực hiện kiểm định kép (tham số + phi tham số + Effect Size) khóa đúng mẫu sạch (`weather_matched == 1` / `is_timestamp_original == 1`); dùng sai số chuẩn vững `HC3` & phương pháp Delta cho `95% CI` điểm cắt `d* = 3.10` mi (`[2.94, 3.26]`); khóa miền `≤ 7.86` mi để không ngoại suy `37.61` mi (`HIGH-02`).              |
|    5 | `src/models.py` (`L285`)                            | `build_model_registry()`                                      | Đóng gói `10` mô hình (`M0a–M7`) trong `sklearn.pipeline.Pipeline` với `SimpleImputer(median)` fit cô lập trong từng fold/tập Train (chống rò rỉ dữ liệu 100%), so sánh từ Baseline (`M0a/M0b`), tuyến tính (`M1–M4`, `M2-Log Duan Smearing`) đến cây (`M5–M7`).                                                                |
|    6 | `src/interpretation.py` (`L109`)                    | Phân hoạch rời `14` nhóm Permutation Importance               | Xáo trộn đồng thời `14` nhóm đặc trưng rời nhau phủ đúng `27/27` cột thực của `X_test` (`assert` không bỏ sót, không cột ảo, không đếm trùng tương tác — `HIGH-01`), chứng minh `name` (`71.62%`) + `distance × surge` (`24.26%`) chiếm `95.88%`, còn 8 nhóm thời tiết/giờ chỉ chiếm `0.01%`.                                   |
|    7 | `src/simulation.py` (`L156`, `L176`)                | `m2_ols_mean_saving`, `economic_assumptions`                  | Mô phỏng phản thực tế `6` ngưỡng trần Surge Cap (`3.00x → 1.00x`) trên `130,421` lượt báo giá Lyft với 3 độ co giãn cầu ($\varepsilon \in \{0.00, -0.50, -0.80\}$), kẹp cột đối chứng `M2 OLS` để bộc lộ hiện tượng **Tree Saturation** của `M7` ở trần `2.00x` (`$5.20` vs `$5.77`) và gắn nhãn `API Quote Proxy` (`HIGH-03`). |
|    8 | `dashboard/validation.py` (`L463`, `L623`)          | `VAL-11_CRIT01`, `run_full_data_validation()`                 | Cổng kiểm toán `17` chốt chặn độc lập (`VAL-01..12`, `TBL-01..05`) xác minh schema `30` cột, cỡ mẫu `≥ 100k`, `0` trùng khóa chính, `0` fan-out, `CRIT-01`, `CRIT-02`, `HIGH-01/03`; tự ném lỗi dừng app nếu phát sinh vi phạm `CRITICAL`.                                                                                      |
|    9 | `dashboard/business_logic.py` (`L64`, `L369`)       | `apply_global_filters()`, `compute_weather_rain_comparison()` | Tầng nghiệp vụ tách biệt 100% khỏi Streamlit UI; xử lý an toàn lát cắt `0` dòng (`Uber + Surge > 1.0`) không crash; cưỡng chế cứng `weather_matched == 1` và `is_timestamp_original == 1` ngay trong hàm tính KPI/biểu đồ.                                                                                                      |
|   10 | `dashboard/model_loader.py` (`L77`, `L133`)         | `load_trained_models()`, `validate_prediction_inputs()`       | Tự gắn `src/` vào `sys.path` để nạp thành công `m2_interaction_ols.joblib` & `best_model.joblib` từ bất kỳ `cwd` nào; tự kiểm tra miền `[0.02, 7.86]` dặm & danh sách `72` tuyến Boston để phát cảnh báo `EXTRAPOLATION WARNING`.                                                                                               |
|   11 | `dashboard/charts.py` (`L109`)                      | `build_executive_uber_vs_lyft_fig()`                          | Hàm thứ 19 trong bộ Plotly thuần dùng chung 100% giữa Streamlit & Offline HTML: gộp 2 góc nhìn so sánh Uber vs. Lyft (giá TB `5` cặp đối xứng + đường chênh lệch `Lyft − Uber` theo cự ly kèm điểm cắt `d* = 3.10` mi) vào đúng 1 biểu đồ chủ đạo Tab 1.                                                                        |
|   12 | `dashboard/export_static_html.py` (`L160`)          | `export_interactive_html()`                                   | Biên dịch tệp báo cáo 5 Tab `reports/dashboard_interactive.html` (`5.26 MB`) nhúng `plotly.js` nội tuyến đúng 1 lần (`0` thẻ CDN ngoài), đồng bộ 1:1 bố cục Executive 5 khối và tích hợp bộ ước lượng `HC3 OLS` + mô phỏng Surge Cap chạy thuần JavaScript.                                                                     |

### B. Tổng hợp Toàn bộ `12` Thí nghiệm *Mutate & Break* Thực nghiệm (Mốc `v3/v4` ngày `21/11`)

#### Nhóm 1 — `6` Thí nghiệm Phá vỡ Mô hình & Pipeline (`reports/tables/stage5_mutate_break.csv`)

| Mã thí nghiệm | Hành động phá vỡ (*Mutation Action*)                                             | Baseline (`R²` / `RMSE` / `MAPE`) | Sau phá vỡ (`R²` / `RMSE` / `MAPE`) | Bài học khoa học rút ra                                                                                                     |
| ------------- | -------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **`MB1`**     | Xóa tương tác `C(name)*distance` & `C(name)*log_distance` (`M2 → M1`)            | `0.9456` / `$2.1755` / `9.46%`    | `0.9166` / `$2.6944` / `13.38%`     | `MAPE` vọt `+3.92%`, trượt ngưỡng SMART `≤ 10%`: chứng minh mỗi phân khúc xe có đơn giá mỗi dặm riêng biệt.                 |
| **`MB2`**     | Tăng phạt L2 của Ridge lên cực đoan (`alpha = 100,000`)                          | `0.9456` / `$2.1755` / `9.46%`    | `0.6736` / `$5.3310` / `35.81%`     | Hệ số bị ép co sát `0` gây *Underfitting* nặng (`RMSE` tăng hơn `2.4×`).                                                    |
| **`MB3`**     | Bỏ giới hạn độ sâu cây quyết định (`max_depth=None`, `min_samples_leaf=1`)       | `0.9497` / `$2.0920` / `8.74%`    | `0.9144` / `$2.7305` / `10.48%`     | *Overfitting* điển hình: `Train R² = 0.9903` nhưng `Test R²` rớt mạnh xuống `0.9144`.                                       |
| **`MB4`**     | Xóa biến `surge_multiplier` (đánh giá riêng nhóm chuyến Lyft `surge > 1.0`)      | `0.9733` / `$2.2697` / `6.06%`    | `0.4801` / `$10.0215` / `23.76%`    | Trên tập sốc giá, thiếu `surge` làm `R²` sụp đổ từ `0.9733 → 0.4801` (`RMSE` tăng `4.4×`).                                  |
| **`MB5`**     | Thay `OneHotEncoder` bằng `OrdinalEncoder/LabelEncoder` cho tên gói xe trong OLS | `0.9166` / `$2.6944` / `13.38%`   | `0.4976` / `$6.6136` / `40.07%`     | Trả lời trực tiếp câu hỏi Vòng 3 trong `projects.pdf`: gán thứ tự số học giả tạo `0..11` làm OLS sụp đổ (`MAPE 40.07%`).    |
| **`MB6`**     | Đổi tham số `test_size` từ `0.20` (80/20) sang `0.40` (60/40)                    | `0.9531` / `$2.0213` / `8.55%`    | `0.9529` / `$2.0303` / `8.52%`      | Trả lời trực tiếp bài kiểm tra đổi `test_size` Vòng 3: nhờ cỡ mẫu lớn `270,898`, `M7` bất biến tuyệt đối (`ΔR² = -0.0002`). |

#### Nhóm 2 — `6` Thí nghiệm Tiêm lỗi Kiểm toán Dashboard (`dashboard/tests/test_validation.py`)

|  STT | Test Case Tiêm lỗi (*Mutation Test*)                                | Lỗi cố tình tiêm vào dữ liệu                                    | Cơ chế bắt lỗi xác minh (`pytest`)                       | Kết quả  |
| ---: | ------------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------- | -------- |
|    7 | `test_validation_catches_missing_required_column`                   | Xóa 1 cột lõi bắt buộc khỏi `master`                            | `VAL-01_SCHEMA` phát hiện `CRITICAL` và chặn khởi chạy   | **PASS** |
|    8 | `test_validation_catches_crit01_fake_rain_label`                    | Gán `is_raining = 1.0` trên dòng `weather_matched == 0`         | `VAL-11_CRIT01_WEATHER_MATCH` bắt đúng vi phạm `CRIT-01` | **PASS** |
|    9 | `test_validation_catches_crit02_ffill_timestamp_leak`               | Điền giá trị `datetime` trên dòng `is_timestamp_original == 0`  | `VAL-10_CRIT02_TIMESTAMP_EST` bắt đúng rò rỉ `ffill`     | **PASS** |
|   10 | `test_validation_catches_duplicate_primary_key`                     | Nhân đôi 1 khóa chính `id`                                      | `VAL-05_PK_DUPLICATES` bắt đúng trùng khóa (`CRITICAL`)  | **PASS** |
|   11 | `test_validation_catches_invalid_price_bound`                       | Chèn mức giá âm (`price = -5.0`)                                | `VAL-06_BOUNDS_OUTLIERS` báo lỗi miền giá không hợp lệ   | **PASS** |
|   12 | `test_result_tables_validation_catches_corrupted_permutation_table` | Làm lệch tổng tỷ trọng bảng `stage6_permutation_importance.csv` | `TBL-03_HIGH01_PERMUTATION` phát hiện sai lệch `≠ 100%`  | **PASS** |

### C. Sổ Theo dõi *Bug Hunting & Validate & Refine* (`14` Lỗi Thực tế Đã Phát hiện & Xử lý)

|    # | Giai đoạn | Lỗi / Bẫy thực tế phát hiện                                                      | Nguyên nhân gốc                                                             | Cách khắc phục triệt để & Bằng chứng kiểm chứng                                                                         |
| ---: | --------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
|    1 | GĐ3       | Giá rẻ bị nhân $10\times$ (`$70–$95`) lọt dưới ngưỡng cắt cứng `$95`             | Dùng ngưỡng cắt tĩnh không phân tầng theo tuyến/gói                         | Chuyển sang khôi phục $\log_{10}(\text{val}/\text{ref})$ theo `(source, destination, name)` → `M7 R²: 0.8508 → 0.9531`. |
|    2 | GĐ4       | Đa cộng tuyến cực nặng `VIF(distance) = 177.05`                                  | Đưa chung `distance` và `distance * surge` khi `97.68%` dòng `surge = 1.0`  | Tâm hóa thành `lyft_surge_excess = (surge - 1.0) * is_lyft` → `VIF = 1.02`.                                             |
|    3 | GĐ4       | Bẫy rò rỉ mục tiêu `price_per_mile` (`R4`)                                       | Biến tạo từ `price / distance` đẩy `R²` giả lên `0.9980`                    | Khóa chặn tuyệt đối trong `FORBIDDEN_IN_X` tại `src/config.py`.                                                         |
|    4 | GĐ4       | `M1 Basic OLS` trượt chỉ tiêu `MAPE ≤ 10%` (`13.38%`, `R5`)                      | Dùng chung 1 độ dốc `β₁` cho cả 12 gói xe                                   | Xây dựng `EconomicInteractionPreprocessor` → `M2 OLS MAPE = 9.46%`.                                                     |
|    5 | Audit     | **`CRIT-01`**: `37,536` dòng mất khớp thời tiết bị gắn nhãn *"Có mưa"*           | `fillna(mean)` cho cột `rain` trước khi gán `is_raining = (rain > 0)`       | Gắn cờ `weather_matched`, khóa `n = 179,280`, `0` nhãn mưa giả (`H1a p = 0.9467`).                                      |
|    6 | Audit     | **`CRIT-02`**: Lệch múi giờ UTC `+5h` & `ffill` `62,864` dòng thiếu TS           | Không truyền `tz_convert("America/New_York")` & lạm dụng `.ffill().bfill()` | Chuyển chuẩn Boston EST, giữ `pd.NA` + cờ `is_timestamp_original` (`n = 208,034`).                                      |
|    7 | Audit     | **`HIGH-01`**: Permutation Importance bỏ sót 3 cột thực, đếm trùng tương tác     | Danh sách cột viết tay chứa 5 cột ảo và ghép chồng biến tương tác           | Phân hoạch rời đúng `14` nhóm = `27/27` cột thực kèm `assert` tự động.                                                  |
|    8 | Audit     | **`HIGH-02`**: Ngoại suy điểm cắt `37.61` mi & ghép `WAV vs Lyft`                | Chia tuyến tính không kiểm tra miền `[0.02, 7.86]` mi                       | Khóa miền `≤ 7.86` mi (`Shared` không cắt nội miền), chuẩn hóa `5` cặp + tách `WAV`/`Lux`.                              |
|    9 | Audit     | **`HIGH-03`**: Gọi `API Quote Proxy` là "doanh thu thực" & thiếu đối chứng `M2`  | Đồng nhất 130k truy vấn API phân tầng với chuyến hoàn thành                 | Đổi toàn bộ nhãn sang `API Quote Proxy` / `SCENARIO`, thêm cột `M2 OLS` & 3 giả định kinh tế.                           |
|   10 | Audit     | **`MED-01..04` & `LOW-01`**: `max(rain) = 1.0`, lỗi 13 hình PNG, chuỗi hardcode  | Chia cứng `/10`, dùng `np.clip` ở histogram phần dư, hardcode số cũ         | Sửa toàn bộ 13 hình (duyệt trực quan từng PNG), `4/4` Notebook chạy sống `PASS`, `0` số cũ.                             |
|   11 | GĐ1 DB    | `AttributeError: 'Index' object has no attribute 'mean'` ở `data_loader.py:63`   | Thay đổi kiểu trả về trên `pandas 3.0.3` cho `cat.categories.str.len()`     | Đổi sang trung bình Python thuần → `15/15` tests PASS.                                                                  |
|   12 | GĐ3 DB    | `AppTest` rerun chậm (`21.15s`) khi cache 41 MB bằng `@st.cache_data`            | `@st.cache_data` pickle/hash toàn bộ DataFrame mỗi lượt rerun               | Tách `@st.cache_resource` cho dữ liệu/mô hình read-only + `@st.cache_data` cho `FilterParams`.                          |
|   13 | GĐ7 DB    | Cảnh báo startup Streamlit từ `enableCORS = false` & test bắt nhầm `cdn.plot.ly` | `enableCORS` xung đột cấu hình mới; chuỗi nội bộ trong `plotly.js`          | Gỡ `enableCORS` khỏi `.streamlit/config.toml`; kiểm tra chính xác thẻ `<script src>`/`<link>` ngoài.                    |
|   14 | GĐ7 DB    | `README.md` gốc còn sót 7 số liệu trước Final Audit (`RSK-01`)                   | Chưa cập nhật lại sau đợt sửa ngày `2026-11-19`                             | Viết lại toàn bộ `README.md` đối soát thẳng từ 4 CSV gốc → `test_root_readme_synchronized_post_audit` PASS.             |
- **Tồn đọng & Bước tiếp theo:**
  - Chuyển sang biên soạn và xuất bản Toàn văn Tiểu luận 5 chương chuẩn UFM / APA 7th ở ca trưa cùng ngày.

---

## [2026-11-19 10:17] - Phiên 12: Khắc Phục Toàn Diện Final Audit Lần 1 & Lần 2 (`CRIT-01`, `CRIT-02`, `HIGH-01..03`, `MED-01..04`, `LOW-01`)

- **Thành viên phụ trách chính:** **Trần Thanh Huy** (MSSV: `2521001060`)
- **Giai đoạn:** Kiểm định độc lập & Khắc phục toàn diện (*Final Audit Round 1 → Remediation → Final Audit Round 2*) trước khi khởi công Dashboard & Tiểu luận.
- **Môi trường:** Windows 64-bit | `Python 3.14.4` (`pandas 3.0.3`, `numpy 2.5.0`, `scipy 1.18.1`, `statsmodels 0.15.0`, `scikit-learn 1.9.1`, `quarto typst`).
- **Mục tiêu phiên:**
  - Sao lưu toàn trạng trước sửa (`92` tệp vào `archive/backup_pre_audit_fix_20261119/`) và khóa kiểm chứng toàn vẹn dữ liệu gốc `data/raw/` bằng hàm băm `SHA-256`.
  - Xử lý triệt để toàn bộ **10 nhóm lỗi kiểm toán** (`CRIT-01`, `CRIT-02`, `HIGH-01`, `HIGH-02`, `HIGH-03`, `MED-01`, `MED-02`, `MED-03`, `MED-04`, `LOW-01`) phát hiện ở *Final Audit Lần 1*.
  - Tái thực thi toàn bộ pipeline `Stage 3 → Stage 4 → Stage 5 → Stage 6`, kiểm chứng chạy sống `4/4` Notebook (`03`–`06`), duyệt trực quan toàn bộ `13` biểu đồ PNG đã chỉnh sửa và lập báo cáo nghiệm thu `reports/Nhat_Ky_Khac_Phuc_Va_Final_Audit_Lan_2.md`.
- **Tương tác Prompt & Quyết định kỹ thuật (P-D-M-V):**
  - *Prompt chiến lược đã thực thi:* "Thực hiện sao lưu an toàn, khắc phục đồng bộ toàn bộ nhóm lỗi Critical/High/Medium/Low từ báo cáo Final Audit Lần 1, chạy lại toàn bộ pipeline GĐ3–GĐ6, kiểm chứng lại từng tệp PNG và Notebook, sau đó thực hiện Final Audit Lần 2 độc lập."
  - *Bảo vệ dữ liệu gốc (SHA-256 `MATCH = True` 100%):*
    * `data/raw/cab_rides.csv`: `3987E0532C06E3D23B62E9F5C483BAC70A9331DE5BB593BA80C75E0A961ACA9A` (không đổi 1 bit).
    * `data/raw/weather.csv`: `F6253513970DEACE50C198EF71ED05FFA31B83F10893CC9FB078A2B7FF34125F` (không đổi 1 bit).
- **Nội dung đã hoàn thành:**
  - **Bảng Đối soát Trước vs. Sau Khắc phục (Chi tiết tại `reports/Nhat_Ky_Khac_Phuc_Va_Final_Audit_Lan_2.md`):**

| Mã lỗi                          | Tệp mã nguồn đã sửa                                                                                  | Hiện trạng Trước sửa (Audit Lần 1)                                                                                                             | Kết quả Sau sửa & Kiểm chứng (Audit Lần 2)                                                                                                                                                                                           | Trạng thái |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| **`CRIT-01`**                   | `src/stage3_pipeline_and_audit.py`, `src/config.py`, `src/features.py`, `src/stats_tests.py`         | `fillna(mean)` trên cột `rain` làm `37,536` dòng mất khớp thời tiết bị gắn nhãn *"Có mưa"* (`n_rain = 69,348`, rò rỉ xuyên Train/Test).        | Tách cờ `weather_matched == 1` (`n = 179,280`, `86.18%` TS gốc), `0` nhãn mưa giả trên `91,618` dòng mất khớp; `H1a` sạch: Mưa `$16.5859` vs Tạnh `$16.5902` (`p = 0.9467`, `d = -0.00046`), `H4c` `ΔR² = 0.0000016` (`p = 0.8344`). | **ĐẠT**    |
| **`CRIT-02`**                   | `src/clean_raw_data.py`, `src/stage3_pipeline_and_audit.py`, `src/features.py`, `src/stats_tests.py` | `time_stamp` ở múi giờ UTC (lệch `+5h` so với Boston EST) và dùng `.ffill().bfill()` điền giả `62,864` dòng thiếu TS (`23.21%`).               | Chuyển chuẩn `America/New_York` (EST), giữ `pd.NA` + cờ `is_timestamp_original == 1` (`n = 208,034`), `0` dòng `ffill`; khôi phục đúng đáy lưu lượng `03:00` & đỉnh `19:00` EST; `Temporal EST` `M7 R² = 0.9546, RMSE = $1.9945`.    | **ĐẠT**    |
| **`HIGH-01`**                   | `src/interpretation.py`, `src/stage6_evaluation.py`                                                  | Permutation liệt kê `5` cột ảo, thiếu `3` cột thực (`is_raining`, `dow_sin`, `dow_cos`) và đếm trùng biến tương tác ở 2 nhóm.                  | Phân hoạch rời đúng **`14` nhóm = `27/27` cột thực `X_test`** (`assert` tự động): `name` `71.62%` + `distance × surge` `24.26%` = `95.88%`; 8 nhóm thời tiết/giờ = `0.01%`.                                                          | **ĐẠT**    |
| **`HIGH-02`**                   | `src/stats_tests.py`, `src/interpretation.py`, `src/stage3_pipeline_and_audit.py`                    | Ngoại suy điểm cắt `UberPool vs. Shared` tại `37.61` dặm (ngoài miền `7.86` mi); ghép khập khiễng `WAV vs Lyft`.                               | Khóa miền `[0.02, 7.86]` dặm (`0` lần xuất hiện `37.61` trong báo cáo); thống nhất `5` cặp đối xứng (`UberX/Lyft 3.10`, `XL 3.75`, `Black 0.23`, `Black SUV 1.08` mi) & tách riêng `WAV`, `Lux`.                                     | **ĐẠT**    |
| **`HIGH-03`**                   | `src/simulation.py`, `src/stage6_evaluation.py`, `src/build_stage6_deliverables.py`                  | Đồng nhất `$2,267,293.70` với "doanh thu thực", thiếu đối chứng `M2 OLS` tại `Cap 2.00x`, `KN5` nêu `12%–22%` thiếu phép tính.                 | Chuẩn hóa thuật ngữ **`API Quote Proxy`**, thêm cột `M2 OLS` (`Cap 2.00x`: `M7 $5.20` vs `M2 $5.77`; `Cap 1.75x`: `$4.60` vs `$5.02`, `-0.081%`), bổ sung `3` giả định kinh tế & gỡ bỏ `12%–22%`.                                    | **ĐẠT**    |
| **`MED-01..04`** & **`LOW-01`** | `src/clean_raw_data.py`, `src/features.py`, `src/models.py`, `src/build_stage3..6_deliverables.py`   | `max(rain) = 1.000`, thiếu đơn vị đo 39 biến, lỗi hiển thị trên `13` PNG (`np.clip` tạo đỉnh giả, trục đôi, che chữ), hardcode chuỗi kết luận. | `max(rain) = 0.7807` in/h; xuất `data_dictionary_complete.csv` đủ `39/39` cột có `unit`; sửa & kiểm tra mắt `13/13` PNG; `4/4` Notebook chạy sống `PASS`; `0` số liệu cũ trong `Giai_Doan_3..6*.md`.                                 | **ĐẠT**    |

- **Kết quả kiểm định độc lập Final Audit Lần 2 (Mốc `v3` ngày `19/11`):**
  - Toàn bộ **`9/9` hạng mục** chuyển trạng thái sang **`ĐẠT` (`READY`)**.
  - `master_rideshare.parquet` nâng chuẩn lên `270,898 × 30` (`41.04 MB` RAM); hiệu năng tập Test `n = 54,180` giữ nguyên vững chắc sau khi loại rò rỉ: `M7` (`R² = 0.9531, RMSE = $2.0213, MAE = $1.1934, MAPE = 8.55%`), `M2` (`R² = 0.9456, RMSE = $2.1755, MAPE = 9.46%`), `Quantile [P10, P90]` (`PICP = 79.20%, MPIW = $3.48`).
  - Đã kiểm chứng và khóa đồng bộ toàn bộ mã nguồn, dữ liệu sạch, báo cáo và mô hình sau khắc phục Final Audit Round 2.
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Tách cờ `weather_matched == 1` (`CRIT-01`) loại bỏ `37,536` nhãn mưa giả (`H1a p = 0.9467`); giữ `pd.NA` + `is_timestamp_original` (`CRIT-02`) khôi phục đúng đáy `03:00` và đỉnh `19:00` EST; phân hoạch rời `14` nhóm = `27/27` cột thực (`HIGH-01`).
  - *Quyết định kiến trúc:* Khóa miền ngoại suy `[0.02, 7.86]` dặm (`HIGH-02`), chuẩn hóa thuật ngữ `API Quote Proxy` kèm đối chứng `M2 OLS` (`HIGH-03`) và lưu bản sao toàn trạng trước sửa vào `archive/backup_pre_audit_fix_20261119/`.
- **Tồn đọng & Bước tiếp theo:**
  - Khởi động kế hoạch xây dựng **Dashboard Phân tích Tương tác (`Python + Streamlit + Plotly`) + Báo cáo HTML Offline** theo quy trình 7 giai đoạn (bắt đầu từ GĐ0 ở Phiên 13).
- **Liên kết mã nguồn cốt lõi (bản báo cáo nội bộ `reports/Nhat_Ky_Khac_Phuc_Va_Final_Audit_Lan_2.md` lưu ngoài gói `.zip`):** [src/stage3_pipeline_and_audit.py](src/stage3_pipeline_and_audit.py) · [src/interpretation.py](src/interpretation.py) · [src/simulation.py](src/simulation.py) · [data/processed/master_rideshare.parquet](data/processed/master_rideshare.parquet)

---

## [2026-11-16 10:40] - Phiên 11: Cô Lập Nhánh Tiền Nhiệm Vào `archive/` & Vá Đồng Bộ Mã Nguồn – Báo Cáo GĐ3–GĐ6

- **Thành viên phụ trách chính:** **Lê Huy Thắng** (MSSV: `2521001092`)
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`pandas 3.0.6`, `numpy 2.4.6`, `scipy 1.17.1`, `statsmodels 0.15.0`, `scikit-learn 1.9.1`, `nbformat 5.11.1`, `quarto typst`; `pyarrow` qua `uv`).
- **Mục tiêu phiên:**
  - Hoàn tất toàn bộ các điểm cần chỉnh sau **Tổng kiểm toán & Bàn giao dự án** trước khi viết Tiểu luận Giai đoạn 7 (xuất bản Word `.docx` + `PDF`):
    1. Cô lập toàn bộ tài liệu/code/notebook/ảnh nhánh Tiền nhiệm (*Legacy* `637,976` dòng), cẩm nang GĐ1–GĐ2 đầu kỳ và 3 script thăm dò `scratch/*.py` có đường dẫn cứng vào `archive/`.
    2. Vá toàn bộ sai lệch văn bản gõ tay và lỗi trường số liệu nhỏ phát hiện trong `src/` và báo cáo `GĐ3–GĐ6`, tái chạy kiểm chứng `H1–H4`, `main.py --stage evaluate` và biên dịch lại trọn bộ `.ipynb`, `.md`, `.typ`, `.pdf`.
- **Nội dung đã hoàn thành:**
  - **1. Cô lập & Dọn sạch Không gian làm việc (`38` mục chuyển vào `archive/`):**
    * Cô lập nhánh Legacy Epoch 1 vào `archive/legacy_epoch1/` (gồm `data_loader.py`, `preprocessing.py`, `build_notebook_01..02.py`, `01_EDA_storytelling.ipynb`, `02_model_training.ipynb`, `01_ke_hoach_...md`, `README_legacy.md`, `MSSV_2521001060_TieuLuan.*`, `2521001060_Nhom02_TieuLuan.*`, `01_bao_cao_data_understanding.*`, `9` ảnh `01..09*.png` cũ).
    * Cô lập cẩm nang GĐ1–2 và 3 script thăm dò có đường dẫn cứng (`deep_audit.py`, `inspect_raw_data.py`, `test_audit_stats.py`) vào `archive/stage1_2_guides_and_forensic/`.
    * Xóa tệp thử nghiệm rác `test_typst.typ` và thư mục rỗng `models/.vscode/`; tạo mới `archive/README.md` và cập nhật [README.md](README.md) phản ánh đúng 15 mô-đun chính thức.
  - **2. Vá đồng bộ Mã nguồn, Số liệu & Báo cáo GĐ3–GĐ6:**
    * **Giai đoạn 3** (`reports/Giai_Doan_3_Bao_Cao_Ky_Thuat_Data_Prep_EDA_Viz.md`, `scratch/stage3_doc.typ`, `src/build_stage3_deliverables.py`): Đồng bộ 100% theo `stage3_audit_summary.json` & `master_rideshare.parquet` (`270,898 × 28`, `2,459` price & `2,552` distance outliers, `263,497` missing price, `233,362` khớp thời tiết, `12,238`/`6,337` scale-fixed, `22,262` distance imputed, `Skew = +1.0517`, `72` tuyến liên quận thực tế).
    * **Giai đoạn 4** (`src/build_stage4_deliverables.py`): Sửa lỗi thoát chuỗi `\t` trong công thức LaTeX `\\text{VIF}` và `\\text{SMD}`, biên dịch lại `04_*.ipynb` và `Giai_Doan_4_Bao_Cao_Ky_Thuat.md/.pdf`.
    * **Giai đoạn 5** ([src/stats_tests.py](src/stats_tests.py), `src/build_stage5_deliverables.py`): Chuyển `H3a` sang f-string động `V = 0.1599`; sửa lỗi in cột `cv_r2_std` (trước đây in nhầm `cv_rmse_std`) trong bảng CV; thay toàn bộ đoạn prose độ bền hardcode sai (`48,011`, `R2 > 0.96`) bằng số đọc động chuẩn từ `stage5_robustness.csv` (`n = 47,965`, `M7 R3a R2 = 0.9536, MAPE = 8.51%`, `route_group_split M7 R2 = 0.9246, MAPE = 10.66%`).
    * **Giai đoạn 6** ([src/stage6_evaluation.py](src/stage6_evaluation.py), `src/build_stage6_deliverables.py`): Sửa `RQ1` thành động (`tổng nhóm thời tiết & khung giờ: 0.03%`), động hóa phụ đề `fig20` từ `sim_df`, và sửa nhận xét sai về `Residual Median` & `Skewness = 4.6825` theo đúng `84.65%` nằm trong `±$2.00`.
  - **3. Xác minh Sau Chỉnh sửa:** Tái chạy kiểm định `H1–H4`, tái chạy `python main.py --stage evaluate` (exit code `0`), tái biên dịch 4 bộ Notebook/Markdown/Typst/PDF (`GĐ3–GĐ6`), kiểm tra `0` đường dẫn cứng và `0` tab lỗi ngoài `archive/`.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Cô lập toàn bộ 38 mục Legacy Epoch 1 & script thăm dò vào archive/, đồng bộ số liệu động giữa CSV và báo cáo GĐ3–GĐ6."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Sửa lỗi thoát chuỗi `\t` trong công thức LaTeX (`\\text{VIF}`), sửa lỗi in nhầm cột `cv_rmse_std` thành `cv_r2_std`, và sửa nhận xét `Residual Median` / `Skewness` theo đúng số liệu thực nghiệm.
  - *Quyết định kiến trúc:* Giữ thư mục chính chỉ chứa pipeline sạch và thay toàn bộ chuỗi hardcode bằng f-string đọc từ CSV/JSON.
- **Tồn đọng & Bước tiếp theo:**
  - Chuyển sang **Giai đoạn 7**: Soạn thảo toàn văn **Tiểu luận Chính thức 5 Chương chuẩn APA 7th** (Word `.docx` và `PDF`).

---

## [2026-11-14 17:54] - Phiên 10: Hoàn Tất Giai Đoạn 5 & 6 (13 Kiểm Định H1–H4, 10 Mô Hình M0a–M7, Quantile P10–P90, XAI & Mô Phỏng Surge Cap; Ca chiều)

- **Thành viên phụ trách chính:** **Lê Huy Thắng** (MSSV: `2521001092`)
- **Giai đoạn:** Giai đoạn 5 (Modeling & Hypothesis Testing) + Giai đoạn 6 (Evaluation, XAI & Policy Simulation), ca chiều.
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`pandas 3.0.6`, `numpy 2.4.6`, `scipy 1.17.1`, `statsmodels 0.15.0`, `scikit-learn 1.9.1`, `joblib 1.6.0`; `pyarrow` qua `uv`)
- **Mục tiêu phiên:**
  - *(Phần I — GĐ5)* Thực hiện 13 kiểm định kép `H1–H4`, ước lượng biểu giá cắt nhau `Wald HC3` (`UberX–Lyft d* = 3.10 mi`), huấn luyện & 5-Fold CV 10 cấu hình `M0a–M7` + kiểm định độ bền R3 (`88.65%`) + 6 thí nghiệm *Mutate & Break* (`MB1–MB6`).
  - *(Phần II — GĐ6)* Chẩn đoán phần dư nâng cao (`Jarque-Bera`, `Breusch-Pagan`), ước lượng dải bất định `Quantile [P10, P90]`, diễn giải `Permutation Importance` + `PDP/ICE`, và mô phỏng trần `Surge Cap` dưới cầu tĩnh/đàn hồi.
- **Nội dung đã hoàn thành:**
  - **Mã nguồn GĐ5–GĐ6:** [src/stats_tests.py](src/stats_tests.py), [src/models.py](src/models.py), [src/stage5_modeling.py](src/stage5_modeling.py), [src/evaluation.py](src/evaluation.py), [src/interpretation.py](src/interpretation.py), [src/simulation.py](src/simulation.py), [src/stage6_evaluation.py](src/stage6_evaluation.py).
  - **Bảng & Mô hình kết xuất:** 11 bảng CSV (`stage5_*.csv`, `stage6_*.csv`) kèm [models/best_model.joblib](models/best_model.joblib) (`M7`: `R² = 0.9532, RMSE = $2.0195, MAPE = 8.55%`), [models/m2_interaction_ols.joblib](models/m2_interaction_ols.joblib) (`M2`: `R² = 0.9456, RMSE = $2.1754, MAPE = 9.46%`), [reports/results/stage5_summary.json](reports/results/stage5_summary.json), [reports/results/stage6_summary.json](reports/results/stage6_summary.json).
  - **Báo cáo & Notebook:** [notebooks/05_Hypothesis_Testing_Modeling.ipynb](notebooks/05_Hypothesis_Testing_Modeling.ipynb), [notebooks/06_Evaluation_Interpretation_Policy.ipynb](notebooks/06_Evaluation_Interpretation_Policy.ipynb), `reports/Giai_Doan_5_Bao_Cao_Ky_Thuat.pdf`, `reports/Giai_Doan_6_Bao_Cao_Ky_Thuat.pdf`.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Chạy trọn GĐ5 & GĐ6: 13 kiểm định H1–H4 kèm Wald HC3, so sánh 10 mô hình M0a–M7 với 5-Fold CV và 6 thí nghiệm Mutate & Break; chẩn đoán phần dư, dải Quantile P10–P90, Permutation Importance + PDP/ICE và mô phỏng Surge Cap."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* 6 biến thời tiết có `ΔR² = 0.0000032` (`p = 0.6284`); PDP phát hiện vùng phẳng cước sàn `< 1.0` mi giải thích vì sao `M7` vượt `M2`; `MB1–MB6` chứng minh tác hại bỏ tương tác/dùng `LabelEncoder` (`MAPE 40.07%`).
  - *Quyết định kiến trúc:* Đóng gói cả `M7` (dự báo chính) lẫn `M2 OLS HC3` (diễn giải kinh tế lượng) và thiết lập công thức mô phỏng đa độ co giãn cầu $\varepsilon_d \in \{0, -0.5, -0.8\}$.
- **Tồn đọng & Bước tiếp theo:**
  - Cô lập toàn bộ nhánh Legacy Epoch 1 vào `archive/` và vá đồng bộ các sai lệch văn bản/báo cáo GĐ3–GĐ6 trước khi kiểm toán cuối và viết Tiểu luận.

---

## [2026-11-14 11:12] - Phiên 9: Hoàn Tất Toàn Diện Giai Đoạn 4 (Feature Engineering, Kiểm Toán Rò Rỉ, Khử VIF & Đóng Gói Pipeline)

- **Thành viên phụ trách chính:** **Trần Thanh Huy** (MSSV: `2521001060`)
- **Môi trường:** Windows 64-bit | `Python 3.14.4` (`pandas 3.0.3`, `numpy 2.5.0`, `scipy 1.18.1`, `statsmodels 0.15.0`, `scikit-learn 1.9.1`).
- **Mục tiêu phiên:**
  - Triển khai trọn vẹn **Giai đoạn 4 (Feature Engineering & Pipeline Preparation)** trên bảng dữ liệu sạch [data/processed/master_rideshare.parquet](data/processed/master_rideshare.parquet) (`270,898` dòng × `28` cột).
  - Xử lý dứt điểm các rủi ro kỹ thuật đã phát hiện: chuẩn hóa tên tệp nhật ký `./nhật ký làm việc.md` (R1), xây dựng CLI một lệnh `main.py` và `src/config.py` không chứa đường dẫn cứng (R2), kiểm toán tỷ lệ bản ghi nguyên bản không qua hiệu chỉnh quy mô $10^m$ (R3), khóa chặn biến rò rỉ mục tiêu `price_per_mile` (R4), và khắc phục việc OLS cơ bản trượt chỉ tiêu MAPE (R5).
- **Nội dung đã hoàn thành:**
  - **1. Tạo mới & chuẩn hóa kiến trúc mã nguồn:**
    * [requirements.txt](requirements.txt): Khóa phiên bản 11 thư viện cốt lõi.
    * [src/config.py](src/config.py): Quản lý tập trung toàn bộ đường dẫn tương đối (`Path(__file__).resolve().parent.parent`), `RANDOM_STATE = 42`, `TEST_SIZE = 0.20` và danh sách cấm `FORBIDDEN_IN_X`.
    * [main.py](main.py): Điểm vào CLI duy nhất hỗ trợ `--stage {clean,eda,features,model,evaluate,all}` và `--test-size` phục vụ trực tiếp bài kiểm tra *Mutate & Break* tại bàn chấm thi Vòng 3.
    * [src/features.py](src/features.py) & `src/build_stage4_deliverables.py`: Thực thi kiểm toán 37 cột, tạo 9 đặc trưng kinh tế & chu kỳ lượng giác, kiểm định VIF, thiết lập 3 chiến lược chia dữ liệu và đóng gói 4 nhánh `ColumnTransformer` / `EconomicInteractionPreprocessor`.
  - **2. Xuất bản bộ bảng biểu, hình ảnh 300 DPI, Notebook và Báo cáo Kỹ thuật Giai đoạn 4:**
    * Bảng CSV & Kết quả JSON: `reports/tables/stage4_feature_audit.csv`, [reports/tables/stage4_vif.csv](reports/tables/stage4_vif.csv), [reports/tables/stage4_split_summary.csv](reports/tables/stage4_split_summary.csv), [reports/results/stage4_summary.json](reports/results/stage4_summary.json).
    * Cờ kiểm toán nguyên bản R3: [data/processed/stage4_audit_flags.parquet](data/processed/stage4_audit_flags.parquet) (`240,139 / 270,898` dòng = `88.65%` nguyên bản tuyệt đối trên cả `price` và `distance`).
    * Hình ảnh 300 DPI: [fig08_feature_correlation_after_fe.png](reports/figures/stage4/fig08_feature_correlation_after_fe.png), [fig09_split_distribution.png](reports/figures/stage4/fig09_split_distribution.png), [fig10_pipeline_diagram.png](reports/figures/stage4/fig10_pipeline_diagram.png).
    * Notebook & Báo cáo: [notebooks/04_Feature_Engineering.ipynb](notebooks/04_Feature_Engineering.ipynb), `reports/Giai_Doan_4_Bao_Cao_Ky_Thuat.md`, `reports/Giai_Doan_4_Bao_Cao_Ky_Thuat.pdf`.
- **Tương tác Prompt & Quyết định kỹ thuật (Chuẩn P-D-M-V):**
  - *Prompt chiến lược đã thực thi:* "Triển khai Giai đoạn 4 theo kế hoạch đã duyệt: kiểm toán vai trò từng cột, chứng minh thực nghiệm bẫy rò rỉ price_per_mile, khử đa cộng tuyến cấu trúc bằng lyft_surge_excess, xây dựng EconomicInteractionPreprocessor và kiểm chứng trên 3 cách chia dữ liệu."
  - *Kinh nghiệm bóc tách code & Mutate/Break:*
    1. **Phát hiện & khử Đa cộng tuyến cấu trúc (VIF):** Khi dùng `distance_x_surge = distance * surge_multiplier` chung với `distance`, vì `97.68%` chuyến xe có `surge = 1.0`, hai cột này trùng nhau trên gần hết tập dữ liệu khiến $\text{VIF}(\text{distance}) = 177.05$ và $\text{VIF}(\text{distance\_x\_surge}) = 183.52$. Khi chuyển sang biến tâm hóa `lyft_surge_excess = (surge - 1.0) * is_lyft`, $\text{VIF}(\text{distance})$ giảm thẳng từ **`177.05` xuống `1.02`**!
    2. **Thí nghiệm Mutate & Break bẫy rò rỉ `price_per_mile` (R4):** Cố tình đưa `price_per_mile` vào `HistGradientBoosting` làm $R^2$ nhảy giả tạo từ `0.9529` lên `0.9980` và `MAPE` tụt xuống `1.45%`. Đã khóa chặn tự động trong `FORBIDDEN_IN_X`.
    3. **Khắc phục triệt để Rủi ro R5 cho mô hình tuyến tính:** Nhờ `EconomicInteractionPreprocessor` cho phép mỗi gói xe có cước mở cửa, đơn giá/dặm (`C(name):distance`), độ cong cự ly ngắn (`C(name):log_distance`) và độ nhạy surge riêng, mô hình tuyến tính `M2 (OLS tương tác)` tăng $R^2$ từ `0.9166` lên **`0.9456`**, giảm `RMSE` từ `$2.6943` xuống **`$2.1754`**, và giảm `MAPE` từ `13.38%` xuống **`9.46%`** (chính thức vượt mục tiêu SMART $\le 10\%$!).
  - *Quyết định kiến trúc:* Khóa danh sách biến cấm `FORBIDDEN_IN_X` trong `src/config.py`, tâm hóa `lyft_surge_excess` để khử đa cộng tuyến cấu trúc và đóng gói `ColumnTransformer` / `EconomicInteractionPreprocessor` tách biệt giữa các chiến lược chia dữ liệu.
- **Tồn đọng & Bước tiếp theo:**
  - Triển khai **Phiên 10 (Giai đoạn 5)**: Kiểm định thống kê chuyên sâu (bổ sung Chi-Square, Mann-Whitney, Kruskal-Wallis, Wald test HC3) và huấn luyện đầy đủ 9 mô hình (`M0a`, `M0b`, `M1`–`M7`) kèm 5-Fold Cross-Validation và kiểm tra độ bền trên `temporal_split`, `route_group_split` và tập con nguyên bản `88.65%`.

---

## [2026-11-08 19:30] - Phiên 8: Hoàn Tất Toàn Diện Giai Đoạn 3 (Data Preparation, `log10` Scale Recovery, EDA & 7 Biểu Đồ 300 DPI)

- **Thành viên phụ trách chính:** **Lê Huy Thắng** (MSSV: `2521001092`)
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`pandas 3.0.6`, `numpy 2.4.6`, `scipy 1.17.1`, `statsmodels 0.15.0`, `scikit-learn 1.9.1`, `matplotlib 3.11.2`, `seaborn 0.13.2`, `pillow 13.0.0`; `pyarrow` qua `uv`).
- **Mục tiêu phiên:**
  - Thực thi trọn vẹn 5 phần của Giai đoạn 3 trên bộ dữ liệu gốc ([data/raw/cab_rides.csv](data/raw/cab_rides.csv), [data/raw/weather.csv](data/raw/weather.csv)):
    1. Kiểm tra dữ liệu ban đầu, làm sạch 7 tầng bẫy nhiễu và đối soát Trước vs Sau xử lý.
    2. Thực hiện EDA có hệ thống (Thống kê mô tả, Phân phối, Ghép nối đa bảng chống nhân bản dòng, Tương quan & So sánh nhóm).
    3. Thiết kế bộ 7 biểu đồ trực quan hóa chuyên sâu (`300 DPI`) từ Nhóm A đến Nhóm F kèm phân tích 10 tiêu chí/biểu đồ.
    4. Trả lời 3 câu hỏi kinh doanh (RQ1, RQ2, RQ3) bằng kiểm định thống kê và mô hình học máy.
    5. Xuất bản Báo cáo Kỹ thuật (`.md`, `.pdf`), Notebook thực hành (`.ipynb`) và bộ dữ liệu sạch (`.parquet`, `.csv`).

- **Nội dung đã hoàn thành:**
  - **1. Cải tiến thuật toán khôi phục tỷ lệ quy mô bậc 10 ($\log_{10}$ Scale Recovery):**
    * Phát hiện bẫy nhiễu tinh vi: các cuốc xe giá rẻ (`Shared`, `UberX` giá `$7.0 - $9.5`) khi bị nhân $10\times$ thành `$70 - $95` nằm dưới ngưỡng cắt cứng `95.0 USD`, và `5,226` dòng `Uber Taxi` MNAR bị chèn mã lính canh (`-1, -100, 1000, 9999, 10000, 100000`).
    * Áp dụng công thức $\log_{10}(\text{val}/\text{ref})$ theo từng tuyến đường `(source, destination)` và từng phân khúc `name` trong [src/clean_raw_data.py](src/clean_raw_data.py), giúp nâng $R^2$ của OLS từ `0.7886` lên **`0.9166`** và `HistGradientBoosting` từ `0.8508` lên **`0.9527`** (`MAPE = 8.58%`).
  - **2. Đóng gói trọn bộ sản phẩm đầu ra Giai đoạn 3:**
    * Dữ liệu sạch: [data/processed/cab_rides_cleaned.parquet](data/processed/cab_rides_cleaned.parquet) (`270,898` dòng), [data/processed/weather_cleaned.csv](data/processed/weather_cleaned.csv) (`5,053` dòng), [data/processed/master_rideshare.parquet](data/processed/master_rideshare.parquet) (`270,898` dòng × `24` cột, giảm `53.79%` RAM).
    * Bộ 7 biểu đồ `300 DPI`: Lưu tại [reports/figures/stage3/](reports/figures/stage3/).
    * Notebook & Báo cáo kỹ thuật: [notebooks/03_Giai_Doan_3_Data_Prep_EDA_Visualization.ipynb](notebooks/03_Giai_Doan_3_Data_Prep_EDA_Visualization.ipynb), `reports/Giai_Doan_3_Bao_Cao_Ky_Thuat_Data_Prep_EDA_Viz.md` và `reports/Giai_Doan_3_Bao_Cao_Ky_Thuat_Data_Prep_EDA_Viz.pdf`.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Nâng cấp khôi phục quy mô log10 theo từng cặp (route, name), ghép bảng chống fan-out và hoàn thiện Giai đoạn 3."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Phát hiện ngưỡng cắt cứng `$95` bỏ sót các cuốc `Shared`/`UberX` bị nhân $10\times$ (`$70–$95`) và phát hiện `5,226` dòng `Uber Taxi` MNAR chứa mã lính canh (`-1, -100, 1000, 9999, 10000, 100000`).
  - *Quyết định kiến trúc:* Thay ngưỡng cắt cứng $95 bằng log10(val/ref) theo tuyến & gói dịch vụ giúp OLS R² tăng từ 0.7886 lên 0.9166.
- **Tồn đọng & Bước tiếp theo:** Chuyển sang Giai đoạn 4 (Feature Engineering, trực giao hóa VIF và đóng gói ColumnTransformer).

---

## [2026-10-22 11:15] - Phiên 7: Khởi Động Lại Từ Đầu, Dọn Dẹp Môi Trường & Quy Hoạch Dữ Liệu Cần Giữ Lại

- **Thành viên phụ trách chính:** **Trần Thanh Huy** (MSSV: `2521001060`)
- **Môi trường:** Windows 64-bit | `Python 3.14.4`
- **Mục tiêu phiên:**
  - Kiểm kê, phân loại toàn diện 54 tệp trong không gian làm việc `tiểu_luận/`: Xác định rõ danh mục tệp cốt lõi bắt buộc giữ lại và danh mục tệp rác/phái sinh cần xóa dọn.
  - Thiết lập lại trạng thái sẵn sàng để thực thi quy trình từ Giai đoạn 3 (Data Preparation & Data Wrangling).

- **Nội dung đã hoàn thành:**
  - **1. Kiểm kê & Phân loại tài nguyên (`54` tệp):** Giữ nguyên toàn bộ dữ liệu gốc [data/raw/](data/raw/), các module nòng cốt tại [src/](src/), tài liệu kế hoạch `01_ke_hoach_va_quy_trinh_nghien_cuu.md`, [README.md](README.md), `projects.pdf` và báo cáo tại [reports/](reports/); xóa tệp nháp thử nghiệm trong `scratch/` và làm trống [data/processed/](data/processed/) để tái chạy sạch từ Giai đoạn 3.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Kiểm kê 54 tệp dự án, phân loại danh mục cốt lõi giữ lại và dọn sạch dữ liệu trung gian để tái chạy từ Giai đoạn 3."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Rà soát phân biệt tệp trung gian phái sinh (`data/processed/`, `scratch/`) với dữ liệu gốc `data/raw/` và các module `src/` để làm sạch trạng thái tái chạy mà không mất lịch sử.
  - *Quyết định kiến trúc:* Bảo toàn nguyên vẹn dữ liệu gốc `data/raw/` và các module `src/` nòng cốt trước khi tái thiết kế pipeline sạch chuẩn.
- **Tồn đọng & Bước tiếp theo:**
  - Sửa đường dẫn chuẩn trong [src/clean_raw_data.py](src/clean_raw_data.py) và chạy pipeline làm sạch xuất dữ liệu sang [data/processed/](data/processed/).

---

## [2026-10-20 09:45] - Phiên 6: Tái Khởi Động & Đối Soát Chuyên Sâu Quy Trình Làm Sạch Dữ Liệu Từ Đầu

- **Thành viên phụ trách chính:** **Lê Huy Thắng** (MSSV: `2521001092`)
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`pandas 3.0.6`, `numpy 2.4.6`)
- **Mục tiêu phiên:**
  - Hệ thống hóa tiến độ tiểu luận, đối chiếu 2 chiều giữa dữ liệu thô bị làm dơ ([data/raw/cab_rides.csv](data/raw/cab_rides.csv), [data/raw/weather.csv](data/raw/weather.csv)) với dữ liệu sau làm sạch và lập danh mục tái lập `data/processed/`.
- **Nội dung đã hoàn thành:**
  - Lập bảng đối chiếu từng trường dữ liệu trước/sau làm sạch (bóc tách 7 bẫy nhiễu: đơn vị tiền tệ/khối lượng, lỗi gõ tắt, đảo dấu âm, phóng đại quy mô x10–x1000, giá trị lính canh) và xác nhận danh mục tệp trung gian cần tái tạo tại `data/processed/`.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Đối chiếu 2 chiều dữ liệu thô bị làm dơ với dữ liệu sau làm sạch và lập danh mục tái lập từ đầu."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Phân lập 7 nhóm nhiễu trên từng cột (hậu tố đơn vị, cắt ngắn chuỗi, đảo dấu âm, phóng đại $10^m$, mã lính canh) để kiểm tra mức độ khôi phục giữa bảng gốc và bảng trung gian.
  - *Quyết định kiến trúc:* Cô lập rõ ranh giới bảng gốc read-only với dữ liệu trung gian tại `data/processed/` để kiểm soát chất lượng từng bước.
- **Tồn đọng & Bước tiếp theo:**
  - Dọn dẹp `data/processed/`, chạy lại từng bước [src/clean_raw_data.py](src/clean_raw_data.py) và tái tạo [data/processed/master_rideshare.parquet](data/processed/master_rideshare.parquet).

---

## [2026-10-15 09:58] - Phiên 5: Rà Soát, Chẩn Đoán Dữ Liệu Bị Làm Dơ & Triển Khai Pipeline Làm Sạch Toàn Diện

- **Thành viên phụ trách chính:** **Trần Thanh Huy** (MSSV: `2521001060`)
- **Môi trường:** Windows 64-bit | `Python 3.14.4` (`pandas`, `numpy`)
- **Mục tiêu phiên:**
  - Tiếp nhận, kiểm toán 2 bộ dữ liệu thô bị làm dơ ([data/raw/cab_rides.csv](data/raw/cab_rides.csv), [data/raw/weather.csv](data/raw/weather.csv)), phân loại toàn bộ 7 dạng nhiễu nhân tạo và xây dựng pipeline làm sạch [src/clean_raw_data.py](src/clean_raw_data.py).
- **Nội dung đã hoàn thành:**
  - **1. Kiểm tra tổng quan:** `cab_rides.csv` (`693,071 × 10`, toàn bộ ban đầu bị ép `object`) và `weather.csv` (`6,276 × 8`, toàn bộ `object`).
  - **2. Bóc tách & Xử lý 7 bẫy nhiễu:** Regex xóa hậu tố đơn vị (`USD`, `VND`, `đ`, `kg`, `%`) và đổi dấu phẩy thập phân; sửa lỗi cắt ngắn `'Northeastern Universi'` và chuẩn hóa 12 quận Boston; khôi phục 100% `cab_type` & `name` từ bảng tra cứu 1-1 `product_id`; lấy `abs()` khử lật dấu âm; khôi phục quy mô bậc 10 và lọc mã lính canh (`9999`, `-1`, `-100`, `100000`, `?`, `-`).
  - **3. Xuất kết quả kiểm chứng:** Xuất [data/processed/cab_rides_cleaned.parquet](data/processed/cab_rides_cleaned.parquet) (`693,071` dòng chuẩn hóa kiểu số/danh mục) và [data/processed/weather_cleaned.csv](data/processed/weather_cleaned.csv) (`5,053` dòng sau loại bản ghi khuyết khóa), đối soát ID với [data/processed/master_rideshare.parquet](data/processed/master_rideshare.parquet).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Bóc tách 7 bẫy dữ liệu bị làm dơ (chuỗi đơn vị, đảo dấu âm, phóng đại 10^m, sentinel, lỗi chính tả) và viết src/clean_raw_data.py."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Phát hiện các trường số bị ép thành kiểu `object` do hậu tố đơn vị (`USD`, `VND`, `đ`, `kg`, `%`), dấu phẩy thập phân và chuỗi cắt ngắn `'Northeastern Universi'`; tách lọc mã lính canh (`9999`, `-1`, `-100`, `100000`, `?`, `-`).
  - *Quyết định kiến trúc:* Chuẩn hóa `product_id` 1-1, phục hồi scale vật lý/kinh tế và làm sạch triệt để `693,071` dòng chuyến đi cùng `6,276` dòng thời tiết.
- **Tồn đọng & Bước tiếp theo:**
  - Cập nhật quy trình ghép nối `process_timestamps_and_merge` và đưa dữ liệu sạch vào pipeline huấn luyện kiểm tra lại $R^2$, RMSE, MAE.

---

## [2026-10-12 23:30] - Phiên 4: Rà Soát, Hiệu Đính Học Thuật Toàn Diện & Cập Nhật Bản Thảo Tiểu Luận (v2.0)

- **Thành viên phụ trách chính:** **Lê Huy Thắng** (MSSV: `2521001092`)
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (`python-docx 1.2.0`, `Microsoft Word Office 16 COM`)
- **Mục tiêu phiên:**
  - Đối soát toàn văn bản thảo hiệu đính `reports/2521001060_Nhom02_TieuLuan_DaSua.docx` so với bản đầu [reports/2521001060_Nhom02_TieuLuan.docx](reports/2521001060_Nhom02_TieuLuan.docx), tăng cường tính khiêm tốn khoa học (Effect Size, đa cộng tuyến biến tương tác, chẩn đoán phần dư và 4 chiều hạn chế).
- **Nội dung đã hoàn thành:**
  - **1. Chuẩn hóa phân cấp & Chương 1–2:** Bỏ đánh số tiêu đề kép; làm rõ Pipeline chỉ fit trên tập Train (`510,380` dòng Epoch 1) và mục tiêu SMART $R^2 \ge 0.90, \text{MAPE} \le 10\%$ đo trên Test; nêu rõ bộ dữ liệu thiếu biến thời lượng (*Duration*), bổ sung trích dẫn *CRISP-DM* / *Scikit-learn* và lý thuyết Effect Size (Cohen's $d$, $\eta^2$) tránh ngộ nhận $p$-value ở cỡ mẫu lớn ($N > 600,000$).
  - **2. Chương 3–4 (EDA & Mô hình hóa Epoch 1):** Đính chính `55,095` dòng `Uber Taxi` khuyết `price` là MNAR (đồng hồ taximeter); bổ sung Effect Size cho $t$-test mưa ($d = 0.013$, không đáng kể thực tiễn) và ANOVA buổi ($\eta^2 = 0.0001$); minh định `surge_multiplier` & `distance_x_surge` là biến cấu thành giá đồng thời (không phải dự báo nhu cầu), giải thích hệ số `distance` âm ($-1.1396$) do đa cộng tuyến với `distance_x_surge`, và ghi nhận phần dư lệch phải/phương sai tăng ở vùng giá $> 35\$$.
  - **3. Chương 5 (Khuyến nghị & 4 nhóm hạn chế):** Chuyển ngôn ngữ khuyến nghị sang sắc thái đề xuất có điều kiện A/B test; rút bỏ nhận định thiếu căn cứ *"thu nhập tài xế tăng 12–18% khi trời mưa"*; bổ sung mục 5.3 với 4 nhóm hạn chế (dữ liệu mùa đông 2018 thiếu `duration`, chia ngẫu nhiên 80/20, kiểm định đơn biến, đạo đức AI) trên bản thảo `190` đoạn + `9` biểu đồ (`HistGB R² = 0.9674, MAPE = 8.32%`).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Hiệu đính tính khiêm tốn khoa học (Effect Size, đa cộng tuyến distance_x_surge, 4 nhóm hạn chế) cho bản thảo v2.0."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Bóc tách hiện tượng hệ số `distance` âm (`-1.1396`) do đa cộng tuyến với `distance_x_surge` và kiểm chứng Effect Size (`d = 0.013`, `η² = 0.0001`) để không ngộ nhận `p-value` trên cỡ mẫu lớn.
  - *Quyết định kiến trúc:* Bổ sung Cohen's $d$, $\eta^2$, phân biệt biến phái sinh surge với dự báo nhu cầu và chuẩn hóa cấu trúc đề mục APA 7th.
- **Tồn đọng & Bước tiếp theo:**
  - Đồng bộ nội dung hiệu đính sang bản Quarto `.qmd` và chuẩn bị dàn ý bảo vệ theo 5 phát hiện & 4 nhóm hạn chế.

---

## [2026-10-10 07:44] - Phiên 3: Thử Nghiệm Pipeline Cơ Sở — Tải Dữ Liệu Gốc, Tiền Xử Lý, Huấn Luyện Baseline & Bản Thảo Đầu (Ca sáng, cùng Lê Huy Thắng)

- **Thành viên phụ trách chính:** **Trần Thanh Huy** (MSSV: `2521001060`)
- **Giai đoạn:** Giai đoạn thử nghiệm cơ sở (Raw Ingestion → Preprocessing → Baseline ML → Bản thảo đầu v1, ca sáng).
- **Môi trường:** Windows 64-bit | `Python 3.14.4` (`pandas`, `scikit-learn`, `scipy`, `matplotlib`, `python-docx`)
- **Mục tiêu phiên:**
  - *(Bước 1 — Lê Huy Thắng)* Tải bộ dữ liệu chuẩn *Uber and Lyft Dataset (Boston, MA)* (`693,071` dòng × `10` cột; `weather.csv` `6,276` dòng) và kiểm định sơ bộ khóa thời gian (`ms` vs `s`).
  - *(Bước 2 — Trần Thanh Huy)* Xây dựng pipeline nạp đa bảng, Memory Downcasting (~`50%` RAM) và kiểm định sơ bộ `t-test` / `ANOVA` trên tập dữ liệu `637,976` dòng.
  - *(Bước 3 — Trần Thanh Huy)* Huấn luyện 3 mô hình baseline (`OLS`, `Ridge`, `HistGradientBoosting`) trên `80/20` (`random_state=42`) đạt `HistGradientBoosting R² = 0.9674, MAPE = 8.32%`.
  - *(Bước 4 — Lê Huy Thắng)* Tổng hợp bản thảo tiểu luận đầu tiên (`MSSV_2521001060_TieuLuan.docx`, `3.06 MB`) và 9 biểu đồ minh họa.
- **Nội dung đã hoàn thành:**
  - Tải và lưu trữ toàn vẹn dữ liệu thô tại [data/raw/weather.csv](data/raw/weather.csv) và thư mục `data/raw/` (`cab_rides.csv` `693,071` dòng).
  - Thực thi pipeline tiền xử lý, huấn luyện 3 mô hình baseline (`OLS R² = 0.9321`, `HistGB R² = 0.9674`) và kết xuất bộ 9 ảnh + bản thảo đầu v1 (`README.md`, `src/data_loader.py`, `notebooks/02_model_training.ipynb`, `reports/MSSV_2521001060_TieuLuan.docx`).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Thực hiện trọn vòng CRISP-DM thử nghiệm Epoch 1 từ nạp dữ liệu Uber & Lyft Boston, kiểm định t-test/ANOVA, huấn luyện OLS/Ridge/HistGB đến xuất bản thảo báo cáo đầu tiên."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Phát hiện bất đối xứng đơn vị timestamp (`ms` vs `s`), thiếu `price` trên `55,095` dòng `Uber Taxi`, và nhận diện rủi ro phóng đại ý nghĩa thống kê khi cỡ mẫu lớn (`N > 600,000`).
  - *Quyết định kiến trúc:* Dùng `ColumnTransformer` (`StandardScaler` + `OneHotEncoder(drop='first')`) làm chuẩn chống Data Leakage cho toàn bộ các vòng nâng cấp về sau.
- **Tồn đọng & Bước tiếp theo:**
  - Chuyển bản thảo Epoch 1 sang rà soát phản biện học thuật chuyên sâu về Effect Size, đa cộng tuyến và giới hạn suy luận nhân quả ở phiên kế tiếp.

---

## [2026-10-06 09:30] - Phiên 2: Thẩm Định Đối Soát & Xác Lập Đề Tài Nghiên Cứu Chính Thức (`PRJ-EXT-01`)

- **Thành viên phụ trách chính:** **Lê Huy Thắng** (MSSV: `2521001092`)
- **Môi trường:** Windows 64-bit | `Python 3.11.9` (Antigravity IDE)
- **Mục tiêu phiên:**
  - Khảo sát danh mục 32 đề tài chuẩn từ tài liệu `projects.pdf` và xác định nhu cầu lựa chọn đề tài ngoài danh mục do tình trạng trùng lặp trong lớp học.
  - Thẩm định toàn diện tính khả thi, cỡ mẫu và độ phủ rubric đối với đề tài độc lập ngoài danh mục.
  - Chính thức xác lập tên đề tài và xây dựng tệp quy hoạch lộ trình nghiên cứu chi tiết: `01_ke_hoach_va_quy_trinh_nghien_cuu.md`.

- **Nội dung đã hoàn thành:**
  - **Chính thức xác lập đề tài tiểu luận:** *"Phân tích Cơ chế Định giá Động (Dynamic Surge Pricing) và Dự báo Cước phí Chuyến đi trong Nền tảng Gọi xe Công nghệ (Ride-Hailing Dynamic Pricing & Demand Analytics)"* (Mã: `PRJ-EXT-01`).
  - **Thẩm định điều kiện tiên quyết (Gatekeeper Check):**
    * Cỡ mẫu dự kiến: ~693,071 quan sát từ tập dữ liệu mở *Uber and Lyft Dataset (Boston, MA)* trên Kaggle, vượt xa mức tối thiểu 100,000 dòng.
    * Cấu trúc đa tệp thực chất: Ghép nối bảng `cab_rides.csv` và `weather.csv` qua khóa thời gian và địa điểm với `pd.merge()`.
    * Kỹ thuật tối ưu bộ nhớ: Lập kế hoạch Downcasting kiểu dữ liệu sang `category`, `int32`, `float32` để giảm 75% RAM và lưu trữ định dạng nhị phân `Parquet`.
  - **Khởi tạo tài liệu quy trình nghiên cứu:** Tạo lập hoàn chỉnh tệp `01_ke_hoach_va_quy_trinh_nghien_cuu.md` phân định rõ 7 giai đoạn CRISP-DM, cấu trúc 5 chương báo cáo chuẩn APA 7th và lộ trình 6 phiên làm việc độc lập.
  - Cập nhật tài liệu tổng quan [README.md](README.md) của không gian nghiên cứu tiểu luận.

- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Khảo sát 32 đề tài trong projects.pdf, thẩm định điều kiện Gatekeeper (≥ 100k dòng, đa bảng, tối ưu RAM) cho bộ dữ liệu Uber & Lyft Boston và lập kế hoạch nghiên cứu PRJ-EXT-01."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Đối chiếu cấu trúc khóa ngoại thời gian-địa điểm giữa `cab_rides.csv` và `weather.csv` để tránh lỗi nhân bản dòng khi `pd.merge()` và ước tính mức giảm RAM khi downcast kiểu dữ liệu.
  - *Quyết định kiến trúc:* Lựa chọn bài toán Hồi quy (Regression) kết hợp kiểm định giả thuyết ($t$-test, ANOVA) trên các biến định lượng thực tế (`price`, `distance`, `surge_multiplier`, `temp`, `rain`), tránh biến mã hóa PCA ẩn danh ($V_1 \dots V_{28}$) để bảo đảm giải trình 100% khi bảo vệ Vòng 3.
- **Tồn đọng & Bước tiếp theo:**
  - Tải bộ dữ liệu `cab_rides.csv` và `weather.csv` từ Kaggle vào [data/raw/](data/raw/) và triển khai module nạp đa bảng, ép kiểu tiết kiệm RAM.

---

## [2026-10-01 09:50] - Phiên 1: Thiết lập Môi trường & Khởi tạo Không gian Nghiên cứu Tiểu luận

- **Thành viên phụ trách chính:** **Trần Thanh Huy** (MSSV: `2521001060`)
- **Môi trường:** Windows 64-bit | `Python 3.14.4` (Antigravity IDE)
- **Mục tiêu phiên:**
  - Khởi tạo không gian làm việc chuyên biệt `tiểu_luận/` phục vụ nghiên cứu và hoàn thành đề tài tiểu luận môn học theo yêu cầu `projects.pdf`.
  - Thiết lập quy trình lưu vết nhật ký làm việc P-D-M-V độc lập cho dự án.
- **Nội dung đã hoàn thành:**
  - Thiết lập không gian làm việc `tiểu_luận/`, khởi tạo hồ sơ nghiên cứu ban đầu (`00_nhật_ký_nghiên_cứu_tiểu_luận.md`), tham chiếu quy chuẩn `projects.pdf` và cấu hình quy tắc lưu vết nhật ký chuyên biệt cho đề tài tại [nhật ký làm việc.md](nh%E1%BA%ADt%20k%C3%BD%20l%C3%A0m%20vi%E1%BB%87c.md).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Khởi tạo không gian làm việc nghiên cứu tiểu luận độc lập và thiết lập cấu trúc nhật ký P-D-M-V theo mẫu Mục 8.2 trong projects.pdf."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Kiểm tra quy chuẩn đặt tên tệp, thứ tự thời gian đảo ngược và các trường bắt buộc trong Template Mục 8.2 của `projects.pdf`.
  - *Quyết định kiến trúc:* Tách biệt hoàn toàn không gian dự án tiểu luận khỏi các bài tập học phần thông thường và chuẩn hóa khung ghi nhận 5 thành phần bắt buộc cho mọi phiên.
- **Tồn đọng & Bước tiếp theo:**
  - Khảo sát danh mục 32 đề tài trong `projects.pdf`, lựa chọn hướng đề tài phù hợp, thu thập dữ liệu vào `data/raw/` và lập đề cương chi tiết.

---
