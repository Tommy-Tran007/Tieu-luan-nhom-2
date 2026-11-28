# `PRJ-EXT-01` — Phân tích Biểu hiện Định giá Động (*Dynamic Surge Pricing*) & Dự báo Giá báo API (Uber vs. Lyft — Boston, MA)

> **Học phần:** Lập trình Python cho Khoa học Dữ liệu (*AI-First*) — Đại học Tài chính - Marketing (**UFM**)  
> **Nhóm thực hiện:** Nhóm 02 — **Trần Thanh Huy** (`2521001060`, STT `1`) & **Lê Huy Thắng** (`2521001092`, STT `2`) — Lớp `25DDS01`  
> **Tóm tắt kiểm chứng kỹ thuật:** `270,898 × 32` quan sát sạch (`232,861` lượt surge quan sát thực; `22,262` dòng khuyết `distance` giữ `NaN` để nội suy strictly-on-train trong `Pipeline`) • `18/18` kiểm toán dữ liệu Dashboard PASS • `40/40` kiểm thử `pytest` PASS • định dạng theo cấu hình PEP 8 `pyproject.toml` (`line-length = 79`, `ruff` PASS trên `50` tệp `.py`/`.ipynb` trong gói nộp) • Dashboard HTML Offline (`5.27 MiB`) • Tiểu luận Quarto/Typst chuẩn UFM **`45` trang** (`43/45` trang nội dung ngoài 2 trang bìa; `27` trang Ch.1–5; `187.1 KiB` QMD → `9.35 MiB` PDF; `22` bảng; `17` hình `≥ 300 DPI`; `23/23` APA 7th)

---

## 1. Tổng quan Đề tài, Ngữ cảnh Dữ liệu Thô Bị Làm Dơ & Câu hỏi Nghiên cứu

Nghiên cứu phân tích cơ chế định giá động, đối sánh cấu trúc biểu giá cạnh tranh và dự báo **giá báo API trước chuyến đi (*Upfront API Fare Quotes*, không phải hóa đơn hoàn thành hay doanh thu kế toán)** giữa **Uber** và **Lyft** tại **Boston, MA** (`12` quận, `72` tuyến $K_{6,6}$, miền cự ly `[0.02, 7.86]` dặm, `25/11–18/12/2018` EST).

### 1.1. Ngữ cảnh Bộ Dữ liệu Thô Bị Làm Dơ (*Instructor-Corrupted Raw Dataset*)
Hai tệp gốc trong `data/raw/` là bộ dữ liệu được **làm dơ sẵn theo yêu cầu học phần** và được giữ nguyên trạng thái chỉ-đọc (`read-only`, kèm nguyên tệp gốc trong gói nộp để bảo đảm tái lập độc lập offline từ đầu bằng mã nguồn Python):
- `data/raw/cab_rides.csv`: `693,071 × 10` (`81.51 MiB`, `SHA-256: 3987E0532C06E3D23B62E9F5C483BAC70A9331DE5BB593BA80C75E0A961ACA9A`).
- `data/raw/weather.csv`: `6,276 × 8` (`325.09 KiB`, `SHA-256: F6253513970DEACE50C198EF71ED05FFA31B83F10893CC9FB078A2B7FF34125F`).

Toàn bộ quá trình xử lý đã giải mã và làm sạch **7 tầng nhiễu nhân tạo** bằng mã nguồn tái lập trong `src/clean_raw_data.py`: (1) hậu tố đơn vị (`USD`, `VND`, `đ`, `kg`, `%`) và dấu phẩy thập phân; (2) cắt ngắn địa danh (`Northeastern Universi`); (3) khôi phục 1-1 `cab_type` & `name` qua `product_id`; (4) đảo dấu âm; (5) phóng đại quy mô bậc $10^m$ (dùng mốc cố định `2.15` dặm cho `distance` để chặn rò rỉ phân phối); (6) mã lính canh (`-1`, `-100`, `1000`, `9999`, `10000`, `100000`, `?`, `-`, `unknown`, `chưa rõ`); (7) khuyết `price` trên tập thô (`263,497` dòng, gồm `55,095` dòng `Uber Taxi` MNAR và `208,402` dòng phi-Taxi) cùng các giá trị khuyết được bảo toàn `NaN` trên tập sạch `270,898` dòng đối với `timestamp` (`62,864` dòng), `surge_multiplier` (`38,037` dòng Lyft) và `distance` (`22,262` dòng, nội suy strictly-on-train qua `RouteDistanceImputer` bên trong `sklearn.pipeline.Pipeline`). Đối chứng phân phối cho thấy `270,898` dòng sạch giữ lại tương đồng sát với `367,078` dòng phi-Taxi bị loại (tỷ lệ Uber `51.86%` vs `51.78%`, tỷ lệ Surge Lyft thực `6.80%` vs `6.91%`, cự ly TB `2.197` vs `2.196` dặm, giá TB `$16.58` vs `$16.50`), không ghi nhận thiên lệch chọn mẫu đáng kể.

### 1.2. Bốn Câu hỏi Nghiên cứu (RQ1–RQ4)
- **RQ1 (Thời tiết & Khung giờ EST):** Trên tập khớp thời tiết sạch sentinel (`n = 152,043`) và mẫu độ nhạy loại sạch mọi ô chứa sentinel (`n = 87,733`), mưa nhẹ–vừa và khung giờ EST có tạo khác biệt đáng kể về mặt thống kê và thực tiễn lên giá báo API hay xác suất Surge Lyft không?
- **RQ2 (Biểu giá & Điểm cắt `d*`):** Hệ số chặn ($\beta_0$), độ dốc ($\beta_1$) và điểm cắt $d^*$ trong miền `[0.02, 7.86]` dặm trên `5` cặp dịch vụ đối xứng thay đổi ra sao giữa Đặc tả A (gộp `d × surge`) và Đặc tả B (thuần `1.0x`) trên cùng `X_train` (`80%`)?
- **RQ3 (Có điều kiện vs. Ex-Ante, Split-CQR & XAI):** `10` cấu hình mô hình có điều kiện (`27` biến) vận hành ra sao giữa Uber (surge ẩn) và Lyft (surge công bố), qua các chiến lược chia dữ liệu và khi chuyển sang bài toán **Ex-Ante** (`23` biến không surge + Bộ phân loại Surge Lyft)? Cụm đặc trưng nào chi phối `ΔRMSE` và `Monotonic Split-CQR` hiệu chỉnh khoảng dự báo `[P10, P90]` như thế nào?
- **RQ4 (Kịch bản Surge Cap Bên ngắn):** Theo nguyên tắc bên ngắn `Q_realized = min(Q_D, Q_S)` (xét hàm cung nhạy giá $\varepsilon_S \in \{0.30, 0.60\}$ với dư cầu $\sum \max(0, Q_D - Q_{\text{realized}})$ và cận trên cầu $\varepsilon \in \{-0.50, -0.80\}$), áp trần `Surge Cap` (`3.00x → 1.00x`) trên `92,384` lượt Lyft thực tạo đánh đổi gì giữa tiết kiệm hành khách, `API Quote Proxy` và công suất thử nghiệm **Triggered Switchback A/B**?

---

## 2. Đối chiếu Quy chuẩn & Quy trình Chấm điểm (`projects.pdf`)

Mục này ánh xạ trực tiếp kho mã nguồn và sản phẩm bàn giao vào **Quy trình 3 Vòng**, **7 Giai đoạn CRISP-DM (`§3.1–3.7`)** và **Barem 5 Trụ cột (`§5.3`)** của `projects.pdf` để thuận tiện tra cứu và kiểm chứng độc lập.

### 2.1. Vòng 1 — Ba Điều kiện Tiên quyết (`§5.2.1` Gatekeeper Check)

| Điều kiện (`projects.pdf` §5.2.1) | Bằng chứng thực tế trong gói nộp | Vị trí kiểm chứng |
| :--- | :--- | :--- |
| **GK-1: Quy mô dữ liệu `≥ 100,000` quan sát sau hợp nhất** | `master_rideshare.parquet` đạt **`270,898 × 32`** (`2.71×` ngưỡng); tập có điều kiện (`is_surge_original == 1`) đạt **`232,861`** dòng (`2.33×` ngưỡng); `0` trùng `id`, `0` fan-out khi `Left Join`. | `data/processed/master_rideshare.parquet`, `src/stage3_pipeline_and_audit.py` |
| **GK-2: Tệp `./nhật ký làm việc.md` tại gốc (`≥ 4–6` phiên đảo ngược)** | Tệp [nhật ký làm việc.md](nh%E1%BA%ADt%20k%C3%BD%20l%C3%A0m%20vi%E1%BB%87c.md) đặt tại thư mục gốc, ghi nhận **`24` phiên** (`2026-10-01` → `2026-11-28`) theo thứ tự thời gian đảo ngược, đủ các mục Template `§8.2`. | [nhật ký làm việc.md](nh%E1%BA%ADt%20k%C3%BD%20l%C3%A0m%20vi%E1%BB%87c.md) |
| **GK-3: Mã nguồn chạy thông suốt, không lỗi cú pháp, không đường dẫn cứng** | Toàn bộ đường dẫn xây dựng tương đối qua `pathlib.Path(__file__).resolve()`; `python main.py --stage all`, `dashboard.validation` (`18/18`), `pytest` (`40/40`), `check_thesis_audit.py` và `4` Notebooks chạy độc lập sạch lỗi. | `main.py`, `src/config.py`, `dashboard/config.py` |

### 2.2. Vòng 2 — Bản đồ Ánh xạ 5 Trụ cột Barem (`§5.3`), 7 Giai đoạn CRISP-DM (`§3.1–3.7`) & Đối chiếu Hai Chiều `Code ↔ Tiểu luận PDF`

*(Quy ước chỉ mục trang: `tr. X` là số trang Ả Rập in tại chân trang `Trang 1–36`, tương ứng PDF `p.10–45` (`36` trang); Trang bìa chính/phụ nằm ở PDF `p.1–2` không đánh số (`2` trang) và phần Mở đầu từ Lời cam đoan đến Tóm tắt đồ án nằm ở PDF `p.3–9`, đánh số La Mã `i–vii` (`7` trang); tổng `2 + 7 + 36 = 45` trang PDF — **`43/45` trang nội dung ngoài 2 trang bìa**.)*

| Trụ cột (`projects.pdf` §5.3) | Tệp mã nguồn & Hàm trọng tâm | Vị trí trong Tiểu luận PDF (`tr.` in / `PDF p.`) | Đầu ra kiểm chứng được |
| :--- | :--- | :--- | :--- |
| **I. Xác định Vấn đề & Dữ liệu Lớn (`1.5đ`; GĐ1–2)** | `src/clean_raw_data.py` (`load_cab_rides_chunked`, `clean_cab_rides`, `clean_weather`); `src/stage3_pipeline_and_audit.py` (`step3_merge_and_eda`, `_downcast_master_dataframe`) | `Ch.1 (1.1–1.5)`, `Ch.2 (2.1–2.5)`, `Mục 3.1–3.2` (`tr. 1–8` / `p.10–17`); `Bảng 2.1`, `3.1` | Hợp nhất đa tệp `pd.merge` (`270,898 × 32`, `0` fan-out), streaming `chunksize = 100,000` (`peak 18.23 MiB`), `8/8` cột phân loại `category`, downcast `cab_rides_cleaned.parquet` (`11.54 MiB`), lưu Parquet giảm `64.37%` RAM so với CSV (`105.15 → 37.46 MiB`; riêng bước downcast hậu ghép nối giảm `37.75%`: `60.23 → 37.49 MiB`) và đọc nhanh `7.6×` so với CSV cùng bảng. |
| **II. Làm sạch & Khám phá Dữ liệu EDA (`2.5đ`; GĐ3–4)** | `src/clean_raw_data.py`; `src/stage3_pipeline_and_audit.py`; `notebooks/03_Giai_Doan_3_Data_Prep_EDA_Visualization.ipynb`; `scripts/prepare_thesis_figures.py`; `dashboard/` | `Mục 3.3–3.5` (`tr. 9–12` / `p.18–21`); `Bảng 3.2–3.4`; `Hình 3.1–3.5` | Xử lý 7 tầng nhiễu có đối chứng chọn mẫu (`preserve_decimals=2`, mốc `2.15` dặm cố định cho `distance`, bảo toàn `22,262` `NaN` cự ly cho `Pipeline`, `0` sai khác `--stage clean` vs `eda`), `IQR` giữ ngoại lai kinh tế có lập luận, `groupby`/`pivot_table`/`crosstab`, `21` hình GĐ3–6 + `17` hình tiểu luận `≥ 300 DPI`, Dashboard Streamlit & HTML Plotly tương tác 5 Tab. |
| **III. Mô hình hóa ML & Đánh giá Hiệu năng (`2.5đ`; GĐ5–6)** | `src/features.py` (`engineer_features_stage4`, `run_vif_analysis`, `create_data_splits`, `RouteDistanceImputer`, `EconomicInteractionPreprocessor`); `src/stats_tests.py` (`run_all_hypothesis_tests`); `src/models.py` (`build_model_registry`); `src/stage5_modeling.py` (`run_stage5`, `run_mutate_and_break_experiments`); `src/evaluation.py` (`diagnose_residuals`, `run_quantile_prediction_intervals`); `src/interpretation.py` (`run_permutation_importance_analysis`, `compute_pdp_and_ice_data`); `scripts/thesis_v5_verification.py` | `Mục 3.6` & `Ch.4 (4.1–4.6)` (`tr. 12–23` / `p.21–32`); `Bảng 3.5–3.6`, `4.1–4.9`; `Hình 3.6`, `4.1–4.10` | Kiểm định `t-test`, `Welch`, `Mann-Whitney`, `ANOVA`, `Kruskal-Wallis`, `Chi-Square`, `Wald HC3`, `Cluster SE` 2 cấp, `TOST`, `Exact WCR` (`B = 1,999`), `Holm/BH-FDR`; `10` mô hình có điều kiện + `Ex-Ante` trong `Pipeline` kèm `RouteDistanceImputer` (`M7 R² = 0.9626`, `RMSE = $1.7830`, `MAPE = 8.19%`; `M1`/`M2` đạt Full Column Rank SVD `49/49` & `82/82`); `6` thí nghiệm *Mutate & Break*; `Grouped XAI`, `PDP/ICE`, `Monotonic Split-CQR`. |
| **IV. Khuyến nghị Quản trị & Thực tiễn (`2.0đ`; GĐ7)** | `src/simulation.py` (`run_surge_cap_policy_simulation`); `scripts/thesis_v5_verification.py` (`compute_rq4_supply_power_and_io`) | `Ch.5 (5.1–5.4)` (`tr. 24–27` / `p.33–36`); `Bảng 5.1–5.2`; `Hình 5.1` | Mô phỏng kịch bản `Surge Cap` bên ngắn `Q_realized = min(Q_D, Q_S)` với cung nhạy giá $\varepsilon_S \in \{0.30, 0.60\}$, lộ trình 3 pha gắn **Triggered Switchback A/B** (`18` ngày quan sát thực), phân tích *driver multi-homing*, Massachusetts DPU TNC, *Geographic Coverage Bias* và *k-anonymity* / H3. |
| **V. Hình thức Báo cáo, Code & Nhật ký AI (`1.5đ`; `§4.1–4.2`)** | `pyproject.toml`; `scripts/build_final_thesis.py`; `scripts/check_thesis_audit.py` (`run_audit`); [reports/2521001060_Nhom02_TieuLuan.qmd](reports/2521001060_Nhom02_TieuLuan.qmd); [nhật ký làm việc.md](nh%E1%BA%ADt%20k%C3%BD%20l%C3%A0m%20vi%E1%BB%87c.md) | Bìa (`p.1–2`), Mở đầu (`i–vii` / `p.3–9`), `TLTK` (`tr. 28–29` / `p.37–38`) & `Phụ lục A–B` (`tr. 30–36` / `p.39–45`); `Bảng A.1`, `B.1–B.3` | Tiểu luận Quarto/Typst chuẩn `A4`, lề `2-2-3-2 cm`, `Times New Roman 12pt`, `45` trang (`43/45` nội dung ngoài bìa; `27` trang Ch.1–5 trong khung `25–45`), `23/23` APA 7th, đủ Phụ lục A–B; mã nguồn tách `src/` & `scripts/` theo `pyproject.toml` (`79` ký tự); nhật ký `24` phiên P-D-M-V (`10 + 11 + 3`). |

### 2.3. Vòng 3 — Phân công Đóng góp (`§5.4`), Bằng chứng *Mutate & Break* Đã Lưu Sẵn (`§8.1`) & Hỗ trợ Kiểm thử Sống (`§5.2.3`, `§5.5`)
- **Phân bổ hệ số đóng góp ($\sum \alpha_i = 2.00$):** **Trần Thanh Huy** (`2521001060`, STT `1`, $\alpha_1 = 1.00$, `10` phiên độc lập + `3` phiên đồng thực hiện — phụ trách chính chuẩn hóa dữ liệu `clean_raw_data.py`, `features.py`, `RouteDistanceImputer`, `models.py`, lõi Dashboard & `pytest`) và **Lê Huy Thắng** (`2521001092`, STT `2`, $\alpha_2 = 1.00$, `11` phiên độc lập + `3` phiên đồng thực hiện — phụ trách chính EDA `stage3_pipeline_and_audit.py`, kiểm định `stats_tests.py`, `simulation.py`, `thesis_v5_verification.py`, biểu đồ `≥ 300 DPI`, xuất bản Quarto/UFM & tái lập tổng hợp).
- **Bằng chứng nhạy cảm tham số (*Mutate & Break*) đã thực thi & lưu sẵn trong hồ sơ:**
  - **`6` thí nghiệm phá vỡ mô hình (`MB1–MB6`):** Lưu đầy đủ kết quả định lượng tại [reports/tables/stage5_mutate_break.csv](reports/tables/stage5_mutate_break.csv) và trình bày tại `Bảng 4.7` (xóa tương tác, phạt Ridge cực đoan `alpha = 100,000`, bỏ giới hạn độ sâu cây, loại `surge_multiplier`, thay `OneHotEncoder` bằng `OrdinalEncoder`, và đổi `test_size = 0.20 → 0.40`).
  - **`6` kiểm thử tiêm lỗi tự động (*Mutation Tests*):** Cài đặt tại [dashboard/tests/test_validation.py](dashboard/tests/test_validation.py) (tiêm lỗi thiếu cột `price`, nhãn mưa giả `CRIT-01`, rò rỉ `.ffill()` `CRIT-02`, trùng khóa chính `id`, giá âm và bảng Permutation đếm trùng cột `distance` tại `TBL-03`).
  - **Tái kiểm chứng trực tiếp qua CLI:** `python main.py --stage model --test-size 0.4` cho phép chạy kiểm tra trực tiếp độ ổn định của toàn bộ mô hình khi thay đổi tỷ lệ phân chia theo `§5.2.3` *(lưu ý: lệnh này ghi đè các tệp `models/*.joblib` và `reports/tables/stage5_*`; sau khi thử nghiệm sống, chạy lại `python main.py --stage model` với mặc định `--test-size 0.2` để khôi phục cấu hình `80/20` trước khi chạy `pytest` hoặc `check_thesis_audit.py`, hoặc đối chiếu trực tiếp kết quả `MB6` đã lưu không phá hủy trong `stage5_mutate_break.csv`)*.

---

## 3. Kết quả Thực nghiệm Chính thức

*(Mọi con số dưới đây được đồng bộ từ `reports/results/thesis_v5_verification.json`, `reports/tables/*.csv`, `data/processed/stage3_audit_summary.json` và báo cáo Tiểu luận; quy ước đơn vị nhị phân `1 MiB = 1,048,576 bytes`.)*

### 3.1. Chất lượng Dữ liệu, Bảo toàn Nguồn gốc & Tối ưu Bộ nhớ

| Chỉ số kiểm toán | Kết quả đo đạc | Quy tắc áp dụng |
| :--- | ---: | :--- |
| Bảng phân tích tổng thể (`master_rideshare.parquet`) | **`270,898 × 32`** | `0` trùng `id`, `0` fan-out `Left Join`, `8/8` cột phân loại ép `category` |
| Bộ nhớ & I/O cùng bảng (`270,898 × 32`) | **`37.46 MiB`** RAM / **`15.97 MiB`** đĩa | `cab_rides_cleaned.parquet` downcast còn **`11.54 MiB`**; giảm **`64.37%`** RAM tổng từ CSV sang Parquet (`105.15 → 37.46 MiB`; riêng ép kiểu sau merge giảm **`37.75%`**: `60.23 → 37.49 MiB`); Parquet `0.026s` vs CSV `0.195s` / `64.23 MiB` (**`7.6×`** nhanh hơn, **`4.02×`** gọn hơn); streaming `chunksize = 100,000` (`peak 18.23 MiB`) |
| Nguồn gốc Surge (`CRIT-03`) | **`232,861`** surge quan sát thực (`140,477` Uber + `92,384` Lyft) | Không điền mù `fillna(1.0)` trên `38,037` lượt Lyft khuyết (`29.16%`, giữ `NaN` + `is_surge_original`); toàn tập sạch: Uber TB `$15.83` (`0.00%` API surge) · Lyft TB `$17.38` (`130,421` dòng); riêng `92,384` Lyft surge thực: TB `$17.36`, **Surge thực `6.80%`** (`6,280`) |
| Top 5 quận Surge Lyft thực | **Back Bay `11.17%` · Fenway `10.57%` · Northeastern `10.10%` · Theatre `9.58%` · BU `9.57%`** | Gấp **`7.2–8.4×`** `North End` (`1.33%`) và **`6.5–7.5×`** `Haymarket Square` (`1.48%`); TB `2.16` lượt surge/ô quận×giờ kích hoạt |
| Timestamp EST (`CRIT-02`) & Thời tiết (`CRIT-01`, `H4`) | **`208,034`** TS gốc · **`179,280`** khớp thực · **`152,043`** mưa/khô sạch | `62,864` khuyết TS giữ `NaN` (không dùng `.ffill()`); điền 5 biến khí tượng theo trung vị cùng giờ toàn Boston; tách `is_rain_sentinel` cô lập `27,237` lượt (`24,495` mưa / `127,548` khô) |
| Bản ghi nguyên bản (`R3`), Khuyết `distance` & VIF | **`88.85%`** (`240,689 / 270,898`) | Khớp `stage4_audit_flags.parquet`; giữ `22,262` `distance = NaN` để nội suy trong `Pipeline`; tâm hóa `lyft_surge_excess` giảm `VIF(distance)` từ `162.04` xuống `1.02` (`max VIF 168.27 → 6.64`) |

### 3.2. Kiểm định Giả thuyết, Sai số Cụm, Exact WCR & TOST (`RQ1`)

| Kiểm định / Mở rộng | Cỡ mẫu / Cụm | Ước lượng / Thống kê | `p-value` (`Holm` / Cụm) | Kết luận thận trọng |
| :--- | :--- | :--- | :--- | :--- |
| **`H1a` (Mưa vs. Tạnh — Sạch mưa)** | `152,043` (`24,495` / `127,548`) | `d = -0.00095` (`-$0.0089`, `r_rb = +0.00085`) | `p = 0.8915` (`p_Holm = 1.0000`) | Chưa đủ bằng chứng bác bỏ `H0` |
| **OLS đa biến gom cụm (quận×giờ / giờ Boston)** | `119,903` có `distance` (`2,793` / `329` cụm) | `β = +$0.0021` (`SE_hr = $0.0176`) | `p_cl = 0.9119` / `p_hr = 0.9036` | **TOST `±$0.10`: `p < 0.0001`** *(Không surge: `n = 139,555`, `2,794` / `329` cụm, `β = -$0.0357`, TOST `p = 0.0004`)* |
| **Độ nhạy loại mọi ô chứa sentinel mưa** | **`87,733`** (`325` giờ) | `β = +$0.0072` (`SE_hr = $0.0206`) | **`p_hr = 0.7277`** (`p_cl = 0.7544`) | **TOST `±$0.10`: `p < 0.0001`** |
| **Mưa lớn `≥ 0.10 in/h` — Exact WCR (`B = 1,999`)** | `N = 103,608` (`treated: 3,021` / `47` cụm / `8` giờ) | `β = -$0.0075` (`t = -0.157 / -0.143`) | **`p_wild = 0.8590 / 0.8850`** | Thăm dò (công suất thấp do chỉ có `8` giờ mưa lớn) |
| **`H2a/c` & Wald `F` cụm 4 buổi EST** | `208,034` (`3,980` cụm EST) | `Max–Min = $0.0753`, `F_cl = 0.2829` | `p_Holm = 1.0000` / `p_cl = 0.8378` | Chưa đủ bằng chứng bác bỏ `H0` |
| **`H4c` (Joint Wald 6 biến thời tiết — Hạng đầy đủ)** | `95,925` có `distance` (`df = 6`, HC3) | `F = 1.0683`, `ΔR² = 4.9e-06` | `p = 0.3789` (`p_Holm = 1.0000`) | Đóng góp phương sai thực tiễn `≈ 0` |
| **`H3a` (Surge ~ Hãng) & `H3b` (Surge Lyft ~ Mưa)** | `232,861` / `52,045` | `H3a: χ² = 9,811.30, V = 0.2053`; `H3b: -0.72 pp` | `H3a: p_Holm < 0.0001`; `H3b: p_hr = 0.0793` | `H3a` bác bỏ `H0` (Lyft `6.80%` vs Uber `0.00%`); `H3b` không ý nghĩa sau cụm/Holm (`0.1897`) |

### 3.3. Đối sánh Song đặc tả `5` Cặp Biểu giá Đối xứng trên `X_train` (`n = 186,288`, `RQ2`)

| Phân khúc đối xứng | Biểu giá Uber (Đặc tả A) | Biểu giá Lyft (Đặc tả A) | Điểm cắt Đặc tả A (`CI_cl`) | Điểm cắt Đặc tả B (`1.0x`, `X_train`) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Đi ghép** | `UberPool`: `$6.14 + $1.19/mi` | `Shared`: `$3.34 + $1.24/mi` | Không cắt trong `[0.02, 7.86]` mi | Không cắt (`Shared` luôn rẻ hơn) |
| **2. Phổ thông 4 chỗ** | `UberX`: `$6.47 + $1.51/mi` | `Lyft`: `$5.40 + $1.84/mi` | **`3.21` mi** `[2.96, 3.46]` (`85.0% < d*`) | **`5.17` mi** (`CI_HC3 [4.38, 5.96]`; `CI_cl [4.16, 6.18]`) *(full: `4.99` mi)* |
| **3. Gia đình 6 chỗ** | `UberXL`: `$9.19 + $2.96/mi` | `Lyft XL`: `$7.49 + $3.41/mi` | **`3.75` mi** `[3.36, 4.14]` | **Không cắt (`8.31` mi `> 7.86`; `Lyft XL` rẻ hơn toàn miền)** |
| **4. Hạng sang Sedan** | `Black`: `$12.23 + $3.77/mi` | `Lux Black`: `$11.74 + $4.96/mi` | `0.41` mi `[0.26, 0.55]` *(vùng sàn)* | **Không cắt** (`-0.42` mi; `Uber Black` luôn rẻ hơn) |
| **5. Hạng sang SUV** | `Black SUV`: `$22.63 + $3.47/mi` | `Lux Black XL`: `$20.73 + $5.09/mi` | **`1.17` mi** `[1.08, 1.26]` | **`0.85` mi** (`CI_HC3 [0.71, 1.00]`; `CI_cl [0.70, 1.01]`) *(trên sàn `d ≥ 1.0 mi`: `0.605 mi < 1.0`, `Black SUV` rẻ hơn)* |

*(Ghi chú: `WAV` và `Lux` được tách riêng đối chứng, không ghép cặp chéo.)*

### 3.4. Mô hình Có điều kiện vs. Ex-Ante (`RQ3`), Split-CQR, XAI & Surge Cap Bên ngắn (`RQ4`)

- **Mô hình có điều kiện (`10` cấu hình, `n_train = 186,288 / n_test = 46,573`, sạch rò rỉ `RouteDistanceImputer`):** **`M7_HistGradientBoosting`** đạt **`Test R² = 0.9626`**, **`RMSE = $1.7830`**, **`MAE = $1.1338`**, **`MAPE = 8.19%`**, `Overfit = +0.0027`, `85.87%` lệch `≤ ±$2.00` — vượt `M6_Random_Forest` (`0.9612 / $1.8167 / 8.14%`) trên `5/5` fold CV (`p = 0.0012`). **`M2_Interaction_OLS`** (Full Column Rank SVD `82/82`, `cond = 122.42`) đạt `R² = 0.9539, RMSE = $1.9802, MAPE = 9.33%`.
- **Uber (surge ẩn) vs. Lyft (surge công bố) & Ex-Ante (`23` biến):** `M7` đạt **`R² = 0.9794, RMSE = $1.4370`** trên Lyft vs **`R² = 0.9474, RMSE = $1.9773`** trên Uber. **Ex-Ante `M7`** (loại biến surge): `R² = 0.9286 / $2.4655` (`MAPE = 9.15%`) trên All Test, giảm xuống `0.4677 / $10.29` trên `1,198` lượt Lyft surge `> 1.0x`, và `0.9060 / $3.0861` trên Holdout `38,037` Lyft khuyết surge. **Lyft Surge Classifier** (prevalence Test `6.50%`): `HistGB ROC-AUC = 0.7383, PR-AUC = 0.1474` (tại $\tau^* = 0.1087$: `Acc = 73.30%, Prec = 12.85%, Rec = 53.76%, F1 = 0.2074`).
- **Kiểm chứng độ bền, CQR & Grouped XAI:** Strict Train-Only (`0/46,573` giá đổi; `M7 0.9630 / $1.7746`; trên `route_group_split` không rò rỉ `RouteDistanceImputer` `15` tuyến mới: `M7 0.9253 / $2.4197 / 11.50%`, `strict_train_only` `0.9379 / $2.2067`, tập nguyên bản có `distance` `0.9538 / $1.9031 / 9.37%`); API-Batch GroupSplit (`0.9639 / $1.7608`); Temporal EST (`0.9646 / $1.7422`); trong phạm vi 12 quận trung tâm Boston, độ lệch trung bình `|bias| ≤ $0.0900` (`MAPE ∈ [7.33%, 8.98%]`). **Monotonic Split-CQR** (`pandas category`, `85%/15%`): `PICP = 79.89%` (`12/12` gói `[78.62%, 81.99%]`, `MPIW = $3.562`, `0` cắt ngược). **Grouped XAI** (`27/27` cột): trên `4` họ / `14` nhóm rời (`stage6_permutation_importance.csv`), Dịch vụ & Hãng xe (`69.20%`, riêng `name 69.10%`) + Định giá Động & Tương tác (`24.70%`) = **`93.90%`** `ΔRMSE`, Thời tiết & Giờ EST = **`0.04%`**; trên `6` cụm gộp chống đa cộng tuyến (`Bảng 4.9`), `name` (`69.23%`) + Cự ly & Surge gộp (`23.37%`) = **`92.60%`** (cộng `cab_type 0.09%` = `92.69%`), Thời tiết & Giờ EST = **`0.02%`**.
- **Mô phỏng kịch bản Surge Cap Bên ngắn `Q_realized = min(Q_D, Q_S)` (`92,384` Lyft thực, `API Quote Proxy = $1,603,696.20`, cung nhạy giá $\varepsilon_S \in \{0.30, 0.60\}$):**
  - **Pha 1 (`τ ≤ 0.10%`) — Trần `2.00x`:** `53` lượt (`2.50x–3.00x`), giảm giá TB `$5.53` (`-18.84%`; `M2 = $5.88`, số học `$6.20`), Proxy: **`-0.018%`** (tĩnh) / **`-0.023%`** ($\varepsilon_S = 0.30$, dư cầu `8.0` lượt) / **`-0.027%`** ($\varepsilon_S = 0.60$, dư cầu `11.0` lượt).
  - **Pha 2 (`τ = 0.30%–0.50%`) — Trần `1.75x`:** `704` lượt (`11.21%` surge), giảm giá TB `$4.61` (`-11.64%`; `M2 = $5.14`, số học `$5.28`), Proxy: **`-0.203%`** (tĩnh) / **`-0.255%`** ($\varepsilon_S = 0.30$, dư cầu `65.5` lượt) / **`-0.307%`** ($\varepsilon_S = 0.60$, dư cầu `90.1` lượt; cận trên cầu $\varepsilon = -0.50$: `-0.116%`), ứng viên **Triggered Switchback A/B** chuẩn $\text{DEFF}_{k\times}$ tính trên **`18` ngày quan sát thực**: toàn 12 quận (`DEFF_25x = 2.627`, MDE `3.0 pp` ~`30.7` ngày ở `25×` / `25.1` ngày ở `50×`); 5 quận điểm nóng (`80.30%` lượt kích hoạt, `DEFF_25x = 2.648`, MDE `3.0 pp` ~`38.5` ngày ở `25×` / `31.6` ngày ở `50×`).

---

## 4. Các Đánh đổi Kỹ thuật Có Chủ Đích, Đối soát Phân tầng Số liệu & Quy ước Lưu vết

| Hạng mục | Quyết định thiết kế | Đánh đổi chấp nhận & Bằng chứng kiểm soát |
| :--- | :--- | :--- |
| **Làm sạch bảo thủ dữ liệu bị làm dơ** | Lọc toàn bộ dòng khuyết mục tiêu/tọa độ, mã lính canh và giữ `NaN` cho `timestamp`/`surge` khuyết. | Loại `60.91%` dòng raw và loại bảo thủ `65` dòng phi-Taxi xung đột không phân biệt được giữa giá `$10.00`/`1.00 mi` với mã lính canh (`48` dòng `price ∈ {1000, 10000}` và `17` dòng `distance = 1000`; `0.009%` tập thô / `0.024%` quy mô tập sạch); đổi lại loại trừ nhiễu nặng, giữ `270,898` dòng sạch có phân phối tương đồng sát mẫu bị loại. |
| **Ghép thời tiết theo `(location, hour)`** | Chỉ gắn nhãn thời tiết trên các chuyến có `timestamp` gốc và khớp trực tiếp theo quận×giờ. | Giữ `NaN` ở các dòng không khớp trực tiếp thay vì nội suy chéo từ quận khác để bảo toàn *data provenance*; kiểm chứng thêm độ nhạy loại sạch mọi ô chứa sentinel (`n = 87,733`). |
| **Tách vai trò `M7`, `M2 OLS` & `Ex-Ante`** | Duy trì song song `M7` (dự báo), `M2` (diễn giải biểu giá) và cụm `Ex-Ante` (`23` biến). | Tách rõ bài toán định giá theo tham số quan sát đồng thời với bài toán dự báo trước chuyến đi, giúp minh bạch vai trò của `surge_multiplier` và hạn chế hiểu nhầm về *Data Leakage*. |
| **Phạm vi Mô phỏng & Địa lý Boston** | Đóng khung mô phỏng `Surge Cap` dưới dạng phân tích kịch bản trên `API Quote Proxy` tại 12 quận lõi Boston. | Không ngoại suy sang giao dịch hoàn thành, phản ứng chuyển nền tảng (*multi-homing*) của tài xế, hay các khu vực ngoại vi ngoài 12 quận trong dữ liệu; cần kiểm chứng thực địa qua Switchback A/B. |
| **Tách `src/` (lõi DS) khỏi `scripts/` (xuất bản)** | Giữ `main.py` + `11` module lõi trong `src/` thuần Python/Scikit-learn; chuyển `5` script kiểm chứng bổ sung & biên dịch Quarto/Word vào `scripts/`. | Giữ pipeline lõi nhẹ, đa nền tảng và không phụ thuộc Word COM hay Quarto/Typst khi chạy `main.py --stage all`. |
| **Đóng gói dữ liệu gốc (`81.51 MiB`) & Đối soát phân tầng kết quả** | Kèm nguyên `data/raw/cab_rides.csv` và tách bạch lớp đầu ra chuẩn GĐ3–GĐ6 (`stage3..6`) khỏi lớp kiểm chứng mở rộng (`thesis_v5_verification.json`). | Tăng dung lượng gói nộp nhưng bảo đảm tái lập độc lập offline từ đúng dữ liệu thô làm dơ; đồng thời đối soát tường minh các cặp số liệu hai tầng: (i) `[P10, P90]` thô GĐ6 (`stage6_summary.json`: `78.44%` chặt / `80.28%` dung sai `10⁻⁶` USD) vs. sau hiệu chuẩn `Split-CQR` (`79.89%`, `12/12` gói `[78.62%, 81.99%]`, `0` cắt ngược); (ii) `Permutation Importance` `14` nhóm rời (`stage6_permutation_importance.csv`: `name 69.10%`, thời tiết/giờ `0.04%`) vs. `6` cụm gộp chống đa cộng tuyến (`Bảng 4.9`: `name 69.23%`, thời tiết/giờ `0.02%`); (iii) phân rã sai số `stage6_segment_errors.csv` đủ `46,573/46,573` dòng Test trên cả `4` chiều (bao gồm nhóm `5. Khuyết distance (NaN thô)` `n = 3,830`); (iv) RAM `master_rideshare` ngay sau downcast GĐ3 (`37.49 MiB`) vs. khi đọc lại từ Parquet (`37.46 MiB`). |
| **Quy ước Nhật ký & Tái lập Toàn trình** | Ghi [nhật ký làm việc.md](nh%E1%BA%ADt%20k%C3%BD%20l%C3%A0m%20vi%E1%BB%87c.md) theo cấp độ phiên làm việc & nhóm hạng mục (milestone-level); phát triển tuần tự bằng đồng bộ thư mục giữa 2 máy (`Python 3.14.4` & `3.11.9`, không dùng Git), riêng các phiên đồng nghiệm thu và chặng chốt bàn giao cuối (`Phiên 21–24`) hai thành viên thống nhất làm việc và tái lập toàn trình trên cùng 1 máy tổng hợp (`Python 3.11.9`, `pandas 3.1.1 / pyarrow 26.0.1`) để tránh xung đột phiên bản thư viện/ghi đè tệp. | Mỗi phiên đại diện cho một mốc công việc trọn vẹn; `Quarto 1.10.19` (`Typst 0.15.1`) được cài đặt sẵn trong môi trường tham khảo từ các buổi hướng dẫn đầu học kỳ trước khi áp dụng chính thức cho ấn phẩm ở Phiên 21; ở chặng chốt cuối (`Phiên 21–24`), nhóm thống nhất làm chung trên một máy tổng hợp (`Python 3.11.9`), chuẩn hóa PEP 8 (`ruff format ≤ 79`) cho toàn bộ `50` tệp `.py`/`.ipynb` trong gói nộp chuẩn (`46 .py + 4 .ipynb`) và tái chạy trọn pipeline cùng `4/4` notebook để khóa đồng bộ số liệu trước khi đóng gói. |

> **Quy ước chi tiết về Nhật ký Làm việc (`nhật ký làm việc.md`):** `nhật ký làm việc.md` tổng hợp tiến trình theo cấp độ phiên làm việc, các quyết định kiến trúc và kết quả kiểm chứng trọng tâm (milestone-level), không liệt kê từng dòng chỉnh sửa nhỏ hay các lần tinh chỉnh lặp lại trong quá trình hoàn thiện; chi tiết cài đặt cụ thể được phản ánh trực tiếp qua mã nguồn và bộ kiểm thử tự động. Mỗi phiên trong `nhật ký làm việc.md` tương ứng với một mốc công việc trọn vẹn (xây dựng pipeline, huấn luyện/kiểm định mô hình, hoàn thiện báo cáo); các tinh chỉnh nhỏ về trình bày, câu chữ hoặc đồng bộ kiểm tra giữa các mốc được thực hiện trực tiếp trên mã nguồn mà không tách thành phiên riêng.

---

## 5. Kiến trúc Luồng Dữ liệu, Hệ thống Dashboard & Sơ đồ Tệp

Hai phiên bản Dashboard 5 Tab dùng chung tầng nghiệp vụ và biểu đồ Plotly: **Streamlit Web (`dashboard/app.py`, RAM `37.46 MiB`, suy luận 12 gói `M7+M2` đo thực tế ~`18–66 ms`, ngưỡng kiểm thử `< 500 ms`)** và **Offline HTML 1-Click (`reports/dashboard_interactive.html`, `5.27 MiB`, nhúng `plotly.js` nội tuyến, không phụ thuộc CDN ngoài)**. Chi tiết xem [`dashboard/README.md`](dashboard/README.md).

### 5.1. Sơ đồ Luồng Xử lý & Phụ thuộc Giữa Các Tệp (*Pipeline-First Decoupled Architecture*)

```text
[1] Dữ liệu thô đầu vào — data/raw/ (Read-Only, SHA-256 cố định)
│   ├── cab_rides.csv (693,071 × 10)
│   └── weather.csv   (6,276 × 8)
│
v
[2] Pipeline lõi CRISP-DM — main.py (--stage clean|eda|features|model|evaluate|all; nạp src/config.py)
│   ├── clean    : src/clean_raw_data.py
│   │              └──> data/processed/{cab_rides_cleaned.parquet, weather_cleaned.csv}
│   ├── eda      : src/stage3_pipeline_and_audit.py (tự gọi clean_* khi chạy --stage all)
│   │              ├──> data/processed/{master_rideshare.parquet, stage3_audit_summary.json}
│   │              └──> reports/figures/stage3/ (7 ảnh)
│   ├── features : src/features.py
│   │              ├──> data/processed/stage4_audit_flags.parquet
│   │              ├──> reports/tables/{data_dictionary_complete.csv, stage4_*.csv} (1 + 3 tệp)
│   │              └──> reports/results/stage4_summary.json · reports/figures/stage4/ (3 ảnh)
│   ├── model    : src/{stats_tests.py, models.py, stage5_modeling.py}
│   │              ├──> models/{best_model, m2_interaction_ols}.joblib
│   │              ├──> reports/tables/stage5_*.csv (7 tệp) · reports/results/stage5_summary.json
│   │              └──> reports/figures/stage5/ (4 ảnh)
│   └── evaluate : src/{evaluation.py, interpretation.py, simulation.py, stage6_evaluation.py}
│                  ├──> reports/tables/stage6_*.csv (5 tệp) · reports/results/stage6_summary.json
│                  └──> reports/figures/stage6/ (7 ảnh)
│
├──> [3A] Nhánh Kiểm chứng Mở rộng & Xuất bản Học thuật (scripts/, độc lập main.py)
│    ├── thesis_v5_verification.py
│    │   └──> reports/results/thesis_v5_verification.json (ngữ cảnh v5 cho tiểu luận & [3B])
│    ├── prepare_thesis_figures.py
│    │   └──> reports/figures/thesis/ (17 ảnh ≥ 300 DPI)
│    ├── thesis_content.py + build_final_thesis.py
│    │   └──> reports/2521001060_Nhom02_TieuLuan.qmd (nhúng assets/ufm_logo.jpeg & figures/thesis/)
│    │        ──> .pdf (Quarto/Typst, 45 trang) & .docx dự phòng
│    └── check_thesis_audit.py
│        └──> Đối soát tự động số liệu, trang, font, DPI & 23/23 APA 7th
│
└──> [3B] Nhánh Kiểm chứng Notebook & Trình bày Tương tác
     ├── notebooks/03..06_*.ipynb (gọi hàm src/ & đối soát GĐ3–6; Py 3.11.9 & 3.14.4)
     └── dashboard/ (chỉ đọc data/processed/, models/, reports/tables/ & results/ gồm v5 JSON)
         ├── config.py + data_loader.py + validation.py (18/18 PASS)
         ├── business_logic.py + model_loader.py + charts.py
         ├── app.py + components/ (styles, sidebar, kpi_cards, 5 tab_*) ──> Streamlit Web (5 Tab)
         ├── export_static_html.py ──> reports/dashboard_interactive.html (5.27 MiB)
         └── tests/ (__init__.py + 9 tệp kiểm thử; 40/40 pytest PASS)
```

### 5.2. Cây Thư mục Chi tiết Gói Nộp Bài

*(Sơ đồ dưới đây mô tả đúng cấu trúc gói nộp chuẩn `2521001060_Nhom02_TieuLuan/` gồm `128` tệp, khớp `Phụ lục A.1` & `Bảng A.1` trong Tiểu luận PDF; các thư mục lưu trữ/kiểm toán nội bộ phát sinh trong môi trường phát triển như `archive/`, `audit_v4/`, `scratch/` được tách ngoài gói nộp và khai báo loại trừ tại `pyproject.toml`.)*

```text
2521001060_Nhom02_TieuLuan/                        # 128 tệp (5 gốc + 1 .streamlit + 7 data + 11 src + 6 scripts
│                                                  #   + 2 models + 4 notebooks + 29 dashboard + 63 reports)
├── main.py                                        # CLI lõi (--stage clean|eda|features|model|evaluate|all, --test-size)
├── pyproject.toml                                 # Cấu hình PEP 8 ruff (line-length = 79; kiểm cả scripts/ & notebooks/)
├── requirements.txt                               # Phụ thuộc Python >= 3.11 (đã kiểm trên Python 3.11.9 & 3.14.4)
├── README.md                                      # Tài liệu tổng quan & ánh xạ tiêu chí projects.pdf (tệp này)
├── nhật ký làm việc.md                            # Nhật ký P-D-M-V nhóm (24 phiên: 10 + 11 + 3, α1 = α2 = 1.00)
├── .streamlit/config.toml                         # 1 tệp cấu hình Executive Light Theme cho Streamlit
├── data/                                          # 7 tệp (2 raw/ chỉ đọc + 5 processed/ sau làm sạch & kiểm toán)
│   ├── raw/                                       # 2 tệp gốc bị làm dơ — chỉ đọc (read-only, SHA-256 cố định)
│   │   ├── cab_rides.csv                          # Dữ liệu chuyến đi gốc (693,071 × 10, 81.51 MiB)
│   │   └── weather.csv                            # Dữ liệu khí tượng gốc (6,276 × 8, 325.09 KiB)
│   └── processed/                                 # 5 tệp đầu ra sạch GĐ3–GĐ4
│       ├── cab_rides_cleaned.parquet              # Bảng chuyến đi sau làm sạch 7 tầng nhiễu (270,898 × 12; 11.54 MiB)
│       ├── weather_cleaned.csv                    # Khí tượng sau khử lặp (location, date_hour) & bậc 10 (5,053 × 9)
│       ├── master_rideshare.parquet               # Bảng tổng thể Left Join (270,898 × 32; 15.97 MiB đĩa / 37.46 MiB RAM)
│       ├── stage4_audit_flags.parquet             # Cờ kiểm toán nguyên bản R3 (270,898 × 4; 88.85% không scale-fix)
│       └── stage3_audit_summary.json              # Tóm tắt đối soát làm sạch, thống kê mô tả & tỷ lệ hao hụt
├── src/                                           # 11 module lõi DS (nạp qua sys.path; không phụ thuộc Word/Quarto)
│   ├── config.py                                  # Đường dẫn pathlib, hằng số tái lập & danh sách cấm FORBIDDEN_IN_X
│   ├── clean_raw_data.py                          # Đọc chunksize, chuẩn hóa đơn vị, khôi phục bậc 10 & mã lính canh
│   ├── stage3_pipeline_and_audit.py               # Ghép nối chống fan-out, chuẩn hóa giờ EST, downcast RAM & EDA GĐ3
│   ├── features.py                                # Tạo 9 biến mới, VIF, 3 kiểu chia & EconomicInteractionPreprocessor
│   ├── stats_tests.py                             # 13 kiểm định H1–H4, Effect Size, Wald HC3 (H4c 47/47) & biểu giá
│   ├── models.py                                  # 10 mô hình (M0a–M7, Smearing, Tuning) trong Scikit-learn Pipeline
│   ├── stage5_modeling.py                         # 5-Fold CV, Conditional vs. Ex-Ante, độ bền & 6 Mutate & Break
│   ├── evaluation.py                              # Chẩn đoán phần dư (JB, BP), sai số phân khúc & Quantile [P10, P90]
│   ├── interpretation.py                          # Grouped Permutation Importance (14 nhóm rời = 27/27 cột) & PDP/ICE
│   ├── simulation.py                              # Mô phỏng kịch bản Surge Cap bên ngắn Q_realized = min(Q_D, Q_S)
│   └── stage6_evaluation.py                       # Tổng hợp GĐ6: bảng kết quả CSV/JSON & hình ảnh đánh giá mô hình
├── scripts/                                       # 6 tệp (__init__.py + 5 script kiểm chứng mở rộng & xuất bản)
│   ├── __init__.py                                # Khai báo gói Python cho thư mục scripts/
│   ├── thesis_v5_verification.py                  # Kiểm chứng phản biện (Cluster SE 2 cấp, WCR, TOST, CQR, Switchback)
│   ├── prepare_thesis_figures.py                  # Kết xuất 17 biểu đồ Tiểu luận chuẩn in ấn ≥ 300 DPI
│   ├── thesis_content.py                          # Nội dung động 5 chương, 21 bảng (+ Bảng B.3) & 23 tài liệu tham khảo APA 7th
│   ├── build_final_thesis.py                      # Sinh .qmd, biên dịch PDF qua Quarto/Typst & đồng bộ .docx dự phòng
│   └── check_thesis_audit.py                      # Cổng kiểm toán toàn văn báo cáo (trang, mục lục, DPI, font, APA 7th)
├── models/                                        # 2 tệp mô hình đã huấn luyện (read-only cho Dashboard)
│   ├── best_model.joblib                          # Mô hình chính M7 (HistGradientBoostingRegressor Pipeline)
│   └── m2_interaction_ols.joblib                  # Mô hình diễn giải kinh tế lượng M2 (Interaction OLS Pipeline)
├── notebooks/                                     # 4 Notebook thực thi kiểm chứng GĐ3–GĐ6 (0 lỗi ruff PEP 8)
│   ├── 03_Giai_Doan_3_Data_Prep_EDA_Visualization.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Hypothesis_Testing_Modeling.ipynb
│   └── 06_Evaluation_Interpretation_Policy.ipynb
├── dashboard/                                     # 29 tệp Dashboard 5 Tab (10 tệp gốc + 9 components/ + 10 tests/)
│   ├── __init__.py, config.py, README.md          # 3 tệp khởi tạo gói, đường dẫn tương đối, schema & hướng dẫn
│   ├── data_loader.py, validation.py              # 2 tệp nạp Parquet/CSV/JSON & cổng kiểm toán dữ liệu (18/18 PASS)
│   ├── business_logic.py, model_loader.py         # 2 tệp tính KPI/bộ lọc nghiệp vụ & suy luận live M7+M2
│   ├── app.py, charts.py, export_static_html.py   # 3 tệp Streamlit 5 Tab, biểu đồ Plotly & xuất HTML offline (10 gốc)
│   ├── components/                                # 9 tệp: __init__.py, styles.py, sidebar.py, kpi_cards.py,
│   │                                              #   tab_overview.py, tab_pricing_rq1.py, tab_tariff_rq2.py,
│   │                                              #   tab_models_rq3.py, tab_simulator_rq4.py
│   └── tests/                                     # 10 tệp (__init__.py + 9 tệp kiểm thử; 40/40 pytest PASS):
│                                                  #   test_{data_loader,validation,business_logic,model,
│                                                  #   export_and_perf}.py & 4 tệp test_ui_stage3..6.py
└── reports/                                       # 63 tệp (3 tiểu luận + 1 ảnh bìa + 1 HTML + 38 PNG + 4 JSON + 16 CSV)
    ├── 2521001060_Nhom02_TieuLuan.{qmd,pdf,docx}  # 3 tệp Tiểu luận Quarto/Typst (45 trang / 43 nội dung ngoài bìa, 9.35 MiB PDF)
    ├── assets/ufm_logo.jpeg                       # 1 ảnh biểu trưng UFM nhúng trên Trang bìa (.qmd/.pdf/.docx)
    ├── dashboard_interactive.html                 # 1 tệp Dashboard HTML 5 Tab chạy offline (5.27 MiB)
    ├── figures/                                   # 38 ảnh PNG (21 ảnh stage3..6 [7+3+4+7] + 17 ảnh thesis/ ≥ 300 DPI)
    ├── results/                                   # 4 tệp JSON (3 tệp stage4..6_summary.json & thesis_v5_verification.json)
    └── tables/                                    # 16 bảng CSV (15 bảng chuẩn GĐ4–GĐ6 & data_dictionary_complete.csv)
```

---

## 6. Lộ trình Kiểm định Nhanh & Tái lập Toàn trình

### A. Kiểm định Nhanh cho Người chấm (~1 phút)
```bash
# 1. Tạo môi trường ảo (Python >= 3.11) & cài đặt thư viện
python -m venv .venv
# Windows PowerShell (macOS/Linux: source .venv/bin/activate):
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Kiểm toán dữ liệu (18/18 PASS), kiểm thử tự động (40/40 pytest PASS) & kiểm tra báo cáo
python -m dashboard.validation
pytest dashboard/tests -v
python scripts/check_thesis_audit.py

# 3. Kiểm tra quy chuẩn PEP 8 theo cấu hình pyproject.toml (line-length = 79)
# (Cài thêm ruff nếu môi trường chưa có: pip install ruff)
ruff check main.py src scripts dashboard notebooks
ruff format --check main.py src scripts dashboard notebooks
```

### B. Khởi chạy Dashboard & Tái lập Toàn trình từ Dữ liệu Thô (~10 phút)
```bash
# Khởi chạy Dashboard Streamlit hoặc tái xuất HTML Offline (5.27 MiB)
streamlit run dashboard/app.py
python -m dashboard.export_static_html

# Tái lập toàn bộ Pipeline GĐ3–GĐ6 từ data/raw/* (~9–10 phút do 5-Fold CV Random Forest)
python main.py --stage all

# Tái lập kiểm chứng mở rộng, 17 hình ≥ 300 DPI và xuất bản Tiểu luận Quarto (.qmd -> .pdf)
python scripts/thesis_v5_verification.py
python scripts/prepare_thesis_figures.py
# (Yêu cầu cài đặt Quarto CLI >= 1.4; --skip-docx tạo .qmd & biên dịch thẳng PDF Typst không cần MS Word COM; bỏ cờ nếu cần đồng bộ thêm .docx trên Windows)
python scripts/build_final_thesis.py --skip-docx
# Hoặc biên dịch trực tiếp tệp .qmd sẵn có sang PDF Typst:
quarto render reports/2521001060_Nhom02_TieuLuan.qmd
```

> **Ghi chú môi trường:** Đã kiểm chứng trên Windows 64-bit với hai phiên bản **Python `3.11.9`** (`pandas 3.1.1 / pyarrow 26.0.1`) và **Python `3.14.4`** cùng **`Quarto 1.10.19` (`Typst 0.15.1`)**. Tệp `.qmd` tích hợp trọn vẹn cấu hình Typst và danh mục tài liệu tham khảo nội tuyến (chỉ tham chiếu tương đối ảnh bìa và `17` biểu đồ trong `reports/assets/` và `reports/figures/thesis/`), biên dịch trực tiếp sang PDF chuẩn UFM bằng `quarto render` mà không cần cài thêm LaTeX. Khi chạy trên các phiên bản mới (`pandas 3.0+` / `matplotlib 3.11+`), thư viện bên thứ ba có thể phát sinh một vài `DeprecationWarning` không ảnh hưởng tới kết quả tính toán.

---

## 7. Danh mục Sản phẩm Bàn giao

- **Nhật ký P-D-M-V nhóm (`24` phiên):** [`nhật ký làm việc.md`](nh%E1%BA%ADt%20k%C3%BD%20l%C3%A0m%20vi%E1%BB%87c.md)
- **Tiểu luận Quarto chuẩn UFM (`45` trang tổng / `43/45` trang nội dung ngoài bìa / `27` Ch.1–5, `187.1 KiB` QMD → `9.35 MiB` PDF Typst / `8.59 MiB` DOCX, `22` bảng):** [`reports/2521001060_Nhom02_TieuLuan.qmd`](reports/2521001060_Nhom02_TieuLuan.qmd) · [`reports/2521001060_Nhom02_TieuLuan.pdf`](reports/2521001060_Nhom02_TieuLuan.pdf) · [`reports/2521001060_Nhom02_TieuLuan.docx`](reports/2521001060_Nhom02_TieuLuan.docx)
- **Bằng chứng *Mutate & Break* & Kiểm thử tự động:** [`reports/tables/stage5_mutate_break.csv`](reports/tables/stage5_mutate_break.csv) · [`dashboard/tests/test_validation.py`](dashboard/tests/test_validation.py)
- **Dashboard Offline & Tài liệu kiến trúc:** [`reports/dashboard_interactive.html`](reports/dashboard_interactive.html) · [`dashboard/README.md`](dashboard/README.md)
