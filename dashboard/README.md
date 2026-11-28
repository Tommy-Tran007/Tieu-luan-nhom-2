# Dashboard Phân tích Định giá Động & Dự báo Giá báo API Uber/Lyft (`PRJ-EXT-01`)

- **Đề tài:** *Uber vs. Lyft: Boston Rideshare Dynamic Pricing & API Fare Quote Analysis*
- **Nhóm thực hiện:** Nhóm 02 — Trần Thanh Huy (`2521001060`) & Lê Huy Thắng (`2521001092`) — Lớp `25DDS01` (UFM)
- **Công nghệ:** `Python >= 3.11` (`3.14.4`) + `Streamlit` + `Plotly` + `Pandas` + `Scikit-learn` + `Joblib` + `Parquet`

---

## 1. Hai Phiên bản Dashboard (Đồng bộ từ cùng một nguồn logic phân tích)

1. **Ứng dụng Web Tương tác Streamlit (`dashboard/app.py` — Phiên bản chính):**
   - Lọc động trực tiếp trên `270,898 × 32` quan sát `master_rideshare.parquet` (`37.46 MiB` RAM, bảo toàn nguyên vẹn `38,037` dòng Lyft khuyết surge `NaN` và `is_surge_original`).
   - Chạy dự báo giá báo API có điều kiện thời gian thực từ `models/best_model.joblib` (`M7`, `R² = 0.9626`, `RMSE = $1.7830`) & `models/m2_interaction_ols.joblib` (`M2`, `R² = 0.9539`) kèm cảnh báo ngoại suy và mô phỏng Surge Cap theo nguyên tắc bên ngắn `Q_realized = min(Q_D, Q_S)` với hàm cung nhạy giá `ε_S ∈ {0.30, 0.60}`.
2. **Báo cáo HTML Tương tác Độc lập (`reports/dashboard_interactive.html` — Phiên bản dự phòng Offline):**
   - Một tệp duy nhất **`5.27 MiB`**, mở thẳng bằng trình duyệt Chrome/Edge **không cần Python server và không cần Internet** (nhúng sẵn `plotly.js` nội tuyến, `0` CDN ngoài).
   - Giữ trọn vẹn `5` Tab, biểu đồ Plotly tương tác (zoom, hover, đổi `5` cặp biểu giá, đổi 3 chỉ số Heatmap), công cụ ước lượng biểu giá `HC3 OLS` và bộ mô phỏng Surge Cap tích hợp bộ chọn cung nhạy giá `ε_S = 0.00 / 0.30 / 0.60` thuần JS.

> **Lưu ý thuật ngữ xuất bản:** Cả hai phiên bản trên là **Dashboard tương tác vector Plotly** (không gọi là `300 DPI`). Bộ **`21` hình tĩnh GĐ3–GĐ6** (`reports/figures/stage3..6/`) và **`17` hình tiểu luận chuẩn `300 DPI PNG`** (`reports/figures/thesis/`) được xuất tách biệt để in ấn tiểu luận.

---

## 2. Hướng dẫn Cài đặt & Chạy trên Môi trường Sạch

Chạy toàn bộ các lệnh dưới đây từ **thư mục gốc dự án (`TieuLuan_PRJ-EXT-01_Nhom02/`)**:

```bash
# 1. (Khuyên dùng) Tạo và kích hoạt môi trường ảo sạch
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 2. Cài đặt đầy đủ thư viện
pip install -r requirements.txt

# 3. Chạy cổng kiểm toán dữ liệu độc lập (18/18 hạng mục PASS)
python -m dashboard.validation

# 4. Chạy bộ kiểm thử tự động (9 tệp / 40 ca pytest PASS)
pytest dashboard/tests -v

# 5. Khởi chạy Dashboard Web Streamlit
streamlit run dashboard/app.py

# 6. Tái tạo tệp Báo cáo HTML Tương tác Offline (reports/dashboard_interactive.html)
python -m dashboard.export_static_html
```

---

## 3. Cấu trúc Kiến trúc Tách lớp (`dashboard/`)

```text
dashboard/
├── app.py                   # Bộ điều phối giao diện 5 Tab (0 công thức hardcode, @st.cache_resource)
├── config.py                # Đường dẫn tương đối pathlib, hợp đồng schema 32 cột, hằng số miền & nhãn Quote Proxy
├── data_loader.py           # Tầng đọc Parquet/CSV/JSON an toàn, đo tối ưu RAM, ném DataLoadError rõ ràng
├── validation.py            # Cổng kiểm toán 18 hạng mục (Schema 32 cột, PK, 0 fan-out, CRIT-01/02/03, H4, HIGH-01..03)
├── business_logic.py        # Công thức KPI, bộ lọc giữ NaN surge, cưỡng chế CRIT-01..03, HIGH-02/03 & C2 Q_realized
├── model_loader.py          # Nạp M7/M2 độc lập cwd, đánh giá trên is_surge_original == 1 (46,573 Test) & cảnh báo ngoại suy
├── charts.py                # Hàm Plotly thuần (dùng chung 100% giữa app.py và export_static_html.py)
├── export_static_html.py    # Trình xuất báo cáo HTML 5 Tab chạy 100% offline tại reports/dashboard_interactive.html
├── components/              # Các mô-đun giao diện Streamlit (styles, sidebar, kpi_cards, tab_1..tab_5)
└── tests/                   # Bộ 9 tệp / 40 ca kiểm thử đơn vị, mutation tests, C2 monotonicity & Streamlit AppTest
```

---

## 4. Hướng dẫn Triển khai & Chạy Cục bộ / Trực tuyến (Streamlit)

1. Đảm bảo toàn bộ thư mục `dashboard/`, `.streamlit/config.toml`, `requirements.txt`, `data/processed/master_rideshare.parquet`, `models/*.joblib`, `reports/tables/*.csv`, `reports/results/*.json` và `src/` nằm đúng cấu trúc trong thư mục gốc dự án.
2. Cài đặt phụ thuộc (nếu chưa có): `pip install -r requirements.txt` và khởi chạy cục bộ bằng lệnh `streamlit run dashboard/app.py`, hoặc mở trực tiếp `reports/dashboard_interactive.html` không cần máy chủ.
3. Khi triển khai lên Streamlit Community Cloud qua [share.streamlit.io](https://share.streamlit.io), trỏ **Main file path** vào **`dashboard/app.py`** trên gói dự án đã đóng gói.
4. Bộ nhớ vận hành của ứng dụng chỉ ~`37.5 MiB` cho dữ liệu sạch + ~`1 MiB` cho 2 tệp mô hình `.joblib`, vận hành thoải mái trong hạn mức `1 GiB RAM` tiêu chuẩn.
