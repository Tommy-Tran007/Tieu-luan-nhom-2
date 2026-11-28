"""
Export, Documentation Consistency & Performance Tests cho Giai đoạn 7.
Kiểm tra:
1. Tệp báo cáo HTML độc lập reports/dashboard_interactive.html tồn tại, dung
lượng hợp lý (< 12 MB),
   chạy 100% offline (nhúng plotly.js nội tuyến, 0 liên kết cdn.plot.ly), đủ 5
   Tab & không còn số liệu lỗi thời.
2. README.md gốc đã đồng bộ sạch 100% số liệu sau Final Audit (0 chuỗi cũ trước
Audit).
3. Kiểm định hiệu năng (Performance Testing): thời gian nạp Parquet, dung lượng
RAM và độ trễ dự báo live M7+M2.
"""

from __future__ import annotations

import re
import time

from dashboard.business_logic import (
    compute_route_catalog,
    compute_weather_reference_medians,
)
from dashboard.config import BASE_DIR, HTML_EXPORT_PATH
from dashboard.data_loader import (
    load_all_result_tables,
    load_master_dataset,
    measure_ram_mb,
)
from dashboard.model_loader import load_trained_models, predict_fares_all_tiers


def test_standalone_html_report_integrity_offline_and_clean() -> None:
    """
    Kiểm chứng báo cáo HTML độc lập reports/dashboard_interactive.html sạch, đủ
    5 Tab và 100% offline.
    """
    assert HTML_EXPORT_PATH.exists(), (
        f"Thiếu tệp báo cáo HTML tại {HTML_EXPORT_PATH}"
    )
    size_mb = HTML_EXPORT_PATH.stat().st_size / (1024**2)
    assert 3.0 <= size_mb <= 12.0, (
        f"Dung lượng HTML bất thường: {size_mb:.2f} MB"
    )

    content = HTML_EXPORT_PATH.read_text(encoding="utf-8")
    # Kiểm tra 100% offline: không có bất kỳ thẻ <script src=...> hoặc <link
    # href="http..."> ngoài cục bộ
    assert re.findall(r"<script[^>]+src=", content, flags=re.I) == [], (
        "HTML chứa thẻ <script src=...> phụ thuộc ngoài!"
    )
    assert (
        re.findall(r"<link[^>]+href=['\"]https?://", content, flags=re.I) == []
    ), "HTML chứa <link> ngoài cục bộ!"
    assert "Plotly" in content

    body = content.split("<body>", 1)[1]
    # Kiểm tra đủ 5 Tab và các dấu hiệu học thuật v5 bắt buộc trong <body>
    for tab_id in (
        'id="tab1"',
        'id="tab2"',
        'id="tab3"',
        'id="tab4"',
        'id="tab5"',
    ):
        assert tab_id in body, f"Thiếu {tab_id} trong báo cáo HTML!"
    assert "API Quote Proxy" in body
    assert "EXTRAPOLATION WARNING" in body
    assert "Q_realized = min(Q_D, Q_S)" in body
    assert "0 Nhãn mưa giả" in body and "Boston EST" in body
    assert "0.9626" in body and "6.80%" in body and "46,573" in body

    # Xác nhận 0 chuỗi số liệu cũ v3/v4 hoặc thuật ngữ cấm trong nội dung báo
    # cáo <body>
    for stale in (
        "37.61",
        "233,362",
        "69,348",
        "$2.0195",
        "$4.54",
        "-11.52%",
        "79.34%",
        "0.9531",
        "54,180",
        "4.82%",
        "$2,267,293.70",
        "3.10 dặm",
        "10 giờ mưa lớn",
        "~15–34 ngày",
        "(l.b0 + l.b1 * d) * s",
        "Cước phí",
        "cước phí",
    ):
        assert stale not in body, (
            f"Phát hiện chuỗi/số liệu cũ '{stale}' trong nội dung báo cáo "
            "HTML!"
        )
    assert "l.b0 + l.b1 * d * s" in body, (
        "Offline HTML Estimator phải dùng đúng công thức HC3 OLS: "
        "l.b0 + l.b1 * d * s!"
    )


def test_root_readme_synchronized_post_audit() -> None:
    """
    Xác nhận README.md gốc đã đồng bộ sạch 100% số liệu v5 (0 chuỗi cũ v3/v4).
    """
    readme_text = (BASE_DIR / "README.md").read_text(encoding="utf-8")
    for stale in (
        "270,898 × 28",
        "270,898 × 30",
        "6` điểm cắt",
        "$2.0195",
        "$4.54",
        "-11.52%",
        "doanh thu ròng Lyft",
        "79.34%",
        "0.9531",
        "54,180",
        "4.82%",
        "$2,267,293.70",
        "thesis_v4_verification.py",
        "17/17",
        "38/38",
    ):
        assert stale not in readme_text, (
            f"README.md vẫn còn số liệu/thuật ngữ cũ: {stale!r}"
        )
    for required in (
        "streamlit run dashboard/app.py",
        "dashboard_interactive.html",
        "API Quote Proxy",
        "270,898 × 32",
        "0.9626",
        "6.80%",
        "18/18",
        "40/40",
        "thesis_v5_verification.py",
        "Q_realized = min(Q_D, Q_S)",
    ):
        assert required in readme_text, (
            f"README.md thiếu nội dung v5 bắt buộc: {required!r}"
        )


def test_performance_load_ram_and_live_inference_latency() -> None:
    """
    Đo hiệu năng thực tế: tốc độ đọc Parquet, RAM <= 50 MB và suy luận 12 gói
    M7+M2 < 0.5 giây.
    """
    t0 = time.perf_counter()
    master = load_master_dataset()
    t_load_parquet = time.perf_counter() - t0
    ram_mb = measure_ram_mb(master)

    assert t_load_parquet < 3.0, (
        f"Thời gian nạp Parquet quá chậm: {t_load_parquet:.2f}s"
    )
    assert ram_mb < 50.0, f"Dung lượng RAM vượt ngưỡng tối ưu: {ram_mb:.2f} MB"

    tables = load_all_result_tables()
    models = load_trained_models()
    routes = compute_route_catalog(master)
    w_ref = compute_weather_reference_medians(master)

    t1 = time.perf_counter()
    res = predict_fares_all_tiers(
        models=models,
        pi_df=tables["prediction_interval"],
        route_catalog=routes,
        weather_ref=w_ref,
        source="Back Bay",
        destination="South Station",
        distance=1.51,
        lyft_surge=1.25,
    )
    t_infer = time.perf_counter() - t1

    assert res["input_check"].is_valid
    assert t_infer < 0.50, (
        f"Độ trễ dự báo live 12 gói (M7+M2) quá cao: {t_infer:.3f}s"
    )
